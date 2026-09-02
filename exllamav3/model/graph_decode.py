from __future__ import annotations
import os
import torch

# Whole-step CUDA/HIP graph capture for single-token decode.
#
# Motivation (measured on gfx1150 / ROCm 7.2.4, qwen3-0.6b-4bpw, Triton EXL3
# linear path): a decode step issues ~591 Triton launches, and 82.7% of the
# step's wall time is host-side dispatch that never touches the GPU. Triton's
# HIP launcher calls hipPointerGetAttribute once per tensor argument per launch
# (triton/backends/amd/driver.py), so the GPU idles between kernels. Replaying
# the step as one graph collapses all of that to a single graph launch.
#
# This is a parallel path: Model.forward dispatches here once, and nothing on
# the eager path changes. Opt in with EXL3_DECODE_GRAPH=1; default is off.
#
# Not captured: the leading CPU-resident module(s). Embedding sets
# caps["prefer_cpu"] = True unconditionally, so the embedding runs on the host
# and its output reaches the GPU through a pageable H2D copy in
# Module.prepare_for_device -- which cannot be captured. The graph therefore
# starts at the first GPU-resident module and the host writes the embedding
# output into a stable device buffer each step (measured at 0.012 ms/token).

enabled = os.environ.get("EXL3_DECODE_GRAPH", "0") != "0"

# Number of eager steps observed for a shape signature before capturing it.
# Triton autotune runs host-side benchmarking on first call for a shape and
# must be settled before capture, so this must be >= 1.
_WARMUP_STEPS = int(os.environ.get("EXL3_DECODE_GRAPH_WARMUP", "2"))

# Distinct shape signatures to keep captured at once. The generator rounds the
# block-table width up to a multiple of 16 pages (generator.py, iterate_gen),
# so the signature changes once every PAGE_SIZE*16 = 4096 tokens of context.
_MAX_GRAPHS = int(os.environ.get("EXL3_DECODE_GRAPH_CACHE", "4"))

# After capturing, replay once and compare bitwise against an eager step with
# identical inputs. A mismatch means something in the step was not static, so
# the graph is discarded and the signature never retried. Cheap (one extra step
# per capture) and the only thing standing between a subtly dynamic module and
# silently wrong logits. Do not turn this off outside benchmarking.
_VERIFY = os.environ.get("EXL3_DECODE_GRAPH_VERIFY", "1") != "0"

_DEBUG = os.environ.get("EXL3_DECODE_GRAPH_DEBUG", "0") != "0"


def _log(msg: str):
    if _DEBUG:
        print(f" -- decode_graph: {msg}", flush = True)


# Module types whose single-token decode forward is known NOT to be static and
# device-resident, i.e. it host-syncs, branches on tensor values, or produces
# data-dependent shapes. A model containing any of these falls back to eager.
#
# To opt a module in once its forward becomes static, either drop its name from
# this set or set caps["graph_capturable"] = True on the module (the cap wins,
# so a module can opt itself in from its own file without touching this one).
#
# BlockSparseMLP: the ROCm fallback path host-syncs on torch.bincount and
# expert_count.tolist(), then iterates range(num_ex) on the host with
# data-dependent index_select/index_add_ shapes. Work to make that loop static
# and device-resident is in flight separately; when it lands, this entry goes.
_UNCAPTURABLE_TYPES = frozenset({
    "BlockSparseMLP",
    "QSAIndexer",       # attn.py: qsa_seqlens_cpu.max().item() picks the kernel
    "MLAttention",      # host-side seqlen reductions in the MLA path
    "DSV4Attention",
})


def _module_capturable(module) -> bool:
    cap = module.caps.get("graph_capturable")
    if cap is not None:
        return bool(cap)
    if type(module).__name__ in _UNCAPTURABLE_TYPES:
        return False
    # Recurrent state is advanced outside forward() and carries its own
    # host-side bookkeeping; not in scope here.
    if module.caps.get("recurrent_cache"):
        return False
    return True


class _CapturedStep:
    """One captured graph plus the buffers a replay reads and writes."""

    def __init__(self, graph, static_h, out, dev_seqlens, dev_block_table, cache):
        self.graph = graph
        self.static_h = static_h
        self.out = out
        self.dev_seqlens = dev_seqlens
        self.dev_block_table = dev_block_table
        # The signature keys on id(cache); holding the reference keeps that id
        # from being recycled by a new cache while this graph is still live
        # (same reason util.tensor.get_for_device keeps its source tensor alive)
        self.cache = cache


class DecodeGraphs:
    """
    Captures the GPU-resident portion of a single-token decode step and replays
    it, falling back to the eager path whenever the step is not capturable.

    Attached to a Model as model.decode_graphs and driven from Model.forward.

    CALLER CONTRACT -- the logits returned by a replay alias the graph's static
    output buffer, which the NEXT replay overwrites. This differs from the eager
    path, which returns a fresh tensor every step. Consume (sample from, or
    clone) the logits before calling forward() again. The generator satisfies
    this: it argmaxes/samples the batch logits in the same iteration.
    """

    def __init__(self, model):
        self.model = model
        self.graphs = {}        # signature -> _CapturedStep
        self.seen = {}          # signature -> eager steps observed so far
        self.rejected = set()   # signatures that failed verification
        self.split_idx = None   # first GPU-resident module index
        self.device = None
        self._eligible = None
        self._reason = ""
        # Instrumentation: a caller can assert replays actually happened rather
        # than the whole run having quietly fallen back to eager
        self.n_captures = 0
        self.n_replays = 0
        self.n_fallbacks = 0

    # ------------------------------------------------------------------ gate

    def eligible(self) -> bool:
        """Whole-model check, evaluated once."""
        if self._eligible is not None:
            return self._eligible
        self._eligible, self._reason = self._check_eligible()
        _log(f"eligible={self._eligible} ({self._reason})")
        return self._eligible

    def _check_eligible(self):
        model = self.model
        if getattr(model, "loaded_tp", False):
            return False, "tensor-parallel"
        fwd = model.fwd_modules
        if not fwd:
            return False, "no modules"

        # Leading CPU-resident prefix runs eagerly (the embedding). A
        # CPU-resident module anywhere after it would split the graph.
        split = 0
        for module, _, _ in fwd:
            if module.caps.get("prefer_cpu") or module.caps.get("x_cpu"):
                split += 1
            else:
                break
        if split == 0:
            return False, "no CPU prefix to start after (unexpected)"
        dev = fwd[split][0].device
        for module, _, _ in fwd[split:]:
            if module.caps.get("prefer_cpu") or module.caps.get("x_cpu"):
                return False, f"CPU-resident module in interior: {module.key}"
            if module.device is None:
                return False, f"unloaded module: {module.key}"
            # One graph is captured on one device's stream; a layer split across
            # GPUs would need one graph per device plus the transfers between
            if module.device != dev:
                return False, f"layer split across devices ({dev} vs {module.device})"

        # Recursive scan: every submodule must be capturable
        for module, _, _ in fwd[split:]:
            for sub in module:
                if not _module_capturable(sub):
                    return False, f"{type(sub).__name__} ({sub.key}) is not capturable"

        self.split_idx = split
        self.device = fwd[split][0].device
        return True, f"capturing modules [{split}..{len(fwd)}) on {self.device}"

    # ------------------------------------------------------------- signature

    def _signature(self, input_ids, params):
        """
        Everything a replay must hold constant. Contents of the cache-length and
        block-table tensors may change freely between replays; their shapes may
        not, nor may anything that steers a kernel's launch configuration.
        """
        bt = params["block_table"]
        cs = params["cache_seqlens"]
        cache = params.get("cache")
        return (
            tuple(input_ids.shape),
            tuple(bt.shape), bt.dtype,
            tuple(cs.shape), cs.dtype,
            id(cache),
            params.get("causal", True),
            params.get("last_tokens_only"),
            bool(params.get("positions") is not None),
            bool(params.get("position_ids") is not None),
        )

    def _capturable_call(self, input_ids, params) -> bool:
        if input_ids.shape[0] != 1 or input_ids.numel() != 1:
            return False                        # single-token, single-sequence only for now
        if params.get("prefill"):
            return False
        if "block_table" not in params or params.get("cache_seqlens") is None:
            return False                        # generator paged mode only; batch_shape mode
                                                # bakes a python-int rope position
        if params.get("position_ids") is not None:
            return False
        if params.get("sim_kvq") is not None or "ovr" in params:
            return False
        if params.get("recurrent_states"):
            return False
        return True

    # ----------------------------------------------------------------- entry

    def forward(self, input_ids, params):
        """
        Returns logits for the step, or None to tell the caller to run eager.

        A returned tensor is only valid until the next call (see the class
        docstring): it is the captured graph's output buffer, not a fresh one.
        """
        if not self.eligible() or not self._capturable_call(input_ids, params):
            self.n_fallbacks += 1
            return None
        sig = self._signature(input_ids, params)
        if sig in self.rejected:
            self.n_fallbacks += 1
            return None

        cap = self.graphs.get(sig)
        if cap is not None:
            return self._replay(cap, input_ids, params)

        n = self.seen.get(sig, 0) + 1
        self.seen[sig] = n
        if n <= _WARMUP_STEPS:
            self.n_fallbacks += 1
            return None                         # eager, settles Triton autotune
        return self._capture(sig, input_ids, params)

    # --------------------------------------------------------------- machinery

    def _run_prefix(self, input_ids, params):
        """Eager CPU-resident prefix (the embedding), returns a device tensor."""
        x = input_ids
        for module, instance, _ in self.model.fwd_modules[:self.split_idx]:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x.to(self.device).contiguous()

    def _run_body(self, x, params):
        """The captured region: every GPU-resident module, in order."""
        for module, instance, _ in self.model.fwd_modules[self.split_idx:]:
            params["layer_instance"] = instance
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x

    def _bind(self, params, cap):
        """Point params at the graph's persistent device buffers."""
        p = dict(params)
        p["cache_seqlens"] = cap.dev_seqlens
        p["block_table"] = cap.dev_block_table
        p["positions"] = cap.dev_seqlens
        p.pop("position", None)
        p.pop("dev_cache", None)
        return p

    def _refresh(self, cap, h, params):
        """Copy this step's inputs into the graph's buffers.

        TransformerBlock.forward does `x += y`, so the graph's input buffer is
        clobbered by every replay and MUST be rewritten before the next one.
        """
        cap.static_h.copy_(h)
        # Async H2D from the caller's pinned staging buffers. Same contract as
        # util.tensor.get_for_device: the caller must not refill those buffers
        # before the next sync point. The generator satisfies this -- it syncs
        # every iteration collecting the sampled token.
        src_cs = params["cache_seqlens"]
        cap.dev_seqlens.copy_(src_cs, non_blocking = src_cs.is_pinned())
        src_bt = params["block_table"]
        cap.dev_block_table.copy_(src_bt, non_blocking = src_bt.is_pinned())

    def _replay(self, cap, input_ids, params):
        h = self._run_prefix(input_ids, params)
        self._refresh(cap, h, params)
        cap.graph.replay()
        self.n_replays += 1
        return cap.out

    def _capture(self, sig, input_ids, params):
        model = self.model
        h = self._run_prefix(input_ids, params)

        dev_seqlens = params["cache_seqlens"].to(self.device).contiguous().clone()
        dev_block_table = params["block_table"].to(self.device).contiguous().clone()
        static_h = h.clone()
        cap = _CapturedStep(None, static_h, None, dev_seqlens, dev_block_table,
                            params.get("cache"))
        p = self._bind(params, cap)

        # Eager reference for this exact step, from the same bound params
        self._refresh(cap, h, params)
        ref = self._run_body(cap.static_h, p)
        torch.cuda.synchronize()
        ref = ref.clone()

        # Warm up on a side stream, then capture. Capture records without
        # executing, so nothing here writes the cache or mutates static_h.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._refresh(cap, h, params)
                self._run_body(cap.static_h, p)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._refresh(cap, h, params)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = self._run_body(cap.static_h, p)
        except Exception as e:
            _log(f"capture failed for {sig}: {type(e).__name__}: {e}")
            self.rejected.add(sig)
            return None
        cap.graph = graph
        cap.out = out

        # Replay produces this step's real output (capture executed nothing)
        self._refresh(cap, h, params)
        graph.replay()
        torch.cuda.synchronize()

        if _VERIFY and not torch.equal(out, ref):
            d = (out.float() - ref.float()).abs().max().item()
            _log(f"VERIFY FAILED for {sig}: max abs diff {d:.3e}; falling back to eager")
            self.rejected.add(sig)
            return ref

        if len(self.graphs) >= _MAX_GRAPHS:
            self.graphs.pop(next(iter(self.graphs)))
        self.graphs[sig] = cap
        self.n_captures += 1
        _log(f"captured {sig} (verified)" if _VERIFY else f"captured {sig}")
        return out
