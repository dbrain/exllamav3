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
# Measured 42.1 -> 101.5 tok/s on qwen3-0.6b (2.41x).
#
# This is a parallel path: Model.forward dispatches here once, and nothing on
# the eager path changes. Opt in with EXL3_DECODE_GRAPH=1; default is off.
#
# The step is split into SPANS of consecutive modules, alternating between runs
# that are replayed as one graph and runs that stay eager. A module forces an
# eager span when it host-syncs, branches on tensor values, or lives on the CPU:
#
#   - Embedding sets caps["prefer_cpu"] = True unconditionally, so the leading
#     span is always eager and its output reaches the GPU through the pageable
#     H2D copy in Module.prepare_for_device (0.012 ms/token, uncapturable).
#   - PLELayer (Flash-Next has one at ple_layer_ids = [2]) hashes n-grams on the
#     host every step, so it becomes an eager island with a captured graph on
#     either side of it.
#
# Each span hands its output to the next; a captured span copies its input into
# a stable device buffer first, which is also what makes the in-place residual
# in TransformerBlock.forward (`x += y`) safe -- every replay would otherwise
# accumulate into the previous step's buffer.

enabled = os.environ.get("EXL3_DECODE_GRAPH", "0") != "0"

# Number of eager steps observed for a shape signature before capturing it.
# Triton autotune runs host-side benchmarking on first call for a shape and
# must be settled before capture, so this must be >= 1.
_WARMUP_STEPS = int(os.environ.get("EXL3_DECODE_GRAPH_WARMUP", "2"))

# Distinct shape signatures to keep captured at once. The generator rounds the
# block-table width up to a multiple of 16 pages (generator.py, iterate_gen),
# so the signature changes once every PAGE_SIZE*16 = 4096 tokens of context.
_MAX_GRAPHS = int(os.environ.get("EXL3_DECODE_GRAPH_CACHE", "4"))

# After capturing a span, replay it and compare bitwise against an eager run of
# the same span from the same input. A mismatch means something in it was not
# static, so the whole signature is discarded and never retried. Cheap and the
# only thing standing between a subtly dynamic module and silently wrong
# logits. Do not turn this off outside benchmarking.
_VERIFY = os.environ.get("EXL3_DECODE_GRAPH_VERIFY", "1") != "0"

_DEBUG = os.environ.get("EXL3_DECODE_GRAPH_DEBUG", "0") != "0"


def _log(msg: str):
    if _DEBUG:
        print(f" -- decode_graph: {msg}", flush = True)


# Module types whose single-token decode forward is known NOT to be static and
# device-resident, i.e. it host-syncs, branches on tensor values, or produces
# data-dependent shapes. Each one forces an eager span around itself.
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
    # Recurrent modules that genuinely host-sync every decode step:
    "SlidingAttention", # sliding_attn.py _decode_state_prep rebuilds
                        # cache_seqlens from host ints (rs.position -
                        # rs.window_beg) and conditionally page-shifts the state
                        # under a host branch on cache_pos + seqlen > state size
    "PLELayer",         # ple.py hashes n-grams host-side: unconditional
                        # ids.to("cpu") + int(ids.max()) every step
    "ShortConv",        # unconditional recurrent_slots.tolist() + python loop
                        # over slots whenever conv_state exists, i.e. at decode
})

# Recurrent modules (caps["recurrent_cache"]) mutate state tensors in place from
# one step to the next. That by itself is no obstacle -- the KV cache does the
# same and replays correctly -- because advance_recurrent_states() is pure host
# bookkeeping that runs OUTSIDE forward(). Whether a recurrent module can be
# captured is a per-module question about its forward, so they are opted in by
# name rather than as a class:
#
#   GatedDeltaNet   CAPTURABLE -- validated on qwen35-35b-a3b-4bpw over 150
#                   consecutive captured steps. At decode (seqlen == 1) the
#                   chunked branch in gated_delta_rule.py, guarded by
#                   `seqlen >= num_v_heads`, is not taken, so the
#                   recurrent_slots.tolist() host sync it performs never runs;
#                   the fused recurrent kernel reads slots as a device tensor.
#                   Same for the conv: conv1d.py's .tolist() loop is the
#                   no-Triton fallback only.
#
#   Mamba2          Structurally identical (chunked prefill guarded by
#                   `seqlen >= num_v_heads`, fused recurrent decode) and no
#                   decode-path sync found by inspection -- but NOT tested, no
#                   model on hand. Add it here once validated.
#
# NOTE on exactness: ext.cuda_recurrent_gated_delta_rule is not deterministic
# run to run. Two eager runs from the same state snapshot drift in the fp32
# recurrent state (measured 3.7e-09 over 24 steps, 7.5e-09 over 150) and
# eventually produce a different token. Graph replay was measured
# indistinguishable from that control -- same divergence step, same state delta
# -- but it does mean bit-identical decode is unattainable on these models by
# ANY route. See _Span.exempt for how verification handles it.
_CAPTURABLE_RECURRENT_TYPES = frozenset({
    "GatedDeltaNet",
})


def _module_capturable(module) -> bool:
    cap = module.caps.get("graph_capturable")
    if cap is not None:
        return bool(cap)
    name = type(module).__name__
    if name in _UNCAPTURABLE_TYPES:
        return False
    if module.caps.get("recurrent_cache"):
        return name in _CAPTURABLE_RECURRENT_TYPES
    return True


def _recurrent_state_tensors(params) -> list:
    """Every recurrent state tensor this step will advance.

    Unlike the KV cache -- which a repeated step rewrites with the same values
    at the same position -- recurrent state ADVANCES on every call, so the
    multi-pass capture procedure below (eager reference, warmups, verify
    replays, final replay) would over-advance it. These are snapshotted and
    restored so the capturing step advances the state exactly once, like any
    other step. Empty for dense models, where the whole mechanism is free.
    """
    rs = params.get("recurrent_states")
    if not rs:
        return []
    out, seen = [], set()
    for r in rs:
        cache = getattr(r, "cache", None)
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        get_all = getattr(cache, "get_all_recurrent_layers", None)
        if get_all is None:
            continue
        for layer in get_all().values():
            ts = layer.get_state_tensors()
            if isinstance(ts, torch.Tensor):
                ts = (ts,)
            for t in ts or ():
                if isinstance(t, torch.Tensor):
                    out.append(t)
    return out


class _Span:
    """One run of consecutive modules, [start, end), replayed or eager."""

    def __init__(self, captured: bool, start: int, end: int, exempt: bool):
        self.captured = captured
        self.start = start
        self.end = end
        # exempt: this span contains a kernel that is not deterministic run to
        # run AND advances state, so an eager reference re-run is neither
        # reproducible nor idempotent. Verification is skipped for it. This is
        # per SPAN, not per model: a dense span in a hybrid model is still
        # checked bitwise.
        self.exempt = exempt
        self.graph = None
        self.static_in = None
        self.out = None

    def __repr__(self):
        kind = "graph" if self.captured else "eager"
        return f"{kind}[{self.start}:{self.end}]" + ("*" if self.exempt else "")


class _CapturedStep:
    def __init__(self, spans, dev_seqlens, dev_block_table, cache, params):
        self.spans = spans
        self.dev_seqlens = dev_seqlens
        self.dev_block_table = dev_block_table
        # The signature keys on id(cache); holding the reference keeps that id
        # from being recycled by a new cache while this graph is still live
        # (same reason util.tensor.get_for_device keeps its source tensor alive)
        self.cache = cache
        self.params = params


class DecodeGraphs:
    """
    Captures the GPU-resident spans of a single-token decode step and replays
    them, running uncapturable modules eagerly in between and falling back to
    the fully eager path whenever the step is not capturable at all.

    Attached to a Model as model.decode_graphs and driven from Model.forward.

    CALLER CONTRACT -- the logits returned by a replay may alias the last
    captured span's static output buffer, which the NEXT replay overwrites.
    This differs from the eager path, which returns a fresh tensor every step.
    Consume (sample from, or clone) the logits before calling forward() again.
    The generator satisfies this: it argmaxes/samples in the same iteration.
    """

    def __init__(self, model):
        self.model = model
        self.graphs = {}        # signature -> _CapturedStep
        self.seen = {}          # signature -> eager steps observed so far
        self.rejected = set()   # signatures that failed verification
        self.device = None
        self.layout = None      # list of (captured, start, end, exempt)
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

        dev = None
        for module, _, _ in fwd:
            if module.caps.get("prefer_cpu") or module.caps.get("x_cpu"):
                continue
            if module.device is None:
                return False, f"unloaded module: {module.key}"
            if dev is None:
                dev = module.device
            elif module.device != dev:
                # One graph is captured on one device's stream; a layer split
                # across GPUs would need one graph per device plus the
                # transfers between, which is not handled here
                return False, f"layer split across devices ({dev} vs {module.device})"
        if dev is None:
            return False, "no GPU-resident modules"
        self.device = dev

        self.layout = self._build_layout()
        spans = [s for s in self.layout if s[0]]
        if not spans:
            return False, "no capturable span"
        n_mod = sum(e - s for _, s, e, _ in spans)
        desc = " ".join(
            ("G" if c else "e") + f"[{s}:{e}]" + ("*" if x else "")
            for c, s, e, x in self.layout
        )
        return True, (f"{len(spans)} captured span(s), {n_mod}/{len(fwd)} modules: {desc}")

    def _build_layout(self):
        """Split fwd_modules into alternating captured / eager runs."""
        fwd = self.model.fwd_modules
        flags = []
        for module, _, _ in fwd:
            on_dev = (
                module.device == self.device
                and not module.caps.get("prefer_cpu")
                and not module.caps.get("x_cpu")
            )
            flags.append(on_dev and all(_module_capturable(sub) for sub in module))

        layout, i = [], 0
        while i < len(flags):
            j = i
            while j < len(flags) and flags[j] == flags[i]:
                j += 1
            exempt = flags[i] and any(
                type(sub).__name__ in _CAPTURABLE_RECURRENT_TYPES
                for module, _, _ in fwd[i:j] for sub in module
            )
            layout.append((flags[i], i, j, exempt))
            i = j
        return layout

    # ------------------------------------------------------------- signature

    def _signature(self, input_ids, params):
        """
        Everything a replay must hold constant. Contents of the cache-length and
        block-table tensors may change freely between replays; their shapes may
        not, nor may anything that steers a kernel's launch configuration.
        """
        bt = params["block_table"]
        cs = params["cache_seqlens"]
        return (
            tuple(input_ids.shape),
            tuple(bt.shape), bt.dtype,
            tuple(cs.shape), cs.dtype,
            id(params.get("cache")),
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
        return True

    # ----------------------------------------------------------------- entry

    def forward(self, input_ids, params):
        """
        Returns logits for the step, or None to tell the caller to run eager.

        A returned tensor is only valid until the next call (see the class
        docstring): it may be a captured span's output buffer, not a fresh one.
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

    def _to_capture_device(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return x.contiguous()

    def _run_range(self, x, start, end, params):
        for module, instance, _ in self.model.fwd_modules[start:end]:
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

    def _refresh_step(self, cap, params):
        """Copy this step's cache lengths / page table into the graph buffers.

        Async H2D from the caller's pinned staging buffers. Same contract as
        util.tensor.get_for_device: the caller must not refill those buffers
        before the next sync point. The generator satisfies this -- it syncs
        every iteration collecting the sampled token.
        """
        src_cs = params["cache_seqlens"]
        cap.dev_seqlens.copy_(src_cs, non_blocking = src_cs.is_pinned())
        src_bt = params["block_table"]
        cap.dev_block_table.copy_(src_bt, non_blocking = src_bt.is_pinned())

    def _walk(self, cap, input_ids):
        """One decode step: replay captured spans, run eager ones in between."""
        x = input_ids
        for span in cap.spans:
            if span.captured:
                # Rewrite the input buffer every time: TransformerBlock.forward
                # does `x += y`, so the previous replay clobbered it
                span.static_in.copy_(x)
                span.graph.replay()
                x = span.out
            else:
                x = self._run_range(x, span.start, span.end, cap.params)
        return x

    def _replay(self, cap, input_ids, params):
        self._refresh_step(cap, params)
        x = self._walk(cap, input_ids)
        self.n_replays += 1
        return x

    def _capture(self, sig, input_ids, params):
        dev = self.device
        dev_seqlens = params["cache_seqlens"].to(dev).contiguous().clone()
        dev_block_table = params["block_table"].to(dev).contiguous().clone()
        spans = [_Span(c, s, e, x) for c, s, e, x in self.layout]
        cap = _CapturedStep(spans, dev_seqlens, dev_block_table,
                            params.get("cache"), None)
        cap.params = self._bind(params, cap)
        p = cap.params
        self._refresh_step(cap, params)

        rec = _recurrent_state_tensors(params)
        snap = [t.clone() for t in rec]
        def restore():
            for t, s in zip(rec, snap):
                t.copy_(s)

        # 1) Eager pass, recording the input and output of every span we intend
        #    to capture, so each can be captured and checked independently
        ins, outs = {}, {}
        x = input_ids
        for k, span in enumerate(spans):
            if span.captured:
                # The hand-off buffer must live on the capture device. The
                # preceding eager span often ends on the CPU (Embedding is
                # prefer_cpu), and a pageable H2D copy cannot be captured -- it
                # has to happen outside the graph, which is what _walk does.
                x = self._to_capture_device(x)
                ins[k] = x.clone()
                x = self._run_range(x, span.start, span.end, p)
                outs[k] = x.clone()
            else:
                x = self._run_range(x, span.start, span.end, p)
        torch.cuda.synchronize()

        # 2) Capture each span from its recorded input
        for k, span in enumerate(spans):
            if not span.captured:
                continue
            span.static_in = ins[k].clone()
            restore()
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(3):
                    span.static_in.copy_(ins[k])
                    self._run_range(span.static_in, span.start, span.end, p)
            torch.cuda.current_stream().wait_stream(s_)
            torch.cuda.synchronize()

            restore()
            span.static_in.copy_(ins[k])
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    span.out = self._run_range(span.static_in, span.start, span.end, p)
            except Exception as e:
                _log(f"capture failed for {sig} span {span}: {type(e).__name__}: {e}")
                self.rejected.add(sig)
                restore()
                return None
            span.graph = graph

        # 3) Verify each span against its eager output, from the same input
        if _VERIFY:
            for k, span in enumerate(spans):
                if not span.captured:
                    continue
                if span.exempt:
                    _log(f"span {span} verify skipped: contains a kernel that is "
                         f"non-deterministic and advances state, so an eager "
                         f"reference re-run is neither reproducible nor idempotent")
                    continue
                restore()
                span.static_in.copy_(ins[k])
                span.graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(span.out, outs[k]):
                    d = (span.out.float() - outs[k].float()).abs().max().item()
                    _log(f"VERIFY FAILED for {sig} span {span}: max abs diff "
                         f"{d:.3e}; falling back to eager")
                    self.rejected.add(sig)
                    restore()
                    return None

        # 4) Produce this step's real output. Capture executed nothing and the
        #    passes above were all rolled back, so the state advances exactly
        #    once, as it would have on the eager path
        restore()
        out = self._walk(cap, input_ids)

        if len(self.graphs) >= _MAX_GRAPHS:
            self.graphs.pop(next(iter(self.graphs)))
        self.graphs[sig] = cap
        self.n_captures += 1
        _log(f"captured {sig}: {' '.join(str(s) for s in spans)}")
        return out
