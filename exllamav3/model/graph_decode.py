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

# Share one private memory pool across all spans of a step (see _capture).
# Set to 0 only to A/B the aliasing hypothesis; the default is the safe setting.
_SHARED_POOL = os.environ.get("EXL3_DECODE_GRAPH_SHARED_POOL", "1") != "0"

# How many multiples of a span's measured eager-vs-eager spread a captured span
# may differ by and still be accepted (see the exempt branch in _capture).
_NOISE_TOLERANCE = float(os.environ.get("EXL3_DECODE_GRAPH_NOISE_TOL", "4"))

# Eager samples used to estimate that spread. Must be >= 2; 3 costs one extra
# forward at capture time and avoids a lucky identical pair reading as 0 noise.
# Six, not three. The spread between two eager runs of the same span is bimodal
# on 35B-A3B -- measured as exactly 0.0 or exactly 3.477e-01, nothing between --
# so a floor estimated from too few samples lands on 0, collapses the budget to
# the 1e-3 absolute floor, and rejects a sound graph. Measured over 12 trials,
# a 3-sample floor (3 pairs) read 0 in 3 of them; 6 samples gives 15 pairs and
# takes that to ~0.1%.
_NOISE_SAMPLES = max(2, int(os.environ.get("EXL3_DECODE_GRAPH_NOISE_SAMPLES", "6")))


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
# BlockSparseMLP: the dense per-expert fallback host-syncs on torch.bincount and
# data-dependent index_select/index_add_ shapes. It opts back in from its own
# file when the grouped Triton mgemm decode path is active, which has none of
# that; this entry remains the correct default for every other configuration.
_UNCAPTURABLE_TYPES = frozenset({
    "BlockSparseMLP",
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


# QSA (Flash-Next full-attention layers) is capturable in ONE of its two
# regimes, so it is decided per step rather than once per model.
#
# Attention.forward picks the regime with
#     qsa_sparse = int(cache_seqlens.max()) + seqlen > indexer.sparse_threshold()
# which is host control flow, not a device sync (the seqlens are read from the
# caller's pinned CPU staging buffer). A graph bakes in whichever branch was
# live at capture, so the regime joins the capture signature and the crossing
# recaptures -- exactly like the block-table width boundary.
#
#   DENSE regime (position < sparse_threshold, 2051 tokens on Flash-Next):
#     CAPTURABLE. Only QSAIndexer.update_planes runs, and it is fully device
#     resident -- it reads cache_seqlens as a DEVICE tensor, its grids are
#     functions of (bsz, seqlen) only, and its workspaces come from
#     g_tensor_cache.get_bucketed at a constant numel, so the same buffer comes
#     back every call. Its in-place writes to layer.raw_k / layer.pooled are
#     idempotent at a fixed position, like the KV cache. Attention then takes
#     the ordinary paged-decode path.
#
#   SPARSE regime (position >= sparse_threshold): NOT CAPTURABLE. The extra
#     work is QSAIndexer.select_indices_paged, which does
#         pos0 = int(cache_seqlens_cpu[b])
#     and then feeds that per-step position into _select_rows as a KERNEL
#     ARGUMENT (dsa_indexer_scores, _dsa_pool_expand_kernel), as a topk bound
#     (ext.dsa_topk with min(block_topk, T_slab)), and as a workspace SHAPE
#     (s_stride = cdiv(T_slab, 128) * 128). pos0 advances every token, so no
#     signature can cover it -- it would need a fresh graph per position.
#     Making this capturable is a kernel-signature change (read the position
#     from the device tensor, allocate the scores workspace at an upper bound,
#     move the topk bound device-side), not something the capture layer can fix.
_QSA_TYPES = frozenset({"QSAIndexer"})


def _module_capturable(module, qsa_dense: bool = False) -> bool:
    cap = module.caps.get("graph_capturable")
    if cap is not None:
        return bool(cap)
    name = type(module).__name__
    if name in _QSA_TYPES:
        return qsa_dense
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
    def __init__(self, spans, dev_seqlens, host_seqlens, dev_block_table, cache, params):
        self.spans = spans
        self.dev_seqlens = dev_seqlens
        # Pinned host mirror of the cache lengths. Attention's QSA branch reads
        # get_for_device(params, "cache_seqlens", "cpu"); binding only a device
        # tensor would turn that into a D2H copy, which is a sync and cannot be
        # captured. Seeded into params["dev_cache"] by _bind.
        self.host_seqlens = host_seqlens
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
        self.layouts = {}       # qsa_dense -> list of (captured, start, end, exempt)
        self.layout = None      # the most recently used layout, for debugging
        self.qsa_threshold = None
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

        self.qsa_threshold = self._find_qsa_threshold()
        for regime in ((True, False) if self.qsa_threshold is not None else (False,)):
            self.layouts[regime] = self._build_layout(regime)
        self.layout = self.layouts[max(self.layouts)]
        if not any(any(sp[0] for sp in lay) for lay in self.layouts.values()):
            return False, "no capturable span"

        def _desc(lay):
            spans = [sp for sp in lay if sp[0]]
            n_mod = sum(e - st for _, st, e, _ in spans)
            body = " ".join(("G" if c else "e") + f"[{st}:{e}]" + ("*" if x else "")
                            for c, st, e, x in lay)
            return f"{len(spans)} span(s), {n_mod}/{len(fwd)} modules: {body}"

        if self.qsa_threshold is None:
            return True, _desc(self.layouts[False])
        return True, (f"QSA threshold {self.qsa_threshold}; "
                      f"dense regime: {_desc(self.layouts[True])} | "
                      f"sparse regime: {_desc(self.layouts[False])}")


    def _find_qsa_threshold(self):
        """Position at which every QSA layer flips to the uncapturable sparse
        regime, or None when the model has no indexer."""
        for module, _, _ in self.model.fwd_modules:
            for sub in module:
                idx = getattr(sub, "qsa_indexer", None)
                if idx is not None:
                    return int(idx.sparse_threshold())
                if type(sub).__name__ in _QSA_TYPES:
                    return int(sub.sparse_threshold())
        return None


    def _qsa_regime(self, params) -> bool:
        """True when this step is below the QSA sparse threshold, i.e. the QSA
        layers can be captured. Mirrors the test in Attention.forward.

        Read from the caller's staging buffer, which is pinned host memory on
        the generator path, so this costs no device sync. If it ever arrives on
        the device, reading it would sync, so treat the step as sparse instead.
        """
        if self.qsa_threshold is None:
            return False
        cs = params.get("cache_seqlens")
        if cs is None or cs.device.type != "cpu":
            return False
        seqlen = 1
        return int(cs.max()) + seqlen <= self.qsa_threshold

    def _build_layout(self, qsa_dense: bool):
        """Split fwd_modules into alternating captured / eager runs."""
        fwd = self.model.fwd_modules
        flags = []
        for module, _, _ in fwd:
            on_dev = (
                module.device == self.device
                and not module.caps.get("prefer_cpu")
                and not module.caps.get("x_cpu")
            )
            flags.append(on_dev and all(_module_capturable(sub, qsa_dense) for sub in module))

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
            # QSA regime: a graph bakes in the branch that was live at capture,
            # so crossing the sparse threshold must mint a new graph
            self._qsa_regime(params),
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
        # Seed the host mirror so Attention's QSA branch, which asks for
        # get_for_device(params, "cache_seqlens", "cpu"), gets a pinned CPU
        # tensor instead of a D2H copy off the bound device buffer (that copy
        # is a sync and would fail the capture). Keyed exactly as
        # get_for_device does: (id(source), device-as-passed).
        p["dev_cache"] = {
            (id(cap.dev_seqlens), "cpu"): (cap.dev_seqlens, cap.host_seqlens),
        }
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
        # Host-side mirror: a plain host memcpy, no stream involved
        cap.host_seqlens.copy_(src_cs)
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
        # All spans of a step share ONE private pool. With a pool per span, each
        # span's captured kernels hold addresses in a pool the OTHER spans'
        # captures (and the eager islands between them) can allocate from, so a
        # replay can read memory something else wrote since. A single shared
        # pool keeps the whole step's captured allocations in one arena that the
        # normal allocator never hands out.
        pool = torch.cuda.graph_pool_handle() if _SHARED_POOL else None
        regime = self._qsa_regime(params)
        layout = self.layouts.get(regime, self.layouts.get(False))
        self.layout = layout
        spans = [_Span(c, s, e, x) for c, s, e, x in layout]
        host_seqlens = params["cache_seqlens"].detach().to("cpu").clone().pin_memory()
        cap = _CapturedStep(spans, dev_seqlens, host_seqlens, dev_block_table,
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
                ctx = (torch.cuda.graph(graph, pool = pool) if pool is not None
                       else torch.cuda.graph(graph))
                with ctx:
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
                    # This span holds kernels that are not deterministic run to
                    # run, so neither "graph == eager" nor "graph == graph"
                    # holds even when the capture is perfectly correct. Measured
                    # on qwen35-35b-a3b with capture OFF: repeating one decode
                    # step from byte-identical state gave bitwise-identical
                    # logits only 2 times in 7, and 8 full greedy decodes
                    # produced 3 distinct token sequences.
                    #
                    # So calibrate instead of assuming: run the span eagerly
                    # _NOISE_SAMPLES times from the same restored state to
                    # measure this span's own noise floor, then accept the graph
                    # only if it lands inside it. Self-calibrating, and it
                    # neither rubber-stamps (the old skip) nor refuses every
                    # hybrid (a bitwise gate would). Three samples, not two: the
                    # noise is intermittent (a pair can measure exactly 0 while
                    # other pairs on the same span measure 2e-2), so one pair
                    # under-estimates the floor and false-rejects a sound graph.
                    refs = []
                    for _ in range(_NOISE_SAMPLES):
                        restore(); span.static_in.copy_(ins[k])
                        r = self._run_range(span.static_in, span.start, span.end, p)
                        torch.cuda.synchronize()
                        refs.append(r.float().clone())
                    noise = max(
                        (refs[a] - refs[b]).abs().max().item()
                        for a in range(len(refs)) for b in range(a + 1, len(refs))
                    )
                    r1 = refs[0]

                    restore(); span.static_in.copy_(ins[k])
                    span.graph.replay(); torch.cuda.synchronize()
                    delta = (span.out.float() - r1).abs().max().item()

                    # Allow a few multiples of the observed spread, plus a small
                    # absolute floor so a span that happens to measure zero
                    # noise on one pair is not held to exact equality.
                    budget = max(noise * _NOISE_TOLERANCE, 1e-3)
                    if delta > budget:
                        _log(f"VERIFY FAILED for {sig} span {span}: graph differs "
                             f"from eager by {delta:.3e}, outside this span's "
                             f"measured noise floor {noise:.3e} (budget "
                             f"{budget:.3e}); falling back to eager")
                        self.rejected.add(sig)
                        restore()
                        return None
                    _log(f"span {span} verified against its own noise floor: "
                         f"delta {delta:.3e} <= budget {budget:.3e} "
                         f"(eager-vs-eager spread {noise:.3e})")
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
