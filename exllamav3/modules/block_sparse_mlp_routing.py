"""
Expert routing for BlockSparseMLP: the RoutingCFG bundle each layer builds at load time and
the per-arch top-k functions the module dispatches to (router_type). Multi-row calls take
bucketed workspaces from the per-device static cache up to ROUTING_CACHE_ROWS rows; the bsz-1
buffers live in the RoutingCFG (CUDA-graph paths bake their addresses).
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from ..util.device_copy import to_device
from ..ext import exllamav3_ext as ext
from ..util.tensor import g_tensor_cache
from ..util.row_inv import ROW_INV, ROW_INV_MAX_ROWS


def _pad_rows(y):
    """router logits GEMM input at a row pitch that is not a multiple of 2 KiB (same values; a no-op when not a hazard)."""
    if y.dim() == 2:
        from .quant.exl3 import pad_copy_in
        return pad_copy_in(y, y.shape[1])
    return y

from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX

ROUTING_CACHE_ROWS = 128


def _routing_buffers(cfg, bsz, device):
    """Router outputs for a multi-row call: (router_logits (bsz, E) half, selected_experts
    (bsz, K) long, routing_weights (bsz, K) half). Decode/MTP-class row counts (verify windows,
    small batches: up to ROUTING_CACHE_ROWS) take bucketed workspaces from the per-device static
    cache, shared by every MoE layer on the device: a layer's routing outputs are consumed by
    its own expert compute (and offload/broadcast hand-offs) before the next layer routes on the
    same stream, so one set per device suffices and nearby row counts share a backing instead
    of one static per shape. Prefill-sized calls allocate per call (not CPU-bound, and the
    static cache is meant to hold only small buffers)."""
    E, K = cfg.num_experts, cfg.num_experts_per_tok
    if bsz <= ROUTING_CACHE_ROWS:
        ws = lambda numel, dtype, tag: g_tensor_cache.get_bucketed(device, numel, dtype, tag)
    else:
        ws = lambda numel, dtype, tag: torch.empty((numel,), dtype = dtype, device = device)
    return (
        ws(bsz * E, torch.half, "moe_route_logits").view(bsz, E),
        ws(bsz * K, torch.long, "moe_route_sel").view(bsz, K),
        ws(bsz * K, torch.half, "moe_route_w").view(bsz, K),
    )

# Score activations for the nogroup routing kernels (must match routing.cu)
ROUTING_ACT_SIGMOID = 0
ROUTING_ACT_SQRTSP = 1

def _esb_h(cfg):
    """fp16 selection-bias copy for the CUDA top-k kernels, built lazily (load may be
    deferred when the RoutingCFG is constructed). Mean-centered when the source is wider than
    fp16: selection is invariant to a constant shift, and centering keeps fp16 rounding well
    below the inter-expert score gaps even when the bias values are large (GLM-5.2 ~34.0)."""
    if cfg.e_score_bias_h is None and cfg.e_score_correction_bias is not None:
        esb = cfg.e_score_correction_bias
        cfg.e_score_bias_h = esb if esb.dtype == torch.half else (esb - esb.mean()).half()
    return cfg.e_score_bias_h


@dataclass
class RoutingCFG:
    gate_tensor: torch.Tensor
    gate_tensor_t: torch.Tensor | None
    num_experts: int
    num_experts_per_tok: int
    router_logits_bsz1: torch.Tensor
    routing_weights_bsz1: torch.Tensor
    selected_experts_bsz1: torch.Tensor
    e_score_correction_bias: torch.Tensor | None
    e_score_bias_h: torch.Tensor | None   # lazy, see _esb_h
    routed_scaling_factor: float | None
    n_group: int | None
    topk_group: int | None
    per_expert_scale: torch.Tensor | None
    router_bias: torch.Tensor | None = None
    tid2eid: torch.Tensor | None = None
    e_score_bias_vl: torch.Tensor | None = None   # DeepSeek-V4 vision: selection bias for image rows
    gate_i8: torch.Tensor | None = None     # (2, E, K) int8 hi/lo slices, lazy (see _gate_i8)
    gate_sb: torch.Tensor | None = None     # (E) fp32 row scales
    gate_tensor_f32: torch.Tensor | None = None       # grouped DS3 router: fp32 gate for multi-row logits, lazy
    e_score_bias_f32: torch.Tensor | None = None      # grouped DS3 router: fp32 selection bias, lazy
    router_logits_f32_bsz1: torch.Tensor | None = None   # grouped DS3 router: single-row fp32 logits, lazy
    router_logits_gfx12: torch.Tensor | None = None
    routing_weights_gfx12: torch.Tensor | None = None
    selected_experts_gfx12: torch.Tensor | None = None


_HIP_ROUTER_HIDDEN = 2560
_HIP_ROUTER_EXPERTS = 512
_HIP_ROUTER_TOP_K = 10
_HIP_ROUTER_MAX_ROWS = 16


def _hip_grouped_max_rows() -> int:
    try:
        return max(1, min(16, int(os.environ.get("EXL3_HIP_GROUPED_MAX_ROWS", "16"))))
    except ValueError:
        return 16


_HIP_GROUPED_MAX_ROWS = _hip_grouped_max_rows()
# The native prefill binding follows the grouped cap boundary and accepts rows 2..512.
# Crossover between the per-(row,expert) decode kernel and the expert-GROUPED
# prefill kernel. The prefill kernel sorts assignments by expert and amortizes
# each expert's weight read over up to 16 rows, which is exactly what a
# multi-row MTP verification batch wants: at 3 rows, 31% of the decode path's
# expert reads are duplicates. Lowering this routes small speculative batches
# through the deduplicating kernel. Default 17 keeps the original split.
_HIP_PREFILL_MIN_ROWS = max(
    2, int(os.environ.get("EXL3_HIP_PREFILL_MIN_ROWS", _HIP_GROUPED_MAX_ROWS + 1)))
_HIP_PREFILL_MAX_ROWS = 2048
_HIP_PREFILL_MAX_EXPERT_ROWS = _HIP_PREFILL_MAX_ROWS * _HIP_ROUTER_TOP_K


# EXL3_MOE_VALU=1: the VALU decode matvec (K2/K4 experts) serves every row count 1..8 through the grouped/dedup
# launches, one reduction order for all of them: plain decode (R=1) and verify rounds (R>1) give identical bits. It
# therefore bypasses the batch-1 fused kernel, the union path and the prefill-kernel yield below. Default off.
_MOE_VALU = os.environ.get("EXL3_MOE_VALU", "0") != "0"
_MOE_VALU_MAX_ROWS = 8


def _hip_grouped_rows_eligible(rows: int, prefill_available: bool = True) -> bool:
    if _MOE_VALU and rows <= _MOE_VALU_MAX_ROWS:
        return 1 <= rows <= min(_HIP_GROUPED_MAX_ROWS, _MOE_VALU_MAX_ROWS)
    # Yield to the expert-grouped prefill kernel once it claims this row count,
    # otherwise the per-(row,expert) decode kernel would take it first and
    # re-read duplicate expert weights. When no prefill kernel serves this model
    # (MiMo-V2: the prefill path is Qwen3.8-only), the grouped decode kernel is
    # the only grouped route and takes every row count it supports.
    cap = _HIP_GROUPED_MAX_ROWS if not prefill_available \
        else min(_HIP_GROUPED_MAX_ROWS, _HIP_PREFILL_MIN_ROWS - 1)
    return 1 <= rows <= cap


def _hip_prefill_rows_eligible(rows: int) -> bool:
    return _HIP_PREFILL_MIN_ROWS <= rows <= _HIP_PREFILL_MAX_ROWS


def _hip_router_device_supported(device):
    if not torch.version.hip or not hasattr(ext, "routing_std_gfx12_bsz1"):
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
    # routing_std_gfx12 uses only wave-width primitives (__shfl_down/__syncwarp),
    # no RDNA4-only instruction, so any wave32 gfx11.5/gfx12 part qualifies.
    return arch in ("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152") and getattr(props, "warp_size", 0) == 32


def _hip_router_config_supported(cfg):
    gate = cfg.gate_tensor
    return (
        cfg.num_experts == _HIP_ROUTER_EXPERTS and
        cfg.num_experts_per_tok == _HIP_ROUTER_TOP_K and
        cfg.per_expert_scale is None and
        cfg.router_bias is None and
        gate.dtype == torch.half and gate.is_contiguous() and
        tuple(gate.shape) == (_HIP_ROUTER_HIDDEN, _HIP_ROUTER_EXPERTS) and
        _hip_router_device_supported(gate.device)
    )


def _hip_router_call_supported(bsz, cfg, y, params):
    return (
        os.environ.get("EXL3_HIP_ROUTER", "1") != "0" and
        1 <= bsz <= _HIP_ROUTER_MAX_ROWS and not params.get("activate_all_experts") and
        _hip_router_config_supported(cfg) and
        y.dtype == torch.half and y.is_contiguous() and
        tuple(y.shape) == (bsz, _HIP_ROUTER_HIDDEN)
    )


def _prepare_hip_router_gate_t(cfg):
    if (
        os.environ.get("EXL3_HIP_ROUTER", "1") != "0" and
        _hip_router_config_supported(cfg) and cfg.gate_tensor_t is None
    ):
        # One persistent E x H allocation per router (~2.5 MiB at 512 x 2560), created
        # with the layer on its assigned device and released with routing_cfg on unload.
        cfg.gate_tensor_t = cfg.gate_tensor.T.contiguous()


def _prepare_hip_router_buffers(cfg):
    if _hip_router_config_supported(cfg) and getattr(cfg, "router_logits_gfx12", None) is None:
        # RoutingCFG owns one persistent multirow workspace. Loaded modules already share
        # non-reentrant workspaces, so Generator batches are stream-ordered rather than callers
        # needing separate buffers for speculative streams.
        device = cfg.gate_tensor.device
        cfg.router_logits_gfx12 = torch.empty(
            (_HIP_ROUTER_MAX_ROWS, _HIP_ROUTER_EXPERTS), device = device, dtype = torch.half)
        cfg.selected_experts_gfx12 = torch.empty(
            (_HIP_ROUTER_MAX_ROWS, _HIP_ROUTER_TOP_K), device = device, dtype = torch.long)
        cfg.routing_weights_gfx12 = torch.empty(
            (_HIP_ROUTER_MAX_ROWS, _HIP_ROUTER_TOP_K), device = device, dtype = torch.half)


def _hip_router_buffers(cfg, rows):
    if rows == 1:
        return cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    _prepare_hip_router_buffers(cfg)
    return (
        cfg.router_logits_gfx12[:rows],
        cfg.selected_experts_gfx12[:rows],
        cfg.routing_weights_gfx12[:rows],
    )

def _routing_std_torch(cfg, y, params, include_bias = False):
    router_logits = torch.matmul(_pad_rows(y), cfg.gate_tensor)
    # EXL3_ROUTER_TOPK (default 1, read per call): prefill rows select top-k with one Triton pass (router_topk.py)
    # instead of the stable sort below over all experts; same ids, same values into the same softmax
    if (router_logits.dim() == 2 and router_logits.shape[0] > 16 and router_logits.is_cuda and router_logits.stride(1) == 1
            and not (include_bias and cfg.router_bias is not None) and not params.get("activate_all_experts")
            and cfg.num_experts_per_tok <= 32 and os.environ.get("EXL3_ROUTER_TOPK", "1") != "0"
            and not torch.cuda.is_current_stream_capturing()):
        from .router_topk import router_topk
        top_v, selected_experts = router_topk(router_logits, cfg.num_experts_per_tok)
        routing_weights = torch.softmax(top_v, dim = -1)
        if cfg.per_expert_scale is not None:
            routing_weights = routing_weights * cfg.per_expert_scale.float()[selected_experts]
        return selected_experts, routing_weights.half()
    router_logits_f = router_logits.float()
    if include_bias and cfg.router_bias is not None:
        router_logits_f = router_logits_f + cfg.router_bias.float()
    if params.get("activate_all_experts"):
        selected_experts = torch.arange(
            cfg.num_experts, dtype = torch.long, device = y.device
        ).expand(y.shape[0], -1)
        routing_weights = torch.softmax(router_logits_f, dim = -1)
    else:
        # torch.topk breaks ties between equal logits in an arbitrary, run-dependent order (and may pick a different
        # expert at the k-th place); fp16 router logits tie often. A stable descending sort keeps the lowest expert
        # index first among equals, so identical inputs give identical routing. Same values and softmax otherwise.
        top_v, selected_experts = torch.sort(router_logits_f, dim = -1, descending = True, stable = True)
        top_v = top_v[:, :cfg.num_experts_per_tok]
        selected_experts = selected_experts[:, :cfg.num_experts_per_tok].contiguous()
        routing_weights = torch.softmax(top_v, dim = -1)
    if cfg.per_expert_scale is not None:
        routing_weights = routing_weights * cfg.per_expert_scale.float()[selected_experts]
    return selected_experts, routing_weights.half()


def _routing_nogroup_torch(cfg, y, params, scores):
    if params.get("activate_all_experts"):
        selected_experts = torch.arange(
            cfg.num_experts, dtype = torch.long, device = y.device
        ).expand(y.shape[0], -1)
    else:
        selection_scores = scores
        if cfg.e_score_correction_bias is not None:
            selection_scores = selection_scores + cfg.e_score_correction_bias.float().unsqueeze(0)
        # torch.topk breaks ties between equal scores in an arbitrary, run-dependent way (both the order and,
        # at the k-th place, which expert is chosen). Router scores tie often, so identical inputs gave
        # different expert sets from run to run. A stable descending sort keeps the lowest expert index first
        # among equals: same values, reproducible routing.
        if (selection_scores.dim() == 2 and selection_scores.shape[0] > 16 and selection_scores.is_cuda
                and selection_scores.stride(1) == 1 and cfg.num_experts_per_tok <= 32
                and os.environ.get("EXL3_ROUTER_TOPK", "1") != "0" and not torch.cuda.is_current_stream_capturing()):
            # EXL3_ROUTER_TOPK: same ids as the stable sort (router_topk.py)
            from .router_topk import router_topk
            selected_experts = router_topk(selection_scores, cfg.num_experts_per_tok)[1]
        else:
            selected_experts = torch.sort(selection_scores, dim = -1, descending = True, stable = True).indices[
                :, :cfg.num_experts_per_tok].contiguous()
    routing_weights = scores.gather(1, selected_experts)
    routing_weights = routing_weights / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
    routing_weights = routing_weights * cfg.routed_scaling_factor
    return selected_experts, routing_weights.half()


def _gate_t(cfg):
    """Transposed (E, K) half gate for the single-row GEMV, plus the int8 hi/lo slices and row
    scales for the deterministic multi-row projection (ext.routing_gemm_det), both built lazily
    (weights may be deferred when the RoutingCFG is constructed). Builds without the deterministic
    projection leave the int8 slices unset and route on the half gate."""
    if cfg.gate_tensor_t is None:
        cfg.gate_tensor_t = cfg.gate_tensor.T.contiguous()
    if getattr(ext, "HAS_DET_GEMM", False) and cfg.gate_i8 is None and cfg.gate_tensor_t.dtype == torch.half:
        E, K = cfg.gate_tensor_t.shape
        cfg.gate_i8 = torch.empty((2, E, K), dtype = torch.int8, device = cfg.gate_tensor_t.device)
        cfg.gate_sb = torch.empty((E,), dtype = torch.float, device = cfg.gate_tensor_t.device)
        ext.det_quant_weight(cfg.gate_tensor_t, cfg.gate_i8, cfg.gate_sb)
    return cfg.gate_tensor_t


def routing_std(bsz, cfg, y, params):
    if _hip_router_call_supported(bsz, cfg, y, params):
        _prepare_hip_router_gate_t(cfg)
        scores, selected_experts, routing_weights = _hip_router_buffers(cfg, bsz)
        ext.routing_std_gfx12_bsz1(
            y,
            cfg.gate_tensor_t,
            scores,
            selected_experts,
            routing_weights,
        )
        return selected_experts, routing_weights
    if not hasattr(ext, "routing_std"):
        return _routing_std_torch(cfg, y, params)
    if bsz == 1:
        _gate_t(cfg)
        ext.routing_std(
            y,
            cfg.gate_tensor,
            cfg.router_logits_bsz1,
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
            cfg.per_expert_scale,
            cfg.gate_tensor_t,
            None,
            cfg.gate_i8,
            cfg.gate_sb,
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            router_logits = torch.matmul(_pad_rows(y), cfg.gate_tensor)
            routing_weights = torch.softmax(router_logits, dim = -1)
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
            if cfg.per_expert_scale is not None:
                routing_weights *= cfg.per_expert_scale.unsqueeze(0)
            return selected_experts, routing_weights
        else:
            router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
            _gate_t(cfg)
            ext.routing_std(
                y,
                cfg.gate_tensor,
                router_logits,
                selected_experts,
                routing_weights,
                cfg.per_expert_scale,
                cfg.gate_tensor_t,
                None,
                cfg.gate_i8,
                cfg.gate_sb,
            )
        return selected_experts, routing_weights


def routing_std_bias(bsz, cfg, y, params):
    """Standard softmax routing with a bias on the router logits (gpt-oss): the bias enters
    before top-k selection, and the weights are the softmax over the selected biased logits
    (equivalent to renormalizing the full biased softmax over the top-k set)."""
    if not hasattr(ext, "routing_std"):
        return _routing_std_torch(cfg, y, params, include_bias = True)
    if params.get("activate_all_experts"):
        if cfg.router_bias is not None:
            router_logits = torch.addmm(cfg.router_bias, y, cfg.gate_tensor)
        else:
            router_logits = torch.matmul(y, cfg.gate_tensor)
        routing_weights = torch.softmax(router_logits.float(), dim = -1).half()
        selected_experts = (
            torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
            .repeat((bsz, 1))
        )
        return selected_experts, routing_weights
    # Every batch size on the deterministic ext path (bias before top-k inside the kernel)
    _gate_t(cfg)
    if bsz == 1:
        router_logits, selected_experts, routing_weights = \
            cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_std(
        y,
        cfg.gate_tensor,
        router_logits,
        selected_experts,
        routing_weights,
        cfg.per_expert_scale,
        cfg.gate_tensor_t,
        cfg.router_bias,
        cfg.gate_i8,
        cfg.gate_sb,
    )
    return selected_experts, routing_weights


# TODO: Optimize top_k groups (for DS3)
def routing_ds3(bsz, cfg, y, params):
    """Group-limited DS3 routing (n_group > 1): fp32 logits (the reference gate's precision;
    with hundreds of experts the selection margins sit below fp16 resolution), group scores
    from each group's two best biased scores, the topk_group best groups eligible, top-K over
    those, weights from the unbiased scores. Single rows project with the fixed-order fp32
    GEMV, other row counts with an fp32 matmul; selection and weights run in one kernel
    (routing_ds3_group). The torch composition stays for the all-experts calibration pass."""
    if params.get("activate_all_experts") or not hasattr(ext, "routing_ds3_group") or \
            not hasattr(ext, "routing_gemv_f32"):
        return _routing_ds3_torch(bsz, cfg, y, params)
    E, K = cfg.num_experts, cfg.num_experts_per_tok
    if cfg.e_score_bias_f32 is None and cfg.e_score_correction_bias is not None:
        cfg.e_score_bias_f32 = cfg.e_score_correction_bias.float().contiguous()
    if bsz == 1:
        _gate_t(cfg)
        if cfg.router_logits_f32_bsz1 is None:
            cfg.router_logits_f32_bsz1 = torch.empty((1, E), dtype = torch.float, device = y.device)
        logits = cfg.router_logits_f32_bsz1
        ext.routing_gemv_f32(y.view(-1), cfg.gate_tensor_t, logits.view(-1))
        selected_experts, routing_weights = cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        if cfg.gate_tensor_f32 is None:
            cfg.gate_tensor_f32 = cfg.gate_tensor.float().contiguous()
        logits = torch.matmul(y.view(-1, y.shape[-1]).float(), cfg.gate_tensor_f32)
        _, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_ds3_group(
        logits, cfg.e_score_bias_f32, selected_experts, routing_weights,
        cfg.routed_scaling_factor, ROUTING_ACT_SIGMOID, cfg.n_group, cfg.topk_group,
    )
    return selected_experts, routing_weights


def _routing_ds3_torch(bsz, cfg, y, params):
    """Torch composition of the group-limited DS3 router (the all-experts calibration pass,
    and the reference the kernel is tested against)"""
    activate_all_experts = params.get("activate_all_experts")
    router_logits = torch.matmul(y.float(), cfg.gate_tensor.float())

    scores = router_logits.sigmoid()
    scores_for_choice = scores.view(-1, cfg.num_experts)
    if cfg.e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + cfg.e_score_correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .topk(2, dim = -1)[0]
        .sum(dim = -1)
    )
    group_idx = torch.topk(group_scores, k = cfg.topk_group, dim = -1, sorted = False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .reshape(-1, cfg.num_experts)
    )
    # -inf, not 0: the selection bias can push eligible scores below zero
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    topk_indices = torch.topk(
        scores_for_choice,
        k = cfg.num_experts if activate_all_experts else cfg.num_experts_per_tok,
        dim = -1,
        sorted = False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    denominator = topk_weights.sum(dim = -1, keepdim = True) + 1e-20
    topk_weights /= denominator
    topk_weights = topk_weights * cfg.routed_scaling_factor
    # half weights, as every other router returns (TP broadcast buffers and the decode kernels)
    return topk_indices, topk_weights.half()


DEC_ROUTER_ROWS_MAX = 4  # loop ~23 us/row vs torch fallback ~120 us flat: loses past R~5
ROUTER_ROWS_KERNEL_MAX = 4  # exl3_dec_router_rows takes NR in 2..4 (exl3_dec.cu)


def _router_row_chunks(bsz: int) -> list:
    """Balanced chunk sizes, each 2..ROUTER_ROWS_KERNEL_MAX, summing to bsz (bsz >= 2)."""
    nch = -(-bsz // ROUTER_ROWS_KERNEL_MAX)
    out, left = [], bsz
    for c in range(nch):
        k = -(-left // (nch - c))
        out.append(k)
        left -= k
    return out


def _router_rows_max() -> int:
    # Row invariance: the torch fallback picks from another score arithmetic than the batch-1 kernel,
    # so every verify row count up to the engine maximum stays on the batch-1 arithmetic
    return ROW_INV_MAX_ROWS if ROW_INV["on"] else DEC_ROUTER_ROWS_MAX


def _dec_router_ok(bsz, cfg, y, params) -> bool:
    return (
        bsz == 1 and hasattr(ext, "exl3_dec_router") and os.environ.get("EXL3_DEC_ROUTER", "1") != "0" and
        not params.get("activate_all_experts") and cfg.tid2eid is None and cfg.router_bias is None and
        cfg.per_expert_scale is None and (cfg.n_group or 1) == 1 and
        y.dtype == torch.half and y.is_contiguous() and y.numel() == y.shape[-1] and
        cfg.gate_tensor.dtype == torch.half and cfg.gate_tensor.is_contiguous() and
        cfg.gate_tensor.dim() == 2 and cfg.gate_tensor.shape[0] == y.shape[-1] and
        y.shape[-1] % 256 == 0 and cfg.num_experts <= 1024 and cfg.num_experts % 8 == 0 and
        cfg.num_experts_per_tok <= 64  # mirrors the kernel TORCH_CHECK (E % 8: 16-byte gate loads; GLM-5.3 E = 288 fits)
    )


def _dec_router(cfg, y):
    from .quant.exl3 import dec_workspace
    bias = getattr(cfg, "dec_bias", False)
    if bias is False:
        esb = cfg.e_score_correction_bias
        bias = esb.float().contiguous() if esb is not None else None
        cfg.dec_bias = bias
    scratch, counters = dec_workspace(y.device)
    scale = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    ext.exl3_dec_router(y.view(1, -1), cfg.gate_tensor, bias, cfg.selected_experts_bsz1,
                        cfg.routing_weights_bsz1, scratch, counters, float(scale))
    return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1


def _dec_router_rows(cfg, y, bsz):
    """DFlash/MTP verify (R = 2..MAX_BSZN rows): the batch-1 exl3_dec_router once per row into
    row slices of the shared routing workspace. Bit-identical routing to plain decode (the
    torch fallback picks from a different score arithmetic, so a near-tie pick can flip and the
    verified ids drift from plain greedy), 47 us vs 120 us at R = 2, and capture-safe."""
    from .quant.exl3 import dec_workspace
    bias = getattr(cfg, "dec_bias", False)
    if bias is False:
        esb = cfg.e_score_correction_bias
        bias = esb.float().contiguous() if esb is not None else None
        cfg.dec_bias = bias
    scratch, counters = dec_workspace(y.device)
    scale = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    _, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    y2 = y.view(bsz, -1)
    # EXL3_DEC_ROUTER_ROWS=2 (default off, read per call): all rows in one launch, G read once,
    # bitwise to the per-row loop below
    if (os.environ.get("EXL3_DEC_ROUTER_ROWS", "1") == "2" and hasattr(ext, "exl3_dec_router_rows") and
            cfg.num_experts <= 320):
        # The kernel takes 2..4 rows: more rows go as balanced chunks of 2..4 (5 -> 3+2, 8 -> 4+4),
        # every row is still the batch-1 arithmetic
        r0 = 0
        for k in _router_row_chunks(bsz):
            ext.exl3_dec_router_rows(y2[r0:r0 + k], cfg.gate_tensor, bias, selected_experts[r0:r0 + k],
                                     routing_weights[r0:r0 + k], scratch, counters, float(scale))
            r0 += k
        return selected_experts, routing_weights
    for r in range(bsz):
        ext.exl3_dec_router(y2[r:r + 1], cfg.gate_tensor, bias, selected_experts[r],
                            routing_weights[r], scratch, counters, float(scale))
    return selected_experts, routing_weights


def _dec_router_rows_ok(bsz, cfg, y, params) -> bool:
    return (
        1 < bsz <= _router_rows_max() and params.get("dflash_verify") and
        os.environ.get("EXL3_DEC_ROUTER_ROWS", "1") != "0" and
        y.is_contiguous() and y.numel() == bsz * y.shape[-1] and
        _dec_router_ok(1, cfg, y[:1] if y.dim() == 2 else y.view(bsz, -1)[:1], params)
    )


def routing_dots(bsz, cfg, y, params):

    if _dec_router_ok(bsz, cfg, y, params):
        return _dec_router(cfg, y)
    if _dec_router_rows_ok(bsz, cfg, y, params):
        return _dec_router_rows(cfg, y, bsz)

    if not hasattr(ext, "routing_ds3_nogroup"):
        if os.environ.get("EXL3_PF_GLUE", "0") == "1" and bsz > 64:
            # (E, K) contiguous weight read as its transpose: hipBLAS picks the NT kernel, ~2x faster
            # cold at E=288 prefill, bit-exact (scratch/pfbig1/mb_dense.py)
            if cfg.gate_tensor_t is None:
                cfg.gate_tensor_t = cfg.gate_tensor.T.contiguous()
            logits = torch.matmul(_pad_rows(y), cfg.gate_tensor_t.T)
        else:
            logits = torch.matmul(_pad_rows(y), cfg.gate_tensor)
        scores = torch.sigmoid(logits.float())
        if params.get("activate_all_experts"):
            routing_weights = scores
            if cfg.e_score_correction_bias is not None:
                routing_weights = routing_weights + cfg.e_score_correction_bias.unsqueeze(0).float()
            factor = cfg.routed_scaling_factor / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
            routing_weights = (routing_weights * factor).half()
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
            return selected_experts, routing_weights
        return _routing_nogroup_torch(cfg, y, params, scores)

    if bsz == 1:
        _gate_t(cfg)
        ext.routing_ds3_nogroup(
            y,
            cfg.gate_tensor,
            cfg.router_logits_bsz1,
            _esb_h(cfg),
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
            cfg.routed_scaling_factor,
            cfg.gate_tensor_t,
            ROUTING_ACT_SIGMOID,
            cfg.gate_i8,
            cfg.gate_sb,
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1

    else:
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            router_logits = torch.matmul(_pad_rows(y), cfg.gate_tensor)
            routing_weights = router_logits.sigmoid().float()
            if cfg.e_score_correction_bias is not None:
                routing_weights = routing_weights + cfg.e_score_correction_bias.unsqueeze(0).float()
            factor = cfg.routed_scaling_factor / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
            routing_weights = (routing_weights * factor).half()
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
        else:
            router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
            _gate_t(cfg)
            ext.routing_ds3_nogroup(
                y,
                cfg.gate_tensor,
                router_logits,
                _esb_h(cfg),
                selected_experts,
                routing_weights,
                cfg.routed_scaling_factor,
                cfg.gate_tensor_t,
                ROUTING_ACT_SIGMOID,
                cfg.gate_i8,
                cfg.gate_sb,
            )
        return selected_experts, routing_weights


def _sqrtsp_scores(cfg, y):
    logits = torch.matmul(y.float(), cfg.gate_tensor.float())
    return F.softplus(logits).sqrt()


def _vl_rows(cfg, params, bsz, device):
    """DeepSeek-V4 vision: mask of image rows (multimodal embedding ids) in the current chunk,
    or None when there are none. Only chunks that carry indexed embeddings (prompt prefill)
    are inspected, computed once per forward per device, so text decode never syncs."""
    if getattr(cfg, "e_score_bias_vl", None) is None or not params.get("indexed_embeddings"):
        return None
    key = ("_vl_rows", str(device))
    if key in params:
        return params[key]
    from .attn import get_for_device
    ids = get_for_device(params, "input_ids", device).reshape(-1)
    assert ids.shape[0] == bsz, f"routing: {bsz} hidden rows but {ids.shape[0]} input ids"
    mask = ids >= FIRST_MM_EMBEDDING_INDEX
    mask = mask if bool(mask.any()) else None
    params[key] = mask
    return mask


def _esb_vl_h(cfg):
    """fp16 copy of the vision selection bias (mean-centered like _esb_h; selection is shift
    invariant), built lazily."""
    if getattr(cfg, "e_score_bias_vl_h", None) is None:
        esb = cfg.e_score_bias_vl
        cfg.e_score_bias_vl_h = esb if esb.dtype == torch.half else (esb - esb.mean()).half()
    return cfg.e_score_bias_vl_h


def _routing_sqrtsp_vl(cfg, y, vl, hash_sel):
    """sqrtsp routing for a chunk with image rows (reference Gate.forward): image rows select
    top-k on scores + bias_vl; text rows use the hash table selection when given (hash layers),
    else scores + bias. Weights: raw scores over the selected set, normalized, times
    routed_scaling_factor. Composed from the deterministic ext kernels (one top-k pass per bias
    over all rows, merged by the row mask), so tensor-parallel ranks agree bit for bit."""
    bsz = y.shape[0]
    _gate_t(cfg)
    vl_col = vl.unsqueeze(-1)

    def topk_with(bias_h):
        logits, sel, w = _routing_buffers(cfg, bsz, y.device)
        ext.routing_ds3_nogroup(
            y, cfg.gate_tensor, logits, bias_h, sel, w, cfg.routed_scaling_factor,
            cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb,
        )
        return sel.clone(), w.clone()

    sel_vl, w_vl = topk_with(_esb_vl_h(cfg))
    if hash_sel is None:
        sel_tx, w_tx = topk_with(_esb_h(cfg))
    else:
        logits, _, w_tx = _routing_buffers(cfg, bsz, y.device)
        sel_tx = hash_sel.long()
        ext.routing_sel_norm(
            y, cfg.gate_tensor, logits, sel_tx, w_tx, cfg.routed_scaling_factor,
            cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb,
        )
    sel = torch.where(vl_col, sel_vl, sel_tx)
    w = torch.where(vl_col, w_vl, w_tx)
    return sel, w


def routing_sqrtsp(bsz, cfg, y, params):
    """DeepSeek-V4 router: sqrt(softplus(logits)) affinity, noaux_tc bias for selection only,
    weights normalized over the selected set, times routed_scaling_factor. The nogroup top-k
    kernel serves every batch size (one block per row); bsz 1 reuses the cached output
    buffers, larger batches allocate per call. activate_all_experts (conversion) stays
    torch-composed."""
    if params.get("activate_all_experts") or not hasattr(ext, "routing_ds3_nogroup"):
        return _routing_nogroup_torch(cfg, y, params, _sqrtsp_scores(cfg, y))
    vl = _vl_rows(cfg, params, bsz, y.device)
    if vl is not None:
        return _routing_sqrtsp_vl(cfg, y, vl, None)
    _gate_t(cfg)
    if bsz == 1:
        router_logits = cfg.router_logits_bsz1
        selected_experts = cfg.selected_experts_bsz1
        routing_weights = cfg.routing_weights_bsz1
    else:
        router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_ds3_nogroup(
        y,
        cfg.gate_tensor,
        router_logits,
        _esb_h(cfg),
        selected_experts,
        routing_weights,
        cfg.routed_scaling_factor,
        cfg.gate_tensor_t,
        ROUTING_ACT_SQRTSP,
        cfg.gate_i8,
        cfg.gate_sb,
    )
    return selected_experts, routing_weights


def routing_sqrtsp_hash(bsz, cfg, y, params):
    """DeepSeek-V4 hash-MoE bootstrap: expert indices come from the frozen tid2eid table
    indexed by the current tokens (params["input_ids"], flattened row-major); the learned
    gate still weights the selected experts."""
    if params.get("activate_all_experts"):
        return routing_sqrtsp(bsz, cfg, y, params)
    # One device copy of the ids per forward via the params cache, shared by every hash
    # layer on that device; batch dims flatten row-major, matching the hidden-state rows
    from .attn import get_for_device
    input_ids = get_for_device(params, "input_ids", cfg.tid2eid.device).reshape(-1)
    assert input_ids.shape[0] == bsz, \
        f"hash routing: {bsz} hidden rows but {input_ids.shape[0]} input ids"
    vl = _vl_rows(cfg, params, bsz, y.device)
    if vl is not None:
        # Image rows are outside the table: look up token 0 for them (discarded) and route
        # them by top-k with the vision bias, as the reference does
        safe_ids = torch.where(to_device(vl, input_ids.device), torch.zeros_like(input_ids), input_ids)
        hash_sel = to_device(cfg.tid2eid[safe_ids], y.device).long()
        return _routing_sqrtsp_vl(cfg, y, vl, hash_sel)
    selected_experts = to_device(cfg.tid2eid[input_ids], y.device).long()
    _gate_t(cfg)
    if bsz == 1:
        routing_weights = cfg.routing_weights_bsz1
        router_logits = cfg.router_logits_bsz1
    else:
        router_logits, _, routing_weights = _routing_buffers(cfg, bsz, y.device)
    if hasattr(ext, "routing_sel_norm"):
        ext.routing_sel_norm(
            y,
            cfg.gate_tensor,
            router_logits,
            selected_experts,
            routing_weights,
            cfg.routed_scaling_factor,
            cfg.gate_tensor_t,
            ROUTING_ACT_SQRTSP,
            cfg.gate_i8,
            cfg.gate_sb,
        )
        return selected_experts, routing_weights
    scores = _sqrtsp_scores(cfg, y)
    routing_weights = scores.gather(1, selected_experts)
    routing_weights = routing_weights / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
    return selected_experts, (routing_weights * cfg.routed_scaling_factor).half()
