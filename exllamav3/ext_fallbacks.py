"""Pure-PyTorch fallback implementations of C++ extension functions.

Used on ROCm where the CUDA-specific kernels (activation.cu, norm.cu, etc.) are
excluded from the build. Each function matches the signature of its C++ counterpart
so it can be monkey-patched onto the extension module transparently.

These are written for correctness, not performance — they use standard PyTorch ops
that compose naturally with CUDA graph capture and (potentially) torch.compile.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# -- Activation fused ops (activation.cu) -------------------------------------

def silu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.silu(x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def silu_oai_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    # OAI variant: silu(x * y) — see activation.cu
    r = F.silu(x * y)
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def gelu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.gelu(x, approximate = "tanh") * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def relu2_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = torch.square(F.relu(x)) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def relu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.relu(x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def xielu(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = (torch.tanh(x.clamp(min = -2.3562, max = 2.3562)) * x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)


# -- In-place gate ops (activation.cu) -----------------------------------------

def mul_sigmoid_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(torch.sigmoid(g))

def mul_sigmoid_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(torch.sigmoid(g))

def mul_softplus_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(F.softplus(g.float(), threshold = 11).to(o.dtype))

def add_sigmoid_gate(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
    # add_sigmoid_kernel_f: z += x * sigmoid(y), with y a per-row gate scalar
    # (size(-1) == 1) broadcast across the row. Accumulates into z; does not
    # overwrite it.
    z.add_(x * torch.sigmoid(y.float()))

def add_sigmoid_gate_proj(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
) -> None:
    # add_sigmoid_proj_kernel_f: the gate projection is folded in, so
    # z += x * sigmoid(y @ w) with w of shape (dim, 1). Accumulates into z.
    z.add_(x * torch.sigmoid(torch.matmul(y.float(), w.float())))


# -- Attention helpers (activation.cu) ----------------------------------------

def deinterleave_qg(
    qg: torch.Tensor,
    q: torch.Tensor,
    g: torch.Tensor,
    head_dim: int,
) -> None:
    bsz, qlen = qg.shape[0], qg.shape[1]
    chunks = qg.view(bsz, qlen, -1, head_dim * 2)
    q.copy_(chunks[..., :head_dim].reshape(q.shape))
    g.copy_(chunks[..., head_dim:].reshape(g.shape))


# -- Norm ops (norm.cu) --------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    eps: float,
    constant_bias: float,
    constant_scale: float,
    span_heads: bool,
    add_residual: bool,
) -> None:
    xf = x.float()
    if w is not None:
        wf = (w + constant_bias).float() if constant_bias != 0.0 else w.float()
    else:
        wf = None
    var = xf.pow(2).mean(dim = -1, keepdim = True) + eps
    xf = xf * torch.rsqrt(var) * constant_scale
    if wf is not None:
        xf = xf * wf
    y.copy_(xf.to(y.dtype))

def rms_norm_res_in(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    constant_bias: float,
    constant_scale: float,
) -> None:
    r.add_(x)
    rf = r.float()
    if w is not None:
        wf = (w + constant_bias).float() if constant_bias != 0.0 else w.float()
    else:
        wf = None
    var = rf.pow(2).mean(dim = -1, keepdim = True) + eps
    rf = rf * torch.rsqrt(var) * constant_scale
    if wf is not None:
        rf = rf * wf
    y.copy_(rf.to(y.dtype))

def gated_rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    eps: float,
    constant_bias: float,
    w_groups: int,
    gate_first: bool,
) -> None:
    xf = x.float()
    gf = g.float()
    if gate_first:
        hidden = xf * F.silu(gf)
        if w_groups > 1:
            wf = w.view(w_groups, -1).float()
            hidden_2d = hidden.view(-1, wf.shape[1])
            var = hidden_2d.pow(2).mean(dim = -1, keepdim = True) + eps
            hidden_2d = hidden_2d * torch.rsqrt(var)
            hidden = (wf * hidden_2d).view(hidden.shape)
        else:
            var = hidden.pow(2).mean(-1, keepdim = True) + eps
            hidden = hidden * torch.rsqrt(var)
            hidden = w.float() * hidden
    else:
        var = xf.pow(2).mean(-1, keepdim = True) + eps
        xf = xf * torch.rsqrt(var)
        if w_groups > 1:
            hidden = w.view(w_groups, -1).float() * xf.view(-1, w.shape[-1] // w_groups if w.dim() > 1 else w.shape[0] // w_groups)
        else:
            hidden = w.float() * xf
        hidden = hidden * F.silu(gf)
    y.copy_(hidden.to(y.dtype))


# -- Softcap (softcap.cu) ------------------------------------------------------

def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    if cap == 0.0:
        return x
    return torch.tanh(x / cap) * cap


# -- Sentinel for missing BC_* classes -----------------------------------------

class _BCNone:
    """Callable that returns None, used as stand-in for missing BC_* constructors."""
    __slots__ = ()
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


# -- MoE routing (routing.cu) -------------------------------------------------
#
# Semantics transcribed from exllamav3_ext/routing.cu. The CUDA kernels fuse the
# router GEMV, the score activation, the top-k selection and the renormalization;
# here they are separate ops with the same observable result. `scores` is an output
# buffer in every case (it receives the raw router logits), and topk_indices /
# topk_weights / weights are preallocated and written in place.

ROUTING_ACT_SIGMOID = 0
ROUTING_ACT_SQRTSP = 1


def _routing_act(x: torch.Tensor, act_fn: int) -> torch.Tensor:
    # routing_act<ACT> in routing.cu: sqrt(softplus(x)) with torch's threshold=20
    # semantics, or a numerically stable sigmoid.
    if act_fn == ROUTING_ACT_SQRTSP:
        return torch.sqrt(F.softplus(x, beta = 1.0, threshold = 20.0))
    return torch.sigmoid(x)


def _routing_gemv(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    # routing_gemv: scores <- hidden @ gate, gate is (hidden_dim, num_experts).
    # Accumulate in fp32 and round once, matching the kernel's fp32 accumulator.
    h = hidden.view(-1, hidden.shape[-1])
    logits = torch.matmul(h.float(), gate.float())
    scores.copy_(logits.to(scores.dtype))
    return logits


def routing_std(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    per_expert_scale: torch.Tensor | None,
    gate_t: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> None:
    logits = _routing_gemv(hidden, gate, scores)
    if bias is not None:
        # Router bias (gpt-oss): biased logits drive both selection and the softmax
        logits = logits + bias.view(1, -1).float()
    k = topk_indices.shape[1]
    top_v, top_i = torch.topk(logits, k, dim = -1)
    # The kernel computes exp(logit - max_over_all_experts) and normalizes over the
    # selected k. The global max is always inside the top-k, so this is a softmax
    # over the selected logits. (The kernel's +1e-20 guard is below fp16 resolution.)
    w = torch.softmax(top_v, dim = -1)
    if per_expert_scale is not None:
        w = w * per_expert_scale.float()[top_i]
    topk_indices.copy_(top_i.to(torch.long))
    topk_weights.copy_(w.to(topk_weights.dtype))


def routing_ds3_nogroup(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    scaling_factor: float,
    gate_t: torch.Tensor | None,
    act_fn: int,
) -> None:
    logits = _routing_gemv(hidden, gate, scores)
    o_all = _routing_act(logits, act_fn)
    # DS3 aux-loss-free bias steers *selection* only; the emitted weight is the
    # activated unbiased score. (The kernel's `v -= min(v)` shift only exists to keep
    # the radix sort on positive floats and does not affect ordering.)
    sel = o_all if bias is None else o_all + bias.view(1, -1).float()
    k = topk_indices.shape[1]
    _, top_i = torch.topk(sel, k, dim = -1)
    o = o_all.gather(-1, top_i)
    o = o * (scaling_factor / (o.sum(dim = -1, keepdim = True) + 1e-20))
    topk_indices.copy_(top_i.to(torch.long))
    topk_weights.copy_(o.to(topk_weights.dtype))


def routing_sel_norm(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    scores: torch.Tensor,
    selected: torch.Tensor,
    weights: torch.Tensor,
    scaling_factor: float,
    gate_t: torch.Tensor | None,
    act_fn: int,
) -> None:
    # Experts are chosen upstream (hash routing); this only scores and renormalizes.
    logits = _routing_gemv(hidden, gate, scores)
    o = _routing_act(logits.gather(-1, selected.long()), act_fn)
    o = o * (scaling_factor / (o.sum(dim = -1, keepdim = True) + 1e-20))
    weights.copy_(o.to(weights.dtype))
