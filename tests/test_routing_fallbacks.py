"""Validate the pure-PyTorch routing fallbacks (ext_fallbacks.routing_*) against
independently-formulated references.

routing.cu is excluded from the ROCm build, so these fallbacks are what actually runs
MoE models there. A wrong normalization here does not crash — it silently degrades
output — so each fallback is checked against the formulation used by the reference
HF implementation of the corresponding architecture, which is written differently
from the CUDA kernel and therefore is a real cross-check rather than a transcription
of the same expression.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from exllamav3 import ext_fallbacks as fb

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BSZ, H, NE, K = 7, 128, 32, 4


def _mk(bsz=BSZ, ne=NE, k=K, seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(bsz, H, device=DEV) * 0.5).half()
    gate = (torch.randn(H, ne, device=DEV) * 0.5).half()
    scores = torch.empty((bsz, ne), dtype=torch.half, device=DEV)
    idx = torch.empty((bsz, k), dtype=torch.long, device=DEV)
    w = torch.empty((bsz, k), dtype=torch.half, device=DEV)
    return hidden, gate, scores, idx, w


def _logits(hidden, gate):
    return torch.matmul(hidden.view(-1, hidden.shape[-1]).float(), gate.float())


# --- routing_std: HF Qwen3-MoE / Mixtral formulation -------------------------
# softmax over ALL experts, then top-k, then renormalize. The kernel instead takes
# top-k of the logits and softmaxes over the selection; these are algebraically
# identical and that equivalence is exactly what this asserts.

@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("use_scale", [False, True])
def test_routing_std_matches_hf(use_bias, use_scale):
    hidden, gate, scores, idx, w = _mk()
    bias = (torch.randn(1, NE, device=DEV) * 0.3).half() if use_bias else None
    pes = (torch.rand(NE, device=DEV) + 0.5).to(torch.bfloat16) if use_scale else None

    fb.routing_std(hidden, gate, scores, idx, w, pes, None, bias)

    ref_logits = _logits(hidden, gate)
    if use_bias:
        ref_logits = ref_logits + bias.view(1, -1).float()
    probs = torch.softmax(ref_logits, dim=-1)
    ref_w, ref_i = torch.topk(probs, K, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
    if use_scale:
        ref_w = ref_w * pes.float()[ref_i]

    assert torch.equal(idx, ref_i.to(torch.long))
    torch.testing.assert_close(w.float(), ref_w.float(), atol=2e-3, rtol=2e-3)


# --- routing_ds3_nogroup: HF DeepSeek-V3 formulation -------------------------
# sigmoid scores; the correction bias steers selection only; weights come from the
# UNBIASED scores, renormalized over the selection, times routed_scaling_factor.

@pytest.mark.parametrize("act_fn", [fb.ROUTING_ACT_SIGMOID, fb.ROUTING_ACT_SQRTSP])
@pytest.mark.parametrize("use_bias", [False, True])
def test_routing_ds3_matches_hf(act_fn, use_bias):
    hidden, gate, scores, idx, w = _mk(seed=1)
    bias = (torch.randn(1, NE, device=DEV) * 0.3).half() if use_bias else None
    sf = 2.5

    fb.routing_ds3_nogroup(hidden, gate, scores, bias, idx, w, sf, None, act_fn)

    lg = _logits(hidden, gate)
    s = torch.sigmoid(lg) if act_fn == fb.ROUTING_ACT_SIGMOID \
        else torch.sqrt(F.softplus(lg, beta=1.0, threshold=20.0))
    choice = s if not use_bias else s + bias.view(1, -1).float()
    ref_i = torch.topk(choice, K, dim=-1)[1]
    ref_w = s.gather(1, ref_i)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True) * sf

    assert torch.equal(idx, ref_i.to(torch.long))
    torch.testing.assert_close(w.float(), ref_w.float(), atol=2e-3, rtol=2e-3)


def test_ds3_bias_steers_selection_but_not_weight():
    """The bug that would silently degrade output: applying the bias to the emitted
    weight instead of only to the choice."""
    hidden, gate, scores, idx, w = _mk(seed=2)
    bias = torch.zeros(1, NE, device=DEV).half()
    bias[0, 3] = 10.0                       # force expert 3 into the selection
    fb.routing_ds3_nogroup(hidden, gate, scores, bias, idx, w, 1.0, None,
                           fb.ROUTING_ACT_SIGMOID)
    assert (idx == 3).any(dim=-1).all(), "biased expert must be selected"

    s = torch.sigmoid(_logits(hidden, gate))
    pos = (idx == 3).float().argmax(dim=-1)
    got = w[torch.arange(BSZ, device=DEV), pos].float()
    unbiased = s[:, 3] / s.gather(1, idx).sum(dim=-1) * 1.0
    torch.testing.assert_close(got, unbiased.float(), atol=2e-3, rtol=2e-3)


# --- routing_sel_norm --------------------------------------------------------

@pytest.mark.parametrize("act_fn", [fb.ROUTING_ACT_SIGMOID, fb.ROUTING_ACT_SQRTSP])
def test_routing_sel_norm(act_fn):
    hidden, gate, scores, _, w = _mk(seed=3)
    torch.manual_seed(4)
    sel = torch.stack([torch.randperm(NE, device=DEV)[:K] for _ in range(BSZ)]).long()
    sf = 1.7

    fb.routing_sel_norm(hidden, gate, scores, sel, w, sf, None, act_fn)

    lg = _logits(hidden, gate)
    s = torch.sigmoid(lg) if act_fn == fb.ROUTING_ACT_SIGMOID \
        else torch.sqrt(F.softplus(lg, beta=1.0, threshold=20.0))
    ref = s.gather(1, sel)
    ref = ref / ref.sum(dim=-1, keepdim=True) * sf
    torch.testing.assert_close(w.float(), ref.float(), atol=2e-3, rtol=2e-3)


# --- shared invariants -------------------------------------------------------

def test_weights_sum_to_scaling_factor():
    hidden, gate, scores, idx, w = _mk(seed=5)
    fb.routing_ds3_nogroup(hidden, gate, scores, None, idx, w, 3.0, None,
                           fb.ROUTING_ACT_SIGMOID)
    torch.testing.assert_close(w.float().sum(-1),
                               torch.full((BSZ,), 3.0, device=DEV), atol=3e-3, rtol=3e-3)

    fb.routing_std(hidden, gate, scores, idx, w, None, None, None)
    torch.testing.assert_close(w.float().sum(-1),
                               torch.ones(BSZ, device=DEV), atol=3e-3, rtol=3e-3)


def test_scores_buffer_receives_raw_logits():
    """scores is an output: downstream code reads the raw router logits from it."""
    hidden, gate, scores, idx, w = _mk(seed=6)
    fb.routing_std(hidden, gate, scores, idx, w, None, None, None)
    torch.testing.assert_close(scores.float(), _logits(hidden, gate).half().float(),
                               atol=1e-2, rtol=1e-2)


def test_sqrtsp_matches_torch_softplus_including_threshold():
    x = torch.tensor([-30.0, -1.0, 0.0, 1.0, 19.9, 20.1, 50.0], device=DEV)
    got = fb._routing_act(x, fb.ROUTING_ACT_SQRTSP)
    ref = torch.sqrt(F.softplus(x, beta=1.0, threshold=20.0))
    torch.testing.assert_close(got, ref)
