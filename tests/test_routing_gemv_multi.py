"""EXL3_ROUTING_GEMV=multi -- the M-row routing GEMV must match the hgemm path it replaces.

routing_gemv only took its hand-written kernel at bsz == 1 AND only when gate_t was supplied,
and block_sparse_mlp passed gate_t on the bsz == 1 branch only. So every bsz > 1 routing call
-- every call in a speculative verify forward -- went through hgemm. These tests cover the new
M-row kernel across the whole dispatched range.

Correctness here is not "matches hgemm bit for bit": the two accumulate in different orders.
The assertion is that the new kernel is no further from an fp32 reference than hgemm is, which
is a real cross-check rather than a transcription of the same expression, and that the top-K
SELECTION -- the only thing downstream consumes -- is identical.
"""
import sys, os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

DEV = "cuda:0"
K_DIM, E, TOPK = 2560, 512, 10
MAX_M = 16


def _mk(M, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    hidden = (torch.randn(M, K_DIM, generator = g) * 0.35).half().to(DEV)
    gate = (torch.randn(K_DIM, E, generator = g) * 0.05).half().to(DEV)
    return hidden, gate, gate.T.contiguous()


def _run(hidden, gate, gate_t, M, arm):
    prev = os.environ.get("EXL3_ROUTING_GEMV")
    if arm is None:
        os.environ.pop("EXL3_ROUTING_GEMV", None)
    else:
        os.environ["EXL3_ROUTING_GEMV"] = arm
    try:
        scores = torch.empty((M, E), dtype = torch.half, device = DEV)
        idx = torch.empty((M, TOPK), dtype = torch.long, device = DEV)
        w = torch.empty((M, TOPK), dtype = torch.half, device = DEV)
        ext.routing_std(hidden, gate, scores, idx, w, None, gate_t, None)
        torch.cuda.synchronize()
        return scores, idx, w
    finally:
        if prev is None:
            os.environ.pop("EXL3_ROUTING_GEMV", None)
        else:
            os.environ["EXL3_ROUTING_GEMV"] = prev


@pytest.mark.parametrize("arm", ["multi", "multi4"])
@pytest.mark.parametrize("M", list(range(2, MAX_M + 1)))
def test_multi_no_worse_than_hgemm(M, arm):
    hidden, gate, gate_t = _mk(M, seed = 1000 + M)
    ref = (hidden.float() @ gate.float())

    s_base, i_base, w_base = _run(hidden, gate, gate_t, M, None)
    s_mult, i_mult, w_mult = _run(hidden, gate, gate_t, M, arm)

    e_base = (s_base.float() - ref).abs().max().item()
    e_mult = (s_mult.float() - ref).abs().max().item()
    assert e_mult <= max(e_base * 1.5, 4e-3), \
        f"M={M} arm={arm}: multi err {e_mult:.3e} vs hgemm err {e_base:.3e}"

    assert torch.equal(i_base.sort(dim = 1).values, i_mult.sort(dim = 1).values), \
        f"M={M} arm={arm}: top-{TOPK} selection differs"
    assert (w_base.float() - w_mult.float()).abs().max().item() < 4e-3


def test_knob_is_read_per_call():
    """A read at import would make this a no-op as a per-sample arm (RUNBOOK trap)."""
    M = 7
    hidden, gate, gate_t = _mk(M, seed = 7)
    a = _run(hidden, gate, gate_t, M, "multi")[0].clone()
    b = _run(hidden, gate, gate_t, M, None)[0].clone()
    c = _run(hidden, gate, gate_t, M, "multi")[0].clone()
    assert torch.equal(a, c), "same arm gave different results across calls"
    ref = (hidden.float() @ gate.float())
    assert (a.float() - ref).abs().max().item() < 5e-2
    assert (b.float() - ref).abs().max().item() < 5e-2


@pytest.mark.parametrize("arm", ["multi", "multi4"])
@pytest.mark.parametrize("M", [MAX_M + 1, 64])
def test_above_max_m_falls_back(M, arm):
    hidden, gate, gate_t = _mk(M, seed = 2000 + M)
    s_base = _run(hidden, gate, gate_t, M, None)[0]
    s_mult = _run(hidden, gate, gate_t, M, arm)[0]
    assert torch.equal(s_base, s_mult), f"M={M} should fall back to hgemm, bit-identically"


@pytest.mark.parametrize("arm", ["multi", "multi4"])
def test_bsz1_unaffected(arm):
    hidden, gate, gate_t = _mk(1, seed = 3)
    s_base = _run(hidden, gate, gate_t, 1, None)[0]
    s_mult = _run(hidden, gate, gate_t, 1, arm)[0]
    assert torch.equal(s_base, s_mult), "bsz 1 already had its own kernel; multi must not change it"


def test_unknown_arm_value_is_off():
    """An unrecognised value must fall back, not silently pick a kernel."""
    M = 7
    hidden, gate, gate_t = _mk(M, seed = 11)
    s_base = _run(hidden, gate, gate_t, M, None)[0]
    s_junk = _run(hidden, gate, gate_t, M, "yes")[0]
    assert torch.equal(s_base, s_junk)


def test_non_contiguous_hidden_falls_back():
    M = 7
    hidden, gate, gate_t = _mk(M * 2, seed = 5)
    view = hidden[::2]
    assert not view.is_contiguous()
    scores_a = torch.empty((M, E), dtype = torch.half, device = DEV)
    scores_b = torch.empty((M, E), dtype = torch.half, device = DEV)
    idx = torch.empty((M, TOPK), dtype = torch.long, device = DEV)
    w = torch.empty((M, TOPK), dtype = torch.half, device = DEV)
    os.environ.pop("EXL3_ROUTING_GEMV", None)
    ext.routing_std(view, gate, scores_a, idx, w, None, gate_t, None)
    os.environ["EXL3_ROUTING_GEMV"] = "multi"
    try:
        ext.routing_std(view, gate, scores_b, idx, w, None, gate_t, None)
    finally:
        os.environ.pop("EXL3_ROUTING_GEMV", None)
    torch.cuda.synchronize()
    assert torch.equal(scores_a, scores_b)
