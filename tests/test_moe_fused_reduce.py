"""Bit-exactness of the fused MoE routed-expert reduction (EXL3_FUSE_MOE_REDUCE).

At decode the grouped-mgemm MoE path finishes with

    final = torch.sum(out.float() * routing_weights.view(-1, 1).float(), dim = 0)

which is FOUR launches -- two casts, a broadcast multiply, and a reduction --
for 51 KB of data, 48 times per token. Two fusions were written:

    EXL3_FUSE_MOE_REDUCE=mul    cast + cast + mul -> one kernel (4 -> 2)  SHIPS
    reduction folded in too                                     (4 -> 1)  WITHDRAWN

``mul`` is exact by construction: half -> float is exact, and the product is a
single fp32 multiply in both arms, so nothing about the reduction changes.

The full fold was predicted at risk and measured at risk. torch materialises the
product tile and a separate kernel sums it, so torch computes
``fl(fl(o*w) + acc)``. A fused kernel writing ``acc += o * w`` invites the
backend to contract that into an FMA -- ``fl(o*w + acc)`` -- which is a
different number, the same mechanism that killed the input-Hadamard fusion at
the heavy-accumulator widths (see exl3_triton._fuse_input_had). Walking E in
torch's own index order with tl.static_range is necessary but not sufficient.
It is therefore not selectable, and the test below records ndiff/max|diff| as a
measurement rather than loosening to a tolerance.
"""
import os

import pytest
import torch

from exllamav3.modules import moe_reduce as MR

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not MR.has_triton, reason="Triton not available"),
]


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


def reference(out, weights):
    """Exactly what block_sparse_mlp does today."""
    return torch.sum(out.float() * weights.view(-1, 1).float(), dim=0)


# (E, N): the production case is top-k 10 into hidden 2560. The others cover a
# non-power-of-two E, a single expert, and an N that is not a multiple of the
# block.
SHAPES = [(10, 2560), (10, 640), (6, 2560), (1, 2560), (10, 100)]


@pytest.mark.parametrize("E,N", SHAPES)
def test_fused_mul_is_bit_exact(E, N):
    """cast + cast + mul folded into one kernel; the reduction is untouched."""
    dev = device()
    torch.manual_seed(E * 31 + N)
    out = torch.randn(E, N, dtype=torch.half, device=dev)
    w = torch.rand(1, E, dtype=torch.half, device=dev)

    ref = reference(out, w)
    prod = MR.moe_weighted_product(out, w)
    got = torch.sum(prod, dim=0)

    assert torch.equal(prod, out.float() * w.view(-1, 1).float()), (
        f"product tile differs at E={E} N={N}")
    assert torch.equal(got, ref), (
        f"E={E} N={N}: ndiff {int((got != ref).sum())}/{got.numel()} "
        f"max|diff| {(got - ref).abs().max().item():.4e}")


@pytest.mark.parametrize("E,N", SHAPES)
def test_full_reduce_is_not_bit_exact(E, N):
    """WITHDRAWN mode: records the drift instead of asserting equality.

    Folding the reduction in as well was predicted at risk and measured so. The
    difference is one fp32 ulp on roughly half the outputs -- torch sums a
    materialised product, a fused kernel contracts the multiply-add -- which is
    bounded rounding, not a decode error, but it is not bitwise identity and so
    it does not ship. E=1 has no reduction to reorder and is exact; if a real
    shape ever goes exact across compiler versions, revisit.
    """
    dev = device()
    torch.manual_seed(E * 31 + N)
    out = torch.randn(E, N, dtype=torch.half, device=dev)
    w = torch.rand(1, E, dtype=torch.half, device=dev)

    ref = reference(out, w)
    got = MR.moe_weighted_sum(out, w)
    d = (got - ref).abs()
    rel = (d / ref.abs().clamp(min=1e-6)).max().item()
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-6), (
        f"E={E} N={N} drift is NOT bounded rounding: max|diff| {d.max().item():.4e}")
    if E == 1:
        assert torch.equal(got, ref), "a single expert has no reduction to reorder"
        return
    if torch.equal(got, ref):
        pytest.xfail(f"E={E} N={N} is bit-exact here; if that holds across shapes "
                     f"and compiler versions, 'full' can be reinstated")
    print(f"\nE={E} N={N}: ndiff {int((got != ref).sum())}/{got.numel()} "
          f"max|diff| {d.max().item():.4e} max_rel {rel:.4e}")


def test_full_reduce_is_not_selectable(monkeypatch):
    """A withdrawn mode must not be reachable by setting the flag."""
    monkeypatch.setenv("EXL3_FUSE_MOE_REDUCE", "full")
    with pytest.raises(ValueError, match="withdrawn"):
        MR.fuse_mode()


def test_fused_reduce_matches_reference_dtype_and_shape():
    dev = device()
    out = torch.randn(10, 2560, dtype=torch.half, device=dev)
    w = torch.rand(1, 10, dtype=torch.half, device=dev)
    ref = reference(out, w)
    for got in (torch.sum(MR.moe_weighted_product(out, w), dim=0),):
        assert got.dtype == ref.dtype == torch.float
        assert got.shape == ref.shape == (2560,)


def test_moe_reduce_mode_gate(monkeypatch):
    monkeypatch.delenv("EXL3_FUSE_MOE_REDUCE", raising=False)
    assert MR.fuse_mode() == "off"
    for v, want in (("mul", "mul"), ("0", "off"), ("off", "off"), ("1", "mul")):
        monkeypatch.setenv("EXL3_FUSE_MOE_REDUCE", v)
        assert MR.fuse_mode() == want, v
    monkeypatch.setenv("EXL3_FUSE_MOE_REDUCE", "nonsense")
    with pytest.raises(ValueError):
        MR.fuse_mode()
