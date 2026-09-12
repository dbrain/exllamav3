"""Fused reduction of the routed experts' outputs (decode path).

``block_sparse_mlp``'s grouped-mgemm decode branch finishes with

    torch.sum(out.float() * routing_weights.view(-1, 1).float(), dim = 0)

which torch executes as four kernels -- two casts, a broadcast multiply and a
reduction -- over 51 KB, once per layer. At decode a launch costs far more than
51 KB of stream, so the four are almost pure overhead.

Two modes, because they carry different risk:

  mul   the two casts and the multiply become one kernel; ``torch.sum`` still
        does the reduction. Bit-exact by construction -- half -> float is
        exact, and the product is the same single fp32 multiply -- because no
        reduction is reordered.

        MEASURED NULL end to end, and kept OFF by default for that reason:

            base_end control          120.759 ms/token   2652 launches
            EXL3_FUSE_MOE_REDUCE=mul  120.940 ms/token   2556 launches
                                      +0.181 ms, inside the 0.50 ms spread of
                                      the run's three null arms

        The launch count falls by 96 and the time does not move, because the
        trade is 144 aten launches for 48 Triton ones and a Triton launch is
        the dearer of the two. Correct, exact, and not worth enabling on its
        own; it is here so a future arm can carry it for free alongside a
        fusion that does pay.

  full  WITHDRAWN. Folding the reduction in as well was predicted at risk and
        measured not bit-exact, on gfx1150 / Triton 3.5.1:

            E=10 N=2560  ndiff 1115/2560  max|diff| 9.537e-07  max_rel 4.82e-05
            E=10 N=640   ndiff  330/640   max|diff| 4.768e-07  max_rel 7.83e-06
            E=6  N=2560  ndiff  486/2560  max|diff| 4.768e-07  max_rel 3.69e-06
            E=1  N=2560  exact (no reduction to reorder)

        One fp32 ulp on ~43% of outputs. torch materialises the product and sums
        it in a second pass -- ``fl(fl(o*w) + acc)`` -- while a fused
        ``acc += o * w`` lets the backend contract into an FMA, ``fl(o*w + acc)``.
        Walking E in torch's own order is necessary but not sufficient. Same
        mechanism as the input-Hadamard fusion at the heavy-accumulator widths
        (exl3_triton._fuse_input_had). ``moe_weighted_sum`` is kept only as the
        subject of the test that records this; ``fuse_mode`` will not return it.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()


# "full" is deliberately absent: MEASURED not bit-exact (see below and
# tests/test_moe_fused_reduce.py::test_full_reduce_is_not_bit_exact), so it must
# not be selectable. Setting it raises rather than silently degrading quality.
_MODES = {
    "": "off", "0": "off", "off": "off", "no": "off", "false": "off",
    "1": "mul", "mul": "mul", "on": "mul", "true": "mul",
}


def fuse_mode() -> str:
    """'off' | 'mul', read per call so an A/B harness can flip it."""
    v = os.environ.get("EXL3_FUSE_MOE_REDUCE", "off").strip().lower()
    try:
        return _MODES[v]
    except KeyError:
        raise ValueError(
            f"EXL3_FUSE_MOE_REDUCE: unknown mode {v!r}, expected off|mul"
            + (" ('full' was withdrawn: measured not bit-exact)"
               if v == "full" else "")) from None


@triton.jit
def _moe_weighted_kernel(
    out_ptr, w_ptr, y_ptr,
    N, stride_oe,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    REDUCE: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    if REDUCE:
        # Serial walk over E in index order, which is what torch's reduction
        # does for a small E and a large N: each output column gets its own
        # thread and loops the E strided elements. Starting the accumulator at
        # zero is exact (0 + x == x), so the two arms sum the same values in
        # the same order.
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for e in tl.static_range(E):
            o = tl.load(out_ptr + e * stride_oe + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + e).to(tl.float32)
            acc += o * w
        tl.store(y_ptr + cols, acc, mask=mask)
    else:
        for e in tl.static_range(E):
            o = tl.load(out_ptr + e * stride_oe + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + e).to(tl.float32)
            tl.store(y_ptr + e * N + cols, o * w, mask=mask)


def _launch(out: torch.Tensor, weights: torch.Tensor, y: torch.Tensor, reduce: bool):
    E, N = out.shape
    assert out.dtype == torch.half and weights.dtype == torch.half
    assert weights.numel() == E and weights.is_contiguous()
    assert out.stride(1) == 1 and y.stride(-1) == 1
    BLOCK = 256 if N >= 256 else triton.next_power_of_2(N)
    _moe_weighted_kernel[(triton.cdiv(N, BLOCK),)](
        out, weights, y,
        N, out.stride(0),
        E=E, BLOCK=BLOCK, REDUCE=reduce,
        num_warps=4, num_stages=2,
    )


def moe_weighted_sum(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sum_e out[e] * weights[e], fp32, in one launch. out [E, N] half.

    NOT bit-exact against the torch chain and therefore not reachable from
    moe_reduce; see the module docstring for the measurement.
    """
    y = torch.empty((out.shape[1],), dtype=torch.float, device=out.device)
    _launch(out, weights, y, True)
    return y


def moe_weighted_product(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """out.float() * weights.view(-1, 1).float(), in one launch. [E, N] fp32."""
    y = torch.empty(out.shape, dtype=torch.float, device=out.device)
    _launch(out, weights, y, False)
    return y


def moe_reduce(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Dispatch on EXL3_FUSE_MOE_REDUCE; the 'off' arm is the original torch chain."""
    mode = fuse_mode() if has_triton else "off"
    if mode == "mul":
        return torch.sum(moe_weighted_product(out, weights), dim=0)
    return torch.sum(out.float() * weights.view(-1, 1).float(), dim=0)
