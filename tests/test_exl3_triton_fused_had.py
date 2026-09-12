"""Bit-exactness of the in-kernel input Hadamard (EXL3_FUSE_INPUT_HAD).

The fused path must be bitwise identical to had_r_128_triton followed by the
unfused GEMM, not merely close: it is an optimization of the decode path, and
any drift would be a silent quality change no tolerance test would catch.

Two levels of coverage:

- pinned launches through _fused_dequant_gemm_kernel.fn, which fix BLOCK_N /
  BLOCK_K so both arms provably enter the same kernel branch (autotune is free
  to pick different tiles for the two arms, and on a shape where one pool
  member takes a fast path and another the generic path that alone would
  change fp32 accumulation order)
- the whole linear through _linear_exl3_triton with the env var flipped, on
  shapes whose K is a multiple of 256 so every pool member takes the same
  branch; these also assert the arms actually differ, by checking whether the
  xh workspace was written
"""
import os

import pytest
import torch

from exllamav3.modules.quant import exl3_triton as T

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not T.has_triton, reason="Triton not available"),
    pytest.mark.skipif(not hasattr(T, "_fuse_input_had"), reason="fused input Hadamard not available"),
]

if T.has_triton:
    import triton


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


def make_trellis(in_features, out_features, K, dev):
    return torch.randint(
        0, 65536, (in_features // 16, out_features // 16, 256 * K // 16),
        dtype=torch.int32, device=dev,
    ).to(torch.short)


def _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, suh=None, num_warps=2):
    dev = x.device
    perm = T._get_perm(dev)
    mrow = T._get_m_row_offsets(K_bits, dev) if K_bits in T._M_ROW_OFFSETS else perm
    y = torch.empty((1, N), dtype=torch.half, device=dev)
    T._fused_dequant_gemm_kernel.fn[(triton.cdiv(N, BN),)](
        x, y, trellis, perm, mrow,
        1, N, K_dim, 1,
        x.stride(0), x.stride(1), trellis.stride(0), trellis.stride(1),
        y.stride(0), y.stride(1), y.stride(0),
        BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=1,
        K_BITS=K_bits, N_PACKED=trellis.shape[-1], CB=cb, M1=True, SPLITS=1,
        # constexpr divisibility gate, as the host entry passes it. Without it
        # FULL is a runtime tensor and Triton compiles all five decode branches
        # into every binary (~3 min/case), which made this file unrunnable.
        N_DIV=T._gate_div(N), K_DIV=T._gate_div(K_dim),
        FUSE_HAD=suh is not None,
        suh_ptr=suh,
        had_r_scale=T._RSCALE_128,
        num_warps=num_warps, num_stages=2,
    )
    return y


# (K_dim, N): 512 divides both tile depths (fast paths); 384 does not divide
# BLOCK_K=256, which routes that case through the generic gather path
PINNED_SHAPES = [(512, 256), (384, 256)]


# Only the light-accumulator widths are bit-exact under the input fusion; see
# exl3_triton._fuse_input_had for the measurement and the mechanism, and
# test_input_had_drifts_at_heavy_widths below for the widths that are not.
# The full cross-product was 108 cases at minutes apiece and could not gate
# anything; the pinned sweep now covers the exact widths across every tile and
# both shapes, and one case per codebook rides along in
# test_fused_had_other_widths.
@pytest.mark.parametrize("K_bits", list(T._IN_HAD_EXACT_BITS))
@pytest.mark.parametrize("cb", [2])
@pytest.mark.parametrize("BN,BK", [(32, 128), (128, 128), (32, 256)])
@pytest.mark.parametrize("K_dim,N", PINNED_SHAPES)
def test_fused_had_bit_exact_pinned(K_dim, N, BN, BK, cb, K_bits):
    dev = device()
    torch.manual_seed(K_dim * 31 + N + K_bits * 7 + BN + BK + cb)
    trellis = make_trellis(K_dim, N, K_bits, dev)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    suh = torch.sign(torch.randn(K_dim, device=dev)).half()

    xh = torch.empty_like(x)
    T.had_r_128_triton(x, xh, suh, None, 1.0)
    y_unfused = _pinned_gemm(xh, trellis, K_bits, cb, N, K_dim, BN, BK)
    y_fused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, suh=suh)

    assert torch.equal(y_fused, y_unfused), (
        f"K_bits={K_bits} cb={cb} BN={BN} BK={BK} shape={K_dim}x{N}: "
        f"max|diff| {(y_fused.float() - y_unfused.float()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("K_bits,cb", [(1, 0), (1, 1), (2, 0), (8, 1)])
def test_fused_had_other_widths(K_bits, cb):
    """Codebook coverage on the widths the gate allows."""
    dev = device()
    K_dim, N, BN, BK = 512, 256, 32, 128
    torch.manual_seed(K_bits * 17 + cb)
    trellis = make_trellis(K_dim, N, K_bits, dev)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    suh = torch.sign(torch.randn(K_dim, device=dev)).half()
    xh = torch.empty_like(x)
    T.had_r_128_triton(x, xh, suh, None, 1.0)
    y_unfused = _pinned_gemm(xh, trellis, K_bits, cb, N, K_dim, BN, BK)
    y_fused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, suh=suh)
    assert torch.equal(y_fused, y_unfused), (
        f"K_bits={K_bits} cb={cb}: max|diff| "
        f"{(y_fused.float() - y_unfused.float()).abs().max().item():.3e}")


# (in_features, out_features), K % 256 == 0 so every autotune pool member takes
# the same kernel branch in both arms; small-N is CTA-starved (split-K plan),
# large-N is not
LINEAR_SHAPES = [
    (2048, 1024),    # classic route, small N
    (2048, 12288),   # classic route, large N
    (4096, 1024),    # split-K route, small N
    (4096, 12288),   # split-K route, large N
]


import contextlib


@contextlib.contextmanager
def _env(**kw):
    prev = {k: os.environ.get(k) for k in kw}
    os.environ.update(kw)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _linear_arm(x, trellis, suh, svh, K, in_f, out_f, fuse):
    y = torch.empty((1, out_f), dtype=torch.half, device=x.device)
    xh = torch.full((1, in_f), float("nan"), dtype=torch.half, device=x.device)
    prev = os.environ.get("EXL3_FUSE_INPUT_HAD")
    os.environ["EXL3_FUSE_INPUT_HAD"] = "1" if fuse else "0"
    try:
        T._linear_exl3_triton(
            x, y, xh, trellis, suh, svh, K, False, False, None, in_f, out_f,
        )
    finally:
        if prev is None:
            os.environ.pop("EXL3_FUSE_INPUT_HAD", None)
        else:
            os.environ["EXL3_FUSE_INPUT_HAD"] = prev
    return y, xh


# bits 2 and 8: the widths the gate allows (see _IN_HAD_EXACT_BITS). With 4 and 6
# every one of these shapes skipped, which made the test vacuous.
@pytest.mark.parametrize("K", [2, 8])
@pytest.mark.parametrize("in_features,out_features", LINEAR_SHAPES)
def test_fused_had_bit_exact_linear(in_features, out_features, K):
    dev = device()
    with _env(EXL3_FUSE_INPUT_HAD="1"):
        eligible = T._fuse_input_had(K, 1, in_features, out_features)
    if not eligible:
        pytest.skip("shape is not eligible for the fused input Hadamard")
    torch.manual_seed(in_features * 13 + out_features + K)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh = torch.sign(torch.randn(in_features, device=dev)).half()
    svh = torch.sign(torch.randn(out_features, device=dev)).half()
    x = torch.randn(1, in_features, dtype=torch.half, device=dev) * 0.1

    y_unfused, xh_unfused = _linear_arm(x, trellis, suh, svh, K, in_features, out_features, False)
    y_fused, xh_fused = _linear_arm(x, trellis, suh, svh, K, in_features, out_features, True)

    # the arms must genuinely differ: only the unfused one writes xh
    assert not xh_unfused.isnan().any()
    assert xh_fused.isnan().all()

    assert torch.equal(y_fused, y_unfused), (
        f"K={K} {in_features}x{out_features}: max|diff| "
        f"{(y_fused.float() - y_unfused.float()).abs().max().item():.3e}"
    )


def test_fuse_input_had_gates(monkeypatch):
    monkeypatch.setenv("EXL3_FUSE_INPUT_HAD", "1")
    assert T._fuse_input_had(2, 1, 4096, 1024)
    assert T._fuse_input_had(8, 1, 4096, 1024)
    for b in (3, 4, 5, 6, 7):
        assert not T._fuse_input_had(b, 1, 4096, 1024), f"bits={b} must be refused"
    assert not T._fuse_input_had(2, 16, 4096, 1024)    # prefill
    assert not T._fuse_input_had(2, 1, 4160, 1024)     # K % 128 (4160 = 32*128 + 64)
    assert not T._fuse_input_had(2, 1, 4096, 1040)     # N % 128
    monkeypatch.setenv("EXL3_FUSE_INPUT_HAD", "0")
    assert not T._fuse_input_had(2, 1, 4096, 1024)
    monkeypatch.delenv("EXL3_FUSE_INPUT_HAD")
    assert not T._fuse_input_had(2, 1, 4096, 1024)


@pytest.mark.skipif(not T.has_triton, reason="Triton not available")
def test_prune_drops_narrow_k_tiles_when_fused():
    configs = T._exl3_gemm_configs()
    args = {"M": 1, "N": 4096, "K_dim": 12288, "K_BITS": 4}
    assert any(c.kwargs["BLOCK_K"] % 128 for c in configs)
    pruned = T._exl3_gemm_early_prune(configs, dict(args, FUSE_HAD=True))
    assert pruned and all(c.kwargs["BLOCK_K"] % 128 == 0 for c in pruned)
    # a shape whose pools collapse to the generic fallback must still be legal
    pruned = T._exl3_gemm_early_prune(
        configs, {"M": 1, "N": 4112, "K_dim": 4096, "K_BITS": 4, "FUSE_HAD": True},
    )
    assert pruned and all(c.kwargs["BLOCK_K"] % 128 == 0 for c in pruned)


# The widths the gate refuses, and the evidence for refusing them. The
# difference is a bounded fp32 reduction-rounding change, not a decode error:
# the transform is provably bit-exact (a probe kernel storing _had_x_tile's
# output matches had_r_128_triton byte for byte at every width), and what moves
# is whether the backend contracts `acc += w * x` into an FMA once the
# transform's live registers join the branch. bits 1/2/8 carry a single
# (BLOCK_N,) accumulator and are stable; 3, 4, 5, 6 and 7 carry 512-2048 fp32
# elements per CTA and are not.
def test_gate_refuses_heavy_widths(monkeypatch):
    monkeypatch.setenv("EXL3_FUSE_INPUT_HAD", "1")
    for b in (3, 4, 5, 6, 7):
        assert T._fuse_input_had(b, 1, 2560, 2560) is False, f"bits={b}"
    for b in T._IN_HAD_EXACT_BITS:
        assert T._fuse_input_had(b, 1, 2560, 2560) is True, f"bits={b}"


@pytest.mark.parametrize("K_bits", [3, 4, 5, 6, 7])
def test_input_had_drifts_at_heavy_widths(K_bits):
    """Records the drift rather than asserting equality. If a width ever starts
    passing across tiles and compiler versions, move it into
    exl3_triton._IN_HAD_EXACT_BITS and into the pinned sweep."""
    dev = device()
    torch.manual_seed(K_bits)
    K_dim, N, BN, BK, cb = 512, 256, 128, 128, 2
    trellis = make_trellis(K_dim, N, K_bits, dev)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    suh = torch.sign(torch.randn(K_dim, device=dev)).half()
    xh = torch.empty_like(x)
    T.had_r_128_triton(x, xh, suh, None, 1.0)
    y_unfused = _pinned_gemm(xh, trellis, K_bits, cb, N, K_dim, BN, BK)
    y_fused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, suh=suh)
    d = (y_fused.float() - y_unfused.float()).abs()
    assert torch.allclose(y_fused, y_unfused, rtol=2e-2, atol=2e-2), (
        f"bits={K_bits} drift is NOT bounded rounding: max|diff| {d.max().item():.3e}")
    if torch.equal(y_fused, y_unfused):
        pytest.xfail(f"bits={K_bits} is bit-exact at BN{BN}/BK{BK}; not enough to "
                     f"lift the exclusion (bits=4 is exact at BK=128 and drifts "
                     f"at BK=256, and BLOCK_K is an autotune outcome)")
    print(f"\nbits={K_bits}: ndiff {int((y_fused != y_unfused).sum())}/{y_fused.numel()} "
          f"max|diff| {d.max().item():.3e}")
