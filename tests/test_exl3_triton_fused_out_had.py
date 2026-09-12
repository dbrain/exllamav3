"""Bit-exactness of the in-kernel OUTPUT Hadamard (EXL3_FUSE_OUTPUT_HAD).

The classic decode route is had_r_128 -> gemm -> had_r_128, three launches per
LinearEXL3. EXL3_FUSE_INPUT_HAD folds the first away; this flag folds the last
one into the GEMM's store, so an eligible decode linear costs one launch.

Bit-exactness is not optional. The unfused route rounds the fp32 accumulator to
half when it stores y, then reads that half back in the Hadamard kernel, so the
fused epilogue has to round-trip through half at exactly the same point and
apply the post-scale as a half multiply. Anything else is a silent quality
change no tolerance test would catch.

The fusion needs BLOCK_N % 128 == 0 (the transform is block-diagonal over
aligned 128-column groups, so a CTA can only finish columns it owns), which
prunes the narrow-N decode tiles. That is a throughput trade, not a
correctness one, and _fuse_output_had's host gate decides it.
"""
import os

import pytest
import torch

from exllamav3.modules.quant import exl3_triton as T

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not T.has_triton, reason="Triton not available"),
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


def _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, svh=None, num_warps=2):
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
        # constexpr divisibility gate, exactly as the host entry passes it.
        # Without it FULL is a runtime tensor and Triton compiles all five
        # decode branches into every binary -- ~3 min per case (the campaign
        # brief's dead-branch finding), which is what made this file unrunnable.
        N_DIV=T._gate_div(N), K_DIV=T._gate_div(K_dim),
        FUSE_OUT_HAD=svh is not None,
        svh_ptr=svh,
        had_r_scale=T._RSCALE_128,
        num_warps=num_warps, num_stages=2,
    )
    return y


# (K_dim, N): 512 divides both tile depths (fast paths); 384 does not divide
# BLOCK_K=256, which routes that case through the generic gather path. N is a
# multiple of 128 in both, which the output fusion requires.
PINNED_SHAPES = [(512, 256), (384, 256)]


# This checkpoint is bits 4 and 6 with codebook mul1 (cb=2), so those get the
# tile and shape sweep; every other width gets one representative case, which is
# all that is needed to keep its branch honest. cb 0/1 ride along on one width.
@pytest.mark.parametrize("K_bits", [4, 6])
@pytest.mark.parametrize("cb", [2])
@pytest.mark.parametrize("BN,BK", [(128, 128), (256, 128), (128, 256)])
@pytest.mark.parametrize("K_dim,N", PINNED_SHAPES)
def test_fused_out_had_bit_exact_pinned(K_dim, N, BN, BK, cb, K_bits):
    """Pinned BLOCK_N/BLOCK_K so both arms provably enter the same branch."""
    dev = device()
    torch.manual_seed(K_dim * 31 + N + K_bits * 7 + BN + BK + cb)
    trellis = make_trellis(K_dim, N, K_bits, dev)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    svh = torch.sign(torch.randn(N, device=dev)).half()

    y_unfused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK)
    T.had_r_128_triton(y_unfused, y_unfused, None, svh, 1.0)
    y_fused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, svh=svh)

    assert torch.equal(y_fused, y_unfused), (
        f"K_bits={K_bits} cb={cb} BN={BN} BK={BK} shape={K_dim}x{N}: "
        f"max|diff| {(y_fused.float() - y_unfused.float()).abs().max().item():.3e}"
    )


# One representative per remaining decode branch and codebook. The output
# fusion only reads the finished accumulator, so unlike the input fusion it
# cannot perturb the decode loop -- including at bits 3 and 7, which is exactly
# why they are covered here and not excluded.
@pytest.mark.parametrize("K_bits,cb", [(1, 2), (2, 2), (3, 2), (5, 2), (7, 2),
                                       (8, 2), (4, 0), (4, 1), (6, 0)])
def test_fused_out_had_other_widths(K_bits, cb):
    dev = device()
    K_dim, N, BN, BK = 512, 256, 128, 128
    torch.manual_seed(K_bits * 17 + cb)
    trellis = make_trellis(K_dim, N, K_bits, dev)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    svh = torch.sign(torch.randn(N, device=dev)).half()
    y_unfused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK)
    T.had_r_128_triton(y_unfused, y_unfused, None, svh, 1.0)
    y_fused = _pinned_gemm(x, trellis, K_bits, cb, N, K_dim, BN, BK, svh=svh)
    assert torch.equal(y_fused, y_unfused), (
        f"K_bits={K_bits} cb={cb}: max|diff| "
        f"{(y_fused.float() - y_unfused.float()).abs().max().item():.3e}")


# K % 256 == 0 so every autotune pool member takes the same kernel branch; the
# split-K shapes exist to prove the fusion stays OFF there (that route already
# folds the output Hadamard into _m1_split_reduce_had).
LINEAR_SHAPES = [
    (2048, 4096),    # classic route
    (2048, 12288),   # classic route, large N
    (4096, 4096),    # split-K route -- output fusion must not engage
]


def _linear_arm(x, trellis, suh, svh, K, in_f, out_f, fuse, count):
    """Run one arm and count had_r_128_triton launches.

    had_r_128_triton is a plain Python function looked up as a module global by
    _linear_exl3_triton, so wrapping the module attribute genuinely intercepts
    it (rule 1 only bites on @triton.jit bodies).
    """
    y = torch.empty((1, out_f), dtype=torch.half, device=x.device)
    xh = torch.empty((1, in_f), dtype=torch.half, device=x.device)
    prev = os.environ.get("EXL3_FUSE_OUTPUT_HAD")
    os.environ["EXL3_FUSE_OUTPUT_HAD"] = "1" if fuse else "0"
    orig = T.had_r_128_triton

    def counting(*a, **kw):
        count.append(1)
        return orig(*a, **kw)

    T.had_r_128_triton = counting
    try:
        T._linear_exl3_triton(
            x, y, xh, trellis, suh, svh, K, False, False, None, in_f, out_f,
        )
    finally:
        T.had_r_128_triton = orig
        if prev is None:
            os.environ.pop("EXL3_FUSE_OUTPUT_HAD", None)
        else:
            os.environ["EXL3_FUSE_OUTPUT_HAD"] = prev
    return y


@pytest.fixture
def pinned_pool():
    """Pin the autotune pool to one config.

    FUSE_OUT_HAD is an autotune KEY, so an unpinned run compiles and benchmarks
    a whole pool per arm. These tests are about bit-exactness, not tile choice.
    """
    kern = T._fused_dequant_gemm_kernel
    prev = kern.early_config_prune
    one = [c for c in T._exl3_gemm_configs()
           if c.kwargs["BLOCK_M"] == 16 and c.kwargs["BLOCK_N"] == 128
           and c.kwargs["BLOCK_K"] == 128 and c.num_warps == 2]
    assert one, "no M1 BN128/BK128 config in the pool"
    kern.early_config_prune = lambda configs, named_args, **kw: one[:1]
    kern.cache.clear()
    try:
        yield
    finally:
        kern.early_config_prune = prev
        kern.cache.clear()


@pytest.mark.parametrize("K", [4, 6])
@pytest.mark.parametrize("in_features,out_features", LINEAR_SHAPES)
def test_fused_out_had_bit_exact_linear(pinned_pool, in_features, out_features, K):
    dev = device()
    torch.manual_seed(in_features * 13 + out_features + K)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh = torch.sign(torch.randn(in_features, device=dev)).half()
    svh = torch.sign(torch.randn(out_features, device=dev)).half()
    x = torch.randn(1, in_features, dtype=torch.half, device=dev) * 0.1

    n_off, n_on = [], []
    y_unfused = _linear_arm(x, trellis, suh, svh, K, in_features, out_features, False, n_off)
    y_fused = _linear_arm(x, trellis, suh, svh, K, in_features, out_features, True, n_on)

    eligible = T._m1_splitk_plan(1, out_features, in_features, K) == 1
    if eligible:
        # the arms must genuinely differ: the fused one issues one Hadamard fewer
        assert len(n_on) == len(n_off) - 1, (
            f"expected one fewer had_r_128 launch, got {len(n_off)} -> {len(n_on)}")
    else:
        # split-K already folds the output Hadamard into its reduce
        assert len(n_on) == len(n_off)

    assert torch.equal(y_fused, y_unfused), (
        f"K={K} {in_features}x{out_features}: max|diff| "
        f"{(y_fused.float() - y_unfused.float()).abs().max().item():.3e}"
    )


def test_fuse_output_had_gates(monkeypatch):
    monkeypatch.setenv("EXL3_FUSE_OUTPUT_HAD", "1")
    assert T._fuse_output_had(4, 1, 2560, 12288)
    assert T._fuse_output_had(6, 1, 2560, 10240)
    assert not T._fuse_output_had(4, 16, 2560, 12288)      # prefill
    assert not T._fuse_output_had(4, 1, 2560, 12352)       # N % 128
    # Starved-N shapes lose more to the forced BLOCK_N=128 tile than a launch
    # is worth; the gate refuses them.
    assert not T._fuse_output_had(6, 1, 2560, 640)
    assert not T._fuse_output_had(6, 1, 2560, 512)
    monkeypatch.setenv("EXL3_FUSE_OUTPUT_HAD", "0")
    assert not T._fuse_output_had(4, 1, 2560, 12288)
    monkeypatch.delenv("EXL3_FUSE_OUTPUT_HAD")
    assert not T._fuse_output_had(4, 1, 2560, 12288)


@pytest.mark.skipif(not T.has_triton, reason="Triton not available")
def test_prune_drops_narrow_n_tiles_when_out_fused():
    configs = T._exl3_gemm_configs()
    args = {"M": 1, "N": 12288, "K_dim": 2560, "K_BITS": 4}
    assert any(c.kwargs["BLOCK_N"] % 128 for c in configs)
    pruned = T._exl3_gemm_early_prune(configs, dict(args, FUSE_OUT_HAD=True))
    assert pruned and all(c.kwargs["BLOCK_N"] % 128 == 0 for c in pruned)
    # both fusions at once must leave a non-empty, legal pool
    pruned = T._exl3_gemm_early_prune(
        configs, dict(args, FUSE_HAD=True, FUSE_OUT_HAD=True))
    assert pruned and all(c.kwargs["BLOCK_N"] % 128 == 0
                          and c.kwargs["BLOCK_K"] % 128 == 0 for c in pruned)
