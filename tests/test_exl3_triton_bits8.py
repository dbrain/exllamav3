"""The K_BITS == 8 M == 1 branch must agree with ext.reconstruct at every tile width.

_fused_dequant_gemm_kernel's 8-bit decode sums to (nj, cl, c3) and then permutes
before the shared tail, which indexes n = 16*nj + 8*c3 + cl and so wants
(c3, nj, cl). The permute was (0, 2, 1), which is the identity for this purpose
only when NN == 1 (BLOCK_N == 16) -- the width autotune tends to pick, which is
why it went unnoticed. Every wider tile decoded to the wrong columns.

The kernel is launched through the raw JITFunction at pinned tiles: going
through exl3_gemm_triton would let autotune choose BLOCK_N == 16 and hide it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, torch, triton
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant import exl3_triton as T

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

K_DIM, N = 512, 256


def _truth(x, trellis, K_bits, cb):
    w = torch.empty((K_DIM, N), dtype = torch.half, device = "cuda")
    ext.reconstruct(w, trellis, K_bits, cb == 1, cb == 2)
    return (x.float() @ w.float()).half()


def _run_pinned(x, trellis, K_bits, cb, BN, BK, nw):
    y = torch.empty((1, N), dtype = torch.half, device = "cuda")
    perm = T._get_perm(x.device)
    mrow = T._get_m_row_offsets(K_bits, x.device) if K_bits in T._M_ROW_OFFSETS else perm
    T._fused_dequant_gemm_kernel.fn[(triton.cdiv(N, BN),)](
        x, y, trellis, perm, mrow,
        1, N, K_DIM, 1,
        x.stride(0), x.stride(1), trellis.stride(0), trellis.stride(1),
        y.stride(0), y.stride(1), y.stride(0),
        BLOCK_M = 16, BLOCK_N = BN, BLOCK_K = BK, GROUP_M = 1,
        K_BITS = K_bits, N_PACKED = trellis.shape[-1], CB = cb, M1 = True, SPLITS = 1,
        num_warps = nw, num_stages = 2)
    return y


# BLOCK_N == 16 is the width where the wrong permute is accidentally correct;
# the wider tiles are the regression this guards.
@pytest.mark.parametrize("BN", [16, 32, 64, 128])
@pytest.mark.parametrize("cb", [0, 1, 2])
def test_bits8_matches_reconstruct(BN, cb):
    torch.manual_seed(0)
    K_bits = 8
    # int16 cannot hold 0..65535 directly; the repo builds these as int32 and casts
    trellis = torch.randint(0, 65536, (K_DIM // 16, N // 16, 256 * K_bits // 16),
                            dtype = torch.int32, device = "cuda").to(torch.short)
    x = torch.randn(1, K_DIM, dtype = torch.half, device = "cuda")
    ref = _truth(x, trellis, K_bits, cb)
    got = _run_pinned(x, trellis, K_bits, cb, BN, 128, 2)
    peak = ref.abs().max().item()
    err = (got.float() - ref.float()).abs().max().item()
    assert err <= 3e-3 * max(peak, 1.0), \
        f"K_bits=8 cb={cb} BLOCK_N={BN}: max|err| {err:.3e} vs peak {peak:.3f}"
