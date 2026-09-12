"""EXL3_FUSE_MOE_GATE_UP: gate_proj and up_proj as one grouped mgemm.

At decode both projections read the SAME hidden row and have identical shapes;
only the weights differ. Running them as one launch over 2E programs instead of
two launches over E halves the routed MLP's launch count for the first two
projections -- with the output-Hadamard fusion on, four launches become two.

Mechanism: the pointer tables are concatenated once at load (gate block, then
up block) and the kernels derive

    eid = expert_ids[pid_e % E_HALF] + (pid_e // E_HALF) * PTRS_SPLIT

so the second half of the grid indexes the up block with the same routed ids.
PTRS_SPLIT == 0 is the unmerged path and compiles to the original single load.

Why this should be bit-exact, and what could break it. Each output row is
computed by the same code over the same K loop with the same per-CTA
accumulator -- only the grid widens, so nothing about a CTA's register pressure
or reduction order changes. The one real hazard is autotune: E_BUCKET is part
of the autotune key, so a merged call at 2E would otherwise mint a fresh entry
and could pick a different BLOCK_N/BLOCK_K, and the heavy-accumulator branches
DO fold their final reduction differently per tile. The host therefore passes
the per-projection bucket, keeping the key -- and so the tile -- identical.
tests here pin the tile anyway, and test_merge_uses_the_unmerged_autotune_key
holds the host side of that.
"""
import os

import pytest
import torch

from exllamav3.modules.quant import exl3_mgemm_triton as MG
from exllamav3.modules.quant import exl3_triton as T

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not T.has_triton, reason="Triton not available"),
]

if T.has_triton:
    import triton

E = 4               # routed set
NUM_EXPERTS = 6     # pointer table length per projection
CB = 2


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


class Proj:
    """One projection's weights for NUM_EXPERTS experts, plus pointer tables."""

    def __init__(self, K_dim, N, K_bits, dev, seed):
        torch.manual_seed(seed)
        self.trellis = [
            torch.randint(0, 65536, (K_dim // 16, N // 16, 16 * K_bits),
                          dtype=torch.int32, device=dev).to(torch.short).contiguous()
            for _ in range(NUM_EXPERTS)]
        self.suh = [torch.sign(torch.randn(K_dim, device=dev)).half().contiguous()
                    for _ in range(NUM_EXPERTS)]
        self.svh = [torch.sign(torch.randn(N, device=dev)).half().contiguous()
                    for _ in range(NUM_EXPERTS)]

    @staticmethod
    def table(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype=torch.long,
                            device=ts[0].device)

    def tables(self):
        return self.table(self.trellis), self.table(self.suh), self.table(self.svh)


def _gemv(x, p_tr, ids, K_bits, N, K_dim, BN, BK, n_prog, ptrs_split=0,
          e_half=0, p_svh=None):
    dev = x.device
    perm = MG._get_perm_i(dev)
    mrow = MG._get_m_row_offsets(K_bits, dev) if K_bits in MG._M_ROW_OFFSETS else perm
    y = torch.empty((n_prog, N), dtype=torch.half, device=dev)
    stride_tn = 16 * K_bits
    stride_tk = (N // 16) * stride_tn
    MG._grouped_dequant_gemv_kernel.fn[(n_prog * triton.cdiv(N, BN),)](
        x, y, p_tr, ids, perm, mrow,
        n_prog, N, K_dim, 4,
        0, x.stride(1),                       # broadcast row: expert stride 0
        stride_tk, stride_tn,
        y.stride(0), y.stride(1),
        BLOCK_N=BN, BLOCK_K=BK,
        K_BITS=K_bits, N_PACKED=16 * K_bits, CB=CB,
        N_DIV=T._gate_div(N), K_DIV=T._gate_div(K_dim),
        FUSE_HAD=False, ptrs_suh=None,
        FUSE_OUT_HAD=p_svh is not None, ptrs_svh=p_svh,
        had_r_scale=MG._RSCALE_128,
        PTRS_SPLIT=ptrs_split, E_HALF=e_half,
        num_warps=2, num_stages=2,
    )
    return y


MOE_SHAPES = [(2560, 640), (640, 2560)]


@pytest.mark.parametrize("fuse_out", [False, True])
@pytest.mark.parametrize("K_bits", [4, 6])
@pytest.mark.parametrize("K_dim,N", MOE_SHAPES)
def test_gate_up_merge_is_bit_exact(K_dim, N, K_bits, fuse_out):
    dev = device()
    BN, BK = 128, 128
    gate = Proj(K_dim, N, K_bits, dev, K_dim + K_bits)
    up = Proj(K_dim, N, K_bits, dev, K_dim + K_bits + 777)
    g_tr, g_suh, g_svh = gate.tables()
    u_tr, u_suh, u_svh = up.tables()
    ids = torch.tensor([3, 0, 5, 1], dtype=torch.long, device=dev)[:E]
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1

    ref_g = _gemv(x, g_tr, ids, K_bits, N, K_dim, BN, BK, E,
                  p_svh=g_svh if fuse_out else None)
    ref_u = _gemv(x, u_tr, ids, K_bits, N, K_dim, BN, BK, E,
                  p_svh=u_svh if fuse_out else None)

    m_tr = torch.cat((g_tr, u_tr))
    m_svh = torch.cat((g_svh, u_svh))
    got = _gemv(x, m_tr, ids, K_bits, N, K_dim, BN, BK, 2 * E,
                ptrs_split=NUM_EXPERTS, e_half=E,
                p_svh=m_svh if fuse_out else None)

    for half, ref, tag in ((got[:E], ref_g, "gate"), (got[E:], ref_u, "up")):
        assert torch.equal(half, ref), (
            f"{tag} {K_dim}x{N} bits={K_bits} fuse_out={fuse_out}: ndiff "
            f"{int((half != ref).sum())}/{ref.numel()} max|diff| "
            f"{(half.float() - ref.float()).abs().max().item():.4e}")


def test_ptrs_split_zero_is_the_unmerged_path():
    """PTRS_SPLIT=0 must reproduce the plain single-table launch exactly."""
    dev = device()
    K_dim, N, K_bits, BN, BK = 2560, 640, 4, 128, 128
    gate = Proj(K_dim, N, K_bits, dev, 5)
    g_tr, _, g_svh = gate.tables()
    ids = torch.tensor([3, 0, 5, 1], dtype=torch.long, device=dev)[:E]
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    a = _gemv(x, g_tr, ids, K_bits, N, K_dim, BN, BK, E)
    b = _gemv(x, g_tr, ids, K_bits, N, K_dim, BN, BK, E, ptrs_split=0, e_half=0)
    assert torch.equal(a, b)


def test_grouped_hadamard_merge_is_bit_exact():
    """The input Hadamard over 2E rows must match two E-row launches."""
    dev = device()
    K_dim = 2560
    gate = Proj(K_dim, 640, 4, dev, 11)
    up = Proj(K_dim, 640, 4, dev, 12)
    _, g_suh, _ = gate.tables()
    _, u_suh, _ = up.tables()
    ids = torch.tensor([3, 0, 5, 1], dtype=torch.long, device=dev)[:E]
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    xb = x.unsqueeze(0).expand(E, -1, -1)

    ref = torch.empty((2 * E, 1, K_dim), dtype=torch.half, device=dev)
    MG.had_r_128_mtriton(xb, ref[:E], g_suh, None, ids, 1.0)
    MG.had_r_128_mtriton(xb, ref[E:], u_suh, None, ids, 1.0)

    got = torch.empty((2 * E, 1, K_dim), dtype=torch.half, device=dev)
    MG.had_r_128_mtriton(x.unsqueeze(0).expand(2 * E, -1, -1), got,
                         torch.cat((g_suh, u_suh)), None, ids, 1.0,
                         ptrs_split=NUM_EXPERTS, e_half=E)
    assert torch.equal(got, ref), (
        f"ndiff {int((got != ref).sum())}/{ref.numel()} max|diff| "
        f"{(got.float() - ref.float()).abs().max().item():.4e}")


def test_merge_uses_the_unmerged_autotune_key():
    """E_BUCKET must not move when the grid doubles.

    It is an autotune key, so a merged call reporting bucket(2E) could pick a
    different tile -- and the heavy-accumulator branches fold their final
    reduction per tile, so that would change the output bits.
    """
    assert MG._e_bucket(10) == MG._e_bucket(10)
    assert MG._e_bucket(20) != MG._e_bucket(10), "test is vacuous if these match"
    # the host must divide the reported count back down when merging
    assert MG._merged_e_bucket(20, 2) == MG._e_bucket(10)
    assert MG._merged_e_bucket(10, 1) == MG._e_bucket(10)


def test_gate_up_merge_gate(monkeypatch):
    monkeypatch.delenv("EXL3_FUSE_MOE_GATE_UP", raising=False)
    assert not MG.fuse_gate_up()
    monkeypatch.setenv("EXL3_FUSE_MOE_GATE_UP", "1")
    assert MG.fuse_gate_up()
    monkeypatch.setenv("EXL3_FUSE_MOE_GATE_UP", "off")
    assert not MG.fuse_gate_up()


@pytest.mark.parametrize("K_dim,N", MOE_SHAPES)
def test_merged_host_path_matches_two_unmerged_calls(K_dim, N, monkeypatch):
    """The whole host entry point, at the default flag state.

    The kernel-level tests above all drive the shared decode row, which is what
    the merged path uses when the input Hadamard is fused into the GEMM. With
    EXL3_FUSE_MGEMM_IN_HAD off -- the default, and permanently so at bits 4/6 --
    the grouped Hadamard writes 2E distinct rows into xh and the GEMM reads one
    row per program instead of broadcasting one. That is a second, untested
    shape through the same entry point, and it is the shape a default load
    takes.
    """
    monkeypatch.delenv("EXL3_FUSE_MGEMM_IN_HAD", raising=False)
    dev = device()
    K_bits = 4
    gate = Proj(K_dim, N, K_bits, dev, 21)
    up = Proj(K_dim, N, K_bits, dev, 22)
    g_tr, g_suh, g_svh = gate.tables()
    u_tr, u_suh, u_svh = up.tables()
    ids = torch.tensor([3, 0, 5, 1], dtype=torch.long, device=dev)[:E]
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1

    xh = torch.empty((E, 1, K_dim), dtype=torch.half, device=dev)
    ref = torch.empty((2 * E, N), dtype=torch.half, device=dev)
    MG._linear_exl3_mgemm_triton(x, xh, ref[:E], g_tr, g_suh, g_svh, ids, K_bits, CB)
    MG._linear_exl3_mgemm_triton(x, xh, ref[E:], u_tr, u_suh, u_svh, ids, K_bits, CB)

    xh2 = torch.empty((2 * E, 1, K_dim), dtype=torch.half, device=dev)
    got = torch.empty((2 * E, N), dtype=torch.half, device=dev)
    MG._linear_exl3_mgemm_gate_up(
        x, xh2, got, torch.cat((g_tr, u_tr)), torch.cat((g_suh, u_suh)),
        torch.cat((g_svh, u_svh)), ids, K_bits, CB, NUM_EXPERTS)

    for half, r, tag in ((got[:E], ref[:E], "gate"), (got[E:], ref[E:], "up")):
        assert torch.equal(half, r), (
            f"{tag} {K_dim}x{N}: ndiff {int((half != r).sum())}/{r.numel()} "
            f"max|diff| {(half.float() - r.float()).abs().max().item():.4e}")


def test_warmup_runs_at_default_flags(monkeypatch):
    """load_local calls this unconditionally; it must not need any flag set."""
    monkeypatch.delenv("EXL3_FUSE_MGEMM_IN_HAD", raising=False)
    monkeypatch.delenv("EXL3_FUSE_MOE_GATE_UP", raising=False)
    dev = device()
    K_dim, N, K_bits = 2560, 640, 4
    gate, up = Proj(K_dim, N, K_bits, dev, 31), Proj(K_dim, N, K_bits, dev, 32)
    tabs = [torch.cat(t) for t in zip(gate.tables(), up.tables())]
    MG.mgemm_prepare_gate_up(E, K_dim, N, K_bits, CB, dev, *tabs, NUM_EXPERTS)
