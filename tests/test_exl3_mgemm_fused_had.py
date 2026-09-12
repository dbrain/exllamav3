"""Bit-exactness of the fused Hadamards in the grouped (MoE) EXL3 GEMM.

A routed decode projection costs three grouped launches today:
had_r_128_mtriton -> exl3_mgemm_triton -> had_r_128_mtriton. The two flags here
fold the transforms into the GEMM:

    EXL3_FUSE_MGEMM_IN_HAD   input  Hadamard inside the K loop  (BLOCK_K % 128)
    EXL3_FUSE_MGEMM_OUT_HAD  output Hadamard in the store       (BLOCK_N % 128)

The grouped Hadamards are tiny (a top-10 routed set at K=2560 moves 51 KB, at
N=640 moves 13 KB), so their cost is essentially the fixed per-launch cost and
removing them is the point. That only holds if the fused result is BITWISE the
same as the three-launch route -- the transforms wrap a quantized GEMM, and a
few ulps of drift there is a silent quality change.

MEASURED, and it splits the two flags apart:

  OUT is exact at every width and tile. It only reads the finished accumulator,
      so it cannot perturb the decode loop.
  IN  is exact only at bits 1, 2 and 8. The transform itself is provably exact
      (a probe kernel storing _had_x_tile's output matches had_r_128_mtriton
      byte for byte at every width), but holding its live registers across the
      K loop changes how the backend rounds the fp32 reduction -- FMA
      contraction flips -- and every heavy-accumulator width drifts:
        bits=6 2560x640  ndiff 1686/2560  max|diff| 1.172e-02
        bits=6 640x2560  ndiff 6511/10240 max|diff| 7.812e-03
        bits=5  512x256  ndiff  762/1024  max|diff| 7.812e-03
        bits=4 @ BK=256  ndiff  986/2560  max|diff| 1.562e-02
      So _fuse_mgemm_in_had refuses them, and on a 4.05bpw checkpoint (bits 4
      and 6) the input fusion never fires at all.

Runtime budget. Two things previously made this file unusable:

  * ``_pinned`` did not pass N_DIV/K_DIV, so ``FULL`` was a runtime tensor and
    Triton compiled ALL FIVE decode branches into every binary -- the campaign
    brief's dead-branch finding, ~3 min per compile. Production always passes
    them (via _gate_div), so passing them here is both faster AND more
    representative.
  * the matrix was 162 cases. This checkpoint is bits 4 and 6 only, so those
    two get the real MoE shapes and every tile; the other widths get one
    representative case each, which is all that is needed to keep their copied
    branches honest. A full run is now ~2 minutes.

The kernels here are generated from exl3_triton.py by gen_exl3_mgemm_triton.py;
test_mgemm_generated_matches_source keeps the checked-in file honest.
"""
import os
import subprocess
import sys

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

E = 4
CB = 2          # mul1: what the flashnext-4.05bpw checkpoint uses


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


class ExpertSet:
    """Weights plus their int64 pointer tables.

    The tables hold raw data_ptr() values, so the tensors they point at MUST
    outlive them. Keeping them as attributes of one object is the whole reason
    this class exists: an earlier version built the scale vectors in a helper
    and returned only the tables, so every suh/svh was freed to the caching
    allocator at the helper's return and the next torch.empty handed the same
    blocks out again. Both arms then read whatever had most recently been
    written there, and every test in this file failed.
    """

    def __init__(self, K_dim, N, K_bits, dev, seed):
        torch.manual_seed(seed)
        self.trellis = [
            torch.randint(0, 65536, (K_dim // 16, N // 16, 16 * K_bits),
                          dtype=torch.int32, device=dev).to(torch.short).contiguous()
            for _ in range(E)]
        self.suh = [torch.sign(torch.randn(K_dim, device=dev)).half().contiguous()
                    for _ in range(E)]
        self.svh = [torch.sign(torch.randn(N, device=dev)).half().contiguous()
                    for _ in range(E)]
        self.ids = torch.arange(E, dtype=torch.int32, device=dev)
        self.p_trellis = self._table(self.trellis)
        self.p_suh = self._table(self.suh)
        self.p_svh = self._table(self.svh)

    @staticmethod
    def _table(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype=torch.long,
                            device=ts[0].device)


def _pinned(s, x, K_bits, N, K_dim, BN, BK, fuse_in=False, fuse_out=False,
            num_warps=2):
    dev = x.device
    perm = MG._get_perm_i(dev)
    mrow = MG._get_m_row_offsets(K_bits, dev) if K_bits in MG._M_ROW_OFFSETS else perm
    y = torch.empty((E, N), dtype=torch.half, device=dev)
    stride_tn = 16 * K_bits
    stride_tk = (N // 16) * stride_tn
    MG._grouped_dequant_gemv_kernel.fn[(E * triton.cdiv(N, BN),)](
        x, y, s.p_trellis, s.ids, perm, mrow,
        E, N, K_dim, 4,
        x.stride(0) if x.shape[0] == E else 0, x.stride(1),
        stride_tk, stride_tn,
        y.stride(0), y.stride(1),
        BLOCK_N=BN, BLOCK_K=BK,
        K_BITS=K_bits, N_PACKED=16 * K_bits, CB=CB,
        # constexpr divisibility gate, exactly as the host entry passes it
        N_DIV=T._gate_div(N), K_DIV=T._gate_div(K_dim),
        FUSE_HAD=fuse_in, ptrs_suh=s.p_suh if fuse_in else None,
        FUSE_OUT_HAD=fuse_out, ptrs_svh=s.p_svh if fuse_out else None,
        had_r_scale=MG._RSCALE_128,
        num_warps=num_warps, num_stages=2,
    )
    return y


def _reference(s, x, K_bits, N, K_dim, BN, BK):
    """The three-launch route: grouped had -> mgemm -> grouped had."""
    dev = x.device
    xh = torch.empty((E, 1, K_dim), dtype=torch.half, device=dev)
    MG.had_r_128_mtriton(x.unsqueeze(0).expand(E, -1, -1), xh, s.p_suh, None, s.ids, 1.0)
    y = _pinned(s, xh.view(E, K_dim), K_bits, N, K_dim, BN, BK)
    pre_out = y.clone()
    MG.had_r_128_mtriton(y.unsqueeze(1), y.unsqueeze(1), None, s.p_svh, s.ids, 1.0)
    return xh, pre_out, y


def _check(tag, got, ref):
    d = (got.float() - ref.float()).abs()
    assert torch.equal(got, ref), (
        f"{tag}: ndiff {int((got != ref).sum())}/{got.numel()} "
        f"max|diff| {d.max().item():.4e} (ref |max| {ref.float().abs().max().item():.3f})")


# The two real routed-expert shapes: gate/up (2560 -> 640) and down (640 -> 2560).
MOE_SHAPES = [(2560, 640), (640, 2560)]
# BLOCK_K=256 needs K % 256; 640 does not divide it, so the down shape exercises
# a narrower pool. Both tiles satisfy BLOCK_N % 128 and BLOCK_K % 128.
TILES = [(128, 128), (128, 256)]


# The OUTPUT fusion only reads the finished accumulator, so it cannot perturb
# the decode loop's register pressure and is exact at every width.
@pytest.mark.parametrize("K_bits", [4, 6])
@pytest.mark.parametrize("K_dim,N", MOE_SHAPES)
def test_mgemm_fused_out_had_bit_exact(K_dim, N, K_bits):
    """Production widths and shapes, pinned tile so both arms take one branch."""
    dev = device()
    BN, BK = 128, 128
    s = ExpertSet(K_dim, N, K_bits, dev, K_dim * 31 + N + K_bits)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    xh, _, ref = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, xh.view(E, K_dim), K_bits, N, K_dim, BN, BK, fuse_out=True)
    _check(f"fused-out {K_dim}x{N} bits={K_bits}", got, ref)


@pytest.mark.parametrize("BN,BK", TILES)
def test_mgemm_fused_out_had_tiles(BN, BK):
    """Tile sweep on one production shape (K % 256, so BLOCK_K=256 is legal)."""
    dev = device()
    K_dim, N, K_bits = 2560, 640, 4
    s = ExpertSet(K_dim, N, K_bits, dev, BN * 7 + BK)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    xh, _, ref = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, xh.view(E, K_dim), K_bits, N, K_dim, BN, BK, fuse_out=True)
    _check(f"BN{BN}/BK{BK}", got, ref)


# The INPUT fusion is bit-exact only at the light-accumulator widths; see
# exl3_triton._fuse_input_had for the mechanism and _fuse_mgemm_in_had for the
# grouped restriction. These are the widths the gate lets through.
@pytest.mark.parametrize("K_bits", list(MG._IN_HAD_EXACT_BITS))
def test_mgemm_fused_in_had_bit_exact(K_bits):
    dev = device()
    K_dim, N, BN, BK = 512, 256, 128, 128
    s = ExpertSet(K_dim, N, K_bits, dev, K_bits)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    _, ref_pre, _ = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, x, K_bits, N, K_dim, BN, BK, fuse_in=True)
    _check(f"fused-in bits={K_bits}", got, ref_pre)
    both = _pinned(s, x, K_bits, N, K_dim, BN, BK, fuse_in=True, fuse_out=True)
    _, _, ref = _reference(s, x, K_bits, N, K_dim, BN, BK)
    _check(f"fused-both bits={K_bits}", both, ref)


# The widths the gate REFUSES, and why. Recorded as measurements rather than
# asserted equal: the difference is a bounded fp32 reduction-rounding change
# (the transform itself is provably bit-exact -- a probe kernel storing
# _had_x_tile's output matches had_r_128_mtriton byte for byte), not a decode
# error, but it is not bitwise identity and so it does not ship.
@pytest.mark.parametrize("K_bits", [3, 4, 5, 6, 7])
def test_mgemm_fused_in_had_drifts_at_heavy_widths(K_bits):
    dev = device()
    K_dim, N, BN, BK = 512, 256, 128, 128
    assert not MG._fuse_mgemm_in_had(K_bits, K_dim), \
        f"bits={K_bits} must be refused by the gate"
    s = ExpertSet(K_dim, N, K_bits, dev, K_bits)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    _, ref_pre, _ = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, x, K_bits, N, K_dim, BN, BK, fuse_in=True)
    d = (got.float() - ref_pre.float()).abs()
    # bounded: a reduction-rounding change, never a decode error
    assert torch.allclose(got, ref_pre, rtol=2e-2, atol=2e-2), (
        f"bits={K_bits} drift is NOT bounded rounding: max|diff| {d.max().item():.3e}")
    if torch.equal(got, ref_pre):
        pytest.xfail(f"bits={K_bits} is bit-exact here; if it holds across tiles "
                     f"and compiler versions, add it to _IN_HAD_EXACT_BITS")
    print(f"\nbits={K_bits}: ndiff {int((got != ref_pre).sum())}/{got.numel()} "
          f"max|diff| {d.max().item():.3e}")


# One representative per remaining decode branch, for the OUTPUT fusion:
#   1 -> the (1, 2, 8) branch, 5 -> the (5, 7) branch, 8 -> its own permutation,
#   3 -> its own run-funnel branch. All must be exact.
@pytest.mark.parametrize("K_bits", [1, 2, 3, 5, 7, 8])
def test_mgemm_fused_out_had_other_widths(K_bits):
    dev = device()
    K_dim, N, BN, BK = 512, 256, 128, 128
    s = ExpertSet(K_dim, N, K_bits, dev, K_bits)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    xh, _, ref = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, xh.view(E, K_dim), K_bits, N, K_dim, BN, BK, fuse_out=True)
    _check(f"bits={K_bits}", got, ref)


def test_mgemm_fused_had_generic_branch():
    """K % BLOCK_K != 0 routes both arms through the staged-gather branch."""
    dev = device()
    K_dim, N, K_bits, BN, BK = 384, 256, 4, 128, 256
    assert T._gate_div(K_dim) % BK != 0, "shape no longer reaches the generic branch"
    s = ExpertSet(K_dim, N, K_bits, dev, 99)
    x = torch.randn(1, K_dim, dtype=torch.half, device=dev) * 0.1
    xh, _, ref = _reference(s, x, K_bits, N, K_dim, BN, BK)
    got = _pinned(s, xh.view(E, K_dim), K_bits, N, K_dim, BN, BK, fuse_out=True)
    _check("generic branch", got, ref)


@pytest.fixture
def pinned_pool():
    """Pin the autotune pool to one config for the whole-linear tests.

    Autotune benchmarks (and therefore compiles) every surviving config, and
    FUSE_HAD/FUSE_OUT_HAD are autotune KEYS, so an unpinned run compiles the
    pool once per flag combination. The tests here are about bit-exactness, not
    tile choice.
    """
    kern = MG._grouped_dequant_gemv_kernel
    prev = kern.early_config_prune
    one = [c for c in MG._mgemm_configs()
           if c.kwargs["BLOCK_N"] == 128 and c.kwargs["BLOCK_K"] == 128]
    assert one, "no BN128/BK128 config in the mgemm pool"
    kern.early_config_prune = lambda configs, named_args, **kw: one[:1]
    kern.cache.clear()
    try:
        yield
    finally:
        kern.early_config_prune = prev
        kern.cache.clear()


def _linear_arm(s, x, K_bits, in_f, out_f, fin, fout, count):
    dev = x.device
    y = torch.empty((E, out_f), dtype=torch.half, device=dev)
    xh = torch.empty((E, 1, in_f), dtype=torch.half, device=dev)
    prev = {k: os.environ.get(k) for k in
            ("EXL3_FUSE_MGEMM_IN_HAD", "EXL3_FUSE_MGEMM_OUT_HAD")}
    os.environ["EXL3_FUSE_MGEMM_IN_HAD"] = "1" if fin else "0"
    os.environ["EXL3_FUSE_MGEMM_OUT_HAD"] = "1" if fout else "0"
    orig = MG.had_r_128_mtriton

    def counting(*a, **kw):
        count.append(1)
        return orig(*a, **kw)

    MG.had_r_128_mtriton = counting
    try:
        MG._linear_exl3_mgemm_triton(x, xh, y, s.p_trellis, s.p_suh, s.p_svh,
                                     s.ids, K_bits, CB)
    finally:
        MG.had_r_128_mtriton = orig
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return y


@pytest.mark.parametrize("in_features,out_features", MOE_SHAPES)
def test_mgemm_fused_out_had_linear(pinned_pool, in_features, out_features):
    """Whole routed projection through the host entry, both flags flipped on.

    bits=4 is a width the input gate refuses, so this also proves the gate is
    what decides: with BOTH env flags set, exactly one grouped Hadamard launch
    disappears, not two.
    """
    dev = device()
    K_bits = 4
    s = ExpertSet(in_features, out_features, K_bits, dev,
                  in_features + out_features + K_bits)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev) * 0.1

    n_ref, n_got = [], []
    ref = _linear_arm(s, x, K_bits, in_features, out_features, False, False, n_ref)
    got = _linear_arm(s, x, K_bits, in_features, out_features, True, True, n_got)

    assert len(n_ref) == 2
    assert len(n_got) == 1, (
        f"expected 1 grouped Hadamard launch (input fusion refused at bits=4), "
        f"got {len(n_got)}")
    _check(f"{in_features}x{out_features} out-fused", got, ref)


def test_mgemm_fuse_gates(monkeypatch):
    monkeypatch.delenv("EXL3_FUSE_MGEMM_IN_HAD", raising=False)
    monkeypatch.delenv("EXL3_FUSE_MGEMM_OUT_HAD", raising=False)
    assert not MG._fuse_mgemm_in_had(2, 2560)
    assert not MG._fuse_mgemm_out_had(2560)
    monkeypatch.setenv("EXL3_FUSE_MGEMM_IN_HAD", "1")
    monkeypatch.setenv("EXL3_FUSE_MGEMM_OUT_HAD", "1")
    assert MG._fuse_mgemm_in_had(2, 2560) and MG._fuse_mgemm_in_had(8, 640)
    assert MG._fuse_mgemm_out_had(2560) and MG._fuse_mgemm_out_had(640)
    # the widths this checkpoint uses are refused: not bit-exact
    for b in (3, 4, 5, 6, 7):
        assert not MG._fuse_mgemm_in_had(b, 2560), f"bits={b} must be refused"
    assert not MG._fuse_mgemm_in_had(2, 96)    # gdn.in_proj_ba class
    assert not MG._fuse_mgemm_out_had(96)


@pytest.mark.parametrize("fused", [False, True])
def test_mgemm_prune_keeps_a_legal_pool(fused):
    cfgs = MG._mgemm_configs()
    for K_dim, N in MOE_SHAPES:
        kw = {"E": 10, "E_BUCKET": 16, "N": N, "K_dim": K_dim, "K_BITS": 4}
        if fused:
            kw.update(FUSE_HAD=True, FUSE_OUT_HAD=True)
        pool = MG._mgemm_prune(cfgs, kw)
        assert pool, f"empty pool for {K_dim}x{N} fused={fused}"
        if fused:
            assert all(c.kwargs["BLOCK_N"] % 128 == 0 and c.kwargs["BLOCK_K"] % 128 == 0
                       for c in pool)


def test_mgemm_generated_matches_source(tmp_path):
    """The checked-in exl3_mgemm_triton.py must be what the generator emits.

    It went stale once already (the generator's leak guard aborts BEFORE
    writing, so a failed run leaves the old file in place and nothing says so).
    """
    import shutil
    here = os.path.dirname(os.path.abspath(MG.__file__))
    gen = os.path.join(here, "gen_exl3_mgemm_triton.py")
    live = os.path.join(here, "exl3_mgemm_triton.py")
    keep = tmp_path / "exl3_mgemm_triton.py"
    shutil.copy(live, keep)
    try:
        r = subprocess.run([sys.executable, gen], cwd=here,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"generator failed:\n{r.stderr}"
        assert open(live).read() == open(keep).read(), (
            "exl3_mgemm_triton.py is stale; rerun gen_exl3_mgemm_triton.py")
    finally:
        shutil.copy(keep, live)
