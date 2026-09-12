"""The constexpr fast-path gate must actually fold, on every production decode shape.

`_fused_dequant_gemm_kernel` and `_grouped_dequant_gemv_kernel` select their
decode fast path on `N % BLOCK_N == 0 and K_dim % BLOCK_K == 0`. N and K_dim are
runtime kernel arguments, so without the gate that predicate is a Triton tensor
and Triton compiles EVERY decode branch into EVERY binary. The dead branches
never execute but they set the register allocation: measured on gfx1150 at
bits=6 / BN32 / BK128, 256 VGPR with 291 VGPR spills and 676 B/lane of scratch,
against 123 VGPR and no spills at all once the predicate folds.

Nothing here launches a kernel or allocates a tensor -- every compile goes
through JITFunction.warmup with dtype stand-ins -- but Triton still needs a
target, hence the GPU skip.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton

from exllamav3.modules.quant import exl3_triton as T
from exllamav3.modules.quant import exl3_mgemm_triton as G

# (label, K_dim, N, bits) -- Flash-Next 4.05bpw decode, from its
# quantization_config.json tensor_storage shapes.
DENSE_SHAPES = [
    ("shared.gate_proj", 2560, 640, 6),
    ("gdn.in_proj_z", 2560, 6144, 6),
    ("gdn.in_proj_qkv", 2560, 10240, 6),
    ("attn.q_proj", 2560, 12288, 6),
    ("lm_head", 2560, 248320, 6),
    ("moe.gate_proj", 2560, 640, 4),
    ("moe.up_proj", 2560, 640, 4),
    ("moe.down_proj", 640, 2560, 4),
]
# The grouped kernel sees the routed set in one launch: top-k 10.
MOE_E = 10
MOE_SHAPES = [
    ("moe.gate_proj", 2560, 640, 4),
    ("moe.up_proj", 2560, 640, 4),
    ("moe.down_proj", 640, 2560, 4),
]
CB = 2  # mul1, the codebook this model was quantized with


def _meta(asm):
    def g(pat, d=0):
        m = re.search(pat, asm)
        return int(m.group(1)) if m else d
    return dict(
        vgpr=g(r"\.vgpr_count:\s+(\d+)"),
        agpr=g(r"num_agpr,\s*(\d+)"),
        vspill=g(r"\.vgpr_spill_count:\s+(\d+)"),
        sspill=g(r"\.sgpr_spill_count:\s+(\d+)"),
        scratch=g(r"\.private_segment_fixed_size:\s+(\d+)"),
        lds=g(r"\.group_segment_fixed_size:\s+(\d+)"),
        occ=g(r"; Occupancy:\s+(\d+)"),
        codelen=g(r"; codeLenInByte = (\d+)"),
    )


def _dense(K, N, bits, cfg, gated):
    a = cfg.kwargs
    k = T._fused_dequant_gemm_kernel.fn.warmup(
        torch.float16, torch.float16, torch.int16, torch.int64, torch.int32,
        1, N, K, T._m_bucket(1),
        K, 1, (N // 16) * 16 * bits, 16 * bits, N, 1, N,
        BLOCK_M=a["BLOCK_M"], BLOCK_N=a["BLOCK_N"], BLOCK_K=a["BLOCK_K"],
        GROUP_M=a["GROUP_M"], K_BITS=bits, N_PACKED=16 * bits, CB=CB,
        M1=True, SPLITS=1,
        N_DIV=T._gate_div(N) if gated else 0,
        K_DIV=T._gate_div(K) if gated else 0,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages, grid=(1,))
    return _meta(k.asm["amdgcn"])


def _grouped(K, N, bits, cfg, gated):
    a = cfg.kwargs
    k = G._grouped_dequant_gemv_kernel.fn.warmup(
        torch.float16, torch.float16, torch.int64, torch.int32,
        torch.int64, torch.int32,
        MOE_E, N, K, G._e_bucket(MOE_E),
        0, 1, (N // 16) * 16 * bits, 16 * bits, N, 1,
        BLOCK_N=a["BLOCK_N"], BLOCK_K=a["BLOCK_K"],
        K_BITS=bits, N_PACKED=16 * bits, CB=CB,
        N_DIV=G._gate_div(N) if gated else 0,
        K_DIV=G._gate_div(K) if gated else 0,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages, grid=(1,))
    return _meta(k.asm["amdgcn"])


def _check(why, off, on, results):
    """Two invariants, because the pool is not uniform.

    Every pruned config must be monotonically better with the gate on -- that is
    what catches the dead branch coming back. The headline claim (no spills, no
    scratch, twice the waves per SIMD) is asserted over the pool rather than per
    config: the deep-K tiles land at 146 VGPR and cap at 6 waves, and the b4
    BLOCK_N=128 tile needs more than 256 VGPR for its live accumulator alone, so
    neither can reach 2x however the branch is compiled.
    """
    ctx = f"{why}: off={off} on={on}"
    assert on["vspill"] <= off["vspill"], "gate added register spills: " + ctx
    assert on["scratch"] <= off["scratch"], "gate added scratch traffic: " + ctx
    assert on["codelen"] < off["codelen"], "gate did not shrink the binary: " + ctx
    assert on["occ"] >= off["occ"], "gate lowered waves/SIMD: " + ctx
    results.append((why, off, on,
                    on["vspill"] == 0 and on["sspill"] == 0 and on["scratch"] == 0
                    and on["occ"] >= 2 * off["occ"]))


def _assert_pool(label, results):
    assert any(ok for *_, ok in results), (
        f"{label}: no pruned config reaches zero spills, zero scratch and 2x waves/SIMD\n"
        + "\n".join(f"  {w}: occ {o['occ']} -> {n['occ']}, spill {o['vspill']} -> {n['vspill']}, "
                     f"scratch {o['scratch']} -> {n['scratch']}"
                     for w, o, n, _ in results))


def _ledger(rows):
    path = os.environ.get("EXL3_GATE_LEDGER")
    if not path:
        return
    import csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["kernel", "shape", "E", "K", "N", "bits", "BN", "BK", "nw", "ns",
                        "gate", "vgpr", "agpr", "vspill", "sspill", "scratch", "lds",
                        "occ_llvm", "codelen"])
        w.writerows(rows)


# --------------------------------------------------------------------------
# CPU-only: the substitution is exact, and the flag is read per call
# --------------------------------------------------------------------------

def test_capped_valuation_is_exact():
    """min(256, v & -v) % B == 0 must agree with v % B == 0 for every pool tile."""
    blocks = sorted({c.kwargs["BLOCK_N"] for c in T._exl3_gemm_configs()}
                    | {c.kwargs["BLOCK_K"] for c in T._exl3_gemm_configs()}
                    | {c.kwargs["BLOCK_N"] for c in G._mgemm_configs()}
                    | {c.kwargs["BLOCK_K"] for c in G._mgemm_configs()})
    assert max(blocks) <= T._GATE_CAP, \
        f"a config tile {max(blocks)} exceeds the gate cap {T._GATE_CAP}"
    vals = {n for _, k, n, _ in DENSE_SHAPES for n in (k, n)}
    vals |= {17, 96, 640, 2560, 5120, 12288, 17408, 248320, 1 << 20}
    for v in vals:
        capped = min(T._GATE_CAP, v & -v)
        for b in blocks:
            assert (capped % b == 0) == (v % b == 0), (v, b)


def test_gate_flag_is_read_per_call(monkeypatch):
    monkeypatch.delenv("EXL3_CONSTEXPR_GATE", raising=False)
    assert T._gate_div(10240) == 256 and G._gate_div(640) == 128
    for off in ("0", "off", "false", "no", "OFF"):
        monkeypatch.setenv("EXL3_CONSTEXPR_GATE", off)
        assert T._gate_div(10240) == 0 and G._gate_div(640) == 0
    monkeypatch.setenv("EXL3_CONSTEXPR_GATE", "1")
    assert T._gate_div(10240) == 256


def test_grouped_conditions_match_the_generator_source():
    """exl3_mgemm_triton.py's branch conditions are copied verbatim out of
    exl3_triton.py by gen_exl3_mgemm_triton.py; a hand edit to one file that
    does not reach the other silently diverges the two kernels."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "exllamav3", "modules", "quant")
    src = open(os.path.join(d, "exl3_triton.py")).read().split("\n")
    gen = open(os.path.join(d, "exl3_mgemm_triton.py")).read()
    conds = [l for l in src
             if re.match(r"^    (if|elif) \(?K_BITS ==", l) and "FULL" in l]
    assert len(conds) == 5, f"expected 5 gated conditions, found {len(conds)}"
    for c in conds:
        assert c in gen, f"condition not carried into the grouped kernel: {c!r}"


# --------------------------------------------------------------------------
# Compile-only: the gate must remove the spills and double the occupancy
# --------------------------------------------------------------------------

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU target")


@needs_gpu
@pytest.mark.parametrize("label,K,N,bits", DENSE_SHAPES, ids=[s[0] for s in DENSE_SHAPES])
def test_dense_gate_removes_spills_and_doubles_occupancy(label, K, N, bits):
    cfgs = T._exl3_gemm_early_prune(T._fused_dequant_gemm_kernel.configs,
                                    dict(M=1, N=N, K_dim=K, K_BITS=bits),
                                    M=1, N=N, K_dim=K, K_BITS=bits)
    assert cfgs, "the prune left no config"
    rows, results = [], []
    for cfg in cfgs:
        off = _dense(K, N, bits, cfg, gated=False)
        on = _dense(K, N, bits, cfg, gated=True)
        a = cfg.kwargs
        for tag, m in (("off", off), ("on", on)):
            rows.append(["dense", label, 1, K, N, bits, a["BLOCK_N"], a["BLOCK_K"],
                         cfg.num_warps, cfg.num_stages, tag, m["vgpr"], m["agpr"],
                         m["vspill"], m["sspill"], m["scratch"], m["lds"],
                         m["occ"], m["codelen"]])
        _check(f"{label} BN{a['BLOCK_N']} BK{a['BLOCK_K']} nw{cfg.num_warps}", off, on, results)
    _ledger(rows)
    _assert_pool(label, results)


@needs_gpu
@pytest.mark.parametrize("label,K,N,bits", MOE_SHAPES, ids=[s[0] for s in MOE_SHAPES])
def test_grouped_gate_removes_spills_and_doubles_occupancy(label, K, N, bits):
    cfgs = G._mgemm_prune(G._grouped_dequant_gemv_kernel.configs,
                          dict(E=MOE_E, N=N, K_dim=K, K_BITS=bits),
                          E=MOE_E, N=N, K_dim=K, K_BITS=bits)
    assert cfgs, "the prune left no config"
    rows, results = [], []
    for cfg in cfgs:
        off = _grouped(K, N, bits, cfg, gated=False)
        on = _grouped(K, N, bits, cfg, gated=True)
        a = cfg.kwargs
        for tag, m in (("off", off), ("on", on)):
            rows.append(["grouped", label, MOE_E, K, N, bits, a["BLOCK_N"], a["BLOCK_K"],
                         cfg.num_warps, cfg.num_stages, tag, m["vgpr"], m["agpr"],
                         m["vspill"], m["sspill"], m["scratch"], m["lds"],
                         m["occ"], m["codelen"]])
        _check(f"{label} E{MOE_E} BN{a['BLOCK_N']} BK{a['BLOCK_K']} nw{cfg.num_warps}",
               off, on, results)
    _ledger(rows)
    _assert_pool(label, results)


@needs_gpu
def test_raw_jit_callers_compile_without_the_gate_arguments():
    """N_DIV/K_DIV default to 0 so the pinned-tile tests that drive the kernel
    through .fn keep working, and get the runtime predicate they expect."""
    k = T._fused_dequant_gemm_kernel.fn.warmup(
        torch.float16, torch.float16, torch.int16, torch.int64, torch.int32,
        1, 256, 512, T._m_bucket(1),
        512, 1, 16 * 16 * 4, 16 * 4, 256, 1, 256,
        BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
        K_BITS=4, N_PACKED=64, CB=CB, M1=True, SPLITS=1,
        num_warps=2, num_stages=2, grid=(1,))
    assert _meta(k.asm["amdgcn"])["vgpr"] > 0
    g = G._grouped_dequant_gemv_kernel.fn.warmup(
        torch.float16, torch.float16, torch.int64, torch.int32,
        torch.int64, torch.int32,
        MOE_E, 256, 512, G._e_bucket(MOE_E),
        0, 1, 16 * 16 * 4, 16 * 4, 256, 1,
        BLOCK_N=32, BLOCK_K=128, K_BITS=4, N_PACKED=64, CB=CB,
        num_warps=2, num_stages=2, grid=(1,))
    assert _meta(g.asm["amdgcn"])["vgpr"] > 0
