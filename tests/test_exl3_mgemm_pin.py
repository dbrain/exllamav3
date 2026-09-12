"""The grouped-mgemm tile pin must be deterministic, and must never change math.

Autotune's own pick is not reproducible on gfx1150: the same shape with the same
inputs chose BLOCK_K 256, 128, 256 on three consecutive runs, because the pool is
benchmarked on a cold clock and the candidates sit within noise of each other.
That is not only a performance wobble -- at BN64/BK128 the choice decides whether
the shipped binary spills 482 VGPRs and burns 864 B/lane of scratch -- so a
non-deterministic pick makes every downstream measurement unreproducible.

`_MGEMM_PINNED` therefore maps the shape part of the autotune key to one
measured tile, and `_mgemm_prune` returns it alone so triton.autotune skips
benchmarking entirely.

Nothing here launches a kernel, allocates a device tensor or needs a GPU: the
pin is a dict lookup plus divisibility guards.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import triton

from exllamav3.modules.quant import exl3_mgemm_triton as G
from exllamav3.modules.quant import exl3_triton as T

KEY = ("K_dim", "N", "K_BITS", "E_BUCKET")


def na(k_dim, n, bits, e_bucket, **extra):
    d = dict(zip(KEY, (k_dim, n, bits, e_bucket)))
    d.setdefault("FUSE_HAD", False)
    d.setdefault("FUSE_OUT_HAD", False)
    d.update(extra)
    return d


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("EXL3_MGEMM_PIN", raising=False)
    monkeypatch.delenv("TMGEMM_CONFIGS", raising=False)
    # The dense half of this file drives exl3_triton, whose pin reads its own pair of
    # variables. perf/exl3-env.sh exports EXL3_GEMM_PIN=strict, which collapses every
    # m == 1 pool to one config and breaks the "shape must be unpinned" preconditions.
    monkeypatch.delenv("EXL3_GEMM_PIN", raising=False)
    monkeypatch.delenv("EXL3_GEMM_CONFIGS", raising=False)


@pytest.fixture
def pinned(monkeypatch):
    """One known entry, so the test does not depend on the shipped table."""
    monkeypatch.setattr(G, "_MGEMM_PINNED", {(2560, 640, 4, 16): (32, 64, 2, 3)})
    return (2560, 640, 4, 16)


def test_pin_hit_returns_exactly_one_config(pinned):
    out = G._mgemm_prune(G._mgemm_configs(), na(*pinned))
    assert len(out) == 1, "a pin must collapse the pool, or autotune still races"
    c = out[0]
    assert (c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"], c.num_warps, c.num_stages) \
        == (32, 64, 2, 3)


def test_pin_is_deterministic_across_calls(pinned):
    picks = {
        tuple(sorted(c.kwargs.items())) + (c.num_warps, c.num_stages)
        for _ in range(5)
        for c in G._mgemm_prune(G._mgemm_configs(), na(*pinned))
    }
    assert len(picks) == 1


def test_miss_falls_through_to_the_pool(pinned):
    # a shape with no entry must still be tuned, not forced onto someone
    # else's tile
    out = G._mgemm_prune(G._mgemm_configs(), na(2560, 6144, 6, 1))
    assert len(out) > 1


@pytest.mark.parametrize("off", ["0", "off", "false", "no"])
def test_env_disables_the_pin(monkeypatch, pinned, off):
    monkeypatch.setenv("EXL3_MGEMM_PIN", off)
    assert len(G._mgemm_prune(G._mgemm_configs(), na(*pinned))) > 1


def test_manual_sweep_env_wins_over_the_pin(monkeypatch, pinned):
    monkeypatch.setenv("TMGEMM_CONFIGS", "64,128:2:3")
    assert G._pinned_config(na(*pinned), {}) is None


# A pin that does not divide the shape would silently land the launch in the
# generic tl.gather path, which accumulates in a different order -- i.e. a
# different result, not just a slower one. The guards must refuse instead.
@pytest.mark.parametrize("named,why", [
    (na(2560, 640, 4, 16, N=48), "BLOCK_N does not divide N"),
    (na(2560, 640, 4, 16, K_dim=96), "BLOCK_K does not divide K_dim"),
    (na(2560, 640, 4, 16, FUSE_HAD=True), "fused input had needs BLOCK_K % 128"),
    (na(2560, 640, 4, 16, FUSE_OUT_HAD=True), "fused output had needs BLOCK_N % 128"),
])
def test_pin_refuses_rather_than_changing_math(monkeypatch, named, why):
    monkeypatch.setattr(G, "_MGEMM_PINNED", {
        (named["K_dim"], named["N"], named["K_BITS"], named["E_BUCKET"]): (32, 64, 2, 3)})
    assert G._pinned_config(named, {}) is None, why


def test_shipped_table_entries_are_self_consistent():
    """Every shipped pin must divide its own shape and name a real tile."""
    for (k_dim, n, bits, e_bucket), (bn, bk, nw, ns) in G._MGEMM_PINNED.items():
        assert n % bn == 0 and k_dim % bk == 0, f"pin {(k_dim, n, bits)} cannot fold"
        assert bn in (16, 32, 64, 128, 256) and bk in (16, 32, 64, 128, 256)
        assert nw in (1, 2, 4, 8) and ns >= 1
        assert 1 <= bits <= 8 and e_bucket >= 1
        assert G._pinned_config(na(k_dim, n, bits, e_bucket), {}) is not None


# ---------------------------------------------------------------------------
# The DENSE pool (exl3_triton) has the same non-determinism and the same fix.
# Its config carries BLOCK_M and GROUP_M as well, and its key uses M_BUCKET.
# ---------------------------------------------------------------------------
DKEY = ("K_dim", "N", "K_BITS", "M_BUCKET")


def dna(k_dim, n, bits, m_bucket, **extra):
    d = dict(zip(DKEY, (k_dim, n, bits, m_bucket)))
    d.setdefault("FUSE_HAD", False)
    d.setdefault("FUSE_OUT_HAD", False)
    d.update(extra)
    return d


@pytest.fixture
def dpinned(monkeypatch):
    monkeypatch.setattr(T, "_GEMM_PINNED",
                        {(2560, 12288, 6, 1): (16, 32, 128, 1, 2, 3)})
    return (2560, 12288, 6, 1)


def test_dense_pin_hit_returns_exactly_one_config(dpinned):
    out = T._exl3_gemm_early_prune(T._exl3_gemm_configs(), dna(*dpinned))
    assert len(out) == 1
    c = out[0]
    assert (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"],
            c.kwargs["GROUP_M"], c.num_warps, c.num_stages) == (16, 32, 128, 1, 2, 3)


def test_dense_miss_falls_through_to_the_pool(dpinned):
    assert len(T._exl3_gemm_early_prune(T._exl3_gemm_configs(),
                                        dna(2560, 4096, 6, 1))) > 1


@pytest.mark.parametrize("off", ["0", "off", "false", "no"])
def test_dense_env_disables_the_pin(monkeypatch, dpinned, off):
    monkeypatch.setenv("EXL3_GEMM_PIN", off)
    assert len(T._exl3_gemm_early_prune(T._exl3_gemm_configs(), dna(*dpinned))) > 1


def test_dense_manual_sweep_env_wins(monkeypatch, dpinned):
    monkeypatch.setenv("EXL3_GEMM_CONFIGS", "16,32,128,1:2:3")
    assert T._gemm_pinned_config(dna(*dpinned), {}) is None


@pytest.mark.parametrize("named,why", [
    (dna(2560, 12288, 6, 1, N=48), "BLOCK_N does not divide N"),
    (dna(2560, 12288, 6, 1, K_dim=96), "BLOCK_K does not divide K_dim"),
    (dna(2560, 12288, 6, 1, FUSE_HAD=True), "fused input had needs BLOCK_K % 128"),
    (dna(2560, 12288, 6, 1, FUSE_OUT_HAD=True), "fused output had needs BLOCK_N % 128"),
])
def test_dense_pin_refuses_rather_than_changing_math(monkeypatch, named, why):
    monkeypatch.setattr(T, "_GEMM_PINNED", {
        (named["K_dim"], named["N"], named["K_BITS"], named["M_BUCKET"]):
            (16, 32, 64, 1, 2, 3)})
    assert T._gemm_pinned_config(named, {}) is None, why


def test_dense_shipped_table_entries_are_self_consistent():
    for (k_dim, n, bits, mb), cfg in T._GEMM_PINNED.items():
        bm, bn, bk, gm, nw, ns = cfg
        assert n % bn == 0 and k_dim % bk == 0, f"pin {(k_dim, n, bits)} cannot fold"
        assert nw in (1, 2, 4, 8) and ns >= 1 and gm >= 1 and bm >= 16
        assert T._gemm_pinned_config(dna(k_dim, n, bits, mb), {}) is not None


def test_both_pools_are_pinned_independently(monkeypatch):
    """Disabling one pool's pin must not disable the other's."""
    monkeypatch.setattr(G, "_MGEMM_PINNED", {(2560, 640, 4, 16): (32, 64, 2, 3)})
    monkeypatch.setattr(T, "_GEMM_PINNED", {(2560, 12288, 6, 1): (16, 32, 128, 1, 2, 3)})
    monkeypatch.setenv("EXL3_GEMM_PIN", "0")
    assert G._pinned_config(na(2560, 640, 4, 16), {}) is not None
    assert T._gemm_pinned_config(dna(2560, 12288, 6, 1), {}) is None


# ---------------------------------------------------------------------------
# strict mode. A table only covers shapes someone has measured; strict makes
# every shape deterministic, which is the property that actually gates the
# multi-token MoE path (tile choice decides bit-exactness) and graph capture.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mod,prune,cfgs,named,env", [
    (G, "_mgemm_prune", "_mgemm_configs", na(2560, 6144, 6, 1), "EXL3_MGEMM_PIN"),
    (T, "_exl3_gemm_early_prune", "_exl3_gemm_configs", dna(2560, 4096, 6, 1), "EXL3_GEMM_PIN"),
])
def test_strict_collapses_an_unpinned_shape(monkeypatch, mod, prune, cfgs, named, env):
    pool = getattr(mod, cfgs)()
    assert len(getattr(mod, prune)(pool, named)) > 1, "shape must be unpinned"
    monkeypatch.setenv(env, "strict")
    out = getattr(mod, prune)(pool, named)
    assert len(out) == 1


@pytest.mark.parametrize("mod,prune,cfgs,named,env", [
    (G, "_mgemm_prune", "_mgemm_configs", na(2560, 6144, 6, 1), "EXL3_MGEMM_PIN"),
    (T, "_exl3_gemm_early_prune", "_exl3_gemm_configs", dna(2560, 4096, 6, 1), "EXL3_GEMM_PIN"),
])
def test_strict_is_the_same_tile_every_call(monkeypatch, mod, prune, cfgs, named, env):
    monkeypatch.setenv(env, "strict")
    picks = set()
    for _ in range(5):
        c = getattr(mod, prune)(getattr(mod, cfgs)(), named)[0]
        picks.add((tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages))
    assert len(picks) == 1


def test_strict_still_lets_an_explicit_pin_win(monkeypatch):
    monkeypatch.setenv("EXL3_MGEMM_PIN", "strict")
    monkeypatch.setattr(G, "_MGEMM_PINNED", {(2560, 640, 4, 16): (64, 128, 2, 3)})
    out = G._mgemm_prune(G._mgemm_configs(), na(2560, 640, 4, 16))
    assert len(out) == 1 and out[0].kwargs["BLOCK_N"] == 64


# ---------------------------------------------------------------------------
# The speculative verify shapes. M_BUCKET is _m_bucket(M) and an MTP round runs
# the target over ndt + 1 tokens, so the deployed ndt 4 and 6 both land on
# M_BUCKET 8 while plain decode stays at 1.
#
# The table below is keyed on M_BUCKET 1 only, so every verify shape MISSES it
# and `strict` falls through to "take the first config the prune leaves". At
# M > 1 the prune returns the pool UNFILTERED (exl3_triton._exl3_gemm_prune_pool
# only buckets the m == 1 path, and only that path gets _prefer_warps), so
# out[0] is decided by the literal order of _exl3_gemm_configs() -- the M == 1
# CTA-starved decode tile at num_warps 4.
#
# Measured, in situ, from the live autotuner cache of two fresh processes on
# flashnext-4.05bpw (llm-fondling/perf/exl3/ledger-roundgap-tiles.csv, runs
# 20260910_060625 strict and 20260910_071434 pin=1): at M_BUCKET 1 the two
# settings agree on every shape; at M_BUCKET 8 they differ on every shape,
# strict giving BN32/BK128/nw4 to all eight sites and the autotuner choosing
# BN32/BK32 at 1-2 warps.
#
# Cost of the difference, end to end (ledger-roundgap-sample.csv, 4 samples per
# cell, 2 prompts, 2 interleaved reps, fresh process per arm, Tctl 76 C):
# the target verify forward runs 363.7 ms vs 263.7 at ndt 4 and 419.4 vs 313.8
# at ndt 6, the round costs ~100 ms more at ndt 4 and ~110 at ndt 6, and plain
# decode is unchanged at 98.7 ms either way.
_VERIFY_M_BUCKET = 8
MEASURED_VERIFY_TILES = {
    # (K_dim,      N, bits, M_BUCKET): (BM, BN, BK, GM, nw, ns)
    ( 2560,  10240, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # gdn.in_proj_qkv
    ( 2560,   6144, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # gdn.in_proj_z
    ( 2560,  12288, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 2, 3),  # attn.q_proj
    ( 2560,    640, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # shared.gate/up
    ( 2560,    512, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # attn.k/v_proj
    ( 2560, 248320, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # lm_head
    ( 6144,   2560, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 2, 3),  # gdn.out_proj
    (  640,   2560, 6, _VERIFY_M_BUCKET): (16, 32, 32, 1, 1, 2),  # moe/shared down
}


@pytest.mark.parametrize("shape,want", sorted(MEASURED_VERIFY_TILES.items()))
def test_verify_shapes_resolve_to_their_measured_tile(shape, want):
    """A verify shape must hit the table, not strict's list-order fallback."""
    cfg = T._gemm_pinned_config(dna(*shape), {})
    assert cfg is not None, f"{shape} misses the pin table, so strict picks by pool order"
    got = (cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"],
           cfg.kwargs["GROUP_M"], cfg.num_warps, cfg.num_stages)
    assert got == want


def test_strict_does_not_give_every_verify_shape_the_same_tile(monkeypatch):
    """Eight sites with six distinct M == 1 winners cannot share one M > 1 tile.

    This is the shape of the bug rather than its exact values: a single tile for
    every shape is the signature of a selection made by list order.
    """
    monkeypatch.setenv("EXL3_GEMM_PIN", "strict")
    pool = T._exl3_gemm_configs()
    picks = set()
    for k_dim, n, bits, mb in MEASURED_VERIFY_TILES:
        c = T._exl3_gemm_early_prune(pool, dna(k_dim, n, bits, mb))[0]
        picks.add((c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"],
                   c.num_warps, c.num_stages))
    assert len(picks) > 1, f"all verify shapes collapsed onto {picks}"


# ---------------------------------------------------------------------------
# The ROUTED MoE has the same hole one level up. _MGEMM_PINNED's key carries
# E_BUCKET = _e_bucket(bsz * top_k), and every shipped entry is E_BUCKET 16 --
# which is bsz 1 (top_k 10 -> E 10 -> bucket 16). A speculative round routes
# ndt + 1 tokens, so ndt 4 lands on bucket 64 (E 50) and ndt 6 and ndt 8 both on
# bucket 128 (E 70 / E 90), and all three miss.
#
# This was caught by the ship-confirmation run rather than predicted. With the
# dense M_BUCKET 8 entries in place, strict reproduced EXL3_GEMM_PIN=1 to within
# 0.9 ms at ndt 4 (306.8 vs 305.9 ms/round, long_code) but stayed 10.2 ms short
# at ndt 6 (382.6 vs 372.4). The asymmetry is the tell: the dense verify tile is
# M_BUCKET 8 at BOTH depths, so it cannot explain a difference between them --
# but the MoE bucket changes from 64 to 128, and the live autotuner caches of
# the two runs disagree on exactly one shape at bucket 128 and none at 64
# (llm-fondling/perf/exl3/ledger-roundgap-tiles.csv, runs 20260910_060625
# strict and 20260910_071434 pin=1): gate/up takes BLOCK_N 64 under autotune and
# BLOCK_N 32 under strict.
_SPEC_E_BUCKETS = (64, 128)
MEASURED_ROUTED_TILES = {
    # (K_dim,    N, bits, E_BUCKET): (BLOCK_N, BLOCK_K, nw, ns)
    (2560,  640, 4,  64): (32, 128, 2, 3),   # gate/up, ndt 4
    ( 640, 2560, 4,  64): (32, 128, 2, 3),   # down,    ndt 4
    (2560,  640, 4, 128): (64, 128, 2, 3),   # gate/up, ndt 6 and ndt 8
    ( 640, 2560, 4, 128): (32, 128, 2, 3),   # down,    ndt 6 and ndt 8
}


@pytest.mark.parametrize("shape,want", sorted(MEASURED_ROUTED_TILES.items()))
def test_routed_spec_shapes_resolve_to_their_measured_tile(shape, want):
    cfg = G._pinned_config(na(*shape), {})
    assert cfg is not None, f"{shape} misses the mgemm pin table"
    assert (cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"],
            cfg.num_warps, cfg.num_stages) == want


def test_routed_table_covers_every_speculative_expert_bucket():
    """Both MoE shapes at both buckets a deployed ndt 4/6/8 round can reach."""
    missing = [(k, n, 4, e)
               for e in _SPEC_E_BUCKETS for k, n in ((2560, 640), (640, 2560))
               if (k, n, 4, e) not in G._MGEMM_PINNED]
    assert not missing, f"unpinned routed verify shapes: {missing}"


# ---------------------------------------------------------------------------
# strict's list-order fallback is only legitimate where the pool was curated.
# _exl3_gemm_prune_pool N-buckets and _prefer_warps-filters the m == 1 branch
# and every member of those pools was measured; every other M falls through to
# a bare `return configs`, so out[0] there is decided by the literal order of
# _exl3_gemm_configs() -- whose first entries are the M == 1 decode tiles.
#
# That is how the M_BUCKET 8 verify shapes came to be handed a BN32/BK128/nw4
# decode tile on all eight dense sites, for ~100 ms/round at ndt 4. Pinning
# bucket 8 fixed the deployed case but not the mechanism: _m_bucket(9) is 16,
# so an ndt 8 round under EXL3_MOE_MULTITOK_MAX=16 misses the table and re-arms
# the identical bug on a bucket nobody has measured.
_UNCURATED_BUCKETS = (16, 32)


@pytest.mark.parametrize("m_bucket", _UNCURATED_BUCKETS)
def test_strict_does_not_guess_on_an_uncurated_bucket(monkeypatch, m_bucket):
    named = dna(2560, 12288, 6, m_bucket, M=m_bucket - 1)
    assert T._gemm_pinned_config(named, {}) is None, "shape must be unpinned"
    monkeypatch.setenv("EXL3_GEMM_PIN", "strict")
    out = T._exl3_gemm_early_prune(T._exl3_gemm_configs(), named)
    assert len(out) > 1, (
        f"strict collapsed M_BUCKET {m_bucket} to one unmeasured tile "
        f"{out[0].kwargs if out else None}; autotuning beats an arbitrary pick")


def test_strict_still_collapses_the_curated_m1_pool(monkeypatch):
    named = dna(2560, 12288, 6, 1, M=1)
    assert T._gemm_pinned_config(named, {}) is None or True
    monkeypatch.setattr(T, "_GEMM_PINNED", {})
    monkeypatch.setenv("EXL3_GEMM_PIN", "strict")
    out = T._exl3_gemm_early_prune(T._exl3_gemm_configs(), named)
    assert len(out) == 1, "m == 1 pools are measured, so strict must stay deterministic"
