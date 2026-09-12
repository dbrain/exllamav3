"""The dedup kernel's tile pin, and the guards that make a bad pin fail loudly.

Why it matters more here than for the dense pin. `_dedup_configs()` contains
(BLOCK_N 64, BLOCK_K 128, num_warps 2), which at MAXR 4 spills the
[MAXR, 2, 2, 2, NN, 8, 4] fp32 accumulator and runs **12x** slower than the same
tile at num_warps 4 (9.66 vs 0.76 ms/launch, llm-fondling/perf/exl3/final/
dedupbound1.log). The two sit in the same autotune pool, and
TRITON_CACHE_AUTOTUNING=1 freezes whichever one a contended benchmark happened to
like. So the pin is not a micro-optimisation, it is the thing that keeps a
12x cliff out of the shipped binary.

Nothing here launches a kernel or needs a GPU: the pin is a dict lookup plus
divisibility guards. The numerics of the kernel itself are gated by
test_moe_multitok_grouped.py::test_expert_dedup_matches_token_major.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from exllamav3.modules.quant import exl3_mgemm_dedup_triton as D

KEY = ("K_dim", "N", "K_BITS", "E_BUCKET", "MAXR")


def na(k_dim, n, bits, e_bucket, maxr):
    return dict(zip(KEY, (k_dim, n, bits, e_bucket, maxr)))


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("EXL3_MGEMM_PIN", raising = False)


@pytest.fixture
def pinned(monkeypatch):
    """One known entry, so the guard tests do not depend on the shipped table."""
    monkeypatch.setattr(D, "_DEDUP_PINNED", {(2560, 640, 4, 128, 4): (64, 128, 4, 3)})
    return (2560, 640, 4, 128, 4)


def test_pin_hit_returns_exactly_one_config(pinned):
    out = D._dedup_prune(D._dedup_configs(), na(*pinned))
    assert len(out) == 1, "a pin must collapse the pool, or autotune still races"
    c = out[0]
    assert (c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"], c.num_warps, c.num_stages) \
        == (64, 128, 4, 3)


def test_pin_off_falls_through_to_the_pool(pinned, monkeypatch):
    for v in ("0", "off", "false", "no", "OFF"):
        monkeypatch.setenv("EXL3_MGEMM_PIN", v)
        assert len(D._dedup_prune(D._dedup_configs(), na(*pinned))) > 1, \
            f"EXL3_MGEMM_PIN={v} must disable the pin"


def test_pin_that_does_not_divide_the_shape_is_refused(monkeypatch):
    """A tile that does not divide N or K_dim lands in a different code path with a
    different accumulation order. Refuse the pin rather than change math."""
    monkeypatch.setattr(D, "_DEDUP_PINNED", {(2560, 640, 4, 128, 4): (48, 128, 4, 3)})
    assert len(D._dedup_prune(D._dedup_configs(), na(2560, 640, 4, 128, 4))) > 1


def test_miss_keeps_the_divisibility_and_accumulator_filters():
    out = D._dedup_prune(D._dedup_configs(), na(640, 2560, 4, 128, 8))
    assert out, "the pool must never come back empty"
    for c in out:
        assert 2560 % c.kwargs["BLOCK_N"] == 0 and 640 % c.kwargs["BLOCK_K"] == 0
        assert 8 * c.kwargs["BLOCK_N"] <= 256


# (K_dim, N, K_BITS, E_BUCKET, MAXR) the deployed Flash-Next MTP verify actually hits:
# routed gate/up 2560x640 and down 640x2560, K_BITS 4, EXL3_MOE_DEDUP=4, and the three
# E buckets an ndt 4/6/8 round lands on (E = (ndt+1)*10 -> _e_bucket 64/128/128, plus 32
# from the shorter accept-prefill rounds).
DEPLOYED = [(2560, 640, 4, eb, 4) for eb in (32, 64, 128)] + \
           [(640, 2560, 4, eb, 4) for eb in (32, 64, 128)]


@pytest.mark.parametrize("key", DEPLOYED)
def test_every_deployed_key_is_pinned(key):
    assert key in D._DEDUP_PINNED, (
        f"{key} is a shape the MTP verify launches every layer; unpinned it re-races the "
        f"autotune pool (which contains a 12x spilling tile) once per process")


@pytest.mark.parametrize("key", sorted(D._DEDUP_PINNED))
def test_shipped_pins_are_self_consistent(key):
    """A pin bypasses _dedup_prune's own filters, so the table must satisfy them here.
    maxr * BLOCK_N is the accumulator width that made EXL3_MOE_DEDUP=8 cost +134%."""
    k_dim, n, _, _, maxr = key
    bn, bk, nw, ns = D._DEDUP_PINNED[key]
    assert n % bn == 0 and k_dim % bk == 0, "pin must divide the shape or it is inert"
    assert maxr * bn <= 256, "pinned accumulator exceeds the width that spills"
    assert nw in (1, 2, 4, 8) and ns >= 1
    assert D._dedup_prune(D._dedup_configs(), na(*key))[0].kwargs["BLOCK_N"] == bn


def _tile_of_last_compiled(dev):
    k = D._grouped_dedup_gemv4_kernel.fn
    c = list(k.device_caches[dev.index][0].values())[-1]
    m = c.metadata
    return m.num_warps, m.num_stages


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason = "CUDA (or ROCm) device required")
def test_pin_reaches_the_compiled_binary():
    """A dict test cannot tell a live pin from a pin the autotuner overrode. Launch the
    real entry point at a deployed key and read num_warps off the binary Triton built:
    the pool's other BLOCK_N 64 entry is num_warps 2, the spilling one."""
    import torch

    dev = torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))
    key = (2560, 640, 4, 128, 4)
    k_dim, n, bits, e_bucket, maxr = key
    e = 70                                    # _e_bucket(70) == 128
    assert D._e_bucket(e) == e_bucket

    g = torch.Generator(device = "cpu").manual_seed(0xD3D0)
    pool = torch.randint(-32768, 32767, (4, (k_dim // 16) * (n // 16) * 16 * bits),
                         dtype = torch.int32, generator = g).to(torch.short).to(dev)
    ptrs = torch.tensor([pool[i % 4].data_ptr() for i in range(8)],
                        dtype = torch.int64, device = dev)
    ids = torch.arange(e, device = dev) % 8
    x = (torch.randn((e, k_dim), generator = g) / 8).half().to(dev)
    y = torch.empty((e, n), dtype = torch.half, device = dev)

    D.exl3_mgemm_dedup(x, ptrs, ids.sort().values, y, bits, 2, max_rows = maxr)
    torch.cuda.synchronize(dev)

    want = D._DEDUP_PINNED[key]
    assert _tile_of_last_compiled(dev) == (want[2], want[3]), \
        "the autotuner did not run the pinned tile"
    assert torch.isfinite(y.float()).all()
