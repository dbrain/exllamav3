"""Every dense GEMM the shipped speculative round launches must resolve to a PIN.

Why this is a correctness-of-deployment test and not a micro-optimisation. With no pin,
`_exl3_gemm_early_prune` hands the whole 13-config pool to `triton.autotune` at M > 1
(exl3_triton.py:866-876), and `TRITON_CACHE_AUTOTUNING=1` -- which perf/exl3-env.sh ships --
persists whatever that benchmark liked and never re-measures it (upstream
triton/runtime/autotuner.py:168-206, `check_disk_cache`). This box has a 30 W package cap
shared between CPU and iGPU and a thermal cliff at 91 -> 94 C, so an autotune pass that runs
while the box is busy measures distorted timings and can freeze a loser on disk permanently.

The exposure is not hypothetical and it is not small. From the round census
(llm-fondling/perf/exl3/final/ledger-specprof3.csv, arm spec_graph1_d3584_ndt6), the dense
Triton GEMM is 82.140 ms of a ~301 ms round, of which these shapes had NO pin:

    architecture/qwen4_exp_mtp.py:233   (2560, 98304, 6)  6 launches  17.134 ms   5.69%
    arch_specific/qwen4_exp_mtp.py:151  (2560,  2560, 5)  6 launches   0.845 ms   0.28%
    arch_specific/qwen4_exp_mtp.py:156  (2560,  2560, 5)  6 launches   0.494 ms   0.16%
    modules/qsa_indexer.py:492          (2560,   640, 4) 12 launches   1.037 ms   0.34%
    (plus the accept_prefill tail of the same three)      ~=  19.9 ms   6.6%

The pinned values are not guesses and no sweep was run for them: they are what the autotuner
has ALREADY chosen, read back out of ~/.triton/cache and cross-checked across every cached
generation. Consistency and the margin over the runner-up, per bucket:

    (2560, 98304, 6, 2)  (16,32,32,1,1,2)   1/1 runs, runner-up 1.39x away
    (2560, 98304, 6, 4)  (16,32,32,1,1,2)   1/1 runs, runner-up 1.41x away
    (2560,  2560, 5, 2)  (16,32,32,1,1,2)   9/9 runs, runner-up 1.09x
    (2560,  2560, 5, 4)  (16,32,32,1,1,2) 10/10 runs, runner-up 1.12x
    (2560,  2560, 5, 8)  (16,32,32,1,1,2)   9/9 runs, runner-up 1.07x
    (2560,   640, 4, 8)  (16,32,128,1,2,3) 12/13 runs, runner-up ~1.00x

So pinning them should be a NULL on speed -- it freezes the choice already in force. What it
buys is that the choice can no longer be re-rolled by a contended autotune pass.

Nothing here launches a kernel or needs a GPU: a pin is a dict lookup plus divisibility guards.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from exllamav3.modules.quant import exl3_triton as T

# (K_dim, N, K_BITS, M_BUCKET, issuer) -- every dense-GEMM launch in the ndt6 round at d3584.
ROUND_SHAPES = [
    (2560,  98304, 6, 2, "mtp draft head (vocab-sliced)"),
    (2560,  98304, 6, 4, "mtp draft head (vocab-sliced)"),
    (2560,   2560, 5, 2, "mtp.fc_hidden / fc_embedding"),
    (2560,   2560, 5, 4, "mtp.fc_hidden / fc_embedding"),
    (2560,   2560, 5, 8, "mtp.fc_hidden / fc_embedding"),
    (2560,    640, 4, 8, "attn.indexer index_qk_proj"),
    (2560,  10240, 6, 8, "gdn.in_proj_qkv"),
    (2560,   6144, 6, 8, "gdn.in_proj_z"),
    (6144,   2560, 6, 8, "gdn.out_proj / attn.o_proj"),
    (2560,  12288, 6, 8, "attn.q_proj"),
    (2560,    512, 6, 8, "attn.k_proj / v_proj"),
    (2560,    640, 6, 8, "shared_expert.gate_proj / up_proj"),
    (640,    2560, 6, 8, "shared_expert.down_proj"),
    (2560, 248320, 6, 8, "lm_head"),
]


def named(k_dim, n, bits, bucket):
    return {"K_dim": k_dim, "N": n, "K_BITS": bits, "M_BUCKET": bucket, "M": bucket,
            "FUSE_HAD": False, "FUSE_OUT_HAD": False}


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("EXL3_GEMM_PIN", raising = False)
    monkeypatch.delenv("EXL3_GEMM_CONFIGS", raising = False)


@pytest.mark.parametrize("k_dim, n, bits, bucket, issuer", ROUND_SHAPES,
                         ids = [f"{s[4]}@M{s[3]}" for s in ROUND_SHAPES])
def test_round_shape_is_pinned(k_dim, n, bits, bucket, issuer):
    cfg = T._gemm_pinned_config(named(k_dim, n, bits, bucket), {})
    assert cfg is not None, (
        f"({k_dim}, {n}, {bits}, {bucket}) -- {issuer} -- has no _GEMM_PINNED entry, so it "
        f"autotunes at M > 1 and TRITON_CACHE_AUTOTUNING freezes the result. Measure it and "
        f"add it to the table."
    )


@pytest.mark.parametrize("key, val", sorted(T._GEMM_PINNED.items()))
def test_every_pin_survives_its_own_divisibility_guard(key, val):
    """A pin the guard rejects is dead weight that reads as coverage."""
    k_dim, n, bits, bucket = key
    cfg = T._gemm_pinned_config(named(k_dim, n, bits, bucket), {})
    assert cfg is not None, f"{key} -> {val} is rejected by its own guard (N % BN or K % BK)"
    bm, bn, bk, gm, nw, ns = val
    assert (cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"],
            cfg.kwargs["GROUP_M"], cfg.num_warps, cfg.num_stages) == (bm, bn, bk, gm, nw, ns)


def test_pin_hit_collapses_the_pool_to_one_config():
    """The point of a pin is that autotune never benchmarks. One config, or it still races."""
    na = named(2560, 98304, 6, 2)
    out = T._exl3_gemm_early_prune(None, na, **na)
    assert len(out) == 1, f"pinned shape still offers {len(out)} configs to the autotuner"


def test_pin_off_still_returns_a_pool():
    """EXL3_GEMM_PIN=0 must remain a real escape hatch, not a no-op."""
    os.environ["EXL3_GEMM_PIN"] = "0"
    try:
        assert T._gemm_pinned_config(named(2560, 98304, 6, 2), {}) is None
    finally:
        os.environ.pop("EXL3_GEMM_PIN", None)
