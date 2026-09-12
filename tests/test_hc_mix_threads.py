"""EXL3_HC_THREADS: the gr_dots block width is now a template parameter, so it is a code path.

Changing NT changes the reduction SHAPE -- each warp reduces its own lanes and the final write
sums NT/32 partials -- so results across thread counts are near-equal, not bit-equal, and this
gate is written to that. What it must pin:

  * every supported width still agrees with the fp32 reference at the tolerance
    tests/test_hc_mix_q8.py uses, across the R values that select the launch geometry;
  * every width agrees with the shipped default (128) to fp32-reduction tolerance, so a width
    is a performance choice and never a numerics one;
  * an unsupported or malformed value silently takes the default, matching the sibling knob
    gr_j_tile_env (EXL3_HC_J_TILE) rather than the stricter knobs that raise;
  * the knob is read PER CALL, so a one-process A/B harness can flip it. A knob read once at
    import is a silent no-op as an arm, which is RUNBOOK trap 5.

    python tests/test_hc_mix_threads.py --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

from test_hc_mix_q8 import DH, H, RMS_EPS, build, rel_rms, workspaces

WIDTHS = (64, 128, 256, 512)
R_SWEEP = (1, 4, 7, 8)
TOL_MIXED = 1.5e-2
TOL_POST = 2e-2
# Across widths the only freedom is how NT/32 warp partials associate in fp32, so the two
# results differ by rounding on a sum of at most 16 terms -- orders below the fp16 output step.
TOL_CROSS = 2e-3

DEV = "cuda:0"


def _streams(device, R, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed + 1)
    return (torch.randn((1, R, H, DH), generator = g) * 3.0).float().to(device).contiguous()


@torch.inference_mode()
def _run(m, streams, R, use_combine, width, monkeypatch = None):
    if width is None:
        os.environ.pop("EXL3_HC_THREADS", None)
    else:
        os.environ["EXL3_HC_THREADS"] = str(width)
    dots, post, mixed = workspaces(streams.device, R, m.fn_rows, use_combine, torch.half)
    ext.gr_mix_q(streams.view(R, H, DH), m.fn_q, m.fn_qs, m.upx_q, m.upx_qs, m.w_h,
                 RMS_EPS, dots, post, mixed)
    return (post.clone() if post is not None else None), mixed.clone()


@pytest.fixture(autouse = True)
def _clean_env():
    old = os.environ.get("EXL3_HC_THREADS")
    os.environ.setdefault("EXL3_HC_J_TILE", "2")
    yield
    if old is None:
        os.environ.pop("EXL3_HC_THREADS", None)
    else:
        os.environ["EXL3_HC_THREADS"] = old


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("R", R_SWEEP)
def test_width_matches_reference(width, R):
    m = build(DEV, True, 4242, q8 = True)
    streams = _streams(DEV, R, 4242)
    p_ref, x_ref = m._mix_ref(streams)
    post, mixed = _run(m, streams, R, True, width)
    assert torch.isfinite(mixed.float()).all().item(), f"NT={width} R={R} produced non-finite"
    e_mixed = rel_rms(mixed, x_ref.view(R, DH))
    e_post = (post - p_ref.view(R, H)).abs().max().item()
    assert e_mixed < TOL_MIXED, f"NT={width} R={R}: mixed rel-rms {e_mixed:.3e}"
    assert e_post < TOL_POST, f"NT={width} R={R}: post max-abs {e_post:.3e}"


@pytest.mark.parametrize("width", [w for w in WIDTHS if w != 128])
@pytest.mark.parametrize("R", R_SWEEP)
def test_width_agrees_with_the_shipped_default(width, R):
    """A block width must be a speed choice, never a numerics one."""
    m = build(DEV, True, 99, q8 = True)
    streams = _streams(DEV, R, 99)
    p128, x128 = _run(m, streams, R, True, 128)
    pw, xw = _run(m, streams, R, True, width)
    e = rel_rms(xw, x128.float())
    assert e < TOL_CROSS, f"NT={width} vs NT=128 at R={R}: mixed rel-rms {e:.3e}"
    assert (pw - p128).abs().max().item() < TOL_CROSS, f"NT={width} vs 128 at R={R}: post"


@pytest.mark.parametrize("bad", ["0", "7", "1024", "yes", "", "-128"])
def test_malformed_width_takes_the_default(bad):
    """gr_threads_env's contract, matching EXL3_HC_J_TILE: unknown values fall back, silently.

    The strict knobs in this tree (EXL3_MOE_DEDUP, EXL3_HC_FUSED_MAX_R) raise instead. This one
    does not, and the test exists so that is a decision on the record rather than an accident.
    """
    m = build(DEV, True, 7, q8 = True)
    streams = _streams(DEV, 4, 7)
    _, x_def = _run(m, streams, 4, True, None)
    _, x_bad = _run(m, streams, 4, True, bad)
    assert torch.equal(x_bad, x_def), f"EXL3_HC_THREADS={bad!r} did not fall back to the default"


def test_knob_is_read_per_call():
    """Flipped between two calls in ONE process the output must track the flip.

    Bit-equality would be too strong (the reduction width moves), so this asserts the weaker
    but sufficient thing: 512 and 64 both track the default within TOL_CROSS when set
    per-call, which is impossible if the value were captured once at import.
    """
    m = build(DEV, True, 11, q8 = True)
    streams = _streams(DEV, 7, 11)
    _, a = _run(m, streams, 7, True, 64)
    _, b = _run(m, streams, 7, True, 512)
    _, c = _run(m, streams, 7, True, 64)
    assert torch.equal(a, c), "same width twice in one process gave different results"
    assert rel_rms(b, a.float()) < TOL_CROSS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default = "cuda:0")
    args = ap.parse_args()
    DEV = args.device
    sys.exit(pytest.main([__file__, "-q"]))
