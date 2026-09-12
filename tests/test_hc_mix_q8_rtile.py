"""
R-sweep parity gate for the int8 GatedResidual mix (hc_mix.cu gr_dots_q8 /
gr_finalize_q8), on Flash-Next shapes (H = 4, hidden 2560, rank 320).

The row count selects the launch geometry, so it is a code path, not a parameter: this
sweeps R across every tile boundary and remainder and pins three things at each one.

  * bit-exact agreement with the fp16 gr_mix at power-of-two scales. Any change to the
    accumulation or reduction ORDER surfaces here as a hard failure rather than as a
    tolerance that could be widened.
  * rel-rms against the fp32 _mix_ref on post and mixed, at the tolerance
    tests/test_hc_mix_q8.py already uses.
  * row invariance: the output for stream row r must not depend on how many rows were
    mixed in the same call. A batched row that disagrees with the same row mixed alone
    means the reduction order moved with the launch config, which the hc_mix.cu header
    forbids.

    python tests/test_hc_mix_q8_rtile.py --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exllamav3.ext import exllamav3_ext as ext

from test_hc_mix_q8 import (DH, H, RMS_EPS, build, q8_pow2, rel_rms, run_pair, workspaces)

R_SWEEP = (1, 2, 3, 4, 5, 7, 8, 16)
TOL_MIXED = 1.5e-2
TOL_POST = 2e-2


def _streams(device, R, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed + 1)
    return (torch.randn((1, R, H, DH), generator = g) * 3.0).float().to(device).contiguous()


@torch.inference_mode()
def _run_q(m, streams, R, use_combine, out_dtype):
    dots, post, mixed = workspaces(streams.device, R, m.fn_rows, use_combine, out_dtype)
    ext.gr_mix_q(streams.view(R, H, DH), m.fn_q, m.fn_qs, m.upx_q, m.upx_qs, m.w_h,
                 RMS_EPS, dots, post, mixed)
    return post, mixed


@torch.inference_mode()
def check_ref(device, R, use_combine, seed, out_dtype = torch.half):
    m = build(device, use_combine, seed, q8 = True)
    streams = _streams(device, R, seed)
    p_ref, x_ref = m._mix_ref(streams)
    post, mixed = _run_q(m, streams, R, use_combine, out_dtype)
    e_mixed = rel_rms(mixed, x_ref.view(R, DH))
    e_post = (post - p_ref.view(R, H)).abs().max().item() if use_combine else 0.0
    ok = e_mixed < TOL_MIXED and e_post < TOL_POST and torch.isfinite(mixed.float()).all().item()
    print(f"  {'PASS' if ok else 'FAIL'} ref   R={R:<3d} combine={int(use_combine)} "
          f"out={str(out_dtype).split('.')[-1]:<5s}: mixed rel-rms {e_mixed:.4e} "
          f"(< {TOL_MIXED:g}), post maxabs {e_post:.4e} (< {TOL_POST:g})")
    return ok


def check_exact(device, R, use_combine, seed):
    (p0, x0), (p1, x1), _ = run_pair(device, R, use_combine, seed, q8_pow2)
    ok = torch.equal(x0, x1) and (not use_combine or torch.equal(p0, p1))
    dp = (p0 - p1).abs().max().item() if use_combine else 0.0
    print(f"  {'PASS' if ok else 'FAIL'} exact R={R:<3d} combine={int(use_combine)}       "
          f": mixed maxdiff {(x0.float() - x1.float()).abs().max().item():.3e}, "
          f"post maxdiff {dp:.3e}")
    return ok


@torch.inference_mode()
def _row_invariance(m, R, use_combine, seed, label):
    streams = _streams(m.w_h.device, R, seed)
    post, mixed = _run_q(m, streams, R, use_combine, torch.half)
    bad = []
    for r in range(R):
        one = streams[:, r : r + 1].contiguous()
        p1, x1 = _run_q(m, one, 1, use_combine, torch.half)
        if not torch.equal(x1[0], mixed[r]) or (use_combine and not torch.equal(p1[0], post[r])):
            bad.append(r)
    ok = not bad
    print(f"  {'PASS' if ok else 'FAIL'} rowinv {label:<7s} R={R:<3d}: rows differing from "
          f"the R=1 result: {bad if bad else 'none'}")
    return ok


@torch.inference_mode()
def check_row_invariance(device, R, use_combine, seed):
    m = build(device, use_combine, seed, q8 = True)
    return _row_invariance(m, R, use_combine, seed, f"int8/c{int(use_combine)}")


@torch.inference_mode()
def check_mixed_width(device, mode, R, seed):
    """int4fn and int4up quantize the two sides differently, and only the int8 side has a
    tiled kernel, so the pair runs at two different grid.y within one call. Neither the q4
    nor the q8 gate builds these modes."""
    from test_hc_mix_q4 import build as build_q4
    return _row_invariance(build_q4(device, True, seed, mode), R, True, seed, mode)


def _device():
    return torch.device(os.environ.get("EXL3_TEST_DEVICE", "cuda:0"))


def test_ref():
    d = _device()
    assert all(check_ref(d, R, True, 900 + R) for R in R_SWEEP)


def test_exact():
    d = _device()
    assert all(check_exact(d, R, True, 900 + R) for R in R_SWEEP)


def test_row_invariance():
    d = _device()
    assert all(check_row_invariance(d, R, True, 900 + R) for R in (2, 5, 7, 16))


def test_mixed_width():
    d = _device()
    assert all(check_mixed_width(d, m, 7, 960) for m in ("int4fn", "int4up"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True
    for R in R_SWEEP:
        ok &= check_exact(device, R, True, seed = 900 + R)
        ok &= check_ref(device, R, True, seed = 900 + R)
    print("  -- no-combine (post = nullptr, M = LR) and fp32 out --")
    for R in (1, 7, 16):
        ok &= check_exact(device, R, False, seed = 950 + R)
        ok &= check_ref(device, R, False, seed = 950 + R)
        ok &= check_ref(device, R, True, seed = 950 + R, out_dtype = torch.float)
    print("  -- row invariance --")
    for R in (2, 5, 7, 16):
        ok &= check_row_invariance(device, R, True, seed = 900 + R)
    for R in (2, 7):
        ok &= check_row_invariance(device, R, False, seed = 970 + R)
    print("  -- mixed int4/int8 widths (only one side is tiled) --")
    for mode in ("int4fn", "int4up"):
        ok &= check_mixed_width(device, mode, 7, seed = 960)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
