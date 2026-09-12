"""
Parity tests for the int4 GatedResidual mix kernel (exllamav3_ext/hc_mix.cu gr_mix_q4,
EXL3_HC_QUANT=int4) on Flash-Next shapes (H = 4, hidden 2560, rank 320).

Levels, weakest assumption first:
  * pack round-trip: _q4_pack_fn / _q4_pack_upx are bijections on the int4 code range, so
    unpacking in torch must reproduce the codes bitwise. Pure host arithmetic, no kernel --
    this is the layout contract the kernel's nibble indexing has to match.
  * layout parity: with the SAME codes dequantized to fp16 and fed to gr_mix, the q4 kernel
    must agree to fp-association noise (< 1e-3 rel RMS). Any transposed stride, swapped
    nibble or misindexed group scale is an O(1) error, not a 1e-3 one, so this is the test
    that actually catches wiring bugs.
  * error bound: with real max-based group scales, symmetric int4 is ~4 bits of weight
    error; the tolerance here is loose on purpose -- it catches something structurally
    broken, not a width judgement. Whether int4 ships is decided by KLD on the real model
    (perf/exl3/hc-kld.py) and by nothing here.

    python tests/test_hc_mix_q4.py --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import (
    GatedResidual, _q4_pack_fn, _q4_pack_upx, _q4_unpack_fn, _q4_unpack_upx,
    Q4_FN_GROUP, Q4_UP_GROUP,
)

H = 4
DH = 2560
RANK = 320
RMS_EPS = 1e-6


def make_weights(device, use_combine, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    n = H * DH
    down = (torch.randn((RANK, n), generator = g) * n ** -0.5).half().to(device)
    up = (torch.randn((n, RANK), generator = g) * RANK ** -0.5).half().to(device)
    inject = (torch.randn((H, n), generator = g) * n ** -0.5).half().to(device) \
        if use_combine else None
    norm_w = (torch.randn((n,), generator = g) * 0.02).half().to(device)
    return down, up, inject, norm_w


def build(device, use_combine, seed, mode):
    m = GatedResidual(config = None, key = "test", hc_mult = H, hidden_size = DH,
                      rms_norm_eps = RMS_EPS, use_combine = use_combine)
    m.qbits = {"off": None, "int8": (8, 8), "int4": (4, 4),
               "int4fn": (4, 8), "int4up": (8, 4)}[mode]
    down, up, inject, norm_w = make_weights(device, use_combine, seed)
    m.norm_w_raw = norm_w
    m._prepare(down, up, inject)
    return m


def workspaces(device, R, M, use_combine, out_dtype = torch.half):
    dots = torch.zeros((R, M + 1, H), dtype = torch.float, device = device)
    post = torch.zeros((R, H), dtype = torch.float, device = device) if use_combine else None
    mixed = torch.zeros((R, DH), dtype = out_dtype, device = device)
    return dots, post, mixed


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-9)).item()


def check_pack_roundtrip(device, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    M = RANK + H
    fn_codes = torch.randint(-7, 8, (M, H, DH), generator = g, dtype = torch.int8).to(device)
    ok = torch.equal(_q4_unpack_fn(_q4_pack_fn(fn_codes), DH), fn_codes)
    up_codes = torch.randint(-7, 8, (H, DH // 4, RANK, 4), generator = g,
                             dtype = torch.int8).to(device)
    ok &= torch.equal(_q4_unpack_upx(_q4_pack_upx(up_codes), RANK), up_codes)
    print(f"  {'PASS' if ok else 'FAIL'} pack round-trip (fn {tuple(fn_codes.shape)}, "
          f"upx {tuple(up_codes.shape)})")
    return ok


@torch.inference_mode()
def run_pair(device, R, use_combine, seed):
    """(dequantized-fp16 gr_mix, gr_mix_q4, unquantized-fp16 gr_mix) on identical codes."""
    m = build(device, use_combine, seed, mode = "int4")
    M = m.fn_rows
    fn_codes = _q4_unpack_fn(m.fn_q, DH)                                # (M, H, DH)
    up_codes = _q4_unpack_upx(m.upx_q, RANK)                            # (H, DH/4, RANK, 4)
    fn_deq = (fn_codes.float()
              * m.fn_qs.view(M, H, DH // Q4_FN_GROUP, 1).float()
              .expand(M, H, DH // Q4_FN_GROUP, Q4_FN_GROUP).reshape(M, H, DH)) \
        .half().view(M, H * DH).contiguous()
    # upx_qs is (LR / G, H * DH); undo to (H, DH/4, LR/G, 4) then broadcast over the group
    us = m.upx_qs.view(RANK // Q4_UP_GROUP, H, DH // 4, 4).permute(1, 2, 0, 3)
    up_deq = (up_codes.float().view(H, DH // 4, RANK // Q4_UP_GROUP, Q4_UP_GROUP, 4)
              * us.float().unsqueeze(3)).half().view(H, DH // 4, RANK, 4).contiguous()

    streams = (torch.randn((R, H, DH), generator = torch.Generator(device = "cpu")
                           .manual_seed(seed + 1)) * 3.0).float().to(device).contiguous()

    d0, p0, x0 = workspaces(device, R, M, use_combine)
    ext.gr_mix(streams, fn_deq, up_deq, m.w_h, RMS_EPS, d0, p0, x0)
    d1, p1, x1 = workspaces(device, R, M, use_combine)
    ext.gr_mix_q(streams, m.fn_q, m.fn_qs, m.upx_q, m.upx_qs, m.w_h, RMS_EPS, d1, p1, x1)

    mf = build(device, use_combine, seed, mode = "off")
    d2, p2, x2 = workspaces(device, R, M, use_combine)
    ext.gr_mix(streams, mf.fn_h, mf.upx_h, mf.w_h, RMS_EPS, d2, p2, x2)
    return (p0, x0), (p1, x1), (p2, x2)


def check_layout(device, R, use_combine, seed, tol = 1e-3):
    (p0, x0), (p1, x1), _ = run_pair(device, R, use_combine, seed)
    e_mixed = rel_rms(x1, x0)
    e_post = rel_rms(p1, p0) if use_combine else 0.0
    ok = e_mixed < tol and e_post < tol
    print(f"  {'PASS' if ok else 'FAIL'} layout R={R} combine={use_combine}: "
          f"mixed rel-rms {e_mixed:.3e}, post rel-rms {e_post:.3e} (< {tol:g})")
    return ok


def check_bound(device, R, use_combine, seed, tol_mixed = 3.5e-1, tol_post = 3.5e-1):
    _, (p1, x1), (p2, x2) = run_pair(device, R, use_combine, seed)
    e_mixed = rel_rms(x1, x2)
    e_post = rel_rms(p1, p2) if use_combine else 0.0
    ok = e_mixed < tol_mixed and e_post < tol_post
    print(f"  {'PASS' if ok else 'FAIL'} bound R={R} combine={use_combine}: "
          f"mixed rel-rms {e_mixed:.3e} (< {tol_mixed:g}), post rel-rms {e_post:.3e} "
          f"(< {tol_post:g})")
    return ok


@torch.inference_mode()
def check_module(device, use_combine, seed):
    m = build(device, use_combine, seed, mode = "int4")
    M = RANK + (H if use_combine else 0)
    ok = (m.fn_h is None and m.upx_h is None
          and m.fn_q.dtype == torch.uint8 and m.upx_q.dtype == torch.uint8
          and m.fn_q.shape == (M, H * DH // 2)
          and m.upx_q.shape == (H, DH // 4, RANK // 2, 4)
          and m.fn_qs.numel() == M * H * (DH // Q4_FN_GROUP)
          and m.upx_qs.numel() == (RANK // Q4_UP_GROUP) * H * DH
          and m.proj_h.dtype == torch.half and m.up_h.dtype == torch.half)
    t = m.get_tensors()
    ok = ok and all(v.dtype == torch.half for v in t.values())
    streams = torch.randn((1, 1, H, DH), dtype = torch.float, device = device)
    post, _, mixed = m.mix(streams, {}) if use_combine else (None, None, m.forward(streams, {}))
    ok = ok and torch.isfinite(mixed.float()).all().item()
    print(f"  {'PASS' if ok else 'FAIL'} module wiring combine={use_combine}")
    return ok


def check_bytes():
    """int4 must actually halve the decode-time weight stream against int8."""
    n_fn = (RANK + H) * H * DH
    n_up = H * DH * RANK
    b8 = n_fn + (RANK + H) * H * 2 + n_up + H * DH * 2
    b4 = (n_fn // 2 + (RANK + H) * H * (DH // Q4_FN_GROUP) * 2
          + n_up // 2 + (RANK // Q4_UP_GROUP) * H * DH * 2)
    ok = b4 < 0.56 * b8
    print(f"  {'PASS' if ok else 'FAIL'} bytes/site int8 {b8 / 1e6:.3f} MB -> int4 "
          f"{b4 / 1e6:.3f} MB ({b4 / b8:.3f}x, scales included)")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    p.add_argument("--host-only", action = "store_true",
                   help = "packing + byte accounting only; no kernel, no device")
    args = p.parse_args()
    ok = check_bytes()
    if not args.host_only:
        assert hasattr(ext, "gr_mix_q"), "extension has no gr_mix_q (rebuild required)"
    device = torch.device("cpu" if args.host_only else args.device)
    if not args.host_only:
        torch.cuda.set_device(device)
    ok &= check_pack_roundtrip(device, seed = 900)
    if not args.host_only:
        for i, (R, uc) in enumerate([(1, True), (1, False), (4, True), (8, True), (32, True)]):
            ok &= check_layout(device, R, uc, seed = 700 + i)
            ok &= check_bound(device, R, uc, seed = 700 + i)
        for i, uc in enumerate([True, False]):
            ok &= check_module(device, uc, seed = 800 + i)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
