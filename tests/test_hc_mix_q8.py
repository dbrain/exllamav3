"""
Parity tests for the int8 GatedResidual mix kernel (exllamav3_ext/hc_mix.cu gr_mix_q8,
EXL3_HC_QUANT=int8) against the fp16 gr_mix, on Flash-Next shapes (H = 4, hidden 2560,
rank 320).

Two levels:
  * bit-exact: with power-of-two scales, q * s is exact in fp16 and scaling commutes with the
    fp32 reduction, so the post-accumulation scale in gr_mix_q8 must reproduce gr_mix bitwise.
    This isolates layout/indexing bugs from quantization error.
  * error bound: with real max-based scales, symmetric int8 gives ~0.8% RMS weight error
    (fn, per (row, stream) slice of 2560) and ~0.75% (up, per rank-320 output channel);
    through the low-rank bottleneck and the sigmoid gate that lands under 1.5e-2 relative
    RMS on the mixed output.

    python tests/test_hc_mix_q8.py --device cuda:1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import GatedResidual, _q8_sym

H = 4
DH = 2560
RANK = 320
RMS_EPS = 1e-6


def q8_pow2(t, dim):
    s = t.float().abs().amax(dim = dim, keepdim = True) / 127.0
    s = torch.exp2(torch.ceil(torch.log2(s.clamp(min = 2.0 ** -24)))).half()
    q = (t.float() / s.float()).round_().clamp_(-127.0, 127.0).to(torch.int8)
    return q.contiguous(), s


def make_weights(device, use_combine, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    n = H * DH
    # Standard fan-in init: with N(0, 1) weights the up-projection logits reach O(250) and the
    # gate saturates, which would make the comparison measure sigmoid flips instead of the
    # kernel
    down = (torch.randn((RANK, n), generator = g) * n ** -0.5).half().to(device)
    up = (torch.randn((n, RANK), generator = g) * RANK ** -0.5).half().to(device)
    inject = (torch.randn((H, n), generator = g) * n ** -0.5).half().to(device) \
        if use_combine else None
    norm_w = (torch.randn((n,), generator = g) * 0.02).half().to(device)
    return down, up, inject, norm_w


def build(device, use_combine, seed, q8):
    m = GatedResidual(config = None, key = "test", hc_mult = H, hidden_size = DH,
                      rms_norm_eps = RMS_EPS, use_combine = use_combine)
    m.qbits = (8, 8) if q8 else None
    down, up, inject, norm_w = make_weights(device, use_combine, seed)
    m.norm_w_raw = norm_w
    m._prepare(down, up, inject)
    return m


def workspaces(device, R, M, use_combine, out_dtype = torch.half):
    dots = torch.zeros((R, M + 1, H), dtype = torch.float, device = device)
    post = torch.zeros((R, H), dtype = torch.float, device = device) if use_combine else None
    mixed = torch.zeros((R, DH), dtype = out_dtype, device = device)
    return dots, post, mixed


@torch.inference_mode()
def run_pair(device, R, use_combine, seed, quantize):
    m = build(device, use_combine, seed, q8 = False)
    M = m.fn_rows
    fn_q, fn_s = quantize(m.fn_h.view(M, H, DH), -1)
    upx_q, upx_s = quantize(m.upx_h, -2)
    fn_deq = (fn_q.float() * fn_s.float()).half().view(M, H * DH).contiguous()
    upx_deq = (upx_q.float() * upx_s.float()).half().contiguous()
    fn_q = fn_q.view(M, H * DH)
    fn_s = fn_s.reshape(M * H).contiguous()
    upx_s = upx_s.reshape(H * DH).contiguous()

    streams = (torch.randn((R, H, DH), generator = torch.Generator(device = "cpu")
                           .manual_seed(seed + 1)) * 3.0).float().to(device).contiguous()

    d0, p0, x0 = workspaces(device, R, M, use_combine)
    ext.gr_mix(streams, fn_deq, upx_deq, m.w_h, RMS_EPS, d0, p0, x0)
    d1, p1, x1 = workspaces(device, R, M, use_combine)
    ext.gr_mix_q(streams, fn_q, fn_s, upx_q, upx_s, m.w_h, RMS_EPS, d1, p1, x1)

    d2, p2, x2 = workspaces(device, R, M, use_combine)
    ext.gr_mix(streams, m.fn_h, m.upx_h, m.w_h, RMS_EPS, d2, p2, x2)
    return (p0, x0), (p1, x1), (p2, x2)


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-9)).item()


def check_exact(device, R, use_combine, seed):
    (p0, x0), (p1, x1), _ = run_pair(device, R, use_combine, seed, q8_pow2)
    ok = torch.equal(x0, x1) and (not use_combine or torch.equal(p0, p1))
    tag = f"bit-exact R={R} combine={use_combine}"
    print(f"  {'PASS' if ok else 'FAIL'} {tag}: mixed maxdiff "
          f"{(x0.float() - x1.float()).abs().max().item():.3e}")
    return ok


def check_bound(device, R, use_combine, seed, tol_mixed = 1.5e-2, tol_post = 2e-2):
    _, (p1, x1), (p2, x2) = run_pair(device, R, use_combine, seed, _q8_sym)
    e_mixed = rel_rms(x1, x2)
    e_post = (p1 - p2).abs().max().item() if use_combine else 0.0
    ok = e_mixed < tol_mixed and e_post < tol_post
    print(f"  {'PASS' if ok else 'FAIL'} bound R={R} combine={use_combine}: "
          f"mixed rel-rms {e_mixed:.3e} (< {tol_mixed:g}), post maxabs {e_post:.3e} "
          f"(< {tol_post:g})")
    return ok


@torch.inference_mode()
def check_module(device, use_combine, seed):
    m = build(device, use_combine, seed, q8 = True)
    ok = (m.fn_h is None and m.upx_h is None
          and m.fn_q.dtype == torch.int8 and m.upx_q.dtype == torch.int8
          and m.fn_q.shape == (RANK + (H if use_combine else 0), H * DH)
          and m.upx_q.shape == (H, DH // 4, RANK, 4)
          and m.fn_qs.numel() == m.fn_rows * H and m.upx_qs.numel() == H * DH
          # the GEMM path and get_tensors must still see unquantized fp16
          and m.proj_h.dtype == torch.half and m.up_h.dtype == torch.half)
    t = m.get_tensors()
    ok = ok and all(v.dtype == torch.half for v in t.values())
    streams = torch.randn((1, 1, H, DH), dtype = torch.float, device = device)
    post, _, mixed = m.mix(streams, {}) if use_combine else (None, None, m.forward(streams, {}))
    ok = ok and torch.isfinite(mixed.float()).all().item()
    print(f"  {'PASS' if ok else 'FAIL'} module wiring combine={use_combine}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True
    for i, (R, uc) in enumerate([(1, True), (1, False), (4, True), (8, True), (32, True)]):
        ok &= check_exact(device, R, uc, seed = 700 + i)
        ok &= check_bound(device, R, uc, seed = 700 + i)
    for i, uc in enumerate([True, False]):
        ok &= check_module(device, uc, seed = 800 + i)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
