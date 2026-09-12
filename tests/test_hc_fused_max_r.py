"""
EXL3_HC_FUSED_MAX_R: route the multi-row GatedResidual mix to the GEMM path.

The fp16 gr_mix kernels grid on (·, R) with r = blockIdx.y, so block (j, r) reads fn row
j for stream row r: every stream row re-reads the whole weight, and a speculative verify
mixing R = ndt + 1 rows at once pays it 7 times. FUSED_MAX_R = 32 is the cutover to the
GEMM path, which reads proj_h/up_h once regardless of R. The int8 pair (gr_mix_q) has
since been R-tiled in hc_mix.cu and no longer re-reads per row, so under the shipping
EXL3_HC_QUANT=int8 this knob no longer trades traffic for anything.

These tests pin the routing knob and the numerics of the path it selects. They do NOT
assert which path is faster; that is a ledger question.

    python tests/test_hc_fused_max_r.py --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.modules import hyperconnections as hc
from exllamav3.modules.hyperconnections import GatedResidual

H = 4
DH = 2560
RANK = 320
RMS_EPS = 1e-6


def build(device, seed, q8):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = H * DH
    down = (torch.randn((RANK, n), generator=g) * n ** -0.5).half().to(device)
    up = (torch.randn((n, RANK), generator=g) * RANK ** -0.5).half().to(device)
    inject = (torch.randn((H, n), generator=g) * n ** -0.5).half().to(device)
    norm_w = (torch.randn((n,), generator=g) * 0.02).half().to(device)
    m = GatedResidual(config=None, key="test", hc_mult=H, hidden_size=DH,
                      rms_norm_eps=RMS_EPS, use_combine=True)
    m.qbits = (8, 8) if q8 else None
    m.norm_w_raw = norm_w
    m._prepare(down, up, inject)
    return m


class CountFused:
    """Which path _mix took, counted at the kernel rather than inferred from timing."""

    def __init__(self):
        self.n = 0
        self._saved = []

    def __enter__(self):
        for name in ("gr_mix", "gr_mix_q"):
            orig = getattr(hc.ext, name, None)
            if orig is None:
                continue

            def make(orig=orig):
                def counting(*a, **kw):
                    self.n += 1
                    return orig(*a, **kw)
                return counting

            setattr(hc.ext, name, make())
            self._saved.append((name, orig))
        return self

    def __exit__(self, *e):
        for name, orig in self._saved:
            setattr(hc.ext, name, orig)
        return False


class Env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def __exit__(self, *e):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-9)).item()


@torch.inference_mode()
def _mix_at(m, device, R, seed):
    streams = (torch.randn((1, R, H, DH),
                           generator=torch.Generator(device="cpu").manual_seed(seed))
               * 3.0).float().to(device).contiguous()
    with CountFused() as c:
        post, mixed = m._mix(streams, cached=False)
    return post, mixed, c.n, streams


def check_routing(device, seed=901):
    """The knob must be read PER CALL: specdepth applies A/B env deltas per sample inside
    one process, so a threshold resolved at import would silently no-op as an arm."""
    m = build(device, seed, q8=True)
    ok = True
    with Env(EXL3_HC_FUSED_MAX_R=None):
        _, _, n7, _ = _mix_at(m, device, 7, seed)
        ok &= n7 == 1
        print(f"  {'PASS' if n7 == 1 else 'FAIL'} default: R=7 takes the fused path "
              f"(fused calls {n7}, want 1)")
    with Env(EXL3_HC_FUSED_MAX_R=4):
        _, _, n7, _ = _mix_at(m, device, 7, seed)
        _, _, n1, _ = _mix_at(m, device, 1, seed)
        ok &= n7 == 0 and n1 == 1
        print(f"  {'PASS' if n7 == 0 else 'FAIL'} MAX_R=4: R=7 takes the GEMM path "
              f"(fused calls {n7}, want 0)")
        print(f"  {'PASS' if n1 == 1 else 'FAIL'} MAX_R=4: R=1 still takes the fused path "
              f"(fused calls {n1}, want 1)")
    with Env(EXL3_HC_FUSED_MAX_R=0):
        _, _, n1, _ = _mix_at(m, device, 1, seed)
        ok &= n1 == 0
        print(f"  {'PASS' if n1 == 0 else 'FAIL'} MAX_R=0: even R=1 takes the GEMM path "
              f"(fused calls {n1}, want 0)")
    return ok


def check_bad_value(device, seed=902):
    """A malformed value must raise, not degrade to a default. EXL3_MOE_DEDUP=16 silently
    running dedup OFF under a label claiming otherwise cost this campaign a retraction."""
    m = build(device, seed, q8=True)
    with Env(EXL3_HC_FUSED_MAX_R="banana"):
        try:
            _mix_at(m, device, 7, seed)
            print("  FAIL bad value: accepted 'banana' silently")
            return False
        except ValueError:
            print("  PASS bad value: 'banana' raises ValueError")
            return True


@torch.inference_mode()
def check_parity(device, seed=903, tol=1.5e-2):
    """The GEMM path the knob selects must be no worse against the fp32 reference than the
    fused int8 path it replaces. Both are compared to _mix_ref at the same R."""
    ok = True
    for R in (1, 7):
        m = build(device, seed + R, q8=True)
        streams = (torch.randn((1, R, H, DH),
                               generator=torch.Generator(device="cpu").manual_seed(seed))
                   * 3.0).float().to(device).contiguous()
        p_ref, x_ref = m._mix_ref(streams)
        p_ref, x_ref = p_ref.view(R, H), x_ref.view(R, DH)
        with Env(EXL3_HC_FUSED_MAX_R=None):
            p_f, x_f = m._mix(streams, cached=False)
        with Env(EXL3_HC_FUSED_MAX_R=0):
            p_g, x_g = m._mix(streams, cached=False)
        e_f, e_g = rel_rms(x_f, x_ref), rel_rms(x_g, x_ref)
        pf = (p_f - p_ref).abs().max().item()
        pg = (p_g - p_ref).abs().max().item()
        good = e_g < tol and e_f < tol and e_g <= e_f * 1.05 and pg < 2e-2
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} parity R={R}: mixed rel-rms vs fp32 ref "
              f"fused(int8) {e_f:.3e} / gemm(fp16) {e_g:.3e} (tol {tol:g}, gemm must not "
              f"be worse); post maxabs {pf:.3e} / {pg:.3e}")
    return ok


@torch.inference_mode()
def check_shapes(device, seed=904):
    """The GEMM path must return the same shapes and dtypes as the fused one, and the
    weights it needs must survive int8 quantization (fn_h/upx_h are freed, proj_h/up_h are
    not -- hyperconnections.py:417)."""
    m = build(device, seed, q8=True)
    ok = m.fn_h is None and m.upx_h is None
    ok &= m.proj_h is not None and m.proj_h.dtype == torch.half
    ok &= m.up_h is not None and m.up_h.dtype == torch.half
    with Env(EXL3_HC_FUSED_MAX_R=None):
        p_f, x_f, _, _ = _mix_at(m, device, 7, seed)
    with Env(EXL3_HC_FUSED_MAX_R=0):
        p_g, x_g, _, _ = _mix_at(m, device, 7, seed)
    ok &= p_f.shape == p_g.shape and x_f.shape == x_g.shape
    ok &= x_f.dtype == x_g.dtype and p_f.dtype == p_g.dtype
    ok &= torch.isfinite(x_g.float()).all().item()
    print(f"  {'PASS' if ok else 'FAIL'} shapes/dtypes match and int8 keeps proj_h/up_h "
          f"(fused {tuple(x_f.shape)}/{x_f.dtype}, gemm {tuple(x_g.shape)}/{x_g.dtype})")
    return ok


def _device():
    return torch.device(os.environ.get("EXL3_TEST_DEVICE", "cuda:0"))


def test_routing():
    assert check_routing(_device())


def test_bad_value():
    assert check_bad_value(_device())


def test_parity():
    assert check_parity(_device())


def test_shapes():
    assert check_shapes(_device())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True
    ok &= check_routing(device)
    ok &= check_bad_value(device)
    ok &= check_parity(device)
    ok &= check_shapes(device)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
