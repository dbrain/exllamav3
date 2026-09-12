"""
EXL3_GDN_BA_GEMV: merge the GatedDeltaNet b/a projections into one fp16 GEMV.

The Torch split-projection path runs b_proj and a_proj as two separate LinearFP16 calls.
Both are built with out_dtype = torch.float against a half weight, so fp16.py:96 takes the
dtype-mismatch branch and issues ext.hgemm: 2 launches per GDN layer, 72 per M=7 verify at
36 layers, for a (M, 2560) x (2560, 48) GEMV each. ext.gdn_ba_gemv is the merged kernel the
C++ BC path uses for exactly this (gated_delta_net.cpp:210) and it is compiled on ROCm even
though the BC classes are not.

The trap is the consumer. ext.gated_delta_net_fused_op_2 indexes b and a as dense [B,S,H]
off data_ptr (gdn.cu:256-264) and TORCH_CHECKs only their shapes, so handing it strided
slices of one packed [rows, 2H] buffer is silently wrong for every row but the first.

    python tests/test_gdn_ba_gemv.py --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.modules import Linear
from exllamav3.modules import gated_delta_net as gdn
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.quant import LinearFP16

HS = 2560
NV = 48
BETA_SCALE = 1.0


def _proj(device, gen, bias):
    w = (torch.randn((HS, NV), generator = gen) * HS ** -0.5).half().to(device)
    b = (torch.randn((NV,), generator = gen) * 0.1).half().to(device) if bias else None
    m = Linear(None, "test", HS, NV, qmap = None, out_dtype = torch.float, pad_to = 1)
    m.inner = LinearFP16(HS, NV, w, b, full_in_features = HS, full_out_features = NV,
                         first_in_feature = 0, first_out_feature = 0,
                         out_dtype = torch.float)
    m.quant_type = "fp16"
    return m


def build(device, seed, bias = False):
    gen = torch.Generator(device = "cpu").manual_seed(seed)
    m = GatedDeltaNet.__new__(GatedDeltaNet)
    m.num_v_heads = NV
    m.hidden_size = HS
    m.device = device
    m.b_proj = _proj(device, gen, bias)
    m.a_proj = _proj(device, gen, bias)
    m.ba_weight_t = None
    m.ba_bias = None
    m.ba_weight_filled = False
    return m


class CountBA:
    """Which path ran, counted at the kernel rather than inferred from timing."""

    def __init__(self):
        self.hgemm = 0
        self.gemv = 0
        self._saved = []

    def __enter__(self):
        for name, field in (("hgemm", "hgemm"), ("gdn_ba_gemv", "gemv")):
            orig = getattr(gdn.ext, name)

            def make(orig = orig, field = field):
                def counting(*a, **kw):
                    setattr(self, field, getattr(self, field) + 1)
                    return orig(*a, **kw)
                return counting

            setattr(gdn.ext, name, make())
            self._saved.append((name, orig))
        return self

    def __exit__(self, *e):
        for name, orig in self._saved:
            setattr(gdn.ext, name, orig)
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


def _x(device, seqlen, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return torch.randn((1, seqlen, HS), generator = g).half().to(device).contiguous()


@torch.inference_mode()
def _ba_at(m, x, mode):
    bsz, seqlen, _ = x.shape
    with Env(EXL3_GDN_BA_GEMV = mode):
        with CountBA() as c:
            b, a = m._ba_proj(x, {}, bsz, seqlen)
    return b, a, c


@torch.inference_mode()
def _ref(m, x):
    xf = x.view(-1, HS).float()
    wb = m.b_proj.inner.get_weight_tensor().float()
    wa = m.a_proj.inner.get_weight_tensor().float()
    bb = m.b_proj.inner.get_bias_tensor()
    ab = m.a_proj.inner.get_bias_tensor()
    b = xf @ wb
    a = xf @ wa
    if bb is not None:
        b = b + bb.float()
    if ab is not None:
        a = a + ab.float()
    return b.view(*x.shape[:-1], NV), a.view(*x.shape[:-1], NV)


def check_routing(device, seed = 801):
    """The knob must be read PER CALL: the campaign's harnesses apply A/B env deltas per
    sample inside one process, so a mode resolved at import silently no-ops as an arm."""
    m = build(device, seed)
    ok = True
    for seqlen in (1, 7):
        x = _x(device, seqlen, seed)
        _, _, c0 = _ba_at(m, x, None)
        good = c0.hgemm == 2 and c0.gemv == 0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} M={seqlen} default: hgemm x{c0.hgemm} "
              f"gemv x{c0.gemv} (want 2/0)")
        _, _, c1 = _ba_at(m, x, 1)
        good = c1.hgemm == 0 and c1.gemv == 1
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} M={seqlen} mode 1: hgemm x{c1.hgemm} "
              f"gemv x{c1.gemv} (want 0/1)")
        _, _, c2 = _ba_at(m, x, "split")
        good = c2.hgemm == 0 and c2.gemv == 2
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} M={seqlen} mode split: hgemm x{c2.hgemm} "
              f"gemv x{c2.gemv} (want 0/2)")
        _, _, c3 = _ba_at(m, x, 0)
        good = c3.hgemm == 2 and c3.gemv == 0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} M={seqlen} mode 0: hgemm x{c3.hgemm} "
              f"gemv x{c3.gemv} (want 2/0)")
    return ok


def check_row_cap(device, seed = 802):
    """gdn_ba_gemv grids one warp per output row per input row, so it re-reads the whole
    weight for every row. Past the measured break-even that is worse than a tiled GEMM and
    the knob must not apply."""
    m = build(device, seed)
    cap = gdn._BA_GEMV_MAX_ROWS
    ok = True
    _, _, c_in = _ba_at(m, _x(device, cap, seed), 1)
    good = c_in.gemv == 1 and c_in.hgemm == 0
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} M={cap} (at cap): gemv x{c_in.gemv} (want 1)")
    _, _, c_out = _ba_at(m, _x(device, cap + 1, seed), 1)
    good = c_out.gemv == 0 and c_out.hgemm == 2
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} M={cap + 1} (over cap): gemv x{c_out.gemv} "
          f"hgemm x{c_out.hgemm} (want 0/2)")
    return ok


def check_bad_value(device, seed = 803):
    """A malformed value must raise, not degrade to a default. EXL3_MOE_DEDUP=16 silently
    running dedup OFF under a label claiming otherwise cost this campaign a retraction."""
    m = build(device, seed)
    x = _x(device, 7, seed)
    try:
        _ba_at(m, x, "banana")
        print("  FAIL bad value: accepted 'banana' silently")
        return False
    except ValueError:
        print("  PASS bad value: 'banana' raises ValueError")
        return True


@torch.inference_mode()
def check_parity(device, seed = 804, tol = 5e-3):
    """Both gemv modes must be no worse than the hgemm baseline against an fp32 reference.
    ext.hgemm accumulates in fp32 (rel-rms ~1e-6 at this shape); a half-output matmul is
    ~200x worse, so a dtype swap is not an acceptable substitute for this kernel."""
    ok = True
    for bias in (False, True):
        for seqlen in (1, 7):
            m = build(device, seed + seqlen, bias = bias)
            x = _x(device, seqlen, seed)
            rb, ra = _ref(m, x)
            e = {}
            for mode in (0, 1, "split"):
                b, a, _ = _ba_at(m, x, mode)
                e[mode] = max(rel_rms(b, rb), rel_rms(a, ra))
            good = all(v < tol for v in e.values()) and \
                   e[1] <= e[0] * 8.0 and e["split"] <= e[0] * 8.0
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'} parity M={seqlen} bias={int(bias)}: "
                  f"rel-rms vs fp32 hgemm {e[0]:.3e} / merged {e[1]:.3e} / "
                  f"split {e['split']:.3e} (tol {tol:g})")
    return ok


@torch.inference_mode()
def check_consumer(device, seed = 805):
    """The b/a the knob produces feed ext.gated_delta_net_fused_op_2, which indexes them as
    dense [B,S,H] off data_ptr and checks only their shapes. Strided slices of a packed
    [rows, 2H] buffer pass every shape check and mis-index every row but the first, so the
    gate is beta/g out of the real consumer, not the projections in isolation."""
    ok = True
    g = torch.Generator(device = "cpu").manual_seed(seed)
    dt_bias = (torch.randn((NV,), generator = g) * 0.5).to(torch.bfloat16).to(device)
    a_log = (torch.randn((NV,), generator = g) * 0.5).float().to(device)
    for seqlen in (1, 7):
        m = build(device, seed + seqlen)
        x = _x(device, seqlen, seed)
        out = {}
        for mode in (0, 1, "split"):
            b, a, _ = _ba_at(m, x, mode)
            beta = torch.empty((1, seqlen, NV), dtype = torch.bfloat16, device = device)
            gg = torch.empty((1, seqlen, NV), dtype = torch.float, device = device)
            gdn.ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta, gg, BETA_SCALE)
            out[mode] = (beta.float().clone(), gg.clone())
        good = True
        for mode in (1, "split"):
            eb = (out[mode][0] - out[0][0]).abs().max().item()
            eg = rel_rms(out[mode][1], out[0][1])
            good &= eb <= 8e-3 and eg < 5e-3
            print(f"  {'PASS' if eb <= 8e-3 and eg < 5e-3 else 'FAIL'} consumer M={seqlen} "
                  f"mode {mode}: beta maxabs {eb:.3e} g rel-rms {eg:.3e} vs hgemm path")
        ok &= good
    return ok


@torch.inference_mode()
def check_layout(device, seed = 806):
    """b and a must come back contiguous, [B,S,H] and float, whatever the mode."""
    m = build(device, seed)
    ok = True
    for seqlen in (1, 7):
        x = _x(device, seqlen, seed)
        for mode in (0, 1, "split"):
            b, a, _ = _ba_at(m, x, mode)
            good = (b.shape == (1, seqlen, NV) and a.shape == b.shape and
                    b.dtype == torch.float and a.dtype == torch.float and
                    b.is_contiguous() and a.is_contiguous())
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'} layout M={seqlen} mode {mode}: "
                  f"{tuple(b.shape)}/{b.dtype} contiguous {b.is_contiguous()}/"
                  f"{a.is_contiguous()}")
    return ok


@torch.inference_mode()
def check_fill_once(device, seed = 807):
    """The packed weight is assembled once, from the final post-load weights, and not
    rebuilt per forward."""
    m = build(device, seed)
    x = _x(device, 7, seed)
    _ba_at(m, x, 1)
    ok = m.ba_weight_t is not None and m.ba_weight_filled
    w = m.ba_weight_t
    _ba_at(m, x, 1)
    ok &= m.ba_weight_t is w
    ok &= torch.equal(w[:NV], m.b_proj.inner.get_weight_tensor().T)
    ok &= torch.equal(w[NV:], m.a_proj.inner.get_weight_tensor().T)
    print(f"  {'PASS' if ok else 'FAIL'} packed weight filled once, b rows then a rows "
          f"({tuple(w.shape)})")
    return ok


def _device():
    return torch.device(os.environ.get("EXL3_TEST_DEVICE", "cuda:0"))


def test_routing():
    assert check_routing(_device())


def test_row_cap():
    assert check_row_cap(_device())


def test_bad_value():
    assert check_bad_value(_device())


def test_parity():
    assert check_parity(_device())


def test_consumer():
    assert check_consumer(_device())


def test_layout():
    assert check_layout(_device())


def test_fill_once():
    assert check_fill_once(_device())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True
    ok &= check_routing(device)
    ok &= check_row_cap(device)
    ok &= check_bad_value(device)
    ok &= check_parity(device)
    ok &= check_consumer(device)
    ok &= check_layout(device)
    ok &= check_fill_once(device)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
