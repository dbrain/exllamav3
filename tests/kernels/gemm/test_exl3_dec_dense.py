"""
Dense exl3_dec decode GEMVs (ROCm): exl3_dec_gemv / _multi (one row) and exl3_dec_gemv_r / _r_multi (2..8 verify rows) against the
reconstruct reference (testlib.exl3.linear_ref) at every dense bitrate the LinearEXL3 gate admits (K 2, 2.5 .. 8), both codebooks the
kernels decode, plus the gemv_r contract the verify path relies on: row r of an R-row launch is bit-equal to the one-row launch on row r.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import _DEC_KB2_DENSE, dec_workspace
from testlib.compare import rel_err
from testlib.exl3 import generator, rand_linear, linear_ref

pytestmark = pytest.mark.rocm_only

KS = [kb2 / 2 for kb2 in _DEC_KB2_DENSE]
SHAPES = [(2560, 1024), (1024, 512)]
TOL = 5e-3


def _require_dec():
    if not (torch.version.hip and hasattr(ext, "exl3_dec_gemv_r_multi")):
        pytest.skip("exl3_dec is ROCm only")


def _mats(device, k, ns, K, codebook, seed):
    gen = generator(seed)
    return [rand_linear(k, n, K, gen, device) for n in ns]


def _ref(x, w, K, codebook):
    return linear_ref(x, w["trellis"], w["suh"], w["svh"], K, codebook).float()


def test_k8_is_a_dense_decode_bitrate():
    assert 16 in _DEC_KB2_DENSE


@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("k, n", SHAPES)
@pytest.mark.parametrize("K", KS)
@torch.inference_mode()
def test_dense_dec_matches_reconstruct(device, K, k, n, codebook):
    _require_dec()
    if codebook == "mcg" and not float(K).is_integer():
        pytest.skip("half-integer bitrates are mul1 only")
    mcg = codebook == "mcg"
    ws = _mats(device, k, (n, n // 2), K, codebook, seed = int(K * 2) * 100 + k + mcg)
    scratch, counters = dec_workspace(device)
    gen = generator(int(K * 2) + k)
    x8 = (torch.randn((8, k), generator = gen) * 0.5).half().to(device)
    singles = []
    for r in range(8):
        y = torch.empty((1, n), dtype = torch.float, device = device)
        ext.exl3_dec_gemv(x8[r: r + 1], ws[0]["trellis"], ws[0]["suh"], ws[0]["svh"], y, scratch, counters, K, mcg)
        singles.append(y)
    singles = torch.cat(singles)
    err = rel_err(singles, _ref(x8, ws[0], K, codebook))
    assert err < TOL, f"K{K} {codebook} gemv: rel err {err:.2e}"

    outs = [torch.empty((1, w["trellis"].shape[1] * 16), dtype = torch.float, device = device) for w in ws]
    ext.exl3_dec_gemv_multi(x8[:1], [w["trellis"] for w in ws], [w["suh"] for w in ws], [w["svh"] for w in ws],
                            outs, scratch, counters, [float(K)] * 2, mcg)
    assert torch.equal(outs[0], singles[:1]), f"K{K} {codebook} gemv_multi != gemv"
    err = rel_err(outs[1], _ref(x8[:1], ws[1], K, codebook))
    assert err < TOL, f"K{K} {codebook} gemv_multi second matrix: rel err {err:.2e}"

    for R in range(2, 9):
        y = torch.empty((R, n), dtype = torch.float, device = device)
        ext.exl3_dec_gemv_r(x8[:R], ws[0]["trellis"], ws[0]["suh"], ws[0]["svh"], y, scratch, counters, K, mcg)
        assert torch.equal(y, singles[:R]), f"K{K} {codebook} gemv_r R={R}: rows differ from the one-row launches"
        outs = [torch.empty((R, w["trellis"].shape[1] * 16), dtype = torch.float, device = device) for w in ws]
        ext.exl3_dec_gemv_r_multi(x8[:R], [w["trellis"] for w in ws], [w["suh"] for w in ws], [w["svh"] for w in ws],
                                  outs, scratch, counters, [float(K)] * 2, mcg)
        assert torch.equal(outs[0], singles[:R]), f"K{K} {codebook} gemv_r_multi R={R}: rows differ from the one-row launches"
        err = rel_err(outs[1], _ref(x8[:R], ws[1], K, codebook))
        assert err < TOL, f"K{K} {codebook} gemv_r_multi R={R} second matrix: rel err {err:.2e}"


@pytest.mark.parametrize("dec1", ["0", "1"])
@torch.inference_mode()
def test_k8_gemv_r_one_decode_pass(device, dec1, monkeypatch):
    _require_dec()
    monkeypatch.setenv("EXL3_GEMV_R_DEC1", dec1)
    k, n = SHAPES[0]
    w = _mats(device, k, (n,), 8, "mul1", seed = 81)[0]
    scratch, counters = dec_workspace(device)
    x = (torch.randn((8, k), generator = generator(82)) * 0.5).half().to(device)
    ref = _ref(x, w, 8, "mul1")
    for R in range(2, 9):
        y = torch.empty((R, n), dtype = torch.float, device = device)
        ext.exl3_dec_gemv_r(x[:R], w["trellis"], w["suh"], w["svh"], y, scratch, counters, 8.0, False)
        err = rel_err(y, ref[:R])
        assert err < TOL, f"DEC1={dec1} R={R}: rel err {err:.2e}"
