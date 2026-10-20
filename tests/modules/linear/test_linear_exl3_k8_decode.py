"""
Small-row routes of LinearEXL3 for the K7 / K8 dense projections (the 4.18 bpw "SC" pack of Qwen3.8-Flash-Next stores every dense
projection at K8) on ROCm, where the BC_* classes are masked: no row count up to 16 may fall back to reconstruct + GEMM.
  - exl3_dec-eligible shapes: one row takes dec_gemv, 2..8 verify rows take dec_gemv_r
  - every other small-row case (non-verify rows, shapes exl3_dec cannot take) takes the native EXL3 GEMM (ext.exl3_gemm)
  - outputs match the reconstruct reference (testlib.exl3.linear_ref)
  - on the real pack (layers 0 GDN and 3 full attention, $EXL3_FLASH_K8_TEST_MODEL) no K8 projection takes reconstruct at 1..8 rows
"""
import os
from pathlib import Path

import pytest
import torch

from exllamav3.modules import Linear
from exllamav3.modules.quant import exl3 as q

from testlib.checkpoint import module_config
from testlib.compare import rel_err
from testlib.exl3 import checkpoint_tensors, generator, linear_ref

pytestmark = pytest.mark.rocm_only

KEY = "model.layers.0.self_attn.q_proj"
MODEL = Path(os.environ.get("EXL3_FLASH_K8_TEST_MODEL", "~/dev/tinyfiddler/models/flashnext-exl3-4.18sc-ng6")).expanduser()
PATHS = ("dec_gemv", "dec_gemv_r", "hip_gemv", "native_gemm", "reconstruct_hgemm")
TOL = 5e-3


class PathSpy:
    def __init__(self, inner):
        self.last = None
        for name in PATHS:
            orig = getattr(inner, name, None)
            if orig is None:
                continue
            def spy(*a, _orig = orig, _name = name, **k):
                self.last = self.last or _name
                return _orig(*a, **k)
            setattr(inner, name, spy)

    def run(self, fn):
        self.last = None
        out = fn()
        return out, self.last


def _verify_params(rows):
    return {"dflash_verify": True} if rows > 1 else {}


@pytest.fixture
def verify_fuse(monkeypatch):
    monkeypatch.setitem(q.VERIFY_FUSE, "on", True)


@pytest.mark.parametrize("K", [6, 7, 8])
@pytest.mark.parametrize("k, n, dec", [(2560, 1024, True), (640, 2560, False)], ids = ["dec-shape", "narrow-k"])
@torch.inference_mode()
def test_small_rows_never_reconstruct(tmp_path, device, K, k, n, dec, verify_fuse):
    if not torch.version.hip:
        pytest.skip("ROCm routes")
    t = checkpoint_tensors(KEY, k, n, K, generator(K * 7 + k))
    module = Linear(module_config(t, tmp_path), KEY, k, n)
    module.load(device)
    inner = module.inner
    assert inner.dec_ok == dec
    spy = PathSpy(inner)
    tt = {name: v.to(device) for name, v in t.items()}
    gen = generator(K + k)
    for rows in range(1, 17):
        x = (torch.randn((1, rows, k), generator = gen) * 0.5).half().to(device)
        ref = linear_ref(x.view(rows, k), tt[f"{KEY}.trellis"], tt[f"{KEY}.suh"], tt[f"{KEY}.svh"], K)
        for params in ({}, _verify_params(rows)):
            y, path = spy.run(lambda: module.forward(x, dict(params)))
            assert path != "reconstruct_hgemm" or (K <= 6 and rows > 8), f"K{K} {k}x{n} rows={rows} params={params}: reconstruct"
            if dec and rows == 1:
                assert path == "dec_gemv"
            elif dec and rows <= 8 and params:
                assert path == "dec_gemv_r"
            elif K > 6:
                assert path == "native_gemm", f"K{K} {k}x{n} rows={rows} params={params}: {path}"
            err = rel_err(y.view(rows, n), ref)
            assert err < TOL, f"K{K} {k}x{n} rows={rows} params={params} ({path}): rel err {err:.2e}"


@pytest.mark.parametrize("layer", [0, 3], ids = ["gdn", "full-attn"])
@torch.inference_mode()
def test_418_dense_k8_decode_routes(device, layer, verify_fuse):
    if not MODEL.is_dir():
        pytest.skip(f"4.18 SC pack not found: {MODEL} (set EXL3_FLASH_K8_TEST_MODEL)")
    from exllamav3 import Config, Model
    model = Model.from_config(Config.from_directory(str(MODEL)))
    blk = model.find_module(f"model.language_model.layers.{layer}")
    mods = [blk.attn, blk.mlp.shared_experts]
    for m in mods:
        m.load(device)
    try:
        lins = [m for mod in mods for m in mod if isinstance(getattr(m, "inner", None), q.LinearEXL3)]
        assert lins and all(l.inner.K == 8 for l in lins)
        gen = generator(layer)
        for lin in lins:
            spy = PathSpy(lin.inner)
            for rows in range(1, 9):
                x = (torch.randn((1, rows, lin.in_features), generator = gen) * 0.5).half().to(device)
                _, path = spy.run(lambda: lin.forward(x, _verify_params(rows)))
                assert path != "reconstruct_hgemm", f"{lin.key} rows={rows}: reconstruct"
                if lin.inner.dec_ok:
                    assert path == ("dec_gemv" if rows == 1 else "dec_gemv_r"), f"{lin.key} rows={rows}: {path}"
    finally:
        for m in mods:
            m.unload()
