"""
K8 shared expert in the fused MoE half-layer (EXL3_MOE_FUSED, ROCm gfx11.5). The 4.18 bpw "SC" pack of Qwen3.8-Flash-Next pairs K5 (layers 0, 1, 47)
or K4 (every other layer) routed experts with a K8 shared expert.

  - tile body (exl3_moe_fused_tile_test) at every bit width against the independent NumPy trellis decode (testlib.trellis): every instantiated
    (K, output dtype), native and repacked layouts, 1..4 rows
  - real layers 0 (routed K5) and 2 (routed K4) of the pack ($EXL3_FLASH_K8_TEST_MODEL), decode rows 1..8: the block must take the fused kernel, and
    its MoE output (workspace ydbg) must match the reconstruct reference built from the kernel's own mixer output, top-k picks and weights
    (routing-independent, so a near-tie in the router cannot flake it); the 8-row one-launch result must equal the 4 + 4 split bitwise
  - prefill rows 16, 128, 1024: the block must decline the fused kernel, and the K8 shared expert (the module path those rows take) must match
    the reconstruct reference
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.compare import rel_err
from testlib.exl3 import dequant
from testlib.trellis import dequant_rotated

pytestmark = pytest.mark.rocm_only

MODEL = Path(os.environ.get("EXL3_FLASH_K8_TEST_MODEL", "~/dev/tinyfiddler/models/flashnext-exl3-4.18sc-ng6")).expanduser()
TILE_KS = [(2560, False), (4096, False), (640, True), (2048, True)]     # (K, float output): the instantiated tile-test table
TILE_N = 64
TILE_TOL = 2e-3
LAYER_TOL = 1e-2


def _require_fused():
    if not (torch.version.hip and torch.cuda.is_available() and hasattr(ext, "exl3_moe_fused_tile_test")):
        pytest.skip("fused MoE half-layer is ROCm only")


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("k, cf", TILE_KS, ids = lambda v: str(v))
@pytest.mark.parametrize("native", [1, 0])
@torch.inference_mode()
def test_tile_matches_numpy_decode(device, bits, k, cf, native):
    from exllamav3.modules.moe_fused import _rep
    _require_fused()
    gen = torch.Generator().manual_seed(bits * 1000 + k + native)
    trellis = torch.randint(-32768, 32768, (k // 16, TILE_N // 16, 16 * bits), dtype = torch.int16, generator = gen)
    w_hat = torch.from_numpy(dequant_rotated(trellis, bits, "mul1").astype(np.float64))
    B = (trellis if native else _rep(trellis)).contiguous().to(device)
    for rows in (1, 2, 3, 4):
        A = (torch.randn((rows, k), generator = gen) * 0.1).half()
        C = torch.full((rows, TILE_N), float("nan"), dtype = torch.float if cf else torch.half, device = device)
        ext.exl3_moe_fused_tile_test(A.to(device), B, C, bits, native)
        ref = A.double() @ w_hat
        err = rel_err(C.cpu(), ref)
        assert err < TILE_TOL, f"K{bits} k={k} cf={cf} native={native} rows={rows}: rel err {err:.2e}"


@pytest.fixture(scope = "module")
def k8_model():
    if not MODEL.is_dir():
        pytest.skip(f"4.18 SC pack not found: {MODEL} (set EXL3_FLASH_K8_TEST_MODEL)")
    from exllamav3 import Config, Model
    return Model.from_config(Config.from_directory(str(MODEL)))


@pytest.fixture(scope = "module", params = [(0, 5), (2, 4)], ids = ["layer0-K5S8", "layer2-K4S8"])
def k8_block(k8_model, device, request):
    _require_fused()
    layer, rb = request.param
    blk = k8_model.find_module(f"model.language_model.layers.{layer}")
    for m in (blk.mlp_hc, blk.mlp):
        m.load(device)
    blk._moe_fused, blk._moe_fused_tried = None, False
    g0, sg = blk.mlp.gates[0].inner, blk.mlp.shared_experts.gates[0].inner
    assert (g0.K, sg.K) == (rb, 8), f"layer {layer}: routed K{g0.K} / shared K{sg.K}, expected K{rb} / K8"
    yield blk
    blk._moe_fused, blk._moe_fused_tried = None, False
    for m in (blk.mlp_hc, blk.mlp):
        m.unload()


def _stack(blk, rows, seed):
    gen = torch.Generator().manual_seed(seed)
    H, D = blk.mlp_hc.hc_mult, blk.mlp.hidden_size
    return (torch.randn((1, rows, H, D), generator = gen) * 0.7 + 0.05).to(blk.mlp.shared_experts.gates[0].inner.trellis.device)


def _forward(blk, x, fused, monkeypatch):
    from exllamav3.modules import moe_fused
    calls = {"n": 0}
    orig = moe_fused.FusedMoEHalf.__call__
    def spy(self, xx):
        calls["n"] += 1
        return orig(self, xx)
    monkeypatch.setattr(moe_fused.FusedMoEHalf, "__call__", spy)
    blk._moe_fused_on = fused
    try:
        out = blk._forward_mlp(x.clone(), None, {})
    finally:
        monkeypatch.setattr(moe_fused.FusedMoEHalf, "__call__", orig)
    torch.cuda.synchronize()
    return out, calls["n"]


def _mlp_ref(gate, up, down, m):
    """fp32 SiLU-gated MLP of one expert on m (rows, D) fp16 through the reconstructed dense weights, gate / up rounded to fp16 like the kernel"""
    w = lambda l: dequant(l.trellis, l.suh, l.svh, l.K).float()
    g = (m.float() @ w(gate)).half().float()
    u = (m.float() @ w(up)).half().float()
    a = (torch.nn.functional.silu(g) * u).half().float()
    return a @ w(down)


def _moe_ref(blk, fh, rows):
    mlp, D, TOPK = blk.mlp, fh.D, fh.TOPK
    mixed = fh.ws_view("mixed", rows)[: rows * D].view(rows, D)
    sgl = fh.ws_view("sgl", rows)[:rows]
    seldbg = fh.ws_view("seldbg", rows)
    sel = seldbg[: rows * TOPK].view(rows, TOPK).long().cpu()
    wb = 128 if rows > 4 else 64
    wts = torch.from_numpy(seldbg[wb: wb + rows * TOPK].cpu().numpy().astype(np.uint16).view(np.float16).astype(np.float32)).view(rows, TOPK)
    sh = mlp.shared_experts
    y = torch.sigmoid(sgl).unsqueeze(1) * _mlp_ref(sh.gates[0].inner, sh.ups[0].inner, sh.downs[0].inner, mixed)
    for r in range(rows):
        for k in range(TOPK):
            e = int(sel[r, k])
            assert 0 <= e < fh.NEXP
            y[r] += float(wts[r, k]) * _mlp_ref(mlp.gates[e].inner, mlp.ups[e].inner, mlp.downs[e].inner, mixed[r: r + 1])[0]
    return y, fh.ws_view("ydbg", rows)[: rows * D].view(rows, D).clone()


@pytest.mark.parametrize("variant", [0x160000, 0], ids = ["mf9-split", "core5"])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@torch.inference_mode()
def test_decode_rows_take_fused_and_match_reconstruct(k8_block, rows, variant, monkeypatch):
    from exllamav3.modules import moe_fused
    if rows > 4 and variant != 0x160000:
        pytest.skip("one-launch 5..8 rows exist for the split-launch variant only")
    monkeypatch.setitem(moe_fused.MF9, "variant", variant)
    monkeypatch.setitem(moe_fused.R8, "on", rows > 4)
    blk = k8_block
    x = _stack(blk, rows, 17 + rows)
    wide0 = moe_fused.R8["calls"]
    out, calls = _forward(blk, x, True, monkeypatch)
    fh = blk._moe_fused
    assert fh is not None and fh.supported and (fh.RB, fh.SB) == (blk.mlp.gates[0].inner.K, 8), "K8 shared expert must take the fused kernel"
    assert calls == 1 and moe_fused.R8["calls"] - wide0 == (1 if rows > 4 else 0)
    assert fh.ctl().tolist() == [0, 0, 0]
    assert torch.isfinite(out).all() and not torch.equal(out, x)
    ref, ydbg = _moe_ref(blk, fh, rows)
    err = rel_err(ydbg, ref)
    assert err < LAYER_TOL, f"rows={rows} variant={variant:#x}: fused MoE output vs reconstruct rel err {err:.2e}"
    if rows > 4:
        monkeypatch.setitem(moe_fused.R8, "on", False)
        split, n = _forward(blk, x, True, monkeypatch)
        assert n == 1 + (rows + 3) // 4
        assert torch.equal(out, split), "one-launch 8 rows != 4 + 4 split"


@pytest.mark.parametrize("rows", [1, 16, 128, 1024])
@torch.inference_mode()
def test_shared_expert_k8_matches_reconstruct(k8_block, rows, monkeypatch):
    blk = k8_block
    if rows > 8:
        x = _stack(blk, rows, 5 + rows)
        out, calls = _forward(blk, x, True, monkeypatch)
        assert calls == 0, f"{rows} rows must decline the fused half-layer"
        assert torch.isfinite(out).all()
        off, _ = _forward(blk, x, False, monkeypatch)
        assert rel_err(out, off) < 1e-5, "the fused flag must not change the prefill path"
    sh = blk.mlp.shared_experts
    gen = torch.Generator().manual_seed(rows)
    m = (torch.randn((1, rows, blk.mlp.hidden_size), generator = gen) * 0.5).half().to(sh.gates[0].inner.trellis.device)
    got = sh.forward(m, {}).float().view(rows, -1)
    ref = _mlp_ref(sh.gates[0].inner, sh.ups[0].inner, sh.downs[0].inner, m.view(rows, -1))
    err = rel_err(got, ref)
    assert err < LAYER_TOL, f"rows={rows}: K8 shared expert vs reconstruct rel err {err:.2e}"
