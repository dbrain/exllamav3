"""GatedResidual row-padded output is for the attention site only: the MLP site's consumers (MoE router,
fused / reconstruct expert tiers) index the block input as contiguous rows, so a padded view there reads
the uninitialized pad columns (NaN rows, out-of-range expert ids, GPU hang in the autosplit forward)."""
from __future__ import annotations

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import hyperconnections as hc
from exllamav3.modules.quant.exl3 import PAD_MIN_ROWS

pytestmark = pytest.mark.skipif(
    not (torch.version.hip and torch.cuda.is_available() and hasattr(ext, "gr_mix")),
    reason = "ROCm GatedResidual HIP mix path")

DEVICE = "cuda:0"
H, D, RANK = 4, 2560, 320


@pytest.fixture(scope = "module")
def gr_module():
    torch.manual_seed(0)
    m = object.__new__(hc.GatedResidual)
    m.hc_mult, m.hidden_size, m.rms_eps, m.use_combine = H, D, 1e-6, True
    m.out_dtype = None
    m.norm_w_raw = (torch.randn((H, D), device = DEVICE) * 0.15).half()
    m.norm_w = m.w_h = None
    m.down_h = m.up_h = m.upx_h = m.inject_h = m.proj_h = None
    m.rank, m.key, m.next_site = 0, "test.gr", None
    m._prepare((torch.randn((RANK, H * D), device = DEVICE) * 0.02).half(),
               (torch.randn((H * D, RANK), device = DEVICE) * 0.02).half(),
               (torch.randn((H, H * D), device = DEVICE) * 0.02).half(),
               keep_source_weights = True)
    m.tiled = False
    return m


@pytest.mark.parametrize("site, padded", [(True, True), ("mlp", False), (None, False)], ids = ["attn", "mlp", "none"])
def test_mix_norm_pads_only_for_attention_site(gr_module, monkeypatch, site, padded):
    if hc._gr_hip() is None:
        pytest.skip("EXL3_GR_HIP kernels unavailable")
    monkeypatch.setenv("EXL3_GR_HIP", "1")
    streams = torch.randn((1, PAD_MIN_ROWS, H, D), device = DEVICE) * 0.25
    params = {} if site is None else {"hc_pad_ok": site}
    _, _, y, _ = gr_module.mix_norm(streams, params, None)
    y2 = y.view(-1, D)
    assert y2.is_contiguous() != padded
    if padded:
        assert y2.stride(0) == hc._pad_pitch_for(D)
