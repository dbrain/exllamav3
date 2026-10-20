"""PLE row slabs (EXL3_PLE_SUB_CHUNK) vs the mid-chunk checkpoint taps (EXL3_PF_NO_TAIL): the conv window after
row r must be the full-stream columns [r, r + win) whichever slab holds them, not an index into the last slab."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import ple

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and hasattr(ext, "ple_forward_streams")),
                                reason = "ple_forward_streams binding")

DEVICE = "cuda:0"
B, S, H, D, E, KS, DIL = 1, 40, 4, 64, 32, 4, 3
WIN = (KS - 1) * DIL


def _layer():
    g = torch.Generator(device = "cpu").manual_seed(0)
    r = lambda *s, scale = 1.0: (torch.randn(*s, generator = g) * scale).to(DEVICE)
    m = object.__new__(ple.PLELayer)
    emb = r(B, S, E).half()
    m.ple_embedding = SimpleNamespace(forward = lambda *_a, **_k: emb)
    proj = lambda w: SimpleNamespace(quant_type = "fp16", inner = SimpleNamespace(weight = w, bias = None))
    m.key_proj = proj(r(E, H * D, scale = E ** -0.5).half())
    m.value_proj = proj(r(E, D, scale = E ** -0.5).half())
    norm = lambda: SimpleNamespace(weight = r(H * D, scale = 0.3).half(), rms_norm_eps = 1e-6)
    m.norm_key, m.norm_query, m.norm_conv = norm(), norm(), norm()
    m.hc_mult, m.hidden_size = H, D
    m.gate_scale, m.conv_dilation, m.conv_state_len = D ** -0.5, DIL, WIN
    m.conv_w = r(H * D, 1, KS, scale = 0.5).half()
    return m, r(B, S, H, D, scale = 2.0).float(), r(B, H * D, WIN).half()


def test_taps_match_unslabbed_full_stream(monkeypatch):
    m, streams, cs = _layer()
    rows = [1, 7, 8, 9, 15, 16, 23, 33, 39]
    history = torch.zeros((B, S + 2), dtype = torch.long)

    monkeypatch.setattr(ple, "_ple_sub_range", 0)
    full_taps = dict.fromkeys(rows)
    s_full = streams.clone()
    _, full = m.forward_streams(s_full, history, {}, conv_state = cs, taps = full_taps)
    assert full.shape[-1] == WIN + S

    monkeypatch.setattr(ple, "_ple_sub_range", 8)
    taps = dict.fromkeys(rows)
    s_slab = streams.clone()
    _, last = m.forward_streams(s_slab, history, {}, conv_state = cs, taps = taps)
    assert last.shape[-1] == WIN + 8

    torch.testing.assert_close(s_slab, s_full, rtol = 0, atol = 0)
    torch.testing.assert_close(last[..., -WIN:], full[..., -WIN:], rtol = 0, atol = 0)
    for r in rows:
        torch.testing.assert_close(full_taps[r], full[0, :, r:r + WIN], rtol = 0, atol = 0)
        torch.testing.assert_close(taps[r], full[0, :, r:r + WIN], rtol = 0, atol = 0)
