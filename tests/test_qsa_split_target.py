"""
EXL3_QSA_SPLIT_TARGET: the split-count heuristic in the gathered-GQA sparse kernel.

qsa_sparse_attend_rows sizes its flash-decoding splits from `2 * multi_processor_count`,
a target inherited from the large-SM parts the kernel was written on. On a 16-CU gfx1150
an M=7 verify launches programs = R * kvh * h_blocks = 7 * 2 * 1 = 14, so the target
admits splits = 2 and the grid is 28 workgroups over 16 CUs -- a 1.75/CU tail on a pure
gather. The target is the only free variable; the K/V bytes are unchanged by it, because
each split reads a disjoint slice of the same index list.

Splitting changes the fp32 reduction order of the online-softmax combine, so the gate is
a tolerance, not bit-identity.

    python -m pytest tests/test_qsa_split_target.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from util import resolve_device

qsa_triton = pytest.importorskip("exllamav3.modules.attention_fn.qsa_triton")
if not qsa_triton.has_triton:
    pytest.skip("triton unavailable", allow_module_level = True)

DEV = resolve_device()


def _case(rows = 7, qh = 24, kvh = 2, hd = 256, k_pad = 2048, n_tok = 4096, seed = 0xA5A):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    q = (torch.randn(rows, qh, hd, generator = g) * 0.1).half().to(DEV).contiguous()
    k = (torch.randn(n_tok, kvh, hd, generator = g) * 0.1).half().to(DEV).contiguous()
    v = (torch.randn(n_tok, kvh, hd, generator = g) * 0.1).half().to(DEV).contiguous()
    idx = torch.randint(0, n_tok, (rows, k_pad), generator = g).int().to(DEV).contiguous()
    return q, k, v, idx


def test_split_target_env_is_read_per_call(monkeypatch):
    """The knob must be read per call. One read at import silently no-ops as an A/B arm."""
    q, k, v, idx = _case()
    seen = []
    orig = qsa_triton._qsa_sparse_splits

    def spy(*a, **kw):
        n = orig(*a, **kw)
        seen.append(n)
        return n

    monkeypatch.setattr(qsa_triton, "_qsa_sparse_splits", spy)
    monkeypatch.delenv("EXL3_QSA_SPLIT_TARGET", raising = False)
    qsa_triton.qsa_sparse_attend_rows(q, k, v, idx, 1.0 / 16.0)
    base = seen[-1]
    monkeypatch.setenv("EXL3_QSA_SPLIT_TARGET", "8")
    qsa_triton.qsa_sparse_attend_rows(q, k, v, idx, 1.0 / 16.0)
    raised = seen[-1]
    assert raised > base, f"split target 8x did not raise splits ({base} -> {raised})"


@pytest.mark.parametrize("target", ["1", "2", "4", "8", "16", "32"])
def test_split_target_preserves_output(monkeypatch, target):
    q, k, v, idx = _case()
    monkeypatch.setenv("EXL3_QSA_SPLIT_TARGET", "0")      # 0 = force splits 1
    ref = qsa_triton.qsa_sparse_attend_rows(q, k, v, idx, 1.0 / 16.0).float()
    monkeypatch.setenv("EXL3_QSA_SPLIT_TARGET", target)
    got = qsa_triton.qsa_sparse_attend_rows(q, k, v, idx, 1.0 / 16.0).float()
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 2e-3 * max(scale, 1e-3), f"target {target}: max abs err {err} vs scale {scale}"


def test_split_target_zero_forces_single_split(monkeypatch):
    q, k, v, idx = _case()
    monkeypatch.setenv("EXL3_QSA_SPLIT_TARGET", "0")
    assert qsa_triton._qsa_sparse_splits(DEV, 14, 2048, 32) == 1
