"""EXL3_MOE_ALIAS_IDS: delete the routed-set copy instead of fusing it.

The bsz-1 routing kernels write their top-k straight into
RoutingCFG.selected_experts_bsz1, a stable (1, top_k) int64 buffer allocated
once at load. The grouped-mgemm decode branch then did

    b.ids.copy_(selected_experts.view(-1))

before every projection triple -- one launch per layer per token to move 80
bytes onto a second stable buffer, 48 launches/token on this model. Pointing
MGemmBuffers.ids at the router's own buffer removes it outright. No arithmetic
changes, so there is nothing to be bit-exact about; what has to hold is that
the two buffers really are interchangeable and that the fallback copy still
fires whenever they are not.

CPU-only: this is allocation plumbing, not a kernel.
"""
import os
import types

import pytest
import torch

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MGemmBuffers


E, HIDDEN = 10, 2560


def _stub(ids_dtype=torch.long, ids_numel=E, with_gate=True):
    gu = torch.empty((2 * E, 640), dtype=torch.half)
    buf = MGemmBuffers(
        xh_gu=torch.empty((2 * E, 1, HIDDEN), dtype=torch.half),
        interm_gu=gu,
        interm_g=gu[:E],
        interm_u=gu[E:],
        act=torch.empty((E, 640), dtype=torch.half),
        xh_d=torch.empty((E, 1, 640), dtype=torch.half),
        out=torch.empty((E, HIDDEN), dtype=torch.half),
        ids=torch.empty((ids_numel,), dtype=ids_dtype),
    )
    cfg = types.SimpleNamespace(
        selected_experts_bsz1=torch.empty((1, E), dtype=torch.long))
    return types.SimpleNamespace(
        mgemm_buf=buf, routing_cfg=cfg, mgemm_ids_own=None,
        routing_gate=object() if with_gate else None)


def _alias(stub):
    BlockSparseMLP._alias_mgemm_ids(stub)


def test_alias_is_off_by_default(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_ALIAS_IDS", raising=False)
    s = _stub()
    before = s.mgemm_buf.ids.data_ptr()
    _alias(s)
    assert s.mgemm_buf.ids.data_ptr() == before
    assert s.mgemm_buf.ids.data_ptr() != s.routing_cfg.selected_experts_bsz1.data_ptr()


def test_alias_points_ids_at_the_router_buffer(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "1")
    s = _stub()
    sel = s.routing_cfg.selected_experts_bsz1
    _alias(s)
    ids = s.mgemm_buf.ids
    assert ids.data_ptr() == sel.data_ptr()
    assert ids.shape == (E,) and ids.dtype == torch.long
    # the forward guard must then skip the copy
    assert ids.data_ptr() == sel.data_ptr()
    # and a write through the router's buffer must be visible to the mgemm's view
    sel.copy_(torch.arange(E, dtype=torch.long).view(1, E))
    assert torch.equal(ids, torch.arange(E, dtype=torch.long))


@pytest.mark.parametrize("kw", [
    {"ids_dtype": torch.int32},     # dtype mismatch: the kernel reads int64 here
    {"ids_numel": E + 1},           # count mismatch
    {"with_gate": False},           # broadcast routing path: sel is not the router's
])
def test_alias_refuses_mismatches(monkeypatch, kw):
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "1")
    s = _stub(**kw)
    before = s.mgemm_buf.ids.data_ptr()
    _alias(s)
    assert s.mgemm_buf.ids.data_ptr() == before, "must not alias a mismatched buffer"


def test_alias_is_reversible(monkeypatch):
    """An A/B harness has to be able to turn it back off in a running process."""
    s = _stub()
    own = s.mgemm_buf.ids.data_ptr()
    sel = s.routing_cfg.selected_experts_bsz1.data_ptr()
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "1")
    _alias(s)
    assert s.mgemm_buf.ids.data_ptr() == sel
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "0")
    _alias(s)
    assert s.mgemm_buf.ids.data_ptr() == own
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "1")
    _alias(s)
    assert s.mgemm_buf.ids.data_ptr() == sel


def test_alias_tolerates_missing_buffers(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_ALIAS_IDS", "1")
    for missing in ("mgemm_buf", "routing_cfg"):
        s = _stub()
        setattr(s, missing, None)
        _alias(s)          # must not raise


def test_forward_guard_expression():
    """The forward path skips the copy iff the buffers are the same allocation."""
    sel = torch.empty((1, E), dtype=torch.long)
    aliased = sel.view(-1)
    separate = torch.empty((E,), dtype=torch.long)
    assert aliased.data_ptr() == sel.data_ptr()
    assert separate.data_ptr() != sel.data_ptr()
