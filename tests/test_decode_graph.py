"""Eligibility gates for whole-step decode graph capture (graph_decode.py).

Pure gate logic: no model load, no GPU. The capture path itself is validated by
running a model, since bit-identity against the eager path is the real gate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.model import graph_decode as gd


class FakeMod:
    def __init__(self, name, key, caps = None, device = "cuda:0", subs = ()):
        self.__class__ = type(name, (FakeMod,), {})
        self.key = key
        self.caps = caps or {}
        self.device = torch.device(device) if device else None
        self.modules = list(subs)

    def __iter__(self):
        yield self
        for m in self.modules:
            yield from m


class FakeModel:
    loaded_tp = False
    def __init__(self, mods):
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(mods)]


def _emb():  return FakeMod("Embedding", "emb", {"prefer_cpu": True}, device = "cpu")
def _blk(inner = ()): return FakeMod("TransformerBlock", "blk", {}, subs = inner)
def _head(): return FakeMod("Linear", "lm_head", {"logits_output": True})


@pytest.mark.parametrize("name,mods,want_ok,want_reason", [
    ("dense", [_emb(), _blk(), _blk(), _head()], True, "capturing modules"),
    # MoE: the ROCm expert loop host-syncs (bincount / tolist / range(num_ex))
    ("moe", [_emb(), _blk([FakeMod("BlockSparseMLP", "blk.mlp")]), _head()],
     False, "BlockSparseMLP"),
    # ...and can opt itself in from its own file once that loop is static
    ("moe_opt_in", [_emb(), _blk([FakeMod("BlockSparseMLP", "blk.mlp",
                                          {"graph_capturable": True})]), _head()],
     True, "capturing modules"),
    ("recurrent", [_emb(), _blk([FakeMod("Mamba2", "blk.ssm", {"recurrent_cache": True})]), _head()],
     False, "Mamba2"),
    ("qsa", [_emb(), _blk([FakeMod("QSAIndexer", "blk.qsa")]), _head()], False, "QSAIndexer"),
    ("cpu_interior", [_emb(), _blk(), FakeMod("MLP", "cpu_mlp", {"prefer_cpu": True}), _head()],
     False, "interior"),
    ("multi_device", [_emb(), _blk(), FakeMod("TransformerBlock", "b2", {}, device = "cuda:1"), _head()],
     False, "across devices"),
])
def test_model_eligibility(name, mods, want_ok, want_reason):
    g = gd.DecodeGraphs(FakeModel(mods))
    assert g.eligible() is want_ok, g._reason
    assert want_reason in g._reason


def test_tensor_parallel_rejected():
    m = FakeModel([_emb(), _blk(), _head()])
    m.loaded_tp = True
    g = gd.DecodeGraphs(m)
    assert not g.eligible()
    assert "tensor-parallel" in g._reason


_BT = torch.zeros((1, 16), dtype = torch.int32)
_CS = torch.zeros(1, dtype = torch.int32)
_BASE = {"block_table": _BT, "cache_seqlens": _CS}
_ID1 = torch.zeros((1, 1), dtype = torch.long)

@pytest.mark.parametrize("name,ids,params,want", [
    ("plain_decode", _ID1, _BASE, True),
    ("multi_token", torch.zeros((1, 8), dtype = torch.long), _BASE, False),
    # batch_shape mode bakes a python-int rope position into the graph
    ("no_block_table", _ID1, {"cache_seqlens": _CS}, False),
    ("prefill_flag", _ID1, {**_BASE, "prefill": True}, False),
    ("position_ids", _ID1, {**_BASE, "position_ids": torch.zeros(1)}, False),
    ("sim_kvq", _ID1, {**_BASE, "sim_kvq": (4, 4)}, False),
    ("batched", torch.zeros((2, 1), dtype = torch.long), _BASE, False),
])
def test_per_call_gate(name, ids, params, want):
    g = gd.DecodeGraphs(FakeModel([_emb(), _blk(), _head()]))
    assert g._capturable_call(ids, params) is want


def test_signature_tracks_block_table_width():
    """Width rounds up in 16-page steps, so context growth mints a new graph."""
    g = gd.DecodeGraphs(FakeModel([_emb(), _blk(), _head()]))
    s1 = g._signature(_ID1, _BASE)
    wide = {**_BASE, "block_table": torch.zeros((1, 32), dtype = torch.int32)}
    assert g._signature(_ID1, wide) != s1

def test_signature_ignores_buffer_contents():
    """Cache length and page indices change every step; that must not recapture."""
    g = gd.DecodeGraphs(FakeModel([_emb(), _blk(), _head()]))
    s1 = g._signature(_ID1, _BASE)
    moved = {"block_table": _BT.clone() + 3, "cache_seqlens": _CS.clone() + 977}
    assert g._signature(_ID1, moved) == s1
