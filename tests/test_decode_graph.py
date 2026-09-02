"""Span layout and eligibility gates for decode graph capture (graph_decode.py).

Pure planning logic: no model load, no GPU. The capture path itself is
validated by running a model, since matching the eager path is the real gate.
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
def _blk(inner = (), key = "blk"): return FakeMod("TransformerBlock", key, {}, subs = inner)
def _head(): return FakeMod("Linear", "lm_head", {"logits_output": True})
def _moe(key = "blk.mlp"): return FakeMod("BlockSparseMLP", key)
def _gdn(key = "blk.linear_attn"):
    return FakeMod("GatedDeltaNet", key, {"recurrent_cache": True})
def _ple(): return FakeMod("PLELayer", "ple", {"recurrent_cache": True})

def _plan(mods):
    g = gd.DecodeGraphs(FakeModel(mods))
    ok = g.eligible()
    return g, ok, (g.layout or [])

def _spans(layout):
    """Readable form: ('G'|'e', start, end)."""
    return [("G" if c else "e", s, e) for c, s, e, _ in layout]


# --------------------------------------------------------------- eligibility

def test_dense_is_one_span_after_the_cpu_embedding():
    g, ok, layout = _plan([_emb(), _blk(), _blk(), _head()])
    assert ok
    assert _spans(layout) == [("e", 0, 1), ("G", 1, 4)]


def test_moe_blocks_become_eager_leaving_only_the_head():
    """Every block host-syncs, so only the trailing norm/head span is captured.
    Correct, but nearly worthless -- the MoE work is what unlocks these models."""
    g, ok, layout = _plan([_emb(), _blk([_moe()]), _blk([_moe()]), _head()])
    assert ok
    assert _spans(layout) == [("e", 0, 3), ("G", 3, 4)]


def test_nothing_capturable_is_rejected():
    g, ok, _ = _plan([_emb(), _blk([_moe()]), _ple()])
    assert not ok
    assert "no capturable span" in g._reason


@pytest.mark.parametrize("name,mods,want", [
    # PLE island in the middle -- the Flash-Next shape (ple_layer_ids = [2])
    ("gap_in_middle", [_emb(), _blk(), _ple(), _blk(), _head()],
     [("e", 0, 1), ("G", 1, 2), ("e", 2, 3), ("G", 3, 5)]),
    # gap immediately at the first capturable module
    ("gap_at_front", [_emb(), _ple(), _blk(), _blk(), _head()],
     [("e", 0, 2), ("G", 2, 5)]),
    # gap at the very end
    ("gap_at_end", [_emb(), _blk(), _blk(), _ple()],
     [("e", 0, 1), ("G", 1, 3), ("e", 3, 4)]),
    # two gaps -> three captured spans
    ("two_gaps", [_emb(), _blk(), _ple(), _blk(), _ple(), _blk(), _head()],
     [("e", 0, 1), ("G", 1, 2), ("e", 2, 3), ("G", 3, 4), ("e", 4, 5), ("G", 5, 7)]),
    # adjacent gaps coalesce into one eager span
    ("adjacent_gaps", [_emb(), _blk(), _ple(), _ple(), _blk(), _head()],
     [("e", 0, 1), ("G", 1, 2), ("e", 2, 4), ("G", 4, 6)]),
])
def test_span_layouts(name, mods, want):
    g, ok, layout = _plan(mods)
    assert ok, g._reason
    assert _spans(layout) == want


@pytest.mark.parametrize("name,mods,want_reason", [
    ("cpu_interior", [_emb(), _blk(), FakeMod("MLP", "cpu_mlp", {"prefer_cpu": True}), _head()],
     None),   # a CPU module in the interior is just another eager span now
    ("multi_device", [_emb(), _blk(), FakeMod("TransformerBlock", "b2", {}, device = "cuda:1"), _head()],
     "across devices"),
])
def test_global_rejections(name, mods, want_reason):
    g, ok, layout = _plan(mods)
    if want_reason is None:
        assert ok and _spans(layout) == [("e", 0, 1), ("G", 1, 2), ("e", 2, 3), ("G", 3, 4)]
    else:
        assert not ok and want_reason in g._reason


def test_tensor_parallel_rejected():
    m = FakeModel([_emb(), _blk(), _head()])
    m.loaded_tp = True
    g = gd.DecodeGraphs(m)
    assert not g.eligible()
    assert "tensor-parallel" in g._reason


# ------------------------------------------------------- per-module capability

@pytest.mark.parametrize("mod,capturable", [
    (_blk(), True),
    (_moe(), False),
    (_gdn(), True),                                        # validated on 35b-a3b
    (FakeMod("Mamba2", "ssm", {"recurrent_cache": True}), False),      # untested
    (FakeMod("SlidingAttention", "swa", {"recurrent_cache": True}), False),
    (FakeMod("ShortConv", "sc", {"recurrent_cache": True}), False),
    (_ple(), False),
    (FakeMod("QSAIndexer", "qsa"), False),
    # a module can opt itself in from its own file
    (FakeMod("BlockSparseMLP", "m", {"graph_capturable": True}), True),
])
def test_module_capability(mod, capturable):
    assert gd._module_capturable(mod) is capturable


# ------------------------------------------------------------ span exemption

def test_exempt_is_per_span_not_per_model():
    """A dense span in a hybrid model must still be gated bitwise; only the
    span holding the non-deterministic recurrent kernel is exempt."""
    mods = [_emb(), _blk([_gdn()]), _ple(), _blk(), _head()]
    g, ok, layout = _plan(mods)
    assert ok
    by_span = {(s, e): x for c, s, e, x in layout if c}
    assert by_span[(1, 2)] is True,  "GDN span should be exempt from bitwise verify"
    assert by_span[(3, 5)] is False, "dense span must still be verified bitwise"


def test_dense_model_has_no_exempt_span():
    _, ok, layout = _plan([_emb(), _blk(), _head()])
    assert ok and not any(x for c, _, _, x in layout if c)


# ---------------------------------------------------------------- call gate

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


def test_recurrent_state_tensors_empty_for_dense():
    assert gd._recurrent_state_tensors({}) == []
    assert gd._recurrent_state_tensors({"recurrent_states": None}) == []
