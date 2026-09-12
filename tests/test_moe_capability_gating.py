"""The mgemm/fused expert paths must stay unreachable on builds lacking their kernels.

quant/exl3_gemm.cu and quant/exl3_moe.cu are excluded from the ROCm build, so
support_quant_paths must be False there or forward() reaches ext.exl3_mgemm and raises.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from exllamav3.modules import block_sparse_mlp as bsm


class _Lin:
    def __init__(self, bias = None, trim = False, out_features = 8, out_unpadded = 8):
        self.inner = type("_Inner", (), {"bias": bias})()
        self.trim_padded_out = trim
        self.out_features = out_features
        self.out_features_unpadded = out_unpadded


def _supports(has_mgemm = True, **over):
    kw = dict(is_quantized = True, gated = True, activation_fn = "silu",
              gates = [_Lin()], ups = [_Lin()], downs = [_Lin()])
    kw.update(over)
    prev = bsm._HAS_MGEMM
    bsm._HAS_MGEMM = has_mgemm
    try:
        return bsm._supports_quant_paths(**kw)
    finally:
        bsm._HAS_MGEMM = prev


def test_missing_mgemm_kernel_disables_quant_paths():
    assert _supports(has_mgemm = False) is False


def test_present_mgemm_kernel_allows_quant_paths():
    assert _supports(has_mgemm = True) is True


def test_missing_kernel_overrides_every_otherwise_eligible_config():
    for over in ({}, {"gated": False, "activation_fn": "relu2"}, {"activation_fn": "gelu"}):
        assert _supports(has_mgemm = False, **over) is False


@pytest.mark.parametrize("over", [
    {"is_quantized": False},
    {"activation_fn": "swiglu_oai"},
    {"gated": False, "activation_fn": "silu"},
    {"downs": [_Lin(trim = True, out_features = 16, out_unpadded = 8)]},
    {"ups": [_Lin(bias = object())]},
])
def test_preexisting_constraints_still_reject(over):
    assert _supports(has_mgemm = True, **over) is False


def test_moe_flag_tracks_the_actual_build():
    from exllamav3.ext import exllamav3_ext as ext
    assert bsm._HAS_MOE == hasattr(ext, "exl3_moe")


# -- grouped Triton mgemm (decode) --------------------------------------------

def _supports_grouped(has_triton = True, has_native = False, **over):
    kw = dict(is_quantized = True, gated = True, activation_fn = "silu",
              gates = [_Lin()], ups = [_Lin()], downs = [_Lin()],
              num_local_experts = 8, num_experts = 8)
    kw.update(over)
    prev_t, prev_n = bsm._HAS_TRITON_MGEMM, bsm._HAS_MGEMM
    bsm._HAS_TRITON_MGEMM, bsm._HAS_MGEMM = has_triton, has_native
    try:
        return bsm._supports_grouped_mgemm(**kw)
    finally:
        bsm._HAS_TRITON_MGEMM, bsm._HAS_MGEMM = prev_t, prev_n


def test_grouped_used_when_native_mgemm_absent():
    assert _supports_grouped() is True


def test_grouped_yields_to_the_native_kernel():
    assert _supports_grouped(has_native = True) is False


def test_grouped_needs_triton_module():
    assert _supports_grouped(has_triton = False) is False


@pytest.mark.parametrize("over", [
    {"num_local_experts": 4},                     # TP / CPU-split shard: sentinel id would deref
    {"is_quantized": False},
    {"gated": False},
    {"activation_fn": "relu2"},
    {"ups": [_Lin(bias = object())]},
    {"downs": [_Lin(trim = True, out_features = 16, out_unpadded = 8)]},
])
def test_grouped_rejects_unsupported(over):
    assert _supports_grouped(**over) is False


# -- EXL3_NO_MGEMM: containing a build that exposes exl3_mgemm ------------------

class _ExtWithMgemm:
    """Proxies the real extension, forcing exl3_mgemm present or absent."""

    def __init__(self, real, present):
        self._real = real
        self._present = present

    def __getattr__(self, name):
        if name == "exl3_mgemm":
            if not self._present:
                raise AttributeError(name)
            return lambda *a, **kw: None
        return getattr(self._real, name)


def _patch_build(monkeypatch, present, env, hip):
    import torch
    from exllamav3 import ext as ext_mod
    monkeypatch.setattr(ext_mod, "exllamav3_ext",
                        _ExtWithMgemm(ext_mod.exllamav3_ext, present))
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    if env is None:
        monkeypatch.delenv("EXL3_NO_MGEMM", raising = False)
    else:
        monkeypatch.setenv("EXL3_NO_MGEMM", env)


@pytest.mark.parametrize("present, env, hip, expected", [
    (False, None, "7.0", False),
    (True,  None, "7.0", False),   # ROCm default: symbol present, path still off
    (True,  "0",  "7.0", True),    # explicit opt-in
    (True,  "1",  None,  False),   # explicit opt-out on CUDA
    (True,  None, None,  True),    # CUDA default unchanged
    (False, "0",  "7.0", False),   # opt-in cannot conjure a missing kernel
])
def test_mgemm_kernels_available(monkeypatch, present, env, hip, expected):
    from exllamav3.model import config as cfg
    _patch_build(monkeypatch, present, env, hip)
    assert cfg.mgemm_kernels_available() is expected


def test_use_mgemm_follows_the_gate(monkeypatch):
    from exllamav3.model.config import InferParams, mgemm_kernels_available
    _patch_build(monkeypatch, present = True, env = None, hip = "7.0")
    assert mgemm_kernels_available() is False
    assert InferParams().use_mgemm(4, 4096) is False


def test_grouped_moe_survives_a_build_exposing_the_symbol(monkeypatch):
    """The regression this gate exists for: _HAS_MGEMM flipping True drops both
    _supports_grouped_mgemm and, with it, caps["graph_capturable"]."""
    import importlib
    _patch_build(monkeypatch, present = True, env = None, hip = "7.0")
    mod = importlib.reload(bsm)
    try:
        assert mod._HAS_MGEMM is False
        assert mod._supports_grouped_mgemm(
            is_quantized = True, gated = True, activation_fn = "silu",
            gates = [_Lin()], ups = [_Lin()], downs = [_Lin()],
            num_local_experts = 8, num_experts = 8) is True
    finally:
        monkeypatch.undo()
        importlib.reload(bsm)


def test_flags_track_the_gate():
    from exllamav3.model.config import mgemm_kernels_available
    assert bsm._HAS_MGEMM == mgemm_kernels_available()
