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


def test_flags_track_the_actual_build():
    from exllamav3.ext import exllamav3_ext as ext
    assert bsm._HAS_MGEMM == hasattr(ext, "exl3_mgemm")
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
