"""
g_tensor_cache must not hand an INFERENCE tensor to a caller outside inference_mode.

A tensor allocated inside `torch.inference_mode()` stays an inference tensor for life, and
any in-place write to one outside that scope raises

    RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.

The cache is global and outlives both scopes, so one workspace first taken during a forward
poisons that tag for every later non-inference user. It bites for real at
`GatedResidual._prepare` (`hyperconnections.py`), which takes the `gr_prep_tmp` bucket and
`copy_`s into it during model load: running any inference-mode test before a load in the same
process makes the load raise. Observed as 2 failures in tests/test_hc_fused_max_r.py when it
runs after tests/test_hc_mix_q8_rtile.py, and passing when it runs alone.

    python -m pytest tests/test_tensor_cache_inference.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.util.tensor import g_tensor_cache


def test_bucket_taken_in_inference_mode_is_writable_outside_it():
    tag = "test_inference_poison"
    with torch.inference_mode():
        a = g_tensor_cache.get_bucketed("cpu", 16, torch.float, tag)
        assert a.is_inference()
    b = g_tensor_cache.get_bucketed("cpu", 16, torch.float, tag)
    assert not b.is_inference(), "cache returned an inference tensor outside inference_mode"
    b.copy_(torch.ones(16))


def test_normal_tensor_is_reused_and_not_reallocated():
    tag = "test_inference_reuse"
    a = g_tensor_cache.get_bucketed("cpu", 16, torch.float, tag)
    b = g_tensor_cache.get_bucketed("cpu", 16, torch.float, tag)
    assert a.data_ptr() == b.data_ptr(), "a non-inference entry must still be cached"
    with torch.inference_mode():
        c = g_tensor_cache.get_bucketed("cpu", 16, torch.float, tag)
    assert c.data_ptr() == a.data_ptr(), "a normal tensor is usable inside inference_mode"
