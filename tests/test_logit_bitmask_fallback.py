"""apply_logit_bitmask is the only path that applies a packed grammar mask in the eager
sampler, and generator/sampler/custom.py calls it unguarded. sampling_fused.cu is in
ROCM_EXCLUDE_FILES and the binding sits inside the `#if !defined(USE_ROCM)` block of
bindings.cpp, so on ROCm every constrained request raises AttributeError.

The kernel (exllamav3_ext/generator/sampling_fused.cu, apply_logit_bitmask_kernel) is the
spec; _ref below transcribes it and the tests hold the fallback to it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallbacks as fb

NI = -float("inf")

# (bsz, dim, words) -- words*32 deliberately lands below, on and above dim
CASES = [
    (1, 64, 2),        # exact fit
    (1, 48, 2),        # dim not a multiple of 32, mask wider than dim
    (3, 100, 4),       # dim not a multiple of 32, mask wider than dim
    (1, 128, 2),       # padded vocab beyond the mask width: tail must be masked out
    (4, 200, 3),       # padded vocab beyond the mask width, batched
    (2, 32, 1),        # single word
    (1, 1, 1),
    (5, 1024, 32),
]


def _ref(logits_in, bitmask):
    """apply_logit_bitmask_kernel, elementwise: keep = i < bits_words*32 && bit i set."""
    bsz, dim = logits_in.shape
    words = bitmask.shape[-1]
    bits = bitmask.to(torch.int64) & 0xFFFFFFFF
    out = torch.empty_like(logits_in)
    for row in range(bsz):
        src = 0 if bitmask.shape[0] == 1 else row
        for i in range(dim):
            keep = i < words * 32 and bool((int(bits[src, i >> 5]) >> (i & 31)) & 1)
            out[row, i] = logits_in[row, i] if keep else NI
    return out


def _mk(bsz, dim, words, dtype, mask_bsz, seed = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bsz, dim, generator = g).to(dtype)
    # Sign bit of the int32 word must survive: word 0 gets bit 31 set unconditionally
    bm = torch.randint(-(2 ** 31), 2 ** 31 - 1, (mask_bsz, words), generator = g, dtype = torch.int32)
    bm[:, 0] |= torch.tensor(-(2 ** 31), dtype = torch.int32)
    return logits, bm


@pytest.mark.parametrize("bsz,dim,words", CASES)
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("mask_bcast", [False, True])
def test_fallback_matches_reference(bsz, dim, words, dtype, mask_bcast):
    logits, bm = _mk(bsz, dim, words, dtype, 1 if mask_bcast else bsz)
    out = torch.empty_like(logits)
    fb.apply_logit_bitmask(logits, out, bm)
    torch.testing.assert_close(out, _ref(logits, bm), rtol = 0, atol = 0, equal_nan = False)


@pytest.mark.parametrize("bsz,dim,words", CASES)
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.skipif(
    not hasattr(ext, "apply_logit_bitmask")
    or type(ext.apply_logit_bitmask).__name__ != "builtin_function_or_method",
    reason = "native apply_logit_bitmask not in this build",
)
def test_fallback_matches_kernel(bsz, dim, words, dtype):
    logits, bm = _mk(bsz, dim, words, dtype, bsz)
    logits_d, bm_d = logits.cuda(), bm.cuda()
    ok, of = torch.empty_like(logits_d), torch.empty_like(logits_d)
    ext.apply_logit_bitmask(logits_d, ok, bm_d)
    fb.apply_logit_bitmask(logits_d, of, bm_d)
    torch.testing.assert_close(ok, of, rtol = 0, atol = 0)


def test_fallback_does_not_modify_input():
    logits, bm = _mk(2, 96, 2, torch.float, 2)
    before = logits.clone()
    fb.apply_logit_bitmask(logits, torch.empty_like(logits), bm)
    torch.testing.assert_close(logits, before, rtol = 0, atol = 0)


def test_all_bits_clear_masks_everything():
    logits = torch.randn(2, 64)
    out = torch.empty_like(logits)
    fb.apply_logit_bitmask(logits, out, torch.zeros(1, 2, dtype = torch.int32))
    assert (out == NI).all()


def test_all_bits_set_is_a_copy():
    logits = torch.randn(2, 64)
    out = torch.empty_like(logits)
    fb.apply_logit_bitmask(logits, out, torch.full((1, 2), -1, dtype = torch.int32))
    torch.testing.assert_close(out, logits, rtol = 0, atol = 0)


# -- wiring: the sampler entry point, not just the ext function ----------------

def _sampler():
    from exllamav3.generator.sampler.custom import CustomSampler, SS_Temperature, SS_Sample_mn
    return CustomSampler([SS_Temperature(1.0), SS_Sample_mn()])


def _bits_for(allowed, words):
    bm = torch.zeros(1, words, dtype = torch.int32)
    acc = [0] * words
    for t in allowed:
        acc[t >> 5] |= 1 << (t & 31)
    for w in range(words):
        bm[0, w] = torch.tensor(acc[w] - (1 << 32) if acc[w] >= (1 << 31) else acc[w], dtype = torch.int32)
    return bm


def test_sampler_forward_suppresses_forbidden_token():
    # The logit peak is at 7 and nothing else is close; only token 40 is allowed, so a
    # sampler that ignores the mask returns 7.
    sampler = _sampler()
    assert not sampler.fused_only, "eager bitmask branch not taken; this test would not cover it"
    logits = torch.full((1, 64), -20.0)
    logits[0, 7] = 30.0
    logits[0, 40] = -19.0
    out = sampler.forward(logits.clone(), rand_u32 = 1234, logit_mask = _bits_for([40], 2))
    assert out.flatten().tolist() == [40], f"sampled {out.flatten().tolist()}, mask allowed only [40]"


def test_sampler_forward_bitmask_matches_dense_mask():
    torch.manual_seed(7)
    logits = torch.randn(1, 256) * 4
    allowed = [3, 31, 32, 33, 100, 255]
    dense = torch.full((1, 256), NI, dtype = torch.half)
    dense[0, allowed] = 0.0
    a = _sampler().forward(logits.clone(), rand_u32 = 99, logit_mask = _bits_for(allowed, 8), return_state = True)
    b = _sampler().forward(logits.clone(), rand_u32 = 99, logit_mask = dense, return_state = True)
    assert a.sample.flatten().tolist() == b.sample.flatten().tolist()
    assert a.sample.flatten().item() in allowed


def test_sampler_forward_masks_tail_beyond_mask_width():
    # Mask covers 64 of 128 logits; the peak sits in the uncovered tail and must be suppressed.
    sampler = _sampler()
    logits = torch.full((1, 128), -20.0)
    logits[0, 120] = 40.0
    logits[0, 5] = -19.0
    out = sampler.forward(logits.clone(), rand_u32 = 1, logit_mask = _bits_for([5], 2))
    assert out.flatten().tolist() == [5], f"sampled {out.flatten().tolist()} from the unmasked tail"
