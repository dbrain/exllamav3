"""Contract guard: dsa_topk scans `scores.size(1)`, so its caller owns the scan bound.

This file exists because that contract LOOKS violated and is not. `dsa_indexer_scores` takes a
score backing whose row stride S_stride is a kernel constexpr, deliberately decoupled from the
visible pool length so the kernel is not recompiled at every pool boundary, and its own comment
says "The kernels only write columns < T either way". `dsa_topk.cu` then does
`int T = scores.size(1)`. Six of the eight call sites pass `t_ptr = None`, which reads like a
scan over unwritten memory -- and `qsa_indexer` takes its backing from `g_tensor_cache`, so that
tail would hold the previous tile's real scores, the values most able to win a top-k.

It is safe because `dsa_indexer_scores` ends with `return scores[:, :T]`. Callers never see the
padded width. The multirow path is the one case where the returned slice (t_cap) can exceed the
device-side written extent, and that is exactly the path that passes a per-row `t_ptr`.

So these tests do not reproduce a bug; they pin the invariant that makes the absence of one
load-bearing. `test_unbounded_scan_is_selectable` builds the unbounded case BY HAND to show the
mechanism is real, so that if anyone ever removes the slice in `dsa_indexer_scores` or hands
`dsa_topk` a raw backing, the guard above fails loudly instead of silently degrading selection.
"""
import sys, os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
DEV = "cuda:0"
POISON = 6.0e4


def _score(R, T, s_stride, cr = 8, H_i = 4, D_i = 128, seed = 3, fill = POISON):
    """Returns (what the function hands back, the raw backing it was given).

    Keeping both apart matters: the backing is s_stride wide with a poisoned tail, and the
    return value is the slice that makes every caller safe. Conflating them is exactly the
    mistake that makes this look like a live bug.
    """
    g = torch.Generator(device = "cpu").manual_seed(seed)
    q = (torch.randn(R, H_i, D_i, generator = g) * 0.1).half().to(DEV)
    w = (torch.randn(R, H_i, generator = g) * 0.1).half().to(DEV)
    k = (torch.randn(T, D_i, generator = g) * 0.1).half().to(DEV)
    backing = torch.full((R, s_stride), fill, dtype = torch.half, device = DEV)
    returned = dsa_indexer_scores(q, w, k, T * cr, cr, T, scores = backing)
    return returned, backing


@CUDA
@pytest.mark.parametrize("T,s_stride", [(448, 8192), (300, 512), (1000, 8192), (129, 256)])
def test_topk_never_selects_beyond_the_written_extent(T, s_stride):
    """The qsa_indexer shape: SEL_TILE 8192 gives s_stride 8192 while the visible pool is
    (pos0 + rows) // compress_rate, so at any depth below ~65k tokens most of the scanned
    width was never written."""
    R, k = 4, 64
    sc, _ = _score(R, T, s_stride)
    idx = torch.empty((R, k), dtype = torch.int32, device = DEV)
    ext.dsa_topk(sc, idx, k, None, 0)
    torch.cuda.synchronize()
    bad = idx[(idx >= T) | (idx < 0)]
    assert bad.numel() == 0, \
        f"T={T} stride={s_stride}: {bad.numel()} of {idx.numel()} selections outside the pool, " \
        f"e.g. {bad[:8].tolist()}"


@CUDA
@pytest.mark.parametrize("T,s_stride", [(448, 8192), (300, 512)])
def test_unbounded_scan_is_selectable(T, s_stride):
    """The mechanism, built by hand: handed the FULL padded row with no t_ptr, the selection
    does reach the unwritten tail. No shipping call site does this -- dsa_indexer_scores slices
    before returning -- but this is what that slice is buying, so it is asserted rather than
    assumed."""
    R, k = 4, 64
    _, backing = _score(R, T, s_stride)
    idx = torch.empty((R, k), dtype = torch.int32, device = DEV)
    ext.dsa_topk(backing, idx, k, None, 0)
    torch.cuda.synchronize()
    assert (idx >= T).any(), \
        "poisoned tail was NOT selected -- this control no longer proves anything; " \
        "check that dsa_topk still takes its scan width from scores.size(1)"


@CUDA
def test_selection_does_not_depend_on_prior_buffer_contents():
    """Two identical selections whose backing differs only in the unwritten tail must agree.
    This is the property the deployed path actually needs, since qsa_indexer reuses one
    g_tensor_cache buffer across tiles, layers and tokens."""
    R, T, s_stride, k = 4, 448, 8192, 64
    a, _ = _score(R, T, s_stride, fill = POISON)
    b, _ = _score(R, T, s_stride, fill = -POISON)
    ia = torch.empty((R, k), dtype = torch.int32, device = DEV)
    ib = torch.empty((R, k), dtype = torch.int32, device = DEV)
    ext.dsa_topk(a, ia, k, None, 0)
    ext.dsa_topk(b, ib, k, None, 0)
    torch.cuda.synchronize()
    assert torch.equal(ia, ib), "selection changed with the tail contents"


@CUDA
@pytest.mark.parametrize("T,s_stride", [(448, 8192), (300, 512), (129, 256), (1000, 8192)])
def test_scores_are_returned_sliced_to_the_written_extent(T, s_stride):
    """THE load-bearing invariant, and the reason none of the `t_ptr = None` call sites are bugs.

    `dsa_indexer_scores` ends with `return scores[:, :T]`, so a caller that hands in a wide
    backing still gets a view of exactly the written width back. Every consumer -- `dsa_topk`,
    which takes its scan bound from `scores.size(1)`, and `dsa_topk_tile`, which has no t_ptr
    parameter at all and therefore CANNOT be bounded any other way -- is correct only because
    of this line. Delete it and six call sites start selecting from stale scores at once.
    """
    R = 4
    sc, backing = _score(R, T, s_stride)
    assert backing.shape == (R, s_stride)
    assert sc.shape == (R, T), \
        f"expected the returned view to be sliced to T={T}, got {tuple(sc.shape)}"
    assert sc.data_ptr() != 0 and sc.stride(1) == 1
