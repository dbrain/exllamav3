"""Expert counting on the multi-token MoE path (block_sparse_mlp.py).

Why this exists. An MTP verification round feeds the target model ndt+1 tokens
where plain decode feeds 1. Measured on flashnext-4.05bpw, that one forward cost
644 ms against a ~125 ms plain decode step, and cProfile put `torch.bincount` at
the top of the whole run by self time -- 3.326 s over 960 calls (48 MoE layers x
~20 rounds), i.e. 3.47 ms for a count over 50 elements. That is not arithmetic,
it is a stall: bincount on CUDA reads max(input) back to the host to size its
bins, which is why graph_decode._UNCAPTURABLE_TYPES already lists BlockSparseMLP
as host-syncing "on torch.bincount".

The count is then handed straight to ext.exl3_moe as a DEVICE tensor
(block_sparse_mlp.py:1241) and, whenever num_tokens * top_k <= TEMP_ROWS_FUSED,
is never read on the host at all (:1276-1278, expert_count_list = None). So the
sync buys nothing on exactly the shapes MTP verify produces.

The bin count is known statically -- flat_expert_local is clamped to [0, E] with
E as the sentinel -- so a scatter_add over a pre-sized zero vector is an exact,
integer-identical, sync-free replacement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

from exllamav3.modules.block_sparse_mlp import _expert_counts

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _assignments(num_tokens, top_k, num_experts, sentinel = False, seed = 0):
    """Flattened [num_tokens * top_k] local expert ids, shaped as the real path
    builds them: selected_experts.reshape(-1), sentinel E for out-of-slice."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, num_experts, (num_tokens * top_k,), generator = g)
    if sentinel:
        ids[::3] = num_experts                      # TP / CPU-split out-of-slice marker
    return ids


# ------------------------------------------------------------------- exactness

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 12, 64])
@pytest.mark.parametrize("sentinel", [False, True])
def test_counts_match_bincount_exactly(num_tokens, sentinel):
    """Integer counts, so 'close enough' is not a category here: any difference
    changes which rows the fused kernel reads."""
    E, top_k = 512, 10
    ids = _assignments(num_tokens, top_k, E, sentinel)
    want = torch.bincount(ids, minlength = E + 1)
    got = _expert_counts(ids, E)
    assert torch.equal(got, want), (got - want).abs().max().item()
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert int(got.sum()) == num_tokens * top_k


def test_counts_cover_the_degenerate_shapes():
    E = 8
    for ids in (torch.zeros(0, dtype = torch.long),
                torch.zeros(7, dtype = torch.long),
                torch.full((7,), E, dtype = torch.long)):
        assert torch.equal(_expert_counts(ids, E),
                           torch.bincount(ids, minlength = E + 1))


# ------------------------------------------------------------------ no syncing

@CUDA
def test_bincount_really_does_sync_and_the_replacement_does_not():
    """The control and the arm, per campaign rule 2: an arm whose behaviour MUST
    differ, checked rather than assumed. If bincount ever stops syncing this test
    fails loudly and the replacement can be reverted."""
    E, dev = 512, torch.device("cuda:0")
    ids = _assignments(5, 10, E).to(dev)

    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match = "(?i)sync"):
            torch.bincount(ids, minlength = E + 1)
        _expert_counts(ids, E)                       # must not raise
    finally:
        torch.cuda.set_sync_debug_mode("default")


@CUDA
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8])
def test_counts_match_on_device_across_verify_shapes(num_tokens):
    """seqlen 1 is plain decode; 5 is an ndt=4 MTP verify; 2/3/8 are the short
    prefill and prompt-cache tails that reach the same branch."""
    E, dev = 512, torch.device("cuda:0")
    ids = _assignments(num_tokens, 10, E, sentinel = True).to(dev)
    assert torch.equal(_expert_counts(ids, E),
                       torch.bincount(ids, minlength = E + 1))
