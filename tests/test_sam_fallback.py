"""BC_SAM semantics: the (start, end) span returned by accept_tensor is the earliest occurrence of
the longest suffix of the sequence that already occurred earlier in it. Checked against a brute
force search, incrementally (one token per call, as generator/job.py drives it), in bulk, and
across the rewind path that shrinks the sequence.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def ref_span(seq):
    total = len(seq)
    if total < 1:
        return -1, -1
    for L in range(total - 1, 0, -1):
        pat = seq[total - L:]
        for i in range(0, total - L):
            if seq[i:i + L] == pat:
                return i, i + L
    return -1, -1


def _t(seq):
    return torch.tensor([seq], dtype = torch.long)


@pytest.mark.parametrize("alphabet,length", [(3, 200), (8, 300), (50, 200), (1, 40)])
def test_incremental_matches_bruteforce(alphabet, length):
    random.seed(alphabet * 1000 + length)
    seq = [random.randrange(alphabet) for _ in range(length)]
    sam = ext.BC_SAM()
    for n in range(1, length + 1):
        got = tuple(sam.accept_tensor(_t(seq[:n])))
        want = ref_span(seq[:n])
        assert got == want, f"n={n}: {got} != {want}"


def test_bulk_equals_incremental():
    random.seed(7)
    seq = [random.randrange(5) for _ in range(400)]
    a = ext.BC_SAM()
    for n in range(1, len(seq) + 1):
        last = a.accept_tensor(_t(seq[:n]))
    b = ext.BC_SAM()
    assert tuple(b.accept_tensor(_t(seq))) == tuple(last)
    assert a.length() == b.length() == len(seq)


def test_repeated_call_no_new_tokens():
    sam = ext.BC_SAM()
    seq = [1, 2, 3, 1, 2]
    assert tuple(sam.accept_tensor(_t(seq))) == ref_span(seq)
    assert tuple(sam.accept_tensor(_t(seq))) == (-1, -1)
    assert sam.length() == len(seq)


def test_rewind_rebuilds():
    random.seed(11)
    seq = [random.randrange(4) for _ in range(120)]
    sam = ext.BC_SAM()
    sam.accept_tensor(_t(seq))
    # job rewind: sequence shrinks, the automaton must be rebuilt from the truncation
    for cut in (90, 40, 5):
        got = tuple(sam.accept_tensor(_t(seq[:cut])))
        assert got == ref_span(seq[:cut]), f"cut={cut}"
        assert sam.length() == cut


def test_repeating_pattern_finds_long_match():
    seq = list(range(20)) * 4
    sam = ext.BC_SAM()
    beg, end = sam.accept_tensor(_t(seq))
    assert end - beg == 60 and (beg, end) == ref_span(seq)


def test_accept_single_token():
    sam = ext.BC_SAM()
    seq = [4, 5, 6, 4, 5, 6, 4]
    spans = [tuple(sam.accept(t)) for t in seq]
    assert spans[-1] == ref_span(seq)
    assert sam.length() == len(seq)


def test_1d_and_2d_accepted():
    seq = [1, 2, 1, 2, 1]
    a = ext.BC_SAM(); b = ext.BC_SAM()
    assert tuple(a.accept_tensor(torch.tensor(seq, dtype = torch.long))) == \
           tuple(b.accept_tensor(_t(seq)))


def test_ngram_draft_reachable():
    """generator/job.py:1602 asserts the object is truthy before using it."""
    sam = ext.BC_SAM()
    assert sam


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
