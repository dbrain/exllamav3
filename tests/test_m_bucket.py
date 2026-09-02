"""M_BUCKET is the autotune key for the row count; see _m_bucket."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from exllamav3.modules.quant.exl3_triton import _m_bucket


@pytest.mark.parametrize("m,want", [
    (0, 1), (1, 1),                     # GEMV path stays its own key
    (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16),
    (1000, 1024), (1024, 1024), (1025, 2048),
])
def test_bucket_values(m, want):
    assert _m_bucket(m) == want


def test_bucket_is_monotonic_and_never_shrinks_m():
    prev = 0
    for m in range(1, 4097):
        b = _m_bucket(m)
        assert b >= m, "bucket must not under-report the row count"
        assert b >= prev, "bucket must be monotonic in M"
        prev = b


def test_bucket_collapses_the_moe_expert_spread():
    """The case this exists for: per-expert token counts across a routed batch.

    Previously each distinct M was its own autotune entry."""
    per_expert_rows = list(range(1, 65))          # 64 experts, 1..64 tokens each
    assert len(set(per_expert_rows)) == 64
    assert len(set(_m_bucket(m) for m in per_expert_rows)) == 7
