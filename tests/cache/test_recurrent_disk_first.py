"""
Disk-first recurrent checkpoints: with a disk tier attached, a RecurrentCache budget below one checkpoint keeps
exactly the entry being stored, and a checkpoint that leaves RAM at the end of its chain (neither a checkpoint
still in RAM nor the incoming one descends from it) is written to disk regardless of spacing, its anchor page
queued first through DiskPageCache.queue_chain when it is not on disk yet. Checkpoints inside a chain keep the
spacing rule. Pages are stored straight into the CPU tier, as in test_disk_cache.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache.recurrent import RecurrentCache
from testlib.cache import NUM_PAGES, build_disk_tiers, page, phash


@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


@pytest.fixture
def tiers(device, tmp_path):
    tier, disk, _ = build_disk_tiers(16, device, tmp_path)
    disk.min_state_distance = 1 << 20
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 0)
    rc.disk_tier = disk
    yield tier, disk, rc
    disk.close()
    tier.close()


def state(pos):
    return {"position": pos, "checkpoint_size": 64, "x": torch.full((16,), float(pos))}


def chain(tier, tags, prev = None):
    for tag in tags:
        tier.store(page(tag, tag % NUM_PAGES, prev), serial = tag)
        prev = phash(tag)


def settle(disk):
    disk.flush()
    disk.drain_ready_states()
    disk.flush()


def test_budget_below_one_entry_keeps_the_entry_being_stored(tiers):
    tier, disk, rc = tiers
    chain(tier, [1, 2])
    rc.put(phash(1), None, stashed = state(100))
    assert phash(1) in rc
    rc.put(phash(2), None, stashed = state(200))
    assert list(rc.keys()) == [phash(2)]


def test_evicted_chain_end_is_written_regardless_of_spacing(tiers):
    tier, disk, rc = tiers
    chain(tier, [1, 2, 3])
    chain(tier, [10, 11])
    rc.put(phash(3), None, stashed = state(300))
    rc.put(phash(11), None, stashed = state(200))
    settle(disk)
    assert disk.has_state(phash(3)), "the end of another conversation's chain was lost"
    assert rc.get_stashed(phash(3))["position"] == 300


def test_evicted_ancestor_keeps_the_spacing_rule(tiers):
    tier, disk, rc = tiers
    chain(tier, [1, 2, 3])
    rc.put(phash(1), None, stashed = state(100))
    rc.put(phash(3), None, stashed = state(300))
    settle(disk)
    assert not disk.has_state(phash(1))


def test_chain_end_whose_page_is_not_on_disk_is_queued_first(tiers):
    tier, disk, rc = tiers
    queued = []

    def queue_chain(h):
        queued.append(h)
        chain(tier, [5, 6])

    disk.queue_chain = queue_chain
    rc.put(phash(6), None, stashed = state(600))
    chain(tier, [20])
    rc.put(phash(20), None, stashed = state(100))
    settle(disk)
    assert queued == [phash(6)]
    assert disk.has_state(phash(6))
