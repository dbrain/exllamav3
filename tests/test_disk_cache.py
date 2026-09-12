import os
import threading
import time
import pytest
import torch
from collections import namedtuple
from types import SimpleNamespace

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.generator.disk_cache import DiskCheckpointTier
from exllamav3.generator.pagetable import PageTable, Sequence
from exllamav3.cache.recurrent import RecurrentCache

TOKENS = PAGE_SIZE
SLOT = 4096
IDENT_A = {"model": "a"}
IDENT_B = {"model": "b"}


@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


class FakeCacheLayer:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence"])


def page(tag, page_index, prev = None):
    return Page(
        tag.to_bytes(16, "big"),
        prev,
        page_index,
        torch.full((1, TOKENS), tag + 1000, dtype = torch.long),
    )


def phash(tag):
    return tag.to_bytes(16, "big")


def build(root, slots = 8, ident = None, width = 64, dtype = torch.float16, num_pages = 8, disk_size = 1 << 30):
    tensor = torch.zeros((num_pages, width), dtype = dtype, device = "cuda")
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])}, model = None)
    tier = CPUPageCache(
        [cache_obj],
        slots * SLOT,
        disk_dir = str(root),
        disk_size = disk_size,
        disk_identity = ident if ident is not None else IDENT_A,
    )
    return tier, tensor


def fill(tensor, n = None):
    n = n if n is not None else tensor.shape[0]
    ref = torch.randn((n, tensor.shape[1]), dtype = torch.float32).to(tensor.dtype)
    tensor[:n].copy_(ref)
    return ref


# ---- restore on startup --------------------------------------------------------------------------------------

def test_a_page_survives_a_restart_byte_for_byte(tmp_path):
    tier, tensor = build(tmp_path)
    ref = fill(tensor, 4)
    pages = [page(i, i, phash(i - 1) if i else None) for i in range(4)]
    for p in pages:
        tier.store(p, serial = p.page_index)
    tier.drain_disk()
    tier.close()

    tier2, tensor2 = build(tmp_path)
    tensor2.fill_(-1.0)
    for p in pages:
        assert p.phash in tier2, "a page written before the restart was not found on disk"
        entry = tier2.fetch(p.phash, p.page_index, serial = 100)
        assert entry["prev_hash"] == p.prev_hash
        assert torch.equal(entry["tokens"], p.sequence)
    torch.cuda.synchronize()

    # The restored cache contents must be bit-identical to what was evicted, not merely close
    assert torch.equal(tensor2[:4], ref.to(tensor2.dtype).cuda())
    assert tier2.metrics["disk_restores"] == 4
    tier2.close()


def test_the_index_is_available_without_touching_payloads(tmp_path):
    tier, tensor = build(tmp_path)
    fill(tensor, 3)
    for i in range(3):
        tier.store(page(i, i, phash(i - 1) if i else None), serial = i)
    tier.drain_disk()
    tier.close()

    tier2, _ = build(tmp_path)
    assert tier2.prev_hash_of(phash(2)) == (True, phash(1))
    assert tier2.prev_hash_of(phash(0)) == (True, None)
    assert tier2.prev_hash_of(phash(9)) == (False, None)
    assert tier2.metrics["disk_restores"] == 0, "a metadata lookup must not page in a payload"
    tier2.close()


# ---- restore mid-session ------------------------------------------------------------------------------------

def test_a_page_evicted_from_the_host_tier_is_restored_from_disk(tmp_path):
    tier, tensor = build(tmp_path, slots = 2)
    ref = fill(tensor, 6)
    for i in range(6):
        tier.store(page(i, i, phash(i - 1) if i else None), serial = i)
    tier.drain_disk()

    assert len(tier.entries) == 2, "test needs the host tier to have spilled"
    spilled = [i for i in range(6) if phash(i) not in tier.entries]
    assert spilled
    i = spilled[0]
    assert phash(i) in tier, "a page spilled out of host RAM must still be reachable on disk"

    tensor.fill_(-1.0)
    tier.fetch(phash(i), i, serial = 100)
    torch.cuda.synchronize()
    assert torch.equal(tensor[i], ref[i].to(tensor.dtype).cuda())
    assert tier.metrics["disk_restores"] == 1
    tier.close()


def test_a_host_tier_hit_does_not_read_the_disk(tmp_path):
    tier, tensor = build(tmp_path, slots = 8)
    fill(tensor, 2)
    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()
    tier.fetch(phash(0), 1, serial = 1)
    assert tier.metrics["restores"] == 1
    assert tier.metrics["disk_restores"] == 0
    tier.close()


# ---- refused restore across builds --------------------------------------------------------------------------

def test_a_mismatched_model_identity_refuses_to_restore(tmp_path):
    tier, tensor = build(tmp_path, ident = IDENT_A)
    fill(tensor, 2)
    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()
    tier.close()

    tier2, _ = build(tmp_path, ident = IDENT_B)
    assert phash(0) not in tier2
    tier2.close()
    assert phash(0) in build(tmp_path, ident = IDENT_A)[0]


@pytest.mark.parametrize("change", [{"width": 128}, {"dtype": torch.bfloat16}, {"num_pages": 16}])
def test_a_changed_cache_geometry_refuses_to_restore(tmp_path, change):
    # num_pages is the one geometry change that does NOT alter the byte layout of a page, so it must stay
    # compatible; width and dtype do, so they must not
    tier, tensor = build(tmp_path)
    fill(tensor, 2)
    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()
    tier.close()

    tier2, _ = build(tmp_path, **change)
    compatible = "num_pages" in change
    assert (phash(0) in tier2) is compatible
    tier2.close()


# ---- corrupt page rejected ----------------------------------------------------------------------------------

def test_a_corrupt_page_is_a_miss_not_wrong_bytes(tmp_path):
    tier, tensor = build(tmp_path)
    fill(tensor, 2)
    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()
    path = tier.disk.store.path_for(phash(0))
    tier.close()

    raw = bytearray(open(path, "rb").read())
    raw[-9] ^= 0xFF
    open(path, "wb").write(bytes(raw))

    tier2, tensor2 = build(tmp_path)
    tensor2.fill_(-1.0)
    assert phash(0) in tier2, "the index is built from headers, so the damage is only visible on read"
    assert tier2.fetch(phash(0), 0, serial = 1) is None, "a corrupt page must not be served"
    torch.cuda.synchronize()
    assert torch.equal(tensor2[0], torch.full_like(tensor2[0], -1.0)), "corrupt bytes reached the cache"
    assert phash(0) not in tier2
    assert tier2.disk.store.metrics["rejected"] == 1
    tier2.close()


# ---- the write/slot-recycle race ---------------------------------------------------------------------------

def test_a_slot_with_a_disk_write_in_flight_is_not_recycled(tmp_path):
    # The writer reads the page straight out of its pinned slot, so handing that slot to another page before
    # the write lands would splice two pages into one record
    tier, tensor = build(tmp_path, slots = 4)
    fill(tensor, 4)
    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()

    gate = threading.Event()
    tier.disk.writer.q.put(lambda: gate.wait(30))
    for i in (1, 2, 3):
        tier.store(page(i, i), serial = i)
    assert len(tier.pending_disk) == 3, "test needs writes actually in flight"
    settled = tier.entries[phash(0)]["slot"]
    assert settled not in tier.pending_disk

    assert tier._evict_one(None) == settled, "a slot with a disk write in flight was recycled"

    # Now every remaining slot has a write in flight, so the policy has to wait for the writer
    threading.Timer(0.4, gate.set).start()
    t0 = time.perf_counter()
    slot = tier._evict_one(None)
    waited = time.perf_counter() - t0
    assert slot is not None
    assert waited >= 0.3, f"handed out a slot with a write in flight after only {waited:.3f}s"
    assert not tier.pending_disk
    tier.drain_disk()
    tier.close()


# ---- recurrent checkpoints ---------------------------------------------------------------------------------

def mk_stash(seed, position = 512):
    # The three container shapes real layer states stash: a tuple (GDN, SWA, PLE), a bare tensor (ShortConv)
    # and a list (DSA), plus an empty tensor, which SWA produces at position 0
    g = torch.Generator().manual_seed(seed)
    return {
        "position": position,
        "checkpoint_size": 4096,
        (0, 0): (torch.randn((2, 3, 4), generator = g), torch.randn((5, 6), generator = g)),
        (7, 1): torch.randn((1, 8), generator = g, dtype = torch.float32).to(torch.bfloat16),
        (9, 0): [torch.randn((4,), generator = g), torch.empty((0, 3))],
    }


def ckpt_tier(root, ident = None, size = 1 << 30, geometry = None):
    return DiskCheckpointTier(str(root), ident if ident is not None else IDENT_A, geometry, size)


def assert_stash_equal(a, b):
    assert a.keys() == b.keys()
    assert a["position"] == b["position"] and a["checkpoint_size"] == b["checkpoint_size"]
    for k in a:
        if not isinstance(k, tuple):
            continue
        assert type(a[k]) is type(b[k]), f"container of {k} changed: {type(a[k])} -> {type(b[k])}"
        xs = a[k] if isinstance(a[k], (list, tuple)) else (a[k],)
        ys = b[k] if isinstance(b[k], (list, tuple)) else (b[k],)
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert torch.equal(x, y), f"checkpoint tensor {k} did not round-trip bit-exactly"


def test_a_checkpoint_survives_a_restart(tmp_path):
    d = ckpt_tier(tmp_path)
    st = mk_stash(1)
    d.put(phash(9), st)
    d.drain()
    d.close()

    d2 = ckpt_tier(tmp_path)
    assert phash(9) in d2
    assert_stash_equal(st, d2.get(phash(9)))
    d2.close()


def test_a_checkpoint_from_a_mismatched_build_is_refused(tmp_path):
    d = ckpt_tier(tmp_path, IDENT_A)
    d.put(phash(9), mk_stash(1))
    d.drain()
    d.close()

    d2 = ckpt_tier(tmp_path, IDENT_B)
    assert phash(9) not in d2
    assert d2.get(phash(9)) is None
    d2.close()


def test_a_corrupt_checkpoint_is_a_miss(tmp_path):
    d = ckpt_tier(tmp_path)
    d.put(phash(9), mk_stash(1))
    d.drain()
    path = d.store.path_for(phash(9))
    d.close()
    raw = bytearray(open(path, "rb").read())
    raw[-5] ^= 0xFF
    open(path, "wb").write(bytes(raw))

    d2 = ckpt_tier(tmp_path)
    assert d2.get(phash(9)) is None
    assert phash(9) not in d2
    d2.close()


def test_a_tp_checkpoint_is_never_written(tmp_path):
    # A tensor-parallel stash is a handle into worker processes, with no bytes in this process to persist
    d = ckpt_tier(tmp_path)
    d.put(phash(9), {"position": 0, "checkpoint_size": 1, "tp_handle": 3})
    d.drain()
    assert phash(9) not in d
    assert d.metrics["unserializable"] == 1
    d.close()


# ---- RecurrentCache integration ----------------------------------------------------------------------------

class FakeState:
    def __init__(self, stashed):
        self.stashed = stashed

    def stash(self):
        return self.stashed


def rc(root, ident = None, max_size = 1 << 30):
    model = SimpleNamespace(loaded_tp = False)
    return RecurrentCache(model, max_size, disk = ckpt_tier(root, ident))


def test_the_recurrent_cache_falls_back_to_disk(tmp_path):
    c = rc(tmp_path)
    st = mk_stash(2)
    c.put(phash(5), FakeState(st))
    c.disk.drain()
    c.clear()
    assert phash(5) not in c, "test needs the RAM tier emptied"

    assert c.has(phash(5))
    got = c.get_stashed(phash(5))
    assert_stash_equal(st, got)
    assert phash(5) in c, "a disk hit must be promoted into RAM"
    c.disk.close()


def test_has_spans_both_tiers(tmp_path):
    c = rc(tmp_path)
    c.put(phash(5), FakeState(mk_stash(2)))
    c.disk.drain()
    assert c.has(phash(5))
    assert not c.has(phash(6))
    c.clear()
    assert c.has(phash(5))
    c.disk.close()


def test_the_recurrent_cache_works_without_a_disk_tier(tmp_path):
    model = SimpleNamespace(loaded_tp = False)
    c = RecurrentCache(model, 1 << 30)
    c.put(phash(5), FakeState(mk_stash(2)))
    assert c.has(phash(5))
    assert not c.has(phash(6))
    assert c.get_stashed(phash(6)) is None


# ---- no stranding ------------------------------------------------------------------------------------------

def test_a_disk_only_chain_keeps_a_checkpoint_anchored(tmp_path):
    tier, tensor = build(tmp_path, slots = 2, num_pages = 4)
    fill(tensor, 4)
    for i in range(4):
        tier.store(page(i, i % 4, phash(i - 1) if i else None), serial = i)
    tier.drain_disk()

    pt = PageTable(SimpleNamespace(recurrent_cache = None), SimpleNamespace(max_num_tokens = PAGE_SIZE * 4))
    pt.cpu_tier = tier
    tier.attach(pt)

    assert len(tier.entries) == 2, "test needs most of the chain to live only on disk"
    assert pt.is_resumable(phash(3)), "a checkpoint anchored on a disk-resident chain was reported stranded"
    assert not pt.is_resumable(phash(99))
    tier.close()


# ---- the real allocation path ---------------------------------------------------------------------------------

def real_chain(n_pages, tail = 5, seed = 7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (1, n_pages * PAGE_SIZE + tail), generator = g, dtype = torch.long)
    seq = Sequence(ids, ids)
    seq.prepare(False, 8)
    assert len(seq.page_hashes) == n_pages
    return ids, seq


def fake_pagetable(tier, num_pages, recurrent_cache = None):
    pt = PageTable(
        SimpleNamespace(recurrent_cache = recurrent_cache),
        SimpleNamespace(max_num_tokens = PAGE_SIZE * num_pages),
    )
    pt.cpu_tier = tier
    tier.attach(pt)
    if recurrent_cache is not None:
        recurrent_cache.pagetable = pt
    return pt


def seed_store(root, n_pages, num_pages = 16):
    ids, seq = real_chain(n_pages)
    tier, tensor = build(root, slots = n_pages + 4, num_pages = num_pages)
    ref = fill(tensor, n_pages)
    for i, h in enumerate(seq.page_hashes):
        tier.store(
            Page(h, seq.page_hashes[i - 1] if i else None, i, ids[:, i * PAGE_SIZE : (i + 1) * PAGE_SIZE]),
            serial = i,
        )
    tier.drain_disk()
    tier.close()
    return ids, seq, ref


def test_allocation_restores_a_whole_chain_from_disk_after_a_restart(tmp_path):
    ids, _, ref = seed_store(tmp_path, 4)

    tier, tensor = build(tmp_path, slots = 16, num_pages = 16)
    tensor.fill_(-1.0)
    pt = fake_pagetable(tier, 16)
    _, seq = real_chain(4)
    allocated, cached, _, stashed = seq.allocate_pages(pt, None)

    assert cached == 4, "a chain persisted by an earlier process was not reused"
    assert seq.kv_position == 4 * PAGE_SIZE
    assert pt.metrics["alloc_tier_pages"] == 4
    assert stashed is None
    torch.cuda.synchronize()
    for i, pg in enumerate(seq.allocated_pages[:4]):
        assert pg.kv_position == PAGE_SIZE
        assert torch.equal(tensor[pg.page_index], ref[i].to(tensor.dtype).cuda()), \
            f"restored page {i} is not byte-identical to the page that was evicted"
        assert torch.equal(pg.sequence, ids[:, i * PAGE_SIZE : (i + 1) * PAGE_SIZE])
    tier.close()


def test_allocation_caps_the_prefix_at_the_deepest_persisted_checkpoint(tmp_path):
    # The whole point of persisting checkpoints: K/V pages alone cap the resumable prefix at zero on a hybrid
    # model, and the cap is the intersection of valid pages and available checkpoints
    ids, seq0, ref = seed_store(tmp_path, 4)

    c = rc(tmp_path)
    c.put(seq0.page_hashes[1], FakeState(mk_stash(3, position = 2 * PAGE_SIZE)))
    c.disk.drain()
    c.clear()

    tier, tensor = build(tmp_path, slots = 16, num_pages = 16)
    tensor.fill_(-1.0)
    pt = fake_pagetable(tier, 16, c)
    _, seq = real_chain(4)
    allocated, cached, _, stashed = seq.allocate_pages(pt, c)

    assert cached == 2, "the prefix was not capped at the deepest checkpoint that survived the restart"
    assert seq.kv_position == 2 * PAGE_SIZE
    assert stashed is not None, "the checkpoint anchoring the prefix was not restored"
    assert stashed["position"] == 2 * PAGE_SIZE
    assert pt.metrics["alloc_tier_pages"] == 2, "pages past the last checkpoint must not be paged in"
    assert pt.metrics["alloc_kv_only_pages"] == 0
    torch.cuda.synchronize()
    for i in range(2):
        assert torch.equal(tensor[seq.allocated_pages[i].page_index], ref[i].to(tensor.dtype).cuda())
    c.disk.close()
    tier.close()


def test_allocation_with_no_persisted_checkpoint_falls_back_to_a_full_prefill(tmp_path):
    ids, seq0, ref = seed_store(tmp_path, 4)
    c = rc(tmp_path)

    tier, tensor = build(tmp_path, slots = 16, num_pages = 16)
    pt = fake_pagetable(tier, 16, c)
    _, seq = real_chain(4)
    _, cached, _, stashed = seq.allocate_pages(pt, c)

    assert cached == 0 and stashed is None
    assert pt.metrics["alloc_tier_pages"] == 0, "restoring pages past the last checkpoint buys nothing"
    c.disk.close()
    tier.close()


def test_allocation_skips_a_corrupt_page_and_prefills_instead(tmp_path):
    ids, seq0, ref = seed_store(tmp_path, 4)
    probe = build(tmp_path, slots = 4, num_pages = 8)[0]
    path = probe.disk.store.path_for(seq0.page_hashes[2])
    probe.close()
    raw = bytearray(open(path, "rb").read())
    raw[-200] ^= 0xFF
    open(path, "wb").write(bytes(raw))

    tier, tensor = build(tmp_path, slots = 16, num_pages = 16)
    tensor.fill_(-1.0)
    pt = fake_pagetable(tier, 16)
    _, seq = real_chain(4)
    _, cached, _, _ = seq.allocate_pages(pt, None)

    assert cached == 2, "the prefix must stop at the damaged page, not skip over it"
    torch.cuda.synchronize()
    for i in range(2):
        assert torch.equal(tensor[seq.allocated_pages[i].page_index], ref[i].to(tensor.dtype).cuda())
    damaged = seq.allocated_pages[2]
    assert damaged.kv_position == 0, "a page that failed verification must be left for prefill"
    assert torch.equal(tensor[damaged.page_index], torch.full_like(tensor[damaged.page_index], -1.0)), \
        "bytes from a damaged record reached the cache"
    tier.close()


def test_a_changed_recurrent_geometry_refuses_to_restore(tmp_path):
    d = ckpt_tier(tmp_path, geometry = [["(0, 0)", 64, [[[2, 3], "torch.float32"]]]])
    d.put(phash(9), mk_stash(1))
    d.drain()
    d.close()

    d2 = ckpt_tier(tmp_path, geometry = [["(0, 0)", 64, [[[2, 4], "torch.float32"]]]])
    assert phash(9) not in d2
    d2.close()


def test_the_kv_and_checkpoint_stores_share_one_generation(tmp_path):
    # They stamp the same directory, so neither can be pruned as a stale generation while the other is in use
    tier, _ = build(tmp_path)
    d = ckpt_tier(tmp_path)
    assert tier.disk.store.generation == d.store.generation
    assert tier.disk.store.root != d.store.root
    d.close()
    tier.close()


# ---- pages that are never evicted ----------------------------------------------------------------------------

def claim_and_fill(tmp_path, n_pages, **kw):
    ids, seq = real_chain(n_pages)
    tier, tensor = build(tmp_path, slots = 16, num_pages = 16, **kw)
    pt = fake_pagetable(tier, 16)
    seq.allocate_pages(pt, None)
    ref = {}
    for i, pg in enumerate(seq.allocated_pages[:n_pages]):
        assert pg.phash == seq.page_hashes[i]
        row = torch.randn((tensor.shape[1],), dtype = torch.float32).to(tensor.dtype)
        tensor[pg.page_index].copy_(row)
        pg.sequence.copy_(ids[:, i * PAGE_SIZE : (i + 1) * PAGE_SIZE])
        pg.prev_hash = seq.page_hashes[i - 1] if i else None
        pg.kv_position = PAGE_SIZE
        ref[pg.phash] = row
    return tier, tensor, pt, seq, ref


def test_a_page_that_is_never_evicted_is_still_persisted(tmp_path):
    # Without this the durable tier only ever sees pages the GPU cache was forced to give up, i.e. never the
    # case it exists for: one conversation that comfortably fits and is then lost to a restart
    tier, tensor, pt, seq, ref = claim_and_fill(tmp_path, 3)
    assert tier.metrics["pushes"] == 0, "test needs a page table under no eviction pressure"

    assert pt.persist_complete_pages() == 3
    tier.drain_disk()
    assert all(h in tier.disk for h in seq.page_hashes)
    assert not tier.entries, "write-through must not spend host tier slots on VRAM-resident pages"

    assert pt.persist_complete_pages() == 0
    assert not pt._persist_dirty, "a sweep that finds nothing must latch off"
    tier.close()

    tier2, tensor2 = build(tmp_path, slots = 16, num_pages = 16)
    tensor2.fill_(-1.0)
    pt2 = fake_pagetable(tier2, 16)
    _, seq2 = real_chain(3)
    _, cached, _, _ = seq2.allocate_pages(pt2, None)
    assert cached == 3
    torch.cuda.synchronize()
    for i, pg in enumerate(seq2.allocated_pages[:3]):
        assert torch.equal(tensor2[pg.page_index], ref[seq.page_hashes[i]].cuda()), \
            "a page persisted while resident did not come back byte-identical"
    tier2.close()


def test_an_incomplete_or_unhashed_page_is_never_persisted(tmp_path):
    tier, tensor, pt, seq, _ = claim_and_fill(tmp_path, 3)
    seq.allocated_pages[1].kv_position = PAGE_SIZE - 1
    seq.allocated_pages[2].make_unique()

    assert pt.persist_complete_pages() == 1
    tier.drain_disk()
    assert seq.page_hashes[0] in tier.disk
    assert seq.page_hashes[1] not in tier.disk, "a partially filled page was persisted"
    assert seq.page_hashes[2] not in tier.disk, "a page without a content hash was persisted"
    tier.close()


def test_the_sweep_is_inert_without_a_durable_tier(tmp_path):
    tier, tensor, pt, seq, _ = claim_and_fill(tmp_path, 3, disk_size = 0)
    assert tier.disk is None
    assert pt.persist_complete_pages() == 0
    tier.close()


# ---- the record covers every plane the cache layer exposes ----------------------------------------------------

# Flash-Next geometry: 12 full-attention layers (the other 36 are GatedDeltaNet and hold no per-token K/V),
# 2 kv_heads x 256 head_dim of fp16 K and V, plus the QSA indexer's side planes, which stay fp16 whatever the
# K/V storage is: raw_k is 128 values per token, pooled is 128 values per 4-token block.
FN_LAYERS = 12
FN_KV_HEADS = 2
FN_HEAD_DIM = 256
FN_INDEX_DIM = 128
FN_COMPRESS = 4
FN_KIB_PER_TOKEN = 27.75


def flashnext_shaped_cache(num_pages = 4):
    layers = {}
    for i in range(FN_LAYERS):
        kv = lambda: torch.zeros(
            (num_pages, PAGE_SIZE, FN_KV_HEADS, FN_HEAD_DIM), dtype = torch.half, device = "cuda"
        )
        raw_k = torch.zeros((num_pages, PAGE_SIZE, FN_INDEX_DIM), dtype = torch.half, device = "cuda")
        pooled = torch.zeros(
            (num_pages, PAGE_SIZE // FN_COMPRESS, FN_INDEX_DIM), dtype = torch.half, device = "cuda"
        )
        layers[i] = FakeCacheLayer([kv(), kv(), raw_k, pooled])
    return SimpleNamespace(layers = layers, model = None)


def test_a_record_covers_every_plane_the_cache_layer_exposes(tmp_path):
    # A page record is built by enumerating the tensors a cache layer actually exposes, never by counting
    # layers. A layer that adds a plane and forgets to extend get_tensors() would leave that plane out of the
    # record, and a restore would then produce correct K/V beside stale indexer planes - plausible-but-wrong
    # block scores rather than an honest cache miss. This pins the measured size against the known geometry.
    tier = CPUPageCache(
        [flashnext_shaped_cache()], 64 * 1024**2,
        disk_dir = str(tmp_path), disk_size = 1 << 30, disk_identity = IDENT_A,
    )
    per_token = tier.slot_size / PAGE_SIZE / 1024
    assert per_token == FN_KIB_PER_TOKEN, \
        f"record holds {per_token} KiB/token, expected {FN_KIB_PER_TOKEN} (K/V 24.00 + indexer planes 3.75)"
    assert len(tier.segments) == FN_LAYERS * 4, "a plane was dropped from the segment table"
    tier.close()


def test_every_plane_round_trips_byte_identically(tmp_path):
    cache_obj = flashnext_shaped_cache()
    tier = CPUPageCache(
        [cache_obj], 64 * 1024**2,
        disk_dir = str(tmp_path), disk_size = 1 << 30, disk_identity = IDENT_A,
    )
    tensors = [t for layer in cache_obj.layers.values() for t in layer.get_tensors()]
    ref = []
    for t in tensors:
        r = torch.randn(t[0].shape, dtype = torch.float32).to(t.dtype)
        t[0].copy_(r)
        ref.append(r)

    tier.store(page(0, 0), serial = 0)
    tier.drain_disk()
    tier.close()

    cache2 = flashnext_shaped_cache()
    tier2 = CPUPageCache(
        [cache2], 64 * 1024**2,
        disk_dir = str(tmp_path), disk_size = 1 << 30, disk_identity = IDENT_A,
    )
    tensors2 = [t for layer in cache2.layers.values() for t in layer.get_tensors()]
    for t in tensors2:
        t.fill_(-1.0)
    assert phash(0) in tier2
    assert tier2.fetch(phash(0), 0, serial = 1) is not None
    torch.cuda.synchronize()
    for i, (t, r) in enumerate(zip(tensors2, ref)):
        assert torch.equal(t[0], r.cuda()), f"plane {i} did not round-trip byte-identically"
    tier2.close()
