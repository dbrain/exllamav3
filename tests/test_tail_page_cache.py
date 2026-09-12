import os
import pytest
import torch
from types import SimpleNamespace

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator import pagetable as pt_mod
from exllamav3.generator.pagetable import PageTable, Sequence, tail_hash_checksum


class FakeCache:
    def __init__(self, max_num_tokens):
        self.max_num_tokens = max_num_tokens
        self.copies = []

    def copy_page(self, target, from_page, to_page, num_tokens):
        assert target is self
        self.copies.append((from_page, to_page, num_tokens))


class FakeRecurrentCache(dict):
    def has(self, key):
        return key in self

    def get_stashed(self, key, default = None):
        return self.get(key, default)

    def put(self, key, state):
        self[key] = state


def build(tokens = PAGE_SIZE * 64, draft = False):
    cache = FakeCache(tokens)
    draft_cache = FakeCache(tokens) if draft else None
    gen = SimpleNamespace(tokenizer = None, cache = cache, draft_cache = draft_cache,
                          mtp_draft = draft, draft_model = None)
    pt = PageTable(gen, cache)
    pt.generator = gen
    if draft:
        return pt, cache, draft_cache
    return pt, cache


def ids_for(length, salt = 0):
    return (torch.arange(length, dtype = torch.long) + salt * 1000000).unsqueeze(0)


def seq_for(length, salt = 0, max_new_tokens = 64):
    ids = ids_for(length, salt)
    s = Sequence(ids, ids)
    s.prepare(False, max_new_tokens)
    return s


def tail_of(s):
    n = len(s.page_hashes) * PAGE_SIZE
    return s.sequence_ids.torch_slice(n, len(s.sequence_ids) - 1)


def run_prefill(pt, rc, s, position_state = "S"):
    """Emulate what Job.prefill does at the end of prompt prefill for a recurrent model: fill every
    allocated page, content-hash the complete ones, then register the partial tail."""
    s.allocate_pages(pt, rc)
    target = len(s.sequence_ids) - 1
    for local_idx, page in enumerate(s.allocated_pages):
        a = local_idx * PAGE_SIZE
        if a >= target:
            break
        b = min(a + PAGE_SIZE, target)
        page.prev_hash = None if local_idx == 0 else s.allocated_pages[local_idx - 1].phash
        page.sequence[:, : b - a].copy_(s.sequence_ids.torch_slice(a, b))
        page.kv_position = b - a
        if page.kv_position == PAGE_SIZE:
            page.update_hash()
            # maybe_stash_recurrent(interval = PAGE_SIZE) checkpoints every complete page of the prompt
            rc.put(page.phash, f"page@{b}")
    s.kv_position = target
    s.register_tail(pt, rc, position_state)
    return s


def forward_one(pt, s, next_token = 5):
    """Emulate one decode step: Job.get_input_ids_list(add_to_cache = True) writes the forwarded token
    into its page and advances page.kv_position, then the accept path appends the sampled token and
    commits any page that just landed on the grid."""
    ids = s.sequence_ids.torch_slice(s.kv_position, None)
    skvp, n = s.kv_position, ids.shape[-1]
    while n:
        page = s.allocated_pages[skvp // PAGE_SIZE]
        k = min(n, PAGE_SIZE - page.kv_position)
        page.sequence[:, page.kv_position : page.kv_position + k] = ids[:, :k]
        page.kv_position += k
        page.can_revert = False
        skvp += k
        ids = ids[:, k:]
        n -= k
    s.sequence_ids.append(torch.full((1, 1), next_token, dtype = torch.long))
    page_before = s.kv_position // PAGE_SIZE
    s.kv_position += 1
    if s.kv_position // PAGE_SIZE > page_before:
        s.commit_page(pt, page_before)


# --- hashing ------------------------------------------------------------------------------------

def test_tail_hash_separates_length_chain_and_content():
    a = ids_for(64)
    b = ids_for(64, salt = 1)
    h = tail_hash_checksum(a, None, 64)
    assert tail_hash_checksum(a, None, 64) == h
    assert tail_hash_checksum(a, None, 63) != h
    assert tail_hash_checksum(a, b"0" * 16, 64) != h
    assert tail_hash_checksum(b, None, 64) != h
    # A tail hash must never be mistaken for a full-page hash of the same tokens
    full = ids_for(PAGE_SIZE)
    assert tail_hash_checksum(full, None, PAGE_SIZE) != pt_mod.tensor_hash_checksum(full, None)


# --- prepare() ----------------------------------------------------------------------------------

@pytest.mark.parametrize("length, exp_pages, exp_tail", [
    (PAGE_SIZE * 14, 13, PAGE_SIZE - 1),
    (PAGE_SIZE * 14 + 1, 14, 0),
    (PAGE_SIZE * 14 + 116, 14, 115),
    (PAGE_SIZE * 157, 156, PAGE_SIZE - 1),
])
def test_prepare_emits_tail_for_the_unhashable_remainder(length, exp_pages, exp_tail):
    s = seq_for(length)
    assert len(s.page_hashes) == exp_pages
    assert s.tail_len == exp_tail
    if exp_tail:
        assert s.tail_hash is not None
    else:
        assert s.tail_hash is None


# --- exact repeat -------------------------------------------------------------------------------

def test_identical_prompt_resumes_to_the_last_token():
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state@3583")
    pt.deallocate_pages(s1.allocated_pages)

    s2 = seq_for(L)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert cached_pages == 13
    assert s2.kv_position == L - 1, "tail page not restored"
    assert stashed == "state@3583"
    assert cache.copies and cache.copies[-1][2] == PAGE_SIZE - 1


def test_extension_resumes_past_the_shared_tail():
    pt, cache = build()
    rc = FakeRecurrentCache()
    S1 = PAGE_SIZE * 14 - 100
    s1 = run_prefill(pt, rc, seq_for(S1), "state@S1")
    pt.deallocate_pages(s1.allocated_pages)

    # Turn 2: same sequence plus 40 new tokens. The shared part ends mid-page.
    ids = torch.cat([ids_for(S1)[:, : S1 - 1], torch.full((1, 41), 7, dtype = torch.long)], dim = -1)
    s2 = Sequence(ids, ids)
    s2.prepare(False, 64)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert cached_pages == 13
    assert s2.kv_position == S1 - 1
    assert stashed == "state@S1"


def test_divergent_tail_is_not_a_hit():
    # A checkpoint exists only at the registered tail length, so a tail that agrees on fewer tokens
    # than that has no resume point and must fall back to the page boundary
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state")
    pt.deallocate_pages(s1.allocated_pages)

    ids = ids_for(L)
    ids[:, PAGE_SIZE * 13 + 5] = 999999999
    s2 = Sequence(ids, ids)
    s2.prepare(False, 64)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert s2.kv_position == PAGE_SIZE * 13
    assert cached_pages == 13


def test_missing_checkpoint_is_not_a_hit():
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state")
    pt.deallocate_pages(s1.allocated_pages)
    for h in list(pt.tail_by_hash):
        del rc[h]

    s2 = seq_for(L)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert s2.kv_position == PAGE_SIZE * 13
    assert stashed == f"page@{PAGE_SIZE * 13}"


def test_overwritten_backing_page_is_not_a_hit():
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state")
    tail_page = s1.allocated_pages[13]
    pt.deallocate_pages(s1.allocated_pages)
    tail_page.clear()

    s2 = seq_for(L)
    _, _, _, stashed = s2.allocate_pages(pt, rc)
    assert s2.kv_position == PAGE_SIZE * 13, "stale tail registration was trusted"


def test_tail_checkpoint_counts_as_resumable():
    pt, cache = build()
    rc = FakeRecurrentCache()
    s1 = run_prefill(pt, rc, seq_for(PAGE_SIZE * 14), "state")
    (tail_hash,) = list(pt.tail_by_hash)
    assert pt.is_resumable(tail_hash)
    pt.deallocate_pages(s1.allocated_pages)
    s1.allocated_pages[13].clear()
    assert not pt.is_resumable(tail_hash)


def test_flag_off_leaves_the_page_aligned_prefix():
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state")
    pt.deallocate_pages(s1.allocated_pages)

    s2 = seq_for(L)
    try:
        pt_mod.TAIL_CACHE = False
        _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    finally:
        pt_mod.TAIL_CACHE = True
    assert s2.kv_position == PAGE_SIZE * 13
    assert cached_pages == 13


@pytest.fixture(autouse = True)
def tail_cache_on():
    prev = pt_mod.TAIL_CACHE
    pt_mod.TAIL_CACHE = True
    yield
    pt_mod.TAIL_CACHE = prev


# --- second entry point: registration after generation, past the prompt's hashed pages ----------

def test_generation_end_registration_serves_the_next_turn():
    """The prompt-prefill hook only ever registers a tail inside the prompt's own page range. A job
    that generated past that range registers through the same chokepoint with kv_position beyond
    len(page_hashes), and the next turn's prompt -- which contains those generated tokens -- must
    resume from it."""
    pt, cache = build()
    rc = FakeRecurrentCache()
    L1 = PAGE_SIZE * 13 + 40
    s1 = seq_for(L1, max_new_tokens = 600)
    run_prefill(pt, rc, s1)

    # Generate 300 tokens: the sequence grows, pages complete and get content hashes
    gen = torch.full((1, 300), 5, dtype = torch.long)
    s1.sequence_ids.append(gen)
    s1.input_ids.append(gen)
    S1 = L1 + 300
    for local_idx, page in enumerate(s1.allocated_pages):
        a = local_idx * PAGE_SIZE
        if a >= S1:
            break
        b = min(a + PAGE_SIZE, S1)
        page.prev_hash = None if local_idx == 0 else s1.allocated_pages[local_idx - 1].phash
        page.sequence[:, : b - a].copy_(s1.sequence_ids.torch_slice(a, b))
        page.kv_position = b - a
        page.can_revert = False
        if page.kv_position == PAGE_SIZE and not pt_mod.is_content_hash(page.phash):
            page.update_hash()
            rc.put(page.phash, f"page@{b}")
    s1.kv_position = S1
    s1.register_tail(pt, rc, "state@S1")
    pt.deallocate_pages(s1.allocated_pages)

    ids = torch.cat([s1.sequence_ids.torch(), torch.full((1, 25), 9, dtype = torch.long)], dim = -1)
    s2 = Sequence(ids, ids)
    s2.prepare(False, 64)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert cached_pages == S1 // PAGE_SIZE
    assert s2.kv_position == S1, "generation-end tail not reused"
    assert stashed == "state@S1"


# --- the two arithmetic invariants a tail resume breaks -----------------------------------------

def _stub_job(kv_position, cached_pages, prompt_len, chunk = 512,
              interval = 2048, interval_pp = 32768):
    from exllamav3.generator.job import Job
    j = Job.__new__(Job)
    ids = torch.zeros((1, prompt_len + 1), dtype = torch.long)
    s = Sequence(ids, ids)
    s.kv_position = kv_position
    j.sequences = [s]
    j.cached_pages = cached_pages
    j.generator = SimpleNamespace(max_chunk_size = chunk,
                                  recurrent_checkpoint_interval = interval,
                                  recurrent_checkpoint_interval_pp = interval_pp)
    return j


@pytest.mark.parametrize("tail", [0, 1, 155, PAGE_SIZE - 1])
def test_checkpoint_boundaries_stay_page_aligned_after_a_tail_resume(tail):
    """maybe_stash_recurrent asserts kv_position % PAGE_SIZE == 0, so a generation that resumed at
    a non-page-aligned tail position must still only hit boundaries on the page grid. cached_pages
    is what the real allocate_pages reports -- the page-aligned prefix, NOT the resume point."""
    prompt_len = PAGE_SIZE * 14 - 1
    cached_pages = 13
    resume = cached_pages * PAGE_SIZE + tail
    hits = [p for p in range(resume, resume + 8192)
            if _stub_job(p, cached_pages, prompt_len).is_checkpoint_boundary()]
    assert hits, "no checkpoint would ever be taken after a tail resume"
    for p in hits:
        assert p % PAGE_SIZE == 0, f"checkpoint at {p} is not page aligned"


def test_restored_state_position_matches_the_resumed_kv_position():
    """Cache.new_from_stashed -> GDNState.unstash asserts position == stashed["position"], so the
    job must restore at the sequence's actual resume point, not at the page-aligned prefix."""
    from exllamav3.generator.job import Job

    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state@3583")
    pt.deallocate_pages(s1.allocated_pages)

    asked = []

    class StateCache(FakeCache):
        def get_new_state(self):
            asked.append(("new", 0))
            return SimpleNamespace(position = 0, free = lambda: None)

        def new_from_stashed(self, stashed, position):
            asked.append((stashed, position))
            return SimpleNamespace(position = position, free = lambda: None)

    sc = StateCache(cache.max_num_tokens)
    pt.cache = sc
    pt.generator = SimpleNamespace(tokenizer = None, cache = sc, draft_cache = None,
                                   recurrent_cache = rc)

    j = Job.__new__(Job)
    j.sequences = [seq_for(L)]
    j.pagetable = pt
    j.generator = pt.generator
    j.all_unique_hashes = list(j.sequences[0].page_hashes)
    j.recurrent_state = None
    j.cached_pages = j.cached_tokens = j.total_pages = j.non_sequential_pages = 0
    j.allocate_pages()

    (stashed, position), = asked
    assert stashed == "state@3583"
    assert position == L - 1, f"restored at {position}, sequence resumed at {j.sequences[0].kv_position}"
    assert position == j.sequences[0].kv_position


def test_tail_hit_is_reported_in_the_jobs_cached_tokens():
    """The job's reported cached_tokens is cached_pages*PAGE_SIZE + cached_tokens (job.py), and the
    tail hit must land in the second term. Without it every harness reads a tail hit as identical to
    a miss, and a falsification criterion keyed on that number fires spuriously on the treated arm."""
    from exllamav3.generator.job import Job

    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state@3583")
    pt.deallocate_pages(s1.allocated_pages)

    class StateCache(FakeCache):
        def get_new_state(self):
            return SimpleNamespace(position = 0, free = lambda: None)

        def new_from_stashed(self, stashed, position):
            return SimpleNamespace(position = position, free = lambda: None)

    sc = StateCache(cache.max_num_tokens)
    pt.cache = sc
    pt.generator = SimpleNamespace(tokenizer = None, cache = sc, draft_cache = None,
                                   recurrent_cache = rc)

    j = Job.__new__(Job)
    j.sequences = [seq_for(L)]
    j.pagetable = pt
    j.generator = pt.generator
    j.all_unique_hashes = list(j.sequences[0].page_hashes)
    j.recurrent_state = None
    j.cached_pages = j.cached_tokens = j.total_pages = j.non_sequential_pages = 0
    j.allocate_pages()

    assert j.cached_pages == 13
    assert j.cached_tokens == PAGE_SIZE - 1
    assert j.cached_pages * PAGE_SIZE + j.cached_tokens == L - 1


def test_tail_page_reuse_under_starvation_degrades_to_a_miss():
    """When the allocation genuinely has no other page to take, the tail page IS reused. Protection
    reorders preference, it cannot conjure pages. The property that must hold is that the stale
    registration is then rejected rather than trusted: a page-aligned resume, not a wrong one."""
    pt, cache = build(tokens = PAGE_SIZE * 16)
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14
    s1 = run_prefill(pt, rc, seq_for(L), "state@3583")
    pt.deallocate_pages(s1.allocated_pages)
    tail_page = s1.allocated_pages[13]

    s2 = seq_for(L, max_new_tokens = 400)
    _, cached_pages, _, stashed = s2.allocate_pages(pt, rc)
    assert tail_page in s2.allocated_pages[13:], "starvation case not reproduced"
    assert s2.kv_position == PAGE_SIZE * 13, "stale tail registration was trusted"
    assert cached_pages == 13
    assert stashed == f"page@{PAGE_SIZE * 13}"


def test_tail_page_survives_eviction_by_root_recency_not_by_protection():
    """Why no explicit protection is wired in: reviving the prefix refreshes the chain root's access
    serial, build_eviction_order orders rooted trees by root recency and prunes leaf-first, so the
    just-revived chain's tail page is already the LAST thing evicted. An added protected_hashes entry
    would change nothing, so there is none."""
    pt, cache = build(tokens = PAGE_SIZE * 8)
    tail, other = pt.all_pages[0], pt.all_pages[1]
    tail.kv_position = other.kv_position = PAGE_SIZE - 1
    tail.access_serial, other.access_serial = 900, 100
    order = list(pt.build_eviction_order())
    assert order[-1] is tail
    assert order.index(other) < order.index(tail)


# --- page completion must not orphan a live tail registration -----------------------------------

def test_page_completion_keeps_the_previous_turns_tail_registration():
    """A turn that hits the tail runs zero prefill chunks, so register_tail_checkpoint -- reachable
    only from inside Job.prefill's loop -- never fires and the registration still points at the
    PREVIOUS turn's page. Completing the identical page then finds that page as an unreferenced
    duplicate and clears it, taking the registration with it: turn 3 re-prefills the whole tail."""
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14

    s1 = run_prefill(pt, rc, seq_for(L), "state@3583")
    turn1_tail_page = s1.allocated_pages[13]
    forward_one(pt, s1)
    assert pt_mod.is_content_hash(turn1_tail_page.phash)
    pt.deallocate_pages(s1.allocated_pages)

    s2 = seq_for(L)
    _, _, _, stashed2 = s2.allocate_pages(pt, rc)
    assert s2.kv_position == L - 1 and stashed2 == "state@3583", "turn 2 did not hit the tail"
    assert s2.allocated_pages[13] is not turn1_tail_page, "turn 2 reused the registered page: scenario not reproduced"
    forward_one(pt, s2)
    assert turn1_tail_page.kv_position == 0, "the duplicate was not cleared: scenario not reproduced"
    pt.deallocate_pages(s2.allocated_pages)

    s3 = seq_for(L)
    _, _, _, stashed3 = s3.allocate_pages(pt, rc)
    assert s3.kv_position == L - 1, "turn 3 re-prefilled the tail: the registration was orphaned"
    assert stashed3 == "state@3583"


def test_rehomed_registration_survives_a_whole_chain_of_turns():
    """Each turn hands the registration to the page it just completed, so the resume point must hold
    for an unbounded number of turns, not just the one after the fix."""
    pt, cache = build()
    rc = FakeRecurrentCache()
    L = PAGE_SIZE * 14

    s = run_prefill(pt, rc, seq_for(L), "state@3583")
    forward_one(pt, s)
    pt.deallocate_pages(s.allocated_pages)

    for turn in range(2, 8):
        s = seq_for(L)
        _, _, _, stashed = s.allocate_pages(pt, rc)
        assert s.kv_position == L - 1, f"turn {turn} re-prefilled the tail"
        assert stashed == "state@3583"
        forward_one(pt, s)
        pt.deallocate_pages(s.allocated_pages)


# --- MTP speculative decode over a chain of warm turns ------------------------------------------

class FakeMTPDraft:
    """The one thing about an MTP draft that matters here: the target forward of every step it
    drafts from has to hand back the target's post-final-norm state through params, which the
    target model is told to export by the draft's own verifier params (architecture/
    qwen3_5_mtp.py, qwen4_exp_mtp.py)."""
    draft_verifier_params = {"export_state_norm_keys": {"model.norm"}}


def mtp_reseed_step_params(pt, s, draft):
    """The params Generator.iterate_gen builds for the non-speculative step that follows a turn
    with no prefill chunk: one token, paged, with the draft's verifier params folded in."""
    params = {
        "attn_mode": "flash_attn",
        "block_table": s.block_index_tensor,
        "cache": pt.cache,
        "cache_seqlens": torch.tensor([s.kv_position], dtype = torch.int32),
        "recurrent_states": None,
        "indexed_embeddings": None,
        "positions": torch.tensor([s.kv_position], dtype = torch.int32),
        "recurrent_history": False,
        "pinned_staging": True,
    }
    params.update(draft.draft_verifier_params)
    return params


def test_mtp_warm_turns_force_an_eager_reseed_step(monkeypatch):
    """A restored tail reaches the last prompt token, so the turn runs ZERO prefill chunks and
    Job.prefill's MTP block -- the only place seq.mtp_carry_hidden is ever set -- never runs.
    MTP recovers by taking one ordinary single-token target step (generator.py,
    iterate_draftmodel_mtp_gen returns None when job.mtp_last_hidden is None) and reading the
    exported hidden state out of params afterwards. A single-token paged step is exactly what
    the decode-graph path captures, and a graph-served step cannot return an out-parameter, so
    the reseed step must be refused by it.

    Six turns, because the refusal is what is under test and the graph path only reaches capture
    after EXL3_DECODE_GRAPH_WARMUP eager steps: turns 2 and 3 fall back for free and turn 4 is
    where a missing rule first costs the generator a TypeError on None.
    """
    from exllamav3.model import graph_decode as gd

    pt, cache, draft_cache = build(draft = True)
    rc = FakeRecurrentCache()
    draft = FakeMTPDraft()
    graphs = gd.DecodeGraphs(SimpleNamespace(modules = [], fwd_modules = [], loaded_tp = False))
    monkeypatch.setattr(graphs, "eligible", lambda: True)
    captured = []
    monkeypatch.setattr(graphs, "_capture", lambda sig, ids, params: captured.append(sig))

    L = PAGE_SIZE * 14
    s = run_prefill(pt, rc, seq_for(L), "state@3583")
    forward_one(pt, s)
    pt.deallocate_pages(s.allocated_pages)

    one_token = torch.zeros((1, 1), dtype = torch.long)
    for turn in range(2, 8):
        s = seq_for(L)
        _, _, _, stashed = s.allocate_pages(pt, rc)
        assert s.kv_position == L - 1 and stashed == "state@3583", f"turn {turn} missed the tail"
        assert s.mtp_carry_hidden is None, "no prefill chunk ran, so nothing could have set it"
        assert draft_cache.copies[-1] == cache.copies[-1], \
            "the draft model's KV did not follow the target's over the restored tail"

        params = mtp_reseed_step_params(pt, s, draft)
        assert graphs.forward(one_token, params) is None, \
            f"turn {turn}: the MTP reseed step was served from a graph"
        assert not captured, f"turn {turn}: a state-exporting step was sent to capture"

        forward_one(pt, s)
        pt.deallocate_pages(s.allocated_pages)
