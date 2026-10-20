import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from ..constants import PAGE_SIZE
import torch
from ..util.memory import malloc_trim, set_host_mmap_threshold


def is_content_hash(h: bytes) -> bool:
    # Mirrors generator.pagetable.is_content_hash (not imported: that module imports this one)
    return h[:8] != bytes(8)

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0


class HostPool:
    """
    Reusable host buffers for recurrent checkpoints. A checkpoint is a few dozen multi-MiB tensors with
    lifetimes interleaved with everything else the process allocates; allocating them fresh per stash
    and freeing them on eviction is exactly the pattern that strands memory in glibc's arenas (the
    dynamic mmap threshold moves them into the heap after the first free, and what malloc_trim can
    then hand back depends on the allocator and the platform). Buffers are handed out by exact shape
    and dtype, returned on eviction, and never freed, so the pool's footprint is bounded by the peak
    checkpoint occupancy and the allocator sees no churn at all.
    """

    def __init__(self):
        self.free = {}
        self.allocated = 0
        self.reused = 0

    def take(self, shape, dtype):
        key = (tuple(shape), dtype)
        lst = self.free.get(key)
        if lst:
            self.reused += 1
            return lst.pop()
        self.allocated += 1
        return torch.empty(shape, dtype = dtype)

    def give(self, obj):
        """Return every tensor inside a stashed structure (dict / list / tuple of tensors) to the pool"""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cpu":
                self.free.setdefault((tuple(obj.shape), obj.dtype), []).append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self.give(o)
        elif isinstance(obj, dict):
            for k, o in obj.items():
                if k not in ("position", "checkpoint_size", "tp_handle"):
                    self.give(o)

    def release(self):
        """Drop the idle buffers (idle-transition housekeeping: the ones pruning stranded checkpoints
        just returned would otherwise hold their RAM for the whole idle period)"""
        self.free.clear()


host_pool = HostPool()


def mp_host_pool_release(local_context: dict):
    host_pool.release()


def host_copy(src: torch.Tensor) -> torch.Tensor:
    """Copy a device tensor (any strides) into a pooled host buffer; the stash-side replacement for .cpu()"""
    dst = host_pool.take(src.shape, src.dtype)
    dst.copy_(src)
    return dst

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class PairedState:
    """
    One sequence's recurrent state across the target model and the draft model, as the generator and
    jobs see it: each side that keeps recurrent state (GDN / Mamba2 pools, SWA rings, DSA pools) has its
    own state object from its own cache, and the pair moves together. Checkpoints stash both sides
    under one key at the target's page boundary, so a prompt-cache resume restores both, and one LRU
    eviction drops both. Either side may be absent (a dense target with a recurrent draft, or the
    common case today, a recurrent target and no recurrent draft).

    The verification pass advances the target side only; the draft side is advanced by the draft's own
    forwards, or by the projected rows a DFlash-family drafter writes, so rewind() after verification
    and after a prefill overshoot rewinds the target side alone. A checkpoint rewind (banned strings)
    brings every side to the same position (rewind_to), in place where each side's rollback capacity
    allows, otherwise the job restores the pair from a stash.
    """

    def __init__(self, target = None, draft = None):
        assert target is not None or draft is not None, "PairedState needs at least one side"
        self.target = target
        self.draft = draft

    @property
    def components(self):
        return [s for s in (self.target, self.draft) if s is not None]

    @property
    def primary(self):
        return self.target if self.target is not None else self.draft

    @property
    def position(self) -> int:
        return self.primary.position

    @property
    def last_history(self) -> int:
        return self.primary.last_history

    @property
    def checkpoint_size(self) -> int:
        return sum(s.checkpoint_size for s in self.components)

    def rewind(self, num_tokens: int):
        """Settle the target side after a speculative pass, or correct its position after a prefill
        overshoot (see the class note)"""
        if self.target is not None:
            self.target.rewind(num_tokens)

    def can_rewind_to(self, position: int) -> bool:
        return all(0 <= s.position - position <= s.rollback_capacity() for s in self.components)

    def rewind_to(self, position: int):
        for s in self.components:
            s.rewind(s.position - position)

    @property
    def supports_async_stash(self) -> bool:
        return all(getattr(s, "supports_async_stash", False) for s in self.components)

    def stash(self, async_: bool = False, pin_pool: dict = None) -> dict:
        """async_ (RecurrentCache.stash_async only): each side stashes without a host sync and the sides' D2H events
        are merged into one top-level "_ev" (recorded after both sides' copies on the current stream)"""
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
        }
        for side in ("target", "draft"):
            s = getattr(self, side)
            if s is None:
                continue
            stashed[side] = s.stash(async_ = True, pin_pool = pin_pool) if async_ else s.stash()
        if async_:
            evs = [stashed[side].pop("_ev") for side in ("target", "draft") if side in stashed and "_ev" in stashed[side]]
            if evs:
                stashed["_ev"] = evs[-1]
        return stashed

    def free(self):
        for s in self.components:
            s.free()

    def reset(self):
        for s in self.components:
            s.reset()


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
        draft_model = None,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model
        # This cache is what churns MB-scale host blocks (see util.memory.set_host_mmap_threshold
        # for why glibc strands them), so fix the allocator policy before the first stash.
        self.mmap_threshold = set_host_mmap_threshold()
        # The models behind each side of a stashed pair, for the tensor-parallel ranks' copies
        self.models = {"target": model, "draft": draft_model}

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        # Optional disk tier (DiskPageCache): checkpoints are written through on put() and read back on a miss
        self.disk_tier = None
        # EXL3_NC2_ASYNC_STASH: stashes whose D2H is still in flight (see settle)
        self._pending = []
        self._pin_pool = {}  # (shape, dtype) -> pinned host tensors free for reuse
        self._worker = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
        }


    def _finish(self, st: dict):
        """Worker thread: wait for the stash's D2H event, replace the pinned copies by pageable clones (the cache
        never retains pinned memory), return the pinned buffers to the pool."""
        st.pop("_ev").synchronize()
        self._unpin(st)

    def _unpin(self, st: dict):
        for k, v in st.items():
            if isinstance(v, dict):
                self._unpin(v)
            elif isinstance(v, tuple) and v and all(hasattr(t, "is_pinned") and t.is_pinned() for t in v):
                st[k] = tuple(t.clone() for t in v)
                for t in v:
                    self._pin_pool.setdefault((tuple(t.shape), t.dtype), []).append(t)

    def settle(self):
        """
        EXL3_NC2_ASYNC_STASH: wait until every pending asynchronous stash is finished (_finish). Called before any
        reader sees an entry; normally a no-op since the worker ran during the next chunk.
        """
        pend = self._pending
        if not pend:
            return
        self._pending = []
        for f in pend:
            f.result()

    def async_stash_ok(self, state) -> bool:
        # The disk tier writes a stash through as soon as it is inserted, so it must not see one still in flight
        return os.environ.get("EXL3_NC2_ASYNC_STASH", "0") == "1" and self.disk_tier is None and \
            not any(m is not None and m.loaded_tp for m in self.models.values()) and \
            getattr(state, "supports_async_stash", False)

    def stash_async(self, state) -> dict:
        """Stash state without a host sync; the copies are pending until settle()."""
        self.settle()
        stashed_state = state.stash(async_ = True, pin_pool = self._pin_pool)
        if "_ev" in stashed_state:
            if self._worker is None:
                self._worker = ThreadPoolExecutor(1, thread_name_prefix = "rc-stash")
            self._pending.append(self._worker.submit(self._finish, stashed_state))
        return stashed_state

    def get(self, key, default = None):
        self.settle()
        return super().get(key, default)

    def __getitem__(self, key):
        self.settle()
        return super().__getitem__(key)

    def items(self):
        self.settle()
        return super().items()

    def values(self):
        self.settle()
        return super().values()

    def has(self, key) -> bool:
        """Whether a checkpoint for key can be produced, from RAM or from the disk tier"""
        return key in self or (self.disk_tier is not None and self.disk_tier.has_state(key))


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue. A checkpoint that is only on disk is read
        back into the cache first.
        """
        self.settle()
        if key in self:
            self.move_to_end(key)
            return self[key]
        if self.disk_tier is not None:
            stashed = self.disk_tier.read_state(key)
            if stashed is not None:
                self._insert(key, stashed, persist = False)
                return stashed
        return default


    def rekey(self, old_key, new_key):
        """Move a checkpoint to the final hash of its anchor page (deferred pages are hashed once their
        tokens are final)"""
        if old_key in self and new_key not in self:
            stashed = self.pop(old_key)
            self[new_key] = stashed
            if self.disk_tier is not None:
                self.disk_tier.rekey_state(old_key, new_key, stashed)


    def put(self, key, state, stashed = None):
        """
        Add state to cache. stashed: an already stashed state (same format as state.stash()), e.g. a mid-chunk
        checkpoint; state is ignored then
        """
        if key in self:
            self.move_to_end(key)
        else:
            self._insert(key, stashed, persist = True, state = state if stashed is None else None)


    def _insert(self, key, stashed, persist: bool, state = None):
        """Make room for a checkpoint, then stash `state` (or take `stashed` as is) under key"""
        # Evict before stashing so the pool's peak occupancy is the cache limit, not the limit
        # plus the incoming checkpoint: the evicted buffers are what the new stash reuses
        state_size = state.checkpoint_size if state is not None else stashed["checkpoint_size"]
        # A checkpoint larger than the whole budget (e.g. -rcs 0) is not stored at all, without evicting or
        # stashing anything; a later restore misses and falls back to replay
        if state_size > self.max_size:
            return
        while self.update_total_size() + state_size > self.max_size:
            assert self.current_size >= 0, "Not enough space in cache for single state"
            pt = self.pagetable

            # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
            # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
            # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
            # extra cost, since the missing pages force a replay past this position either way.
            popped_key = None
            if pt is not None:
                for k in self:
                    if not pt.is_resumable(k):
                        popped_key = k
                        break
            if popped_key is not None:
                popped = self.pop(popped_key)
                self.metrics["stash_evictions_stranded"] += 1
            else:
                popped_key, popped = self.popitem(last = False)
                if pt is not None:
                    page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                    if page is not None and page.kv_position == PAGE_SIZE:
                        self.metrics["stash_evictions_live_kv"] += 1

            self.metrics["stash_evictions"] += 1
            self._release(popped)

        if state is not None:
            stashed = self.stash_async(state) if self.async_stash_ok(state) else state.stash()
        self[key] = stashed
        self.update_total_size()
        if persist and self.disk_tier is not None and is_content_hash(key):
            self.disk_tier.note_state(key, stashed)


    def drop(self, key) -> None:
        """Remove one checkpoint (rolling-checkpoint cleanup)."""
        popped = OrderedDict.pop(self, key, None)
        if popped is None:
            return
        if not any(v is popped for v in OrderedDict.values(self)):
            self._release(popped)
        self.update_total_size()


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k in self if not self.pagetable.is_resumable(k)]
        for k in stranded:
            popped = self.pop(k)
            self.metrics["stash_pruned"] += 1
            self._release(popped)
        if stranded:
            self.update_total_size()
        return len(stranded)


    def close(self):
        """
        Drop every checkpoint, return its buffers to the stash pool and release the pool, so the RAM goes
        back to the OS now rather than when this object is garbage collected. For a generator being retired:
        nothing restores from a closed cache, and a replacement generator's own cache would otherwise fill up
        alongside the checkpoints still stashed here. Safe to call more than once.
        """
        seen = set()
        freed = 0
        while len(self):
            _, popped = self.popitem(last = False)
            # Several keys may share one stash
            if id(popped) in seen:
                continue
            seen.add(id(popped))
            host_pool.give(popped)
            freed += popped["checkpoint_size"]
            self._release_tp(popped)
        self.current_size = 0
        self.pagetable = None
        if freed:
            note_freed(freed)
        host_pool.release()
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_host_pool_release, ())
        malloc_trim()


    def _release_tp(self, stashed: dict):
        """Drop the tensor-parallel ranks' copies of a stash: one handle per side of a pair (or at
        the top level of a bare state stash)"""
        for key in ("target", "draft", None):
            s = stashed if key is None else stashed.get(key)
            if not isinstance(s, dict) or "tp_handle" not in s:
                continue
            model = self.model if key is None else self.models[key]
            if model is not None and model.loaded_tp:
                model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), s["tp_handle"]))

    def _release(self, stashed: dict):
        # A stash the disk tier is still writing keeps its buffers until the write is done
        if self.disk_tier is not None:
            self.disk_tier.release_after_write(stashed, self._release_now)
        else:
            self._release_now(stashed)

    def _release_now(self, stashed: dict):
        host_pool.give(stashed)
        note_freed(stashed["checkpoint_size"])
        self._release_tp(stashed)

    def update_total_size(self):
        seen = set()
        total = 0
        for v in OrderedDict.values(self):  # sizes only: must not settle in-flight stashes
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    host_pool.give(stashed)
    note_freed(_stashed_bytes(stashed))
