from __future__ import annotations
import atexit
import logging
import queue
import threading
import time
import torch
from .disk_store import RecordStore, store_fingerprint

logger = logging.getLogger(__name__)

_DTYPES = {
    str(d): d for d in (
        torch.float64, torch.float32, torch.float16, torch.bfloat16,
        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
        torch.float8_e4m3fn, torch.float8_e5m2,
    )
}
_STOP = object()


def _bytes_of(t: torch.Tensor) -> memoryview:
    return memoryview(t.contiguous().flatten().view(torch.uint8).numpy())


def _empty_like_spec(shape, dtype_str, nbytes) -> torch.Tensor:
    dtype = _DTYPES[dtype_str]
    flat = torch.empty((nbytes,), dtype = torch.uint8)
    return flat.view(dtype).view(*shape)


class _Writer:
    """
    Single background thread draining a bounded queue of records to disk. Writes are deferred because they sit
    on the generator's critical path: a page is handed over during allocation, while jobs wait.

    A full queue blocks the submitter rather than dropping the record. Dropping is not a local cost: a missing
    page breaks the hash chain for every page above it, so the whole suffix of that context stops being
    restorable. Blocking costs one disk write, and only happens when pages are completing faster than the disk
    can absorb them. The bound exists so a failing or full disk cannot stall the generator indefinitely.
    """

    def __init__(self, name: str, depth: int, block_s: float = 5.0):
        self.q = queue.Queue(maxsize = depth)
        self.block_s = block_s
        self._thread = threading.Thread(target = self._run, name = name, daemon = True)
        self._thread.start()
        atexit.register(self._at_exit)


    def submit(self, fn) -> bool:
        try:
            self.q.put(fn, timeout = self.block_s)
            return True
        except queue.Full:
            return False


    def _run(self):
        with torch.inference_mode():
            while True:
                fn = self.q.get()
                try:
                    if fn is _STOP:
                        return
                    fn()
                except Exception:
                    logger.exception("Disk cache writer failed")
                finally:
                    self.q.task_done()


    def drain(self, timeout: float | None = None) -> bool:
        # queue.join() has no timeout, and an exit handler must not be able to wedge the interpreter
        with self.q.all_tasks_done:
            if timeout is None:
                while self.q.unfinished_tasks:
                    self.q.all_tasks_done.wait()
                return True
            deadline = time.monotonic() + timeout
            while self.q.unfinished_tasks:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.q.all_tasks_done.wait(left)
        return True


    def close(self):
        atexit.unregister(self._at_exit)
        self.q.put(_STOP)
        self._thread.join(timeout = 30)


    def _at_exit(self):
        self.drain(10.0)


class DiskPageTier:
    """
    Durable third tier under the pinned-host CPUPageCache: complete hashed K/V pages written to one file each,
    named from the page hash, under a directory named for the layout fingerprint they were produced by.

    Nothing is repopulated at startup. The index is rebuilt from record headers when the store opens, which is
    enough for the existing allocation path to find a returning context's pages by hash and page them in on
    demand; eagerly loading would stall startup and spend host RAM on contexts that may never come back.
    """

    def __init__(
        self,
        root: str,
        fingerprint: str,
        max_size: int,
        slab_size: int,
        generation: str | None = None,
        queue_depth: int = 8,
    ):
        self.store = RecordStore(root, fingerprint, "kv", max_size, generation)
        self.slab_size = slab_size
        self.writer = _Writer("exl3-disk-kv", queue_depth)
        # Hashes queued but not yet written, so a repeated sweep does not schedule the same page twice
        self.pending = set()
        self.metrics = {
            "writes": 0,
            "restores": 0,
            "dropped": 0,
            "dedup_hits": 0,
        }


    def __contains__(self, phash: bytes):
        return phash in self.store


    def __len__(self):
        return len(self.store)


    def will_store(self, phash: bytes) -> bool:
        return phash in self.store or phash in self.pending


    def prev_hash(self, phash: bytes):
        e = self.store.index.get(phash)
        if e is None:
            return False, None
        pv = e[3].get("pv")
        return True, (bytes.fromhex(pv) if pv else None)


    def links(self) -> dict:
        out = {}
        for phash, manifest in self.store.snapshot():
            pv = manifest.get("pv")
            out[phash] = bytes.fromhex(pv) if pv else None
        return out


    def schedule(self, phash, prev_hash, tokens: torch.Tensor, slab: torch.Tensor, event, release) -> bool:
        """
        Persist one page once the pending device-to-host copy into slab has landed. The caller guarantees the
        slab is not recycled until release() is called.
        """
        if phash in self.store or phash in self.pending:
            self.metrics["dedup_hits"] += 1
            release()
            return False

        manifest = {
            "pv": prev_hash.hex() if prev_hash is not None else None,
            "ts": list(tokens.shape),
            "td": str(tokens.dtype),
            "tn": tokens.numel() * tokens.element_size(),
        }
        slab_mv = memoryview(slab.numpy())[:self.slab_size]
        tokens = tokens.contiguous()

        def run():
            try:
                if event is not None:
                    event.synchronize()
                if self.store.write(phash, manifest, [slab_mv, _bytes_of(tokens)]):
                    self.metrics["writes"] += 1
            finally:
                self.pending.discard(phash)
                release()

        self.pending.add(phash)
        if not self.writer.submit(run):
            self.metrics["dropped"] += 1
            self.pending.discard(phash)
            release()
            return False
        return True


    def load(self, phash: bytes, slab: torch.Tensor):
        """
        Page one record back into slab. Returns (prev_hash, tokens) or None on a miss or a record that failed
        verification.
        """
        e = self.store.index.get(phash)
        if e is None:
            return None
        manifest = e[3]
        try:
            tokens = _empty_like_spec(manifest["ts"], manifest["td"], manifest["tn"])
        except (KeyError, ValueError, RuntimeError):
            self.store.remove(phash)
            return None

        slab_mv = memoryview(slab.numpy())[:self.slab_size]
        got, _ = self.store.read(phash, into = [slab_mv, _bytes_of(tokens)])
        if got is None:
            return None
        self.metrics["restores"] += 1
        pv = got.get("pv")
        return (bytes.fromhex(pv) if pv else None), tokens


    def drain(self):
        self.writer.drain()


    def close(self):
        self.writer.close()


class DiskCheckpointTier:
    """
    Durable tier under RecurrentCache: one file per recurrent checkpoint, keyed by the same page hash that
    anchors it in host RAM. Without this, a restored K/V chain on a hybrid model caps the resumable prefix at
    zero, since the prefix cap is the intersection of valid pages and available checkpoints.
    """

    def __init__(
        self,
        root: str,
        identity: dict,
        geometry,
        max_size: int,
        queue_depth: int = 4,
    ):
        self.store = RecordStore(
            root,
            store_fingerprint({"t": "recurrent", "state": geometry, **identity}),
            "rec",
            max_size,
            store_fingerprint(identity),
        )
        self.writer = _Writer("exl3-disk-rec", queue_depth)
        self.metrics = {
            "writes": 0,
            "restores": 0,
            "dropped": 0,
            "dedup_hits": 0,
            "unserializable": 0,
        }


    def __contains__(self, phash: bytes):
        return phash in self.store


    def __len__(self):
        return len(self.store)


    def put(self, phash: bytes, stashed: dict) -> bool:
        if phash in self.store:
            self.metrics["dedup_hits"] += 1
            return False
        if "tp_handle" in stashed:
            # A tensor-parallel stash is a handle into the worker processes; this process holds no bytes for it
            self.metrics["unserializable"] += 1
            return False

        layers, parts = [], []
        for k, v in stashed.items():
            if k in ("position", "checkpoint_size"):
                continue
            # Layer states stash a bare tensor, a tuple or a list, and unstash() is written against whichever
            # one it produced, so the container is part of the record
            container = "l" if isinstance(v, list) else "t" if isinstance(v, tuple) else "x"
            tensors = v if container != "x" else (v,)
            specs = []
            for t in tensors:
                if not isinstance(t, torch.Tensor) or str(t.dtype) not in _DTYPES:
                    self.metrics["unserializable"] += 1
                    return False
                t = t.contiguous()
                specs.append([list(t.shape), str(t.dtype), t.numel() * t.element_size()])
                parts.append(t)
            kk = ["T", list(k)] if isinstance(k, tuple) else ["S", k]
            layers.append([kk[0], kk[1], container, specs])

        manifest = {
            "pos": int(stashed["position"]),
            "cs": int(stashed["checkpoint_size"]),
            "ls": layers,
        }
        parts = [_bytes_of(t) for t in parts]

        def run():
            if self.store.write(phash, manifest, parts):
                self.metrics["writes"] += 1

        if not self.writer.submit(run):
            self.metrics["dropped"] += 1
            return False
        return True


    def get(self, phash: bytes):
        e = self.store.index.get(phash)
        if e is None:
            return None
        manifest = e[3]
        try:
            out = {"position": manifest["pos"], "checkpoint_size": manifest["cs"]}
            parts = []
            for keykind, keyvalue, container, specs in manifest["ls"]:
                tensors = []
                for shape, dtype_str, nbytes in specs:
                    t = _empty_like_spec(shape, dtype_str, nbytes)
                    tensors.append(t)
                    parts.append(_bytes_of(t))
                key = tuple(keyvalue) if keykind == "T" else keyvalue
                out[key] = tensors if container == "l" else tuple(tensors) if container == "t" else tensors[0]
        except (KeyError, ValueError, TypeError, RuntimeError):
            self.store.remove(phash)
            return None

        got, _ = self.store.read(phash, into = parts)
        if got is None:
            return None
        self.metrics["restores"] += 1
        return out


    def drain(self):
        self.writer.drain()


    def close(self):
        self.writer.close()
