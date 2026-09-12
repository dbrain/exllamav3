from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import struct
import threading
import time

logger = logging.getLogger(__name__)

MAGIC = b"EXL3DSK2"
HEADER_SIZE = 48
_HEADER = struct.Struct("<8sIIQ16s8x")
_DIGEST_SIZE = 16
_SCAN_READ = 4096

# SHA-NI makes sha256 3.6x faster than hashlib's blake2b here (1.18 vs 0.33 GiB/s measured under load), and the
# digest is the dominant cost of a restore: 5.3 GiB of K/V for a 200k context. Truncated to the header's 16-byte
# field, which is the integrity margin this needs, not a signature. MAGIC encodes the choice, so records written
# by a build using a different one are rejected structurally rather than failing verification one at a time.
def _digest():
    return hashlib.sha256()


def store_fingerprint(fields: dict) -> str:
    """
    Identity of the byte layout a record was written under. Any difference in it must produce a different
    value, since the alternative to refusing a restore is reinterpreting foreign bytes as cache state.
    """
    blob = json.dumps(fields, sort_keys = True, separators = (",", ":"), default = str).encode()
    return hashlib.blake2b(blob, digest_size = 8).hexdigest()


def _as_parts(parts) -> list:
    if parts is None:
        return []
    if isinstance(parts, (bytes, bytearray, memoryview)):
        return [memoryview(parts).cast("B")]
    return [memoryview(p).cast("B") for p in parts]


class RecordStore:
    """
    Durable, content-addressed record store: one file per key, named from the key, grouped under a directory
    named for the layout fingerprint of the process that wrote it.

    A record is a fixed header (magic, manifest length, payload length, payload+manifest digest), a JSON
    manifest and a raw payload. Writes are staged under a temporary name in the same directory and renamed
    into place, so a reader never observes a half-written record. Reads verify the digest before the payload
    is handed back.

    The store is bounded in bytes and evicted oldest-mtime-first. Reads touch the file, so recency survives a
    restart; the index itself is rebuilt at open by reading headers only.

    The byte budget covers one fingerprint. Directories belonging to other fingerprints are generations of a
    superseded model or build; each keeps a stamp file refreshed while it is in use, and generations whose stamp
    has gone stale are deleted at open so a sequence of rebuilds cannot grow the directory without bound.
    """

    def __init__(
        self,
        root: str,
        fingerprint: str,
        kind: str,
        max_size: int,
        generation: str | None = None,
        stale_days: float = 7.0,
    ):
        self.fingerprint = fingerprint
        self.kind = kind
        self.max_size = int(max_size)
        # One generation per model/build, shared by every kind of record it produced, so the stamp that keeps a
        # generation alive is refreshed by all of them and pruning acts on the natural unit
        self.generation = os.path.join(str(root), f"exl3-cache-{generation or fingerprint}")
        self.root = os.path.join(self.generation, f"{kind}-{fingerprint[:12]}")
        os.makedirs(self.root, exist_ok = True)
        self.stamp = os.path.join(self.generation, "stamp")
        self._stamped = 0.0
        self._touch_stamp()
        self._prune_generations(str(root), stale_days)

        # key -> [path, size, mtime_ns, manifest]. Mutated by the background writer and by readers on the
        # generator thread, so every iteration of it is taken under the lock
        self.index = {}
        self.lock = threading.RLock()
        self.total_size = 0
        self._seq = 0
        self.metrics = {
            "written": 0,      # records renamed into place
            "read": 0,         # records read back and verified
            "rejected": 0,     # records refused for damage or a foreign fingerprint
            "evicted": 0,      # records dropped to stay inside the budget
            "write_errors": 0,
        }
        self._scan()


    def _touch_stamp(self):
        now = time.time()
        if now - self._stamped < 60.0:
            return
        self._stamped = now
        try:
            with open(self.stamp, "a"):
                os.utime(self.stamp, (now, now))
        except OSError:
            pass


    def _prune_generations(self, root: str, stale_days: float):
        if stale_days <= 0:
            return
        cutoff = time.time() - stale_days * 86400
        try:
            siblings = os.listdir(root)
        except OSError:
            return
        for name in siblings:
            d = os.path.join(root, name)
            if not name.startswith("exl3-cache-") or d == self.generation or not os.path.isdir(d):
                continue
            try:
                if os.stat(os.path.join(d, "stamp")).st_mtime >= cutoff:
                    continue
            except OSError:
                pass
            logger.info(f"Disk cache: dropping stale generation {name}")
            shutil.rmtree(d, ignore_errors = True)


    def __contains__(self, key: bytes):
        return key in self.index


    def __len__(self):
        return len(self.index)


    def path_for(self, key: bytes) -> str:
        h = key.hex()
        return os.path.join(self.root, h[:2], h + ".rec")


    def _reject(self, key, path, reason: str, unlink: bool):
        self.metrics["rejected"] += 1
        logger.warning(f"Disk cache ({self.kind}): refusing record {path}: {reason}")
        with self.lock:
            e = self.index.pop(key, None)
            if e is not None:
                self.total_size -= e[1]
        if unlink:
            try:
                os.unlink(path)
            except OSError:
                pass


    def _scan(self):
        for shard in os.listdir(self.root):
            d = os.path.join(self.root, shard)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if not name.endswith(".rec"):
                    # Staging files belong to a process that died mid-write
                    if ".tmp-" in name:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    continue
                try:
                    key = bytes.fromhex(name[:-4])
                except ValueError:
                    self._reject(None, path, "unparseable name", True)
                    continue
                try:
                    st = os.stat(path)
                    with open(path, "rb", buffering = 0) as f:
                        head = f.read(_SCAN_READ)
                        magic, mlen, _, plen, _ = _HEADER.unpack_from(head)
                        if magic != MAGIC:
                            raise ValueError("bad magic")
                        if st.st_size != HEADER_SIZE + mlen + plen:
                            raise ValueError("size does not match header")
                        if len(head) < HEADER_SIZE + mlen:
                            f.seek(HEADER_SIZE)
                            mb = f.read(mlen)
                        else:
                            mb = head[HEADER_SIZE : HEADER_SIZE + mlen]
                        manifest = json.loads(mb.decode())
                except (OSError, ValueError, struct.error, UnicodeDecodeError) as e:
                    self._reject(key, path, str(e), True)
                    continue
                if manifest.get("fp") != self.fingerprint:
                    self._reject(key, path, "foreign layout fingerprint", False)
                    continue
                self.index[key] = [path, st.st_size, st.st_mtime_ns, manifest]
                self.total_size += st.st_size


    def write(self, key: bytes, manifest_extra: dict, payload_parts) -> bool:
        parts = _as_parts(payload_parts)
        manifest = dict(manifest_extra)
        manifest["fp"] = self.fingerprint
        mb = json.dumps(manifest, sort_keys = True, separators = (",", ":")).encode()

        hasher = _digest()
        hasher.update(mb)
        plen = 0
        for p in parts:
            hasher.update(p)
            plen += p.nbytes
        header = _HEADER.pack(MAGIC, len(mb), 0, plen, hasher.digest()[:_DIGEST_SIZE])

        path = self.path_for(key)
        self._seq += 1
        stage = f"{path}.tmp-{os.getpid()}-{self._seq}"
        try:
            os.makedirs(os.path.dirname(path), exist_ok = True)
            with open(stage, "wb", buffering = 0) as f:
                f.write(header)
                f.write(mb)
                for p in parts:
                    f.write(p)
            os.replace(stage, path)
        except OSError as e:
            self.metrics["write_errors"] += 1
            logger.warning(f"Disk cache ({self.kind}): write failed for {path}: {e}")
            try:
                os.unlink(stage)
            except OSError:
                pass
            return False

        self._touch_stamp()
        size = HEADER_SIZE + len(mb) + plen
        mtime = os.stat(path).st_mtime_ns
        with self.lock:
            old = self.index.get(key)
            if old is not None:
                self.total_size -= old[1]
            self.index[key] = [path, size, mtime, manifest]
            self.total_size += size
            self.metrics["written"] += 1
            self._enforce_budget(keep = key)
        return True


    def read(self, key: bytes, into = None):
        """
        Verified read. Returns (manifest, payload_bytes), or (manifest, None) when the payload was written into
        the caller's buffers, or (None, None) on a miss or a record that failed verification.
        """
        e = self.index.get(key)
        if e is None:
            return None, None
        path = e[0]
        dest = _as_parts(into) if into is not None else None
        try:
            with open(path, "rb", buffering = 0) as f:
                head = f.read(HEADER_SIZE)
                magic, mlen, _, plen, digest = _HEADER.unpack(head)
                if magic != MAGIC:
                    raise ValueError("bad magic")
                mb = f.read(mlen)
                if len(mb) != mlen:
                    raise ValueError("truncated manifest")
                manifest = json.loads(mb.decode())
                if manifest.get("fp") != self.fingerprint:
                    raise ValueError("foreign layout fingerprint")
                hasher = _digest()
                hasher.update(mb)
                if dest is None:
                    payload = f.read(plen)
                    if len(payload) != plen:
                        raise ValueError("truncated payload")
                    hasher.update(payload)
                else:
                    if sum(p.nbytes for p in dest) != plen:
                        raise ValueError("payload size does not match destination buffers")
                    payload = None
                    for p in dest:
                        if f.readinto(p) != p.nbytes:
                            raise ValueError("truncated payload")
                        hasher.update(p)
                # The digest is the only thing standing between a torn or rotted file and plausible-looking
                # cache state, so it gates the payload rather than merely being logged
                if hasher.digest()[:_DIGEST_SIZE] != digest:
                    raise ValueError("digest mismatch")
        except (OSError, ValueError, struct.error, UnicodeDecodeError) as ex:
            self._reject(key, path, str(ex), True)
            return None, None

        self.metrics["read"] += 1
        self.touch(key)
        return manifest, payload


    def touch(self, key: bytes):
        e = self.index.get(key)
        if e is None:
            return
        try:
            os.utime(e[0])
            e[2] = os.stat(e[0]).st_mtime_ns
        except OSError:
            pass


    def remove(self, key: bytes):
        with self.lock:
            e = self.index.pop(key, None)
            if e is None:
                return
            self.total_size -= e[1]
        try:
            os.unlink(e[0])
        except OSError:
            pass


    def _enforce_budget(self, keep: bytes | None = None):
        with self.lock:
            if self.total_size <= self.max_size:
                return
            order = sorted(
                ((e[2], k) for k, e in self.index.items() if k != keep),
                key = lambda t: (t[0], t[1]),
            )
            for _, k in order:
                if self.total_size <= self.max_size:
                    break
                self.remove(k)
                self.metrics["evicted"] += 1


    def snapshot(self) -> list:
        with self.lock:
            return [(k, e[3]) for k, e in self.index.items()]
