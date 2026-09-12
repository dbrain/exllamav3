import json
import time
import os
from pathlib import Path
import pytest
import torch

from exllamav3.generator.disk_store import RecordStore, store_fingerprint

FP_A = store_fingerprint({"model": "a", "page_size": 256})
FP_B = store_fingerprint({"model": "b", "page_size": 256})


def mk(root, fp = FP_A, max_size = 1 << 30, kind = "kv"):
    return RecordStore(root, fp, kind, max_size)


def read_manifest(path):
    raw = bytearray(open(path, "rb").read())
    mlen = int.from_bytes(raw[8:12], "little")
    return raw, mlen, json.loads(raw[48:48 + mlen].decode())


def test_fingerprint_is_stable_and_discriminating():
    assert store_fingerprint({"model": "a", "page_size": 256}) == FP_A
    assert store_fingerprint({"page_size": 256, "model": "a"}) == FP_A, "key order must not matter"
    assert FP_A != FP_B


def test_record_roundtrip(tmp_path):
    s = mk(tmp_path)
    key = bytes(range(16))
    payload = os.urandom(4096)
    s.write(key, {"pv": None}, [payload])

    s2 = mk(tmp_path)
    assert key in s2
    manifest, data = s2.read(key)
    assert manifest["pv"] is None
    assert data == payload


def test_read_into_fills_a_caller_buffer(tmp_path):
    s = mk(tmp_path)
    key = b"k" * 16
    payload = os.urandom(8192)
    s.write(key, {}, [payload])

    buf = torch.empty((8192,), dtype = torch.uint8)
    manifest, data = mk(tmp_path).read(key, into = memoryview(buf.numpy()))
    assert data is None
    assert buf.numpy().tobytes() == payload


def test_a_different_fingerprint_does_not_see_records(tmp_path):
    key = b"x" * 16
    mk(tmp_path, FP_A).write(key, {}, [b"\xaa" * 1024])

    other = mk(tmp_path, FP_B)
    assert key not in other
    assert other.read(key) == (None, None)

    # The stores coexist rather than one invalidating the other
    assert key in mk(tmp_path, FP_A)


def test_a_record_bearing_a_foreign_fingerprint_is_refused(tmp_path):
    key = b"y" * 16
    s = mk(tmp_path, FP_A)
    s.write(key, {}, [b"\xbb" * 512])
    path = s.path_for(key)
    raw, mlen, manifest = read_manifest(path)
    manifest["fp"] = FP_B
    nm = json.dumps(manifest, sort_keys = True, separators = (",", ":")).encode()
    assert len(nm) == mlen, "test needs an equal-length manifest so only fp changes"
    raw[48:48 + mlen] = nm
    open(path, "wb").write(bytes(raw))

    s2 = mk(tmp_path, FP_A)
    assert s2.read(key) == (None, None)
    assert s2.metrics["rejected"] == 1


@pytest.mark.parametrize("damage", ["flip_payload", "flip_manifest", "truncate", "bad_magic", "short_header"])
def test_a_damaged_record_is_rejected_not_served(tmp_path, damage):
    key = b"z" * 16
    s = mk(tmp_path)
    payload = os.urandom(4096)
    s.write(key, {"pv": None}, [payload])
    path = s.path_for(key)

    raw, mlen, body = read_manifest(path)
    if damage == "flip_payload":
        raw[-17] ^= 0x01
    elif damage == "flip_manifest":
        body["pv"] = "00" * 16
        nm = json.dumps(body, sort_keys = True, separators = (",", ":")).encode()
        raw = bytearray(raw[:8] + len(nm).to_bytes(4, "little") + raw[12:48] + nm + raw[48 + mlen:])
    elif damage == "truncate":
        raw = raw[:len(raw) // 2]
    elif damage == "bad_magic":
        raw[0:1] = b"Q"
    elif damage == "short_header":
        raw = raw[:20]
    open(path, "wb").write(bytes(raw))

    s2 = mk(tmp_path)
    assert s2.read(key) == (None, None), f"{damage} was served"
    assert s2.metrics["rejected"] == 1
    assert key not in s2, "a record that failed verification must be dropped from the index"
    assert not os.path.exists(path), "a record that failed verification must be unlinked"


def test_a_crash_mid_write_leaves_nothing_indexable(tmp_path):
    s = mk(tmp_path)
    key = b"w" * 16
    stage = Path(str(s.path_for(key)) + ".tmp-999-0")
    stage.parent.mkdir(parents = True, exist_ok = True)
    stage.write_bytes(b"\x00" * 4096)

    s2 = mk(tmp_path)
    assert key not in s2
    assert not stage.exists(), "stale staging files must be swept at open"


def test_store_is_bounded_and_evicts_oldest_first(tmp_path):
    rec = 4096
    budget = 10 * (rec + 512)
    s = mk(tmp_path, max_size = budget)
    keys = [bytes([i]) * 16 for i in range(40)]
    for k in keys:
        s.write(k, {}, [os.urandom(rec)])

    assert s.total_size <= budget
    assert len(s.index) < 40, "store grew past its budget"
    assert keys[-1] in s, "the newest record must survive"
    assert keys[0] not in s, "the oldest record must be the first evicted"
    assert s.metrics["evicted"] == 40 - len(s.index)
    assert mk(tmp_path, max_size = budget).total_size == s.total_size


def test_reopening_rebuilds_the_index_from_the_files(tmp_path):
    s = mk(tmp_path)
    keys = [bytes([i]) * 16 for i in range(8)]
    for k in keys:
        s.write(k, {"n": len(k)}, [os.urandom(1024)])
    size, count = s.total_size, len(s.index)

    s2 = mk(tmp_path)
    assert set(s2.index) == set(keys)
    assert (s2.total_size, len(s2.index)) == (size, count)


def test_kinds_are_separate_namespaces(tmp_path):
    key = b"q" * 16
    mk(tmp_path, kind = "kv").write(key, {}, [b"\x01" * 64])
    assert key not in mk(tmp_path, kind = "rec")
    assert key in mk(tmp_path, kind = "kv")


def test_a_stale_generation_is_dropped_but_a_live_one_is_kept(tmp_path):
    old = mk(tmp_path, FP_B)
    old.write(b"o" * 16, {}, [b"\x01" * 256])
    live = mk(tmp_path, store_fingerprint({"model": "c"}))
    live.write(b"l" * 16, {}, [b"\x02" * 256])

    stale = 8 * 86400
    now = time.time()
    os.utime(old.stamp, (now - stale, now - stale))

    s = mk(tmp_path, FP_A)
    assert not os.path.isdir(old.generation), "a stale generation was not reclaimed"
    assert os.path.isdir(live.generation), "a generation still in use was deleted"
    assert os.path.isdir(s.generation)


def test_generations_are_never_pruned_when_disabled(tmp_path):
    old = RecordStore(tmp_path, FP_B, "kv", 1 << 30, stale_days = 0)
    now = time.time()
    os.utime(old.stamp, (now - 99 * 86400, now - 99 * 86400))
    RecordStore(tmp_path, FP_A, "kv", 1 << 30, stale_days = 0)
    assert os.path.isdir(old.generation)
