import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import torch

"""
EXL3_NGRAM_ROW_CACHE: keeping recently read rows of the streamed n-gram table resident must be
bit-exact against the uncached preads, including across ring wraparound and from two threads at
once, and must actually elide reads (the behaviour that MUST differ: syscall count).

Runs against the real checkpoint's table; no GPU needed -- the row gather is pure host I/O.
"""

MODEL = os.environ.get("EXL3_TEST_MODEL",
                       "/home/dbrain/dev/tinyfiddler/exllamav3/_models/flashnext-4.05bpw")
KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import NGramEmbedding



def syscr():
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("syscr:"):
                return int(line.split(":")[1])
    return -1


def make_module():
    stc = SafetensorsCollection(MODEL)
    assert stc.has_tensor(f"{KEY}.trellis") or stc.has_tensor(f"{KEY}.shard_0.trellis"), \
        f"no n-gram table under {KEY} in {MODEL}"
    mod = NGramEmbedding(config = SimpleNamespace(stc = stc, infer_params = None), key = KEY,
                         ngram_size = 3, heads_per_ngram = 8, ple_embed_dim = 2560,
                         eos_token_id = 248044, stream_from_disk = True)
    mod.load(torch.device("cpu"))
    assert mod.mode.endswith("disk") and mod.handles is not None and mod.tables is None, \
        f"expected the streamed handles branch, got mode={mod.mode}"
    return mod


def uid_set(rng, n, num_rows):
    return torch.tensor(sorted(set(int(rng.integers(0, num_rows)) for _ in range(n))),
                        dtype = torch.int64)


def gather(mod, uids, words, dtype):
    out = torch.empty((uids.numel(), words), dtype = dtype)
    mod._gather_rows(uids, out)
    return out


def main():
    import numpy as np
    rng = np.random.default_rng(0xE313)

    oracle = make_module()
    assert oracle._cache_rows == 0, "oracle must have the cache off (EXL3_NGRAM_ROW_CACHE unset)"
    from exllamav3.modules.quant.exl3_lib.ngram_codec import words_per_row
    words = words_per_row(oracle.K)
    dtype = torch.int16

    mod = make_module()
    mod._cache_rows = 4096

    sets = [uid_set(rng, k, oracle.num_rows) for k in (16, 16, 5, 64, 16, 300, 16)]

    # 1. bit-exact, first (all-miss) and second (all-hit) pass
    for u in sets:
        ref = gather(oracle, u, words, dtype)
        a = gather(mod, u, words, dtype)
        b = gather(mod, u, words, dtype)
        assert torch.equal(a, ref), "cached gather differs from the uncached rows"
        assert torch.equal(b, ref), "cache replay differs from the uncached rows"
    st = dict(mod._cache_stats)
    assert st["hit"] > 0 and st["miss"] > 0, st

    # 2. MUST-DIFFER control: a fully cached repeat issues no read syscalls at all.
    # syscr() reads /proc/self/io, which is itself a read syscall -- subtract that.
    p0 = syscr(); probe = syscr() - p0
    u = sets[0]
    gather(mod, u, words, dtype)
    c0 = syscr(); gather(mod, u, words, dtype); c_hit = syscr() - c0 - probe
    c0 = syscr(); gather(oracle, u, words, dtype); c_miss = syscr() - c0 - probe
    assert c_hit == 0 and c_miss >= u.numel(), \
        f"cache did not elide reads: hit-path syscr={c_hit}, uncached={c_miss}"

    # 3. ring wraparound: a cap far smaller than the traffic must still never serve a wrong row
    small = make_module()
    small._cache_rows = 32
    for _ in range(24):
        u = uid_set(rng, int(rng.integers(4, 40)), oracle.num_rows)
        assert torch.equal(gather(small, u, words, dtype), gather(oracle, u, words, dtype))
    assert len(small._cache_map) <= 32 and small._cache_slab.shape[0] == 32
    assert sum(1 for x in small._cache_ring if x >= 0) <= 32

    # 4. two threads gathering at once (prefetch worker + inline stage)
    par = make_module()
    par._cache_rows = 2048
    us = [uid_set(rng, 16, oracle.num_rows) for _ in range(64)]
    refs = [gather(oracle, u, words, dtype) for u in us]
    errs = []

    def run(lo, hi):
        try:
            for _ in range(4):
                for i in range(lo, hi):
                    assert torch.equal(gather(par, us[i], words, dtype), refs[i]), f"row mismatch at {i}"
        except Exception as e:
            errs.append(e)

    ts = [threading.Thread(target = run, args = (0, 40)), threading.Thread(target = run, args = (24, 64))]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errs, errs

    # 5. a gather larger than the ring is passed through, not wrapped onto itself
    big = make_module()
    big._cache_rows = 64
    u = uid_set(rng, 400, oracle.num_rows)
    assert torch.equal(gather(big, u, words, dtype), gather(oracle, u, words, dtype))
    assert not big._cache_map

    print(f" -- row cache bit-exact; stats {st}; hit-path syscr {c_hit} vs uncached {c_miss}")
    print("ok")


if __name__ == "__main__":
    # Scoped to the run, NOT entered at module scope. This file is named test_* but defines no
    # test functions, so pytest imports it during collection and never runs anything in it -- a
    # module-scope inference_mode that is never exited therefore leaks into every test collected
    # afterwards, and g_tensor_cache then hands out inference tensors for the rest of the
    # session (see tests/test_tensor_cache_inference.py, which is what catches it).
    with torch.inference_mode():
        main()
