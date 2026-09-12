"""Semantics of the KV-cache quantization fallbacks (exllamav3/ext_fallback_qcache.py) against a
literal transcription of exllamav3_ext/cache/q_cache_kernels.cuh.

The reference here simulates the warp arithmetic the kernel actually performs -- the in-register
H4 plus the 3-round subgroup H8, and pack_plane's (sg, sl, off) word/shift computation -- rather
than the collapsed forms the fallback uses (one 32x32 matmul, bit w*j of the plane). So it checks
the collapses, not just the code against itself. The other half of the validation is upstream:
tests/test_triton_paged_hdpad.py::test_qc_hdpad and tests/test_mla.py's quant-cache tests run the
independently written Triton plane loaders over bytes this code packed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallback_qcache as fbq

device = os.environ.get("EXL_TEST_DEVICE", "cuda:0")
R32 = 0.17677669529663688110
PAGE = 256


# -- literal kernel transcription ---------------------------------------------

def _had_4_inreg(v):
    s0, d0, s1, d1 = v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3]
    return [s0 + s1, d0 + d1, s0 - s1, d0 - d1]


def _had_8_subgroup(reg):
    # reg[sl] = [v0..v3]; one __shfl_xor round per stage, sign flip where (lane & i)
    i = 1
    while i < 8:
        out = [None] * 8
        for sl in range(8):
            p = reg[sl ^ i]
            sgn = -1.0 if (sl & i) else 1.0
            out[sl] = [sgn * reg[sl][t] + p[t] for t in range(4)]
        reg = out
        i <<= 1
    return reg


def _ref_rotate(x32):
    reg = [_had_4_inreg([x32[sl * 4 + t] for t in range(4)]) for sl in range(8)]
    reg = _had_8_subgroup(reg)
    return [reg[sl][t] * R32 for sl in range(8) for t in range(4)]


def _lm_encode(x, bits, a):
    b = 1.0 - a
    p3 = a / b / 3.0
    qh = x / b * 0.5
    s = math.sqrt(qh * qh + p3 ** 3)
    cb = lambda z: math.copysign(abs(z) ** (1.0 / 3.0), z)
    t = cb(qh + s) + cb(qh - s)
    hn = float(1 << (bits - 1))
    return min(max(int(math.floor(t * hn + hn)), 0), (1 << bits) - 1)


def _lm_decode(q, bits, a):
    t = (2.0 * q + 1.0) / (1 << bits) - 1.0
    return t * (a + (1.0 - a) * t * t)


def _ref_quant_group(x32, bits, compand_a):
    y = _ref_rotate(x32)
    s = max(abs(v) for v in y) + 1e-10
    inv = 1.0 / s
    if compand_a > 0.0:
        q = [_lm_encode(v * inv, bits, compand_a) for v in y]
    else:
        m = float(1 << (bits - 1))
        q = [min(max(int(math.floor(v * inv * m + m)), 0), (1 << bits) - 1) for v in y]
    return q, s


def _ref_pack_group(q, bits):
    # pack_plane(): 8 lanes x 4 consecutive values, field at bit sl*4*w of the plane
    words = [0] * bits
    rem, wb = bits, 0
    for w in (8, 4, 2, 1):
        if bits & w:
            rem -= w
            for sl in range(8):
                field = 0
                for t in range(4):
                    field |= ((q[sl * 4 + t] >> rem) & ((1 << w) - 1)) << (t * w)
                off = sl * 4 * w
                words[wb + (off >> 5)] |= field << (off & 31)
            wb += w
    return words


def _ref_dequant_group(words, scale, bits, compand_a):
    rem, wb = bits, 0
    q = [0] * 32
    for w in (8, 4, 2, 1):
        if bits & w:
            rem -= w
            for sl in range(8):
                off = sl * 4 * w
                word = words[wb + (off >> 5)] >> (off & 31)
                for t in range(4):
                    q[sl * 4 + t] = (q[sl * 4 + t] << w) | ((word >> (t * w)) & ((1 << w) - 1))
            wb += w
    s = scale * R32
    if compand_a > 0.0:
        v = [_lm_decode(qq, bits, compand_a) * s for qq in q]
    else:
        mh = float(1 << (bits - 1)) - 0.5
        v = [(qq - mh) * (s / (1 << (bits - 1))) for qq in q]
    reg = [_had_4_inreg([v[sl * 4 + t] for t in range(4)]) for sl in range(8)]
    reg = _had_8_subgroup(reg)
    return [reg[sl][t] for sl in range(8) for t in range(4)]


def _as_u32(w):
    return [x & 0xFFFFFFFF for x in w]


# -- tests --------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_pack_layout_is_the_kernels(bits):
    """Bit layout only -- integer codes in, words out, no floating point anywhere, so this is an
    exact check on both the plane order and the within-word slot order."""
    torch.manual_seed(bits)
    groups = 11
    q = torch.randint(0, 1 << bits, (groups, 32), dtype = torch.int64)
    w = fbq._pack(q, bits)
    assert w.dtype == torch.int32 and w.shape == (groups, bits)
    for g in range(groups):
        assert _as_u32(w[g].tolist()) == _ref_pack_group(q[g].tolist(), bits), f"group {g}"
    assert torch.equal(fbq._unpack(w, bits).to(torch.int64), q)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("compand_a", [0.0, 0.65])
def test_cont_matches_kernel_transcription(bits, compand_a):
    torch.manual_seed(bits)
    groups = 7                      # not a multiple of the kernel's 4-group warp chunk
    x = torch.randn((groups, 32), dtype = torch.half, device = device)
    q = torch.empty((groups, bits), dtype = torch.int32, device = device)
    s = torch.empty((groups,), dtype = torch.half, device = device)
    ext.quant_cache_cont(x, q, s, compand_a)

    xl = x.float().tolist()
    got_codes = fbq._unpack(q.cpu(), bits).tolist()
    for g in range(groups):
        rq, rs = _ref_quant_group(xl[g], bits, compand_a)
        # Codes, not words: the layout is checked exactly above, and a value sitting on a grid
        # boundary can land either side of it under fma vs separate mul/add rounding
        assert all(abs(a - b) <= 1 for a, b in zip(got_codes[g], rq)), f"group {g} codes"
        assert abs(float(s[g]) - rs) <= 1e-3 * max(rs, 1e-3)

    deq = torch.empty((groups, 32), dtype = torch.half, device = device)
    ext.dequant_cache_cont(q, s, deq, compand_a)
    for g in range(groups):
        ref = _ref_dequant_group(_as_u32(q[g].tolist()), float(s[g]), bits, compand_a)
        got = deq[g].float().tolist()
        for a, b in zip(got, ref):
            assert abs(a - b) <= 3e-3 + 3e-3 * abs(b), f"group {g}: {a} vs {b}"


@pytest.mark.parametrize("bits", [2, 4, 5, 8])
def test_cont_roundtrip_accuracy(bits):
    torch.manual_seed(0)
    x = torch.randn((512, 128), dtype = torch.half, device = device)
    q = torch.empty((512, 128 // 32 * bits), dtype = torch.int32, device = device)
    s = torch.empty((512, 4), dtype = torch.half, device = device)
    out = torch.empty_like(x)
    ext.quant_cache_cont(x, q, s, 0.0)
    ext.dequant_cache_cont(q, s, out, 0.0)
    d = (out.float() - x.float()).abs()
    err, rms = d.max().item(), d.pow(2).mean().sqrt().item()
    # 32-value group, absmax-normalized midpoint grid: the error is a fixed multiple of the code
    # step s/2^bits, so both statistics track 1/2^bits across the whole bitrate range. The rms
    # bound is tight enough that a wrong plane order or a transposed rotation breaks it.
    assert err < 9.0 / (1 << bits), f"bits={bits} max abs err {err}"
    assert rms < 1.6 / (1 << bits), f"bits={bits} rms err {rms}"


def test_cont_head_dim_not_multiple_of_four_groups():
    # head_dim 96 -> 3 groups per row: the kernel's warp covers 4, so the tail is partial
    torch.manual_seed(1)
    x = torch.randn((37, 96), dtype = torch.half, device = device)
    q = torch.empty((37, 3 * 5), dtype = torch.int32, device = device)
    s = torch.empty((37, 3), dtype = torch.half, device = device)
    out = torch.empty_like(x)
    ext.quant_cache_cont(x, q, s, 0.0)
    ext.dequant_cache_cont(q, s, out, 0.0)
    assert (out.float() - x.float()).abs().max().item() < 9.0 / 32


def _paged_ref(k, bits, compand_a = 0.0):
    """Quantize every row of a flat (rows, dim) fp16 tensor through the cont path."""
    rows, dim = k.shape
    q = torch.empty((rows, dim // 32 * bits), dtype = torch.int32, device = k.device)
    s = torch.empty((rows, dim // 32), dtype = torch.half, device = k.device)
    ext.quant_cache_cont(k.contiguous(), q, s, compand_a)
    return q, s


@pytest.mark.parametrize("in_contiguous", [False, True])
def test_paged_quant_positions_and_partial_page(in_contiguous):
    torch.manual_seed(2)
    bits, kvh, hd = 6, 2, 96
    dim = kvh * hd
    bsz, pps, pages = 3, 3, 16
    # permuted block table: physical pages are not the logical order
    bt = torch.randperm(bsz * pps, device = device)[:bsz * pps].to(torch.int32).view(bsz, pps)
    seqlens = torch.tensor([PAGE + 5, 0, 2 * PAGE - 1], dtype = torch.int32, device = device)
    seq_len = 7

    pool = torch.zeros((pages, PAGE, kvh, hd), dtype = torch.half, device = device)
    src = torch.randn((bsz, seq_len, kvh, hd), dtype = torch.half, device = device)
    if not in_contiguous:
        for b in range(bsz):
            for t in range(seq_len):
                tok = int(seqlens[b]) + t
                pool[int(bt[b, tok // PAGE]), tok % PAGE] = src[b, t]
    k_in = src if in_contiguous else pool

    qk = torch.zeros((pages, PAGE, dim // 32 * bits), dtype = torch.int32, device = device)
    sk = torch.zeros((pages, PAGE, dim // 32), dtype = torch.half, device = device)
    ext.quant_cache_paged(k_in, qk, sk, k_in, qk.clone(), sk.clone(), seqlens, bt,
                          PAGE, seq_len, 0.0, in_contiguous)

    rq, rs = _paged_ref(src.reshape(bsz * seq_len, dim), bits)
    rq = rq.view(bsz, seq_len, -1)
    rs = rs.view(bsz, seq_len, -1)
    for b in range(bsz):
        for t in range(seq_len):
            tok = int(seqlens[b]) + t
            pos = (int(bt[b, tok // PAGE]), tok % PAGE)
            assert torch.equal(qk[pos[0], pos[1]], rq[b, t]), f"b={b} t={t}"
            assert torch.equal(sk[pos[0], pos[1]], rs[b, t])
    # nothing outside the written positions was touched
    written = {(int(bt[b, (int(seqlens[b]) + t) // PAGE]), (int(seqlens[b]) + t) % PAGE)
               for b in range(bsz) for t in range(seq_len)}
    nz = (qk != 0).any(-1).nonzero().tolist()
    assert {(p, o) for p, o in nz} <= written


@pytest.mark.parametrize("sliding_window", [-1, 300])
def test_paged_dequant_ragged_and_window(sliding_window):
    torch.manual_seed(3)
    bits, kvh, hd = 4, 1, 96
    dim = kvh * hd
    bsz, pps, pages = 3, 3, 12
    bt = torch.randperm(bsz * pps, device = device)[:bsz * pps].to(torch.int32).view(bsz, pps)
    seqlens = torch.tensor([1, PAGE + 3, 3 * PAGE], dtype = torch.int32, device = device)

    pool = torch.randn((pages, PAGE, kvh, hd), dtype = torch.half, device = device)
    qk, sk = _paged_ref(pool.reshape(pages * PAGE, dim), bits)
    qk = qk.view(pages, PAGE, -1).contiguous()
    sk = sk.view(pages, PAGE, -1).contiguous()

    out = torch.zeros((pages, PAGE, kvh, hd), dtype = torch.half, device = device)
    ext.dequant_cache_paged(qk, sk, out, qk, sk, out.clone(), seqlens, bt, PAGE,
                            sliding_window, 0.0)

    ref = torch.empty((pages * PAGE, dim), dtype = torch.half, device = device)
    ext.dequant_cache_cont(qk.view(pages * PAGE, -1), sk.view(pages * PAGE, -1), ref, 0.0)
    ref = ref.view(pages, PAGE, kvh, hd)

    lo_all = {}
    for b in range(bsz):
        n = int(seqlens[b])
        lo = max(0, n - sliding_window) if sliding_window > 0 else 0
        for tok in range(lo, n):
            p, o = int(bt[b, tok // PAGE]), tok % PAGE
            lo_all[(p, o)] = True
            torch.testing.assert_close(out[p, o], ref[p, o], atol = 0, rtol = 0)
    if sliding_window > 0:
        # outside the window the kernel may skip; it must never write garbage
        for b in range(bsz):
            n = int(seqlens[b])
            for tok in range(0, max(0, n - sliding_window)):
                p, o = int(bt[b, tok // PAGE]), tok % PAGE
                if (p, o) in lo_all:
                    continue
                assert torch.equal(out[p, o], torch.zeros_like(out[p, o])) or \
                       torch.equal(out[p, o], ref[p, o])


def test_paged_dequant_window_compact_layout():
    torch.manual_seed(4)
    bits, kvh, hd = 8, 2, 64
    dim = kvh * hd
    bsz, pps, pages = 2, 2, 8
    bt = torch.tensor([[5, 1], [3, 6]], dtype = torch.int32, device = device)
    seqlens = torch.tensor([PAGE - 4, 10], dtype = torch.int32, device = device)
    bonus = 6

    pool = torch.randn((pages, PAGE, kvh, hd), dtype = torch.half, device = device)
    qk, sk = _paged_ref(pool.reshape(pages * PAGE, dim), bits)
    qk = qk.view(pages, PAGE, -1).contiguous()
    sk = sk.view(pages, PAGE, -1).contiguous()

    scratch = torch.zeros((bsz * pps, PAGE, kvh, hd), dtype = torch.half, device = device)
    ext.dequant_cache_paged_window(qk, sk, scratch, qk, sk, scratch.clone(), seqlens, bt,
                                   PAGE, bonus, 0.0)

    ref = torch.empty((pages * PAGE, dim), dtype = torch.half, device = device)
    ext.dequant_cache_cont(qk.view(pages * PAGE, -1), sk.view(pages * PAGE, -1), ref, 0.0)
    ref = ref.view(pages, PAGE, kvh, hd)

    for b in range(bsz):
        for tok in range(int(seqlens[b]) + bonus):
            pg, off = tok // PAGE, tok % PAGE
            src = ref[int(bt[b, pg]), off]
            assert torch.equal(scratch[b * pps + pg, off], src), f"b={b} tok={tok}"


@pytest.mark.skipif(device == "cpu", reason="needs a CUDA/HIP device")
def test_paged_quant_survives_graph_capture():
    """model/graph_decode.py captures whole Attention spans, so a quantized cache's update_kv runs
    inside a graph: no host syncs, and cache_seqlens must be read on replay, not baked."""
    bits, kvh, hd = 4, 2, 64
    dim = kvh * hd
    pages = 4
    bt = torch.arange(pages, dtype = torch.int32, device = device).view(1, pages)
    seqlens = torch.zeros((1,), dtype = torch.int32, device = device)
    k = torch.randn((1, 1, kvh, hd), dtype = torch.half, device = device)
    qk = torch.zeros((pages, PAGE, dim // 32 * bits), dtype = torch.int32, device = device)
    sk = torch.zeros((pages, PAGE, dim // 32), dtype = torch.half, device = device)

    def step():
        ext.quant_cache_paged(k, qk, sk, k, qk, sk, seqlens, bt, PAGE, 1, 0.0, True)

    side = torch.cuda.Stream(device = device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream(device).wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    want, _ = _paged_ref(k.reshape(1, dim), bits)
    for pos in (17, 260, 900):
        qk.zero_()
        seqlens.fill_(pos)
        g.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(qk[pos // PAGE, pos % PAGE], want[0]), f"replay at {pos}"
        assert int((qk != 0).any(-1).sum()) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
