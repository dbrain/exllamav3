"""Pure-PyTorch fallbacks for the KV cache quantization kernels (cache/q_cache.cu).

Semantics transcribed from exllamav3_ext/cache/q_cache_kernels.cuh, whose layout is also read
directly by the Triton attention loaders in modules/attention_fn/triton_paged.py
(_qc_load_kt/_qc_load_v) -- those are the independent reference this must agree with, and
tests/test_triton_paged_hdpad.py::test_qc_hdpad checks exactly that.

Per group of 32 values along the head dim:
  y = H32 x / sqrt(32)             unnormalized Sylvester H32, then scaled
  s = max|y| + 1e-10               stored as one fp16 per group
  q = floor(y/s * m + m)           clamped to [0, 2^bits - 1], m = 2^(bits-1)
                                   (or the cubic compander of cache/lmq.cuh when compand_a > 0)
Dequant inverts with the midpoint grid (q - (m - 0.5)) / m, folds the 1/sqrt(32) into the scale
and applies the unnormalized H32 again (H32 H32 = 32 I).
"""
from __future__ import annotations

import torch

R32 = 0.17677669529663688110    # 1/sqrt(32)

# Groups per quant/dequant pass. The kernels stream; this materializes, so a whole-cache call has
# to be split or the fp32 rotation temporaries dwarf the cache itself.
_CHUNK_GROUPS = 1 << 17

_h32_cache = {}


def h32(device: torch.device) -> torch.Tensor:
    h = _h32_cache.get(device)
    if h is None:
        m = torch.ones((1, 1), dtype = torch.float32)
        while m.shape[0] < 32:
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        h = m.to(device = device).contiguous()
        _h32_cache[device] = h
    return h


def _planes(bits: int):
    # num_bits splits into power-of-two bit planes, widest first, each plane w words wide and
    # carrying bits [sh, sh + w) of every code in the group (MSB plane first).
    out = []
    rem = bits
    wb = 0
    for w in (8, 4, 2, 1):
        if bits & w:
            rem -= w
            out.append((w, wb, rem))
            wb += w
    return out


def _to_int32(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0x80000000, x - 0x100000000, x).to(torch.int32)


def _pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    # q: (M, 32) int64 codes -> (M, bits) int32. Value j of a group sits at bit w*j of its
    # plane, so word w*j//32 holds it at bit (w*j)%32 -- no field ever straddles a word.
    m = q.shape[0]
    words = q.new_zeros((m, bits))
    for w, wb, sh in _planes(bits):
        vpw = 32 // w
        sl = ((q >> sh) & ((1 << w) - 1)).view(m, w, vpw)
        shifts = torch.arange(vpw, device = q.device, dtype = q.dtype) * w
        words[:, wb:wb + w] = (sl << shifts).sum(-1)
    return _to_int32(words)


def _unpack(words: torch.Tensor, bits: int) -> torch.Tensor:
    # (M, bits) int32 -> (M, 32) int32 codes, planes recombined MSB first. torch's right shift on
    # int32 is arithmetic, but the last slot's shift is 32 - w, so the sign fill lands entirely
    # above bit w - 1 and the mask removes it -- no widening to int64 needed.
    m = words.shape[0]
    q = None
    for w, wb, _ in _planes(bits):
        vpw = 32 // w
        shifts = torch.arange(vpw, device = words.device, dtype = torch.int32) * w
        sl = ((words[:, wb:wb + w].unsqueeze(-1) >> shifts) & ((1 << w) - 1)).reshape(m, 32)
        q = sl if q is None else ((q << w) | sl)
    return q


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


def _lm_encode(x: torch.Tensor, bits: int, a: float) -> torch.Tensor:
    # Cardano root of b t^3 + a t = x, then the same floor/clamp as the linear grid
    b = 1.0 - a
    inv_b = 1.0 / b
    p3 = a * inv_b / 3.0
    q_half = x * (inv_b * 0.5)
    s = torch.sqrt(q_half * q_half + p3 * p3 * p3)
    t = _cbrt(q_half + s) + _cbrt(q_half - s)
    half_n = float(1 << (bits - 1))
    return torch.floor(t * half_n + half_n).clamp_(0, (1 << bits) - 1)


def _lm_decode(q: torch.Tensor, bits: int, a: float) -> torch.Tensor:
    t = (2.0 * q + 1.0) * (1.0 / (1 << bits)) - 1.0
    return t * (a + (1.0 - a) * t * t)


def quant_groups(x: torch.Tensor, bits: int, compand_a: float):
    """(M, 32) -> packed (M, bits) int32, scales (M,) fp32."""
    y = torch.matmul(x.float(), h32(x.device)) * R32
    s = y.abs().amax(-1, keepdim = True) + 1e-10
    t = y * (1.0 / s)
    if compand_a > 0.0:
        q = _lm_encode(t, bits, compand_a)
    else:
        m = float(1 << (bits - 1))
        q = torch.floor(t * m + m).clamp_(0, (1 << bits) - 1)
    return _pack(q.to(torch.int64), bits), s.squeeze(-1)


def dequant_groups(words: torch.Tensor, scales: torch.Tensor, bits: int, compand_a: float):
    """packed (M, bits) int32 + scales (M,) -> (M, 32) fp32."""
    q = _unpack(words, bits).to(torch.float32)
    s = scales.to(torch.float32).unsqueeze(-1) * R32
    if compand_a > 0.0:
        v = _lm_decode(q, bits, compand_a) * s
    else:
        mh = float(1 << (bits - 1)) - 0.5
        v = (q - mh) * (s * (1.0 / (1 << (bits - 1))))
    return torch.matmul(v, h32(words.device))


# -- Contiguous variants ------------------------------------------------------

def quant_cache_cont(
    in_: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor,
    compand_a: float = 0.0,
) -> None:
    dim = in_.shape[-1]
    assert dim % 32 == 0, "head_dim must be a multiple of 32"
    bits = out.shape[-1] // (dim // 32)
    x = in_.contiguous().view(-1, 32)
    wv = out.view(-1, bits)
    sv = out_scales.view(-1)
    for i in range(0, x.shape[0], _CHUNK_GROUPS):
        j = min(i + _CHUNK_GROUPS, x.shape[0])
        w, s = quant_groups(x[i:j], bits, compand_a)
        wv[i:j] = w
        sv[i:j] = s.to(sv.dtype)


def dequant_cache_cont(
    in_: torch.Tensor,
    in_scales: torch.Tensor,
    out: torch.Tensor,
    compand_a: float = 0.0,
) -> None:
    dim = out.shape[-1]
    assert dim % 32 == 0, "head_dim must be a multiple of 32"
    bits = in_.shape[-1] // (dim // 32)
    wv = in_.contiguous().view(-1, bits)
    sv = in_scales.contiguous().view(-1)
    ov = out.view(-1, 32)
    for i in range(0, wv.shape[0], _CHUNK_GROUPS):
        j = min(i + _CHUNK_GROUPS, wv.shape[0])
        ov[i:j] = dequant_groups(wv[i:j], sv[i:j], bits, compand_a).to(ov.dtype)


# -- Paged variants -----------------------------------------------------------

def _flat_dim(t: torch.Tensor) -> int:
    if t.dim() == 4:
        return t.shape[2] * t.shape[3]
    assert t.dim() == 3, "paged cache must be 3D or 4D"
    return t.shape[2]


def quant_cache_paged(
    k_in: torch.Tensor,
    k_out: torch.Tensor,
    k_out_scales: torch.Tensor,
    v_in: torch.Tensor,
    v_out: torch.Tensor,
    v_out_scales: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    seq_len: int,
    compand_a: float = 0.0,
    in_contiguous: bool = False,
) -> None:
    seq_len = int(seq_len)
    if seq_len <= 0:
        return
    dim = _flat_dim(k_in)
    gpt = dim // 32
    assert dim == 32 * gpt, "dim must be a multiple of 32"
    bsz, blocks_per_seq = block_table.shape
    dev = k_in.device

    t = torch.arange(seq_len, device = dev, dtype = torch.int64)
    tok = cache_seqlens.to(device = dev, dtype = torch.int64).reshape(bsz, 1) + t.reshape(1, -1)
    # A token past the block table's span has no physical page; the kernel would read the table
    # out of bounds, a gather raises. Those positions are outside every sequence either way.
    page = (tok // page_size).clamp_(0, blocks_per_seq - 1)
    mapped = block_table.to(torch.int64).gather(1, page)
    tok_pos = (mapped * page_size + tok % page_size).reshape(-1)
    if in_contiguous:
        in_pos = (torch.arange(bsz, device = dev, dtype = torch.int64).reshape(-1, 1) * seq_len
                  + t.reshape(1, -1)).reshape(-1)
    else:
        in_pos = tok_pos

    rows = max(1, _CHUNK_GROUPS // gpt)
    for src, wdst, sdst in ((k_in, k_out, k_out_scales), (v_in, v_out, v_out_scales)):
        bits = wdst.shape[2] // gpt
        sf = src.reshape(-1, dim)
        wf = wdst.view(-1, gpt * bits)
        sfo = sdst.view(-1, gpt)
        for i in range(0, tok_pos.numel(), rows):
            ip = in_pos[i:i + rows]
            tp = tok_pos[i:i + rows]
            x = sf.index_select(0, ip).reshape(-1, 32)
            w, s = quant_groups(x, bits, compand_a)
            wf.index_copy_(0, tp, w.reshape(ip.numel(), gpt * bits))
            sfo.index_copy_(0, tp, s.reshape(ip.numel(), gpt).to(sfo.dtype))


def _dequant_paged(
    k_in, k_in_scales, k_out, v_in, v_in_scales, v_out,
    cache_seqlens, block_table, page_size, sliding_window, compand_a,
    compact_out: bool, bonus_len: int,
) -> None:
    dim = _flat_dim(k_out)
    gpt = dim // 32
    assert dim == 32 * gpt, "dim must be a multiple of 32"
    bsz, pages_per_seq = block_table.shape
    dev = k_in.device

    lens = (cache_seqlens.to(device = dev, dtype = torch.int64).reshape(-1) + bonus_len) \
        .clamp_(0, pages_per_seq * page_size)
    if sliding_window > 0:
        lo = (lens - sliding_window).clamp_(min = 0)
    else:
        lo = torch.zeros_like(lens)
    counts = (lens - lo).clamp_(min = 0)
    total = int(counts.sum())
    if total == 0:
        return

    b_idx = torch.repeat_interleave(torch.arange(bsz, device = dev, dtype = torch.int64), counts)
    starts = torch.cumsum(counts, 0) - counts
    tok = torch.arange(total, device = dev, dtype = torch.int64) - starts[b_idx] + lo[b_idx]
    page = tok // page_size
    in_page = tok % page_size
    src_pos = block_table.to(torch.int64)[b_idx, page] * page_size + in_page
    dst_pos = ((b_idx * pages_per_seq + page) * page_size + in_page) if compact_out else src_pos

    rows = max(1, _CHUNK_GROUPS // gpt)
    for src, ssrc, dst in ((k_in, k_in_scales, k_out), (v_in, v_in_scales, v_out)):
        bits = src.shape[2] // gpt
        wf = src.view(-1, gpt * bits)
        sf = ssrc.view(-1, gpt)
        of = dst.view(-1, dim)
        for i in range(0, total, rows):
            sp = src_pos[i:i + rows]
            dp = dst_pos[i:i + rows]
            n = sp.numel()
            w = wf.index_select(0, sp).reshape(-1, bits)
            s = sf.index_select(0, sp).reshape(-1)
            y = dequant_groups(w, s, bits, compand_a).reshape(n, dim)
            of.index_copy_(0, dp, y.to(of.dtype))


def dequant_cache_paged(
    k_in: torch.Tensor,
    k_in_scales: torch.Tensor,
    k_out: torch.Tensor,
    v_in: torch.Tensor,
    v_in_scales: torch.Tensor,
    v_out: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    sliding_window: int,
    compand_a: float = 0.0,
) -> None:
    _dequant_paged(k_in, k_in_scales, k_out, v_in, v_in_scales, v_out, cache_seqlens,
                   block_table, page_size, sliding_window, compand_a, False, 0)


def dequant_cache_paged_window(
    k_in: torch.Tensor,
    k_in_scales: torch.Tensor,
    k_out: torch.Tensor,
    v_in: torch.Tensor,
    v_in_scales: torch.Tensor,
    v_out: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    bonus_len: int,
    compand_a: float = 0.0,
) -> None:
    _dequant_paged(k_in, k_in_scales, k_out, v_in, v_in_scales, v_out, cache_seqlens,
                   block_table, page_size, -1, compand_a, True, int(bonus_len))
