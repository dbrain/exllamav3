"""Grouped (multi-expert) EXL3 Triton GEMM for ROCm / gfx1150.

A standalone replacement for the CUDA-only ``ext.exl3_mgemm`` cooperative
kernel, covering the decode case (one token, ``M == 1``). Where the CUDA
kernel is unavailable, block_sparse_mlp falls back to a per-expert Python
loop of ``had_r_128 -> exl3_gemm -> had_r_128``; at 48 layers x top-k 10 x 3
linears that is ~4300 launches per token and launch overhead dominates.

At ``bsz == 1`` every routed expert sees the same input row and identical
shapes, so the routed set is a batched GEMV against E different weight
matrices. Each expert's trellis is a separate allocation, so the kernels use
the grouped-GEMM pointer-indirection pattern: an int64 device tensor of
weight base addresses, loaded per program and cast to a Triton pointer.
``MultiLinear`` already builds exactly these tables (``ptrs_trellis``,
``ptrs_suh``, ``ptrs_svh``) on every backend.

Entry points:

    exl3_mgemm_triton(x, ptrs_trellis, expert_ids, y, K_bits, cb)
        y[e] = dequant(trellis[expert_ids[e]]).T @ x[e]   (one launch)

    had_r_128_mtriton(x, y, ptrs_pre, ptrs_post, expert_ids, scale)
        grouped row Hadamard with a per-expert scale vector (one launch)

    linear_exl3_mgemm_triton(...)
        had -> mgemm -> had, three launches for the whole expert set

Device residency / CUDA-graph capture. Nothing in this module reads a device
tensor's *contents* on the host: ``expert_ids`` is dereferenced only inside the
kernels, and the grid depends solely on E (= top_k, a Python int from the
output tensor's shape) and on N. So a routed decode step is fully static and
capturable, provided the caller:
  * passes pre-allocated output buffers with stable addresses (use
    ``_linear_exl3_mgemm_triton``, which allocates nothing);
  * calls ``mgemm_prepare()`` once per shape BEFORE capture, so Triton's
    autotune benchmarking and the cached perm/mrow index tensors are done
    outside the graph;
  * writes new routing into ``expert_ids`` in place (``copy_``/``index_``) so
    the captured pointer stays valid; the replayed graph picks up the new
    experts with no host round-trip.

The M == 1 decode branch tree of the GEMM kernel is lifted verbatim from
exllamav3's ``_fused_dequant_gemm_kernel`` by ``_gen.py`` (line ranges are
listed there); only the trellis base pointer and the output row differ. Do
not hand-edit the region between the BEGIN/END markers -- rerun _gen.py.

M > 1 is NOT implemented: the grouped-tiling problem (per-expert row counts,
segment offsets) is a different kernel and the win is at decode.

One deliberate deviation from the copied source: the K_BITS==8 M1 output
permutation is fixed here (see FIXES in _gen.py). exl3_triton.py's version is
wrong for BLOCK_N > 16 -- so at 8 bpw this module does NOT match the per-expert
path; it matches ext.reconstruct instead, and the per-expert path does not.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helpers copied from exllamav3/modules/quant/exl3_triton.py
# ---------------------------------------------------------------------------

_RSCALE_128 = 0.088388347648  # 1/sqrt(128), matches the C++ literal

_M_ROW_OFFSETS = {
    3: [29, 17, 26, 14, 5, 57, 2, 54, 45, 33, 42, 94, 85, 73, 82, 70,
        23, 11, 20, 8, 63, 51, 60, 48, 39, 91, 36, 88, 79, 67, 76, 64],
    5: [27, 7, 22, 2, 51, 95, 46, 90, 75, 119, 70, 114, 99, 143, 158, 138,
        17, 61, 12, 56, 41, 85, 36, 80, 65, 109, 124, 104, 153, 133, 148, 128],
    7: [25, 61, 18, 54, 33, 69, 90, 126, 105, 141, 98, 134, 177, 213, 170, 206,
        11, 47, 4, 40, 83, 119, 76, 112, 155, 191, 148, 184, 163, 199, 220, 192],
}
_M_ROW_CACHE = {}
_PERM_CACHE = {}
_DEV_CACHE = {}


def _get_m_row_offsets(K_bits: int, device) -> torch.Tensor:
    key = (K_bits, str(device))
    if key not in _M_ROW_CACHE:
        _M_ROW_CACHE[key] = torch.tensor(
            _M_ROW_OFFSETS[K_bits], device=device, dtype=torch.int32
        )
    return _M_ROW_CACHE[key]


def _get_perm_i(device) -> torch.Tensor:
    key = str(device)
    if key not in _PERM_CACHE:
        perm = [0] * 256
        for t in range(32):
            r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
            c0 = t // 4; c1 = c0 + 8
            perm[t * 8 + 0] = r0 * 16 + c0; perm[t * 8 + 1] = r1 * 16 + c0
            perm[t * 8 + 2] = r2 * 16 + c0; perm[t * 8 + 3] = r3 * 16 + c0
            perm[t * 8 + 4] = r0 * 16 + c1; perm[t * 8 + 5] = r1 * 16 + c1
            perm[t * 8 + 6] = r2 * 16 + c1; perm[t * 8 + 7] = r3 * 16 + c1
        perm_i = [0] * 256
        for i, p in enumerate(perm):
            perm_i[p] = i
        _PERM_CACHE[key] = torch.tensor(perm_i, device=device, dtype=torch.long)
    return _PERM_CACHE[key]


def _dev_caps(device=None):
    idx = torch.cuda.current_device() if device is None else device
    if idx not in _DEV_CACHE:
        p = torch.cuda.get_device_properties(idx)
        is_amd = torch.version.hip is not None
        wave = getattr(p, "warp_size", None) or (64 if is_amd else 32)
        _DEV_CACHE[idx] = (p.multi_processor_count, wave, is_amd)
    return _DEV_CACHE[idx]


@triton.jit
def _decode_u16(w_u32, CB: tl.constexpr):
    """Inline arithmetic decode of 16-bit codebook indices (matches
    decode_3inst in the C++ reference)."""
    if CB == 0:
        w_u32 = (w_u32 * 89226354 + 64248484) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    elif CB == 1:
        w_u32 = (w_u32 * 0xCBAC1FED) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    else:  # CB == 2 (mul1)
        w_u32 = (w_u32 * 0x83DCD12D) & 0xFFFFFFFF
        db0 = w_u32 & 0xFF
        db1 = (w_u32 >> 8) & 0xFF
        db2 = (w_u32 >> 16) & 0xFF
        db3 = (w_u32 >> 24) & 0xFF
        w_u32 = (db0 + db1 + db2 + db3 + 0x6400) & 0xFFFF

    if CB == 0 or CB == 1:
        lo = w_u32 & 0xFFFF
        hi = (w_u32 >> 16) & 0xFFFF
        lo_h = tl.cast(lo.to(tl.int16), tl.float16, bitcast=True)
        hi_h = tl.cast(hi.to(tl.int16), tl.float16, bitcast=True)
        return lo_h + hi_h
    else:
        sum16 = w_u32 & 0xFFFF
        h = tl.cast(sum16.to(tl.int16), tl.float16, bitcast=True)
        k_inv_h = tl.full((1,), 0x1eee, dtype=tl.int16)
        k_inv_h = tl.cast(k_inv_h, tl.float16, bitcast=True)
        k_bias_h = tl.full((1,), 0xc931, dtype=tl.int16)
        k_bias_h = tl.cast(k_bias_h, tl.float16, bitcast=True)
        return h * k_inv_h + k_bias_h


@triton.jit
def _decode_word_pair(low_u32, high_u32, shift,
                      SHIFT_FITS_32: tl.constexpr, CB: tl.constexpr):
    if SHIFT_FITS_32:
        neg_shift = tl.minimum(32 - shift, 31)
        windows = ((low_u32 >> shift) | (high_u32 << neg_shift)) & 0xFFFF
    else:
        low64 = (low_u32.to(tl.int64) & 0xFFFFFFFF) | ((high_u32.to(tl.int64) & 0xFFFFFFFF) << 32)
        windows = ((low64 >> shift) & 0xFFFF).to(tl.uint32)
    return _decode_u16(windows.to(tl.uint32), CB)


@triton.jit
def _funnel6(lo, hi, s):
    sel = s >= 32
    s32 = s & 31
    ns = tl.minimum(32 - s32, 31)
    base = tl.where(sel[:, None, None], hi[None, :, :], lo[None, :, :])
    second = tl.where(sel[:, None, None], lo[None, :, :], hi[None, :, :])
    return ((base >> s32[:, None, None]) | (second << ns[:, None, None])) & 0xFFFF


@triton.jit
def _had_stage(v, BLOCK_R: tl.constexpr, SPAN: tl.constexpr):
    G: tl.constexpr = 128 // (2 * SPAN)
    pair = tl.permute(v.reshape(BLOCK_R, G, 2, SPAN), (0, 1, 3, 2))
    lo, hi = tl.split(pair)
    pair = tl.join(lo + hi, lo - hi)
    return tl.permute(pair, (0, 1, 3, 2)).reshape(BLOCK_R, 128)


# ---------------------------------------------------------------------------
# Grouped row Hadamard: one launch for the whole routed expert set
#
# Bit-identical to E separate had_r_128_triton calls: the butterfly, the fp32
# evaluation and the round-to-half ordering are the same expression tree; only
# the scale vector pointer is fetched indirectly, per expert.
# ---------------------------------------------------------------------------

@triton.jit
def _grouped_had_r_128_kernel(
    x_ptr, y_ptr, ptrs_scale, expert_ids_ptr,
    n_rows,
    stride_xe, stride_xr, stride_ye, stride_yr,
    r_scale,
    IO_FP32: tl.constexpr,
    PRE_SCALED: tl.constexpr,
    POST_SCALED: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    eid = tl.load(expert_ids_ptr + pid_e)
    s_ptr = tl.cast(tl.load(ptrs_scale + eid), tl.pointer_type(tl.float16))

    xb = x_ptr + pid_e * stride_xe
    yb = y_ptr + pid_e * stride_ye

    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = rows < n_rows
    col = tl.arange(0, 128)

    x = tl.load(
        xb + rows[:, None] * stride_xr + (pid_c * 128 + col)[None, :],
        mask=mask_r[:, None], other=0.0,
    )

    if PRE_SCALED:
        pre = tl.load(s_ptr + pid_c * 128 + col)
        if IO_FP32:
            x = x * pre.to(tl.float32)
        else:
            x = x * pre

    v = x.to(tl.float32)
    v = _had_stage(v, BLOCK_R, 1)
    v = _had_stage(v, BLOCK_R, 2)
    v = _had_stage(v, BLOCK_R, 4)
    v = _had_stage(v, BLOCK_R, 8)
    v = _had_stage(v, BLOCK_R, 16)
    v = _had_stage(v, BLOCK_R, 32)
    v = _had_stage(v, BLOCK_R, 64)
    v = v * r_scale

    if POST_SCALED:
        post = tl.load(s_ptr + pid_c * 128 + col)
        if IO_FP32:
            out = v * post.to(tl.float32)
        else:
            out = v.to(x_ptr.dtype.element_ty) * post
    else:
        out = v

    tl.store(
        yb + rows[:, None] * stride_yr + (pid_c * 128 + col)[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=mask_r[:, None],
    )


def had_r_128_mtriton(
    input: torch.Tensor,
    output: torch.Tensor,
    ptrs_pre_scale: torch.Tensor | None,
    ptrs_post_scale: torch.Tensor | None,
    expert_ids: torch.Tensor,
    scale: float = 1.0,
) -> None:
    """Grouped y[e] = (x[e].view(-1, 128) @ H128) * scale_vec[expert_ids[e]].

    ``input`` is [E, rows, cols] or [rows, cols] broadcast across experts (any
    tensor whose expert stride is 0, e.g. ``x.unsqueeze(0).expand(E, -1, -1)``).
    ``output`` is [E, rows, cols]. Scale pointer tables are int64 device
    tensors of ``half`` vector base addresses, as built by MultiLinear.
    """
    assert input.dtype == output.dtype
    assert input.dtype in (torch.half, torch.float)
    assert (ptrs_pre_scale is None) or (ptrs_post_scale is None)
    assert output.dim() == 3 and output.shape[-1] % 128 == 0
    assert input.stride(-1) == 1 and output.stride(-1) == 1

    E, rows, cols = output.shape
    if input.dim() == 2:
        input = input.unsqueeze(0).expand(E, -1, -1)

    BLOCK_R = 4
    grid = (E, triton.cdiv(rows, BLOCK_R), cols // 128)
    _grouped_had_r_128_kernel[grid](
        input, output,
        ptrs_pre_scale if ptrs_pre_scale is not None else ptrs_post_scale,
        expert_ids,
        rows,
        input.stride(0), input.stride(1), output.stride(0), output.stride(1),
        scale * _RSCALE_128,
        IO_FP32=input.dtype == torch.float,
        PRE_SCALED=ptrs_pre_scale is not None,
        POST_SCALED=ptrs_post_scale is not None,
        BLOCK_R=BLOCK_R,
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Autotune pool
#
# Only the M == 1 tiles are relevant. The grouped grid is E x cdiv(N, BLOCK_N)
# programs, i.e. E times the single-expert grid, so the CTA-starvation that
# forces narrow tiles on the per-expert path is much weaker here; the pool
# keeps both the narrow and the wide tiles and lets autotune decide.
#
# _prefer_warps mirrors the repo: on parts with few CUs (gfx1150 has 8) narrow
# CTAs keep more blocks resident. RDNA runs wave32, so num_warps counts 32-lane
# waves exactly like CUDA -- this is not a wave64 lane-count correction.
# ---------------------------------------------------------------------------

def _prefer_warps(configs):
    cu, wave, is_amd = _dev_caps()
    if cu > 16:
        return configs
    want = [c for c in configs if c.num_warps <= 4]
    return want if want else configs


def _mgemm_configs():
    spec = os.environ.get("TMGEMM_CONFIGS")
    if spec:
        # "BN,BK:nw:ns;..." for manual sweeps
        out = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            dims, _, rest = part.partition(":")
            bn, bk = (int(v) for v in dims.split(","))
            nw = int(rest.split(":")[0]) if rest else 2
            ns = int(rest.split(":")[1]) if ":" in rest else 3
            out.append(triton.Config({"BLOCK_N": bn, "BLOCK_K": bk},
                                     num_warps=nw, num_stages=ns))
        return out
    return [
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        # generic-path / odd-shape fallback, never pruned away
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
    ]


def _mgemm_prune(configs, named_args, **kwargs):
    bits = kwargs.get("K_BITS", named_args.get("K_BITS"))
    n = kwargs.get("N", named_args.get("N"))
    k = kwargs.get("K_dim", named_args.get("K_dim"))
    fast_ok = n % 128 == 0 and k % 128 == 0

    out = [c for c in configs
           if n % c.kwargs["BLOCK_N"] == 0 and k % c.kwargs["BLOCK_K"] == 0]
    if not out:
        out = list(configs)

    if fast_ok:
        # Every width has a gather-free fast path for full tiles; drop the
        # tiny generic fallback so it can't win a cold-clock autotune pass.
        wide = [c for c in out if c.kwargs["BLOCK_N"] >= 32]
        out = wide or out
        if bits != 4:
            # Mirror the single-expert pool's BLOCK_N ceiling per width class:
            # the heavy M1 accumulators (2-4 fp32 tensors of 256*NN elements per
            # CTA) collapse occupancy at wide BLOCK_N, and those are the tiles
            # the repo actually measured for these widths.
            lim = 32 if n <= 16384 else 64
            narrow = [c for c in out if c.kwargs["BLOCK_N"] <= lim]
            out = narrow or out
        if bits == 3:
            narrow = [c for c in out if c.kwargs["BLOCK_N"] <= 32]
            out = narrow or out
    else:
        # Generic tl.gather path: this Triton build's LLVM aborts on large
        # gather tiles.
        small = [c for c in out
                 if c.kwargs["BLOCK_N"] <= 64 and c.kwargs["BLOCK_K"] <= 64]
        out = small or out

    return _prefer_warps(out) or configs


_PRUNE = {"early_config_prune": _mgemm_prune}


# ---------------------------------------------------------------------------
# Grouped fused dequant + GEMV kernel (M == 1)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_mgemm_configs(),
    key=["E_BUCKET", "N", "K_dim", "K_BITS", "N_PACKED", "CB"],
    prune_configs_by=_PRUNE,
)
@triton.jit
def _grouped_dequant_gemv_kernel(
    x_ptr, y_ptr,
    ptrs_trellis,          # [num_experts] int64, trellis base addresses
    expert_ids_ptr,        # [E] int32/int64, indexes ptrs_trellis
    perm_i_ptr,
    mrow_ptr,
    E, N, K_dim,
    E_BUCKET,              # autotune key only; unused in the body
    stride_xe, stride_xk,
    stride_tk, stride_tn,
    stride_ye, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_BITS: tl.constexpr,
    N_PACKED: tl.constexpr,
    CB: tl.constexpr,
):
    NK: tl.constexpr = BLOCK_K // 16   # k-sub-tiles per weight tile
    NN: tl.constexpr = BLOCK_N // 16   # n-sub-tiles per weight tile
    N_U32: tl.constexpr = K_BITS * 256 // 32
    SHIFT_FITS_32: tl.constexpr = (K_BITS == 1) | (K_BITS == 2) | (K_BITS == 4)

    # Grid is flat over (expert, n-tile) so the whole routed set is one launch.
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_e = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Grouped-GEMM pointer indirection: this program's expert selects a base
    # address out of the int64 table, which becomes a Triton pointer. The
    # tensors are separate allocations, so no single strided view spans them.
    eid = tl.load(expert_ids_ptr + pid_e)
    tbase = tl.load(ptrs_trellis + eid)
    tu32_ptr = tl.cast(tbase, tl.pointer_type(tl.uint32))

    # x rows may be shared across experts (stride_xe == 0 for a broadcast view)
    x_ptr = x_ptr + pid_e * stride_xe
    y_ptr = y_ptr + pid_e * stride_ye

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    stride_tk_u32 = stride_tk // 2
    stride_tn_u32 = stride_tn // 2
    base_n = (pid_n * NN) * stride_tn_u32

    n_k_tiles_total = K_dim // 16
    k_base = 0
    n_outer = tl.cdiv(n_k_tiles_total, NK)

    # --- BEGIN generated by _gen.py from exl3_triton.py (sha256[:16] 37356b4fa0f3fbd8) ---
    if K_BITS == 4 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=4 fast path (full tiles only): coalesced staging + gather-free
        # algebraic decode.
        #
        # Staging: the packed words of all NN sub-tile columns of one k-tile
        # are contiguous, so two linear u32 loads (the row and the same row
        # shifted one word back) fetch everything dword-vectorized. The m1
        # row is wrapped within each sub-tile in registers (word -1 == word
        # 31), so no rotated/global scattered loads are ever issued.
        #
        # Decode: for sub-tile element (r, c) the codebook index comes from
        # trellis word pair (t-1, t) at shift s where
        #   t(r, c) = 4*(c%8) + (r%8)//2,   s(r, c) = 28 - 4*j(r, c),
        #   j(r, c) = 4*(c//8) + 2*(r//8) + (r%2),
        # a bijection (r, c) <-> (j, t) verified against _get_perm /
        # _dq_indices. Decoding the [8j, NN*32t] table of every (shift, word)
        # pair computes each weight exactly once; the permutation back to
        # (r, c) order is pure axis algebra.
        # ------------------------------------------------------------------
        j8 = tl.arange(0, 8)
        sh = 28 - 4 * j8                       # funnel shift per j row
        neg_sh = tl.minimum(32 - sh, 31)       # neighbor shift, masked to 0
        wc = tl.arange(0, NN * 32)             # staged word row
        nj8 = tl.arange(0, NN)

        # Decode path: pure GEMV reduction in fp32. The permuted weight
        # tile is never materialized: because (r, c) -> (j, t) is a
        # bijection, sum_r x[r] * W[r, c] == sum_{(j,t): c(j,t)=c}
        # Q[j,t] * X[j,t] with X[j, t] = x[r(j, t)] built from the 16 x
        # values by pure reshape/broadcast over the axis algebra
        #   j = 4*ch + 2*rh + p,  t = 4*cl + q,  r = 8*rh + 2*q + p,
        #   c = 16*nj + 8*ch + cl,
        # so the whole permutation lives in X's layout — free.
        #
        # The [ch, rh, p, nj, cl, q]-shaped product tile is accumulated
        # elementwise across the whole K loop (no cross-lane traffic per
        # iteration); the reduction over (rh, p, q) happens once at the
        # end. The m1 wrap word (t == 0 needs word 31) is a tiny [NN]
        # load instead of a per-subtile reduction.
        r16 = tl.arange(0, 16)
        acc6 = tl.zeros((2, 2, 2, NN, 8, 4), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_base + k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n
                words = tl.load(row + wc)                          # [NN*32]
                safe = (ktb > 0) | (base_n > 0)
                m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                w31 = tl.load(row + nj8 * 32 + 31)                # [NN]
                w31_bcast = tl.reshape(
                    tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
                )
                m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
                q = ((words[None, :] >> sh[:, None]) |
                     (m1[None, :] << neg_sh[:, None])) & 0xFFFF    # [8, NN*32]
                w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                # X over (rh, p, q): r = 8*rh + 2*q + p
                xpat = tl.permute(tl.reshape(xk, (2, 4, 2)), (0, 2, 1))
                xb6 = tl.broadcast_to(
                    tl.reshape(xpat, (1, 2, 2, 1, 1, 4)), (2, 2, 2, NN, 8, 4)
                )
                acc6 += tl.reshape(w_dec, (2, 2, 2, NN, 8, 4)) * xb6
        s = tl.sum(acc6, 5)      # q    -> (ch, rh, p, nj, cl)
        s = tl.sum(s, 2)         # p    -> (ch, rh, nj, cl)
        s = tl.sum(s, 1)         # rh   -> (ch, nj, cl)
        acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
        tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
    elif K_BITS == 6 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=6 fast path (full tiles only): gather-free algebraic decode,
        # twin of the bits=4 path. Verified against _dq_indices/_get_perm:
        #
        # e = 4*tg + jj,  tg = 4a + b      (jj = e%4, b = (e//4)%4, a = e//16)
        # code(e) = funnel(word(u), word(u-1), s) with
        #   u(e) = 3a + f(b),  f = [0,1,2,2]   (word index within the 48-word
        #                                      tile; u-1 wraps mod 48)
        #   s(e) = C_b - 6*jj,  C = [26, 34, 42, 18]
        # target position of e under the _get_perm permutation (verified
        # bijective bit-field assignment; e's bits are 32*cl + 16*(a&1) +
        # 8*(b>>1) + 4*(b&1) + 2*j1 + j0):
        #   r = 8*j1 + 4*(a&1) + 2*(b>>1) + j0
        #   c = 8*(b&1) + (a>>1)
        #
        # Only four linear word-slice loads are needed (all contiguous over
        # (nj, a), so everything stays coalesced, no tl.gather):
        #   b=0: (word 3a,   word 3a-1)  shift 26-6jj
        #   b=1: (word 3a+1, word 3a)    shift 34-6jj
        #   b=2: (word 3a+2, word 3a+1)  shift 42-6jj
        #   b=3: (word 3a+2, word 3a+1)  shift 18-6jj   (same words as b=2)
        # ------------------------------------------------------------------
        a16 = tl.arange(0, 16)
        nj8 = tl.arange(0, NN)
        j8 = tl.arange(0, 4)
        # word-slice addresses relative to the subtile base (mod 48 in-tile)
        wbase = tl.reshape(nj8[:, None] * 48 + 3 * a16[None, :], (NN * 16,))       # word 3a
        wone = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 1) % 48, (NN * 16,))  # 3a+1
        wtwo = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 2) % 48, (NN * 16,))  # 3a+2
        wneg = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 47) % 48, (NN * 16,)) # 3a-1
        # per-b constant shifts for the 4 jj rows
        C0 = tl.full((4,), 26, tl.int32); C1 = tl.full((4,), 34, tl.int32)
        C2 = tl.full((4,), 42, tl.int32); C3 = tl.full((4,), 18, tl.int32)
        sh6 = 6 * j8

        # GEMV: fold the permutation into the x broadcast. With
        # r = 8*j1 + 4*a0 + 2*b1 + j0, the (j1,j0,a0,b1)-indexed x
        # pattern comes from a reshape + permute + split of the 16
        # values; the b=0/1 decodes multiply x[..., b1=0], b=2/3 the
        # b1=1 half. Decodes reshape to (j1, j0, nj, cA, a0) since the
        # a axis factors as a = 8*cA + a0 (a0 = a&1, cA = a>>1 = c%8).
        r16 = tl.arange(0, 16)
        acc0 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
        acc1 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
        acc2 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
        acc3 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_base + k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n
                words2 = tl.reshape(tl.load(row + wbase), (NN, 16))
                wone2 = tl.reshape(tl.load(row + wone), (NN, 16))
                wtwo2 = tl.reshape(tl.load(row + wtwo), (NN, 16))
                wneg2 = tl.reshape(tl.load(row + wneg), (NN, 16))
                d0 = _decode_u16(_funnel6(words2, wneg2, C0 - sh6), CB).to(tl.float32)
                d1 = _decode_u16(_funnel6(wone2, words2, C1 - sh6), CB).to(tl.float32)
                d2 = _decode_u16(_funnel6(wtwo2, wone2, C2 - sh6), CB).to(tl.float32)
                d3 = _decode_u16(_funnel6(wtwo2, wone2, C3 - sh6), CB).to(tl.float32)
                xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                # r = 8*j1 + 4*a0 + 2*b1 + j0  =>  (r3,r2,r1,r0)=(j1,a0,b1,j0)
                xr = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (0, 3, 1, 2))
                x_lo, x_hi = tl.split(xr)
                x_lo = tl.broadcast_to(tl.reshape(x_lo, (2, 2, 1, 1, 2)), (2, 2, NN, 8, 2))
                x_hi = tl.broadcast_to(tl.reshape(x_hi, (2, 2, 1, 1, 2)), (2, 2, NN, 8, 2))
                acc0 += tl.reshape(d0, (2, 2, NN, 8, 2)) * x_lo
                acc1 += tl.reshape(d1, (2, 2, NN, 8, 2)) * x_lo
                acc2 += tl.reshape(d2, (2, 2, NN, 8, 2)) * x_hi
                acc3 += tl.reshape(d3, (2, 2, NN, 8, 2)) * x_hi
        # reduce over (j1, j0, a0); leaves (nj, cA) per b; output
        # n = 16*nj + 8*b0 + cA with b0 = b&1 (b=0,2 -> 0; b=1,3 -> 1)
        s0 = tl.sum(tl.sum(tl.sum(acc0, 0), 0), 2)
        s1 = tl.sum(tl.sum(tl.sum(acc1, 0), 0), 2)
        s2v = tl.sum(tl.sum(tl.sum(acc2, 0), 0), 2)
        s3 = tl.sum(tl.sum(tl.sum(acc3, 0), 0), 2)
        h0 = s0 + s2v
        h1 = s1 + s3
        out = tl.permute(tl.join(h0, h1), (0, 2, 1))
        tl.store(y_ptr + offs_n * stride_yn, tl.reshape(out, (BLOCK_N,)).to(y_ptr.dtype.element_ty), mask=mask_n)
    elif (K_BITS == 1 or K_BITS == 2 or K_BITS == 8) and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # Power-of-two widths (K = 1, 2, 8): same gather-free structure as
        # the bits=4 path, generalized. The (r, c) -> (word, shift) lookup is
        # a pure bit-field map: with r = 8*r3 + 4*r2 + 2*r1 + r0 and
        # c = 8*c3 + cl (cl = c%8), the 5 bits (r2, r1, c3, r3, r0) split —
        # the first log2(K_BITS) of them pack into the sub-tile word index
        #   word = K_BITS*cl + g,      g = (r2, r1, c3)[:log2(K)] packed MSB-first
        # and the remaining bits form the shift row
        #   row  = the (5 - log2(K)) remaining bits, MSB-first
        #   shift(row) = 32 - K_BITS - K_BITS*row
        # (verified element-exactly against the C++ reconstruct kernel; the
        # K_BITS == 4 case is the branch above). Every (row, word) pair is one
        # sub-tile element exactly once, so like bits=4 the packed row loads
        # linear and the permutation is realized by static reshapes; the m1
        # neighbor word (needed when shift > 16) is the row shifted one word
        # back, wrapped inside the sub-tile in registers.
        # ------------------------------------------------------------------
        ROWS: tl.constexpr = 32 // K_BITS
        rows = tl.arange(0, ROWS)
        sh = (32 - K_BITS) - K_BITS * rows
        neg_sh = tl.minimum(32 - sh, 31)
        wc = tl.arange(0, NN * N_U32)

        # GEMV: fold the permutation into the x broadcast (see bits=4).
        r16 = tl.arange(0, 16)
        if K_BITS == 1:
            acc7 = tl.zeros((2, 2, 2, 2, 2, NN, 8), dtype=tl.float32)  # (r2,r1,c3,r3,r0,nj,cl)
        elif K_BITS == 2:
            acc7 = tl.zeros((2, 2, 2, 2, NN, 8, 2), dtype=tl.float32)  # (r1,c3,r3,r0,nj,cl,r2)
        else:
            acc7 = tl.zeros((2, 2, NN, 8, 2, 2, 2), dtype=tl.float32)  # (r3,r0,nj,cl,r2,r1,c3)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n
                words = tl.load(row + wc)
                safe = (ktb > 0) | (base_n > 0)
                m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                wlast = tl.load(row + (wc // N_U32) * N_U32 + (N_U32 - 1))
                m1 = tl.where((wc % N_U32) == 0, wlast, m1_lin)
                q = ((words[None, :] >> sh[:, None]) |
                     (m1[None, :] << neg_sh[:, None])) & 0xFFFF     # [ROWS, NN*N]
                w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                if K_BITS == 1:
                    # r = 8*r3 + 4*r2 + 2*r1 + r0
                    xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (1, 2, 0, 3))
                    xb = tl.broadcast_to(
                        tl.reshape(xpat, (2, 2, 1, 2, 2, 1, 1)), (2, 2, 2, 2, 2, NN, 8)
                    )
                    acc7 += tl.reshape(w_dec, (2, 2, 2, 2, 2, NN, 8)) * xb
                elif K_BITS == 2:
                    xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (2, 0, 3, 1))
                    xb = tl.broadcast_to(
                        tl.reshape(xpat, (2, 1, 2, 2, 1, 1, 2)), (2, 2, 2, 2, NN, 8, 2)
                    )
                    acc7 += tl.reshape(w_dec, (2, 2, 2, 2, NN, 8, 2)) * xb
                else:
                    xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (0, 3, 1, 2))
                    xb = tl.broadcast_to(
                        tl.reshape(xpat, (2, 2, 1, 1, 2, 2, 1)), (2, 2, NN, 8, 2, 2, 2)
                    )
                    acc7 += tl.reshape(w_dec, (2, 2, NN, 8, 2, 2, 2)) * xb
        if K_BITS == 1:
            s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 0), 1), 1)     # -> (c3, nj, cl)
        elif K_BITS == 2:
            s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 1), 1), 3)     # -> (c3, nj, cl)
        else:
            s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 0), 2), 2)     # -> (nj, cl, c3)
            # the shared tail indexes n = 16*nj + 8*c3 + cl, so it wants
            # (c3, nj, cl); (0, 2, 1) is only right when NN == 1
            s = tl.permute(s, (2, 0, 1))
        acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))       # n = 16*nj + 8*c3 + cl
        tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
    elif K_BITS == 3 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=3 fast path (full tiles only). The D-table rows regroup into 8
        # run-groups g = 2v + c3 (rows r = 2v + 8q + p, m = 2q + p) whose four
        # 16-bit windows are consecutive 3-bit steps of ONE 32-bit funnel
        # Q_g = (W[a_g] >> b_g) | (W[a_g - 1] << (32 - b_g)) of subtile words
        # (word indices mod 24; b_g = (84 - 12*g) % 32, a_g = g // 4),
        # window(g, m) = (Q_g >> (9 - 3*m)) & 0xFFFF. Verified bit-exact
        # against the D-table decode. Two strided u32 slice loads (the g/4
        # word and its -1 neighbor, stride 3 over the 24-word subtile) feed
        # all eight funnels, so each weight costs one load-lane instead of
        # two and one funnel instead of four.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        g8 = tl.arange(0, 8)
        col = tl.arange(0, NN * 8)
        njc = col // 8
        clc = col % 8
        base_n3 = (pid_n * NN) * stride_tn_u32

        base_g = (84 - 12 * g8) % 32
        neg_g = tl.minimum(32 - base_g, 31)
        a_g = 2 - (84 - 12 * g8) // 32
        w_a = njc[None, :] * N_U32 + 3 * clc[None, :] + a_g[:, None]
        w_b = njc[None, :] * N_U32 + (3 * clc[None, :] + a_g[:, None] + N_U32 - 1) % N_U32

        # acc[g, nj*8 + clc]; each of the 4 rows of a group accumulates
        # into the shared (g, column) slot with its own x element.
        acc8 = tl.zeros((8, NN * 8), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n3
                A = tl.load(row + w_a)
                B = tl.load(row + w_b)
                # no 32-bit mask on Q: the extraction masks drop every bit
                # above 24, including the bit-31 pollution B<<31 at base 0
                Q = (A >> base_g[:, None]) | (B << neg_g[:, None])
                xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                # X_m[g] = xk[2*(g//2) + 8*(m//2) + m%2]: pairs (e, o) of
                # the rows xk[2i+p], then halves v<4 / v>=4, interleaved
                # over c3 by the final (4, 2) -> 8 broadcast.
                e, o = tl.split(tl.reshape(xk, (8, 2)))
                e_lo, e_hi = tl.split(tl.permute(tl.reshape(e, (2, 4)), (1, 0)))
                o_lo, o_hi = tl.split(tl.permute(tl.reshape(o, (2, 4)), (1, 0)))
                x_m0 = tl.broadcast_to(tl.reshape(tl.broadcast_to(e_lo[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                x_m1 = tl.broadcast_to(tl.reshape(tl.broadcast_to(o_lo[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                x_m2 = tl.broadcast_to(tl.reshape(tl.broadcast_to(e_hi[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                x_m3 = tl.broadcast_to(tl.reshape(tl.broadcast_to(o_hi[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                acc8 += (
                    _decode_u16(((Q >> 9) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m0
                    + _decode_u16(((Q >> 6) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m1
                    + _decode_u16(((Q >> 3) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m2
                    + _decode_u16((Q & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m3
                )
        # (v, c3, nj, clc) -> sum v -> n = 16*nj + 8*c3 + clc
        s = tl.sum(tl.reshape(acc8, (4, 2, NN, 8)), 0)   # (c3, nj, clc)
        acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
        tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
    elif (K_BITS == 5 or K_BITS == 7) and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # Odd widths (K = 3, 5, 7): the (word, shift) lookup does not factor
        # into independent per-axis bit fields (the 16-bit decode window ends
        # inside a code, so word index and shift carry into each other).
        # Instead, each of the 32 (r, c3) rows of a sub-tile has ONE fixed
        # window offset D = 32*f + sh (see _M_ROW_OFFSETS): element (r, c)
        # reads word K_BITS*(c%8) + f(r, c//8) at funnel shift sh(r, c//8),
        # neighbor word -1 when sh > 16. The decode tile is [32 rows,
        # NN*8 (c%8)] with affine word addresses (stride K_BITS in the
        # column axis, constant row offset), so there is still no
        # data-dependent gather; every word of the sub-tile is used exactly
        # once per f-slice.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        mrow = tl.arange(0, 32)                     # row = 2*r + c3
        D_vec = tl.load(mrow_ptr + mrow)
        f_vec = D_vec // 32
        sh_vec = D_vec % 32
        neg_vec = tl.minimum(32 - sh_vec, 31)
        col = tl.arange(0, NN * 8)
        njc = col // 8
        clc = col % 8
        w_lo = njc[None, :] * N_U32 + K_BITS * clc[None, :] + f_vec[:, None]
        t_hi = K_BITS * clc[None, :] + f_vec[:, None] - 1
        w_hi = njc[None, :] * N_U32 + (t_hi + N_U32) % N_U32

        accm = tl.zeros((32, NN * 8), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n
                lo = tl.load(row + w_lo)                       # [32, NN*8]
                hi = tl.load(row + w_hi)
                q = ((lo >> sh_vec[:, None]) |
                     (hi << neg_vec[:, None])) & 0xFFFF
                w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                xb = tl.reshape(
                    tl.broadcast_to(tl.reshape(xk, (16, 1, 1, 1)), (16, 2, NN, 8)),
                    (32, NN * 8),
                )
                accm += w_dec * xb
        # (r, c3, nj, cl) -> sum over r -> (c3, nj, cl) -> n = 16*nj + 8*c3 + cl
        s = tl.sum(tl.reshape(accm, (16, 2, NN, 8)), 0)
        acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
        tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
    else:
        # ------------------------------------------------------------------
        # Generic path (other bit widths / non-full tiles): staged row load +
        # tl.gather decode. The packed words for all NN sub-tile columns of
        # one k-sub-tile are contiguous, so one u32 load fetches them
        # coalesced; per-element decode words are then gathered from the
        # staged row via shared memory instead of scattered global scalars.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        n_idx = tl.arange(0, BLOCK_N)
        elem_flat = r16[:, None] * 16 + (n_idx % 16)[None, :]  # [16, BLOCK_N]
        elem_idx = tl.load(perm_i_ptr + elem_flat).to(tl.int32)

        if K_BITS == 4:
            lane = elem_idx // 8
            r = elem_idx % 8
            word_low_idx = lane
            word_high_idx = (lane + 31) % 32
            shift = (7 - r) * 4
        elif K_BITS == 2:
            q16 = elem_idx // 16
            i1 = q16
            i0 = (i1 + 15) % 16
            r = elem_idx % 8
            shift0 = ((~(elem_idx // 8 * 8)) & 8) * 2
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r) * 2
        elif K_BITS == 1:
            q32 = elem_idx // 32
            i1 = q32
            i0 = (i1 + 7) % 8
            r = elem_idx % 8
            shift0 = (~(elem_idx // 8 * 8)) & 24
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r)
        elif K_BITS == 3:
            t_offset = elem_idx // 8 * 8
            r = elem_idx % 8
            b1 = (t_offset + 257) * K_BITS
            b0 = b1 - 16
            b2 = b1 + K_BITS * 7
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % N_U32
            word_high_idx = i0 % N_U32
            shift = s2 + (7 - r) * K_BITS
        elif K_BITS == 7:
            # dq2x2 widths: the C++ decode pairs consecutive codes across the
            # word boundary, so the per-element window does not follow the
            # t_offset/j algebra of the dq4 widths. Use the verified per-row
            # window offsets (same tables as the odd-width fast path).
            row = r16[:, None] * 2 + ((n_idx[None, :] % 16) // 8)
            d = tl.load(mrow_ptr + row)
            word_low_idx = K_BITS * (n_idx[None, :] % 8) + d // 32
            word_high_idx = (word_low_idx - 1 + N_U32) % N_U32
            shift = d % 32
        else:
            t = (elem_idx // 4) * 4
            j = elem_idx % 4
            b0 = (t + 257) * K_BITS - 16
            b2 = (t + 260) * K_BITS
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % N_U32
            word_high_idx = i0 % N_U32
            shift = s2 + (3 - j) * K_BITS

        # Gather indices into the staged row: sub-tile nj occupies words
        # [nj*N_U32, (nj+1)*N_U32).
        tile_off = (n_idx // 16) * N_U32
        idx_low = word_low_idx + tile_off[None, :]
        idx_high = word_high_idx + tile_off[None, :]

        WCOLS: tl.constexpr = NN * N_U32
        WCOLS_P2: tl.constexpr = triton.next_power_of_2(WCOLS)
        wcols = tl.arange(0, WCOLS_P2)
        tiles_n = N // 16
        n_words_valid = min(WCOLS, max(tiles_n - pid_n * NN, 0) * N_U32)
        wmask = wcols < n_words_valid

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_outer * NK + ki
                k_ok = ktb < n_k_tiles_total
                words = tl.load(
                    tu32_ptr + ktb * stride_tk_u32 + base_n + wcols,
                    mask=wmask & k_ok, other=0,
                )
                src = tl.broadcast_to(words[None, :], (16, WCOLS_P2))
                low_u32 = tl.gather(src, idx_low, 1)
                high_u32 = tl.gather(src, idx_high, 1)
                w = _decode_word_pair(low_u32, high_u32, shift, SHIFT_FITS_32, CB)
                xk = tl.load(
                    x_ptr + (ktb * 16 + r16) * stride_xk,
                    mask=k_ok & (r16 < 16), other=0.0,
                )
                acc += tl.sum(w.to(tl.float32) * xk.to(tl.float32)[:, None], 0)
        tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
    # --- END generated ---


def exl3_mgemm_triton(
    x: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    expert_ids: torch.Tensor,
    y: torch.Tensor,
    K_bits: int,
    cb: int = 0,
    trellis_sample: torch.Tensor | None = None,
) -> None:
    """One-launch grouped EXL3 dequant + GEMV over a routed expert set.

    ``y[e] = dequant(trellis[expert_ids[e]]).T @ x[e]`` for e in [0, E).

    x            [E, K_dim] or [1, K_dim] half; an expert stride of 0
                 (``x.expand(E, -1)``) broadcasts one decode row to all experts
    ptrs_trellis [num_experts] int64 device tensor of trellis base addresses
                 (MultiLinear.ptrs_trellis)
    expert_ids   [E] int32/int64 device tensor indexing ptrs_trellis
    y            [E, N] half or float output, written in full
    K_bits       1..8
    cb           0 plain / 1 mcg / 2 mul1

    All experts must share shape and bit width (MultiLinear asserts this). The
    trellis layout is the packed EXL3 one: int16 [K_dim/16, N/16, 16*K_bits],
    contiguous; strides are derived from the shapes and cross-checked against
    ``trellis_sample`` when one is supplied.
    """
    assert x.dim() == 2 and y.dim() == 2, "exl3_mgemm_triton: M == 1 only"
    E = y.shape[0]
    N = y.shape[1]
    K_dim = x.shape[1]
    assert x.shape[0] in (E, 1), f"x rows {x.shape[0]} vs E {E}"
    assert expert_ids.numel() == E
    assert K_dim % 16 == 0 and N % 16 == 0

    # int16 element strides of the packed trellis
    stride_tn = 16 * K_bits
    stride_tk = (N // 16) * stride_tn
    if trellis_sample is not None:
        assert tuple(trellis_sample.shape) == (K_dim // 16, N // 16, 16 * K_bits), \
            f"unexpected trellis shape {tuple(trellis_sample.shape)}"
        assert trellis_sample.stride(0) == stride_tk and trellis_sample.stride(1) == stride_tn, \
            "exl3_mgemm_triton: non-contiguous trellis is not supported"

    stride_xe = x.stride(0) if x.shape[0] == E else 0
    grid = lambda meta: (E * triton.cdiv(N, meta["BLOCK_N"]),)
    perm_i = _get_perm_i(x.device)

    _grouped_dequant_gemv_kernel[grid](
        x, y,
        ptrs_trellis, expert_ids,
        perm_i,
        _get_m_row_offsets(K_bits, x.device) if K_bits in _M_ROW_OFFSETS else perm_i,
        E, N, K_dim,
        _e_bucket(E),
        stride_xe, x.stride(1),
        stride_tk, stride_tn,
        y.stride(0), y.stride(1),
        K_BITS=K_bits,
        N_PACKED=16 * K_bits,
        CB=cb,
    )


def _e_bucket(E: int) -> int:
    """Autotune key for the expert count (same idea as exl3_triton._m_bucket).

    The compiled code does not depend on E -- only the grid does -- but the
    best tile does, so E belongs in the key. Bucketing to powers of two keeps
    that from minting a fresh autotune entry for every routed-set size.
    """
    if E <= 1:
        return 1
    return 1 << (E - 1).bit_length()


# ---------------------------------------------------------------------------
# Full grouped linear: had -> mgemm -> had (3 launches for E experts)
# ---------------------------------------------------------------------------

def _linear_exl3_mgemm_triton(
    x: torch.Tensor,
    xh: torch.Tensor,
    y: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    expert_ids: torch.Tensor,
    K_bits: int,
    cb: int,
) -> None:
    """Grouped EXL3 linear into pre-allocated buffers. Allocates nothing.

    Graph-capture entry point: every buffer address is the caller's, the grid
    is a function of E and N only, and ``expert_ids`` is read on the device.

    x   [1, in_features] (shared decode row) or [E, in_features] half
    xh  [E, 1, in_features] half workspace
    y   [E, out_features] half or float, written in full
    """
    E, _, in_features = xh.shape
    out_features = y.shape[1]
    assert x.dim() == 2 and x.shape[1] == in_features and x.shape[0] in (1, E)
    assert x.dtype == torch.half and xh.dtype == torch.half
    assert y.shape[0] == E

    xin = x.unsqueeze(0).expand(E, -1, -1) if x.shape[0] == 1 else x.unsqueeze(1)
    had_r_128_mtriton(xin, xh, ptrs_suh, None, expert_ids, 1.0)
    exl3_mgemm_triton(xh.view(E, in_features), ptrs_trellis, expert_ids, y, K_bits, cb)
    had_r_128_mtriton(y.unsqueeze(1), y.unsqueeze(1), None, ptrs_svh, expert_ids, 1.0)


def linear_exl3_mgemm_triton(
    x: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    expert_ids: torch.Tensor,
    K_bits: int,
    cb: int,
    in_features: int,
    out_features: int,
    xh: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    """Allocating convenience wrapper over ``_linear_exl3_mgemm_triton``.

    x is [in_features] or [1, in_features] (shared decode row) or
    [E, in_features] (per-expert rows, e.g. the down_proj input). Returns
    [E, out_features]. Pass ``xh``/``y`` to keep it allocation-free.
    """
    E = expert_ids.shape[0]
    x2 = x.view(-1, in_features)
    if xh is None:
        xh = torch.empty((E, 1, in_features), dtype=torch.half, device=x.device)
    if y is None:
        y = torch.empty((E, out_features), dtype=out_dtype, device=x.device)
    _linear_exl3_mgemm_triton(x2, xh, y, ptrs_trellis, ptrs_suh, ptrs_svh,
                              expert_ids, K_bits, cb)
    return y


def mgemm_prepare(
    E: int,
    in_features: int,
    out_features: int,
    K_bits: int,
    cb: int,
    device: torch.device,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor | None = None,
    ptrs_svh: torch.Tensor | None = None,
) -> None:
    """Run every kernel once so nothing is deferred into a CUDA-graph capture.

    Triton autotune benchmarks a config pool on the first call for a new key,
    and the perm / m-row index tensors are built lazily; both must happen
    before capture. Safe to call repeatedly.
    """
    _get_perm_i(device)
    if K_bits in _M_ROW_OFFSETS:
        _get_m_row_offsets(K_bits, device)
    eids = torch.zeros((E,), dtype=torch.int32, device=device)
    x = torch.zeros((1, in_features), dtype=torch.half, device=device)
    xh = torch.zeros((E, 1, in_features), dtype=torch.half, device=device)
    y = torch.zeros((E, out_features), dtype=torch.half, device=device)
    if ptrs_suh is not None and ptrs_svh is not None:
        _linear_exl3_mgemm_triton(x, xh, y, ptrs_trellis, ptrs_suh, ptrs_svh,
                                  eids, K_bits, cb)
    else:
        exl3_mgemm_triton(x, ptrs_trellis, eids, y, K_bits, cb)
    torch.cuda.synchronize()
