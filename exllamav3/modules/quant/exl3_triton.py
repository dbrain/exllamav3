"""Triton implementation of the EXL3 linear layer (fused dequant + GEMM).

An opt-in alternative to the C++ BC/reconstruct paths, selected per layer with
the EXL3_PREFER_TRITON_LINEAR=1 environment variable (see LinearEXL3.forward).
Where the BC_* classes are unavailable (e.g. ROCm builds) this is the fast
path; on CUDA it is an alternative for comparison and benchmarking.

Three entry points, called directly like every other Triton path in this
project (no torch.library registration — the dispatcher overhead is not wanted
on the decode path):

    had_r_128_triton(x, y, suh, None, 1.0)   # row Hadamard transform
    exl3_gemm_triton(xh, t, y, ...)          # fused dequant + GEMM
    linear_exl3_triton(...)                  # full linear forward: hadamard
                                             # -> fused dequant-gemm -> hadamard
                                             # (+ optional bias)

A decode linear costs three launches by default. Two flags remove one each:

    EXL3_FUSE_INPUT_HAD   folds the input Hadamard into the GEMM's K loop
                          (pass ``suh`` to exl3_gemm_triton). Costs no tile
                          quality, but is only BIT-EXACT for bits 1, 2 and 8:
                          the heavy-accumulator widths change fp32 reduction
                          rounding under the added register pressure. See
                          _fuse_input_had.
    EXL3_FUSE_OUTPUT_HAD  folds the output Hadamard + post-scale into the
                          GEMM's store (pass ``svh``). NOT free: it needs
                          BLOCK_N % 128 == 0, which forfeits the narrow decode
                          tiles the heavy-accumulator widths want, so
                          _fuse_output_had gates it on N. On the split-K route
                          it never engages -- _m1_split_reduce_had already
                          folds the same transform into the reduce.

The fused kernel decodes the EXL3 trellis tile-by-tile inside the K-loop
without materializing the weight matrix. Every bit width K = 1..8 has a
dedicated M==1 fast decode whose per-element word/shift lookup is realized
without data-dependent gathers (linear/affine u32 row loads + static-reshape
permutations, or per-(row, c>>3) constexpr window offsets for the odd
widths), so packed rows load as linear u32 vectors. Non-divisible shapes
fall back to a generic staged-row tl.gather decode that covers all widths.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    # Triton dummy shims so importing this module doesn't fail when Triton is
    # unavailable (mirrors attention_fn/triton_paged.py)
    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()

# ---------------------------------------------------------------------------
# Triton Hadamard transform (128-element rows, radix-2 butterfly)
#
# Mirrors the C++ had_hf/had_ff_r_128 kernels exactly: the transform is
# evaluated in fp32 regardless of I/O dtype (with a single round to half at
# the very end for half output), r_scale = scale / sqrt(128) is applied in
# fp32 after the transform, and pre/post scales are applied in the I/O
# dtype. The butterfly runs sequentially over masks 1..64, which reproduces
# the C++ expression tree (4-point transform in registers, then 32-lane
# xor-shuffles) term-for-term, so results are bit-identical.
# ---------------------------------------------------------------------------

@triton.jit
def _had_stage(v, BLOCK_R: tl.constexpr, SPAN: tl.constexpr):
    """One radix-2 butterfly stage over a [BLOCK_R, 128] fp32 tile.

    Elements whose bit log2(SPAN) is 0 receive a+b, those with the bit set
    receive a-b, where b is the partner element at distance SPAN.
    """
    G: tl.constexpr = 128 // (2 * SPAN)
    pair = tl.permute(v.reshape(BLOCK_R, G, 2, SPAN), (0, 1, 3, 2))
    lo, hi = tl.split(pair)
    pair = tl.join(lo + hi, lo - hi)
    return tl.permute(pair, (0, 1, 3, 2)).reshape(BLOCK_R, 128)


@triton.jit
def _had_r_128_kernel(
    x_ptr, y_ptr, scale_ptr,
    n_rows,
    stride_xr, stride_yr,
    r_scale,
    IO_FP32: tl.constexpr,
    PRE_SCALED: tl.constexpr,
    POST_SCALED: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # One program transforms a [BLOCK_R, 128] tile: pid_m rows, pid_c the
    # 128-column block within each row. Scales are indexed by flat column
    # position (row-independent), matching the C++ kernel.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = rows < n_rows
    col = tl.arange(0, 128)

    x = tl.load(
        x_ptr + rows[:, None] * stride_xr + (pid_c * 128 + col)[None, :],
        mask=mask_r[:, None], other=0.0,
    )

    # Pre-scale, applied in the I/O dtype (half multiply for the half path,
    # exactly like the C++ __hmul2 version)
    if PRE_SCALED:
        pre = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            x = x * pre.to(tl.float32)
        else:
            x = x * pre

    # fp32 radix-2 butterfly over bit masks 1..64, unrolled via a constexpr
    # helper. Each stage pairs element j with j^span: the bit-0 element gets
    # a+b, the bit-1 element a-b — the same expression tree as the C++
    # register H4 + xor-shuffle network, so the result is bit-identical.
    v = x.to(tl.float32)
    v = _had_stage(v, BLOCK_R, 1)
    v = _had_stage(v, BLOCK_R, 2)
    v = _had_stage(v, BLOCK_R, 4)
    v = _had_stage(v, BLOCK_R, 8)
    v = _had_stage(v, BLOCK_R, 16)
    v = _had_stage(v, BLOCK_R, 32)
    v = _had_stage(v, BLOCK_R, 64)
    v = v * r_scale

    # Post-scale. The C++ half kernel rounds the scaled transform to half
    # first and then multiplies in half; reproduce that ordering exactly.
    if POST_SCALED:
        post = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            out = v * post.to(tl.float32)
        else:
            out = v.to(x_ptr.dtype.element_ty) * post
    else:
        out = v

    tl.store(
        y_ptr + rows[:, None] * stride_yr + (pid_c * 128 + col)[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=mask_r[:, None],
    )

# ---------------------------------------------------------------------------
# had_r_128_triton: Triton row Hadamard transform
#
# A Triton twin of the C++ pybind ``ext.had_r_128`` (quant/hadamard.cu),
# bit-identical by construction. Used ONLY inside the Triton EXL3 linear
# path so that path depends on Triton alone; every other caller keeps
# using the C++ kernel through pybind.
#
# Scope note: its kernel time matches or beats the C++ kernel at every shape
# (measured via CUDA-graph replay), but every call — captured or not — costs a
# launch, and on this part a launch is ~16 us of fixed DEVICE cost (Triton
# pure-stream fit t = 16.3us + bytes/76.6GB/s, R^2 0.999999) plus host power.
# At decode these transforms move 5-40 KB, so the launch IS the cost: a
# 2560-wide input Hadamard is ~0.07 us of stream against ~16 us of fixed cost.
# The decode path avoids the input one when EXL3_FUSE_INPUT_HAD is set
# (_fuse_input_had / _had_x_tile) and the output one on the split-K route
# (_m1_split_reduce_had) or under EXL3_FUSE_OUTPUT_HAD (_store_out_had); don't
# adopt this kernel anywhere a fusion is available instead.
# ---------------------------------------------------------------------------

_RSCALE_128 = 0.088388347648  # 1/sqrt(128), matches the C++ literal


def had_r_128_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    scale: float,
) -> None:
    """y = (x.view(-1, 128) @ H128) * (pre|post)_scale, scaled by scale/sqrt(128).

    Matches the C++ ``had_r_128`` contract: input/output must be 2D, the same
    dtype (half or float), contiguous in the last dim, with last dim a
    multiple of 128; scales are half tensors with one element per column
    (flat, row-independent). Pre-scale multiplies before the transform in the
    I/O dtype; post-scale multiplies after it (for half output, after the
    round to half), like the C++ kernels.
    """
    assert input.dtype == output.dtype, "had_r_128_triton: input/output dtype mismatch"
    assert input.dtype in (torch.half, torch.float), \
        f"had_r_128_triton: unsupported dtype {input.dtype}"
    assert input.dim() == 2 and input.shape[-1] % 128 == 0
    # The kernel indexes the last dim with an implicit unit stride
    assert input.stride(-1) == 1, \
        f"had_r_128_triton: input last dim must be contiguous, got stride {input.stride(-1)}"
    assert output.stride(-1) == 1, \
        f"had_r_128_triton: output last dim must be contiguous, got stride {output.stride(-1)}"
    assert (pre_scale is None) or (post_scale is None)
    rows, cols = input.shape

    # Tiling (swept on RDNA3 via graph-of-64 replay timing): BLOCK_R=4 with a
    # single warp is optimal or tied-for-optimal at every shape from rows==1
    # (decode) through rows==512+ (prefill). The 128-wide tile leaves extra
    # warps idle; wider row tiles only help shapes too small to matter.
    BLOCK_R = 4
    num_warps = 1

    grid = (triton.cdiv(rows, BLOCK_R), cols // 128)
    _had_r_128_kernel[grid](
        input, output,
        pre_scale if pre_scale is not None else post_scale,
        rows,
        input.stride(0), output.stride(0),
        scale * _RSCALE_128,
        IO_FP32=input.dtype == torch.float,
        PRE_SCALED=pre_scale is not None,
        POST_SCALED=post_scale is not None,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# EXL3 dequantization in pure PyTorch
# ---------------------------------------------------------------------------

_TENSOR_CORE_PERM = None
_TENSOR_CORE_PERM_I = None

def _get_perm(device):
    global _TENSOR_CORE_PERM, _TENSOR_CORE_PERM_I
    if _TENSOR_CORE_PERM is None or _TENSOR_CORE_PERM.device != device:
        perm = [0] * 256
        for t in range(32):
            r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
            c0 = t // 4; c1 = c0 + 8
            perm[t*8+0] = r0*16+c0; perm[t*8+1] = r1*16+c0
            perm[t*8+2] = r2*16+c0; perm[t*8+3] = r3*16+c0
            perm[t*8+4] = r0*16+c1; perm[t*8+5] = r1*16+c1
            perm[t*8+6] = r2*16+c1; perm[t*8+7] = r3*16+c1
        perm_i = [0]*256
        for i, p in enumerate(perm):
            perm_i[p] = i
        _TENSOR_CORE_PERM = torch.tensor(perm, device=device, dtype=torch.long)
        _TENSOR_CORE_PERM_I = torch.tensor(perm_i, device=device, dtype=torch.long)
    return _TENSOR_CORE_PERM_I


_DQ_CACHE = {}
_LUT_CACHE = {}

# Per-row window offsets for the odd bit widths (K = 3, 5, 7).
#
# For those widths the per-element (word, shift) lookup does not factor into
# independent per-axis bit fields (the window end falls inside a code, so the
# word index and the funnel shift carry into each other). Instead each of the
# 32 (r, c//8) combinations of a 16x16 sub-tile has one fixed window offset
# D(r, c3) = 32*f + sh: element (r, c) reads the 16-bit code window starting
# at stream bit 32*(K_BITS*(c%8) + f) + sh, i.e. word (K_BITS*(c%8) + f) of
# the subtile at funnel shift sh (neighbor word -1 when sh > 16). These
# tables were recovered element-exactly from the C++ reconstruct kernel by
# differential probing and verified to reproduce its output bit-for-bit.
_M_ROW_OFFSETS = {
    3: [29, 17, 26, 14, 5, 57, 2, 54, 45, 33, 42, 94, 85, 73, 82, 70,
        23, 11, 20, 8, 63, 51, 60, 48, 39, 91, 36, 88, 79, 67, 76, 64],
    5: [27, 7, 22, 2, 51, 95, 46, 90, 75, 119, 70, 114, 99, 143, 158, 138,
        17, 61, 12, 56, 41, 85, 36, 80, 65, 109, 124, 104, 153, 133, 148, 128],
    7: [25, 61, 18, 54, 33, 69, 90, 126, 105, 141, 98, 134, 177, 213, 170, 206,
        11, 47, 4, 40, 83, 119, 76, 112, 155, 191, 148, 184, 163, 199, 220, 192],
}
_M_ROW_CACHE = {}


def _get_m_row_offsets(K_bits: int, device) -> torch.Tensor:
    key = (K_bits, str(device))
    if key not in _M_ROW_CACHE:
        _M_ROW_CACHE[key] = torch.tensor(
            _M_ROW_OFFSETS[K_bits], device=device, dtype=torch.int32
        )
    return _M_ROW_CACHE[key]


def _decode_lut(cb: int, device) -> torch.Tensor:
    key = (cb, str(device))
    if key not in _LUT_CACHE:
        x = torch.arange(65536, device=device, dtype=torch.int64)
        M = 0xFFFFFFFF
        if cb == 0:
            x = (x * 89226354) & M; x = (x + 64248484) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 1:
            x = (x * 0xCBAC1FED) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 2:
            x = (x * 0x83DCD12D) & M
            acc = torch.full_like(x, 0x6400)
            s = (acc + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)) & 0xFFFF
            sum_h = s.to(torch.int16).view(torch.float16)
            k_inv = torch.tensor([0x1eee], dtype=torch.int16, device=device).view(torch.float16)
            k_bias_data = torch.tensor([0xc931], dtype=torch.int32, device=device).to(torch.int16).view(torch.float16)
            lut = sum_h * k_inv + k_bias_data
        _LUT_CACHE[key] = lut
    return _LUT_CACHE[key]


# ---------------------------------------------------------------------------
# M == 1 split-K GEMV plan (starved-N shapes)
#
# The decode GEMV's only lever on a CTA-starved shape (N/BLOCK_N under one
# wave, e.g. MLP down_proj N=4096-5120) is memory-level parallelism: CTAs x
# staged bytes in flight. Splitting the K loop across SPLITS CTAs per N tile
# multiplies the CTA count without shrinking the staged window, at the cost
# of SPLITS x N fp32 partials (tens of KB) and one tiny reduce kernel that
# also performs the output Hadamard transform (fusing away the separate
# had_r_128 launch for these linears).
#
# Numerics: the partial sums are fp32 and the reduce runs the butterfly in
# fp32, so the split path skips the one intermediate round-to-half the
# classic path takes between GEMV and Hadamard (slightly more accurate, same
# 2e-2 relative tolerance regime; accumulation order within fp32 changes).
# ---------------------------------------------------------------------------

_SPLITK_BUFS: dict = {}


def _m_bucket(M: int) -> int:
    """Autotune key for the row count.

    M is data-dependent on the MoE per-expert path: with the fused exl3_moe kernel
    unavailable (ROCm), block_sparse_mlp falls back to an index_select loop that
    calls each expert with however many tokens routed to it, so nearly every expert
    presents a different M and mints a fresh autotune entry. Measured on a
    35B-A3B load: 155 autotune passes over 5 modules, 1276 s of a 1277 s load.

    Bucketing to powers of two collapses that to at most one entry per octave while
    keeping M == 1 (the GEMV fast path, which has its own config pool) distinct.
    """
    if M <= 1:
        return 1
    return 1 << (M - 1).bit_length()


def _m1_splitk_plan(M: int, N: int, K_dim: int, K_bits: int) -> int:
    """Number of K-splits for an M == 1 invocation, or 1 (classic path).

    Split-K applies where the bits=4 fast path is guaranteed for the whole
    autotune pool (N and K divisible by 256 covers BLOCK_K up to 256) and
    each split still gets a meaningful K slice. It helps CTA-starved shapes
    (down_proj class, N=4096-5120) massively and large-N shapes (gate/up /
    qkv-class, N=10240-17408) moderately: measured composite GB/s incl.
    hadamards + reduce (RX 7900 XTX, L2-cold layer sweeps):
      9B  down N=4096  K=12288: 242 -> 445 (S=4) / 455 (S=8, BN32/BK256)
      27B down N=5120  K=17408: 239 -> 353 (S=4) / 385 (S=8)
      27B g/u  N=17408 K=5120:  310 -> 378 (S=4, BN64/BK256)
    EXL3_SPLITK=off (or =n) overrides for experiments; default behavior
    needs no environment variable.
    """
    import os
    if M != 1 or K_bits not in (4, 6) or N % 256 or K_dim % 256:
        return 1
    # bits=4 splits linear-class shapes; bits=6 splits the linear-class b6
    # shapes too (6bpw MLP: down 442 -> 559 GB/s) but NOT the lm_head stream
    # (N=248320 loses at S=4: 628 vs 666 GB/s classic — already parallel)
    if N > (16384 if K_bits == 6 else 32768):
        return 1
    k_tiles = K_dim // 16
    splits = 8 if k_tiles >= 512 else (4 if k_tiles >= 256 else 1)
    env = os.environ.get("EXL3_SPLITK")
    if env is not None:
        splits = 0 if env.lower() in ("off", "0", "none") else int(env)
    # Every split needs at least ~4 outer iterations of a BK256 tile.
    if k_tiles < splits * 64:
        return 1
    return max(splits, 1)


def _get_splitk_buf(N: int, splits: int, device) -> torch.Tensor:
    key = (N, splits, str(device))
    buf = _SPLITK_BUFS.get(key)
    if buf is None:
        buf = torch.empty((splits, N), dtype=torch.float, device=device)
        _SPLITK_BUFS[key] = buf
    return buf


@triton.jit
def _m1_split_reduce_had_kernel(
    partials_ptr, y_ptr, scale_ptr,
    N, stride_ps, stride_yn,
    r_scale,
    SPLITS: tl.constexpr,
    IO_FP32: tl.constexpr,
):
    """Sum the split-K partials for one 128-column block and apply the output
    Hadamard transform + post-scale, reproducing _had_r_128_kernel's fp32
    butterfly and rounding order exactly (scale 1.0)."""
    pid = tl.program_id(0)
    col = pid * 128 + tl.arange(0, 128)
    acc = tl.zeros((128,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(partials_ptr + s * stride_ps + col)
    v = tl.reshape(acc, (1, 128))
    v = _had_stage(v, 1, 1)
    v = _had_stage(v, 1, 2)
    v = _had_stage(v, 1, 4)
    v = _had_stage(v, 1, 8)
    v = _had_stage(v, 1, 16)
    v = _had_stage(v, 1, 32)
    v = _had_stage(v, 1, 64)
    v = tl.reshape(v, (128,)) * r_scale
    post = tl.load(scale_ptr + col)
    if IO_FP32:
        out = v * post.to(tl.float32)
    else:
        out = v.to(y_ptr.dtype.element_ty) * post
    tl.store(y_ptr + col * stride_yn, out.to(y_ptr.dtype.element_ty))


def _m1_split_reduce_had(
    partials: torch.Tensor, y: torch.Tensor, post_scale: torch.Tensor, splits: int,
) -> None:
    N = partials.shape[1]
    assert y.stride(-1) == 1, "split reduce: output must be contiguous in the last dim"
    _m1_split_reduce_had_kernel[(N // 128,)](
        partials, y, post_scale,
        N, partials.stride(0), y.stride(-1),
        _RSCALE_128,
        SPLITS=splits,
        IO_FP32=y.dtype == torch.float,
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Triton matmul kernel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused dequant + GEMM Triton kernel
#
# Each block computes a [BLOCK_M, BLOCK_N] output tile. The weight tile
# [BLOCK_K, BLOCK_N] is decoded on-the-fly from the compressed trellis every
# K-iteration (no full weight matrix is ever materialized).
#
# Memory strategy (mirrors the C++ kernel's sh_b staging): for every 16x16
# trellis sub-tile, the packed words of all covered n-sub-tile columns are
# fetched with linear, dword-vectorized u32 loads (128 B per sub-tile row,
# contiguous in the trellis), never scattered scalar gathers.
#
# Fast paths (full tiles): every bit width decodes the full [shift, word]
# code table with NO data-dependent gather, so each sub-tile weight is
# computed exactly once:
# - K = 1/2/4/8: the 16x16 sub-tile permutation factors into per-axis index
#   bits (see the kernel branches), realized by static reshape/permute
#   (tensor-core path) or folded into a broadcast pattern of the x vector
#   (M==1 GEMV path).
# - K = 3/5/7: per-(row, c>>3) window-offset tables (_M_ROW_OFFSETS) with
#   affine word-slice loads.
# - K = 6: four word-slice loads + the _funnel6 word-pair decode.
# For M == 1 the decoded tile is reduced in fp32 with the product tile
# accumulated elementwise over the whole K loop, so the k-loop issues no
# cross-lane reductions; the result is summed once per block at the end.
#
# Generic path (non-divisible tiles): staged row load + tl.gather decode,
# which lowers to LDS (shared-memory) reads.
# ---------------------------------------------------------------------------


_DEV_CACHE = {}


def _dev_caps(device = None):
    """(cu_count, wave_size, is_amd) for the active device, cached.

    The decode pools below were measured on an RX 7900 XTX: 48 CUs, and Triton
    num_warps counted in 32-wide waves. Both assumptions break on other parts —
    notably RDNA3.5 APUs (gfx1150: 8 WGPs, wave64), where a num_warps that is
    nominally 4 is 256 lanes and starves occupancy.
    """
    idx = torch.cuda.current_device() if device is None else device
    if idx not in _DEV_CACHE:
        p = torch.cuda.get_device_properties(idx)
        is_amd = torch.version.hip is not None
        wave = getattr(p, "warp_size", None) or (64 if is_amd else 32)
        _DEV_CACHE[idx] = (p.multi_processor_count, wave, is_amd)
    return _DEV_CACHE[idx]


def _prefer_warps(configs):
    """Narrow the CTAs on parts with few CUs.

    Warps per CTA trades latency hiding within a CTA against how many CTAs stay
    resident per CU. The pools pick num_warps=4/8 because 48 CUs give the XTX
    plenty of CTAs from N-tiling alone. A part with a handful of CUs is the
    other way round: it needs narrow CTAs to keep several resident.

    Measured on gfx1150 (8 WGPs), 1x4096x12288 bits=4, both hadamards:
    num_warps=2 takes the top five results outright — 67.6 GB/s at
    BLOCK_N=32/BLOCK_K=128 — against 45-58 for the pools' native nw=4/8.
    num_warps=1 collapses to 6.8, so this is an occupancy optimum, not
    "narrower is better".

    Note RDNA runs wave32, so this is not a wave64 lane-count correction.
    """
    cu, wave, is_amd = _dev_caps()
    if cu > 16:
        return configs
    want = [c for c in configs if c.num_warps == 2]
    return want if want else configs


# Measured tile winners for the DENSE pool, keyed like the autotune key's shape
# part: (K_dim, N, K_BITS, M_BUCKET) -> (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
# num_warps, num_stages).
#
# Same rationale as _MGEMM_PINNED in exl3_mgemm_triton.py: on gfx1150 the pool
# is benchmarked on a cold clock and the candidates sit within noise, so the
# pick is not reproducible -- the dense b6 pool chose BLOCK_K 256 one run and
# 128 the next. Non-deterministic tile selection is not just a reproducibility
# annoyance: it changes the fp32 accumulator reduction order, so it decides
# whether a result is bit-exact or ~1 ULP off, and it perturbs graph-capture
# engagement. Pinning makes the shipped kernel a function of the shape alone.
#
# Escape hatches, in precedence order:
#   EXL3_GEMM_CONFIGS=...     explicit pool, bypasses the table (manual sweeps)
#   EXL3_GEMM_PIN=0           ignore the table, benchmark the pool as before
#   EXL3_GEMM_PIN=strict      never benchmark: on a table miss take the first
#                             config the prune leaves, a pure function of the
#                             shape. Covers shapes nobody has measured yet,
#                             which a table cannot.
_GEMM_PINNED = {
    # (K_dim,     N, bits, M_bucket): (BM,  BN,  BK, GM, nw, ns)
    # Flash-Next 4.05bpw dense decode (M == 1). Winners of a 13-tile sweep,
    # 3 interleaved reps, cache warm, GPU edge gated, zero compiles inside any
    # timing window (llm-fondling/perf/exl3/ledger-gemm.csv). Every entry is a
    # member of the pool the autotuner would have chosen from anyway, so pinning
    # cannot pick a tile production could not already have landed on -- it only
    # stops the choice being a coin flip.
    #
    # Confirmed end to end through this code path, not just as isolated arms
    # (ledger-dpin.csv, 3 reps, fresh process per arm, depth 4096):
    #   EXL3_GEMM_PIN=0   9.472 t/s   105.6 ms/tok   rep spread 1.55% / 6.12%
    #   pinned           10.037       99.6           rep spread 0.18% / 0.04%
    #   strict           10.226       97.7           rep spread 2.02% / 2.09%
    # (the two spreads are sync=none / sync=token). +6.0% from the table alone.
    # Note the isolated per-shape sweep predicted -7.5 ms and the real gain was
    # -4.3 ms: kernel rates measured in isolation do not fully carry into decode.
    ( 2560,  10240, 6, 1): ( 16,  64, 128, 1, 2, 3),  # gdn.in_proj_qkv 35.4 vs 32.1
    ( 2560,   6144, 6, 1): ( 16,  64, 128, 1, 2, 3),  # gdn.in_proj_z   35.2 vs 31.9
    ( 2560,  12288, 6, 1): ( 16, 128,  64, 1, 2, 3),  # attn.q_proj     38.0 vs 33.5
    ( 2560,    640, 6, 1): ( 16,  16,  64, 1, 2, 3),  # shared.gate/up  15.6 vs 13.0
    ( 2560,    512, 6, 1): ( 16,  16,  64, 1, 2, 3),  # attn.k/v_proj   14.3 vs 12.7
    ( 2560,    640, 4, 1): ( 16,  32, 256, 1, 2, 2),  # attn.indexer_qk 12.2 vs 11.5
    ( 2560, 248320, 6, 1): ( 16, 128,  64, 1, 2, 3),  # lm_head         56.2 vs 47.9

    # Flash-Next 4.05bpw SPECULATIVE VERIFY (M_BUCKET 8). An MTP round runs the
    # target over ndt + 1 tokens, so the deployed ndt 4 and 6 both land here
    # while plain decode stays at M_BUCKET 1. Without these entries every verify
    # shape misses the table and `strict` takes the first config the prune
    # leaves -- and at M > 1 the prune returns the pool UNFILTERED (only the
    # m == 1 path is N-bucketed, and only it gets _prefer_warps), so out[0] is
    # decided by the literal order of _exl3_gemm_configs(): the M == 1
    # CTA-starved decode tile at num_warps 4, handed to all eight sites at once.
    #
    # Values are the autotuner's own picks, read out of the live cache of a
    # fresh EXL3_GEMM_PIN=1 process on this checkpoint
    # (llm-fondling/perf/exl3/ledger-roundgap-tiles.csv, run 20260910_071434);
    # the strict process of the same run (20260910_060625) agrees with the
    # M_BUCKET 1 entries above on every shape and disagrees here on every one.
    #
    # End to end, ledger-roundgap-sample.csv -- median of 4 samples per cell,
    # 2 prompts, 2 interleaved reps, fresh process per pin arm, 0 compilations
    # in any timed window, both arms opening at Tctl 99 C and closing at 76 C
    # (the harness's host spin thread is what heats the die; the timed decode is
    # memory-bound and the package cools through it):
    #   target verify forward   363.7 -> 263.7 ms (ndt 4), 419.4 -> 313.8 (ndt 6)
    #   round                   405.2 -> 305.9 ms (ndt 4), 482.1 -> 372.4 (ndt 6)
    #   plain decode round       98.8 ->  98.7 ms -- unchanged, as it must be
    #   long_code at acceptance 3.0/4:  9.88 -> 13.08 t/s
    ( 2560,  10240, 6, 8): ( 16,  32,  32, 1, 1, 2),  # gdn.in_proj_qkv
    ( 2560,   6144, 6, 8): ( 16,  32,  32, 1, 1, 2),  # gdn.in_proj_z
    ( 2560,  12288, 6, 8): ( 16,  32,  32, 1, 2, 3),  # attn.q_proj
    ( 2560,    640, 6, 8): ( 16,  32,  32, 1, 1, 2),  # shared.gate/up
    ( 2560,    512, 6, 8): ( 16,  32,  32, 1, 1, 2),  # attn.k/v_proj
    ( 2560, 248320, 6, 8): ( 16,  32,  32, 1, 1, 2),  # lm_head
    ( 6144,   2560, 6, 8): ( 16,  32,  32, 1, 2, 3),  # gdn.out_proj
    (  640,   2560, 6, 8): ( 16,  32,  32, 1, 1, 2),  # moe / shared down_proj

    # The speculative round's remaining dense GEMM, 19.9 ms of ~301 (6.6%), which ran
    # AUTOTUNED because nobody had tabled it -- and TRITON_CACHE_AUTOTUNING freezes an
    # autotune result on disk without ever re-measuring it, so a pass that happened to run
    # while this 30 W box was busy could pin a loser permanently. These values were not swept:
    # they are the picks the autotuner has already been making, read back out of
    # ~/.triton/cache and cross-checked across every cached generation, so shipping them is a
    # determinism fix and should be a null on speed.
    #   consistency across cached generations / margin over the runner-up:
    ( 2560,  98304, 6, 2): ( 16,  32,  32, 1, 1, 2),  # mtp sliced draft head   1/1,  1.39x
    ( 2560,  98304, 6, 4): ( 16,  32,  32, 1, 1, 2),  #   17.134 ms/round, 5.69% 1/1,  1.41x
    ( 2560,   2560, 5, 2): ( 16,  32,  32, 1, 1, 2),  # mtp.fc_hidden/_embedding 9/9, 1.09x
    ( 2560,   2560, 5, 4): ( 16,  32,  32, 1, 1, 2),  #                        10/10, 1.12x
    ( 2560,   2560, 5, 8): ( 16,  32,  32, 1, 1, 2),  #                         9/9,  1.07x
    ( 2560,    640, 4, 8): ( 16,  32, 128, 1, 2, 3),  # attn.indexer_qk        12/13, ~1.00x
}

_GEMM_PIN_OFF = ("0", "off", "false", "no")


def _gemm_pinned_config(named_args, kwargs):
    """The pinned tile for this shape, or None to fall through to autotune."""
    if os.environ.get("EXL3_GEMM_PIN", "1").lower() in _GEMM_PIN_OFF:
        return None
    if os.environ.get("EXL3_GEMM_CONFIGS"):
        return None

    def arg(nm):
        return kwargs.get(nm, named_args.get(nm))

    hit = _GEMM_PINNED.get((arg("K_dim"), arg("N"), arg("K_BITS"),
                            arg("M_BUCKET")))
    if hit is None:
        return None
    bm, bn, bk, gm, nw, ns = hit
    # A pin must never override a correctness constraint. A tile that does not
    # divide the shape lands in the generic gather path with a different
    # accumulation order -- a different result, not just a slower one -- and the
    # fused transforms are block-diagonal over aligned 128-element groups.
    n_, k_ = arg("N"), arg("K_dim")
    if n_ % bn or k_ % bk:
        return None
    if kwargs.get("FUSE_HAD", named_args.get("FUSE_HAD")) and bk % 128:
        return None
    if kwargs.get("FUSE_OUT_HAD", named_args.get("FUSE_OUT_HAD")) and bn % 128:
        return None
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                          "GROUP_M": gm}, num_warps=nw, num_stages=ns)


def _exl3_gemm_prune_pool(configs, named_args, **kwargs):
    """Restrict the config set per invocation:
    - Other bit widths run the generic tl.gather path, where this Triton
      build's LLVM aborts on large gather tiles.
    - Shapes whose N/K are not divisible by a config's tile fall back to the
      generic path too, so apply the same cap there.
    - M == 1 never benefits from BLOCK_M > 16 (the grid is a single block and
      the GEMV branch ignores BLOCK_M); pruning them also works around Triton
      compile failures for some narrow-BLOCK_N decode tiles at high warp
      counts.
    - M == 1 full-tile shapes are bucketed by N (RDNA3 starved-N rule): with
      N/BLOCK_N CTAs under ~one wave (48 CUs), only small-BLOCK_N tiles have
      enough parallelism, and at large N the wide tiles amortize the staged
      decode better. Every pool member was measured at or above the previous
      default pick's rate on its bucket's shapes, so a cold-clock autotune
      pass cannot lock in a regression.
    Every bit width has a gather-free fast path that handles the large tiles."""
    bits = kwargs.get("K_BITS", named_args.get("K_BITS"))
    n = kwargs.get("N", named_args.get("N"))
    k = kwargs.get("K_dim", named_args.get("K_dim"))
    m = kwargs.get("M", named_args.get("M"))
    fast_ok = n % 128 == 0 and k % 128 == 0
    if kwargs.get("FUSE_HAD", named_args.get("FUSE_HAD")):
        # The in-kernel input Hadamard needs whole 128-element K groups per
        # tile, so narrower K tiles cannot serve it. Applied to the base list
        # so every "empty pool -> configs" fallback below stays legal.
        configs = [c for c in configs if c.kwargs["BLOCK_K"] % 128 == 0]
    if kwargs.get("FUSE_OUT_HAD", named_args.get("FUSE_OUT_HAD")):
        # Same argument on the N axis: the in-kernel output Hadamard can only
        # finish 128-column groups the CTA owns. This is the fusion's whole
        # cost -- it forfeits the narrow-BLOCK_N decode tiles that starved-N
        # shapes want -- which is why _fuse_output_had gates on N, not just on
        # divisibility.
        configs = [c for c in configs if c.kwargs["BLOCK_N"] % 128 == 0]
    else:
        configs = [c for c in configs
                   if (c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"], c.num_warps)
                   not in _FUSE_OUT_ONLY_CFG]
    if bits == 1:
        # This Triton build's TTGIR pass crashes on the K_BITS=1 decode tiles
        # ([*, 8] per sub-tile) with 8 warps at BLOCK_N=16.
        configs = [c for c in configs if not (c.num_warps >= 8 and c.kwargs["BLOCK_N"] <= 16)]
    if m == 1:
        out = [c for c in configs if c.kwargs["BLOCK_M"] == 16]
        if bits != 3:
            # The BLOCK_N=32/K=32 pair exists only for the bits=3 M1 run-funnel
            # decode; other widths never see it (identical autotune behaviour
            # to before it was added).
            out = [c for c in out if not (c.kwargs["BLOCK_N"] == 32 and c.kwargs["BLOCK_K"] == 32)]
            if fast_ok:
                # N-bucketed decode pools (bits != 3), bucketed per width class:
                #
                # bits=4 (light M1 accumulator, [2,2,2,NN,8,4]): starved-N
                # shapes (down_proj class, N<=8192 at BN128 = 32-40 CTAs)
                # take BN32 tiles (measured 265-280 vs 242-244 GB/s for
                # BN128/BK128); large N keeps BN128/BK128 and gains the
                # BN64/BK256 deep-K tile (329-425 vs 321-410 GB/s). BN64/BK128
                # stays in both pools (weakest b4 member for starved-N at
                # 235-239 but needed nowhere else to regress).
                #
                # Split-K-eligible bits=4 shapes (see _m1_splitk_plan): the
                # CTA count comes from the K splits, so the widest windows
                # win outright (measured at S=8: BN32/BK256 383-455 GB/s,
                # BN64/BK256 386-452, vs 367-375 for the BK128 tiles).
                #
                # Other widths (1,2,5,6,7,8) run the heavy-accumulator M1
                # kernels (2-4 fp32 tensors of 256*NN elements per CTA, or
                # the staged-gather generic path): per-CTA occupancy collapses
                # at BN>=64 unless N is huge, so BN32 tiles win everywhere up
                # to large N. Measured (bits=6, N=12288): BN32/BK128 554 vs
                # BN64/BK128 203 GB/s; (bits=2, N=4096): 114 vs 34 GB/s; at
                # the huge lm_head N=248320 (bits=6) BN64/BK128 wins (685 vs
                # 667), so the large-N pool keeps it.
                cu, wave, is_amd = _dev_caps()
                # The pools below bucket by "CTAs per wave" against the XTX's 48
                # CUs. A part with far fewer CUs is never CTA-starved at these N,
                # so the narrow deep-K tiles that win there also win at large N;
                # (32,128) measured 67.6 GB/s vs 65.5 for the (32,256) pick on
                # gfx1150 at 1x4096x12288, and is pruned out entirely below.
                low_cu = cu <= 16
                if bits == 4:
                    if n <= 32768 and n % 256 == 0 and k % 256 == 0 and k // 16 >= 256:
                        # Split-K-eligible bits=4 shape (see _m1_splitk_plan):
                        # the CTA count comes from the K splits, so the widest
                        # windows win outright (down S=8: BN32/BK256 383-455
                        # GB/s, BN64/BK256 386-452, vs 367-375 for the BK128
                        # tiles; g/u N=17408 S=4: BN64/BK256 378 vs 354-358
                        # for BN128).
                        pool = ((32, 256), (64, 256))
                        if low_cu:
                            pool += ((32, 128),)
                    elif n <= 8192:
                        pool = ((32, 128), (32, 256), (64, 128))
                    else:
                        pool = ((128, 128), (64, 256), (64, 128))
                elif bits == 6 and n <= 16384 and n % 256 == 0 and k % 256 == 0 and k // 16 >= 256:
                    # Split-K-eligible bits=6 linear-class shape: BK128 stays
                    # the winner for the b6 funnel decode (6bpw MLP split-4:
                    # 559-570 GB/s vs 442-553 classic)
                    pool = ((64, 128), (32, 128))
                elif n <= 16384:
                    pool = ((32, 128), (32, 256))
                else:
                    pool = ((64, 128), (32, 128))
                out = [c for c in out
                       if (c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"]) in pool]
                return _prefer_warps(out) if out else configs
        else:
            # bits=3 run-funnel decode: BLOCK_N=128 tiles collapse to the old
            # per-code path's rate (measured 229 GB/s at 1x5120x248320 vs 477+
            # for the narrow tiles), so keep the M1 list tight enough that a
            # cold-clock autotune pass cannot lock one in.
            out = [c for c in out if c.kwargs["BLOCK_N"] != 128]
        if not fast_ok:
            out = [c for c in out if c.kwargs["BLOCK_N"] <= 64 and c.kwargs["BLOCK_K"] <= 64]
        return out if out else configs
    if not fast_ok:
        small = [c for c in configs if c.kwargs["BLOCK_N"] <= 64 and c.kwargs["BLOCK_K"] <= 64]
        return small if small else configs
    return configs


def _exl3_gemm_configs():
    import os
    cfg_spec = os.environ.get("EXL3_GEMM_CONFIGS")
    if cfg_spec:
        # Format: "BM,BN,BK,GM:nw:ns;..." for quick manual sweeps.
        configs = []
        for part in cfg_spec.split(";"):
            part = part.strip()
            if not part:
                continue
            dims, _, rest = part.partition(":")
            bm, bn, bk, gm = (int(x) for x in dims.split(","))
            nw = int(rest.split(":")[0]) if rest else 4
            ns = int(rest.split(":")[1]) if ":" in rest else 3
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm}, num_warps=nw, num_stages=ns))
        return configs
    # Default autotune set. The decode path (M=1, bits=4) is pure bandwidth:
    # wide N/K tiles amortize the staged decode, and the no-dot reduction
    # path removes the tensor-core shape constraint. All M==1 configs are
    # within a few percent of each other at operating clocks, so a cold-clock
    # autotune pass cannot lock in a slow one.
    #
    # M == 1 (decode GEMV). The per-N pools are enforced by
    # _exl3_gemm_early_prune: starved-N shapes (down_proj-class, N/BLOCK_N
    # under one wave on 48 CUs) get the BN32 tiles; large-N shapes (gate/up,
    # lm_head) keep BN128/BK128 and gain the BN64/BK256 deep-K tile. Measured
    # (RX 7900 XTX, 4 bpw, L2-cold layer sweeps incl. both hadamards):
    #   down N=4096:  BN32/BK128 265, BN32/BK256 280 GB/s (was 242 at BN128)
    #   down N=5120:  BN32/BK128 252, BN32/BK256 251 GB/s (was 239-244)
    #   g/u   N=12288: BN128/BK128 406-410, BN64/BK256 425 GB/s
    #   g/u   N=17408: BN128/BK128 321-322, BN64/BK256 334 GB/s
    return [
        # M == 1 (decode GEMV), small-N pool (CTA-starved shapes)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # wave64 twins of the two decode tiles above. On a 64-wide wave these
        # are the same 128 lanes the nw=4 entries give a wave32 part, and
        # _prefer_warps selects between them by device. Measured on gfx1150,
        # 1x4096x12288 bits=4: 67.6 and 65.5 GB/s, vs 45.7 for the stock pick.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=2, num_stages=2),
        # M == 1, large-N pool
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=8, num_stages=3),
        # M == 1 fallback for shapes outside both pools (never autotuned away:
        # kept so the pruned list is never empty on unusual shapes)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M == 1, FUSED-OUTPUT-HADAMARD-ONLY tiles. The output fusion forces
        # BLOCK_N % 128 == 0, which leaves the stock list with exactly one
        # member ((128,128) at 8 warps) and no autotune choice at all. These
        # give it one. They are pruned out of every non-fused invocation
        # (_FUSE_OUT_ONLY_CFG) so the measured pools above are untouched.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M == 1, bits=3 run-funnel decode: the shared-funnel tile is narrow
        # (8 x BLOCK_N/2), so single-warp narrow blocks win the wide-N stream
        # (measured 484 vs 431 GB/s at 1x5120x248320). Pruned out for every
        # other invocation by _exl3_gemm_early_prune.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=2, num_stages=3),
        # Generic path (other bit widths) and small shapes
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M > 1 (prefill)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=8, num_stages=2),
    ]


# (BLOCK_N, BLOCK_K, num_warps) of the pool members that exist ONLY to give the
# fused-output-Hadamard route something to autotune over. Dropped everywhere
# else so the measured pools stay exactly as they were.
_FUSE_OUT_ONLY_CFG = {(128, 128, 2), (128, 128, 4), (128, 256, 4)}


_WARNED_UNCURATED = set()


def _warn_uncurated_strict(named_args, kwargs, m):
    """Name the shape once, so it gets pinned rather than silently autotuned."""
    def g(k):
        return kwargs.get(k, named_args.get(k))
    key = (g("K_dim"), g("N"), g("K_BITS"), g("M_BUCKET"))
    if key in _WARNED_UNCURATED:
        return
    _WARNED_UNCURATED.add(key)
    print(f" !! EXL3_GEMM_PIN=strict: (K_dim, N, K_BITS, M_BUCKET)={key} (M={m}) has "
          f"no _GEMM_PINNED entry and M > 1 pools are uncurated; autotuning instead "
          f"of pinning. Measure it and add it to _GEMM_PINNED for a deterministic run.",
          flush=True)


def _exl3_gemm_early_prune(configs, named_args, **kwargs):
    """Pin, else strict-first, else the measured pool selector.

    Wrapping rather than editing _exl3_gemm_prune_pool's four return paths keeps
    the pool logic exactly as it was measured, and guarantees the pin applies on
    every path rather than only the one someone remembered to patch.
    """
    pin = _gemm_pinned_config(named_args, kwargs)
    if pin is not None:
        # A single config means triton.autotune skips benchmarking entirely, so
        # the cold-clock race that produced the non-deterministic pick is gone.
        return [pin]
    out = _exl3_gemm_prune_pool(configs, named_args, **kwargs)
    if os.environ.get("EXL3_GEMM_PIN", "1").lower() == "strict" and out:
        # out[0] is a pure function of the shape either way, but it is only a
        # GOOD tile where the pool was curated: _exl3_gemm_prune_pool N-buckets
        # and _prefer_warps-filters the m == 1 branch, and every member of those
        # pools was measured. Every other M falls through to a bare
        # `return configs`, where out[0] is just the head of the literal config
        # list -- an m == 1 decode tile. Collapsing there handed all eight dense
        # verify sites a CTA-starved tile for ~100 ms/round at ndt 4, and
        # _m_bucket(9) == 16 re-arms it on any bucket nobody has pinned.
        # Autotuning an unmeasured shape beats deterministically guessing it.
        m = kwargs.get("M", named_args.get("M"))
        if m is None:
            m = kwargs.get("M_BUCKET", named_args.get("M_BUCKET"))
        if m == 1:
            return [out[0]]
        _warn_uncurated_strict(named_args, kwargs, m)
    return out


_PRUNE = {"early_config_prune": _exl3_gemm_early_prune}


@triton.jit
def _had_x_tile(
    x_ptr, suh_ptr, k0, stride_xk, K_dim, r_scale,
    FUSE_HAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Input Hadamard for the K-tile this CTA is about to consume, replacing the
    # separate had_r_128_triton launch. The transform is block-diagonal over
    # aligned 128-element groups of K, so a CTA owning [k0, k0 + BLOCK_K) can
    # derive its own slice — hence the BLOCK_K % 128 == 0 and k0 % 128 == 0
    # requirements enforced by _exl3_gemm_early_prune and _m1_splitk_plan's
    # NK-aligned k_base. Every CTA along N repeats this work; that is one
    # BLOCK_K-wide transform against a whole tile of weight decodes, and x is
    # L2-resident.
    #
    # Bit-exactness with _had_r_128_kernel is mandatory (the unfused path feeds
    # the GEMM a half xh): pre-scale in half, butterfly in fp32, * r_scale,
    # then round to half HERE — keeping fp32 across the multiply would change
    # the result.
    if FUSE_HAD:
        R: tl.constexpr = BLOCK_K // 128
        off = k0 + tl.arange(0, R)[:, None] * 128 + tl.arange(0, 128)[None, :]
        m = off < K_dim
        xv = tl.load(x_ptr + off * stride_xk, mask=m, other=0.0)
        pre = tl.load(suh_ptr + off, mask=m, other=0.0)
        v = (xv * pre).to(tl.float32)
        v = _had_stage(v, R, 1)
        v = _had_stage(v, R, 2)
        v = _had_stage(v, R, 4)
        v = _had_stage(v, R, 8)
        v = _had_stage(v, R, 16)
        v = _had_stage(v, R, 32)
        v = _had_stage(v, R, 64)
        v = v * r_scale
        return tl.reshape(v.to(tl.float16), (BLOCK_K // 16, 16))
    else:
        return tl.zeros((BLOCK_K // 16, 16), dtype=tl.float16)


@triton.jit
def _store_out_had(
    y_ptr, y_off, offs_n, mask_n, acc, svh_ptr, r_scale,
    FUSE_OUT_HAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Output Hadamard for the N-tile this CTA just finished, replacing the
    # trailing had_r_128_triton launch. Block-diagonal over aligned 128-element
    # groups of N, so a CTA owning [n0, n0 + BLOCK_N) can finish its own
    # columns -- hence BLOCK_N % 128 == 0, enforced by _exl3_gemm_early_prune,
    # and n0 % 128 == 0, which follows from it.
    #
    # Bit-exactness with gemm-store + _had_r_128_kernel(POST_SCALED) is
    # mandatory: the unfused route ROUNDS the fp32 accumulator to half when it
    # stores y and reads that half back, so the round-trip through half has to
    # happen here too, before the butterfly. The post-scale is then a half
    # multiply, matching the C++ kernel's ordering.
    if FUSE_OUT_HAD:
        R: tl.constexpr = BLOCK_N // 128
        v = tl.reshape(acc.to(tl.float16).to(tl.float32), (R, 128))
        v = _had_stage(v, R, 1)
        v = _had_stage(v, R, 2)
        v = _had_stage(v, R, 4)
        v = _had_stage(v, R, 8)
        v = _had_stage(v, R, 16)
        v = _had_stage(v, R, 32)
        v = _had_stage(v, R, 64)
        v = v * r_scale
        post = tl.load(svh_ptr + offs_n, mask=mask_n, other=0.0)
        out = v.to(tl.float16) * tl.reshape(post, (R, 128))
        tl.store(y_ptr + y_off, tl.reshape(out, (BLOCK_N,)), mask=mask_n)
    else:
        tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _x_sub16(
    xhad, x_ptr, ktb, stride_xk,
    KI: tl.constexpr,
    NK: tl.constexpr,
    FUSE_HAD: tl.constexpr,
):
    if FUSE_HAD:
        # Select through the bit pattern, not the value: a float masked sum
        # would flush a -0.0 element to +0.0.
        b = tl.cast(xhad, tl.int16, bitcast=True).to(tl.int32)
        sel = (tl.arange(0, NK) == KI)[:, None]
        return tl.cast(tl.sum(tl.where(sel, b, 0), 0).to(tl.int16), tl.float16, bitcast=True)
    else:
        return tl.load(x_ptr + (ktb * 16 + tl.arange(0, 16)) * stride_xk)


# ---------------------------------------------------------------------------
# Constexpr fast-path gate
#
# The decode fast paths are selected on `N % BLOCK_N == 0 and K_dim % BLOCK_K
# == 0`. N and K_dim are RUNTIME kernel arguments, so that predicate is a
# Triton tensor, not a Python bool: Triton emits an scf.if and compiles EVERY
# decode branch into EVERY binary. Measured on gfx1150, bits=6 / BN32 / BK128:
# the dead generic tl.gather branch is 7808 of 13698 instructions and drags the
# kernel to 256 VGPR / 291 VGPR spills / 676 B-per-lane scratch / 94 KB code /
# 4 waves per SIMD. Folding the predicate gives 123 VGPR / 0 spills / 9 KB /
# 8 waves, executing exactly the same instructions.
#
# For power-of-two BLOCK_N, `N % BLOCK_N == 0` <=> `pow2div(N) % BLOCK_N == 0`,
# so the capped 2-adic valuation of N and K_dim is an EXACT constexpr
# substitute -- provided BLOCK_N and BLOCK_K do not exceed the cap, which the
# kernel checks before trusting it. A 0 disables the gate and restores the
# runtime-argument codegen unchanged, so EXL3_CONSTEXPR_GATE=0 is a real
# control arm rather than a different slow path.
# ---------------------------------------------------------------------------

_GATE_CAP = 256
_GATE_OFF = ("0", "off", "false", "no")


def _gate_div(v: int) -> int:
    # read per call, not at import, so one process can A/B both codegens
    if os.environ.get("EXL3_CONSTEXPR_GATE", "1").lower() in _GATE_OFF:
        return 0
    return min(_GATE_CAP, v & -v)


@triton.jit
def _decode_word_pair(
    low_u32, high_u32, shift,
    SHIFT_FITS_32: tl.constexpr,
    CB: tl.constexpr,
):
    """Funnel-shift a (low, high) u32 word pair into the 16-bit codebook index
    (generic-path helper) and decode it via _decode_u16."""
    if SHIFT_FITS_32:
        # 32-bit funnel: shift is guaranteed in [0,31] for K_BITS in {1,2,4}.
        neg_shift = tl.minimum(32 - shift, 31)
        windows = ((low_u32 >> shift) | (high_u32 << neg_shift)) & 0xFFFF
    else:
        low64 = (low_u32.to(tl.int64) & 0xFFFFFFFF) | ((high_u32.to(tl.int64) & 0xFFFFFFFF) << 32)
        windows = ((low64 >> shift) & 0xFFFF).to(tl.uint32)
    return _decode_u16(windows.to(tl.uint32), CB)


@triton.jit
def _funnel6(lo, hi, s):
    """bits=6 funnel: 16-bit code window from a (lo, hi) u32 word pair where
    hi is the word *preceding* lo in the tile's virtual bit stream, so the
    window can start past bit 31 of lo and the base word flips. lo, hi are
    [NN, 16] u32; s is an [S] shift vector; returns [S, NN, 16] u32 codes."""
    sel = s >= 32
    s32 = s & 31
    ns = tl.minimum(32 - s32, 31)
    base = tl.where(sel[:, None, None], hi[None, :, :], lo[None, :, :])
    second = tl.where(sel[:, None, None], lo[None, :, :], hi[None, :, :])
    return ((base >> s32[:, None, None]) | (second << ns[:, None, None])) & 0xFFFF


@triton.jit
def _decode_u16(w_u32, CB: tl.constexpr):
    """Inline arithmetic decode of 16-bit codebook indices (matches
    decode_3inst in the C++ reference): ~3 ALU ops instead of a 65536-entry
    LUT gather. Elementwise over u32 codes; returns fp16 weights."""
    if CB == 0:
        w_u32 = (w_u32 * 89226354 + 64248484) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    elif CB == 1:
        w_u32 = (w_u32 * 0xCBAC1FED) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    else:  # CB == 2 (mul1)
        w_u32 = (w_u32 * 0x83DCD12D) & 0xFFFFFFFF
        # byte sum: dp4a(x, 0x01010101, 0x6400) emulated
        db0 = w_u32 & 0xFF
        db1 = (w_u32 >> 8) & 0xFF
        db2 = (w_u32 >> 16) & 0xFF
        db3 = (w_u32 >> 24) & 0xFF
        w_u32 = (db0 + db1 + db2 + db3 + 0x6400) & 0xFFFF

    # bitcast low/high 16 bits to fp16 then add (cb 0/1), or fma (cb 2)
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


@triton.autotune(configs=_exl3_gemm_configs(),
                 key=["M_BUCKET", "N", "K_dim", "K_BITS", "N_PACKED", "CB", "FUSE_HAD",
                      "FUSE_OUT_HAD"],
                 prune_configs_by=_PRUNE)
@triton.jit
def _fused_dequant_gemm_kernel(
    x_ptr, y_ptr,
    trellis_ptr,
    perm_i_ptr,
    mrow_ptr,
    M, N, K_dim,
    M_BUCKET,          # autotune key only (see _m_bucket); unused in the body
    stride_xm, stride_xk,
    stride_tk, stride_tn,
    stride_ym, stride_yn,
    stride_ys,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_BITS: tl.constexpr,
    N_PACKED: tl.constexpr,
    CB: tl.constexpr,
    M1: tl.constexpr,
    SPLITS: tl.constexpr,
    N_DIV: tl.constexpr = 0,
    K_DIV: tl.constexpr = 0,
    FUSE_HAD: tl.constexpr = False,
    suh_ptr = None,
    FUSE_OUT_HAD: tl.constexpr = False,
    svh_ptr = None,
    had_r_scale = _RSCALE_128,
):
    NK: tl.constexpr = BLOCK_K // 16   # k-sub-tiles per weight tile
    NN: tl.constexpr = BLOCK_N // 16   # n-sub-tiles per weight tile
    N_U32: tl.constexpr = K_BITS * 256 // 32
    # For K_BITS in {1,2,4} the funnel shift never exceeds 31, so the 64-bit
    # funnel (high<<32 | low) >> shift can be computed with 32-bit ops only,
    # avoiding expensive emulated 64-bit arithmetic on RDNA3.
    SHIFT_FITS_32: tl.constexpr = (K_BITS == 1) | (K_BITS == 2) | (K_BITS == 4)
    if N_DIV > 0 and BLOCK_N <= 256 and BLOCK_K <= 256:
        FULL = ((N_DIV % BLOCK_N) + (K_DIV % BLOCK_K)) == 0
    else:
        FULL = (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if SPLITS == 1:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_split = 0
    else:
        # M == 1 split-K GEMV: axis 0 tiles N, axis 1 slices the K loop. The
        # generic pid math above is unused (M == 1, GROUP_M irrelevant).
        pid_m = 0
        pid_n = pid % num_pid_n
        pid_split = pid // num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    tu32_ptr = trellis_ptr.to(tl.pointer_type(tl.uint32))
    stride_tk_u32 = stride_tk // 2
    stride_tn_u32 = stride_tn // 2
    base_n = (pid_n * NN) * stride_tn_u32

    n_k_tiles_total = K_dim // 16
    if SPLITS == 1:
        k_base = 0
        n_outer = tl.cdiv(n_k_tiles_total, NK)
    else:
        # Contiguous K slice for this split; splits beyond the remainder get
        # an empty range (their partial stays unwritten only if fully empty —
        # avoided by requiring K to have at least SPLITS * NK k-sub-tiles).
        tiles_per_split = tl.cdiv(n_k_tiles_total, SPLITS * NK) * NK
        k_base = pid_split * tiles_per_split
        n_outer = tl.cdiv(min(tiles_per_split, n_k_tiles_total - k_base), NK)

    if K_BITS == 4 and FULL:
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

        if M1:
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
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_base + k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
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
                    xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD).to(tl.float32)
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
            if SPLITS == 1:
                _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,
                               had_r_scale, FUSE_OUT_HAD, BLOCK_N)
            else:
                # Split-K partial: row pid_split of the [SPLITS, N] fp32 buffer.
                # stride_yn is the (unit) column stride of the partials buffer.
                tl.store(y_ptr + pid_split * stride_ys + offs_n * stride_yn, acc, mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words = tl.load(row + wc)
                    safe = (ktb > 0) | (base_n > 0)
                    m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                    w31 = tl.load(row + nj8 * 32 + 31)                # [NN]
                    w31_bcast = tl.reshape(
                        tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
                    )
                    m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
                    q = ((words[None, :] >> sh[:, None]) |
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # reorder (ch, rh, p, nj, cl, q_) -> (r, c) statically
                    w = tl.reshape(w, (2, 2, 2, NN, 8, 4))
                    w = tl.permute(w, (1, 5, 2, 3, 0, 4))   # (rh, q_, p, nj, ch, cl)
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif K_BITS == 6 and FULL:
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

        if M1:
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
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_base + k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
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
                    xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD).to(tl.float32)
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
            if SPLITS == 1:
                _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, tl.reshape(out, (BLOCK_N,)),
                               svh_ptr, had_r_scale, FUSE_OUT_HAD, BLOCK_N)
            else:
                # Split-K partial (see the bits=4 M1 path): fp32 row of the
                # [SPLITS, N] buffer, summed + output-Hadamard by the reduce.
                tl.store(y_ptr + pid_split * stride_ys + offs_n * stride_yn,
                         tl.reshape(out, (BLOCK_N,)), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words2 = tl.reshape(tl.load(row + wbase), (NN, 16))
                    wone2 = tl.reshape(tl.load(row + wone), (NN, 16))
                    wtwo2 = tl.reshape(tl.load(row + wtwo), (NN, 16))
                    wneg2 = tl.reshape(tl.load(row + wneg), (NN, 16))
                    d0 = _decode_u16(_funnel6(words2, wneg2, C0 - sh6), CB)
                    d1 = _decode_u16(_funnel6(wone2, words2, C1 - sh6), CB)
                    d2 = _decode_u16(_funnel6(wtwo2, wone2, C2 - sh6), CB)
                    d3 = _decode_u16(_funnel6(wtwo2, wone2, C3 - sh6), CB)
                    # reorder to (r, n): decode is [jj, nj, a]; reshape to
                    # (j1, j0, nj, cA, a0) — a = 8*cA + a0 with a0 = a&1 and
                    # cA = a>>1 = c%8 — then permute to
                    # (j1, a0, b1, j0, nj, b0, cA) and fold
                    # r = 8*j1 + 4*a0 + 2*b1 + j0, n = 16*nj + 8*b0 + cA.
                    P0 = tl.permute(tl.reshape(d0, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P1 = tl.permute(tl.reshape(d1, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P2 = tl.permute(tl.reshape(d2, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P3 = tl.permute(tl.reshape(d3, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    J0 = tl.join(P0, P2)      # (j1, j0, a0, nj, cA, b1)
                    J1 = tl.join(P1, P3)
                    Wt = tl.join(J0, J1)      # (j1, j0, a0, nj, cA, b1, b0)
                    Wt = tl.permute(Wt, (0, 2, 5, 1, 3, 6, 4))
                    w = tl.reshape(Wt, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif (K_BITS == 1 or K_BITS == 2 or K_BITS == 8) and FULL:
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

        if M1:
            # GEMV: fold the permutation into the x broadcast (see bits=4).
            r16 = tl.arange(0, 16)
            if K_BITS == 1:
                acc7 = tl.zeros((2, 2, 2, 2, 2, NN, 8), dtype=tl.float32)  # (r2,r1,c3,r3,r0,nj,cl)
            elif K_BITS == 2:
                acc7 = tl.zeros((2, 2, 2, 2, NN, 8, 2), dtype=tl.float32)  # (r1,c3,r3,r0,nj,cl,r2)
            else:
                acc7 = tl.zeros((2, 2, NN, 8, 2, 2, 2), dtype=tl.float32)  # (r3,r0,nj,cl,r2,r1,c3)
            for k_outer in range(n_outer):
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
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
                    xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD).to(tl.float32)
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
            _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,
                           had_r_scale, FUSE_OUT_HAD, BLOCK_N)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # static reorder (row bits, word bits) -> (r, c)
                    if K_BITS == 1:
                        w = tl.reshape(w, (2, 2, 2, 2, 2, NN, 8))        # (r2,r1,c3,r3,r0,nj,cl)
                        w = tl.permute(w, (3, 0, 1, 4, 5, 2, 6))         # (r3,r2,r1,r0,nj,c3,cl)
                    elif K_BITS == 2:
                        w = tl.reshape(w, (2, 2, 2, 2, NN, 8, 2))        # (r1,c3,r3,r0,nj,cl,r2)
                        w = tl.permute(w, (2, 6, 0, 3, 4, 1, 5))         # (r3,r2,r1,r0,nj,c3,cl)
                    else:
                        w = tl.reshape(w, (2, 2, NN, 8, 2, 2, 2))        # (r3,r0,nj,cl,r2,r1,c3)
                        w = tl.permute(w, (0, 4, 5, 1, 2, 6, 3))         # (r3,r2,r1,r0,nj,c3,cl)
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif K_BITS == 3 and FULL:
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

        if M1:
            # acc[g, nj*8 + clc]; each of the 4 rows of a group accumulates
            # into the shared (g, column) slot with its own x element.
            acc8 = tl.zeros((8, NN * 8), dtype=tl.float32)
            for k_outer in range(n_outer):
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n3
                    A = tl.load(row + w_a)
                    B = tl.load(row + w_b)
                    # no 32-bit mask on Q: the extraction masks drop every bit
                    # above 24, including the bit-31 pollution B<<31 at base 0
                    Q = (A >> base_g[:, None]) | (B << neg_g[:, None])
                    xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD).to(tl.float32)
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
            _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,
                           had_r_scale, FUSE_OUT_HAD, BLOCK_N)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n3
                    A = tl.load(row + w_a)
                    B = tl.load(row + w_b)
                    Q = (A >> base_g[:, None]) | (B << neg_g[:, None])
                    # (v, c3, nj, clc, p, q) -> (r = 8q + 2v + p, n = 16*nj +
                    # 8*c3 + clc) with m = 2q + p
                    q0 = tl.reshape((Q >> 9) & 0xFFFF, (4, 2, NN, 8))
                    q1 = tl.reshape((Q >> 6) & 0xFFFF, (4, 2, NN, 8))
                    q2 = tl.reshape((Q >> 3) & 0xFFFF, (4, 2, NN, 8))
                    q3 = tl.reshape(Q & 0xFFFF, (4, 2, NN, 8))
                    wq = tl.join(tl.join(q0, q1), tl.join(q2, q3))
                    wq = tl.permute(wq, (5, 0, 4, 2, 1, 3)).reshape(16, BLOCK_N)
                    w = _decode_u16(wq.to(tl.uint32), CB)
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif (K_BITS == 5 or K_BITS == 7) and FULL:
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

        if M1:
            accm = tl.zeros((32, NN * 8), dtype=tl.float32)
            for k_outer in range(n_outer):
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    lo = tl.load(row + w_lo)                       # [32, NN*8]
                    hi = tl.load(row + w_hi)
                    q = ((lo >> sh_vec[:, None]) |
                         (hi << neg_vec[:, None])) & 0xFFFF
                    w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                    xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD).to(tl.float32)
                    xb = tl.reshape(
                        tl.broadcast_to(tl.reshape(xk, (16, 1, 1, 1)), (16, 2, NN, 8)),
                        (32, NN * 8),
                    )
                    accm += w_dec * xb
            # (r, c3, nj, cl) -> sum over r -> (c3, nj, cl) -> n = 16*nj + 8*c3 + cl
            s = tl.sum(tl.reshape(accm, (16, 2, NN, 8)), 0)
            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
            _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,
                           had_r_scale, FUSE_OUT_HAD, BLOCK_N)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    lo = tl.load(row + w_lo)
                    hi = tl.load(row + w_hi)
                    q = ((lo >> sh_vec[:, None]) |
                         (hi << neg_vec[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # (r, c3, nj, cl) -> (r, nj, c3, cl) -> [16, BLOCK_N]
                    w = tl.permute(tl.reshape(w, (16, 2, NN, 8)), (0, 2, 1, 3))
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
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

        if M1:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_outer in range(n_outer):
                xhad = _had_x_tile(x_ptr, suh_ptr, (k_outer * NK) * 16, stride_xk,
                                   K_dim, had_r_scale, FUSE_HAD, BLOCK_K)
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
                    if FUSE_HAD:
                        # K_dim % 128 == 0 is required for fusion, so a 128
                        # group is entirely in or out of range and the tile's
                        # own bounds mask covers k_ok.
                        xk = _x_sub16(xhad, x_ptr, ktb, stride_xk, ki, NK, FUSE_HAD)
                    else:
                        xk = tl.load(
                            x_ptr + (ktb * 16 + r16) * stride_xk,
                            mask=k_ok & (r16 < 16), other=0.0,
                        )
                    acc += tl.sum(w.to(tl.float32) * xk.to(tl.float32)[:, None], 0)
            _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,
                           had_r_scale, FUSE_OUT_HAD, BLOCK_N)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
                    k_off = ktb * 16 + r16
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None] & (k_off < K_dim)[None, :],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )


def exl3_gemm_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    y: torch.Tensor,
    lut: torch.Tensor,
    perm_i: torch.Tensor,
    K_bits: int,
    tiles_n: int,
    cb: int = 0,
    splits: int = 1,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
) -> None:
    if not has_triton:
        raise RuntimeError("exl3_gemm_triton requires Triton")
    """Fused EXL3 dequant + fp16 matmul. Does NOT materialize the weight matrix.

    ``splits > 1`` (M == 1 bits=4 full-tile shapes only) runs the split-K GEMV:
    ``y`` must then be the [splits, N] fp32 partials buffer from
    _get_splitk_buf, to be summed by _m1_split_reduce_had afterwards.

    Pass ``suh`` to run the input Hadamard inside the kernel: ``x`` is then the
    raw (untransformed) input and the caller's had_r_128_triton launch goes
    away. M == 1 only, and only for shapes whose K is a multiple of 128.

    Pass ``svh`` to run the output Hadamard + post-scale inside the kernel's
    store, so ``y`` comes out finished and the trailing had_r_128_triton
    launch goes away too. M == 1, half output, N a multiple of 128, and never
    together with ``splits > 1`` (that route folds the same transform into
    _m1_split_reduce_had instead).
    """
    M, K_dim = x.shape
    N = y.shape[1]

    # The split store exists only in the bits=4/6 M==1 fast branches; never
    # let a split request reach any other path.
    if not (M == 1 and K_bits in (4, 6)):
        splits = 1

    if suh is not None:
        assert M == 1 and K_dim % 128 == 0 and N % 128 == 0, \
            f"exl3_gemm_triton: fused input Hadamard needs M==1 and 128-divisible N/K, got {M}x{K_dim}x{N}"
        assert x.dtype == torch.half and suh.dtype == torch.half
    if svh is not None:
        assert M == 1 and N % 128 == 0 and splits == 1, \
            f"exl3_gemm_triton: fused output Hadamard needs M==1, splits==1 and " \
            f"128-divisible N, got {M}x{K_dim}x{N} splits={splits}"
        # the epilogue reproduces _had_r_128_kernel's HALF path exactly (round the
        # accumulator to half, butterfly in fp32, half post-scale); the fp32 I/O
        # variant has a different rounding order and is not implemented here
        assert y.dtype == torch.half and svh.dtype == torch.half

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]) * splits,
    )

    _fused_dequant_gemm_kernel[grid](
        x, y,
        trellis,
        perm_i,
        _get_m_row_offsets(K_bits, x.device) if K_bits in _M_ROW_OFFSETS else perm_i,
        M, N, K_dim,
        _m_bucket(M),
        x.stride(0), x.stride(1),
        trellis.stride(0), trellis.stride(1),
        y.stride(0), y.stride(1),
        y.stride(0),
        N_DIV=_gate_div(N),
        K_DIV=_gate_div(K_dim),
        K_BITS=K_bits,
        N_PACKED=trellis.shape[-1],
        CB=cb,
        M1=(M == 1),
        SPLITS=splits,
        FUSE_HAD=suh is not None,
        suh_ptr=suh,
        FUSE_OUT_HAD=svh is not None,
        svh_ptr=svh,
        had_r_scale=_RSCALE_128,
    )


# ---------------------------------------------------------------------------
# LinearEXL3 composition: had_r_128_triton -> gemm -> had_r_128_triton
# ---------------------------------------------------------------------------

# Widths whose M1 branch carries a LIGHT accumulator (a single (BLOCK_N,) fp32
# vector) and so keeps the same fp32 reduction rounding when the fused input
# transform's registers join the branch. See _fuse_input_had.
_IN_HAD_EXACT_BITS = (1, 2, 8)


def _fuse_input_had(K_bits: int, M: int, in_features: int, out_features: int) -> bool:
    """Whether to run the input Hadamard inside the GEMM kernel (one launch
    fewer per linear on the decode path).

    Read per call, not at import, so an A/B harness can flip
    EXL3_FUSE_INPUT_HAD in one process.

    ONLY bits 1, 2 and 8 are bit-exact, and that restriction is measured, not
    theoretical. The transform itself is provably exact -- a probe kernel storing
    _had_x_tile's output matches had_r_128_triton byte for byte at every width
    tested, including the failing ones. The divergence is entirely in the fp32
    reduction: with identical x and identical weights, `acc += w * x` rounds
    differently once the transform's live registers are added to the branch, i.e.
    the backend's decision to contract the multiply-add into an FMA flips. So the
    exact widths are exactly the ones with a LIGHT accumulator -- 1/2/8 carry a
    single (BLOCK_N,) fp32 vector, while 3, 4, 5, 6 and 7 carry 512-2048 fp32
    elements per CTA and all drift.

    Measured on gfx1150, Triton 3.5.1, pinned tiles (tests/test_exl3_triton_fused_had.py):
        bits=6 BK128  max|diff| 3.906e-03   bits=5 BK128  max|diff| 4.517e-03
        bits=4 BK256  max|diff| 7.812e-03 (bits=4 at BK128 happens to be exact,
                                           but BLOCK_K is an autotune outcome the
                                           host gate cannot pin, so 4 is excluded)
        bits=3, bits=7                     the originally documented pair

    The earlier "bits 3 and 7 only" reading came from a test that did not pass
    N_DIV/K_DIV, so FULL was a runtime predicate and Triton compiled all five
    decode branches into every binary (291 VGPR spills, campaign brief). Under
    that much spilling the pressure difference between the arms disappears and
    the drift hides. With the constexpr gate -- what production actually runs --
    it does not.
    """
    if os.environ.get("EXL3_FUSE_INPUT_HAD", "0").lower() in ("0", "", "off", "no", "false"):
        return False
    return (M == 1 and K_bits in _IN_HAD_EXACT_BITS
            and in_features % 128 == 0 and out_features % 128 == 0)


# Below this N the forced BLOCK_N=128 tile costs more than the launch it saves:
# the grid is N/128 CTAs against 8 WGPs, so N=4096 is 32 CTAs (4 per WGP) and
# N=1024 is 8 (one each, no latency hiding at all). Starved-N shapes are exactly
# the ones whose measured M1 winner is a BN32 tile (265-280 vs 242-244 GB/s,
# ledger-arms2). Overridable for A/B: EXL3_FUSE_OUTPUT_HAD_MIN_N.
_FUSE_OUT_MIN_N = 4096


def _fuse_output_had(K_bits: int, M: int, in_features: int, out_features: int) -> bool:
    """Whether to run the output Hadamard + post-scale inside the GEMM's store.

    Read per call, not at import, so an A/B harness can flip the flag in one
    process. Unlike the input fusion this one is not free: it prunes the pool to
    BLOCK_N % 128 == 0, so it is gated on N being wide enough for a 128-column
    tile to keep the part busy. Every bit width is bit-exact here -- the
    epilogue only reads the finished accumulator, so it cannot perturb the
    decode loop's register pressure the way the input fusion does at bits 3/7.
    """
    if os.environ.get("EXL3_FUSE_OUTPUT_HAD", "0").lower() in ("0", "", "off", "no", "false"):
        return False
    min_n = int(os.environ.get("EXL3_FUSE_OUTPUT_HAD_MIN_N", _FUSE_OUT_MIN_N))
    return M == 1 and out_features % 128 == 0 and out_features >= min_n


def _linear_exl3_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    xh: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
) -> None:
    """Complete EXL3 linear forward into pre-allocated buffers.

    Writes the result to ``y``, and the Hadamard-transformed input to ``xh``
    unless the input Hadamard runs inside the GEMM (see _fuse_input_had), in
    which case ``xh`` is left untouched — no caller reads it back. All tensors
    must be pre-allocated with stable addresses for CUDA graph capture;
    nothing is allocated inside this call.
    """
    # A cast here would allocate and silently break a capturing graph; callers
    # that need dtype conversion must cast before calling.
    assert x.dtype == torch.half, f"_linear_exl3_triton: expected half input, got {x.dtype}"

    fuse_had = _fuse_input_had(K, x.shape[0], in_features, out_features)
    fuse_out = (y.dtype == torch.half
                and _fuse_output_had(K, x.shape[0], in_features, out_features))
    if fuse_had:
        gemm_x, gemm_suh = x, suh
    else:
        # Phase 1: input Hadamard transform -> xh
        had_r_128_triton(x, xh, suh, None, 1.0)
        gemm_x, gemm_suh = xh, None

    # Phase 2 + 3: fused dequant + Triton GEMM -> y
    cb = 1 if mcg else (2 if mul1 else 0)
    splits = _m1_splitk_plan(x.shape[0], out_features, in_features, K)
    if splits > 1:
        # Split-K GEMV into fp32 partials, then a fused reduce + output
        # Hadamard (replaces the separate had_r_128_triton launch).
        partials = _get_splitk_buf(out_features, splits, x.device)
        exl3_gemm_triton(
            gemm_x, trellis, partials,
            _decode_lut(cb, x.device), _get_perm(x.device),
            K, trellis.shape[1], cb, splits, gemm_suh,
        )
        _m1_split_reduce_had(partials, y, svh, splits)
    else:
        exl3_gemm_triton(
            gemm_x, trellis, y,
            _decode_lut(cb, x.device), _get_perm(x.device),
            K, trellis.shape[1], cb, 1, gemm_suh,
            svh if fuse_out else None,
        )
        if not fuse_out:
            # Phase 4: output Hadamard transform (in place)
            had_r_128_triton(y, y, None, svh, 1.0)

    if bias is not None:
        y.add_(bias)


def linear_exl3_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.half,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused EXL3 dequant + GEMM, allocating and returning the output.

    Convenience wrapper over ``_linear_exl3_triton`` for uncaptured callers;
    graph-capturing paths (BC) call the preallocated-buffer function directly
    to keep tensor addresses stable across replays.

    All bit widths K = 1..8 are decoded in-kernel (fast gather-free paths for
    full tiles, a staged-row gather path for the rest).

    The per-call ``xh`` workspace allocation is deliberate and cheap: the
    caching allocator serves it from its free list (no cudaMalloc, no sync)
    after the first few calls, and this wrapper is off the decode hot path —
    prefill is dominated by the GEMM itself.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    xh = torch.empty_like(x_half)

    _linear_exl3_triton(
        x_half, y, xh, trellis, suh, svh, K, mcg, mul1, bias,
        in_features, out_features,
    )

    return y.view(original_shape[:-1] + (out_features,))
