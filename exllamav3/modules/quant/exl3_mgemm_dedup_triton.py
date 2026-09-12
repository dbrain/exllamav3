"""Expert-deduplicating grouped dequant + GEMV for the multi-token (verify) MoE path.

WHY. The grouped kernel in exl3_mgemm_triton.py reads one expert weight matrix per
routed ENTRY. A speculative verify routes M * top_k entries -- 70 at ndt 6 -- but those
cover only ~42 DISTINCT experts per layer (measured: perf/exl3/final/ledger-final-probe.csv,
42.1 of 70 at M=7, 33.0 of 50 at M=5), because different tokens of one draft window route
to overlapping experts. So ~40% of the trellis traffic AND ~40% of the decode ALU in a
verify is a second decode of a matrix another row in the same launch already decoded.
Sorting the entries so the repeats are adjacent and letting L2 catch them recovers only
1.6% of the round, which is the evidence that the cost is the decode, not just DRAM.
This kernel decodes each expert's tile ONCE and applies it to every row routed to it.

SHAPE. Entries arrive sorted by expert id, so equal ids are contiguous. A program owns
(entry position p, n-tile) and is a "chunk head" when its position within its run is a
multiple of MAXR; a head serves rows p .. p + nrows - 1 with nrows <= MAXR. A run can be
no longer than the token count (an expert is routed at most once per token) and the
multi-token path caps that at EXL3_MOE_MULTITOK_MAX <= 16, so a 16-wide look-back
resolves the position within the run in-kernel -- no host-side segment table, no
host sync, and the grid stays E * cdiv(N, BLOCK_N), a static function of the shapes,
which is what graph capture needs. Non-head programs exit immediately.

MAXR IS A KNOB, NOT A CONSTANT. A head pays MAXR rows of FMA whether or not the rows
exist, while the saving is one tile decode per entry it absorbs. Which way that trades
is a measurement, so it is selected per call by EXL3_MOE_DEDUP=2|4|8. MEASURED: 4 wins
(round -4.8/-5.2% in two processes, routed MoE -15%); 2 is inside noise; 8 spills the
accumulator and costs +134%. A 16-row tl.dot/WMMA variant was built and LOST (+7.0%
round, +21.6% MoE) -- a run averages 1.67 rows, so the matrix units mostly multiply
padding while the per-subtile permute into natural (k, n) order is paid every tile.
Don't rebuild it; see llm-fondling/FINALRUN.md.

THE BOUND IS DECODE, NOT DRAM, AND IT IS SETTLED -- do not price this kernel against a
bandwidth roofline. Collapsing ptrs_trellis from 275 MB of distinct tiles to one
L2-resident tile, with the grid, head count, decode and FMA byte-identical, buys 8.7% on
down and 7.1% on gate; deleting _decode_u16 with the loads and FMA untouched buys 32-37%.
So ~49 GB/s against the device's 86.4 GB/s read ceiling is NOT ~38 ms/round of missing
bandwidth: infinite DRAM is worth <=10% of this kernel. Time tracks the HEAD count almost
exactly (heads -36% -> time -33%), and MAXR 4 makes ~45 heads against a floor of 42.1
distinct experts, so the algorithm is within ~7% of its own floor. Two decode-side levers
are measured dead: v_dot4_u32_u8 for the CB 2 byte sum is bitwise identical and verifiably
compiled (64 emitted, v_bfe_u32 153 -> 25) and 1.47-1.58x SLOWER; dropping the rmask from
the x load is safe-looking but 1.04-1.05x slower. See llm-fondling/perf/exl3/
ledger-dedupbound.csv and final/dedupbound{1,3,4}.log.

NUMERICS. Each row accumulates the same expression tree as the per-entry kernel -- same
decode, same k-order accumulation into the same [ch, rh, p, nj, cl, q] tile -- so the
only freedom is how the final in-register reduction associates, which the leading row
axis can change. Expect bitwise or a few fp16 ULP, never more.

Scope: K_BITS == 4 (the routed expert width on flashnext) with full tiles and unfused
Hadamards, which is what the multi-token path runs (both EXL3_FUSE_MGEMM_*_HAD default
off). Anything else falls back to the per-entry kernel.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .exl3_mgemm_triton import _decode_u16, _e_bucket, had_r_128_mtriton

# Longest run of equal ids = most tokens a multi-token MoE call can carry.
_WIN = 16


def dedup_supported(K_bits: int, K_dim: int, N: int) -> bool:
    return K_bits == 4 and K_dim % 128 == 0 and N % 16 == 0


def max_rows_env() -> int:
    """EXL3_MOE_DEDUP, read per call so one process can A/B it. 0 = off."""
    try:
        v = int(os.environ.get("EXL3_MOE_DEDUP", "0") or 0)
    except ValueError:
        return 0
    return v if v in (2, 4, 8) else 0


def _dedup_configs():
    return [
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128}, num_warps=2, num_stages=3),
    ]


# Same contract as _MGEMM_PINNED: (K_dim, N, bits, E_BUCKET, MAXR) -> tile.
#
# The pool below is not a set of near-ties. (BLOCK_N 64, BLOCK_K 128, num_warps 2)
# spills the [MAXR, 2, 2, 2, NN, 8, 4] accumulator at MAXR 4 and runs 12x slower than
# the SAME tile at num_warps 4 -- 9.66 vs 0.76 ms/launch on gate, 154 VGPRs of scratch
# vs none. num_warps is not in the autotune key, both are in the pool, and
# TRITON_CACHE_AUTOTUNING=1 freezes whichever a contended benchmark liked. So this
# table exists to keep a 12x cliff out of the binary, not to shave a few percent.
#
# Values are the autotuner's own pick, read from the live on-disk autotune cache of
# real Flash-Next processes (~/.triton/cache/*/_grouped_dedup_gemv4_kernel.autotune.json)
# the way _MGEMM_PINNED documents. All six deployed keys agree across independent
# processes, and an independent sweep at the production working set puts the runner-up
# 1.45-1.55x behind (llm-fondling/perf/exl3/ledger-dedupbound.csv, final/dedupbound1.log).
# E_BUCKET is _e_bucket(bsz * top_k): an MTP round routes ndt + 1 tokens, so ndt 4 lands
# on 64 (E 50), ndt 6 and 8 on 128 (E 70 / 90), and the shorter accept-prefill rounds
# on 32. Only MAXR 4 is listed because that is what EXL3_MOE_DEDUP ships; 2 and 8 keep
# autotuning, and 8 is a measured loser either way (+134%, the same accumulator spill).
_DEDUP_PINNED: dict = {
    # (K_dim,  N,   bits, E_bucket, MAXR): (BLOCK_N, BLOCK_K, num_warps, num_stages)
    (2560,  640, 4,  32, 4): (64, 128, 4, 3),   # gate_proj / up_proj
    (2560,  640, 4,  64, 4): (64, 128, 4, 3),
    (2560,  640, 4, 128, 4): (64, 128, 4, 3),
    ( 640, 2560, 4,  32, 4): (64, 128, 4, 3),   # down_proj
    ( 640, 2560, 4,  64, 4): (64, 128, 4, 3),
    ( 640, 2560, 4, 128, 4): (64, 128, 4, 3),
}

_PIN_OFF = ("0", "off", "false", "no")


def _dedup_prune(configs, named_args, **kwargs):
    def arg(nm):
        return kwargs.get(nm, named_args.get(nm))

    n, k, maxr = arg("N"), arg("K_dim"), arg("MAXR")
    if os.environ.get("EXL3_MGEMM_PIN", "1").lower() not in _PIN_OFF:
        hit = _DEDUP_PINNED.get((k, n, arg("K_BITS"), arg("E_BUCKET"), maxr))
        if hit is not None:
            bn, bk, nw, ns = hit
            if n % bn == 0 and k % bk == 0:
                return [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk},
                                      num_warps=nw, num_stages=ns)]
    out = [c for c in configs
           if n % c.kwargs["BLOCK_N"] == 0 and k % c.kwargs["BLOCK_K"] == 0]
    out = out or list(configs)
    # The accumulator is MAXR * 16 * BLOCK_N fp32 per program; wide tiles at a wide
    # MAXR spill. Keep the product bounded, and never return an empty pool.
    fit = [c for c in out if maxr * c.kwargs["BLOCK_N"] <= 256]
    return fit or [min(out, key=lambda c: c.kwargs["BLOCK_N"])]


@triton.autotune(
    configs=_dedup_configs(),
    key=["E_BUCKET", "N", "K_dim", "K_BITS", "CB", "MAXR"],
    prune_configs_by={"early_config_prune": _dedup_prune},
)
@triton.jit
def _grouped_dedup_gemv4_kernel(
    x_ptr, y_ptr,
    ptrs_trellis,          # [num_experts] int64, trellis base addresses
    sorted_ids_ptr,        # [E] expert id per routed entry, SORTED ascending
    E, N, K_dim,
    E_BUCKET,              # autotune key only; unused in the body
    stride_xe, stride_xk,
    stride_tk, stride_tn,
    stride_ye, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_BITS: tl.constexpr,
    CB: tl.constexpr,
    MAXR: tl.constexpr,
    WIN: tl.constexpr,
):
    NK: tl.constexpr = BLOCK_K // 16
    NN: tl.constexpr = BLOCK_N // 16

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_e = pid // num_pid_n
    pid_n = pid % num_pid_n

    eid = tl.load(sorted_ids_ptr + pid_e)
    # Sorted ids ⇒ the entries equal to mine are contiguous, so counting equal
    # predecessors inside a WIN-wide window gives my position in the run exactly.
    back = pid_e - 1 - tl.arange(0, WIN)
    prev = tl.load(sorted_ids_ptr + back, mask=back >= 0, other=-1)
    pos = tl.sum((prev == eid).to(tl.int32), 0)
    fwd = pid_e + tl.arange(0, MAXR)
    nxt = tl.load(sorted_ids_ptr + fwd, mask=fwd < E, other=-1)
    nrows = tl.sum((nxt == eid).to(tl.int32), 0)

    if pos % MAXR == 0:
        tbase = tl.load(ptrs_trellis + eid)
        tu32_ptr = tl.cast(tbase, tl.pointer_type(tl.uint32))

        rows = tl.arange(0, MAXR)
        rmask = rows < nrows
        row_e = tl.minimum(pid_e + rows, E - 1)      # masked lanes stay in bounds
        xrow = x_ptr + row_e * stride_xe
        yrow = y_ptr + row_e * stride_ye

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        stride_tk_u32 = stride_tk // 2
        stride_tn_u32 = stride_tn // 2
        base_n = (pid_n * NN) * stride_tn_u32
        n_k_tiles_total = K_dim // 16
        n_outer = tl.cdiv(n_k_tiles_total, NK)

        # Decode algebra lifted verbatim from the K_BITS == 4 FULL branch of
        # _grouped_dequant_gemv_kernel; only the row axis is new.
        j8 = tl.arange(0, 8)
        sh = 28 - 4 * j8
        neg_sh = tl.minimum(32 - sh, 31)
        wc = tl.arange(0, NN * 32)
        nj8 = tl.arange(0, NN)
        r16 = tl.arange(0, 16)

        acc7 = tl.zeros((MAXR, 2, 2, 2, NN, 8, 4), dtype=tl.float32)
        for k_outer in range(n_outer):
            for ki in tl.static_range(NK):
                ktb = k_outer * NK + ki
                row = tu32_ptr + ktb * stride_tk_u32 + base_n
                words = tl.load(row + wc)
                safe = (ktb > 0) | (base_n > 0)
                m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                w31 = tl.load(row + nj8 * 32 + 31)
                w31_bcast = tl.reshape(
                    tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
                )
                m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
                q = ((words[None, :] >> sh[:, None]) |
                     (m1[None, :] << neg_sh[:, None])) & 0xFFFF
                w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                xk = tl.load(xrow[:, None] + (ktb * 16 + r16)[None, :] * stride_xk,
                             mask=rmask[:, None], other=0.0).to(tl.float32)
                xpat = tl.permute(tl.reshape(xk, (MAXR, 2, 4, 2)), (0, 1, 3, 2))
                xb7 = tl.broadcast_to(
                    tl.reshape(xpat, (MAXR, 1, 2, 2, 1, 1, 4)),
                    (MAXR, 2, 2, 2, NN, 8, 4)
                )
                acc7 += tl.reshape(w_dec, (1, 2, 2, 2, NN, 8, 4)) * xb7
        s = tl.sum(acc7, 6)      # q  -> (r, ch, rh, p, nj, cl)
        s = tl.sum(s, 3)         # p  -> (r, ch, rh, nj, cl)
        s = tl.sum(s, 2)         # rh -> (r, ch, nj, cl)
        acc = tl.reshape(tl.permute(s, (0, 2, 1, 3)), (MAXR, BLOCK_N))
        tl.store(yrow[:, None] + offs_n[None, :] * stride_yn,
                 acc.to(y_ptr.dtype.element_ty),
                 mask=rmask[:, None] & mask_n[None, :])


def exl3_mgemm_dedup(
    x: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    sorted_ids: torch.Tensor,
    y: torch.Tensor,
    K_bits: int,
    cb: int = 0,
    max_rows: int = 4,
) -> None:
    """``y[e] = dequant(trellis[sorted_ids[e]]).T @ x[e]``, one decode per expert run.

    ``sorted_ids`` MUST be sorted ascending: the kernel derives each entry's position
    within its run from that order. Allocates nothing; every buffer is the caller's.
    """
    assert x.dim() == 2 and y.dim() == 2, "exl3_mgemm_dedup: 2-D x and y"
    E, K_dim = x.shape
    N = y.shape[1]
    assert y.shape[0] == E and sorted_ids.numel() == E
    assert max_rows in (2, 4, 8), f"max_rows {max_rows} must be 2, 4 or 8"
    assert dedup_supported(K_bits, K_dim, N), \
        f"exl3_mgemm_dedup: unsupported shape bits={K_bits} K={K_dim} N={N}"

    stride_tn = 16 * K_bits
    stride_tk = (N // 16) * stride_tn
    grid = lambda meta: (E * triton.cdiv(N, meta["BLOCK_N"]),)
    _grouped_dedup_gemv4_kernel[grid](
        x, y, ptrs_trellis, sorted_ids,
        E, N, K_dim, _e_bucket(E),
        x.stride(0), x.stride(1),
        stride_tk, stride_tn,
        y.stride(0), y.stride(1),
        K_BITS=K_bits, CB=cb, MAXR=max_rows, WIN=_WIN,
    )


def linear_exl3_mgemm_dedup(
    x: torch.Tensor,
    xh: torch.Tensor,
    y: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    sorted_ids: torch.Tensor,
    K_bits: int,
    cb: int,
    max_rows: int,
) -> None:
    """Dedup twin of _linear_exl3_mgemm_triton: grouped input Hadamard, GEMV, output
    Hadamard. Same three launches, same buffers; only the middle kernel differs."""
    E, _, in_features = xh.shape
    had_r_128_mtriton(x.unsqueeze(1), xh, ptrs_suh, None, sorted_ids, 1.0)
    exl3_mgemm_dedup(xh.view(E, in_features), ptrs_trellis, sorted_ids, y,
                     K_bits, cb, max_rows=max_rows)
    had_r_128_mtriton(y.unsqueeze(1), y.unsqueeze(1), None, ptrs_svh, sorted_ids, 1.0)
