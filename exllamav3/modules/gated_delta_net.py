from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import get_for_device, to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..util.backend import QKV_SLICE
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm
from .quant.exl3 import row_pad_pitch as _row_pad_pitch

_MIDCKPT_STREAM = None  # EXL3_MIDCHUNK_CKPT=2 side stream for the checkpoint D2H
from ..cache import Cache
from ..util.tensor import g_tensor_cache
from .multilinear import SlicedMultiLinear
import os

# Sliced qkv+z projection bundle at decode for the split-projection GDN (Qwen3.5 / Qwen3.8 style):
# one mgemm over equal-width column slices, see attn.py. EXL3_QKV_SLICE=0 disables it
# (off by default on ROCm: the RDNA multi-matrix GEMVs do not take sliced bundles, which would fall back to
# the cooperative GEMM)
_qkv_slice_enable = QKV_SLICE
# EXL3_BC_GDN=0 disables the graph-captured decode paths (torch path only), for A/B testing
_bc_gdn_enable = os.environ.get("EXL3_BC_GDN", "1") != "0"
# Prefill projections (qkv, z, forget/beta gates) come out of the GEMM in fp16 rather than fp32.
# EXL3_GDN_PROJ_FP32=1 restores fp32
_proj_dtype = torch.float if os.environ.get("EXL3_GDN_PROJ_FP32", "0") != "0" else torch.half
# KDA's per-channel forget gate and beta logits: separate knob, the decay chain they feed is
# where fp16 rounding compounds (see the A/B in the docs)
_gate_dtype = torch.float if os.environ.get("EXL3_GDN_GATE_FP32", "1") != "0" else torch.half
# Prefill conv reads the projection output in place (token-major) instead of a transposed bf16 copy
_conv_token_major = os.environ.get("EXL3_GDN_CONV_TOKEN_MAJOR", "1") != "0"
_QKVZ_DEC_MULTI = os.environ.get("EXL3_GDN_QKVZ_MULTI", "1") != "0"   # (qkv+z one launch)
# EXL3_QWD_AB (default 0 until proven) folds the fp16 b / a projections, and with a bf16 dt_bias / fp32 a_log also
# gated_delta_net_fused_op_2 (beta, g), into the qkv|z launch as extra blocks (bit-identical to the separate launches);
# EXL3_QWD_AB_FIRST=1 puts those blocks first in the grid
_QWD_AB_FLAGS = 1 if os.environ.get("EXL3_QWD_AB_FIRST", "0") != "0" else 0
def _qwd_ab_on() -> bool:   # read per call so a served A/B run can flip it (serve_ab arm switch)
    return os.environ.get("EXL3_QWD_AB", "0") != "0"
from ..model.model_tp_shared import TPTensorWrapper
from .gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn
from ..cache.recurrent import (
    mp_cache_recurrent_stash,
    mp_cache_recurrent_unstash,
    mp_cache_recurrent_clear,
    new_checkpoint_handle,
)
from ..util import profile_opt
from .attention_fn.bc_attn import MAX_BSZ as _BC_MAX_BSZ, MAX_QLEN as _BC_MAX_QLEN
from ..cache.recurrent import host_copy

# Fused KDA beta/g gate kernel (ROCm fused_elt_rocm.cu); read at import
_FUSE_KDA_GATE = hasattr(ext, "kda_gate") and os.environ.get("EXL3_FUSE_KDA_GATE", "1") != "0"
# Decode-graph KDA gate: keep dt_bias in fp32 (as prefill and the HF ForgetGate do) instead of a bf16
# copy. GLM dt_bias is stored fp32 and is not bf16-exact (max rounding error 0.03)
_KDA_DT_F32 = os.environ.get("EXL3_KDA_DT_F32", "0") != "0"
KDA_KNOBS = {
    "half_a": os.environ.get("EXL3_KDA_HALF_A", "1") != "0",
    "dec_cat": os.environ.get("EXL3_KDA_DEC_CAT", "1") != "0",
    "lr_batched": os.environ.get("EXL3_KDA_DEC_LR_BATCHED", "1") != "0",
    # kda-dec step 2: f_b fused into the gate kernel, g_b fused into the gated norm (bit-exact)
    "lr_fused": os.environ.get("EXL3_KDA_DEC_LR_FUSED", "1") != "0",
}

def _rewind_layers(layers, slot: int, last_history: int, num_tokens: int,
                   spec_row: int | None, spec_shape: tuple | None, base_row: int, scratch_row: int):
    """Roll one sequence's recurrent layer states back `num_tokens` tokens after a speculative
    pass that recorded `last_history` tokens of history, in a handful of launches per device.
    With layer-split loading a single cache's GDN layers span multiple devices, so jobs are
    grouped by the device each layer's state lives on (launching them all under one device index
    dereferences foreign pointers). Only GDNLayerState instances (GDN and Mamba2 alike) are
    batched; any other recurrent-state type sharing the same cache (e.g. SWA, short-conv) falls
    back to its own .rewind() call, unchanged.

    Two halves. The conv ring is copied back (batched_conv_rewind). The recurrent state has no
    per-token snapshots: the pass advanced the scratch row from the untouched base row, so the
    state after the accepted prefix is rebuilt by replaying the scan over the staged inputs of
    that pass (batched_scan_replay, one launch per device and kernel geometry). A rewind of the
    whole pass (prefix 0) never happens: the first token of a pass is always kept"""
    prefix = last_history + 1 - num_tokens
    replay = last_history > 0 and num_tokens > 0
    if replay:
        assert spec_shape is not None and spec_row is not None, \
            "GDN rewind: history was recorded but the pass staged no scan inputs to replay"
    conv_by_device = {}
    replay_by_key = {}
    for l in layers:
        if isinstance(l, GDNLayerState):
            # l.device may be a plain string ("cuda:0") in some TP contexts rather than a
            # torch.device, so normalize rather than assume a .index attribute
            device_index = torch.device(l.device).index
            cj = l.rewind_conv_job(slot, last_history, num_tokens)
            if cj is not None:
                conv_by_device.setdefault(device_index, []).append(cj)
            if replay:
                key = (device_index,) + l.scan_geometry()
                replay_by_key.setdefault(key, []).append(
                    l.replay_job(spec_row, prefix, spec_shape, base_row, scratch_row))
        else:
            l.rewind(slot, last_history, num_tokens)
    for device_index, jobs in conv_by_device.items():
        ext.batched_conv_rewind(jobs, device_index)
    for (device_index, kind, nk, nv, dk, dv, state_half), jobs in replay_by_key.items():
        ext.batched_scan_replay(jobs, device_index, kind, nk, nv, dk, dv, state_half)


def mp_cache_recurrent_rewind(local_context: dict, cache_id: int, slot: int, last_history, num_tokens,
                              spec_row, spec_shape, base_row, scratch_row):
    recurrent_modules = local_context["recurrent_modules"]
    layers = [module.tp_recurrent_lookup[cache_id] for module in recurrent_modules]
    _rewind_layers(layers, slot, last_history, num_tokens, spec_row, spec_shape, base_row, scratch_row)


def mp_cache_recurrent_stash_gdn(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int, parity: int):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position, parity) if isinstance(l, GDNLayerState) else l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash_gdn(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int, parity: int):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, st in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        if isinstance(l, GDNLayerState):
            l.unstash(slot, st, position, parity)
        else:
            l.unstash(slot, st, position)


_ABCAT = os.environ.get("EXL3_QWD4_ABCAT", "1") != "0"
QWD3 = {"skip": 0}   # harness-only kill test: 1 = reuse the last b / a outputs (skip the skinny launches), 2 = also reuse beta / g. NOT bit-identical, never committed.

class GDNState:
    """One sequence's recurrent state across every recurrent layer of a cache. The layer pools
    hold two rows per sequence slot; `parity` names the live (base) row, 2 * slot + parity. A
    speculative pass reads the base row and advances the other (scratch) row, and rewind()
    settles it: a full accept flips the parity, a partial accept replays the accepted prefix
    from the base row into the scratch row and then flips. Until rewind() runs, the base row is
    the committed state and the scratch row is provisional"""

    # Prompt-end checkpoints (Job._tail_stash) are validated for this state class only
    supports_tail_ckpt = True
    # RecurrentCache.put may call stash(async_ = True) (EXL3_NC2_ASYNC_STASH)
    supports_async_stash = True

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
        exported: bool = False,
    ):
        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.parity = 0
        # Batch row and (bsz, seqlen) of the last speculative pass: the staged scan inputs the
        # rewind replays are batch-indexed (set by advance_recurrent_states)
        self.spec_row = None
        self.spec_shape = None
        self.exported = exported

        if not exported:
            assert test_state or position == 0 or stashed is not None, \
                "State must be new, restored from checkpoint or marked as a test state."

            if clear and stashed is None:
                if not self.cache.model.loaded_tp:
                    for l in self.cache.get_all_recurrent_layers().values():
                        l.clear(slot)
                else:
                    self.cache.model.tp_dispatch_all(mp_cache_recurrent_clear, (id(self.cache), self.slot))

            if stashed is not None:
                self.unstash(stashed)

            self.checkpoint_size = sum(
                l.get_checkpoint_size()
                for l in self.cache.get_all_recurrent_layers().values()
            )


    def free(self):
        self.cache.release_state(self)


    def base_row(self) -> int:
        return 2 * self.slot + self.parity


    def scratch_row(self) -> int:
        return 2 * self.slot + 1 - self.parity


    def rewind(self, num_tokens: int):
        """Settle the state after the last forward pass, keeping its first (last_history + 1 -
        num_tokens) tokens. After a pass without history (no speculative tokens) this is a
        position correction only: nothing was provisional, and re-fed tokens land on the same
        state again"""
        assert num_tokens >= 0, f"GDN rewind: negative num_tokens {num_tokens}"
        last_history = self.last_history
        if last_history > 0:
            assert num_tokens <= last_history, \
                f"GDN rewind: {num_tokens} tokens exceeds the recorded history of {last_history}"
            args = (self.slot, last_history, num_tokens, self.spec_row, self.spec_shape,
                    self.base_row(), self.scratch_row())
            if not self.cache.model.loaded_tp:
                _rewind_layers(self.cache.get_all_recurrent_layers().values(), *args)
            else:
                self.cache.model.tp_dispatch_all(mp_cache_recurrent_rewind, (id(self.cache),) + args)
            # The scratch row now holds the committed state: the whole pass, or the replayed
            # accepted prefix. It becomes the base row
            self.parity = 1 - self.parity
        self.position -= num_tokens
        self.last_history = 0
        self.spec_row = None
        self.spec_shape = None


    def rollback_capacity(self):
        # The state advances destructively; rewinding is only possible immediately after a forward pass that
        # recorded per-token history (speculative decoding), never at an arbitrary later point
        return 0


    def stash(self, async_: bool = False, pin_pool: dict = None):
        """async_ (EXL3_NC2_ASYNC_STASH, via RecurrentCache.stash_async only): non-blocking D2H into pinned host tensors on the
        current stream plus one event in stashed["_ev"]; the cache syncs it and moves the copies to pageable memory
        (RecurrentCache._finish, worker thread) before any reader sees the entry."""
        # A speculative pass whose outcome was never settled has its committed state in the
        # base row still, but its position already includes the provisional tokens
        assert self.last_history == 0, "GDN stash: state has an unsettled speculative pass (rewind() first)"
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size
        }
        if not self.cache.model.loaded_tp:
            layers = self.cache.get_all_recurrent_layers()
            if async_ and all(isinstance(l, GDNLayerState) for l in layers.values()):
                for k, l in layers.items():
                    stashed[k] = l.stash_async(self.slot, pin_pool, self.parity)
                ev = torch.cuda.Event()
                ev.record()
                stashed["_ev"] = ev
                return stashed
            for k, l in layers.items():
                stashed[k] = l.stash(self.slot, self.position, self.parity) if isinstance(l, GDNLayerState) \
                    else l.stash(self.slot, self.position)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_stash_gdn,
                                             (id(self.cache), cp_handle, self.slot, self.position, self.parity))
            stashed["tp_handle"] = cp_handle
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                if isinstance(l, GDNLayerState):
                    l.unstash(self.slot, stashed[k], self.position, self.parity)
                else:
                    l.unstash(self.slot, stashed[k], self.position)
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_unstash_gdn,
                                             (id(self.cache), cp_handle, self.slot, self.position, self.parity))


    def post_advance(self):
        pass


    def tp_export(self):
        return GDNState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            exported = True,
        )


    def reset(self):
        self.position = 0


class GDNLayerState:
    """One recurrent layer's state for every sequence slot of a cache (GDN and Mamba2 share it):
    the conv ring (conv_kernel_size + max_history columns, so a speculative pass of up to
    max_history + 1 tokens can be rolled back), the recurrent-state pool with two rows per slot
    (see GDNState), and the staged scan inputs of the last speculative pass, which a rewind
    replays. The staging buffers are sized for max_batch_size passes of max_history + 1 tokens
    and allocated with the rest of the state, so a long speculative pass never allocates at
    run time"""

    def __init__(
        self,
        module: GatedDeltaNet,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
        state_dtype: torch.dtype = torch.float,
    ):
        self.module = module
        self.conv_state = torch.empty(
            (max_batch_size, module.fdim_qkv, module.conv_kernel_size + max_history),
            dtype = torch.bfloat16,
            device = "meta"
        )
        # The recurrent state is stored in fp16 (the cache's default for GDN and KDA layers) or in
        # fp32 (Cache(recurrent_dtype = torch.float); Mamba2 layers always). The scan kernels
        # compute in fp32 either way and round once per pass on the store; the chunked prefill
        # kernels take and return fp32
        assert state_dtype in (torch.float, torch.half), "recurrent state dtype must be float32 or float16"
        self.recurrent_state = torch.empty(
            (2 * max_batch_size, 1, module.num_v_heads, module.k_head_dim, module.v_head_dim),
            dtype = state_dtype,
            device = "meta"
        )
        # Staged scan inputs, flat rows. A pass of (bsz, seqlen) stages row-major: the first
        # bsz * seqlen rows as its contiguous (bsz, seqlen, ...) conv output / beta / g, batch row
        # b's token t at row b * seqlen + t. A pass fed one token at a time (a recurrent draft
        # model drafting) stages token-major: step t writes rows [t * bsz, (t + 1) * bsz), so each
        # step's inputs are contiguous and row b's token t is at t * bsz + b. The replay walks a
        # row's tokens with the matching stride (spec_shape). g is per head (GDN, Mamba2) or per
        # k-channel (KDA)
        rows = max_batch_size * (max_history + 1)
        self.g_width = module.num_v_heads * (module.k_head_dim if getattr(module, "kda", False) else 1)
        self.spec_conv = torch.empty((rows, module.fdim_qkv), dtype = torch.bfloat16, device = "meta")
        self.spec_beta = torch.empty((rows, module.num_v_heads), dtype = torch.bfloat16, device = "meta")
        self.spec_g = torch.empty((rows, self.g_width), dtype = torch.float, device = "meta")
        # (bsz, tokens, token stride) of the pass the staging buffers currently hold
        self.spec_shape = None
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def set_state_dtype(self, dtype: torch.dtype):
        """Storage dtype of the recurrent-state pool, settable until the state is allocated"""
        assert self.device is None, "recurrent state dtype must be set before the state is allocated"
        assert dtype in (torch.float, torch.half), "recurrent state dtype must be float32 or float16"
        self.recurrent_state = torch.empty_like(self.recurrent_state, dtype = dtype)


    def get_checkpoint_size(self):
        return (
            self.module.fdim_qkv * self.module.conv_kernel_size * 2 +
            self.module.num_v_heads * self.module.k_head_dim * self.module.v_head_dim * self.recurrent_state.element_size()
        )


    def _tensors(self):
        return [self.conv_state, self.recurrent_state, self.spec_conv, self.spec_beta, self.spec_g]


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in self._tensors())


    def alloc(self, device):
        self.conv_state = torch.empty_like(self.conv_state, device = device)
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = device)
        self.spec_conv = torch.empty_like(self.spec_conv, device = device)
        self.spec_beta = torch.empty_like(self.spec_beta, device = device)
        self.spec_g = torch.empty_like(self.spec_g, device = device)
        self.conv_state.zero_()
        self.recurrent_state.zero_()
        self.spec_shape = None
        self.device = device


    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = "meta")
        self.spec_conv = torch.empty_like(self.spec_conv, device = "meta")
        self.spec_beta = torch.empty_like(self.spec_beta, device = "meta")
        self.spec_g = torch.empty_like(self.spec_g, device = "meta")
        self.spec_shape = None
        self.device = None


    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()
            self.recurrent_state[2 * idx : 2 * idx + 2].zero_()


    def get_state_tensors(self):
        return (
            self.conv_state,
            self.recurrent_state,
        )


    def staged_views(self, bsz: int, seqlen: int):
        """The (bsz, seqlen, ...) conv output, beta and g buffers a speculative pass of that
        shape writes its scan inputs into (contiguous views over the staging rows), and the
        record of that shape for the rewind"""
        assert seqlen <= self.max_history + 1 and bsz <= self.max_batch_size, \
            f"speculative pass of {seqlen} tokens x {bsz} rows exceeds the cache's max_history {self.max_history} / max_batch_size {self.max_batch_size}"
        n = bsz * seqlen
        self.spec_shape = (bsz, seqlen, 1)
        return (
            self.spec_conv[:n].view(bsz, seqlen, -1),
            self.spec_beta[:n].view(bsz, seqlen, -1),
            self.spec_g[:n].view(bsz, seqlen, -1),
        )


    def staged_step_views(self, bsz: int, step: int):
        """The (bsz, 1, ...) buffers step `step` of a pass fed one token at a time writes its scan
        inputs into (token-major rows), and the record of the pass so far for the rewind. Step 0
        is the same rows as a one-token pass, so it may run through the one-token speculative
        graph slot"""
        assert step <= self.max_history and bsz <= self.max_batch_size, \
            f"step {step} of a token-by-token speculative pass exceeds the cache's max_history {self.max_history} / max_batch_size {self.max_batch_size}"
        if step > 0:
            assert self.spec_shape == (bsz, step, bsz), \
                f"step {step} continues a pass staged as {self.spec_shape}, expected ({bsz}, {step}, {bsz})"
        a, b = step * bsz, (step + 1) * bsz
        self.spec_shape = (bsz, step + 1, bsz)
        return (
            self.spec_conv[a:b].view(bsz, 1, -1),
            self.spec_beta[a:b].view(bsz, 1, -1),
            self.spec_g[a:b].view(bsz, 1, -1),
        )


    def scan_geometry(self):
        """(kind, num_k_heads, num_v_heads, k_head_dim, v_head_dim, state_half): the replay
        kernel's selection; every job of one batched_scan_replay launch shares it"""
        m = self.module
        kind = 2 if getattr(m, "kda", False) else (1 if getattr(m, "mamba2", False) else 0)
        return kind, m.num_k_heads, m.num_v_heads, m.k_head_dim, m.v_head_dim, self.recurrent_state.dtype == torch.half


    def rewind_conv_job(self, slot: int, last_history: int, num_tokens: int):
        """Job descriptor for the batched conv-state rewind kernel (ext.batched_conv_rewind),
        computed without performing any copy. Addresses are integer arithmetic on the base
        pointer: a rewind touches every GDN layer of the model, and indexing views cost ~10 us
        per layer in Python."""
        if last_history == 0:
            return None
        assert 0 <= num_tokens <= last_history, \
            f"GDN conv rewind: {num_tokens} tokens outside the recorded history of {last_history}"
        cs = self.conv_state
        cdim = self.module.conv_kernel_size
        p = cs.shape[-1] - num_tokens
        es = cs.element_size()
        base = cs.data_ptr() + slot * cs.stride(0) * es
        return ext.ConvRewindJob(
            base + (p - cdim) * cs.stride(2) * es,
            base,
            cs.shape[1],
            cdim,
            cs.stride(1),
        )


    def replay_job(self, spec_row: int, prefix: int, spec_shape: tuple, base_row: int, scratch_row: int):
        """Job descriptor for the batched scan replay (ext.batched_scan_replay): rerun the scan
        over the first `prefix` staged tokens of batch row `spec_row` of the last speculative
        pass, from the pool row `base_row` into `scratch_row`. Same kernel, same inputs, same
        initial state as that pass, so the result is the state it had after `prefix` tokens"""
        assert self.spec_shape == spec_shape, \
            f"GDN replay: staged inputs are from a pass of shape {self.spec_shape}, rewind expects {spec_shape}"
        bsz, seqlen, tstride = spec_shape
        assert 1 <= prefix <= seqlen and 0 <= spec_row < bsz
        # Row-major pass: row b's tokens start at b * seqlen, one row apart; token-major: at b, bsz apart
        r0 = spec_row * seqlen if tstride == 1 else spec_row
        rs = self.recurrent_state
        row_bytes = rs.stride(0) * rs.element_size()
        m = self.module
        D = getattr(m, "d_skip_f", None)
        return ext.ScanReplayJob(
            self.spec_conv.data_ptr() + r0 * self.spec_conv.stride(0) * self.spec_conv.element_size(),
            self.spec_g.data_ptr() + r0 * self.spec_g.stride(0) * self.spec_g.element_size(),
            self.spec_beta.data_ptr() + r0 * self.spec_beta.stride(0) * self.spec_beta.element_size(),
            rs.data_ptr() + base_row * row_bytes,
            rs.data_ptr() + scratch_row * row_bytes,
            D.data_ptr() if D is not None else 0,
            prefix,
            tstride,
        )


    def stash(self, slot, position: int = 0, parity: int = 0):
        cdim = self.module.conv_kernel_size
        row = 2 * slot + parity
        return (
            host_copy(self.recurrent_state[row : row + 1]),
            host_copy(self.conv_state[slot, :, :cdim])
        )


    def stash_async(self, slot, pin_pool: dict = None, parity: int = 0):
        """Non-blocking D2H into pinned host tensors, reused from pin_pool ((shape, dtype) -> free list) when given:
        a pinned allocation costs ~1 ms of host time, more than the blocking copy it replaces."""
        cdim = self.module.conv_kernel_size
        row = 2 * slot + parity
        out = []
        for src in (self.recurrent_state[row : row + 1], self.conv_state[slot, :, :cdim]):
            free = None if pin_pool is None else pin_pool.get((tuple(src.shape), src.dtype))
            dst = free.pop() if free else torch.empty(src.shape, dtype = src.dtype, pin_memory = True)
            dst.copy_(src, non_blocking = True)
            out.append(dst)
        return tuple(out)


    def unstash(self, slot, stashed, position: int = 0, parity: int = 0):
        cdim = self.module.conv_kernel_size
        s, c = stashed
        row = 2 * slot + parity
        self.recurrent_state[row : row + 1].copy_(s)
        self.conv_state[slot, :, :cdim].copy_(c)


    def tp_stash_pieces(self):
        """
        How this rank's stash maps into the layout-independent checkpoint the disk tier stores, one entry per
        stashed tensor: the split dimension, its full size, and (local start, local end, canonical start)
        pieces. The recurrent state is split by v head; the conv state's channels are the q, k and v
        projections' rows in sequence, each split by its own head range. None for a tensor that is not split.
        """
        m = self.module
        first = getattr(m, "tp_kv_first", None)
        if first is None:
            return [None, None]
        nk, nv, kd, vd = m.num_k_heads, m.num_v_heads, m.k_head_dim, m.v_head_dim
        gk = m.tp_k_heads_total
        G = nv // nk
        assert m.fdim_qkv == 2 * nk * kd + nv * vd, "conv channel layout differs from q|k|v, cannot map the stash"
        return [
            {"dim": 2, "total": gk * G, "pieces": [(0, nv, first * G)]},
            {"dim": 0, "total": 2 * gk * kd + gk * G * vd, "pieces": [
                (0, nk * kd, first * kd),
                (nk * kd, 2 * nk * kd, (gk + first) * kd),
                (2 * nk * kd, 2 * nk * kd + nv * vd, 2 * gk * kd + first * G * vd),
            ]},
        ]


    def tp_export(self, plan):
        return {
            "cls": GDNLayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
                "state_dtype": self.recurrent_state.dtype,
            }
        }


class GatedDeltaNet(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        beta_scale: float = 1.0,
        key_a_log: str | None = None,
        key_dt_bias: str | None = None,
        key_conv1d: str | None = None,
        key_conv1d_q: str | None = None,
        key_conv1d_k: str | None = None,
        key_conv1d_v: str | None = None,
        key_fused_ba: str | None = None,
        key_fused_qkvz: str | None = None,
        key_qkv: str | None = None,
        key_qkv_alt: list | None = None,
        key_z: str | None = None,
        key_b: str | None = None,
        key_a: str | None = None,
        key_f_a: str | None = None,
        key_f_b: str | None = None,
        key_g_a: str | None = None,
        key_g_b: str | None = None,
        key_f: str | None = None,
        key_g: str | None = None,
        gate_lower_bound: float | None = None,
        key_norm: str | None = None,
        key_o: str | None = None,
        a_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        conv1d_weight: torch.Tensor | None = None,
        conv1d_bias: torch.Tensor | None = None,
        qkv_proj: Linear | None = None,
        z_proj: Linear | None = None,
        b_proj: Linear | None = None,
        a_proj: Linear | None = None,
        norm: GatedRMSNorm | None = None,
        o_proj: Linear | None = None,
        f_a_proj: Linear | None = None,
        f_b_proj: Linear | None = None,
        g_a_proj: Linear | None = None,
        g_b_proj: Linear | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        self.module_name = "GatedDeltaNet"

        self.q_priority = 1 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads if num_k_heads else 0
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads
        self.beta_scale = beta_scale

        self.out_dtype = out_dtype

        self.fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
        self.fdim_ba = 2 * self.num_v_heads
        self.fdim_qkv = 2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim

        # KDA mode (GLM5.3/Kimi linear attention): per-k-channel decay from a low-rank forget
        # gate (f_a/f_b + per-channel dt_bias + per-head A_log, "safe gate" when
        # gate_lower_bound is set), a low-rank sigmoid output gate (g_a/g_b) in place of z,
        # and in-kernel q/k l2norm. Shares the conv, projections, cache and rewind machinery
        # Direct form (Ling 3.x, no_kda_lora): full-rank forget- and output-gate projections
        # (key_f/key_g), quantized like the other projections
        self.kda = key_f_a is not None or key_f is not None
        self.kda_direct = key_f is not None
        self.gate_lower_bound = gate_lower_bound
        self.key_f_a = key_f_a
        self.key_f_b = key_f_b
        self.key_g_a = key_g_a
        self.key_g_b = key_g_b

        if self.num_k_heads == 0:
            return

        if self.kda_direct:
            assert key_g and not (key_f_a or key_f_b or key_g_a or key_g_b), \
                "Direct KDA gates take key_f and key_g alone"
        elif self.kda:
            assert key_f_b and key_g_a and key_g_b, \
                "KDA mode requires key_f_a, key_f_b, key_g_a and key_g_b"
            assert (key_b or b_proj is not None) and not key_a, \
                "KDA mode takes key_b for beta; decay comes from the f projections"
            assert num_k_heads == num_v_heads and k_head_dim == v_head_dim, \
                "KDA mode requires uniform head geometry"

        if key_qkv or key_z:
            assert key_qkv and (key_z or self.kda), \
                "GatedDeltaNet split qkv/z projections require both key_qkv and key_z"
        if (key_b or key_a) and not self.kda:
            assert key_b and key_a, \
                "GatedDeltaNet split b/a projections require both key_b and key_a"

        if key_fused_qkvz:
            self.qkvz_proj = Linear(
                config,
                f"{key}.{key_fused_qkvz}",
                hidden_size,
                self.fdim_qkvz,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkv_proj = None
            self.z_proj = None
            self.register_submodule(self.qkvz_proj)
        elif qkv_proj:
            self.qkv_proj = qkv_proj
            self.z_proj = z_proj
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            self.register_submodule(self.z_proj)
        elif key_qkv:
            self.qkv_proj = Linear(
                config,
                f"{key}.{key_qkv}",
                hidden_size,
                self.fdim_qkv,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                alt_key = None if not key_qkv_alt else [f"{key}.{x}" for x in key_qkv_alt],
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            if key_z:
                self.z_proj = Linear(
                    config,
                    f"{key}.{key_z}",
                    hidden_size,
                    self.v_dim,
                    qmap = qmap + ".input",
                    out_dtype = torch.float,
                    select_hq_bits = select_hq_bits,
                    qgroup = key + ".qkvz",
                )
                self.register_submodule(self.z_proj)
            else:
                self.z_proj = None
        else:
            self.qkvz_proj = None
            self.qkv_proj = None
            self.z_proj = None

        if key_fused_ba:
            self.ba_proj = Linear(config, f"{key}.{key_fused_ba}", hidden_size, self.fdim_ba, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.b_proj = None
            self.a_proj = None
            self.register_submodule(self.ba_proj)
        elif b_proj:
            self.b_proj = b_proj
            self.a_proj = a_proj
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        elif key_b and self.kda:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = None
            self.ba_proj = None
            self.register_submodule(self.b_proj)
        elif key_b:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = Linear(config, f"{key}.{key_a}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        else:
            self.b_proj = None
            self.a_proj = None
            self.ba_proj = None

        # KDA low-rank forget-gate and output-gate projections. Kept in fp16 (qmap = None):
        # the reference fp8 checkpoints exclude them from quantization, which is a strong
        # sensitivity signal
        self.f_proj = self.g_proj = None
        if self.kda_direct:
            self.f_proj = Linear(config, f"{key}.{key_f}", hidden_size, self.k_dim, qmap = qmap, out_dtype = torch.float, select_hq_bits = select_hq_bits)
            self.g_proj = Linear(config, f"{key}.{key_g}", hidden_size, self.v_dim, qmap = qmap, out_dtype = torch.float, select_hq_bits = select_hq_bits)
            self.register_submodule(self.f_proj)
            self.register_submodule(self.g_proj)
            self.f_a_proj = self.f_b_proj = self.g_a_proj = self.g_b_proj = None
        elif self.kda and f_a_proj is not None:
            # Prebuilt (TP import)
            self.f_a_proj = f_a_proj
            self.f_b_proj = f_b_proj
            self.g_a_proj = g_a_proj
            self.g_b_proj = g_b_proj
            for m in (self.f_a_proj, self.f_b_proj, self.g_a_proj, self.g_b_proj):
                self.register_submodule(m)
        elif self.kda:
            self.f_a_proj = Linear(config, f"{key}.{key_f_a}", hidden_size, self.k_head_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.f_b_proj = Linear(config, f"{key}.{key_f_b}", self.k_head_dim, self.k_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.g_a_proj = Linear(config, f"{key}.{key_g_a}", hidden_size, self.v_head_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.g_b_proj = Linear(config, f"{key}.{key_g_b}", self.v_head_dim, self.v_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            for m in (self.f_a_proj, self.f_b_proj, self.g_a_proj, self.g_b_proj):
                self.register_submodule(m)
        else:
            self.f_a_proj = self.f_b_proj = self.g_a_proj = self.g_b_proj = None

        if o_proj:
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)
        else:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                self.v_head_dim * self.num_v_heads,
                hidden_size,
                qmap = qmap + ".output",
                out_dtype = self.out_dtype,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o",
            )
            self.register_submodule(self.o_proj)

        if norm is not None:
            self.norm = norm
            self.register_submodule(self.norm)
        else:
            self.norm = GatedRMSNorm(
                config, f"{key}.{key_norm}", self.rms_norm_eps, out_dtype = torch.half,
                gate_activation = "sigmoid" if self.kda else "silu")
            self.register_submodule(self.norm)

        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_weight_flat = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None

        if dt_bias is not None:
            self.a_log = a_log
            self.dt_bias = dt_bias
            self.key_a_log = None
            self.key_dt_bias = None
        else:
            self.key_a_log = f"{key}.{key_a_log}"
            self.key_dt_bias = f"{key}.{key_dt_bias}"
        if conv1d_weight is not None:
            self.conv1d_weight = conv1d_weight
            self.conv1d_bias = conv1d_bias
            self.key_conv1d_weight = None
            self.key_conv1d_bias = None
            self.key_conv1d_q_weight = None
            self.key_conv1d_k_weight = None
            self.key_conv1d_v_weight = None
        else:
            self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
            self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"
            self.key_conv1d_q_weight = f"{key}.{key_conv1d_q}.weight" if key_conv1d_q else None
            self.key_conv1d_k_weight = f"{key}.{key_conv1d_k}.weight" if key_conv1d_k else None
            self.key_conv1d_v_weight = f"{key}.{key_conv1d_v}.weight" if key_conv1d_v else None

        self.conv_dim = self.k_head_dim * self.num_k_heads

        self.caps.update({
            "recurrent_cache": True
        })
        self.layer_state_cls = GDNLayerState

        self.bc = None
        self.bc_split = False
        self.bsz1_pa_args = []
        self.ba_weight_t = None
        self.ba_bias = None
        self.ba_weight_filled = False

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        if self.qkvz_proj is not None:
            return [[
                self.qkvz_proj.optimizer_targets(),
                self.o_proj.optimizer_targets(),
            ]]

        targets = []
        if self.qkv_proj is not None:
            targets.append(self.qkv_proj.optimizer_targets())
        if self.z_proj is not None:
            targets.append(self.z_proj.optimizer_targets())
        if self.kda_direct:
            targets.append(self.f_proj.optimizer_targets())
            targets.append(self.g_proj.optimizer_targets())
        targets.append(self.o_proj.optimizer_targets())
        return [targets]


    def load_local(self, device, **kwargs):

        if self.num_k_heads == 0:
            return

        # Recurrent states
        for rl in self.recurrent_layers:
            rl.alloc(device)

        is_quantized = (
            self.qkvz_proj is not None and self.qkvz_proj.quant_format_id() == "exl3" and
            self.ba_proj is not None and self.ba_proj.quant_format_id() is None and
            self.o_proj is not None and self.o_proj.quant_format_id() == "exl3"
        )

        if is_quantized:
            self.bsz1_pa_args = [
                (device, (1, self.fdim_qkv, 1), torch.bfloat16),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "a"),
                (device, (1, 1, self.num_v_heads), torch.bfloat16),
                (device, (1, 1, self.num_v_heads), torch.float),
                (device, (1, 1, self.fdim_qkvz), torch.float),
                (device, (1, 1, self.fdim_ba), torch.float),
                (device, (1, self.fdim_qkv, self.conv_kernel_size + 1), torch.bfloat16, "a"),
                (device, (1, self.fdim_qkv, 2), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads * self.v_head_dim), torch.half),
            ]

            self.bc = ext.BC_GatedDeltaNet(
                *(g_tensor_cache.get(*arg) for arg in self.bsz1_pa_args),
                self.qkvz_proj.inner.bc,
                self.ba_proj.inner.bc,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.conv1d_weight,
                self.conv1d_bias,
                self.norm.bc,
                self.o_proj.inner.bc,
                self.beta_scale
            )

        # Fuse conv1d weights and cache the flattened weight (normally done lazily in forward,
        # needed here for the split-projection batched call)
        if self.conv1d_weight is None and self.conv1d_q_weight is not None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None
        if self.conv1d_weight_flat is None and self.conv1d_weight is not None:
            self.conv1d_weight_flat = self.conv1d_weight.squeeze(1).contiguous()

        is_quantized_split = (
            _bc_gdn_enable and device != torch.device("cpu") and
            self.qkvz_proj is None and self.ba_proj is None and
            self.qkv_proj is not None and self.qkv_proj.quant_type == "exl3" and
            self.z_proj is not None and self.z_proj.quant_type == "exl3" and
            self.b_proj is not None and self.b_proj.quant_type == "fp16" and
            self.a_proj is not None and self.a_proj.quant_type == "fp16" and
            self.o_proj is not None and self.o_proj.quant_type == "exl3" and
            self.conv1d_weight_flat is not None and
            self.conv1d_weight_flat.dtype == torch.bfloat16 and
            (self.conv1d_bias is None or self.conv1d_bias.dtype == torch.bfloat16) and
            self.dt_bias is not None and self.dt_bias.dtype == torch.bfloat16
        )

        if is_quantized_split:
            # Merge the small unquantized b/a projections into a single fp16 GEMV. The weights may
            # not be materialized yet (deferred load), so only allocate here — the BC keeps a
            # reference — and copy the actual values in on the first forward pass
            nv, hv = self.num_v_heads, self.v_head_dim
            self.ba_weight_t = torch.empty((2 * nv, self.hidden_size), dtype = torch.half, device = device)
            has_bias = (
                self.b_proj.inner.get_bias_tensor() is not None or
                self.a_proj.inner.get_bias_tensor() is not None
            )
            self.ba_bias = torch.empty((2 * nv,), dtype = torch.half, device = device) if has_bias else None
            self.ba_weight_filled = False

            self.bc = ext.BC_GatedDeltaNetSplit(
                self.qkv_proj.inner.bc,
                self.z_proj.inner.bc,
                self.o_proj.inner.bc,
                self.ba_weight_t,
                self.ba_bias,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                nv,
                self.k_head_dim,
                hv,
                self.conv1d_weight_flat,
                self.conv1d_bias,
                self.norm.bc,
                self.beta_scale
            )
            self.bc_split = self.bc is not None

            # Sliced qkv+z bundle: both projections read x and are cut into equal-width column
            # slices run as one launch (SlicedMultiLinear); the graph object gets the tables, the
            # eager path (m <= 32) uses them directly
            self.multi_qkvz = None
            if (
                _qkv_slice_enable and
                self.qkv_proj.inner.bias is None and self.z_proj.inner.bias is None and
                self.qkv_proj.inner.K == self.z_proj.inner.K and
                self.qkv_proj.in_features == self.z_proj.in_features
            ):
                try:
                    self.multi_qkvz = SlicedMultiLinear(self.device, [self.qkv_proj, self.z_proj])
                except (ValueError, AssertionError):
                    self.multi_qkvz = None
                # Unfusing policy judged on the slice width (see attn.py)
                if self.multi_qkvz is not None and not self.config.infer_params.use_mgemm(
                    self.multi_qkvz.K, self.multi_qkvz.width, self.multi_qkvz.mul1, device,
                ):
                    self.multi_qkvz = None
            if self.multi_qkvz is not None:
                mq = self.multi_qkvz
                self.bc.set_qkvz_bundle(mq.ptrs_trellis, mq.ptrs_suh, mq.ptrs_svh, mq.meta, mq.K, bool(mq.mcg), bool(mq.mul1))
                # fp32 outputs: the mgemm C argument carries the dtype and slice width only
                self.prealloc_qkvz_carrier = g_tensor_cache.get(device, (mq.num_slices, 1, mq.width), torch.float, "qkvzc_1")

        is_quantized_kda = (
            _bc_gdn_enable and device != torch.device("cpu") and self.kda and
            self.qkv_proj is not None and self.qkv_proj.quant_type == "exl3" and
            self.o_proj is not None and self.o_proj.quant_type == "exl3" and
            all(p is not None and p.quant_type == "fp16" for p in
                (self.b_proj, self.f_a_proj, self.f_b_proj, self.g_a_proj, self.g_b_proj)) and
            self.conv1d_weight_flat is not None and
            self.conv1d_weight_flat.dtype == torch.bfloat16 and
            (self.conv1d_bias is None or self.conv1d_bias.dtype == torch.bfloat16) and
            self.dt_bias is not None
        )

        if is_quantized_kda:
            # The gate op reads dt_bias as bf16 (default) or fp32 (EXL3_KDA_DT_F32=1; GLM stores it
            # fp32 and it is not bf16-exact). Allocated here, FILLED in the deferred-fill
            # block on the first forward: at this point dt_bias may still be an unfilled
            # deferred tensor, and a .to() would copy garbage
            self.dt_bias_bc = torch.empty_like(self.dt_bias, dtype = torch.float if _KDA_DT_F32 else torch.bfloat16)
            # Transposed fp16 copies of the small projections for the graph GEMVs; weights may
            # not be materialized yet (deferred load) so only allocate here and fill on the
            # first forward (the BC keeps references)
            nv, hk, hv = self.num_v_heads, self.k_head_dim, self.v_head_dim
            hs = self.hidden_size
            self.kda_b_t = torch.empty((nv, hs), dtype = torch.half, device = device)
            self.kda_fa_t = torch.empty((hk, hs), dtype = torch.half, device = device)
            self.kda_fb_t = torch.empty((nv * hk, hk), dtype = torch.half, device = device)
            self.kda_ga_t = torch.empty((hv, hs), dtype = torch.half, device = device)
            self.kda_gb_t = torch.empty((nv * hv, hv), dtype = torch.half, device = device)
            self.ba_weight_filled = False

            self.bc = ext.BC_GatedDeltaNetSplit(
                self.qkv_proj.inner.bc,
                self.o_proj.inner.bc,
                self.kda_b_t,
                self.kda_fa_t,
                self.kda_fb_t,
                self.kda_ga_t,
                self.kda_gb_t,
                self.dt_bias_bc,
                self.a_log,
                float(self.gate_lower_bound or 0.0),
                self.num_k_heads,
                nv,
                self.k_head_dim,
                hv,
                self.conv1d_weight_flat,
                self.conv1d_bias,
                self.norm.bc,
                self.beta_scale
            )
            self.bc_split = self.bc is not None


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        if self.key_a_log is not None:
            self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
            # Kimi Linear stores A_log as (1, 1, H, 1); the kernels take it as (H,)
            self.a_log = self.a_log.reshape(-1)
            self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        if self.key_conv1d_weight is not None:
            # no_defer: load_local concatenates/flattens (copies) these immediately, which a
            # deferred (unfilled) tensor would corrupt
            self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = True, allow_bf16 = True, no_defer = True)
            self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True, no_defer = True)
            if self.conv1d_weight is None:
                self.conv1d_q_weight = self.config.stc.get_tensor(self.key_conv1d_q_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
                self.conv1d_k_weight = self.config.stc.get_tensor(self.key_conv1d_k_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
                self.conv1d_v_weight = self.config.stc.get_tensor(self.key_conv1d_v_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
        self.norm.load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        if self.bc is not None:
            # for arg in self.bsz1_pa_args:
            #     g_tensor_cache.drop(*arg)
            self.bc = None
            self.bc_split = False
            self.bsz1_pa_args = []
        self.ba_weight_t = None
        self.ba_bias = None
        self.ba_weight_filled = False
        self.multi_qkvz = None
        self.prealloc_qkvz_carrier = None
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_weight_flat = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None
        self.norm.unload()
        for cl in self.recurrent_layers:
            cl.free()
        super().unload()


    def split_fused_inputs(self, mixed_qkvz, mixed_ba):
        # mixed_qkvz and mixed_ba have same (bsz, seqlen)
        # both are contiguous
        bsz, seqlen, _ = mixed_qkvz.shape

        mixed_qkvz = mixed_qkvz.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads // self.num_k_heads,
        )
        mixed_ba = mixed_ba.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.num_v_heads // self.num_k_heads
        )

        split_arg_list_qkvz = [
            self.k_head_dim,
            self.k_head_dim,
            (self.num_v_groups * self.v_head_dim),
            (self.num_v_groups * self.v_head_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]
        q, k, v, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim = 3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim = 3)

        q = q.reshape(bsz, seqlen, -1)
        k = k.reshape(bsz, seqlen, -1)
        v = v.reshape(bsz, seqlen, -1)
        z = z.reshape(bsz, seqlen, -1, self.v_head_dim)
        b = b.reshape(bsz, seqlen, self.num_v_heads)
        a = a.reshape(bsz, seqlen, self.num_v_heads)
        mixed_qkv = torch.cat((q, k, v), dim = -1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        return mixed_qkv, z, b, a


    def _bc_configure_slot_kda(self, bsz: int, seqlen: int, history: bool, rsl = None):
        device = self.device
        f = self.fdim_qkv
        nv, hk, hv = self.num_v_heads, self.k_head_dim, self.v_head_dim
        qkv             = g_tensor_cache.get(device, (bsz, seqlen, f), torch.float, "s_qkv")
        z               = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.float, "s_z")
        b_out           = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.float, "s_kb")
        fa_out          = g_tensor_cache.get(device, (bsz, seqlen, hk), torch.float, "s_kfa")
        fb_out          = g_tensor_cache.get(device, (bsz, seqlen, nv * hk), torch.float, "s_kfb")
        ga_out          = g_tensor_cache.get(device, (bsz, seqlen, hv), torch.float, "s_kga")
        if history:
            # Speculative-pass slot: the scan inputs land in the cache's staging buffers, which
            # the rewind replays from (see GDNLayerState); the slot is bound to that cache
            conv_out, beta, g = rsl.staged_views(bsz, seqlen)
            g = g.view(bsz, seqlen, nv, hk)
        else:
            beta            = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.bfloat16, "s_beta")
            g               = g_tensor_cache.get(device, (bsz, seqlen, nv, hk), torch.float, "s_kg4")
            conv_out        = g_tensor_cache.get(device, (bsz, seqlen, f), torch.bfloat16, "s_conv")
        mixed_qkv       = g_tensor_cache.get(device, (bsz, f, seqlen), torch.bfloat16, "s_mqkv")
        core_attn_out   = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.bfloat16, "s_cao")
        core_attn_out_f = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_caof")
        qkv_xh = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_qkv_xh")
        o_xh   = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_o_xh")
        self.bc.configure_slot_kda(
            bsz, seqlen, history,
            qkv, z, b_out, fa_out, fb_out, ga_out, beta, g, mixed_qkv, conv_out,
            core_attn_out, core_attn_out_f, qkv_xh, o_xh,
        )
        if history:
            self.bc.set_stage_key(bsz, seqlen, True, id(rsl))

    def _dec_qkvz_linears(self):
        """(qkv_proj, z_proj) when one exl3_dec_gemv_multi launch can replace their two forwards (same K, mcg and
        input width, no bias), else None. one launch instead of two, one grid of both matrices."""
        lins = getattr(self, "_dec_qkvz", False)
        if lins is False:
            from .mla_attn import dec_proj_ok, dec_proj_shapes
            lins = None
            if self.qkv_proj is not None and self.z_proj is not None and self.qkvz_proj is None:
                cand = (self.qkv_proj, self.z_proj)
                if all(dec_proj_ok(l) for l in cand) and dec_proj_shapes(cand):
                    lins = cand
            self._dec_qkvz = lins
        return lins

    def project_qkvz_dec(self, x: torch.Tensor, params: dict, bsz: int, seqlen: int):
        """Decode qkv + z in ONE exl3_dec launch (both Hadamard stages inside): batch 1, or the R-row verify
        launch (dflash_verify). Returns (qkv, z) with the Linears' shapes and dtype, or None to fall back."""
        rows = bsz * seqlen
        if rows > 8 or x.dtype != torch.half or "capture" in params or "ovr" in params or not x.is_contiguous():
            return None
        if rows > 1 and not (params.get("dflash_verify") and os.environ.get("EXL3_VERIFY_GEMV_R", "0") != "0"
                             and hasattr(ext, "exl3_dec_gemv_r_multi")):
            return None
        lins = self._dec_qkvz_linears()
        if lins is None or x.shape[-1] != lins[0].in_features:
            return None
        from .mla_attn import dec_proj_multi
        qkv, z = dec_proj_multi(lins, x, rows)
        return (qkv.view(bsz, seqlen, -1), z.view(bsz, seqlen, -1))


    def _ab_cat(self, x: torch.Tensor, params: dict, bsz: int, seqlen: int):
        """decode b and a skinny projections in ONE launch (ext.skinny_cat_w, same per-column reduction order as the two
        separate skinny launches, bit-identical). Returns (b, a) fp32 [bsz, seqlen, heads] or None to fall back. EXL3_QWD4_ABCAT=0 = off."""
        ok = getattr(self, "_ab_cat_ok_v", None)
        if ok is None:
            mods = (self.b_proj, self.a_proj)
            ok = (hasattr(ext, "skinny_cat_w") and not self.kda and
                  all(m is not None and m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
                      m.inner._pinned_store is None and not m.lora_a_tensors and m.pre_scale == 1.0 and m.post_scale == 1.0 and
                      m.softcap == 0.0 and m.out_features == m.out_features_unpadded and m.out_features == self.num_v_heads and
                      m.inner.weight.is_contiguous() and m.inner.weight.dtype == torch.half and m.inner.out_dtype == torch.float for m in mods))
            self._ab_cat_ok_v = ok
        if not ok or bsz * seqlen != 1 or x.dtype != torch.half or not x.is_contiguous() or "capture" in params or "ovr" in params:
            return None
        h = self.num_v_heads
        b = torch.empty((bsz, seqlen, h), dtype = torch.float, device = x.device)
        a = torch.empty((bsz, seqlen, h), dtype = torch.float, device = x.device)
        ext.skinny_cat_w(x.view(1, -1), [self.b_proj.inner.weight, self.a_proj.inner.weight], [b.view(1, h), a.view(1, h)])
        return b, a

    def _qwd_ab_ok(self) -> bool:
        ok = getattr(self, "_qwd_ab_ok_v", None)
        if ok is None:
            lins = self._dec_qkvz_linears()
            mods = (self.b_proj, self.a_proj)
            ok = (lins is not None and hasattr(ext, "exl3_dec_gemv_x") and not self.kda and
                  all(m is not None and m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
                      m.inner._pinned_store is None and not m.lora_a_tensors and m.pre_scale == 1.0 and m.post_scale == 1.0 and
                      m.softcap == 0.0 and m.out_features == m.out_features_unpadded and m.out_features == self.num_v_heads and
                      m.inner.weight.is_contiguous() and m.inner.weight.dtype == torch.half and
                      m.inner.weight.shape[0] == lins[0].in_features and m.inner.out_dtype == torch.float for m in mods) and
                  float(lins[0].inner.K) == 5.0 and lins[0].inner.in_features // 128 <= 64)
            self._qwd_ab_ok_v = ok
        return ok

    def project_qkvz_ab_dec(self, x: torch.Tensor, params: dict, bsz: int, seqlen: int):
        """qkv + z + b + a (+ beta / g) in ONE launch (exl3_dec_gemv_x), batch 1 only. Returns (qkv, z, b, a, None, None) with b, a fp32
        [bsz, seqlen, num_v_heads], or (qkv, z, None, None, beta, g) when the gate math rides along, or None to fall back."""
        if bsz * seqlen != 1 or x.dtype != torch.half or "capture" in params or "ovr" in params or not x.is_contiguous():
            return None
        if not self._qwd_ab_ok():
            return None
        lins = self._dec_qkvz_linears()
        if x.shape[-1] != lins[0].in_features:
            return None
        from .quant.exl3 import dec_workspace
        scratch, counters = dec_workspace(x.device)
        outs = [torch.empty((1, l.inner.out_features), dtype = l.inner.default_out_dtype, device = x.device) for l in lins]
        h = self.num_v_heads
        gate = h % 16 == 0 and self.dt_bias.dtype == torch.bfloat16 and self.a_log.dtype == torch.float and \
               self.dt_bias.is_contiguous() and self.a_log.is_contiguous() and self.dt_bias.numel() == h and self.a_log.numel() == h
        if gate:
            o1 = torch.empty((1, h), dtype = torch.bfloat16, device = x.device)
            o2 = torch.empty((1, h), dtype = torch.float, device = x.device)
            gargs = [self.dt_bias, self.a_log]
        else:
            o1 = torch.empty((1, h), dtype = torch.float, device = x.device)
            o2 = torch.empty((1, h), dtype = torch.float, device = x.device)
            gargs = []
        ext.exl3_dec_gemv_x(
            x.reshape(1, -1), [l.inner.trellis for l in lins], [l.inner.suh for l in lins], [l.inner.svh for l in lins],
            outs, scratch, counters, [float(l.inner.K) for l in lins], bool(lins[0].inner.mcg), _QWD_AB_FLAGS,
            [self.b_proj.inner.weight, self.a_proj.inner.weight], [o1, o2], gargs, float(self.beta_scale))
        qkv, z = outs[0].view(bsz, seqlen, -1), outs[1].view(bsz, seqlen, -1)
        if gate:
            return (qkv, z, None, None, o1.view(bsz, seqlen, -1), o2.view(bsz, seqlen, -1))
        return (qkv, z, o1.view(bsz, seqlen, -1), o2.view(bsz, seqlen, -1), None, None)

    def project_qkvz_sliced(self, x: torch.Tensor, bsz: int, seqlen: int) -> tuple:
        """qkv and z projections as one sliced mgemm (fp32 outputs, like the Linears); m <= 32"""
        mq = self.multi_qkvz
        m = bsz * seqlen
        hidden = self.qkv_proj.in_features
        x = x.half()
        if x.shape[-1] < hidden:
            x = torch.nn.functional.pad(x, (0, hidden - x.shape[-1]))
        x = x.contiguous().view(1, m, hidden)
        xh = torch.empty((mq.num_src, m, hidden), dtype = torch.half, device = x.device)
        qkv = torch.empty((bsz, seqlen, self.qkv_proj.out_features), dtype = torch.float, device = x.device)
        z = torch.empty((bsz, seqlen, self.z_proj.out_features), dtype = torch.float, device = x.device)
        c_ptrs = mq.c_ptrs([qkv.view(m, -1), z.view(m, -1)])
        ext.exl3_mgemm(
            x,
            mq.ptrs_trellis,
            self.prealloc_qkvz_carrier.expand(mq.num_slices, m, mq.width),
            mq.ptrs_suh,
            xh,
            mq.ptrs_svh,
            None,
            None,
            mq.K,
            -1,
            mq.mcg,
            mq.mul1,
            -1,
            -1,
            0,
            1,
            mq.size_n_list,
            c_ptrs,
            mq.n_stride_list,
            mq.had_src_list,
            mq.num_src,
        )
        return qkv, z


    def _bc_configure_slot(self, bsz: int, seqlen: int, history: bool, rsl = None):
        """Allocate (or fetch, if already cached at this exact shape) the per-(bsz, seqlen)
        statics for the BC_GatedDeltaNetSplit graph slot and hand them to C++. Called at most
        once per (bsz, seqlen, history) combination per layer instance, and again for a
        speculative-pass slot when a different cache's state runs through it"""
        device = self.device
        f = self.fdim_qkv
        nv, hv = self.num_v_heads, self.v_head_dim
        qkv             = g_tensor_cache.get(device, (bsz, seqlen, f), torch.float, "s_qkv")
        z               = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.float, "s_z")
        ba              = g_tensor_cache.get(device, (bsz, seqlen, 2 * nv), torch.float, "s_ba")
        if history:
            # Speculative-pass slot: the scan inputs land in the cache's staging buffers, which
            # the rewind replays from (see GDNLayerState); the slot is bound to that cache
            conv_out, beta, g = rsl.staged_views(bsz, seqlen)
        else:
            beta            = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.bfloat16, "s_beta")
            g               = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.float, "s_g")
            conv_out        = g_tensor_cache.get(device, (bsz, seqlen, f), torch.bfloat16, "s_conv")
        mixed_qkv       = g_tensor_cache.get(device, (bsz, f, seqlen), torch.bfloat16, "s_mqkv")
        core_attn_out   = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.bfloat16, "s_cao")
        core_attn_out_f = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_caof")
        qkv_xh = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_qkv_xh")
        z_xh   = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_z_xh")
        o_xh   = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_o_xh")
        self.bc.configure_slot(
            bsz, seqlen, history,
            qkv, z, ba, beta, g, mixed_qkv, conv_out, core_attn_out, core_attn_out_f,
            qkv_xh, z_xh, o_xh,
        )
        if history:
            self.bc.set_stage_key(bsz, seqlen, True, id(rsl))


    def _pfe_f16_out(self, m, x: torch.Tensor, params: dict, env: str) -> bool:
        # EXL3_PFE_QKVZ / EXL3_PFE_OPROJ (default 1): request this projection's output in fp16 instead of fp32.
        # Where the fp32 output is an fp16 GEMM plus a widen (_f32_via_f16, HIP reconstruct path) the fp16
        # buffer holds the same values and the consumer widens in-kernel: drops the widen copy, bit-exact
        if os.environ.get(env, "1") != "1" or "capture" in params or "quant_preserve" in params:
            return False
        from .quant import exl3 as _exl3
        rows = x.numel() // x.shape[-1]
        if not (_exl3._f32_via_f16 and rows >= _exl3._f32_via_f16_min_rows):
            return False
        if type(m.inner).__name__ != "LinearEXL3" or m.inner.bias is not None:
            return False
        if rows <= _exl3.AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct \
                or params.get("reconstruct"):
            return False
        return (
            not m.lora_a_tensors and m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
            m.out_features == m.out_features_unpadded
        )

    def _kda_qkv_f16(self, x: torch.Tensor) -> bool:
        # EXL3_KDA_QKV_F16=1 (default): take qkv_proj's output in fp16 instead of fp32. On HIP the fp32 output
        # is an fp16 GEMM plus a widen (_f32_via_f16), so the fp16 buffer holds the same values; the
        # conv casts to bf16 in-kernel and accumulates in fp32. Drops a ~200 MB fp32 write per KDA
        # layer per 2K chunk (and the MALL flush it causes). Bit-exact only on the reconstruct path
        if os.environ.get("EXL3_KDA_QKV_F16", "1") != "1":
            return False
        from .quant import exl3 as _exl3
        m = self.qkv_proj
        rows = x.numel() // x.shape[-1]
        if not (_exl3._f32_via_f16 and rows >= _exl3._f32_via_f16_min_rows):
            return False
        if type(m.inner).__name__ != "LinearEXL3" or m.inner.bias is not None:
            return False
        if rows <= _exl3.AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
            return False
        return (
            not m.lora_a_tensors and m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
            m.out_features == m.out_features_unpadded
        )

    def _kda_f16_gates(self, x: torch.Tensor) -> bool:
        # EXL3_KDA_F16_GATES=1: keep the KDA gate projections in fp16 (no fp32 widen of the
        # 2048x8192 g_b/f_b outputs, ~0.5 ms each). Bit-exact only where LinearFP16 already
        # computes fp32 outputs as an fp16 GEMM plus a widen (HIP, rows >= the via-f16 threshold)
        # z is then fp16: the fused gated norm must take an fp16 gate (ext.py ROCM_KNOBS["gnorm_f16g"]),
        # else it drops to the torch fallback (5.9 vs 0.5 ms per call, -3 % 4K prefill; glm-next r3)
        if os.environ.get("EXL3_KDA_F16_GATES", "1") != "1":
            return False
        from .quant import fp16 as _fp16
        if not (_fp16._f32_via_f16 and x.numel() // x.shape[-1] >= _fp16._f32_via_f16_min_rows):
            return False
        mods = (self.g_a_proj, self.g_b_proj, self.b_proj, self.f_a_proj, self.f_b_proj)
        return all(
            m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
            m.inner._pinned_store is None and not m.lora_a_tensors and
            m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
            m.out_features == m.out_features_unpadded
            for m in mods
        )

    def _kda_dec_cat_ok(self) -> bool:
        ok = getattr(self, "_kda_dec_cat_ok_v", None)
        if ok is None:
            mods = (self.g_a_proj, self.g_b_proj, self.b_proj, self.f_a_proj, self.f_b_proj)
            ok = hasattr(ext, "skinny_cat") and all(
                m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
                m.inner._pinned_store is None and not m.lora_a_tensors and
                m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
                m.out_features == m.out_features_unpadded and m.inner.weight.is_contiguous()
                for m in mods
            ) and self.g_a_proj.out_features == self.f_a_proj.out_features \
              and self.g_a_proj.out_features * 2 + self.b_proj.out_features <= 512
            self._kda_dec_cat_ok_v = ok
        return ok

    def _kda_lr_fused_ok(self) -> bool:
        # kda_fb_gate / kda_gb_norm preconditions that do not depend on the input (cached)
        ok = getattr(self, "_kda_lr_fused_ok_v", None)
        if ok is None:
            wg, wf = self.g_b_proj.inner.weight, self.f_b_proj.inner.weight
            hd = self.num_v_heads * self.k_head_dim
            # kda_fb_gate scales b before the sigmoid: exact only for beta_scale == 1
            ok = hasattr(ext, "kda_fb_gate") and hasattr(ext, "kda_gb_norm") and _FUSE_KDA_GATE and \
                self.beta_scale == 1.0 and os.environ.get("EXL3_FUSE_GNORM", "1") != "0" and \
                self.g_a_proj.out_features == 128 and self.f_a_proj.out_features == 128 and \
                wg.shape == wf.shape == (128, hd) and self.v_head_dim == 128 and self.k_head_dim * self.num_v_heads == hd and \
                hd % 64 == 0 and wg.data_ptr() % 16 == 0 and wf.data_ptr() % 16 == 0 and \
                self.b_proj.out_features == self.num_v_heads and \
                self.dt_bias.dtype == torch.float and self.a_log.dtype == torch.float and \
                self.dt_bias.is_contiguous() and self.a_log.is_contiguous() and self.dt_bias.numel() == hd and \
                self.norm.gate_activation == "sigmoid" and \
                self.norm.out_dtype in (torch.half, torch.float) and \
                self.norm.weight.dtype in (torch.bfloat16, torch.float) and self.norm.weight.is_contiguous() and \
                self.norm.weight.numel() == self.norm.groups * self.v_head_dim
            self._kda_lr_fused_ok_v = ok
        return ok

    def _kda_dec_cat(self, x: torch.Tensor, params: dict):
        # EXL3_KDA_DEC_CAT=1 (decode, rows <= 8, fp16 x): g_a | f_a | b in ONE skinny launch
        # (ext.skinny_cat: same per-column reduction order as the separate skinny launches, so
        # bit-identical; g_a/f_a rounded to fp16 RNE in-kernel exactly like ha). skinny_cat_w reads
        # the three weights in place (no concat copy). With EXL3_KDA_DEC_LR_FUSED=1 the g_b/f_b second
        # stages move into kda_fb_gate / kda_gb_norm: returns (None, b, None, gaf) and the caller
        # runs them. Else EXL3_KDA_DEC_LR_BATCHED=1 runs g_b/f_b as one strided-batched GEMM
        lead = x.shape[:-1]
        rows = x.numel() // x.shape[-1]
        na = self.g_a_proj.out_features
        gaf = torch.empty((2, rows, na), dtype = torch.half, device = x.device)
        b = torch.empty((*lead, self.b_proj.out_features), dtype = torch.float, device = x.device)
        if hasattr(ext, "skinny_cat_w"):
            ext.skinny_cat_w(x.view(rows, x.shape[-1]),
                             [m.inner.weight for m in (self.g_a_proj, self.f_a_proj, self.b_proj)],
                             [gaf[0], gaf[1], b])
        else:
            w = getattr(self, "kda_dec_cat_w", None)
            if w is None:
                w = torch.cat([m.inner.weight for m in (self.g_a_proj, self.f_a_proj, self.b_proj)], 1).contiguous()
                self.kda_dec_cat_w = w
            ext.skinny_cat(x.view(rows, x.shape[-1]), w, [gaf[0], gaf[1], b])
        if KDA_KNOBS["lr_fused"] and self._kda_lr_fused_ok():
            return None, b, None, gaf
        if KDA_KNOBS["lr_batched"] and self.g_b_proj.inner.weight.shape == self.f_b_proj.inner.weight.shape:
            lw = getattr(self, "kda_dec_lr_w", None)
            if lw is None:
                lw = torch.stack([self.g_b_proj.inner.weight, self.f_b_proj.inner.weight]).contiguous()
                self.kda_dec_lr_w = lw
            zf = torch.empty((2, rows, lw.shape[-1]), dtype = torch.float, device = x.device)
            ext.hgemm_batched(gaf, lw, zf)
            z = zf[0].view(*lead, lw.shape[-1])
            f = zf[1].view(*lead, lw.shape[-1])
        else:
            z = self.g_b_proj.forward(gaf[0].view(*lead, na), params)
            f = self.f_b_proj.forward(gaf[1].view(*lead, na), params)
        return z, b, f, None

    def _kda_gb_norm(self, core_attn_out: torch.Tensor, gaf: torch.Tensor, params: dict) -> torch.Tensor:
        # g_b folded into the gated norm (z only feeds the norm). Same conditions as the fused norm
        # path GatedRMSNorm.forward would take; otherwise materialize z and use the norm as before
        n = self.norm
        if core_attn_out.dtype == torch.bfloat16 and core_attn_out.is_contiguous():
            y = torch.empty_like(core_attn_out, dtype = n.out_dtype)
            ext.kda_gb_norm(core_attn_out, gaf[0], self.g_b_proj.inner.weight, n.weight, y,
                            n.rms_norm_eps, n.constant_bias, n.groups, n.gate_first, 1)
            return y
        z = self.g_b_proj.forward(gaf[0].view(*core_attn_out.shape[:-2], gaf.shape[-1]), params)
        z = z.view(core_attn_out.shape)
        # EXL3_NC2_KDA_OPAD=1 (read per call): write the norm output at the o_proj row pitch, so o_proj skips its
        # pad_copy_in (same kernel, only the store address moves: bit-exact)
        o_pitch = 0
        _pad_ok = getattr(ext.gated_rms_norm, "pad_ok", None)
        if os.environ.get("EXL3_NC2_KDA_OPAD", "0") == "1" and _pad_ok is not None and not n.gate_first and \
                _pad_ok(core_attn_out, n.weight, n.out_dtype or torch.half, z, n.groups,
                        1 if n.gate_activation == "sigmoid" else 0):
            o_pitch = _row_pad_pitch(self.o_proj, self.num_v_heads * self.v_head_dim,
                                     core_attn_out.numel() // (self.num_v_heads * self.v_head_dim), params)
        return n.forward(core_attn_out, params, gate = z, out_pitch = o_pitch)

    def _kda_gates_f16(self, x: torch.Tensor, params: dict, small_first: bool):
        h = torch.half
        if os.environ.get("EXL3_KDA_SMALL_CAT", "0") == "1":
            # g_a | b | f_a share x: one N=320 GEMM instead of three N<=128 ones (48 vs 3x16 WGs)
            w = getattr(self, "kda_small_cat_w", None)
            if w is None:
                w = torch.cat([m.inner.weight for m in (self.g_a_proj, self.b_proj, self.f_a_proj)], 1).contiguous()
                self.kda_small_cat_w = w
            x2 = x.view(-1, x.shape[-1])
            y = torch.matmul(x2, w).view(*x.shape[:-1], w.shape[1])
            n_ga, n_b = self.g_a_proj.out_features, self.b_proj.out_features
            ga, b, fa = y[..., :n_ga], y[..., n_ga:n_ga + n_b], y[..., n_ga + n_b:]
        elif small_first:
            # x-consumers back to back while x is still in the MALL
            ga = self.g_a_proj.forward(x, params, out_dtype = h)
            b = self.b_proj.forward(x, params, out_dtype = h)
            fa = self.f_a_proj.forward(x, params, out_dtype = h)
        else:
            ga = self.g_a_proj.forward(x, params, out_dtype = h)
            b = fa = None
        z = self.g_b_proj.forward(ga, params, out_dtype = h)
        if b is None:
            b = self.b_proj.forward(x, params, out_dtype = h)
        beta = (torch.sigmoid(b.float()) * self.beta_scale).to(torch.bfloat16)
        if fa is None:
            fa = self.f_a_proj.forward(x, params, out_dtype = h)
        f = self.f_b_proj.forward(fa, params, out_dtype = h)
        return z, beta, f

    @override
    def _midchunk_segments(self, params: dict, bsz: int, seqlen: int, save_state: bool, save_history: bool):
        """Row bounds [0, r1, .., T] for a mid-chunk checkpoint split, or None. Only the KDA chunk path, bsz 1,
        rows % 64 == 0 (fla chunk BT) and every segment > 256 rows (same conv kernel as the unsplit call)."""
        rows = params.get("checkpoint_rows")
        if not rows or not self.kda or bsz != 1 or not save_state or save_history or params.get("midchunk_mode", 1) != 1:
            return None
        rows = sorted(set(int(r) for r in rows))
        segs = [0] + rows + [seqlen]
        if any(r % 64 for r in rows) or any(b - a <= 256 for a, b in zip(segs[:-1], segs[1:])):
            return None
        return segs


    def _midchunk_kda_ckpt(self, params: dict, bsz: int, seqlen: int, save_state: bool, save_history: bool,
                           mixed_qkv: torch.Tensor):
        """EXL3_MIDCHUNK_CKPT=2: one unsplit call; the HIP KDA kernel side-writes S at the checkpoint row, conv state
        there = the last cdim pre-conv inputs (bf16). Single row, rows % 64 == 0 only. Returns the ckpt dict or None."""
        rows = params.get("checkpoint_rows")
        if not rows or params.get("midchunk_mode") != 2 or bsz != 1 or not save_state or save_history:
            return None
        # GDN (not KDA): only the fused HIP chain (EXL3_GDN_FUSE=1 + EXL3_GDN_PF=1, chunk path) side-writes the state
        if not self.kda and (os.environ.get("EXL3_GDN_FUSE", "0") != "1" or os.environ.get("EXL3_GDN_PF", "1") != "1"
                             or seqlen < self.num_v_heads):
            return None
        rows = sorted(set(int(r) for r in rows))
        cdim = self.conv_kernel_size
        # EXL3_PF_NO_TAIL=1: the job also asks for the prompt's last full page, so up to 2 rows (kernel side-writes 2)
        max_rows = 2 if os.environ.get("EXL3_PF_NO_TAIL", "0") == "1" else 1
        if self.kda and os.environ.get("EXL3_NC2_CKPT4", "0") == "1":
            max_rows = 4  # CK4 kernel build (vendor/fla/hip/kda_fused_h_hip.py): 8192-row chunks
        if len(rows) > max_rows or any(r % 64 or not cdim <= r < seqlen for r in rows):
            return None
        cks = [{
            "row": r,
            "chunk": r // 64,
            "s": torch.empty((1, self.num_v_heads, self.k_head_dim, self.v_head_dim), dtype = torch.float, device = self.device),
            "conv": mixed_qkv[0, :, r - cdim:r].to(torch.bfloat16).contiguous(),
        } for r in rows]
        return cks[0] if len(cks) == 1 else cks


    def _midchunk_kda_d2h(self, params: dict, ck: dict, state_dtype: torch.dtype):
        """Async D2H of a written mode-2 checkpoint into pinned buffers on a side stream; the event is synced at
        stash time (Job.stash_midchunk). S lands in the pool row layout and dtype (GDNLayerState.stash)."""
        if not ck.get("ok"):
            return
        global _MIDCKPT_STREAM
        if _MIDCKPT_STREAM is None:
            _MIDCKPT_STREAM = torch.cuda.Stream(device = self.device)
        side = _MIDCKPT_STREAM
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            s_dev = ck["s"].unsqueeze(0).to(state_dtype)
            hs = torch.empty(s_dev.shape, dtype = s_dev.dtype, pin_memory = True)
            hc = torch.empty(ck["conv"].shape, dtype = ck["conv"].dtype, pin_memory = True)
            hs.copy_(s_dev, non_blocking = True)
            hc.copy_(ck["conv"], non_blocking = True)
            ev = torch.cuda.Event()
            ev.record(side)
        ck["s"].record_stream(side)
        s_dev.record_stream(side)
        ck["conv"].record_stream(side)
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        params.setdefault("midchunk_states", {}).setdefault(ck["row"], {})[layer_instance] = (hs, hc, ev)


    def _midchunk_conv_core(self, mixed_qkv, beta, g, conv_state, recurrent_state, recurrent_slots, slots_scan,
                            params, segs, g_cumsum):
        """Conv + KDA core as one call per segment, state chained in place through the base row. After each
        segment but the last, save GPU clones of the slices GDNLayerState.stash() copies (S, conv) in
        params["midchunk_states"][row][layer_instance]."""
        rs0 = params["recurrent_states"][0]
        slot = rs0.slot
        row = 2 * slot + getattr(rs0, "parity", 0)
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        cdim = self.conv_kernel_size
        saved = params.setdefault("midchunk_states", {})
        outs = []
        for i, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
            last = i == len(segs) - 2
            qkv_s = causal_conv1d_update(
                mixed_qkv = mixed_qkv[:, :, a:b],
                conv_state = conv_state,
                recurrent_slots = recurrent_slots,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                history = False,
                params = params,
            )
            if not last:
                conv_snap = conv_state[slot, :, :cdim].clone()
            outs.append(gated_delta_rule_fn(
                mixed_qkv = qkv_s,
                beta = beta[:, a:b],
                g = g[:, a:b],
                recurrent_state = recurrent_state,
                recurrent_slots = slots_scan,
                history = False,
                save_state = True,
                num_k_heads = self.num_k_heads,
                num_v_heads = self.num_v_heads,
                k_dim = self.k_dim,
                v_dim = self.v_dim,
                k_head_dim = self.k_head_dim,
                v_head_dim = self.v_head_dim,
                params = params,
                channelwise_g = True,
                g_cumsum = g_cumsum,
            ))
            if not last:
                saved.setdefault(b, {})[layer_instance] = (recurrent_state[row : row + 1].clone(), conv_snap)
        return torch.cat(outs, dim = 1)


    def note_graph_replay(self, params: dict, bsz: int, seqlen: int):
        """Host-side effects of a forward replayed from a captured graph (block_graph) instead of run: a speculative
        pass records the shape of its staged scan inputs for the rewind's replay, as forward() does"""
        rsg = params.get("recurrent_states")
        if self.num_k_heads == 0 or not rsg or not params.get("recurrent_history"):
            return
        assert params.get("recurrent_spec_step") is None, "token-by-token speculative steps are not graph-replayable"
        rsl = rsg[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
        rsl.staged_views(bsz, seqlen)


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_k_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                self.tp_collect(params["backend"], x, False)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)
        # A speculative pass fed one token at a time (recurrent draft models): step 0 is a plain
        # one-token speculative pass, later steps continue it in place on the scratch row with
        # incremental conv and token-major staging (GDNLayerState.staged_step_views)
        spec_step = params.get("recurrent_spec_step") if save_history else None
        scan_history = save_history and not spec_step

        # Post load, fuse conv1d weights if needed
        if self.conv1d_weight is None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None
        if self.conv1d_weight_flat is None:
            self.conv1d_weight_flat = self.conv1d_weight.squeeze(1).contiguous()

        # Previous state
        rsg = params.get("recurrent_states")
        if rsg:
            # Three slot tensors (prepare_for_recurrence): the cache slot for the conv ring, the
            # pool row the scan writes, and on a speculative pass the base row it starts from
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            slots_scan = get_for_device(params, "recurrent_slots_scan", self.device)
            slots_scan_in = get_for_device(params, "recurrent_slots_scan_in", self.device) if scan_history else None
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, recurrent_state = rsl.get_state_tensors()
            save_state = True
        else:
            recurrent_slots = slots_scan = slots_scan_in = None
            rsl = None
            conv_state, recurrent_state = None, None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity
            scan_history = False
            spec_step = None

        def stage(bsz, seqlen):
            # Staging rows (and their record for the rewind) of this pass
            return rsl.staged_step_views(bsz, spec_step) if spec_step is not None else rsl.staged_views(bsz, seqlen)

        # Deferred fill of the merged b/a projection (weights are materialized by now)
        if self.bc_split and not self.ba_weight_filled and self.kda:
            self.dt_bias_bc.copy_(self.dt_bias)
            self.kda_b_t.copy_(self.b_proj.inner.get_weight_tensor().T)
            self.kda_fa_t.copy_(self.f_a_proj.inner.get_weight_tensor().T)
            self.kda_fb_t.copy_(self.f_b_proj.inner.get_weight_tensor().T)
            self.kda_ga_t.copy_(self.g_a_proj.inner.get_weight_tensor().T)
            self.kda_gb_t.copy_(self.g_b_proj.inner.get_weight_tensor().T)
            self.ba_weight_filled = True
        elif self.bc_split and not self.ba_weight_filled:
            self.ba_weight_t.copy_(torch.cat([
                self.b_proj.inner.get_weight_tensor(),
                self.a_proj.inner.get_weight_tensor(),
            ], dim = -1).T)
            if self.ba_bias is not None:
                nv = self.num_v_heads
                b_bias = self.b_proj.inner.get_bias_tensor()
                a_bias = self.a_proj.inner.get_bias_tensor()
                if b_bias is None: b_bias = torch.zeros(nv, dtype = torch.half, device = self.device)
                if a_bias is None: a_bias = torch.zeros(nv, dtype = torch.half, device = self.device)
                self.ba_bias.copy_(torch.cat([b_bias, a_bias]))
            self.ba_weight_filled = True

        # Fused C++ path for decode with split projections, generalized over (bsz, seqlen) up to
        # (_BC_MAX_BSZ, _BC_MAX_QLEN) and over save_history (needed for MTP draft/verify). Runs
        # the entire layer in one call, replayed through an internal CUDA graph per (bsz, seqlen,
        # history) shape from the third invocation of that shape on
        if (
            self.bc_split and save_state and
            recurrent_slots is not None and
            1 <= bsz <= _BC_MAX_BSZ and 1 <= seqlen <= _BC_MAX_QLEN and
            not spec_step
        ):
            stage_key = id(rsl) if save_history else 0
            if self.bc.needs_configure(bsz, seqlen, save_history, stage_key):
                if self.kda:
                    self._bc_configure_slot_kda(bsz, seqlen, save_history, rsl)
                else:
                    self._bc_configure_slot(bsz, seqlen, save_history, rsl)
            if save_history:
                stage(bsz, seqlen)   # records the pass shape for the rewind (step 0 shares the one-token slot's rows)
            y = torch.empty_like(x, dtype = self.out_dtype or torch.half)
            self.bc.run_bszN(x, y, conv_state, recurrent_state, recurrent_slots, slots_scan, save_history, slots_scan_in)
            if self.tp_reduce:
                self.tp_collect(params["backend"], y)
            return to2(y, out_dtype, self.out_dtype)

        # Torch path. A speculative pass writes its scan inputs (conv output, beta, g) into the
        # cache's staging buffers for the rewind's replay; the gates are produced there where a
        # branch allows it, else copied in, and the conv output is copied in
        if save_history:
            staged_conv, staged_beta, staged_g = stage(bsz, seqlen)
        else:
            staged_conv = staged_beta = staged_g = None
        # Qwen3.5 uses split projections (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a),
        # while Qwen3-Next uses fused projections. The fused C++ helper expects the
        # packed layout used by fused projections; applying it to split qkv tensors
        # causes incorrect head ordering and broken generations.
        g_cumsum = False
        lr_gaf = None
        if self.qkvz_proj is not None and self.ba_proj is not None:
            qkvz = self.qkvz_proj.forward(x, params)
            ba = self.ba_proj.forward(x, params)

            mixed_qkv = torch.empty((bsz, self.fdim_qkv, seqlen), dtype = torch.bfloat16, device = self.device)
            z = torch.empty((bsz, seqlen, self.num_v_heads, self.v_head_dim), dtype = torch.bfloat16, device = self.device)
            beta = staged_beta if save_history else \
                torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
            g = staged_g if save_history else \
                torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

            ext.gated_delta_net_fused_op(
                qkvz, ba,
                self.dt_bias,
                self.a_log,
                mixed_qkv, z, beta, g,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.beta_scale
            )
            del qkvz, ba
        elif self.kda:
            # EXL3_KDA_SMALL_FIRST=1: run the small x-consumers (g_a, b, f_a) before qkv_proj.
            # qkv_proj writes a ~200 MB output that evicts x from the 32 MB MALL; the small-N
            # GEMMs (16 workgroups, DRAM-latency bound) then take 1.0-1.5 ms instead of 0.2 ms.
            # Same kernels, same inputs: bit-exact. End-to-end gain alone was noise (+0.1%, step 3 A/B)
            small_first = os.environ.get("EXL3_KDA_SMALL_FIRST", "1") == "1"
            qkv_dtype = torch.half if self._kda_qkv_f16(x) else _proj_dtype
            if not small_first:
                qkv = self.qkv_proj.forward(x, params, qkv_dtype)

            b = None
            beta = None
            if self.kda_direct:
                # Direct gates (Ling 3.x): full-rank forget / output gate projections
                # (z stays fp32: the gated norm takes a bf16 or fp32 gate, not fp16)
                z = self.g_proj.forward(x, params)
                b = self.b_proj.forward(x, params, out_dtype = _gate_dtype)
                f = self.f_proj.forward(x, params, out_dtype = _gate_dtype)
            elif self._kda_f16_gates(x):
                z, beta, f = self._kda_gates_f16(x, params, small_first)
            else:
                # Low-rank sigmoid output gate stands in for z (applied by the gated norm)
                # r35: at decode (rows <= 8, half x) the skinny hgemm rounds its fp32 sum straight to
                # fp16 (RNE), identical to fp32 out + .to(half), minus one copy launch per projection
                ha = torch.half if (KDA_KNOBS["half_a"] and x.dtype == torch.half and
                                    x.numel() // x.shape[-1] <= 8) else None
                if ha is not None and KDA_KNOBS["dec_cat"] and self._kda_dec_cat_ok():
                    z, b, f, lr_gaf = self._kda_dec_cat(x, params)
                else:
                    z = self.g_b_proj.forward(self.g_a_proj.forward(x, params, ha or torch.half).to(torch.half), params)
                    b = self.b_proj.forward(x, params, out_dtype = _gate_dtype)
                    f = self.f_b_proj.forward(self.f_a_proj.forward(x, params, ha or torch.half).to(torch.half), params,
                                              out_dtype = _gate_dtype)
            if z is not None:
                z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)

            if small_first:
                qkv = self.qkv_proj.forward(x, params, qkv_dtype)
            # EXL3_KDA_FUSED_CAST=1 (default): hand the conv the fp16 (bsz, f, seq) view; it reads the
            # projection output in place (the strided Triton conv, or one cast-copy for the short CUDA conv)
            if _conv_token_major and os.environ.get("EXL3_KDA_FUSED_CAST", "1") != "0":
                mixed_qkv = qkv.transpose(1, 2)
            else:
                mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
            del qkv

            # Per-k-channel log decay from the low-rank forget gate: "safe gate" form when a
            # lower bound is configured, else softplus (per the Kimi/GLM5.3 reference).
            # f may be fp16 (EXL3_KDA_F16_GATES). beta_scale multiplies after the sigmoid, as in the
            # kernels and the reference (beta in [0, 2] for negative eigenvalues); the fused ROCm gate
            # kernels (kda_gate, kda_fb_gate) scale before it, so they only take beta_scale == 1
            # EXL3_KDA_G_FUSED=1 (default, +2.0% 4K prefill, bit-exact): when the fla chunk path will run
            # (prefill, no history), one Triton pass computes the gate AND fla's chunk-local cumsum
            # (bit-equal to the torch chain + chunk_local_cumsum; kda_gate.py uses BS=32 tiles for that).
            # Saves ~2.9 ms per KDA layer per 2048-row chunk (3.5 -> 0.6 ms, scratch/dg/s5_mb.py)
            g_cumsum = (
                os.environ.get("EXL3_KDA_G_FUSED", "1") == "1"
                and seqlen >= self.num_v_heads and not save_history
            )
            if lr_gaf is not None:
                # EXL3_KDA_DEC_LR_FUSED: f = f_a @ f_b and the gate in one launch (kda_gate math)
                g_cumsum = False
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = x.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads, self.k_head_dim), dtype = torch.float, device = x.device)
                lb = self.gate_lower_bound
                ext.kda_fb_gate(lr_gaf[1], self.f_b_proj.inner.weight, b, self.dt_bias, self.a_log,
                                float(self.beta_scale), float(lb or 0.0), lb is not None, beta, g)
            elif g_cumsum:
                if beta is None:
                    beta = (torch.sigmoid(b.float()) * self.beta_scale).to(torch.bfloat16)
                from .gated_delta_net_fn.kda_gate import kda_gate
                from ..vendor.fla import RCP_LN2
                g = kda_gate(
                    f.contiguous(), self.dt_bias.float(), torch.exp(self.a_log.float()),
                    self.num_v_heads, self.k_head_dim, self.gate_lower_bound,
                    cumsum_chunk = 64, cumsum_scale = RCP_LN2,
                )
            elif beta is None and _FUSE_KDA_GATE and self.beta_scale == 1.0 \
                    and b.dtype == torch.float and f.dtype == torch.float \
                    and self.dt_bias.dtype == torch.float and self.a_log.dtype == torch.float \
                    and b.is_contiguous() and f.is_contiguous():
                # One launch for beta and g (fused_elt_rocm.cu kda_gate), same per-element math
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = f.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads, self.k_head_dim), dtype = torch.float, device = f.device)
                lb = self.gate_lower_bound
                ext.kda_gate(b, f, self.dt_bias, self.a_log, float(self.beta_scale),
                             float(lb or 0.0), lb is not None, beta, g)
            else:
                if beta is None:
                    beta = (torch.sigmoid(b.float()) * self.beta_scale).to(torch.bfloat16)
                # One fp32 (rows, v_heads, k_head_dim) tensor for the whole chain: F.softplus with the
                # threshold is the where/log1p/exp expression in one kernel, and the decay scaling is
                # in place. The op-by-op form held five of these (2.5 GB per 16k rows on GLM-5.3)
                gf = f.float() if f.dtype != torch.float else f.clone()
                gf = gf.add_(self.dt_bias.float().view(1, 1, -1)).view(bsz, seqlen, self.num_v_heads, self.k_head_dim)
                decay = torch.exp(self.a_log.float()).view(1, 1, self.num_v_heads, 1)
                if self.gate_lower_bound is not None:
                    g = torch.sigmoid(gf.mul_(decay)).mul_(self.gate_lower_bound)
                else:
                    g = F.softplus(gf, threshold = 20.0).mul_(-decay)
                del gf
            del b, f
        else:
            # EXL3_QWG_ABFIRST=1 runs the two skinny b / a projections before qkv / z (independent kernels, bit-exact)
            ab_first = os.environ.get("EXL3_QWG_ABFIRST", "0") == "1"
            if ab_first:
                b = self.b_proj.forward(x, params)
                a = self.a_proj.forward(x, params)
            qzab = self.project_qkvz_ab_dec(x, params, bsz, seqlen) if (_QKVZ_DEC_MULTI and not ab_first and _qwd_ab_on()) else None
            qz = None if qzab is not None else (self.project_qkvz_dec(x, params, bsz, seqlen) if _QKVZ_DEC_MULTI else None)
            if qzab is not None:
                qkv, z, b, a, beta_ab, g_ab = qzab
            elif qz is not None:
                qkv, z = qz
            elif getattr(self, "multi_qkvz", None) is not None and bsz * seqlen <= 32:
                qkv, z = self.project_qkvz_sliced(x, bsz, seqlen)
            else:
                qkv_dt = torch.half if self._pfe_f16_out(self.qkv_proj, x, params, "EXL3_PFE_QKVZ") else _proj_dtype
                z_dt = torch.half if self._pfe_f16_out(self.z_proj, x, params, "EXL3_PFE_QKVZ") else None
                qkv = self.qkv_proj.forward(x, params, qkv_dt)
                z = self.z_proj.forward(x, params, z_dt)
            z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)
            if not ab_first and qzab is None:
                _c = getattr(self, "_qd3_ba", None)
                if QWD3["skip"] and _c is not None and _c[0] == x.shape:
                    b, a = _c[1], _c[2]
                else:
                    ba = self._ab_cat(x, params, bsz, seqlen) if QWD3.get("abcat", _ABCAT) else None
                    if ba is not None:
                        b, a = ba
                    else:
                        b = self.b_proj.forward(x, params)
                        a = self.a_proj.forward(x, params)
                    self._qd3_ba = (x.shape, b, a)

            # EXL3_GDN_FUSED_CAST=1 (default): the conv reads the fp16 transposed view in place (strided Triton
            # conv, or one cast-copy for the short CUDA conv), replacing two full passes (strided cast +
            # transpose copy) over qkv
            if _conv_token_major and os.environ.get("EXL3_GDN_FUSED_CAST", "1") != "0":
                mixed_qkv = qkv.transpose(1, 2)
            else:
                mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
            del qkv

            if qzab is not None and beta_ab is not None:
                beta, g = beta_ab, g_ab
            else:
                _c = getattr(self, "_qd3_bg", None)
                if QWD3["skip"] >= 2 and _c is not None and _c[0] == x.shape:
                    beta, g = _c[1], _c[2]
                else:
                    beta = staged_beta if save_history else \
                        torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
                    g = staged_g if save_history else \
                        torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)
                    ext.gated_delta_net_fused_op_2(
                        b, a,
                        self.dt_bias,
                        self.a_log,
                        beta, g,
                        self.beta_scale
                    )
                    self._qd3_bg = (x.shape, beta, g)

        # Speculative pass: scan inputs not produced in the staging rows are copied in
        if save_history:
            if beta is not staged_beta:
                beta = staged_beta.copy_(beta.view(staged_beta.shape))
            if g is not staged_g:
                g = staged_g.view(g.shape).copy_(g)

        # Mid-chunk recurrent checkpoints (EXL3_MIDCHUNK_CKPT, set by the job via params): split conv + KDA
        # core at the given rows, chaining conv state and S, and save the state at each split row
        segs = self._midchunk_segments(params, bsz, seqlen, save_state, save_history)
        if segs is not None:
            core_attn_out = self._midchunk_conv_core(
                mixed_qkv, beta, g, conv_state, recurrent_state, recurrent_slots, slots_scan, params, segs, g_cumsum
            )
        else:
            ck = self._midchunk_kda_ckpt(params, bsz, seqlen, save_state, save_history, mixed_qkv)
            if ck is not None:
                params["kda_ckpt"] = ck
            # Convolution. EXL3_PF_GLUE=1 (default off, bit-exact): KDA chunk path writes q | k | v planes
            plane = (self.k_dim if os.environ.get("EXL3_PF_GLUE", "0") == "1" and self.kda
                     and self.k_dim == self.v_dim and not save_history and seqlen >= self.num_v_heads else 0)
            mixed_qkv = causal_conv1d_update(
                mixed_qkv = mixed_qkv,
                conv_state = conv_state,
                recurrent_slots = recurrent_slots,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                history = save_history,
                params = params,
                incremental = bool(spec_step),
                plane = plane,
            )
            if save_history:
                mixed_qkv = staged_conv.copy_(mixed_qkv)

            # Delta rule
            core_attn_out = gated_delta_rule_fn(
                mixed_qkv = mixed_qkv,
                beta = beta,
                g = g,
                recurrent_state = recurrent_state,
                recurrent_slots = slots_scan,
                slots_in = slots_scan_in,
                history = scan_history,
                save_state = save_state,
                num_k_heads = self.num_k_heads,
                num_v_heads = self.num_v_heads,
                k_dim = self.k_dim,
                v_dim = self.v_dim,
                k_head_dim = self.k_head_dim,
                v_head_dim = self.v_head_dim,
                params = params,
                channelwise_g = self.kda,
                g_cumsum = g_cumsum,
            )
            if ck is not None:
                params.pop("kda_ckpt")
                for c in (ck if isinstance(ck, list) else [ck]):
                    self._midchunk_kda_d2h(params, c, recurrent_state.dtype)
        del mixed_qkv, beta, g

        # Norm
        if lr_gaf is not None:
            core_attn_out = self._kda_gb_norm(core_attn_out, lr_gaf, params)
        else:
            o_pitch = 0
            _pad_ok = getattr(ext.gated_rms_norm, "pad_ok", None)
            # EXL3_NC2_KDA_NPAD=1 (read per call): KDA prefill too (GLM: 34 layers, o_proj K 8192), so o_proj skips its
            # pad_copy_in; same fused kernel, only the store address moves (bit-exact). The pitched forward needs fp16 out
            kda_ok = not self.kda or (os.environ.get("EXL3_NC2_KDA_NPAD", "0") == "1" and self.norm.out_dtype == torch.half)
            if _pad_ok is not None and kda_ok and not self.norm.gate_first and \
                    _pad_ok(core_attn_out, self.norm.weight, self.norm.out_dtype or torch.half, z, self.norm.groups,
                            1 if self.norm.gate_activation == "sigmoid" else 0):
                o_pitch = _row_pad_pitch(self.o_proj, self.num_v_heads * self.v_head_dim,
                                         core_attn_out.numel() // (self.num_v_heads * self.v_head_dim), params)
            core_attn_out = self.norm.forward(core_attn_out, params, gate = z, out_pitch = o_pitch)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)
        del z

        # Output projection (fp16 when the block's residual apply reads fp16: params["pfe_y_half"])
        o_dt = torch.half if params.get("pfe_y_half") and \
            self._pfe_f16_out(self.o_proj, core_attn_out, params, "EXL3_PFE_OPROJ") else None
        x = self.o_proj.forward(core_attn_out, params, o_dt)

        # TP reduction
        if self.tp_reduce:
            self.tp_collect(params["backend"], x)

        # EXL3_NC2_ATTN_F16 (default off): keep o_proj's fp16 output when the block asked for it (see MLAttention)
        if out_dtype is None and x.dtype == torch.half and params.get("pfe_y_half") and \
                os.environ.get("EXL3_NC2_ATTN_F16", "0") == "1":
            return x
        return to2(x, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        assert self.qkv_proj is not None
        assert self.b_proj is not None
        assert not self.kda_direct, "Tensor parallelism does not split direct KDA gates yet"
        # Head-split projections. KDA has no z/a projections; its low-rank gate pairs split on
        # the expanding side (f_b/g_b) while the contracting side (f_a/g_a) is replicated
        if self.kda:
            split_projs = [self.qkv_proj, self.b_proj, self.f_b_proj, self.g_b_proj, self.o_proj]
            storage_dev = self.f_a_proj.storage_size() + self.g_a_proj.storage_size()
        else:
            assert self.z_proj is not None
            assert self.a_proj is not None
            split_projs = [self.qkv_proj, self.z_proj, self.b_proj, self.a_proj, self.o_proj]
            storage_dev = 0
        storage = sum(p.storage_size() for p in split_projs)
        for cl in self.recurrent_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += 2 * self.num_k_heads * self.k_head_dim * torch.half.itemsize
        overhead_s += 2 * self.num_v_heads * self.v_head_dim * torch.half.itemsize
        recons = max(p.recons_size() for p in split_projs)
        channel_width = 1
        channels_to_split = self.num_k_heads
        assert self.num_v_heads % self.num_k_heads == 0, \
            "num_k_heads doesn't divide num_v_heads"
        while channel_width * self.k_head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.k_head_dim) % 128 == 0 and (channel_width * self.v_head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "K-heads",
            storage_per_device = storage_dev,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = channels_to_split,
            limit_key = "linear_attn"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            if child is None:
                return None
            if isinstance(child, torch.Tensor):
                return TPTensorWrapper.tp_export(child, plan, producer)
            else:
                return child.tp_export(plan, producer)

        return {
            "cls": GatedDeltaNet,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "k_head_dim": self.k_head_dim,
                "v_head_dim": self.v_head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "beta_scale": self.beta_scale,
                "out_dtype": self.out_dtype,
                # KDA mode is keyed on key_f_a; the projections themselves arrive prebuilt
                "key_f_a": self.key_f_a,
                "key_f_b": self.key_f_b,
                "key_g_a": self.key_g_a,
                "key_g_b": self.key_g_b,
                "gate_lower_bound": self.gate_lower_bound,
            },
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "num_kv_group": self.num_v_heads // self.num_k_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "qkv_proj",
                "z_proj",
                "b_proj",
                "a_proj",
                "o_proj",
                "norm",
                "conv1d_weight",
                "conv1d_bias",
                "a_log",
                "dt_bias",
                "f_a_proj",
                "f_b_proj",
                "g_a_proj",
                "g_b_proj",
            )},
            "device": self.device,
            "recurrent_layers": [
                rl.tp_export(plan) for rl in self.recurrent_layers
            ]
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        k_head_dim = exported["kwargs"]["k_head_dim"]
        v_head_dim = exported["kwargs"]["v_head_dim"]
        G = exported["num_kv_group"]
        global_num_k_heads = exported["num_k_heads"]
        global_num_v_heads = exported["num_v_heads"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "K-heads"
        num_k_heads = last - first
        num_v_heads = (last - first) * G

        q_split = (True, first * k_head_dim, last * k_head_dim) \
            if num_k_heads else None
        k_split = (True, (global_num_k_heads + first) * k_head_dim, (global_num_k_heads + last) * k_head_dim) \
            if num_k_heads else None
        # V rows follow the q and k rows, which are k_head_dim wide
        v_base = global_num_k_heads * 2 * k_head_dim
        v_split = (True, v_base + first * G * v_head_dim, v_base + last * G * v_head_dim) \
            if num_k_heads else None
        z_split = (True, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        o_split = (False, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        a_split = (True, first * G, last * G) \
            if num_k_heads else None
        b_split = (True, first * G, last * G) \
            if num_k_heads else None
        # KDA: f_b/g_b expand a head_dim-wide latent to the per-channel k/v widths (column split
        # by head range), f_a/g_a are replicated, and dt_bias is per k-channel, not per head
        kda = exported["kwargs"].get("key_f_a") is not None
        fb_split = (True, first * k_head_dim, last * k_head_dim) \
            if num_k_heads else None
        gb_split = (True, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        dt_split = fb_split if kda else a_split

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        def _import_split_3(name, split_0, split_1, split_2):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split_3(local_context, exported[name], plan, split_0, split_1, split_2) \
                if split_0 and exported.get(name) else None

        module = GatedDeltaNet(
            config = None,
            **exported["kwargs"],
            num_k_heads = num_k_heads,
            num_v_heads = num_v_heads,
            conv1d_weight = _import_split_3("conv1d_weight", q_split, k_split, v_split),
            conv1d_bias = _import_split_3("conv1d_bias", q_split, k_split, v_split),
            qkv_proj = _import_split_3("qkv_proj", q_split, k_split, v_split),
            z_proj = _import_split("z_proj", z_split),
            o_proj = _import_split("o_proj", o_split),
            b_proj = _import_split("b_proj", b_split),
            a_proj = _import_split("a_proj", a_split),
            norm = _import("norm"),
            a_log = _import_split("a_log", a_split),
            dt_bias = _import_split("dt_bias", dt_split),
            f_a_proj = _import("f_a_proj") if num_k_heads else None,
            f_b_proj = _import_split("f_b_proj", fb_split),
            g_a_proj = _import("g_a_proj") if num_k_heads else None,
            g_b_proj = _import_split("g_b_proj", gb_split),
        )

        if num_k_heads:
            recurrent_layers = exported["recurrent_layers"]
            if len(recurrent_layers):
                module.has_split_cache = True
                for rl in exported["recurrent_layers"]:
                    rli = rl["cls"](module, **rl["args"])
                    module.recurrent_layers.append(rli)
                    module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli

        # This rank's k-head range within the layer, for the disk tier's canonical checkpoint layout
        module.tp_kv_first = first
        module.tp_k_heads_total = global_num_k_heads
        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
            module.tp_owner = module.tp_single_owner(local_context, key)

        module.load_local(device)
        torch.cuda.synchronize()
        return module
