import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from util import resolve_device

device = resolve_device()


def _l2norm(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def _torch_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    slots: torch.Tensor | None,
    history: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
):
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim
    scale = k_head_dim ** -0.5

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    v = v.float().view(bsz, seqlen, num_v_heads, v_head_dim)
    g = g.float().exp()
    beta = beta.float()

    out = torch.empty((bsz, seqlen, num_v_heads, v_head_dim), dtype = torch.bfloat16, device = mixed_qkv.device)
    state_out = recurrent_state.clone()

    for bi in range(bsz):
        slot = int(slots[bi].item()) if slots is not None else bi
        state = state_out[slot, 0].clone()

        for t in range(seqlen):
            next_state = torch.empty_like(state)

            for vh in range(num_v_heads):
                kh = vh // group
                kv_mem = (state[vh] * k[bi, t, kh].unsqueeze(-1)).sum(dim = -2)
                v_t = v[bi, t, vh] - kv_mem * g[bi, t, vh]
                next_state[vh] = state[vh] * g[bi, t, vh] + \
                    k[bi, t, kh].unsqueeze(-1) * v_t.unsqueeze(-2) * beta[bi, t, vh]
                out[bi, t, vh] = ((next_state[vh] * q[bi, t, kh].unsqueeze(-1)).sum(dim = -2) * scale).bfloat16()

            state = next_state
            if history and t < seqlen - 1:
                state_out[slot, t + 1].copy_(state)

        state_out[slot, 0].copy_(state)

    return out, state_out


def _run_cuda_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    slots: torch.Tensor | None,
    history: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
):
    from exllamav3.ext import exllamav3_ext as ext

    out = torch.empty(
        (mixed_qkv.shape[0], mixed_qkv.shape[1], num_v_heads, v_head_dim),
        dtype = torch.bfloat16,
        device = mixed_qkv.device,
    )
    state = recurrent_state.clone()
    ext.cuda_recurrent_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        state,
        out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
    )
    torch.cuda.synchronize()
    return out, state


def _run_chunk_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
):
    chunk_gated_delta_rule = pytest.importorskip(
        "fla.ops.gated_delta_rule",
        reason = "fla-core required",
    ).chunk_gated_delta_rule

    bsz, _, _ = mixed_qkv.shape
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = q.view(bsz, -1, num_k_heads, k_head_dim)
    k = k.view(bsz, -1, num_k_heads, k_head_dim)
    v = v.view(bsz, -1, num_v_heads, v_head_dim)

    out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g = g,
        beta = beta,
        initial_state = recurrent_state[:, 0],
        output_final_state = True,
        use_qk_l2norm_in_kernel = True,
    )
    torch.cuda.synchronize()
    return out.to(torch.bfloat16), final_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 1, 1, 1, 64, 64),
        (2, 5, 2, 4, 64, 64),
        (3, 7, 2, 4, 128, 128),
        (1, 7, 16, 48, 128, 128),
        (1, 8, 16, 48, 128, 128),
        (1, 15, 16, 32, 128, 128),
        (2, 17, 16, 32, 128, 128),
        (1, 128, 4, 8, 256, 256),
        (2, 1024, 4, 8, 128, 128),
        (1, 2047, 2, 8, 128, 128),
        (1, 2049, 2, 8, 128, 128),
    ],
)
@torch.inference_mode()
def test_cuda_recurrent_gated_delta_rule_matches_torch(
    history,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
):
    torch.manual_seed(1234)

    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    state_len = seqlen if history else 1
    num_slots = bsz + 2

    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn(
        (num_slots, state_len, num_v_heads, k_head_dim, v_head_dim),
        dtype = torch.float,
        device = device,
    ) * 0.05
    slots = torch.arange(bsz, dtype = torch.int32, device = device) + 1

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    torch.testing.assert_close(cuda_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(cuda_state[:, :state_len], ref_state[:, :state_len], rtol = 5e-2, atol = 5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 64, 1, 1, 64, 64),
        (2, 65, 2, 4, 64, 64),
        (3, 127, 2, 4, 128, 128),
    ],
)
@torch.inference_mode()
def test_chunk_gated_delta_rule_matches_torch(
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
):
    torch.manual_seed(5678)

    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn(
        (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
        dtype = torch.float,
        device = device,
    ) * 0.05

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        None,
        False,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )
    chunk_out, chunk_state = _run_chunk_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        None,
        False,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    torch.testing.assert_close(chunk_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_state, ref_state[:, 0], rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_out, cuda_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_state, cuda_state[:, 0], rtol = 5e-2, atol = 5e-2)


def _torch_channelwise_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    slots: torch.Tensor | None,
    history: bool,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
):
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    scale = head_dim ** -0.5

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, head_dim))
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, head_dim))
    v = v.float().view(bsz, seqlen, num_v_heads, head_dim)
    g = g.float().exp()
    beta = beta.float()

    out = torch.empty((bsz, seqlen, num_v_heads, head_dim), dtype = torch.bfloat16, device = mixed_qkv.device)
    state_out = recurrent_state.clone()

    for bi in range(bsz):
        slot = int(slots[bi].item()) if slots is not None else bi
        state = state_out[slot, 0].clone()

        for t in range(seqlen):
            next_state = torch.empty_like(state)

            for vh in range(num_v_heads):
                kh = vh // group
                decayed = state[vh] * g[bi, t, vh].unsqueeze(-1)
                kv_mem = (decayed * k[bi, t, kh].unsqueeze(-1)).sum(dim = -2)
                v_t = v[bi, t, vh] - kv_mem
                next_state[vh] = decayed + \
                    k[bi, t, kh].unsqueeze(-1) * v_t.unsqueeze(-2) * beta[bi, t, vh]
                out[bi, t, vh] = ((next_state[vh] * q[bi, t, kh].unsqueeze(-1)).sum(dim = -2) * scale).bfloat16()

            state = next_state
            if history and t < seqlen - 1:
                state_out[slot, t + 1].copy_(state)

        state_out[slot, 0].copy_(state)

    return out, state_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("seqlen", [1, 2, 7])
@pytest.mark.parametrize("num_heads", [4, 48])
@torch.inference_mode()
def test_cuda_channelwise_gated_delta_rule_matches_torch(history, seqlen, num_heads):
    torch.manual_seed(4321)
    head_dim = 128
    bsz = 1

    qkv_dim = 2 * num_heads * head_dim + num_heads * head_dim
    state_len = seqlen if history else 1
    num_slots = bsz + 2

    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_heads, head_dim), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn(
        (num_slots, state_len, num_heads, head_dim, head_dim),
        dtype = torch.float,
        device = device,
    ) * 0.05
    slots = torch.arange(bsz, dtype = torch.int32, device = device) + 1

    ref_out, ref_state = _torch_channelwise_delta_rule(
        mixed_qkv, g, beta, recurrent_state, slots, history,
        num_heads, num_heads, head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv, g, beta, recurrent_state, slots, history,
        num_heads, num_heads, head_dim, head_dim,
    )

    torch.testing.assert_close(cuda_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(cuda_state[:, :state_len], ref_state[:, :state_len], rtol = 5e-2, atol = 5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("seqlen", [2, 4, 7, 8])
@torch.inference_mode()
def test_cuda_gated_delta_rule_history_slots_are_per_step(seqlen):
    """Speculative rewind reads slot t+1 to restore the state after t accepted tokens, so
    every intermediate slot -- not just the final one -- has to be the real running state.
    A kernel that carries the state in registers across timesteps loses exactly this if it
    stops writing a slot, and the final state stays correct while the rewind silently does
    not."""
    torch.manual_seed(99)
    bsz, num_k_heads, num_v_heads, head_dim = 1, 16, 48, 128
    qkv_dim = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    num_slots = 3

    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn(
        (num_slots, seqlen, num_v_heads, head_dim, head_dim), dtype = torch.float, device = device,
    ) * 0.05
    slots = torch.tensor([1], dtype = torch.int32, device = device)

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv, g, beta, recurrent_state, slots, True,
        num_k_heads, num_v_heads, head_dim, head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv, g, beta, recurrent_state, slots, True,
        num_k_heads, num_v_heads, head_dim, head_dim,
    )

    torch.testing.assert_close(cuda_out, ref_out, rtol = 5e-2, atol = 5e-2)
    for t in range(seqlen):
        # slot 0 is the final state; slot t (t >= 1) is the state after step t - 1
        torch.testing.assert_close(
            cuda_state[1, t], ref_state[1, t], rtol = 5e-2, atol = 5e-2,
            msg = lambda m, t = t: f"history slot {t} (of {seqlen}) diverged\n{m}",
        )

    # A slot left holding its input, or one step's state copied into every slot, would pass
    # every assert above if the reference were degenerate. It is not.
    untouched = recurrent_state[1, 1:]
    assert not torch.allclose(cuda_state[1, 1:], untouched, rtol = 1e-3, atol = 1e-3), \
        "history slots were not written at all"
    for t in range(1, seqlen - 1):
        assert not torch.allclose(cuda_state[1, t], cuda_state[1, t + 1], rtol = 1e-3, atol = 1e-3), \
            f"history slots {t} and {t + 1} are identical; the per-step state is not advancing"
