import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pytest
import torch
import torch.nn.functional as F

from exllamav3.modules import MLAttention
from exllamav3.cache import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE
from exllamav3.util.rope import RopeSettings, RopeStyle
from exllamav3.modules.attention_fn.mla_triton import has_triton

from test_mla import FakeConfig, rms_norm

# DSA-on-MLA (GLM-5.2): the lightning indexer selects index_topk tokens per query and the
# attention core gathers only those latent rows. These tests transcribe the reference indexer
# (transformers glm_moe_dsa) in plain torch and check, on random weights:
#
#   - the module's top-k selection against the reference scores (allowing a small overlap
#     slack at the k-th-score boundary, where fp16 kernel scores and fp32 reference scores
#     can order ties differently),
#   - the sparse attention output against a masked dense reference driven by the MODULE's own
#     selection (tight: this isolates the gather/attention math from boundary selection noise),
#   - dense equivalence when index_topk >= T,
#   - cross-layer sharing ("shared" layers consume the published selection),
#   - the cached path (paged indexer-key plane) against the cache-less path, chunked.

device = "cuda:0"
pytestmark = pytest.mark.skipif(not has_triton, reason = "requires Triton")


def build_dsa(H = 8, hidden = 512, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128,
              q_lora = 256, idx_heads = 4, idx_dim = 128, topk = 64, mode = "full",
              seed = 0, wscale = 0.085):
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape, scale = wscale):
        return (torch.randn(*shape, generator = g) * scale).half()

    key = "model.layers.0.self_attn"
    qk_head = nope + rope_dim
    t = {
        f"{key}.q_a_proj.weight": rnd(q_lora, hidden),
        f"{key}.q_a_layernorm.weight": (torch.randn(q_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.q_b_proj.weight": rnd(H * qk_head, q_lora),
        f"{key}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{key}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{key}.o_proj.weight": rnd(hidden, H * v_head),
    }
    if mode == "full":
        # Indexer weights run hotter than the attention ones so the relu keeps a healthy
        # mix of active and clamped scores and the top-k boundary is not pure noise
        t[f"{key}.indexer.wq_b.weight"] = rnd(idx_heads * idx_dim, q_lora, scale = 0.25)
        t[f"{key}.indexer.wk.weight"] = rnd(idx_dim, hidden, scale = 0.25)
        t[f"{key}.indexer.k_norm.weight"] = (torch.randn(idx_dim, generator = g) * 0.1 + 1).half()
        t[f"{key}.indexer.k_norm.bias"] = (torch.randn(idx_dim, generator = g) * 0.05).half()
        t[f"{key}.indexer.weights_proj.weight"] = rnd(idx_heads, hidden, scale = 0.25)

    rope_settings = RopeSettings(
        head_dim = rope_dim, rope_theta = 10000.0, rope_style = RopeStyle.GPTJ,
    )
    module = MLAttention(
        config = FakeConfig(t), key = key, layer_idx = 0, hidden_size = hidden,
        num_q_heads = H, kv_lora_rank = kv_lora, qk_nope_head_dim = nope,
        qk_rope_head_dim = rope_dim, v_head_dim = v_head, rope_settings = rope_settings,
        q_lora_rank = q_lora, rms_norm_eps = 1e-6,
        indexer_mode = mode, index_n_heads = idx_heads, index_head_dim = idx_dim,
        index_topk = topk,
    )
    module.load(torch.device(device))
    return module, {k: v.to(device) for k, v in t.items()}, key


def ref_index_scores(module, t, key, x, positions):
    """Reference lightning-indexer scores (B, S, S): relu(q . k) * D**-0.5, head-weighted,
    fp32, interleaved rope on the first rope_dim dims, -inf past the causal bound."""
    m = module
    bsz, S, _ = x.shape
    Hi, Di, rd = m.index_n_heads, m.index_head_dim, m.qk_rope_head_dim
    xf = x.float()

    q_resid = rms_norm(xf @ t[f"{key}.q_a_proj.weight"].float().T,
                       t[f"{key}.q_a_layernorm.weight"], m.norm_eps).half().float()
    q = (q_resid @ t[f"{key}.indexer.wq_b.weight"].float().T).view(bsz, S, Hi, Di)
    k = F.layer_norm(xf @ t[f"{key}.indexer.wk.weight"].float().T, (Di,),
                     t[f"{key}.indexer.k_norm.weight"].float(),
                     t[f"{key}.indexer.k_norm.bias"].float(), eps = 1e-6)
    k = k.view(bsz, S, 1, Di)

    q_rot, k_rot = m.rope.apply(
        q[..., :rd].half().contiguous(), k[..., :rd].half().contiguous(),
        0, positions, None, False, None, None, m.norm_eps, 0.0, None,
    )
    q = torch.cat([q_rot.float(), q[..., rd:]], dim = -1)
    k = torch.cat([k_rot.float(), k[..., rd:]], dim = -1).squeeze(2)

    scores = torch.einsum("bqhd,bkd->bqhk", q, k) * Di ** -0.5
    scores = F.relu(scores)
    w = (xf @ t[f"{key}.indexer.weights_proj.weight"].float().T) * Hi ** -0.5
    scores = torch.einsum("bqh,bqhk->bqk", w, scores)

    pos = positions.view(bsz, 1) + torch.arange(S, device = x.device).view(1, S)
    causal = pos.view(bsz, 1, S) > pos.view(bsz, S, 1)
    return scores.masked_fill(causal, -float("inf"))


def _sim_cache_quant(v, bits):
    """Round-trip a (..., D) fp16 tensor through the cache quantizer at `bits` (same CUDA
    kernels that fill CacheLayer_MLA_quant), so a reference can carry the cache's exact
    latent values."""
    D = v.shape[-1]
    rows = v.numel() // D
    pq = torch.empty((rows, D // 32 * bits), dtype = torch.int32, device = v.device)
    ps = torch.empty((rows, D // 32), dtype = torch.half, device = v.device)
    ext.quant_cache_cont(v.reshape(rows, D).half().contiguous(), pq, ps, 0.0)
    out = torch.empty((rows, D), dtype = torch.half, device = v.device)
    ext.dequant_cache_cont(pq, ps, out, 0.0)
    return out.view(v.shape)


def ref_forward_indices(module, t, key, x, positions, indices, ckv_quant_bits = 0):
    """Dense reference MLA restricted to a given per-query selection: the module's gathered
    output must match this regardless of how the selection was made. With ckv_quant_bits the
    latent is round-tripped through the cache quantizer (the values a CacheLayer_MLA_quant
    at that width holds; the fp16 rope key is exact in both)."""
    m = module
    bsz, S, _ = x.shape

    # Rebuild the dense score path (same transcription as test_mla.ref_forward) with an
    # additional selection mask
    H, nope, rope_dim, v_head = m.num_q_heads, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim
    xf = x.float()
    q = xf @ t[f"{key}.q_a_proj.weight"].float().T
    q = rms_norm(q, t[f"{key}.q_a_layernorm.weight"], m.norm_eps)
    q = q @ t[f"{key}.q_b_proj.weight"].float().T
    q = q.view(bsz, S, H, m.qk_head_dim)
    q_nope, q_pe = q[..., :nope], q[..., nope:]

    ckv_kpe = xf @ t[f"{key}.kv_a_proj_with_mqa.weight"].float().T
    ckv = rms_norm(ckv_kpe[..., :m.kv_lora_rank], t[f"{key}.kv_a_layernorm.weight"], m.norm_eps)
    if ckv_quant_bits:
        ckv = _sim_cache_quant(ckv.half(), ckv_quant_bits).float()
    k_pe = ckv_kpe[..., m.kv_lora_rank:].view(bsz, S, 1, rope_dim)

    q_pe, k_pe = m.rope.apply(
        q_pe.half().contiguous(), k_pe.half().contiguous(),
        0, positions, None, False, None, None, m.norm_eps, 0.0, None,
    )
    q_pe, k_pe = q_pe.float(), k_pe.float()

    kv = (ckv.half().float() @ t[f"{key}.kv_b_proj.weight"].float().T).view(bsz, S, H, nope + v_head)
    k_nope, v = kv[..., :nope], kv[..., nope:]
    k = torch.cat([k_nope, k_pe.expand(bsz, S, H, rope_dim)], dim = -1)
    q_full = torch.cat([q_nope, q_pe], dim = -1)

    scores = torch.einsum("bqhd,bkhd->bhqk", q_full, k) * m.sm_scale
    pos = positions.view(bsz, 1) + torch.arange(S, device = x.device).view(1, S)
    sel = torch.zeros((bsz, S, S), dtype = torch.bool, device = x.device)
    idx = indices.view(bsz, S, -1).long()
    bb, qq, kk = (idx >= 0).nonzero(as_tuple = True)
    sel[bb, qq, idx[bb, qq, kk]] = True
    allowed = (pos.view(bsz, 1, S) <= pos.view(bsz, S, 1)) & sel
    scores = scores.masked_fill(~allowed.unsqueeze(1), -float("inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v).reshape(bsz, S, H * v_head)
    return o @ t[f"{key}.o_proj.weight"].float().T


def rel_err(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)


def nc_forward(module, x, positions = None, params = None):
    p = {"attn_mode": "flash_attn_nc"}
    if positions is not None:
        p["positions"] = positions
    if params is not None:
        p.update(params)
    out = module.forward(x, p)
    return out, p


def rnd_x(shape, seed, scale = 0.5):
    """Seeded input activations.

    build_dsa carefully seeds a CPU generator for the WEIGHTS, but every test then drew x from
    the GLOBAL CUDA RNG, so the activations depended on whatever ran earlier in the process and
    the whole file was order-dependent. test_dsa_selection[300] is simply the one whose
    tolerance is tight enough to notice: on identical invocations it returned 12-passed, then
    59/64, then 57/64, always failing at the deepest query row -- which is where near-ties at
    the top-k boundary are densest, so it is the row a changing input perturbs first.
    """
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return (torch.randn(*shape, generator = g) * scale).half().to(device)


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_selection_never_leaves_the_valid_pool(S):
    """Selection must never return an entry that is not in the pool.

    `dsa_indexer_scores` allocates (R, S_stride) with S_stride a kernel constexpr deliberately
    decoupled from the visible pool length, so at S=300 the backing is 512 wide against 300 real
    columns and the tail is never written. `dsa_topk.cu` takes `int T = scores.size(1)`, and six
    of eight call sites pass `t_ptr = None`, so the scan bound is whatever width it is handed.

    That is safe only because `dsa_indexer_scores` ends with `return scores[:, :T]` -- callers
    never receive the padded width. This test pins that, because the property is load-bearing
    and invisible at the call site. See test_dsa_topk_bounds.py, which builds the unbounded case
    by hand to show the mechanism is real.

    The allocator is poisoned first so a regression surfaces as large-positive garbage winning
    the top-k rather than depending on whatever happened to be resident."""
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = S)
    bsz = 2
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + S)
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    poison = [torch.full((S, w), 6.0e4, dtype = torch.half, device = device)
              for w in (128, 256, 512, 1024, 2048)]
    del poison
    torch.cuda.synchronize()

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"].view(bsz, S, -1)

    bad = indices[indices >= S]
    assert bad.numel() == 0, \
        f"{bad.numel()} selected entries outside the {S}-entry pool, e.g. {bad[:8].tolist()}"


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_selection(S):
    """Module top-k against the reference scores, compared by SCORE rather than by index.

    The indexer's scores are ReLU-sparse, so a row can hold fewer than topk positive candidates
    and the k-th reference score is then exactly 0.0 -- a tie plateau dozens of entries wide.
    Which of those zeros a given implementation returns is arbitrary and every choice is equally
    correct, so an index-overlap assertion is ill-posed there: it failed 58/64 at row 289 of
    S=300 with all six missing AND all six substituted entries scoring exactly 0.00000.

    The well-posed invariant is the one top-k actually promises: every entry scoring strictly
    better than the k-th must be selected, and nothing scoring strictly worse may be. That is
    stricter than the old overlap tolerance wherever scores are distinct, and correctly
    indifferent across a tie."""
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = S)
    bsz = 2
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (S))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"].view(bsz, S, -1)

    ref_scores = ref_index_scores(module, t, key, x, positions)
    for b in range(bsz):
        for q_row in range(0, S, 17):
            k_eff = min(topk, q_row + 1)
            ref_row = ref_scores[b, q_row].float()
            kth = ref_row.topk(k_eff).values[-1].item()
            got = set(i for i in indices[b, q_row].tolist() if i >= 0)
            assert len(got) == k_eff, f"row {q_row}: {len(got)} selected, expected {k_eff}"

            # fp16 scores, so a strict comparison would re-litigate the boundary it is meant to
            # tolerate; scale the band by the row's magnitude rather than using an absolute eps.
            # Masked entries are -inf, so the scale must come from the finite side or tol is inf
            # and BOTH assertions below silently accept everything.
            top = ref_row.max().item()
            tol = 1e-3 * max(1.0, abs(top))
            missed = [i for i in (ref_row > kth + tol).nonzero().flatten().tolist() if i not in got]
            assert not missed, \
                f"row {q_row}: {len(missed)} entries scoring above the k-th were not selected, " \
                f"e.g. {[(i, round(ref_row[i].item(), 5)) for i in missed[:4]]} vs kth={kth:.5f}"
            worse = [i for i in got if ref_row[i].item() < kth - tol]
            assert not worse, \
                f"row {q_row}: {len(worse)} selected entries score below the k-th, " \
                f"e.g. {[(i, round(ref_row[i].item(), 5)) for i in worse[:4]]} vs kth={kth:.5f}"


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_sparse_output(S):
    """Gathered attention output against the masked dense reference, driven by the module's
    own selection (so a boundary tie cannot fail this test; only the attention math can)."""
    module, t, key = build_dsa(topk = 64, seed = 100 + S)
    bsz = 2
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (100 + S))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"]
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def test_dsa_dense_equivalence():
    """T <= index_topk: the sparse machinery must stand down and reproduce dense MLA."""
    module, t, key = build_dsa(topk = 64, seed = 7)
    bsz, S = 2, 64
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (7))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    assert "dsa_topk_indices" not in params, "selection ran below the sparse threshold"

    module.index_topk = 1 << 30
    dense, _ = nc_forward(module, x, positions)
    module.index_topk = 64
    assert rel_err(out, dense) == 0.0, "dense-regime forward diverged from plain dense MLA"


def test_dsa_sharing():
    """A "shared" module must consume the published selection, and must refuse to run
    without one."""
    S = 200
    full, t, key = build_dsa(topk = 64, seed = 11)
    shared, t2, _ = build_dsa(topk = 64, mode = "shared", seed = 11)

    bsz = 1
    x = rnd_x((bsz, S, full.hidden_size), seed = 9000 + (11))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out_full, params = nc_forward(full, x, positions)
    indices = params["dsa_topk_indices"]

    out_shared, _ = nc_forward(shared, x, positions, {"dsa_topk_indices": indices})
    ref = ref_forward_indices(shared, t2, key, x, positions, indices)
    assert rel_err(out_shared, ref) < 5e-3, f"rel err {rel_err(out_shared, ref):.3e}"

    with pytest.raises(AssertionError, match = "shared-indexer"):
        nc_forward(shared, x, positions)


def test_dsa_cached_vs_nc():
    """Cached path with the paged indexer plane, fed in chunks. Chunk 2 scores over the paged
    plane (past + current). The output is checked against the masked dense reference driven by
    the cached path's own per-chunk selections (the paged and contiguous scoring kernels can
    order fp16 ties at the k-th score differently, so raw output comparison against the nc
    path is only held to selection-overlap standards)."""
    S, chunk = 384, 128
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = 23)
    bsz = 2
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (23))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    _, nc_params = nc_forward(module, x, positions)
    nc_indices = nc_params["dsa_topk_indices"].view(bsz, S, -1)

    layer = CacheLayer_MLA_fp16(None, module, 0, 4 * PAGE_SIZE * bsz)
    layer.alloc(torch.device(device))
    assert layer.k_idx is not None, "full-indexer layer allocated no indexer plane"
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    outs = []
    chunk_indices = []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        chunk_indices.append(params["dsa_topk_indices"].view(bsz, b - a, -1))
        seqlens = seqlens + (b - a)
    out = torch.cat(outs, dim = 1)

    # Attention math over the paged pool, given the selection actually made
    k_pad = max(ci.shape[-1] for ci in chunk_indices)
    indices = torch.cat(
        [F.pad(ci, (0, k_pad - ci.shape[-1]), value = -1) for ci in chunk_indices], dim = 1)
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"

    # Selection agreement with the cache-less path, allowing boundary ties to differ
    for b in range(bsz):
        for q_row in range(topk, S, 37):
            a_set = set(i for i in indices[b, q_row].tolist() if i >= 0)
            b_set = set(i for i in nc_indices[b, q_row].tolist() if i >= 0)
            overlap = len(a_set & b_set)
            assert overlap >= topk - max(2, topk // 16), \
                f"row {q_row}: cached/nc selection overlap {overlap}/{topk}"


def test_dsa_cached_decode():
    """Single-token decode steps over a sparse context: cached selection + gather at
    seqlen 1, against the cache-less forward's last row."""
    S = 200
    module, t, key = build_dsa(topk = 64, seed = 31)
    bsz = 1
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (31))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = CacheLayer_MLA_fp16(None, module, 0, 4 * PAGE_SIZE)
    layer.alloc(torch.device(device))
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    prefill = S - 8
    params = {
        "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
        "cache_seqlens": seqlens, "positions": seqlens.clone(),
    }
    module.forward(x[:, :prefill].contiguous(), params)
    seqlens += prefill
    outs = []
    step_indices = []
    for i in range(prefill, S):
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, i:i + 1].contiguous(), params))
        step_indices.append(params["dsa_topk_indices"].view(bsz, 1, -1))
        seqlens += 1
    out = torch.cat(outs, dim = 1)

    # Full-sequence reference driven by the decode steps' own selections. Attention rows are
    # independent, so rows outside the compared region just select everything (causality is
    # intersected inside the reference)
    k_pad = step_indices[0].shape[-1]
    indices = torch.arange(S, dtype = torch.int32, device = device) \
        .view(1, 1, S).expand(bsz, S, S).contiguous()
    indices[:, prefill:, :] = F.pad(
        torch.cat(step_indices, dim = 1), (0, S - k_pad), value = -1)
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref[:, prefill:]) < 5e-3, \
        f"rel err {rel_err(out, ref[:, prefill:]):.3e}"


@pytest.mark.parametrize("bits", [8, 4])
def test_dsa_cached_quant_prefill(bits):
    """Sparse cached prefill over the packed-quantized latent (CacheLayer_MLA_quant): the
    gather kernel dequantizes online in the H32-rotated domain. Reference: the masked dense
    attention driven by the module's own selection, with the latent round-tripped through the
    same quantizer (the selection is identical to the fp16-cache case: indexer planes stay
    fp16). Residual error is the reference's fp16 ckv rounding flipping a few quantization
    levels, hence the width-dependent tolerance."""
    S, chunk = 384, 128
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = 41)
    bsz = 2
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (41))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = CacheLayer_MLA_quant(None, module, 0, 4 * PAGE_SIZE * bsz, k_bits = bits)
    layer.alloc(torch.device(device))
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    outs, chunk_indices = [], []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        chunk_indices.append(params["dsa_topk_indices"].view(bsz, b - a, -1))
        seqlens = seqlens + (b - a)
    out = torch.cat(outs, dim = 1)
    k_pad = max(ci.shape[-1] for ci in chunk_indices)
    indices = torch.cat(
        [F.pad(ci, (0, k_pad - ci.shape[-1]), value = -1) for ci in chunk_indices], dim = 1)
    ref = ref_forward_indices(module, t, key, x, positions, indices, ckv_quant_bits = bits)
    tol = {8: 6e-3, 4: 2.5e-2}[bits]
    assert rel_err(out, ref) < tol, f"rel err {rel_err(out, ref):.3e} (tol {tol})"


@pytest.mark.parametrize("bits", [8, 4])
def test_dsa_cached_quant_decode(bits):
    """Single-token decode steps over a sparse context held in the packed-quantized cache."""
    S = 200
    module, t, key = build_dsa(topk = 64, seed = 43)
    bsz = 1
    x = rnd_x((bsz, S, module.hidden_size), seed = 9000 + (43))
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = CacheLayer_MLA_quant(None, module, 0, 4 * PAGE_SIZE, k_bits = bits)
    layer.alloc(torch.device(device))
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    prefill = S - 8
    params = {
        "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
        "cache_seqlens": seqlens, "positions": seqlens.clone(),
    }
    module.forward(x[:, :prefill].contiguous(), params)
    seqlens += prefill
    outs, step_indices = [], []
    for i in range(prefill, S):
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, i:i + 1].contiguous(), params))
        step_indices.append(params["dsa_topk_indices"].view(bsz, 1, -1))
        seqlens += 1
    out = torch.cat(outs, dim = 1)
    k_pad = step_indices[0].shape[-1]
    indices = torch.arange(S, dtype = torch.int32, device = device) \
        .view(1, 1, S).expand(bsz, S, S).contiguous()
    indices[:, prefill:, :] = F.pad(
        torch.cat(step_indices, dim = 1), (0, S - k_pad), value = -1)
    ref = ref_forward_indices(module, t, key, x, positions, indices, ckv_quant_bits = bits)
    tol = {8: 6e-3, 4: 2.5e-2}[bits]
    e = rel_err(out, ref[:, prefill:])
    assert e < tol, f"rel err {e:.3e} (tol {tol})"
