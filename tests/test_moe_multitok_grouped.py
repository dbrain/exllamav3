"""The grouped Triton mgemm, T tokens at a time.

Why this path has to exist on ROCm. `exl3_moe`, the fused multi-token MoE
kernel, is bound in bindings.cpp and NOT in bindings_hip.cpp, so `_HAS_MOE` is
False here and `support_fused` can never be true. Every MoE forward with more
than one token therefore falls through to the dense branch, whose fallback is a
Python loop over ALL num_local_experts (512 on flashnext) per layer, plus one
`expert_count.tolist()` host sync per layer. Measured on flashnext-4.05bpw, the
48 MoE layers cost 45 ms for one token and 306 ms for two.

The grouped kernel already accepts one x row per routed entry (its contract is
`x.shape[0] in (E, 1)`), so T tokens are T * top_k entries with the matching x
row gathered per entry: the same three launches per layer the decode path uses.

What this test pins: for every token, the grouped multi-token call must produce
what T separate single-token grouped calls produce -- the decode path, run once
per token. Bitwise if the autotune tile holds (E_BUCKET is part of the key and
T * top_k lands in a different bucket than top_k, so it need not), otherwise
within a tolerance far tighter than the fp16 output's own resolution.
"""
import os

import pytest
import torch

from exllamav3.modules.block_sparse_mlp import (
    BlockSparseMLP, MGemmBuffers, MTGemmBuffers,
)
from exllamav3.modules.quant import exl3_triton as T

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not T.has_triton, reason="Triton not available"),
]

NUM_EXPERTS = 6
TOP_K = 3
HI, I, HO = 256, 128, 256
K_BITS, CB = 4, 2


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


class Proj:
    """One projection's per-expert weights plus the pointer tables the kernel reads."""

    def __init__(self, k_dim, n, dev, seed):
        g = torch.Generator(device = "cpu").manual_seed(seed)
        self.K = K_BITS
        self.in_features = k_dim
        self.out_features = n
        self._tr = [torch.randint(0, 65536, (k_dim // 16, n // 16, 16 * K_BITS),
                                  dtype = torch.int32, generator = g).to(torch.short)
                    .contiguous().to(dev) for _ in range(NUM_EXPERTS)]
        self._suh = [torch.sign(torch.randn(k_dim, generator = g)).half().contiguous().to(dev)
                     for _ in range(NUM_EXPERTS)]
        self._svh = [torch.sign(torch.randn(n, generator = g)).half().contiguous().to(dev)
                     for _ in range(NUM_EXPERTS)]
        self.ptrs_trellis = self._table(self._tr)
        self.ptrs_suh = self._table(self._suh)
        self.ptrs_svh = self._table(self._svh)

    @staticmethod
    def _table(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype = torch.long,
                            device = ts[0].device)


def _act(g, u, a, limit):
    a.copy_((torch.nn.functional.silu(g.float()) * u.float()).half())


def _stub(dev, rows):
    """A BlockSparseMLP with exactly the attributes _grouped_multitok touches."""
    half = torch.half
    mt = MTGemmBuffers(
        xrows = torch.empty((rows, HI), dtype = half, device = dev),
        xh = torch.empty((rows, 1, HI), dtype = half, device = dev),
        interm_g = torch.empty((rows, I), dtype = half, device = dev),
        interm_u = torch.empty((rows, I), dtype = half, device = dev),
        act = torch.empty((rows, I), dtype = half, device = dev),
        xh_d = torch.empty((rows, 1, I), dtype = half, device = dev),
        out = torch.empty((rows, HO), dtype = half, device = dev),
        ids = torch.empty((rows,), dtype = torch.long, device = dev),
    )
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.num_experts_per_tok = TOP_K
    m.mgemm_cb = CB
    m.act_limit = 0.0
    m.activation_fn_call = _act
    m.multi_gate = Proj(HI, I, dev, 1)
    m.multi_up = Proj(HI, I, dev, 2)
    m.multi_down = Proj(I, HO, dev, 3)
    m.mgemm_buf = MGemmBuffers(
        xh_gu = torch.empty((2 * TOP_K, 1, HI), dtype = half, device = dev),
        interm_gu = torch.empty((2 * TOP_K, I), dtype = half, device = dev),
        interm_g = torch.empty((TOP_K, I), dtype = half, device = dev),
        interm_u = torch.empty((TOP_K, I), dtype = half, device = dev),
        act = torch.empty((TOP_K, I), dtype = half, device = dev),
        xh_d = torch.empty((TOP_K, 1, I), dtype = half, device = dev),
        out = torch.empty((TOP_K, HO), dtype = half, device = dev),
        ids = torch.empty((TOP_K,), dtype = torch.long, device = dev),
        mt = mt,
    )
    return m


def _per_token_reference(m, y, sel, w):
    """T single-token grouped calls: the decode path, once per token."""
    from exllamav3.modules.quant.exl3_mgemm_triton import _linear_exl3_mgemm_triton
    b = m.mgemm_buf
    out = []
    for t in range(y.shape[0]):
        ids = sel[t].contiguous()
        row = y[t:t + 1].contiguous()
        _linear_exl3_mgemm_triton(row, b.xh_gu[:TOP_K], b.interm_g,
                                  m.multi_gate.ptrs_trellis, m.multi_gate.ptrs_suh,
                                  m.multi_gate.ptrs_svh, ids, m.multi_gate.K, CB)
        _linear_exl3_mgemm_triton(row, b.xh_gu[:TOP_K], b.interm_u,
                                  m.multi_up.ptrs_trellis, m.multi_up.ptrs_suh,
                                  m.multi_up.ptrs_svh, ids, m.multi_up.K, CB)
        _act(b.interm_g, b.interm_u, b.act, 0.0)
        _linear_exl3_mgemm_triton(b.act, b.xh_d, b.out,
                                  m.multi_down.ptrs_trellis, m.multi_down.ptrs_suh,
                                  m.multi_down.ptrs_svh, ids, m.multi_down.K, CB)
        out.append(torch.sum(b.out.float() * w[t].view(-1, 1).float(), dim = 0))
    return torch.stack(out)


@pytest.mark.parametrize("tokens", [2, 3, 5, 8])
def test_multitok_matches_one_grouped_call_per_token(tokens):
    dev = device()
    m = _stub(dev, TOP_K * 8)
    g = torch.Generator(device = "cpu").manual_seed(0x5EED + tokens)
    y = (torch.randn((tokens, HI), generator = g) / 8).half().to(dev)
    sel = torch.stack([torch.randperm(NUM_EXPERTS, generator = g)[:TOP_K]
                       for _ in range(tokens)]).long().to(dev)
    w = torch.rand((tokens, TOP_K), generator = g).half().to(dev)

    ref = _per_token_reference(m, y, sel, w)
    got = BlockSparseMLP._grouped_multitok(
        m, torch.empty((tokens, HO), device = dev), y, tokens, sel, w)
    torch.cuda.synchronize()

    assert got.shape == ref.shape == (tokens, HO)
    d = (got - ref).abs()
    scale = ref.abs().max().clamp(min = 1e-6)
    rel = (d.max() / scale).item()
    # The routed rows are fp16, so one ULP is 2**-10 = 9.8e-4 of the scale. The
    # two arms are the same arithmetic in a different grid, and E_BUCKET is part
    # of the autotune key (top_k here, tokens * top_k there), so the tile -- and
    # with it the fp32 reduction order inside a CTA -- need not match. Bitwise
    # when the tiles coincide, a few ULP when they do not; anything larger would
    # mean rows are being paired with the wrong expert or the wrong token.
    ulp = rel / 2 ** -10
    assert ulp < 5, (f"tokens={tokens}: max|diff| {d.max().item():.4e} = {ulp:.1f} "
                     f"fp16 ULP of scale {scale.item():.4e} -- too large to be "
                     f"reduction order")
    print(f"\ntokens={tokens}: {'bitwise' if torch.equal(got, ref) else 'not bitwise'}; "
          f"ndiff {int((got != ref).sum())}/{got.numel()} max|diff| "
          f"{d.max().item():.4e} = {ulp:.2f} fp16 ULP")


@pytest.mark.parametrize("tokens", [2, 5, 8])
def test_expert_sorted_rows_match_token_major_bitwise(tokens, monkeypatch):
    """EXL3_MOE_SORT_ROWS=1 lays the routed entries out by expert id, so entries that
    share an expert land in adjacent programs of the entry-major grid and the second
    read of that expert's tiles can hit L2. Each entry is still its own GEMV under the
    same E_BUCKET tile, so the result must be BITWISE the token-major one; any other
    difference means a row was paired with the wrong token on the way back."""
    dev = device()
    m = _stub(dev, TOP_K * 8)
    g = torch.Generator(device = "cpu").manual_seed(0x50F7 + tokens)
    y = (torch.randn((tokens, HI), generator = g) / 8).half().to(dev)
    # 6 experts, top 3: consecutive tokens are forced to share experts
    sel = torch.stack([torch.randperm(NUM_EXPERTS, generator = g)[:TOP_K]
                       for _ in range(tokens)]).long().to(dev)
    w = torch.rand((tokens, TOP_K), generator = g).half().to(dev)

    monkeypatch.delenv("EXL3_MOE_SORT_ROWS", raising = False)
    ref = BlockSparseMLP._grouped_multitok(
        m, torch.empty((tokens, HO), device = dev), y, tokens, sel, w).clone()
    monkeypatch.setenv("EXL3_MOE_SORT_ROWS", "1")
    got = BlockSparseMLP._grouped_multitok(
        m, torch.empty((tokens, HO), device = dev), y, tokens, sel, w)
    torch.cuda.synchronize()

    assert torch.equal(m.mgemm_buf.mt.ids[:tokens * TOP_K],
                       sel.view(-1).sort(stable = True).values), \
        "EXL3_MOE_SORT_ROWS=1 must hand the kernel expert-sorted entries"
    assert torch.equal(got, ref)


@pytest.mark.parametrize("maxr", [2, 4, 8])
@pytest.mark.parametrize("tokens", [2, 5, 8])
def test_expert_dedup_matches_token_major(tokens, maxr, monkeypatch):
    """EXL3_MOE_DEDUP=<maxr> decodes each routed expert's weight tile ONCE and applies
    it to up to maxr rows routed to that expert, instead of once per routed entry.
    6 experts / top 3 forces runs of 1..8 equal ids, so every maxr exercises both a
    single chunk and a run split across several chunk heads. Each row sees the same
    weights and the same k-order accumulation as the per-entry kernel; only the final
    in-register reduction may associate differently, so allow a few ULP."""
    from exllamav3.modules.quant import exl3_mgemm_dedup_triton as D
    calls = []
    real = D.exl3_mgemm_dedup
    monkeypatch.setattr(D, "exl3_mgemm_dedup",
                        lambda *a, **kw: calls.append(kw.get("max_rows")) or real(*a, **kw))

    dev = device()
    m = _stub(dev, TOP_K * 8)
    g = torch.Generator(device = "cpu").manual_seed(0xDED0 + 16 * tokens + maxr)
    y = (torch.randn((tokens, HI), generator = g) / 8).half().to(dev)
    sel = torch.stack([torch.randperm(NUM_EXPERTS, generator = g)[:TOP_K]
                       for _ in range(tokens)]).long().to(dev)
    w = torch.rand((tokens, TOP_K), generator = g).half().to(dev)

    monkeypatch.delenv("EXL3_MOE_DEDUP", raising = False)
    ref = BlockSparseMLP._grouped_multitok(
        m, torch.empty((tokens, HO), device = dev), y, tokens, sel, w).clone()
    assert not calls
    monkeypatch.setenv("EXL3_MOE_DEDUP", str(maxr))
    got = BlockSparseMLP._grouped_multitok(
        m, torch.empty((tokens, HO), device = dev), y, tokens, sel, w)
    torch.cuda.synchronize()

    assert calls == [maxr] * 3, f"dedup kernel must serve gate, up and down, got {calls}"
    d = (got - ref).abs()
    scale = ref.abs().max().clamp(min = 1e-6)
    ulp = (d.max() / scale).item() / 2 ** -10
    assert ulp < 5, f"tokens={tokens} maxr={maxr}: {ulp:.1f} fp16 ULP -- a row got the wrong expert or token"
    print(f"\ntokens={tokens} maxr={maxr}: {'bitwise' if torch.equal(got, ref) else f'{ulp:.2f} ULP'}")


def test_rows_are_token_major_so_the_reduce_needs_no_counts():
    """The reduction is a plain per-token sum only because row t*top_k+k belongs
    to token t. If the row order ever changes, this is the assertion that says so."""
    dev = device()
    m = _stub(dev, TOP_K * 8)
    sel = torch.arange(2 * TOP_K, device = dev).remainder(NUM_EXPERTS).view(2, TOP_K)
    y = torch.zeros((2, HI), dtype = torch.half, device = dev)
    w = torch.zeros((2, TOP_K), dtype = torch.half, device = dev)
    BlockSparseMLP._grouped_multitok(
        m, torch.empty((2, HO), device = dev), y, 2, sel, w)
    assert torch.equal(m.mgemm_buf.mt.ids[:2 * TOP_K], sel.view(-1)), \
        "expert ids must be laid out token-major"
