"""Validate the pure-PyTorch activation fallbacks for the shared-expert gate.

These shipped with the wrong arity (2 and 3 params against the C++ bindings' 3 and
4) and the wrong arithmetic, which raises TypeError at the block_sparse_mlp call
site for any MoE with a gated shared expert. Reference is add_sigmoid_kernel_f /
add_sigmoid_proj_kernel_f in exllamav3_ext/activation_kernels.cuh.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: F401
import torch
import torch.nn.functional as F

from exllamav3 import ext_fallbacks as fb

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BSZ, H, NE, K = 7, 128, 32, 4


def _mk(bsz=BSZ, ne=NE, k=K, seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(bsz, H, device=DEV) * 0.5).half()
    gate = (torch.randn(H, ne, device=DEV) * 0.5).half()
    scores = torch.empty((bsz, ne), dtype=torch.half, device=DEV)
    idx = torch.empty((bsz, k), dtype=torch.long, device=DEV)
    w = torch.empty((bsz, k), dtype=torch.half, device=DEV)
    return hidden, gate, scores, idx, w


def _logits(hidden, gate):
    return torch.matmul(hidden.view(-1, hidden.shape[-1]).float(), gate.float())


# --- shared-expert gate fallbacks (activation.cu) ----------------------------
# These shipped in PR #283 with both the wrong arity and the wrong arithmetic
# (`o.add_(g).mul_(sigmoid(o))`), which breaks any MoE with a gated shared expert.
# Reference is add_sigmoid_kernel_f / add_sigmoid_proj_kernel_f.

def test_add_sigmoid_gate_accumulates():
    torch.manual_seed(10)
    x = torch.randn(BSZ, H, device=DEV)
    y = torch.randn(BSZ, 1, device=DEV)
    z = torch.randn(BSZ, H, device=DEV)
    ref = z + x * torch.sigmoid(y)          # broadcast over the row
    fb.add_sigmoid_gate(x, y, z)
    torch.testing.assert_close(z, ref)


def test_add_sigmoid_gate_proj_accumulates():
    torch.manual_seed(11)
    x = torch.randn(BSZ, H, device=DEV)
    y = (torch.randn(BSZ, H, device=DEV) * 0.3).half()
    z = torch.randn(BSZ, H, device=DEV)
    w = (torch.randn(H, 1, device=DEV) * 0.3).half()
    ref = z + x * torch.sigmoid(torch.matmul(y.float(), w.float()))
    fb.add_sigmoid_gate_proj(x, y, z, w)
    torch.testing.assert_close(z, ref, atol=1e-4, rtol=1e-4)


def test_gate_fallbacks_match_ext_arity():
    """Arity must match the C++ binding or the call site raises TypeError."""
    import inspect
    assert len(inspect.signature(fb.add_sigmoid_gate).parameters) == 3
    assert len(inspect.signature(fb.add_sigmoid_gate_proj).parameters) == 4


# --- rms_norm (norm.cu) ------------------------------------------------------
# add_residual selects RES_POST (y += norm(x)*w), NOT overwrite; span_heads flattens
# the trailing head dim; w_groups cycles the weight by row. #283's version ignored
# all three, so any residual call silently discarded the residual.

def _ref_norm(x, w, eps, cb, cs):
    xf = x.float()
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * cs
    return o * ((w + cb).float() if cb else w.float()) if w is not None else o


def test_rms_norm_plain():
    torch.manual_seed(20)
    x = torch.randn(BSZ, H, device=DEV); w = torch.randn(H, device=DEV)
    y = torch.zeros(BSZ, H, device=DEV)
    fb.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False)
    torch.testing.assert_close(y, _ref_norm(x, w, 1e-6, 0.0, 1.0), atol=1e-5, rtol=1e-5)


def test_rms_norm_add_residual_accumulates():
    torch.manual_seed(21)
    x = torch.randn(BSZ, H, device=DEV); w = torch.randn(H, device=DEV)
    resid = torch.randn(BSZ, H, device=DEV)
    y = resid.clone()
    fb.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, True)
    torch.testing.assert_close(y, resid + _ref_norm(x, w, 1e-6, 0.0, 1.0),
                               atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y, _ref_norm(x, w, 1e-6, 0.0, 1.0)), \
        "add_residual must accumulate, not overwrite"


def test_rms_norm_w_groups_cycles_weight_by_row():
    torch.manual_seed(22)
    g, rows = 4, 12
    x = torch.randn(rows, H, device=DEV); w = torch.randn(g * H, device=DEV)
    y = torch.zeros(rows, H, device=DEV)
    fb.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False, g)
    wg = w.view(g, H)
    for r in range(rows):
        torch.testing.assert_close(y[r], _ref_norm(x[r:r+1], wg[r % g], 1e-6, 0.0, 1.0)[0],
                                   atol=1e-5, rtol=1e-5)


def test_rms_norm_span_heads_flattens():
    torch.manual_seed(23)
    heads, hd = 4, 32
    x = torch.randn(BSZ, heads, hd, device=DEV)
    y = torch.zeros(BSZ, heads, hd, device=DEV)
    fb.rms_norm(x, None, y, 1e-6, 0.0, 1.0, True, False)
    ref = _ref_norm(x.flatten(-2), None, 1e-6, 0.0, 1.0).view_as(y)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_softcap_arity_and_inplace():
    import inspect
    assert len(inspect.signature(fb.softcap).parameters) == 3
    torch.manual_seed(24)
    x = torch.randn(BSZ, H, device=DEV)
    ref = torch.tanh(x.float() / 30.0) * 30.0
    fb.softcap(x, x, 30.0)                      # linear.py calls it in place
    torch.testing.assert_close(x, ref.to(x.dtype), atol=1e-5, rtol=1e-5)
