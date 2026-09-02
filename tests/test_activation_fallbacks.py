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
