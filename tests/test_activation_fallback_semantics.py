"""The PyTorch fallbacks must match the kernels they stand in for.

ext_fallbacks is what runs wherever activation.cu is unavailable, so a fallback
that disagrees with the kernel is a silent numerical divergence, not a slow path.
The kernel is the spec; these compare against it directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallbacks as fb

pytestmark = pytest.mark.skipif(
    not hasattr(ext, "silu_mul") or type(ext.silu_mul).__name__ != "builtin_function_or_method",
    reason = "native activation kernels not in this build",
)

SHAPES = [(1, 2048), (7, 512), (128, 1024)]


def _pair(shape, scale, dtype = torch.half):
    torch.manual_seed(1234)
    x = torch.randn(*shape, dtype = dtype, device = "cuda") * scale
    y = torch.randn(*shape, dtype = dtype, device = "cuda") * scale
    return x, y


# scale 20 puts most values outside a limit of 7, so the clamp actually binds.
# At scale 3 the limit is inert and every candidate implementation agrees --
# such a test passes without testing anything.
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("act_limit,scale", [(0.0, 3.0), (7.0, 3.0), (7.0, 20.0)])
@pytest.mark.parametrize("name", ["silu_mul", "gelu_mul", "relu2_mul", "relu_mul", "silu_oai_mul"])
def test_gated_activation_matches_kernel(name, shape, act_limit, scale):
    x, y = _pair(shape, scale)
    zk = torch.empty_like(x)
    zf = torch.empty_like(x)
    getattr(ext, name)(x, y, zk, act_limit)
    getattr(fb, name)(x, y, zf, act_limit)
    assert torch.allclose(zk.float(), zf.float(), rtol = 2e-2, atol = 3e-2), \
        f"{name} limit={act_limit} scale={scale} max_abs={(zk.float()-zf.float()).abs().max().item():.4g}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ap,an", [(0.8, -0.3), (0.0, 0.0), (25.0, 1.0)])
def test_xielu_matches_kernel(shape, ap, an):
    torch.manual_seed(99)
    # xielu takes float input and writes half; the half-input branch raises
    x = torch.randn(*shape, dtype = torch.float, device = "cuda") * 3.0
    alpha_p = torch.tensor([ap], dtype = torch.float)   # CPU tensors, per the kernel
    alpha_n = torch.tensor([an], dtype = torch.float)
    yk = torch.empty(*shape, dtype = torch.half, device = "cuda")
    yf = torch.empty(*shape, dtype = torch.half, device = "cuda")
    ext.xielu(x, yk, alpha_p, alpha_n)
    fb.xielu(x, yf, alpha_p, alpha_n)
    assert torch.allclose(yk.float(), yf.float(), rtol = 2e-2, atol = 3e-2), \
        f"xielu ap={ap} an={an} max_abs={(yk.float()-yf.float()).abs().max().item():.4g}"
