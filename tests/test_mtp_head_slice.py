"""The MTP draft head may serve a VOCAB SLICE of the target's lm_head.

WHY. A draft step reads the whole 248320-column head -- 455 MiB of the ~567 MiB it touches --
only to argmax it. ids < 98304 cover 99.88-100% of real corpora, and the TARGET still scores
the full vocabulary, so a token the slice cannot propose is a REJECTED draft, never a wrong
token: quality is preserved by construction, only acceptance can move.

WHAT THIS PINS. A head built from the first `cut` columns must return exactly what the full
head returns for those columns. That holds because EXL3's output transform is block-diagonal
over aligned 128-column groups, so with `cut % 128 == 0` no column outside the slice can
contribute. Two ways to get this wrong are caught here: slicing at a non-multiple of 128, and
leaving the sliced trellis non-contiguous -- the kernels derive their strides from the SHAPE,
so a view would be read as interleaved garbage rather than failing loudly.
"""

import pytest
import torch

from exllamav3.architecture.qwen4_exp_mtp import build_sliced_head
from exllamav3.modules.quant import exl3_triton as T
from exllamav3.modules.quant.exl3 import LinearEXL3

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not T.has_triton, reason="Triton not available"),
]

IN_F, OUT_F, K = 512, 512, 4
CUT = 256  # a multiple of 128, like the real 98304 of 248320


def _head(dev):
    g = torch.Generator(device="cpu").manual_seed(0x11EAD)
    trellis = (torch.randint(0, 65536, (IN_F // 16, OUT_F // 16, 256 * K // 16),
                             dtype=torch.int32, generator=g)
               .to(torch.short).contiguous().to(dev))
    suh = torch.sign(torch.randn(IN_F, generator=g)).half().to(dev)
    svh = torch.sign(torch.randn(OUT_F, generator=g)).half().to(dev)
    # Positional order as in Linear.load_exl3: config, in, out, scale, su, sv, suh, svh,
    # trellis, mcg, mul1, bias, out_dtype. Codebook tensors are absent here (plain cb); the
    # slicing property is codebook-independent, since cb only decides how a code decodes.
    return LinearEXL3(None, IN_F, OUT_F, None, None, None, suh, svh, trellis,
                      None, None, None, torch.half)


def test_sliced_head_matches_the_full_head_columns():
    dev = torch.device("cuda:0")
    full = _head(dev)
    sliced = build_sliced_head(full, CUT)

    g = torch.Generator(device="cpu").manual_seed(0x51CE)
    x = ((torch.randn((4, IN_F), generator=g) / 8).half().to(dev).contiguous())
    y_full = full.forward(x, {})
    y_sliced = sliced.forward(x, {})
    torch.cuda.synchronize()

    assert sliced.out_features == CUT
    assert sliced.trellis.is_contiguous(), "a sliced trellis VIEW would be read with the wrong stride"
    assert tuple(sliced.trellis.shape) == (IN_F // 16, CUT // 16, 256 * K // 16)
    assert tuple(sliced.svh.shape) == (CUT,)
    assert y_sliced.shape == (4, CUT)

    ref = y_full[:, :CUT]
    d = (y_sliced.float() - ref.float()).abs().max().item()
    scale = max(ref.float().abs().max().item(), 1e-6)
    ulp = (d / scale) / 2 ** -10
    assert ulp < 5, (f"sliced head differs from the full head's first {CUT} columns by "
                     f"{ulp:.1f} fp16 ULP -- the slice is not column-exact")
    print(f"\nsliced vs full: {'bitwise' if torch.equal(y_sliced, ref) else f'{ulp:.2f} ULP'}")


def test_sliced_head_refuses_a_cut_that_breaks_the_128_groups():
    """The output Hadamard is block-diagonal over aligned 128-column groups, so a cut that
    splits a group would silently return wrong logits rather than a smaller head."""
    dev = torch.device("cuda:0")
    full = _head(dev)
    with pytest.raises(AssertionError):
        build_sliced_head(full, 200)
