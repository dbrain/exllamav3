from __future__ import annotations
import os
from typing_extensions import override
import torch
import torch.nn.functional as F

from ..util.tensor import to2
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation


def conv_gemm_enabled() -> bool:
    """
    EXL3_CONV_GEMM=1: run the reshape2d 2D convolutions (the spatial-merge downsample of the GLM-4V
    family vision tower) as one hgemm GEMM instead of F.conv2d.

    F.conv2d goes to MIOpen, which has to pick a solver for every new convolution shape the first
    time it sees it. On a platform that ships no MIOpen find-db (gfx1151: the ROCm SDK's
    share/miopen/db has no .fdb.txt at all), that search benchmarks the candidate solvers on the
    real tensors, and its cost is linear in the batch -- which here is the number of vision tokens,
    so it grows with the image:

        400x400  (225 vision tokens)   ~67 s
        1024x1024 (1369 vision tokens) ~409 s
        2888x4177 (7992 vision tokens) ~40 min (extrapolated)

    The shape is cached per (arch, MIOpen version, shape) in the user find-db, so a repeat of the
    same image is fast, but every new image size pays again. The GEMM below is the same
    contraction without MIOpen: it removes the search entirely and costs milliseconds.

    Default off ("0" or unset): the original F.conv2d path, byte for byte.
    """
    return os.environ.get("EXL3_CONV_GEMM", "0") != "0"


class Conv(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
        flat: bool = False,
        reshape2d: bool = False,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Conv"
        self.module_name = "Conv"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = None
        self.bias = None
        self.out_dtype = out_dtype
        self._numel = None
        self.flat = flat
        self.reshape2d = reshape2d
        # EXL3_CONV_GEMM: (dim, out_channels) weight in GEMM layout, built on first use so the
        # default F.conv2d path pays neither the memory nor the transpose
        self._gemm_b = None

        self.dims = len(kernel_size)
        assert self.dims in [2, 3], "Convolution must be 2D or 3D"

        self.flat = flat
        self.dim = in_channels
        for d in self.kernel_size: self.dim *= d


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, float2half = True)
        bias = self.config.stc.get_tensor(f"{self.key}.bias", self.device, optional = True, float2half = True)
        self._numel = weight.numel() + (bias.numel() if bias is not None else 0)
        self.weight = weight
        self.bias = bias
        self._gemm_b = None   # rebuilt from these weights on the next EXL3_CONV_GEMM forward


    @override
    def unload(self):
        self.device = None
        self.weight = None
        self.bias = None
        self._gemm_b = None


    @override
    def get_tensors(self):
        t = {}
        t[f"{self.key}.weight"] = self.weight.contiguous()
        if self.bias is not None:
            t[f"{self.key}.bias"] = self.bias.contiguous()
        return t


    @override
    def weights_numel(self):
        return self._numel


    def forward_2d_gemm(self, x: torch.Tensor, out_dtype: torch.dtype | None) -> torch.Tensor:
        """
        The reshape2d 2D convolution as one GEMM (see conv_gemm_enabled).

        The input is a patch sequence (batch, tokens * kh * kw, in_channels) laid out merge-major:
        the kh * kw patches of one merged token are consecutive rows, and each row is
        (in_channels, ...) inside its patch. With stride == kernel the output is 1x1, so the
        convolution is a plain sum over (c, kh, kw) of x * weight -- i.e. x_flat @ w_flat.T with
        both operands flattened in (kh, kw, c) order. That is the same contraction the flat 3D
        patch embedding already runs through ext.hgemm, and `weight` is stored (out, c, kh, kw),
        so the (kh, kw, c) layout is one permute done once per load, never per forward.
        """
        assert x.numel() % self.dim == 0, \
            f"EXL3_CONV_GEMM: {x.shape} does not hold whole {self.kernel_size} patches of {self.in_channels} channels"
        # The activations reaching this layer are fp32 (the 3D patch embedding runs out_dtype=float
        # and the blocks keep it); the F.conv2d branch casts with x.to(self.weight.dtype) and so does
        # this one, so both compute on the same fp16 operand.
        x_flat = x.reshape(-1, self.dim).to(torch.half)
        if self._gemm_b is None:
            w = self.weight.permute(0, *range(2, 2 + self.dims), 1).reshape(self.out_channels, self.dim)
            self._gemm_b = w.T.contiguous()
        dt = out_dtype or self.out_dtype or torch.half
        y = torch.empty((x_flat.shape[0], self.out_channels), dtype = dt, device = x_flat.device)
        ext.hgemm(x_flat, self._gemm_b, y)
        if self.bias is not None:
            y += self.bias
        # Same shape the F.conv2d branch returns: 1x1 output spatial, so (tokens, 1, out_channels)
        return y.view(-1, 1, self.out_channels)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if self.dims == 2:
            if self.reshape2d and conv_gemm_enabled():
                y = self.forward_2d_gemm(x, out_dtype)
            else:
                if self.reshape2d:
                    x = x.view(-1, self.kernel_size[0], self.kernel_size[1], x.shape[-1]).permute(0, 3, 1, 2)
                y = F.conv2d(x.to(self.weight.dtype), self.weight, self.bias, self.kernel_size)
                y = y.flatten(2).permute(0, 2, 1).contiguous()

        elif self.dims == 3:
            bsz, seqlen, dim = x.shape

            if self.flat:
                x_flat = x.view(-1, self.dim)
                w_flat = self.weight.view(self.weight.shape[0], -1).T.contiguous()
                out_dtype = out_dtype or self.out_dtype
                assert x_flat.dtype == torch.half
                assert w_flat.dtype == torch.half
                y = torch.empty((x_flat.shape[0], w_flat.shape[1]), dtype = out_dtype, device = x_flat.device)
                ext.hgemm(x_flat, w_flat, y)
                y = y.view(bsz, seqlen, self.out_channels)
                if self.bias is not None:
                    y += self.bias
            else:
                # Extremely inefficient for Qwen3
                x = x.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
                y = F.conv3d(x, self.weight, self.bias, self.kernel_size)
                y = y.view(bsz, seqlen, self.out_channels)

        return to2(y, out_dtype, self.out_dtype)


    @override
    def optimizer_targets(self):
        raise NotImplementedError()


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError("TP not implemented for Conv layer.")