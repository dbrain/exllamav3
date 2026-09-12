import os
import sys

import torch


def _unavailable(reason):
    # Under pytest this must be a clean skip; as a plain script these files are run
    # directly (python tests/test_reconstruct_had.py), where Skipped would be noise.
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason, allow_module_level = True)
    raise RuntimeError(reason)


def resolve_device(default = "cuda:0"):
    """EXL_TEST_DEVICE, or `default`, skipping the module if that device is absent.

    Several tests were written on a box with three GPUs and hardcode cuda:1 / cuda:2,
    which is an "invalid device ordinal" AcceleratorError anywhere else. The kernels
    under test are device-agnostic, so the ordinal is incidental.
    """
    name = os.environ.get("EXL_TEST_DEVICE", default)
    if not torch.cuda.is_available():
        _unavailable("no CUDA (or ROCm) device")
    index = torch.device(name).index or 0
    if index >= torch.cuda.device_count():
        _unavailable(f"{name} not present (device_count = {torch.cuda.device_count()})")
    return name


def skip_module(reason):
    _unavailable(reason)


def assert_close_mr(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        mismatch_ratio: float = 0.0,
        check_device: bool = True,
        check_dtype: bool = True,
        msg: str = None,
):

    # 1) Check shape
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: {actual.shape} vs {expected.shape}"
        )

    # 2) (Optional) Check device
    if check_device and (actual.device != expected.device):
        raise AssertionError(
            f"Device mismatch: {actual.device} vs {expected.device}"
        )

    # 3) (Optional) Check dtype
    if check_dtype and (actual.dtype != expected.dtype):
        raise AssertionError(
            f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
        )

    # 4) Compare element-wise closeness
    #    close_mask[i] = True if actual[i] ~ expected[i] within rtol/atol
    close_mask = torch.isclose(actual, expected, rtol = rtol, atol = atol)

    # 5) Compute fraction of elements that are out of tolerance
    total_elements = close_mask.numel()
    mismatch_count = total_elements - close_mask.sum().item()
    fraction_mismatched = mismatch_count / total_elements

    if fraction_mismatched > mismatch_ratio:
        default_msg = (
            f"Too many values are out of tolerance:\n"
            f"  Mismatch ratio = {fraction_mismatched:.6f} "
            f"(allowed <= {mismatch_ratio:.6f})\n"
            f"  Mismatched elements = {mismatch_count} / {total_elements}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        error_msg = f"{msg}\n{default_msg}" if msg else default_msg
        raise AssertionError(error_msg)