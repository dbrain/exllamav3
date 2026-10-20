"""ROCm coverage for portable compiled rewind/cache bindings."""
from __future__ import annotations

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm rewind binding tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext


def test_portable_rewind_and_cache_bindings_exist():
    for name in (
        "ConvRewindJob", "ScanReplayJob", "batched_conv_rewind",
        "batched_scan_replay", "dspark_write_rows",
    ):
        assert hasattr(ext, name), f"missing ROCm binding: {name}"
