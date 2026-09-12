"""CPU MoE expert offload on ROCm: cpu/moe_handoff.cu and cpu/moe_mul1.cpp are excluded from the
build, so only the tuning setters and ISA probes get fallbacks. Everything that would dispatch
work raises at the call site rather than silently doing nothing.

The load-bearing part is model/moe_cpu_host.py calling ext.exl3_moe_cpu_set_memops at module
level (line 109): without a fallback, importing it -- which modules/block_sparse_mlp_cpu.py does
lazily when cpu_unload is configured -- raises AttributeError.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

rocm_only = pytest.mark.skipif(torch.version.hip is None, reason="ROCm fallbacks only")


@rocm_only
def test_moe_cpu_host_imports():
    from exllamav3.model.moe_cpu_host import MoeCpuHost, TUNING
    assert TUNING is not None and MoeCpuHost is not None


@rocm_only
def test_isa_probes_report_no_tier():
    assert ext.exl3_moe_cpu_has_avx2() is False
    assert ext.exl3_moe_cpu_has_avx512_vnni() is False
    assert ext.exl3_moe_cpu_has_avx512_vbmi() is False


@rocm_only
def test_tuning_setters_are_noops():
    assert ext.exl3_moe_cpu_set_memops(True) is None
    assert ext.exl3_moe_cpu_set_prof(False) is None


@rocm_only
@pytest.mark.parametrize("name", [
    "exl3_moe_cpu_make_layer", "exl3_moe_cpu_free_layer", "exl3_moe_cpu_forward",
    "exl3_moe_cpu_worker_run", "exl3_moe_cpu_pool_stress",
])
def test_dispatch_entry_points_raise(name):
    with pytest.raises(NotImplementedError, match="not built on ROCm"):
        getattr(ext, name)()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
