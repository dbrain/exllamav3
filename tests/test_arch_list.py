from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.nogpu

_ARCH_LIST = Path(__file__).resolve().parents[1] / "exllamav3" / "util" / "arch_list.py"
_spec = importlib.util.spec_from_file_location("exllamav3_arch_list_contract", _ARCH_LIST)
assert _spec is not None and _spec.loader is not None
arch_list = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch_list)


def _fake_rocm_torch(props: list):
    return SimpleNamespace(
        version = SimpleNamespace(hip = "7.2", cuda = None),
        cuda = SimpleNamespace(
            device_count = lambda: len(props),
            get_device_properties = lambda i: props[i],
        ),
    )


@pytest.mark.parametrize(
    "gcn_arch_names,preset,expected",
    [
        (["gfx1150"], None, "gfx1150"),
        (["gfx1201:sramecc+:xnack-", "gfx1100", "gfx1201"], None, "gfx1100;gfx1201"),
        (["gfx1151"], "gfx1150", "gfx1150"),
    ],
)
def test_rocm_arch_env_from_visible_devices(monkeypatch, gcn_arch_names, preset, expected):
    props = [SimpleNamespace(name = "dev", gcnArchName = n, warp_size = 32) for n in gcn_arch_names]
    monkeypatch.setattr(arch_list, "torch", _fake_rocm_torch(props))
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising = False)
    if preset is None:
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising = False)
    else:
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", preset)

    arch_list.maybe_set_arch_list_env()

    assert arch_list.os.environ["PYTORCH_ROCM_ARCH"] == expected


def test_rocm_arch_env_rejects_wave64(monkeypatch):
    props = [SimpleNamespace(name = "mi300", gcnArchName = "gfx942", warp_size = 64)]
    monkeypatch.setattr(arch_list, "torch", _fake_rocm_torch(props))
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising = False)
    with pytest.raises(RuntimeError, match = "wave32"):
        arch_list.maybe_set_arch_list_env()
