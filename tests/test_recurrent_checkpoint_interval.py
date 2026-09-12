import pytest

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import resolve_recurrent_checkpoint_interval as resolve

CAPS = {"default_recurrent_checkpoint_interval": 2048}


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("EXL3_RECURRENT_CKPT", raising = False)


def test_default_comes_from_model_caps():
    assert resolve(None, CAPS) == 2048
    assert resolve(None, {}) == 2048


def test_explicit_argument_outranks_the_env(monkeypatch):
    monkeypatch.setenv("EXL3_RECURRENT_CKPT", str(PAGE_SIZE))
    assert resolve(512, CAPS) == 512


def test_env_override_takes_effect(monkeypatch):
    monkeypatch.setenv("EXL3_RECURRENT_CKPT", str(PAGE_SIZE))
    assert resolve(None, CAPS) == PAGE_SIZE


@pytest.mark.parametrize("bad", ["", " ", "nonsense", "0", "-256", "300", "255", "1.5", "256x"])
def test_malformed_or_unaligned_values_fall_back_silently(monkeypatch, bad):
    """generator.py asserts interval % PAGE_SIZE == 0, so an unaligned env value would otherwise
    abort the server at load. A typo in a deployment env file must not be fatal."""
    monkeypatch.setenv("EXL3_RECURRENT_CKPT", bad)
    assert resolve(None, CAPS) == 2048
