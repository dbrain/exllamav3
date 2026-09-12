"""End-to-end cover for the packed-bitmask constraint path:
LLGuidanceFilter -> Job.prepare_logit_mask -> CustomSampler.forward -> apply_logit_bitmask.

test_logit_bitmask_fallback.py pins the op's semantics; this pins the wiring inside a real
generation loop -- that a bitmask set at the grammar layer changes which token is selected,
and that the resulting text conforms to the schema.

Needs a GPU and a model (EXL3_TEST_MODEL_DIR, default _models/qwen3-0.6b-4bpw).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("EXL3_TEST_MODEL_DIR", os.path.join(ROOT, "_models", "qwen3-0.6b-4bpw"))

pytest.importorskip("llguidance")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "no GPU"),
    pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason = f"no model at {MODEL_DIR}"),
]

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["name", "year"],
    "additionalProperties": False,
}

PROMPT = "Vital information about the city of Paris, in JSON format:\n\n"
MAX_NEW = 100


@pytest.fixture(scope = "module")
def gen():
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)
    yield generator, tokenizer
    model.unload()


def _bit_set(bitmask, token):
    return bool((int(bitmask[0, token >> 5].item()) >> (token & 31)) & 1)


def _recording_sampler():
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    class Recording(ArgmaxSampler):
        def __init__(self):
            super().__init__()
            self.trace = []

        def forward(self, logits, sequence_ids = None, rand_u32 = None, tokenizer = None,
                    logit_mask = None, return_state = False):
            row = logits.view(-1, logits.shape[-1])[0].float()
            if tokenizer is not None:
                row = row[:tokenizer.actual_vocab_size]
            free = int(row.argmax().item())
            mask = None if logit_mask is None else logit_mask.cpu().clone()
            out = super().forward(logits, sequence_ids, rand_u32, tokenizer, logit_mask, return_state)
            self.trace.append((free, int(out.flatten()[0].item()), mask))
            return out

    return Recording()


def _generate(generator, tokenizer, sampler, filters = None):
    return generator.generate(
        prompt = PROMPT,
        max_new_tokens = MAX_NEW,
        add_bos = True,
        completion_only = True,
        sampler = sampler,
        filters = filters,
    )


def test_grammar_bitmask_suppresses_the_token_the_model_wanted(gen):
    from exllamav3 import LLGuidanceFilter

    generator, tokenizer = gen
    sampler = _recording_sampler()
    text = _generate(generator, tokenizer, sampler,
                     [LLGuidanceFilter(tokenizer, eos_after_completed = True, json_schema = SCHEMA)])

    masks = [m for _, _, m in sampler.trace if m is not None]
    assert masks, "no logit mask reached the sampler; the constraint path was not exercised"
    assert all(m.dtype == torch.int32 for m in masks), \
        f"mask dtypes {set(str(m.dtype) for m in masks)}; this test only covers the packed path"

    suppressed = [
        (i, free, taken, m) for i, (free, taken, m) in enumerate(sampler.trace)
        if m is not None and not _bit_set(m, free)
    ]
    assert suppressed, (
        "the grammar never forbade the unconstrained argmax, so no suppression is observable "
        "in this run; the test cannot distinguish a working mask from an ignored one"
    )
    print(f"\nconstrained output: {text!r}")
    for i, free, taken, m in suppressed[:8]:
        print(f"  step {i}: unconstrained argmax {free} {tokenizer.decode(torch.tensor([[free]]))!r} mask bit CLEAR "
              f"-> emitted {taken} {tokenizer.decode(torch.tensor([[taken]]))!r} mask bit "
              f"{'SET' if _bit_set(m, taken) else 'CLEAR'}")
    for i, free, taken, m in suppressed:
        assert taken != free, f"step {i}: forbidden token {free} was emitted anyway"
        assert _bit_set(m, taken), f"step {i}: emitted token {taken} is forbidden by the mask"


def test_schema_constrained_generation_conforms(gen):
    from exllamav3 import LLGuidanceFilter

    generator, tokenizer = gen
    text = _generate(generator, tokenizer, None,
                     [LLGuidanceFilter(tokenizer, eos_after_completed = True, json_schema = SCHEMA)])
    print(f"\nconstrained output: {text!r}")
    obj = json.loads(text)
    assert set(obj) == {"name", "year"}
    assert isinstance(obj["name"], str)
    assert isinstance(obj["year"], int) and not isinstance(obj["year"], bool)


def test_unconstrained_generation_does_not_conform(gen):
    # Control: without the filter the same prompt must NOT already satisfy the schema,
    # otherwise the test above would pass with the mask ignored entirely.
    generator, tokenizer = gen
    text = _generate(generator, tokenizer, None, None)
    print(f"\nunconstrained output: {text!r}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    assert not (isinstance(obj, dict) and set(obj) == {"name", "year"})


def test_regex_constrained_generation_conforms(gen):
    from exllamav3 import LLGuidanceFilter

    generator, tokenizer = gen
    text = _generate(generator, tokenizer, None,
                     [LLGuidanceFilter(tokenizer, eos_after_completed = True,
                                       regex = r"Paris, France, population [0-9]{1,9}\.")])
    print(f"\nregex-constrained output: {text!r}")
    import re
    assert re.fullmatch(r"Paris, France, population [0-9]{1,9}\.", text), text
