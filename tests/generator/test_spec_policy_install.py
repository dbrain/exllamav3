"""
install_draft_policy on a model-free stand-in for an MTP Generator: the shipped tier rule and n-gram lookup are what
a server gets from the one public call, a dynamic-draft or drafter-less generator is refused, and an n-gram copy never
carries multimodal embedding ids (they are not text tokens: the target never emits them and a verify row cannot embed them).
"""

import pytest
import torch

from exllamav3.generator import spec_policy as sp
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from exllamav3.util.tensor import SeqTensor

pytestmark = pytest.mark.nogpu

VOCAB = 1000


class Job:
    def __init__(self, ids):
        sid = SeqTensor((1, 1), torch.long, -1, init_cap = 256)
        sid.append(torch.tensor([ids], dtype = torch.long))
        self.sequences = [type("Seq", (), {"sequence_ids": sid})()]

    def is_prefill_done(self):
        return True


class Drafter:
    caps = {"mtp_draft": True}

    def __init__(self, ps):
        self.ps, self.k, self.forwards = ps, 0, 0

    def forward(self, ids, params):
        self.forwards += 1
        return torch.zeros(1)

    def sample_from_state(self, state, params):
        self.k += 1
        return torch.tensor([[100 + self.k - 1]])


class Head:
    def __init__(self, drafter):
        self.drafter = drafter

    def forward(self, *a, **k):
        p = self.drafter.ps[self.drafter.k]
        lg = torch.full((1, 1, VOCAB), float(torch.log(torch.tensor((1 - p) / (VOCAB - 1) + 1e-30))))
        lg[..., 0] = float(torch.log(torch.tensor(p)))
        return lg


class Target:
    class config:
        vocab_size = VOCAB

    def __init__(self, head):
        self.modules = [head]
        self.logit_layer_idx = 0


class Gen:
    mtp_draft = True

    def __init__(self, ids, ps, ndt = 7, dynamic = False):
        self.num_draft_tokens = ndt
        self.dynamic_draft = dynamic
        self.draft_ids_pinned = torch.zeros((1, ndt), dtype = torch.long)
        self.draft_model = Drafter(list(ps) + [0.5] * ndt)
        self.model = Target(Head(self.draft_model))
        self.active_jobs = [Job(ids)]

    def enqueue(self, job):
        return None

    def iterate_draftmodel_mtp_gen(self, results, window):
        for idx in range(window):
            state = self.draft_model.forward(None, {})
            self.model.modules[0].forward(state)
            self.draft_ids_pinned[:1, idx:idx + 1].copy_(self.draft_model.sample_from_state(state, {}))
        return self.draft_ids_pinned[:, :window]


def draft(ids, ps, **kw):
    g = Gen(ids, ps, **kw)
    uninstall = sp.install_draft_policy(g)
    return g.iterate_draftmodel_mtp_gen([], g.num_draft_tokens), g, uninstall


ROW = list(range(1, 21))


@pytest.mark.parametrize("ps, keep, forwards", [
    ([0.999] * 7, 7, 7),
    ([0.99] * 7, 6, 6),
    ([0.95, 0.95, 0.95, 0.9], 3, 3),
    ([0.99, 0.99, 0.99, 0.5, 0.9], 3, 4),
    ([0.5, 0.4], 1, 1),
])
def test_tier_rule_by_default(ps, keep, forwards):
    out, g, _ = draft([1, 2, 3], ps)
    assert out.tolist() == [[100 + i for i in range(keep)]]
    assert g.draft_model.forwards == forwards
    assert g.num_draft_tokens == 7


def test_ngram_lookup_by_default():
    out, g, _ = draft(ROW + ROW[:16], [0.999] * 7)
    assert out.tolist() == [ROW[16:20] + ROW[:3]]
    assert g.draft_model.forwards == 1


def test_ngram_copy_stops_at_embedding_ids():
    out, _, _ = draft(ROW + [FIRST_MM_EMBEDDING_INDEX + 5, 7, 8] + ROW[:16], [0.999] * 7)
    assert out.tolist() == [ROW[16:20]]


def test_ngram_copy_of_only_embedding_ids_is_a_drafter_round():
    mm = list(range(FIRST_MM_EMBEDDING_INDEX, FIRST_MM_EMBEDDING_INDEX + 8))
    out, g, _ = draft(ROW[:16] + mm + ROW[:16], [0.999] * 7)
    assert out.tolist() == [[100 + i for i in range(7)]]
    assert g.draft_model.forwards == 7


def test_uninstall_restores_the_generator():
    g = Gen([1, 2, 3], [0.99] * 7)
    uninstall = sp.install_draft_policy(g)
    uninstall()
    assert "iterate_draftmodel_mtp_gen" not in g.__dict__ and "enqueue" not in g.__dict__
    assert g.iterate_draftmodel_mtp_gen([], g.num_draft_tokens).shape[-1] == 7


@pytest.mark.parametrize("attr, value", [("dynamic_draft", True), ("mtp_draft", False)])
def test_refuses_generators_it_cannot_drive(attr, value):
    g = Gen([1, 2, 3], [0.99] * 7)
    setattr(g, attr, value)
    with pytest.raises(ValueError):
        sp.install_draft_policy(g)


@pytest.mark.parametrize("uninstall_old_after", [False, True])
def test_reinstall_on_a_new_generator_over_the_same_models(uninstall_old_after):
    old = Gen([1, 2, 3], [0.5, 0.4])
    uninstall_old = sp.install_draft_policy(old)
    assert old.iterate_draftmodel_mtp_gen([], old.num_draft_tokens).shape[-1] == 1
    new = Gen([1, 2, 3], [])
    new.draft_model, new.model = old.draft_model, old.model
    old.draft_model.ps, old.draft_model.k = [0.95, 0.95, 0.95, 0.9] + [0.5] * 4, 0
    sp.install_draft_policy(new)
    if uninstall_old_after:
        uninstall_old()
    assert new.iterate_draftmodel_mtp_gen([], new.num_draft_tokens).tolist() == [[100, 101, 102]]
    assert old.draft_model.forwards == 1 + 3
