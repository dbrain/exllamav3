"""Decode-graph replay against inputs that arrive through params (graph_decode.py).

The MTP draft head is the case this exists for: every draft step within one
verification round carries a FRESH target_hidden (generator.py,
iterate_draftmodel_mtp_gen), and the MTP input layer reads it out of params
rather than receiving it as x (modules/arch_specific/qwen4_exp_mtp.py). Binding a
step's params once at capture time therefore froze the draft's only real input
and acceptance collapsed to noise, with capture, replay and verification all
reporting success.

No MTP model is small: the only MTP head on hand is inside the 63 GB
flashnext-4.05bpw directory (its config's text_config.mtp), and neither
qwen3-0.6b-4bpw nor qwen35-35b-a3b-4bpw has one. So the mechanism is driven
directly instead -- the first test needs no GPU at all, the second captures real
graphs over a synthetic three-module model and needs no weights.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.model import graph_decode as gd

HID = 8


class HiddenConsumer:
    """Stands in for an MTP input layer: output depends on a per-step tensor
    that arrives through params, not through x. Eager (x_cpu), as the real ones
    are -- they call the target's CPU-resident embedding."""

    def __init__(self, device):
        self.key = "mtp"
        self.caps = {"x_cpu": True}
        self.device = device

    def __iter__(self):
        yield self

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        th = params["target_hidden"]
        emb = x.to(th.device).float().view(*x.shape, 1).expand(*x.shape, HID)
        return emb + th


class MatMul:
    """A capturable device-resident block."""

    def __init__(self, device, seed):
        self.key = f"blk{seed}"
        self.caps = {}
        self.device = device
        g = torch.Generator(device = "cpu").manual_seed(seed)
        self.w = torch.randn((HID, HID), generator = g).to(device)

    def __iter__(self):
        yield self

    def prepare_for_device(self, x, params):
        return x if x.device == self.device else x.to(self.device)

    def forward(self, x, params):
        return torch.tanh(x @ self.w)


class FakeModel:
    loaded_tp = False

    def __init__(self, mods):
        self.modules = mods
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(mods)]

    def run_eager(self, input_ids, params):
        x = input_ids
        for module, instance, _ in self.fwd_modules:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x


def _step_params(cs, bt, target_hidden):
    """A fresh params dict per step, as the generator's draft loop builds one."""
    return {
        "attn_mode": "flash_attn",
        "block_table": bt,
        "cache_seqlens": cs,
        "target_hidden": target_hidden,
    }


# ------------------------------------------------- replay sees the current step

def test_replay_runs_eager_spans_against_the_current_steps_params():
    """A replayed step's eager spans must read THIS step's params.

    Pure bookkeeping, so it runs on CPU: no capture, one eager span, four steps
    with four different target_hidden values. Binding params once at capture
    time returns the capture-time value four times over.
    """
    dev = torch.device("cpu")
    model = FakeModel([HiddenConsumer(dev)])
    g = gd.DecodeGraphs(model)
    g.device = dev

    cs = torch.zeros(1, dtype = torch.int32)
    bt = torch.zeros((1, 16), dtype = torch.int32)
    th0 = torch.zeros((1, 1, HID))
    span = gd._Span(captured = False, start = 0, end = 1, exempt = False)
    cap = gd._CapturedStep(
        [span], cs.clone(), cs.clone(), bt.clone(),
        {"target_hidden": th0.clone()}, None, None,
    )
    cap.params = g._bind(_step_params(cs, bt, th0), cap)

    ids = torch.zeros((1, 1), dtype = torch.long)
    got = []
    for step in range(1, 5):
        th = torch.full((1, 1, HID), float(step))
        got.append(g._replay(cap, ids, _step_params(cs, bt, th)).flatten()[0].item())
    assert got == [1.0, 2.0, 3.0, 4.0]


def test_step_inputs_are_in_the_signature_by_shape_only():
    """Contents change every step and must not recapture; shape must."""
    g = gd.DecodeGraphs(FakeModel([HiddenConsumer(torch.device("cpu"))]))
    cs = torch.zeros(1, dtype = torch.int32)
    bt = torch.zeros((1, 16), dtype = torch.int32)
    ids = torch.zeros((1, 1), dtype = torch.long)
    sig = g._signature(ids, _step_params(cs, bt, torch.zeros((1, 1, HID))))
    assert g._signature(ids, _step_params(cs, bt, torch.ones((1, 1, HID)))) == sig
    assert g._signature(ids, _step_params(cs, bt, torch.zeros((1, 1, HID * 2)))) != sig
    assert g._signature(ids, {"block_table": bt, "cache_seqlens": cs}) != sig


def test_perturb_changes_every_element_without_leaving_range():
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    p = gd._perturb(t)
    assert not torch.equal(p, t)
    assert torch.equal(p.sort().values, t.sort().values)
    # degenerate inputs roll to themselves rather than fabricating values
    assert torch.equal(gd._perturb(torch.zeros(1)), torch.zeros(1))


# ------------------------------------------------------- real capture + replay

@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU for graph capture")
def test_captured_steps_match_eager_across_varying_target_hidden():
    """Several consecutive steps, each with a different target_hidden, must come
    out bitwise equal to the eager path -- the shape of an MTP draft round."""
    dev = torch.device("cuda:0")
    mods = [HiddenConsumer(dev), MatMul(dev, 1), MatMul(dev, 2)]
    model = FakeModel(mods)
    g = gd.DecodeGraphs(model)
    assert g.eligible(), g._reason

    cs = torch.zeros(1, dtype = torch.int32)
    bt = torch.zeros((1, 16), dtype = torch.int32)
    ids = torch.zeros((1, 1), dtype = torch.long)

    hiddens = [torch.randn((1, 1, HID), generator = torch.Generator().manual_seed(s)).to(dev)
               for s in range(8)]
    want = [model.run_eager(ids, _step_params(cs, bt, th)).clone() for th in hiddens]

    got = []
    for th in hiddens:
        y = g.forward(ids, _step_params(cs, bt, th))
        if y is None:                                   # warmup steps run eager
            y = model.run_eager(ids, _step_params(cs, bt, th))
        got.append(y.clone())

    assert g.n_replays > 0, "nothing was replayed, the test proved nothing"
    for i, (a, b) in enumerate(zip(got, want)):
        assert torch.equal(a, b), f"step {i} diverged: max {(a - b).abs().max().item():.3e}"


def test_varied_verification_perturbs_bound_inputs_in_place_and_restores_them():
    """The second verification pass has to vary BOTH channels a step's inputs
    arrive on, and rewrite the bound params buffers in place -- a correct capture
    reads through exactly those addresses, so a fresh tensor would prove nothing
    -- then leave them holding the real step's values."""
    g = gd.DecodeGraphs(FakeModel([HiddenConsumer(torch.device("cpu"))]))
    th = torch.arange(float(HID)).view(1, 1, HID)
    cap = gd._CapturedStep([], None, None, None, {"target_hidden": th.clone()}, None, None)

    seen = []
    g._verify_span = lambda span, x_in, p, restore, ref = None: seen.append(
        (cap.step_inputs["target_hidden"].clone(), x_in.clone())
    )

    x = torch.arange(float(HID))
    assert g._verify_span_varied(cap, None, x, {}, lambda: None) is None
    varied_hidden, varied_x = seen[0]
    assert not torch.equal(varied_hidden, th), "bound params buffer was not varied"
    assert not torch.equal(varied_x, x), "span input was not varied"
    assert torch.equal(cap.step_inputs["target_hidden"], th), "buffer not restored"


# ----------------------------------------- steps whose caller reads exported state

class ExportingNorm:
    """Stands in for the target's final RMSNorm (modules/rmsnorm.py) and for the
    exporting TransformerBlock (modules/transformer.py): when the step asks for it,
    the module appends its output to params["export_states"] as an OUT-parameter."""

    def __init__(self, device):
        self.key = "norm"
        self.caps = {}
        self.device = device

    def __iter__(self):
        yield self

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        if any(params.get(k) for k in gd._OUT_PARAM_REQUESTS):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            states.append(x)
        return x


def _verify_step_params(cs, bt, target_hidden, key):
    p = _step_params(cs, bt, target_hidden)
    p[key] = {"norm"} if key == "export_state_norm_keys" else \
             {0} if key == "export_state_layers" else True
    return p


@pytest.mark.parametrize("key", gd._OUT_PARAM_REQUESTS)
def test_a_step_that_exports_states_is_never_served_by_the_graph_path(key):
    """Both channels a graph-served step reaches a module on lose an out-parameter:
    a captured span runs no Python at all, and every span -- captured or eager --
    is handed _bind's COPY of params, so a module's append lands in a dict the
    caller never sees. The MTP verify path indexes that list unconditionally
    (generator.py, iterate_gen: `target_hidden = p_export_states[-1]`), so such a
    step has to stay eager.

    Five steps, because the failure is delayed: _WARMUP_STEPS steps fall back on
    their own and only the one after them reaches capture.
    """
    g = gd.DecodeGraphs(FakeModel([ExportingNorm(torch.device("cpu"))]))
    g.eligible = lambda: True
    attempts = []
    g._capture = lambda sig, ids, params: attempts.append(sig)

    cs = torch.zeros(1, dtype = torch.int32)
    bt = torch.zeros((1, 16), dtype = torch.int32)
    ids = torch.zeros((1, 1), dtype = torch.long)
    for step in range(gd._WARMUP_STEPS + 3):
        p = _verify_step_params(cs, bt, torch.full((1, 1, HID), float(step)), key)
        assert g.forward(ids, p) is None, f"step {step} was served by the graph path"
        assert not attempts, f"step {step} tried to capture a state-exporting step"
        FakeModel([ExportingNorm(torch.device("cpu"))]).run_eager(ids, p)
        assert p.get("export_states"), "eager fallback did not export"


def test_bound_params_cannot_carry_an_out_parameter_back_to_the_caller():
    """Why the rule above is a refusal and not a fix inside the graph path."""
    g = gd.DecodeGraphs(FakeModel([ExportingNorm(torch.device("cpu"))]))
    cs = torch.zeros(1, dtype = torch.int32)
    bt = torch.zeros((1, 16), dtype = torch.int32)
    th = torch.zeros((1, 1, HID))
    cap = gd._CapturedStep([], cs.clone(), cs.clone(), bt.clone(),
                           {"target_hidden": th.clone()}, None, None)
    outer = _verify_step_params(cs, bt, th, "export_state_norm_keys")
    inner = g._bind(outer, cap)
    inner["export_states"] = ["written by a module"]
    assert "export_states" not in outer
