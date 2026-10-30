"""Unit tests for the IterativeReweighting module group."""

from __future__ import annotations

import math

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.model.base import ModuleGroup
from salt.model.bind import ResolvedSchema
from salt.model.modules import IterativeReweighting, KinematicWeight
from salt.model.modules.tasks import ClassificationTaskModule


def _ce(reduction):
    return {"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"reduction": reduction}}


PRIOR = {
    "stream": "jets", "input": "prior_input.jets", "label": "flavour_label",
    "class_names": ["b", "c", "u"], "dense": {"hidden_layers": [8]},
    "sample_weight": "flavour_class_weight", "loss": _ce("none")}
WEIGHT = {
    "mode": "nn", "q_f": None, "n_classes": 3, "balanced_prior": True,
    "variables": {"pt": {"edges": [0.0, 1.0, 2.0], "scale": 1e-3}}}
FIXED = WEIGHT | {"q_f": [0.5, 0.3, 0.2], "n_classes": None, "w_max": math.inf}
TARGETS = ("inclusive", "mean", "harmonic", "harmonic_equal")
SCHEMA = ResolvedSchema(widths={"prior_input.jets": 4}, fields={})
ITER = ("nn_kinematic_weight_iter_1", "nn_kinematic_weight_iter_2")
LAST = "labels.jets.nn_kinematic_weight"
N_ROWS = 64


def group(prior=PRIOR, weight=WEIGHT, **kw) -> dict:
    """Expand an IterativeReweighting (default N=1) and name its members by key."""
    cfg = {"n_iterations": 1} | kw
    members = IterativeReweighting(prior=prior, weight=weight, **cfg).expand()
    for key, m in members.items():
        m.name = key
    return members


def bind(members: dict, seed=0) -> dict:
    """Bind every member, reseeding before each so equal configs get equal heads."""
    for m in members.values():
        torch.manual_seed(seed)
        m.bind(SCHEMA)
    return members


def weights(members: dict) -> dict:
    return {k: m for k, m in members.items() if isinstance(m, KinematicWeight)}


def bundle(seed=1, extra=None) -> Bundle:
    """A random jets batch with prior inputs, labels, class weights and pt."""
    g = torch.Generator().manual_seed(seed)
    b = Bundle()
    leaves = {
        "prior_input.jets": torch.randn(N_ROWS, 4, generator=g),
        "labels.jets.flavour_label": torch.randint(0, 3, (N_ROWS,), generator=g),
        "labels.jets.flavour_class_weight": torch.rand(N_ROWS, generator=g) + 0.5,
        "labels.jets.pt": torch.rand(N_ROWS, generator=g) * 2e3,
    }
    for key, value in (leaves | (extra or {})).items():
        b.set(key, value)
    return b


def run(members: dict, b: Bundle, mode: Mode) -> dict:
    """Run members in order, writing every produced leaf into `b` (the executor's role)."""
    out = {}
    for m in members.values():
        for key, value in m(b, mode).items():
            b.set(key, value)
            out[key] = value
    return out


@pytest.mark.parametrize("target", TARGETS)
def test_n1_is_todays_pair(target):
    weight = WEIGHT | {"target": target}
    got = group(weight=weight)
    hand = {
        "kinematic_prior": ClassificationTaskModule(**PRIOR),
        "kinematic_weight": KinematicWeight(**weight, task_name="kinematic_prior"),
    }
    for key, m in hand.items():
        m.name = key
    assert list(got) == list(hand) == ["kinematic_prior", "kinematic_weight"]
    bind(got)
    bind(hand)
    for key, m in hand.items():
        assert list(got[key].state_dict()) == list(m.state_dict())
    kw = got["kinematic_weight"]
    assert kw.eta == 1.0 and kw.chain == ()
    for phase, mode in enumerate((Mode.FIT, None, Mode.FIT, Mode.VAL)):
        if mode is None:  # stage boundary: FIT running -> frozen
            for members in (got, hand):
                members["kinematic_weight"].freeze_accumulators(1, 0)
            continue
        a, b = run(got, bundle(seed=phase), mode), run(hand, bundle(seed=phase), mode)
        assert a.keys() == b.keys()
        for key, value in a.items():
            assert torch.equal(value, b[key]), (phase, key)


@pytest.mark.parametrize(
    ("avg_last", "chain", "exps", "eta"),
    [(1, ITER[1:], (1.0,), 1.0), (2, ITER[1:], (1.0,), 0.5), (3, ITER, (1 / 3, 2 / 3), 1 / 3)],
)
def test_n3_names_fields_expose_and_chains(avg_last, chain, exps, eta):
    members = group(n_iterations=3, avg_last=avg_last)
    assert list(members) == [
        "kinematic_prior_1", "kinematic_weight_1", "kinematic_prior_2", "kinematic_weight_2",
        "kinematic_prior", "kinematic_weight"]
    p1, w1, p2, w2, p3, w3 = members.values()
    assert p1.sample_weights == ("flavour_class_weight",)
    assert p2.sample_weights == ("flavour_class_weight", ITER[0])
    assert p3.sample_weights == ("flavour_class_weight", ITER[1])
    assert not (p1.expose_modes & Mode.TEST) and not (p2.expose_modes & Mode.TEST)
    assert p3.expose_modes == Mode.ALL
    assert [w.task_name for w in (w1, w2, w3)] == [
        "kinematic_prior_1", "kinematic_prior_2", "kinematic_prior"]
    assert [w.weight_field for w in (w1, w2, w3)] == [*ITER, "nn_kinematic_weight"]
    assert (w1.chain, w1.eta) == ((), 1.0)
    assert (w2.chain, w2.chain_exponents, w2.eta) == (ITER[:1], (1.0,), 1.0)
    assert w3.chain == chain
    assert w3.chain_exponents == pytest.approx(exps)
    assert w3.eta == pytest.approx(eta)
    assert w1.variables is None and w2.variables is None
    assert set(w3.variables) == {"pt"}


@pytest.mark.parametrize(
    ("avg_last", "e"), [(1, (1, 1, 1)), (2, (1, 1, 0.5)), (3, (1, 2 / 3, 1 / 3))]
)
def test_fixed_weight_algebra(avg_last, e):
    g = torch.Generator().manual_seed(7)
    heads = ("kinematic_prior_1", "kinematic_prior_2", "kinematic_prior")
    preds = {f"preds.jets.{h}": torch.randn(N_ROWS, 3, generator=g) for h in heads}
    members = group(weight=FIXED, n_iterations=3, avg_last=avg_last)
    last = run(bind(weights(members)), bundle(extra=preds), Mode.VAL)[LAST].double()
    # standalone eta=1, unchained residuals w_j = t / p_j from the same logits
    w = torch.stack([
        KinematicWeight(**FIXED, task_name=h)(bundle(extra=preds), Mode.VAL)[LAST].double()
        for h in heads])
    expected = (w ** torch.tensor(e, dtype=torch.float64)[:, None]).prod(0)
    torch.testing.assert_close(last, expected, rtol=1e-6, atol=0)
    geo = torch.cumprod(w, 0)[-avg_last:].log().mean(0).exp()  # mean log of last M cumulatives
    torch.testing.assert_close(last, geo, rtol=1e-6, atol=0)


def test_cumulative_cap():
    y, rows = bundle().get("labels.jets.flavour_label"), torch.arange(N_ROWS)
    l1, l2 = torch.zeros(N_ROWS, 3), torch.zeros(N_ROWS, 3)
    l1[rows, y] = -30.0  # P(f|x) floored at p_floor -> w_1 = q_f / p_floor >> w_max
    l2[rows, y] = -5.0  # head-2 residual well above 1
    preds = {"preds.jets.kinematic_prior_1": l1, "preds.jets.kinematic_prior": l2}
    members = group(weight=FIXED | {"w_max": 50.0}, n_iterations=2)
    out = run(weights(members), bundle(extra=preds), Mode.VAL)
    w1, last = out[f"labels.jets.{ITER[0]}"], out[LAST]
    assert torch.equal(w1, torch.full_like(w1, 50.0))
    assert torch.isfinite(last).all() and (last <= 50.0).all()
    assert torch.equal(last, torch.full_like(last, 50.0))


def test_no_gradient_leaks_and_head2_weight_is_cumulative():
    members = bind(group(n_iterations=2))
    b = bundle()
    out = run(members, b, Mode.FIT)
    prior, loss_2 = members["kinematic_prior"], out["losses.kinematic_prior"]
    loss_2.backward()
    assert all(p.grad is None for p in members["kinematic_prior_1"].parameters())
    assert all(p.grad is not None for p in prior.parameters())
    w_cls, w_1 = b.get("labels.jets.flavour_class_weight"), b.get(f"labels.jets.{ITER[0]}")
    assert not w_1.requires_grad
    labels = b.get("labels.jets.flavour_label")
    hand = prior.calculate_loss(out["preds.jets.kinematic_prior"], labels, w_cls * w_1)
    torch.testing.assert_close(loss_2, hand, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_iterations": 0}, {"n_iterations": True}, {"n_iterations": 1.5},
        {"n_iterations": 2, "avg_last": 0}, {"n_iterations": 2, "avg_last": 3},
        {"eta": 0}, {"eta": 1.5},
        {"n_iterations": 2, "prior": PRIOR | {"loss": _ce("mean")}},
        {"prior": ["not", "a", "mapping"]},
        {"weight": WEIGHT | {"mode": "binned"}},
        {"weight": WEIGHT | {"task_name": "other"}},
        {"weight": WEIGHT | {"eta": 0.5}}, {"weight": WEIGHT | {"chain": ["x"]}},
        {"weight": WEIGHT | {"chain_exponents": [1.0]}},
        {"n_iterations": 2, "weight": WEIGHT | {"target": "harmonic"}},
        {"n_iterations": 2, "weight": WEIGHT | {"target": "harmonic_equal"}},
    ],
)
def test_config_rejected(kwargs):
    with pytest.raises(ConfigError, match="IterativeReweighting"):
        IterativeReweighting(**{"n_iterations": 1, "prior": PRIOR, "weight": WEIGHT} | kwargs)


def test_n1_permits_mean_reduction_and_harmonic_and_is_a_module_group():
    assert issubclass(IterativeReweighting, ModuleGroup)
    IterativeReweighting(n_iterations=1, prior=PRIOR | {"loss": _ce("mean")}, weight=WEIGHT)
    harmonic = group(weight=WEIGHT | {"target": "harmonic"})
    assert harmonic["kinematic_weight"].target == "harmonic"
    assert list(group(n_iterations=2, avg_last=2)) == [
        "kinematic_prior_1", "kinematic_weight_1", "kinematic_prior", "kinematic_weight"]
