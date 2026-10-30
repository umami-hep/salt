"""Unit tests for KinematicWeight and its variable spec."""

from __future__ import annotations

import functools
import logging
import math
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError, ConnectivityError
from salt.graph.planner import compile_plan
from salt.graph.spec import Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.model.bind import ResolvedSchema
from salt.model.modules import KinematicWeight
from salt.model.modules.kinematic_weight import cell_index, parse_variables
from salt.model.modules.tasks import ClassificationTaskModule
from salt.model.multistage_training import TrainingController
from salt.schedule import TrainingSchedule

Q_F_3 = [0.5, 0.3, 0.2]
TARGETS = ("inclusive", "mean", "harmonic", "harmonic_equal")
_ONEHOT, _X = "inputs.container_onehot", {"edges": [0.0, 1.0]}
_VARS = {"x": {"edges": [0.0, 1.0, 2.0, 3.0]}, "c": {"categories": ["a", "b"]}}  # cell 2*xbin+c
_VALID = {"pt": {"edges": [0.0, 1.0, 2.5, math.inf], "scale": 1e-3}, "cat": {"categories": "abcd"}}
_NNV = {
    "pt": _VALID["pt"], "eta": {"edges": [0.0, 1.0, 2.5], "abs": True},
    "subcampaign": {"categories": "abc"}}
_SUB = tuple(f"subcampaign_{c}" for c in "abc")  # its spec-order one-hot block
_VAL = tuple(map(torch.tensor, (  # (x, c, y); rows 5, 6: empty cell
    [0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 0.5], [0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 2, 0, 1, 2, 0, 1])))
_DEFAULTS = {  # configured nn, accumulate nn, binned
    "nn": {"mode": "nn", "task_name": "kinematic_prior", "q_f": Q_F_3},
    "acc": {"mode": "nn", "task_name": "head", "q_f": None, "n_classes": 3},
    "binned": {"mode": "binned", "n_classes": 3, "variables": _VARS}}


def _oh(specs, **codes):  # float32 one-hot of each categorical's codes, blocks in spec order
    return torch.cat([
        F.one_hot(torch.as_tensor(codes[s.name]).long(), s.n_bins).float()
        for s in specs.values() if s.categorical], -1)


def make_kw(kind="acc", fields=None, **overrides) -> KinematicWeight:
    """Build and bind; `fields` overrides the schema's spec-order one-hot field names."""
    kw = KinematicWeight(**{**_DEFAULTS[kind], **overrides})
    kw.name = "kw"
    specs = (kw.variables or {}).values()
    names = tuple(f"{s.name}_{c}" for s in specs for c in s.categories or ())
    kw.bind(ResolvedSchema(widths={}, fields={_ONEHOT: names} if fields is None else fields))
    return kw


def _bundle(labels, logits=None, task="head", onehot=None, **fields):
    b = Bundle()
    keyed = {f"labels.jets.{k}": v for k, v in {"flavour_label": labels, **fields}.items()}
    for key, value in {**keyed, f"preds.jets.{task}": logits, _ONEHOT: onehot}.items():
        if value is not None:
            b.set(key, torch.as_tensor(value))
    return b


def _xc(x, c, y):  # a `_VARS` bundle: c enters only through the one-hot
    return _bundle(y, x=x, onehot=F.one_hot(torch.as_tensor(c).long(), 2).float())


def _w(kw, b, mode=Mode.FIT):
    return kw(b, mode)[f"labels.jets.{kw.weight_field}"]


def _expected(q, p, p_floor=1e-4, w_max=50.0):  # the legacy inclusive formula
    return torch.clamp(q / torch.clamp(p, min=p_floor, max=1.0), min=0.0, max=w_max)


def _np_s(target, pi, P):  # harmonic-family S: α = π (harmonic) or α = 1 (harmonic_equal)
    return ((pi**2 if target == "harmonic" else pi) / P).sum(-1)


def _np_w(target, pi, P, y, z=None):  # numpy t/p_f from pi (K,), clipped P (N, K), labels y
    base = pi[y] / P[np.arange(len(y)), y]
    if target in ("inclusive", "mean"):
        return base if target == "inclusive" else base / len(pi) * (P / pi).sum(-1)
    return base / (z * _np_s(target, pi, P))


def _small_batches(seed, n_batches=3, rows=60):
    g, out = torch.Generator().manual_seed(seed), []
    for _ in range(n_batches):
        x, c = torch.rand(rows, generator=g) * 3, torch.randint(0, 2, (rows,), generator=g)
        c[x >= 2] = 0  # leaves cell (x-bin 2, c=1) empty
        out.append((x, c, torch.randint(0, 3, (rows,), generator=g)))
    return out


def _np_binned(target, batches, val=_VAL, p_floor=1e-4, w_max=50.0):
    """`_VARS` cell counts of `batches` and the frozen-table weights of `val`, in numpy."""
    counts = np.zeros((6, 3))
    cells = [
        (np.minimum(x.numpy() // 1, 2) * 2 + c.numpy()).astype(int) for x, c, _ in [*batches, val]]
    for cell, (_, _, y) in zip(cells, batches, strict=False):
        np.add.at(counts, (cell, y.numpy()), 1)
    n_c, total = counts.sum(1), counts.sum()
    P, pi = np.clip(counts / np.maximum(n_c, 1)[:, None], p_floor, 1), counts.sum(0) / total
    z = ((n_c / total) / _np_s(target, pi, P))[n_c > 0].sum() if "harmonic" in target else None
    return counts, np.clip(_np_w(target, pi, P[cells[-1]], val[2].numpy(), z), 0, w_max)


def _patch_ddp(monkeypatch, fake):
    for name, value in (("is_available", True), ("is_initialized", True), ("get_world_size", 2)):
        monkeypatch.setattr(torch.distributed, name, lambda value=value: value)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake)


@pytest.mark.parametrize("kwargs", [
    *({"variables": {"x": v}} for v in (
        {**_X, "categories": "ab"}, {"scale": 1.0}, {"edges": [0, 2, 1]}, {"edges": [0]},
        {"edges": [0, math.inf, 5]}, {**_X, "scale": 0.0}, {**_X, "bogus": 1},
        {"categories": "a"}, {"categories": "aa"},
    )),  # both/neither edges|categories, non-increasing, interior inf, scale, key, categories
    {"q_f": Q_F_3, "n_classes": None}, {"variables": None},
    {"mode": "nn", "q_f": Q_F_3, "n_classes": None, "variables": None, "target": "harmonic"},
    {"mode": "nn", "variables": None, "target": "bogus"},
    *({"min_cell_count": bad} for bad in (0, True, 2.5)),
])
def test_variables_spec_errors(kwargs):
    with pytest.raises(ConfigError):
        make_kw("binned", **kwargs)
    specs = parse_variables(_VALID, "test")
    pt = specs["pt"]
    cat = specs["cat"]
    assert list(specs) == ["pt", "cat"] and [float(e) for e in pt.finite_edges] == [0.0, 1.0, 2.5]
    assert (pt.n_bins, pt.dtype, cat.n_bins, cat.dtype) == (3, "float32", 4, "int64")


def test_nd_cell_counts_match_histogramdd():
    inf = math.inf
    g = torch.Generator().manual_seed(11)
    cols = []
    labels = []
    variables = {
        "pt": {"edges": [20.0, 50.0, 100.0, 300.0], "scale": 1e-3},  # finite: clamp
        "eta": {"edges": [0.0, 1.0, 2.5, inf], "abs": True},
        "phi": {"edges": [-inf, -1.0, 1.0, inf]}, "cat": {"categories": "abcd"}}
    kw = make_kw("binned", variables=variables)
    requires = flatten_spec(kw.declare_io(Mode.FIT).requires)
    n_cells = kw.n_cells
    keys = (*(f"labels.jets.{v}" for v in ("pt", "eta", "phi")), _ONEHOT)  # cat: via the one-hot
    assert [requires[k].dtype for k in keys] == ["float32"] * 4
    assert "labels.jets.cat" not in requires
    assert n_cells == 3 * 3 * 3 * 4
    for _ in range(3):
        pt = torch.empty(400).uniform_(math.log(5e3), math.log(1e6), generator=g).exp()  # MeV
        eta, phi = (torch.empty(400).uniform_(-4.0, 4.0, generator=g) for _ in range(2))
        cat = torch.randint(0, 4, (400,), generator=g)
        y = torch.randint(-1, 3, (400,), generator=g)  # -1: none
        kw(_bundle(y, pt=pt, eta=eta, phi=phi, onehot=_oh(kw.variables, cat=cat)), Mode.FIT)
        raw = (pt.double() * 1e-3, eta.double().abs(), phi.double(), cat.double())
        cols.append(np.stack([t.numpy() for t in raw], 1))
        labels.append(y.numpy())
    data = np.concatenate(cols)
    y = np.concatenate(labels)
    edges = []
    for i, v in enumerate(["pt", "eta", "phi"]):
        e = np.array([
            math.copysign(1e30, x) if math.isinf(x) else x for x in variables[v]["edges"]])
        data[:, i] = np.clip(data[:, i], e[0], np.nextafter(e[-1], -np.inf))  # right-open + clamp
        edges.append(e)
    expected = np.stack([
        np.histogramdd(data[y == f], bins=[*edges, np.arange(5) - 0.5])[0] for f in range(3)], -1)
    assert np.array_equal(kw.cell_counts.view(3, 3, 3, 4, 3).numpy(), expected)
    assert torch.equal(kw.q_f_counts, torch.from_numpy(expected.reshape(-1, 3).sum(0)))


def test_binned_categorical_axis_uses_onehot_codes():
    kw = make_kw("binned")
    batch = _small_batches(seed=8, n_batches=1)[0]
    b = _xc(*batch)
    assert "labels.jets.c" not in b and torch.equal(b.get(_ONEHOT).argmax(-1), batch[1])
    kw(b, Mode.FIT)
    x, c, y = (t.numpy() for t in batch)
    bins = [_VARS["x"]["edges"], [-0.5, 0.5, 1.5]]
    hist = [np.histogramdd(np.stack([x, c], 1)[y == f], bins=bins)[0] for f in range(3)]
    assert np.array_equal(kw.cell_counts.view(3, 2, 3).numpy(), np.stack(hist, -1))


@pytest.mark.parametrize("code", [4, -1])
def test_categorical_code_out_of_range_raises(code):
    specs = parse_variables({"x": _X, "cat": {"categories": "abcd"}}, "test")
    values = {"x": torch.tensor([0.5, 0.5])}
    edges = {"x": torch.tensor(_X["edges"])}
    assert cell_index({**values, "cat": torch.tensor([0, 3])}, specs, edges)[0].tolist() == [0, 3]
    with pytest.raises(ValueError, match="cat"):
        cell_index({**values, "cat": torch.tensor([0, code])}, specs, edges)


def test_categorical_codes_decode_from_onehot_fields():
    variables = {"subcampaign": {"categories": "abc"}, "shower": {"categories": ["py", "hw"]}}
    # shower first, unrelated around
    fields = ("ntrk", "shower_py", "shower_hw", "flag", *_SUB, "z")
    kw = make_kw(variables=variables, fields={_ONEHOT: fields})
    g = torch.Generator().manual_seed(2)
    x = torch.rand(50, len(fields), generator=g)  # argmax within each block, not a strict one-hot
    y = torch.randint(0, 3, (50,), generator=g)
    out = kw(_bundle(y, torch.randn(50, 3), onehot=x), Mode.FIT)
    assert torch.equal(out[kw.published["shower"]], x[:, 1:3].argmax(-1))
    assert torch.equal(out[kw.published["subcampaign"]], x[:, 4:7].argmax(-1))


@pytest.mark.parametrize(("fields", "match"), [
    *(({_ONEHOT: f}, "subcampaign") for f in (  # missing, wrong order, gap, extra, no block
        _SUB[::2], (*_SUB[1::-1], _SUB[2]), (_SUB[0], "x_y", *_SUB[1:]), (*_SUB, "subcampaign_x"),
        ("shower_a", "shower_b", "shower_c"))),
    ({}, None),  # the schema declares no fields for inputs.container_onehot
])
def test_onehot_fields_mismatch_is_config_error(fields, match):
    with pytest.raises(ConfigError, match=match):
        make_kw(variables={"subcampaign": {"categories": "abc"}}, fields=fields)


@pytest.mark.parametrize("target", TARGETS)
def test_nn_mode_publishes_variables_after_transform(target):
    """NN + variables publishes the transformed axes and leaves weight and accumulators alone."""
    kw = make_kw(target=target, variables=_NNV)
    plain = make_kw(target=target)
    g = torch.Generator().manual_seed(9)
    pub = list(kw.published.values())
    produces = flatten_spec(kw.declare_io(Mode.FIT).produces)
    keys = {kw.weight_key, *pub}
    assert pub == [f"labels.jets.nn_kinematic_weight_{v}" for v in _NNV] and plain.published == {}
    assert set(produces) == keys and [produces[k].dtype for k in pub] == ["float32"] * 2 + ["int64"]
    assert flatten_spec(kw.declare_io(Mode.TEST).produces) == {}
    assert set(kw.state_dict()) == set(plain.state_dict())  # nn mode registers no new buffers

    def check(module, mode):
        y, sub = (torch.randint(0, 3, (40,), generator=g) for _ in range(2))
        logits, eta = torch.randn(40, 3, generator=g), torch.randn(40, generator=g) * 2  # signed
        pt = torch.rand(40, generator=g) * 3e3  # MeV
        onehot = _oh(kw.variables, subcampaign=sub)
        out = module(_bundle(y, logits, pt=pt, eta=eta, onehot=onehot), mode)
        ref = plain(_bundle(y, logits), mode)
        assert set(out) == keys and list(ref) == [plain.weight_key]
        assert torch.equal(out[kw.weight_key], ref[plain.weight_key])
        gev, abs_eta, codes = (out[k] for k in pub)  # codes: argmax of the one-hot block
        dtypes = (gev.dtype, abs_eta.dtype, codes.dtype)
        assert dtypes == (torch.float32, torch.float32, torch.int64)
        assert torch.allclose(gev, pt * 1e-3) and torch.equal(abs_eta, eta.abs())
        assert torch.equal(codes, sub)

    check(kw, Mode.FIT)
    check(kw, Mode.FIT)
    for module in (kw, plain):
        module.freeze_accumulators(2, 0)
    check(kw, Mode.VAL)
    fresh = make_kw(target=target, variables=_NNV)
    fresh.load_state_dict(kw.state_dict())
    check(fresh, Mode.VAL)
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in fresh.state_dict().items())


@pytest.mark.parametrize("target", TARGETS)
def test_binned_posterior_after_freeze_matches_numpy(target):
    kw = make_kw("binned", target=target)
    batches = _small_batches(seed=1)
    for batch in batches:
        kw(_xc(*batch), Mode.FIT)
    kw.freeze_accumulators(3, 0)
    counts, expected = _np_binned(target, batches)
    assert torch.equal(kw.cell_counts, torch.from_numpy(counts))
    w = _w(kw, _xc(*_VAL), Mode.VAL)
    np.testing.assert_allclose(w.numpy(), expected, rtol=1e-5)
    assert torch.isfinite(w).all() and (target != "inclusive" or w[5] == w[6] == 50.0)  # empty
    cells = kw.cell_counts.clone()
    counts_q = kw.q_f_counts.clone()
    kw(_xc(*batches[0]), Mode.FIT)
    assert torch.equal(kw.cell_counts, cells) and torch.equal(kw.q_f_counts, counts_q)


def test_sparse_cells_warn_at_freeze(caplog):
    with pytest.raises(RuntimeError, match="no accumulated"):
        make_kw("binned").freeze_accumulators(0, 0)
    kw = make_kw("binned", variables={"x": {"edges": [0, 1, 2, 3, 4]}}, min_cell_count=5)
    x = torch.repeat_interleave(torch.tensor([0.5, 1.5, 2.5, 3.5]), torch.tensor([10, 3, 2, 10]))
    kw(_bundle(torch.arange(25) % 3, x=x), Mode.FIT)
    with caplog.at_level(logging.INFO, logger="salt"):
        kw.freeze_accumulators(7, 0)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2" in warnings[0] and "0.200" in warnings[0]  # 2 cells, 5/25
    assert "4" in caplog.text and kw.accumulators_frozen is True


def test_binned_scheduled_stage_boundary_freeze_checkpoint_roundtrip(tmp_path):
    kw = make_kw("binned", target="harmonic")
    batches = _small_batches(seed=5)
    kw.name = "kinematic_weight"
    stages = {"priors": {"epochs": 1, "frozen": []}, "main": {"frozen": []}}
    schedule = TrainingSchedule.from_config({"stages": stages}, ("kinematic_weight",))
    controller = TrainingController(schedule, torch.nn.ModuleDict({kw.name: kw}), {}, "AdamW")
    for batch in batches:
        kw(_xc(*batch), Mode.FIT)
    controller.advance_to_stage(1, global_step=3, epoch=0, reason="epochs")
    assert kw.accumulators_frozen is True
    _, expected = _np_binned("harmonic", batches)
    np.testing.assert_allclose(_w(kw, _xc(*_VAL), Mode.VAL).numpy(), expected, rtol=1e-5)
    torch.save(kw.state_dict(), tmp_path / "kw.pt")
    fresh = make_kw("binned", target="harmonic")
    fresh.load_state_dict(torch.load(tmp_path / "kw.pt"))
    for key in ("cell_counts", "q_f", "z", "q_f_frozen"):
        assert torch.equal(getattr(fresh, key), getattr(kw, key)), key
    frozen = {k: getattr(kw, k).clone() for k in ("cell_counts", "q_f_counts", "z")}
    for module in (kw, fresh):  # stage-2 FIT forwards
        module(_xc(*batches[1]), Mode.FIT)
        assert all(torch.equal(getattr(module, k), v) for k, v in frozen.items())
        np.testing.assert_allclose(_w(module, _xc(*_VAL)).numpy(), expected, rtol=1e-5)


@pytest.mark.parametrize("target", TARGETS)
def test_targets_match_numpy_reference(target):
    # exact posteriors: 1-D Gaussians 1.5σ apart x a class-dependent category, unequal priors
    rng = np.random.default_rng(7)
    n = 20000
    prior = np.array([0.6, 0.3, 0.1])
    mu = np.array([0.0, 1.5, 3.0])
    cat_p = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    y = rng.choice(3, size=n, p=prior)
    x = rng.normal(mu[y], 1.0)
    c = (rng.random(n) < cat_p[y, 1]).astype(int)
    log_p = np.log(prior) - 0.5 * (x[:, None] - mu) ** 2 + np.log(cat_p[:, c].T)
    log_p -= np.logaddexp.reduce(log_p, axis=1, keepdims=True)
    kw = make_kw(target=target, p_floor=1e-12, w_max=math.inf)
    w = _w(kw, _bundle(torch.from_numpy(y), torch.tensor(log_p, dtype=torch.float32)))
    pi, P = np.bincount(y, minlength=3) / n, np.clip(np.exp(log_p), 1e-12, 1)  # q_f after 1 batch
    z = (1 / _np_s(target, pi, P)).mean()
    np.testing.assert_allclose(w.double().numpy(), _np_w(target, pi, P, y, z), rtol=1e-5, atol=1e-5)
    assert "harmonic" not in target or kw.z.item() == pytest.approx(z, rel=1e-5)


@functools.cache
def _population_jets():
    """Exact finite population: integer multiplicities n[f, x] on 8 x-points x 2 categories."""
    m, c = np.divmod(np.arange(16), 2)
    n = np.stack([(8 - m) * 10 + 5 * c + 1, (m + 1) * 3 + 2 * (1 - c) + 1, 2 + m % 3 + 3 * c])
    K, M = n.shape
    rows = np.repeat(np.arange(K), M)  # one row per (f, x), f-major
    y, x = np.repeat(rows, n.ravel()), np.repeat(np.tile(np.arange(M), K), n.ravel())
    P, first = (n / n.sum(0)).T, np.concatenate([[0], np.cumsum(n.ravel())[:-1]]).reshape(K, M)
    return n, rows, y, x, P, first


@functools.cache
def _population():
    """`_population_jets`' prior, p(x|f) and per-(f, x) FIT weights vs closed form, per target."""
    n, rows, y, x, P, first = _population_jets()
    K, M = n.shape
    pi = n.sum(1) / n.sum()
    px = n.sum(0) / n.sum()
    pfx = n / n.sum(1, keepdims=True)
    logits = torch.tensor(np.log(P[x]), dtype=torch.float32)
    weights = {}
    closed = {}
    for t in TARGETS:
        kw = make_kw(target=t, p_floor=1e-12, w_max=math.inf)
        weights[t] = _w(kw, _bundle(torch.from_numpy(y), logits)).double().numpy()[first]
        z = (px / _np_s(t, pi, P)).sum() if "harmonic" in t else None
        closed[t] = _np_w(t, pi, np.tile(P, (K, 1)), rows, z).reshape(K, M)
    return pi, pfx, weights, closed


def test_per_class_mean_weight_is_one_for_every_target():
    _, pfx, weights, closed = _population()
    for target in TARGETS:
        np.testing.assert_allclose(weights[target], closed[target], rtol=1e-5)
        np.testing.assert_allclose((pfx * weights[target]).sum(1), 1.0, atol=1e-5)


def test_harmonic_maximises_pooled_ess():
    pi, pfx, weights, _ = _population()
    ess = {t: 1 / (pi[:, None] * pfx * w**2).sum() for t, w in weights.items()}
    assert all(ess["harmonic"] > ess[t] * (1 + 1e-9) for t in TARGETS if t != "harmonic"), ess


def test_harmonic_equal_minimises_summed_inverse_class_ess():
    _, pfx, weights, _ = _population()
    inv = {t: (pfx * w**2).sum(1).sum() for t, w in weights.items()}  # Σ_f 1/ESS_f
    others = [t for t in TARGETS if t != "harmonic_equal"]
    assert all(inv["harmonic_equal"] < inv[t] * (1 - 1e-9) for t in others), inv


def test_balanced_prior_recovers_the_posterior_for_every_target():
    pi, pfx, _, closed = _population()
    _, _, y, x, P, first = _population_jets()
    P_bal = (P / pi) / (P / pi).sum(1, keepdims=True)  # what a class-balanced prior learns
    logits_bal = torch.tensor(np.log(P_bal[x]), dtype=torch.float32)
    clip = {"p_floor": 1e-12, "w_max": math.inf, "balanced_prior": True}
    kws = [(t, make_kw(target=t, **clip)) for t in TARGETS]  # running q_f = pi after one batch
    kws.append(("inclusive", make_kw("nn", task_name="head", q_f=pi.tolist(), **clip)))  # frozen
    for target, kw in kws:
        w = _w(kw, _bundle(torch.from_numpy(y), logits_bal)).double().numpy()[first]
        np.testing.assert_allclose(w, closed[target], rtol=1e-5)
        np.testing.assert_allclose((pfx * w).sum(1), 1.0, atol=1e-5)


def test_balanced_prior_false_is_bitwise_todays_weight():
    y = torch.tensor([0] * 12 + [1] * 5 + [2] * 3)  # unequal class counts: non-uniform q_f
    logits = torch.randn(len(y), 3, generator=torch.Generator().manual_seed(13))
    for t in TARGETS:
        opts = ({}, {"balanced_prior": False}, {"balanced_prior": True})
        kws = [make_kw(target=t, **o) for o in opts]
        for step, mode in (("fit", Mode.FIT), ("frozen", Mode.FIT), ("val", Mode.VAL)):
            if step == "frozen":
                for kw in kws:
                    kw.freeze_accumulators(1, 0)
            today, off, on = (_w(kw, _bundle(y, logits), mode) for kw in kws)
            assert torch.equal(today, off) and not torch.equal(today, on), (t, step)


def test_balanced_prior_needs_nn_mode():
    with pytest.raises(ConfigError, match="balanced_prior"):
        make_kw("binned", balanced_prior=True)
    assert make_kw("binned").balanced_prior is False
    assert make_kw("nn", balanced_prior=True).balanced_prior is True


def test_default_field_names_and_explicit_override():
    for kind, field in (("nn", "nn_kinematic_weight"), ("binned", "binned_kinematic_weight")):
        kw = make_kw(kind)
        axes = _VARS if kind == "binned" else ()
        assert (kw.weight_field, kw.weight_key) == (field, f"labels.jets.{field}")
        assert kw.published == {v: f"labels.jets.{field}_{v}" for v in axes}
        produces = set(flatten_spec(kw.declare_io(Mode.FIT).produces))
        assert produces == {kw.weight_key, *kw.published.values()}
    custom = make_kw("binned", weight_field="custom")
    assert (custom.weight_field, custom.published["x"]) == ("custom", "labels.jets.custom_x")


class TestDeclareIoModeGating:
    def test_fit_val_produce_the_key_test_onnx_are_empty_and_requires(self):
        for mode in (Mode.FIT, Mode.VAL):
            produces = flatten_spec(make_kw("nn").declare_io(mode).produces)
            s = produces["labels.jets.nn_kinematic_weight"]
            expected = (("B",), "float32", "label", Mode.TRAINING)
            assert (s.shape, s.dtype, s.kind, s.modes) == expected
        ios = [make_kw(k).declare_io(m) for k in ("nn", "binned") for m in (Mode.TEST, Mode.ONNX)]
        assert all(flatten_spec(io.requires) == flatten_spec(io.produces) == {} for io in ios)
        requires = flatten_spec(make_kw("nn").declare_io(Mode.FIT).requires)
        assert set(requires) == {"labels.jets.flavour_label", "preds.jets.kinematic_prior"}
        assert requires["preds.jets.kinematic_prior"].shape == ("B", 3)
        assert make_kw("nn").variables is None
        requires = flatten_spec(make_kw("binned").declare_io(Mode.FIT).requires)
        assert set(requires) == {"labels.jets.flavour_label", "labels.jets.x", _ONEHOT}
        assert [requires[k].dtype for k in ("labels.jets.x", _ONEHOT)] == ["float32", "float32"]


class TestWiringProof:
    SOURCES = unflatten_spec({
        "labels.jets.flavour_label": TensorSpec(("B",), "int64", kind="label", modes=Mode.TRAINING),
        "preds.jets.kinematic_prior": TensorSpec(shape=("B", 3), dtype="float32"),
        _ONEHOT: TensorSpec(("B", 11), "float32"),
        "pooled.global": TensorSpec(("B", 16), "float32"),
    })

    @staticmethod
    def _modules(sample_weight="nn_kinematic_weight"):
        kw = make_kw("nn", variables={"subcampaign": {"categories": "abc"}})
        kw.name = "kinematic_weight"
        loss = {"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"reduction": "none"}}
        head = ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["b", "c", "u"],
            input="pooled.global", sample_weight=sample_weight, loss=loss)
        head.name = "main_head"
        return {"kinematic_weight": kw, "main_head": head}

    def test_weight_module_ordered_before_head_and_key_satisfied_in_model_plan(self):
        plan = compile_plan(self._modules(), Mode.FIT, self.SOURCES)
        assert plan.module_names.index("kinematic_weight") < plan.module_names.index("main_head")
        edges = {(e.producer, e.key, e.consumer) for e in plan.edges}  # in-model, not boundary
        assert ("kinematic_weight", "labels.jets.nn_kinematic_weight", "main_head") in edges

    def test_published_key_is_a_model_plan_sink_anchor(self):
        """A published key demanded as a sink is produced in-model (a sink anchor, not an Edge)."""
        modules = self._modules()
        key = modules["kinematic_weight"].published["subcampaign"]
        sinks = [key, "labels.jets.nn_kinematic_weight"]
        plan = compile_plan(modules, Mode.FIT, self.SOURCES, sinks=sinks)
        assert key in plan.step("kinematic_weight").produces
        with pytest.raises(ConnectivityError):  # a misspelt published sink has no producer
            misspelt = key.replace("campaign", "campiagn")
            compile_plan(modules, Mode.FIT, self.SOURCES, sinks=[misspelt])

    def test_mismatched_mode_naming_fails_to_compile(self):
        """A head naming the OTHER mode's field must fail: the guard the mode-specific name buys."""
        with pytest.raises(ConnectivityError, match="binned_kinematic_weight"):
            compile_plan(self._modules("binned_kinematic_weight"), Mode.FIT, self.SOURCES)


class TestNnMaths:
    @pytest.mark.parametrize(("q_f", "w_max", "labels", "logits", "hand"), [
        (Q_F_3, 50.0, [0, 2], [[2.0, 0.5, -1.0], [0.1, 0.1, 0.1]], None),
        ([0.001, 0.001, 0.998], 50.0, [0], [[0.0, 0.0, 50.0]], 0.001 / 1e-4),  # p_floor binds
        ([0.9, 0.1], 1.0, [0], [[0.0, 0.0]], 1.0),  # w_max binds: raw 0.9 / 0.5 = 1.8
    ])
    def test_inclusive_equals_legacy_formula(self, q_f, w_max, labels, logits, hand):
        w = _w(make_kw("nn", task_name="head", q_f=q_f, w_max=w_max), _bundle(labels, logits))
        y = torch.tensor(labels)
        p = torch.softmax(torch.tensor(logits), dim=-1).gather(1, y[:, None]).squeeze(1)
        assert torch.allclose(w, _expected(torch.tensor(q_f)[y], p, w_max=w_max), atol=1e-6)
        assert hand is None or w.item() == pytest.approx(hand, rel=1e-4)

    def test_output_carries_no_grad_and_invalid_label_is_clamped(self):
        kw = make_kw("nn", task_name="head", q_f=[0.5, 0.5])
        w = _w(kw, _bundle([0, -1], torch.tensor([[1.0, -1.0]] * 2, requires_grad=True)))
        assert (w.shape, w.requires_grad, w.grad_fn) == ((2,), False, None)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"q_f": [1.0]}, "at least 2 entries"), ({"q_f": [0.5, 0.5, 0.0]}, "must all be > 0"),
    ({"q_f": [0.5, 0.6]}, "must sum to 1"), ({"mode": "bogus"}, "mode must be one of"),
    ({"p_floor": 1.5}, "p_floor"), ({"w_max": 0.0}, "w_max"),
    ({"n_classes": 3}, "exactly one"), ({"q_f": None}, "exactly one"),  # q_f / n_classes XOR
    *(({"q_f": None, "n_classes": bad}, "n_classes") for bad in (1, True, 2.5)),
])
def test_config_rejected(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        make_kw("nn", **kwargs)


class TestAccumulateMode:
    @staticmethod
    def _fit_batch(kw, labels, *, mode=Mode.FIT) -> dict:
        return kw(_bundle(labels, torch.zeros(len(labels), 3)), mode)

    def test_accumulate_then_mid_epoch_freeze_is_idempotent_and_later_batches_do_not_leak(
        self, caplog):
        """A num_samples/frac_samples stage freezes mid-epoch; later FIT batches must not leak."""
        with pytest.raises(RuntimeError, match="no accumulated"):
            make_kw().freeze_accumulators(0, 0)
        kw = make_kw()
        q3 = torch.tensor([3 / 7, 2 / 7, 2 / 7])
        flat = torch.full((3,), 1 / 3)

        def close(out, ys):  # the weight from q3 at the flat posterior
            expected = _expected(q3[ys], flat)
            return torch.allclose(out["labels.jets.nn_kinematic_weight"], expected, atol=1e-6)

        assert torch.allclose(kw.q_f, flat) and kw.q_f_frozen.item() is False
        assert (kw.q_f_counts.tolist(), kw.z_sum.item(), kw.z.item()) == ([0.0] * 3, 0.0, 1.0)
        for labels in ([0, 0, 1], [2, -1, 1], [0, 5, 2]):  # -1 and 5 are excluded from the counts
            out = self._fit_batch(kw, labels)
        assert torch.equal(kw.q_f_counts, torch.tensor([3.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(kw.q_f, q3, atol=1e-6)
        assert close(out, [0, 2, 2])  # the 3rd batch's weight uses the q_f that batch updated
        with caplog.at_level(logging.INFO, logger="salt"):
            kw.freeze_accumulators(global_step=3, epoch=0)
        assert torch.allclose(kw.q_f, q3, atol=1e-6) and kw.q_f_frozen.item() is True
        assert "q_f frozen at global step 3 (epoch 0)" in caplog.text
        counts_after, q_f_after = kw.q_f_counts.clone(), kw.q_f.clone()
        kw.freeze_accumulators(9, 2)  # idempotent
        for labels in ([1, 1, 1], [2, 2, 2]):
            out = self._fit_batch(kw, labels)
        assert torch.equal(kw.q_f_counts, counts_after) and torch.equal(kw.q_f, q_f_after)
        assert close(out, [2, 2, 2])

    def test_val_batch_does_not_update_counts_or_q_f(self):
        kw, binned = make_kw(), make_kw("binned")
        self._fit_batch(kw, [0, 0, 1])
        binned(_xc(*_VAL), Mode.FIT)
        before = [t.clone() for t in (kw.q_f_counts, kw.q_f, binned.cell_counts, binned.q_f_counts)]
        self._fit_batch(kw, [2, 2], mode=Mode.VAL)
        binned(_xc(*_small_batches(seed=2, n_batches=1)[0]), Mode.VAL)
        after = (kw.q_f_counts, kw.q_f, binned.cell_counts, binned.q_f_counts)
        assert all(torch.equal(a, b) for a, b in zip(after, before, strict=True))

    @pytest.mark.parametrize("target", TARGETS)
    def test_class_that_never_appeared(self, target, caplog):
        """Inclusive warns (frozen q_f 0); a zero prior breaks every other target."""
        kw, inclusive = make_kw(target=target), target == "inclusive"
        raises = nullcontext() if inclusive else pytest.raises(RuntimeError, match=r"\b2\b")
        with caplog.at_level(logging.WARNING, logger="salt"), raises:
            self._fit_batch(kw, [0, 0, 1, 1])
            kw.freeze_accumulators(5, 0)
        assert not inclusive or (kw.q_f[2].item() == 0.0 and "class index 2" in caplog.text)

    @pytest.mark.parametrize("binned", [False, True])
    def test_ddp_all_reduce_runs_every_fit_batch_on_fresh_increments_only(
        self, binned, monkeypatch):
        calls: list[object] = []
        _patch_ddp(monkeypatch, lambda tensor, op=None: (calls.append(op), tensor.mul_(2)))
        if binned:
            kw, batches = make_kw("binned"), _small_batches(seed=4, n_batches=2)
            for batch in batches:
                kw(_xc(*batch), Mode.FIT)
            c1 = _np_binned("inclusive", batches[:1])[0]
            c2 = _np_binned("inclusive", batches[1:])[0]
            kw(_xc(*_VAL), Mode.VAL)
            assert torch.equal(kw.cell_counts, torch.from_numpy(2 * (c1 + c2)))  # not 2*(2*c1+c2)
            expected = torch.from_numpy(2 * (c1 + c2).sum(0))
        else:
            kw = make_kw()
            self._fit_batch(kw, [0, 0, 1])  # local [2, 1, 0] -> reduced [4, 2, 0]
            self._fit_batch(kw, [2])  # local [0, 0, 1] -> reduced [0, 0, 2]
            self._fit_batch(kw, [0], mode=Mode.VAL)
            expected = torch.tensor([4.0, 2.0, 2.0], dtype=torch.float64)
        assert torch.equal(kw.q_f_counts, expected)
        kw.freeze_accumulators(1, 0)
        assert calls == [torch.distributed.ReduceOp.SUM] * 2  # neither VAL nor the freeze
        assert torch.allclose(kw.q_f, (expected / expected.sum()).float(), atol=1e-6)

    def test_ddp_ranks_with_unequal_rows_freeze_the_same_z(self, monkeypatch):
        g = torch.Generator().manual_seed(3)
        labels = {0: torch.tensor([0, 1, 2, 0, 1]), 1: torch.tensor([2, 2, 1, 0, 0, 1, 2, 0, 2])}
        logits = {r: torch.randn(len(y), 3, generator=g) for r, y in labels.items()}
        y_all = torch.cat([labels[0], labels[1]]).numpy()  # bincount below: the union q_f
        pi = np.bincount(y_all, minlength=3) / len(y_all)
        P = {r: np.clip(torch.softmax(logits[r].double(), -1).numpy(), 1e-4, 1) for r in labels}
        z_inc = {r: (1 / _np_s("harmonic_equal", pi, P[r])).sum() for r in labels}
        counts = {r: torch.bincount(labels[r], minlength=3).double() for r in labels}
        kws = {r: make_kw(target="harmonic_equal") for r in labels}
        for r, kw in kws.items():  # each fake all_reduce adds the other rank's fresh increment
            inc = {1: torch.as_tensor(z_inc[1 - r]), 3: counts[1 - r]}
            _patch_ddp(
                monkeypatch, lambda tensor, op=None, inc=inc: tensor.add_(inc[tensor.numel()]))
            kw(_bundle(labels[r], logits[r]), Mode.FIT)
            kw.freeze_accumulators(1, 0)
        assert torch.allclose(kws[0].z, kws[1].z, rtol=1e-6)
        assert kws[0].z.item() == pytest.approx((z_inc[0] + z_inc[1]) / len(y_all), rel=1e-5)


class TestConfiguredModeAndStateDict:
    def test_configured_mode_starts_frozen_and_ignores_fit_and_freeze(self):
        kw = make_kw("nn")
        assert (kw.q_f_frozen.item(), kw.accumulators_frozen) == (True, True)
        kw(_bundle([0, 1], torch.zeros(2, 3), task="kinematic_prior"), Mode.FIT)
        assert torch.equal(kw.q_f_counts, torch.zeros(3, dtype=torch.float64))
        kw.freeze_accumulators(1, 0)  # must not raise despite zero counts
        assert torch.allclose(kw.q_f, torch.tensor(Q_F_3)) and kw.q_f_frozen.item() is True

    def test_state_dict_has_accumulator_keys(self):
        keys = {"q_f", "q_f_counts", "q_f_frozen", "z_sum", "z"}
        assert keys <= set(make_kw().state_dict())
        binned = set(make_kw("binned", variables={"pt": _X}).state_dict())
        assert keys | {"cell_counts"} <= binned and "edges_pt" not in binned
        assert set(make_kw(balanced_prior=True).state_dict()) == set(make_kw().state_dict())
        assert set(make_kw(chain=("c1",), eta=0.5).state_dict()) == set(make_kw().state_dict())

    @pytest.mark.parametrize("target", ["inclusive", "harmonic"])
    def test_load_state_dict_restores_frozen_state_and_stops_accumulation(self, target):
        kw = make_kw(target=target)
        fresh = make_kw(target=target)
        g = torch.Generator()
        kw(_bundle([0, 1, 2, 1], torch.randn(4, 3, generator=g.manual_seed(0))), Mode.FIT)
        kw.freeze_accumulators(1, 0)
        fresh.load_state_dict(kw.state_dict())
        assert fresh.q_f_frozen.item() is True and torch.allclose(fresh.q_f, kw.q_f)
        snap = {k: getattr(kw, k).clone() for k in ("q_f_counts", "q_f", "z_sum", "z")}
        for module in (kw, fresh):  # stage-2 FIT forwards leave q_f and the frozen Z alone
            module(_bundle([2, 2], torch.randn(2, 3, generator=g)), Mode.FIT)
            assert all(torch.equal(getattr(module, k), v) for k, v in snap.items())


def test_eta_one_no_chain_is_bitwise_todays_weight():
    y = torch.tensor([0] * 12 + [1] * 5 + [2] * 3)  # unequal class counts: non-uniform q_f
    logits = torch.randn(len(y), 3, generator=torch.Generator().manual_seed(13))
    for t in TARGETS:
        kws = [make_kw(target=t), make_kw(target=t, eta=1.0, chain=None)]
        assert (kws[1].eta, kws[1].chain, kws[1].chain_exponents) == (1.0, (), ())
        for step, mode in (("fit", Mode.FIT), ("frozen", Mode.FIT), ("val", Mode.VAL)):
            for kw in kws if step == "frozen" else ():
                kw.freeze_accumulators(1, 0)
            today, new = (_w(kw, _bundle(y, logits), mode) for kw in kws)
            assert torch.equal(today, new), (t, step)


def test_eta_shrinks_before_the_clamp():
    y = torch.tensor([0, 1, 2, 0])  # raw weights ~276, ~128, ~438, ~0.87
    logits = torch.tensor([[-4.0, 2.0, 1.0], [3.0, -3.0, 0.0], [2.0, 2.0, -5.0], [1.0, 0.0, 0.0]])

    def w(**kw):
        return _w(make_kw("nn", task_name="head", q_f=Q_F_3, **kw), _bundle(y, logits))
    w1 = w(w_max=math.inf)
    torch.testing.assert_close(w(w_max=math.inf, eta=0.5), w1.sqrt(), rtol=1e-6, atol=0)
    capped = w(w_max=15.0, eta=0.5)
    assert (w1[:3] > 15.0).all() and torch.equal(w(w_max=15.0), w1.clamp(max=15.0))
    torch.testing.assert_close(capped, w1.sqrt().clamp(max=15.0), rtol=1e-6, atol=0)
    assert not torch.allclose(capped, w1.clamp(max=15.0).sqrt())  # eta sits inside the clamp


def test_chain_multiplies_after_the_clamp():
    g = torch.Generator().manual_seed(21)
    y, logits = torch.randint(0, 3, (30,), generator=g), torch.randn(30, 3, generator=g) * 3
    y[0], logits[0] = 0, torch.tensor([-5.0, 5.0, 5.0])  # own weight far above any w_max
    c1, c2 = torch.rand(30, generator=g) * 4, torch.tensor([4.0] * 15 + [1e-4] * 15)  # sqrt 2|.01
    chain = {"chain": ("c1", "c2"), "chain_exponents": (1.0, 0.5)}

    def w(c1_in, w_max, **kw):
        module = make_kw("nn", task_name="head", q_f=Q_F_3, w_max=w_max, **kw)
        return _w(module, _bundle(y, logits, c1=c1_in, c2=c2))
    own = w(c1, math.inf)
    torch.testing.assert_close(w(c1, math.inf, **chain), c1 * c2**0.5 * own, rtol=1e-6, atol=0)
    full = torch.full((30,), 50.0)
    capped, raw = w(full, 50.0, **chain), w(full, 50.0) * 50.0 * c2**0.5  # own clamped first
    hit = raw >= 50.0
    assert bool(hit[0]) and not bool(hit[15:].any())  # both branches exercised
    assert (capped <= 50.0).all() and (capped[hit] == 50.0).all()
    torch.testing.assert_close(capped[~hit], raw[~hit], rtol=1e-6, atol=0)


def test_declare_io_requires_chain_keys():
    kw = make_kw("nn", chain=("c1", "c2"))
    assert kw.chain_exponents == (1.0, 1.0)
    assert kw.chain_keys == ("labels.jets.c1", "labels.jets.c2")
    for mode in (Mode.FIT, Mode.VAL):
        requires = flatten_spec(kw.declare_io(mode).requires)
        specs = [(s.shape, s.dtype, s.kind, s.modes) for s in map(requires.get, kw.chain_keys)]
        assert specs == [(("B",), "float32", "label", Mode.TRAINING)] * 2
    ios = [kw.declare_io(m) for m in (Mode.TEST, Mode.ONNX)]
    assert all(flatten_spec(io.requires) == flatten_spec(io.produces) == {} for io in ios)


@pytest.mark.parametrize(("kwargs", "match"), [
    *(({"eta": e}, "eta must satisfy") for e in (0.0, 1.5)), ({"chain": []}, "at least one field"),
    *(({"chain": c, "chain_exponents": [1.0]}, "pair one exponent") for c in (None, ["a", "b"])),
    ({"chain": ["a"], "chain_exponents": [0.0]}, "chain_exponents must be > 0"),
])
def test_chain_config_rejected(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        make_kw("nn", **kwargs)
