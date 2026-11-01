"""Tests for `salt.callbacks.KinematicReweightPlotter` (reads `KinematicWeight`'s weights)."""

from __future__ import annotations

import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning.pytorch.loggers.comet import CometLogger
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer
from matplotlib.patches import StepPatch

from salt.callbacks.kinematic_reweight_plotter import KinematicReweightPlotter, _figure
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.model.modules.kinematic_weight import KinematicWeight
from salt.tests.unit.callbacks.conftest import make_trainer

inf = math.inf
CLASS_NAMES = ["bjets", "cjets", "ujets"]
CATEGORIES = ["mc20a", "mc23a", "mc23d"]
VARIABLES = {
    "pt": {"edges": [20.0, 50.0, 100.0, 300.0, inf], "scale": 1e-3},
    "eta": {"edges": [0.0, 1.0, 2.5], "abs": True},
    "subcampaign": {"categories": CATEGORIES},
}
PLOTS = [{"x": "pt"}, {"x": "eta"}, {"x": "pt", "split_by": "subcampaign"}]
PLOTS += [{"x": "eta", "split_by": "subcampaign"}, {"x": "subcampaign"}]
SPLIT_KEYS = [f"pt_by_subcampaign_{c}" for c in CATEGORIES]
ETA_SPLIT_KEYS = [f"eta_by_subcampaign_{c}" for c in CATEGORIES]
FIGURE_KEYS = {"pt", "eta", *SPLIT_KEYS, *ETA_SPLIT_KEYS, "subcampaign"}
KW_ARGS = {"task_name": "prior", "mode": "nn", "n_classes": 3}
KW = KinematicWeight(**KW_ARGS, variables=VARIABLES)  # key strings come from the module
LABEL_KEY, WEIGHT_KEY, PUB = KW.label_key, KW.weight_key, KW.published
COLS = (LABEL_KEY, PUB["subcampaign"], PUB["pt"])  # label, subcampaign code, published pt
ERROR_NOTE = (
    "±1σ statistical, own-class only: before sqrt(bin_n)/n_in_range; after sqrt(bin_sum_w2)/"
    "sum_w_in_range; ratio = err/mean over classes (denominator and normalisation not "
    "propagated); empty bins carry 0"
)
OLD_KEYS = ["before", "after", "ratio_before", "ratio_after", "n", "sum_w", "sum_w2", "bin_n"]
OLD_KEYS += ["bin_sum_w", "mean_w"]
RATIO_ERR_KEYS = {"ratio_before_err", "ratio_after_err"}
ERR_KEYS = {"bin_sum_w2", "before_err", "after_err", *RATIO_ERR_KEYS}


def make_plotter(**kwargs) -> KinematicReweightPlotter:
    return KinematicReweightPlotter(**{"plots": PLOTS, **kwargs})


def stub_pl_module(w_max: float = 50.0, target: str = "inclusive") -> SimpleNamespace:
    """SaltModule-shaped stub: a real nn `KinematicWeight`, the prior head, a non-weight module."""
    kw = KinematicWeight(**KW_ARGS, variables=VARIABLES, w_max=w_max, target=target)
    kw.name = "kinematic_weight"
    prior = SimpleNamespace(class_names=CLASS_NAMES)
    schedule = SimpleNamespace(stages=[SimpleNamespace(name="priors")])
    modules = {"kinematic_weight": kw, "prior": prior, "norm": SimpleNamespace()}
    controller = SimpleNamespace(schedule=schedule, current_stage_index=0)
    return SimpleNamespace(_graph_modules=modules, training_controller=controller)


def make_bundle(seed, n=300, *, w_scale=1.0, n_bad=0, n_cats=3, skew=False) -> Bundle:
    """Label, weight + PUBLISHED values; ``skew``: ~50/50/200 classes, pt x (1 + 2 label)."""
    gen = torch.Generator().manual_seed(seed)
    label = torch.randint(0, 6 if skew else 3, (n,), generator=gen, dtype=torch.int64).clamp(max=2)
    label[:n_bad] = -1
    weight = torch.exp(0.4 * torch.randn(n, generator=gen)) * w_scale
    log_pt = math.log(10.0) + torch.rand(n, generator=gen) * (math.log(500.0) - math.log(10.0))
    log_pt += skew * torch.log(1.0 + 2.0 * label.clamp(min=0))  # class 2 overflows 300 GeV
    bundle = Bundle()
    bundle.set(LABEL_KEY, label)
    bundle.set(WEIGHT_KEY, weight.to(torch.float32))
    bundle.set(PUB["pt"], torch.exp(log_pt).to(torch.float32))  # GeV, 10-500
    bundle.set(PUB["eta"], (torch.rand(n, generator=gen) * 3.0).to(torch.float32))  # |eta|
    bundle.set(PUB["subcampaign"], torch.randint(0, n_cats, (n,), generator=gen, dtype=torch.int64))
    return bundle


def run_epoch(cb, pl_module, trainer, bundles) -> None:
    cb.setup(trainer, pl_module, stage="fit")
    for i, bundle in enumerate(bundles):
        cb.on_validation_batch_end(trainer, pl_module, {"bundle": bundle}, None, i)
    cb.on_validation_epoch_end(trainer, pl_module)


def one_epoch(seed: int = 1, **kwargs) -> KinematicReweightPlotter:
    cb = make_plotter(**kwargs)
    run_epoch(cb, stub_pl_module(), make_trainer(), [make_bundle(seed)])
    return cb


def assert_class_ratios(classes, edges, truth=None) -> None:
    """Unit-sum curves; ``ratio_k[b] == curve[b] / class-mean[b]``, ``None`` iff the mean is 0."""
    assert list(classes) == CLASS_NAMES
    for k in ("before", "after"):
        values = np.stack([np.asarray(classes[n][k], dtype=np.float64) for n in CLASS_NAMES])
        assert len(edges) == (mean := values.mean(axis=0)).size + 1
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
        for i, name in enumerate(CLASS_NAMES):
            for b, ratio in enumerate(classes[name][f"ratio_{k}"]):
                assert (ratio is None) == (mean[b] == 0.0)
                if ratio is not None:
                    assert ratio == pytest.approx(values[i, b] / mean[b], rel=1e-9)
    for e in classes.values():  # mean_w = bin_sum_w / bin_n, None iff empty; after ∝ before·mean_w
        mw = [None if v is None or math.isnan(v) else v for v in e["mean_w"]]  # last_plots: NaN
        assert len(mw) == len(s := e["bin_sum_w"]) == len(n := e["bin_n"]) == len(edges) - 1
        assert [v is None for v in mw] == [k == 0 for k in n]
        ok = [a / k for a, k in zip(s, n, strict=True) if k > 0]
        assert [v for v in mw if v is not None] == pytest.approx(ok, rel=1e-9)
        prod = np.array([bv * (mv or 0.0) for bv, mv in zip(e["before"], mw, strict=True)])
        if prod.sum() > 0:
            np.testing.assert_allclose(prod / prod.sum(), e["after"], rtol=1e-6, atol=1e-12)
    for c in range(len(CLASS_NAMES) if truth else 0):  # truth = (label, w, x, extra selection)
        (label, w, x, sel), e = truth, classes[CLASS_NAMES[c]]  # n/sum_w/sum_w2 incl. out-of-range
        wm, hist = w[(m := (label == c) & sel)], np.histogram(x[m], bins=edges)[0].tolist()
        assert (e["n"], e["bin_n"]) == (int(m.sum()), hist)  # bin_n: raw in-range counts
        assert (e["sum_w"], e["sum_w2"]) == pytest.approx((wm.sum(), (wm**2).sum()), rel=1e-6)
        assert e["bin_sum_w"] == pytest.approx(np.histogram(x[m], edges, weights=wm)[0], rel=1e-9)
    assert_class_errors(classes, edges, truth)


def _none(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def assert_class_errors(classes, edges, truth=None) -> None:
    """``before_err = sqrt(bin_n)/n_in``, ``after_err = sqrt(bin_sum_w2)/s_in``, ratio err/mean."""
    for e in classes.values():
        assert len(e["bin_sum_w2"]) == len(e["before_err"]) == len(e["after_err"])
        assert len(e["bin_sum_w2"]) == len(edges) - 1
        n_in, s_in = sum(e["bin_n"]), sum(e["bin_sum_w"])
        exp_b = [math.sqrt(n) / n_in if n_in else 0.0 for n in e["bin_n"]]
        exp_a = [math.sqrt(w2) / s_in if s_in else 0.0 for w2 in e["bin_sum_w2"]]
        assert e["before_err"] == pytest.approx(exp_b, rel=1e-9)
        assert e["after_err"] == pytest.approx(exp_a, rel=1e-9)
    for k in ("before", "after") if all("ratio_before" in e for e in classes.values()) else ():
        values = np.stack([np.asarray(classes[n][k], dtype=np.float64) for n in classes])
        mean = values.mean(axis=0)
        for e in classes.values():
            for b, r in enumerate(map(_none, e[f"ratio_{k}_err"])):
                assert (r is None) == (mean[b] == 0.0)
                if r is not None:
                    assert r == pytest.approx(e[f"{k}_err"][b] / mean[b], rel=1e-9)
    for c in range(len(CLASS_NAMES) if truth else 0):
        (label, w, x, sel), e = truth, classes[CLASS_NAMES[c]]
        wm, xm = w[(m := (label == c) & sel)], x[m]
        expected = np.histogram(xm, edges, weights=wm**2)[0]
        assert e["bin_sum_w2"] == pytest.approx(expected, rel=1e-9)


def test_fit_val_demand_is_label_weight_and_published_keys_only():
    pl_module, cb = stub_pl_module(), make_plotter()
    modules = pl_module._graph_modules
    kw = modules["kinematic_weight"]
    demand, used = cb.fit_val_demand(modules), ("pt", "eta", "subcampaign")  # PLOTS order
    assert demand == (kw.label_key, kw.weight_key, *[kw.published[v] for v in used])
    assert not {"labels.jets.pt", "labels.jets.eta", "labels.jets.subcampaign"} & set(demand)
    assert cb.requires == ()  # setup never ran
    only_pt = make_plotter(plots=[{"x": "pt"}]).fit_val_demand(modules)
    assert only_pt == (kw.label_key, kw.weight_key, kw.published["pt"])  # no subcampaign
    with pytest.raises(ConfigError):  # `variables=None`: nothing published to plot
        cb.fit_val_demand({**modules, "kinematic_weight": KinematicWeight(**KW_ARGS)})
    cb.setup(make_trainer(), pl_module, stage="fit")
    assert cb.requires == cb.fit_val_demand(modules)
    got = (cb.stream, cb.label, cb.weight_field, cb.w_max, cb.n_classes, list(cb.class_names))
    assert got == ("jets", "flavour_label", "nn_kinematic_weight", 50.0, 3, CLASS_NAMES)


def test_metrics_come_from_read_weights():
    previous = None
    for w_scale in (1.0, 3.0):  # Bundle is write-once: rebuild the same rows, weights x3
        bundle, cb = make_bundle(7, w_scale=w_scale), make_plotter()
        run_epoch(cb, stub_pl_module(w_max=1.5), make_trainer(), [bundle])
        m, pt = cb.last_metrics, cb.last_plots["pt"]
        label, w = bundle.get(LABEL_KEY).numpy(), bundle.get(WEIGHT_KEY).numpy().astype(np.float64)
        x = bundle.get(PUB["pt"]).numpy().astype(np.float64)  # published GeV, AS GIVEN
        assert m["n_jets"] == len(w)
        assert m["mean_w"] == pytest.approx(w.mean(), rel=1e-6)
        assert m["ess_ratio"] == pytest.approx(w.sum() ** 2 / (len(w) * (w**2).sum()), rel=1e-6)
        assert m["clip_fraction"] == pytest.approx(np.mean(w >= 1.5), rel=1e-6)
        for c, name in enumerate(CLASS_NAMES):
            wc = w[label == c]
            assert m[f"mean_w_{name}"] == pytest.approx(wc.mean(), rel=1e-6)
            assert m[f"ess_{name}"] == pytest.approx(wc.sum() ** 2 / (wc**2).sum(), rel=1e-6)
            hist = np.histogram(x[label == c], bins=pt["edges"])[0]  # never re-scaled
            np.testing.assert_allclose(pt["classes"][name]["before"], hist / hist.sum(), atol=1e-12)
        assert (outside := np.mean((x < 20.0) | (x > 300.0))) > 0
        assert m["frac_outside_pt"] == pytest.approx(outside, rel=1e-6)
        if previous is not None:  # w -> 3w: mean triples, ESS is scale-free, more rows clip
            assert m["mean_w"] == pytest.approx(3 * previous["mean_w"], rel=1e-6)
            assert m["ess_ratio"] == pytest.approx(previous["ess_ratio"], rel=1e-6)
            assert m["clip_fraction"] > previous["clip_fraction"]
            assert m["frac_outside_pt"] == previous["frac_outside_pt"]
        previous = m


def test_continuous_plot_ratio_panels_reproduce_from_json(tmp_path):
    cb = one_epoch(seed=11, output_dir=str(tmp_path))
    text = (tmp_path / "step000030.json").read_text()
    assert "NaN" not in text  # nulls must be JSON null, never a bare NaN token
    payload, split = (plots := json.loads(text)["plots"])["pt"], plots[SPLIT_KEYS[0]]
    bundle, keys = make_bundle(11), (LABEL_KEY, WEIGHT_KEY, PUB["pt"], PUB["subcampaign"])
    label, w, x, code = (bundle.get(k).numpy().astype(np.float64) for k in keys)  # one_epoch's rows
    assert_class_ratios(payload["classes"], payload["edges"], (label, w, x, True))
    assert all(sum(e["bin_n"]) < e["n"] for e in payload["classes"].values())  # pt overflows edges
    assert_class_ratios(split["classes"], split["edges"], (label, w, x, code == 0))  # mc20a
    assert np.asarray(payload["target"]).sum() == pytest.approx(1.0)
    assert len(payload["target"]) == len(payload["edges"]) - 1
    fig = cb.last_figures["pt"]
    assert len(fig.axes) == 4
    assert all((ax.get_yscale(), ax.get_ylim()) == ("linear", (0.0, 2.0)) for ax in fig.axes[2:])
    lo, hi = fig.axes[1].get_ylim()  # the <w> panel: log y, symmetric about 1
    assert (fig.axes[1].get_yscale(), lo) == ("log", pytest.approx(1 / hi, rel=1e-9)) and hi >= 2
    assert len([p for p in fig.axes[1].patches if isinstance(p, StepPatch)]) == 3
    steps = [p for p in fig.axes[0].patches if isinstance(p, StepPatch)]
    assert len(steps) == 2 * 3 + 1  # + the black inclusive-target step
    assert sum(p.get_linestyle() in ("--", "dashed") for p in steps) == 3
    assert fig.axes[0].get_xscale() == "log"  # every pt edge > 0
    assert cb.last_figures["eta"].axes[0].get_xscale() == "linear"  # eta edges start at 0


def test_split_by_plot_one_figure_per_category_with_classes_as_curves():
    cb, bundle = one_epoch(seed=12), make_bundle(12)
    label, code, x = (bundle.get(k).numpy().astype(np.float64) for k in COLS)
    assert len({id(cb.last_figures[k]) for k in SPLIT_KEYS}) == 3  # none overwrote another
    for k, key in enumerate(SPLIT_KEYS):
        assert len(axes := cb.last_figures[key].axes) == 4
        assert set(CLASS_NAMES) <= {t.get_text() for t in axes[0].get_legend().get_texts()}
        classes, edges = cb.last_plots[key]["classes"], cb.last_plots[key]["edges"]
        assert_class_ratios(classes, edges)
        for c, name in enumerate(CLASS_NAMES):  # curves = the classes, restricted to category k
            hist = np.histogram(x[(label == c) & (code == k)], bins=edges)[0]
            np.testing.assert_allclose(classes[name]["before"], hist / hist.sum(), atol=1e-12)
    cb, absent = make_plotter(), {SPLIT_KEYS[2], ETA_SPLIT_KEYS[2]}  # category with no jets
    run_epoch(cb, stub_pl_module(), make_trainer(), [make_bundle(12, n_cats=2)])
    assert set(cb.last_figures) == set(cb.last_plots) == FIGURE_KEYS - absent


@pytest.mark.parametrize("target", ["inclusive", "mean", "harmonic", "harmonic_equal"])
def test_target_curve_matches_definition(target):
    bundle, cb = make_bundle(21, skew=True), make_plotter()
    run_epoch(cb, stub_pl_module(target=target), make_trainer(), [bundle])
    assert cb.target == target and "target" not in cb.last_plots["subcampaign"]
    label, code, x = (bundle.get(k).numpy().astype(np.float64) for k in COLS)
    sels = {"pt": (label >= 0) & (label < 3)} | {k: code == i for i, k in enumerate(SPLIT_KEYS)}
    for key in ("pt", "eta", *SPLIT_KEYS):
        plot, ax = cb.last_plots[key], cb.last_figures[key].axes[0]
        steps = [p for p in ax.patches if isinstance(p, StepPatch)]
        if target != "inclusive":  # joint target: not reconstructible from marginals
            assert (plot["target"], len(steps)) == (None, 2 * 3) and target in plot["target_note"]
            continue
        assert (len(t := np.asarray(plot["target"])), len(steps)) == (len(plot["edges"]) - 1, 7)
        assert t.sum() == pytest.approx(1.0, abs=1e-9)
        assert sum(tuple(p.get_edgecolor()) == to_rgba("k") for p in steps) == 1  # 3 + 3 + target
        if key in sels:  # count-weighted pooled before, NOT the plain mean of class shapes
            hist = np.histogram(x[sels[key]], bins=plot["edges"])[0]
            np.testing.assert_allclose(t, hist / hist.sum(), atol=1e-12)
            assert not np.allclose(t, np.mean([c["before"] for c in plot["classes"].values()], 0))


def test_categorical_plot_bars_per_category():
    bundle, cb = make_bundle(13), make_plotter()
    run_epoch(cb, stub_pl_module(), make_trainer(), [bundle])
    fig = cb.last_figures["subcampaign"]
    assert len(fig.axes) == 2
    bars = [c for c in fig.axes[0].containers if isinstance(c, BarContainer)]
    assert [len(c) for c in bars] == [3] * (2 * 3)  # before + after per class, 3 categories
    wbars = [c for c in fig.axes[1].containers if isinstance(c, BarContainer)]
    assert [len(c) for c in wbars] == [3] * 3 and fig.axes[1].get_yscale() == "log"  # <w> panel
    payload = cb.last_plots["subcampaign"]
    assert list(payload["categories"]) == CATEGORIES
    label, code = bundle.get(LABEL_KEY).numpy(), bundle.get(PUB["subcampaign"]).numpy()
    w = bundle.get(WEIGHT_KEY).numpy().astype(np.float64)
    for c, name in enumerate(CLASS_NAMES):
        entry, code_c, w_c = payload["classes"][name], code[label == c], w[label == c]
        expected_mean = [w_c[code_c == k].mean() for k in range(3)]  # seed 13: every cell filled
        np.testing.assert_allclose(entry["mean_w"], expected_mean, rtol=1e-9)
        assert entry["bin_sum_w"] == pytest.approx([w_c[code_c == k].sum() for k in range(3)])
        assert np.sum(entry["before"]) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(entry["after"]) == pytest.approx(1.0, abs=1e-6)
        expected = np.bincount(code[label == c], minlength=3) / np.sum(label == c)
        np.testing.assert_allclose(entry["before"], expected, atol=1e-9)


SHAPE_ERRORS = [[{"x": "pt", "bins": 3}], [{"split_by": "subcampaign"}], [{"x": "pt"}] * 2, []]
VARIABLE_ERRORS = [  # only resolvable against `kw.variables`
    [{"x": "phi"}],
    [{"x": "pt", "split_by": "shower"}],
    [{"x": "pt", "split_by": "eta"}],  # continuous split_by
    [{"x": "subcampaign", "split_by": "subcampaign"}],
    [{"x": "subcampaign", "edges": [0.0, 1.0, 2.0]}],
    [{"x": "pt", "edges": [0.0, 50.0, inf]}],
    [{"x": "pt", "edges": [0.0, 50.0, 40.0]}],
]


@pytest.mark.parametrize("plots", SHAPE_ERRORS + VARIABLE_ERRORS)
def test_plot_config_errors(plots):
    pl_module, cb = stub_pl_module(), make_plotter(weight_module="norm")
    modules = pl_module._graph_modules
    built = None if plots in SHAPE_ERRORS else make_plotter(plots=plots)  # variable-level: builds
    with pytest.raises(ConfigError):  # shape errors raise at construction, the rest on resolve
        (built or make_plotter(plots=plots)).fit_val_demand(modules)
    for edges in ([-inf, inf], [0.0, inf]):  # < 2 finite edges: needs a plot `edges:`
        mods = {"kinematic_weight": KinematicWeight(**KW_ARGS, variables={"pt": {"edges": edges}})}
        with pytest.raises(ConfigError, match="'pt'"):  # names the variable
            make_plotter(plots=[{"x": "pt"}]).fit_val_demand(mods)
        make_plotter(plots=[{"x": "pt", "edges": [0.0, 50.0, 100.0]}]).fit_val_demand(mods)
    with pytest.raises(ConfigError, match="kinematic_weight"):  # candidates listed
        cb.setup(make_trainer(), pl_module, stage="fit")
    with pytest.raises(ConfigError, match="kinematic_weight"):
        cb.fit_val_demand(modules)
    with pytest.raises(ConfigError):
        make_plotter().setup(make_trainer(), SimpleNamespace(_graph_modules=None), stage="fit")


def test_comet_log_figure_and_metrics_called():
    trainer, cb = make_trainer(logger=MagicMock(spec=CometLogger)), make_plotter()
    run_epoch(cb, stub_pl_module(), trainer, [make_bundle(51)])
    exp = trainer.logger.experiment
    assert exp.log_figure.call_count == 1 + 1 + 3 + 3 + 1  # pt, eta, 3+3 split, subcampaign
    calls = [c.kwargs for c in exp.log_figure.call_args_list]
    names = {f"kinematic_reweight/{k}/step_000030" for k in FIGURE_KEYS}
    assert {c["figure_name"] for c in calls} == names
    assert all(c["step"] == 30 for c in calls)
    exp.log_metrics.assert_called_once()
    assert exp.log_metrics.call_args.kwargs == {"step": 30}
    expected = {f"kinematic_reweight/{k}": v for k, v in cb.last_metrics.items()}
    assert exp.log_metrics.call_args.args[0] == pytest.approx(expected)


def test_output_dir_writes_png_and_json(tmp_path):
    cb = one_epoch(seed=61, output_dir=str(tmp_path))
    assert set(cb.last_figures) == FIGURE_KEYS
    assert {p.name for p in tmp_path.glob("*.png")} == {f"{k}_step000030.png" for k in FIGURE_KEYS}
    assert [p.name for p in tmp_path.glob("*.json")] == ["step000030.json"]
    loaded = json.loads((tmp_path / "step000030.json").read_text())
    assert set(loaded) == set(cb.last_metrics) | {"class_names", "plots"}
    assert {k: loaded[k] for k in cb.last_metrics} == pytest.approx(cb.last_metrics)
    assert loaded["class_names"] == CLASS_NAMES
    assert set(loaded["plots"]) == set(cb.last_figures)
    assert {"mean_w", "bin_sum_w", "bin_n"} <= set(loaded["plots"]["pt"]["classes"][CLASS_NAMES[0]])


def test_mean_weight_panel_range_tracks_w_max():
    cb = make_plotter()
    run_epoch(cb, stub_pl_module(w_max=1.5), make_trainer(), [make_bundle(7, w_scale=3.0)])
    assert cb.last_figures["pt"].axes[1].get_ylim()[1] == pytest.approx(2.0)  # floor 2 > w_max
    cb = make_plotter()
    run_epoch(cb, stub_pl_module(), make_trainer(), [make_bundle(7, w_scale=3.0)])
    classes = cb.last_plots["pt"]["classes"].values()
    m_max = max(max(v, 1 / v) for e in classes for v in e["mean_w"] if v is not None and v > 0)
    assert cb.last_figures["pt"].axes[1].get_ylim()[1] == pytest.approx(max(2.0, min(50.0, m_max)))


def test_sanity_check_and_reset():
    pl_module, cb = stub_pl_module(), make_plotter()
    run_epoch(cb, pl_module, make_trainer(sanity_checking=True), [make_bundle(41)])
    assert cb.last_metrics is None
    cb.on_validation_epoch_end(make_trainer(), pl_module)  # nothing was stashed -> no rows
    assert cb.last_metrics is None
    run_epoch(cb, pl_module, make_trainer(), [make_bundle(42), make_bundle(43)])
    stash = cb.last_metrics
    assert stash["n_jets"] == 600
    cb.on_validation_epoch_end(make_trainer(), pl_module)  # accumulators were reset
    assert cb.last_metrics is stash
    assert set(cb.last_figures) == FIGURE_KEYS
    run_epoch(cb, pl_module, make_trainer(), [make_bundle(44, n=60, n_bad=5)])
    assert (cb.last_metrics["n_ignored"], cb.last_metrics["n_jets"]) == (5, 55)


def test_on_fit_start_writes_comet_experiment_json(tmp_path):
    info = {"key": "k1", "url": "u1", "name": "n1", "project_name": "p1", "workspace": "w1"}
    logger_mock = MagicMock(spec=CometLogger)
    exp = logger_mock.experiment
    exp.get_key.return_value, exp.get_name.return_value = info["key"], info["name"]
    exp.url, exp.project_name, exp.workspace = info["url"], "p1", "w1"
    cb, pl_module = make_plotter(output_dir=str(tmp_path)), stub_pl_module()
    cb.setup(make_trainer(), pl_module, stage="fit")
    cb.on_fit_start(make_trainer(logger=logger_mock), pl_module)
    assert json.loads((tmp_path / "comet_experiment.json").read_text()) == info


def test_no_logger_no_output_dir_warns_once_over_two_epochs(caplog):
    pl_module, cb = stub_pl_module(), make_plotter()
    cb.setup(make_trainer(), pl_module, stage="fit")
    with caplog.at_level(logging.WARNING, logger="salt"):
        for step, seed in ((30, 71), (60, 72)):
            trainer = make_trainer(global_step=step)
            cb.on_validation_batch_end(trainer, pl_module, {"bundle": make_bundle(seed)}, None, 0)
            cb.on_validation_epoch_end(trainer, pl_module)
    assert len([r for r in caplog.records if "nowhere to log" in r.getMessage()]) == 1


def test_class_names_fall_back_without_prior_entry():
    pl_module, cb = stub_pl_module(), make_plotter()
    del pl_module._graph_modules["prior"]
    cb.setup(make_trainer(), pl_module, stage="fit")
    assert list(cb.class_names) == ["class_0", "class_1", "class_2"]


def test_error_bands_match_hand_computed_case():
    f = lambda v: np.asarray(v, dtype=np.float64)  # noqa: E731
    groups = {
        "A": (f([0.5] * 4), f([1.0] * 4)),
        "B": (f([0.5, 0.5, 1.5]), f([1.0, 3.0, 2.0])),
        "C": (f([1.5, 1.5]), f([1.0, 1.0])),
        "D": (f([5.0]), f([1.0])),  # out of range: every curve and error is 0
    }
    bins = np.array([0.0, 1.0, 2.0, 3.0])
    entries, _, extra = _figure(bins, groups, "t", "x", None, "inclusive")
    a, b, c, d = (entries[k] for k in "ABCD")
    tol = {"rel": 1e-12, "abs": 1e-12}
    assert a["before"] == pytest.approx([1.0, 0.0, 0.0], **tol)
    assert a["before_err"] == pytest.approx([0.5, 0.0, 0.0], **tol)  # sqrt(4)/4
    assert a["after_err"] == pytest.approx([0.5, 0.0, 0.0], **tol)  # sqrt(4)/4 (w = 1)
    assert (b["bin_n"], b["bin_sum_w"]) == ([2, 1, 0], pytest.approx([4.0, 2.0, 0.0], **tol))
    assert b["bin_sum_w2"] == pytest.approx([10.0, 4.0, 0.0], **tol)  # 1 + 9, 4
    assert b["before"] == pytest.approx([2 / 3, 1 / 3, 0.0], **tol)
    assert b["before_err"] == pytest.approx([math.sqrt(2) / 3, 1 / 3, 0.0], **tol)
    assert b["after"] == pytest.approx([4 / 6, 2 / 6, 0.0], **tol)
    assert b["after_err"] == pytest.approx([math.sqrt(10) / 6, 2 / 6, 0.0], **tol)
    assert c["before_err"] == pytest.approx([0.0, math.sqrt(2) / 2, 0.0], **tol)
    assert c["after_err"] == pytest.approx([0.0, math.sqrt(2) / 2, 0.0], **tol)
    assert d["n"] == 1 and d["bin_n"] == [0, 0, 0]
    for k in ("before", "after", "before_err", "after_err", "bin_sum_w2", "bin_sum_w"):
        assert d[k] == [0.0, 0.0, 0.0]
    mean = {
        "before": [(1 + 2 / 3 + 0 + 0) / 4, (0 + 1 / 3 + 1 + 0) / 4],
        "after": [(1 + 4 / 6 + 0 + 0) / 4, (0 + 2 / 6 + 1 + 0) / 4],
    }
    assert a["ratio_before_err"][0] == pytest.approx(0.5 / ((1 + 2 / 3 + 0 + 0) / 4), **tol)
    for e in (a, b, c, d):
        for k in ("before", "after"):
            assert math.isnan(e[f"ratio_{k}_err"][2])  # bin 2 is empty in every class
            assert math.isnan(e[f"ratio_{k}"][2])
            for i in (0, 1):
                assert e[f"ratio_{k}_err"][i] == pytest.approx(e[f"{k}_err"][i] / mean[k][i], **tol)
    assert extra["error_note"] == ERROR_NOTE
    assert "target" in extra


def test_error_keys_are_additive_and_old_keys_unchanged():
    cb, bundle = one_epoch(seed=31), make_bundle(31)
    keys = (LABEL_KEY, WEIGHT_KEY, PUB["pt"], PUB["subcampaign"])
    label, w, x, code = (bundle.get(k).numpy().astype(np.float64) for k in keys)
    for key in ("pt", *SPLIT_KEYS):
        for entry in cb.last_plots[key]["classes"].values():
            assert set(entry) == set(OLD_KEYS) | ERR_KEYS
            assert list(entry)[:10] == OLD_KEYS
    pt = cb.last_plots["pt"]
    assert_class_ratios(pt["classes"], pt["edges"], (label, w, x, True))
    split = cb.last_plots[SPLIT_KEYS[0]]
    assert_class_ratios(split["classes"], split["edges"], (label, w, x, code == 0))
    assert pt["error_note"] == ERROR_NOTE
    assert all(cb.last_plots[k]["error_note"] == ERROR_NOTE for k in SPLIT_KEYS)
    cat = cb.last_plots["subcampaign"]
    cat_keys = (set(OLD_KEYS) - {"ratio_before", "ratio_after"}) | (ERR_KEYS - RATIO_ERR_KEYS)
    for entry in cat["classes"].values():
        assert set(entry) == cat_keys
    assert_class_errors(cat["classes"], [0, 1, 2, 3])
    assert cat["error_note"] == ERROR_NOTE and "target" not in cat


def test_bands_drawn_on_main_and_ratio_panels():
    fig = one_epoch(seed=32).last_figures["pt"]
    polys = lambda ax: [c for c in ax.collections if isinstance(c, PolyCollection)]  # noqa: E731
    assert len(polys(fig.axes[0])) == 2 * 3
    assert len(polys(fig.axes[1])) == 0  # the <w> panel carries no band
    assert len(polys(fig.axes[2])) == 3 == len(polys(fig.axes[3]))
    assert len([p for p in fig.axes[0].patches if isinstance(p, StepPatch)]) == 2 * 3 + 1
    assert len([p for p in fig.axes[1].patches if isinstance(p, StepPatch)]) == 3
    assert "±1σ stat." in {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert all(ax.get_ylim() == (0.0, 2.0) for ax in fig.axes[2:])


def test_categorical_bars_carry_yerr():
    fig = one_epoch(seed=33).last_figures["subcampaign"]
    assert len(fig.axes) == 2
    bars = [c for c in fig.axes[0].containers if isinstance(c, BarContainer)]
    assert len(bars) == 6 and all(c.errorbar is not None for c in bars)
    assert len([c for c in fig.axes[0].collections if isinstance(c, LineCollection)]) == 6
    wbars = [c for c in fig.axes[1].containers if isinstance(c, BarContainer)]
    assert len(wbars) == 3 and all(c.errorbar is None for c in wbars)


def test_setup_runs_for_validate_stage_not_test():
    pl_module, cb = stub_pl_module(), make_plotter()
    modules = pl_module._graph_modules
    cb.setup(make_trainer(), pl_module, stage="validate")
    assert cb.requires == cb.fit_val_demand(modules)
    assert list(cb.class_names) == CLASS_NAMES
    other = make_plotter()
    other.setup(make_trainer(), pl_module, stage="test")
    assert other.requires == ()


def test_on_validation_start_writes_comet_experiment_json(tmp_path):
    info = {"key": "k1", "url": "u1", "name": "n1", "project_name": "p1", "workspace": "w1"}
    logger_mock = MagicMock(spec=CometLogger)
    exp = logger_mock.experiment
    exp.get_key.return_value, exp.get_name.return_value = info["key"], info["name"]
    exp.url, exp.project_name, exp.workspace = info["url"], "p1", "w1"
    cb, pl_module = make_plotter(output_dir=str(tmp_path)), stub_pl_module()
    cb.setup(make_trainer(), pl_module, stage="validate")
    for _ in range(2):  # idempotent rewrite
        cb.on_validation_start(make_trainer(logger=logger_mock), pl_module)
        assert json.loads((tmp_path / "comet_experiment.json").read_text()) == info
