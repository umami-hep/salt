"""Tests for `salt.callbacks.InputNormalisationPlotter` (raw vs standardised input histograms)."""

from __future__ import annotations

import json
import logging
import math
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml
from lightning.pytorch.loggers.comet import CometLogger
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba

from salt.callbacks._plot_io import CLASS_COLOURS
from salt.callbacks.input_normalisation_plotter import InputNormalisationPlotter as Plotter
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.model.bind import ResolvedSchema
from salt.model.modules.norm import MaskedInputNormaliser, Normaliser
from salt.model.modules.tasks import ClassificationTaskModule
from salt.tests.unit.callbacks.conftest import make_trainer

FIELDS = {"inputs.jets": ("pt", "eta"), "inputs.tracks": ("d0", "z0")}
SCHEMA = ResolvedSchema(widths={"inputs.jets": 2, "inputs.tracks": 2}, fields=FIELDS)
STATS = {"jets": {"pt": (3.0, 2.0), "eta": (0.0, 0.5)}, "tracks": {"d0": (1.0, 4.0)}}
STATS["tracks"]["z0"] = (-2.0, 1.0)  # == the population moments of make_bundle()'s values
KEYS = ("jets.pt", "jets.eta", "tracks.d0", "tracks.z0")
MOMENTS = ("mean_raw", "std_raw", "mean_std", "std_std")
ENTRY = {"edges", "counts", "underflow", "overflow", "n", "norm_mean", "norm_std", "stats_warm"}
ENTRY |= {"n_unlabelled", "classes", "integer_bins", "n_nonintegral"}
GAUSS, CLASSES = "N(0,1) in standardised units", ("b", "c", "l")
TASK = partial(Plotter, class_task="task")  # every per-class test names the "task" stub
PT, ETA, D0, Z0 = (1.0, 5.0), (0.5, -0.5), (-3.0, 5.0), (-3.0, -1.0)  # exact in float32


def make_normaliser(tmp_path) -> Normaliser:
    entries = {s: {v: {"mean": m, "std": sd} for v, (m, sd) in d.items()} for s, d in STATS.items()}
    (path := tmp_path / "norm.yaml").write_text(yaml.safe_dump(entries))  # the norm dictionary
    norm = Normaliser(norm_dict=path, streams=["jets", "tracks"], global_object="jets")
    norm.name = "norm"
    norm.bind(SCHEMA)
    norm.materialise()
    return norm


def make_masked(eps=1e-5, warm_bundle=None) -> MaskedInputNormaliser:
    norm = MaskedInputNormaliser(["jets", "tracks"], "jets", momentum=None, eps=eps)
    norm.name = "norm"
    norm.bind(SCHEMA)
    if warm_bundle is not None:  # momentum=None: one training forward sets the exact moments
        norm.train()(warm_bundle, Mode.FIT)
    return norm.eval()


def make_bundle(n_valid=(1, 2, 3, 2), t=3, pt=PT, eta=ETA, pad=1e6, lab=(0, 1, 2, 0)) -> Bundle:
    """jets [B, 2]; tracks [B, t, 2] whose padded rows hold `pad` (mask True == padded)."""
    b, k = len(n_valid), 0  # valid tracks alternate: d0 mean 1 std 4, z0 mean -2 std 1
    jets = torch.tensor([[pt[i % len(pt)], eta[i % len(eta)]] for i in range(b)])
    tracks, mask = torch.full((b, t, 2), pad), torch.ones(b, t, dtype=torch.bool)
    for i, nv in enumerate(n_valid):
        for j in range(nv):
            tracks[i, j], mask[i, j], k = torch.tensor([D0[k % 2], Z0[k % 2]]), False, k + 1
    data = {"inputs": {"jets": jets, "tracks": tracks}, "masks": {"tracks": mask}}
    return Bundle(data | {"labels": {"jets": {"flavour_label": torch.tensor(lab)}}})  # int64 [B]


def stub_pl_module(norm, schema=SCHEMA) -> SimpleNamespace:
    task = partial(ClassificationTaskModule, label="flavour_label", class_names=CLASSES)
    mods = {"norm": norm, "other": SimpleNamespace(), "task": task("jets"), "wrong": task("tracks")}
    mods["mapped"] = task("jets", label_map={0: 0, 1: 1, 2: 2})
    return SimpleNamespace(_graph_modules=mods, schema=schema)


def run_epoch(cb, pl_module, bundles, **trainer_kw) -> Plotter:
    cb.setup(trainer := make_trainer(**trainer_kw), pl_module, stage="fit")
    for i, bundle in enumerate(bundles):
        cb.on_validation_batch_end(trainer, pl_module, {"bundle": bundle}, None, i)
    cb.on_validation_epoch_end(trainer, pl_module)
    return cb


def axes(fig):  # (primary, secondary) after an Agg draw: a bare Figure's draw is a no-op
    FigureCanvasAgg(fig).draw()
    ax = fig.axes[0]
    return ax, next(a for a in (*fig.axes, *ax.child_axes) if isinstance(a, SecondaryAxis))


def column(bundle, key, out=None) -> torch.Tensor:
    """Unpadded ``inputs.<stream>`` (or ``out["normed.<stream>"]``) values of ``stream.var``."""
    stream, var = key.split(".")
    x = bundle.get(f"inputs.{stream}") if out is None else out[f"normed.{stream}"]
    x = x if stream == "jets" else x[~bundle.get(f"masks.{stream}")]
    return x[:, FIELDS[f"inputs.{stream}"].index(var)].double()


def parents(bundle, key) -> np.ndarray:  # parent jet's flavour_label per column(bundle, key) row
    lab, mask = bundle.get("labels.jets.flavour_label"), bundle.get("masks.tracks")
    return (lab if key.startswith("jets") else lab[:, None].expand(mask.shape)[~mask]).numpy()


def stairs(fig) -> dict:  # {label: (values, edges, baseline)} per stairs patch, after an Agg draw
    FigureCanvasAgg(fig).draw()
    return {patch.get_label(): patch.get_data() for patch in fig.axes[0].patches}


def test_fit_val_demand_is_inputs_and_masks_of_normaliser_streams(tmp_path):
    demand = ("inputs.jets", "inputs.tracks", "masks.tracks")
    for norm in (make_normaliser(tmp_path), make_masked()):
        cb, pl_module = Plotter(), stub_pl_module(norm)
        assert cb.requires == ()  # setup never ran
        assert cb.fit_val_demand(pl_module._graph_modules) == demand
        cb.setup(make_trainer(), pl_module, stage="fit")
        assert cb.requires == demand
    with pytest.raises(ConfigError, match=r"normaliser 'other'.*\['norm'\]"):  # candidates
        Plotter(normaliser="other").fit_val_demand(pl_module._graph_modules)
    with pytest.raises(ConfigError):
        Plotter().setup(make_trainer(), SimpleNamespace(_graph_modules=None), stage="fit")
    bad = ({"bins": 0}, {"bins": True}, {"log_y": 1}, {"gaussian": "yes"}, {"max_batches": 0})
    for kwargs in bad:
        with pytest.raises(ConfigError):
            Plotter(**kwargs)


def test_top_axis_mapping_equals_normaliser_forward(tmp_path):
    bundle = make_bundle(pt=(1.0, 5.0, 1.0, 43.0))  # 43: outside the first batch's [1, 5]
    norms = {"norm": make_normaliser(tmp_path), "masked": make_masked(4.0, make_bundle())}
    batches = [make_bundle(), bundle]  # the x-range comes from the first batch: 43 overflows
    runs = {k: run_epoch(Plotter(), stub_pl_module(n), batches) for k, n in norms.items()}
    for name, norm in norms.items():  # every figure exists before any is checked
        cb, normed = runs[name], norm(bundle, Mode.VAL)
        for key in KEYS:
            stream, var = key.split(".")
            e, j = cb.last_plots[key], FIELDS[f"inputs.{stream}"].index(var)
            m, s, (e0, e1) = e["norm_mean"], e["norm_std"], (e["edges"][0], e["edges"][-1])
            z = (column(bundle, key) - m) / s
            torch.testing.assert_close(z, column(bundle, key, normed), atol=1e-5, rtol=0)
            bufs = ("means", "stds") if name == "norm" else ("running_mean", "running_var")
            bm, bs = (float(getattr(norm, f"{b}_{stream}")[j]) for b in bufs)
            assert (m, s) == pytest.approx((bm, bs if name == "norm" else math.sqrt(bs + 4.0)))
            ax, top = axes(cb.last_figures[key])
            assert ax.get_xlim() == (e0, e1)
            assert top.get_xlim() == pytest.approx(((e0 - m) / s, (e1 - m) / s), abs=1e-6)
            assert sum(e["counts"]) + e["underflow"] + e["overflow"] == e["n"]
        assert cb.last_plots["jets.pt"]["overflow"] >= 1
    e = runs["norm"].last_plots["jets.pt"]  # known stats: x = 5 -> 1.0
    assert (e["norm_mean"], e["norm_std"], (5 - e["norm_mean"]) / e["norm_std"]) == (3, 2, 1.0)


def test_padded_constituents_are_excluded(tmp_path):
    norm = make_normaliser(tmp_path)
    padded = make_bundle((1, 2, 3, 2), t=5, pad=1e6)  # 12 padded rows of absurd values
    packed = make_bundle((2, 2, 2, 2), t=2)  # the same 8 valid tracks, nothing padded
    a, b = (run_epoch(Plotter(), stub_pl_module(norm), [x]).last_plots for x in (padded, packed))
    for key in ("tracks.d0", "tracks.z0"):
        assert (e := a[key]) == b[key]
        assert (e["n"], sum(e["counts"]) + e["underflow"] + e["overflow"]) == (8, 8)
    d0 = a["tracks.d0"]
    assert (d0["mean_raw"], d0["std_raw"], d0["overflow"]) == (1.0, 4.0, 0)


def test_standardised_sample_stats_are_zero_one_when_stats_match(tmp_path):
    bundle, eps = make_bundle(), 4.0  # eps comparable to var: a plotter dropping eps fails
    for norm, masked in ((make_normaliser(tmp_path), False), (make_masked(eps, bundle), True)):
        plots = run_epoch(Plotter(), stub_pl_module(norm), [bundle]).last_plots
        for key in KEYS:
            var = column(bundle, key).var(unbiased=False).item()
            std_std = math.sqrt(var / (var + eps)) if masked else 1.0
            assert plots[key]["mean_std"] == pytest.approx(0.0, abs=1e-6)
            assert plots[key]["std_std"] == pytest.approx(std_std, abs=1e-6)


def test_gaussian_overlay_peak_and_toggle(tmp_path):
    norm, bundle = make_normaliser(tmp_path), make_bundle()
    for gaussian in (True, False):
        cb = run_epoch(Plotter(gaussian=gaussian), stub_pl_module(norm), [bundle])
        e, ax = cb.last_plots["jets.pt"], cb.last_figures["jets.pt"].axes[0]
        lines = [line for line in ax.lines if line.get_label() == GAUSS]
        assert len(lines) == int(gaussian)
        assert any("raw:" in t.get_text() and "standardised:" in t.get_text() for t in ax.texts)
        if gaussian:
            x, y = lines[0].get_data()
            peak = 1 / (e["norm_std"] * math.sqrt(2 * math.pi))  # the plain pdf
            assert y.max() == pytest.approx(peak, rel=1e-6)
            assert x[np.argmax(y)] == pytest.approx(e["norm_mean"], abs=1e-9)


def test_x_range_is_the_first_batch_range_widened_5pct(tmp_path):
    eps, s0 = 4.0, math.sqrt(5.0)  # never trained: mean 0, scale sqrt(1 + eps)
    empty = make_bundle((0, 0, 0, 0), pt=(1.25, 5.25), eta=(0.5,))  # jets: the first batch
    full = make_bundle(pt=(1.25, 5.25), eta=(0.5,))
    norms = ((make_normaliser(tmp_path), 0.5, True), (make_masked(eps), s0, False))
    runs = []
    for norm, s_eta, warm in norms:  # warm and unwarmed: the same data-driven edges
        cb = run_epoch(Plotter(), stub_pl_module(norm), [empty, full])
        runs.append(p := cb.last_plots)
        assert all(e["stats_warm"] is warm for e in p.values())
        edges = {k: (p[k]["edges"][0], p[k]["edges"][-1]) for k in KEYS}
        assert edges["jets.pt"] == pytest.approx((1.05, 5.45))  # [1.25, 5.25] widened 5 %
        assert (len(p["jets.pt"]["edges"]), p["jets.pt"]["integer_bins"]) == (101, False)
        half = 0.05 * max(0.5, s_eta)  # constant 0.5: positive half-width
        assert edges["jets.eta"] == pytest.approx((0.5 - half, 0.5 + half))
        assert edges["tracks.d0"] == pytest.approx((-3.5, 5.5))  # deferred past all-padded batch
        d0 = p["tracks.d0"]
        assert (len(d0["edges"]), d0["integer_bins"], d0["n"]) == (10, True, 8)
    assert all(runs[0][k]["edges"] == runs[1][k]["edges"] for k in KEYS if k != "jets.eta")
    ax, top = axes(cb.last_figures["jets.pt"])  # the unwarmed run: mean 0, scale s0
    assert "unwarmed" in ax.get_title()
    assert top.get_xlim() == pytest.approx((1.05 / s0, 5.45 / s0), abs=1e-6)
    cb = run_epoch(Plotter(gaussian=True), stub_pl_module(make_masked(eps)), [empty, empty])
    e, ax = cb.last_plots["tracks.z0"], cb.last_figures["tracks.z0"].axes[0]  # zero valid tracks
    assert (e["n"], any(e["counts"]), [e[k] for k in MOMENTS]) == (0, False, [None] * 4)
    assert (e["edges"][0], e["edges"][-1]) == pytest.approx((-s0, s0))  # mean +/- 1 std
    assert [t.get_text() for t in ax.texts] == ["no valid data"]
    assert not [line for line in ax.lines if line.get_label() == GAUSS]
    norm, bundle = make_normaliser(tmp_path), make_bundle()  # value errors
    with pytest.raises(RuntimeError, match="schema"):
        run_epoch(Plotter(), stub_pl_module(norm, schema=None), [bundle])
    with pytest.raises(RuntimeError, match=r"jets\.pt"):  # a NaN VALID value
        run_epoch(Plotter(), stub_pl_module(norm), [make_bundle(pt=(math.nan, 5.0))])
    with torch.no_grad():
        norm.stds_jets[0] = 0.0
    with pytest.raises(RuntimeError, match=r"jets\.pt"):  # a zero normaliser scale
        run_epoch(Plotter(), stub_pl_module(norm), [bundle])


def test_max_batches_sanity_check_and_reset(tmp_path):
    pl_module = stub_pl_module(make_normaliser(tmp_path))
    first, second = make_bundle(), make_bundle((3, 3, 3, 3))  # 8 and 12 valid tracks
    plots = run_epoch(Plotter(max_batches=1), pl_module, [first, second]).last_plots
    assert (plots["tracks.d0"]["n"], plots["jets.pt"]["n"]) == (8, 4)  # the first batch only
    cb = run_epoch(Plotter(), pl_module, [first], sanity_checking=True)
    assert cb.last_plots is None
    stash = run_epoch(cb, pl_module, [first, second]).last_plots
    assert stash["tracks.d0"]["n"] == 20
    cb.on_validation_epoch_end(make_trainer(), pl_module)  # accumulators were reset
    assert cb.last_plots is stash
    assert run_epoch(cb, pl_module, [second]).last_plots["tracks.d0"]["n"] == 12  # recomputed


def test_comet_log_figure_and_output_dir_write(tmp_path):
    logger, out = MagicMock(spec=CometLogger), tmp_path / "plots"
    cb, pl_module = Plotter(output_dir=str(out)), stub_pl_module(make_normaliser(tmp_path))
    run_epoch(cb, pl_module, [make_bundle()], logger=logger)
    exp = logger.experiment
    assert exp.log_figure.call_count == 4
    calls = [c.kwargs for c in exp.log_figure.call_args_list]
    assert {c["figure_name"] for c in calls} == {f"input_norm/{k}/step_000030" for k in KEYS}
    assert all(c["step"] == 30 for c in calls)
    exp.log_metrics.assert_called_once()
    assert exp.log_metrics.call_args.kwargs == {"step": 30}
    names = {f"input_norm/{k}/{m}" for k in KEYS for m in MOMENTS[2:]}
    assert set(exp.log_metrics.call_args.args[0]) == names
    pngs = {f"{k.replace('.', '_')}_step000030.png" for k in KEYS}
    assert {p.name for p in out.glob("*.png")} == pngs
    assert [p.name for p in out.glob("*.json")] == ["step000030.json"]
    loaded = json.loads(text := (out / "step000030.json").read_text())
    top = {"step", "epoch", "normaliser", "normaliser_class", "n_batches", "plots"}
    assert set(loaded) == top | {"class_task", "class_names"}
    assert (loaded["normaliser_class"], set(loaded["plots"])) == ("Normaliser", set(KEYS))
    assert (loaded["class_task"], loaded["class_names"]) == (None, [])
    assert all(set(e) == ENTRY | set(MOMENTS) for e in loaded["plots"].values())
    assert all((e["classes"], e["n_unlabelled"]) == ({}, 0) for e in loaded["plots"].values())
    assert "NaN" not in text


def test_no_logger_no_output_dir_warns_once(tmp_path, caplog):
    cb, pl_module = Plotter(), stub_pl_module(make_normaliser(tmp_path))
    with caplog.at_level(logging.WARNING, logger="salt"):
        for step in (30, 60):
            run_epoch(cb, pl_module, [make_bundle()], global_step=step)
    assert len([r for r in caplog.records if "nowhere to log" in r.getMessage()]) == 1


def test_class_task_demand_adds_label_key_and_validates(tmp_path):
    demand = ("inputs.jets", "inputs.tracks", "masks.tracks", "labels.jets.flavour_label")
    for norm in (make_normaliser(tmp_path), make_masked()):
        cb, mods = TASK(), (pl_module := stub_pl_module(norm))._graph_modules
        assert (cb.fit_val_demand(mods), run_epoch(cb, pl_module, []).requires) == (demand, demand)
    for bad in ("wrong", "mapped", "other", "missing"):  # tracks stream, label_map, not a task
        with pytest.raises(ConfigError, match=rf"class_task '{bad}'.*candidates: \['task'\]$"):
            Plotter(class_task=bad).fit_val_demand(mods)


def test_per_class_histograms_equal_a_hand_split(tmp_path):
    pl, bundle = stub_pl_module(make_normaliser(tmp_path)), make_bundle()  # lab (0, 1, 2, 0)
    a, b = (run_epoch(TASK(), pl, [x]).last_plots for x in (bundle, make_bundle(t=6)))
    assert a == b  # t=6: the same tracks under the same parent jets, plus padding
    for key, e in a.items():  # tracks follow their PARENT jet's flavour
        col, par, cls = column(bundle, key).numpy(), parents(bundle, key), e["classes"].values()
        hand = [np.histogram(col[par == c], e["edges"])[0].tolist() for c in range(3)]
        assert [c["counts"] for c in cls] == hand and np.sum(hand, 0).tolist() == e["counts"]
        assert sum(c["n"] for c in cls) == e["n"] and e["n_unlabelled"] == 0


def test_per_class_stats_and_unlabelled_total_only(tmp_path):
    pl = stub_pl_module(make_normaliser(tmp_path))  # -1, K = 3: total only
    for key, e in run_epoch(TASK(), pl, [make_bundle(lab=[0, -1, 3, 1])]).last_plots.items():
        cls = e["classes"].values()  # jets 1, 2 (labels -1, 3) hold 2 + 3 tracks
        assert e["n_unlabelled"] == e["n"] - sum(c["n"] for c in cls) == (5 if key[0] == "t" else 2)
        assert sum(e["counts"]) - sum(sum(c["counts"]) for c in cls) == e["n_unlabelled"]
    batches = [make_bundle(pt=p, lab=[0, 0, 1, 1]) for p in ((1.0, 5.0), (100.25, 100.5, 1.0, 5.0))]
    for pl in map(stub_pl_module, (make_normaliser(tmp_path), make_masked(4.0))):  # 2nd: unwarmed
        per_class, total = (run_epoch(cb, pl, batches).last_plots for cb in (TASK(), Plotter()))
        for key, e in per_class.items():  # class 0: narrow and far from the mean in batch 2
            assert {**e, "classes": {}} == total[key]  # classes never perturb the total
            x, p = (np.concatenate([f(b, key) for b in batches]) for f in (column, parents))
            xz = np.stack([x, (x - e["norm_mean"]) / e["norm_std"]])
            for c, s in enumerate(e["classes"].values()):  # class 2 ("l") is empty
                ref = [f(r[p == c]) for r in xz for f in (np.mean, np.std)] if c < 2 else [None] * 4
                assert [s[k] for k in MOMENTS] == pytest.approx(ref, abs=1e-9)
                assert (s["n"], any(s["counts"])) == ((p == c).sum(), c < 2)


def test_per_class_curves_are_unit_area(tmp_path):
    pl, names = stub_pl_module(make_normaliser(tmp_path)), {"total", *CLASSES}
    cb = run_epoch(TASK(gaussian=True), pl, [make_bundle()])
    for fig in cb.last_figures.values():  # K + 1 curves, every one of unit area; full legend
        areas = {k: np.sum(y * np.diff(x)) for k, (y, x, _) in stairs(fig).items()}
        assert areas == pytest.approx(dict.fromkeys(names, 1.0), rel=1e-9)
        legend = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
        assert legend == names | {"normaliser mean", "sample mean", GAUSS}
    bundle = make_bundle((0,) * 4, pt=(13.5, 43.5, 1.5, 5.5), lab=[0, 1, 0, 1])  # float pt
    cb = run_epoch(TASK(bins=4), pl, [bundle])
    e, st = cb.last_plots["jets.pt"], stairs(fig := cb.last_figures["jets.pt"])
    assert (len(e["edges"]), e["integer_bins"], e["underflow"], e["overflow"]) == (5, False, 0, 0)
    assert sum(e["counts"]) == 4
    areas = {k: np.sum(y * np.diff(x)) for k, (y, x, _) in st.items()}  # "l" empty: JSON only
    assert areas == pytest.approx(dict.fromkeys(("total", "b", "c"), 1.0), rel=1e-9)
    peak = max(s.values.max() for s in st.values())  # dx = 11.55: far below the old 1.0 floor
    assert fig.axes[0].get_ylim()[1] == pytest.approx(1.4 * peak)
    tracks = stairs(cb.last_figures["tracks.d0"])  # an entirely empty stream: no ZeroDivision
    assert [(k, v.any()) for k, (v, *_) in tracks.items()] == [("total", False)]


def test_class_colours_fixed_by_class_names_index(tmp_path):
    pl = stub_pl_module(make_normaliser(tmp_path))  # "c" (the MIDDLE class) empty: "l" keeps 2
    cb = run_epoch(TASK(), pl, [make_bundle(lab=[0, 2, 2, 0])])
    FigureCanvasAgg(fig := cb.last_figures["jets.pt"]).draw()
    colours = {p.get_label(): p.get_edgecolor() for p in fig.axes[0].patches}  # public API
    assert set(colours) == {"total", "b", "l"}  # the empty class is drawn nowhere
    assert (colours["b"], colours["l"]) == (to_rgba(CLASS_COLOURS[0]), to_rgba(CLASS_COLOURS[2]))


def test_integer_variables_get_centred_unit_bins(tmp_path):
    pl, edges = stub_pl_module(make_normaliser(tmp_path)), np.arange(-3.5, 6.0)
    bundle = make_bundle(pt=(0.0, 1.0), eta=(0.0, 5.0, 2.0, 3.0))  # a 0/1 and a 0..5 column
    p = run_epoch(Plotter(), pl, [bundle]).last_plots
    assert p["jets.pt"]["edges"] == pytest.approx([-0.5, 0.5, 1.5])
    assert p["jets.eta"]["edges"] == pytest.approx(np.arange(-0.5, 6.0).tolist())  # 7 edges
    assert p["tracks.d0"]["edges"] == pytest.approx(edges.tolist())
    assert all(p[k]["integer_bins"] for k in ("jets.pt", "jets.eta", "tracks.d0"))
    assert p["jets.pt"]["counts"] == [2, 2]
    later = run_epoch(Plotter(), pl, [bundle, make_bundle(pt=(0.5, 7.0))]).last_plots["jets.pt"]
    assert (later["n_nonintegral"], later["overflow"]) == (2, 2)  # recorded, never discarded
    assert (later["edges"], later["counts"]) == (p["jets.pt"]["edges"], [2, 4])  # 0.5: an edge
    for pt, n_edges in (((3.0,), 2), ((0.0, 50.0), 52), ((1.5, 5.0), 101), ((0.0, 51.0), 101)):
        e = run_epoch(Plotter(), pl, [make_bundle(pt=pt)]).last_plots["jets.pt"]
        assert (len(e["edges"]), e["integer_bins"]) == (n_edges, n_edges < 101)  # range <= 50
        assert e["edges"][0] < min(pt) <= max(pt) < e["edges"][-1]  # finite, nonzero width


def test_log_y_toggle(tmp_path):
    pl = stub_pl_module(make_normaliser(tmp_path))
    assert Plotter().log_y is False
    for log_y in (True, False):
        ax = run_epoch(Plotter(log_y=log_y), pl, [make_bundle()]).last_figures["jets.pt"].axes[0]
        assert ax.get_yscale() == ("log" if log_y else "linear")
        assert ax.get_ylim()[0] > 0 if log_y else ax.get_ylim()[0] == 0
