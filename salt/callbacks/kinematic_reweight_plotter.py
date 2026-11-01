"""`KinematicReweightPlotter` — validation plots of a `KinematicWeight` module's weights.

Reads the weight that module produces in VAL (the key the loss consumes) and the variable
values it publishes, never recomputing either; the variable spec is the module's. Plots:
continuous ``x``, continuous ``x`` split by a categorical (a figure per category), categorical
``x``; figures/metrics go to Comet and/or ``output_dir``. Per-rank accumulation, rank-0 logs.
JSON per figure/class: unit-sum curves, ratios, selected ``n``/``sum_w``/``sum_w2``, ``bin_n``,
``bin_sum_w`` and ``mean_w`` (= ``bin_sum_w / bin_n``, null where a bin is empty), plus per-bin
``bin_sum_w2`` and ±1 sigma own-class stat. errors ``before_err``/``after_err`` (+ ``ratio_*_err``
on continuous figures; shaded bands, bar error bars); class-mean denominator and normalisation
not propagated; empty bins carry 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from salt.callbacks._plot_io import (
    CLASS_COLOURS,
    _candidates_error,
    _require_graph_dict,
    log_figures,
    write_comet_experiment_json,
)
from salt.graph.errors import ConfigError
from salt.model.modules.kinematic_weight import KinematicWeight, VariableSpec

_PREFIX = "KinematicReweightPlotter"
_PLOT_KEYS = frozenset({"x", "split_by", "edges"})
_NO_TARGET = "joint target is not reconstructible from these marginal histograms"
_ERROR_NOTE = (
    "±1σ statistical, own-class only: before sqrt(bin_n)/n_in_range; "  # noqa: RUF001
    "after sqrt(bin_sum_w2)/sum_w_in_range; ratio = err/mean over classes "
    "(denominator and normalisation not propagated); empty bins carry 0"
)
_Plot = tuple[str, VariableSpec, VariableSpec | None, np.ndarray | None]
_Groups = dict[str, tuple[np.ndarray, np.ndarray]]


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(den, np.nan), where=den > 0)


def _ess_ratio(weights: np.ndarray) -> float:
    den = weights.shape[0] * (weights**2).sum()  # Kish (sum w)^2 / (N sum w^2); 0 when empty
    return float(weights.sum() ** 2 / den) if den > 0 else 0.0


def _err(sq: np.ndarray, total: float) -> np.ndarray:
    """``sqrt(sq) / total``; all zeros when ``total`` is 0."""
    return np.sqrt(sq) / total if total > 0 else np.zeros_like(sq)


def _band(ax: Any, bins: np.ndarray, y: np.ndarray, err: np.ndarray, ok: Any, **kw: Any) -> None:
    """Step band ``y ± err`` clipped at 0; zero height where not ``ok`` or not finite.

    ``where`` covers both edges of every ok bin: a post-step run ending at edge ``i`` stops at
    ``bins[i]``, so masking by bin alone would drop each ok bin just before a masked one.
    """
    ok = ok & np.isfinite(y) & np.isfinite(err)
    lo = np.clip(np.nan_to_num(y - err), 0, None)
    hi = np.where(ok, np.nan_to_num(y + err), lo)
    where = np.append(ok, False) | np.append(False, ok)
    lo, hi = np.append(lo, lo[-1]), np.append(hi, hi[-1])
    ax.fill_between(bins, lo, hi, step="post", lw=0, where=where, **kw)


def _unit_norm(x: np.ndarray, bins: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    hist = np.histogram(x, bins=bins, weights=w)[0].astype(np.float64)
    return hist / hist.sum() if hist.sum() > 0 else hist


def _parse_plot(plot: Mapping[str, Any], variables: dict[str, VariableSpec]) -> _Plot:
    name, split_name, given = plot["x"], plot.get("split_by"), plot.get("edges")
    for n in (name, split_name)[: 1 if split_name is None else 2]:
        if n not in variables:
            raise ConfigError(f"{_PREFIX}: plot variable {n!r} is not a weight-module variable")
    x, split = variables[name], None if split_name is None else variables[split_name]
    if split is not None and not split.categorical:
        raise ConfigError(f"{_PREFIX}: split_by {split_name!r} is not a categorical variable")
    if x.categorical and (split is not None or given is not None):
        raise ConfigError(f"{_PREFIX}: categorical x {name!r} takes no split_by/edges")
    if x.categorical:
        return name, x, None, None
    edges = np.asarray(x.finite_edges if given is None else given, dtype=np.float64)
    bad = edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
    if bad or not (np.diff(edges) > 0).all():
        raise ConfigError(f"{_PREFIX}: {name!r} has < 2 finite increasing edges; give plot edges:")
    return (name if split is None else f"{name}_by_{split_name}"), x, split, edges


def _figure(
    bins: np.ndarray,
    groups: _Groups,
    title: str,
    xlab: str,
    cats: Any,
    target: str | None = None,
    w_max: float = 50.0,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Per-group before (--) / after (-) steps + ratio panels (+ black target step), or bars,
    + a ⟨w⟩ = sum_w/n per-bin panel directly under the main panel (log y, symmetric about 1).
    Each group's JSON entry also carries its selected ``n``/``sum_w``/``sum_w2`` + ``bin_n``,
    ``bin_sum_w`` and ``mean_w`` (``w_max`` caps only the panel's y-range). ±1 sigma own-class
    errors (before ``sqrt(bin_n)/n_in``, after ``sqrt(bin_sum_w2)/s_in``, ratio = err/class mean;
    mean and normalisation not propagated; empty bins 0) are shaded step bands, or bar ``yerr``.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ratios = ("before", "after") if cats is None else ()
    hists = {"before": [_unit_norm(x, bins) for x, _ in groups.values()]}
    hists["after"] = [_unit_norm(x, bins, w) for x, w in groups.values()]
    means = {k: np.mean(np.stack(hists[k]), axis=0) for k in ratios}
    rel = {k: [_ratio(h, means[k]) for h in hists[k]] for k in ratios}
    arrays = {**hists, **{f"ratio_{k}": rel[k] for k in ratios}}
    xs, ws = [x for x, _ in groups.values()], [w for _, w in groups.values()]
    counts: dict[str, list[Any]] = {"n": [len(x) for x in xs]}  # selected rows, incl. out-of-range
    counts |= {"sum_w": [float(w.sum()) for w in ws], "sum_w2": [float((w**2).sum()) for w in ws]}
    bin_n = [np.histogram(x, bins=bins)[0] for x in xs]  # raw in-range counts (int)
    hw = [np.histogram(x, bins=bins, weights=w)[0].astype(np.float64) for x, w in groups.values()]
    hw2 = [np.histogram(x, bins, weights=w**2)[0].astype(np.float64) for x, w in groups.values()]
    errs = {"before": [_err(n.astype(np.float64), float(n.sum())) for n in bin_n]}
    errs["after"] = [_err(s2, float(s.sum())) for s2, s in zip(hw2, hw, strict=True)]
    rel_err = {k: [_ratio(e, means[k]) for e in errs[k]] for k in ratios}
    mean_w = [_ratio(s, n.astype(np.float64)) for s, n in zip(hw, bin_n, strict=True)]
    counts["bin_n"] = [a.tolist() for a in bin_n]
    counts |= {"bin_sum_w": [a.tolist() for a in hw], "mean_w": [a.tolist() for a in mean_w]}
    err_cols = {"bin_sum_w2": hw2, "before_err": errs["before"], "after_err": errs["after"]}
    err_cols |= {f"ratio_{k}_err": rel_err[k] for k in ratios}
    cols = {**{k: [a.tolist() for a in v] for k, v in arrays.items()}, **counts}
    cols |= {k: [a.tolist() for a in v] for k, v in err_cols.items()}
    entries = {n: {k: v[i] for k, v in cols.items()} for i, n in enumerate(groups)}
    n_ax, width = 2 + len(ratios), 0.8 / len(groups)
    fig = Figure(figsize=(9, 3 + 2 * n_ax))
    axes = fig.subplots(n_ax, sharex=True, squeeze=False, height_ratios=[3, 1, 1, 1][:n_ax])[:, 0]
    colours = [CLASS_COLOURS[i % len(CLASS_COLOURS)] for i in range(len(groups))]
    curves = zip(colours, hists["before"], hists["after"], mean_w, strict=True)
    for j, (c, b, a, m) in enumerate(curves):
        eb, ea = errs["before"][j], errs["after"][j]
        if cats is None:
            axes[0].stairs(np.where(b > 0, b, np.nan), bins, color=c, ls="--", baseline=None)
            axes[0].stairs(np.where(a > 0, a, np.nan), bins, color=c, baseline=None)
            axes[1].stairs(m, bins, color=c, baseline=None)  # empty (NaN) bins draw nothing
            _band(axes[0], bins, b, eb, b > 0, color=c, alpha=0.12)  # masked like the curves
            _band(axes[0], bins, a, ea, a > 0, color=c, alpha=0.2)
        else:
            left = np.arange(len(cats)) - 0.4 + j * width
            kb, ka = ({"yerr": e, "capsize": 2, "ecolor": c} for e in (eb, ea))
            axes[0].bar(left + width / 4, b, width / 2, ec=c, fill=False, ls="--", hatch="//", **kb)
            axes[0].bar(left + 3 * width / 4, a, width / 2, color=c, **ka)
            axes[1].bar(left + width / 2, m, width, color=c)
    m_all = np.concatenate(mean_w)
    m_all = m_all[np.isfinite(m_all) & (m_all > 0)]  # filter before 1/m; floor 2 beats w_max
    lim = max(2.0, min(float(np.max(np.maximum(m_all, 1 / m_all))), w_max) if m_all.size else 0.0)
    axes[1].axhline(1.0, color="grey", ls=":", lw=1.0)
    axes[1].set(ylabel="⟨w⟩ per bin", yscale="log", ylim=(1.0 / lim, lim))
    handles = [Line2D([], [], color=c, label=n) for n, c in zip(groups, colours, strict=True)]
    styles = (("--", "before (unweighted)"), ("-", "after (weighted)"))
    handles += [Line2D([], [], color="grey", ls=s, label=lbl) for s, lbl in styles]
    handles.append(Patch(facecolor="grey", alpha=0.3, label="±1σ stat."))  # noqa: RUF001
    extra: dict[str, Any] = {"target": None, "target_note": f"{target}: {_NO_TARGET}"}
    extra["error_note"] = _ERROR_NOTE
    if cats is not None:
        extra = {"error_note": _ERROR_NOTE}
    elif target == "inclusive":  # pooled unweighted in-range counts: the shape every class gets
        t = _unit_norm(np.concatenate([x for x, _ in groups.values()]), bins)
        axes[0].stairs(np.where(t > 0, t, np.nan), bins, color="k", lw=1.5, baseline=None)
        handles.append(Line2D([], [], color="k", lw=1.5, label="target (inclusive)"))
        extra = {"target": t.tolist(), "error_note": _ERROR_NOTE}
    axes[0].legend(handles=handles, fontsize="small", ncol=2)
    axes[0].set(ylabel="normalised (unit sum)", title=title)
    axes[0].set_xscale("log" if cats is None and bins[0] > 0 else "linear")
    if cats is not None:
        axes[0].set_xticks(np.arange(len(cats)), labels=cats)
    for ax, key in zip(axes[2:], ratios, strict=True):
        for c, r, e in zip(colours, rel[key], rel_err[key], strict=True):
            ax.stairs(r, bins, color=c, baseline=None)
            _band(ax, bins, r, e, np.ones(r.shape, dtype=bool), color=c, alpha=0.2)
        ax.axhline(1.0, color="grey", ls=":", lw=1.0)
        ax.set(ylabel=f"{key} / mean", ylim=(0, 2))
    for ax in axes[3:]:
        ax.sharey(axes[2])
    axes[-1].set_xlabel(xlab)
    fig.tight_layout()
    return entries, fig, extra


class KinematicReweightPlotter(Callback):
    """Plot a `KinematicWeight` module's produced weights each validation-epoch end.

    The variable spec is the `KinematicWeight` module's (one source of truth); the demand is
    its ``label_key``, ``weight_key`` and the ``published`` key of every plotted variable.
    Runs under ``salt fit`` and ``salt validate``; under ``salt validate`` the title's stage reads
    ``?`` (the schedule stage is not restored off-fit).

    Parameters
    ----------
    plots : Sequence[Mapping[str, Any]]
        Non-empty ``{x, split_by?, edges?}`` list (categorical ``split_by``; ``edges`` override).
        ``split_by`` (e.g. subcampaign, shower): a figure per category, curves = classes + target.
    weight_module : str, optional
        ``model.modules`` key of the `KinematicWeight`, by default ``"kinematic_weight"``.
    output_dir : str | None, optional
        Where PNG/JSON (and ``comet_experiment.json``) are written; by default ``None``.
    figure_name : str, optional
        Comet figure/metric prefix, by default ``"kinematic_reweight"``.

    Raises
    ------
    ConfigError
        Bad plot shape (construction); a plot the module's variables do not support, a module
        without variables, or a non-`KinematicWeight` ``weight_module`` (``fit_val_demand``).
    """

    def __init__(
        self,
        plots: Sequence[Mapping[str, Any]],
        weight_module: str = "kinematic_weight",
        output_dir: str | None = None,
        figure_name: str = "kinematic_reweight",
    ) -> None:
        if not plots:
            raise ConfigError(f"{_PREFIX}: plots must list at least one plot")
        for plot in plots:
            if not isinstance(plot, Mapping) or "x" not in plot or set(plot) - _PLOT_KEYS:
                raise ConfigError(f"{_PREFIX}: plot {plot!r} needs x; keys {sorted(_PLOT_KEYS)}")
        pairs = [(plot["x"], plot.get("split_by")) for plot in plots]
        if len(set(pairs)) != len(pairs):
            raise ConfigError(f"{_PREFIX}: duplicate plot definitions (same x and split_by)")
        self._plot_cfgs = list(plots)
        self._plots: list[_Plot] = []  # parsed against the module's spec in fit_val_demand
        self.used_variables = tuple(dict.fromkeys(n for pair in pairs for n in pair if n))
        self.weight_module, self.figure_name = weight_module, figure_name
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.requires: tuple[str, ...] = ()
        self.last_metrics: dict[str, float] | None = None
        self.last_plots: dict[str, Any] | None = None
        self.last_figures: dict[str, Any] = {}
        self._warned_nowhere = False
        self._reset()

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        _require_graph_dict(model_modules, _PREFIX)
        kw = model_modules.get(self.weight_module)
        if not isinstance(kw, KinematicWeight):
            found = sorted(k for k, v in model_modules.items() if isinstance(v, KinematicWeight))
            msg = f"weight_module {self.weight_module!r} does not name a KinematicWeight module"
            raise _candidates_error(_PREFIX, msg, found)
        if kw.variables is None:
            raise ConfigError(f"{_PREFIX}: {self.weight_module!r} configures no variables to plot")
        self._plots = [_parse_plot(plot, kw.variables) for plot in self._plot_cfgs]
        return (kw.label_key, kw.weight_key, *(kw.published[v] for v in self.used_variables))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        del trainer
        if stage not in {"fit", "validate"}:
            return
        self._reset()
        modules: Any = getattr(pl_module, "_graph_modules", None)
        self.requires = self.fit_val_demand(modules)
        kw = modules[self.weight_module]
        self.stream, self.label, self.weight_field = kw.stream, kw.label, kw.weight_field
        self.w_max, self.n_classes, self.target = float(kw.w_max), int(kw.n_classes), str(kw.target)
        names = list(getattr(modules.get(kw.task_name), "class_names", None) or [])
        ok = len(names) == kw.n_classes
        self.class_names = names if ok else [f"class_{i}" for i in range(kw.n_classes)]

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, *_: Any, **__: Any
    ) -> None:
        del pl_module
        if getattr(trainer, "sanity_checking", False):
            return
        outputs_map: Any = outputs
        cols = [outputs_map["bundle"].get(k).detach().cpu().reshape(-1) for k in self.requires]
        cols[:2] = [cols[0].long(), cols[1].float()]
        valid = (cols[0] >= 0) & (cols[0] < self.n_classes)
        self._n_ignored += int((~valid).sum())
        stores = [self._labels, self._weights, *(self._values[n] for n in self.used_variables)]
        for store, col in zip(stores, cols, strict=True):
            store.append(col[valid])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False) or not sum(t.numel() for t in self._labels):
            self._reset()
            return
        label, w = torch.cat(self._labels).numpy(), torch.cat(self._weights).double().numpy()
        # published values are already scaled/abs'd; categorical codes stay exact in float64
        values = {n: torch.cat(c).double().numpy() for n, c in self._values.items()}
        masks = {n: label == i for i, n in enumerate(self.class_names)}
        metrics = {"n_jets": len(label), "n_ignored": self._n_ignored, "mean_w": float(w.mean())}
        metrics |= {"ess_ratio": _ess_ratio(w), "clip_fraction": float(np.mean(w >= self.w_max))}
        metrics |= {f"mean_w_{n}": float(w[m].sum() / max(m.sum(), 1)) for n, m in masks.items()}
        metrics |= {f"ess_{n}": _ess_ratio(w[m]) * int(m.sum()) for n, m in masks.items()}
        controller: Any = getattr(pl_module, "training_controller", None)
        try:
            stage_name = controller.schedule.stages[controller.current_stage_index].name
        except (AttributeError, IndexError, TypeError):
            stage_name = "?"
        fn = getattr(getattr(trainer, "state", None), "fn", None)
        if fn is not None and fn != TrainerFn.FITTING:  # off-fit the stage is not restored
            stage_name = "?"
        step, epoch = getattr(trainer, "global_step", 0), getattr(trainer, "current_epoch", 0)
        info = f"step {step}, epoch {epoch}, stage {stage_name}, N={len(label)} val jets"
        plots: dict[str, Any] = {}
        figures: dict[str, Any] = {}
        for key, x_spec, split, edges in self._plots:
            x, xlab = values[x_spec.name], x_spec.name
            if edges is None:
                cats = list(x_spec.categories or ())
                bins = np.arange(len(cats) + 1) - 0.5
                groups = {n: (x[m], w[m]) for n, m in masks.items()}
                args_c = (bins, groups, f"{key}: {info}", xlab, cats)
                hists, figures[key], extra = _figure(*args_c, w_max=self.w_max)
                plots[key] = {"categories": cats, "classes": hists, **extra}
                continue
            metrics[f"frac_outside_{key}"] = 1 - float(((x >= edges[0]) & (x <= edges[-1])).mean())
            views = [("", "", np.ones(len(x), dtype=bool))]  # (key suffix, title tag, selection)
            if split is not None:  # a figure per category; its curves are the classes
                cats, codes = list(split.categories or ()), values[split.name]
                views = [(f"_{c}", f" [{c}]", codes == k) for k, c in enumerate(cats)]
            for suffix, tag, sel in views:
                if not sel.any():
                    continue  # a category with no validation jets gets no figure / plots entry
                sub = {n: (x[m & sel], w[m & sel]) for n, m in masks.items()}
                fkey, title = f"{key}{suffix}", f"{key}{tag}: {info}"
                args = (edges, sub, title, xlab, None, self.target)
                hists, figures[fkey], extra = _figure(*args, w_max=self.w_max)
                plots[fkey] = {"edges": edges.tolist(), "classes": hists, **extra}
        if getattr(trainer, "is_global_zero", True):
            payload: dict[str, Any] = {**metrics, "class_names": self.class_names, "plots": plots}
            kw: dict[str, Any] = {"prefix": self.figure_name, "output_dir": self.output_dir}
            kw |= {"who": _PREFIX, "warned": self._warned_nowhere}
            self._warned_nowhere = log_figures(trainer, figures, metrics, payload, **kw)
        self.last_metrics, self.last_plots, self.last_figures = metrics, plots, figures
        self._reset()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        write_comet_experiment_json(trainer, self.output_dir)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write the Comet receipt under ``salt validate`` too (rank-0, idempotent rewrite)."""
        del pl_module
        write_comet_experiment_json(trainer, self.output_dir)

    def _reset(self) -> None:
        self._labels: list[torch.Tensor] = []
        self._weights: list[torch.Tensor] = []
        self._values: dict[str, list[torch.Tensor]] = {v: [] for v in self.used_variables}
        self._n_ignored = 0
