"""`InputNormalisationPlotter` — raw input histograms with a normaliser's standardised top axis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from salt.callbacks._plot_io import (
    CLASS_COLOURS,
    _candidates_error,
    _require_graph_dict,
    log_figures,
    write_comet_experiment_json,
)
from salt.graph.errors import ConfigError
from salt.model.modules.norm import MaskedInputNormaliser, Normaliser
from salt.model.modules.tasks import ClassificationTaskModule

_PREFIX, _FIG = "InputNormalisationPlotter", "input_norm"  # _FIG: Comet figure/metric prefix
_NORMS = (Normaliser, MaskedInputNormaliser)
_MOMENTS = ("mean_raw", "std_raw", "mean_std", "std_std")


class _Acc:  # one stream: per-field normaliser (mean, std); per curve (0 = total, 1..k = classes)
    def __init__(self, fields, m, s, warm, bins, k) -> None:  # ragged histograms, Chan moments
        self.fields, self.m, self.s, self.warm, self.bins = fields, m, s, warm, bins
        self.integer, self.nonint = np.zeros(len(fields), bool), np.zeros(len(fields), np.int64)
        self.k, self.unlabelled, self.pending, self.n = k, 0, True, np.zeros(k + 1, np.int64)
        self.under, self.over, self.mean, self.m2 = np.zeros((4, k + 1, len(fields)))
        self._bin(m, m, s)  # no valid data yet: the normaliser's mean +/- 1 std

    def _bin(self, lo, hi, pad) -> None:  # integer column: a unit bin per k in [lo, hi] (k +/- 0.5)
        pad, nb = np.where(self.integer, 0.5, pad), np.where(self.integer, hi - lo + 1, self.bins)
        self.lo, self.hi = lo, hi = lo - pad, hi + pad  # == each field's first/last edge
        self.edges = [np.linspace(a, b, int(n) + 1) for a, b, n in zip(lo, hi, nb, strict=True)]
        self.counts = [np.zeros((self.k + 1, int(n)), np.int64) for n in nb]

    def add(self, x: np.ndarray, lab: np.ndarray) -> None:  # lab: class index per row of x
        if self.pending and len(x):  # data range of the first non-empty batch, widened 5 %
            lo, hi = x.min(0), x.max(0)  # integer-valued with range <= 50: unpadded unit bins
            self.integer, self.pending = (x == np.round(x)).all(0) & (hi - lo <= 50), False
            self._bin(lo, hi, 0.05 * np.where(hi > lo, hi - lo, np.maximum(np.abs(lo), self.s)))
        self.nonint += ((x != np.round(x)) & self.integer).sum(0)  # recorded, still histogrammed
        self.unlabelled += int(((lab < 0) | (lab >= self.k)).sum()) if self.k else 0
        for i, rows in enumerate((x, *(x[lab == c] for c in range(self.k)))):  # one update path
            if not len(rows):
                continue
            for j, col in enumerate(rows.T):  # last bin closed: counts + under + over == n exactly
                self.counts[j][i] += np.histogram(col, self.edges[j])[0]
            self.under[i] += (rows < self.lo).sum(0)
            self.over[i] += (rows > self.hi).sum(0)
            nb, mb = len(rows), rows.mean(0)
            delta, nn = mb - self.mean[i], self.n[i] + nb
            self.mean[i] += delta * nb / nn
            self.m2[i] += ((rows - mb) ** 2).sum(0) + delta**2 * self.n[i] * nb / nn
            self.n[i] = nn


def _stats(a: _Acc, i: int, j: int) -> dict[str, Any]:
    """Curve ``i`` (0 = total) of field ``j``: counts, tails and population (ddof=0) moments."""
    n, mu, sd = int(a.n[i]), a.mean[i, j], math.sqrt(a.m2[i, j] / max(a.n[i], 1))
    cols = (mu, sd, (mu - a.m[j]) / a.s[j], sd / a.s[j])  # in _MOMENTS order
    tails = {"underflow": int(a.under[i, j]), "overflow": int(a.over[i, j])}
    moments = {k: float(c) if n else None for k, c in zip(_MOMENTS, cols, strict=True)}
    return {"counts": a.counts[j][i].tolist(), "n": n, **tails, **moments}


def _figure(key: str, e: dict[str, Any], info: str, gaussian: bool, log_y: bool) -> Any:
    """Unit-area densities (total filled, class lines); top axis ``(x - mean)/std``; stats box."""
    from matplotlib.figure import Figure

    n, edges = e["n"], np.asarray(e["edges"])
    dx, curves = np.diff(edges), {"total": e} | e["classes"]  # classes in class_names order
    m, s = e["norm_mean"], e["norm_std"]  # locals: every figure keeps its own affine map
    ax = (fig := Figure(figsize=(8, 5))).subplots()
    for i, (name, c) in enumerate(curves.items()):  # n includes the tails: in-range area <= 1
        if i and not c["n"]:  # empty class: JSON only, but it keeps its CLASS_COLOURS slot
            continue
        colour = CLASS_COLOURS[(i - 1) % len(CLASS_COLOURS)]  # fixed by class_names position
        kw = {"color": colour, "lw": 1.2} if i else {"fill": True, "alpha": 0.5, "color": "0.6"}
        ax.stairs(np.asarray(c["counts"]) / (max(c["n"], 1) * dx), edges, label=name, **kw)
    top = ax.secondary_xaxis("top", functions=(lambda x: (x - m) / s, lambda z: z * s + m))
    top.set_xlabel("standardised (x - mean)/std")
    ax.axvline(m, color="k", ls=":", label="normaliser mean")
    text = "no valid data"
    if n:
        ax.axvline(e["mean_raw"], color="k", ls="--", label="sample mean")
        rows = (("raw", "mean_raw", "std_raw"), ("standardised", "mean_std", "std_std"))
        text = "\n".join(f"{t}: mean={e[a]:.4g} std={e[b]:.4g}" for t, a, b in rows)
        text += f"\nN={n}  out-of-range={100 * (e['underflow'] + e['overflow']) / n:.2f}%"
    if gaussian and n:
        xs = np.linspace(edges[0], edges[-1], 4 * len(dx) + 1)
        ys = np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        ax.plot(xs, ys, color="C3", lw=1.2, label="N(0,1) in standardised units")
    style = {"family": "monospace", "size": "small", "bbox": {"facecolor": "w", "alpha": 0.8}}
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", **style)
    unwarmed = "" if e["stats_warm"] else " — normaliser stats: initial buffers (unwarmed)"
    ax.set(xlabel=f"{key} (raw)", ylabel="density (unit area per curve)", xlim=edges[[0, -1]])
    ax.set(title=f"{key}: {info}{unwarmed}", yscale="log" if log_y else "linear")
    ymax = ax.dataLim.y1 or 1.0  # headroom over the tallest curve; log: a positive floor
    ax.set_ylim((1e-5 * ymax, 4 * ymax) if log_y else (0, 1.4 * ymax))
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


class InputNormalisationPlotter(Callback):
    """Plot each raw input variable against its normaliser's standardised scale per val epoch.

    One unweighted unit-area density per ``(stream, variable)`` of raw ``inputs.<stream>`` in raw
    units, total + one line per ``class_task`` class (``masks`` drop padded objects; none on the
    ``global_object``), ``(x - mean)/std`` from a `Normaliser`/`MaskedInputNormaliser`'s own
    buffers at plot time on a top ``secondary_xaxis``; the x-range is the first non-empty
    validation batch's range widened 5 % (tails: ``underflow``/``overflow`` share); the total's
    sample mean/std in both, opt-in N(0,1). Rank 0 logs its rank only.

    Parameters
    ----------
    normaliser : str, optional
        ``model.modules`` key of the normaliser, by default ``"norm"``.
    bins : int, optional
        Bins for a float variable; an integer-valued variable with range <= 50 gets one bin per
        integer, by default 100.
    max_batches : int | None, optional
        Validation batches accumulated per epoch; by default ``None`` (all).
    gaussian : bool, optional
        Overlay the N(0,1) pdf in standardised units; by default False.
    log_y : bool, optional
        Log y-axis (density), by default False.
    output_dir : str | None, optional
        Where PNG/JSON (and ``comet_experiment.json``) go; by default ``None``.
    class_task : str | None, optional
        ``model.modules`` key of a `ClassificationTaskModule` on the normaliser's
        ``global_object`` stream; one extra unit-area curve per class, by default None (total only).

    Raises
    ------
    ConfigError
        Bad ``bins``/``max_batches``/``gaussian``/``log_y``, or bad ``normaliser``/``class_task``.
    """

    def __init__(
        self,
        normaliser: str = "norm",
        bins: int = 100,
        max_batches: int | None = None,
        gaussian: bool = False,
        log_y: bool = False,
        output_dir: str | None = None,
        class_task: str | None = None,
    ) -> None:
        ints = {"bins": bins, "max_batches": 1 if max_batches is None else max_batches}
        if bad := [k for k, v in ints.items() if type(v) is not int or v < 1]:  # bool excluded
            raise ConfigError(f"{_PREFIX}: {bad} must be an int > 0 (max_batches may be None)")
        if any(type(v) is not bool for v in (gaussian, log_y)):
            raise ConfigError(f"{_PREFIX}: gaussian/log_y must be bool")
        self.normaliser, self.bins, self.log_y = normaliser, bins, log_y
        self.max_batches, self.gaussian, self.class_task = max_batches, gaussian, class_task
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.requires: tuple[str, ...] = ()
        self.last_plots: dict[str, Any] | None = None
        self.last_figures: dict[str, Any] = {}
        self._warned_nowhere = False
        self._norm: Any = None
        self._label: str | None = None
        self._class_names: tuple[str, ...] = ()
        self._reset()

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        _require_graph_dict(model_modules, _PREFIX)
        if not isinstance(norm := model_modules.get(self.normaliser), _NORMS):
            found = sorted(k for k, v in model_modules.items() if isinstance(v, _NORMS))
            msg = f"normaliser {self.normaliser!r} names no Normaliser/MaskedInputNormaliser module"
            raise _candidates_error(_PREFIX, msg, found)
        seq, g = [s for s in norm.streams if s != norm.global_object], norm.global_object
        demand = (*[f"inputs.{s}" for s in norm.streams], *[f"masks.{s}" for s in seq])
        tasks = {k: v for k, v in model_modules.items() if isinstance(v, ClassificationTaskModule)}
        found = sorted(k for k, v in tasks.items() if v.stream == g and v.label_map is None)
        if self.class_task is not None and self.class_task not in found:
            msg = f"class_task {self.class_task!r} names no ClassificationTaskModule on stream "
            raise _candidates_error(_PREFIX, f"{msg}{g!r} without label_map", found)
        return demand if self.class_task is None else (*demand, tasks[self.class_task].label_key)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        del trainer
        if stage != "fit":
            return
        self._reset()
        modules: Any = getattr(pl_module, "_graph_modules", None)
        self.requires, self._norm = self.fit_val_demand(modules), modules[self.normaliser]
        if (task := modules.get(self.class_task)) is not None:  # class names only from the task
            self._label, self._class_names = task.label_key, tuple(task.class_names)

    def _reset(self) -> None:
        self._acc: dict[str, _Acc] | None = None
        self._n_batches = 0

    def _first_batch(self, pl_module: LightningModule) -> dict[str, _Acc]:
        """Field names + the stats the normaliser's forward applies, once per epoch."""
        if (schema := getattr(pl_module, "schema", None)) is None:  # never read in `setup`
            raise RuntimeError(f"{_PREFIX}: pl_module.schema is None at the first validation batch")
        acc: dict[str, _Acc] = {}
        for stream in (norm := self._norm).streams:
            fields = tuple(schema.fields_of(f"inputs.{stream}"))
            if isinstance(norm, Normaliser):
                m, s, warm = getattr(norm, f"means_{stream}"), getattr(norm, f"stds_{stream}"), True
            else:  # MaskedInputNormaliser divides by sqrt(running_var + eps)
                warm = int(getattr(norm, f"num_batches_tracked_{stream}")) > 0
                m = getattr(norm, f"running_mean_{stream}")
                s = (getattr(norm, f"running_var_{stream}").double() + norm.eps).sqrt()
            m, s = (t.detach().cpu().double().numpy() for t in (m, s))
            if not (ok := np.isfinite(m) & np.isfinite(s) & (s > 0)).all():
                bad = f"{stream}.{fields[int(np.argmin(ok))]}"
                raise RuntimeError(f"{_PREFIX}: {bad} has a non-finite or non-positive scale")
            acc[stream] = _Acc(fields, m, s, warm, self.bins, len(self._class_names))
        return acc

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, *_: Any, **__: Any
    ) -> None:
        full = self.max_batches is not None and self._n_batches >= self.max_batches
        if getattr(trainer, "sanity_checking", False) or full:
            return
        bundle = cast("dict[str, Any]", outputs)["bundle"]
        acc = self._acc = self._first_batch(pl_module) if self._acc is None else self._acc
        lab = bundle.get(self._label).detach().cpu().long() if self._label else None
        for stream, a in acc.items():
            x, mask = bundle.get(f"inputs.{stream}").detach().cpu(), f"masks.{stream}"
            lb = x.new_full(x.shape[:1], -1) if lab is None else lab  # -1: no class task
            lb = lb.reshape(-1, *[1] * (x.dim() - 2)).expand(x.shape[:-1])  # parent label over T
            if stream != self._norm.global_object and mask in bundle:  # True == padded
                keep = ~bundle.get(mask).detach().cpu()  # as MaskedInputNormaliser
                x, lb = x[keep], lb[keep]
            flat = x.reshape(-1, x.shape[-1]).double().numpy()
            if not (finite := np.isfinite(flat).all(axis=0)).all():
                var = a.fields[int(np.argmin(finite))]
                raise RuntimeError(f"{_PREFIX}: non-finite valid values in inputs.{stream}.{var}")
            a.add(flat, lb.reshape(-1).numpy())
        self._n_batches += 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if getattr(trainer, "sanity_checking", False) or self._acc is None:
            self._reset()
            return
        step, epoch = getattr(trainer, "global_step", 0), getattr(trainer, "current_epoch", 0)
        cls = type(self._norm).__name__
        plots: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        for stream, a in self._acc.items():
            for j, var in enumerate(a.fields):
                key, e = f"{stream}.{var}", {"edges": a.edges[j].tolist(), **_stats(a, 0, j)}
                e |= {"norm_mean": float(a.m[j]), "norm_std": float(a.s[j]), "stats_warm": a.warm}
                e |= {"n_unlabelled": a.unlabelled, "integer_bins": bool(a.integer[j])}
                e["n_nonintegral"] = int(a.nonint[j])  # later batches' values in an integer column
                e["classes"] = {c: _stats(a, i, j) for i, c in enumerate(self._class_names, 1)}
                ok = [k for k in _MOMENTS[2:] if e["n"] and math.isfinite(e[k])]
                plots[key], metrics = e, metrics | {f"{key}/{k}": e[k] for k in ok}
        info = f"step {step}, epoch {epoch}, {cls}"
        figures = {k: _figure(k, e, info, self.gaussian, self.log_y) for k, e in plots.items()}
        data = {"step": step, "epoch": epoch, "normaliser": self.normaliser}
        data |= {"normaliser_class": cls, "n_batches": self._n_batches, "plots": plots}
        data |= {"class_task": self.class_task, "class_names": list(self._class_names)}
        if getattr(trainer, "is_global_zero", True):
            w, out = self._warned_nowhere, self.output_dir
            self._warned_nowhere = log_figures(
                trainer, figures, metrics, data, prefix=_FIG, output_dir=out, who=_PREFIX, warned=w
            )
        self.last_plots, self.last_figures = plots, figures
        self._reset()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        write_comet_experiment_json(trainer, self.output_dir)
