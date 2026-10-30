"""`PriorClosureMetrics` — per-iteration AUC / CE of the nn `KinematicWeight` prior heads."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from scipy.stats import entropy  # H of (unnormalised) class masses, 0 log 0 = 0

from salt.callbacks._plot_io import CLASS_COLOURS, log_figures, write_comet_experiment_json
from salt.graph.errors import ConfigError
from salt.model.modules.kinematic_weight import KinematicWeight

_PREFIX = "PriorClosureMetrics"


def _auc(s: np.ndarray, pos: np.ndarray, w: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    """Weighted one-vs-rest AUC (ties count 1/2) via the sorted cumulative negative mass."""
    n_pos, n_neg, den = int(pos.sum()), int((~pos).sum()), float(w[pos].sum() * w[~pos].sum())
    bad = "no_positives" if not n_pos else "no_negatives" if not n_neg else "zero_weight"
    support = {"n_pos": n_pos, "n_neg": n_neg, "status": "ok" if den > 0 else bad}
    if den <= 0:
        return None, support
    u, inv = np.unique(s, return_inverse=True)
    neg_at = np.bincount(inv, weights=w * ~pos, minlength=len(u))
    below = (np.cumsum(neg_at) - neg_at)[inv][pos] + 0.5 * neg_at[inv][pos]
    return float((w[pos] * below).sum() / den), support


def _figure(heads: dict[str, dict[str, Any]], class_names: list[str], ce_floor: float) -> Any:
    from matplotlib.figure import Figure

    rows, it = list(heads.values()), np.arange(1, len(heads) + 1)
    top, bot = (fig := Figure(figsize=(8, 6), layout="constrained")).subplots(2, sharex=True)
    lines = [[r["auc"][n] for r in rows] for n in class_names] + [[r["auc_macro"] for r in rows]]
    colours = [CLASS_COLOURS[c % len(CLASS_COLOURS)] for c in range(len(class_names))] + ["k"]
    for vals, c, lbl in zip(lines, colours, [*class_names, "macro"], strict=True):
        top.plot(it, [np.nan if v is None else v for v in vals], "o-", color=c, label=lbl)
    top.axhline(0.5, color="grey", ls=":", lw=1.0)
    top.set(ylabel="weighted one-vs-rest AUC", title="prior heads (closure: AUC -> 0.5)")
    bot.plot(it, [r["ce"] for r in rows], "o-", color="k", label="CE (weighted)")
    bot.axhline(ce_floor, color="grey", ls="--", lw=1.0, label="ce_floor = H(q_val)")
    bot.set(xlabel="iteration", ylabel="cross-entropy", xticks=it)
    fig.legend(fontsize="small", loc="outside right upper")  # collects both panels' labels
    return fig


class PriorClosureMetrics(Callback):
    """Per prior head: weighted AUC / CE of its recovered posterior on validation jets.

    Head ``k`` is weighted by the detached cumulative weight ``C_{k-1}`` it trained on (last
    ``chain`` field, 1 for head 1; no class-balancing weights). Closure: AUC -> 0.5, CE ->
    ``H(q_w)``; but an UNTRAINED constant head also gives AUC 0.5, and ``ce_const_q == ce_floor``
    only if the weighted marginal is ``q_f``: head diagnostics; closure evidence is the plots.
    Also runs under ``salt validate`` (one checkpoint-restored validation pass).

    Parameters
    ----------
    weight_modules : Sequence[str] | None, optional
        `KinematicWeight` names in iteration order; default every nn one, by ``_<k>`` suffix.
    output_dir, figure_name : str | None, str, optional
        PNG/``step*.json`` directory (default ``None``); Comet prefix (``"prior_closure"``).

    Raises
    ------
    ConfigError
        No module dict / named non-`KinematicWeight` / no head / mixed label keys or classes.
    """

    def __init__(
        self,
        weight_modules: Sequence[str] | None = None,
        output_dir: str | None = None,
        figure_name: str = "prior_closure",
    ) -> None:
        self.weight_modules = None if weight_modules is None else list(weight_modules)
        self.output_dir, self.figure_name = Path(output_dir) if output_dir else None, figure_name
        self.label_key, self.n_classes, self._warned_nowhere = "", 0, False
        self._heads: list[tuple[str, str, str | None]] = []  # (name, task, incoming weight key)
        self._mods: list[KinematicWeight] = []
        self.requires: tuple[str, ...] = ()
        self.class_names: list[str] = []
        self._cols: list[dict[str, torch.Tensor]] = []  # per val batch: requires -> owned CPU copy
        self.last_metrics: dict[str, Any] | None = None
        self.last_payload: dict[str, Any] | None = None
        self.last_figures: dict[str, Any] = {}

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        if not isinstance(model_modules, dict):
            raise ConfigError(f"{_PREFIX} needs a LightningModule with a graph-module dict")
        kws = {k: v for k, v in model_modules.items() if isinstance(v, KinematicWeight)}
        nn = [n for n, v in kws.items() if v.mode == "nn"]
        nn.sort(key=lambda n: int(t) if (t := n.rpartition("_")[2]).isdigit() else np.inf)
        names = nn if self.weight_modules is None else self.weight_modules
        if not names or any(n not in kws for n in names):
            raise ConfigError(f"{_PREFIX}: {names} must name nn KinematicWeights of {list(kws)}")
        mods = [kws[n] for n in names]
        if len({m.label_key for m in mods}) != 1 or len({m.n_classes for m in mods}) != 1:
            raise ConfigError(f"{_PREFIX}: {names} disagree on label key / n_classes")
        inc = [f"labels.{m.stream}.{m.chain[-1]}" if m.chain else None for m in mods]
        self._heads = [(n, m.task_name, k) for n, m, k in zip(names, mods, inc, strict=True)]
        self._mods, self.label_key, self.n_classes = mods, mods[0].label_key, mods[0].n_classes
        preds = [f"preds.{m.stream}.{m.task_name}" for m in mods]
        return tuple(dict.fromkeys((self.label_key, *preds, *(k for k in inc if k))))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        del trainer
        if stage in {"fit", "validate"}:
            modules: Any = getattr(pl_module, "_graph_modules", None)
            self.requires, self._cols, k = self.fit_val_demand(modules), [], self.n_classes
            names = list(getattr(modules.get(self._heads[0][1]), "class_names", None) or [])
            self.class_names = names if len(names) == k else [f"class_{i}" for i in range(k)]

    def on_validation_batch_end(self, trainer: Trainer, *args: Any, **__: Any) -> None:
        if not getattr(trainer, "sanity_checking", False):  # args = (pl_module, outputs, ...)
            b = args[1]["bundle"]
            cols = {key: b.get(key).detach().to("cpu", copy=True) for key in self.requires}
            self._cols.append(cols)

    def on_validation_epoch_end(self, trainer: Trainer, *_: Any) -> None:
        cols, self._cols, k, names = self._cols, [], self.n_classes, self.class_names
        y = torch.cat([c[self.label_key].reshape(-1) for c in cols] or [torch.zeros(0)]).long()
        if getattr(trainer, "sanity_checking", False) or not (valid := (y >= 0) & (y < k)).any():
            return
        y, t = y[valid].numpy(), {key: torch.cat([c[key] for c in cols]) for key in self.requires}
        n, ce_floor = len(y), float(entropy(np.bincount(y, minlength=k)))  # unweighted H(q_val)
        metrics: dict[str, Any] = {"ce_floor": ce_floor, "n_jets": n}
        heads: dict[str, dict[str, Any]] = {}
        for i, ((name, task, wkey), mod) in enumerate(zip(self._heads, self._mods, strict=True)):
            p = torch.softmax(t[f"preds.{mod.stream}.{task}"].double().reshape(-1, k)[valid], -1)
            if mod.balanced_prior:  # recovered P(f|x) = q_f P_bal / sum_g q_g P_bal (live q_f)
                p = p * mod.q_f.detach().double().cpu()
                p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
            w = np.ones(n) if wkey is None else t[wkey].double().reshape(-1)[valid].numpy()
            q_w, p = np.bincount(y, weights=w, minlength=k) / (sw := float(w.sum())), p.numpy()
            cq = float(-(q_w * np.log(np.clip(mod.q_f.double().cpu().numpy(), 1e-12, None))).sum())
            aucs = [_auc(p[:, c], y == c, w) for c in range(k)]
            d = [a for a, _ in aucs if a is not None]  # macro = mean of the defined classes
            ce = float(-(w * np.log(np.clip(p[np.arange(n), y], 1e-12, None))).sum() / sw)
            h: dict[str, Any] = {"iteration": i + 1, "task": task, "n": n, "sum_w": sw, "ce": ce}
            h |= {"weight_field": (mod.chain or (None,))[-1], "ce_floor": float(entropy(q_w))}
            h |= {"ce_const_q": cq, "auc_macro": float(np.mean(d)) if d else None}
            h["auc"] = {c: a for c, (a, _) in zip(names, aucs, strict=True)}
            h["auc_support"] = {c: s for c, (_, s) in zip(names, aucs, strict=True)}
            metrics |= {f"{name}/{m}": h[m] for m in ("ce", "ce_floor", "ce_const_q", "auc_macro")}
            metrics |= {f"{name}/auc_{c}": a for c, a in h["auc"].items()}
            heads[name] = h
        metrics = {key: v for key, v in metrics.items() if v is not None}
        payload = {"class_names": names, "n_jets": n, "ce_floor": ce_floor, "heads": heads}
        figures = {"auc_ce_vs_iteration": _figure(heads, names, ce_floor)}
        if getattr(trainer, "is_global_zero", True):
            kw: dict[str, Any] = {"prefix": self.figure_name, "output_dir": self.output_dir}
            kw |= {"who": _PREFIX, "warned": self._warned_nowhere}
            self._warned_nowhere = log_figures(trainer, figures, metrics, payload, **kw)
        self.last_metrics, self.last_payload, self.last_figures = metrics, payload, figures

    def on_fit_start(self, trainer: Trainer, *_: Any) -> None:
        write_comet_experiment_json(trainer, self.output_dir)

    def on_validation_start(self, trainer: Trainer, *_: Any) -> None:
        """`on_fit_start` never fires under ``salt validate``; rank-0, idempotent."""
        write_comet_experiment_json(trainer, self.output_dir)
