"""Shared Comet/disk/warn-once figure I/O for the plotter callbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from lightning import Trainer
from lightning.pytorch.loggers.comet import CometLogger

from salt.utils.logging import get_logger

CLASS_COLOURS = tuple(f"C{i}" for i in range(10))  # tab10, shared by both plotter callbacks


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a Comet API change must never crash the fit
        return None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return None if isinstance(obj, float) and not np.isfinite(obj) else obj


def log_figures(
    trainer, figures, metrics, payload, *, prefix: str, output_dir: Path | None, who, warned: bool
) -> bool:
    """Comet figures/metrics; PNG + ``step{N}.json`` in `output_dir`; else warn once (returned).

    A PNG name swaps ``.`` in the figure key for ``_``; the Comet name keeps the key.
    """
    step, logger = int(getattr(trainer, "global_step", 0)), getattr(trainer, "logger", None)
    if isinstance(logger, CometLogger):
        exp = logger.experiment
        for key, fig in figures.items():
            exp.log_figure(figure_name=f"{prefix}/{key}/step_{step:06d}", figure=fig, step=step)
        exp.log_metrics({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, fig in figures.items():
            fig.savefig(output_dir / f"{key.replace('.', '_')}_step{step:06d}.png", dpi=130)
        with (output_dir / f"step{step:06d}.json").open("w") as f:
            json.dump(_json_safe(payload), f, indent=2)
    elif not isinstance(logger, CometLogger) and not warned:
        warned = True
        msg = "figures computed but nowhere to log them (no CometLogger, no output_dir)"
        get_logger(__name__).warning(f"{who}: {msg}")
    return warned


def write_comet_experiment_json(trainer: Trainer, output_dir: Path | None) -> None:
    """Write the attached Comet experiment's identity to ``comet_experiment.json`` (rank 0)."""
    rank0 = getattr(trainer, "is_global_zero", True)
    if output_dir is None or not rank0 or not isinstance(trainer.logger, CometLogger):
        return
    exp = trainer.logger.experiment
    info = {a: _safe(lambda a=a: getattr(exp, a)) for a in ("url", "project_name", "workspace")}
    info |= {"key": _safe(exp.get_key), "name": _safe(exp.get_name)}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comet_experiment.json").open("w") as f:
        json.dump(info, f, indent=2)
