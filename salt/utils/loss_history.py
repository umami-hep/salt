"""Per-epoch loss history as a plain CSV — a stack-neutral measurement instrument.

Snapshots ``trainer.callback_metrics`` each epoch into ``<log_dir>/loss_history.csv``
(columns: ``stage, epoch, step, metric, value``).
"""

from __future__ import annotations

import csv
from pathlib import Path

from lightning import Callback, LightningModule, Trainer

__all__ = ["LossHistoryWriter"]

_FIELDS = ("stage", "epoch", "step", "metric", "value")


class LossHistoryWriter(Callback):
    """Append per-epoch ``train/*`` / ``val/*`` metrics to a CSV file.

    Parameters
    ----------
    fname : str, optional
        File name created inside the trainer log dir (falls back to
        ``trainer.default_root_dir``), by default ``"loss_history.csv"``.
    """

    def __init__(self, fname: str = "loss_history.csv") -> None:
        self.fname = fname

    def _path(self, trainer: Trainer) -> Path:
        out_dir = Path(trainer.log_dir or trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / self.fname

    def _write(self, trainer: Trainer, stage: str) -> None:
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        rows = []
        for name, value in trainer.callback_metrics.items():
            if not name.startswith(f"{stage}/"):
                continue
            rows.append({
                "stage": stage,
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "metric": name,
                "value": float(value),
            })
        if not rows:
            return
        path = self._path(trainer)
        new_file = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record the aggregated ``val/*`` metrics for this epoch."""
        del pl_module
        self._write(trainer, "val")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record the ``train/*`` metrics as of the last step of this epoch."""
        del pl_module
        self._write(trainer, "train")
