"""`Checkpoint` — the per-epoch ``ckpts/`` checkpoint callback."""

from __future__ import annotations

from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from salt.graph.errors import ConfigError


class Checkpoint(ModelCheckpoint):
    """Save a checkpoint per epoch under ``ckpts/`` with the ``loss=`` filename stem.

    Filename and directory are a contract: ``salt test`` without
    ``--ckpt_path`` resolves the best epoch by globbing
    ``<config dir>/{ckpts,checkpoints}/*.ckpt`` and parsing the smallest
    ``loss=<value>`` out of each name (`salt.main._best_checkpoint`).
    Keep the filename stem and the glob in sync.

    Parameters
    ----------
    monitor_loss : str
        The metric key to monitor and embed in the filename, e.g.
        ``val/jets_classification_loss`` or ``val/loss``. Must exist in
        ``trainer.callback_metrics`` at checkpoint time.
    fname_string : str, optional
        The filename loss tag, by default ``"loss"`` — the ``loss=`` stem the
        best-epoch glob keys on.
    mode : str, optional
        ``min``/``max`` selection direction passed to `ModelCheckpoint`, by
        default ``"min"``.
    save_top_k : int, optional
        How many checkpoints to keep, by default ``-1`` (every epoch).
    dirname : str, optional
        The log-dir sub-directory checkpoints land in, by default ``"ckpts"``.
    """

    def __init__(
        self,
        monitor_loss: str = "val/loss",
        fname_string: str = "loss",
        mode: str = "min",
        save_top_k: int = -1,
        dirname: str = "ckpts",
    ) -> None:
        filename = "epoch={epoch:03d}-" + fname_string + "={" + monitor_loss + ":.5f}"
        super().__init__(
            monitor=monitor_loss,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
            auto_insert_metric_name=False,
        )
        self.dirname = dirname

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Fix the checkpoint dir to ``<log_dir>/<dirname>`` on a real fit; raises
        `ConfigError` for an unsupported ``s3://`` log dir.
        """
        if stage == "fit" and not trainer.fast_dev_run:
            log_dir = trainer.log_dir or trainer.default_root_dir
            if log_dir is not None and str(log_dir).startswith(("s3://", "s3:/")):
                raise ConfigError(
                    "salt.callbacks.Checkpoint does not support s3:// log dirs yet "
                    "(rides with the Comet/run-dir wiring); use a local trainer.log_dir "
                    "(v1 checkpoint.py:34-43 s3 branch deferred)"
                )
            self.dirpath = str(Path(log_dir) / self.dirname)
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
