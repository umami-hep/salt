"""`Checkpoint` — the per-epoch ``ckpts/`` checkpoint callback.

`StepCheckpoint` — the opt-in intra-epoch sibling for crash/requeue recovery.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from salt.graph.errors import ConfigError


def _forced_ckpt_dir(trainer: Trainer, dirname: str) -> str:
    """Resolve ``<log_dir>/<dirname>``; raises `ConfigError` for an unsupported
    ``s3://`` log dir.
    """
    log_dir = trainer.log_dir or trainer.default_root_dir
    if log_dir is not None and str(log_dir).startswith(("s3://", "s3:/")):
        raise ConfigError(
            "salt.callbacks.Checkpoint/StepCheckpoint do not support s3:// log dirs "
            "yet (rides with the Comet/run-dir wiring); use a local trainer.log_dir "
            "(v1 checkpoint.py:34-43 s3 branch deferred)"
        )
    return str(Path(log_dir) / dirname)


class Checkpoint(ModelCheckpoint):
    """Save a checkpoint per epoch under ``ckpts/`` with the ``loss=`` filename stem.

    Filename and directory are a contract: ``salt test`` without
    ``--ckpt_path`` resolves the best epoch by globbing
    ``<config dir>/{ckpts,checkpoints}/*.ckpt`` and parsing the smallest
    ``loss=<value>`` out of each name (`salt.main._best_checkpoint`); native
    ``--auto_resume`` (`salt.main.latest_checkpoint`) instead parses the
    ``epoch=`` and ``step=`` tokens to rank by ``(epoch, global_step)``. The
    stem is ``epoch=NNN-step=N-loss=<value>.ckpt`` — keep it, the best-epoch
    glob, and the auto-resume parser in sync.

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
        filename = "epoch={epoch:03d}-step={step}-" + fname_string + "={" + monitor_loss + ":.5f}"
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
            self.dirpath = _forced_ckpt_dir(trainer, self.dirname)
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)


class StepCheckpoint(ModelCheckpoint):
    """Opt-in intra-epoch checkpoint for crash/requeue recovery between epoch ends.

    Saves under the same forced ``ckpts/`` directory as `Checkpoint`, with the
    unmonitored stem ``epoch=NNN-step=N.ckpt`` — no ``loss=`` tag, so it is
    invisible to `salt.main._best_checkpoint`'s best-epoch glob, but eligible
    for native ``--auto_resume`` (`salt.main.latest_checkpoint`), which ranks
    checkpoints of both callbacks by ``(epoch, global_step)``. Defaults to
    keeping only the latest step checkpoint (``save_top_k=1``) so a
    month-long run does not silently fill the disk; pass ``save_top_k=-1`` to
    keep every one instead.

    Fires on exactly one of a step count or a wall-clock interval, never both
    and never neither — Lightning would otherwise either fall back to
    every-epoch saving (neither set) or raise its own `MisconfigurationException`
    (both set); this callback fails loudly with a salt `ConfigError` instead.

    Examples
    --------
    Every 500 optimizer steps::

        callbacks:
          step_checkpoint:
            class_path: salt.callbacks.StepCheckpoint
            init_args: {every_n_train_steps: 500}

    Every 30 minutes::

        callbacks:
          step_checkpoint:
            class_path: salt.callbacks.StepCheckpoint
            init_args: {train_time_interval: 1800}

    Parameters
    ----------
    every_n_train_steps : int | None, optional
        Save every N optimizer steps. Mutually exclusive with
        ``train_time_interval`` — exactly one of the two must be set.
    train_time_interval : float | None, optional
        Save every N seconds of wall-clock training time, converted to a
        `datetime.timedelta`. Mutually exclusive with ``every_n_train_steps``.
    save_top_k : int, optional
        ``1`` (default) keeps only the latest step checkpoint; ``-1`` keeps
        every one. Any other value raises `ConfigError` — Lightning only
        prunes unmonitored saves when ``save_top_k == 1``.
    dirname : str, optional
        The log-dir sub-directory checkpoints land in, by default ``"ckpts"``.
    """

    def __init__(
        self,
        every_n_train_steps: int | None = None,
        train_time_interval: float | None = None,
        save_top_k: int = 1,
        dirname: str = "ckpts",
    ) -> None:
        if (every_n_train_steps is None) == (train_time_interval is None):
            raise ConfigError(
                "salt.callbacks.StepCheckpoint requires exactly one of "
                "every_n_train_steps or train_time_interval (Lightning silently "
                "falls back to every-epoch saving with neither set, and raises "
                "its own error with both set) — set exactly one."
            )
        if save_top_k not in {1, -1}:
            raise ConfigError(
                "salt.callbacks.StepCheckpoint save_top_k must be 1 (keep latest) "
                f"or -1 (keep all) — got {save_top_k}. Lightning only prunes "
                "unmonitored saves when save_top_k == 1, so any other positive "
                "value would silently keep unbounded checkpoints on disk."
            )
        interval = (
            timedelta(seconds=train_time_interval) if train_time_interval is not None else None
        )
        super().__init__(
            monitor=None,
            filename="epoch={epoch:03d}-step={step}",
            auto_insert_metric_name=False,
            save_top_k=save_top_k,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=interval,
        )
        self.dirname = dirname

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Fix the checkpoint dir to ``<log_dir>/<dirname>`` on a real fit; raises
        `ConfigError` for an unsupported ``s3://`` log dir.
        """
        if stage == "fit" and not trainer.fast_dev_run:
            self.dirpath = _forced_ckpt_dir(trainer, self.dirname)
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
