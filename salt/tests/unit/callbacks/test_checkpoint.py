"""Tests for `salt.core.callbacks.Checkpoint` (split from test_callbacks.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.core.callbacks import Checkpoint
from salt.core.graph.errors import ConfigError


# Checkpoint (the v1 salt.callbacks.Checkpoint port — D2 slice) + ProgressBar


def _ckpt_trainer(log_dir: str, *, fast_dev_run: bool = False) -> SimpleNamespace:
    """A minimal trainer satisfying the real `ModelCheckpoint.setup`."""
    return SimpleNamespace(
        fast_dev_run=fast_dev_run,
        log_dir=log_dir,
        default_root_dir=log_dir,
        is_global_zero=True,
        loggers=[],
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )


class TestCheckpoint:
    def test_filename_monitor_contract(self):
        # the v1 ctor string (checkpoint.py:26): epoch=NNN-loss={monitor:.5f}
        cb = Checkpoint(monitor_loss="val/jets_classification_loss")
        assert cb.filename == "epoch={epoch:03d}-loss={val/jets_classification_loss:.5f}"
        assert cb.save_top_k == -1  # keep every epoch (v1 :27)
        assert cb.monitor == "val/jets_classification_loss"
        assert cb.auto_insert_metric_name is False

    def test_formatted_name_keeps_loss_tag(self):
        # the per-task metric (carrying a '/') renders into the loss= stem the
        # salt-test best-epoch glob keys on
        cb = Checkpoint(monitor_loss="val/loss")
        name = cb.format_checkpoint_name({"epoch": 9, "val/loss": 0.64624})
        assert name == "epoch=009-loss=0.64624.ckpt"

    def test_fname_string_override(self):
        cb = Checkpoint(monitor_loss="val/loss", fname_string="val_loss")
        assert cb.filename == "epoch={epoch:03d}-val_loss={val/loss:.5f}"

    def test_setup_fit_forces_ckpts_dir(self, tmp_path):
        # the real setup forces dirpath to <log_dir>/ckpts (v1 :44-46), driven
        # through ModelCheckpoint.setup's pre-set-dirpath short-circuit
        cb = Checkpoint(monitor_loss="val/loss")
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        assert str(cb.dirpath) == str(tmp_path / "ckpts")

    def test_setup_non_fit_does_not_force_ckpts(self, tmp_path):
        # non-fit setup is a no-op for our branch (v1 :30-32): super resolves to
        # the Lightning default 'checkpoints/', not the v1 'ckpts/'
        cb = Checkpoint()
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="test")
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_fast_dev_run_does_not_force_ckpts(self, tmp_path):
        cb = Checkpoint()
        cb.setup(_ckpt_trainer(str(tmp_path), fast_dev_run=True), SimpleNamespace(), stage="fit")
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_s3_log_dir_raises(self):
        # the v1 s3 branch (checkpoint.py:34-43) is deferred to M6 — loud error
        cb = Checkpoint()
        with pytest.raises(ConfigError, match="s3://"):
            cb.setup(_ckpt_trainer("s3://bucket/run"), SimpleNamespace(), stage="fit")

    def test_best_checkpoint_resolves_what_setup_writes(self, tmp_path):
        # the salt-test run-dir contract end-to-end: a checkpoint named by the
        # callback under its setup-forced ckpts/ dir is discovered by
        # salt.core.main._best_checkpoint (the no-ckpt_path fallback)
        from salt.core.main import _best_checkpoint

        cb = Checkpoint(monitor_loss="val/loss")
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        ckpt_dir = Path(cb.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for epoch, loss in ((8, 0.70123), (9, 0.64624)):
            name = cb.format_checkpoint_name({"epoch": epoch, "val/loss": loss})
            (ckpt_dir / name).write_text("ckpt")
        (tmp_path / "config.yaml").write_text("class_path: salt.core.SaltModule\n")
        best = _best_checkpoint(tmp_path / "config.yaml")
        assert Path(best).name == "epoch=009-loss=0.64624.ckpt"  # lowest loss
