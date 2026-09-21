"""Tests for `salt.callbacks.Checkpoint` (split from test_callbacks.py)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.callbacks import Checkpoint, StepCheckpoint
from salt.graph.errors import ConfigError


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
        # the v1 ctor filename stem: epoch=NNN-step=N-loss={monitor:.5f}
        cb = Checkpoint(monitor_loss="val/jets_classification_loss")
        assert (
            cb.filename
            == "epoch={epoch:03d}-step={step}-loss={val/jets_classification_loss:.5f}"
        )
        assert cb.save_top_k == -1  # keep every epoch (v1 :27)
        assert cb.monitor == "val/jets_classification_loss"
        assert cb.auto_insert_metric_name is False

    def test_formatted_name_keeps_loss_tag(self):
        # the per-task metric (carrying a '/') renders into the loss= stem the
        # salt-test best-epoch glob keys on; step= carries global_step for
        # auto-resume ranking (salt.main.latest_checkpoint)
        cb = Checkpoint(monitor_loss="val/loss")
        name = cb.format_checkpoint_name({"epoch": 9, "step": 40, "val/loss": 0.64624})
        assert name == "epoch=009-step=40-loss=0.64624.ckpt"

    def test_fname_string_override(self):
        cb = Checkpoint(monitor_loss="val/loss", fname_string="val_loss")
        assert cb.filename == "epoch={epoch:03d}-step={step}-val_loss={val/loss:.5f}"

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
        # the v1 s3 branch (checkpoint.py:34-43) is deferred — loud error
        cb = Checkpoint()
        with pytest.raises(ConfigError, match="s3://"):
            cb.setup(_ckpt_trainer("s3://bucket/run"), SimpleNamespace(), stage="fit")

    def test_best_checkpoint_resolves_what_setup_writes(self, tmp_path):
        # the salt-test run-dir contract end-to-end: a checkpoint named by the
        # callback under its setup-forced ckpts/ dir is discovered by
        # salt.main._best_checkpoint (the no-ckpt_path fallback); a step-only
        # (StepCheckpoint-style) file in the same dir must stay invisible to it
        from salt.main import _best_checkpoint

        cb = Checkpoint(monitor_loss="val/loss")
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        ckpt_dir = Path(cb.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for epoch, step, loss in ((8, 80, 0.70123), (9, 90, 0.64624)):
            name = cb.format_checkpoint_name({"epoch": epoch, "step": step, "val/loss": loss})
            (ckpt_dir / name).write_text("ckpt")
        # a step-only checkpoint (no loss= tag) must not confuse the best-epoch glob
        (ckpt_dir / "epoch=009-step=45.ckpt").write_text("ckpt")
        (tmp_path / "config.yaml").write_text("class_path: salt.model.SaltModule\n")
        best = _best_checkpoint(tmp_path / "config.yaml")
        assert Path(best).name == "epoch=009-step=90-loss=0.64624.ckpt"  # lowest loss


class TestStepCheckpoint:
    def test_no_trigger_raises(self):
        with pytest.raises(ConfigError, match="exactly one"):
            StepCheckpoint()

    def test_both_triggers_raises(self):
        with pytest.raises(ConfigError, match="exactly one"):
            StepCheckpoint(every_n_train_steps=5, train_time_interval=30)

    def test_invalid_save_top_k_raises(self):
        with pytest.raises(ConfigError, match="save_top_k"):
            StepCheckpoint(every_n_train_steps=5, save_top_k=2)

    def test_step_mode_config(self):
        cb = StepCheckpoint(every_n_train_steps=5)
        assert cb.monitor is None
        assert cb.save_top_k == 1
        assert cb.auto_insert_metric_name is False
        assert cb._every_n_train_steps == 5
        assert cb._every_n_epochs == 0
        name = cb.format_checkpoint_name({"epoch": 0, "step": 7})
        assert name == "epoch=000-step=7.ckpt"

    def test_time_mode_config(self):
        cb = StepCheckpoint(train_time_interval=30)
        assert cb._train_time_interval == timedelta(seconds=30)
        assert cb._every_n_train_steps == 0

    def test_save_top_k_minus_one_accepted(self):
        cb = StepCheckpoint(every_n_train_steps=5, save_top_k=-1)
        assert cb.save_top_k == -1

    def test_setup_fit_forces_ckpts_dir(self, tmp_path):
        cb = StepCheckpoint(every_n_train_steps=5)
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        assert str(cb.dirpath) == str(tmp_path / "ckpts")

    def test_setup_non_fit_does_not_force_ckpts(self, tmp_path):
        cb = StepCheckpoint(every_n_train_steps=5)
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="test")
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_fast_dev_run_does_not_force_ckpts(self, tmp_path):
        cb = StepCheckpoint(every_n_train_steps=5)
        cb.setup(
            _ckpt_trainer(str(tmp_path), fast_dev_run=True), SimpleNamespace(), stage="fit"
        )
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_s3_log_dir_raises(self):
        cb = StepCheckpoint(every_n_train_steps=5)
        with pytest.raises(ConfigError, match="s3://"):
            cb.setup(_ckpt_trainer("s3://bucket/run"), SimpleNamespace(), stage="fit")

    def test_shares_dirpath_with_checkpoint_but_distinct_state_key(self, tmp_path):
        # both callback instances coexist on the same trainer: same forced
        # ckpts/ dir, distinct Lightning state_key so neither clobbers the other
        epoch_cb = Checkpoint(monitor_loss="val/loss")
        step_cb = StepCheckpoint(every_n_train_steps=5)
        trainer = _ckpt_trainer(str(tmp_path))
        epoch_cb.setup(trainer, SimpleNamespace(), stage="fit")
        step_cb.setup(trainer, SimpleNamespace(), stage="fit")
        assert epoch_cb.dirpath == step_cb.dirpath
        assert epoch_cb.state_key != step_cb.state_key


def test_step_checkpoint_importable_from_callbacks():
    from salt.callbacks import StepCheckpoint as ImportedStepCheckpoint

    assert ImportedStepCheckpoint is StepCheckpoint
