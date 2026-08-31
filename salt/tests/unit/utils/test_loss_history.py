"""Tests for `salt.utils.loss_history.LossHistoryWriter`."""

from __future__ import annotations

import csv
from types import SimpleNamespace

from salt.utils.loss_history import LossHistoryWriter


def _trainer(tmp_path, *, sanity_checking=False, fast_dev_run=False, metrics=None):
    return SimpleNamespace(
        log_dir=str(tmp_path),
        default_root_dir=str(tmp_path),
        sanity_checking=sanity_checking,
        fast_dev_run=fast_dev_run,
        current_epoch=3,
        global_step=42,
        callback_metrics=metrics or {},
    )


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestLossHistoryWriter:
    def test_val_epoch_end_writes_only_val_prefixed_metrics(self, tmp_path):
        metrics = {"val/loss": 1.5, "train/loss": 2.5, "other/metric": 0.1}
        trainer = _trainer(tmp_path, metrics=metrics)
        cb = LossHistoryWriter()
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        rows = _rows(tmp_path / "loss_history.csv")
        assert len(rows) == 1
        assert rows[0]["stage"] == "val"
        assert rows[0]["metric"] == "val/loss"
        assert rows[0]["epoch"] == "3"
        assert rows[0]["step"] == "42"
        assert float(rows[0]["value"]) == 1.5

    def test_train_epoch_end_writes_only_train_prefixed_metrics(self, tmp_path):
        metrics = {"val/loss": 1.5, "train/loss": 2.5}
        trainer = _trainer(tmp_path, metrics=metrics)
        cb = LossHistoryWriter()
        cb.on_train_epoch_end(trainer, SimpleNamespace())
        rows = _rows(tmp_path / "loss_history.csv")
        assert len(rows) == 1
        assert rows[0]["stage"] == "train"
        assert rows[0]["metric"] == "train/loss"

    def test_both_hooks_append_to_the_same_file(self, tmp_path):
        metrics = {"val/loss": 1.5, "train/loss": 2.5}
        trainer = _trainer(tmp_path, metrics=metrics)
        cb = LossHistoryWriter()
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        cb.on_train_epoch_end(trainer, SimpleNamespace())
        rows = _rows(tmp_path / "loss_history.csv")
        assert {r["stage"] for r in rows} == {"val", "train"}

    def test_sanity_checking_writes_nothing(self, tmp_path):
        trainer = _trainer(tmp_path, sanity_checking=True, metrics={"val/loss": 1.5})
        cb = LossHistoryWriter()
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        assert not (tmp_path / "loss_history.csv").exists()

    def test_fast_dev_run_writes_nothing(self, tmp_path):
        trainer = _trainer(tmp_path, fast_dev_run=True, metrics={"train/loss": 2.5})
        cb = LossHistoryWriter()
        cb.on_train_epoch_end(trainer, SimpleNamespace())
        assert not (tmp_path / "loss_history.csv").exists()

    def test_no_matching_metrics_writes_no_file(self, tmp_path):
        trainer = _trainer(tmp_path, metrics={"other/metric": 0.1})
        cb = LossHistoryWriter()
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        assert not (tmp_path / "loss_history.csv").exists()

    def test_falls_back_to_default_root_dir_when_log_dir_is_none(self, tmp_path):
        trainer = _trainer(tmp_path, metrics={"val/loss": 1.5})
        trainer.log_dir = None
        cb = LossHistoryWriter()
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        assert (tmp_path / "loss_history.csv").is_file()

    def test_custom_fname(self, tmp_path):
        trainer = _trainer(tmp_path, metrics={"val/loss": 1.5})
        cb = LossHistoryWriter(fname="custom.csv")
        cb.on_validation_epoch_end(trainer, SimpleNamespace())
        assert (tmp_path / "custom.csv").is_file()
        assert not (tmp_path / "loss_history.csv").exists()
