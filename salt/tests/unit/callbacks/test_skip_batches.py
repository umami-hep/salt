"""Unit tests for `salt.callbacks.skip_batches` (SkipBatchesWithoutLabels)."""

from __future__ import annotations

from types import SimpleNamespace

import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from salt.callbacks.skip_batches import SkipBatchesWithoutLabels
from salt.graph.errors import ConfigError

KEYS = ("labels.jets.small", "labels.jets.large")


def _batch(small: list[int], large: list[int]) -> dict:
    return {
        "x": torch.ones(len(small), 2),
        "labels": {"jets": {"small": torch.tensor(small), "large": torch.tensor(large)}},
    }


def test_needs_label_keys() -> None:
    with pytest.raises(ConfigError, match="non-empty"):
        SkipBatchesWithoutLabels([])
    with pytest.raises(ConfigError, match="non-empty"):
        SkipBatchesWithoutLabels("labels.jets.small")


def test_empty_labels_nested_batch() -> None:
    cb = SkipBatchesWithoutLabels(KEYS)
    assert cb.empty_labels(_batch([0, 1, -1], [-1, -1, 3])) == []
    assert cb.empty_labels(_batch([0, 1, 2], [-1, -1, -1])) == ["labels.jets.large"]
    assert cb.empty_labels(_batch([-1, -1], [-1, -1])) == list(KEYS)


def test_empty_labels_flat_batch() -> None:
    cb = SkipBatchesWithoutLabels(KEYS)
    flat = {"labels.jets.small": torch.tensor([2, -1]), "labels.jets.large": torch.tensor([-1, -1])}
    assert cb.empty_labels(flat) == ["labels.jets.large"]


def test_nan_regression_target_counts_as_unlabelled() -> None:
    cb = SkipBatchesWithoutLabels(["labels.jets.pt_target"])
    nan = float("nan")
    assert cb.empty_labels({"labels.jets.pt_target": torch.tensor([nan, nan])}) == ["labels.jets.pt_target"]
    assert cb.empty_labels({"labels.jets.pt_target": torch.tensor([nan, 0.97])}) == []


def test_missing_label_raises() -> None:
    cb = SkipBatchesWithoutLabels(["labels.jets.other"])
    with pytest.raises(KeyError, match="labels.jets.other"):
        cb.empty_labels(_batch([0], [1]))


def test_skip_drops_gradients_and_counts() -> None:
    cb = SkipBatchesWithoutLabels(KEYS)
    module = torch.nn.Linear(2, 3)
    module.device = torch.device("cpu")
    trainer = SimpleNamespace(current_epoch=0)
    module(torch.ones(1, 2)).sum().backward()

    cb.on_train_batch_start(trainer, module, _batch([0, 1], [2, 3]), 0)
    cb.on_before_optimizer_step(trainer, module, None)
    assert all(p.grad is not None for p in module.parameters())

    cb.on_train_batch_start(trainer, module, _batch([0, 1], [-1, -1]), 1)
    cb.on_before_optimizer_step(trainer, module, None)
    assert all(p.grad is None for p in module.parameters())
    assert (cb.skipped_this_epoch, cb.skipped_total) == (1, 1)
    assert not cb.skip_pending


def test_skip_covers_the_accumulation_window() -> None:
    cb = SkipBatchesWithoutLabels(KEYS)
    module = torch.nn.Linear(2, 3)
    module.device = torch.device("cpu")
    trainer = SimpleNamespace(current_epoch=0)
    cb.on_train_batch_start(trainer, module, _batch([0], [-1]), 0)  # empty, no step yet
    cb.on_train_batch_start(trainer, module, _batch([0], [1]), 1)  # fine, step follows
    module(torch.ones(1, 2)).sum().backward()
    cb.on_before_optimizer_step(trainer, module, None)
    assert all(p.grad is None for p in module.parameters())


class _ToyModule(lightning.LightningModule):
    """Two classification heads on one layer; -1 labels ignored, an all-ignored head reports 0."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(2, 3)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.net(batch["x"])
        losses = []
        for name in ("small", "large"):
            labels = batch["labels"]["jets"][name]
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
            losses.append(torch.where((labels != -1).any(), loss, torch.zeros_like(loss)))
        return torch.pow(losses[0] * losses[1], 0.5)  # geometric mean, as in LossGLS

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _train_one_step(batch: dict, callbacks: list) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _ToyModule()
    before = model.net.weight.detach().clone()
    trainer = lightning.Trainer(
        max_steps=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
    )
    trainer.fit(model, DataLoader([batch], batch_size=None))
    return before, model.net.weight.detach().clone()


def test_training_skips_an_empty_batch() -> None:
    before, after = _train_one_step(_batch([0, 1, 2], [-1, -1, -1]), [SkipBatchesWithoutLabels(KEYS)])
    assert torch.equal(before, after)
    assert torch.isfinite(after).all()


def test_without_the_callback_an_empty_batch_corrupts_the_weights() -> None:
    _, after = _train_one_step(_batch([0, 1, 2], [-1, -1, -1]), [])
    assert not torch.isfinite(after).all()


def test_training_updates_on_a_normal_batch() -> None:
    before, after = _train_one_step(_batch([0, 1, 2], [2, -1, 0]), [SkipBatchesWithoutLabels(KEYS)])
    assert not torch.equal(before, after)
    assert torch.isfinite(after).all()
