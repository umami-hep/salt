"""Tests for `salt.core.callbacks.MaskformerConfusionMatrix` (split from test_callbacks.py)."""

from __future__ import annotations

from types import SimpleNamespace

from salt.tests.unit.callbacks.conftest import make_matched_bundle


# MaskformerConfusionMatrix (v1 port) — matched.objects.* FIT/VAL sink + CM


class TestMaskformerConfusionMatrix:
    def test_fit_val_demand_declares_matched_class_keys(self):
        from salt.core.callbacks import MaskformerConfusionMatrix

        callback = MaskformerConfusionMatrix()
        assert callback.fit_val_demand({}) == (
            "matched.objects.class_logits",
            "matched.objects.object_class",
        )

    def test_accumulate_then_stash_confusion_matrix(self):
        # two VAL batches accumulate, epoch-end stashes a [n_cls, n_cls] matrix
        # whose entries sum to all accumulated objects (logger absent -> no crash)
        from salt.core.callbacks import MaskformerConfusionMatrix

        callback = MaskformerConfusionMatrix()
        trainer = SimpleNamespace(fast_dev_run=False, current_epoch=0, logger=None)
        for seed in (1, 2):
            callback.on_validation_batch_end(
                trainer, None, {"bundle": make_matched_bundle(seed=seed, batch=6, m=5, n_cls=3)}, None, 0
            )
        callback.on_validation_epoch_end(trainer, None)
        assert callback.last_matrix is not None
        assert tuple(callback.last_matrix.shape) == (3, 3)
        assert int(callback.last_matrix.sum()) == 2 * 6 * 5  # 2 batches * B * M objects
        # accumulators reset for the next epoch, stash retained
        assert callback.truth_labels == [] and callback.pred_labels == []

    def test_fast_dev_run_skips_accumulation(self):
        from salt.core.callbacks import MaskformerConfusionMatrix

        callback = MaskformerConfusionMatrix()
        callback.on_validation_batch_end(
            SimpleNamespace(fast_dev_run=True), None, {"bundle": make_matched_bundle()}, None, 0
        )
        assert callback.truth_labels == [] and callback.pred_labels == []

    def test_log_every_n_epochs_gates_the_stash(self):
        # off-cadence epochs reset the accumulators WITHOUT computing/stashing
        from salt.core.callbacks import MaskformerConfusionMatrix

        callback = MaskformerConfusionMatrix(log_every_n_epochs=2)
        trainer = SimpleNamespace(fast_dev_run=False, current_epoch=1, logger=None)
        callback.on_validation_batch_end(trainer, None, {"bundle": make_matched_bundle()}, None, 0)
        callback.on_validation_epoch_end(trainer, None)
        assert callback.last_matrix is None  # epoch 1 % 2 != 0 -> not logged
        assert callback.truth_labels == [] and callback.pred_labels == []
