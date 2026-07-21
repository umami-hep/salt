"""Tests for `salt.callbacks.MaskformerMetrics` (split from test_callbacks.py)."""

from __future__ import annotations

from types import SimpleNamespace

from salt.graph.spec import Mode
from salt.model.saltmodule import SaltModule
from salt.tests.unit.callbacks.conftest import LRS, make_matched_bundle


# MaskformerMetrics (FD 1200-1202) — matched.objects.* FIT/VAL sink + metrics


class TestMaskformerMetrics:
    def test_fit_val_demand_declares_matched_keys(self):
        # the DP2 declaration: the matcher-permuted class/object-class/mask keys
        # the callback reads each VAL epoch (config-only, no setup needed)
        from salt.callbacks import MaskformerMetrics

        callback = MaskformerMetrics()
        assert callback.fit_val_demand({}) == (
            "matched.objects.class_logits",
            "matched.objects.object_class",
            "matched.objects.masks",
            "matched.objects.target_masks",
        )

    def test_enters_fit_val_sinks_and_keeps_matched_loss_alive(self):
        # the criterion: the REAL MaskformerMetrics callback, attached to a
        # SaltModule carrying a MaskFormerMatchedLoss, makes the matched.* keys
        # FIT/VAL plan sinks (DP2) AND keeps the matched loss alive under pruning
        from salt.callbacks import MaskformerMetrics
        from salt.graph.planner import compile_plan
        from salt.model.modules import LossSum
        from salt.tests._fixtures.v2_builders import (
            build_matched_loss_module,
        )

        loss_module = build_matched_loss_module()
        model = SaltModule({"mf_matched_loss": loss_module, "loss": LossSum()}, lrs=LRS)
        model._trainer = SimpleNamespace(  # noqa: SLF001 - duck-typed attach
            callbacks=[MaskformerMetrics()], datamodule=SimpleNamespace(reader=None)
        )
        for mode in (Mode.FIT, Mode.VAL):
            sinks = model._model_sinks(mode)  # noqa: SLF001
            assert "matched.objects.class_logits" in sinks
            assert "matched.objects.object_class" in sinks
        # the matched loss survives pruning to a FIT plan with those sinks: the
        # matched.* products keep its module alive (it has no loss anchoring them
        # beyond losses.* — the callback sink is what keeps matched.* reachable)
        m = 5
        n_cls = 3
        from salt.graph.spec import TensorSpec, unflatten_spec

        f = Mode.FIT
        sources = unflatten_spec({
            "objects.class_logits": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
            "objects.class_probs": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
            "objects.embed": TensorSpec(shape=("B", m, 16), dtype="float32", modes=f),
            "objects.masks": TensorSpec(shape=("B", m, "T:tracks"), dtype="float32", modes=f),
            "preds.objects.regression": TensorSpec(shape=("B", m, 3), dtype="float32", modes=f),
            "targets.objects.regression": TensorSpec(shape=("B", m, 3), dtype="float32", modes=f),
            "labels.objects.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            "labels.objects.masks": TensorSpec(
                shape=("B", m, "T:tracks"), dtype="bool", kind="label", modes=f
            ),
        })
        plan = compile_plan(
            {"mf_matched_loss": loss_module},
            Mode.FIT,
            sources=sources,
            sinks=[
                "matched.objects.class_logits",
                "matched.objects.object_class",
                "matched.objects.masks",
                "matched.objects.target_masks",
            ],
        )
        assert "mf_matched_loss" in plan.module_names

    def test_compute_metrics_from_matched_bundle(self):
        # the metrics are computed from matched.objects.* (bundle-native) and stashed
        from salt.callbacks import MaskformerMetrics

        callback = MaskformerMetrics()
        trainer = SimpleNamespace(fast_dev_run=False)
        logged: dict[str, float] = {}
        pl_module = SimpleNamespace(log=lambda name, value: logged.__setitem__(name, float(value)))
        callback.on_validation_batch_end(
            trainer, pl_module, {"bundle": make_matched_bundle()}, None, 0
        )
        # the v1 metric set is present, logged under the val/ prefix and stashed
        assert callback.last_metrics
        assert "class_exact_match" in callback.last_metrics
        assert "notnull_eff" in callback.last_metrics and "notnull_pur" in callback.last_metrics
        assert "query_perfect_match_eff" in callback.last_metrics
        assert "query_regression_mae" in callback.last_metrics
        assert any(k.startswith("val/") for k in logged)
        # efficiencies / purities are valid fractions in [0, 1]
        assert 0.0 <= callback.last_metrics["notnull_eff"] <= 1.0
        assert 0.0 <= callback.last_metrics["class_exact_match"] <= 1.0

    def test_only_val_skips_train_logging(self):
        from salt.callbacks import MaskformerMetrics

        callback = MaskformerMetrics(only_val=True)
        logged: dict[str, float] = {}
        pl_module = SimpleNamespace(log=lambda n, v: logged.__setitem__(n, v))
        callback.on_train_batch_end(
            SimpleNamespace(fast_dev_run=False),
            pl_module,
            {"bundle": make_matched_bundle()},
            None,
            0,
        )
        assert logged == {}  # only_val=True -> no train logging
