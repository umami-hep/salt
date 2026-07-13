"""Unit tests for LossSum and LossGLS (mirror of salt/core/nn/losses.py)."""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import (
    Bundle,
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.core.nn import (
    LossGLS,
    LossSum,
)
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
)


class TestLossSum:
    def test_unnarrowed_declare_raises(self):
        loss = LossSum()
        loss.name = "loss"
        with pytest.raises(ConfigError, match="framework"):
            loss.declare_io(Mode.FIT)

    def test_explicit_losses_config(self):
        loss = LossSum(losses=["jets_classification", "losses.track_origin"])
        loss.name = "loss"
        io = loss.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {
            "losses.jets_classification",
            "losses.track_origin",
        }
        assert set(flatten_spec(io.produces)) == {"loss.total"}

    def test_inactive_outside_training(self):
        loss = LossSum(losses=["a"])
        loss.name = "loss"
        io = loss.declare_io(Mode.TEST)
        assert not flatten_spec(io.requires) and not flatten_spec(io.produces)

    def test_unknown_weight_key_rejected(self):
        with pytest.raises(ConfigError, match="unknown loss keys"):
            LossSum(losses=["a"], weights={"b": 2.0})

    def test_weighted_sum(self):
        loss = LossSum(losses=["a", "b"], weights={"b": 2.0})
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(1.0))
        b.set("losses.b", torch.tensor(3.0))
        out = loss(b, Mode.FIT)
        assert out["loss.total"].item() == pytest.approx(7.0)

    def test_collect_loss_keys(self, gn2v2):
        modules, _, _ = gn2v2
        assert LossSum.collect_loss_keys(modules) == (
            "losses.jets_classification",
            "losses.track_origin",
            "losses.track_vertexing",
        )

    def test_double_narrow_rejected(self, gn2v2):
        modules, _, _ = gn2v2
        with pytest.raises(ConfigError, match="already fixed"):
            modules["loss"].narrow(["losses.x"])


class TestLossGLS:
    """`LossGLS` — geometric-mean combination + the all-weights==1.0 guard."""

    def test_is_a_losssum_subclass(self):
        # the SaltModule narrow loop keys off isinstance(_, LossSum), so a
        # LossGLS must be picked up for free (collect_loss_keys/narrow/declare_io)
        assert issubclass(LossGLS, LossSum)
        assert isinstance(LossGLS(losses=["a"]), LossSum)

    def test_geometric_mean_forward(self):
        loss = LossGLS(losses=["a", "b", "c"])
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(2.0))
        b.set("losses.b", torch.tensor(8.0))
        b.set("losses.c", torch.tensor(4.0))
        out = loss(b, Mode.FIT)
        # geometric mean of 2, 8, 4 is the cube root of 64, which is 4.0
        assert out["loss.total"].item() == pytest.approx(4.0)

    def test_two_task_geometric_mean_not_sum(self):
        # the smallest genuinely-combined case (the GN3 journey): >= 2 losses
        loss = LossGLS(losses=["a", "b"])
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(2.0))
        b.set("losses.b", torch.tensor(8.0))
        total = loss(b, Mode.FIT)["loss.total"]
        assert total.item() == pytest.approx(4.0)  # sqrt(16), NOT the sum 10
        assert total.item() != pytest.approx(10.0)

    def test_inherits_losssum_skeleton(self):
        # declare_io is the LossSum skeleton: losses.* -> loss.total (TRAINING),
        # empty outside TRAINING
        loss = LossGLS(losses=["a", "b"])
        loss.name = "loss"
        io = loss.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"losses.a", "losses.b"}
        assert set(flatten_spec(io.produces)) == {"loss.total"}
        assert not flatten_spec(loss.declare_io(Mode.TEST).requires)

    def test_module_weight_not_one_rejected(self):
        # GLS does not utilise weights — the per-loss weights surface is guarded
        with pytest.raises(ConfigError, match="not utilised by the geometric mean"):
            LossGLS(losses=["a", "b"], weights={"b": 2.0})

    def test_module_weight_one_accepted(self):
        # an explicit weight of exactly 1.0 is a no-op, not an error
        loss = LossGLS(losses=["a", "b"], weights={"a": 1.0, "b": 1.0})
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(4.0))
        b.set("losses.b", torch.tensor(9.0))
        assert loss(b, Mode.FIT)["loss.total"].item() == pytest.approx(6.0)  # sqrt(36)

    def test_check_task_weights_rejects_weighted_task(self):
        # the task-side weight: float surface — the v2 home of v1's ctor assert
        # (modelwrapper.py:139-142). check_task_weights inspects sibling tasks.
        weighted = ClassificationTaskModule(
            stream="jets", label="flav", class_names=["u", "b"], weight=2.0
        )
        weighted.name = "jets_classification"
        unit = RegressionTaskModule(stream="jets", targets="pt", input="pooled.global", weight=1.0)
        unit.name = "reg"
        gls = LossGLS()
        gls.name = "loss"
        modules = {"jets_classification": weighted, "reg": unit, "loss": gls}
        with pytest.raises(ConfigError, match="does not utilise task weights"):
            LossGLS.check_task_weights(modules)

    def test_check_task_weights_accepts_unit_weights(self):
        # all task weights 1.0 (the GLS validity domain) passes cleanly; the
        # loss module itself (a LossSum subclass) is skipped, not flagged
        t1 = ClassificationTaskModule(
            stream="jets", label="flav", class_names=["u", "b"], weight=1.0
        )
        t1.name = "jets_classification"
        t2 = RegressionTaskModule(stream="jets", targets="pt", input="pooled.global", weight=1.0)
        t2.name = "reg"
        gls = LossGLS()
        gls.name = "loss"
        LossGLS.check_task_weights({"jets_classification": t1, "reg": t2, "loss": gls})

    def test_check_task_weights_catches_int_weight(self):
        # hardening: the guard is duck-typed numeric (int OR float), so a future
        # task module storing an un-coerced int weight is still caught — int 2
        # raises, int 1 passes, a non-numeric weight is ignored (no numeric
        # weight to guard). The shipped task base float-coerces (tasks.py:88), so
        # this is forward-robustness, not a config-reachable path today.
        class _IntWeightTask:
            def __init__(self, w):
                self.weight = w

        with pytest.raises(ConfigError, match="does not utilise task weights"):
            LossGLS.check_task_weights({"t": _IntWeightTask(2)})  # int, not float
        # int 1 (== 1.0) and a non-numeric weight both pass cleanly
        LossGLS.check_task_weights({"t": _IntWeightTask(1)})
        LossGLS.check_task_weights({"t": _IntWeightTask("not-a-number")})
