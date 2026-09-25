"""Unit tests for TrainingController."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from salt.graph.errors import ConfigError
from salt.model.multistage_training import TrainingController
from salt.schedule import EarlyStopTracker, TrainingSchedule

MODULES = ("a", "b")
LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}
OPTIMIZER_NAME = "AdamW"


def make_net() -> nn.ModuleDict:
    return nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)})


def make_trainer(**over: Any) -> SimpleNamespace:
    defaults = {
        "max_epochs": 4,
        "strategy": None,
        "limit_val_batches": 1,
        "state": None,
        "estimated_stepping_batches": 40,
    }
    defaults.update(over)
    return SimpleNamespace(**defaults)


def single_stage() -> TrainingSchedule:
    return TrainingSchedule.desugar_legacy(MODULES)


def two_stage_no_es() -> TrainingSchedule:
    raw = {"stages": {"warmup": {"epochs": 2, "frozen": ["a"]}, "full": {}}}
    return TrainingSchedule.from_config(raw, MODULES)


def two_stage_es() -> TrainingSchedule:
    raw = {
        "stages": {
            "warmup": {"epochs": 2, "frozen": ["a"], "lrs": {"max": 5e-3}, "optimizer": "lion"},
            "full": {"early_stop": {"monitor": "val/loss", "patience": 2}},
        }
    }
    return TrainingSchedule.from_config(raw, MODULES)


def make_controller(schedule: TrainingSchedule) -> TrainingController:
    return TrainingController(schedule, make_net(), LRS, OPTIMIZER_NAME)


class TestConstruction:
    def test_defaults(self):
        schedule = single_stage()
        net = make_net()
        c = TrainingController(schedule, net, LRS, OPTIMIZER_NAME)
        assert c.current_stage_index == 0
        assert c.frozen_module_names == set()
        assert c.stage_start_epoch == 0
        assert c.early_stop_tracker is None
        assert c.boundary_records == []
        assert c.pending_early_advance is False
        assert c.reducer_safe_freeze is False
        assert c.schedule is schedule
        assert c.net is net
        assert c.lrs is LRS
        assert c.optimizer is OPTIMIZER_NAME

    def test_not_an_nn_module_and_net_unregistered(self):
        net = make_net()
        c = make_controller(single_stage())
        c.net = net
        assert not isinstance(c, nn.Module)
        assert len(list(net.parameters())) == 4


class TestPreflightAndStartFit:
    def test_start_fit_freezes_stage_zero_and_seeds_state(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        assert c.frozen_module_names == {"a"}
        for param in c.net["a"].parameters():
            assert param.requires_grad is False
        for param in c.net["b"].parameters():
            assert param.requires_grad is True
        assert c.net["a"].training is False
        assert c.reducer_safe_freeze is False
        assert c.early_stop_tracker is None
        assert c.current_stage_index == 0
        assert c.boundary_records == []

    def test_preflight_raises_when_validation_disabled_and_early_stop_declared(self):
        c = make_controller(two_stage_es())
        with pytest.raises(ConfigError, match="limit_val_batches=0"):
            c.preflight(make_trainer(limit_val_batches=0))

    def test_preflight_does_not_raise_without_early_stop(self):
        c = make_controller(two_stage_no_es())
        c.preflight(make_trainer(limit_val_batches=0))

    def test_preflight_raises_on_multi_stage_with_no_max_epochs(self):
        c = make_controller(two_stage_no_es())
        with pytest.raises(ConfigError, match="multiple epoch-delimited stages"):
            c.preflight(make_trainer(max_epochs=None))


class TestAdvanceToStage:
    def test_advance_with_early_stop_records_boundary_and_rebuilds_tracker(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.current_stage_index == 1
        assert c.frozen_module_names == set()
        for param in c.net["a"].parameters():
            assert param.requires_grad is True
        assert c.net["a"].training is True
        assert c.stage_start_epoch == 2
        assert c.pending_early_advance is False
        assert len(c.boundary_records) == 1
        record = c.boundary_records[0]
        assert record["stage_name"] == "full"
        assert record["stage_index"] == 1
        assert record["global_step"] == 10
        assert record["epoch"] == 2
        assert record["reason"] == "epochs"
        assert isinstance(c.early_stop_tracker, EarlyStopTracker)

    def test_advance_without_early_stop_unfreezes_but_records_nothing(self):
        c = make_controller(two_stage_no_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.frozen_module_names == set()
        assert c.boundary_records == []
        assert c.stage_start_epoch == 0
        assert c.early_stop_tracker is None


class TestCheckpointState:
    def test_no_early_stop_schedule_has_two_keys(self):
        c = make_controller(two_stage_no_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        assert c.checkpoint_state() == {"stage_index": 0, "stage_name": "warmup"}

    def test_single_stage_has_two_keys(self):
        assert make_controller(single_stage()).checkpoint_state() == {
            "stage_index": 0,
            "stage_name": "fit",
        }

    def test_early_stop_schedule_stage_zero_has_no_early_stop_state(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        state = c.checkpoint_state()
        assert set(state) == {"stage_index", "stage_name", "stage_start_epoch", "boundaries"}

    def test_early_stop_schedule_after_advance_has_early_stop_state(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        state = c.checkpoint_state()
        assert set(state) == {
            "stage_index",
            "stage_name",
            "stage_start_epoch",
            "boundaries",
            "early_stop_state",
        }
        stage = c.schedule.stages[1]
        assert state["early_stop_state"] == {
            "best_score": None,
            "wait_count": 0,
            "check_count": 0,
            "fingerprint": stage.early_stop.fingerprint(),
        }


class TestRestoreRoundTrip:
    def test_round_trip_at_stage_one_with_early_stop_counters(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.evaluate_early_stop(1.0) is False
        assert c.evaluate_early_stop(1.0) is False
        assert c.early_stop_tracker.wait_count == 1
        state = c.checkpoint_state()

        c2 = make_controller(two_stage_es())
        c2.restore_schedule_stage(state, make_trainer())
        assert c2.current_stage_index == 1
        assert c2.frozen_module_names == set()
        assert c2.stage_start_epoch == 2
        assert c2.boundary_records == c.boundary_records
        assert c2.early_stop_tracker.state_dict() == c.early_stop_tracker.state_dict()
        assert c2.checkpoint_state() == state

    def test_round_trip_at_stage_zero_reapplies_freeze(self):
        c3 = make_controller(two_stage_es())
        state0 = {
            "stage_index": 0,
            "stage_name": "warmup",
            "stage_start_epoch": 0,
            "boundaries": [],
        }
        c3.restore_schedule_stage(state0, make_trainer())
        assert c3.frozen_module_names == {"a"}
        for param in c3.net["a"].parameters():
            assert param.requires_grad is False

    def test_none_state_is_a_no_op(self):
        c = make_controller(two_stage_es())
        c.restore_schedule_stage(None, make_trainer())
        assert c.current_stage_index == 0

    def test_out_of_range_stage_index_raises(self):
        c = make_controller(two_stage_es())
        with pytest.raises(ConfigError, match="out of range"):
            c.restore_schedule_stage({"stage_index": 5, "stage_name": "x"}, make_trainer())

    def test_mismatched_stage_name_raises(self):
        c = make_controller(two_stage_es())
        with pytest.raises(ConfigError, match="names it"):
            c.restore_schedule_stage({"stage_index": 1, "stage_name": "wrong"}, make_trainer())

    def test_criterion_fingerprint_mismatch_raises(self):
        c = make_controller(two_stage_es())
        state = {
            "stage_index": 1,
            "stage_name": "full",
            "stage_start_epoch": 2,
            "boundaries": [],
            "early_stop_state": {
                "best_score": None,
                "wait_count": 0,
                "check_count": 0,
                "fingerprint": {
                    "monitor": "val/loss",
                    "mode": "min",
                    "min_delta": 0.0,
                    "patience": 99,
                },
            },
        }
        with pytest.raises(ConfigError, match="criterion"):
            c.restore_schedule_stage(state, make_trainer())

    @pytest.mark.parametrize(
        "missing", ["stage_name", "stage_start_epoch", "boundaries", "early_stop_state"]
    )
    def test_missing_schedule_payload_key_raises_config_error(self, missing):
        c = make_controller(two_stage_es())
        state = {
            "stage_index": 1,
            "stage_name": "full",
            "stage_start_epoch": 2,
            "boundaries": [],
            "early_stop_state": {
                "best_score": None,
                "wait_count": 0,
                "check_count": 0,
                "fingerprint": c.schedule.stages[1].early_stop.fingerprint(),
            },
        }
        del state[missing]
        with pytest.raises(ConfigError, match=f"missing '{missing}'"):
            c.restore_schedule_stage(state, make_trainer())

    def test_missing_fingerprint_raises_config_error(self):
        c = make_controller(two_stage_es())
        state = {
            "stage_index": 1,
            "stage_name": "full",
            "stage_start_epoch": 2,
            "boundaries": [],
            "early_stop_state": {"best_score": None, "wait_count": 0, "check_count": 0},
        }
        with pytest.raises(ConfigError, match="missing 'fingerprint'"):
            c.restore_schedule_stage(state, make_trainer())

    def test_missing_stage_index_raises_config_error(self):
        c = make_controller(two_stage_es())
        with pytest.raises(ConfigError, match="missing 'stage_index'"):
            c.restore_schedule_stage({"stage_name": "full"}, make_trainer())

    def test_single_stage_restore_leaves_net_and_index_untouched(self):
        c = make_controller(single_stage())
        c.restore_schedule_stage({"stage_index": 0, "stage_name": "fit"}, make_trainer())
        assert c.current_stage_index == 0


class TestPendingEarlyAdvance:
    def test_cap_and_flag_driven_advance(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        assert c.next_stage_index_early_stop(1) == 0
        assert c.next_stage_index_early_stop(2) == 1
        c.mark_pending_early_advance()
        assert c.pending_early_advance is True
        assert c.next_stage_index_early_stop(1) == 1

    def test_final_stage_never_advances_and_flag_resets_on_entry(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.mark_pending_early_advance()
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.next_stage_index_early_stop(99) == 1
        assert c.pending_early_advance is False


class TestEvaluateEarlyStop:
    def test_stage_without_early_stop_never_raises_or_stops(self):
        c = make_controller(two_stage_es())
        assert c.evaluate_early_stop(0.5) is False
        assert c.evaluate_early_stop(None) is False

    def test_missing_monitor_raises_after_advancing_to_a_scored_stage(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        with pytest.raises(ConfigError, match="absent from trainer.callback_metrics"):
            c.evaluate_early_stop(None)

    def test_patience_exhaustion_then_an_improvement_resets_wait_count(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.evaluate_early_stop(1.0) is False
        assert c.evaluate_early_stop(1.0) is False
        assert c.evaluate_early_stop(1.0) is True
        assert c.evaluate_early_stop(0.5) is False
        assert c.early_stop_tracker.wait_count == 0


class TestActiveOptimConfig:
    def test_single_stage_returns_top_level_lrs_by_identity(self):
        c = make_controller(single_stage())
        lrs, optimizer_name = c.active_optim_config()
        assert lrs is LRS
        assert optimizer_name == OPTIMIZER_NAME

    def test_stage_zero_overrides_lrs_and_optimizer(self):
        c = make_controller(two_stage_es())
        lrs, optimizer_name = c.active_optim_config()
        assert lrs == {**LRS, "max": 5e-3}
        assert optimizer_name == "lion"

    def test_stage_one_falls_back_to_top_level(self):
        c = make_controller(two_stage_es())
        c.current_stage_index = 1
        lrs, optimizer_name = c.active_optim_config()
        assert lrs is LRS
        assert optimizer_name == OPTIMIZER_NAME


class TestStageTotalSteps:
    def test_single_stage_uses_the_whole_run_estimate(self):
        c = make_controller(single_stage())
        assert c.stage_total_steps(make_trainer()) == 40

    def test_no_early_stop_multi_stage_uses_the_arithmetic_allocation(self):
        schedule = two_stage_no_es()
        allocations = schedule.stage_step_allocations(40, 4)
        c = make_controller(schedule)
        t = make_trainer()
        assert c.stage_total_steps(t) == allocations[0]
        c.current_stage_index = 1
        assert c.stage_total_steps(t) == allocations[1]

    def test_early_stop_stage_zero_uses_its_epoch_budget(self):
        c = make_controller(two_stage_es())
        assert c.stage_total_steps(make_trainer()) == 20

    def test_early_stop_final_stage_uses_the_actual_start_epoch(self):
        c = make_controller(two_stage_es())
        c.current_stage_index = 1
        c.stage_start_epoch = 2
        assert c.stage_total_steps(make_trainer()) == 20
        c.stage_start_epoch = 1
        assert c.stage_total_steps(make_trainer()) == 30
