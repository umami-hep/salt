"""Unit tests for TrainingController."""

# Guarantee -> test: N lazy/once/rank-0 broadcast, loader length read only for frac_samples,
#   boundary + checkpoint -> TestSampleStages; stop_after -> TestStopAfter; TestFreezeAccumulators.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from salt.graph.errors import ConfigError
from salt.model.multistage_training import TrainingController
from salt.schedule import EarlyStopTracker, TrainingSchedule, boundary_record

MODULES = ("a", "b")
LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}
OPTIMIZER_NAME = "AdamW"
SAMPLE = {"num_samples": 300, "frozen": ["a"]}


def make_net() -> nn.ModuleDict:
    return nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)})


class _FakeStrategy:
    broadcasts = 0
    setup_optimizers = staticmethod(lambda trainer: None)
    reduce_boolean_decision = staticmethod(lambda decision, all=True: decision)  # noqa: A002

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        self.broadcasts += 1
        return obj


def make_trainer(**over: Any) -> SimpleNamespace:
    defaults = {
        "max_epochs": 4,
        "strategy": _FakeStrategy(),
        "datamodule": SimpleNamespace(batch_size=100),
        "world_size": 1,
        "accumulate_grad_batches": 1,
        "num_training_batches": 10,
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
        assert c.next_stage_index(1) == 0
        assert c.next_stage_index(2) == 1
        c.pending_early_advance = True  # the callback sets it on a non-final trigger
        assert c.next_stage_index(1) == 1

    def test_final_stage_never_advances_and_flag_resets_on_entry(self):
        c = make_controller(two_stage_es())
        t = make_trainer()
        c.preflight(t)
        c.start_fit(t)
        c.pending_early_advance = True
        c.advance_to_stage(1, 10, 2, "epochs")
        assert c.next_stage_index(99) == 1
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


def sched(prior: dict[str, Any], **top: Any) -> TrainingSchedule:
    return TrainingSchedule.from_config({"stages": {"prior": prior, "main": {}}, **top}, MODULES)


def started(schedule: TrainingSchedule) -> TrainingController:
    c, t = make_controller(schedule), make_trainer()
    c.preflight(t)
    c.start_fit(t)
    return c


class _NoLoaderLength(SimpleNamespace):
    @property
    def num_training_batches(self) -> float:
        raise AssertionError("num_training_batches read for a num_samples-only schedule")


class TestSampleStages:
    def test_resolve_num_samples_once_and_never_reads_the_loader_length(self):
        c = make_controller(sched(SAMPLE))
        assert (c.stage_start_step, c.stage_step_targets, c.stop_after_record) == (0, None, None)
        assert c.next_stage_index(99) == 0  # a sample stage never advances by epoch
        with pytest.raises(RuntimeError, match="sample-stage step targets not resolved"):
            c.step_target(0)
        over = {k: v for k, v in vars(make_trainer()).items() if k != "num_training_batches"}
        c.resolve_step_targets(t := _NoLoaderLength(**over))
        assert c.stage_step_targets == {0: 3}  # ceil(300 / 100)
        assert t.strategy.broadcasts == 1
        c.resolve_step_targets(make_trainer(datamodule=SimpleNamespace(batch_size=1)))
        assert c.stage_step_targets == {0: 3}  # idempotent
        c.start_fit(make_trainer())
        assert c.stage_step_targets == {0: 3}  # start_fit keeps the resolved targets

    def test_resolve_frac_samples_reads_a_finite_loader_length(self):
        c = make_controller(sched({"frac_samples": 0.1}))
        for length in (float("inf"), None, 0):
            with pytest.raises(ConfigError, match="needs a finite train loader length"):
                c.resolve_step_targets(make_trainer(num_training_batches=length))
        c.resolve_step_targets(make_trainer(num_training_batches=37, accumulate_grad_batches=2))
        assert c.step_target(0) == 2  # steps_per_epoch = ceil(37 / 2) = 19 -> ceil(1.9)

    def test_advance_by_steps_total_steps_and_checkpoint_round_trip(self):
        c = started(sched(SAMPLE))
        assert c.stage_total_steps(make_trainer()) == 3  # the sample stage runs exactly N
        c.advance_to_stage(1, 3, 0, "steps")
        assert c.stage_total_steps(make_trainer()) == 37  # the rest takes the remainder
        assert (c.stage_start_step, c.stage_start_epoch, c.frozen_module_names) == (3, 0, set())
        assert c.boundary_records == [boundary_record("main", 1, 3, 0, "steps")]
        state = c.checkpoint_state()
        assert set(state) == {
            "stage_index", "stage_name", "stage_start_epoch", "boundaries", "stage_start_step"
        }
        restored = started(sched(SAMPLE))
        restored.restore_schedule_stage(state, make_trainer())
        assert (restored.current_stage_index, restored.stage_start_step) == (1, 3)
        assert restored.boundary_records == state["boundaries"]
        del state["stage_start_step"]
        with pytest.raises(ConfigError, match="missing 'stage_start_step'"):
            make_controller(sched(SAMPLE)).restore_schedule_stage(state, make_trainer())

    def test_resumed_epoch_stage_after_sample_stop_is_sized_from_its_actual_start(self):
        """A sample stop_after resume enters the next epoch stage at an epoch start."""
        raw = {"stages": {"prior": SAMPLE, "mid": {"epochs": 1}, "main": {}}, "stop_after": "prior"}
        t = make_trainer(max_epochs=3, estimated_stepping_batches=30)  # N=3, 10 steps/epoch
        c = started(TrainingSchedule.from_config(raw, MODULES))
        assert c.stage_total_steps(t) == 3
        c.advance_to_stage(1, 3, 0, "steps")
        assert c.stage_total_steps(t) == 7  # the rest of epoch 0
        c.epoch_start_step = 10
        c.advance_to_stage(2, 10, 1, "epochs")
        assert c.stage_total_steps(t) == 20
        c = started(TrainingSchedule.from_config(raw, MODULES))
        c.record_stop_after(3, 0)
        resumed = started(TrainingSchedule.from_config(raw, MODULES))
        resumed.restore_schedule_stage(c.checkpoint_state(), make_trainer())
        assert resumed.epoch_start_step is None
        resumed.epoch_start_step = 3  # set by the callback at epoch 1's start
        resumed.advance_to_stage(1, 3, 1, "steps")
        assert resumed.stage_total_steps(t) == 10  # a full epoch, not 7
        resumed.epoch_start_step = 13
        resumed.advance_to_stage(2, 13, 2, "epochs")
        assert resumed.stage_total_steps(t) == 10

    def test_sample_targets_exhausting_the_run_raise_before_training(self):
        c = make_controller(sched({"frac_samples": 0.1}))
        t = make_trainer(max_epochs=1, num_training_batches=1, estimated_stepping_batches=1)
        with pytest.raises(ConfigError, match="leaving the final stage"):
            c.stage_total_steps(t)


class TestStopAfter:
    def test_payload_records_not_reached_then_reached(self):
        c = started(sched({"epochs": 2, "frozen": ["a"]}, stop_after="prior"))
        blk = {"stage": "prior", "reached": False, "boundary": None}
        assert c.checkpoint_state() == {"stage_index": 0, "stage_name": "prior", "stop_after": blk}
        rec = c.record_stop_after(10, 1)
        assert rec == c.stop_after_record == boundary_record("prior", 0, 10, 1, "stop_after")
        assert c.boundary_records == []  # no early_stop switch: not a completed boundary
        reached = {**blk, "reached": True, "boundary": rec}
        assert c.checkpoint_state()["stop_after"] == reached

    def test_record_is_never_restored_start_fit_clears_it_and_early_stop_appends_it(self):
        c = started(sched({"epochs": 2}, stop_after="prior"))
        c.record_stop_after(10, 1)
        resumed = started(sched({"epochs": 2}, stop_after="prior"))
        resumed.restore_schedule_stage(c.checkpoint_state(), make_trainer())
        assert resumed.stop_after_record is None
        assert resumed.checkpoint_state()["stop_after"]["reached"] is False
        c.start_fit(make_trainer())
        assert c.stop_after_record is None
        c = started(sched({"epochs": 2, "early_stop": {"monitor": "val/loss"}}, stop_after="prior"))
        assert c.boundary_records == [c.record_stop_after(10, 1)]

    @pytest.mark.parametrize(
        ("prior", "stop_epoch", "pending"),
        [(SAMPLE, 0, True), ({"epochs": 3}, 0, True), ({"epochs": 3}, 2, False)],
    )
    def test_resume_after_an_early_stopped_stop_stage_rearms_the_advance(
        self, prior, stop_epoch, pending
    ):
        """An early_stop-ended stop_after stage advances on resume; a cap-ended one by cap."""
        es = {"monitor": "val/loss", "patience": 1}
        c = started(sched({**prior, "early_stop": es}, stop_after="prior"))
        c.record_stop_after(2, stop_epoch)
        assert c.current_stage_index == 0
        resumed = make_controller(sched({**prior, "early_stop": es}, stop_after="prior"))
        resumed.restore_schedule_stage(c.checkpoint_state(), make_trainer())
        assert resumed.pending_early_advance is pending
        assert resumed.next_stage_index(stop_epoch + 1) == 1


class _Accumulator(nn.Module):
    def __init__(self, *, freezes: bool = True) -> None:
        super().__init__()
        self.freezes, self.calls, self.accumulators_frozen = freezes, [], False

    def freeze_accumulators(self, global_step: int, epoch: int) -> None:
        self.calls.append((global_step, epoch))
        self.accumulators_frozen = self.freezes


class TestFreezeAccumulators:
    def test_boundary_freezes_accumulators_and_rejects_one_left_unfrozen(self):
        c = started(two_stage_no_es())
        c.net["acc"] = acc = _Accumulator()
        c.advance_to_stage(1, 10, 2, "epochs")
        assert acc.calls == [(10, 2)]
        c.net["acc"] = _Accumulator(freezes=False)
        with pytest.raises(RuntimeError, match="unfrozen accumulators"):
            c.advance_to_stage(1, 10, 2, "epochs")
