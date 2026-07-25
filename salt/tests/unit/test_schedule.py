"""W2 schema + instantiation-time validation for `training_schedule` (plan D1).

Covers gate **G2c** (an unknown module name in a freeze spec → `ConfigError` at
`SaltModule.__init__`) plus the rest of the fail-loud parse/validate surface:
`frozen` XOR `trainable`, empty schedule, duplicate `order`, epoch allocation,
and stage ordering. All CPU-safe — no DataLoader is built.
"""

from __future__ import annotations

import pytest

from salt.graph.errors import ConfigError
from salt.model.saltmodule import SaltModule
from salt.schedule import (
    EarlyStopConfig,
    EarlyStopTracker,
    LRSchedulerConfig,
    StageConfig,
    TrainingSchedule,
    boundary_record,
)
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, write_parity_norm_dict

LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}
MODULE_NAMES = ("norm", "track_embed", "encoder", "pool", "jets_classification")


@pytest.fixture
def norm_dict(tmp_path):
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


class TestParse:
    def test_single_stage_defaults(self):
        sched = TrainingSchedule.from_config({"stages": {"fit": {}}}, MODULE_NAMES)
        assert not sched.is_multi_stage
        assert sched.initial_stage.name == "fit"
        assert sched.frozen_names(sched.initial_stage) == set()  # default frozen=()

    def test_frozen_resolves_directly(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"frozen": ["norm", "encoder"]}}}, MODULE_NAMES
        )
        assert sched.frozen_names(sched.initial_stage) == {"norm", "encoder"}

    def test_trainable_resolves_to_complement(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"trainable": ["jets_classification"]}}}, MODULE_NAMES
        )
        expect = set(MODULE_NAMES) - {"jets_classification"}
        assert sched.frozen_names(sched.initial_stage) == expect

    def test_per_stage_optimizer_and_lrs_captured(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"optimizer": "lion", "lrs": {"max": 5e-4}}}}, MODULE_NAMES
        )
        assert sched.initial_stage.optimizer == "lion"
        assert sched.initial_stage.lrs == {"max": 5e-4}

    def test_declaration_order_preserved(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 1}, "b": {"epochs": 1}, "c": {}}}, MODULE_NAMES
        )
        assert [s.name for s in sched.stages] == ["a", "b", "c"]

    def test_explicit_order_pins_position(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"order": 2, "epochs": 1}, "b": {"order": 1, "epochs": 1}, "c": {}}},
            MODULE_NAMES,
        )
        # b (order 1) before a (order 2); c is unpinned → declaration index 2
        assert [s.name for s in sched.stages] == ["b", "a", "c"]


class TestValidation:
    def test_not_a_mapping(self):
        with pytest.raises(ConfigError, match="must be a mapping with a 'stages' key"):
            TrainingSchedule.from_config({"nope": {}}, MODULE_NAMES)

    def test_empty_schedule(self):
        with pytest.raises(ConfigError, match="is empty"):
            TrainingSchedule.from_config({"stages": {}}, MODULE_NAMES)

    def test_null_stage_dropped(self):
        # a `stage: null` override deletes it; the rest remain valid
        sched = TrainingSchedule.from_config(
            {"stages": {"drop_me": None, "fit": {}}}, MODULE_NAMES
        )
        assert [s.name for s in sched.stages] == ["fit"]

    def test_both_freeze_keys_rejected(self):
        with pytest.raises(ConfigError, match="BOTH 'frozen' and 'trainable'"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"frozen": ["norm"], "trainable": ["encoder"]}}}, MODULE_NAMES
            )

    def test_unknown_frozen_name(self):
        with pytest.raises(ConfigError, match=r"unknown module\(s\) \['ghost'\]"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"frozen": ["ghost"]}}}, MODULE_NAMES
            )

    def test_unknown_trainable_name(self):
        with pytest.raises(ConfigError, match=r"unknown module\(s\)"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"trainable": ["ghost"]}}}, MODULE_NAMES
            )

    def test_unknown_stage_key(self):
        with pytest.raises(ConfigError, match="unknown key"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"frozn": ["norm"]}}}, MODULE_NAMES
            )

    def test_frozen_must_be_list(self):
        with pytest.raises(ConfigError, match="must be a list of module names"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"frozen": "norm"}}}, MODULE_NAMES
            )

    def test_duplicate_explicit_order(self):
        with pytest.raises(ConfigError, match="duplicate explicit 'order'"):
            TrainingSchedule.from_config(
                {"stages": {"a": {"order": 1, "epochs": 1}, "b": {"order": 1}}}, MODULE_NAMES
            )

    def test_non_integer_epochs(self):
        with pytest.raises(ConfigError, match="'epochs' must be an integer"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"epochs": 1.5}}}, MODULE_NAMES
            )

    def test_non_positive_epochs(self):
        with pytest.raises(ConfigError, match="positive integer"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"epochs": 0}}}, MODULE_NAMES
            )


class TestEpochAllocation:
    def test_final_stage_takes_remainder(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 3}, "b": {}}}, MODULE_NAMES
        )
        sched.validate_epochs(5)  # 3 + remainder(2) == 5, ok

    def test_over_allocation_with_open_final(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 5}, "b": {}}}, MODULE_NAMES
        )
        with pytest.raises(ConfigError, match="over-allocates epochs"):
            sched.validate_epochs(5)  # 5 explicit, no remainder for b

    def test_over_allocation_all_explicit(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 4}, "b": {"epochs": 4}}}, MODULE_NAMES
        )
        with pytest.raises(ConfigError, match="over-allocates epochs"):
            sched.validate_epochs(5)

    def test_non_final_stage_must_have_epochs(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {}, "b": {"epochs": 2}}}, MODULE_NAMES
        )
        with pytest.raises(ConfigError, match="only the final stage may omit"):
            sched.validate_epochs(5)

    def test_infinite_training_single_stage_skips_check(self):
        # a single (desugared/only) stage has no per-stage allocation to do, so
        # infinite max_epochs is fine (matches the plain-training semantics).
        sched = TrainingSchedule.from_config({"stages": {"fit": {}}}, MODULE_NAMES)
        sched.validate_epochs(-1)  # max_epochs=-1 (infinite) → nothing to bound
        sched.validate_epochs(None)

    def test_infinite_training_multi_stage_rejected(self):
        # W3: epoch-delimited multi-stage training needs a finite max_epochs to
        # allocate per-stage epochs/steps — infinite is a hard error.
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 99}, "b": {}}}, MODULE_NAMES
        )
        for infinite in (-1, None):
            with pytest.raises(ConfigError, match="finite positive max_epochs"):
                sched.validate_epochs(infinite)


class TestDesugarLegacy:
    """The single-`fit`-stage schedule a plain config desugars to (plan D2)."""

    def test_desugar_is_single_non_freezing_fit_stage(self):
        sched = TrainingSchedule.desugar_legacy(MODULE_NAMES)
        assert not sched.is_multi_stage
        assert not sched.has_freezing
        assert sched.initial_stage == StageConfig(name="fit")
        assert sched.frozen_names(sched.initial_stage) == set()


class TestStepAllocation:
    """Per-stage OneCycle step allocation from estimated_stepping_batches
    (Gotcha #1: never the whole-run figure for a sub-stage)."""

    def test_single_stage_returns_total_unchanged(self):
        sched = TrainingSchedule.desugar_legacy(MODULE_NAMES)
        assert sched.stage_step_allocations(1234, 10) == [1234]

    def test_multi_stage_allocations_sum_to_total(self):
        # 2 stages, warmup=2 epochs of 10, so ~1/5 of the steps go to warmup.
        sched = TrainingSchedule.from_config(
            {"stages": {"warmup": {"epochs": 2}, "full": {}}}, MODULE_NAMES
        )
        allocs = sched.stage_step_allocations(1000, 10)
        assert sum(allocs) == 1000  # exact — final stage takes the remainder
        assert allocs == [200, 800]

    def test_three_stage_allocation_rounds_and_sums(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 3}, "b": {"epochs": 3}, "c": {}}}, MODULE_NAMES
        )
        allocs = sched.stage_step_allocations(101, 10)  # non-divisible on purpose
        assert sum(allocs) == 101
        assert len(allocs) == 3
        # boundaries: round(101*3/10)=30, round(101*6/10)=61 -> [30, 31, 40]
        assert allocs == [30, 31, 40]


class TestStageIndexForEpoch:
    def test_epoch_maps_to_owning_stage(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 2}, "b": {"epochs": 3}, "c": {}}}, MODULE_NAMES
        )
        # a owns [0,2), b owns [2,5), c owns [5, max)
        got = [sched.stage_index_for_epoch(e, 10) for e in range(10)]
        assert got == [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]


class TestChangesFreezeAcrossStages:
    def test_true_when_freeze_set_differs(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"warmup": {"epochs": 2, "frozen": ["encoder"]}, "full": {"frozen": []}}},
            MODULE_NAMES,
        )
        assert sched.changes_freeze_across_stages()

    def test_false_when_freeze_set_constant(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 2, "frozen": ["encoder"]}, "b": {"frozen": ["encoder"]}}},
            MODULE_NAMES,
        )
        assert not sched.changes_freeze_across_stages()

    def test_false_for_single_stage(self):
        assert not TrainingSchedule.desugar_legacy(MODULE_NAMES).changes_freeze_across_stages()


class TestSaltModuleInstantiation:
    """G2c + schedule storage at SaltModule.__init__ (no trainer, no DataLoader)."""

    def test_unknown_module_name_errors_at_init(self, norm_dict):
        modules = build_gn2v2_modules(norm_dict)
        with pytest.raises(ConfigError, match=r"unknown module\(s\) \['nonexistent'\]"):
            SaltModule(
                modules, lrs=LRS,
                training_schedule={"stages": {"fit": {"frozen": ["nonexistent"]}}},
            )

    def test_no_schedule_desugars_to_single_fit_stage(self, norm_dict):
        # plan D2: a plain config (no training_schedule) desugars to one `fit`
        # stage that overrides neither lrs nor optimizer nor freezes anything.
        model = SaltModule(build_gn2v2_modules(norm_dict), lrs=LRS)
        sched = model._schedule  # noqa: SLF001
        assert sched is not None
        assert not sched.is_multi_stage
        assert sched.initial_stage.name == "fit"
        assert not sched.has_freezing
        assert sched.frozen_names(sched.initial_stage) == set()
        assert model._frozen_module_names == set()  # noqa: SLF001
        assert model._current_stage_index == 0  # noqa: SLF001

    def test_valid_single_stage_stored(self, norm_dict):
        model = SaltModule(
            build_gn2v2_modules(norm_dict), lrs=LRS,
            training_schedule={"stages": {"fit": {"frozen": ["encoder"]}}},
        )
        assert model._schedule is not None  # noqa: SLF001
        assert not model._schedule.is_multi_stage  # noqa: SLF001

    def test_multi_stage_accepted_at_init(self, norm_dict):
        # multi-stage passes instantiation validation (rejection is at fit, W3)
        model = SaltModule(
            build_gn2v2_modules(norm_dict), lrs=LRS,
            training_schedule={
                "stages": {"warmup": {"epochs": 2, "frozen": ["encoder"]}, "full": {}}
            },
        )
        assert model._schedule.is_multi_stage  # noqa: SLF001

    def test_schedule_names_validate_against_model_modules_only(self, norm_dict):
        # 'loss'/'concat'/'split' ARE model.modules keys and so are valid targets
        model = SaltModule(
            build_gn2v2_modules(norm_dict), lrs=LRS,
            training_schedule={"stages": {"fit": {"frozen": ["concat", "split"]}}},
        )
        assert model._schedule.frozen_names(model._schedule.initial_stage) == {  # noqa: SLF001
            "concat", "split"
        }


def test_stageconfig_is_frozen_dataclass():
    stage = StageConfig(name="fit")
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass rejects any mutation
        stage.epochs = 3  # type: ignore[misc]


# --- W7: per-stage early stopping -------------------------------------------


class TestEarlyStopParse:
    def test_absent_by_default(self):
        sched = TrainingSchedule.from_config({"stages": {"fit": {}}}, MODULE_NAMES)
        assert sched.initial_stage.early_stop is None
        assert not sched.has_early_stop

    def test_minimal_defaults(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"early_stop": {"monitor": "val/loss"}}}}, MODULE_NAMES
        )
        es = sched.initial_stage.early_stop
        assert es == EarlyStopConfig(monitor="val/loss", mode="min", patience=3, min_delta=0.0)
        assert es.check_finite is True
        assert sched.has_early_stop

    def test_all_fields(self):
        sched = TrainingSchedule.from_config(
            {
                "stages": {
                    "fit": {
                        "early_stop": {
                            "monitor": "val/acc", "mode": "max", "patience": 5,
                            "min_delta": 0.01, "check_finite": False,
                        }
                    }
                }
            },
            MODULE_NAMES,
        )
        es = sched.initial_stage.early_stop
        assert (es.monitor, es.mode, es.patience, es.min_delta, es.check_finite) == (
            "val/acc", "max", 5, 0.01, False
        )

    def test_has_early_stop_true_if_any_stage(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 1}, "b": {"early_stop": {"monitor": "val/loss"}}}},
            MODULE_NAMES,
        )
        assert sched.has_early_stop

    def test_missing_monitor_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.monitor' is required"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"patience": 2}}}}, MODULE_NAMES
            )

    def test_empty_monitor_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.monitor' is required"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "  "}}}}, MODULE_NAMES
            )

    def test_bad_mode_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.mode' must be 'min' or 'max'"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "val/loss", "mode": "lower"}}}},
                MODULE_NAMES,
            )

    def test_non_positive_patience_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.patience' must be a positive integer"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "val/loss", "patience": 0}}}},
                MODULE_NAMES,
            )

    def test_bool_patience_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.patience' must be a positive integer"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "val/loss", "patience": True}}}},
                MODULE_NAMES,
            )

    def test_negative_min_delta_rejected(self):
        with pytest.raises(ConfigError, match="early_stop.min_delta' must be a non-negative"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "val/loss", "min_delta": -0.1}}}},
                MODULE_NAMES,
            )

    def test_unknown_early_stop_key_rejected(self):
        with pytest.raises(ConfigError, match="early_stop' has unknown key"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": {"monitor": "val/loss", "paticence": 3}}}},
                MODULE_NAMES,
            )

    def test_early_stop_not_a_mapping_rejected(self):
        with pytest.raises(ConfigError, match="'early_stop' must be a mapping"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"early_stop": ["val/loss"]}}}, MODULE_NAMES
            )


class TestEarlyStopTracker:
    def test_first_check_seeds_best_never_stops(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/loss", patience=2))
        assert t.check(1.0) is False
        assert t.best_score == 1.0
        assert t.wait_count == 0
        assert t.check_count == 1

    def test_patience_exhausted_min_mode(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/loss", mode="min", patience=2))
        assert t.check(1.0) is False  # best=1.0
        assert t.check(1.0) is False  # no improvement, wait=1
        assert t.check(1.0) is True  # wait=2 == patience → stop

    def test_improvement_resets_wait(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/loss", mode="min", patience=2))
        t.check(1.0)
        t.check(1.0)  # wait=1
        assert t.check(0.5) is False  # improvement → wait reset
        assert t.wait_count == 0
        assert t.best_score == 0.5

    def test_max_mode(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/acc", mode="max", patience=1))
        assert t.check(0.5) is False
        assert t.check(0.6) is False  # improvement
        assert t.check(0.6) is True  # no improvement, wait=1 == patience

    def test_min_delta_requires_meaningful_improvement(self):
        cfg = EarlyStopConfig(monitor="val/loss", mode="min", patience=1, min_delta=0.1)
        t = EarlyStopTracker(cfg)
        t.check(1.0)
        # 0.95 is lower but not by min_delta=0.1 → not an improvement → stop at patience 1
        assert t.check(0.95) is True

    def test_check_finite_stops_on_nan(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/loss", patience=99, check_finite=True))
        t.check(1.0)
        assert t.check(float("nan")) is True

    def test_check_finite_disabled_ignores_nan(self):
        t = EarlyStopTracker(EarlyStopConfig(monitor="val/loss", patience=1, check_finite=False))
        t.check(1.0)
        # nan is not < best - 0 → no improvement → wait=1 == patience → stop (but not via finite)
        assert t.check(float("nan")) is True

    def test_state_dict_round_trip(self):
        cfg = EarlyStopConfig(monitor="val/loss", mode="min", patience=3, min_delta=0.02)
        t = EarlyStopTracker(cfg)
        t.check(1.0)
        t.check(0.9)
        t.check(0.9)  # wait=1
        state = t.state_dict()
        assert state["best_score"] == 0.9
        assert state["wait_count"] == 1
        assert state["check_count"] == 3
        assert state["fingerprint"] == cfg.fingerprint()
        restored = EarlyStopTracker.from_state_dict(cfg, state)
        assert (restored.best_score, restored.wait_count, restored.check_count) == (0.9, 1, 3)
        # continuing from the restored counters matches an uninterrupted run
        assert restored.check(0.9) is False  # wait=2
        assert restored.check(0.9) is True  # wait=3 == patience

    def test_fingerprint_captures_criterion(self):
        cfg = EarlyStopConfig(monitor="val/loss", mode="max", patience=7, min_delta=0.5)
        assert cfg.fingerprint() == {
            "monitor": "val/loss", "mode": "max", "min_delta": 0.5, "patience": 7
        }


class TestBoundaryRecord:
    def test_encodes_all_fields(self):
        rec = boundary_record("full", 1, 250, 5, "early_stop")
        assert rec == {
            "stage_name": "full", "stage_index": 1, "global_step": 250,
            "epoch": 5, "reason": "early_stop",
        }


# --- W7: per-stage scoped callbacks (schema) --------------------------------


class TestStageCallbacksParse:
    def test_absent_by_default(self):
        sched = TrainingSchedule.from_config({"stages": {"fit": {}}}, MODULE_NAMES)
        assert sched.initial_stage.callbacks is None
        assert not sched.has_stage_callbacks

    def test_parsed_specs(self):
        sched = TrainingSchedule.from_config(
            {
                "stages": {
                    "fit": {
                        "callbacks": [
                            {"class_path": "pkg.A"},
                            {"class_path": "pkg.B", "init_args": {"x": 1}},
                        ]
                    }
                }
            },
            MODULE_NAMES,
        )
        assert sched.has_stage_callbacks
        specs = sched.initial_stage.callbacks
        assert len(specs) == 2
        assert specs[0]["class_path"] == "pkg.A"
        assert specs[1]["init_args"] == {"x": 1}

    def test_has_stage_callbacks_true_if_any_stage(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 1}, "b": {"callbacks": [{"class_path": "pkg.A"}]}}},
            MODULE_NAMES,
        )
        assert sched.has_stage_callbacks

    def test_not_a_list_rejected(self):
        with pytest.raises(ConfigError, match="'callbacks' must be a list"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"callbacks": {"class_path": "pkg.A"}}}}, MODULE_NAMES
            )

    def test_item_not_a_mapping_rejected(self):
        with pytest.raises(ConfigError, match=r"callbacks\[0\] must be a mapping"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"callbacks": ["pkg.A"]}}}, MODULE_NAMES
            )

    def test_missing_class_path_rejected(self):
        with pytest.raises(ConfigError, match="needs a non-empty string 'class_path'"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"callbacks": [{"init_args": {"x": 1}}]}}}, MODULE_NAMES
            )

    def test_unknown_spec_key_rejected(self):
        with pytest.raises(ConfigError, match=r"callbacks\[0\] has unknown key"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"callbacks": [{"class_path": "pkg.A", "args": {}}]}}},
                MODULE_NAMES,
            )

    def test_init_args_not_a_mapping_rejected(self):
        with pytest.raises(ConfigError, match="'init_args' must be a mapping"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"callbacks": [{"class_path": "pkg.A", "init_args": [1]}]}}},
                MODULE_NAMES,
            )


# --- W8: per-stage lr_scheduler (schema) ------------------------------------

_COSINE = "torch.optim.lr_scheduler.CosineAnnealingLR"
_PLATEAU = "torch.optim.lr_scheduler.ReduceLROnPlateau"


class TestLRSchedulerParse:
    def test_absent_by_default(self):
        sched = TrainingSchedule.from_config({"stages": {"fit": {}}}, MODULE_NAMES)
        assert sched.initial_stage.lr_scheduler is None
        assert not sched.has_lr_scheduler

    def test_minimal_defaults(self):
        spec = {"class_path": _COSINE, "init_args": {"T_max": 5}}
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"lr_scheduler": spec}}}, MODULE_NAMES
        )
        cfg = sched.initial_stage.lr_scheduler
        assert isinstance(cfg, LRSchedulerConfig)
        assert cfg.class_path == _COSINE
        assert cfg.init_args == {"T_max": 5}
        assert cfg.interval == "epoch"  # default
        assert cfg.frequency == 1
        assert cfg.monitor is None
        assert sched.has_lr_scheduler

    def test_all_fields(self):
        sched = TrainingSchedule.from_config(
            {
                "stages": {
                    "fit": {
                        "lr_scheduler": {
                            "class_path": _PLATEAU,
                            "init_args": {"mode": "min", "patience": 2},
                            "interval": "epoch",
                            "frequency": 2,
                            "monitor": "val/loss",
                        }
                    }
                }
            },
            MODULE_NAMES,
        )
        cfg = sched.initial_stage.lr_scheduler
        assert (cfg.interval, cfg.frequency, cfg.monitor) == ("epoch", 2, "val/loss")

    def test_has_lr_scheduler_true_if_any_stage(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 1}, "b": {"lr_scheduler": {"class_path": _COSINE}}}},
            MODULE_NAMES,
        )
        assert sched.has_lr_scheduler

    def test_missing_class_path_rejected(self):
        with pytest.raises(ConfigError, match="lr_scheduler.class_path' is required"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"init_args": {"T_max": 5}}}}}, MODULE_NAMES
            )

    def test_init_args_optimizer_rejected(self):
        with pytest.raises(ConfigError, match="must not set 'optimizer'"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE,
                 "init_args": {"optimizer": "x"}}}}}, MODULE_NAMES,
            )

    def test_bad_interval_rejected(self):
        with pytest.raises(ConfigError, match="lr_scheduler.interval' must be 'epoch' or 'step'"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE, "interval": "batch"}}}},
                MODULE_NAMES,
            )

    def test_non_positive_frequency_rejected(self):
        with pytest.raises(ConfigError, match="lr_scheduler.frequency' must be a positive integer"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE, "frequency": 0}}}},
                MODULE_NAMES,
            )

    def test_unknown_key_rejected(self):
        with pytest.raises(ConfigError, match="lr_scheduler' has unknown key"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE, "gamma": 0.1}}}},
                MODULE_NAMES,
            )

    def test_not_a_mapping_rejected(self):
        with pytest.raises(ConfigError, match="'lr_scheduler' must be a mapping"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": [_COSINE]}}}, MODULE_NAMES
            )

    def test_onecycle_lrs_key_clash_rejected(self):
        # a stage's OWN lrs override may not set OneCycle-only keys alongside a
        # custom lr_scheduler (contradiction) — initial/weight_decay stay allowed.
        with pytest.raises(ConfigError, match="OneCycle-only 'lrs' key"):
            TrainingSchedule.from_config(
                {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE},
                 "lrs": {"max": 1e-3}}}}, MODULE_NAMES,
            )

    def test_initial_lrs_key_allowed_with_scheduler(self):
        # `initial` (optimizer base LR) stays meaningful with a custom scheduler
        sched = TrainingSchedule.from_config(
            {"stages": {"fit": {"lr_scheduler": {"class_path": _COSINE},
             "lrs": {"initial": 1e-4, "weight_decay": 1e-5}}}}, MODULE_NAMES,
        )
        assert sched.initial_stage.lr_scheduler.class_path == _COSINE
