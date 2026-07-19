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
from salt.schedule import StageConfig, TrainingSchedule
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
