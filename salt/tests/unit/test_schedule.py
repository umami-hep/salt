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

    def test_infinite_training_skips_check(self):
        sched = TrainingSchedule.from_config(
            {"stages": {"a": {"epochs": 99}, "b": {}}}, MODULE_NAMES
        )
        sched.validate_epochs(-1)  # max_epochs=-1 (infinite) → no bound to check
        sched.validate_epochs(None)


class TestSaltModuleInstantiation:
    """G2c + schedule storage at SaltModule.__init__ (no trainer, no DataLoader)."""

    def test_unknown_module_name_errors_at_init(self, norm_dict):
        modules = build_gn2v2_modules(norm_dict)
        with pytest.raises(ConfigError, match=r"unknown module\(s\) \['nonexistent'\]"):
            SaltModule(
                modules, lrs=LRS,
                training_schedule={"stages": {"fit": {"frozen": ["nonexistent"]}}},
            )

    def test_no_schedule_leaves_state_clean(self, norm_dict):
        model = SaltModule(build_gn2v2_modules(norm_dict), lrs=LRS)
        assert model._schedule is None  # noqa: SLF001
        assert model._frozen_module_names == set()  # noqa: SLF001

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
