"""Parse-level validation of the shipped fine-tuning overlay configs.

The `finetune_gn3large*.yaml` templates are referenced by
`docs/tutorials/finetuning.md`. These tests prove the fit parser accepts them —
the `training_schedule:` block parses + validates its epoch allocation, its
freeze spec names real modules of the GN3 base it targets, and (for the
new-head overlay) the added module's `class_path` resolves. No model is
instantiated, no data or checkpoint is touched, so the placeholder paths in the
templates need not exist.
"""

from __future__ import annotations

import importlib

import yaml

from salt.main import CONFIG_DIR
from salt.schedule import TrainingSchedule

# GN3V00 is the in-repo, production-faithful GN3Large stand-in the overlays
# target (same five task heads); its module names are the freeze-spec universe.
BASE_CFG = CONFIG_DIR / "GN3/GN3V00.yaml"


def _base_module_names() -> list[str]:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    return list(cfg["model"]["init_args"]["modules"])


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.is_file(), f"shipped tutorial config missing: {path}"
    return yaml.safe_load(path.read_text())


def _resolve(class_path: str) -> type:
    module_path, _, cls_name = class_path.rpartition(".")
    return getattr(importlib.import_module(module_path), cls_name)


class TestSameHeadsOverlay:
    """`finetune_gn3large.yaml` — worked example A."""

    def test_schedule_parses_and_epochs_valid(self):
        cfg = _load("finetune/finetune_gn3large.yaml")
        names = _base_module_names()
        sched = TrainingSchedule.from_config(cfg["training_schedule"], names)
        # two ordered stages, head_warmup then full_finetune
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        assert sched.is_multi_stage
        # epoch allocation is valid against the overlay's own max_epochs
        sched.validate_epochs(cfg["trainer"]["max_epochs"])

    def test_freeze_spec_names_real_base_modules(self):
        cfg = _load("finetune/finetune_gn3large.yaml")
        names = set(_base_module_names())
        assert "jets_classification" in names  # the flavour head the warm-up trains
        # head_warmup trains only the flavour head → everything else frozen
        sched = TrainingSchedule.from_config(cfg["training_schedule"], list(names))
        warmup = sched.stages[0]
        assert sched.frozen_names(warmup) == names - {"jets_classification"}


class TestNewHeadOverlay:
    """`finetune_gn3large_new_head.yaml` — worked example B."""

    def test_new_module_class_path_resolves(self):
        cfg = _load("finetune/finetune_gn3large_new_head.yaml")
        head = cfg["model"]["init_args"]["modules"]["large_r_jet_classification"]
        cls = _resolve(head["class_path"])
        assert cls.__name__ == "ClassificationTaskModule"
        # the init_args the tutorial documents are present
        init = head["init_args"]
        for key in ("stream", "input", "label", "class_names", "dense"):
            assert key in init

    def test_schedule_references_the_new_head(self):
        cfg = _load("finetune/finetune_gn3large_new_head.yaml")
        # the schedule's freeze universe is the base modules PLUS the added head
        names = [*_base_module_names(), "large_r_jet_classification"]
        sched = TrainingSchedule.from_config(cfg["training_schedule"], names)
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        warmup = sched.stages[0]
        # only the new head trains during warm-up
        assert sched.frozen_names(warmup) == set(names) - {"large_r_jet_classification"}
        sched.validate_epochs(cfg["trainer"]["max_epochs"])
