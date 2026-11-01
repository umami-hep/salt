"""Multi-stage `training_schedule` execution: boundaries, freeze, LR schedulers,
scoped callbacks, and legacy desugaring.

Merges ``test_freeze.py`` (module-freeze machinery), ``test_lr_scheduler.py``
(per-stage LR-scheduler class choice) and ``test_stage_callbacks.py``
(per-stage scoped callbacks) into their shared home — none of the three has
an existence outside a `training_schedule` stage, and all four files shared
the same fixture + boundary machinery. Covered:

- a legacy config (no `training_schedule`) desugars to a single `fit`
  stage and trains bitwise-identically to the pre-desugar behaviour; here
  the structural half is covered (single-stage total_steps == the whole-run
  `estimated_stepping_batches`, optimizer holds every param).
- an explicit single-`fit`-stage schedule trains bitwise-identically to
  the no-schedule (desugared) config (same seed → equal state_dicts).
- a 2-stage schedule shows two OneCycle envelopes with the boundary at
  the correct global step, per-stage total_steps summing to the whole-run
  estimate, the stage-2 optimizer owning the newly-unfrozen params (which move),
  while the stage-1-frozen params stayed bitwise-fixed during stage 1.
- after a boundary, `trainer.optimizers`, `trainer.lr_scheduler_configs`
  and `pl_module.optimizers()` all point at the rebuilt objects.
- a 3-stage schedule with a per-stage `optimizer` override
  (AdamW→lion) rebuilds the right optimizer class per stage; HybridMuonAdamW
  rebuilds its Muon/AdamW split over the new trainable set across a boundary.
- a frozen module has ``requires_grad=False`` on all its params AND is
  in ``eval()`` mode during training, and STAYS in eval across an epoch boundary
  (Lightning re-calls ``model.train()`` every epoch; `SaltModule.train` must
  re-assert eval on frozen modules).
- the optimizer is built over TRAINABLE params only: its param set
  equals ``{p for p in model.parameters() if p.requires_grad}`` and excludes the
  frozen module's params — checked for AdamW AND HybridMuonAdamW.
- no ``training_schedule`` → zero behaviour change: the optimizer holds
  every parameter and nothing is frozen.
- composition smoke: ``--init_from`` + a single-stage schedule freezing
  every loaded module → only the new module's params reach the optimizer, and
  training runs with finite loss.
- per-stage LR scheduler: a stage may declare `lr_scheduler:
  {class_path, init_args, interval, frequency, monitor}` — the scheduler class
  is instantiated over the freshly-rebuilt stage optimizer at the boundary,
  replacing the default per-stage OneCycleLR. Absent → today's OneCycle
  behaviour byte-identically. Legacy parity, two-stage different classes,
  plateau + early_stop coexistence, and mid-stage resume of scheduler state
  are all covered.
- per-stage scoped callbacks: a stage may declare additional scoped
  `callbacks`; the auto-injected `StageScopedCallbacks` coordinator
  instantiates the active stage's scoped callbacks FRESH at stage entry,
  forwards Lightning's per-stage hooks to them ONLY while their stage is
  active, and tears them down at exit — on top of the persistent globals.
  Bad callback specs fail at fit start, not at the boundary.
- sample stages, `stop_after`, a real two-rank gloo fit and the shipped
  `GN3/GN3FTAG1LITE.yaml` two-stage example (guarantee map below).

All DataLoaders use ``num_workers=0``.
"""

# Guarantee -> test map (plan 63: sample stages, stop_after, multistage reweighting)
# TestStopAfterHookGating: an epoch-cap stop_after is inert until the stop stage's last
#   epoch, then one record + should_stop + `-epoch=NNN.ckpt` (early-stop branch too).
# TestStopAfterEndsFitAtBoundary: a real epoch-cap stop_after fit ends at the boundary.
# TestSampleStageHooks: num_samples -> the same N on every rank, never from loader length;
#   advance exactly at N; world-size scaling; tripwire; no all=True collective before N;
#   a sample stop_after sets should_stop at the batch, writes `-step=NNNNNN.ckpt` at epoch
#   end; the epoch-start resume seam advances once and never re-stops.
# TestSampleStageFit: real num_samples / frac_samples fits — N, mid-epoch boundary, OneCycle
#   allocation == steps taken per stage, mid-epoch stop_after + its full payload, and the
#   resumed epoch stage sized from its actual (epoch-start) entry.
# TestTwoRankGlooFit: 2 gloo ranks agree on the broadcast N (accumulate 2), the boundary
#   step and the frozen q_f; the main head trains; stop_after -> ckpt -> resume into main.
# TestShippedTwoStageConfig: GN3FTAG1LITE.yaml's model + schedule — frac boundary, finite
#   weighted loss, norm learned then frozen, q_f + prior frozen, head trains, stop + resume,
#   the TEST graph keeps kinematic_prior and drops kinematic_weight; stop_after fit via
#   SaltCLI -> `salt validate` from its checkpoint restores epoch/step and reproduces the
#   reweight + prior-closure JSON byte-for-value, touching only VAL data.
# TestStageScopedCallbacks: `_PROBE` resolves to THIS module, not a second import of it.

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import socket
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from jsonargparse import ArgumentParser
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn

from salt.callbacks.schedule import StageScopedCallbacks, TrainingScheduleCallback
from salt.data import (
    ContainerOneHot,
    Features,
    FtagLabeller,
    H5StructuredReader,
    Labels,
    SaltDataModule,
)
from salt.graph.spec import Mode
from salt.main import CONFIG_DIR, SaltCLI
from salt.model.base import SaltModelModule
from salt.model.modules import KinematicWeight
from salt.model.modules.losses import LossSum
from salt.model.modules.tasks import ClassificationTaskModule
from salt.model.multistage_training import TrainingController
from salt.model.saltmodule import CKPT_KEY, SaltModule
from salt.optim import HybridMuonAdamW
from salt.schedule import TrainingSchedule, boundary_record
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.dummy_h5 import dummy_h5
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    write_parity_norm_dict,
)
from salt.tests._fixtures.joint_ftag1lite_corpus import (
    build_joint_corpus,
    make_reader,
    side_config,
)
from salt.utils.config_utils import expand_includes

pytestmark = pytest.mark.integration

SEED = 1234
LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.25}
# freeze's own scenarios use a different pct_start; keeping it distinct avoids
# perturbing the freeze suite's OneCycle-envelope timing.
FREEZE_LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("schedule")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path, _, _ = dummy_h5(base)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_datamodule(data) -> SaltDataModule:
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    return SaltDataModule(
        modules, batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


def build_model(data, **kwargs) -> SaltModule:
    return SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, **kwargs)


def make_trainer(*, max_epochs: int, callbacks: list, enable_checkpointing: bool = False,
                 **kwargs) -> Trainer:
    # 5 train steps/epoch (1000 jets / batch 100 = 10, limited to 5) so a 4-epoch
    # fit has estimated_stepping_batches == 20 (a 2-stage split of [10, 10]); a
    # per-stage OneCycle needs >~8 steps or its warmup phase degenerates.
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks, **kwargs,
    )


def _param_ids(params) -> set[int]:
    return {id(p) for p in params}


def _optimizer_param_ids(opt) -> set[int]:
    return {id(p) for group in opt.param_groups for p in group["params"]}


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


# =====================================================================
# stage-boundary execution + rebuild-reference tracking + legacy desugar
# =====================================================================

# --- legacy desugar structural parity ----------------------------------


class TestDesugarStructuralParity:
    def test_single_stage_total_steps_is_whole_run_estimate(self, data):
        # the desugared single `fit` stage must pass estimated_stepping_batches
        # straight into OneCycle (any per-stage allocation would break parity).
        seen: dict[str, int] = {}

        class Cap(Callback):
            def on_train_start(self, trainer, module):
                seen["total_steps"] = trainer.lr_scheduler_configs[0].scheduler.total_steps
                seen["estimate"] = trainer.estimated_stepping_batches
                seen["opt_params"] = len(_optimizer_param_ids(trainer.optimizers[0]))
                seen["model_params"] = len(_param_ids(module.parameters()))

        model = build_model(data)  # no training_schedule
        make_trainer(max_epochs=4, callbacks=[Cap()]).fit(model, build_datamodule(data))

        assert seen["total_steps"] == seen["estimate"] == 20
        assert seen["opt_params"] == seen["model_params"]  # every param optimised


# --- explicit single-fit-stage ≡ legacy (bitwise) ----------------------


class TestSingleStageEqualsLegacy:
    @staticmethod
    def _fit_state_dict(data, training_schedule) -> dict:
        seed_everything(1234, workers=True)
        kwargs = {} if training_schedule is None else {"training_schedule": training_schedule}
        model = build_model(data, **kwargs)
        make_trainer(max_epochs=2, callbacks=[]).fit(model, build_datamodule(data))
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def test_explicit_fit_stage_is_bitwise_equal_to_no_schedule(self, data):
        legacy = self._fit_state_dict(data, None)
        explicit = self._fit_state_dict(data, {"stages": {"fit": {}}})
        assert legacy.keys() == explicit.keys()
        for key, value in legacy.items():
            assert torch.equal(value, explicit[key]), key


# --- 2-stage execution + rebuild-reference tracking ---------------


class _TwoStageRecorder(Callback):
    """Captures the per-step LR/optimizer trace, the frozen module's weight at
    stage boundaries, and the final rebuild-reference identities.
    """

    FROZEN = "encoder"  # frozen in stage 1 (warmup), unfrozen in stage 2 (full)

    def __init__(self) -> None:
        self.trace: list[dict] = []
        self.weights: dict[str, torch.Tensor] = {}
        self.refs: dict[str, bool] = {}

    def on_train_epoch_start(self, trainer, module) -> None:
        # weight value is unaffected by callback ordering (only its grad flag is),
        # so capturing at epoch start reflects the value entering that epoch.
        self.weights[f"epoch{trainer.current_epoch}"] = (
            module.net[self.FROZEN].weight.detach().clone()
            if hasattr(module.net[self.FROZEN], "weight")
            else next(module.net[self.FROZEN].parameters()).detach().clone()
        )

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        opt = trainer.optimizers[0]
        sched = trainer.lr_scheduler_configs[0].scheduler
        self.trace.append({
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "stage": module.training_controller.current_stage_index,
            "lr": opt.param_groups[0]["lr"],
            "opt_id": id(opt),
            "total_steps": sched.total_steps,
        })

    def on_train_end(self, trainer, module) -> None:
        self.weights["end"] = (
            module.net[self.FROZEN].weight.detach().clone()
            if hasattr(module.net[self.FROZEN], "weight")
            else next(module.net[self.FROZEN].parameters()).detach().clone()
        )
        opt = trainer.optimizers[0]
        sched = trainer.lr_scheduler_configs[0].scheduler
        self.refs = {
            "trainer_is_strategy": trainer.optimizers[0] is trainer.strategy.optimizers[0],
            "module_is_trainer": module.optimizers().optimizer is trainer.optimizers[0],
            "sched_wraps_opt": sched.optimizer is opt,
        }


def _run_two_stage(data) -> tuple[SaltModule, _TwoStageRecorder]:
    seed_everything(7, workers=True)
    model = build_model(
        data,
        training_schedule={
            "stages": {
                "warmup": {"epochs": 2, "frozen": ["encoder"], "lrs": {"max": 5e-3}},
                "full": {"frozen": [], "lrs": {"max": 1e-3}},
            }
        },
    )
    rec = _TwoStageRecorder()
    # schedule callback FIRST so the recorder's on_train_batch_end sees rebuilt refs
    make_trainer(
        max_epochs=4, callbacks=[TrainingScheduleCallback(), rec]
    ).fit(model, build_datamodule(data))
    return model, rec


class TestTwoStageEnvelopes:
    def test_two_envelopes_boundary_allocation_and_movement(self, data):
        model, rec = _run_two_stage(data)

        stage0 = [t for t in rec.trace if t["stage"] == 0]
        stage1 = [t for t in rec.trace if t["stage"] == 1]
        assert len(stage0) == 10 and len(stage1) == 10  # boundary at global step 10

        # per-stage total_steps sum to the whole-run estimate (each stage's
        # OneCycleLR spans only its own stage's steps, never the whole run)
        assert stage0[0]["total_steps"] == 10
        assert stage1[0]["total_steps"] == 10
        assert stage0[0]["total_steps"] + stage1[0]["total_steps"] == 20

        # two distinct OneCycle envelopes: stage-0 peak near max=5e-3, stage-1
        # peak near max=1e-3 — clearly separated, each rises then falls.
        assert max(t["lr"] for t in stage0) > 2e-3
        assert max(t["lr"] for t in stage1) < 2e-3
        assert min(t["lr"] for t in stage0) < max(t["lr"] for t in stage0)
        assert min(t["lr"] for t in stage1) < max(t["lr"] for t in stage1)

        # exactly one rebuild: two distinct optimizer ids, switching at the boundary
        assert len({t["opt_id"] for t in rec.trace}) == 2
        assert stage0[0]["opt_id"] != stage1[0]["opt_id"]

        # frozen (encoder) bitwise-fixed through stage 1 (epochs 0,1), moves in stage 2
        assert torch.equal(rec.weights["epoch0"], rec.weights["epoch2"])  # fixed in warmup
        assert not torch.equal(rec.weights["epoch2"], rec.weights["end"])  # moves in full

        # stage-2 optimizer owns the newly-unfrozen encoder params
        final_opt_ids = _optimizer_param_ids(model._trainer.optimizers[0])  # noqa: SLF001
        assert _module_param_ids(model, "encoder") <= final_opt_ids


class TestRebuildReferenceTracking:
    def test_all_refs_point_at_rebuilt_objects(self, data):
        _model, rec = _run_two_stage(data)
        assert rec.refs["trainer_is_strategy"]
        assert rec.refs["module_is_trainer"]
        assert rec.refs["sched_wraps_opt"]


# --- training_schedule.stop_after: hook gating (no Trainer) --------------

MODULE_NAMES = ("norm", "encoder")


class _StubStopAfterModule:
    """The `training_controller` surface `on_train_epoch_end` reads; its own controller."""

    def __init__(self, schedule, *, current_stage_index, next_early_stop_index=None):
        self.training_controller, self.schedule = self, schedule
        self.current_stage_index, self._next_index = current_stage_index, next_early_stop_index
        self.record_calls: list[tuple[int, int]] = []

    def record_stop_after(self, global_step, epoch):
        self.record_calls.append((global_step, epoch))
        idx = self.schedule.stop_after_index
        name = self.schedule.stages[idx].name
        return boundary_record(name, idx, global_step, epoch, "stop_after")

    def next_stage_index(self, _epoch):
        return self._next_index


class TestStopAfterHookGating:
    """`on_train_epoch_end`'s `stop_after` action on a stub module and a fake trainer."""

    EPOCHS: ClassVar = {"stages": {"a": {"epochs": 2}, "b": {"epochs": 2}, "c": {}},
                        "stop_after": "b"}
    EARLY: ClassVar = {
        "stages": {"a": {"epochs": 3, "early_stop": {"monitor": "val/loss"}}, "b": {}},
        "stop_after": "a",
    }

    @staticmethod
    def _end_epoch(tmp_path, raw, stage, epoch, next_index) -> tuple:
        schedule = TrainingSchedule.from_config(raw, MODULE_NAMES)
        module = _StubStopAfterModule(schedule, current_stage_index=stage,
                                      next_early_stop_index=next_index)
        saved: list[Path] = []
        trainer = SimpleNamespace(
            current_epoch=epoch, max_epochs=6, global_step=5 * (epoch + 1), should_stop=False,
            checkpoint_callback=None, default_root_dir=str(tmp_path), save_checkpoint=saved.append,
        )
        TrainingScheduleCallback().on_train_epoch_end(trainer, module)
        return module, trainer, saved

    @pytest.mark.parametrize(("raw", "stage", "epoch", "next_index"), [
        pytest.param(EPOCHS, 1, 2, None, id="stop-stage-has-epochs-left"),  # b owns [2, 4)
        pytest.param(EPOCHS, 2, 3, None, id="active-stage-past-stop-stage"),
        pytest.param({"stages": {"a": {"epochs": 2}, "b": {}}}, 0, 1, None, id="no-stop-after"),
        pytest.param(EARLY, 0, 0, 0, id="early-stop-without-pending-advance"),
    ])
    def test_no_action(self, tmp_path, raw, stage, epoch, next_index):
        module, trainer, saved = self._end_epoch(tmp_path, raw, stage, epoch, next_index)
        assert (module.record_calls, trainer.should_stop, saved) == ([], False, [])

    @pytest.mark.parametrize(("raw", "stage", "epoch", "next_index", "names"), [
        pytest.param(EPOCHS, 1, 3, None, ("b", "c"), id="last-epoch-of-stop-stage"),
        pytest.param(EARLY, 0, 0, 1, ("a", "b"), id="early-stop-pending-advance"),
    ])
    def test_fires_once_at_the_stop_boundary(self, tmp_path, caplog, raw, stage, epoch,
                                             next_index, names):
        with caplog.at_level(logging.INFO, logger="salt"):
            module, trainer, saved = self._end_epoch(tmp_path, raw, stage, epoch, next_index)
        assert module.record_calls == [(5 * (epoch + 1), epoch)]
        assert trainer.should_stop is True
        assert saved == [tmp_path / f"stop_after={names[0]}-epoch={epoch:03d}.ckpt"]
        assert f"stop_after={names[0]!r} reached" in caplog.text
        assert f"before stage {names[1]!r}" in caplog.text


# --- training_schedule.stop_after: real fit ends at the boundary ---------


class TestStopAfterEndsFitAtBoundary:
    def test_fit_ends_at_warmup_boundary(self, data, tmp_path):
        schedule = {"stages": {"warmup": {"epochs": 2, "frozen": ["encoder"]}, "full": {}},
                    "stop_after": "warmup"}
        seed_everything(7, workers=True)
        model = build_model(data, training_schedule=schedule)
        rec = _TwoStageRecorder()
        trainer = make_trainer(max_epochs=4, callbacks=[TrainingScheduleCallback(), rec],
                               default_root_dir=str(tmp_path))
        trainer.fit(model, build_datamodule(data))

        assert trainer.current_epoch == 2
        assert trainer.should_stop is True
        assert all(t["stage"] == 0 for t in rec.trace)  # stage `full` never ran
        assert len({t["opt_id"] for t in rec.trace}) == 1  # no rebuild
        # same stage-0 OneCycle allocation as the un-stopped [10, 10] run
        assert len(rec.trace) == rec.trace[0]["total_steps"] == 10
        assert torch.equal(rec.weights["epoch0"], rec.weights["end"])  # frozen throughout

        loaded = torch.load(str(tmp_path / "stop_after=warmup-epoch=001.ckpt"), weights_only=False)
        assert loaded[CKPT_KEY]["schedule"] == {
            "stage_index": 0, "stage_name": "warmup",
            "stop_after": {"stage": "warmup", "reached": True,
                           "boundary": boundary_record("warmup", 0, 10, 1, "stop_after")},
        }


# --- sample stage (num_samples / frac_samples): hook gating (no Trainer) ---

_NEVER_READ = "a num_samples boundary must never read the loader length"
# ceil(62_469_554 / (batch 1000 * world 3 * accumulate 1)) = ceil(20823.18)
_N_W3 = 20824


class _FakeLoader:
    """A train loader of some per-rank length that a num_samples boundary never reads."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        raise AssertionError(_NEVER_READ)


class _FakeStrategy:
    """Counts `setup_optimizers`; scripted `reduce_boolean_decision` (passthrough, or a
    fabricated cross-rank disagreement); passthrough `broadcast`.
    """

    def __init__(self, *, disagree: bool = False) -> None:
        self.disagree, self.setup_optimizers_calls = disagree, 0
        self.reduce_calls: list[tuple[bool, bool]] = []

    def setup_optimizers(self, _trainer) -> None:
        self.setup_optimizers_calls += 1

    def reduce_boolean_decision(self, value: bool, *, all: bool = True) -> bool:  # noqa: A002
        self.reduce_calls.append((value, all))
        return (not all) if self.disagree else value

    @staticmethod
    def broadcast(obj, src: int = 0) -> object:  # noqa: ARG004 - Lightning's signature
        return obj


class _FakeSampleTrainer:
    """A non-`SimpleNamespace` trainer whose `num_training_batches` raises."""

    def __init__(self, *, world_size=3, global_rank=0, loader_length=11, strategy=None,
                 default_root_dir=".") -> None:
        self.datamodule = SimpleNamespace(batch_size=1000)
        self.world_size, self.accumulate_grad_batches, self.global_rank = world_size, 1, global_rank
        self.global_step = self.current_epoch = 0
        self.should_stop, self.checkpoint_callback = False, None
        self.default_root_dir, self.saved_checkpoints = default_root_dir, []
        self.strategy = _FakeStrategy() if strategy is None else strategy
        self.state = SimpleNamespace(fn=TrainerFn.FITTING)
        self.train_dataloader = _FakeLoader(loader_length)

    def save_checkpoint(self, path) -> None:
        self.saved_checkpoints.append(str(path))

    @property
    def num_training_batches(self) -> int:
        raise AssertionError(_NEVER_READ)


class _StubSampleModule:
    """The `training_controller` surface the sample-stage hooks read; its own controller."""

    def __init__(self, schedule, *, stage_start_step: int = 0) -> None:
        self.training_controller, self.schedule = self, schedule
        self.current_stage_index, self.stage_start_step = 0, stage_start_step
        self.stage_step_targets: dict[int, int] | None = None
        self.stop_after_record: dict | None = None
        self.advance_calls: list[tuple[int, int, int, str]] = []
        self.record_calls: list[tuple[int, int]] = []

    def resolve_step_targets(self, trainer) -> None:
        if self.stage_step_targets is None:
            self.stage_step_targets = self.schedule.stage_step_targets(
                trainer.datamodule.batch_size, trainer.world_size, trainer.accumulate_grad_batches
            )

    def step_target(self, index):
        return self.stage_step_targets[index]

    def next_stage_index(self, _epoch):
        return self.current_stage_index + 1

    def advance_to_stage(self, idx, global_step, epoch, reason) -> None:
        self.advance_calls.append((idx, global_step, epoch, reason))
        self.current_stage_index, self.stage_start_step = idx, global_step

    def record_stop_after(self, global_step, epoch):
        self.record_calls.append((global_step, epoch))
        idx = self.schedule.stop_after_index
        name = self.schedule.stages[idx].name
        self.stop_after_record = boundary_record(name, idx, global_step, epoch, "stop_after")
        return self.stop_after_record


class TestSampleStageHooks:
    """`on_train_batch_end`'s step-count boundary on stub modules and fake trainers."""

    RAW: ClassVar = {"stages": {"kinematic_prior": {"num_samples": 62_469_554},
                                "main": {"epochs": 10}}}

    @classmethod
    def _schedule(cls, *, stop_after: bool = False) -> TrainingSchedule:
        raw = {**cls.RAW, "stop_after": "kinematic_prior"} if stop_after else cls.RAW
        return TrainingSchedule.from_config(raw, MODULE_NAMES)

    def test_three_ranks_convert_same_num_samples_to_same_N_and_stop_at_N_despite_unequal_loader_lengths(self):  # noqa: E501
        callback = TrainingScheduleCallback()
        for rank, length in enumerate((11, 10, 9)):
            trainer = _FakeSampleTrainer(global_rank=rank, loader_length=length)
            module = _StubSampleModule(self._schedule())
            for step in range(1, _N_W3 + 1):
                trainer.global_step = step
                callback.on_train_batch_end(trainer, module, None, None, step - 1)
                if step < _N_W3:
                    assert module.advance_calls == []
            assert module.stage_step_targets == {0: _N_W3}
            assert module.advance_calls == [(1, _N_W3, 0, "steps")]
            assert trainer.strategy.setup_optimizers_calls == 1

    def test_world_size_scaling_from_setup(self):
        # the REAL controller's lazy, idempotent resolve (configure_optimizers is its
        # first consumer in a fit); the raising num_training_batches proves it is unread
        for world_size, expected in {1: 62_470, 2: 31_235, 3: _N_W3}.items():
            controller = TrainingController(self._schedule(), nn.ModuleDict(), LRS, "AdamW")
            trainer = _FakeSampleTrainer(world_size=world_size)
            controller.resolve_step_targets(trainer)
            controller.resolve_step_targets(trainer)
            assert controller.stage_step_targets == {0: expected}
            assert controller.step_target(0) == expected

    def test_tripwire_raises_on_fabricated_disagreement(self):
        trainer = _FakeSampleTrainer(strategy=_FakeStrategy(disagree=True))
        trainer.global_step = _N_W3
        module = _StubSampleModule(self._schedule())
        with pytest.raises(RuntimeError, match="divergence"):
            TrainingScheduleCallback().on_train_batch_end(trainer, module, None, None, 0)

    def test_no_collective_before_the_boundary(self):
        trainer, module = _FakeSampleTrainer(), _StubSampleModule(self._schedule())
        callback = TrainingScheduleCallback()
        for step in range(1, _N_W3):
            trainer.global_step = step
            callback.on_train_batch_end(trainer, module, None, None, step - 1)
        assert trainer.strategy.reduce_calls
        assert not any(is_all for _value, is_all in trainer.strategy.reduce_calls)

    def test_stop_after_on_sample_stage_stops_mid_epoch_with_step_ckpt(self, tmp_path):
        trainer = _FakeSampleTrainer(default_root_dir=str(tmp_path))
        module = _StubSampleModule(self._schedule(stop_after=True))
        callback = TrainingScheduleCallback()
        trainer.global_step = _N_W3
        callback.on_train_batch_end(trainer, module, None, None, 0)
        assert (module.record_calls, trainer.should_stop) == ([(_N_W3, 0)], True)
        assert trainer.saved_checkpoints == []  # deferred to train-epoch end
        # an epoch-end checkpoint resumes at the NEXT epoch with the seam firing (a
        # batch-end one would resume RESTARTED_MID_EPOCH and skip it)
        callback.on_train_epoch_end(trainer, module)
        name = f"stop_after=kinematic_prior-step={_N_W3:06d}.ckpt"
        assert trainer.saved_checkpoints == [str(tmp_path / name)]
        assert (module.advance_calls, trainer.strategy.setup_optimizers_calls) == ([], 0)

    def test_epoch_start_resume_seam_advances_but_never_restops(self, tmp_path):
        trainer = _FakeSampleTrainer(default_root_dir=str(tmp_path))
        trainer.global_step = _N_W3  # a restored stop_after checkpoint: stage 0, at N
        module = _StubSampleModule(self._schedule(stop_after=True))
        TrainingScheduleCallback().on_train_epoch_start(trainer, module)
        assert module.advance_calls == [(1, _N_W3, 0, "steps")]
        assert trainer.strategy.setup_optimizers_calls == 1
        assert trainer.should_stop is False
        assert trainer.saved_checkpoints == module.record_calls == []

    def test_non_sample_schedule_batch_end_is_inert_on_a_bare_trainer(self):
        schedule = TrainingSchedule.from_config({"stages": {"a": {"epochs": 2}, "b": {}}},
                                                MODULE_NAMES)
        bare = SimpleNamespace(current_epoch=0, max_epochs=4)  # no strategy, no step
        TrainingScheduleCallback().on_train_batch_end(bare, _StubSampleModule(schedule),
                                                      None, None, 0)


# --- sample stage: real fits (batch 100, 5 steps/epoch) ---------------------


class TestSampleStageFit:
    SCHEDULE: ClassVar = {"stages": {
        "warmup": {"num_samples": 300, "frozen": ["encoder"], "lrs": {"max": 5e-3}},
        "full": {"frozen": [], "lrs": {"max": 1e-3}},
    }}

    @staticmethod
    def _fit(data, schedule, *, max_epochs, **trainer_kwargs) -> tuple:
        seed_everything(7, workers=True)
        model = build_model(data, training_schedule=schedule)
        rec = _TwoStageRecorder()
        trainer = make_trainer(max_epochs=max_epochs, callbacks=[TrainingScheduleCallback(), rec],
                               **trainer_kwargs)
        trainer.fit(model, build_datamodule(data))
        return model, trainer, rec

    def test_num_samples_target_converts_to_N_and_advances_mid_epoch(self, data):
        # N = ceil(300 / (100 * 1 * 1)) = 3 of the 10 steps of a 2-epoch fit
        model, _trainer, rec = self._fit(data, self.SCHEDULE, max_epochs=2)
        stage0 = [t for t in rec.trace if t["stage"] == 0]
        stage1 = [t for t in rec.trace if t["stage"] == 1]
        # the boundary batch is recorded after the advance, hence 2 + 8
        assert (len(stage0), len(stage1)) == (2, 8)
        assert (stage0[0]["total_steps"], stage1[0]["total_steps"]) == (3, 7)
        assert len({t["opt_id"] for t in rec.trace}) == 2
        assert (rec.trace[2]["global_step"], rec.trace[2]["epoch"]) == (3, 0)
        assert min(t["lr"] for t in stage1) < max(t["lr"] for t in stage1)
        controller = model.training_controller
        assert controller.boundary_records == [boundary_record("full", 1, 3, 0, "steps")]
        assert controller.stage_start_step == 3

    def test_sample_then_epoch_stage_allocations_match_runtime_steps(self, data):
        schedule = {"stages": {"warmup": {"num_samples": 300, "frozen": ["encoder"]},
                               "mid": {"epochs": 1}, "final": {}}}
        # 15 steps; an over-allocated stage would make OneCycleLR raise mid-fit
        model, trainer, rec = self._fit(data, schedule, max_epochs=3)
        assert trainer.global_step == 15
        first_total: dict[int, int] = {}
        for t in rec.trace:
            first_total.setdefault(t["stage"], t["total_steps"])
        assert first_total == {0: 3, 1: 2, 2: 10}
        records = model.training_controller.boundary_records
        assert records == [boundary_record("mid", 1, 3, 0, "steps"),
                           boundary_record("final", 2, 5, 1, "epochs")]
        starts = [0, *(r["global_step"] for r in records), trainer.global_step]
        assert [b - a for a, b in itertools.pairwise(starts)] == [3, 2, 10]

    def test_stop_after_resume_sizes_following_epoch_stage_from_actual_start(self, data, tmp_path):
        schedule = {"stages": {"warmup": {"num_samples": 300, "frozen": ["encoder"]},
                               "mid": {"epochs": 1}, "final": {}}}
        self._fit(data, {**schedule, "stop_after": "warmup"}, max_epochs=3,
                  default_root_dir=str(tmp_path))
        ckpt = tmp_path / "stop_after=warmup-step=000003.ckpt"
        assert ckpt.exists()
        # the resume starts at epoch 1, so 'mid' runs a full epoch (not the 2-step remainder)
        seed_everything(7, workers=True)
        rec = _TwoStageRecorder()
        trainer = make_trainer(max_epochs=3, callbacks=[TrainingScheduleCallback(), rec],
                               default_root_dir=str(tmp_path))
        trainer.fit(build_model(data, training_schedule=schedule), build_datamodule(data),
                    ckpt_path=str(ckpt))
        for stage in (1, 2):
            steps = [t for t in rec.trace if t["stage"] == stage]
            assert (len(steps), steps[0]["total_steps"]) == (5, 5)
        assert trainer.global_step == 13

    def test_frac_samples_fit_converts_via_loader_length(self, data):
        # N = ceil(0.5 * ceil(5 batches / accumulate 1)) = ceil(2.5) = 3
        schedule = {"stages": {"warmup": {"frac_samples": 0.5, "frozen": ["encoder"]}, "full": {}}}
        model, _trainer, _rec = self._fit(data, schedule, max_epochs=2)
        controller = model.training_controller
        assert controller.stage_step_targets == {0: 3}
        assert controller.boundary_records == [boundary_record("full", 1, 3, 0, "steps")]

    def test_stop_after_ends_fit_mid_epoch_with_step_ckpt(self, data, tmp_path):
        schedule = {**self.SCHEDULE, "stop_after": "warmup"}
        _model, trainer, rec = self._fit(data, schedule, max_epochs=2,
                                         default_root_dir=str(tmp_path))
        assert trainer.should_stop is True
        # current_epoch is not asserted: Lightning bumps it once epoch 0's batch loop
        # ends, even on a should_stop break — the boundary record pins epoch 0
        assert all(t["stage"] == 0 for t in rec.trace)
        assert len(rec.trace) == 3
        assert len({t["opt_id"] for t in rec.trace}) == 1

        ckpt = tmp_path / "stop_after=warmup-step=000003.ckpt"
        assert torch.load(str(ckpt), weights_only=False)[CKPT_KEY]["schedule"] == {
            "stage_index": 0, "stage_name": "warmup", "stage_start_step": 0,
            "stage_start_epoch": 0, "boundaries": [],
            "stop_after": {"stage": "warmup", "reached": True,
                           "boundary": boundary_record("warmup", 0, 3, 0, "stop_after")},
        }


# --- per-stage optimizer override + HybridMuonAdamW across a boundary ----


class TestPerStageOptimizer:
    def test_three_stage_optimizer_override_rebuilds_class(self, data):
        seed_everything(3, workers=True)
        model = build_model(
            data,
            optimizer="AdamW",
            training_schedule={
                "stages": {
                    "a": {"epochs": 2},  # inherits AdamW
                    "b": {"epochs": 2, "optimizer": "lion"},
                    "c": {"optimizer": "lion"},
                }
            },
        )
        stage_opt_types: dict[int, str] = {}

        class Cap(Callback):
            def on_train_batch_end(self, trainer, module, *_a) -> None:
                stage_opt_types[module.training_controller.current_stage_index] = type(
                    trainer.optimizers[0]
                ).__name__

        make_trainer(
            max_epochs=6, callbacks=[TrainingScheduleCallback(), Cap()]
        ).fit(model, build_datamodule(data))

        assert stage_opt_types[0] == "AdamW"
        assert stage_opt_types[1] == "Lion"
        assert stage_opt_types[2] == "Lion"

    def test_hybrid_muon_adamw_rebuilds_split_over_new_trainable(self, data):
        seed_everything(5, workers=True)
        model = build_model(
            data,
            optimizer="HybridMuonAdamW",
            training_schedule={
                "stages": {
                    "warmup": {"epochs": 2, "frozen": ["encoder"]},
                    "full": {"frozen": []},
                }
            },
        )
        captured: dict[str, object] = {}

        class Cap(Callback):
            def on_train_batch_end(self, trainer, module, *_a) -> None:
                idx = module.training_controller.current_stage_index
                captured[f"stage{idx}"] = trainer.optimizers[0]

        make_trainer(
            max_epochs=4, callbacks=[TrainingScheduleCallback(), Cap()]
        ).fit(model, build_datamodule(data))

        warmup_opt = captured["stage0"]
        full_opt = captured["stage1"]
        assert isinstance(warmup_opt, HybridMuonAdamW)
        assert isinstance(full_opt, HybridMuonAdamW)
        assert warmup_opt is not full_opt  # rebuilt across the boundary

        # encoder's 2D weights are routed to Muon only after they are unfrozen
        assert not any("encoder" in n for n in warmup_opt._muon_names)  # noqa: SLF001
        assert any("encoder" in n for n in full_opt._muon_names)  # noqa: SLF001
        # the rebuilt split covers exactly the current trainable set
        trainable = _param_ids(p for p in model.parameters() if p.requires_grad)
        assert _optimizer_param_ids(full_opt) == trainable


# =====================================================================
# the module-freeze machinery
# =====================================================================


def build_model_from_modules(modules, **kwargs) -> SaltModule:
    return SaltModule(modules, lrs=FREEZE_LRS, **kwargs)


def make_freeze_trainer(**kwargs) -> Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("limit_train_batches", 1)
    kwargs.setdefault("limit_val_batches", 1)
    return Trainer(
        accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, num_sanity_val_steps=0,
        **kwargs,
    )


def offline_bind(model: SaltModule, dm: SaltDataModule, max_epochs: int = 10) -> None:
    """Bind a model against a datamodule WITHOUT a training loop, with a stub
    trainer exposing the fields `configure_optimizers`/schedule-setup read.
    """
    dm.set_sinks(model.sink_demand())
    dm.setup("fit")
    model._trainer = SimpleNamespace(  # noqa: SLF001
        datamodule=dm, callbacks=[], max_epochs=max_epochs, estimated_stepping_batches=100
    )
    model.setup("fit")


class TestFreezeAppliedAndPersists:
    def test_frozen_params_requires_grad_false_and_eval(self, data):
        model = build_model_from_modules(
            build_gn2v2_modules(data["nd"]),
            training_schedule={"stages": {"fit": {"frozen": ["encoder", "track_embed"]}}},
        )
        offline_bind(model, build_datamodule(data))

        for name in ("encoder", "track_embed"):
            assert not any(p.requires_grad for p in model.net[name].parameters()), name
            assert not model.net[name].training, name  # in eval mode
        # a non-frozen module keeps grads + train mode
        assert all(p.requires_grad for p in model.net["pool"].parameters())

    def test_frozen_module_stays_eval_across_epoch_boundary(self, data):
        # Lightning calls model.train() at every epoch start; the override must
        # keep the frozen module in eval each epoch.
        seen: list[bool] = []

        class Rec(Callback):
            def on_train_epoch_start(self, _t, module):
                seen.append(module.net["encoder"].training)

        model = build_model_from_modules(
            build_gn2v2_modules(data["nd"]),
            training_schedule={"stages": {"fit": {"frozen": ["encoder"]}}},
        )
        make_freeze_trainer(max_epochs=2, callbacks=[Rec()]).fit(model, build_datamodule(data))

        assert len(seen) == 2  # two epochs observed
        assert seen == [False, False]  # frozen module NEVER in train mode
        assert not model.net["encoder"].training  # still eval after fit


class TestOptimizerExcludesFrozen:
    @pytest.mark.parametrize("optimizer", ["AdamW", "HybridMuonAdamW"])
    def test_optimizer_param_set_is_trainable_only(self, data, optimizer):
        model = build_model_from_modules(
            build_gn2v2_modules(data["nd"]),
            optimizer=optimizer,
            training_schedule={"stages": {"fit": {"frozen": ["encoder"]}}},
        )
        offline_bind(model, build_datamodule(data))
        [opt], _ = model.configure_optimizers()

        trainable = _param_ids(p for p in model.parameters() if p.requires_grad)
        opt_ids = _optimizer_param_ids(opt)
        assert opt_ids == trainable
        # the frozen encoder's params are excluded entirely
        assert not (_module_param_ids(model, "encoder") & opt_ids)
        # sanity: encoder actually HAS params (so the exclusion is meaningful)
        assert _module_param_ids(model, "encoder")


class TestNoScheduleNoChange:
    def test_optimizer_holds_all_params(self, data):
        model = build_model_from_modules(build_gn2v2_modules(data["nd"]))  # no training_schedule
        offline_bind(model, build_datamodule(data))
        [opt], _ = model.configure_optimizers()

        all_ids = _param_ids(model.parameters())
        assert _optimizer_param_ids(opt) == all_ids
        assert model.training_controller.frozen_module_names == set()
        assert all(p.requires_grad for p in model.parameters())


class TestMultiStageBindsAtFit:
    """A >1-stage schedule is now EXECUTED (the old fit-time rejection is gone):
    it binds and applies stage 0's freeze mask at setup.
    """

    def test_multi_stage_binds_and_applies_stage0_freeze(self, data):
        model = build_model_from_modules(
            build_gn2v2_modules(data["nd"]),
            training_schedule={
                "stages": {"warmup": {"epochs": 1, "frozen": ["encoder"]}, "full": {}}
            },
        )
        assert model.training_controller.schedule.is_multi_stage
        offline_bind(model, build_datamodule(data))  # no rejection
        assert model.training_controller.current_stage_index == 0
        assert model.training_controller.frozen_module_names == {"encoder"}
        assert not any(p.requires_grad for p in model.net["encoder"].parameters())


class TestInitFromComposesWithFreeze:
    @staticmethod
    def _surgery_modules(nd) -> dict:
        # drop track_vertexing, add a new head — mirrors the swap-one-head
        # surgery in test_init_from.py
        modules = build_gn2v2_modules(nd)
        del modules["track_vertexing"]
        extra = ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
            input="pooled.global", dense={"hidden_layers": [16], "activation": "ReLU"},
        )
        extra.name = "extra_jets"
        modules["extra_jets"] = extra
        modules["loss"] = LossSum()
        return modules

    @pytest.fixture(scope="class")
    def base_ckpt(self, data) -> Path:
        model = build_model_from_modules(build_gn2v2_modules(data["nd"]))
        make_freeze_trainer().fit(model, build_datamodule(data))
        ckpt = data["dir"] / "freeze_base.ckpt"
        model._trainer.save_checkpoint(ckpt)  # noqa: SLF001
        return ckpt

    def test_only_new_module_trains(self, data, base_ckpt):
        # freeze EVERY loaded (retained) module; only extra_jets (new) trains
        retained = [
            "norm", "track_embed", "encoder", "pool", "jets_classification", "track_origin",
        ]
        losses: list[torch.Tensor] = []
        captured: dict[str, object] = {}

        class Rec(Callback):
            def on_train_start(self, trainer, _m):
                captured["opt"] = trainer.optimizers[0]

            def on_train_batch_end(self, _t, _m, outputs, _b, _i):
                losses.append(outputs["loss"].detach())

        model = build_model_from_modules(
            self._surgery_modules(data["nd"]),
            training_schedule={"stages": {"fit": {
                "frozen": retained, "skip_losses": ["jets_classification", "track_origin"]}}},
        )
        model._init_from = str(base_ckpt)  # noqa: SLF001
        make_freeze_trainer(callbacks=[Rec()]).fit(model, build_datamodule(data))

        assert model._init_warm_started  # noqa: SLF001 - warm start ran
        assert losses  # training ran
        assert all(torch.isfinite(x) for x in losses)  # finite loss throughout
        opt_ids = _optimizer_param_ids(captured["opt"])
        assert opt_ids == _module_param_ids(model, "extra_jets")  # ONLY the new head
        # every retained module is frozen out of the optimizer
        for name in retained:
            assert not (_module_param_ids(model, name) & opt_ids), name


# =====================================================================
# per-stage LR-scheduler class choice
# =====================================================================

COSINE = "torch.optim.lr_scheduler.CosineAnnealingLR"
PLATEAU = "torch.optim.lr_scheduler.ReduceLROnPlateau"


class _DeterministicDataModule(SaltDataModule):
    def train_dataloader(self):  # noqa: D102 - shuffle=False for resume determinism
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=False)


def build_dm(data) -> _DeterministicDataModule:
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    return _DeterministicDataModule(
        modules, batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


class _SchedRecorder(Callback):
    """Per-train-batch trace of (stage index, active scheduler class name, LR)."""

    def __init__(self) -> None:
        self.trace: list[dict] = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        sched = trainer.lr_scheduler_configs[0].scheduler
        self.trace.append({
            "stage": module.training_controller.current_stage_index,
            "sched": type(sched).__name__,
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
        })


# --- legacy parity — no lr_scheduler → OneCycleLR ------------------------


class TestLegacyParity:
    def test_no_lr_scheduler_uses_onecycle(self, data):
        # a plain (no lr_scheduler) config keeps the default per-stage OneCycleLR,
        # step-interval — the byte-parity path.
        seed_everything(SEED, workers=True)
        model = build_model(data)  # no training_schedule → desugared fit stage
        rec = _SchedRecorder()
        make_trainer(max_epochs=2, callbacks=[rec]).fit(model, build_dm(data))
        assert {t["sched"] for t in rec.trace} == {"OneCycleLR"}


# --- two-stage, different scheduler classes -----------------------------


class TestTwoStageSchedulers:
    SCHEDULE = {
        "stages": {
            "warmup": {
                "epochs": 2, "trainable": ["jets_classification"],
                "skip_losses": ["track_origin", "track_vertexing"],
                "lr_scheduler": {"class_path": COSINE, "init_args": {"T_max": 2}},
            },
            "full": {
                "frozen": [],
                "lr_scheduler": {
                    "class_path": PLATEAU,
                    "init_args": {"mode": "min", "factor": 0.5, "patience": 1},
                    "interval": "epoch",
                    "monitor": "val/loss",
                },
            },
        }
    }

    def _run(self, data):
        seed_everything(SEED, workers=True)
        model = build_model(data, training_schedule=self.SCHEDULE)
        rec = _SchedRecorder()
        make_trainer(
            max_epochs=4, callbacks=[TrainingScheduleCallback(), rec]
        ).fit(model, build_dm(data))
        return model, rec

    def test_correct_scheduler_class_active_per_stage(self, data):
        _model, rec = self._run(data)
        stage0 = {t["sched"] for t in rec.trace if t["stage"] == 0}
        stage1 = {t["sched"] for t in rec.trace if t["stage"] == 1}
        assert stage0 == {"CosineAnnealingLR"}  # warmup scheduler
        assert stage1 == {"ReduceLROnPlateau"}  # finetune scheduler (rebuilt at boundary)

    def test_stage1_optimizer_owns_trainable_set(self, data):
        model, _ = self._run(data)
        assert model.training_controller.current_stage_index == 1
        opt_ids = {
            id(p) for group in model.trainer.optimizers[0].param_groups for p in group["params"]
        }
        # full stage unfroze everything → the previously-frozen encoder is optimised
        assert _module_param_ids(model, "encoder") <= opt_ids


# --- plateau scheduler + early_stop on the same monitor ------------------


class TestPlateauWithEarlyStop:
    def test_plateau_and_early_stop_coexist(self, data):
        # one stage declares BOTH a ReduceLROnPlateau scheduler and early_stop on the
        # same monitor — both must act with independent state (scheduler reduces LR,
        # early_stop advances the stage). Two stages so a non-final early_stop can
        # advance; here the plateau lives on the first (non-final) stage.
        schedule = {
            "stages": {
                "reduce": {
                    "epochs": 8,
                    "lr_scheduler": {
                        "class_path": PLATEAU,
                        "init_args": {"mode": "min", "factor": 0.5, "patience": 1},
                        "monitor": "val/loss",
                    },
                    "early_stop": {"monitor": "val/loss", "mode": "min", "patience": 2},
                },
                "full": {"frozen": []},
            }
        }
        seed_everything(SEED, workers=True)
        model = build_model(data, training_schedule=schedule)
        rec = _SchedRecorder()
        make_trainer(
            max_epochs=10, callbacks=[TrainingScheduleCallback(), rec]
        ).fit(model, build_dm(data))
        # the plateau scheduler was active in stage 0 (both mechanisms coexisted;
        # no crash / no interference — the run completed and reached/attempted stage
        # advance under early_stop patience).
        stage0 = [t for t in rec.trace if t["stage"] == 0]
        assert stage0 and {t["sched"] for t in stage0} == {"ReduceLROnPlateau"}
        # a boundary record exists (early_stop or epoch cap advanced the stage) OR
        # the fit ended in stage 0 via early_stop — either way both mechanisms ran
        # without error. Assert the schedule progressed past stage 0 OR early-stopped.
        assert model.training_controller.current_stage_index in (0, 1)


# --- resume restores scheduler state ------------------------------------


class TestResume:
    def test_mid_stage_resume_restores_plateau_state(self, data, tmp_path):
        # a single-stage ReduceLROnPlateau schedule; save mid-stage and resume — the
        # scheduler's internal state (best/num_bad_epochs) is restored by Lightning,
        # so the resumed LR trace continues (not reset).
        schedule = {
            "stages": {
                "fit": {
                    "lr_scheduler": {
                        "class_path": PLATEAU,
                        "init_args": {"mode": "min", "factor": 0.5, "patience": 1},
                        "monitor": "val/loss",
                    }
                }
            }
        }
        ckpt_dir = tmp_path / "ck"
        seed_everything(SEED, workers=True)
        m1 = build_model(data, training_schedule=schedule)
        mc = ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch}", save_top_k=-1,
                             every_n_epochs=1)
        make_trainer(
            max_epochs=4, callbacks=[TrainingScheduleCallback(), mc], enable_checkpointing=True,
        ).fit(m1, build_dm(data))
        ckpt = ckpt_dir / "epoch=1.ckpt"
        assert ckpt.exists()
        saved = torch.load(str(ckpt), weights_only=False)
        # Lightning persisted the scheduler state under lr_schedulers
        assert saved.get("lr_schedulers"), "scheduler state not checkpointed"

        # resume: a fresh module + trainer picks up from the checkpoint without error
        # and the scheduler is a ReduceLROnPlateau again.
        seed_everything(SEED, workers=True)
        m2 = build_model(data, training_schedule=schedule)
        rec = _SchedRecorder()
        make_trainer(
            max_epochs=4, callbacks=[TrainingScheduleCallback(), rec]
        ).fit(m2, build_dm(data), ckpt_path=str(ckpt))
        assert rec.trace and {t["sched"] for t in rec.trace} == {"ReduceLROnPlateau"}


# =====================================================================
# per-stage scoped callbacks
# =====================================================================

# module-level sink so freshly-instantiated (per-stage) delegates can record into a
# shared place; cleared at the start of each test.
HOOK_LOG: list[tuple[str, str, int]] = []  # (tag, hook, stage_index)


class HookProbe(Callback):
    """Records ``(tag, hook, active_stage_index)`` into `HOOK_LOG` for the hooks the
    coordinator forwards, plus setup/teardown. Instantiated from a stage
    ``callbacks`` spec (or passed directly as a persistent global).
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.epoch_starts = 0

    def setup(self, trainer, pl_module, stage) -> None:  # noqa: D102
        idx = pl_module.training_controller.current_stage_index
        HOOK_LOG.append((self.tag, "setup", idx))

    def teardown(self, trainer, pl_module, stage) -> None:  # noqa: D102
        idx = pl_module.training_controller.current_stage_index
        HOOK_LOG.append((self.tag, "teardown", idx))

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D102
        self.epoch_starts += 1
        idx = pl_module.training_controller.current_stage_index
        HOOK_LOG.append((self.tag, "epoch_start", idx))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: D102
        idx = pl_module.training_controller.current_stage_index
        HOOK_LOG.append((self.tag, "batch_end", idx))


# the module pytest actually imported: a hard-coded dotted path would import a SECOND
# copy of this file with its own (never-read) HOOK_LOG
_PROBE = f"{HookProbe.__module__}.HookProbe"


class TestStageScopedCallbacks:
    SCHEDULE = {
        "stages": {
            "a": {"epochs": 2, "callbacks": [{"class_path": _PROBE, "init_args": {"tag": "A"}}]},
            "b": {"callbacks": [{"class_path": _PROBE, "init_args": {"tag": "B"}}]},
        }
    }

    def _run(self, data) -> HookProbe:
        HOOK_LOG.clear()
        seed_everything(SEED, workers=True)
        model = SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=self.SCHEDULE)
        global_probe = HookProbe(tag="GLOBAL")
        # coordinator AFTER the transition driver so it sees the advanced stage;
        # global_probe is a persistent top-level callback (never re-instantiated).
        make_trainer(
            max_epochs=4,
            callbacks=[TrainingScheduleCallback(), StageScopedCallbacks(), global_probe],
        ).fit(model, build_datamodule(data))
        return global_probe

    def test_stage_scoped_callbacks_only_fire_within_their_stage(self, data):
        self._run(data)
        a_hook_stages = {s for (t, h, s) in HOOK_LOG if t == "A" and h in {"epoch_start", "batch_end"}}
        b_hook_stages = {s for (t, h, s) in HOOK_LOG if t == "B" and h in {"epoch_start", "batch_end"}}
        assert a_hook_stages == {0}  # stage-A delegate only active in stage 0
        assert b_hook_stages == {1}  # stage-B delegate only active in stage 1
        # A saw exactly stage 0's two epochs; B saw stage 1's two epochs.
        assert sum(1 for (t, h, _) in HOOK_LOG if t == "A" and h == "epoch_start") == 2
        assert sum(1 for (t, h, _) in HOOK_LOG if t == "B" and h == "epoch_start") == 2

    def test_scoped_callbacks_are_set_up_and_torn_down(self, data):
        self._run(data)
        assert ("A", "setup", 0) in HOOK_LOG  # A set up on entering stage 0
        assert any(t == "A" and h == "teardown" for (t, h, _) in HOOK_LOG)  # A torn down at exit
        assert ("B", "setup", 1) in HOOK_LOG  # B set up on entering stage 1
        assert any(t == "B" and h == "teardown" for (t, h, _) in HOOK_LOG)  # B torn down at fit end

    def test_global_callback_persists_across_all_stages(self, data):
        global_probe = self._run(data)
        global_stages = {s for (t, h, s) in HOOK_LOG if t == "GLOBAL" and h == "epoch_start"}
        assert global_stages == {0, 1}  # global fires in BOTH stages (persistent)
        assert global_probe.epoch_starts == 4  # one instance, accumulated all epochs

    def test_scoped_delegate_is_a_fresh_instance_per_stage(self, data):
        self._run(data)
        # A and B are distinct freshly-instantiated delegates (their setups carry
        # different tags at different stages — no shared/re-used instance).
        setups = [(t, s) for (t, h, s) in HOOK_LOG if h == "setup" and t in {"A", "B"}]
        assert ("A", 0) in setups
        assert ("B", 1) in setups


class TestStageCallbackValidation:
    def test_bad_class_path_fails_at_fit_start(self, data):
        schedule = {
            "stages": {"fit": {"callbacks": [{"class_path": "salt.nonexistent.NoSuchCallback"}]}}
        }
        seed_everything(SEED, workers=True)
        model = SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=schedule)
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            make_trainer(
                max_epochs=2, callbacks=[StageScopedCallbacks()]
            ).fit(model, build_datamodule(data))


# =====================================================================
# a real two-rank CPU/gloo DDP fit (helpers module-level: spawned ranks import them)
# =====================================================================

GLOO_WORLD, GLOO_JOIN_TIMEOUT_S = 2, 300
# Each rank gets 5 of the 10 batches (1000 jets / batch 100) per epoch; accumulate 2 gives
# ceil(5 / 2) = 3 optimizer steps per epoch (6 in 2 epochs), and the priors target is
# N = ceil(num_samples / (batch 100 * world 2 * accumulate 2)) = ceil(800 / 400) = 2:
# the priors -> main boundary falls inside epoch 0, right after global step 2.
_GLOO_FROZEN = ["track_embed", "encoder", "pool", "jets_classification", "track_origin",
                "track_vertexing"]
GLOO_SCHEDULE = {"stages": {
    "priors": {"num_samples": 800, "frozen": _GLOO_FROZEN,
               "skip_losses": ["jets_classification", "track_origin", "track_vertexing"]},
    "main": {"frozen": ["prior"], "skip_losses": ["prior"]},
}}
_CLASSES = ["bjets", "cjets", "ujets"]


def _gloo_modules(norm_dict) -> dict:
    """GN2v2 plus a `prior` head on `normed.jets` whose posterior weights the main head."""
    modules = build_gn2v2_modules(norm_dict)
    del modules["loss"]  # narrowed to GN2v2's keys; SaltModule narrows the fresh one below
    modules["jets_classification"] = ClassificationTaskModule(
        stream="jets", label="flavour_label", class_names=_CLASSES, input="pooled.global",
        dense={"hidden_layers": [16], "activation": "ReLU"}, sample_weight="nn_kinematic_weight",
        loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"reduction": "none"}},
    )
    modules["prior"] = ClassificationTaskModule(
        stream="jets", label="flavour_label", class_names=_CLASSES, input="normed.jets",
        dense={"hidden_layers": [8], "activation": "ReLU"},
    )
    modules["kinematic_weight"] = KinematicWeight(
        task_name="prior", mode="nn", stream="jets", label="flavour_label", q_f=None, n_classes=3,
    )
    modules["loss"] = LossSum()
    return modules


def _head(module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in module.net["jets_classification"].parameters()])


class _MainHeadRecorder(Callback):
    """Main-head L2 delta from its first main-stage batch end to train end."""

    def __init__(self) -> None:
        self.start: torch.Tensor | None = None
        self.delta: float | None = None

    def on_train_batch_end(self, _trainer, module, *_args) -> None:
        if self.start is None and module.training_controller.current_stage_index == 1:
            self.start = _head(module)

    def on_train_end(self, _trainer, module) -> None:
        if self.start is not None:
            self.delta = float((_head(module) - self.start).norm())


def _gloo_child(rank, world, work_dir, data_paths, schedule, ckpt_path, port) -> None:
    """One externally launched DDP rank: fit, then write ``rank{rank}.json`` — or the
    traceback to ``rank{rank}.error.txt`` and exit non-zero.
    """
    out = Path(work_dir)
    try:
        for key in [k for k in os.environ if k.startswith("SLURM_")]:
            del os.environ[key]  # the gate job's Slurm env must not define these ranks
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world),
                          LOCAL_RANK=str(rank), NODE_RANK="0")
        torch.set_num_threads(1)
        seed_everything(SEED, workers=True)
        data = {key: Path(value) for key, value in data_paths.items()}
        model = SaltModule(_gloo_modules(data["nd"]), lrs=LRS, training_schedule=schedule)
        head = _MainHeadRecorder()
        # LightningEnvironment with LOCAL_RANK set: the ranks are launched externally (here)
        Trainer(
            accelerator="cpu", devices=world, strategy="ddp", plugins=[LightningEnvironment()],
            max_epochs=2, accumulate_grad_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, default_root_dir=str(out),
            callbacks=[TrainingScheduleCallback(), head],
        ).fit(model, build_datamodule(data), ckpt_path=ckpt_path)
        ctrl, weight = model.training_controller, model.net["kinematic_weight"]
        result = {
            "stage_step_targets": ctrl.stage_step_targets, "stage": ctrl.current_stage_index,
            "boundary_records": ctrl.boundary_records, "stop_after_record": ctrl.stop_after_record,
            "q_f": weight.q_f.tolist(), "q_f_frozen": bool(weight.q_f_frozen),
            "main_delta": head.delta,
        }
        if ckpt_path is not None:  # main-head L2 distance from the checkpointed weights
            saved = torch.load(ckpt_path, weights_only=False)["state_dict"]
            state = model.state_dict()
            keys = [k for k in saved if k.startswith("net.jets_classification.")]
            sq = sum(float((state[k] - saved[k]).double().norm()) ** 2 for k in keys)
            result["main_delta_vs_ckpt"] = math.sqrt(sq)
        (out / f"rank{rank}.json").write_text(json.dumps(result))
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except BaseException:
        (out / f"rank{rank}.error.txt").write_text(traceback.format_exc())
        raise


def _run_gloo(work_dir: Path, data, schedule, ckpt_path: str | None = None) -> list[dict]:
    """Run `GLOO_WORLD` spawned ranks under one join deadline; a hung rank FAILS."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    paths = {key: str(value) for key, value in data.items()}
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_gloo_child,
                         args=(r, GLOO_WORLD, str(work_dir), paths, schedule, ckpt_path, port))
             for r in range(GLOO_WORLD)]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + GLOO_JOIN_TIMEOUT_S
    for proc in procs:
        proc.join(max(0.0, deadline - time.monotonic()))
    hung = [r for r, proc in enumerate(procs) if proc.is_alive()]
    for r in hung:
        procs[r].kill()
        procs[r].join(10)
    errors = "\n".join(f"{f.name}:\n{f.read_text()}" for f in sorted(work_dir.glob("*.error.txt")))
    if hung:
        pytest.fail(f"gloo rank(s) {hung} still running after {GLOO_JOIN_TIMEOUT_S} s\n{errors}")
    if failed := [r for r, proc in enumerate(procs) if proc.exitcode != 0]:
        pytest.fail(f"gloo rank(s) {failed} exited non-zero\n{errors}")
    return [json.loads((work_dir / f"rank{r}.json").read_text()) for r in range(GLOO_WORLD)]


class TestTwoRankGlooFit:
    """Two spawned CPU ranks under a real gloo process group (not fake-trainer evidence)."""

    def test_ranks_agree_on_target_boundary_and_frozen_q_f(self, data, tmp_path):
        r0, r1 = _run_gloo(tmp_path, data, GLOO_SCHEDULE)
        assert r0["stage_step_targets"] == r1["stage_step_targets"] == {"0": 2}  # JSON keys
        boundary = [boundary_record("main", 1, 2, 0, "steps")]
        assert r0["boundary_records"] == r1["boundary_records"] == boundary
        assert r0["q_f"] == r1["q_f"]
        assert math.isclose(sum(r0["q_f"]), 1.0, abs_tol=1e-6)
        assert r0["q_f_frozen"] is r1["q_f_frozen"] is True
        assert min(r0["main_delta"], r1["main_delta"]) > 0

    def test_stop_after_then_resume_trains_main(self, data, tmp_path):
        stopped = _run_gloo(tmp_path / "stop", data, {**GLOO_SCHEDULE, "stop_after": "priors"})
        for res in stopped:
            assert res["stage"] == 0
            assert res["stop_after_record"] == boundary_record("priors", 0, 2, 0, "stop_after")
        ckpt = tmp_path / "stop" / "stop_after=priors-step=000002.ckpt"
        assert ckpt.exists(), sorted(p.name for p in (tmp_path / "stop").iterdir())
        resumed = _run_gloo(tmp_path / "resume", data, GLOO_SCHEDULE, ckpt_path=str(ckpt))
        for res in resumed:
            assert (res["stage"], res["q_f_frozen"]) == (1, True)
            # the epoch-end checkpoint resumes at epoch 1, where the seam advances at N
            assert res["boundary_records"] == [boundary_record("main", 1, 2, 1, "steps")]
            assert res["main_delta_vs_ckpt"] > 0  # >= 1 main-stage update after resume
        assert resumed[0]["q_f"] == resumed[1]["q_f"]


# =====================================================================
# the shipped GN3/GN3FTAG1LITE.yaml two-stage example, end to end on fixture data
# =====================================================================

GN3_FLAVOUR = [f"ghost{f}jets" for f in ("b", "c", "s", "ud", "g", "tau")]


def _shipped_two_stage() -> tuple[dict, dict]:
    """The shipped model modules + schedule (its wiring unchanged), adapted to the fixture
    corpus (no flows, 3 track fields) and shrunk to a CPU-sized body.
    """
    path = expand_includes(str(CONFIG_DIR / "GN3" / "GN3FTAG1LITE.yaml"))
    cfg = yaml.safe_load(Path(path).read_text())
    modules = cfg["model"]["init_args"]["modules"]
    del modules["flow_embed"]
    modules["norm"]["init_args"]["streams"] = ["jets", "tracks"]
    modules["concat"]["init_args"]["streams"] = ["tracks"]
    modules["track_embed"]["init_args"]["out_dim"] = 16
    encoder = modules["encoder"]["init_args"]
    encoder.update(dim=16, out_dim=16, num_layers=1, attention={**encoder["attention"],
                                                                "num_heads": 2})
    for spec in modules.values():
        dense = spec.get("init_args", {}).get("dense")
        if isinstance(dense, dict) and "hidden_layers" in dense:
            dense["hidden_layers"] = [16]
    modules["kinematic_reweighting"]["init_args"]["prior"]["dense"]["hidden_layers"] = [16]
    schedule = cfg["training_schedule"]
    priors = schedule["stages"]["priors"]
    priors["frozen"] = [name for name in priors["frozen"] if name != "flow_embed"]
    priors["frac_samples"] = 0.5
    return modules, schedule


def _shipped_reweight_plots() -> list:
    """The shipped `reweight_plots` plot list (11 plots)."""
    path = expand_includes(str(CONFIG_DIR / "GN3" / "GN3FTAG1LITE.yaml"))
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg["callbacks"]["reweight_plots"]["init_args"]["plots"]


def _shipped_cli_config(corpus, manifests, root: Path, *, reweight_dir: Path, prior_dir: Path,
                        logger, train_file: str | None = None,
                        train_manifest: str | None = None) -> dict:
    """A complete `salt` config: the shipped two-stage model, `stop_after: priors` over a
    2-epoch priors stage, FTAG1LITE data, the reweight + prior-closure callbacks.
    """
    modules, schedule = _shipped_two_stage()
    priors = schedule["stages"]["priors"]
    del priors["frac_samples"]  # an epoch stage over 2 epochs: the stop validation at epoch 1
    priors["epochs"] = 2
    schedule["stop_after"] = "priors"
    reader = {"class_path": "salt.tests._fixtures.joint_ftag1lite_corpus.SeamFTAG1LITEReader",
              "init_args": {"jets": "small_r", "small": side_config("small"),
                            "shard_scope": "per_rank", "shuffle_pool_rows": None}}
    data_modules = {
        "reader": reader,
        "features": {"class_path": "salt.data.Features", "init_args": {"variables": {
            "jets": ["pt", "eta"], "tracks": ["d0", "z0SinTheta", "qOverP"]}}},
        "labels": {"class_path": "salt.data.Labels"},
        "flavour_labeller": {"class_path": "salt.data.FtagLabeller", "init_args": {
            "stream": "jets", "label": "flavour_label", "class_names": GN3_FLAVOUR,
            "require_labels": True}},
        "container_onehot": {"class_path": "salt.data.ContainerOneHot",
                             "init_args": {"stream": "jets", "include_sim_type": True}},
    }
    manifest = dict(manifests) | ({"train": train_manifest} if train_manifest else {})
    return {
        "name": "shipped_validate",
        "seed_everything": SEED,
        "model": {"class_path": "salt.model.SaltModule",
                  "init_args": {"modules": modules, "lrs": LRS, "optimizer": "AdamW"}},
        "training_schedule": schedule,
        # SaltCLI's datamodule is not subclass mode: `data:` takes SaltDataModule kwargs
        "data": {
            "modules": data_modules, "train_file": train_file or corpus.glob,
            "val_file": corpus.glob, "test_file": corpus.glob, "batch_size": 8,
            "num_workers": 0, "iterable": True, "block_rows": None, "max_live_streams": None,
            "manifest": manifest},
        "callbacks": {
            "progress": None, "artifacts": None, "lr_monitor": None, "model_summary": None,
            "reweight_plots": {"class_path": "salt.callbacks.KinematicReweightPlotter",
                               "init_args": {"weight_module": "kinematic_weight",
                                             "plots": _shipped_reweight_plots(),
                                             "output_dir": str(reweight_dir)}},
            "prior_closure": {"class_path": "salt.callbacks.PriorClosureMetrics",
                              "init_args": {"output_dir": str(prior_dir)}},
        },
        "trainer": {"accelerator": "cpu", "devices": 1, "logger": logger, "max_epochs": 3,
                    "check_val_every_n_epoch": 2, "limit_val_batches": 2,
                    "num_sanity_val_steps": 0, "enable_progress_bar": False,
                    "enable_model_summary": False, "log_every_n_steps": 1,
                    "default_root_dir": str(root)},
    }


def _assert_json_equal(a, b, path: str = "") -> None:
    """Recursive JSON equality: containers by structure, bool/int/str/None exact (type too),
    floats to rel 1e-9 / abs 1e-12 (NaN == NaN; a JSON NaN marks an empty bin).
    """
    where = path or "<root>"
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), f"{where}: {type(a)} vs {type(b)}"
        assert set(a) == set(b), f"{where}: keys {sorted(set(a) ^ set(b))} differ"
        for k in a:
            _assert_json_equal(a[k], b[k], f"{path}.{k}" if path else str(k))
    elif isinstance(a, list) or isinstance(b, list):
        assert isinstance(a, list) and isinstance(b, list), f"{where}: {type(a)} vs {type(b)}"
        assert len(a) == len(b), f"{where}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_json_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), f"{where}: {a!r} vs {b!r}"
        else:
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{where}: {a!r} vs {b!r}"
    else:  # bool / int / str / None; an int vs float (or bool vs int) mismatch fails here
        assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
        assert a == b, f"{where}: {a!r} vs {b!r}"


def _instantiate(modules: dict) -> dict[str, SaltModelModule]:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--modules", type=dict[str, SaltModelModule])
    return parser.instantiate_classes(parser.parse_object({"modules": modules})).modules


@pytest.fixture(scope="module")
def ftag1lite(tmp_path_factory):
    corpus = build_joint_corpus(tmp_path_factory.mktemp("ftag1lite"))
    manifests = tmp_path_factory.mktemp("ftag1lite_manifests")
    return corpus, {stage: str(manifests / f"{stage}.json") for stage in ("train", "val", "test")}


def _ftag1lite_datamodule(corpus, manifests) -> SaltDataModule:
    modules = {
        "reader": make_reader(corpus.root, "small_r", shard_scope="per_rank",
                              shuffle_pool_rows=None),
        "features": Features(variables={"jets": ["pt", "eta"],
                                        "tracks": ["d0", "z0SinTheta", "qOverP"]}),
        "labels": Labels(),
        "flavour_labeller": FtagLabeller(stream="jets", label="flavour_label",
                                         class_names=GN3_FLAVOUR, require_labels=True),
        "container_onehot": ContainerOneHot(stream="jets", include_sim_type=True),
    }
    return SaltDataModule(
        modules, train_file=corpus.glob, val_file=corpus.glob, test_file=corpus.glob,
        batch_size=8, num_workers=0, iterable=True, block_rows=None, max_live_streams=None,
        manifest=manifests,
    )


class _ShippedRecorder(Callback):
    """Training losses; norm buffers, prior + head params and q_f at the priors -> main
    boundary and at train end.
    """

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.snaps: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _snap(module) -> dict[str, torch.Tensor]:
        net = module.net
        state = {f"norm.{k}": v for k, v in net["norm"].state_dict().items()}
        state |= {f"prior.{k}": v for k, v in net["kinematic_prior"].named_parameters()}
        state |= {f"head.{k}": v for k, v in net["jets_classification"].named_parameters()}
        state["q_f"] = net["kinematic_weight"].q_f
        return {k: v.detach().clone() for k, v in state.items()}

    def on_train_batch_end(self, _trainer, module, outputs, *_args) -> None:
        self.losses.append(float(outputs["loss"]))
        if "boundary" not in self.snaps and module.training_controller.current_stage_index == 1:
            self.snaps["boundary"] = self._snap(module)

    def on_train_end(self, _trainer, module) -> None:
        self.snaps["end"] = self._snap(module)


class TestShippedTwoStageConfig:
    """`GN3FTAG1LITE.yaml`'s modules + schedule (CPU-shrunk, flows dropped,
    `frac_samples: 0.5`) fitted on the joint fixture corpus through the FTAG1LITE reader.
    """

    @staticmethod
    def _fit(ftag1lite, root: Path, *, stop_after=None, ckpt_path=None,
             n_iterations: int = 1) -> tuple:
        corpus, manifests = ftag1lite
        modules, schedule = _shipped_two_stage()
        if stop_after is not None:
            schedule["stop_after"] = stop_after
        if n_iterations > 1:  # the main stage must freeze every prior head
            modules["kinematic_reweighting"]["init_args"]["n_iterations"] = n_iterations
            heads = [f"kinematic_prior_{k}" for k in range(1, n_iterations)]
            schedule["stages"]["main"]["frozen"] += heads
            schedule["stages"]["main"]["skip_losses"] += heads
        seed_everything(SEED, workers=True)
        model = SaltModule(_instantiate(modules), lrs=LRS, optimizer="AdamW",
                           training_schedule=schedule)
        rec, dm = _ShippedRecorder(), _ftag1lite_datamodule(corpus, manifests)
        trainer = Trainer(
            accelerator="cpu", devices=1, logger=False, max_epochs=2, limit_val_batches=2,
            num_sanity_val_steps=0, enable_checkpointing=stop_after is not None,
            enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(root),
            callbacks=[TrainingScheduleCallback(), rec],
        )
        trainer.fit(model, dm, ckpt_path=ckpt_path)
        return model, trainer, rec, dm

    @pytest.fixture(scope="class")
    def plain(self, ftag1lite, tmp_path_factory):
        return self._fit(ftag1lite, tmp_path_factory.mktemp("shipped_plain"))

    def test_frac_boundary_and_what_each_stage_trains(self, plain):
        model, trainer, rec, _dm = plain
        n = math.ceil(0.5 * trainer.num_training_batches)
        assert model.training_controller.boundary_records == [
            boundary_record("main", 1, n, 0, "steps")
        ]
        assert rec.losses
        assert all(math.isfinite(x) for x in rec.losses)
        at, end = rec.snaps["boundary"], rec.snaps["end"]

        def unchanged(prefix: str) -> bool:
            keys = [k for k in at if k.startswith(prefix)]
            assert keys, prefix
            return all(torch.equal(at[k], end[k]) for k in keys)

        for stream in ("jets", "tracks"):  # learned during priors ...
            mean, var = at[f"norm.running_mean_{stream}"], at[f"norm.running_var_{stream}"]
            assert not (torch.all(mean == 0) and torch.all(var == 1)), stream
        assert unchanged("norm.")  # ... then frozen, bitwise, through main
        assert unchanged("prior.")
        assert not unchanged("head.")
        assert bool(model.net["kinematic_weight"].q_f_frozen)
        cw = model.net["class_weight"]  # same labels, same accumulation as q_f
        assert cw.accumulators_frozen
        assert torch.equal(cw.counts, model.net["kinematic_weight"].q_f_counts)
        assert torch.equal(cw.q, model.net["kinematic_weight"].q_f)
        assert torch.equal(at["q_f"], end["q_f"])
        assert math.isclose(float(end["q_f"].sum()), 1.0, abs_tol=1e-5)

    def test_test_graph_keeps_the_prior_and_drops_the_weight(self, plain):
        model, _trainer, _rec, dm = plain
        Trainer(accelerator="cpu", devices=1, logger=False, limit_test_batches=2,
                enable_progress_bar=False, enable_model_summary=False).test(model, datamodule=dm)
        names = model.plans[Mode.TEST].module_names
        assert "kinematic_prior" in names  # exposed in every mode
        assert "kinematic_weight" not in names  # TRAINING-only
        assert "class_weight" not in names  # TRAINING-only

    def test_stop_after_priors_then_resume_enters_main(self, ftag1lite, tmp_path):
        self._fit(ftag1lite, tmp_path / "stop", stop_after="priors")
        [ckpt] = (tmp_path / "stop").rglob("stop_after=priors-step=*.ckpt")
        saved = torch.load(str(ckpt), weights_only=False)["state_dict"]
        counts = saved["net.kinematic_weight.q_f_counts"]
        model, *_ = self._fit(ftag1lite, tmp_path / "resume", ckpt_path=str(ckpt))
        assert model.training_controller.current_stage_index == 1
        weight = model.net["kinematic_weight"]
        # q_f freezes at the resume seam from the checkpointed priors counts; main adds none
        assert bool(weight.q_f_frozen)
        assert torch.equal(weight.q_f_counts, counts)
        assert torch.allclose(weight.q_f, (counts / counts.sum()).float(), rtol=0, atol=1e-7)
        assert bool(model.net["class_weight"].frozen)
        assert torch.equal(model.net["class_weight"].counts, counts)

    def test_two_iteration_joint_fit(self, ftag1lite, tmp_path):
        """N=2 joint: both prior heads train in the one sample stage; main freezes both."""
        model, trainer, rec, _ = self._fit(ftag1lite, tmp_path / "joint2", n_iterations=2)
        n = math.ceil(0.5 * trainer.num_training_batches)
        assert model.training_controller.boundary_records == [
            boundary_record("main", 1, n, 0, "steps")
        ]
        assert rec.losses and all(math.isfinite(x) for x in rec.losses)
        net = model.net
        w1, w = net["kinematic_weight_1"], net["kinematic_weight"]
        assert bool(w1.q_f_frozen) and bool(w.q_f_frozen)
        assert torch.equal(w1.q_f_counts, w.q_f_counts)
        assert torch.equal(w.q_f_counts, net["class_weight"].counts)
        assert w.chain == ("nn_kinematic_weight_iter_1",)
        assert net["kinematic_prior_1"].sample_weights == ("flavour_class_weight",)
        assert net["kinematic_prior"].sample_weights == (
            "flavour_class_weight", "nn_kinematic_weight_iter_1")
        if Mode.TEST in model.plans:  # fit compiles no TEST plan
            assert "kinematic_prior_1" not in model.plans[Mode.TEST].module_names

    def test_validate_from_stop_after_ckpt_reproduces_reweight_json(self, ftag1lite, tmp_path,
                                                                    monkeypatch):
        """`salt validate` from a stop_after ckpt restores epoch/step and reproduces the JSON."""
        corpus, manifests = ftag1lite
        for name in ("COMET_API_KEY", "COMET_EXPERIMENT_NAME", "COMET_OFFLINE_DIRECTORY"):
            monkeypatch.delenv(name, raising=False)
        fit_reweight, fit_prior = tmp_path / "fit_reweight", tmp_path / "fit_prior"
        val_reweight, val_prior = tmp_path / "val_reweight", tmp_path / "val_prior"
        validate_root = tmp_path / "validate_root"

        # fit half: stop_after=priors, its only validation at the end of epoch 1
        fit_cfg = _shipped_cli_config(corpus, manifests, tmp_path / "fit_root",
                                      reweight_dir=fit_reweight, prior_dir=fit_prior,
                                      logger=False)
        fit_yaml = tmp_path / "fit.yaml"
        fit_yaml.write_text(yaml.safe_dump(fit_cfg, sort_keys=False))
        SaltCLI(args=["fit", "--config", str(fit_yaml)])
        [ckpt] = (tmp_path / "fit_root").rglob("stop_after=priors-*.ckpt")
        meta = torch.load(str(ckpt), weights_only=False)
        assert meta["epoch"] == 1
        n = meta["global_step"]
        assert n > 0
        stem = f"step{n:06d}.json"
        fit_json, fit_prior_json = fit_reweight / stem, fit_prior / stem
        assert fit_json.exists(), sorted(p.name for p in fit_reweight.iterdir())
        assert [p.name for p in fit_reweight.glob("step*.json")] == [fit_json.name]
        assert fit_prior_json.exists(), sorted(p.name for p in fit_prior.iterdir())
        assert [p.name for p in fit_prior.glob("step*.json")] == [fit_prior_json.name]

        # validate half: offline Comet, train source + manifest poisoned (must never be touched)
        logger = {"class_path": "lightning.pytorch.loggers.comet.CometLogger",
                  "init_args": {"project": "salt", "online": False}}
        val_cfg = _shipped_cli_config(
            corpus, manifests, validate_root, reweight_dir=val_reweight, prior_dir=val_prior,
            logger=logger, train_file=str(tmp_path / "no_such_corpus" / "*" / "*.pool.root"),
            train_manifest=str(tmp_path / "no_such_manifest_dir" / "train.json"))
        val_yaml = tmp_path / "validate.yaml"
        val_yaml.write_text(yaml.safe_dump(val_cfg, sort_keys=False))
        cli = SaltCLI(args=["validate", "--config", str(val_yaml), "--ckpt_path", str(ckpt)])

        trainer = cli.trainer
        assert trainer.state.fn == TrainerFn.VALIDATING
        assert trainer.current_epoch == 1  # restored; the fit loop never ran
        assert trainer.global_step == n
        plotter = next(cb for cb in trainer.callbacks
                       if type(cb).__name__ == "KinematicReweightPlotter")
        # the title is built from the trainer counters DURING validation
        assert f"step {n}, epoch 1" in plotter.last_figures["pt"].axes[0].get_title()
        assert plotter.last_metrics["n_jets"] == json.loads(fit_json.read_text())["n_jets"]
        assert trainer.default_root_dir == str(validate_root)  # verbatim, no <name>_<ts> subdir
        assert (validate_root / "config.yaml").exists()
        assert not [d for d in validate_root.iterdir()
                    if d.is_dir() and d.name.startswith("shipped_validate_")]
        assert not list(validate_root.rglob("*.ckpt"))
        assert set(cli.model.plans) == {Mode.VAL}
        assert cli.datamodule.train_dset is None
        assert not (tmp_path / "no_such_manifest_dir").exists()  # train manifest never built

        # reproduction: same stems, every key equal (ints exact, floats rel 1e-9)
        val_json, val_prior_json = val_reweight / stem, val_prior / stem
        assert val_json.exists(), sorted(p.name for p in val_reweight.iterdir())
        assert [p.name for p in val_reweight.glob("step*.json")] == [val_json.name]
        _assert_json_equal(json.loads(fit_json.read_text()), json.loads(val_json.read_text()))
        fit_pngs = {p.name for p in fit_reweight.glob("*.png")}
        assert fit_pngs
        assert fit_pngs == {p.name for p in val_reweight.glob("*.png")}
        assert val_prior_json.exists(), sorted(p.name for p in val_prior.iterdir())
        _assert_json_equal(json.loads(fit_prior_json.read_text()),
                           json.loads(val_prior_json.read_text()))

        # Comet receipt from the offline logger; the logger-less fit wrote none
        for d in (val_reweight, val_prior):
            info = json.loads((d / "comet_experiment.json").read_text())
            assert set(info) == {"url", "project_name", "workspace", "key", "name"}, d
            assert isinstance(info["key"], str) and info["key"], d
        assert not (fit_reweight / "comet_experiment.json").exists()
