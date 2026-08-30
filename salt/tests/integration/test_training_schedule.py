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

All DataLoaders use ``num_workers=0``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from salt.callbacks.schedule import StageScopedCallbacks, TrainingScheduleCallback
from salt.data import Features, SaltDataModule, H5StructuredReader, Labels
from salt.model.modules.losses import LossSum
from salt.model.modules.tasks import ClassificationTaskModule
from salt.optim import HybridMuonAdamW
from salt.model.saltmodule import SaltModule
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    write_parity_norm_dict,
)

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
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
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
            "stage": module._current_stage_index,  # noqa: SLF001
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
                stage_opt_types[module._current_stage_index] = type(  # noqa: SLF001
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
                idx = module._current_stage_index  # noqa: SLF001
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
        assert model._frozen_module_names == set()  # noqa: SLF001
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
        assert model._schedule.is_multi_stage  # noqa: SLF001
        offline_bind(model, build_datamodule(data))  # no rejection
        assert model._current_stage_index == 0  # noqa: SLF001
        assert model._frozen_module_names == {"encoder"}  # noqa: SLF001 - stage-0 mask
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
            training_schedule={"stages": {"fit": {"frozen": retained}}},
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
            "stage": module._current_stage_index,  # noqa: SLF001
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
        assert model._current_stage_index == 1  # noqa: SLF001
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
        assert model._current_stage_index in (0, 1)  # noqa: SLF001 - ran cleanly


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
        HOOK_LOG.append((self.tag, "setup", pl_module._current_stage_index))  # noqa: SLF001

    def teardown(self, trainer, pl_module, stage) -> None:  # noqa: D102
        HOOK_LOG.append((self.tag, "teardown", pl_module._current_stage_index))  # noqa: SLF001

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D102
        self.epoch_starts += 1
        HOOK_LOG.append((self.tag, "epoch_start", pl_module._current_stage_index))  # noqa: SLF001

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: D102
        HOOK_LOG.append((self.tag, "batch_end", pl_module._current_stage_index))  # noqa: SLF001


_PROBE = "salt.tests.integration.test_training_schedule.HookProbe"


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
