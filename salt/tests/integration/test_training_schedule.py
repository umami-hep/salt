"""W3 gates: multi-stage `training_schedule` execution + legacy desugaring.

Gates (plan 01 W3):

- **G3a** — a legacy config (no `training_schedule`) desugars to a single `fit`
  stage and trains bitwise-identically to the pre-desugar behaviour. The
  cross-version bitwise check against base @ 7bee757 is run out-of-suite; here
  the structural half is covered (single-stage total_steps == the whole-run
  `estimated_stepping_batches`, optimizer holds every param).
- **G3b** — an explicit single-`fit`-stage schedule trains bitwise-identically to
  the no-schedule (desugared) config (same seed → equal state_dicts).
- **G3c** — a 2-stage schedule shows two OneCycle envelopes with the boundary at
  the correct global step, per-stage total_steps summing to the whole-run
  estimate, the stage-2 optimizer owning the newly-unfrozen params (which move),
  while the stage-1-frozen params stayed bitwise-fixed during stage 1.
- **G3d** — after a boundary, `trainer.optimizers`, `trainer.lr_scheduler_configs`
  and `pl_module.optimizers()` all point at the rebuilt objects (S1 identity).
- **G3e** — a 3-stage schedule with a per-stage `optimizer` override
  (AdamW→lion) rebuilds the right optimizer class per stage; HybridMuonAdamW
  rebuilds its Muon/AdamW split over the new trainable set across a boundary.

All DataLoaders use ``num_workers=0`` (agent memcg gotcha).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lightning import Callback, Trainer, seed_everything

from salt.callbacks.schedule import TrainingScheduleCallback
from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
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

LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.25}


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


def build_datamodule(data) -> GraphDataModule:
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    return GraphDataModule(
        modules, batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


def build_model(data, **kwargs) -> SaltModule:
    return SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, **kwargs)


def make_trainer(*, max_epochs: int, callbacks: list, **kwargs) -> Trainer:
    # 5 train steps/epoch (1000 jets / batch 100 = 10, limited to 5) so a 4-epoch
    # fit has estimated_stepping_batches == 20 (a 2-stage split of [10, 10]); a
    # per-stage OneCycle needs >~8 steps or its warmup phase degenerates.
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks, **kwargs,
    )


def _param_ids(params) -> set[int]:
    return {id(p) for p in params}


def _optimizer_param_ids(opt) -> set[int]:
    return {id(p) for group in opt.param_groups for p in group["params"]}


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


# --- G3a: legacy desugar structural parity ----------------------------------


class TestG3aDesugarStructuralParity:
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


# --- G3b: explicit single-fit-stage ≡ legacy (bitwise) ----------------------


class TestG3bSingleStageEqualsLegacy:
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


# --- G3c / G3d: 2-stage execution + rebuild-reference tracking ---------------


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


class TestG3cTwoStageEnvelopes:
    def test_two_envelopes_boundary_allocation_and_movement(self, data):
        model, rec = _run_two_stage(data)

        stage0 = [t for t in rec.trace if t["stage"] == 0]
        stage1 = [t for t in rec.trace if t["stage"] == 1]
        assert len(stage0) == 10 and len(stage1) == 10  # boundary at global step 10

        # per-stage total_steps sum to the whole-run estimate (Gotcha #1)
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


class TestG3dRebuildReferenceTracking:
    def test_all_refs_point_at_rebuilt_objects(self, data):
        _model, rec = _run_two_stage(data)
        assert rec.refs["trainer_is_strategy"]
        assert rec.refs["module_is_trainer"]
        assert rec.refs["sched_wraps_opt"]


# --- G3e: per-stage optimizer override + HybridMuonAdamW across a boundary ----


class TestG3ePerStageOptimizer:
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
