"""W7 gates: per-stage early stopping (plan 12).

A stage may declare `early_stop`; the stage ends at whichever comes first — its
epoch cap or the early-stop trigger. A non-final trigger advances to the next
stage (freeze flip + optimizer/LR rebuild, the same path as an epoch-cap
boundary); a final-stage trigger ends the fit. Boundaries become data-dependent,
so completed transitions + the live patience counters are checkpointed and a
resume reconstructs them (not epoch arithmetic). A config with no `early_stop`
anywhere behaves — and checkpoints — bitwise-identically to the pre-W7 tip.

Gates (plan 12 W7):

- **G7a** legacy parity — a no-early-stop config's checkpoint `schedule` payload is
  exactly ``{stage_index, stage_name}`` (no new keys); the full W1–W6 + resume
  suites cover the bitwise-state parity.
- **G7b** two-stage: stage-0 `early_stop` triggers at epoch k < its epoch cap →
  the transition fires at k, the stage-1 optimizer owns the (now-unfrozen) stage-1
  trainable set, stage-1's OneCycle envelope is sized from its own budget, and the
  boundary record reads ``reason="early_stop"``. Stage 0's OneCycle envelope was
  truncated (fewer steps taken than its `total_steps`).
- **G7c** final-stage `early_stop` ends the fit before `max_epochs`; a non-final
  early-stop never sets `trainer.should_stop` (it advances instead).
- **G7d** resume: a mid-stage checkpoint with a partially-consumed patience counter
  restores `best_score`/`wait_count` exactly (patience continues identically); a
  checkpoint saved after an early-stopped boundary resumes into the correct stage.
  A changed `early_stop` criterion since the checkpoint is a hard `ConfigError`.

All DataLoaders use ``num_workers=0`` (agent memcg gotcha). Early-stop decisions
are driven by an injected, fully-deterministic monitor metric (`MetricInjector`),
so the stop epoch is independent of the toy model's actual val loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from salt.callbacks.schedule import TrainingScheduleCallback
from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
from salt.graph.errors import ConfigError
from salt.model.saltmodule import CKPT_KEY, SaltModule
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
MONITOR = "injected/metric"
FROZEN = "encoder"


# --- fixtures / builders -----------------------------------------------------


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("earlystop")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_dm(data) -> GraphDataModule:
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


def build_model(data, *, schedule=None) -> SaltModule:
    kwargs = {} if schedule is None else {"training_schedule": schedule}
    return SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, **kwargs)


def make_trainer(*, max_epochs: int, callbacks: list, enable_checkpointing: bool = False,
                 **kwargs) -> Trainer:
    # 5 train steps/epoch (1000 jets / batch 100 = 10, limited to 5).
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks, **kwargs,
    )


# --- instrumentation ---------------------------------------------------------


class MetricInjector(Callback):
    """Writes a deterministic `MONITOR` value into `callback_metrics` at each real
    validation-epoch end, so the early-stop decision is independent of the toy
    model's loss. Placed BEFORE the schedule callback so its value is present when
    the schedule callback reads it in `on_validation_end`.
    """

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.i = 0

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = self.values[min(self.i, len(self.values) - 1)]
        trainer.callback_metrics[MONITOR] = torch.tensor(float(value))
        self.i += 1


class StageRecorder(Callback):
    """Per-train-batch trace of the active stage, epoch, LR, OneCycle total_steps,
    and `trainer.should_stop`.
    """

    def __init__(self) -> None:
        self.trace: list[dict] = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        sched = trainer.lr_scheduler_configs[0].scheduler
        self.trace.append({
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "stage": module._current_stage_index,  # noqa: SLF001
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "total_steps": sched.total_steps,
            "should_stop": trainer.should_stop,
        })


class TrackerCapture(Callback):
    """Captures the module's early-stop tracker state at `on_train_start` (used to
    prove a mid-stage resume restored the counters).
    """

    def __init__(self) -> None:
        self.state: dict | None = None

    def on_train_start(self, trainer, module) -> None:
        tracker = module._early_stop_tracker  # noqa: SLF001
        self.state = tracker.state_dict() if tracker is not None else None


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


# --- G7a: legacy parity — checkpoint payload unchanged -----------------------


class TestG7aLegacyPayloadParity:
    def _saved_schedule_payload(self, data, tmp_path, schedule, *, max_epochs: int = 1) -> dict:
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=schedule)
        mc = ModelCheckpoint(dirpath=str(tmp_path), filename="{epoch}", save_top_k=-1)
        cbs: list = [mc]
        if schedule is not None:
            cbs.insert(0, TrainingScheduleCallback())
        make_trainer(max_epochs=max_epochs, callbacks=cbs, enable_checkpointing=True).fit(
            model, build_dm(data)
        )
        ckpt = torch.load(str(tmp_path / "epoch=0.ckpt"), weights_only=False)
        return ckpt[CKPT_KEY]["schedule"]

    def test_no_schedule_payload_has_only_index_and_name(self, data, tmp_path):
        payload = self._saved_schedule_payload(data, tmp_path, None)
        assert set(payload) == {"stage_index", "stage_name"}
        assert payload == {"stage_index": 0, "stage_name": "fit"}

    def test_multi_stage_no_early_stop_payload_has_only_index_and_name(self, data, tmp_path):
        # a freezing multi-stage schedule WITHOUT early_stop must not gain any new
        # W7 checkpoint keys (the has_early_stop master switch is off). epoch 0 is in
        # stage a (owns [0,1)); saved at end of epoch 0 → stage_index 0.
        schedule = {"stages": {"a": {"epochs": 1, "frozen": [FROZEN]}, "b": {"frozen": []}}}
        payload = self._saved_schedule_payload(data, tmp_path, schedule, max_epochs=2)
        assert set(payload) == {"stage_index", "stage_name"}

    def test_early_stop_payload_carries_w7_keys(self, data, tmp_path):
        schedule = {"stages": {"fit": {"early_stop": {"monitor": MONITOR, "patience": 5}}}}
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=schedule)
        mc = ModelCheckpoint(dirpath=str(tmp_path), filename="{epoch}", save_top_k=-1)
        make_trainer(
            max_epochs=1, callbacks=[MetricInjector([1.0]), TrainingScheduleCallback(), mc],
            enable_checkpointing=True,
        ).fit(model, build_dm(data))
        ckpt = torch.load(str(tmp_path / "epoch=0.ckpt"), weights_only=False)
        payload = ckpt[CKPT_KEY]["schedule"]
        assert {"stage_start_epoch", "boundaries", "early_stop_state"} <= set(payload)
        assert payload["early_stop_state"]["best_score"] == 1.0


# --- G7b: two-stage, stage-0 early-stops before its epoch cap -----------------


class TestG7bStageZeroEarlyStop:
    SCHEDULE = {
        "stages": {
            # cap = 6 epochs, but patience=2 with a flat monitor stops it at epoch 3
            "warmup": {
                "epochs": 6, "frozen": [FROZEN], "lrs": {"max": 5e-3},
                "early_stop": {"monitor": MONITOR, "mode": "min", "patience": 2},
            },
            "full": {"frozen": [], "lrs": {"max": 1e-3}},
        }
    }

    def _run(self, data) -> tuple[SaltModule, StageRecorder]:
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=self.SCHEDULE)
        rec = StageRecorder()
        # flat metric → no improvement → stop after patience+1 checks (epoch 2 val)
        injector = MetricInjector([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        make_trainer(
            max_epochs=8, callbacks=[injector, TrainingScheduleCallback(), rec]
        ).fit(model, build_dm(data))
        return model, rec

    def test_transition_fires_before_cap(self, data):
        model, rec = self._run(data)
        stage0 = [t for t in rec.trace if t["stage"] == 0]
        stage1 = [t for t in rec.trace if t["stage"] == 1]
        # stage 0 = epochs 0,1,2 (15 steps) — stopped at epoch 3 START, well before
        # its 6-epoch cap; stage 1 owns the rest to max_epochs (epochs 3..7).
        assert {t["epoch"] for t in stage0} == {0, 1, 2}
        assert len(stage0) == 15
        assert min(t["epoch"] for t in stage1) == 3

    def test_boundary_record_reason_is_early_stop(self, data):
        model, _ = self._run(data)
        records = model._boundary_records  # noqa: SLF001
        assert len(records) == 1
        assert records[0]["reason"] == "early_stop"
        assert records[0]["stage_index"] == 1
        assert records[0]["epoch"] == 3
        assert records[0]["global_step"] == 15

    def test_stage0_onecycle_envelope_truncated(self, data):
        _model, rec = self._run(data)
        stage0 = [t for t in rec.trace if t["stage"] == 0]
        # the stage-0 OneCycle was sized for its 6-epoch cap (30 steps) but only 15
        # steps were taken — the envelope is truncated mid-curve (D-ES).
        assert stage0[0]["total_steps"] == 30
        assert len(stage0) == 15

    def test_stage1_optimizer_owns_unfrozen_encoder(self, data):
        model, _ = self._run(data)
        assert model._current_stage_index == 1  # noqa: SLF001
        opt_ids = {
            id(p) for group in model.trainer.optimizers[0].param_groups for p in group["params"]
        }
        # encoder was frozen in stage 0, unfrozen in stage 1 → in the optimizer now
        assert _module_param_ids(model, FROZEN) <= opt_ids

    def test_non_final_early_stop_never_sets_should_stop(self, data):
        _model, rec = self._run(data)
        # a non-final early-stop advances the stage; it must never end the fit.
        assert not any(t["should_stop"] for t in rec.trace)
        # the fit ran to max_epochs (stage 1 has no early_stop)
        assert max(t["epoch"] for t in rec.trace) == 7


# --- G7c: final-stage early-stop ends the fit --------------------------------


class TestG7cFinalStageEarlyStop:
    def test_single_stage_early_stop_ends_before_max_epochs(self, data):
        schedule = {"stages": {"fit": {"early_stop": {"monitor": MONITOR, "patience": 2}}}}
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=schedule)
        rec = StageRecorder()
        make_trainer(
            max_epochs=10,
            callbacks=[MetricInjector([1.0] * 10), TrainingScheduleCallback(), rec],
        ).fit(model, build_dm(data))
        # patience 2 on a flat metric → stop after epoch 2's validation; the fit
        # ends well before max_epochs=10.
        assert max(t["epoch"] for t in rec.trace) == 2
        assert model.trainer.should_stop

    def test_two_stage_final_early_stop_ends_fit(self, data):
        schedule = {
            "stages": {
                "warmup": {"epochs": 2},
                "full": {"early_stop": {"monitor": MONITOR, "patience": 1}},
            }
        }
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=schedule)
        rec = StageRecorder()
        # improving during stage 0 (irrelevant — no early_stop there), then flat in
        # stage 1: epoch2 seeds best, epoch3 no-improve wait=1==patience → stop.
        make_trainer(
            max_epochs=10,
            callbacks=[MetricInjector([9, 9, 5, 5, 5, 5]), TrainingScheduleCallback(), rec],
        ).fit(model, build_dm(data))
        assert max(t["stage"] for t in rec.trace) == 1  # reached the final stage
        assert max(t["epoch"] for t in rec.trace) < 9  # ended before max_epochs


# --- G7d: resume across data-dependent boundaries ----------------------------


class TestG7dResume:
    SCHEDULE = {
        "stages": {
            "warmup": {
                "epochs": 6, "frozen": [FROZEN],
                "early_stop": {"monitor": MONITOR, "mode": "min", "patience": 2},
            },
            "full": {"frozen": []},
        }
    }

    def _leg1(self, data, ckpt_dir: Path, save_epoch: int, values: list[float]) -> Path:
        seed_everything(SEED, workers=True)
        model = build_model(data, schedule=self.SCHEDULE)
        mc = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="{epoch}", save_top_k=-1, every_n_epochs=1
        )
        # schedule callback BEFORE ModelCheckpoint so the tracker is updated before
        # the checkpoint serialises it.
        make_trainer(
            max_epochs=8,
            callbacks=[MetricInjector(values), TrainingScheduleCallback(), mc],
            enable_checkpointing=True,
        ).fit(model, build_dm(data))
        ckpt = ckpt_dir / f"epoch={save_epoch}.ckpt"
        assert ckpt.exists()
        return ckpt

    def test_mid_stage_patience_restored_exactly(self, data, tmp_path):
        # flat metric, patience 2: epoch0 seeds best(wait0), epoch1 wait1, epoch2
        # wait2==patience → stop. Save at end of epoch1 (INSIDE stage 0, wait=1).
        values = [1.0] * 8
        ckpt = self._leg1(data, tmp_path / "ck", save_epoch=1, values=values)
        payload = torch.load(str(ckpt), weights_only=False)[CKPT_KEY]["schedule"]
        assert payload["stage_index"] == 0
        assert payload["early_stop_state"]["best_score"] == 1.0
        assert payload["early_stop_state"]["wait_count"] == 1

        # resume: the tracker must restore wait=1/best=1.0 at on_train_start.
        seed_everything(SEED, workers=True)
        m2 = build_model(data, schedule=self.SCHEDULE)
        cap = TrackerCapture()
        make_trainer(
            max_epochs=8, callbacks=[MetricInjector(values[2:]), TrainingScheduleCallback(), cap]
        ).fit(m2, build_dm(data), ckpt_path=str(ckpt))
        assert cap.state is not None
        assert cap.state["best_score"] == 1.0
        assert cap.state["wait_count"] == 1

    def test_resume_after_early_stopped_boundary_enters_correct_stage(self, data, tmp_path):
        # stop stage 0 at epoch 2 val (patience 2, flat); the boundary crosses at
        # epoch 3 start, so epoch=3.ckpt is the FIRST checkpoint written in stage 1.
        values = [1.0] * 8
        ckpt = self._leg1(data, tmp_path / "ck2", save_epoch=3, values=values)
        payload = torch.load(str(ckpt), weights_only=False)[CKPT_KEY]["schedule"]
        assert payload["stage_index"] == 1  # already advanced into stage 1
        assert any(r["reason"] == "early_stop" for r in payload["boundaries"])

        seed_everything(SEED, workers=True)
        m2 = build_model(data, schedule=self.SCHEDULE)
        make_trainer(max_epochs=8, callbacks=[TrainingScheduleCallback()]).fit(
            m2, build_dm(data), ckpt_path=str(ckpt)
        )
        assert m2._current_stage_index == 1  # noqa: SLF001 - resumed into stage 1
        # the stage-1 optimizer owns the unfrozen encoder
        opt_ids = {
            id(p) for group in m2.trainer.optimizers[0].param_groups for p in group["params"]
        }
        assert _module_param_ids(m2, FROZEN) <= opt_ids

    def test_changed_criterion_since_checkpoint_is_config_error(self, data, tmp_path):
        ckpt = self._leg1(data, tmp_path / "ck3", save_epoch=1, values=[1.0] * 8)
        # resume with a DIFFERENT patience → fingerprint mismatch → fail loud.
        changed = {
            "stages": {
                "warmup": {
                    "epochs": 6, "frozen": [FROZEN],
                    "early_stop": {"monitor": MONITOR, "mode": "min", "patience": 5},
                },
                "full": {"frozen": []},
            }
        }
        seed_everything(SEED, workers=True)
        m2 = build_model(data, schedule=changed)
        with pytest.raises(ConfigError, match="criterion changed since the checkpoint"):
            make_trainer(max_epochs=8, callbacks=[TrainingScheduleCallback()]).fit(
                m2, build_dm(data), ckpt_path=str(ckpt)
            )
