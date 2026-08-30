"""Checkpoint/resume state identity across `training_schedule` stage boundaries,
and per-stage early stopping.

Merges ``test_early_stop.py`` — early-stop's whole difficulty is that stage
boundaries become *data-dependent*, so completed transitions + the live
patience counters have to be checkpointed and reconstructed on resume; half
that file was already resume tests on the same axis as this one.

`salt2 fit --ckpt_path <ckpt>` must resume a multi-stage schedule from any
epoch — inside stage 0, exactly at a stage boundary, or inside a later stage —
producing state (weights, optimizer moments, per-step LR trace) identical to the
uninterrupted run.

Covered:

- **probe** — pins the Lightning 2.6.5 restore order these tests rely on:
  ``setup("fit")`` -> ``on_load_checkpoint`` -> ``configure_optimizers`` ->
  optimizer-state restore. The stage restore in `SaltModule.on_load_checkpoint`
  is only correct if it runs *before* the optimizer is rebuilt.
- resume inside stage 0 (from a real val-loss `ModelCheckpoint`) ≡
  uninterrupted: final `state_dict` bitwise-equal, optimizer moments equal.
- resume exactly at a stage boundary: exactly one rebuild fires
  post-restore; the stage-1 optimizer owns exactly the stage-1 trainable set;
  final state bitwise-equal. Covered for AdamW and HybridMuonAdamW.
- resume mid-stage-1: NO rebuild at the resume epoch, the restored
  optimizer moments are kept (not reset), frozen modules are frozen + eval from
  the first restored step; final state bitwise-equal.
- per-step LR trace across the resume point identical to the
  uninterrupted run (asserted inside each resume case).
- parity guard — a no-schedule (legacy) config resumes bitwise-identically
  to its uninterrupted run (the checkpoint save/restore additions do not perturb
  the desugared single-stage path).
- a stage may declare `early_stop`; the stage ends at whichever comes first
  — its epoch cap or the early-stop trigger. A non-final trigger advances to
  the next stage (freeze flip + optimizer/LR rebuild, the same path as an
  epoch-cap boundary); a final-stage trigger ends the fit. A config with no
  `early_stop` anywhere behaves — and checkpoints — bitwise-identically to a
  build without the early-stop machinery.
- legacy parity — a no-early-stop config's checkpoint `schedule` payload is
  exactly ``{stage_index, stage_name}`` (no new keys).
- two-stage: stage-0 `early_stop` triggers at epoch k < its epoch cap →
  the transition fires at k, the stage-1 optimizer owns the (now-unfrozen)
  stage-1 trainable set, stage-1's OneCycle envelope is sized from its own
  budget, and the boundary record reads ``reason="early_stop"``. Stage 0's
  OneCycle envelope was truncated (fewer steps taken than its `total_steps`).
- final-stage `early_stop` ends the fit before `max_epochs`; a non-final
  early-stop never sets `trainer.should_stop` (it advances instead).
- early-stop resume: a mid-stage checkpoint with a partially-consumed
  patience counter restores `best_score`/`wait_count` exactly (patience
  continues identically); a checkpoint saved after an early-stopped boundary
  resumes into the correct stage. A changed `early_stop` criterion since the
  checkpoint is a hard `ConfigError`.

Determinism contract (what the bitwise equalities rely on / do NOT rely on):
the datamodule uses a **shuffle=False** train loader (`_DeterministicDataModule`)
and the fixture model has **no dropout** (default), so epoch e produces identical
batches and identical forward passes in the uninterrupted run and the resumed
run regardless of global-RNG state. Equality therefore does NOT rely on Lightning
restoring the global torch/numpy RNG across a fit resume (it does not by default)
— it relies only on the deterministic loader + dropout-free model + a fixed seed
for identical initial weights. Early-stop decisions are driven by an injected,
fully-deterministic monitor metric (`MetricInjector`), so the stop epoch is
independent of the toy model's actual val loss. All DataLoaders use
``num_workers=0``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from salt.callbacks.schedule import TrainingScheduleCallback
from salt.data import Features, SaltDataModule, H5StructuredReader, Labels
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
# stage-0 freezes the encoder (warmup), stage-1 unfreezes it (full finetune).
FROZEN = "encoder"
WARMUP_FULL = {
    "stages": {
        "warmup": {"epochs": 2, "frozen": [FROZEN], "lrs": {"max": 5e-3}},
        "full": {"frozen": [], "lrs": {"max": 1e-3}},
    }
}
MONITOR = "injected/metric"


# --- fixtures / builders -----------------------------------------------------


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("resume")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


class _DeterministicDataModule(SaltDataModule):
    """A SaltDataModule whose train loader does NOT shuffle — so epoch e yields
    the same batches every run (resume-equivalence needs a deterministic loader).
    """

    def train_dataloader(self):  # noqa: D102 - see class docstring
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


def build_model(data, *, schedule=None, optimizer="AdamW") -> SaltModule:
    # the fixture model is dropout-free (build_gn2v2_modules default dropout=0),
    # a precondition of the determinism contract in the module docstring.
    kwargs = {} if schedule is None else {"training_schedule": schedule}
    return SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, optimizer=optimizer, **kwargs)


def make_trainer(*, max_epochs: int, callbacks: list, enable_checkpointing: bool = False,
                 **kwargs) -> Trainer:
    # 5 train steps/epoch (1000 jets / batch 100 = 10, limited to 5).
    # enable_checkpointing must be True on the leg that carries a ModelCheckpoint.
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks, **kwargs,
    )


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


# =====================================================================
# checkpoint / resume
# =====================================================================

# --- instrumentation ---------------------------------------------------------


class _RebuildCounter(TrainingScheduleCallback):
    """A schedule callback that also counts the boundary rebuilds it performs."""

    def __init__(self) -> None:
        super().__init__()
        self.rebuild_epochs: list[int] = []

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        before = pl_module._current_stage_index  # noqa: SLF001
        super().on_train_epoch_start(trainer, pl_module)
        if pl_module._current_stage_index != before:  # noqa: SLF001
            self.rebuild_epochs.append(trainer.current_epoch)


class _Tracer(Callback):
    """Records the per-step LR trace + the first-epoch / freeze / optimizer state
    entering the run (used to prove resume continuity).
    """

    def __init__(self) -> None:
        self.lr: list[tuple[int, float]] = []
        self.first_epoch: int | None = None
        self.frozen_training_at_start: bool | None = None
        self.frozen_requires_grad_at_start: bool | None = None
        self.opt_state_at_start: dict | None = None

    def on_train_start(self, trainer, module) -> None:
        self.frozen_training_at_start = module.net[FROZEN].training
        self.frozen_requires_grad_at_start = any(
            p.requires_grad for p in module.net[FROZEN].parameters()
        )
        self.opt_state_at_start = copy.deepcopy(trainer.optimizers[0].state_dict())

    def on_train_epoch_start(self, trainer, module) -> None:
        if self.first_epoch is None:
            self.first_epoch = trainer.current_epoch

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        self.lr.append((trainer.global_step, trainer.optimizers[0].param_groups[0]["lr"]))


def _leg1_checkpoint(data, *, schedule, save_epoch, max_epochs, ckpt_dir: Path,
                     optimizer="AdamW", monitor=None) -> Path:
    """Run a fresh fit over the FULL `max_epochs` with a real `ModelCheckpoint`
    saving every epoch (``epoch={N}.ckpt``), and return ``epoch={save_epoch}.ckpt``
    — the checkpoint written at the END of `save_epoch` (Lightning's own epoch
    encoding, so resume starts at ``save_epoch + 1``).

    Leg-1 MUST use the same full `max_epochs` as the uninterrupted / resumed runs:
    the schedule validates its epoch budget and allocates each stage's OneCycle
    ``total_steps`` from `max_epochs`, so a truncated leg-1 would over-allocate
    epochs (ConfigError) and mis-size the per-stage step envelopes. It is
    deterministic (shuffle=False, dropout-free), so epochs 0..save_epoch reproduce
    the uninterrupted run's first epochs.
    """
    seed_everything(SEED, workers=True)
    model = build_model(data, schedule=schedule, optimizer=optimizer)
    mc = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="{epoch}", save_top_k=-1,
        every_n_epochs=1, monitor=monitor,
    )
    callbacks: list = [mc]
    if schedule is not None:
        callbacks.insert(0, TrainingScheduleCallback())
    make_trainer(
        max_epochs=max_epochs, callbacks=callbacks, enable_checkpointing=True
    ).fit(model, build_dm(data))
    ckpt = ckpt_dir / f"epoch={save_epoch}.ckpt"
    assert ckpt.exists(), f"leg-1 did not write {ckpt.name} (have: {sorted(p.name for p in ckpt_dir.glob('*.ckpt'))})"
    return ckpt


# --- snapshot + comparison helpers -------------------------------------------


def _clone_sd(sd) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


def _snapshot(model: SaltModule, tracer: _Tracer) -> dict:
    return {
        "state_dict": _clone_sd(model.state_dict()),
        "opt_state": copy.deepcopy(model.trainer.optimizers[0].state_dict()),
        "lr": list(tracer.lr),
    }


def _assert_state_dicts_equal(a: dict, b: dict, ctx: str) -> None:
    assert a.keys() == b.keys(), f"{ctx}: state_dict keys differ"
    for key in a:
        assert torch.equal(a[key], b[key]), f"{ctx}: tensor {key} differs"


def _collect_tensors(obj, prefix: str = "") -> dict:
    """Recursively gather every tensor in an optimizer state_dict, keyed by path.

    Works for both AdamW (``{"state": {...}, "param_groups": [...]}``) and the
    nested `HybridMuonAdamW` state_dict (``{"wrapper", "muon", "adamw", ...}``);
    non-tensor leaves (hyperparams, param-index lists, name lists) are skipped.
    """
    out: dict[str, torch.Tensor] = {}
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, Mapping):
        for key, val in obj.items():
            out.update(_collect_tensors(val, f"{prefix}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            out.update(_collect_tensors(val, f"{prefix}.{i}"))
    return out


def _assert_moments_equal(a: dict, b: dict, ctx: str) -> None:
    ta, tb = _collect_tensors(a), _collect_tensors(b)
    assert ta.keys() == tb.keys(), f"{ctx}: optimizer moment keys differ"
    assert ta, f"{ctx}: no optimizer moment tensors (empty state)"
    for key in ta:
        assert torch.equal(ta[key], tb[key]), f"{ctx}: optimizer moment {key} differs"


def _assert_lr_tail_matches(full: list[tuple[int, float]], tail: list[tuple[int, float]],
                            ctx: str) -> None:
    fmap = dict(full)
    assert tail, f"{ctx}: resumed LR trace is empty"
    for gstep, lr in tail:
        assert gstep in fmap, f"{ctx}: resumed step {gstep} absent from uninterrupted trace"
        assert fmap[gstep] == lr, (
            f"{ctx}: LR mismatch at global step {gstep}: resumed {lr} vs uninterrupted {fmap[gstep]}"
        )


# --- run harness -------------------------------------------------------------


def _run_uninterrupted(data, *, schedule, max_epochs, optimizer="AdamW") -> dict:
    seed_everything(SEED, workers=True)
    model = build_model(data, schedule=schedule, optimizer=optimizer)
    tracer = _Tracer()
    callbacks: list = [tracer]
    if schedule is not None:
        callbacks.insert(0, _RebuildCounter())  # schedule cb first, tracer sees rebuilt refs
    make_trainer(max_epochs=max_epochs, callbacks=callbacks).fit(model, build_dm(data))
    return _snapshot(model, tracer)


def _run_resume(data, *, schedule, max_epochs, save_epoch, optimizer="AdamW",
                tmp_path: Path) -> tuple[dict, _Tracer, _RebuildCounter | None, Path]:
    # leg 1: reproduce epochs 0..save_epoch, checkpoint the boundary (ModelCheckpoint).
    ckpt = _leg1_checkpoint(
        data, schedule=schedule, save_epoch=save_epoch, max_epochs=max_epochs,
        optimizer=optimizer, ckpt_dir=tmp_path / "ckpts",
    )
    # leg 2: fresh module + trainer, resume from the checkpoint.
    seed_everything(SEED, workers=True)
    m2 = build_model(data, schedule=schedule, optimizer=optimizer)
    tracer = _Tracer()
    counter = _RebuildCounter() if schedule is not None else None
    cbs2: list = [tracer]
    if counter is not None:
        cbs2.insert(0, counter)
    make_trainer(max_epochs=max_epochs, callbacks=cbs2).fit(m2, build_dm(data), ckpt_path=str(ckpt))
    return _snapshot(m2, tracer), tracer, counter, ckpt


# --- probe: Lightning restore order ------------------------------------------


class _ProbeModule(SaltModule):
    """Records the order of setup / on_load_checkpoint / configure_optimizers."""

    EVENTS: list[tuple] = []

    def setup(self, stage: str) -> None:
        _ProbeModule.EVENTS.append(("setup", stage))
        super().setup(stage)

    def on_load_checkpoint(self, checkpoint) -> None:
        _ProbeModule.EVENTS.append((
            "on_load_checkpoint",
            checkpoint.get("epoch"),
            getattr(self._trainer, "current_epoch", None),
            checkpoint.get(CKPT_KEY, {}).get("schedule"),
        ))
        super().on_load_checkpoint(checkpoint)

    def configure_optimizers(self):
        _ProbeModule.EVENTS.append(("configure_optimizers", self._current_stage_index))
        return super().configure_optimizers()


class TestRestoreOrderProbe:
    def test_restore_order_is_setup_then_load_then_configure(self, data, tmp_path):
        # leg 1: full [2,2] schedule over 4 epochs, ModelCheckpoint saves every
        # epoch. We resume from epoch=1.ckpt (end of epoch 1, still stage 0; the
        # boundary is at epoch 2).
        seed_everything(SEED, workers=True)
        m1 = _ProbeModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=WARMUP_FULL)
        ckpt_dir = tmp_path / "probe"
        mc = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="{epoch}", save_top_k=-1, every_n_epochs=1
        )
        make_trainer(
            max_epochs=4, callbacks=[TrainingScheduleCallback(), mc], enable_checkpointing=True
        ).fit(m1, build_dm(data))
        ckpt = ckpt_dir / "epoch=1.ckpt"
        assert ckpt.exists()

        _ProbeModule.EVENTS = []
        seed_everything(SEED, workers=True)
        m2 = _ProbeModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=WARMUP_FULL)
        make_trainer(max_epochs=4, callbacks=[TrainingScheduleCallback()]).fit(
            m2, build_dm(data), ckpt_path=str(ckpt)
        )

        names = [e[0] for e in _ProbeModule.EVENTS]
        print("RESTORE-ORDER PROBE events:", _ProbeModule.EVENTS)  # noqa: T201
        assert "setup" in names and "on_load_checkpoint" in names
        assert "configure_optimizers" in names
        # the ordering these tests depend on: stage is restored (on_load) BEFORE the
        # optimizer is (re)built (configure_optimizers), and both AFTER setup.
        assert names.index("setup") < names.index("on_load_checkpoint")
        assert names.index("on_load_checkpoint") < names.index("configure_optimizers")
        # the saved schedule payload rode along in the checkpoint under CKPT_KEY.
        load_events = [e for e in _ProbeModule.EVENTS if e[0] == "on_load_checkpoint"]
        assert load_events[0][3] is not None, "no schedule payload in the checkpoint"
        assert load_events[0][3]["stage_index"] == 0  # saved end of epoch 1 = still stage 0


# --- resume inside stage 0 from a real val-loss ModelCheckpoint ----------


class TestResumeWithinStage0:
    def test_val_loss_checkpoint_resume_within_stage0_equals_uninterrupted(self, data, tmp_path):
        max_epochs = 4  # [2,2] schedule, boundary at epoch 2
        ref = _run_uninterrupted(data, schedule=WARMUP_FULL, max_epochs=max_epochs)

        # leg 1: a REAL val-loss-monitored ModelCheckpoint over the full run,
        # proving Lightning's default val-loss checkpoint path is not excluded.
        # We resume from epoch=0.ckpt (end of epoch 0, inside stage 0).
        ckpt = _leg1_checkpoint(
            data, schedule=WARMUP_FULL, save_epoch=0, max_epochs=max_epochs,
            ckpt_dir=tmp_path / "ckpts", monitor="val/loss",
        )

        # leg 2: resume to the full run.
        seed_everything(SEED, workers=True)
        m2 = build_model(data, schedule=WARMUP_FULL)
        tracer = _Tracer()
        counter = _RebuildCounter()
        make_trainer(max_epochs=max_epochs, callbacks=[counter, tracer]).fit(
            m2, build_dm(data), ckpt_path=str(ckpt)
        )
        got = _snapshot(m2, tracer)

        assert tracer.first_epoch == 1  # resumed at epoch 1 (inside stage 0)
        assert counter.rebuild_epochs == [2]  # single boundary rebuild at epoch 2
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "stage-0 resume state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "stage-0 resume moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "stage-0 resume LR trace")


# --- resume exactly at a stage boundary ---------------------------------


class TestResumeAtBoundary:
    def _run(self, data, tmp_path, optimizer):
        max_epochs = 4  # [2,2], boundary at epoch 2; save at end of epoch 1 (stage-0 last)
        ref = _run_uninterrupted(
            data, schedule=WARMUP_FULL, max_epochs=max_epochs, optimizer=optimizer
        )
        got, tracer, counter, _ = _run_resume(
            data, schedule=WARMUP_FULL, max_epochs=max_epochs, save_epoch=1,
            optimizer=optimizer, tmp_path=tmp_path,
        )
        return ref, got, tracer, counter

    def test_boundary_resume_one_rebuild_and_equivalence_adamw(self, data, tmp_path):
        ref, got, tracer, counter = self._run(data, tmp_path, "AdamW")
        assert tracer.first_epoch == 2  # resumed exactly at the boundary epoch
        assert counter.rebuild_epochs == [2]  # exactly ONE rebuild, post-restore, at the boundary
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "boundary state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "boundary moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "boundary LR trace")

    def test_boundary_resume_stage1_optimizer_owns_stage1_trainable_set(self, data, tmp_path):
        # after the boundary the encoder is unfrozen, so the rebuilt stage-1
        # optimizer must own exactly the current trainable set (encoder included).
        ckpt = _leg1_checkpoint(
            data, schedule=WARMUP_FULL, save_epoch=1, max_epochs=4, optimizer="AdamW",
            ckpt_dir=tmp_path / "own",
        )
        seed_everything(SEED, workers=True)
        m2 = build_model(data, schedule=WARMUP_FULL)
        make_trainer(max_epochs=4, callbacks=[TrainingScheduleCallback()]).fit(
            m2, build_dm(data), ckpt_path=str(ckpt)
        )
        opt_ids = {id(p) for group in m2.trainer.optimizers[0].param_groups for p in group["params"]}
        trainable = {id(p) for p in m2.parameters() if p.requires_grad}
        assert m2._current_stage_index == 1  # noqa: SLF001 - reached stage 1
        assert opt_ids == trainable  # optimizer owns exactly the trainable set
        assert _module_param_ids(m2, FROZEN) <= opt_ids  # incl. the unfrozen encoder

    def test_boundary_resume_hybrid_muon_adamw(self, data, tmp_path):
        ref, got, tracer, counter = self._run(data, tmp_path, "HybridMuonAdamW")
        assert tracer.first_epoch == 2
        assert counter.rebuild_epochs == [2]
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "hybrid boundary state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "hybrid boundary moments")


# --- resume mid-stage-1 (after the boundary) ----------------------------


class TestResumeMidStage1:
    def test_mid_stage1_resume_keeps_moments_no_rebuild_and_equivalence(self, data, tmp_path):
        # [2,3] schedule over 5 epochs: stage 1 = epochs 2,3,4. Stage 1 FREEZES the
        # encoder (a freeze-backbone / head-finetune stage) so we can check the
        # freeze persists across the resume. Save at end of epoch 3 (INSIDE
        # stage 1); resume at epoch 4 (still stage 1, encoder frozen).
        schedule = {
            "stages": {
                "joint": {"epochs": 2, "lrs": {"max": 5e-3}},  # train everything
                "head_finetune": {"frozen": [FROZEN], "lrs": {"max": 1e-3}},  # freeze encoder
            }
        }
        max_epochs = 5
        ref = _run_uninterrupted(data, schedule=schedule, max_epochs=max_epochs)
        got, tracer, counter, ckpt = _run_resume(
            data, schedule=schedule, max_epochs=max_epochs, save_epoch=3, tmp_path=tmp_path,
        )

        assert tracer.first_epoch == 4  # resumed inside stage 1
        assert counter.rebuild_epochs == []  # NO spurious rebuild — stage unchanged

        # frozen module frozen + eval from the first restored step.
        assert tracer.frozen_requires_grad_at_start is False
        assert tracer.frozen_training_at_start is False

        # restored optimizer moments are KEPT (not reset): the optimizer state
        # entering the resumed run equals the checkpoint's saved optimizer state
        # (a fresh optimizer would have an empty `state`).
        saved = torch.load(str(ckpt), weights_only=False)["optimizer_states"][0]
        _assert_moments_equal(saved, tracer.opt_state_at_start, "restored-at-start moments")

        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "mid-stage-1 state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "mid-stage-1 moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "mid-stage-1 LR trace")


# --- parity guard — no-schedule (legacy) resume unchanged ---------------


class TestLegacyParityGuard:
    def test_no_schedule_resume_is_bitwise_equal_to_uninterrupted(self, data, tmp_path):
        max_epochs = 4
        ref = _run_uninterrupted(data, schedule=None, max_epochs=max_epochs)
        got, tracer, counter, _ = _run_resume(
            data, schedule=None, max_epochs=max_epochs, save_epoch=1, tmp_path=tmp_path,
        )
        assert counter is None  # no schedule callback on the legacy path
        assert tracer.first_epoch == 2
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "legacy resume state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "legacy resume optimizer moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "legacy resume LR trace")


# =====================================================================
# per-stage early stopping
# =====================================================================


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


# --- legacy parity — checkpoint payload unchanged -----------------------


class TestLegacyPayloadParity:
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
        # early-stop checkpoint keys (the has_early_stop master switch is off). epoch 0 is in
        # stage a (owns [0,1)); saved at end of epoch 0 → stage_index 0.
        schedule = {"stages": {"a": {"epochs": 1, "frozen": [FROZEN]}, "b": {"frozen": []}}}
        payload = self._saved_schedule_payload(data, tmp_path, schedule, max_epochs=2)
        assert set(payload) == {"stage_index", "stage_name"}

    def test_early_stop_payload_carries_state_keys(self, data, tmp_path):
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


# --- two-stage, stage-0 early-stops before its epoch cap -----------------


class TestStageZeroEarlyStop:
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


# --- final-stage early-stop ends the fit --------------------------------


class TestFinalStageEarlyStop:
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


# --- early-stop resume across data-dependent boundaries ----------------


class TestEarlyStopResume:
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
