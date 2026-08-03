"""Gates: checkpoint resume across `training_schedule` stages.

`salt2 fit --ckpt_path <ckpt>` must resume a multi-stage schedule from any
epoch — inside stage 0, exactly at a stage boundary, or inside a later stage —
producing state (weights, optimizer moments, per-step LR trace) identical to the
uninterrupted run.

Gates:

- **probe** — pins the Lightning 2.6.5 restore order these gates rely on:
  ``setup("fit")`` -> ``on_load_checkpoint`` -> ``configure_optimizers`` ->
  optimizer-state restore. The stage restore in `SaltModule.on_load_checkpoint`
  is only correct if it runs *before* the optimizer is rebuilt.
- **G4a** resume inside stage 0 (from a real val-loss `ModelCheckpoint`) ≡
  uninterrupted: final `state_dict` bitwise-equal, optimizer moments equal.
- **G4b** resume exactly at a stage boundary: exactly one rebuild fires
  post-restore; the stage-1 optimizer owns exactly the stage-1 trainable set;
  final state bitwise-equal. Covered for AdamW and HybridMuonAdamW.
- **G4c** resume mid-stage-1: NO rebuild at the resume epoch, the restored
  optimizer moments are kept (not reset), frozen modules are frozen + eval from
  the first restored step; final state bitwise-equal.
- **G4d** per-step LR trace across the resume point identical to the
  uninterrupted run (asserted inside G4a/G4b/G4c).
- **G4e** parity guard — a no-schedule (legacy) config resumes bitwise-identically
  to its uninterrupted run (the checkpoint save/restore additions do not perturb
  the desugared single-stage path).

Determinism contract (what the bitwise equalities rely on / do NOT rely on):
the datamodule uses a **shuffle=False** train loader (`_DeterministicDataModule`)
and the fixture model has **no dropout** (default), so epoch e produces identical
batches and identical forward passes in the uninterrupted run and the resumed
run regardless of global-RNG state. Equality therefore does NOT rely on Lightning
restoring the global torch/numpy RNG across a fit resume (it does not by default)
— it relies only on the deterministic loader + dropout-free model + a fixed seed
for identical initial weights. All DataLoaders use ``num_workers=0`` (agent
memcg gotcha).
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
from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
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


class _DeterministicDataModule(GraphDataModule):
    """A GraphDataModule whose train loader does NOT shuffle — so epoch e yields
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


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


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
        # the ordering these gates depend on: stage is restored (on_load) BEFORE the
        # optimizer is (re)built (configure_optimizers), and both AFTER setup.
        assert names.index("setup") < names.index("on_load_checkpoint")
        assert names.index("on_load_checkpoint") < names.index("configure_optimizers")
        # the saved schedule payload rode along in the checkpoint under CKPT_KEY.
        load_events = [e for e in _ProbeModule.EVENTS if e[0] == "on_load_checkpoint"]
        assert load_events[0][3] is not None, "no schedule payload in the checkpoint"
        assert load_events[0][3]["stage_index"] == 0  # saved end of epoch 1 = still stage 0


# --- G4a: resume inside stage 0 from a real val-loss ModelCheckpoint ----------


class TestG4aResumeWithinStage0:
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
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "G4a state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "G4a optimizer moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "G4a LR trace")  # G4d


# --- G4b: resume exactly at a stage boundary ---------------------------------


class TestG4bResumeAtBoundary:
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
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "G4b state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "G4b optimizer moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "G4b LR trace")  # G4d

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
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "G4b-hybrid state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "G4b-hybrid optimizer moments")


# --- G4c: resume mid-stage-1 (after the boundary) ----------------------------


class TestG4cResumeMidStage1:
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
        _assert_moments_equal(saved, tracer.opt_state_at_start, "G4c restored-at-start moments")

        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "G4c state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "G4c optimizer moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "G4c LR trace")  # G4d


# --- G4e: parity guard — no-schedule (legacy) resume unchanged ---------------


class TestG4eLegacyParityGuard:
    def test_no_schedule_resume_is_bitwise_equal_to_uninterrupted(self, data, tmp_path):
        max_epochs = 4
        ref = _run_uninterrupted(data, schedule=None, max_epochs=max_epochs)
        got, tracer, counter, _ = _run_resume(
            data, schedule=None, max_epochs=max_epochs, save_epoch=1, tmp_path=tmp_path,
        )
        assert counter is None  # no schedule callback on the legacy path
        assert tracer.first_epoch == 2
        _assert_state_dicts_equal(ref["state_dict"], got["state_dict"], "G4e legacy state_dict")
        _assert_moments_equal(ref["opt_state"], got["opt_state"], "G4e legacy optimizer moments")
        _assert_lr_tail_matches(ref["lr"], got["lr"], "G4e legacy LR trace")
