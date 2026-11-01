"""Tests for `SaltModule`."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from lightning import Callback, Trainer
from torch import nn

from salt.data import Features, SaltDataModule, H5StructuredReader, Labels
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError, NonFiniteError
from salt.graph.spec import IO, Mode
from salt.model.base import ModuleGroup, SaltModelModule
from salt.model.modules import GlobalAttentionPooling, KinematicWeight
from salt.model.modules.losses import LossGLS, LossSum
from salt.model.modules.tasks import ClassificationTaskModule
from salt.outputs import RunTaskOutput
from salt.model.saltmodule import CKPT_KEY, SaltModule, _expand_module_groups, _upstream
from salt.model.sink_prep import boundary_demand, callback_demand, select_fitval_callbacks
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests._fixtures.dummy_h5 import dummy_h5
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules

LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}
TASKS = ["jets_classification", "track_origin", "track_vertexing"]
FIT_DEMAND = {
    "inputs.jets",
    "inputs.tracks",
    "masks.tracks",
    "labels.jets.flavour_label",
    "labels.tracks.ftagTruthOriginLabel",
    "labels.tracks.ftagTruthVertexIndex",
}
TEST_DEMAND = {"inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    # dummy H5 + parity norm dict (distinct per-variable constants) + schema artifact
    base = tmp_path_factory.mktemp("saltmodule")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path, _, _ = dummy_h5(base)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_datamodule(data, **kwargs) -> SaltDataModule:
    # fresh data modules per call — the dm clones/copies them per stage
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    kwargs.setdefault("train_file", data["h5"])
    kwargs.setdefault("val_file", data["h5"])
    kwargs.setdefault("test_file", data["h5"])
    return SaltDataModule(modules, batch_size=100, num_workers=0, **kwargs)


def build_model(data, norm_dict=None, **kwargs) -> SaltModule:
    # GN2v2 module dict with an UN-narrowed LossSum (SaltModule must narrow it)
    modules = build_gn2v2_modules(norm_dict or data["nd"])
    modules["loss"] = LossSum()
    kwargs.setdefault("lrs", LRS)
    return SaltModule(modules, **kwargs)


def make_trainer(**kwargs) -> Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("limit_train_batches", 2)
    kwargs.setdefault("limit_val_batches", 2)
    kwargs.setdefault("num_sanity_val_steps", 1)
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        **kwargs,
    )


class StepRecorder(Callback):
    """Record per-step training losses and step-output shapes."""

    def __init__(self) -> None:
        self.losses: list[torch.Tensor] = []
        self.bundles_ok: list[bool] = []

    def on_train_batch_end(self, _trainer, _module, outputs, _batch, _batch_idx) -> None:
        self.losses.append(outputs["loss"].detach())
        self.bundles_ok.append(isinstance(outputs.get("bundle"), Bundle))


class StopAfterFirstEpoch(Callback):
    """Stop a (nominally 2-epoch) fit after epoch 0, leaving room to resume."""

    def on_train_epoch_end(self, trainer, _module) -> None:
        trainer.should_stop = True


@pytest.fixture(scope="module")
def fitted(data) -> dict:
    # one short CPU fit (epoch 0 of 2: 2 train steps + sanity/val) + saved checkpoint
    model = build_model(data)
    dm = build_datamodule(data)
    recorder = StepRecorder()
    trainer = make_trainer(max_epochs=2, callbacks=[recorder, StopAfterFirstEpoch()])
    trainer.fit(model, dm)
    ckpt_path = data["dir"] / "after_fit.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return {
        "model": model,
        "dm": dm,
        "trainer": trainer,
        "recorder": recorder,
        "ckpt": ckpt_path,
    }


class TestConstruction:
    def test_names_assigned_and_losssum_narrowed(self, data):
        model = build_model(data)
        assert model.net["track_embed"].name == "track_embed"
        loss = model.net["loss"]
        assert isinstance(loss, LossSum)
        assert loss.narrowed
        keys = set(loss.declare_io(Mode.FIT).requires["losses"])
        assert keys == set(TASKS)

    def test_sink_demand(self, data):
        demand = build_model(data).sink_demand()
        assert set(demand[Mode.FIT]) == FIT_DEMAND
        assert set(demand[Mode.VAL]) == FIT_DEMAND
        assert set(demand[Mode.TEST]) == TEST_DEMAND

    def test_missing_lrs_keys_rejected(self, data):
        with pytest.raises(ConfigError, match="lrs is missing"):
            build_model(data, lrs={"initial": 1e-3})

    def test_bad_optimizer_rejected(self, data):
        with pytest.raises(ConfigError, match="optimizer"):
            build_model(data, optimizer="SGD")

    def test_setup_without_datamodule_rejected(self, data):
        with pytest.raises(ConfigError, match="SaltDataModule"):
            build_model(data).setup("fit")

    def test_validate_stage_accepted_predict_rejected(self, data):
        """``setup("predict")`` raises; ``setup("validate")`` compiles VAL only."""
        with pytest.raises(ConfigError, match="predict"):
            build_model(data).setup("predict")
        model, dm = build_model(data), build_datamodule(data)
        dm.set_sinks(model.sink_demand())
        dm.setup("validate")
        assert dm.val_dset is not None
        assert getattr(dm, "train_dset", None) is None
        model._trainer = SimpleNamespace(callbacks=[], datamodule=dm)  # noqa: SLF001
        model.setup("validate")
        assert set(model.plans) == {Mode.VAL}
        assert model.bound

    def test_forward_without_plan_rejected(self, data):
        with pytest.raises(ConfigError, match="no compiled plan"):
            build_model(data)({}, Mode.TEST)

    def test_lossgls_narrowed_like_losssum(self, data):
        # a LossGLS is a LossSum subclass, so the SaltModule narrow loop fixes
        # its loss keys for free (no explicit losses: needed). The gn2v2 fixture
        # carries weighted tasks, so set them to 1.0 first (the GLS domain).
        modules = build_gn2v2_modules(data["nd"])
        for task in ("track_origin", "track_vertexing"):
            modules[task].weight = 1.0
        modules["loss"] = LossGLS()
        model = SaltModule(modules, lrs=LRS)
        loss = model.net["loss"]
        assert isinstance(loss, LossGLS)
        assert loss.narrowed
        assert set(loss.declare_io(Mode.FIT).requires["losses"]) == set(TASKS)

    def test_lossgls_rejects_weighted_task_at_construction(self, data):
        # the all-weights==1.0 guard fires at assembly (the v2 home of v1's ctor
        # assert) — the gn2v2 fixture's track_origin weight is 0.5
        modules = build_gn2v2_modules(data["nd"])
        modules["loss"] = LossGLS()
        with pytest.raises(ConfigError, match="does not utilise task weights"):
            SaltModule(modules, lrs=LRS)

    def test_raw_nn_module_in_modules_rejected(self, data):
        # a raw nn.Module (not a SaltModelModule) is rejected at
        # construction, naming the offending config key.
        modules = build_gn2v2_modules(data["nd"])
        modules["loss"] = LossSum()
        modules["track_embed"] = nn.Linear(4, 4)
        with pytest.raises(ConfigError, match=r"track_embed.*not a SaltModelModule"):
            SaltModule(modules, lrs=LRS)

    def test_valid_output_section_writer_accepted(self, data):
        # RunTaskOutput (an OutputSectionWriter, hence a SaltModelModule) is a
        # legitimate outputs: entry — composes without error.
        model = build_model(data)
        model.compose_output_section({"jets_out": RunTaskOutput(tasks=["jets_classification"])})
        assert "jets_out" in model._output_section  # noqa: SLF001 - asserting composition landed

    def test_raw_nn_module_in_outputs_rejected(self, data):
        # a raw nn.Module in outputs: is neither a SaltModelModule nor a
        # SinkModule — rejected at composition, naming the offending key.
        model = build_model(data)
        with pytest.raises(ConfigError, match=r"bogus_out.*neither a SaltModelModule"):
            model.compose_output_section({"bogus_out": nn.Linear(4, 4)})

    def test_sink_module_protocol_accepted_in_outputs(self, data):
        # audited defensive branch: a terminal callback-style sink
        # (SinkModule-protocol, not nn.Module) is ALSO accepted here — real
        # shipped configs never exercise this (terminal sinks are wired via
        # trainer.callbacks:, see salt.model.base.SaltModelModule docstring),
        # but the validation must not reject the protocol shape outright.
        # By design, `compose_output_section` partitions any `is_sink()`
        # entry OUT of `_output_section` and into `_section_sinks` — the
        # protocol shape is accepted (no rejection), not ridden on
        # `_output_section` (pipeline #15650554 diagnosis item 6).
        class _FakeManifestOnlySink:
            name = "fake_sink"

            def is_sink(self) -> bool:
                return True

            def declare_io(self, mode):  # the SinkModule Protocol also carries this
                del mode
                return IO(requires={}, produces={})

            def is_manifest_only(self) -> bool:
                # manifest-only so compose_output_section never tries to ride
                # it on the nn.ModuleDict (it isn't an nn.Module)
                return True

        model = build_model(data)
        model.compose_output_section({"fake_sink": _FakeManifestOnlySink()})
        assert model._section_sinks[0].name == "fake_sink"  # noqa: SLF001


class TestDataModuleConstruction:
    """SaltDataModule validates its module dict against SaltDatasetModule."""

    def test_raw_object_in_data_modules_rejected(self, data):
        class _NotADatasetModule:
            pass

        modules = {
            "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
            "features": Features(
                variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
            ),
            "labels": Labels(),
            "bogus": _NotADatasetModule(),
        }
        with pytest.raises(ConfigError, match=r"bogus.*not a SaltDatasetModule"):
            SaltDataModule(
                modules,
                train_file=data["h5"],
                val_file=data["h5"],
                test_file=data["h5"],
                batch_size=100,
                num_workers=0,
            )


class TestFit:
    """A short `trainer.fit` on the dummy file, CPU."""

    def test_two_steps_finite_loss(self, fitted):
        losses = fitted["recorder"].losses
        assert len(losses) == 2
        assert all(torch.isfinite(loss) for loss in losses)
        assert all(fitted["recorder"].bundles_ok)

    def test_loss_logging_convention(self, fitted):
        """v1 monitor keys: {stage}/loss + {stage}/{task}_loss (modelwrapper.py:256-270)."""
        metrics = fitted["trainer"].callback_metrics
        for stage in ("train", "val"):
            assert f"{stage}/loss" in metrics
            for task in TASKS:
                assert f"{stage}/{task}_loss" in metrics
        per_task = sum(metrics[f"train/{task}_loss"] for task in TASKS)
        assert torch.isclose(metrics["train/loss"], per_task, atol=1e-6)

    def test_plans_compiled_and_bound(self, fitted):
        model = fitted["model"]
        assert {Mode.FIT, Mode.VAL} <= set(model.plans)
        assert model.schema is not None
        assert model.schema.width("inputs.jets") == len(JET_VARIABLES)
        assert model.schema.width("inputs.tracks") == len(TRACK_VARIABLES)
        assert model.schema.fields_of("inputs.tracks") == tuple(TRACK_VARIABLES)

    def test_fresh_fit_materialised_norm_from_file(self, fitted):
        """materialise() ran before the first step and read the parity constants."""
        norm = fitted["model"].net["norm"]
        assert bool(norm.materialised)
        expect_means = torch.tensor([0.1 * (i + 1) for i in range(len(JET_VARIABLES))])
        expect_stds = torch.tensor([1.0 + 0.05 * (i + 1) for i in range(len(JET_VARIABLES))])
        assert torch.allclose(norm.means_jets, expect_means)
        assert torch.allclose(norm.stds_jets, expect_stds)

    def test_sinks_auto_adopted_by_datamodule(self, fitted):
        """SaltDataModule.setup adopted the model's sink_demand (no set_sinks call)."""
        dm = fitted["dm"]
        assert dm._sinks is not None  # noqa: SLF001 - asserting the auto-wiring itself
        assert set(dm._sinks[Mode.FIT]) == FIT_DEMAND  # noqa: SLF001


class TestTestLoop:
    def test_test_step_returns_bundle(self, fitted):
        bundles: list[Bundle] = []

        class TestRecorder(Callback):
            def on_test_batch_end(self, _trainer, _module, outputs, _batch, _batch_idx) -> None:
                bundles.append(outputs)

        trainer = make_trainer(limit_test_batches=2, callbacks=[TestRecorder()])
        trainer.test(fitted["model"], fitted["dm"])
        assert len(bundles) == 2
        bundle = bundles[0]
        assert isinstance(bundle, Bundle)
        for stream, task in zip(["jets", "tracks", "tracks"], TASKS, strict=True):
            assert f"preds.{stream}.{task}" in bundle
        assert "meta.rows" in bundle
        assert Mode.TEST in fitted["model"].plans


class TestCheckpoint:
    """Schema + plan hashes round-trip; resume skips materialise."""

    def test_payload_written(self, fitted):
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        payload = ckpt[CKPT_KEY]
        model = fitted["model"]
        assert payload["schema"]["widths"] == dict(model.schema.widths)
        stored_fields = {k: tuple(v) for k, v in payload["schema"]["fields"].items()}
        assert stored_fields == dict(model.schema.fields)
        hashes = payload["plan_hashes"]
        for mode in (Mode.FIT, Mode.VAL):
            assert hashes[mode.name] == model.plans[mode].plan_hash
        # the modules dict is NOT pickled into hparams
        assert "modules" not in ckpt.get("hyper_parameters", {})

    def test_load_from_checkpoint_roundtrip_bit_identical(self, data, fitted):
        """Data-less load: fresh unbound modules bind from the STORED schema."""
        fresh = build_gn2v2_modules(data["nd"])
        fresh["loss"] = LossSum()
        loaded = SaltModule.load_from_checkpoint(fitted["ckpt"], modules=fresh, map_location="cpu")
        assert loaded.bound
        assert loaded.loaded_from_checkpoint
        assert dict(loaded.schema.widths) == dict(fitted["model"].schema.widths)
        saved_sd = fitted["model"].state_dict()
        loaded_sd = loaded.state_dict()
        assert set(saved_sd) == set(loaded_sd)
        for key, value in saved_sd.items():
            assert torch.equal(value, loaded_sd[key]), key
        assert bool(loaded.net["norm"].materialised)

    def test_fit_hash_mismatch_is_fatal(self, fitted):
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        tampered = copy.deepcopy(ckpt)
        tampered[CKPT_KEY]["plan_hashes"]["FIT"] = "0" * 64
        model = fitted["model"]
        before = model.loaded_from_checkpoint
        try:
            with pytest.raises(ConfigError, match="plan-hash mismatch for mode FIT"):
                model.on_load_checkpoint(tampered)
        finally:
            model.loaded_from_checkpoint = before  # restore fixture state
            model._ckpt_plan_hashes = {}  # noqa: SLF001

    def test_val_hash_mismatch_warns(self, fitted):
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        tampered = copy.deepcopy(ckpt)
        tampered[CKPT_KEY]["plan_hashes"]["VAL"] = "0" * 64
        model = fitted["model"]
        before = model.loaded_from_checkpoint
        try:
            with pytest.warns(UserWarning, match="plan-hash mismatch for mode VAL"):
                model.on_load_checkpoint(tampered)
        finally:
            model.loaded_from_checkpoint = before  # restore fixture state
            model._ckpt_plan_hashes = {}  # noqa: SLF001

    def test_resume_skips_materialise(self, data, fitted):
        """Resume continues training WITHOUT materialise overwriting loaded buffers."""
        model = build_model(data, norm_dict=data["dir"] / "does_not_exist.yaml")
        dm = build_datamodule(data)
        trainer = make_trainer(max_epochs=2)
        trainer.fit(model, dm, ckpt_path=fitted["ckpt"])
        assert trainer.global_step == 4  # 2 saved + 2 resumed steps
        assert model.loaded_from_checkpoint
        assert not model.materialised
        # buffers came from the checkpoint, not from any file
        saved_norm = fitted["model"].net["norm"]
        assert torch.equal(model.net["norm"].means_tracks, saved_norm.means_tracks)
        assert torch.equal(model.net["norm"].stds_tracks, saved_norm.stds_tracks)
        assert bool(model.net["norm"].materialised)

    def test_resume_with_changed_graph_fails_fast(self, data, fitted):
        """A config change between save and resume trips the FIT hash check."""
        modules = build_gn2v2_modules(data["nd"], embed_dim=24)
        modules["loss"] = LossSum()
        model = SaltModule(modules, lrs=LRS)
        dm = build_datamodule(data)
        trainer = make_trainer(max_epochs=2)
        with pytest.raises(ConfigError, match="plan-hash mismatch for mode FIT"):
            trainer.fit(model, dm, ckpt_path=fitted["ckpt"])

    def test_checkpoint_without_payload_warns(self, fitted):
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        stripped = copy.deepcopy(ckpt)
        del stripped[CKPT_KEY]
        model = fitted["model"]
        before = model.loaded_from_checkpoint
        try:
            with pytest.warns(UserWarning, match="no 'salt_core' payload"):
                model.on_load_checkpoint(stripped)
        finally:
            model.loaded_from_checkpoint = before  # restore fixture state

    def test_compiled_checkpoint_orig_mod_prefix_stripped(self, fitted):
        """A ``--compile``-trained checkpoint carries torch.compile's"""
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        prefixed = copy.deepcopy(ckpt)
        clean_keys = set(prefixed["state_dict"])
        # emulate torch._dynamo.OptimizedModule prefixing every key
        prefixed["state_dict"] = {
            f"_orig_mod.{k}": v for k, v in prefixed["state_dict"].items()
        }
        model = fitted["model"]
        before = model.loaded_from_checkpoint
        try:
            model.on_load_checkpoint(prefixed)
            assert set(prefixed["state_dict"]) == clean_keys
            assert not any("_orig_mod." in k for k in prefixed["state_dict"])
        finally:
            model.loaded_from_checkpoint = before  # restore fixture state
            model._ckpt_plan_hashes = {}  # noqa: SLF001

    def test_noncompiled_checkpoint_state_dict_untouched(self, fitted):
        """No-op: a normally-trained checkpoint has no ``_orig_mod.`` keys,"""
        ckpt = torch.load(fitted["ckpt"], weights_only=False)
        clean = copy.deepcopy(ckpt)
        keys_before = set(clean["state_dict"])
        model = fitted["model"]
        before = model.loaded_from_checkpoint
        try:
            model.on_load_checkpoint(clean)
            assert set(clean["state_dict"]) == keys_before
        finally:
            model.loaded_from_checkpoint = before  # restore fixture state
            model._ckpt_plan_hashes = {}  # noqa: SLF001


class _EpochStepRecorder(Callback):
    """Record ``(current_epoch, global_step)`` at validation start."""

    def __init__(self) -> None:
        self.seen: tuple[int, int] | None = None

    def on_validation_start(self, trainer, _module) -> None:
        self.seen = (trainer.current_epoch, trainer.global_step)


class TestValidateFromCheckpoint:
    """``trainer.validate(ckpt_path=...)`` restores the checkpoint's fit progress."""

    def test_validate_restores_checkpoint_epoch_and_global_step(self, data, fitted, tmp_path):
        """Validation sees the checkpoint's epoch/global_step; no checkpoint is written."""
        rec = _EpochStepRecorder()
        meta = torch.load(str(fitted["ckpt"]), weights_only=False)
        model = build_model(data)
        trainer = make_trainer(callbacks=[rec], default_root_dir=str(tmp_path))
        trainer.validate(model, build_datamodule(data), ckpt_path=str(fitted["ckpt"]))
        assert meta["global_step"] > 0
        assert rec.seen == (meta["epoch"], meta["global_step"])
        assert set(model.plans) == {Mode.VAL}
        assert model.loaded_from_checkpoint
        assert not list(tmp_path.rglob("*.ckpt"))

    def test_validate_rejects_checkpoint_without_fit_loop_progress(self, data, fitted, tmp_path):
        """A checkpoint without ``loops`` is a `ConfigError`, never silent zero progress."""
        ckpt = torch.load(str(fitted["ckpt"]), weights_only=False)
        del ckpt["loops"]
        path = tmp_path / "noloops.ckpt"
        torch.save(ckpt, path)
        with pytest.raises(ConfigError, match="fit_loop"):
            make_trainer().validate(build_model(data), build_datamodule(data), ckpt_path=str(path))


class TestNormGarbageGuard:
    def test_fresh_fit_reads_current_norm_file(self, data, tmp_path):
        """Control for the resume test: a FRESH fit DOES materialise from file."""
        with open(data["nd"]) as fh:
            norm = yaml.safe_load(fh)
        for variables in norm.values():
            for entry in variables.values():
                entry["mean"] = float(entry["mean"]) + 1.0
        shifted_path = tmp_path / "norm_shifted.yaml"
        with open(shifted_path, "w") as fh:
            yaml.dump(norm, fh, sort_keys=False)
        model = build_model(data, norm_dict=shifted_path)
        dm = build_datamodule(data)
        make_trainer(limit_train_batches=1, limit_val_batches=1).fit(model, dm)
        expect_means = torch.tensor([0.1 * (i + 1) + 1.0 for i in range(len(JET_VARIABLES))])
        assert torch.allclose(model.net["norm"].means_jets, expect_means)


class TestBoundaryDemandGuards:
    """Demand provenance + early missing-producer errors."""

    def test_sink_origins_name_demanding_modules(self, data):
        # per-mode since the review (a merged map mis-attributed
        # writer-demanded TEST keys to their inactive FIT demander)
        origins = build_model(data).sink_origins()[Mode.FIT]
        assert (
            origins["labels.jets.flavour_label"]
            == "'jets_classification' (config: model.modules.jets_classification)"
        )
        assert "track_vertexing" in origins["labels.tracks.ftagTruthVertexIndex"]
        # feature demand is attributed too (norm is the first requirer)
        assert "model.modules." in origins["inputs.jets"]

    # test_sink_origins_attribute_writer_demand_per_mode removed:
    # WriterCallback/TaskWriter were deleted with callback.py/modules.py.

    def test_deleted_producer_raises_named_error(self, data):
        # the ergonomics journey-(d) repro: --model.modules.pool=null used to
        # surface as a confusing DATASET-side "sink key 'pooled.global'"
        # error listing only source keys; it must now fail at sink_demand,
        # naming the consuming modules and their config addresses
        modules = build_gn2v2_modules(data["nd"])
        del modules["pool"]
        model = SaltModule(modules, lrs=LRS)
        with pytest.raises(ConfigError, match="pooled.global") as excinfo:
            model.sink_demand()
        message = str(excinfo.value)
        assert "'jets_classification' (config: model.modules.jets_classification)" in message
        assert "dataset boundary" in message
        assert "add or restore a module" in message


class _AuxProbe(SaltModelModule):
    """An aux head producing a preds key NO loss consumes — pruned in FIT/VAL"""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)  # carries params so it is a real model module
        self.name = "aux"

    def declare_io(self, mode):
        from salt.graph.spec import IO, TensorSpec, unflatten_spec

        del mode
        return IO(
            requires=unflatten_spec({"pooled.global": TensorSpec()}),
            produces=unflatten_spec({"preds.jets.aux_probe": TensorSpec(shape=("B", 2))}),
        )


class _ProbeMetrics:
    """Fixture FIT/VAL-sink callback (duck-typed `fit_val_demand`)."""

    def __init__(self, *keys: str) -> None:
        self._keys = keys

    def fit_val_demand(self, model_modules):
        del model_modules
        return self._keys


class TestCallbackSinks:
    """FIT/VAL callback-declared sinks."""

    def _model_with_aux(self, data) -> SaltModule:
        modules = build_gn2v2_modules(data["nd"])
        modules["aux"] = _AuxProbe()
        modules["loss"] = LossSum()
        return SaltModule(modules, lrs=LRS)

    def _attach(self, model: SaltModule, *callbacks) -> None:
        from types import SimpleNamespace

        model._trainer = SimpleNamespace(  # noqa: SLF001 - duck-typed attach
            callbacks=list(callbacks), datamodule=SimpleNamespace(reader=None)
        )

    def test_no_callback_fit_val_sinks_are_loss_only(self, data):
        # the unchanged baseline: with no FIT/VAL-sink callback, sinks stay
        # ['loss.total'] (the previous behaviour) in BOTH training modes
        model = self._model_with_aux(data)
        assert model._sink_prep().by_mode[Mode.FIT].anchors == ["loss.total"]  # noqa: SLF001
        assert model._sink_prep().by_mode[Mode.VAL].anchors == ["loss.total"]  # noqa: SLF001

    def test_callback_declaring_nothing_is_a_noop(self, data):
        # a callback whose fit_val_demand returns no keys must not alter sinks
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics())
        assert model._sink_prep().by_mode[Mode.FIT].anchors == ["loss.total"]  # noqa: SLF001

    def test_declared_preds_becomes_fit_val_sink(self, data):
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics("preds.jets.aux_probe"))
        for mode in (Mode.FIT, Mode.VAL):
            sinks = model._sink_prep().by_mode[mode].anchors  # noqa: SLF001
            assert sinks[0] == "loss.total"  # loss anchor stays first
            assert "preds.jets.aux_probe" in sinks

    def test_callback_demand_is_training_only(self, data):
        # the TRAINING-mode mirror: callback demand never leaks into TEST/ONNX
        # (those sinks are the writer manifest, not callback requires)
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics("preds.jets.aux_probe"))
        fitval_callbacks = select_fitval_callbacks(model._trainer.callbacks)  # noqa: SLF001
        assert callback_demand(model._graph_modules, Mode.TEST, fitval_callbacks) == {}  # noqa: SLF001
        assert callback_demand(model._graph_modules, Mode.ONNX, fitval_callbacks) == {}  # noqa: SLF001

    def test_pruned_producer_kept_alive_by_callback_sink(self, data):
        # THE behaviour the deferral note describes: a preds.* key no task keeps
        # alive is demand-PRUNED in FIT without the callback, and ALIVE with it
        from salt.graph.planner import compile_plan
        from salt.tests._fixtures.gn2v2_fixture import gn2v2_sources

        model = self._model_with_aux(data)
        src = gn2v2_sources()
        pruned = compile_plan(
            model._graph_modules,  # noqa: SLF001
            Mode.FIT,
            sources=src,
            sinks=model._sink_prep().by_mode[Mode.FIT].anchors,  # noqa: SLF001
        )
        assert "aux" not in pruned.module_names  # negative control: pruned

        self._attach(model, _ProbeMetrics("preds.jets.aux_probe"))
        kept = compile_plan(
            model._graph_modules,  # noqa: SLF001
            Mode.FIT,
            sources=src,
            sinks=model._sink_prep().by_mode[Mode.FIT].anchors,  # noqa: SLF001
        )
        assert "aux" in kept.module_names  # callback sink keeps the producer alive

    def test_callback_label_demand_extends_boundary(self, data):
        # a callback's dataset-namespace require not already task-demanded
        # extends the FIT/VAL boundary demand, attributed to the callback
        model = build_gn2v2_modules(data["nd"])
        model["loss"] = LossSum()
        model = SaltModule(model, lrs=LRS)
        self._attach(model, _ProbeMetrics("labels.jets.extra_truth"))
        demand, origins = boundary_demand(model._graph_modules, model._sink_prep())[  # noqa: SLF001
            Mode.FIT
        ]
        assert "labels.jets.extra_truth" in demand
        assert "callback" in origins["labels.jets.extra_truth"]
        assert "_ProbeMetrics" in origins["labels.jets.extra_truth"]

    def test_already_task_demanded_label_not_duplicated(self, data):
        # a callback re-declaring an existing task label is a no-op on the
        # boundary demand list (no duplicate, no error)
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics("labels.jets.flavour_label"))
        demand, _ = boundary_demand(model._graph_modules, model._sink_prep())[  # noqa: SLF001
            Mode.FIT
        ]
        assert demand.count("labels.jets.flavour_label") == 1

    def test_undecidable_callback_key_raises_quality_error(self, data):
        # negative control: a callback demanding a key that is neither
        # model-produced nor a dataset namespace fails loudly, naming the
        # key AND the callback (mirror of the TEST writer guard)
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics("bogus.namespace.key"))
        with pytest.raises(ConfigError, match="bogus.namespace.key") as excinfo:
            boundary_demand(model._graph_modules, model._sink_prep())  # noqa: SLF001
        message = str(excinfo.value)
        assert "callback" in message
        assert "_ProbeMetrics" in message

    def test_wildcard_callback_key_rejected(self, data):
        # callback requires are concrete keys; a wildcard fails
        model = self._model_with_aux(data)
        self._attach(model, _ProbeMetrics("labels.jets.*"))
        with pytest.raises(ConfigError, match="wildcard"):
            boundary_demand(model._graph_modules, model._sink_prep())  # noqa: SLF001


class TestLazyModeAnchors:
    """`_sink_prep`/`compile_mode` only derive anchors for the modes actually asked for."""

    def test_fit_val_only_tasks_prepare_and_compile_fit_without_test_anchors(self, data):
        from types import SimpleNamespace

        from salt.model.modules.tasks import _parse_expose  # noqa: PLC0415, PLC2701 - test-local

        modules = build_gn2v2_modules(data["nd"])
        for name in TASKS:
            modules[name].expose_modes = _parse_expose(["fit", "val"], type(modules[name]).__name__)
        modules["loss"] = LossSum()
        model = SaltModule(modules, lrs=LRS)
        dm = build_datamodule(data)
        model._trainer = SimpleNamespace(callbacks=[], datamodule=dm)  # noqa: SLF001
        dm.set_sinks(model.sink_demand())
        model.sink_origins()
        dm.setup("fit")
        model.compile_mode(Mode.FIT, model._boundary(dm.train_dset, "train"))  # noqa: SLF001
        model.compile_mode(Mode.VAL, model._boundary(dm.val_dset, "val"))  # noqa: SLF001
        with pytest.raises(ConfigError, match="anchor on predictions"):
            model._sink_prep((Mode.TEST,))  # noqa: SLF001

    def test_no_loss_module_prepares_and_compiles_test_without_fit_anchors(self, data):
        from types import SimpleNamespace

        modules = build_gn2v2_modules(data["nd"])
        del modules["loss"]
        model = SaltModule(modules, lrs=LRS)
        dm = build_datamodule(data)
        model._trainer = SimpleNamespace(callbacks=[], datamodule=dm)  # noqa: SLF001
        dm.set_sinks(model.sink_demand())
        dm.setup("test")
        model.compile_mode(Mode.TEST, model._boundary(dm.test_dset, "test"))  # noqa: SLF001
        with pytest.raises(ConfigError, match="no module produces 'loss.total'"):
            model._sink_prep((Mode.FIT,))  # noqa: SLF001

    def test_compile_mode_onnx_still_rejected(self, data):
        with pytest.raises(ConfigError, match="compile_mode prepares FIT/VAL/TEST"):
            build_model(data).compile_mode(Mode.ONNX, {})


class TestClassNamesCheck:
    """The default-on class_names ↔ schema-attrs cross-check."""

    def make_reader(self, data) -> H5StructuredReader:
        return H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"])

    def test_matching_lists_pass_and_are_counted(self, data):
        from salt.model.validation import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        # the dummy file carries jets.attrs['flavour_label'] = [bjets, cjets,
        # ujets] (matching the fixture); tracks carry no class-name attrs
        assert check_class_names(modules, self.make_reader(data)) == 1

    def test_reordered_class_names_fail_set_and_order(self, data):
        from salt.model.validation import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        with pytest.raises(ConfigError, match="DIFFERENT ORDER") as excinfo:
            check_class_names(modules, self.make_reader(data))
        message = str(excinfo.value)
        assert "model.modules.jets_classification.init_args.class_names" in message
        assert "['bjets', 'ujets', 'cjets']" in message  # configured
        assert "['bjets', 'cjets', 'ujets']" in message  # schema

    def test_wrong_class_set_fails(self, data):
        from salt.model.validation import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "cjets", "ujets", "taujets")
        with pytest.raises(ConfigError, match="class sets differ"):
            check_class_names(modules, self.make_reader(data))

    def test_no_schema_artifact_is_noop(self, data):
        from salt.model.validation import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        reader = H5StructuredReader(groups={"jets": {"global_object": True}, "tracks": {"global_object": False}})
        assert check_class_names(modules, reader) == 0

    def test_default_on_at_fit_setup(self, data):
        # the check must fire on every fit, not only in the validate CLI
        model = build_model(data)
        model.net["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        with pytest.raises(ConfigError, match="DIFFERENT ORDER"):
            make_trainer().fit(model, build_datamodule(data))


# TestExposeSilencesDeadPreds removed:
# WriterCallback/TaskWriter were deleted with callback.py/modules.py.


class TestOriginWeightingResolvedAtSetup:
    """Name-based origin_weighting resolves at fit/test setup."""

    @staticmethod
    def _schema_with_origin_attr(data):
        # the dummy file's schema carries no tracks origin class-name attr;
        # build an in-memory Schema that does (umami-preprocessing convention)
        from salt.schema import GroupSchema, Schema, load_schema

        schema = load_schema(data["schema"])
        groups = dict(schema.groups)
        tracks = groups["tracks"]
        groups["tracks"] = GroupSchema(
            fields=dict(tracks.fields),
            attrs={**tracks.attrs, "ftagTruthOriginLabel": list(ORIGIN_CLASSES)},
        )
        return Schema(groups=groups, attrs=dict(schema.attrs))

    def _datamodule_with_origin_schema(self, data) -> SaltDataModule:
        schema = self._schema_with_origin_attr(data)
        modules = {
            "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=schema),
            "features": Features(
                variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
            ),
            "labels": Labels(),
        }
        return SaltDataModule(
            modules,
            batch_size=100,
            num_workers=0,
            train_file=data["h5"],
            val_file=data["h5"],
            test_file=data["h5"],
        )

    def test_setup_resolves_names_to_v1_ids(self, data):
        from salt.model.modules.tasks import VertexingTaskModule

        modules = build_gn2v2_modules(data["nd"])
        modules["track_vertexing"] = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel", context="pooled.global", weight=1.5,
            dense={"hidden_layers": [16], "activation": "ReLU"},
            origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
        )
        modules["track_vertexing"].name = "track_vertexing"
        modules["loss"] = LossSum()
        model = SaltModule(modules, lrs=LRS)
        assert model.net["track_vertexing"].heavy_ids is None  # unresolved pre-fit
        # a real fit: setup resolves the names against the schema before bind
        make_trainer().fit(model, self._datamodule_with_origin_schema(data))
        vtx = model.net["track_vertexing"]
        assert vtx.heavy_ids == (3, 4, 5)
        assert vtx.fake_ids == (1,)

    def test_name_based_without_schema_fails_loudly(self, data):
        # a name-based config but the standard datamodule schema has no origin
        # class-name attr → loud setup error (resolve_origin_names raises)
        from salt.model.modules.tasks import VertexingTaskModule

        modules = build_gn2v2_modules(data["nd"])
        modules["track_vertexing"] = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel", context="pooled.global", weight=1.5,
            dense={"hidden_layers": [16], "activation": "ReLU"},
            origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
        )
        modules["track_vertexing"].name = "track_vertexing"
        modules["loss"] = LossSum()
        model = SaltModule(modules, lrs=LRS)
        with pytest.raises(ConfigError, match="no string-list attr"):
            make_trainer().fit(model, build_datamodule(data))


class _Pair(ModuleGroup):
    def __init__(self, members: dict) -> None:
        super().__init__()
        self.members = members

    def expand(self):
        return dict(self.members)


class TestModuleGroups:
    def test_group_expands_in_place_and_frozen_names_resolve(self, data):
        modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        keys, items = list(modules), list(modules.items())  # [6:8]: the two class heads
        g = dict(items[:6] + [("heads", _Pair(dict(items[6:8])))] + items[8:])
        model = SaltModule(g, lrs=LRS, training_schedule={"stages": {"fit": {"frozen": [keys[6]]}}})
        assert list(model.net) == keys and "heads" not in model.net  # members replace it in place
        assert model.net["jets_classification"].name == "jets_classification"

    def test_duplicate_and_nested_group_rejected(self):
        with pytest.raises(ConfigError, match="already defines"):  # group output vs LATER entry
            _expand_module_groups({"g": _Pair({"norm": _AuxProbe()}), "norm": _AuxProbe()})
        with pytest.raises(ConfigError, match="already defines"):  # group output vs earlier entry
            _expand_module_groups({"norm": _AuxProbe(), "g": _Pair({"norm": _AuxProbe()})})
        with pytest.raises(ConfigError, match="cannot nest"):
            _expand_module_groups({"g": _Pair({"inner": _Pair({})})})

    def test_group_never_joins_the_plan(self):
        with pytest.raises(ConfigError, match="module group"):
            _Pair({}).declare_io(Mode.FIT)
        with pytest.raises(NotImplementedError):
            ModuleGroup().expand()


class TestStagePlanPruning:
    def test_stage_executors_swap_advance_and_resume(self, data, caplog):
        """Stage 0 runs the pruned plans; advance installs the full ones; resume reinstalls."""
        modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        keys = list(modules)
        sched = {"stages": {"a": {"epochs": 2, "skip_losses": [keys[7]]}, "b": {}}}
        model = SaltModule(modules, lrs=LRS, training_schedule=sched)
        trainer = make_trainer(max_epochs=3, callbacks=[StopAfterFirstEpoch()])
        with caplog.at_level(logging.INFO, logger="salt"):
            trainer.fit(model, build_datamodule(data))
        assert "stage 'a' (0): skipping [track_origin]" in caplog.text
        assert "stage 'a' losses.jets_classification: 0/" in caplog.text
        assert "params frozen (0%)" in caplog.text
        assert "stage 'a' losses.track_origin:" not in caplog.text  # skipped: no diagnostics
        stage_plans, loss = model._stage_plans, model.net["loss"]  # noqa: SLF001
        full, pruned = model.plans[Mode.FIT], stage_plans[0][Mode.FIT]
        active = tuple(k for k in loss._loss_keys if k != f"losses.{keys[7]}")  # noqa: SLF001
        assert model._executors[Mode.FIT].plan is pruned  # noqa: SLF001
        assert keys[7] not in pruned.module_names and keys[7] in full.module_names
        assert stage_plans[1][Mode.FIT] is full and loss._active == active  # noqa: SLF001
        ckpt = data["dir"] / "stage_a.ckpt"
        trainer.save_checkpoint(ckpt)
        model.training_controller.advance_to_stage(1, trainer.global_step, 1, "epochs")
        assert model._executors[Mode.FIT].plan is full and loss._active is None  # noqa: SLF001
        model.setup("fit")  # a repeated setup recompiles the full plans, then re-enters stage 0
        assert model.plans[Mode.FIT].plan_hash == full.plan_hash
        assert model._executors[Mode.VAL].plan is model._stage_plans[0][Mode.VAL]  # noqa: SLF001
        fresh_modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        fresh = SaltModule(fresh_modules, lrs=LRS, training_schedule=sched)
        resumed = make_trainer(max_epochs=3, callbacks=[StopAfterFirstEpoch()])
        resumed.fit(fresh, build_datamodule(data), ckpt_path=ckpt)
        executors, plans = fresh._executors, fresh._stage_plans[0]  # noqa: SLF001
        assert all(executors[mode].plan is plans[mode] for mode in (Mode.FIT, Mode.VAL))
        assert keys[7] not in executors[Mode.FIT].plan.module_names
        assert fresh.net["loss"]._active == active  # noqa: SLF001

    @staticmethod
    def _spy_optimizer(monkeypatch) -> list[int]:
        calls: list[int] = []
        monkeypatch.setattr(
            SaltModule, "configure_optimizers", lambda _s: calls.append(1) or pytest.fail("built")
        )
        return calls

    def test_fully_frozen_active_loss_errors_at_setup(self, data, monkeypatch):
        """An active loss with every upstream parameter frozen fails before any optimizer."""
        keys, calls = list(build_gn2v2_modules(data["nd"])), self._spy_optimizer(monkeypatch)
        stage = {"trainable": [keys[6]]}
        model = SaltModule(
            build_gn2v2_modules(data["nd"]) | {"loss": LossSum()},
            lrs=LRS,
            training_schedule={"stages": {"fit": stage}},
        )
        with pytest.raises(ConfigError, match="skip_losses") as info:
            make_trainer().fit(model, build_datamodule(data))
        msg = str(info.value)
        assert "stage 'fit'" in msg and any(f"losses.{k}'" in msg for k in keys[7:9])
        assert not calls
        monkeypatch.undo()
        stage["skip_losses"] = [keys[7], keys[8]]
        modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        model = SaltModule(modules, lrs=LRS, training_schedule={"stages": {"fit": stage}})
        trainer = make_trainer()
        trainer.fit(model, build_datamodule(data))
        assert trainer.state.finished

    @pytest.mark.parametrize(
        "stage",
        [{"frozen": ["jets_classification"]}, {"trainable": TASKS}],
        ids=["frozen_head", "frozen_encoder"],
    )
    def test_partially_frozen_chains_fit(self, data, stage):
        """A frozen head over a trainable encoder, or trainable heads over a frozen one, fit."""
        modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        model = SaltModule(modules, lrs=LRS, training_schedule={"stages": {"fit": stage}})
        (trainer := make_trainer()).fit(model, build_datamodule(data))
        assert trainer.state.finished

    def test_frozen_share_logged_from_materialised_counts(self, data, caplog):
        """The INFO line reports the frozen/total upstream parameter counts of each loss."""
        modules = build_gn2v2_modules(data["nd"]) | {"loss": LossSum()}
        sched = {"stages": {"fit": {"frozen": ["encoder"]}}}
        model = SaltModule(modules, lrs=LRS, training_schedule=sched)
        with caplog.at_level(logging.INFO, logger="salt"):
            make_trainer().fit(model, build_datamodule(data))
        plan, key = model._stage_plans[0][Mode.FIT], "losses.jets_classification"  # noqa: SLF001
        up = [k for k in _upstream(plan, key) if k in model.net]
        n = {k: sum(p.numel() for p in model.net[k].parameters()) for k in up}
        cold, total = n["encoder"], sum(n.values())
        assert 0 < cold < total
        line = f"stage 'fit' {key}: {cold}/{total} params frozen ({100 * cold / total:.0f}%)"
        assert line in caplog.text

    def test_invalid_later_stage_rejected_at_initial_setup(self, data, monkeypatch):
        """A fully frozen active loss in a LATER stage fails the initial fit setup."""
        calls, modules = self._spy_optimizer(monkeypatch), build_gn2v2_modules(data["nd"])
        sched = {"stages": {"a": {"epochs": 1}, "b": {"trainable": [list(modules)[6]]}}}
        model = SaltModule(modules | {"loss": LossSum()}, lrs=LRS, training_schedule=sched)
        with pytest.raises(ConfigError, match="stage 'b'"):
            make_trainer(max_epochs=2).fit(model, build_datamodule(data))
        assert not calls

    def test_detached_sample_weight_is_not_a_gradient_path(self, data):
        """A trainable prior feeding only a label-kind weight leaves the frozen main loss frozen."""
        modules = build_gn2v2_modules(data["nd"])
        del modules["loss"]
        cls = dict(stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"])
        modules["jets_classification"] = ClassificationTaskModule(
            **cls, input="pooled.global", dense={"hidden_layers": [16], "activation": "ReLU"},
            sample_weight="nn_kinematic_weight",
            loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"reduction": "none"}},
        )
        modules["prior"] = ClassificationTaskModule(
            **cls, input="normed.jets", dense={"hidden_layers": [8], "activation": "ReLU"}
        )
        modules["kinematic_weight"] = KinematicWeight(
            task_name="prior", mode="nn", stream="jets", label="flavour_label", q_f=None,
            n_classes=3,
        )
        stage = {"trainable": ["prior"], "skip_losses": ["track_origin", "track_vertexing"]}
        model = SaltModule(
            modules | {"loss": LossSum()}, lrs=LRS, training_schedule={"stages": {"fit": stage}}
        )
        with pytest.raises(ConfigError, match="losses.jets_classification"):
            make_trainer().fit(model, build_datamodule(data))


class _PoisonedPool(GlobalAttentionPooling):
    """GN2v2 pooling whose output is poisoned (``+ value``) on chosen (mode, call) pairs."""

    def __init__(self, poison, value: float) -> None:
        super().__init__(input="encoded.seq", out="pooled.global")
        self.poison = poison
        self.value = value
        self.calls: dict[Mode, int] = {}

    def forward(self, b, mode):
        out = super().forward(b, mode)
        self.calls[mode] = n = self.calls.get(mode, 0) + 1
        if self.poison(mode, n):
            out = {key: value + self.value for key, value in out.items()}
        return out


SAVED = re.compile(r"batch saved to (\S+\.pt)$")


class TestNonFiniteLoss:
    """Every-step finite-loss check: localised by a debug replay, batch saved."""

    @staticmethod
    def fit_failure(data, tmp_path, poison, value=torch.nan, **trainer_kwargs) -> RuntimeError:
        modules = build_gn2v2_modules(data["nd"])
        modules["pool"] = _PoisonedPool(poison, value)
        modules["loss"] = LossSum()
        trainer_kwargs.setdefault("num_sanity_val_steps", 0)
        trainer = make_trainer(default_root_dir=tmp_path, **trainer_kwargs)
        with pytest.raises(RuntimeError) as info:
            trainer.fit(SaltModule(modules, lrs=LRS), build_datamodule(data))
        return info.value

    def test_nan_loss_replay_names_module_and_saves_batch(self, data, tmp_path):
        # FIT call 1 is the guarded first batch (clean); call 2 poisons the loss and
        # the replay (call 3) reproduces it under debug
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.FIT and n >= 2)
        msg = str(err)
        assert isinstance(err.__cause__, NonFiniteError)
        assert "[mode=FIT] module 'pool' produced non-finite values in 'pooled.global'" in msg
        assert "inputs it read were all finite" in msg
        path = Path(SAVED.search(msg).group(1))
        assert path == tmp_path / "non_finite_batch_fit_step1_rank0.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert set(payload) == {"batch", "rng", "global_step", "mode"}
        assert (payload["mode"], payload["global_step"]) == ("FIT", 1)
        assert "inputs" in payload["batch"]
        assert payload["rng"]["cpu"] is not None

    def test_guarded_first_batch_fails_directly(self, data, tmp_path):
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.FIT)
        assert isinstance(err.__cause__, NonFiniteError)
        assert "module 'pool' produced non-finite values" in str(err)
        assert Path(SAVED.search(str(err)).group(1)).name == "non_finite_batch_fit_step0_rank0.pt"

    def test_inf_is_detected(self, data, tmp_path):
        err = self.fit_failure(
            data, tmp_path, lambda mode, n: mode is Mode.FIT and n >= 2, value=torch.inf
        )
        assert re.search(r"module 'pool'.*\(NaN 0, \+Inf [1-9]\d*, -Inf 0 of \d+\)", str(err))

    def test_validation_step_is_checked(self, data, tmp_path):
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.VAL and n >= 2)
        assert isinstance(err.__cause__, NonFiniteError)
        assert "[mode=VAL] module 'pool' produced non-finite values" in str(err)
        assert "non_finite_batch_val_step" in str(err)

    def test_finite_replay_reports_not_reproduced(self, data, tmp_path):
        # one-shot poison: the replay (call 3) is clean
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.FIT and n == 2)
        msg = str(err)
        assert "loss.total is non-finite but a diagnostic forward re-run was finite" in msg
        assert "the failure was not reproduced and its origin is unresolved" in msg
        assert "loss.total = nan" in msg
        assert err.__cause__ is None
        assert Path(SAVED.search(msg).group(1)).is_file()

    def test_save_failure_preserves_the_original_error(self, data, tmp_path, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", refuse)
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.FIT and n >= 2)
        msg = str(err)
        assert "[mode=FIT] module 'pool' produced non-finite values in 'pooled.global'" in msg
        assert "failed (OSError: disk full)" in msg
        assert not list(tmp_path.glob("non_finite_batch_*"))

    def test_handler_calls_no_distributed_collective(self, data, tmp_path, monkeypatch):
        def collective(*_args, **_kwargs):
            raise AssertionError("the failure handler called a torch.distributed collective")

        for name in ("all_reduce", "barrier", "broadcast", "broadcast_object_list", "all_gather"):
            monkeypatch.setattr(torch.distributed, name, collective, raising=False)
        err = self.fit_failure(data, tmp_path, lambda mode, n: mode is Mode.FIT and n >= 2)
        assert "module 'pool' produced non-finite values" in str(err)
