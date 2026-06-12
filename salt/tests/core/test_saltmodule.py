"""Tests for `SaltModule` (plan 05, stage B) — seeds of gates G2 and G5.

End-to-end on CPU with the dummy-file generators: the A2 GN2v2 module dict
(`gn2v2_fixture`) is trained through a real ``trainer.fit`` over the A1 data
pipeline (`GraphDataModule`), then checkpoint round-trips, plan-hash
verification (FIT strict / others warn, design risk 9), and the
resume-skips-materialise semantics (design §2.3) are exercised.

The fitted model/checkpoint are module-scoped fixtures: one short fit feeds
all read-only assertions; mutation-prone paths (resume, tampered payloads)
build their own instances.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from lightning import Callback, Trainer

from salt.core.data import Features, GraphDataModule, H5StructuredReader, Labels
from salt.core.graph import Bundle, ConfigError, Mode
from salt.core.nn.modules import LossSum
from salt.core.saltmodule import CKPT_KEY, SaltModule, bundle_as_v1_outputs
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

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
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_datamodule(data, **kwargs) -> GraphDataModule:
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
    return GraphDataModule(modules, batch_size=100, num_workers=0, **kwargs)


def build_model(data, norm_dict=None, **kwargs) -> SaltModule:
    # GN2v2 module dict with an UN-narrowed LossSum (SaltModule must narrow it)
    modules = build_gn2v2_modules(norm_dict or data["nd"])
    modules["loss"] = LossSum()
    kwargs.setdefault("lrs_config", LRS)
    return SaltModule(modules, **kwargs)


def make_trainer(**kwargs) -> Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("limit_train_batches", 2)
    kwargs.setdefault("limit_val_batches", 2)
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
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
    """Stop a (nominally 2-epoch) fit after epoch 0, leaving room to resume.

    The OneCycleLR total_steps is fixed at first compile from the trainer's
    ``estimated_stepping_batches`` and restored verbatim from the checkpoint
    (v1 semantics) — stopping early (rather than ``max_epochs=1``) keeps the
    resumed run inside the schedule.
    """

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
        with pytest.raises(ConfigError, match="lrs_config"):
            build_model(data, lrs_config={"initial": 1e-3})

    def test_bad_optimizer_rejected(self, data):
        with pytest.raises(ConfigError, match="optimizer"):
            build_model(data, optimizer="SGD")

    def test_setup_without_datamodule_rejected(self, data):
        with pytest.raises(ConfigError, match="GraphDataModule"):
            build_model(data).setup("fit")

    def test_unsupported_stage_rejected(self, data):
        with pytest.raises(ConfigError, match="validate"):
            build_model(data).setup("validate")

    def test_forward_without_plan_rejected(self, data):
        with pytest.raises(ConfigError, match="no compiled plan"):
            build_model(data)({}, Mode.TEST)


class TestFit:
    """Gate G2 seed: a short `trainer.fit` on the dummy file, CPU."""

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
        """GraphDataModule.setup adopted the model's sink_demand (no set_sinks call)."""
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
        # the migration shim exposes the v1 outputs view (design §3.4)
        v1_view = bundle_as_v1_outputs(bundle)
        assert set(v1_view["preds"]) == {"jets", "tracks"}
        assert set(v1_view["pad_masks"]) == {"tracks"}
        assert v1_view["labels"] == {}


class TestCheckpoint:
    """Gate G5 seed: schema + plan hashes round-trip; resume skips materialise."""

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
        # the modules dict is NOT pickled into hparams (design §3.4)
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
            model._loaded_from_checkpoint = before  # noqa: SLF001 - restore fixture state
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
            model._loaded_from_checkpoint = before  # noqa: SLF001 - restore fixture state
            model._ckpt_plan_hashes = {}  # noqa: SLF001

    def test_resume_skips_materialise(self, data, fitted):
        """Resume continues training WITHOUT materialise overwriting loaded buffers.

        The resumed model's Normaliser points at a NONEXISTENT norm dict: if
        materialise ran on resume it would raise FileNotFoundError; instead
        the buffers must arrive from the checkpoint (design §2.3).
        """
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
        """A config change between save and resume trips the FIT hash gate."""
        modules = build_gn2v2_modules(data["nd"], embed_dim=24)
        modules["loss"] = LossSum()
        model = SaltModule(modules, lrs_config=LRS)
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
            model._loaded_from_checkpoint = before  # noqa: SLF001 - restore fixture state


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
    """Stage-E fixes: demand provenance + early missing-producer errors (§4.1)."""

    def test_sink_origins_name_demanding_modules(self, data):
        # per-mode since the M3 review (a merged map mis-attributed
        # writer-demanded TEST keys to their inactive FIT demander)
        origins = build_model(data).sink_origins()[Mode.FIT]
        assert (
            origins["labels.jets.flavour_label"]
            == "'jets_classification' (config: model.modules.jets_classification)"
        )
        assert "track_vertexing" in origins["labels.tracks.ftagTruthVertexIndex"]
        # feature demand is attributed too (norm is the first requirer)
        assert "model.modules." in origins["inputs.jets"]

    def test_sink_origins_attribute_writer_demand_per_mode(self, data):
        # M3-review fix: a writer-demanded TEST label is attributed to the
        # WRITER in Mode.TEST while Mode.FIT keeps the task attribution —
        # the old merged map won FIT-first for both
        from types import SimpleNamespace

        from salt.core.graph.spec import TensorSpec
        from salt.core.writers import TaskWriter, Writer, WriterCallback

        class TruthWriter(Writer):
            def requires(self, ctx):
                del ctx
                return {"labels.jets.flavour_label": TensorSpec(kind="label")}

            def columns(self, ctx):
                raise NotImplementedError  # static demand only in this test

            def write(self, bundle, rows):
                raise NotImplementedError

        wcb = WriterCallback(modules={"truth": TruthWriter(), "tasks": TaskWriter()})
        reader = H5StructuredReader(groups={"jets": {"vector": True}, "tracks": {"vector": False}})
        model = build_model(data)
        model._trainer = SimpleNamespace(  # noqa: SLF001 - duck-typed attach
            callbacks=[wcb], datamodule=SimpleNamespace(reader=reader)
        )
        origins = model.sink_origins()
        assert (
            origins[Mode.TEST]["labels.jets.flavour_label"]
            == "writer 'truth' (config: writers.modules.truth)"
        )
        assert (
            origins[Mode.FIT]["labels.jets.flavour_label"]
            == "'jets_classification' (config: model.modules.jets_classification)"
        )

    def test_deleted_producer_raises_named_error(self, data):
        # the ergonomics journey-(d) repro: --model.modules.pool=null used to
        # surface as a confusing DATASET-side "sink key 'pooled.global'"
        # error listing only source keys; it must now fail at sink_demand,
        # naming the consuming modules and their config addresses
        modules = build_gn2v2_modules(data["nd"])
        del modules["pool"]
        model = SaltModule(modules, lrs_config=LRS)
        with pytest.raises(ConfigError, match="pooled.global") as excinfo:
            model.sink_demand()
        message = str(excinfo.value)
        assert "'jets_classification' (config: model.modules.jets_classification)" in message
        assert "dataset boundary" in message
        assert "add or restore a module" in message


class TestClassNamesCheck:
    """The §2.6 default-on class_names ↔ schema-attrs cross-check (stage-E HIGH fix)."""

    def make_reader(self, data) -> H5StructuredReader:
        return H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"])

    def test_matching_lists_pass_and_are_counted(self, data):
        from salt.core.saltmodule import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        # the dummy file carries jets.attrs['flavour_label'] = [bjets, cjets,
        # ujets] (matching the fixture); tracks carry no class-name attrs
        assert check_class_names(modules, self.make_reader(data)) == 1

    def test_reordered_class_names_fail_set_and_order(self, data):
        from salt.core.saltmodule import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        with pytest.raises(ConfigError, match="DIFFERENT ORDER") as excinfo:
            check_class_names(modules, self.make_reader(data))
        message = str(excinfo.value)
        assert "model.modules.jets_classification.init_args.class_names" in message
        assert "['bjets', 'ujets', 'cjets']" in message  # configured
        assert "['bjets', 'cjets', 'ujets']" in message  # schema

    def test_wrong_class_set_fails(self, data):
        from salt.core.saltmodule import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "cjets", "ujets", "taujets")
        with pytest.raises(ConfigError, match="class sets differ"):
            check_class_names(modules, self.make_reader(data))

    def test_no_schema_artifact_is_noop(self, data):
        from salt.core.saltmodule import check_class_names

        modules = build_gn2v2_modules(data["nd"])
        modules["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        reader = H5StructuredReader(groups={"jets": {"vector": True}, "tracks": {"vector": False}})
        assert check_class_names(modules, reader) == 0

    def test_default_on_at_fit_setup(self, data):
        # the check must fire on every fit, not only in the validate CLI
        model = build_model(data)
        model.net["jets_classification"].class_names = ("bjets", "ujets", "cjets")
        with pytest.raises(ConfigError, match="DIFFERENT ORDER"):
            make_trainer().fit(model, build_datamodule(data))
