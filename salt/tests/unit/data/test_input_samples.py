"""Tests for `InputSamples` + the datamodule data-sourcing setup pass (W3.A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from salt.data import (
    Features,
    GraphDataModule,
    H5StructuredReader,
    InputSamples,
    Labels,
)
from salt.data.input_samples import (
    SOURCE_REGISTRY,
    deepest_source_path,
    source_num,
)
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_setup_plan
from salt.graph.setup_executor import run_setup_plan
from salt.graph.spec import Mode
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

JET_VARS = ["pt_btagJes", "eta_btagJes"]
TRACK_VARS = ["d0", "z0SinTheta", "dphi", "deta"]
JET_LABELS = ["flavour_label"]
TRACK_LABELS = ["ftagTruthOriginLabel", "ftagTruthVertexIndex", "ftagTruthTypeLabel"]

FIT_SINKS = [
    "inputs.jets",
    "inputs.tracks",
    "masks.tracks",
    *[f"labels.jets.{label}" for label in JET_LABELS],
    *[f"labels.tracks.{label}" for label in TRACK_LABELS],
]
TEST_SINKS = ["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"]
SINKS = {Mode.FIT: FIT_SINKS, Mode.VAL: FIT_SINKS, Mode.TEST: TEST_SINKS}
N_JETS = 1000


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("input_samples")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_reader(data) -> H5StructuredReader:
    return H5StructuredReader(
        groups={"jets": {}, "tracks": {}},
        schema=data["schema"],
    )


def build_modules_with_input_samples(data, *, num=None) -> dict:
    """Module dict with an explicit InputSamples + a fileless reader prototype."""
    return {
        "input_samples": InputSamples(
            files={"train": data["h5"], "val": data["h5"], "test": data["h5"]},
            num=num,
        ),
        "reader": build_reader(data),
        "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
        "labels": Labels(),
    }


def build_modules_no_input_samples(data) -> dict:
    """Module dict WITHOUT InputSamples — the deprecated-alias path."""
    return {
        "reader": H5StructuredReader(
            groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=data["h5"]
        ),
        "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
        "labels": Labels(),
    }


# InputSamples unit behaviour — declare + setup (path arithmetic only)


class TestInputSamplesUnit:
    def test_declare_produces_pattern_and_num(self):
        inp = InputSamples(files={"train": "/a.h5", "val": "/b.h5", "test": "/c.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        io = inp.declare_setup_io("train")
        flat = _flat(io.produces)
        assert "source.reader.train.pattern" in flat
        assert flat["source.reader.train.pattern"].kind == "path"
        # whole-dict num SCALAR on the first stage
        assert "artifacts.reader.num" in flat
        assert flat["artifacts.reader.num"].kind == "scalar"

    def test_num_emitted_once_across_stages(self):
        inp = InputSamples(files={"train": "/a.h5", "val": "/b.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        train_flat = _flat(inp.declare_setup_io("train").produces)
        val_flat = _flat(inp.declare_setup_io("val").produces)
        assert "artifacts.reader.num" in train_flat  # first stage carries it
        assert "artifacts.reader.num" not in val_flat  # second stage does NOT

    def test_setup_is_pure_path_arithmetic(self):
        inp = InputSamples(
            files={"train": "/data/train.h5", "val": "/data/val.h5"},
            num={"train": 500, "val": -1},
        )
        inp.name = "input_samples"
        inp._reader = "reader"
        ctx = Bundle()
        inp.setup(ctx, "train")
        inp.setup(ctx, "val")
        assert ctx.get("source.reader.train.pattern") == "/data/train.h5"
        assert ctx.get("source.reader.val.pattern") == "/data/val.h5"
        # whole-dict opaque num leaf, indexed [stage]
        assert ctx.get("artifacts.reader.num") == {"train": 500, "val": -1}

    def test_wildcard_passes_through_verbatim(self):
        inp = InputSamples(files={"train": "/data/pp_*.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        ctx = Bundle()
        inp.setup(ctx, "train")
        assert ctx.get("source.reader.train.pattern") == "/data/pp_*.h5"

    def test_reader_name_required(self):
        inp = InputSamples(files={"train": "/a.h5"})
        inp.name = "input_samples"
        with pytest.raises(RuntimeError, match="no reader name wired"):
            inp.declare_setup_io("train")

    def test_unknown_stage_rejected(self):
        with pytest.raises(ValueError, match="unknown stage"):
            InputSamples(files={"fit": "/a.h5"})

    def test_empty_files_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            InputSamples(files={})

    def test_inactive_stage_empty(self):
        inp = InputSamples(files={"train": "/a.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        assert inp.declare_setup_io("test").is_empty()
        assert inp.setup(Bundle(), "test") is not None  # no-op, returns ctx

    def test_declare_io_per_batch_empty(self):
        inp = InputSamples(files={"train": "/a.h5"})
        io = inp.declare_io(Mode.FIT)
        assert not io.requires
        assert not io.produces


def _flat(nested):
    from salt.graph.setup_spec import flatten_source_spec

    return flatten_source_spec(nested)


# the setup pass end-to-end (compile_setup_plan + run_setup_plan)


class TestSetupPassResolution:
    def test_resolves_files_per_stage(self, data):
        inp = InputSamples(files={"train": data["h5"], "val": data["h5"]}, num={"train": 200})
        inp.name = "input_samples"
        inp._reader = "reader"
        ctx = Bundle()
        for stage in ("train", "val"):
            plan = compile_setup_plan({"input_samples": inp}, stage)
            run_setup_plan(plan, stage, ctx)
        assert ctx.get("source.reader.train.pattern") == str(data["h5"])
        assert ctx.get("source.reader.val.pattern") == str(data["h5"])
        assert source_num(ctx, "reader", "train") == 200
        assert source_num(ctx, "reader", "val") == -1  # whole-dict default

    def test_deepest_path_picks_pattern_in_w3a(self, data):
        # W3.A: only `pattern` exists in the registry, so deepest == pattern.
        inp = InputSamples(files={"train": data["h5"]})
        inp.name = "input_samples"
        inp._reader = "reader"
        ctx = Bundle()
        run_setup_plan(compile_setup_plan({"input_samples": inp}, "train"), "train", ctx)
        assert deepest_source_path(ctx, "reader", "train") == str(data["h5"])
        assert SOURCE_REGISTRY == ("pattern", "vds_path", "staged_path")


# datamodule integration — binds the reader from the resolved ctx


class TestDatamoduleBinding:
    def test_setup_binds_reader_from_ctx(self, data):
        dm = GraphDataModule(
            modules=build_modules_with_input_samples(data),
            batch_size=256,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        # the per-stage reader was bound to the ctx-resolved path
        assert dm.train_dset is not None
        assert dm.val_dset is not None
        assert len(dm.train_dset) == N_JETS
        assert len(dm.val_dset) == N_JETS
        # the resolved ctx carries the per-stage pattern keys
        assert dm._setup_ctx is not None
        assert dm._setup_ctx.get("source.reader.train.pattern") == str(data["h5"])
        assert dm._setup_ctx.get("source.reader.val.pattern") == str(data["h5"])

    def test_num_cap_applies_from_input_samples(self, data):
        dm = GraphDataModule(
            modules=build_modules_with_input_samples(data, num={"train": 300, "val": 400}),
            batch_size=64,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        assert len(dm.train_dset) == 300
        assert len(dm.val_dset) == 400

    def test_input_samples_is_setup_only_not_in_batch_dict(self, data):
        dm = GraphDataModule(
            modules=build_modules_with_input_samples(data), sinks=SINKS, pin_memory=False
        )
        assert "input_samples" in dm._setup_modules
        assert "input_samples" not in dm._batch_modules
        # and the per-batch compile (via _make_dataset) never sees it
        dm.setup("fit")
        assert "input_samples" not in dm.train_dset.plan.module_names

    def test_test_stage_resolves(self, data):
        dm = GraphDataModule(
            modules=build_modules_with_input_samples(data, num={"test": 250}),
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("test")
        assert dm.test_dset is not None
        assert len(dm.test_dset) == 250

    def test_two_input_samples_rejected(self, data):
        modules = build_modules_with_input_samples(data)
        modules["input_samples2"] = InputSamples(files={"train": data["h5"]})
        with pytest.raises(ConfigError, match="at most one InputSamples"):
            GraphDataModule(modules=modules, sinks=SINKS)


# deprecated-alias path (implicit InputSamples) + byte-identity parity


class TestAliasMigrationWindow:
    def test_alias_synthesises_implicit_input_samples(self, data):
        dm = GraphDataModule(
            modules=build_modules_no_input_samples(data),
            train_file=data["h5"],
            val_file=data["h5"],
            sinks=SINKS,
            pin_memory=False,
        )
        # an implicit InputSamples was synthesised into the setup namespace
        assert "input_samples" in dm._setup_modules
        assert isinstance(dm._setup_modules["input_samples"], InputSamples)
        dm.setup("fit")
        assert len(dm.train_dset) == N_JETS
        assert len(dm.val_dset) == N_JETS

    def test_alias_num_cap_carried_through(self, data):
        dm = GraphDataModule(
            modules=build_modules_no_input_samples(data),
            train_file=data["h5"],
            val_file=data["h5"],
            num_train=350,
            num_val=450,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        assert len(dm.train_dset) == 350
        assert len(dm.val_dset) == 450

    def test_byte_identical_input_samples_vs_alias(self, data):
        """The parity-preserving default: served bytes identical via either path."""
        dm_is = GraphDataModule(
            modules=build_modules_with_input_samples(data),
            batch_size=128,
            sinks=SINKS,
            pin_memory=False,
        )
        dm_is.setup("fit")
        dm_alias = GraphDataModule(
            modules=build_modules_no_input_samples(data),
            train_file=data["h5"],
            val_file=data["h5"],
            batch_size=128,
            sinks=SINKS,
            pin_memory=False,
        )
        dm_alias.setup("fit")
        rows = np.s_[0:500]
        a = dm_is.train_dset[rows]
        b = dm_alias.train_dset[rows]
        assert torch.equal(a["inputs"]["jets"], b["inputs"]["jets"])
        assert torch.equal(a["inputs"]["tracks"], b["inputs"]["tracks"])
        assert torch.equal(a["masks"]["tracks"], b["masks"]["tracks"])
        for label in JET_LABELS:
            assert torch.equal(
                a["labels"]["jets"][label], b["labels"]["jets"][label]
            )
        for label in TRACK_LABELS:
            assert torch.equal(
                a["labels"]["tracks"][label], b["labels"]["tracks"][label]
            )
