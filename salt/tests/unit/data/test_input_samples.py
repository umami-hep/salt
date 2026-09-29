"""Tests for `InputSamples` + the datamodule's per-stage source resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from salt.data import (
    Features,
    SaltDataModule,
    H5StructuredReader,
    InputSamples,
    Labels,
)
from salt.graph.errors import ConfigError
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


# InputSamples unit behaviour — a per-stage (file, num) config holder


class TestInputSamplesUnit:
    def test_source_wildcard_passes_through_verbatim(self):
        inp = InputSamples(files={"train": "/data/train_*.h5"})
        assert inp.source("train") == ("/data/train_*.h5", -1)

    def test_source_str_and_num_default(self):
        inp = InputSamples(
            files={"train": Path("/data/train.h5"), "val": "/data/val.h5"},
            num={"train": 500},
        )
        assert inp.source("train") == ("/data/train.h5", 500)
        assert isinstance(inp.source("train")[0], str)
        assert inp.source("val") == ("/data/val.h5", -1)

    def test_unconfigured_stage_is_none(self):
        inp = InputSamples(files={"train": "/a.h5"})
        assert inp.source("test") == (None, -1)

    def test_unknown_stage_rejected(self):
        with pytest.raises(ValueError, match="unknown stage"):
            InputSamples(files={"fit": "/a.h5"})

    def test_empty_files_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            InputSamples(files={})

    def test_declare_io_per_batch_empty(self):
        inp = InputSamples(files={"train": "/a.h5"})
        io = inp.declare_io(Mode.FIT)
        assert not io.requires
        assert not io.produces


# datamodule integration — binds each stage's reader from InputSamples.source


class TestDatamoduleBinding:
    def test_setup_binds_reader_from_ctx(self, data):
        dm = SaltDataModule(
            modules=build_modules_with_input_samples(data),
            batch_size=256,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        # the per-stage reader was bound to the InputSamples-resolved path
        assert dm.train_dset is not None
        assert dm.val_dset is not None
        assert len(dm.train_dset) == N_JETS
        assert len(dm.val_dset) == N_JETS
        assert dm._resolve_source(Mode.FIT) == (str(data["h5"]), -1)
        assert dm._resolve_source(Mode.VAL) == (str(data["h5"]), -1)

    def test_num_cap_applies_from_input_samples(self, data):
        dm = SaltDataModule(
            modules=build_modules_with_input_samples(data, num={"train": 300, "val": 400}),
            batch_size=64,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        assert len(dm.train_dset) == 300
        assert len(dm.val_dset) == 400

    def test_input_samples_excluded_from_batch_modules(self, data):
        dm = SaltDataModule(
            modules=build_modules_with_input_samples(data), sinks=SINKS, pin_memory=False
        )
        assert "input_samples" in dm.modules
        assert "input_samples" not in dm.batch_modules
        # and the per-batch compile (via _make_dataset) never sees it
        dm.setup("fit")
        assert "input_samples" not in dm.train_dset.plan.module_names

    def test_test_stage_resolves(self, data):
        dm = SaltDataModule(
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
            SaltDataModule(modules=modules, sinks=SINKS)

    def test_mixed_explicit_and_legacy_kwarg_is_unconfigured(self, data):
        """An explicit InputSamples omitting `test` ignores a legacy test_file (G9-03)."""
        modules = build_modules_with_input_samples(data)
        modules["input_samples"] = InputSamples(files={"train": data["h5"], "val": data["h5"]})
        dm = SaltDataModule(modules=modules, test_file=data["h5"], sinks=SINKS, pin_memory=False)
        assert dm._resolve_source(Mode.TEST) == (None, -1)
        with pytest.raises(ConfigError, match="no file configured"):
            dm._make_dataset(Mode.TEST)
        with pytest.raises(ConfigError, match="No test file specified"):
            dm.setup("test")


# deprecated-alias path (implicit InputSamples) + byte-identity parity


class TestAliasMigrationWindow:
    def test_alias_synthesises_implicit_input_samples(self, data):
        dm = SaltDataModule(
            modules=build_modules_no_input_samples(data),
            train_file=data["h5"],
            val_file=data["h5"],
            sinks=SINKS,
            pin_memory=False,
        )
        assert isinstance(dm.modules["input_samples"], InputSamples)
        assert "input_samples" not in dm.batch_modules
        dm.setup("fit")
        assert len(dm.train_dset) == N_JETS
        assert len(dm.val_dset) == N_JETS

    def test_alias_num_cap_carried_through(self, data):
        dm = SaltDataModule(
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
        dm_is = SaltDataModule(
            modules=build_modules_with_input_samples(data),
            batch_size=128,
            sinks=SINKS,
            pin_memory=False,
        )
        dm_is.setup("fit")
        dm_alias = SaltDataModule(
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
