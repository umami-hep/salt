"""Unit tests for `salt.data.readers.reader` (H5StructuredReader + its private
wildcard->VDS builder, folded in from the deleted VDS setup module).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from salt.data import Features, H5StructuredReader, InputSamples, Labels, SaltDataModule
from salt.data.readers.reader import _create_vds, _default_vds_path, _has_wildcard
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
    base = tmp_path_factory.mktemp("reader_vds")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_reader(data) -> H5StructuredReader:
    return H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"])


def build_modules(data, files: dict[str, Path] | None = None) -> dict:
    if files is None:
        files = {"train": data["h5"], "val": data["h5"], "test": data["h5"]}
    return {
        "input_samples": InputSamples(files=files),
        "reader": build_reader(data),
        "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
        "labels": Labels(),
    }


# --------------------------------------------------------------------------- #
# private VDS-builder helpers
# --------------------------------------------------------------------------- #


def test_has_wildcard() -> None:
    assert _has_wildcard(Path("/data/pp_output_*.h5"))
    assert _has_wildcard(Path("/data/pp_output_?.h5"))
    assert _has_wildcard(Path("/data/pp_output_[0-9].h5"))
    assert not _has_wildcard(Path("/data/*/pp_output.h5"))  # wildcard only in a parent dir
    assert not _has_wildcard(Path("/data/pp_output.h5"))


def test_default_vds_path_layout(tmp_path: Path) -> None:
    pattern = tmp_path / "pp_output_train_split_*.h5"
    out = _default_vds_path(pattern)
    assert out == tmp_path / "pp_output_train_split_vds" / "vds.h5"
    assert out.parent.is_dir()


def test_create_vds_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _create_vds(tmp_path / "no_such_*.h5")


def test_create_vds_reuses_marker_and_rebuilds_when_stale(data, tmp_path: Path) -> None:
    out = tmp_path / "v.h5"
    pattern = data["dir"] / "pp_output_*.h5"
    built = _create_vds(pattern, out)
    assert built == out
    assert out.is_file()
    assert out.with_suffix(".h5.done").exists()

    mtime = out.stat().st_mtime_ns
    _create_vds(pattern, out)
    assert out.stat().st_mtime_ns == mtime  # marker fresh -> no rebuild

    future = time.time() + 2
    os.utime(data["h5"], (future, future))
    _create_vds(pattern, out)
    assert out.stat().st_mtime_ns != mtime  # member newer than the VDS -> rebuilt


# --------------------------------------------------------------------------- #
# H5StructuredReader wildcard resolution
# --------------------------------------------------------------------------- #


def test_non_wildcard_source_is_the_file_itself(data) -> None:
    reader = H5StructuredReader(
        groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=data["h5"]
    )
    assert reader.source_path == data["h5"]


def test_wildcard_filename_builds_a_vds_in_prepare(data) -> None:
    pattern = data["dir"] / "pp_output_*.h5"
    reader = H5StructuredReader(
        groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=pattern
    )
    reader.prepare()
    assert reader.source_path != pattern
    assert reader.source_path.is_file()
    with h5py.File(reader.source_path, "r") as f:
        assert "jets" in f
    assert len(reader) == N_JETS


def test_wildcard_uses_explicit_vds_path(data, tmp_path: Path) -> None:
    pattern = data["dir"] / "pp_output_*.h5"
    explicit = tmp_path / "explicit.h5"
    reader = H5StructuredReader(
        groups={"jets": {}, "tracks": {}},
        schema=data["schema"],
        filename=pattern,
        vds_path=explicit,
    )
    assert reader.source_path == explicit
    clone = reader.with_source(filename=pattern, stage="val")
    assert clone.vds_path == reader.vds_path


def test_datamodule_wildcard_source_serves_the_same_bytes_as_a_direct_file(data) -> None:
    """Served bytes via a wildcard-resolved VDS == via the direct-file path."""
    pattern = data["dir"] / "pp_output_*.h5"
    dm_glob = SaltDataModule(
        modules=build_modules(data, files={"train": pattern, "val": pattern, "test": pattern}),
        batch_size=128,
        sinks=SINKS,
        pin_memory=False,
    )
    dm_glob.setup("fit")
    dm_file = SaltDataModule(
        modules=build_modules(data), batch_size=128, sinks=SINKS, pin_memory=False
    )
    dm_file.setup("fit")
    rows = np.s_[0:500]
    a = dm_glob.train_dset[rows]
    b = dm_file.train_dset[rows]
    assert torch.equal(a["inputs"]["jets"], b["inputs"]["jets"])
    assert torch.equal(a["inputs"]["tracks"], b["inputs"]["tracks"])
    assert torch.equal(a["masks"]["tracks"], b["masks"]["tracks"])
    for label in JET_LABELS:
        assert torch.equal(a["labels"]["jets"][label], b["labels"]["jets"][label])
    for label in TRACK_LABELS:
        assert torch.equal(a["labels"]["tracks"][label], b["labels"]["tracks"][label])
    r, _num = dm_glob._stage_reader(Mode.FIT)
    assert "*" not in str(r.source_path)
    assert r.source_path.is_file()
