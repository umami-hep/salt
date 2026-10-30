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

from salt.data import (
    Features,
    GlobalObjectCuts,
    H5StructuredReader,
    InputSamples,
    Labels,
    SaltDataModule,
)
from salt.data.base import WorkerCtx
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


# --------------------------------------------------------------------------- #
# H5StructuredReader per_split row cuts — eventNumber % 10 train/val/test split
# --------------------------------------------------------------------------- #


N_EVENTS = 53  # every residue 0..9 present; residues 0-2 six times, 3-9 five times
SPLIT_READ_FIELDS = {"jets": {"eventNumber": "test", "pt": "test"}}


def _write_split_file(path: Path) -> np.ndarray:
    event_numbers = np.arange(N_EVENTS, dtype=np.int64)
    jets = np.zeros(N_EVENTS, dtype=[("eventNumber", "i8"), ("pt", "f4")])
    jets["eventNumber"] = event_numbers
    jets["pt"] = np.linspace(1.0, 2.0, N_EVENTS, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets)
    return event_numbers


def _split_reader(path: Path, stage: str | None) -> H5StructuredReader:
    return H5StructuredReader(
        groups={"jets": {"global_object": True}},
        filename=path,
        cuts=GlobalObjectCuts(
            per_split={
                "train": ("eventNumber % 10 < 8",),
                "val": ("eventNumber % 10 == 8",),
                "test": ("eventNumber % 10 == 9",),
            }
        ),
        stage=stage,
    )


def _served_event_numbers(reader: H5StructuredReader) -> np.ndarray:
    reader.bind(WorkerCtx(mode=Mode.FIT, read_fields=SPLIT_READ_FIELDS, seed=0))
    return reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]["eventNumber"]


def test_per_split_partition_train_val_test_is_disjoint_and_covers_everything(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.h5"
    event_numbers = _write_split_file(path)
    residues = event_numbers % 10
    masks = {"train": residues < 8, "val": residues == 8, "test": residues == 9}

    readers = {stage: _split_reader(path, stage) for stage in masks}
    lens = {stage: len(r) for stage, r in readers.items()}
    assert lens == {"train": 43, "val": 5, "test": 5}
    assert sum(lens.values()) == N_EVENTS

    served: dict[str, np.ndarray] = {}
    for stage, mask in masks.items():
        assert lens[stage] == int(mask.sum())
        served[stage] = _served_event_numbers(readers[stage])
        np.testing.assert_array_equal(served[stage], event_numbers[mask])
        served_residues = served[stage] % 10
        if stage == "train":
            assert np.all(served_residues < 8)
        elif stage == "val":
            assert np.all(served_residues == 8)
        else:
            assert np.all(served_residues == 9)

    sets = {stage: set(ev.tolist()) for stage, ev in served.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sets["train"] | sets["val"] | sets["test"] == set(event_numbers.tolist())


def test_per_split_stage_threads_through_with_source(tmp_path: Path) -> None:
    path = tmp_path / "split.h5"
    event_numbers = _write_split_file(path)
    base = _split_reader(path=path, stage=None)
    clone = base.with_source(filename=path, stage="val")
    assert clone.stage == "val"
    np.testing.assert_array_equal(
        _served_event_numbers(clone), event_numbers[event_numbers % 10 == 8]
    )


def test_per_split_without_stage_serves_every_row(tmp_path: Path) -> None:
    path = tmp_path / "split.h5"
    event_numbers = _write_split_file(path)
    reader = _split_reader(path, stage=None)
    assert len(reader) == N_EVENTS
    assert reader._kept is None  # no cut applies to stage=None -> identity read path
    np.testing.assert_array_equal(_served_event_numbers(reader), event_numbers)
