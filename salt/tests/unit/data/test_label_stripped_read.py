"""A label-stripped inference read serves batches (RUN level).

The static half of the gate (`output_time_requires(ONNX)` names no labels) lives in
``test_target_labels.py``; here the demand actually DRIVES a `SaltDataset` over a
real H5 file whose label fields have been physically removed, proving no label
dataset is demanded or read. The full command-level run is covered separately
by the ``salt inference`` gate on a label-stripped copy.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import repack_fields

from salt.data import Features, SaltDataset, H5StructuredReader, Labels
from salt.graph.errors import GraphError
from salt.graph.spec import Mode
from salt.model.bind import ResolvedSchema
from salt.model.modules.tasks import ClassificationTaskModule
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

JET_VARS = ["pt_btagJes", "eta_btagJes"]
TRACK_VARS = ["d0", "z0SinTheta", "dphi", "deta"]

LABEL_FIELDS = {
    "flavour_label",
    "HadronConeExclTruthLabelID",
    "HadronGhostInitialTruthLabelPdgId",
    "ftagTruthOriginLabel",
    "ftagTruthTypeLabel",
    "ftagTruthVertexIndex",
    "ftagTruthParentBarcode",
}

# the dataset-boundary input demand of an inference run (model inputs only —
# the label demand, if any, is harvested from the tasks per mode below)
INPUT_DEMAND = ["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"]


def _strip_labels(src: Path, dst: Path) -> None:
    """Copy ``src`` dropping every LABEL_FIELDS column from the structured datasets."""
    with h5py.File(src) as fin, h5py.File(dst, "w") as fout:
        for name, ds in fin.items():
            arr = ds[:]
            keep = [f for f in arr.dtype.names if f not in LABEL_FIELDS]
            out = fout.create_dataset(name, data=repack_fields(arr[keep]))
            for k, v in ds.attrs.items():
                if k not in LABEL_FIELDS:
                    out.attrs[k] = v


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("stripped")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    labelled = base / "labelled.h5"
    write_dummy_file(labelled, nd_path)
    stripped = base / "stripped.h5"
    _strip_labels(labelled, stripped)
    schemas: dict[str, Path] = {}
    for key, path in (("labelled", labelled), ("stripped", stripped)):
        schemas[key] = base / f"{key}_schema.yaml"
        save_schema(dump_schema(path), schemas[key])
    return {"labelled": labelled, "stripped": stripped, "schemas": schemas}


def _tasks() -> list[ClassificationTaskModule]:
    """Real Phase-C task heads whose ``output_time_requires`` defines the label demand."""  # noqa: DOC201 - test helper
    jets = ClassificationTaskModule(
        stream="jets",
        label="flavour_label",
        class_names=["bjets", "cjets", "ujets"],
        input="pooled.global",
    )
    jets.name = "jets_classification"
    tracks = ClassificationTaskModule(
        stream="tracks",
        label="ftagTruthOriginLabel",
        class_names=["Pileup", "Fake", "Primary", "FromB", "FromBC", "FromC", "FromTau", "Other"],
        sequence=True,
    )
    tracks.name = "track_origin"
    for task in (jets, tracks):
        task.bind(ResolvedSchema(widths={task.input_key: 8}))
    return [jets, tracks]


def _label_demand(mode: Mode) -> list[str]:
    return [
        dep
        for task in _tasks()
        for dep in task.output_time_requires(mode)
        if dep.startswith("labels.")
    ]


def _dataset(data: dict[str, Path], file_key: str, sinks: list[str]) -> SaltDataset:
    return SaltDataset(
        modules={
            "reader": H5StructuredReader(
                groups={"jets": {}, "tracks": {}},
                schema=data["schemas"][file_key],
                filename=data[file_key],
            ),
            "features": Features(variables={"jets": JET_VARS, "tracks": TRACK_VARS}),
            "labels": Labels(),
        },
        mode=Mode.TEST,
        sinks=sinks,
    )


def test_stripped_file_really_lacks_label_fields(data):
    """Fixture sanity: no LABEL_FIELDS column survives in any stripped dataset."""
    with h5py.File(data["stripped"]) as f:
        for name in f:
            assert not set(f[name].dtype.names) & LABEL_FIELDS, name


def test_label_stripped_read_serves_batches_under_inference_demand(data):
    """The run-level gate: with the tasks' export-mode (label-free) demand, a
    SaltDataset over the label-stripped file narrows its read set to input
    fields only and serves real batches with no ``labels.*`` leaf.
    """
    assert _label_demand(Mode.ONNX) == []  # export/inference demands no labels
    ds = _dataset(data, "stripped", INPUT_DEMAND)
    for stream, fields in ds.read_fields.items():
        assert not set(fields) & LABEL_FIELDS, f"label field in {stream} read set"
    batch = ds[np.s_[0:128]]
    assert "labels" not in batch
    assert batch["inputs"]["jets"].shape == (128, len(JET_VARS))
    assert batch["inputs"]["tracks"].shape[0] == 128
    assert str(batch["masks"]["tracks"].dtype) == "torch.bool"


def test_label_demand_on_stripped_file_fails_loudly(data):
    """Sensitivity control: the same stripped file with the tasks' TEST-mode label
    demand fails at plan/schema validation — the gate above cannot pass
    vacuously, and any label demand leaking into inference would be caught.
    """
    test_labels = _label_demand(Mode.TEST)
    assert test_labels == [
        "labels.jets.flavour_label",
        "labels.tracks.ftagTruthOriginLabel",
    ]

    def build_and_read():
        """Construct + read: if construction validates lazily, the read must fail."""  # noqa: DOC201 - test helper
        return _dataset(data, "stripped", [*INPUT_DEMAND, *test_labels])[np.s_[0:128]]

    with pytest.raises(GraphError, match=r"flavour_label|ftagTruthOriginLabel"):
        build_and_read()


def test_labelled_file_serves_the_same_test_demand(data):
    """Control: the un-stripped twin serves the full TEST demand, labels included."""
    ds = _dataset(data, "labelled", [*INPUT_DEMAND, *_label_demand(Mode.TEST)])
    batch = ds[np.s_[0:128]]
    assert batch["labels"]["jets"]["flavour_label"].shape == (128,)
    assert batch["labels"]["tracks"]["ftagTruthOriginLabel"].shape[0] == 128
