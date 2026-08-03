"""The MaskFormer eval-H5 objects group via the unified path: the
`MaskFormerObjects` TEST reconstruction node + the generic H5OutputSink
``object_groups`` capability (byte parity vs the v1 op chain).

Replaces the retired sink-hosted writer (`MaskFormerObjectsSink`): the object
math now lives ONLY in the graph node, the sink is a dumb terminal that packs
declared leaves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.graph.bundle import Bundle
from salt.graph.spec import Mode, flatten_spec
from salt.outputs import MaskFormerObjects, ObjectGroup, ObjectGroupField
from salt.outputs.sinks.h5_sink import H5OutputSink
from salt.outputs.output_schema import OutputColumn
from salt.utils.mask_utils import indices_from_mask

pytestmark = pytest.mark.cpu_always

OBJECT_CLASSES = ["b", "c", "null"]
RUN_NAME = "MFrun"
# the unified MaskFormer track->object index suffix (eval-H5 and ONNX share it)
INDEX_SUFFIX = "HadronIndex"
_M = 5   # num objects (query bank)
_N_TRACKS = 10
_F4, _I8 = np.dtype("f4"), np.dtype("i8")


# The MaskFormer eval-H5 object groups (mirror salt/configs/MaskFormer.yaml).
def _object_groups() -> list[ObjectGroup]:
    return [
        ObjectGroup(
            name="objects",
            shape=[_M],
            fields=[
                ObjectGroupField(
                    leaf="objects.class_probs", suffixes=["pb", "pc", "pnull"], dtype="f4"
                ),
                ObjectGroupField(
                    leaf="labels.objects.object_class", suffixes=["class_label"],
                    dtype="i8", prefix=False, kind="label",
                ),
            ],
        ),
        ObjectGroup(
            name="object_masks",
            shape=[_M, "tracks"],
            fields=[
                ObjectGroupField(
                    leaf="labels.objects.masks", suffixes=["truth_mask"],
                    dtype="i8", prefix=False, kind="label",
                ),
                ObjectGroupField(
                    leaf="objects.masks", suffixes=["mask_logits"], dtype="f4", prefix=False
                ),
            ],
        ),
        ObjectGroup(
            name="tracks",
            fields=[
                ObjectGroupField(
                    leaf="outputs.tracks.object_index", suffixes=[INDEX_SUFFIX], dtype="i8"
                ),
            ],
        ),
    ]


def _node() -> MaskFormerObjects:
    node = MaskFormerObjects(n_reg=3, stream="objects", constituent_stream="tracks")
    node.name = "maskformer_objects_onnx"
    return node


def _batch_bundle(batch_size=6, n_tracks=_N_TRACKS):
    from salt.tests._fixtures.v2_builders import make_maskformer_writer_batch

    batch = make_maskformer_writer_batch(batch_size=batch_size, n_tracks=n_tracks)
    b = Bundle()
    for key, value in batch.items():
        b.set(key, value)
    # run the node's TEST reconstruction to mint outputs.tracks.object_index,
    # exactly as the graph would in a TEST plan.
    b.set("outputs.tracks.object_index", _node().forward(b, Mode.TEST)["outputs.tracks.object_index"])
    return b, batch


def _sink(*, run_name=RUN_NAME, n_tracks=_N_TRACKS, half=False) -> H5OutputSink:
    sink = H5OutputSink(object_groups=_object_groups(), half_precision=half)
    # seed one reader-stream task column (the explicit-outputs surface was
    # retired; in production the columns come from the outputs: section).
    sink._columns = (OutputColumn(key="outputs.jets.cls", suffixes=["pb"]),)  # noqa: SLF001
    sink._columns_resolved = True  # noqa: SLF001
    sink._run_name = run_name  # noqa: SLF001
    sink._seq_lengths = {"tracks": n_tracks}  # noqa: SLF001
    sink._object_shapes = sink._resolve_object_shapes(("jets", "tracks"))  # noqa: SLF001
    return sink


# 1. the reconstruction node's TEST mode (the object math's ONE home)


class TestMaskFormerObjectsTestMode:
    def test_test_object_index_byte_parity_vs_v1_opchain(self):
        """object_index == indices_from_mask(masks.sigmoid()>0.5) with padded -> -1."""
        b, batch = _batch_bundle()
        idx = b.get("outputs.tracks.object_index")
        masks, pad = batch["objects.masks"], batch["masks.tracks"]
        v1 = indices_from_mask(masks.sigmoid() > 0.5).int().numpy()
        v1 = np.where(~pad.numpy(), v1, -1)
        assert np.array_equal(idx.numpy(), v1)
        assert (idx == -1).any() and (idx == -2).any()  # padded + no-object sentinels

    def test_declare_io_test_produces_only_object_index(self):
        io = _node().declare_io(Mode.TEST)
        assert list(flatten_spec(io.produces)) == ["outputs.tracks.object_index"]
        assert set(flatten_spec(io.requires)) == {"objects.masks", "masks.tracks"}

    def test_declare_io_fit_val_declare_nothing(self):
        for mode in (Mode.FIT, Mode.VAL):
            io = _node().declare_io(mode)
            assert flatten_spec(io.requires) == {} and flatten_spec(io.produces) == {}

    def test_declare_io_onnx_leaves_unchanged(self):
        """The ONNX ports are byte-unchanged (the 4-leaf export set)."""
        io = _node().declare_io(Mode.ONNX)
        assert set(flatten_spec(io.produces)) == {
            "outputs.objects.leading_object",
            "outputs.tracks.object_index",
            "outputs.objects.vertices_class_probs",
            "outputs.objects.vertices_regression",
        }


# 2. the eval-H5 objects group byte parity vs the v1 op chain (the retired
# MaskFormerObjectWriter.write chain, predictionwriter.py:276-308)


class TestEvalH5ByteParity:
    def test_object_group_fragments_byte_parity(self):
        b, batch = _batch_bundle()
        frags = _sink()._object_group_fragments(b)
        assert set(frags) == {"objects", "object_masks", "tracks"}

        cp, masks = batch["objects.class_probs"], batch["objects.masks"]
        oc, tm, pad = (
            batch["labels.objects.object_class"],
            batch["labels.objects.masks"],
            batch["masks.tracks"],
        )
        # objects group: per-class probs + truth class
        v1_probs = u2s(cp.numpy(), np.dtype([(f"{RUN_NAME}_p{c}", "f4") for c in OBJECT_CLASSES]))
        for n in v1_probs.dtype.names:
            assert frags["objects"][n].tobytes() == v1_probs[n].tobytes()
        v1_class = u2s(oc.unsqueeze(-1).numpy(), np.dtype([("class_label", "i8")]))
        assert frags["objects"]["class_label"].tobytes() == v1_class["class_label"].tobytes()
        # tracks HadronIndex: indices_from_mask(sigmoid>0.5) (-2), padded -> -1
        v1_idx = indices_from_mask(masks.sigmoid() > 0.5).int().numpy()
        v1_idx = np.where(~pad.numpy(), v1_idx, -1)
        col = f"{RUN_NAME}_{INDEX_SUFFIX}"
        assert (
            frags["tracks"][col].tobytes()
            == u2s(np.expand_dims(v1_idx, -1), np.dtype([(col, "i8")]))[col].tobytes()
        )
        # object_masks group: truth mask + raw logits
        assert (
            frags["object_masks"]["truth_mask"].tobytes()
            == u2s(tm.unsqueeze(-1).numpy(), np.dtype([("truth_mask", "i8")]))["truth_mask"].tobytes()
        )
        assert (
            frags["object_masks"]["mask_logits"].tobytes()
            == u2s(masks.float().unsqueeze(-1).numpy(), np.dtype([("mask_logits", "f4")]))[
                "mask_logits"
            ].tobytes()
        )

    def test_column_schema(self):
        """The merged column dtypes/order match the v1 eval schema exactly."""
        dtypes, shapes = _sink()._merge_columns(  # noqa: SLF001
            ("jets", "tracks"), ("tracks",), {"jets": "jets", "tracks": "tracks"}, 64
        )
        assert dtypes["objects"].names == (
            f"{RUN_NAME}_pb", f"{RUN_NAME}_pc", f"{RUN_NAME}_pnull", "class_label",
        )
        assert dtypes["objects"][f"{RUN_NAME}_pb"] == _F4
        assert dtypes["objects"]["class_label"] == _I8
        assert dtypes["object_masks"].names == ("truth_mask", "mask_logits")
        assert dtypes["object_masks"]["truth_mask"] == _I8
        assert dtypes["object_masks"]["mask_logits"] == _F4
        assert f"{RUN_NAME}_{INDEX_SUFFIX}" in dtypes["tracks"].names
        assert shapes["objects"] == (64, _M)
        assert shapes["object_masks"] == (64, _M, _N_TRACKS)

    def test_half_precision_demotes_float_object_columns(self):
        dtypes, _ = _sink(half=True)._merge_columns(  # noqa: SLF001
            ("jets", "tracks"), ("tracks",), {"jets": "jets", "tracks": "tracks"}, 64
        )
        assert dtypes["objects"][f"{RUN_NAME}_pb"] == np.dtype("f2")
        assert dtypes["object_masks"]["mask_logits"] == np.dtype("f2")
        assert dtypes["object_masks"]["truth_mask"] == _I8  # int stays i8

    def test_sink_demands_all_object_leaves_in_test(self):
        req = flatten_spec(_sink().declare_io(Mode.TEST).requires)
        for leaf in (
            "objects.class_probs",
            "objects.masks",
            "labels.objects.object_class",
            "labels.objects.masks",
            "outputs.tracks.object_index",
        ):
            assert leaf in req, f"{leaf} not demanded (its producer would prune)"


# 3. ONNX tuple order: the model-graph object node's leaves AFTER the section block

_MF_REG_TARGETS = ("pt", "Lxy", "deta", "dphi", "mass")
_MF_LEADING_LEAF_KEY = "outputs.objects.leading_object"
_MF_LEADING_LEAF_NAMES = [f"leading_objects_{t}" for t in _MF_REG_TARGETS]
_MF_INDEX_LEAF_NAME = "HadronIndex"
_MF_INDEX_LEAF_KEY = f"outputs.tracks.{_MF_INDEX_LEAF_NAME}"

EXPECTED_MFV2_OUTPUT_NAMES = [
    "MFv2_pb",
    "MFv2_pc",
    "MFv2_pu",
    "MFv2_TrackOrigin",
    "MFv2_leading_objects_pt",
    "MFv2_leading_objects_Lxy",
    "MFv2_leading_objects_deta",
    "MFv2_leading_objects_dphi",
    "MFv2_leading_objects_mass",
    "MFv2_HadronIndex",
]


class TestMaskFormerOnnxTupleOrder:
    """The MaskFormer object node's leaves appear AFTER the section block."""

    def _onnx_sink(self, tmp_path):
        from types import SimpleNamespace  # noqa: PLC0415 - test-local

        from salt.outputs import OnnxExportSink  # noqa: PLC0415 - test-local
        from salt.outputs.run_task_output import RunTaskOutput  # noqa: PLC0415 - test-local
        from salt.tests._fixtures.gn2v2_fixture import (  # noqa: PLC0415 - test-local
            build_gn2v2_modules,
            write_parity_norm_dict,
        )

        nd_path = tmp_path / "norm_dict.yaml"
        cd_path = tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd_path, cd_path)
        modules = build_gn2v2_modules(nd_path)
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin"])
        rt.name = "run_tasks"
        rt.bind_model_modules(modules)
        # the object node names its own leaves off the regression task's targets
        modules["regression"] = SimpleNamespace(targets=_MF_REG_TARGETS)
        mf = MaskFormerObjects(n_reg=len(_MF_REG_TARGETS), index_name=_MF_INDEX_LEAF_NAME)
        mf.name = "maskformer_objects_onnx"
        modules["maskformer_objects_onnx"] = mf
        sink = OnnxExportSink(model_name="MFv2")
        sink.bind_output_section({"run_tasks": rt})
        sink.bind_model_modules(modules)
        mf.bind_model_modules(modules)
        return sink

    def test_leading_names_derive_from_the_regression_targets(self, tmp_path):
        """The leading-object suffixes are minted per target, not typed in the config."""
        leaves = {leaf.key: leaf for leaf in self._onnx_sink(tmp_path).leaves}
        assert list(leaves[_MF_LEADING_LEAF_KEY].names) == _MF_LEADING_LEAF_NAMES
        assert leaves[_MF_INDEX_LEAF_KEY].dtype == "int8"
        assert leaves[_MF_INDEX_LEAF_KEY].per_token is True

    def test_output_names_full_ordered_list(self, tmp_path):
        assert self._onnx_sink(tmp_path).output_names() == EXPECTED_MFV2_OUTPUT_NAMES

    def test_trackorigin_before_leading_object_globals(self, tmp_path):
        names = self._onnx_sink(tmp_path).output_names()
        assert names.index("MFv2_TrackOrigin") < names.index("MFv2_leading_objects_pt")

    def test_hadron_index_last(self, tmp_path):
        assert self._onnx_sink(tmp_path).output_names()[-1] == "MFv2_HadronIndex"
