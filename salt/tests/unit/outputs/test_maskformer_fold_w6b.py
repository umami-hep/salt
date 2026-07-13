"""The MaskFormer objects sink on the H5OutputSink ``extra_groups`` seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salt.core.graph.bundle import Bundle
from salt.core.outputs import MaskFormerObjectsSink
from salt.core.outputs.h5_sink import _ExtraGroupCtx
from salt.core.outputs.maskformer import MaskFormerObjectWriter
from salt.core.outputs.names import OBJECT_INDEX
from salt.tests._fixtures.writers_common import (  # noqa: F401  (pytest fixtures)
    L_FILE,
    data,
)

pytestmark = pytest.mark.cpu_always

# the v2 object classes (null LAST, quoted) — the MaskFormer_pb/_pc/_pnull
# object-class columns.
OBJECT_CLASSES = ["b", "c", "null"]
# the run name == the model name (the {run_name}_ column prefix).
RUN_NAME = "MaskFormer"
# the upstream group name the v2 "objects" group deliberately diverges from
# (recorded decision 2026-06-30; upstream equivalence closed at pin 6570e85,
# see the parity-closure section of salt/core/README.md).
UPSTREAM_OBJECT_GROUP = "truth_hadrons"
# the deferred per-object regression eval columns — KNOWN-ABSENT from v2's objects
# group (no v2 writer emits per-object regression in TEST yet; MaskFormer.yaml).
DEFERRED_REGRESSION_COLUMNS = (
    "MaskFormer_pt",
    "MaskFormer_Lxy",
    "MaskFormer_deta",
    "MaskFormer_dphi",
    "MaskFormer_mass",
)
_F4, _I8 = np.dtype("f4"), np.dtype("i8")
_M_MAX_OBJECTS = 5  # the builder modules' max_objects (query bank size)


def _mf_modules(data):  # noqa: ANN001
    from salt.tests._fixtures.v2_builders import build_maskformer_writer_modules

    return build_maskformer_writer_modules(data["nd"])


def _sink(data) -> MaskFormerObjectsSink:  # noqa: ANN001
    sink = MaskFormerObjectsSink(
        object_classes=OBJECT_CLASSES, object_stream="objects", constituent_stream="tracks"
    )
    sink.bind_model_modules(_mf_modules(data))
    return sink


def _ctx(precision: str = "full") -> _ExtraGroupCtx:
    return _ExtraGroupCtx(
        streams=("jets", "tracks"),
        seq_lengths={"tracks": L_FILE},
        total=64,
        run_name=RUN_NAME,
        precision=precision,
    )


# 1. the sink's H5 schema (upstream equivalence closed at pin 6570e85 —
# expectations are asserted from first principles, not a vendored schema)


class TestSinkSchema:
    def test_objects_group_class_columns(self, data):
        """objects: MaskFormer_pb/_pc/_pnull (f4) + class_label (i8)."""
        cols = _sink(data).columns(_ctx())
        objects = cols["objects"]
        assert list(objects.names) == ["MaskFormer_pb", "MaskFormer_pc", "MaskFormer_pnull",
                                       "class_label"]
        assert objects["MaskFormer_pb"] == _F4
        assert objects["MaskFormer_pc"] == _F4
        assert objects["MaskFormer_pnull"] == _F4
        assert objects["class_label"] == _I8

    def test_object_masks_group_columns(self, data):
        """object_masks: truth_mask (i8) + mask_logits (f4) — exact names and dtypes."""
        cols = _sink(data).columns(_ctx())
        om = cols["object_masks"]
        assert list(om.names) == ["truth_mask", "mask_logits"]
        assert om["truth_mask"] == _I8
        assert om["mask_logits"] == _F4

    def test_maskindex_on_tracks(self, data):
        """MaskFormer_MaskIndex (i8) rides the tracks reader stream (PINNED suffix)."""
        cols = _sink(data).columns(_ctx())
        tracks = cols["tracks"]
        index_col = f"{RUN_NAME}_{OBJECT_INDEX.test}"  # MaskFormer_MaskIndex
        assert list(tracks.names) == [index_col]
        assert tracks[index_col] == _I8

    def test_extra_group_shapes_BM_and_BMT(self, data):
        """extra_groups sizes objects (M,) and object_masks (M, T) — the [B, M] compactness."""
        groups = _sink(data).extra_groups(_ctx())
        assert groups == {
            "objects": (_M_MAX_OBJECTS,),
            "object_masks": (_M_MAX_OBJECTS, L_FILE),
        }


class TestRecordedDivergences:
    """The honest assert-as-known allowlist: the divergences are intentional, on record."""

    def test_v2_group_name_is_objects_not_truth_hadrons(self, data):
        """(a) v2 emits the object group as 'objects', NOT 'truth_hadrons' (user decision)."""
        groups = _sink(data).extra_groups(_ctx())
        assert "objects" in groups
        # the upstream writer carries the object preds under truth_hadrons; the
        # v2 group name is a recorded, intentional divergence (merge deferred).
        assert UPSTREAM_OBJECT_GROUP not in groups

    def test_deferred_regression_columns_known_absent(self, data):
        """(b) the per-object regression eval columns are KNOWN-ABSENT from v2 objects."""
        objects = _sink(data).columns(_ctx())["objects"]
        for col in DEFERRED_REGRESSION_COLUMNS:
            assert col not in objects.names, f"{col} unexpectedly emitted (deferral broken)"


# 2. byte parity vs the delegated MaskFormerObjectWriter (drift-proof delegation)


def _bundle(batch_size=6, n_tracks=10):
    from salt.tests._fixtures.v2_builders import make_maskformer_writer_batch

    batch = make_maskformer_writer_batch(batch_size=batch_size, n_tracks=n_tracks)
    bundle = Bundle()
    for key, value in batch.items():
        bundle.set(key, value)
    return bundle


def _legacy_writer(data):  # noqa: ANN001
    from salt.tests.unit.writers.test_maskformer import mf_write_ctx

    w = MaskFormerObjectWriter(object_classes=OBJECT_CLASSES, regression_task="regression")
    w.name = "object_writer"
    w.setup(mf_write_ctx(data, _mf_modules(data), n_tracks=10, total=6))
    return w


class TestByteParityVsDelegatedWriter:
    def test_write_byte_identical_to_writer(self, data):
        """The sink node's write() is byte-identical to MaskFormerObjectWriter.write()."""
        bundle = _bundle()
        rows = slice(0, 6)
        legacy = _legacy_writer(data).write(bundle, rows)
        sink = _sink(data).write(bundle, rows, run_name="MFrun", precision="full")
        assert set(sink) == set(legacy) == {"objects", "tracks", "object_masks"}
        for group in legacy:
            assert sink[group].dtype == legacy[group].dtype, f"{group} dtype drift"
            for col in legacy[group].dtype.names:
                assert sink[group][col].tobytes() == legacy[group][col].tobytes(), (
                    f"{group}.{col} bytes drift from MaskFormerObjectWriter"
                )
        # the -2 (no-object) / -1 (padded) MaskIndex sentinels are both present
        idx = sink["tracks"][f"MFrun_{OBJECT_INDEX.test}"]
        assert (idx == -1).any() and (idx == -2).any()

    def test_columns_byte_identical_to_writer(self, data):
        """columns() is byte-identical to the delegated writer's (same group dtypes)."""
        ctx = _ctx()
        legacy = _legacy_writer(data).columns(ctx)
        sink = _sink(data).columns(ctx)
        assert set(sink) == set(legacy)
        for group in legacy:
            assert sink[group] == legacy[group]


# 3. ONNX tuple order: explicit object leaves AFTER the section block

# The MaskFormer explicit leaves (the two object reduces that the
# outputs: section cannot mint). In MaskFormer.yaml emission order:
_MF_LEADING_LEAF_KEY = "outputs.objects.leading_object"
_MF_LEADING_LEAF_NAMES = [
    "leading_objects_pt",
    "leading_objects_Lxy",
    "leading_objects_deta",
    "leading_objects_dphi",
    "leading_objects_mass",
]
_MF_INDEX_LEAF_KEY = "outputs.tracks.object_index"
_MF_INDEX_LEAF_NAME = "HadronIndex"

# The PINNED full ordered output_names for the MaskFormer OnnxExportSink with
# model_name="MFv2", section tasks=[jets_classification, track_origin], and the
# two explicit MF object leaves (leading_object + object_index). Hand-derived
# from the v1 manifest / writer order: 1:1 head section leaves first (globals
# pb/pc/pu then per-token TrackOrigin), THEN the object-reduce explicit leaves
# (global split_scalars leading_objects_* then per-token HadronIndex).
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
    """Explicit MaskFormer object leaves appear AFTER the section block."""

    def _onnx_sink(self, tmp_path: Path):
        from salt.core.outputs import OnnxExportLeaf, OnnxExportSink
        from salt.core.outputs.run_task_output import RunTaskOutput
        from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
        from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

        nd_path = tmp_path / "norm_dict.yaml"
        cd_path = tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd_path, cd_path)
        modules = build_gn2v2_modules(nd_path)
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin"])
        rt.name = "run_tasks"
        rt.bind_model_modules(modules)
        # The two explicit MaskFormer object leaves in MaskFormer.yaml order.
        explicit_leaves = [
            OnnxExportLeaf(key=_MF_LEADING_LEAF_KEY, names=_MF_LEADING_LEAF_NAMES),
            OnnxExportLeaf(
                key=_MF_INDEX_LEAF_KEY,
                name=_MF_INDEX_LEAF_NAME,
                dtype="int8",
                per_token=True,
            ),
        ]
        sink = OnnxExportSink(outputs=explicit_leaves, model_name="MFv2")
        sink.bind_output_section({"run_tasks": rt})
        return sink

    def test_output_names_full_ordered_list(self, tmp_path):
        """The full flat output_names list equals EXPECTED_MFV2_OUTPUT_NAMES exactly."""
        sink = self._onnx_sink(tmp_path)
        assert sink.output_names() == EXPECTED_MFV2_OUTPUT_NAMES

    def test_trackorigin_before_leading_object_globals(self, tmp_path):
        """TrackOrigin (section per-token) comes BEFORE leading_objects_* (explicit globals)."""
        names = self._onnx_sink(tmp_path).output_names()
        to_idx = names.index("MFv2_TrackOrigin")
        lo_idx = names.index("MFv2_leading_objects_pt")
        assert to_idx < lo_idx, (
            f"TrackOrigin at {to_idx} must precede leading_objects_pt at {lo_idx}"
        )

    def test_hadron_index_last(self, tmp_path):
        """HadronIndex (explicit per-token) is the last element of the output tuple."""
        names = self._onnx_sink(tmp_path).output_names()
        assert names[-1] == "MFv2_HadronIndex"


# 4. object_masks shape guard: mismatch raises ConfigError loudly


class TestObjectMasksShapeGuard:
    """_extra_group_fragments rejects a fragment with wrong per-row shape."""

    _M = 5    # num objects
    _T = 40   # T_file (full file constituent width)

    def _sink_with_state(self, extra_shapes=None):
        """Minimal H5OutputSink with _extra_shapes and a stub extra-group node set up."""
        from salt.core.outputs import H5OutputSink

        sink = H5OutputSink(extra_groups=["mf_objects"])
        # Set the internal state that open_schema would normally populate.
        sink._extra_shapes = extra_shapes or {"object_masks": (self._M, self._T)}  # noqa: SLF001
        sink._seq_lengths = {"tracks": self._T}  # noqa: SLF001
        sink._run_name = "MFv2"  # noqa: SLF001
        return sink

    def _stub_node(self, arr_shape):
        """A minimal extra-group node stub whose write() returns a fixed-shape array."""
        dtype = np.dtype([("truth_mask", "i8"), ("mask_logits", "f4")])
        arr = np.zeros(arr_shape, dtype=dtype)

        class _Stub:
            def write(self, bundle, rows, run_name, precision):  # noqa: ANN001, ARG002
                return {"object_masks": arr}

        return _Stub()

    def _sink_and_node(self, arr_shape):
        sink = self._sink_with_state()
        stub = self._stub_node(arr_shape)
        sink._output_section = {"mf_objects": stub}  # noqa: SLF001
        return sink

    def test_truncated_constituent_axis_raises_config_error(self):
        """object_masks with T_model < T_file raises ConfigError (not an h5py broadcast crash)."""
        from salt.core.graph.errors import ConfigError

        B, T_model = 4, 30  # T_model < T_file=40 — a truncated constituent axis
        sink = self._sink_and_node(arr_shape=(B, self._M, T_model))
        bundle = Bundle()
        with pytest.raises(ConfigError, match="object_masks"):
            sink._extra_group_fragments(bundle, slice(0, B))  # noqa: SLF001

    def test_error_message_names_declared_and_actual_shape(self):
        """The ConfigError message includes both the declared and actual per-row shapes."""
        from salt.core.graph.errors import ConfigError

        B, T_model = 4, 30
        sink = self._sink_and_node(arr_shape=(B, self._M, T_model))
        bundle = Bundle()
        with pytest.raises(ConfigError) as exc_info:
            sink._extra_group_fragments(bundle, slice(0, B))  # noqa: SLF001
        msg = str(exc_info.value)
        # The error names the declared shape (M, T_file) and the actual shape (M, T_model).
        assert f"({self._M}, {self._T})" in msg, f"declared shape not in: {msg}"
        assert f"({self._M}, {T_model})" in msg, f"actual shape not in: {msg}"

    def test_shipped_maskformer_yaml_width_does_not_raise(self):
        """The shipped MaskFormer.yaml case (T_model == T_file == 40) does NOT raise."""
        B = 4
        sink = self._sink_and_node(arr_shape=(B, self._M, self._T))  # T_model == T_file
        bundle = Bundle()
        result = sink._extra_group_fragments(bundle, slice(0, B))  # noqa: SLF001
        assert "object_masks" in result
