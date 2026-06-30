"""W6b — the MaskFormer object writer FOLD onto the H5OutputSink ``extra_groups`` seam.

W6b relocates the legacy `salt.core.writers.MaskFormerObjectWriter` TEST role onto
`salt.core.outputs.MaskFormerObjectsSink` — a manifest-only ``outputs:``-section
node hosted by `H5OutputSink` via ``extra_groups`` (the W6a seam). The
object-prediction group is emitted as the v2-native **``objects``** group (named by
``object_stream``), NOT merged into the ``truth_hadrons`` stream group (USER
DECISION 2026-06-30); the per-object regression eval columns stay DEFERRED.

This is the AUTOMATED parity gate. Two halves:

1. **schema parity vs the upstream 6570e85 fixture** — the migrated sink's emitted
   columns (the v2-owned ones) match the fixture's field NAMES + DTYPES:
   - ``objects`` group: ``MaskFormer_pb/_pc/_pnull`` (f4) + ``class_label`` (i8),
     compared against the fixture's ``truth_hadrons`` object-class fields;
   - ``object_masks`` group: ``truth_mask`` (i8) + ``mask_logits`` (f4) — an exact
     name/dtype match of the fixture's ``object_masks`` group;
   - ``MaskFormer_MaskIndex`` (i8) on the ``tracks`` reader stream.
   The gate RECORDS, as assert-as-known divergences (so it is honest, not vacuous):
   (a) the v2 group name ``objects`` diverges from the fixture's ``truth_hadrons``
   (intentional, the user decision), and (b) the deferred ``MaskFormer_<regression>``
   columns (``MaskFormer_pt/_Lxy/_deta/_dphi/_mass``) are known-ABSENT from the v2
   ``objects`` group.

2. **byte parity vs MFU-6's writer** — the sink node's ``write`` is byte-identical
   to `MaskFormerObjectWriter.write` on the SAME bundle (drift-proof: the sink node
   delegates to an internal writer instance, so this pins that delegation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from salt.core.graph.bundle import Bundle
from salt.core.outputs import MaskFormerObjectsSink
from salt.core.outputs.sinks import _ExtraGroupCtx
from salt.core.writers import OBJECT_INDEX, MaskFormerObjectWriter
from salt.tests._fixtures.writers_common import (  # noqa: F401  (pytest fixtures)
    L_FILE,
    data,
)

pytestmark = pytest.mark.cpu_always

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "_fixtures"
    / "mf_writer_parity"
    / "upstream_6570e85_schema.json"
)

# the v2 object classes (v1 object.class_names; null LAST, quoted), == the
# fixture's MaskFormer_pb/_pc/_pnull object-class columns.
OBJECT_CLASSES = ["b", "c", "null"]
# the run name == the upstream model name (the {run_name}_ column prefix).
RUN_NAME = "MaskFormer"
# the upstream group the v2 "objects" group diverges from (USER DECISION 2026-06-30).
UPSTREAM_OBJECT_GROUP = "truth_hadrons"
# the deferred per-object regression eval columns — KNOWN-ABSENT from v2's objects
# group (no v2 writer emits per-object regression in TEST yet; MaskFormer.yaml :197).
DEFERRED_REGRESSION_COLUMNS = (
    "MaskFormer_pt",
    "MaskFormer_Lxy",
    "MaskFormer_deta",
    "MaskFormer_dphi",
    "MaskFormer_mass",
)
# numpy descriptor <- fixture dtype string.
_NP = {"float32": np.dtype("f4"), "int64": np.dtype("i8")}


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _mf_modules(data):  # noqa: ANN001
    from salt.tests._fixtures.regression_fixture import build_maskformer_writer_modules

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


# ---------------------------------------------------------------------------
# 1. schema parity vs the upstream 6570e85 fixture (the v2-owned columns)
# ---------------------------------------------------------------------------


class TestSchemaParityVsFixture:
    def test_objects_group_class_columns_match_fixture(self, data):
        """objects: MaskFormer_pb/_pc/_pnull (f4) + class_label (i8) == fixture object-class fields.

        RECORDED divergence (a): v2 emits these under "objects", the fixture under
        "truth_hadrons" — intentional (USER DECISION 2026-06-30). The gate compares
        the COLUMN names/dtypes the writer owns, NOT the group name.
        """
        cols = _sink(data).columns(_ctx())
        objects = cols["objects"]
        assert list(objects.names) == ["MaskFormer_pb", "MaskFormer_pc", "MaskFormer_pnull",
                                       "class_label"]
        assert objects["MaskFormer_pb"] == _NP["float32"]
        assert objects["MaskFormer_pc"] == _NP["float32"]
        assert objects["MaskFormer_pnull"] == _NP["float32"]
        assert objects["class_label"] == _NP["int64"]

        # the SAME field names/dtypes live in the fixture's truth_hadrons group
        fx = _fixture()["groups"][UPSTREAM_OBJECT_GROUP]["fields"]
        for col in ("MaskFormer_pb", "MaskFormer_pc", "MaskFormer_pnull", "class_label"):
            assert col in fx, f"{col} missing from upstream {UPSTREAM_OBJECT_GROUP}"
            assert objects[col] == _NP[fx[col]], f"{col} dtype diverges from upstream"

    def test_object_masks_group_exact_match_fixture(self, data):
        """object_masks: truth_mask (i8) + mask_logits (f4) — EXACT name/dtype match of fixture."""
        cols = _sink(data).columns(_ctx())
        om = cols["object_masks"]
        assert list(om.names) == ["truth_mask", "mask_logits"]
        fx = _fixture()["groups"]["object_masks"]["fields"]
        assert list(fx) == ["truth_mask", "mask_logits"]  # same group name AND fields upstream
        for col in om.names:
            assert om[col] == _NP[fx[col]]

    def test_maskindex_on_tracks_match_fixture(self, data):
        """MaskFormer_MaskIndex (i8) rides the tracks reader stream — matches fixture tracks."""
        cols = _sink(data).columns(_ctx())
        tracks = cols["tracks"]
        index_col = f"{RUN_NAME}_{OBJECT_INDEX.test}"  # MaskFormer_MaskIndex (PINNED suffix)
        assert list(tracks.names) == [index_col]
        assert tracks[index_col] == _NP["int64"]
        fx = _fixture()["groups"]["tracks"]["fields"]
        assert index_col in fx and _NP[fx[index_col]] == np.dtype("i8")

    def test_extra_group_shapes_BM_and_BMT(self, data):
        """extra_groups sizes objects (M,) and object_masks (M, T) — the [B, 5] compactness."""
        groups = _sink(data).extra_groups(_ctx())
        m = _fixture()["dims"]["M_max_objects"]  # 5
        t = _fixture()["dims"]["C_max_constituents"]  # 40 == L_FILE
        assert groups == {"objects": (m,), "object_masks": (m, t)}


class TestRecordedDivergences:
    """The honest assert-as-known allowlist: the divergences are intentional, on record."""

    def test_v2_group_name_diverges_from_truth_hadrons(self, data):
        """(a) v2 emits the object group as 'objects', NOT 'truth_hadrons' (user decision)."""
        groups = _sink(data).extra_groups(_ctx())
        assert "objects" in groups
        assert UPSTREAM_OBJECT_GROUP not in groups  # NOT merged into truth_hadrons (DEFERRED)
        # the upstream fixture DOES carry the object preds under truth_hadrons —
        # this is the recorded, intentional group-name divergence.
        assert UPSTREAM_OBJECT_GROUP in _fixture()["groups"]

    def test_deferred_regression_columns_known_absent(self, data):
        """(b) the per-object regression eval columns are KNOWN-ABSENT from v2 objects.

        Upstream's truth_hadrons group carries MaskFormer_pt/_Lxy/_deta/_dphi/_mass;
        v2 DELIBERATELY defers per-object regression eval columns (MaskFormer.yaml
        :197-211). The gate asserts they are absent from v2 AND present upstream, so
        the deferral is recorded, not silently lost.
        """
        objects = _sink(data).columns(_ctx())["objects"]
        fx = _fixture()["groups"][UPSTREAM_OBJECT_GROUP]["fields"]
        for col in DEFERRED_REGRESSION_COLUMNS:
            assert col not in objects.names, f"{col} unexpectedly emitted (deferral broken)"
            assert col in fx, f"{col} not in upstream fixture — stale deferral allowlist"


# ---------------------------------------------------------------------------
# 2. byte parity vs MFU-6's MaskFormerObjectWriter (drift-proof delegation)
# ---------------------------------------------------------------------------


def _bundle(batch_size=6, n_tracks=10):
    from salt.tests._fixtures.regression_fixture import make_maskformer_writer_batch

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


class TestByteParityVsLegacyWriter:
    def test_write_byte_identical_to_mfu6_writer(self, data):
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
                    f"{group}.{col} bytes drift from MFU-6 writer"
                )
        # the -2 (no-object) / -1 (padded) MaskIndex sentinels are both present
        idx = sink["tracks"][f"MFrun_{OBJECT_INDEX.test}"]
        assert (idx == -1).any() and (idx == -2).any()

    def test_columns_byte_identical_to_mfu6_writer(self, data):
        """columns() is byte-identical to the legacy writer's (same group dtypes)."""
        ctx = _ctx()
        legacy = _legacy_writer(data).columns(ctx)
        sink = _sink(data).columns(ctx)
        assert set(sink) == set(legacy)
        for group in legacy:
            assert sink[group] == legacy[group]
