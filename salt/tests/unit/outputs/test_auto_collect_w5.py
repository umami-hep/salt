"""W5.1 unit gate — sink auto-collect resolution + the static dup-name guard.

Plan 31 W5.1: the H5 and ONNX sinks gain an AUTO-COLLECT mode (omitted
``outputs:``) that discovers the active conversion producers from the model
module dict and assembles each per-stream H5 group / the ONNX Athena tuple from
their `output_columns()` field manifest — with a STATIC HARD dup-name guard.

This file covers the RESOLUTION + the dup guard at the unit level (no run):

- H5 auto-collect resolves the SAME `OutputColumn` table the explicit form lists,
  in the same order, for the cutover producer set.
- ONNX auto-collect resolves the SAME leaf set + ORDER (globals -> combines ->
  per-token aux) as the explicit form for a folded GN2v2-style producer set.
- the dup-name guard fires at resolve (compile) time, before any run, for two
  producers minting the same flat column.

The end-to-end H5 auto==explicit + ONNX auto==golden gates run in the integration
suite (`test_auto_collect_h5_parity.py` / the ONNX fixture gate).
"""

from __future__ import annotations

import pytest

from salt.core.graph.errors import ConfigError
from salt.core.nn.tasks import ClassificationTaskModule, VertexingTaskModule
from salt.core.outputs import (
    ClassProbs,
    Combination,
    H5OutputSink,
    OnnxExportSink,
    SeqClassIndex,
    SeqClassProbs,
    VertexUnionFind,
)


def _named(mods: dict) -> dict:
    for name, module in mods.items():
        module.name = name
    return mods


def _cutover_modules() -> dict:
    """The gn2v2-dummy-cutover producer set: jet ClassProbs + track SeqClassProbs."""
    return _named({
        "jets_classification": ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
            input="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks", label="ftagTruthOriginLabel",
            class_names=["Pileup", "Fake", "Primary", "FromB", "FromBC", "FromC",
                         "FromTau", "OtherSecondary"],
            context="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "jet_probs": ClassProbs(task="jets_classification", stream="jets"),
        "track_origin_probs": SeqClassProbs(task="track_origin", stream="tracks"),
    })


# ---------------------------------------------------------------------------
# H5 auto-collect resolution == explicit columns (the cutover reference)
# ---------------------------------------------------------------------------


def test_h5_auto_collect_resolves_cutover_columns():
    """Auto-collect produces the EXACT column table the explicit cutover form lists."""
    mods = _cutover_modules()
    sink = H5OutputSink(write_pad_mask=["tracks"])  # auto-collect (no `outputs`)
    sink.bind_model_modules(mods)
    cols = sink._resolve_columns("GN2v2_dummy")  # noqa: SLF001
    # one OutputColumn per producer leaf, in module declaration order
    assert [c.key for c in cols] == [
        "outputs.jets.jets_classification",
        "outputs.tracks.track_origin",
    ]
    assert list(cols[0].suffixes) == ["pb", "pc", "pu"]
    assert list(cols[1].suffixes) == [
        "pPileup", "pFake", "pPrimary", "pFromB", "pFromBC", "pFromC", "pFromTau",
        "pOtherSecondary",
    ]
    # the flat H5 column names match the explicit reference exactly
    assert cols[0].column_names("GN2v2_dummy") == ["GN2v2_dummy_pb", "GN2v2_dummy_pc",
                                                   "GN2v2_dummy_pu"]
    # the per-token mask stream is discovered from the producer streams
    assert sink._pad_mask_streams() == ("tracks",)  # noqa: SLF001


def test_h5_auto_collect_collections_filter():
    """`collections` narrows auto-collect to the named streams."""
    mods = _cutover_modules()
    sink = H5OutputSink(collections=["jets"])  # only the jets group
    sink.bind_model_modules(mods)
    cols = sink._resolve_columns("R")  # noqa: SLF001
    assert [c.key for c in cols] == ["outputs.jets.jets_classification"]


def test_h5_auto_collect_dup_name_guard_fires_at_compile():
    """Two producers minting the same flat H5 column -> ConfigError at resolve (pre-run)."""
    mods = _cutover_modules()
    # a SECOND jet-stream ClassProbs on a task with the SAME class suffixes (pb/pc/pu)
    mods["jets_classification_2"] = ClassificationTaskModule(
        stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
        input="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
    )
    mods["jets_classification_2"].name = "jets_classification_2"
    dup = ClassProbs(task="jets_classification_2", stream="jets", name="dup_probs")
    dup.name = "dup_probs"
    mods["dup_probs"] = dup
    sink = H5OutputSink()
    sink.bind_model_modules(mods)
    with pytest.raises(ConfigError, match="static dup-name guard"):
        sink._resolve_columns("R")  # noqa: SLF001


def test_h5_auto_collect_no_producers_errors():
    """Auto-collect with no conversion producers raises an actionable error."""
    mods = _named({
        "jets_classification": ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets"],
            input="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
    })
    sink = H5OutputSink()
    sink.bind_model_modules(mods)
    with pytest.raises(ConfigError, match="no producer with a final H5 output column"):
        sink._resolve_columns("R")  # noqa: SLF001


# ---------------------------------------------------------------------------
# ONNX auto-collect resolution == explicit leaf set + ORDER
# ---------------------------------------------------------------------------


def _onnx_modules() -> dict:
    """A GN2v2-style folded ONNX producer set: jet probs + track argmax + vertexing + combine."""
    mods = _named({
        "jets_classification": ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
            input="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks", label="ftagTruthOriginLabel",
            class_names=["Pileup", "Fake", "Primary", "FromB"],
            context="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "track_vertexing": VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="ftagTruthOriginLabel",
            context="pooled.global", dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "jet_probs": ClassProbs(task="jets_classification", stream="jets"),
        "track_origin_index": SeqClassIndex(task="track_origin", stream="tracks"),
        "track_vertex_index": VertexUnionFind(task="track_vertexing", stream="tracks"),
        "pbc": Combination(source="outputs.jets.jets_classification", name="pbc",
                           terms={0: 1.0, 1: 1.0}),
    })
    return mods


def test_onnx_auto_collect_ordered_tuple():
    """ONNX auto-collect orders the tuple: globals -> combines -> per-token aux (§6.3)."""
    mods = _onnx_modules()
    sink = OnnxExportSink(model_name="GN2v2")
    sink.bind_model_modules(mods)
    leaves = sink._ensure_leaves()  # noqa: SLF001
    # globals (jet probs split_scalars) first, then the combination, then per-token aux
    assert [leaf.key for leaf in leaves] == [
        "outputs.jets.jets_classification",  # global float split_scalars
        "outputs.jets.pbc",                  # combination
        "outputs.tracks.track_origin",       # per-token argmax index
        "outputs.tracks.track_vertexing",    # per-token union-find index
    ]
    # the flat Athena names, in tuple order, prefixed with model_name
    assert sink.output_names() == [
        "GN2v2_pb", "GN2v2_pc", "GN2v2_pu",  # split_scalars globals
        "GN2v2_pbc",                          # combine
        "GN2v2_TrackOrigin",                  # argmax (pascal_case)
        "GN2v2_VertexIndex",                  # union-find (VERTEX_INDEX)
    ]
    assert sink.output_dtypes() == [
        "float32", "float32", "float32", "float32", "int8", "int8",
    ]
    # per-token leaves carry a dynamic axis; globals/combines do not
    axes = sink.dynamic_axes()
    assert set(axes) == {"GN2v2_TrackOrigin", "GN2v2_VertexIndex"}
    assert axes["GN2v2_TrackOrigin"] == {0: "n_tracks"}


def test_onnx_auto_collect_dup_name_guard():
    """Two ONNX producers minting the same flat suffix -> ConfigError at resolve.

    A combination named ``pb`` (a DISTINCT output leaf ``outputs.jets.pb``) collides
    with the jet `ClassProbs` split scalar ``pb`` — the flat Athena namespace dup
    guard fires across producers, at resolve (compile) time, before any trace.
    """
    mods = _onnx_modules()
    dup = Combination(source="outputs.jets.jets_classification", name="pb", terms={0: 1.0})
    dup.name = "pb_combine"
    mods["pb_combine"] = dup
    sink = OnnxExportSink(model_name="GN2v2")
    sink.bind_model_modules(mods)
    with pytest.raises(ConfigError, match="duplicate flat ONNX output name"):
        sink._ensure_leaves()  # noqa: SLF001
