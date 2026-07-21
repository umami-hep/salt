"""Unit gates for the `OnnxExportSink` declare-only terminal node (design §4.2)."""

from __future__ import annotations

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, SinkModule, flatten_spec
from salt.outputs import OnnxExportLeaf, OnnxExportSink

_JET = "outputs.jets.jets_classification"
_PBC = "outputs.jets.pbc"
_TRK = "outputs.tracks.track_origin"


def _sink(model_name="GN2v2"):
    """A representative folded export sink: split_scalars + a combine + a per-token int8 leaf."""
    return OnnxExportSink(
        outputs=[
            OnnxExportLeaf(key=_JET, names=["pb", "pc", "pu"]),
            OnnxExportLeaf(key=_PBC, name="pbc"),
            OnnxExportLeaf(key=_TRK, name="TrackOrigin", dtype="int8", per_token=True),
        ],
        model_name=model_name,
    )


# the SinkModule marker + declare_io


def test_onnx_export_sink_is_a_sink_module():
    """`OnnxExportSink` satisfies the `SinkModule` Protocol (``is_sink() -> True``)."""
    sink = _sink()
    assert isinstance(sink, SinkModule)
    assert sink.is_sink() is True


def test_declare_io_onnx_requires_leaves_empty_produces():
    """ONNX requires the conversion leaves (kind=data, dtype unconstrained); empty produces."""
    sink = _sink()
    io = sink.declare_io(Mode.ONNX)
    req = flatten_spec(io.requires)
    assert list(req) == [_JET, _PBC, _TRK]
    assert all(spec.kind == "data" for spec in req.values())
    # dtype is unconstrained — the sink consumes whatever the conversion emits
    assert all(spec.dtype is None for spec in req.values())
    assert flatten_spec(io.produces) == {}


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.TEST])
def test_declare_io_empty_outside_onnx(mode):
    """In FIT/VAL/TEST the sink declares nothing — the planner prunes it (plan_hash safe)."""
    sink = _sink()
    io = sink.declare_io(mode)
    assert flatten_spec(io.requires) == {}
    assert flatten_spec(io.produces) == {}


# generated metadata: output_names / dtypes / dynamic_axes (design §6.3)


def test_output_names_are_in_declared_tuple_order():
    """The export node's leaf list IS the Athena tuple order — split expands, combine in place."""
    sink = _sink()
    assert sink.output_names() == [
        "GN2v2_pb",
        "GN2v2_pc",
        "GN2v2_pu",  # split_scalars: one leaf -> 3 names
        "GN2v2_pbc",  # the combination leaf, at its declared position
        "GN2v2_TrackOrigin",  # the per-token aux leaf, last
    ]


def test_output_dtypes_align_with_names():
    """Per-output dtypes are 1:1 with the names (floats for split/combine, int8 for the index)."""
    sink = _sink()
    assert sink.output_dtypes() == ["float32", "float32", "float32", "float32", "int8"]
    assert len(sink.output_dtypes()) == len(sink.output_names())


def test_dynamic_axes_only_for_per_token_leaves():
    """Only the per-token int8 leaf registers a dynamic axis; globals/combines carry none."""
    sink = _sink()
    assert sink.dynamic_axes() == {"GN2v2_TrackOrigin": {0: "n_tracks"}}


def test_dynamic_axis_default_and_override():
    """The per-token dynamic axis defaults to ``n_<stream>`` and is overridable."""
    default = OnnxExportSink(
        outputs=[OnnxExportLeaf(key=_TRK, name="T", dtype="int8", per_token=True)],
        model_name="M",
    )
    assert default.dynamic_axes() == {"M_T": {0: "n_tracks"}}
    override = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(key=_TRK, name="T", dtype="int8", per_token=True, dyn_axis="n_custom")
        ],
        model_name="M",
    )
    assert override.dynamic_axes() == {"M_T": {0: "n_custom"}}


def test_model_name_required_for_names():
    """Deriving the Athena names needs a model_name (config or adapter-supplied)."""
    sink = OnnxExportSink(outputs=[OnnxExportLeaf(key=_JET, names=["pb", "pc", "pu"])])
    with pytest.raises(ConfigError, match="no model_name"):
        sink.output_names()
    sink.model_name = "Late"
    assert sink.output_names() == ["Late_pb", "Late_pc", "Late_pu"]


# named_outputs: the split realisation (no math), pass-through (design §6.2)


def test_named_outputs_splits_probs_and_passes_through_singles():
    """``named_outputs`` splits the probs leaf into named scalars; single leaves pass through."""
    sink = _sink()
    probs = torch.rand(1, 3)
    pbc = torch.rand(1)
    trk = torch.arange(5, dtype=torch.int8)
    bundle = Bundle({
        "outputs": {
            "jets": {"jets_classification": probs, "pbc": pbc},
            "tracks": {"track_origin": trk},
        }
    })
    named = sink.named_outputs(bundle)
    assert set(named) == {"GN2v2_pb", "GN2v2_pc", "GN2v2_pu", "GN2v2_pbc", "GN2v2_TrackOrigin"}
    # the split is the v1 torch.split(probs, 1, -1) + squeeze — bitwise per channel
    for i, suffix in enumerate(["pb", "pc", "pu"]):
        torch.testing.assert_close(
            named[f"GN2v2_{suffix}"], probs[..., i].squeeze(), rtol=0, atol=0
        )
    # single leaves pass through untouched (no compute on the sink)
    assert named["GN2v2_pbc"] is pbc
    assert named["GN2v2_TrackOrigin"] is trk


def test_named_outputs_split_count_mismatch_errors_eagerly():
    """A split leaf whose last dim contradicts its names count is a loud eager error."""
    sink = OnnxExportSink(
        outputs=[OnnxExportLeaf(key=_JET, names=["pb", "pc", "pu"])], model_name="M"
    )
    bundle = Bundle({"outputs": {"jets": {"jets_classification": torch.rand(1, 2)}}})  # 2 != 3
    with pytest.raises(ConfigError, match="2 channels but declares 3 names"):
        sink.named_outputs(bundle)


# config validation


def test_leaf_requires_exactly_one_of_name_names():
    with pytest.raises(ConfigError, match="exactly one of 'name'"):
        OnnxExportLeaf(key=_JET, name="x", names=["a", "b"])
    with pytest.raises(ConfigError, match="exactly one of 'name'"):
        OnnxExportLeaf(key=_JET)


def test_leaf_rejects_non_outputs_key():
    with pytest.raises(ConfigError, match="not under the 'outputs'"):
        OnnxExportLeaf(key="preds.jets.jets_classification", names=["pb", "pc"])


def test_leaf_rejects_per_token_with_names():
    with pytest.raises(ConfigError, match="per_token applies to single-name"):
        OnnxExportLeaf(key=_JET, names=["pb", "pc"], per_token=True)


def test_leaf_rejects_unknown_dtype():
    with pytest.raises(ConfigError, match="dtype must be"):
        OnnxExportLeaf(key=_TRK, name="T", dtype="float16")


def test_sink_empty_outputs_defers_to_section():
    """An OMITTED/empty `outputs` defers to a bound `outputs:` section (not an error)."""
    sink = OnnxExportSink(outputs=[], model_name="M")
    with pytest.raises(ConfigError, match="has no export leaves"):
        sink.output_names()


def test_sink_rejects_duplicate_leaf_key():
    with pytest.raises(ConfigError, match="duplicate output key"):
        OnnxExportSink(
            outputs=[OnnxExportLeaf(key=_JET, name="a"), OnnxExportLeaf(key=_JET, name="b")]
        )


def test_sink_rejects_duplicate_flat_suffix():
    with pytest.raises(ConfigError, match="duplicate flat ONNX output name"):
        OnnxExportSink(
            outputs=[
                OnnxExportLeaf(key=_JET, names=["pb", "pc", "pu"]),
                OnnxExportLeaf(key=_PBC, name="pb"),  # collides with the split 'pb'
            ]
        )


def test_sink_builds_leaves_from_mappings():
    """jsonargparse-style mapping entries are built into `OnnxExportLeaf` instances."""
    sink = OnnxExportSink(
        outputs=[
            {"key": _JET, "names": ["pb", "pc", "pu"]},
            {"key": _TRK, "name": "TrackOrigin", "dtype": "int8", "per_token": True},
        ],
        model_name="M",
    )
    assert all(isinstance(leaf, OnnxExportLeaf) for leaf in sink.leaves)
    assert sink.outputs == (_JET, _TRK)
