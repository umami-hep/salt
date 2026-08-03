"""Unit gates for the `OnnxExportSink` declare-only terminal node (design §4.2)."""

from __future__ import annotations

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, SinkModule, flatten_spec
from salt.outputs import OnnxExportLeaf, OnnxExportSink, OutputField

_JET = "outputs.jets.jets_classification"
_PBC = "outputs.jets.pbc"
_TRK = "outputs.tracks.track_origin"


class _StubProducer:
    """A minimal ONNX-only producer: it declares its own `(leaf_key, OutputField)` list."""

    def __init__(self, *fields):
        self._fields = list(fields)

    def manifest_fields(self, mode):
        """ONNX-only, mirroring every shipped model-graph producer."""
        return list(self._fields) if mode & Mode.ONNX else []


def _global(name):
    """One global float32 ONNX field."""
    return OutputField(h5_name=None, onnx_name=name, dtype="f4", axis="global")


def _per_token(name):
    """One per-token int8 ONNX field."""
    return OutputField(h5_name=None, onnx_name=name, dtype="i1", axis="per_token")


def _sink(model_name="GN2v2", **kwargs):
    """A representative folded export sink: split_scalars + a combine + a per-token int8 leaf."""
    sink = OnnxExportSink(model_name=model_name, **kwargs)
    sink.bind_model_modules({
        "jet_probs": _StubProducer(
            (_JET, _global("pb")), (_JET, _global("pc")), (_JET, _global("pu"))
        ),
        "pbc": _StubProducer((_PBC, _global("pbc"))),
        "track_origin_index": _StubProducer((_TRK, _per_token("TrackOrigin"))),
    })
    return sink


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


# auto-collection: leaf shape is derived from the producers' fields


def test_several_fields_on_one_key_build_a_split_leaf():
    """N global fields sharing a leaf key -> one plural-``names`` split leaf."""
    sink = _sink()
    jet = next(leaf for leaf in sink.leaves if leaf.key == _JET)
    assert jet.names == ["pb", "pc", "pu"]
    assert jet.per_token is False
    assert jet.dtype == "float32"


def test_single_field_builds_a_single_name_leaf():
    """One field on a key -> a single-``name`` leaf carrying the field's axis and dtype."""
    sink = _sink()
    by_key = {leaf.key: leaf for leaf in sink.leaves}
    assert by_key[_PBC].name == "pbc"
    assert by_key[_PBC].names is None
    assert by_key[_PBC].per_token is False
    assert by_key[_TRK].name == "TrackOrigin"
    assert by_key[_TRK].dtype == "int8"
    assert by_key[_TRK].per_token is True


def test_split_leaf_rejects_a_per_token_field():
    """A key mixing several fields with a per-token one has no valid leaf shape."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({"p": _StubProducer((_JET, _global("pb")), (_JET, _per_token("idx")))})
    with pytest.raises(ConfigError, match="including a per-token one"):
        sink.output_names()


def test_non_final_fields_are_dropped():
    """A ``final=False`` field is declared but never serialised (the collector drops it)."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({
        "p": _StubProducer(
            (_PBC, _global("pbc")),
            (
                "outputs.objects.vertices_regression",
                OutputField(h5_name=None, onnx_name="vertices_regression", final=False),
            ),
        )
    })
    assert sink.outputs == (_PBC,)


def test_section_fields_come_before_model_module_fields():
    """Source order is the tuple order: the section's writers, then the model's modules."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({"late": _StubProducer((_TRK, _per_token("TrackOrigin")))})
    sink.bind_output_section({"early": _StubProducer((_PBC, _global("pbc")))})
    assert sink.outputs == (_PBC, _TRK)


def test_a_writer_in_both_sources_is_collected_once():
    """A section writer folded into the model graph appears twice but is taken once."""
    writer = _StubProducer((_PBC, _global("pbc")))
    sink = OnnxExportSink(model_name="M")
    sink.bind_output_section({"run_tasks": writer})
    sink.bind_model_modules({"run_tasks": writer, "encoder": object()})
    assert sink.output_names() == ["M_pbc"]


# generated metadata: output_names / dtypes / dynamic_axes (design §6.3)


def test_output_names_are_in_declared_tuple_order():
    """The collected leaf list IS the Athena tuple order — split expands, combine in place."""
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


def test_dynamic_axis_defaults_to_the_leaf_stream():
    """A collected per-token leaf takes the ``n_<stream>`` default (a field has no override)."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({"p": _StubProducer((_TRK, _per_token("T")))})
    assert sink.dynamic_axes() == {"M_T": {0: "n_tracks"}}
    override = OnnxExportLeaf(key=_TRK, name="T", per_token=True, dyn_axis="n_custom")
    assert override.resolved_dyn_axis() == "n_custom"


def test_model_name_required_for_names():
    """Deriving the Athena names needs a model_name (config or adapter-supplied)."""
    sink = _sink(model_name=None)
    with pytest.raises(ConfigError, match="no model_name"):
        sink.output_names()
    sink.model_name = "Late"
    assert sink.output_names()[:3] == ["Late_pb", "Late_pc", "Late_pu"]


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
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({
        "p": _StubProducer((_JET, _global("pb")), (_JET, _global("pc")), (_JET, _global("pu")))
    })
    bundle = Bundle({"outputs": {"jets": {"jets_classification": torch.rand(1, 2)}}})  # 2 != 3
    with pytest.raises(ConfigError, match="2 channels but declares 3 names"):
        sink.named_outputs(bundle)


# config validation


def test_leaf_rejects_both_name_and_names():
    with pytest.raises(ConfigError, match="BOTH"):
        OnnxExportLeaf(key=_JET, name="x", names=["a", "b"])


def test_leaf_defaults_name_to_leaf_terminal_segment():
    """Single-source naming: omitting name/names defaults the suffix to the leaf terminal."""
    leaf = OnnxExportLeaf(key=_JET)
    assert leaf.name == "jets_classification"
    assert leaf.suffixes == ("jets_classification",)


def test_leaf_rejects_non_outputs_key():
    with pytest.raises(ConfigError, match="not under the 'outputs'"):
        OnnxExportLeaf(key="preds.jets.jets_classification", names=["pb", "pc"])


def test_leaf_rejects_per_token_with_names():
    with pytest.raises(ConfigError, match="per_token applies to single-name"):
        OnnxExportLeaf(key=_JET, names=["pb", "pc"], per_token=True)


def test_leaf_rejects_unknown_dtype():
    with pytest.raises(ConfigError, match="dtype must be"):
        OnnxExportLeaf(key=_TRK, name="T", dtype="float16")


def test_explicit_outputs_list_is_retired():
    """The config-level `outputs:` leaf list is a hard error naming the replacement."""
    with pytest.raises(ConfigError, match="manifest_fields"):
        OnnxExportSink(outputs=[OnnxExportLeaf(key=_JET, names=["pb", "pc", "pu"])])
    with pytest.raises(ConfigError, match="no longer accepts an explicit"):
        OnnxExportSink(outputs=[{"key": _JET, "names": ["pb", "pc", "pu"]}])


def test_sink_with_no_declared_field_errors():
    """A sink no producer feeds names the sources it searched."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({"encoder": object()})
    with pytest.raises(ConfigError, match="collected no ONNX output"):
        sink.output_names()


def test_sink_rejects_duplicate_leaf_key():
    """The dup-key guard (unreachable through the collector, which groups by key)."""
    with pytest.raises(ConfigError, match="duplicate output key"):
        OnnxExportSink._validate_leaves([  # noqa: SLF001 - guard has no collected trigger
            OnnxExportLeaf(key=_JET, name="a"),
            OnnxExportLeaf(key=_JET, name="b"),
        ])


def test_sink_rejects_duplicate_flat_suffix():
    """Two producers minting the same Athena suffix collide in the flat namespace."""
    sink = OnnxExportSink(model_name="M")
    sink.bind_model_modules({
        "jet_probs": _StubProducer((_JET, _global("pb"))),
        "pbc": _StubProducer((_PBC, _global("pb"))),  # collides with 'pb'
    })
    with pytest.raises(ConfigError, match="duplicate flat ONNX output name"):
        sink.output_names()
