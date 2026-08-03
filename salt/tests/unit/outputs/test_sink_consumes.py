"""The `consumes:` surface: `modes:` picks WHEN a sink runs, `consumes:` picks WHAT it takes."""

from __future__ import annotations

import pytest

from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.outputs import H5OutputSink, JSONLOutputSink, OnnxExportSink, OutputField
from salt.outputs.sinks.sink import collect_manifest_fields

pytestmark = pytest.mark.cpu_always

_JET = "outputs.jets.jets_classification"
_PBC = "outputs.jets.pbc"
_TRK = "outputs.tracks.HadronIndex"


class _StubProducer:
    """An ONNX-only producer declaring a fixed `(leaf_key, OutputField)` list."""

    def __init__(self, *fields):
        self._fields = list(fields)

    def manifest_fields(self, mode):
        """ONNX-only, mirroring every shipped model-graph producer."""
        return list(self._fields) if mode & Mode.ONNX else []


def _producers():
    """Three leaves across two streams, for the pattern tests."""
    return {
        "jet_probs": _StubProducer(
            (_JET, OutputField(h5_name=None, onnx_name="pb")),
            (_JET, OutputField(h5_name=None, onnx_name="pc")),
        ),
        "pbc": _StubProducer((_PBC, OutputField(h5_name=None, onnx_name="pbc"))),
        "objects": _StubProducer((
            _TRK,
            OutputField(h5_name=None, onnx_name="HadronIndex", dtype="i1", axis="per_token"),
        )),
    }


def _onnx_sink(**kwargs):
    sink = OnnxExportSink(model_name="M", **kwargs)
    sink.bind_model_modules(_producers())
    return sink


# -- the filter ---------------------------------------------------------------


def test_no_consumes_takes_everything():
    """The default is byte-identical to no filter at all."""
    assert _onnx_sink().outputs == (_JET, _PBC, _TRK)
    assert _onnx_sink(consumes=None).outputs == (_JET, _PBC, _TRK)


def test_a_wildcard_pattern_narrows_by_stream():
    assert _onnx_sink(consumes=["outputs.jets.*"]).outputs == (_JET, _PBC)


def test_an_exact_pattern_narrows_to_one_leaf():
    assert _onnx_sink(consumes=[_TRK]).outputs == (_TRK,)


def test_patterns_are_ored():
    assert _onnx_sink(consumes=[_PBC, _TRK]).outputs == (_PBC, _TRK)


def test_a_zero_match_pattern_names_the_pattern_and_the_keys():
    """A typo in ANY pattern is loud, even when the other patterns match."""
    sink = _onnx_sink(consumes=["outputs.jets.*", "outputs.typo.*"])
    with pytest.raises(ConfigError) as err:
        sink.output_names()
    assert "outputs.typo.*" in str(err.value)
    assert _JET in str(err.value)


def test_an_empty_consumes_list_points_at_the_delete_syntax():
    """An empty list is not "take nothing" — the section deletes an entry with null."""
    for cls in (OnnxExportSink, H5OutputSink, JSONLOutputSink):
        with pytest.raises(ConfigError, match="empty list"):
            cls(consumes=[])


def test_consumes_is_case_sensitive():
    """fnmatchcase, so a leaf key never matches by accident of platform casing."""
    sink = _onnx_sink(consumes=["outputs.TRACKS.*"])
    with pytest.raises(ConfigError, match="matches none"):
        sink.output_names()


def test_narrowing_changes_the_declared_demand():
    """`consumes:` is load-bearing: the sink stops demanding the leaves it drops."""
    from salt.graph.spec import flatten_spec  # noqa: PLC0415 - test-local

    narrowed = _onnx_sink(consumes=["outputs.jets.*"])
    assert set(flatten_spec(narrowed.declare_io(Mode.ONNX).requires)) == {_JET, _PBC}


# -- the collector ------------------------------------------------------------


def test_collect_skips_sources_without_a_manifest():
    """A source list may mix producers with plain graph modules."""
    fields = collect_manifest_fields([object(), *_producers().values()], Mode.ONNX)
    assert [key for key, _ in fields] == [_JET, _JET, _PBC, _TRK]


def test_collect_drops_non_final_fields():
    intermediate = _StubProducer((_PBC, OutputField(h5_name=None, onnx_name="pbc", final=False)))
    assert collect_manifest_fields([intermediate], Mode.ONNX) == []
