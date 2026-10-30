"""`InputCopyWriter` — mirror test for `salt/outputs/input_copy_writer.py`.

Ctor validation, per-mode `declare_io`/`copy_leaves` gating, `forward` clone
semantics, `manifest_fields` shape, the `streams:` form + `bind_stream_variables`,
and the `OutputColumn.np_dtype` guard.
"""

from __future__ import annotations

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.outputs.input_copy_writer import InputCopyWriter
from salt.outputs.output_schema import OutputColumn

pytestmark = pytest.mark.cpu_always


# -- constructor validation ---------------------------------------------------


def test_none_variables_is_a_config_error():
    """`variables=None` behaves like an empty mapping — no field list at all."""
    with pytest.raises(ConfigError, match="explicit, non-empty field list"):
        InputCopyWriter(variables=None)


def test_empty_variables_mapping_is_a_config_error():
    with pytest.raises(ConfigError, match="explicit, non-empty field list"):
        InputCopyWriter(variables={})


def test_empty_stream_field_list_is_a_config_error():
    with pytest.raises(ConfigError, match=r"stream 'jets' is empty"):
        InputCopyWriter(variables={"jets": []})


def test_duplicate_field_name_is_a_config_error():
    with pytest.raises(ConfigError, match="repeats"):
        InputCopyWriter(variables={"jets": ["pt", "pt"]})


def test_valid_variables_construct_without_error():
    icw = InputCopyWriter(variables={"jets": ["pt", "eta"], "tracks": ["d0"]})
    assert icw.name == "inputs_copy"
    assert icw.variables == {"jets": ["pt", "eta"], "tracks": ["d0"]}


# -- declare_io: TEST requires labels.*, produces outputs.*; other modes empty -


def test_declare_io_test_requires_labels_and_produces_outputs():
    icw = InputCopyWriter(variables={"jets": ["pt", "eta"], "tracks": ["d0"]})
    io = icw.declare_io(Mode.TEST)
    req = flatten_spec(io.requires)
    assert set(req) == {"labels.jets.pt", "labels.jets.eta", "labels.tracks.d0"}
    assert all(spec.kind == "label" for spec in req.values())
    prod = flatten_spec(io.produces)
    assert set(prod) == {"outputs.jets.pt", "outputs.jets.eta", "outputs.tracks.d0"}
    assert all(spec.kind == "data" for spec in prod.values())


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.ONNX])
def test_declare_io_is_empty_outside_test(mode):
    """Copies are eval-H5 only — export/fit/val declare nothing to demand."""
    icw = InputCopyWriter(variables={"jets": ["pt"]})
    io = icw.declare_io(mode)
    assert flatten_spec(io.requires) == {}
    assert flatten_spec(io.produces) == {}


# -- modes: gating (test vs export) is symmetric between declare_io/copy_leaves


def test_modes_test_only_is_inert_under_export():
    icw = InputCopyWriter(variables={"jets": ["pt"]}, modes=["test"])
    assert icw.runs_in_mode(Mode.TEST) is True
    assert icw.runs_in_mode(Mode.ONNX) is False
    assert icw.copy_leaves(Mode.TEST) == [("jets", "pt")]
    assert icw.copy_leaves(Mode.ONNX) == []
    assert flatten_spec(icw.declare_io(Mode.TEST).produces) == {
        "outputs.jets.pt": next(iter(flatten_spec(icw.declare_io(Mode.TEST).produces).values()))
    }


def test_modes_export_only_never_declares_test_io():
    """`declare_io` gates on BOTH `runs_in_mode` AND `mode & Mode.TEST`."""
    icw = InputCopyWriter(variables={"jets": ["pt"]}, modes=["export"])
    assert icw.runs_in_mode(Mode.ONNX) is True
    assert icw.runs_in_mode(Mode.TEST) is False
    io = icw.declare_io(Mode.TEST)
    assert flatten_spec(io.requires) == {}
    assert flatten_spec(io.produces) == {}
    assert icw.copy_leaves(Mode.TEST) == []
    assert icw.copy_leaves(Mode.ONNX) == [("jets", "pt")]


def test_copy_leaves_pairs_in_declaration_order():
    icw = InputCopyWriter(variables={"jets": ["pt", "eta"], "tracks": ["d0"]})
    assert icw.copy_leaves(Mode.TEST) == [("jets", "pt"), ("jets", "eta"), ("tracks", "d0")]


# -- forward: clones the label leaf into the output leaf ---------------------


def test_forward_clones_label_leaves_into_output_leaves():
    icw = InputCopyWriter(variables={"jets": ["pt"], "tracks": ["d0"]})
    pt = torch.tensor([1.0, 2.0, 3.0])
    d0 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    bundle = Bundle({"labels": {"jets": {"pt": pt}, "tracks": {"d0": d0}}})
    out = icw.forward(bundle, Mode.TEST)
    assert set(out) == {"outputs.jets.pt", "outputs.tracks.d0"}
    assert torch.equal(out["outputs.jets.pt"], pt)
    assert out["outputs.jets.pt"].data_ptr() != pt.data_ptr()  # a real clone, not a view
    assert torch.equal(out["outputs.tracks.d0"], d0)
    assert out["outputs.tracks.d0"].data_ptr() != d0.data_ptr()


# -- manifest_fields: order, bare names, prefix=False, dtype=None, final ------


def test_manifest_fields_order_names_and_flags():
    icw = InputCopyWriter(variables={"jets": ["pt", "eta"], "tracks": ["d0"]})
    fields = icw.manifest_fields(Mode.TEST)
    assert [key for key, _ in fields] == [
        "outputs.jets.pt",
        "outputs.jets.eta",
        "outputs.tracks.d0",
    ]
    for (_, field), var in zip(fields, ("pt", "eta", "d0"), strict=True):
        assert field.h5_name == var
        assert field.dtype is None
        assert field.prefix is False
        assert field.final is True


def test_manifest_fields_empty_when_not_running_in_mode():
    icw = InputCopyWriter(variables={"jets": ["pt"]}, modes=["test"])
    assert icw.manifest_fields(Mode.ONNX) == []
    assert icw.manifest_fields(Mode.TEST) != []


# -- OutputColumn.np_dtype refuses a bundle-dtype (dtype=None) column --------


def test_output_column_np_dtype_raises_for_none_dtype():
    col = OutputColumn(key="outputs.jets.pt", suffixes=["pt"], dtype=None)
    with pytest.raises(ConfigError):
        col.np_dtype("salt")


# -- streams form: copy every inputs.<stream> field, bound from the pipeline --

_STREAM_FIELDS = {"jets": ["pt", "eta", "mass"], "tracks": ["d0", "z0"]}


def test_streams_only_bind_requires_every_bound_field_in_order():
    icw = InputCopyWriter(streams=["jets", "tracks"])
    icw.bind_stream_variables(_STREAM_FIELDS)
    expected = [
        ("jets", "pt"),
        ("jets", "eta"),
        ("jets", "mass"),
        ("tracks", "d0"),
        ("tracks", "z0"),
    ]
    assert icw.copy_leaves(Mode.TEST) == expected
    assert icw.variables == _STREAM_FIELDS
    io = icw.declare_io(Mode.TEST)
    assert list(flatten_spec(io.requires)) == [f"labels.{s}.{v}" for s, v in expected]
    assert list(flatten_spec(io.produces)) == [f"outputs.{s}.{v}" for s, v in expected]


def test_streams_order_sets_the_copy_order():
    icw = InputCopyWriter(streams=["tracks", "jets"])
    icw.bind_stream_variables(_STREAM_FIELDS)
    assert list(icw.variables) == ["tracks", "jets"]
    assert icw.copy_leaves(Mode.TEST)[0] == ("tracks", "d0")


def test_streams_with_variables_xbb_shape_explicit_list_wins():
    """`streams: [jets]` + `variables: {jets: [...]}`: the explicit list is kept, no tracks."""
    xbb = ["pt", "eta", "mass", "R10TruthLabel_R22v1", "flavour_label"]
    icw = InputCopyWriter(streams=["jets"], variables={"jets": xbb})
    icw.bind_stream_variables(_STREAM_FIELDS)
    assert icw.variables == {"jets": xbb}
    assert icw.copy_leaves(Mode.TEST) == [("jets", v) for v in xbb]
    assert not any(s == "tracks" for s, _ in icw.copy_leaves(Mode.TEST))


def test_streams_with_partial_variables_narrows_one_and_binds_the_other():
    icw = InputCopyWriter(streams=["jets", "tracks"], variables={"jets": ["pt"]})
    icw.bind_stream_variables(_STREAM_FIELDS)
    assert icw.variables == {"jets": ["pt"], "tracks": ["d0", "z0"]}
    assert icw.copy_leaves(Mode.TEST) == [("jets", "pt"), ("tracks", "d0"), ("tracks", "z0")]


def test_neither_streams_nor_variables_is_a_config_error():
    with pytest.raises(ConfigError, match="explicit, non-empty field list"):
        InputCopyWriter()
    with pytest.raises(ConfigError, match="`streams:`"):
        InputCopyWriter(streams=None, variables=None)


def test_empty_streams_list_is_a_config_error():
    with pytest.raises(ConfigError, match=r"streams: \[\]` is empty"):
        InputCopyWriter(streams=[], variables={"jets": ["pt"]})
    with pytest.raises(ConfigError, match="explicit, non-empty field list"):
        InputCopyWriter(streams=[])


def test_duplicate_stream_in_streams_is_a_config_error():
    with pytest.raises(ConfigError, match=r"repeats stream 'jets'"):
        InputCopyWriter(streams=["jets", "tracks", "jets"])


def test_bare_string_streams_is_a_config_error():
    with pytest.raises(ConfigError, match=r"bare string 'jets'"):
        InputCopyWriter(streams="jets")


def test_variables_key_outside_streams_is_a_config_error():
    with pytest.raises(ConfigError, match=r"stream 'tracks' is not in `streams:`"):
        InputCopyWriter(streams=["jets"], variables={"tracks": ["d0"]})


def test_unresolved_streams_form_raises_at_copy_leaves():
    icw = InputCopyWriter(streams=["jets"])
    with pytest.raises(ConfigError, match=r"\['jets'\] have no field list.*inputs\.<stream>"):
        icw.copy_leaves(Mode.TEST)


def test_unresolved_streams_form_raises_at_declare_io_test():
    icw = InputCopyWriter(streams=["jets", "tracks"], variables={"jets": ["pt"]})
    with pytest.raises(ConfigError, match=r"\['tracks'\] have no field list.*`variables:`"):
        icw.declare_io(Mode.TEST)


def test_unresolved_streams_form_raises_at_manifest_fields_and_forward():
    icw = InputCopyWriter(streams=["jets"])
    with pytest.raises(ConfigError, match="have no field list"):
        icw.manifest_fields(Mode.TEST)
    bundle = Bundle({"labels": {"jets": {"pt": torch.tensor([1.0])}}})
    with pytest.raises(ConfigError, match="have no field list"):
        icw.forward(bundle, Mode.TEST)


def test_bind_with_missing_stream_is_a_config_error():
    icw = InputCopyWriter(streams=["jets", "hits"])
    with pytest.raises(
        ConfigError, match=r"'hits'.*no `inputs\.hits`.*available: \['jets', 'tracks'\]"
    ):
        icw.bind_stream_variables(_STREAM_FIELDS)


def test_bind_with_empty_stream_field_list_is_a_config_error():
    icw = InputCopyWriter(streams=["jets"])
    with pytest.raises(ConfigError, match=r"'jets'.*no `inputs\.jets`"):
        icw.bind_stream_variables({"jets": []})


def test_bind_is_idempotent_and_never_overrides():
    icw = InputCopyWriter(streams=["jets", "tracks"], variables={"jets": ["pt"]})
    icw.bind_stream_variables(_STREAM_FIELDS)
    first = icw.copy_leaves(Mode.TEST)
    icw.bind_stream_variables(_STREAM_FIELDS)
    icw.bind_stream_variables({"jets": ["eta"], "tracks": ["qOverP"]})
    assert icw.copy_leaves(Mode.TEST) == first
    assert icw.variables == {"jets": ["pt"], "tracks": ["d0", "z0"]}


def test_bind_is_a_noop_for_a_variables_only_writer():
    icw = InputCopyWriter(variables={"jets": ["pt", "eta"]})
    icw.bind_stream_variables({})
    assert icw.variables == {"jets": ["pt", "eta"]}


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.ONNX])
def test_streams_form_declare_io_is_empty_outside_test(mode):
    """Outside TEST nothing is demanded, bound or not."""
    unbound = InputCopyWriter(streams=["jets", "tracks"])
    assert flatten_spec(unbound.declare_io(mode).requires) == {}
    assert flatten_spec(unbound.declare_io(mode).produces) == {}
    bound = InputCopyWriter(streams=["jets", "tracks"])
    bound.bind_stream_variables(_STREAM_FIELDS)
    assert flatten_spec(bound.declare_io(mode).requires) == {}
    assert flatten_spec(bound.declare_io(mode).produces) == {}


def test_streams_form_manifest_fields_after_bind():
    icw = InputCopyWriter(streams=["tracks"])
    icw.bind_stream_variables(_STREAM_FIELDS)
    fields = icw.manifest_fields(Mode.TEST)
    assert [key for key, _ in fields] == ["outputs.tracks.d0", "outputs.tracks.z0"]
    assert [field.h5_name for _, field in fields] == ["d0", "z0"]
    assert all(field.dtype is None for _, field in fields)
