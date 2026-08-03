"""Tests for salt.schema."""

import h5py
import numpy as np
import pytest
import yaml

from salt.graph.errors import ConnectivityError, SchemaError
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import (
    SCHEMA_VERSION,
    GroupSchema,
    Schema,
    dump_schema,
    load_schema,
    save_schema,
)

# toy fixtures (no physics)


@pytest.fixture
def h5_path(tmp_path):
    """A tiny structured H5 file modeled on salt.testing.inputs.write_dummy_file."""
    path = tmp_path / "train.h5"
    jets_dtype = np.dtype([("pt", "f4"), ("eta", "f4"), ("flavour_label", "i4")])
    tracks_dtype = np.dtype([("d0", "f4"), ("ftagTruthOriginLabel", "i4"), ("valid", "?")])
    rng = np.random.default_rng(42)
    jets = np.zeros(10, dtype=jets_dtype)
    jets["pt"] = rng.random(10)
    tracks = np.zeros((10, 5), dtype=tracks_dtype)
    tracks["valid"] = True
    with h5py.File(path, "w") as f:
        f.attrs["unique_jets"] = 10
        f.attrs["config"] = "{}"
        f.create_dataset("jets", data=jets)
        f["jets"].attrs["flavour_label"] = ["bjets", "cjets", "ujets"]
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("plain", data=np.zeros(4))  # non-structured: ignored
    return path


# dump_schema


class TestDumpSchema:
    def test_groups_fields_and_dtypes(self, h5_path):
        schema = dump_schema(h5_path)
        assert set(schema.groups) == {"jets", "tracks"}  # 'plain' skipped
        assert schema.groups["jets"].fields == {
            "pt": "float32",
            "eta": "float32",
            "flavour_label": "int32",
        }
        assert schema.groups["tracks"].fields["valid"] == "bool"

    def test_attrs_are_plain_python(self, h5_path):
        schema = dump_schema(h5_path)
        labels = schema.groups["jets"].attrs["flavour_label"]
        assert labels == ["bjets", "cjets", "ujets"]
        assert all(type(name) is str for name in labels)
        assert schema.attrs["unique_jets"] == 10
        assert type(schema.attrs["unique_jets"]) is int

    def test_valid_presence(self, h5_path):
        schema = dump_schema(h5_path)
        assert schema.groups["tracks"].has_valid
        assert not schema.groups["jets"].has_valid

    def test_version_is_current(self, h5_path):
        assert dump_schema(h5_path).schema_version == SCHEMA_VERSION

    def test_unreadable_file_raises_schema_error(self, tmp_path):
        bad = tmp_path / "not_h5.h5"
        bad.write_text("this is not hdf5")
        with pytest.raises(SchemaError, match="cannot open"):
            dump_schema(bad)

    def test_dotted_field_name_skipped_with_warning(self, tmp_path, capsys):
        # '.' is legal in numpy/H5 names but unaddressable as a bundle key —
        # dump must not produce an artifact the rest of the tooling cannot read
        path = tmp_path / "dotted.h5"
        dtype = np.dtype([("kin.pt", "f4"), ("eta", "f4")])
        with h5py.File(path, "w") as f:
            f.create_dataset("jets", data=np.zeros(3, dtype=dtype))
        schema = dump_schema(path)
        assert schema.groups["jets"].fields == {"eta": "float32"}
        assert "skipping field jets/'kin.pt'" in capsys.readouterr().err
        # the full chain stays usable: keys() and validate_keys() work
        assert schema.keys() == ("jets.eta",)
        report = schema.validate_keys(["jets.eta"])
        assert report.ok

    def test_dotted_dataset_name_skipped_with_warning(self, tmp_path, capsys):
        path = tmp_path / "dotted_ds.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("weird.group", data=np.zeros(3, dtype=np.dtype([("x", "f4")])))
            f.create_dataset("jets", data=np.zeros(3, dtype=np.dtype([("pt", "f4")])))
        schema = dump_schema(path)
        assert set(schema.groups) == {"jets"}
        assert "skipping dataset 'weird.group'" in capsys.readouterr().err


# YAML round-trip


class TestRoundTrip:
    def test_dump_save_load(self, h5_path, tmp_path):
        schema = dump_schema(h5_path)
        out = tmp_path / "schema.yaml"
        save_schema(schema, out)
        loaded = load_schema(out)
        assert loaded.schema_version == schema.schema_version
        assert loaded.groups == schema.groups
        assert loaded.attrs == schema.attrs
        assert loaded.keys() == schema.keys()

    def test_keys_are_dotted_group_field(self, h5_path):
        schema = dump_schema(h5_path)
        assert "jets.pt" in schema.keys()
        assert "tracks.ftagTruthOriginLabel" in schema.keys()


# load tolerance (tolerant of additive change)


class TestLoadTolerance:
    def _write(self, tmp_path, payload):
        path = tmp_path / "schema.yaml"
        path.write_text(yaml.safe_dump(payload))
        return path

    def test_unknown_keys_ignored_and_newer_version_accepted(self, tmp_path):
        payload = {
            "schema_version": 2,  # newer than SCHEMA_VERSION
            "future_top_level": {"whatever": 1},
            "attrs": {"unique_jets": 5},
            "groups": {
                "jets": {
                    "fields": {"pt": "float32"},
                    "attrs": {"flavour_label": ["bjets"]},
                    "future_group_key": [1, 2, 3],
                },
            },
        }
        schema = load_schema(self._write(tmp_path, payload))
        assert schema.schema_version == 2
        assert schema.groups["jets"].fields == {"pt": "float32"}
        assert schema.groups["jets"].attrs == {"flavour_label": ["bjets"]}

    def test_missing_version_raises(self, tmp_path):
        path = self._write(tmp_path, {"groups": {}})
        with pytest.raises(SchemaError, match="schema_version"):
            load_schema(path)

    def test_non_int_version_raises(self, tmp_path):
        path = self._write(tmp_path, {"schema_version": "one", "groups": {}})
        with pytest.raises(SchemaError, match="schema_version"):
            load_schema(path)

    def test_missing_groups_raises(self, tmp_path):
        path = self._write(tmp_path, {"schema_version": 1})
        with pytest.raises(SchemaError, match="groups"):
            load_schema(path)

    def test_group_without_fields_raises(self, tmp_path):
        path = self._write(tmp_path, {"schema_version": 1, "groups": {"jets": {}}})
        with pytest.raises(SchemaError, match="fields"):
            load_schema(path)

    def test_not_a_mapping_raises(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(SchemaError, match="mapping"):
            load_schema(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SchemaError, match="cannot read"):
            load_schema(tmp_path / "nope.yaml")

    def test_dotted_field_name_in_artifact_raises(self, tmp_path):
        # a hand-written artifact with dotted names must fail at load with a
        # SchemaError (clean CLI exit), not a raw ValueError from keys()
        payload = {"schema_version": 1, "groups": {"jets": {"fields": {"kin.pt": "float32"}}}}
        with pytest.raises(SchemaError, match=r"field 'kin.pt' contains '\.'"):
            load_schema(self._write(tmp_path, payload))

    def test_dotted_group_name_in_artifact_raises(self, tmp_path):
        payload = {"schema_version": 1, "groups": {"a.b": {"fields": {"x": "float32"}}}}
        with pytest.raises(SchemaError, match=r"group 'a.b' contains '\.'"):
            load_schema(self._write(tmp_path, payload))


# validate_keys (the planner-facing missing/unknown report)


class TestValidateKeys:
    @pytest.fixture
    def schema(self):
        return Schema(
            groups={
                "jets": GroupSchema(fields={"pt": "float32", "eta": "float32"}),
                "tracks": GroupSchema(fields={"d0": "float32", "valid": "bool"}),
            }
        )

    def test_all_present(self, schema):
        report = schema.validate_keys(["jets.pt", "tracks.d0", "tracks.valid"])
        assert report.ok
        assert report.present == ("jets.pt", "tracks.d0", "tracks.valid")
        assert not report.missing
        assert not report.unknown

    def test_missing_field_with_suggestion(self, schema):
        report = schema.validate_keys(["jets.ptt"])
        assert not report.ok
        assert report.missing == ("jets.ptt",)
        assert "jets.pt" in report.suggestions["jets.ptt"]

    def test_unknown_group(self, schema):
        report = schema.validate_keys(["trcks.d0"])
        assert report.unknown == ("trcks.d0",)
        assert "tracks.d0" in report.suggestions["trcks.d0"]

    def test_single_component_key_is_unknown(self, schema):
        report = schema.validate_keys(["jets"])
        assert report.unknown == ("jets",)

    def test_malformed_key_is_unknown_not_a_raise(self, schema):
        # the docstring promise: malformed keys are classified, not raised
        report = schema.validate_keys(["jets..pt", ""])
        assert set(report.unknown) == {"jets..pt", ""}
        assert not report.ok

    def test_keys_with_dotted_field_raises_schema_error(self):
        # in-memory Schema with a dotted field: keys() must not leak ValueError
        schema = Schema(groups={"jets": GroupSchema(fields={"kin.pt": "float32"})})
        with pytest.raises(SchemaError, match="cannot form a dotted bundle key"):
            schema.keys()


# integration: schema.keys() feeds planner wildcard narrowing (rule (d))


class Wild:
    """Framework-style wildcard producer over a whole group."""

    allow_wildcards = True

    def __init__(self, name, pattern):
        self.name = name
        self._io = IO(produces=unflatten_spec({pattern: TensorSpec()}))

    def declare_io(self, mode):
        return self._io


class Consumer:
    def __init__(self, name, key):
        self.name = name
        self._io = IO(
            requires=unflatten_spec({key: TensorSpec()}),
            produces=unflatten_spec({"preds.x": TensorSpec()}),
        )

    def declare_io(self, mode):
        return self._io


class TestPlannerIntegration:
    def test_narrowing_validated_against_dumped_schema(self, h5_path):
        schema = dump_schema(h5_path)
        wild = Wild("reader", "jets.*")
        use = Consumer("use", "jets.pt")
        plan = compile_plan({"reader": wild, "use": use}, Mode.FIT, {}, schema=schema.keys())
        assert set(plan.step("reader").produces) == {"jets.pt"}

    def test_typo_raises_connectivity_error_with_suggestion(self, h5_path):
        schema = dump_schema(h5_path)
        wild = Wild("reader", "jets.*")
        use = Consumer("use", "jets.ptt")
        with pytest.raises(ConnectivityError, match="jets.pt"):
            compile_plan({"reader": wild, "use": use}, Mode.FIT, {}, schema=schema.keys())
