"""H5StructuredReader group→dataset mapping against the shipped GN3EPCLV01 config (N9)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import salt
from salt.data import H5StructuredReader
from salt.graph.errors import SchemaError
from salt.schema import dump_schema
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict


def _shipped_groups() -> dict:
    """The `data.modules.reader.init_args.groups` mapping from the shipped GN3EPCLV01 config."""
    cfg_path = Path(salt.__file__).parent / "configs" / "GN3EPCLV01.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg["data"]["modules"]["reader"]["init_args"]["groups"]


class TestGN3EPCLV01ReaderGroups:
    def test_gn3epclv01_global_group_reads_jets_dataset(self):
        """N9: the shipped 'global' group carries dataset: jets, not a literal 'global' H5 group."""
        groups = _shipped_groups()
        assert groups["global"] == {"global_object": True, "dataset": "jets"}
        assert groups["tracks"]["dataset"] == "tracks_ghost"

    def test_reader_from_shipped_groups_resolves_global_without_schema_group(self, tmp_path):
        """H5StructuredReader resolves the shipped 'global' group against a schema with no
        literal 'global' H5 group, via dataset: jets.
        """
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = dump_schema(h5)
        assert "global" not in schema.groups  # precondition: no literal 'global' H5 group

        groups = _shipped_groups()
        reader = H5StructuredReader(groups=groups, schema=schema)  # must not raise

        assert reader.groups["global"].dataset == "jets"
        assert reader.groups["global"].global_object is True
        assert reader.schema_group("global") is reader.schema_group("jets")
        assert reader.groups["tracks"].dataset == "tracks_ghost"
        assert reader.streams == ("jets", "tracks", "flows", "electrons", "global")

    def test_global_group_without_dataset_hits_schema_error(self, tmp_path):
        """Without dataset: jets, the reader hunts a literal 'global' H5 group — the N9 failure
        mode the config fix removes.
        """
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = dump_schema(h5)

        groups = _shipped_groups()
        bad = dict(groups)
        bad["global"] = {"global_object": True}
        with pytest.raises(SchemaError, match="group 'global'.*dataset 'global'"):
            H5StructuredReader(groups=bad, schema=schema)
