"""Shared fixtures for the MaskFormer objects-sink unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.dummy_h5 import dummy_h5
from salt.tests._fixtures.gn2v2_fixture import (
    build_gn2v2_modules,
    write_parity_norm_dict,
)

N_JETS, L_FILE = 1000, 40  # salt-v1 datagen schema geometry (jets, tracks per jet)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("m3_writers")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # exactly four underscore parts -> sample heuristic yields 'ttbar' (PW:169)
    h5_path, _, _ = dummy_h5(base, name="pp_output_test_ttbar.h5")
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.fixture(scope="module")
def modules(data):
    # config-only module dict: requires/columns need attrs, not bound layers
    return build_gn2v2_modules(data["nd"])
