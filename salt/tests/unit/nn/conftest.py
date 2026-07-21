"""Shared fixtures/helpers for the per-module nn unit tests (split from test_modules.py)."""

from __future__ import annotations

import pytest

from salt.graph import (
    Bundle,
    Mode,
)
from salt.model.modules import (
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.tests._fixtures.gn2v2_fixture import (
    make_gn2_batch,
    write_parity_norm_dict,
)
from salt.tests._fixtures.gn2v2_fixture import (
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_labels,
)

B, T = 6, 10


@pytest.fixture
def norm_paths(tmp_path):
    """Write the parity norm/class dicts; return (norm_dict, class_dict) paths."""
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd, cd


@pytest.fixture
def gn2v2(norm_paths):
    """Build, compile (FIT), bind, and materialise the small GN2v2."""
    modules = build_gn2v2_modules(norm_paths[0])
    plan = compile_gn2v2(modules, Mode.FIT)
    schema = resolve_bind_schema(plan)
    bind_all(modules, schema)
    materialise_all(modules)
    return modules, plan, schema


def fit_bundle(n_electrons: int = 0) -> Bundle:
    """Build a FIT-mode bundle from the deterministic v1 batch + labels."""
    inputs, masks = make_gn2_batch(B, T, n_electrons=n_electrons)
    labels = make_gn2_labels(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    for stream, m in masks.items():
        b.set(f"masks.{stream}", m)
    for stream, fields in labels.items():
        for name, val in fields.items():
            b.set(f"labels.{stream}.{name}", val)
    return b
