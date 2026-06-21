"""Parity adapters: drop-in replacements for salt.utils.inputs writers.

These reproduce the call signatures of ``write_dummy_file`` and
``write_dummy_norm_dict`` but back them with the schema-driven generator. They
let the EXISTING salt test suite (test_pipeline.py / test_cli.py) run unmodified
against schema-generated data, proving the generated file is a drop-in.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .engine import generate_data
from .io import compute_class_dict, compute_norm_dict, write_h5
from .schema import load_schema

# the bundled schema reproducing salt v1 write_dummy_file (committed beside the
# pipeline recipes; robust __file__-relative path, overridable via env)
SCHEMA_PATH = Path(__file__).resolve().parent / "recipes" / "salt-v1-schema.yaml"


def _resolve_schema_path(schema_path: str | Path | None) -> Path:
    if schema_path is not None:
        return Path(schema_path)
    # default: bundled example schema (overridable via env)
    import os

    env = os.environ.get("SALT_DATAGEN_SCHEMA")
    if env:
        return Path(env)
    return SCHEMA_PATH


def _flags(make_xbb=False, inc_taus=False, inc_params=False, is_gn3=False) -> dict[str, bool]:
    return {
        "make_xbb": bool(make_xbb),
        "inc_taus": bool(inc_taus),
        "inc_params": bool(inc_params),
        "is_gn3": bool(is_gn3),
    }


def schema_write_dummy_file(
    fname: str | Path,
    sd_fname: str | Path,
    make_xbb: bool = False,
    inc_taus: bool = False,
    inc_params: bool = False,
    is_gn3: bool = False,
    schema_path: str | Path | None = None,
) -> None:
    """Drop-in for ``salt.utils.inputs.write_dummy_file`` (schema-backed).

    ``sd_fname`` is accepted for signature compatibility but the variable lists
    come from the schema, not the norm-dict file.
    """
    schema = load_schema(_resolve_schema_path(schema_path))
    flags = _flags(make_xbb, inc_taus, inc_params, is_gn3)
    data = generate_data(schema, flags=flags)
    write_h5(data, fname, schema=schema, attrs={"_flags": flags})


def schema_write_dummy_norm_dict(
    nd_path: str | Path,
    cd_path: str | Path,
    is_gn3: bool = False,
    schema_path: str | Path | None = None,
) -> None:
    """Drop-in for ``salt.utils.inputs.write_dummy_norm_dict`` (schema-backed).

    Generates the data once, then derives the norm/class dicts from it.
    """
    schema = load_schema(_resolve_schema_path(schema_path))
    flags = _flags(is_gn3=is_gn3)
    data = generate_data(schema, flags=flags)
    nd = compute_norm_dict(data, schema)
    cd = compute_class_dict(data, schema, flags=flags)

    with open(nd_path, "w") as fh:
        yaml.dump(nd, fh, sort_keys=False)
    with open(cd_path, "w") as fh:
        yaml.dump(cd, fh, sort_keys=False)


def patch_salt_inputs() -> None:
    """Monkeypatch ``salt.utils.inputs`` writers with the schema-backed ones.

    After this call, any test that imports
    ``salt.utils.inputs.write_dummy_file`` / ``write_dummy_norm_dict`` (or the
    re-exports in test modules) will exercise the schema generator.
    """
    import salt.utils.inputs as inputs_mod

    inputs_mod.write_dummy_file = schema_write_dummy_file
    inputs_mod.write_dummy_norm_dict = schema_write_dummy_norm_dict
