"""Per-config gate — the ``outputs:``-section H5 column schema == the committed schema goldens.

Closure evidence (plan 50 Phase E, 2026-07-14): this gate previously compared the
section schema against the tasks' legacy ``output_names`` rendering (the G1
oracle, retired with ``get_h5``/``onnx_outputs``). The contract is now the
committed per-config schema goldens at ``salt/tests/_fixtures/output_goldens/``
(captured at 96d88d8 via ``generate_goldens.py``, regenerated green through
Phase C with the per-task target-label columns) — never self-consistency alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from salt.main import CONFIG_DIR
from salt.onnx.export import _run_free_cli

pytestmark = pytest.mark.cpu_always

GOLDEN_DIR = Path(__file__).parents[1] / "_fixtures" / "output_goldens"

_NORM = "model.modules.norm.init_args.norm_dict=unused.yaml"
# disable the logger so the run-free parse does not hit the keyless CometLogger
# instantiate failure (no COMET_API_KEY in CI/local) — _run_free_cli does not apply
# the disable_logger_in_config patch the salt graph/test entry points do.
_NO_LOGGER = "trainer.logger=false"
MIGRATED = {
    "regression": [_NORM, _NO_LOGGER],
    "regression_gaussian": [_NORM, _NO_LOGGER],
}


def _golden_columns(config_name: str) -> list[dict]:
    """The committed golden's resolved H5 column table, in schema order."""
    golden = json.loads((GOLDEN_DIR / f"{config_name}.json").read_text())
    return [
        {
            "key": col["key"],
            "stream": col["stream"],
            "column_names": list(col["column_names"]),
            "dtype": col["dtype"],
            "prefix": bool(col["prefix"]),
            "suffixes": list(col["suffixes"]),
        }
        for col in golden["h5"]["columns"]
    ]


def _section_columns(sink, run_name: str) -> list[dict]:
    """The live sink's resolved column table, normalised like the golden capture."""
    return [
        {
            "key": col.key,
            "stream": col.stream,
            "column_names": list(col.column_names(run_name)),
            "dtype": col.dtype,
            "prefix": bool(col.prefix),
            "suffixes": list(col.suffixes),
        }
        for col in sink._resolve_columns(run_name)  # noqa: SLF001
    ]


@pytest.mark.parametrize("config_name", sorted(MIGRATED))
def test_section_h5_schema_matches_committed_golden(config_name):
    """The migrated config's ``outputs:``-section H5 schema == the committed golden table."""
    from salt.cli import _as_sink_node, _static_writer_sink_callback

    config = CONFIG_DIR / f"{config_name}.yaml"
    assert config.is_file(), f"missing config {config}"
    golden = _golden_columns(config_name)
    assert golden, f"{config_name}: committed golden carries no H5 columns"

    cli = _run_free_cli([config], MIGRATED[config_name])
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001
    sink = _as_sink_node(_static_writer_sink_callback(cli))
    assert sink is not None, f"{config_name}: no H5OutputSink wired at callbacks"
    # the section is composed + bound onto the sink by SaltCLI.instantiate_classes
    # (the run-free CLI path); assert it really is the dumb-section path.
    assert sink._is_dumb_section(), (  # noqa: SLF001
        f"{config_name}: H5OutputSink is not driven by the outputs: section"
    )

    section = _section_columns(sink, run_name)
    assert section == golden, (
        f"[{config_name}] section H5 schema drifted from the committed golden\n"
        f"  golden ({GOLDEN_DIR / (config_name + '.json')}):\n    {golden}\n"
        f"  section (H5OutputSink._resolve_columns):\n    {section}"
    )
