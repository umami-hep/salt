"""Per-config gate — dumb ``outputs:``-section H5 column schema == legacy TaskWriter schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.graph.spec import KEY_SEP
from salt.core.main import CONFIG_DIR
from salt.core.onnx.export import _run_free_cli

pytestmark = pytest.mark.cpu_always

# the dumb ``outputs:``-section H5OutputSink schema gate. All production
# classification/regression/gaussian configs ride the plan-34 ``outputs:`` section +
# dumb sinks (the auto-collect producer-discovery path was removed in W34.4d). The
# classification family's section H5 schema is gated byte-for-byte against the legacy
# WriterCallback by the cutover34 tests (test_w34_outputs_section.py /
# test_w34_regression_cutover.py / test_w34_gaussian_cutover.py); this gate guards the
# ``regression`` / ``regression_gaussian`` section schema == legacy task.output_names.
_NORM = "model.modules.norm.init_args.norm_dict=unused.yaml"
# disable the logger so the run-free parse does not hit the keyless CometLogger
# instantiate failure (no COMET_API_KEY in CI/local) — _run_free_cli does not apply
# the disable_logger_in_config patch the salt2 graph/test entry points do.
_NO_LOGGER = "trainer.logger=false"
MIGRATED = {
    "regression": [_NORM, _NO_LOGGER],
    "regression_gaussian": [_NORM, _NO_LOGGER],
}

# heads whose H5 eval column is a DEFERRED family (no W5 conversion producer):
# their columns are excluded from the parity comparison (documented deferral).
_DEFERRED_TASK_TYPES = {"VertexingTaskModule"}


def _legacy_columns(modules, run_name: str) -> dict[str, list[tuple[str, str]]]:
    """The legacy TaskWriter H5 schema per stream: {stream: [(column, dtype), ...]}."""
    out: dict[str, list[tuple[str, str]]] = {}
    for module in modules.values():
        pred_key = getattr(module, "pred_key", None)
        stream = getattr(module, "stream", None)
        # a task head: has pred_key + stream + output_names, and is NOT a conversion producer
        if (
            not isinstance(pred_key, str)
            or not isinstance(stream, str)
            or isinstance(getattr(module, "output_key", None), str)
            or not callable(getattr(module, "output_names", None))
        ):
            continue
        if type(module).__name__ in _DEFERRED_TASK_TYPES:
            continue
        try:
            names = module.output_names(run_name)
        except Exception:  # noqa: BLE001 — a head with no TEST rendering (shouldn't happen here)
            continue
        out.setdefault(stream, []).extend((str(c), str(d)) for c, d in names)
    return out


def _section_columns(sink, run_name: str) -> dict[str, list[tuple[str, str]]]:
    """The dumb-section H5OutputSink schema per stream: {stream: [(column, dtype), ...]}."""
    out: dict[str, list[tuple[str, str]]] = {}
    for col in sink._resolve_columns(run_name):  # noqa: SLF001
        stream = col.key.split(KEY_SEP)[1]
        for column in col.column_names(run_name):
            out.setdefault(stream, []).append((column, col.dtype))
    return out


@pytest.mark.parametrize("config_name", sorted(MIGRATED))
def test_section_h5_schema_matches_legacy_taskwriter(config_name):
    """The migrated config's dumb-section H5 schema == legacy TaskWriter schema (per stream)."""
    from salt.core.cli import _as_sink_node, _static_writer_sink_callback

    config = CONFIG_DIR / f"{config_name}.yaml"
    assert config.is_file(), f"missing config {config}"
    cli = _run_free_cli([config], MIGRATED[config_name])
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001
    modules = dict(cli.model._graph_modules)  # noqa: SLF001
    sink = _as_sink_node(_static_writer_sink_callback(cli))
    assert sink is not None, f"{config_name}: no H5OutputSink wired at callbacks"
    # the section is composed + bound onto the sink by Salt2CLI.instantiate_classes
    # (the run-free CLI path); assert it really is the dumb-section path now.
    assert sink._is_dumb_section(), (  # noqa: SLF001
        f"{config_name}: H5OutputSink is not driven by the outputs: section "
        "(the auto-collect path was removed in W34.4d)"
    )

    legacy = _legacy_columns(modules, run_name)
    section = _section_columns(sink, run_name)

    # every NON-deferred stream the legacy writer serialises must be reproduced
    # EXACTLY by the section (same columns, same order, same dtypes). Compare the
    # legacy streams as the authority.
    diffs: list[str] = []
    for stream, legacy_cols in legacy.items():
        section_cols = section.get(stream, [])
        if section_cols != legacy_cols:
            diffs.append(
                f"[{config_name}] stream {stream!r} column schema mismatch\n"
                f"  legacy (TaskWriter):   {legacy_cols}\n"
                f"  section (H5OutputSink): {section_cols}"
            )
    assert not diffs, "SECTION H5 SCHEMA PARITY FAILED:\n" + "\n".join(diffs)
