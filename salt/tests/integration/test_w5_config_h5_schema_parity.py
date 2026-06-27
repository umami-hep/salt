"""W5.2 per-config gate — auto-collect H5 column schema == legacy TaskWriter schema.

Plan 31 W5.2: each migrated config's auto-collect `H5OutputSink` must reproduce the
legacy `TaskWriter` H5 column schema (names + dtype + ORDER) for the families it
serialises — classification (global + per-token probs) + regression. The legacy
schema is `task.output_names(run_name)` per task head (the authority); the
auto-collect schema is the sink's resolved `OutputColumn` table. This is a STATIC,
data-free gate (no run): it loads each config through the run-free CLI, resolves the
auto-collect columns, and asserts per-stream column-name + dtype equality against the
legacy `task.output_names` of every NON-DEFERRED head.

DEFERRED families (excluded, by design — no H5 conversion producer in W5): vertexing
(VertexIndex; the head demand-prunes from TEST) and MaskFormer (W6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.graph.spec import KEY_SEP
from salt.core.main import CONFIG_DIR
from salt.core.onnx.export import _run_free_cli

pytestmark = pytest.mark.cpu_always

# the W5.2 auto-collect H5OutputSink schema gate. The production Class-a-prime
# classification/regression/gaussian configs were RELOCATED onto the plan-34
# ``outputs:`` section + dumb sinks in W34.4b (the auto-collect producer path no
# longer applies to them — their section H5 schema is gated byte-for-byte against
# the legacy WriterCallback by the cutover34 tests, test_w34_outputs_section.py /
# test_w34_regression_cutover.py / test_w34_gaussian_cutover.py). The only configs
# still on the W5 auto-collect path are the PRESERVED oracle configs ``regression``
# and ``regression_gaussian`` (the parity oracles the cutover34 tests compare
# AGAINST — they keep the RegressionDescaleOp/Regression producers + auto-collect
# sink); this gate continues to guard their auto-collect schema == legacy.
_NORM = "model.modules.norm.init_args.norm_dict=unused.yaml"
MIGRATED = {
    "regression": [_NORM],
    "regression_gaussian": [_NORM],
}

# heads whose H5 eval column is a DEFERRED family (no W5 conversion producer):
# their columns are excluded from the parity comparison (documented deferral).
_DEFERRED_TASK_TYPES = {"VertexingTaskModule"}


def _legacy_columns(modules, run_name: str) -> dict[str, list[tuple[str, str]]]:
    """The legacy TaskWriter H5 schema per stream: {stream: [(column, dtype), ...]}.

    Mirrors `TaskWriter.columns`: each task head's `output_names(run_name)`, grouped
    by stream in module declaration order. Deferred (vertexing) heads excluded.
    """
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


def _auto_columns(sink, run_name: str) -> dict[str, list[tuple[str, str]]]:
    """The auto-collect H5OutputSink schema per stream: {stream: [(column, dtype), ...]}."""
    out: dict[str, list[tuple[str, str]]] = {}
    for col in sink._resolve_columns(run_name):  # noqa: SLF001
        stream = col.key.split(KEY_SEP)[1]
        for column in col.column_names(run_name):
            out.setdefault(stream, []).append((column, col.dtype))
    return out


@pytest.mark.parametrize("config_name", sorted(MIGRATED))
def test_auto_collect_h5_schema_matches_legacy_taskwriter(config_name):
    """The migrated config's auto-collect H5 schema == legacy TaskWriter schema (per stream)."""
    from salt.core.cli import _as_sink_node, _static_writer_sink_callback

    config = CONFIG_DIR / f"{config_name}.yaml"
    assert config.is_file(), f"missing config {config}"
    cli = _run_free_cli([config], MIGRATED[config_name])
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001
    modules = dict(cli.model._graph_modules)  # noqa: SLF001
    sink = _as_sink_node(_static_writer_sink_callback(cli))
    assert sink is not None, f"{config_name}: no H5OutputSink wired at callbacks"
    sink.bind_model_modules(modules)

    legacy = _legacy_columns(modules, run_name)
    auto = _auto_columns(sink, run_name)

    # every NON-deferred stream the legacy writer serialises must be reproduced
    # EXACTLY by auto-collect (same columns, same order, same dtypes). Auto-collect
    # may also carry streams the legacy writer didn't (it shouldn't here), so compare
    # the legacy streams as the authority.
    diffs: list[str] = []
    for stream, legacy_cols in legacy.items():
        auto_cols = auto.get(stream, [])
        if auto_cols != legacy_cols:
            diffs.append(
                f"[{config_name}] stream {stream!r} column schema mismatch\n"
                f"  legacy (TaskWriter): {legacy_cols}\n"
                f"  auto   (H5OutputSink): {auto_cols}"
            )
    assert not diffs, "W5.2 H5 SCHEMA PARITY FAILED:\n" + "\n".join(diffs)
