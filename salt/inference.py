"""``salt inference``: config + checkpoint + (optionally unlabelled) H5 -> the
EXPORT output set, written to H5. Compiles the SAME ``Mode.ONNX`` plan/selection
as ``salt export`` and executes it eagerly per jet through the `OnnxAdapter`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError, GraphError
from salt.graph.spec import Mode
from salt.onnx.adapter import OnnxAdapter
from salt.onnx.config import (
    ExportConfig,
    resolve_export_config,
    stream_of_input_port,
)
from salt.onnx.export import (
    _cross_check_schema,
    _features_variables,
    _run_free_cli,
    compile_onnx_plan,
)

__all__ = ["INFERENCE_OUTPUT", "build_inference_sink", "inference_demand", "main", "run_inference"]

INFERENCE_OUTPUT = "{ckpt_dir}/{ckpt_stem}__inference_{sample}.h5"
"""Default output template — ``__inference_`` so a ``salt test`` eval H5 is never clobbered."""

# WHY eager per-jet (not the batched Lightning test loop): the export-mode graph
# steps assume the Athena calling convention — batch 1, valid tokens only, an
# all-valid pad mask. That is exactly how the eager ONNX-mode execution that
# EXISTS today runs: `OnnxAdapter.forward` (the check_onnx torch reference and
# the goldens generator's static selection both anchor on it). Concretely,
# batched eager ONNX execution is UNSAFE: the seq-classification ONNX branch
# appends a [1, 1, C] zero row (classification.py get_output) and `.squeeze(0)`s
# to [L]; the vertexing/union-find branch `reshape(-1)`s the batch away
# (edge.py get_output). Running the adapter per jet
# guarantees `salt inference` == Athena semantics by construction, at the
# check_onnx tolerance.


def inference_demand(export: ExportConfig) -> list[str]:
    """The dataset-boundary demand of an inference run.

    The export block's positional input ports (+ each sequence stream's pad
    mask) + the ``meta.rows`` row anchor. Alias pseudo-inputs are synthesised
    by the adapter from their source port and consume nothing. Label-free by
    construction: no ``labels.*`` key can appear here, so `Labels` narrows to
    nothing and a label-stripped file reads fine (plan 50 §D / 50a Task 2).
    """
    demand: list[str] = []
    for entry in export.inputs:
        if entry.alias is not None:
            continue
        demand.append(entry.port)
        if entry.sequence:
            demand.append(f"masks.{stream_of_input_port(entry.port)}")
    demand.append("meta.rows")
    return demand


def build_inference_sink(section: Any, output: str | Path | None = None) -> Any:
    """The implicit inference H5 sink over the composed ``outputs:`` section.

    An `H5OutputSink` switched to the EXPORT-mode column selection
    (`use_export_selection`): its columns resolve from
    ``manifest_fields(Mode.ONNX)`` — one single-suffix column per ONNX leaf —
    and the section's `InputCopyWriter`/`PadMaskWriter` contribute their
    copy/mask columns only when their ``modes:`` include ``export``.
    Raises `ConfigError` on an empty/missing section (no WHAT to write).
    """
    from salt.outputs import H5OutputSink  # noqa: PLC0415 - heavy/circular

    if not section:
        raise ConfigError(
            "salt inference needs a top-level `outputs:` section with at least one "
            "export-mode RunTaskOutput — the export selection IS the inference output "
            "set (plan 50 decision 2). Add `modes: [test, export]` (or omit `modes:`) "
            "on the RunTaskOutput to export"
        )
    sink = H5OutputSink(output=str(output) if output is not None else INFERENCE_OUTPUT)
    sink.use_export_selection()
    sink.bind_output_section(section)
    return sink


def _jet_args(adapter: OnnxAdapter, bundle: Bundle, i: int) -> tuple[torch.Tensor, ...]:
    """One jet's positional adapter inputs (the Athena calling convention).

    Globals ``[1, F]``; sequences ``[L_valid, F]`` — padded positions stripped
    via the batch pad mask (Athena feeds valid tokens only; valid tokens are
    the leading rows, the reader's pad layout — enforced per batch by
    `_check_leading_valid`).
    """
    args: list[torch.Tensor] = []
    for entry in adapter._positional:  # noqa: SLF001 - same-package (check.py precedent)
        x = bundle.get(entry.port)
        if entry.sequence:
            mask = bundle.get(f"masks.{stream_of_input_port(entry.port)}")[i]
            args.append(x[i][~mask])
        else:
            args.append(x[i : i + 1])
    return tuple(args)


def _check_leading_valid(mask: Any, stream: str) -> None:
    """Reject a batch whose valid tokens are not the leading rows.

    The per-token H5 writeback places each jet's values at the LEADING
    positions and writes the file's pad mask verbatim (`_consume_batch`), so
    the two agree only when every valid token precedes every padded one (the
    dumper/reader pad layout). A file with interior padded tokens would
    otherwise yield silently value/mask-misaligned per-token columns.
    """
    if bool((mask[..., :-1] & ~mask[..., 1:]).any()):
        raise ConfigError(
            f"masks.{stream}: valid tokens are not the leading rows (a padded token "
            "precedes a valid one) — salt inference writes per-token values at "
            "leading positions against the file's pad mask, so this file would "
            "produce misaligned per-token columns. Re-order each jet's tokens "
            "valid-first (the training-dataset-dumper layout)"
        )


def _column_plan(sink: Any, export_sink: Any) -> list[tuple[Any, str, bool]]:
    """Pair each resolved H5 column with its ONNX tuple name and axis.

    Both the export-selection sink columns and the `OnnxExportSink` leaves
    derive from the SAME section ``manifest_fields(Mode.ONNX)`` walk, so they
    are 1:1 by leaf key; a mismatch raises `ConfigError` (never a silent drop).
    """
    prefix = export_sink.resolved_model_name()
    leaves = {leaf.key: leaf for leaf in export_sink.leaves}
    plan: list[tuple[Any, str, bool]] = []
    for col in sink.columns:
        leaf = leaves.get(col.key)
        if leaf is None or len(col.suffixes) != 1:
            raise ConfigError(
                f"inference H5 column {col.key!r} has no 1:1 ONNX tuple counterpart — "
                f"export leaves are {sorted(leaves)} (plan 50 Phase D invariant: the H5 "
                "selection IS the export selection)"
            )
        plan.append((col, f"{prefix}_{leaf.suffixes[0]}", bool(leaf.per_token)))
    return plan


def _consume_batch(
    sink: Any,
    adapter: OnnxAdapter,
    column_plan: list[tuple[Any, str, bool]],
    batch: dict[str, Any],
) -> None:
    """Run the adapter per jet over one dataset batch and feed the sink.

    Re-batches the per-jet named outputs into ``outputs.*`` leaves the H5
    sink packs: global scalars stack to ``[B]``; per-token values are placed
    into a zero-padded ``[B, L]`` block (the sink re-expands to the file
    length — padded positions read 0, with the pad-mask column marking them).
    Leading placement is checked, not assumed: `_check_leading_valid` rejects
    any batch whose valid tokens are not the leading rows.
    """
    bundle_in = Bundle(dict(batch))
    seq_streams = {
        stream_of_input_port(entry.port)
        for entry in adapter._positional  # noqa: SLF001 - same-package (check.py precedent)
        if entry.sequence
    }
    for stream in sorted(seq_streams | set(sink._mask_streams)):  # noqa: SLF001 - sink drive
        _check_leading_valid(bundle_in.get(f"masks.{stream}"), stream)
    rows = bundle_in.get("meta.rows")
    n = int(rows[1]) - int(rows[0])
    per_col: dict[str, list[torch.Tensor]] = {col.key: [] for col, _, _ in column_plan}
    for i in range(n):
        with torch.no_grad():
            outputs = adapter(*_jet_args(adapter, bundle_in, i))
        named = dict(zip(adapter.output_names, outputs, strict=True))
        for col, onnx_name, _ in column_plan:
            per_col[col.key].append(named[onnx_name])
    out = Bundle()
    out.set("meta.rows", rows)
    for col, _, per_token in column_plan:
        vals = per_col[col.key]
        if per_token:
            length = bundle_in.get(f"inputs.{col.stream}").shape[1]
            block = torch.zeros((n, length), dtype=vals[0].dtype)
            for i, v in enumerate(vals):
                block[i, : v.shape[0]] = v
            out.set(col.key, block)
        else:
            out.set(col.key, torch.stack([v.reshape(()) for v in vals]))
    for stream in sink._mask_streams:  # noqa: SLF001 - same-package sink drive
        out.set(f"outputs.{stream}.mask", bundle_in.get(f"masks.{stream}"))
    sink.consume(out)


def run_inference(
    config_paths: Sequence[Path],
    ckpt_path: Path,
    test_file: str | Path,
    *,
    output: str | Path | None = None,
    set_overrides: Sequence[str] = (),
    batch_size: int | None = None,
) -> Path:
    """The programmatic core of ``salt inference``.

    Parses the run config through the real salt surface (run-free — the
    Phase B implicit-sink wiring runs, so the export-mode `OnnxExportSink` is
    discovered exactly as ``salt export`` finds it), loads the checkpoint,
    compiles the ``Mode.ONNX`` plan through `compile_onnx_plan`, and executes
    the `OnnxAdapter` eagerly per jet over the test file, writing the named
    export outputs through the export-selection `H5OutputSink`.

    Parameters
    ----------
    config_paths : Sequence[Path]
        The run config stack (deep-merged left-to-right, the fit semantics).
    ckpt_path : Path
        Checkpoint to load (required — inference names its output after it).
    test_file : str | Path
        The H5 to run over. May be LABEL-FREE: the dataset demand is derived
        from ``export.inputs`` only, so no label dataset is demanded — a
        label-stripped file runs green. Demand-free is not redaction: an
        export-mode ``InputCopyWriter`` copies the source fields it selects
        verbatim, labels included when the file carries them.
    output : str | Path | None, optional
        Output H5 path; default `INFERENCE_OUTPUT` next to the checkpoint.
    set_overrides : Sequence[str], optional
        ``KEY=VALUE`` config overrides (the ``--set`` flag), by default ().
    batch_size : int | None, optional
        Read-slab size; default the datamodule's configured ``batch_size``.

    Returns
    -------
    Path
        The written H5 path.

    Raises
    ------
    ConfigError
        On a missing export block / export-mode selection, an explicit-leaf
        (MaskFormer escape hatch) config, or any sink schema error.
    """
    from salt.cli import _static_onnx_export_sink  # noqa: PLC0415 - heavy/circular
    from salt.model.saltmodule import SaltModule  # noqa: PLC0415 - heavy/circular

    overrides = [f"data.test_file={test_file}", *set_overrides]
    cli = _run_free_cli(config_paths, overrides)
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - main.py precedent
    if export_cfg is None:
        raise ConfigError(
            f"config {config_paths[0]} has no export: block — salt inference feeds the "
            "model through the Athena input contract (export.inputs), so declare it (or "
            "stack an override file carrying only the export: block as a second -c)"
        )
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    export_sink = _static_onnx_export_sink(cli)
    if export_sink is None:
        raise ConfigError(
            "config assembles no ONNX export selection — salt inference's task columns "
            "ARE the export output set (plan 50 decision 2). Give at least one outputs: "
            "section RunTaskOutput `export` in its modes: list (or omit modes: for both)"
        )
    if export_sink._explicit_leaves:  # noqa: SLF001 - same-package scope guard
        raise ConfigError(
            "salt inference supports the outputs:-section export selection only — this "
            "config declares explicit OnnxExportLeaf entries (the MaskFormer object-reduce "
            "escape hatch), which have no H5 counterpart here (plan 50 Phase D scope)"
        )
    variables = _features_variables(cli)
    resolved = resolve_export_config(export_cfg, run_name)
    if export_sink.model_name is None:
        export_sink.model_name = resolved.model_name
    model = SaltModule.load_from_checkpoint(
        ckpt_path,
        modules=cli.model._graph_modules,  # noqa: SLF001 - export.py precedent
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    _cross_check_schema(model, resolved, variables)
    modules = dict(model._graph_modules)  # noqa: SLF001 - export.py precedent
    if export_sink.name in modules:
        raise ConfigError(
            f"OnnxExportSink name {export_sink.name!r} collides with a model module — rename "
            "the callbacks key (design §2.2)"
        )
    modules[export_sink.name] = export_sink
    plan = compile_onnx_plan(modules, resolved, variables)
    feature_fields = {
        entry.port: tuple(variables[stream_of_input_port(entry.port)]) for entry in resolved.inputs
    }
    adapter = OnnxAdapter(plan, resolved, feature_fields)
    adapter.eval()
    adapter.float()  # the export_graph precision contract
    # the dataset: the CLI datamodule with the INFERENCE demand — export input
    # ports + pad masks + meta.rows, never labels (Labels narrows to nothing,
    # so a label-stripped file binds and reads green; 50a Task 2).
    dm = cli.datamodule
    dm.set_sinks({Mode.TEST: inference_demand(resolved)})
    dm.setup("test")
    dset = dm.test_dset
    sink = build_inference_sink(cli.model._output_section, output)  # noqa: SLF001 - section home
    # the sink's lifecycle is driven directly (no Lightning test loop runs
    # here); it reads only these trainer facts, duck-typed:
    trainer = SimpleNamespace(
        lightning_module=model,
        datamodule=dm,
        ckpt_path=str(ckpt_path),
        num_test_batches=None,
    )
    sink.open_schema(trainer)
    try:
        column_plan = _column_plan(sink, export_sink)
        total = len(dset)
        step = batch_size or dm.batch_size
        print("-" * 100)
        print(f"salt inference: {total:,} rows from {test_file} (export selection, eager)")
        for start in range(0, total, step):
            stop = min(start + step, total)
            _consume_batch(sink, adapter, column_plan, dset[np.s_[start:stop]])
        sink.flush()
    finally:
        sink.close_if_open()
    assert sink.output_path is not None
    return sink.output_path


def _parse_args(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse the ``salt inference`` CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="salt inference",
        description=(
            "Label-free inference (labels are never demanded, so label-stripped files "
            "run green; export-mode InputCopyWriter columns still pass source fields "
            "through verbatim, labels included): write the EXPORT output set to H5 "
            "(plan 50). Compiles the same Mode.ONNX selection as salt export and "
            "executes it eagerly per jet — offline inference == Athena semantics by "
            "construction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", type=Path, required=True, help="checkpoint path")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        default=None,
        help="run config; inferred at <ckpt>/../../config.yaml when omitted. Repeatable: "
        "later configs deep-merge on top (the fit semantics)",
    )
    parser.add_argument(
        "--data.test_file",
        dest="test_file",
        type=Path,
        required=True,
        help="H5 file to run over — labelled or label-free (labels are never demanded)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output H5 path; defaults to {ckpt_dir}/{ckpt_stem}__inference_{sample}.h5",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="read-slab size (default: the datamodule's configured batch_size)",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override applied on the run-free parse (repeatable), e.g. "
        "--set data.num_test=1000",
    )
    return parser.parse_args(args)


def _resolve_config_paths(parsed: argparse.Namespace) -> list[Path]:
    """The run-config stack: explicit ``-c`` files, or the ``salt export`` sibling
    inference (``<ckpt>/../../config.yaml``); raises `ConfigError` when neither
    resolves.
    """
    if parsed.config:
        return list(parsed.config)
    inferred = parsed.ckpt_path.parents[1] / "config.yaml"
    if not inferred.is_file():
        raise ConfigError(f"could not find a run config at {inferred} — pass --config")
    return [inferred]


def main(args: Sequence[str] | None = None) -> int:
    """``salt inference`` entry point.

    Returns
    -------
    int
        0 on success, 1 on a config/graph error (printed as one clean block).
    """
    parsed = _parse_args(args)
    for entry in parsed.set_overrides:
        if "=" not in entry:
            print(
                f"salt inference: --set entries must be KEY=VALUE, got {entry!r}", file=sys.stderr
            )
            return 1
    try:
        out = run_inference(
            _resolve_config_paths(parsed),
            parsed.ckpt_path,
            parsed.test_file,
            output=parsed.output,
            set_overrides=parsed.set_overrides,
            batch_size=parsed.batch_size,
        )
    except GraphError as err:
        print(f"salt.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    print("-" * 100)
    print(f"Done! Wrote inference H5 at {out}")
    print("-" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
