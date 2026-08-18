"""``salt export``: config + checkpoint -> validated ``.onnx``.

Compiles the ONNX plan, traces via `OnnxAdapter`, writes ``gnn_config``
metadata, and sweep-checks torch vs onnxruntime.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from salt.graph.errors import ConfigError, GraphError, ShapeError
from salt.graph.planner import Plan, compile_plan
from salt.graph.spec import (
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.logging import console
from salt.outputs.sinks.onnx.adapter import OnnxAdapter
from salt.outputs.sinks.onnx.check import CheckResult, check_onnx
from salt.outputs.sinks.onnx.config import (
    ExportConfig,
    reject_declared_outputs,
    resolve_export_config,
    stream_of_input_port,
)
from salt.outputs.sinks.onnx.metadata import build_gnn_config, load_run_metadata, write_metadata

__all__ = [
    "ExportResult",
    "compile_onnx_plan",
    "derive_onnx_sources",
    "export_graph",
    "main",
]

OPSET_VERSION = 20
"""ONNX opset version; beyond 20 needs the dynamo exporter."""


@dataclass
class ExportResult:
    """What `export_graph` produced: the adapter, plan and written artifacts.

    `plan_txt_path` is the rendered ONNX plan table written next to the ``.onnx``
    file — the authoritative view of the graph Athena will run (the static
    ``salt graph plan --mode onnx`` view is dataset-fed and its plan hash
    legitimately differs).
    """

    adapter: OnnxAdapter
    plan: Plan
    export: ExportConfig
    onnx_path: Path
    gnn_config: dict[str, Any]
    plan_txt_path: Path | None = None


def derive_onnx_sources(export: ExportConfig, variables: Mapping[str, Sequence[str]]) -> NestedSpec:
    """Config-only ONNX boundary sources from the export block + `Features` variables.

    Mirrors the dataset boundary the fit plans compile against (``inputs.<stream>``
    global/sequence tensors with ``fields`` from the `Features` declaration;
    per-sequence ``masks.<stream>`` bool pad masks) WITHOUT constructing a dataset —
    export works from config + checkpoint alone.

    Raises
    ------
    ConfigError
        When an export input stream has no `Features` variable list.
    """
    flat: dict[str, TensorSpec] = {}
    for entry in export.inputs:
        stream = stream_of_input_port(entry.port)
        if stream not in variables:
            raise ConfigError(
                f"export input {entry.port!r}: stream {stream!r} has no Features variable "
                "declaration (config: data.modules.features.init_args.variables) — export "
                "input widths derive from it"
            )
        fields = tuple(variables[stream])
        if entry.sequence:
            flat[entry.port] = TensorSpec(
                shape=("B", sym_dim("T", stream), len(fields)), dtype="float32", fields=fields
            )
            flat[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        else:
            flat[entry.port] = TensorSpec(shape=("B", len(fields)), dtype="float32", fields=fields)
    return unflatten_spec(flat)


def _folded_output_table(adapter: OnnxAdapter) -> str:
    """Render the folded export-sink output table from the adapter.

    The "what does Athena see" table, rendered from the adapter's generated
    names/dtypes. One row per flat ONNX output: name, dtype, the
    folded-source note.
    """
    rows = list(zip(adapter.output_names, adapter.output_dtypes, strict=True))
    width = max((len(name) for name, _ in rows), default=1)
    lines = [f"ONNX output manifest (folded conversion nodes, model_name={adapter.model_name}):"]
    lines += [
        f"  {name:<{width}}  {dtype:<7}  folded conversion node (outputs.* leaf)"
        for name, dtype in rows
    ]
    return "\n".join(lines)


def _onnx_export_sink(modules: Mapping[str, GraphModule]) -> Any:
    """Find the folded `OnnxExportSink` node among the model modules, if any.

    An export-node config wires an `OnnxExportSink`
    (``salt.outputs.OnnxExportSink``) into ``model.modules``; it anchors the
    folded conversion leaves (argmax/split/combine) as a terminal node.

    Raises `ConfigError` if more than one `OnnxExportSink` is configured
    (the Athena tuple has a single ordering authority).
    """
    from salt.outputs import OnnxExportSink

    found = [m for m in modules.values() if isinstance(m, OnnxExportSink)]
    if len(found) > 1:
        raise ConfigError(
            "more than one OnnxExportSink is configured — the ONNX output tuple has a single "
            "ordering authority; declare exactly one export sink"
        )
    return found[0] if found else None


def compile_onnx_plan(
    modules: dict[str, GraphModule],
    export: ExportConfig,
    variables: Mapping[str, Sequence[str]],
) -> Plan:
    """Compile the ``Mode.ONNX`` plan demanded by the folded export sink.

    A folded `OnnxExportSink` in ``model.modules`` anchors the conversion
    leaves (argmax/split/combine) as a terminal node — its
    ``declare_io(Mode.ONNX).requires`` are the ONNX sinks.

    Demand pruning removes labels/losses/matcher automatically; a missing
    output producer raises the planner's `ConnectivityError`. A `ShapeError`
    on an ``export.inputs`` port is re-raised with the config address and the
    concrete fix appended.

    Raises
    ------
    ConfigError
        When no folded export sink supplies demand.
    ShapeError
        On a rank/shape mismatch — augmented with the ``export.inputs``
        attribution when the offending key is an export input port.
    """
    export_sink = _onnx_export_sink(modules)
    if export_sink is None:
        raise ConfigError(
            "compile_onnx_plan needs a folded OnnxExportSink in model.modules — "
            "the off-graph reduce manifest was retired. "
            "Declare an OnnxExportSink naming the conversion outputs.* leaves; the "
            "conversion nodes (ClassProbs/SeqClassIndex/MaskFormerObjects/"
            "Combination) own the math inside the traced graph."
        )
    # the folded OnnxExportSink anchors its conversion leaves as a terminal node
    # (the planner pulls them in), so no flat `sinks=` are needed.
    sinks: list[str] = []
    sink_origins: dict[str, str] = {}
    try:
        return compile_plan(
            modules,
            Mode.ONNX,
            sources=derive_onnx_sources(export, variables),
            sinks=sinks,
            sink_origins=sink_origins,
        )
    except ShapeError as err:
        message = str(err)
        for i, entry in enumerate(export.inputs):
            if f"'{entry.port}'" not in message:
                continue
            stream = stream_of_input_port(entry.port)
            fix = (
                "set 'sequence: false' (a global [1, F] vector has no token axis)"
                if entry.sequence
                else f"set 'sequence: true' (and optionally dyn_axis, default 'n_{stream}')"
            )
            raise ShapeError(
                f"{message}\n  this source comes from export.inputs (config: "
                f"export.inputs[{i}], port {entry.port!r}) — if {stream!r} is a "
                f"{'fixed-width global' if entry.sequence else 'variable-length sequence'} "
                f"stream, {fix}"
            ) from err
        raise


def export_graph(
    modules: dict[str, GraphModule],
    export: ExportConfig,
    variables: Mapping[str, Sequence[str]],
    onnx_path: str | Path,
    *,
    outputs: Sequence[Any] = (),
    run_name: str = "salt",
    config: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    ckpt_path: str | Path | None = None,
    seed: int = 42,
) -> ExportResult:
    """Export a bound, weight-loaded module dict to ONNX (the programmatic core).

    Resolves+validates the export block (`model_name` rules apply HERE, never at
    fit), compiles the ONNX plan demanded by the folded `OnnxExportSink`, builds
    the `OnnxAdapter` (torch-math forced), traces at ``opset_version=20,
    dynamo=False`` with example inputs sized from the `Features` declaration, and
    writes the ``gnn_config`` metadata. The checker is a separate step
    (`check_onnx` — the CLI runs it by default).

    Parameters
    ----------
    modules : dict[str, GraphModule]
        Bound module instances carrying the weights to export.
    export : ExportConfig
        The parsed (unresolved is fine) export block — export-only half; a
        config-declared ``outputs`` section raises.
    variables : Mapping[str, Sequence[str]]
        Per-stream `Features` variable lists.
    onnx_path : str | Path
        Output ``.onnx`` path.
    outputs : Sequence[Any]
        RETIRED — must be empty (the folded `OnnxExportSink` supplies the
        output demand); a non-empty value raises.
    run_name : str, optional
        The run ``name:`` — the default `model_name` source, by default
        ``"salt"``.
    config : Mapping[str, Any] | None, optional
        Resolved run config embedded in the metadata, by default ``{}``.
    run_metadata : Mapping[str, Any] | None, optional
        The run-dir ``metadata.yaml`` content, by default ``{}``.
    ckpt_path : str | Path | None, optional
        Recorded in the metadata, by default None (fixture exports).
    seed : int, optional
        Seed for the example-input draw, by default 42.

    Returns
    -------
    ExportResult
        The adapter (reusable as the checker reference), plan and written
        metadata.
    """
    # the folded OnnxExportSink supplies the entire output demand; `outputs` is
    # always empty here — only the export-only half (model_name/inputs) is
    # resolved below.
    if outputs:
        raise ConfigError(
            "export_graph no longer accepts a reduce-manifest `outputs=` list — the off-graph "
            "reduce manifest was retired. Wire an OnnxExportSink naming the "
            "conversion outputs.* leaves instead."
        )
    resolved = resolve_export_config(export, run_name)
    plan = compile_onnx_plan(modules, resolved, variables)
    feature_fields = {
        entry.port: tuple(variables[stream_of_input_port(entry.port)]) for entry in resolved.inputs
    }
    adapter = OnnxAdapter(plan, resolved, feature_fields)
    adapter.eval()
    adapter.float()  # export in full precision
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        # bool-mask gathers (union-find fake-track strip, zero-token pooling) trace
        # through aten::index with bool indices, which torch warns about
        # ('indices of type Byte ... incorrect ONNX graph'). The post-export sweep
        # checker proves the traced graph correct, so this is suppressed deliberately.
        warnings.filterwarnings(
            "ignore", message=r".*aten::index operator with indices of type Byte.*"
        )
        torch.onnx.export(
            adapter,
            adapter.example_inputs(),
            str(onnx_path),
            opset_version=OPSET_VERSION,
            input_names=adapter.input_names,
            output_names=adapter.output_names,
            dynamic_axes=adapter.dynamic_axes,
            dynamo=False,
        )
    gnn_config = build_gnn_config(
        resolved,
        variables,
        adapter.output_names,
        config or {},
        run_metadata or {},
        ckpt_path,
        plan.plan_hash,
    )
    write_metadata(onnx_path, gnn_config, str(resolved.model_name))
    # the authoritative rendering of the graph Athena will run (the static
    # `salt graph plan --mode onnx` view is dataset-fed and may differ); the
    # output manifest is appended.
    from salt.graph.render import plan_table

    plan_txt_path = onnx_path.parent / "plan_onnx.txt"
    # the folded export-sink's output table renders from the adapter's generated
    # names/dtypes.
    output_table = _folded_output_table(adapter)
    plan_txt_path.write_text(plan_table(plan) + "\n\n" + output_table + "\n")
    return ExportResult(
        adapter=adapter,
        plan=plan,
        export=resolved,
        onnx_path=onnx_path,
        gnn_config=gnn_config,
        plan_txt_path=plan_txt_path,
    )


# ---------------------------------------------------------------------------
# the salt export CLI (dispatched from salt.main)
# ---------------------------------------------------------------------------


def _parse_args(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse the ``salt export`` CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="salt export",
        description="Export a trained salt model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="checkpoint path (required except with --manifest)",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="print the writer-derived output manifest (full ONNX names, dtypes, reduces) "
        "and exit — no checkpoint needed",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        default=None,
        help="saved run config; inferred at <ckpt>/../../config.yaml when omitted "
        "(v1 contract, to_onnx.py:629-631). Repeatable: later configs deep-merge on top "
        "(the fit semantics) — e.g. stack an export-block override file onto a run "
        "config trained without one",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="export model name override (no '_'/'-'); defaults to export.model_name, "
        "then the sanitised run name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .onnx path; defaults to network.onnx next to the run config "
        "(v1 contract, to_onnx.py:708-709)",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="overwrite an existing ONNX file"
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override applied on the run-free parse (repeatable), e.g. "
        "--set outputs.onnx_export.init_args.model_name=GN2v2",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the torch-vs-ONNX sweep checker after export (v1 always checks)",
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="random draws per sequence length (check.py:177)"
    )
    parser.add_argument(
        "--max-length", type=int, default=40, help="sweep lengths 0..N-1 (check.py:176)"
    )
    parser.add_argument(
        "--float-atol",
        type=float,
        default=1e-4,
        help="checker float atol/rtol (v1 global bar 1e-4; gates use 1e-6)",
    )
    return parser.parse_args(args)


def _run_free_cli(config_paths: Sequence[Path], set_overrides: Sequence[str]) -> Any:
    """Parse the run config(s) through the REAL salt surface, run-free.

    Multiple configs deep-merge left-to-right (the ``salt fit`` stacking
    semantics) — the supported way to add an ``export:`` block to a run
    config trained without one. Returns the run-free `SaltCLI`
    (``cli.model``/``cli.datamodule`` constructed, nothing executed, no
    data touched). Raises `ConfigError` when the parse fails (with the
    ``--set`` hint, mirroring ``salt graph``).
    """
    from salt.config_utils import disable_logger_in_config
    from salt.main import SaltCLI

    args: list[str] = []
    for path in config_paths:
        # a saved run config.yaml carries fit/test-only top-level keys (`ckpt_path`,
        # lightning 2.6.5+ `weights_only`) the top-level parser rejects, plus a
        # default-ON CometLogger that fails keyless. Strip both + disable the logger
        # via a /tmp copy; the embedded ONNX payload is still read from the ORIGINAL
        # path in _export_from_cli, so this affects parsing only.
        args.extend(["--config", disable_logger_in_config(str(path))])
    for entry in set_overrides:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        with warnings.catch_warnings():
            # programmatic argv triggers Lightning's 'args parameter is intended...'
            # warning — filtered exactly as the salt fit/test entry point does.
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            return SaltCLI(args=args, run=False)
    except SystemExit as err:
        raise ConfigError(
            f"run config(s) {[str(p) for p in config_paths]} failed to parse through the "
            f"salt surface (parser exit {err.code}; the parser error is printed above). "
            "Supply required init_args data-free via --set if needed"
        ) from err


def _features_variables(cli: Any) -> dict[str, list[str]]:
    """The `Features` variable declaration from the parsed data modules.

    Returns ``{stream: ordered variable list}``. Raises `ConfigError` when
    the config has no `Features` processor.
    """
    from salt.data.processors.features import Features

    for module in cli.datamodule.modules.values():
        if isinstance(module, Features):
            return dict(module.variables)
    raise ConfigError(
        "the run config declares no salt.data.Features module — export input widths "
        "derive from its variable lists"
    )


_ALIAS_KEYS = ("model_name", "inputs", "track_selection", "rename", "combine")
"""The export-contract keys the deprecated top-level ``export:`` block still fills."""

_TRACK_SELECTION_DEFAULT = ExportConfig().track_selection
"""The unset sentinel for the one non-empty-defaulted alias key."""

_ALIAS_MERGED = "_salt_export_alias_merged"
"""Marks a sink the alias already folded onto, so a second merge is a no-op
rather than a spurious both-homes error."""


def _alias_is_set(key: str, value: Any) -> bool:
    """Whether an export-contract field carries a user-set value, not its default."""
    if key == "track_selection":
        return value != _TRACK_SELECTION_DEFAULT
    return bool(value)


def _merge_export_alias(cli: Any, export_sink: Any) -> None:
    """Fold a parsed top-level ``export:`` block onto the ONNX sink.

    The block is a deprecated alias for the sink's own export keys: each one it
    sets fills a field the sink LEFT UNSET, and a key carried by both homes
    raises `ConfigError` naming it and both homes rather than picking a winner
    silently. A declared ``export.outputs`` stays the hard error it is. A
    no-op when the config declares no block.
    """
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - the main.py _get precedent
    if export_cfg is None or getattr(export_sink, _ALIAS_MERGED, False):
        return
    reject_declared_outputs(export_cfg.outputs)
    warnings.warn(
        "the top-level `export:` block is deprecated — its keys "
        f"({', '.join(_ALIAS_KEYS)}) are now init_args of the ONNX sink, e.g.\n"
        "  outputs:\n"
        "    onnx_export:\n"
        "      class_path: salt.outputs.OnnxExportSink\n"
        "      init_args: {model_name: ..., inputs: [...]}\n"
        "Move the block onto the sink; it is read for one deprecation window and a key "
        "set in both homes is an error.",
        DeprecationWarning,
        stacklevel=3,
    )
    for key in _ALIAS_KEYS:
        block_value = getattr(export_cfg, key, None)
        if not _alias_is_set(key, block_value):
            continue
        if _alias_is_set(key, getattr(export_sink, key, None)):
            raise ConfigError(
                f"export.{key} is set in BOTH homes — the deprecated top-level `export:` "
                f"block and the OnnxExportSink's `{key}:` init_arg. Delete the top-level "
                f"export.{key}; the sink is the export contract."
            )
        setattr(export_sink, key, block_value)
    setattr(export_sink, _ALIAS_MERGED, True)


def _resolve_export_contract(
    cli: Any, export_sink: Any, model_name: str | None = None
) -> ExportConfig:
    """The resolved export contract for a parsed run config.

    The single seam every export-side caller goes through: folds the deprecated
    top-level ``export:`` block onto the sink (`_merge_export_alias`), applies a
    ``-n/--name`` override on top, and resolves the sink's contract against the
    run ``name:``.

    A `ConfigError` propagates from the alias merge when a key is set in both
    homes, and from the sink's own resolution when the contract is incomplete
    or malformed.

    Parameters
    ----------
    cli : Any
        The run-free `SaltCLI` (`_run_free_cli` output).
    export_sink : Any
        The config's `salt.outputs.OnnxExportSink`.
    model_name : str | None, optional
        CLI model-name override, applied after the alias merge, by default None.

    Returns
    -------
    ExportConfig
        The resolved export-only half.
    """
    _merge_export_alias(cli, export_sink)
    if model_name is not None:
        export_sink.model_name = model_name
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    return export_sink.export_config(run_name)


def _cross_check_schema(model: Any, export: ExportConfig, variables: Mapping[str, Any]) -> None:
    """Cross-check config-derived widths/fields against the checkpoint schema.

    The checkpoint's ``salt_core`` payload bound the modules; a config whose
    `Features` lists drifted from the trained widths must fail loudly here
    instead of tracing a width-mismatched graph. Raises `ConfigError`
    naming the drifted stream and both widths.
    """
    schema = getattr(model, "schema", None)
    if schema is None:
        return
    for entry in export.inputs:
        stream = stream_of_input_port(entry.port)
        configured = len(variables[stream])
        stored = schema.widths.get(entry.port)
        if stored is not None and stored != configured:
            raise ConfigError(
                f"export input {entry.port!r}: the config declares {configured} variables but "
                f"the checkpoint schema stores width {stored} — the Features list drifted "
                "since training"
            )


def _print_check_result(result: CheckResult) -> None:
    """Print the checker verdict table: one row per output, floats AND int8."""
    console("-" * 100)
    console(f"ONNX check: {result.n_cases} cases")
    for name, diff in sorted(result.worst_abs_diff.items()):
        console(f"  {name:50s} worst abs diff {diff:.3e}")
    for name, values in sorted(result.int8_distinct.items()):
        # int8 outputs are exact-or-fail (never tolerant); give them a positive
        # verdict row so every declared output is visibly checked
        status = (
            f"int8 exact over {result.n_cases} cases"
            if result.passed
            else "int8 exact-or-fail (see failures)"
        )
        console(f"  {name:50s} {status} (values seen: {values})")
    for failure in result.failures[:20]:
        console(f"  FAIL: {failure}")
    if len(result.failures) > 20:
        console(f"  ... and {len(result.failures) - 20} more failures")
    verdict = "consistent" if result.passed else "INCONSISTENT"
    console(f"Torch and ONNX models are {verdict}.")
    console("-" * 100)


def main(args: Sequence[str] | None = None) -> int:
    """``salt export`` entry point (config + checkpoint -> checked ``.onnx``).

    Returns
    -------
    int
        0 on success, 1 on a config/graph error, an existing output without
        ``--overwrite`` (printed as one actionable line instead of a traceback),
        or a failed check.
    """
    parsed = _parse_args(args)
    if not parsed.manifest and parsed.ckpt_path is None:
        console("salt export: --ckpt_path is required (except with --manifest)", file=sys.stderr)
        return 1
    try:
        if parsed.manifest:
            return _print_manifest_from_cli(parsed)
        result, adapter = _export_from_cli(parsed)
    except GraphError as err:
        console(f"salt.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    except FileExistsError as err:
        console(
            f"{err} Pass -o/--overwrite to replace it, or --output <path> for a new path "
            "(v1 refusal contract, to_onnx.py:710-711).",
            file=sys.stderr,
        )
        return 1
    if parsed.check:
        check = check_onnx(
            adapter,
            result.onnx_path,
            max_length=parsed.max_length,
            trials=parsed.trials,
            float_rtol=parsed.float_atol,
            float_atol=parsed.float_atol,
        )
        _print_check_result(check)
        if not check.passed:
            console(f"removing inconsistent export? NO — kept at {result.onnx_path} for debugging")
            return 1
    console("-" * 100)
    console(f"Done! Saved ONNX model at {result.onnx_path}")
    if result.plan_txt_path is not None:
        console(f"ONNX plan table (the traced graph): {result.plan_txt_path}")
    console("-" * 100)
    return 0


def _resolve_config_paths(parsed: argparse.Namespace) -> list[Path]:
    """The run-config stack: explicit ``-c`` files, or the sibling inference.

    Returns at least one config path; the FIRST is the run config (it
    anchors the default output path, ``metadata.yaml`` lookup and the
    embedded ``config.yaml`` payload). Raises `ConfigError` when no config
    is given and none can be inferred.
    """
    config_paths: list[Path] = list(parsed.config or [])
    if config_paths:
        return config_paths
    if parsed.ckpt_path is None:
        raise ConfigError("salt export --manifest needs a config — pass -c <config.yaml>")
    inferred = parsed.ckpt_path.parents[1] / "config.yaml"
    if not inferred.is_file():
        raise ConfigError(f"could not find a run config at {inferred} — pass --config")
    return [inferred]


# the ONNX output manifest derives from the folded OnnxExportSink's declared
# leaves, found via `salt.cli._static_onnx_export_sink`.


def _print_manifest_from_cli(parsed: argparse.Namespace) -> int:
    """``salt export --manifest``: print the assembled output manifest and exit.

    Returns 0 on success (errors raise `GraphError`, handled by `main`).
    """
    from salt.cli import _static_onnx_export_sink

    config_paths = _resolve_config_paths(parsed)
    cli = _run_free_cli(config_paths, parsed.set_overrides)
    # the manifest derives from the folded OnnxExportSink's declared leaves.
    export_sink = _static_onnx_export_sink(cli)
    if export_sink is None:
        raise ConfigError(
            "config has no OnnxExportSink — the ONNX output manifest is declared by an "
            "OnnxExportSink in the outputs: section, naming the conversion outputs.* "
            "leaves. Add the conversion nodes + the OnnxExportSink."
        )
    resolved = _resolve_export_contract(cli, export_sink, parsed.name)
    if export_sink.model_name is None:
        export_sink.model_name = resolved.model_name
    rows = list(zip(export_sink.output_names(), export_sink.output_dtypes(), strict=True))
    width = max((len(name) for name, _ in rows), default=1)
    console(f"ONNX output manifest (folded conversion nodes, model_name={export_sink.model_name}):")
    for name, dtype in rows:
        console(f"  {name:<{width}}  {dtype:<7}  folded conversion node (outputs.* leaf)")
    return 0


def _export_from_cli(parsed: argparse.Namespace) -> tuple[ExportResult, OnnxAdapter]:
    """The CLI export flow: parse config, derive manifest, load checkpoint, export.

    Returns the export result and the eager checker reference. Raises
    `ConfigError` on a missing config / export sink, an incomplete export
    contract, a config-declared ``export.outputs``, or schema drift;
    `FileExistsError` on an existing output without ``--overwrite``.
    """
    from salt.model.saltmodule import SaltModule

    ckpt_path: Path = parsed.ckpt_path
    config_paths = _resolve_config_paths(parsed)
    config_path = config_paths[0]
    cli = _run_free_cli(config_paths, parsed.set_overrides)
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    variables = _features_variables(cli)
    # the folded OnnxExportSink is the sole ONNX authority — the output tuple AND
    # the export contract (model_name/inputs/rename/combine). Find it on the
    # parsed CLI and fold it into the planning module dict so its declared leaves
    # anchor the ONNX plan demand.
    from salt.cli import _static_onnx_export_sink

    export_sink = _static_onnx_export_sink(cli)
    if export_sink is None:
        raise ConfigError(
            f"config {config_path} has no OnnxExportSink — the ONNX output manifest and the "
            "export contract are declared by an OnnxExportSink in the outputs: section, "
            "naming the conversion outputs.* leaves the folded nodes mint (ClassProbs/"
            "SeqClassIndex/MaskFormerObjects/Combination). Add the conversion nodes + the "
            "OnnxExportSink to the run config, or stack an override file carrying only the "
            f"sink as a second config:\n  salt export --ckpt_path {ckpt_path} "
            f"-c {config_path} -c my_export_sink.yaml"
        )
    resolved = _resolve_export_contract(cli, export_sink, parsed.name)
    # data-less checkpoint load: binds from the stored salt_core schema before the
    # strict state-dict load
    model = SaltModule.load_from_checkpoint(
        ckpt_path,
        modules=cli.model._graph_modules,  # noqa: SLF001 - same-package adapter (cli.py precedent)
        map_location=torch.device("cpu"),
        weights_only=False,  # pytorch 2.6+ flipped this default to True
    )
    if export_sink.model_name is None:
        export_sink.model_name = resolved.model_name
    # fold the export sink into the modules the planner/adapter see
    modules = dict(model._graph_modules)  # noqa: SLF001 - same-package adapter
    if export_sink.name in modules:
        raise ConfigError(
            f"OnnxExportSink name {export_sink.name!r} collides with a model module — rename "
            "the sink's outputs: section key"
        )
    modules[export_sink.name] = export_sink
    _cross_check_schema(model, resolved, variables)
    onnx_path: Path = parsed.output or (config_path.parent / "network.onnx")
    if onnx_path.exists() and not parsed.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")
    console("-" * 100)
    console(f"Converting model to ONNX (model_name={resolved.model_name})...")
    console("-" * 100)
    with open(config_path) as fh:
        config_payload = yaml.safe_load(fh) or {}
    result = export_graph(
        modules,
        resolved,  # already-resolved export-only half (outputs=[] below)
        variables,
        onnx_path,
        outputs=[],
        run_name=run_name,
        config=config_payload,
        run_metadata=load_run_metadata(config_path),
        ckpt_path=ckpt_path,
    )
    return result, result.adapter


if __name__ == "__main__":
    sys.exit(main())
