"""``salt2 export``: config + checkpoint -> validated ``.onnx`` (design §7; M4.5).

The exporter compiles the ``Mode.ONNX`` plan from CONFIG-DERIVED sources
("no data needed", design §7.1 — boundary specs mirror the dataset
`Features` declaration instead of touching the H5), demands exactly the
ports of the WRITER-DERIVED output manifest (the M4.5 unified manifest:
``WriterCallback.onnx_manifest`` assembles the writers' `ExportOutput`
declarations — pruning removes labels/losses/the writers' own TEST role
automatically), wraps the plan in the traceable `OnnxAdapter`, traces with
``torch.onnx.export(opset_version=20, dynamo=False)`` (v1
``to_onnx.py:714-721``), writes the v1-bit-compatible ``gnn_config``
metadata, and sweep-checks torch vs onnxruntime (design §7.6).

Programmatic surface (used by the fixture-driven gates, no checkpoint/CLI
required): `compile_onnx_plan` + `export_graph` on a bound, weight-loaded
module dict, with the manifest passed explicitly (``outputs=``). CLI
surface: ``salt2 export --ckpt_path <ckpt> [-c config ...]`` (``-c``
repeatable — later files deep-merge on top, e.g. an export-block override
onto a run config trained without one) derives the manifest from the parsed
``writers:`` block; ``salt2 export --manifest -c <config>`` prints the
assembled manifest and exits (no checkpoint needed). Dispatched from
`salt.core.main` exactly like the graph subcommands. ``--run-dir`` is the
design §7 spelling — it lands with the M6 run-dir layout; until then
``--ckpt_path`` + the inferred sibling config reproduces the v1 contract
(``to_onnx.py:629-631``).
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

from salt.core.graph.errors import ConfigError, GraphError, ShapeError
from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import (
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.core.onnx.adapter import OnnxAdapter
from salt.core.onnx.check import CheckResult, check_onnx
from salt.core.onnx.config import (
    ExportConfig,
    resolve_export_config,
    sanitised_model_name,
    stream_of_input_port,
    validate_model_name,
)
from salt.core.onnx.metadata import build_gnn_config, load_run_metadata, write_metadata

__all__ = [
    "ExportResult",
    "compile_onnx_plan",
    "derive_onnx_sources",
    "export_graph",
    "main",
]

OPSET_VERSION = 20
"""v1's opset (``to_onnx.py:716``) — beyond 20 needs the dynamo exporter."""


@dataclass
class ExportResult:
    """What `export_graph` produced: the adapter, plan and written artifacts.

    `plan_txt_path` is the rendered ONNX plan table written next to the
    ``.onnx`` file (the M3 run-dir ``plan_<mode>.txt`` contract extended to
    export, design §4.4) — the AUTHORITATIVE view of the graph Athena will
    run (the static ``salt2 graph plan --mode onnx`` view is the dataset-fed
    approximation; its plan hash legitimately differs).
    """

    adapter: OnnxAdapter
    plan: Plan
    export: ExportConfig
    onnx_path: Path
    gnn_config: dict[str, Any]
    plan_txt_path: Path | None = None


def derive_onnx_sources(export: ExportConfig, variables: Mapping[str, Sequence[str]]) -> NestedSpec:
    """Config-only ONNX boundary sources from the export block + `Features` variables.

    Mirrors the dataset boundary the fit plans compile against
    (``inputs.<stream>`` ``("B", F)`` global / ``("B", "T:<stream>", F)``
    sequence with ``fields`` from the `Features` declaration; per-sequence
    ``masks.<stream>`` bool pad masks) WITHOUT constructing a dataset —
    export works from config + checkpoint alone (design §7.1; v1 parity:
    ``to_onnx`` never reads data).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.

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
                "input widths derive from it (design §7.1)"
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
    """Render the folded export-sink output table from the adapter (plan-29 W2 plan_onnx.txt).

    The folded-path counterpart to `manifest_table`: a pure-folded config carries
    no legacy ``resolved.outputs`` manifest, so the "what does Athena see" table is
    rendered from the adapter's generated names/dtypes (the OnnxExportSink's output
    table). One row per flat ONNX output: name, dtype, the folded-source note.

    Returns
    -------
    str
        The rendered output table.
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
    """Find the folded `OnnxExportSink` node among the model modules, if any (R8 dispatch).

    The plan-29 W2 hybrid: an export-NODE config wires an `OnnxExportSink`
    (``salt.core.outputs.OnnxExportSink``) into ``model.modules``; it anchors the
    folded conversion leaves (argmax/split/combine) as a terminal node, while any
    legacy reduce outputs (union_find/maskformer) still ride ``export.outputs``.
    A config with no such node is the pure legacy path (unchanged).

    Returns
    -------
    OnnxExportSink | None
        The single export sink, or None for a pure legacy-reduce config.

    Raises
    ------
    ConfigError
        When more than one `OnnxExportSink` is configured (the Athena tuple has
        one ordering authority).
    """
    from salt.core.outputs import OnnxExportSink  # noqa: PLC0415 - heavy/circular

    found = [m for m in modules.values() if isinstance(m, OnnxExportSink)]
    if len(found) > 1:
        raise ConfigError(
            "more than one OnnxExportSink is configured — the ONNX output tuple has a single "
            "ordering authority; declare exactly one export sink (design §6.3)"
        )
    return found[0] if found else None


def compile_onnx_plan(
    modules: dict[str, GraphModule],
    export: ExportConfig,
    variables: Mapping[str, Sequence[str]],
) -> Plan:
    """Compile the ``Mode.ONNX`` plan demanded by the export sinks (folded + legacy, R8).

    Two demand sources, dispatched per config (plan-29 W2 hybrid, design §6.4 /
    R8): a folded `OnnxExportSink` in ``model.modules`` anchors the conversion
    leaves (argmax/split/combine) as a terminal node — its
    ``declare_io(Mode.ONNX).requires`` are the ONNX sinks; any LEGACY reduce
    outputs (union_find/maskformer, NOT folded in W2) still ride
    ``export.outputs`` ports. A config may MIX both without drift; a pure-legacy
    config (the W0 oracle fixtures) has no export sink and uses
    ``[out.port for out in export.outputs]`` EXACTLY as before — byte-identical
    plan.

    `export` must carry the legacy manifest (`attach_manifest`) UNLESS a folded
    export sink supplies the demand. Demand pruning removes labels/losses/matcher
    automatically (design §7); a missing output producer raises the planner's
    §4.1-quality `ConnectivityError`. A `ShapeError` on an ``export.inputs`` port
    is re-raised with the config address and the concrete fix appended.

    Returns
    -------
    Plan
        The frozen ONNX plan.

    Raises
    ------
    ConfigError
        When neither a legacy manifest nor a folded export sink supplies demand.
    ShapeError
        On a rank/shape mismatch — augmented with the ``export.inputs``
        attribution when the offending key is an export input port.
    """
    export_sink = _onnx_export_sink(modules)
    if export_sink is None:
        raise ConfigError(
            "compile_onnx_plan needs a folded OnnxExportSink in model.modules (plan-29 W4) — "
            "the off-graph reduce manifest (WriterCallback.onnx_manifest + attach_manifest) "
            "was retired. Declare an OnnxExportSink naming the conversion outputs.* leaves; the "
            "conversion nodes (ClassProbs/SeqClassIndex/VertexUnionFind/MaskFormerObjects/"
            "Combination) own the math inside the traced graph (design §4.2/§6)."
        )
    # the folded OnnxExportSink anchors ALL its conversion leaves as a terminal
    # node (the planner keeps it via _demand_closure/_is_terminal_consumer +
    # pulls in the folded conversion nodes), so no flat `sinks=` are needed.
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
                f"stream, {fix} (design §5.1)"
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

    Resolves+validates the export block (`model_name` rules apply HERE,
    never at fit — design §7), attaches the writer-derived output manifest
    (M4.5: `outputs` is the assembled ``WriterCallback.onnx_manifest`` —
    or a hand-built `ExportOutput` list in fixture code), compiles the
    ONNX plan, builds the `OnnxAdapter` (torch-math forced via the
    ``set_export_mode`` protocol), traces at ``opset_version=20,
    dynamo=False`` with example inputs sized from the `Features`
    declaration, and writes the ``gnn_config`` metadata. The checker is
    the caller's separate step (`check_onnx` — the CLI runs it by
    default).

    Parameters
    ----------
    modules : dict[str, GraphModule]
        Bound module instances carrying the weights to export.
    export : ExportConfig
        The parsed (unresolved is fine) export block — export-only half;
        a config-declared ``outputs`` section raises the M4.5 migration
        error.
    variables : Mapping[str, Sequence[str]]
        Per-stream `Features` variable lists.
    onnx_path : str | Path
        Output ``.onnx`` path.
    outputs : Sequence[ExportOutput]
        The writer-derived output manifest (``WriterCallback.
        onnx_manifest``), in manifest order.
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
        Seed for the example-input draw (v1 seeds 42 at module import,
        ``to_onnx.py:22``), by default 42.

    Returns
    -------
    ExportResult
        The adapter (reusable as the checker reference), plan and written
        metadata.
    """
    # plan-29 W4: the folded OnnxExportSink supplies the entire output demand —
    # the off-graph reduce manifest is retired, so `outputs` is always empty and
    # only the export-only half (model_name/inputs) is resolved here.
    if outputs:
        raise ConfigError(
            "export_graph no longer accepts a reduce-manifest `outputs=` list — the off-graph "
            "reduce manifest was retired at plan-29 W4. Wire an OnnxExportSink naming the "
            "conversion outputs.* leaves instead (design §4.2/§6)."
        )
    resolved = resolve_export_config(export, run_name)
    plan = compile_onnx_plan(modules, resolved, variables)
    feature_fields = {
        entry.port: tuple(variables[stream_of_input_port(entry.port)]) for entry in resolved.inputs
    }
    adapter = OnnxAdapter(plan, resolved, feature_fields)
    adapter.eval()
    adapter.float()  # v1 exports in full precision (to_onnx.py:669,699)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        # the v1-verbatim bool-mask gathers (union-find fake-track strip,
        # zero-token pooling) trace through aten::index with bool indices;
        # torch warns 'Exporting aten::index operator with indices of type
        # Byte ... incorrect ONNX graph' on them. v1's exporter emits the
        # SAME warning on the SAME ops, and the post-export sweep checker
        # (incl. L=0) proves the traced graph correct — suppressed
        # deliberately so a clean export prints no alarming noise.
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
    # the authoritative rendering of the graph Athena will run — the M3
    # run-dir plan_<mode>.txt contract extended to export (design §4.4); the
    # static `salt2 graph plan --mode onnx` view is dataset-fed and differs.
    # The writer-derived output manifest is appended (amendment §2.3: the
    # generated-artifact answer to "what exactly does Athena see")
    from salt.core.render import plan_table  # noqa: PLC0415 - lazy: keeps onnx import light

    plan_txt_path = onnx_path.parent / "plan_onnx.txt"
    # plan-29 W4: the folded export-sink's output table renders from the adapter's
    # generated names/dtypes (the off-graph reduce manifest is retired).
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
# the salt2 export CLI (dispatched from salt.core.main)
# ---------------------------------------------------------------------------


def _parse_args(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse the ``salt2 export`` CLI arguments.

    Returns
    -------
    argparse.Namespace
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="salt2 export",
        description="Export a trained salt2 model to ONNX (design §7).",
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
        "and exit — no checkpoint needed (M4.5 unified manifest)",
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
        "--set export.model_name=GN2v2",
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
    """Parse the run config(s) through the REAL salt2 surface, run-free.

    Multiple configs deep-merge left-to-right (the ``salt2 fit`` stacking
    semantics) — the supported way to add an ``export:`` block to a run
    config trained without one.

    Returns
    -------
    Salt2CLI
        The run-free CLI (``cli.model`` / ``cli.datamodule`` constructed,
        nothing executed, no data touched).

    Raises
    ------
    ConfigError
        When the parse fails (with the ``--set`` hint, mirroring
        ``salt2 graph``).
    """
    from salt.core.main import Salt2CLI  # noqa: PLC0415 - heavy/circular (main dispatches here)

    args: list[str] = []
    for path in config_paths:
        args.extend(["--config", str(path)])
    for entry in set_overrides:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        with warnings.catch_warnings():
            # programmatic argv triggers Lightning's 'args parameter is
            # intended...' warning — filtered exactly as the salt2 fit/test
            # entry point does in salt.core.main
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            return Salt2CLI(args=args, run=False)
    except SystemExit as err:
        raise ConfigError(
            f"run config(s) {[str(p) for p in config_paths]} failed to parse through the "
            f"salt2 surface (parser exit {err.code}; the parser error is printed above). "
            "Supply required init_args data-free via --set if needed"
        ) from err


def _features_variables(cli: Any) -> dict[str, list[str]]:
    """The `Features` variable declaration from the parsed data modules.

    Returns
    -------
    dict[str, list[str]]
        Stream -> ordered variable list.

    Raises
    ------
    ConfigError
        When the config has no `Features` processor.
    """
    from salt.core.data.processors import Features  # noqa: PLC0415 - heavy/circular

    for module in cli.datamodule.modules.values():
        if isinstance(module, Features):
            return dict(module.variables)
    raise ConfigError(
        "the run config declares no salt.core.data.Features module — export input widths "
        "derive from its variable lists (design §7.1)"
    )


def _cross_check_schema(model: Any, export: ExportConfig, variables: Mapping[str, Any]) -> None:
    """Cross-check config-derived widths/fields against the checkpoint schema.

    The checkpoint's ``salt_core`` payload bound the modules
    (``saltmodule.py:861-893``); a config whose `Features` lists drifted
    from the trained widths must fail loudly here instead of tracing a
    width-mismatched graph (design §7.1 "the serialised schema ... is
    cross-checked against the config").

    Raises
    ------
    ConfigError
        Naming the drifted stream and both widths.
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
                "since training (design §7.1 schema cross-check)"
            )


def _print_check_result(result: CheckResult) -> None:
    """Print the checker verdict table: one row per output, floats AND int8."""
    print("-" * 100)
    print(f"ONNX check: {result.n_cases} cases")
    for name, diff in sorted(result.worst_abs_diff.items()):
        print(f"  {name:50s} worst abs diff {diff:.3e}")
    for name, values in sorted(result.int8_distinct.items()):
        # int8 outputs are exact-or-fail (never tolerant) — give them a
        # positive verdict row so every declared output is visibly checked
        status = (
            f"int8 exact over {result.n_cases} cases"
            if result.passed
            else "int8 exact-or-fail (see failures)"
        )
        print(f"  {name:50s} {status} (values seen: {values})")
    for failure in result.failures[:20]:
        print(f"  FAIL: {failure}")
    if len(result.failures) > 20:
        print(f"  ... and {len(result.failures) - 20} more failures")
    verdict = "consistent" if result.passed else "INCONSISTENT"
    print(f"Torch and ONNX models are {verdict}.")
    print("-" * 100)


def main(args: Sequence[str] | None = None) -> int:
    """``salt2 export`` entry point (config + checkpoint -> checked ``.onnx``).

    Returns
    -------
    int
        0 on success, 1 on a config/graph error, an existing output
        without ``--overwrite`` (the v1 refusal, ``to_onnx.py:710-711``,
        printed as one actionable line instead of a traceback), or a
        failed check.
    """
    parsed = _parse_args(args)
    if not parsed.manifest and parsed.ckpt_path is None:
        print("salt2 export: --ckpt_path is required (except with --manifest)", file=sys.stderr)
        return 1
    try:
        if parsed.manifest:
            return _print_manifest_from_cli(parsed)
        result, adapter = _export_from_cli(parsed)
    except GraphError as err:
        print(f"salt.core.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    except FileExistsError as err:
        print(
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
            print(f"removing inconsistent export? NO — kept at {result.onnx_path} for debugging")
            return 1
    print("-" * 100)
    print(f"Done! Saved ONNX model at {result.onnx_path}")
    if result.plan_txt_path is not None:
        print(f"ONNX plan table (the traced graph, design §4.4): {result.plan_txt_path}")
    print("-" * 100)
    return 0


def _resolve_config_paths(parsed: argparse.Namespace) -> list[Path]:
    """The run-config stack: explicit ``-c`` files, or the v1 sibling inference.

    Returns
    -------
    list[Path]
        At least one config path; the FIRST is the run config (it anchors
        the default output path, ``metadata.yaml`` lookup and the embedded
        ``config.yaml`` payload).

    Raises
    ------
    ConfigError
        When no config is given and none can be inferred.
    """
    config_paths: list[Path] = list(parsed.config or [])
    if config_paths:
        return config_paths
    if parsed.ckpt_path is None:
        raise ConfigError("salt2 export --manifest needs a config — pass -c <config.yaml>")
    inferred = parsed.ckpt_path.parents[1] / "config.yaml"  # to_onnx.py:629-631
    if not inferred.is_file():
        raise ConfigError(f"could not find a run config at {inferred} — pass --config")
    return [inferred]


# plan-29 W4: `_writer_manifest_from_cli` (the M4.5 writer-derived ONNX manifest
# assembly) is RETIRED — the ONNX output manifest now derives from the folded
# OnnxExportSink's declared leaves, found via `salt.core.cli._static_onnx_export_sink`.


def _print_manifest_from_cli(parsed: argparse.Namespace) -> int:
    """``salt2 export --manifest``: print the assembled output manifest and exit.

    Returns
    -------
    int
        0 on success (errors raise `GraphError`, handled by `main`).
    """
    from salt.core.cli import _static_onnx_export_sink  # noqa: PLC0415 - heavy/circular

    config_paths = _resolve_config_paths(parsed)
    cli = _run_free_cli(config_paths, parsed.set_overrides)
    export_cfg = cli._get(cli.config_init, "export") or ExportConfig()  # noqa: SLF001 - main.py precedent
    if parsed.name is not None:
        export_cfg.model_name = parsed.name
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    # plan-29 W4: the manifest derives from the folded OnnxExportSink's declared
    # leaves (the off-graph writer manifest is retired).
    export_sink = _static_onnx_export_sink(cli)
    if export_sink is None:
        raise ConfigError(
            "config has no OnnxExportSink — since plan-29 W4 the ONNX output manifest is "
            "declared by an OnnxExportSink (callbacks.onnx_export) naming the conversion "
            "outputs.* leaves. Add the conversion nodes + the OnnxExportSink (design §4.2/§6)."
        )
    if export_sink.model_name is None:
        export_sink.model_name = validate_model_name(
            export_cfg.model_name or sanitised_model_name(run_name)
        )
    rows = list(zip(export_sink.output_names(), export_sink.output_dtypes(), strict=True))
    width = max((len(name) for name, _ in rows), default=1)
    print(f"ONNX output manifest (folded conversion nodes, model_name={export_sink.model_name}):")
    for name, dtype in rows:
        print(f"  {name:<{width}}  {dtype:<7}  folded conversion node (outputs.* leaf)")
    return 0


def _export_from_cli(parsed: argparse.Namespace) -> tuple[ExportResult, OnnxAdapter]:
    """The CLI export flow: parse config, derive manifest, load checkpoint, export.

    Returns
    -------
    tuple[ExportResult, OnnxAdapter]
        The export result and the eager checker reference.

    Raises
    ------
    ConfigError
        On a missing config / export block, a config-declared
        ``export.outputs`` (the M4.5 migration error), or schema drift.
    FileExistsError
        On an existing output without ``--overwrite``.
    """
    from salt.core.saltmodule import SaltModule  # noqa: PLC0415 - heavy/circular

    ckpt_path: Path = parsed.ckpt_path
    config_paths = _resolve_config_paths(parsed)
    config_path = config_paths[0]
    cli = _run_free_cli(config_paths, parsed.set_overrides)
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - the main.py _get precedent
    if export_cfg is None:
        raise ConfigError(
            f"config {config_path} has no export: block — declare export.inputs (and "
            "optionally model_name/rename/combine; outputs derive from the writers, M4.5) "
            "in the run config, or stack an override file carrying only the export: block "
            f"as a second config:\n  salt2 export --ckpt_path {ckpt_path} "
            f"-c {config_path} -c my_export_block.yaml"
        )
    if parsed.name is not None:
        export_cfg.model_name = parsed.name
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    variables = _features_variables(cli)
    # plan-29 W4: the folded OnnxExportSink (wired at callbacks:) is the SOLE ONNX
    # output authority (the off-graph writer manifest is retired). Find it on the
    # parsed CLI and fold it into the planning module dict so its declared leaves
    # anchor the ONNX plan demand.
    from salt.core.cli import _static_onnx_export_sink  # noqa: PLC0415 - heavy/circular

    export_sink = _static_onnx_export_sink(cli)
    if export_sink is None:
        raise ConfigError(
            "config has no OnnxExportSink — since plan-29 W4 the ONNX output manifest is "
            "declared by an OnnxExportSink (callbacks.onnx_export) naming the conversion "
            "outputs.* leaves the folded nodes mint (ClassProbs/SeqClassIndex/VertexUnionFind/"
            "MaskFormerObjects/Combination). The off-graph reduce manifest was retired; add the "
            "conversion nodes + the OnnxExportSink to the run config (design §4.2/§6)."
        )
    # data-less checkpoint load: binds from the stored salt_core schema
    # BEFORE the strict state-dict load (saltmodule.py:861-893)
    model = SaltModule.load_from_checkpoint(
        ckpt_path,
        modules=cli.model._graph_modules,  # noqa: SLF001 - same-package adapter (cli.py precedent)
        map_location=torch.device("cpu"),
        weights_only=False,  # pytorch 2.6+ default flip (v1 to_onnx.py:666)
    )
    resolved = resolve_export_config(export_cfg, run_name)
    if export_sink.model_name is None:
        export_sink.model_name = resolved.model_name
    # fold the export sink into the modules the planner/adapter see
    modules = dict(model._graph_modules)  # noqa: SLF001 - same-package adapter
    if export_sink.name in modules:
        raise ConfigError(
            f"OnnxExportSink name {export_sink.name!r} collides with a model module — rename "
            "the callbacks key (design §2.2)"
        )
    modules[export_sink.name] = export_sink
    _cross_check_schema(model, resolved, variables)
    onnx_path: Path = parsed.output or (config_path.parent / "network.onnx")
    if onnx_path.exists() and not parsed.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")  # to_onnx.py:711
    print("-" * 100)
    print(f"Converting model to ONNX (model_name={resolved.model_name})...")
    print("-" * 100)
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
