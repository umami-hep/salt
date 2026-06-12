"""``salt2 export``: config + checkpoint -> validated ``.onnx`` (design §7).

The exporter compiles the ``Mode.ONNX`` plan from CONFIG-DERIVED sources
("no data needed", design §7.1 — boundary specs mirror the dataset
`Features` declaration instead of touching the H5), demands exactly the
``export.outputs`` ports (pruning removes labels/losses/writers
automatically), wraps the plan in the traceable `OnnxAdapter`, traces with
``torch.onnx.export(opset_version=20, dynamo=False)`` (v1
``to_onnx.py:714-721``), writes the v1-bit-compatible ``gnn_config``
metadata, and sweep-checks torch vs onnxruntime (design §7.6).

Programmatic surface (used by the fixture-driven gates, no checkpoint/CLI
required): `compile_onnx_plan` + `export_graph` on a bound, weight-loaded
module dict. CLI surface: ``salt2 export --ckpt_path <ckpt> [-c config ...]``
(``-c`` repeatable — later files deep-merge on top, e.g. an export-block
override onto a run config trained without one), dispatched from
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
    stream_of_input_port,
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


def compile_onnx_plan(
    modules: dict[str, GraphModule],
    export: ExportConfig,
    variables: Mapping[str, Sequence[str]],
) -> Plan:
    """Compile the ``Mode.ONNX`` plan demanded by the ``export.outputs`` ports.

    Demand pruning removes labels/losses/matcher automatically (design §7);
    a missing output producer raises the planner's §4.1-quality
    `ConnectivityError`. A `ShapeError` on an ``export.inputs`` port (the
    classic mis-flagged ``sequence:`` mistake — a variable-length stream
    declared as a ``[1, F]`` global, or vice versa) is re-raised with the
    config address and the concrete fix appended (§4.1 quality bar).

    Returns
    -------
    Plan
        The frozen ONNX plan.

    Raises
    ------
    ShapeError
        On a rank/shape mismatch — augmented with the ``export.inputs``
        attribution when the offending key is an export input port.
    """
    try:
        return compile_plan(
            modules,
            Mode.ONNX,
            sources=derive_onnx_sources(export, variables),
            sinks=[out.port for out in export.outputs],
            sink_origins={
                out.port: f"export output {out.port!r} (config: export.outputs)"
                for out in export.outputs
            },
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
    run_name: str = "salt",
    config: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    ckpt_path: str | Path | None = None,
    seed: int = 42,
) -> ExportResult:
    """Export a bound, weight-loaded module dict to ONNX (the programmatic core).

    Resolves+validates the export block (`model_name` rules apply HERE,
    never at fit — design §7), compiles the ONNX plan, builds the
    `OnnxAdapter` (torch-math forced via the ``set_export_mode`` protocol),
    traces at ``opset_version=20, dynamo=False`` with example inputs sized
    from the `Features` declaration, and writes the ``gnn_config``
    metadata. The checker is the caller's separate step (`check_onnx` —
    the CLI runs it by default).

    Parameters
    ----------
    modules : dict[str, GraphModule]
        Bound module instances carrying the weights to export.
    export : ExportConfig
        The parsed (unresolved is fine) export block.
    variables : Mapping[str, Sequence[str]]
        Per-stream `Features` variable lists.
    onnx_path : str | Path
        Output ``.onnx`` path.
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
    # static `salt2 graph plan --mode onnx` view is dataset-fed and differs
    from salt.core.render import plan_table  # noqa: PLC0415 - lazy: keeps onnx import light

    plan_txt_path = onnx_path.parent / "plan_onnx.txt"
    plan_txt_path.write_text(plan_table(plan) + "\n")
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
    parser.add_argument("--ckpt_path", type=Path, required=True, help="checkpoint path")
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
    try:
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


def _export_from_cli(parsed: argparse.Namespace) -> tuple[ExportResult, OnnxAdapter]:
    """The CLI export flow: parse config, load checkpoint, export.

    Returns
    -------
    tuple[ExportResult, OnnxAdapter]
        The export result and the eager checker reference.

    Raises
    ------
    ConfigError
        On a missing config / export block, or schema drift.
    FileExistsError
        On an existing output without ``--overwrite``.
    """
    from salt.core.saltmodule import SaltModule  # noqa: PLC0415 - heavy/circular

    ckpt_path: Path = parsed.ckpt_path
    config_paths: list[Path] = list(parsed.config or [])
    if not config_paths:
        inferred = ckpt_path.parents[1] / "config.yaml"  # to_onnx.py:629-631
        if not inferred.is_file():
            raise ConfigError(f"could not find a run config at {inferred} — pass --config")
        config_paths = [inferred]
    # the FIRST config is the run config: it anchors the default output path,
    # metadata.yaml lookup and the embedded config.yaml payload; later -c
    # files deep-merge overrides on top (the fit stacking semantics)
    config_path = config_paths[0]
    cli = _run_free_cli(config_paths, parsed.set_overrides)
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - the main.py _get precedent
    if export_cfg is None:
        raise ConfigError(
            f"config {config_path} has no export: block — declare export.inputs/outputs "
            "(design §5.1) in the run config, or stack an override file carrying only the "
            f"export: block as a second config:\n  salt2 export --ckpt_path {ckpt_path} "
            f"-c {config_path} -c my_export_block.yaml"
        )
    if parsed.name is not None:
        export_cfg.model_name = parsed.name
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - main.py precedent
    variables = _features_variables(cli)
    # data-less checkpoint load: binds from the stored salt_core schema
    # BEFORE the strict state-dict load (saltmodule.py:861-893)
    model = SaltModule.load_from_checkpoint(
        ckpt_path,
        modules=cli.model._graph_modules,  # noqa: SLF001 - same-package adapter (cli.py precedent)
        map_location=torch.device("cpu"),
        weights_only=False,  # pytorch 2.6+ default flip (v1 to_onnx.py:666)
    )
    resolved = resolve_export_config(export_cfg, run_name)
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
        model._graph_modules,  # noqa: SLF001 - same-package adapter
        resolved,
        variables,
        onnx_path,
        run_name=run_name,
        config=config_payload,
        run_metadata=load_run_metadata(config_path),
        ckpt_path=ckpt_path,
    )
    return result, result.adapter


if __name__ == "__main__":
    sys.exit(main())
