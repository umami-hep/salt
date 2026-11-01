r"""``salt graph``/``salt schema`` — static graph tooling CLI.

Validates, plans, and renders the module graph of a salt trainer config
(``model:``/``data:``, adapted run-free through `SaltCLI` — see `load_config`).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import warnings
import warnings as stdlib_warnings  # stable handle; `warnings` is shadowed by a local list
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

from salt.graph.errors import SUGGESTION_CUTOFF, ConfigError, GraphError
from salt.graph.planner import SOURCES, Plan, Sinks, compile_plan, deadcode
from salt.graph.render import graph_dot, plan_table, pruned_modules, render_image
from salt.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    NestedSpec,
    _has_wildcard,
    _pattern_matches,
    flatten_spec,
    split_key,
)
from salt.model.bind import resolve_bind_schema
from salt.schema import dump_schema, save_schema
from salt.utils.logging import console

__all__ = ["GraphConfig", "load_config", "main"]


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """A loaded trainer config's pipeline graph: live modules plus planner inputs.

    `modules` combines ``data.modules`` and the model-side modules (plus the
    folded sink nodes); `sources` is empty (the reader is the source node).
    `schema` is the reader's label universe (None without a schema artifact);
    `reader` carries that artifact for the validate-time class-names check.
    `mode_errors` carries per-mode config errors found while deriving sinks
    (TEST dead-preds errors, ONNX export-block resolution errors):
    ``validate``/``deadcode`` report them as error-level findings for that
    mode, ``plan``/``plot``/``why`` raise them when the broken mode is
    requested — the affected mode's sinks fall back to anchor-on-all-preds
    so the other modes stay inspectable. `mode_warnings` carries per-mode
    warnings (a trainer config without an ``export:`` block leaves the ONNX
    contract unchecked); ``validate`` reports them (promotable with
    ``--strict``). `sink_origins` enriches missing-sink planner errors with
    the demanding config address (e.g. ``export.outputs``), per mode.
    `model_modules` is the model-side subdict kept separate from
    the combined `modules` exactly as the runtime path passes
    `SaltModule._graph_modules`.
    """

    modules: dict[str, GraphModule]
    sources: NestedSpec
    sinks: Sinks
    schema: tuple[str, ...] | None
    reader: Any | None = None
    mode_errors: dict[Mode, str] = field(default_factory=dict)
    mode_warnings: dict[Mode, str] = field(default_factory=dict)
    sink_origins: dict[Mode, dict[str, str]] = field(default_factory=dict)
    model_modules: dict[str, GraphModule] | None = None
    mup_cfg: dict[str, Any] | None = None


def load_config(
    path: str | Path | Sequence[str | Path], set_overrides: Sequence[str] | None = None
) -> GraphConfig:
    """Load a salt trainer config (stack) as one full-pipeline `GraphConfig`.

    Every file is read as YAML first, so a missing, invalid or non-mapping
    file fails naming it; at least one must carry top-level ``model:``/
    ``data:`` blocks (the fit surface). The stack is then parsed run-free
    through `SaltCLI` and adapted by `_load_fit_config`.

    Parameters
    ----------
    path : str | Path | Sequence[str | Path]
        The config YAML, or a stack of configs deep-merged left-to-right
        through the real `SaltCLI` surface (the repeatable ``-c`` flag); only
        one member need be a full trainer config, the rest may be partial
        overrides.
    set_overrides : Sequence[str] | None, optional
        ``KEY=VALUE`` entries forwarded to the trainer parser (the ``--set``
        CLI flag), by default None.

    Returns
    -------
    GraphConfig
        Live pipeline modules plus the derived per-mode sinks and the
        reader's label universe.

    Raises
    ------
    ConfigError
        On an unreadable or invalid file, a stack with no ``model:``/``data:``
        member, or a trainer-parse failure.
    """
    # local import: avoids a cli<->config_utils cycle
    from salt.utils.config_utils import expand_includes

    paths = [Path(p) for p in (path if isinstance(path, (list, tuple)) else [path])]
    # a config may declare the configs it stacks on; resolve those before reading
    paths = [Path(expand_includes(str(p))) if p.is_file() else p for p in paths]
    raws: list[dict[str, Any]] = []
    for one in paths:
        if not one.is_file():
            raise ConfigError(f"config file not found: {one}")
        try:
            with open(one) as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(f"config file {one} is not valid YAML: {err}") from err
        if not isinstance(raw, dict):
            raise ConfigError(f"config file {one} must contain a mapping")
        raws.append(raw)
    if not any("model" in raw or "data" in raw for raw in raws):
        removed = any("modules" in raw for raw in raws)
        hint = (
            "; the standalone modules:/sources:/sinks: graph format was removed — put the "
            "modules under a trainer config's data.modules / model.init_args.modules"
            if removed
            else ""
        )
        raise ConfigError(
            f"config {' + '.join(str(p) for p in paths)} is not a salt trainer config: "
            f"salt graph needs top-level 'model:'/'data:' blocks (the fit surface){hint}"
        )
    return _load_fit_config(paths, set_overrides)


def _load_fit_config(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> GraphConfig:
    """Adapt a trainer config (stack) into one full-pipeline `GraphConfig`: parse
    through `SaltCLI` run-free, combine ``data.modules`` + ``model.modules`` (the
    reader is the source node, so ``sources`` is empty) and derive per-mode sinks
    through the shared `salt.model.sink_prep.prepare_sinks` — the SAME preparation
    the runtime `SaltModule` runs, with errors collected into
    `GraphConfig.mode_errors` instead of raised. Raises `ConfigError` on a
    trainer-parse failure or a module-name collision.
    """
    # local import: the trainer surface (lightning/jsonargparse) is heavy
    # and circular with this module (salt.main dispatches to cli.main)
    from salt.data.processors.labels import Labels
    from salt.model.sink_prep import prepare_sinks
    from salt.outputs.sinks.registry import iter_sinks

    cli = _parse_trainer_cli(paths, set_overrides)
    model, dm = cli.model, cli.datamodule
    # InputSamples is a config holder, not a graph node: compile batch_modules only
    data_modules, reader = dm.batch_modules, dm.reader
    for module in data_modules.values():
        if isinstance(module, Labels):
            module.bind_streams(reader.streams)
    model_modules: dict[str, GraphModule] = dict(model._graph_modules)  # noqa: SLF001 - same-package adapter
    if overlap := sorted(set(data_modules) & set(model_modules)):
        raise ConfigError(
            f"config {' + '.join(str(p) for p in paths)}: module name(s) {overlap} appear "
            "in BOTH data.modules and model.modules — instance names must be unique "
            "across the pipeline graph"
        )
    prep = prepare_sinks(
        model_modules,
        iter_sinks(cli.trainer),
        output_section=model._output_section,  # noqa: SLF001 - same-package adapter
        reader=reader,
        callbacks=cli.trainer.callbacks,
        run_name=cli._get(cli.config_init, "name") or "salt",  # noqa: SLF001 - same-package adapter
        modes=PRIMARY_MODES,
        collect_errors=True,
        anchor_meta_rows=True,
        writer_demand=False,
    )
    for name in set(prep.modules) - set(model_modules):  # the folded sink nodes
        if name in data_modules:
            raise ConfigError(
                f"sink node name {name!r} collides with a pipeline module — instance "
                "names must be unique across the graph; rename the callback key"
            )
    by_mode = prep.by_mode.items()
    return GraphConfig(
        modules={**data_modules, **prep.modules},
        sources={},
        sinks={mode: tuple(ms.anchors) for mode, ms in by_mode},
        schema=reader.label_universe(),
        reader=reader,
        mode_errors={mode: ms.error for mode, ms in by_mode if ms.error},
        mode_warnings={mode: ms.warning for mode, ms in by_mode if ms.warning},
        sink_origins={m: ms.anchor_origins for m, ms in by_mode if ms.anchor_origins is not None},
        model_modules=model_modules,
        mup_cfg=getattr(model, "mup_cfg", None),
    )


def _run_free_cli_argv(
    config_paths: Sequence[str | Path], set_overrides: Sequence[str] | None
) -> list[str]:
    """Build the argv for a run-free `SaltCLI` parse of a trainer config stack:
    one sanitised ``--config`` per path (repeated configs deep-merge
    left-to-right, as ``salt fit``/``salt export`` do) followed by the
    ``--set`` overrides as ``--KEY=VALUE``. Shared by `_parse_trainer_cli`,
    ``salt export``'s ``_run_free_cli`` and `salt.model.mup._parse_cli`.
    Raises `ConfigError` on a malformed ``--set`` entry.
    """
    from salt.utils.config_utils import disable_logger_in_config

    args: list[str] = []
    for path in config_paths:
        # Disable the logger in keyless envs (no COMET_API_KEY) so run-free
        # parsing (graph tools, tests) doesn't fail at instantiate_classes with
        # "Comet.ml requires an API key"
        args.extend(["--config", disable_logger_in_config(str(path))])
    for entry in set_overrides or []:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    return args


def _build_run_free_cli(args: Sequence[str]) -> Any:
    """Construct the run-free `SaltCLI` from a ready argv (`_run_free_cli_argv`
    output) — the shared core of `_parse_trainer_cli`, ``salt export``'s
    ``_run_free_cli``, `salt.model.mup._parse_cli` and ``salt merge-config``'s
    ``--print_config`` dump, which add their command-specific error translation.
    The parser's `SystemExit` and jsonargparse's instantiate-time `ValueError`
    propagate.
    """
    from salt.main import SaltCLI

    args = list(args)
    with warnings.catch_warnings():
        # programmatic argv triggers Lightning's 'args parameter is
        # intended...' warning — filtered exactly as salt.main and
        # the salt export run-free parse do (noise on tooling whose
        # output users are told to read)
        warnings.filterwarnings(
            "ignore", message=r".*args parameter is intended to run from within Python.*"
        )
        return SaltCLI(args=args, run=False)


def _parse_trainer_cli(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> Any:
    """Parse a trainer config (stack) through the real salt surface, run-free
    (repeated configs deep-merge left-to-right, as `salt fit`/`salt export`
    do). Returns the constructed `SaltCLI` (nothing executed, no data
    touched); raises `ConfigError` on a parse/instantiate failure.
    """
    args = _run_free_cli_argv(paths, set_overrides)
    try:
        return _build_run_free_cli(args)
    except SystemExit as err:
        raise ConfigError(
            f"trainer config {' '.join(str(p) for p in paths)} failed to parse through the "
            f"salt surface (parser exit {err.code}; the parser error is printed above). "
            "Required init_args left as overrides in the YAML header can be supplied "
            "data-free via --set, e.g. --set model.modules.norm.init_args.norm_dict=unused.yaml"
        ) from err
    except ValueError as err:
        # jsonargparse raises a ValueError at instantiate_classes when a module's
        # __init__ validation fails (e.g. a salt writer/module ConfigError raised
        # in its constructor, wrapped as "Does not validate against any of the
        # Union subtypes"). Surface its message as a clean ConfigError so the
        # graph tooling reports rc=1 with the underlying error instead of crashing
        # with an uncaught traceback.
        raise ConfigError(
            f"trainer config {' '.join(str(p) for p in paths)} failed to instantiate "
            f"through the salt surface:\n{err}"
        ) from err


def _static_onnx_export_sink(cli: Any) -> Any | None:
    """The configured `OnnxExportSink`, or None — the ONNX counterpart to
    `_static_writer_sink_callback`, folded into the planning module dict so
    ``salt graph plot --mode onnx`` renders it and keeps the folded conversion
    nodes alive.
    """
    from salt.outputs import OnnxExportSink, iter_sinks

    sinks = iter_sinks(getattr(cli, "trainer", None))
    return next((sink for sink in sinks if isinstance(sink, OnnxExportSink)), None)


def _modes_for(args: argparse.Namespace) -> tuple[Mode, ...]:
    """The requested primary mode(s) from ``--mode`` (all primary modes if
    unset).
    """
    return PRIMARY_MODES if getattr(args, "mode", None) is None else (Mode[args.mode.upper()],)


def _fail(message: str) -> int:
    """Print an error to stderr and return exit code 1."""
    console(message, file=sys.stderr)
    return 1


def _format_graph_error(err: GraphError) -> str:
    """Format a kernel error with its class prefixed, e.g.
    ``"salt.graph.ConnectivityError: ..."``.
    """
    return f"salt.graph.{type(err).__name__}: {err}"


# ---------------------------------------------------------------------------
# graph validate
# ---------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    """``salt graph validate``: compile every requested mode; report findings.

    Graph errors, error-level deadcode findings (an unconsumed ``preds.*``
    port in TEST), and stored per-mode sink errors (`GraphConfig.mode_errors`)
    are always fatal; warnings (no schema, warning-level dead outputs) promote
    to errors under ``--strict``. Info-level findings (unconsumed FIT/VAL
    preds — the normal no-metric-callback case) are report-only.
    """
    cfg = load_config(args.config, args.set)
    infos: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    if cfg.schema is None:
        warnings.append(
            "field spellings cannot be checked statically (the reader has no schema "
            "artifact); a misspelled key will fail at the first batch"
        )
    if cfg.reader is not None:
        # default-on class-names <-> schema-attrs cross-check (set AND order);
        # raises ConfigError -> formatted by main()
        from salt.model.validation import check_class_names

        if checked := check_class_names(cfg.modules, cfg.reader):
            console(f"OK class_names ↔ schema attrs: {checked} list(s) match, set and order")
    if cfg.model_modules is not None:
        # `model_modules` is set only by `_load_fit_config`, whose SaltModule
        # construction already ran both validators, so these re-runs cannot newly fail.
        # muP routing: re-run to surface its warnings (promotable under --strict).
        from salt.model.mup import validate_mup_routing

        with stdlib_warnings.catch_warnings(record=True) as caught:
            stdlib_warnings.simplefilter("always")
            normalised = validate_mup_routing(cfg.mup_cfg, cfg.model_modules)
        warnings.extend(str(w.message) for w in caught)
        if normalised is not None:  # None = no `mup:` configured
            console(
                f"OK muP routing: apply_to={normalised['apply_to']} — every target has a mup "
                "init_arg, no mup:true module left out"
            )
        from salt.model.validation import validate_edge_port

        if n_edge := validate_edge_port(cfg.model_modules):
            console(
                f"OK edge port: {n_edge} edge encoder(s) — edge stream is Concat.streams[0] and "
                "the attention backend is edge-compatible (no silent flash bypass)"
            )
    # duck-typed `preflight()` checks file-backed materialise sources at warning
    # level — `validate` must run on data-less machines; a real fit/test promotes
    # them to hard errors (SaltModule.setup). datasets= (the reader's stream->dataset
    # map) is passed explicitly because validate never binds, so both look up the same keys.
    from salt.model.bind import reader_stream_datasets

    datasets = reader_stream_datasets(cfg.reader)
    for name, module in cfg.modules.items():
        if callable(preflight := getattr(module, "preflight", None)):
            try:
                preflight(datasets=datasets)
            except GraphError as err:
                warnings.append(f"preflight of module {name!r}: {err}")
    for mode in _modes_for(args):
        if stored := cfg.mode_errors.get(mode):
            errors.append(stored)
        if warned := cfg.mode_warnings.get(mode):
            warnings.append(warned)
        try:
            plan = _compile(cfg, mode)
        except GraphError as err:
            return _fail(_format_graph_error(err))
        console(
            f"OK [mode={mode.name}] {len(plan.steps)} steps, {len(plan.edges)} edges, "
            f"plan_hash={plan.plan_hash[:12]}"
        )
        for finding in deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks):
            line = f"[mode={mode.name}] {finding.module}/{finding.key}: {finding.reason}"
            {"error": errors, "info": infos}.get(finding.severity, warnings).append(line)
    for info in infos:
        console(f"info: {info}")
    for warning in warnings:
        console(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        console(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return _fail(f"{len(errors)} error-level deadcode finding(s)")
    if warnings and args.strict:
        return _fail(f"--strict: {len(warnings)} warning(s) promoted to errors")
    return 0


# ---------------------------------------------------------------------------
# graph deadcode
# ---------------------------------------------------------------------------


def _cmd_deadcode(args: argparse.Namespace) -> int:
    """``salt graph deadcode``: mode-aware dead-output report. An unconsumed
    ``preds.*`` port in TEST (or a stored per-mode sink error) is an error
    (exit 1); unconsumed FIT/VAL preds are info; the rest are warnings. A
    per-task ``expose: [fit, val]`` opt-out surfaces here as warning-level
    module pruning instead of a TEST preds error.
    """
    cfg = load_config(args.config, args.set)
    rc = 0
    for mode in _modes_for(args):
        if stored := cfg.mode_errors.get(mode):
            console(f"[deadcode] mode={mode.name}:")
            console(f"  ERROR {stored}")
            rc = 1
            continue
        findings = deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)
        if not findings:
            console(f"[deadcode] mode={mode.name}: OK — all produced ports consumed.")
            continue
        console(f"[deadcode] mode={mode.name}:")
        for finding in findings:
            level = finding.severity.upper()
            if finding.key == "*":
                console(f"  {level} module {finding.module!r}: {finding.reason}")
            else:
                console(f"  {level} {finding.key!r} ({finding.module}): {finding.reason}")
            if finding.severity == "error":
                rc = 1
    return rc


# ---------------------------------------------------------------------------
# graph plan
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    """``salt graph plan``: print the ordered plan table for one mode — each
    step's binding constraint and narrowed wildcard results.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    _require_mode_ok(cfg, mode)
    plan = _compile(cfg, mode)
    console(plan_table(plan))
    _print_onnx_static_caveat(cfg, mode)
    return 0


def _require_mode_ok(cfg: GraphConfig, mode: Mode) -> None:
    """Raise the stored `GraphConfig.mode_errors` entry for `mode` (e.g. a TEST
    dead-preds or ONNX export-block error) when the requested mode is broken.
    """
    if stored := cfg.mode_errors.get(mode):
        raise ConfigError(stored)


def _compile(cfg: GraphConfig, mode: Mode) -> Plan:
    """Compile `cfg`'s graph for `mode` with its schema, sinks and per-mode sink origins."""
    return compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )


def _print_onnx_static_caveat(cfg: GraphConfig, mode: Mode) -> None:
    """Print the dataset-fed-approximation caveat for static ONNX renderings.

    The static ONNX-mode plan/plot of a trainer config includes the dataset
    modules and anchors on dataset sources; the graph `salt export`
    actually traces has positional export inputs, no dataset modules, and
    in-graph reduces — its authoritative rendering is the `plan_onnx.txt`
    written next to ``network.onnx`` at export time (the two plan hashes
    legitimately differ).
    """
    if mode is Mode.ONNX and cfg.reader is not None:
        console(
            "note: this is the dataset-fed STATIC view of the ONNX graph (reader/features "
            "included, no export reduces). The traced export graph is rendered to "
            "plan_onnx.txt next to network.onnx by `salt export`."
        )


# ---------------------------------------------------------------------------
# graph plot
# ---------------------------------------------------------------------------


_PLOT_IMAGE_SUFFIXES = frozenset({".png", ".pdf", ""})


def _cmd_plot(args: argparse.Namespace) -> int:
    """``salt graph plot``: render the mode graph. ``-o`` is a stem: ``x.dot`` writes the
    Graphviz DOT only; ``x.png``/``x.pdf``/``x`` write ``x.dot`` and rasterise it to
    ``x.png`` + ``x.pdf`` with ``dot`` (`render_image`, strict: a missing/failing ``dot``
    raises `GraphError`); any other suffix is a `ConfigError`.
    """
    out_path = Path(args.output)
    if out_path.suffix != ".dot" and out_path.suffix not in _PLOT_IMAGE_SUFFIXES:
        raise ConfigError(
            "salt graph plot -o accepts .dot, .png, .pdf or no suffix "
            f"(got {out_path.suffix!r}): the Graphviz render always writes <stem>.png + "
            "<stem>.pdf next to <stem>.dot"
        )
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    _require_mode_ok(cfg, mode)
    plan = _compile(cfg, mode)
    _print_onnx_static_caveat(cfg, mode)
    pruned = pruned_modules(cfg.modules, plan)
    dot_text = graph_dot(plan, cfg.modules, pruned=pruned, widths=_resolve_widths(cfg))
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_path.with_suffix(".dot")
    dot_path.write_text(dot_text)
    console(f"wrote DOT to {dot_path}")
    if out_path.suffix != ".dot":
        _render_with_dot(dot_path, out_path)
    return 0


def _resolve_widths(cfg: GraphConfig) -> dict[str, int]:
    """Resolve concrete feature widths statically for the plot: compiles every
    inspectable primary-mode plan and feeds them to `resolve_bind_schema` —
    the same static resolution that builds the width-dependent nn.Linear
    layers, data-free. All compilable modes are unified (widths are
    config-fixed); a mode that fails to compile is skipped rather than
    aborting the plot.
    """
    plans: list[Plan] = []
    for plan_mode in PRIMARY_MODES:
        # skip non-requested modes that fail to compile (irrelevant to the widths);
        # the requested mode is compiled by the caller, so it is always present
        if plan_mode not in cfg.mode_errors:
            with contextlib.suppress(GraphError):
                plans.append(_compile(cfg, plan_mode))
    return dict(resolve_bind_schema(plans).widths)


def _render_with_dot(dot_path: Path, out_path: Path) -> None:
    """Strict `render_image` to ``<out_path stem>.{png,pdf}``, printing each written image
    (the CLI report line; shared with ``salt merge-config --merged.plots``).
    """
    images = render_image(dot_path, out_path.with_suffix(""), strict=True)
    for target in images or ():
        console(f"wrote {target.suffix.lstrip('.').upper()} to {target} (graphviz/dot)")


# ---------------------------------------------------------------------------
# graph why
# ---------------------------------------------------------------------------


def _cmd_why(args: argparse.Namespace) -> int:
    """``salt graph why``: explain one key's producer/consumers. For a present
    key: producer, spec, consumers. For an absent key: why (demand-pruned,
    mode-gated, or an undemanded wildcard) — unknown keys exit 1 with
    nearest-key suggestions. Raises `ConfigError` for an invalid ``--key``.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    _require_mode_ok(cfg, mode)
    try:
        key_parts = split_key(args.key)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"invalid --key {args.key!r}: {err}") from err
    key = KEY_SEP.join(key_parts)
    plan = _compile(cfg, mode)
    return 0 if _explain_present(plan, key, mode) else _explain_absent(cfg, plan, key, mode)


def _explain_present(plan: Plan, key: str, mode: Mode) -> bool:
    """Print producer/spec/consumers for a key that is in the plan; True if
    found (and explained).
    """
    if key in plan.sources:
        producer, spec = SOURCES, plan.sources[key]
    elif (step := next((s for s in plan.steps if key in s.produces), None)) is not None:
        producer, spec = f"{step.name} ({type(step.module).__name__})", step.produces[key]
    else:
        return False
    consumers = sorted({edge.consumer for edge in plan.edges if edge.key == key})
    console(f"[mode={mode.name}] {key!r}")
    console(f"  producer:  {producer}")
    console(f"  spec:      kind={spec.kind}, shape={spec.shape}, dtype={spec.dtype}")
    if consumers:
        console(f"  consumers: {', '.join(consumers)}")
    else:
        console(f"  consumers: none — dead output in mode {mode.name} (see salt graph deadcode)")
    return True


def _explain_absent(cfg: GraphConfig, plan: Plan, key: str, mode: Mode) -> int:
    """Explain why `key` is absent from the mode's plan (pruned / mode-gated /
    undemanded wildcard); 1 if the key is unknown everywhere.
    """
    findings = deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)
    pruned = {finding.module: finding.reason for finding in findings if finding.key == "*"}
    producers_in: dict[str, list[Mode]] = {}
    wildcard_hits: list[tuple[str, str]] = []
    gated_wildcards: dict[tuple[str, str], list[Mode]] = {}
    universe: set[str] = set(plan.sources)
    for name in sorted(cfg.modules):
        module = cfg.modules[name]
        for probe_mode in PRIMARY_MODES:
            for pkey, spec in flatten_spec(module.declare_io(probe_mode).produces).items():
                if not spec.active_in(probe_mode):
                    continue
                if _has_wildcard(pkey):
                    if not _pattern_matches(pkey, key):
                        continue
                    if probe_mode == mode:
                        wildcard_hits.append((name, pkey))
                    else:
                        gated_wildcards.setdefault((name, pkey), []).append(probe_mode)
                elif pkey == key:
                    producers_in.setdefault(name, []).append(probe_mode)
                else:
                    universe.add(pkey)
    lines = [f"[mode={mode.name}] {key!r} is not in the plan."]  # explained iff more follow
    for name, active_modes in producers_in.items():
        if mode in active_modes and name in pruned:
            lines.extend((
                f"  produced by {name!r}, but {name!r} is demand-pruned: {pruned[name]}",
                f"  fix: add a sink or consumer that demands {key!r} in mode {mode.name}, "
                "or remove the module",
            ))
        elif mode not in active_modes:
            mode_names = "/".join(m.name for m in active_modes)
            lines.append(
                f"  {name!r} produces {key!r} only in mode(s) {mode_names} — it is "
                f"mode-gated out of {mode.name}"
            )
    for name, pattern in sorted(set(wildcard_hits)):
        lines.append(
            f"  matches wildcard {pattern!r} of {name!r}, but nothing demands {key!r} in "
            f"mode {mode.name} — wildcard producers only materialise demanded keys"
        )
    for (name, pattern), active_modes in sorted(gated_wildcards.items()):
        if (name, pattern) in set(wildcard_hits):
            continue  # already explained as undemanded in this mode
        mode_names = "/".join(m.name for m in active_modes)
        lines.append(
            f"  matches wildcard {pattern!r} of {name!r}, which is active only in mode(s) "
            f"{mode_names} — it is mode-gated out of {mode.name}"
        )
    if len(lines) > 1:
        console("\n".join(lines))
        return 0
    universe |= set(cfg.schema or ())
    near = get_close_matches(key, sorted(universe), n=3, cutoff=SUGGESTION_CUTOFF)
    hint = f"\n  did you mean: {', '.join(repr(k) for k in near)}?" if near else ""
    return _fail(f"[mode={mode.name}] {key!r}: no module or source produces it in any mode.{hint}")


# ---------------------------------------------------------------------------
# schema dump
# ---------------------------------------------------------------------------


def _cmd_schema_dump(args: argparse.Namespace) -> int:
    """``salt schema dump``: scrape an H5 file into a schema artifact."""
    h5_path = Path(args.file)
    if not h5_path.is_file():
        return _fail(f"input file not found: {h5_path}")
    schema = dump_schema(h5_path)
    if not schema.groups:
        console(f"WARNING: no structured datasets found in {h5_path}", file=sys.stderr)
    save_schema(schema, args.output)
    console(f"wrote schema for {len(schema.groups)} group(s) to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# parser / entry point
# ---------------------------------------------------------------------------


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-c/--config`` and ``--set`` arguments."""
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        action="append",
        help="salt trainer config YAML. Repeatable: configs deep-merge left-to-right "
        "(the salt fit/export stacking semantics)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="dotted override forwarded to the trainer parser (repeatable) — supplies "
        "required init_args data-free, e.g. "
        "model.modules.norm.init_args.norm_dict=unused.yaml",
    )


def _add_mode_arg(parser: argparse.ArgumentParser, default: str | None) -> None:
    """Add the shared ``--mode`` argument (None default = all primary modes)."""
    parser.add_argument(
        "--mode",
        choices=("fit", "val", "test", "onnx"),
        default=default,
        help="primary mode" + ("" if default else " (default: all modes)"),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``salt`` argument parser; each subcommand sets ``func``."""
    parser = argparse.ArgumentParser(prog="salt", description="salt v2 static graph tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    graph = sub.add_parser("graph", help="static graph tooling")
    gsub = graph.add_subparsers(dest="graph_command", required=True)

    # the five compile-a-mode subcommands share -c/--set then --mode; the per-command
    # extra flag is added after --mode (creation order = help/listing order)
    sp: dict[str, argparse.ArgumentParser] = {}
    for name, help_text, func, mode_default in (
        ("validate", "connectivity validation", _cmd_validate, None),
        ("deadcode", "mode-aware dead-output report", _cmd_deadcode, None),
        ("plan", "print the ordered plan table", _cmd_plan, "fit"),
        ("plot", "render the graph via Graphviz DOT", _cmd_plot, "fit"),
        ("why", "explain one key's producer/consumers", _cmd_why, "fit"),
    ):
        sp[name] = gsub.add_parser(name, help=help_text)
        _add_config_arg(sp[name])
        _add_mode_arg(sp[name], default=mode_default)
        sp[name].set_defaults(func=func)
    sp["validate"].add_argument("--strict", action="store_true", help="promote warnings to errors")
    sp["plot"].add_argument(
        "-o",
        "--output",
        required=True,
        help="output stem: x.dot writes DOT only; x.png, x.pdf or x writes x.dot + x.png + x.pdf",
    )
    sp["why"].add_argument("--key", required=True, help="dotted bundle key to explain")

    schema = sub.add_parser("schema", help="dataset schema artifact tooling")
    ssub = schema.add_subparsers(dest="schema_command", required=True)
    dump = ssub.add_parser("dump", help="scrape an H5 file into schema.yaml")
    dump.add_argument("file", help="input HDF5 file")
    dump.add_argument("-o", "--output", required=True, help="output schema.yaml path")
    dump.set_defaults(func=_cmd_schema_dump)

    _add_mup_parsers(sub)

    return parser


def _add_mup_parsers(sub: Any) -> None:
    """Add the ``mup-shapes`` / ``mup-coord-check`` subcommands.

    Deferred to `salt.model.mup` handlers (heavy mup/pandas imports stay out of
    the graph-tooling startup path). The ``setup_mup`` console entry forwards
    to ``mup-shapes`` (pyproject.toml).
    """
    from salt.model.mup import cmd_mup_coord_check, cmd_mup_shapes

    shapes = sub.add_parser(
        "mup-shapes",
        help="generate muP base/delta infshapes for a config (the setup_mup console "
        "entry forwards here)",
    )
    _add_config_arg(shapes)
    shapes.add_argument(
        "--save-path",
        default=None,
        help="output infshape file (defaults to the config's model.init_args.mup.shape_path)",
    )
    shapes.add_argument("--base-width", type=int, default=None, help="base (narrow) apply_to width")
    shapes.add_argument(
        "--delta-width", type=int, default=None, help="delta (wider) apply_to width"
    )
    shapes.set_defaults(func=cmd_mup_shapes)

    coord = sub.add_parser(
        "mup-coord-check",
        help="run the muP coordinate-check at several widths; write coord-data CSV + plot",
    )
    _add_config_arg(coord)
    coord.add_argument(
        "--widths",
        nargs="+",
        required=True,
        help="apply_to widths to sweep, e.g. --widths 16 32 64 128",
    )
    coord.add_argument("-o", "--output", required=True, help="output plot path (.png/.pdf)")
    coord.add_argument("--nsteps", type=int, default=3, help="training steps per width")
    coord.add_argument("--nseeds", type=int, default=1, help="random-seed repeats")
    coord.add_argument("--lr", type=float, default=1e-2, help="coord-check learning rate (large)")
    coord.add_argument(
        "--shape-file",
        default=None,
        help="SHARED base/delta infshape file applied at EVERY swept width (salt mup-shapes "
        "output). Omit to auto-generate one (base=min width, delta=max width) — the shared-base "
        "protocol that gives a correct width_mult so the MuReadout is damped (a per-width "
        "self-base would force width_mult==1)",
    )
    coord.set_defaults(func=cmd_mup_coord_check)


def main(argv: Sequence[str] | None = None) -> int:
    """``salt`` entry point (pyproject ``[project.scripts]``).

    Returns
    -------
    int
        Process exit code: 0 success, 1 on kernel/config errors, 2 on
        argparse usage errors (raised by argparse as `SystemExit`).
    """
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except GraphError as err:
        return _fail(_format_graph_error(err))


if __name__ == "__main__":
    sys.exit(main())
