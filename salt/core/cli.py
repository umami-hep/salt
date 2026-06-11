r"""``salt2 graph``/``salt2 schema`` — static graph tooling CLI (design §4).

The ``salt2`` console script lives in `salt.core.main` (M2): trainer
subcommands (``fit``/``test``) run through `Salt2CLI` there, and ``graph``/
``schema`` invocations are dispatched unchanged to this module's `main`.

Subcommands (design §4.1-§4.4, §2.6):

- ``salt2 graph validate -c cfg.yaml [--mode fit|val|test|onnx] [--strict]``
- ``salt2 graph deadcode -c cfg.yaml [--mode ...]``
- ``salt2 graph plan     -c cfg.yaml --mode fit``
- ``salt2 graph plot     -c cfg.yaml --mode fit -o graph.svg``
- ``salt2 graph why      -c cfg.yaml --mode test --key labels.tracks.origin``
- ``salt2 schema dump <file.h5> -o schema.yaml``

All graph tooling operates on declarations only: instantiate the config,
compile plans per mode, analyse — no data, no GPU (design §4). Errors are
always fatal; warnings are promotable with ``--strict`` (the CI default).

**Two config formats are accepted** (auto-detected by `load_config`):

1. **§5.1 trainer configs** (the ``salt2 fit`` YAML surface, top-level
   ``model:``/``data:`` blocks): parsed through the REAL `Salt2CLI` in
   run-free mode (``base2.yaml`` auto-loaded, deep-merge semantics intact),
   then adapted into one full-pipeline graph — dataset modules + model
   modules, per-mode sinks from the model's declared anchors, the label-key
   universe from the reader's schema artifact. Init args that the YAML
   leaves as required overrides (e.g. ``norm_dict``) can be supplied
   data-free with repeated ``--set KEY=VALUE`` flags::

       salt2 graph validate -c salt/core/configs/gn2v2-dummy.yaml \\
         --set model.modules.norm.init_args.norm_dict=unused.yaml

   ``validate`` additionally runs the default-on §2.6 class-names ↔
   schema-attrs cross-check when the reader has a schema artifact.
2. **M1 toy configs** (top-level ``modules:``/``sources:``/``sinks:``) — a
   small instantiate-from-``class_path`` helper using `importlib` (see
   `instantiate`); kept for unit/toy graphs.

M1 config format (YAML)::

    modules:                    # name -> module; names become instance names
      embed:
        class_path: my_pkg.toys.Embed
        init_args: {out_dim: 16}        # optional ctor kwargs
    sources:                    # framework-provided boundary (design §3.1)
      inputs.x: {}                      # dotted key -> TensorSpec kwargs
      masks.x: {kind: pad_mask, modes: [fit, val]}
    sinks:                      # demand anchors per mode (design §3.1)
      fit: [losses.total]               # mapping mode -> keys ...
      test: [preds.x]
    # sinks: [preds.x]                  # ... or a flat list (compiled mode only)
    schema: schema.yaml         # optional: path (relative to this file) or
                                # a flat list of dotted group.field keys

``sources`` entries map dotted keys to `TensorSpec` keyword dicts
(``shape``/``dtype``/``kind``/``modes``/``optional``/``fields``; ``modes`` is
a name or list of names among fit/val/test/onnx/training/all). A list of bare
keys (or single-pair mappings) is also accepted.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any

import yaml

from salt.core.graph.errors import ConfigError, GraphError
from salt.core.graph.planner import SINKS, SOURCES, Plan, Sinks, compile_plan, deadcode
from salt.core.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    flatten_spec,
    split_key,
    unflatten_spec,
)
from salt.core.schema import dump_schema, load_schema, save_schema

try:
    import graphviz
except ImportError:  # pragma: no cover - graphviz is strictly optional
    graphviz = None

__all__ = ["GraphConfig", "instantiate", "load_config", "main"]

_SUGGESTION_CUTOFF = 0.5
_WILDCARD_PARTS = frozenset({"*", "**"})
_MODE_CHOICES = ("fit", "val", "test", "onnx")
_SPEC_KEYS = frozenset({"shape", "dtype", "kind", "modes", "optional", "fields"})


# ---------------------------------------------------------------------------
# config loading (M1-minimal; jsonargparse lands in M2, design §5.3)
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """A loaded graph config: live modules plus planner boundary inputs.

    `reader` is set only for §5.1 trainer configs (the adapter path) — it
    carries the schema artifact for the validate-time class-names check.
    """

    modules: dict[str, GraphModule]
    sources: NestedSpec
    sinks: Sinks
    schema: tuple[str, ...] | None
    reader: Any | None = None


def instantiate(class_path: str, init_args: Mapping[str, Any] | None = None) -> Any:
    """Instantiate ``pkg.mod.Class(**init_args)`` from a dotted class path.

    The M1 stand-in for jsonargparse instantiation (design §5.3).

    Returns
    -------
    Any
        The new instance.

    Raises
    ------
    ConfigError
        On unimportable modules, missing attributes, or ctor errors.
    """
    if not isinstance(class_path, str) or "." not in class_path:
        raise ConfigError(
            f"class_path must be a dotted import path like 'pkg.mod.Class', got {class_path!r}"
        )
    module_name, _, attr = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ConfigError(
            f"cannot import module {module_name!r} for class_path {class_path!r}: {err}"
        ) from err
    try:
        cls = getattr(module, attr)
    except AttributeError as err:
        raise ConfigError(f"module {module_name!r} has no attribute {attr!r}") from err
    try:
        return cls(**dict(init_args or {}))
    except (TypeError, ValueError) as err:
        raise ConfigError(f"instantiating {class_path!r} failed: {err}") from err


def load_config(path: str | Path, set_overrides: Sequence[str] | None = None) -> GraphConfig:
    """Load and instantiate a graph config (both formats — module docstring).

    A top-level ``model:``/``data:`` mapping is a §5.1 trainer config and is
    adapted through `Salt2CLI` (`_load_fit_config`); a top-level ``modules:``
    mapping is the M1 toy format. Instance names are assigned from the
    module-dict keys (design §2.2 — names must match config keys; the
    planner re-checks this invariant).

    Parameters
    ----------
    path : str | Path
        The config YAML.
    set_overrides : Sequence[str] | None, optional
        ``KEY=VALUE`` entries forwarded to the trainer parser (the ``--set``
        CLI flag) — trainer configs only, by default None.

    Returns
    -------
    GraphConfig
        Live modules plus parsed sources/sinks/schema.

    Raises
    ------
    ConfigError
        On unreadable files or structurally invalid configs.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as err:
        raise ConfigError(f"config file {path} is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping")
    if "modules" not in raw and ("model" in raw or "data" in raw):
        return _load_fit_config(path, set_overrides)
    if set_overrides:
        raise ConfigError(
            "--set overrides apply to salt2 trainer configs only "
            f"({path} is an M1 toy graph config)"
        )
    modules_raw = raw.get("modules")
    if not isinstance(modules_raw, dict) or not modules_raw:
        raise ConfigError(
            f"config file {path} must declare either a salt2 trainer config "
            "(top-level 'model:'/'data:' blocks, the §5.1 fit surface) or an M1 toy graph "
            "config (a non-empty 'modules' mapping of name -> {class_path, init_args})"
        )
    modules: dict[str, GraphModule] = {}
    for name, node in modules_raw.items():
        if not isinstance(node, dict) or "class_path" not in node:
            raise ConfigError(
                f"module {name!r} must be a mapping with 'class_path' (and optional 'init_args')"
            )
        instance = instantiate(node["class_path"], node.get("init_args"))
        try:
            instance.name = name
        except AttributeError as err:
            raise ConfigError(
                f"cannot assign instance name {name!r} on {node['class_path']!r}: {err}"
            ) from err
        modules[str(name)] = instance
    return GraphConfig(
        modules=modules,
        sources=_parse_sources(raw.get("sources"), path),
        sinks=_parse_sinks(raw.get("sinks"), path),
        schema=_parse_schema(raw.get("schema"), path),
    )


def _load_fit_config(path: Path, set_overrides: Sequence[str] | None) -> GraphConfig:
    """Adapt a §5.1 trainer config into one full-pipeline `GraphConfig` (design §4).

    Parses the config through the REAL `Salt2CLI` surface in run-free mode
    (``base2.yaml`` auto-loaded, deep-merge/null-deletion semantics intact),
    then builds the combined dataset + model graph: ``data.modules`` and
    ``model.modules`` form one module dict (the reader is the source node,
    so ``sources`` is empty), the per-mode sinks are the model's declared
    anchors (``loss.total`` / ``preds.*`` + TEST ``meta.rows``), and the
    wildcard-narrowing universe comes from the reader's schema artifact.
    Everything stays config-only — no data file is touched (design §2.3).

    Returns
    -------
    GraphConfig
        The adapted config, with `reader` set for the validate-time
        class-names check.

    Raises
    ------
    ConfigError
        When the trainer parse fails (with the ``--set`` hint — required
        init_args like ``norm_dict`` can be supplied data-free), or when a
        module name appears in both ``data.modules`` and ``model.modules``.
    """
    # local imports: the trainer surface (lightning/jsonargparse) is heavy
    # and circular with this module (salt.core.main dispatches to cli.main)
    from salt.core.data.processors import Labels  # noqa: PLC0415 - heavy/circular (docstring)
    from salt.core.main import Salt2CLI  # noqa: PLC0415 - heavy/circular (docstring)

    args = ["--config", str(path)]
    for entry in set_overrides or []:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        cli = Salt2CLI(args=args, run=False)
    except SystemExit as err:
        raise ConfigError(
            f"trainer config {path} failed to parse through the salt2 surface "
            f"(parser exit {err.code}; the parser error is printed above). Required "
            "init_args left as overrides in the YAML header can be supplied data-free "
            "via --set, e.g. --set model.modules.norm.init_args.norm_dict=unused.yaml"
        ) from err
    model, dm = cli.model, cli.datamodule
    data_modules = dm.modules
    reader = dm.reader
    for module in data_modules.values():
        if isinstance(module, Labels):
            module.bind_streams(reader.streams)
    if overlap := sorted(set(data_modules) & set(model._graph_modules)):  # noqa: SLF001 - same-package adapter
        raise ConfigError(
            f"config {path}: module name(s) {overlap} appear in BOTH data.modules and "
            "model.modules — instance names must be unique across the pipeline graph "
            "(design §2.2)"
        )
    modules: dict[str, GraphModule] = {**data_modules, **model._graph_modules}  # noqa: SLF001 - same-package adapter
    sinks: dict[Mode, tuple[str, ...]] = {}
    for mode in PRIMARY_MODES:
        keys = list(model._model_sinks(mode))  # noqa: SLF001 - same-package adapter
        if mode is Mode.TEST:
            keys.append("meta.rows")  # writer row alignment (design §8)
        sinks[mode] = tuple(keys)
    return GraphConfig(
        modules=modules,
        sources={},
        sinks=sinks,
        schema=reader.label_universe(),
        reader=reader,
    )


def _parse_mode(name: str) -> Mode:
    """Parse a mode name (``fit``/``training``/``all``/...) to a `Mode` flag.

    Returns
    -------
    Mode
        The parsed flag (may be composite for sinks mappings).

    Raises
    ------
    ConfigError
        On unknown mode names.
    """
    if isinstance(name, str) and name.upper() in Mode.__members__:
        return Mode[name.upper()]
    valid = "/".join(member.lower() for member in Mode.__members__)
    raise ConfigError(f"unknown mode {name!r}: expected one of {valid}")


def _parse_spec(key: str, node: Mapping[str, Any] | None) -> TensorSpec:
    """Build a `TensorSpec` for source `key` from a plain YAML mapping.

    Returns
    -------
    TensorSpec
        The parsed spec; an empty/absent mapping gives the default spec.

    Raises
    ------
    ConfigError
        On unknown spec keys or invalid spec values.
    """
    kwargs = dict(node or {})
    if unknown := set(kwargs) - _SPEC_KEYS:
        raise ConfigError(
            f"source {key!r}: unknown spec key(s) {sorted(unknown)} — expected {sorted(_SPEC_KEYS)}"
        )
    if "shape" in kwargs:
        kwargs["shape"] = tuple(kwargs["shape"])
    if "modes" in kwargs:
        names = kwargs["modes"]
        if isinstance(names, str):
            names = [names]
        kwargs["modes"] = reduce(or_, (_parse_mode(name) for name in names))
    try:
        return TensorSpec(**kwargs)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"source {key!r}: invalid spec: {err}") from err


def _parse_sources(raw: Any, path: Path) -> NestedSpec:
    """Parse the ``sources:`` section to a `NestedSpec` (design §3.1 boundary).

    Returns
    -------
    NestedSpec
        The nested source spec tree (empty when the section is absent).

    Raises
    ------
    ConfigError
        On malformed entries or clashing dotted keys.
    """
    if raw is None:
        return {}
    flat: dict[str, TensorSpec] = {}
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, str):
                items.append((entry, None))
            elif isinstance(entry, dict) and len(entry) == 1:
                items.append(next(iter(entry.items())))
            else:
                raise ConfigError(
                    f"config {path}: 'sources' list entries must be dotted keys or "
                    f"single-pair mappings, got {entry!r}"
                )
    else:
        raise ConfigError(f"config {path}: 'sources' must be a mapping or list")
    for key, node in items:
        if node is not None and not isinstance(node, dict):
            raise ConfigError(f"config {path}: source {key!r} spec must be a mapping")
        flat[str(key)] = _parse_spec(str(key), node)
    try:
        return unflatten_spec(flat)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"config {path}: invalid 'sources' keys: {err}") from err


def _parse_sinks(raw: Any, path: Path) -> Sinks:
    """Parse the ``sinks:`` section to the planner `Sinks` type (design §3.1).

    Returns
    -------
    Sinks
        ``None`` when absent, a key list (flat form, compiled mode only), or
        a ``{Mode: keys}`` mapping for full per-mode demand analysis.

    Raises
    ------
    ConfigError
        On malformed entries.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        if not all(isinstance(key, str) for key in raw):
            raise ConfigError(f"config {path}: 'sinks' list entries must be dotted keys")
        return list(raw)
    if isinstance(raw, dict):
        sinks: dict[Mode, tuple[str, ...]] = {}
        for mode_name, keys in raw.items():
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ConfigError(
                    f"config {path}: sinks for mode {mode_name!r} must be a list of dotted keys"
                )
            sinks[_parse_mode(mode_name)] = tuple(keys)
        return sinks
    raise ConfigError(f"config {path}: 'sinks' must be a list or a mode -> keys mapping")


def _parse_schema(raw: Any, path: Path) -> tuple[str, ...] | None:
    """Parse the ``schema:`` section to a flat key universe (design §2.2 rule (d)).

    Returns
    -------
    tuple[str, ...] | None
        Dotted ``group.field`` keys, or None when no schema is configured.

    Raises
    ------
    ConfigError
        On malformed entries (a bad schema *file* raises `SchemaError`).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        schema_path = Path(raw)
        if not schema_path.is_absolute():
            schema_path = path.parent / schema_path
        return load_schema(schema_path).keys()
    if isinstance(raw, list) and all(isinstance(key, str) for key in raw):
        return tuple(raw)
    raise ConfigError(
        f"config {path}: 'schema' must be a schema.yaml path or a flat list of dotted keys"
    )


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _has_wildcard(key: str) -> bool:
    """Check whether a dotted key contains a wildcard component (design §2.2).

    Returns
    -------
    bool
        True if any component is ``"*"`` or ``"**"``.
    """
    return any(part in _WILDCARD_PARTS for part in key.split(KEY_SEP))


def _pattern_matches(pattern: str, key: str) -> bool:
    """Match a concrete key against a wildcard pattern (design §2.2 semantics).

    ``"*"`` matches exactly one component, ``"**"`` one or more — mirrors the
    planner's matcher for `why` explanations.

    Returns
    -------
    bool
        True if `key` matches `pattern`.
    """

    def match(pat: tuple[str, ...], parts: tuple[str, ...]) -> bool:
        if not pat:
            return not parts
        head, rest = pat[0], pat[1:]
        if head == "**":
            return any(match(rest, parts[i:]) for i in range(1, len(parts) + 1))
        if not parts:
            return False
        if head in {"*", parts[0]}:
            return match(rest, parts[1:])
        return False

    return match(tuple(pattern.split(KEY_SEP)), tuple(key.split(KEY_SEP)))


def _modes_for(args: argparse.Namespace) -> tuple[Mode, ...]:
    """Resolve the requested primary mode(s) from ``--mode``.

    Returns
    -------
    tuple[Mode, ...]
        A single mode if ``--mode`` was given, else all primary modes.
    """
    if getattr(args, "mode", None) is None:
        return PRIMARY_MODES
    return (Mode[args.mode.upper()],)


def _fail(message: str) -> int:
    """Print an error to stderr and return exit code 1.

    Returns
    -------
    int
        Always 1.
    """
    print(message, file=sys.stderr)
    return 1


def _format_graph_error(err: GraphError) -> str:
    """Format a kernel error with its class, §4.1-style.

    Returns
    -------
    str
        E.g. ``"salt.core.graph.ConnectivityError: ..."``.
    """
    return f"salt.core.graph.{type(err).__name__}: {err}"


# ---------------------------------------------------------------------------
# graph validate (design §4.1)
# ---------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    """``salt2 graph validate``: compile every requested mode; report findings.

    Graph errors are always fatal, as are error-level deadcode findings (an
    unconsumed ``preds.*`` port in TEST, design §4.2); warnings (no schema
    configured, warning-level dead outputs) are promoted to errors under
    ``--strict`` (design §4).

    Returns
    -------
    int
        0 on success, 1 on error (or warnings under ``--strict``).
    """
    cfg = load_config(args.config, args.set)
    warnings: list[str] = []
    errors: list[str] = []
    if cfg.schema is None:
        warnings.append(
            "field spellings cannot be checked statically (no 'schema:' in the config); "
            "a misspelled key will fail at the first batch (design §2.6)"
        )
    if cfg.reader is not None:
        # default-on §2.6 class-names ↔ schema-attrs cross-check (set AND
        # order); raises ConfigError -> formatted by main()
        from salt.core.saltmodule import check_class_names  # noqa: PLC0415 - heavy/circular

        checked = check_class_names(cfg.modules, cfg.reader)
        if checked:
            print(f"OK class_names ↔ schema attrs: {checked} list(s) match, set and order (§2.6)")
    for mode in _modes_for(args):
        try:
            plan = compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
        except GraphError as err:
            return _fail(_format_graph_error(err))
        print(
            f"OK [mode={mode.name}] {len(plan.steps)} steps, {len(plan.edges)} edges, "
            f"plan_hash={plan.plan_hash[:12]}"
        )
        for finding in deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks):
            line = f"[mode={mode.name}] {finding.module}/{finding.key}: {finding.reason}"
            (errors if finding.severity == "error" else warnings).append(line)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return _fail(f"{len(errors)} error-level deadcode finding(s) (design §4.2)")
    if warnings and args.strict:
        return _fail(f"--strict: {len(warnings)} warning(s) promoted to errors")
    return 0


# ---------------------------------------------------------------------------
# graph deadcode (design §4.2)
# ---------------------------------------------------------------------------


def _cmd_deadcode(args: argparse.Namespace) -> int:
    """``salt2 graph deadcode``: mode-aware dead-output report (design §4.2).

    Findings carry severity levels: an unconsumed ``preds.*`` port in TEST is
    an error by default and makes the command exit 1 (design §4.2); the rest
    are warnings (report-only). TODO(M2): the per-task ``expose:`` opt-out.

    Returns
    -------
    int
        0 when no error-level finding exists; 1 on error-level findings or
        if the graph itself fails to resolve.
    """
    cfg = load_config(args.config, args.set)
    rc = 0
    for mode in _modes_for(args):
        findings = deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)
        if not findings:
            print(f"[deadcode] mode={mode.name}: OK — all produced ports consumed.")
            continue
        print(f"[deadcode] mode={mode.name}:")
        for finding in findings:
            level = finding.severity.upper()
            if finding.key == "*":
                print(f"  {level} module {finding.module!r}: {finding.reason}")
            else:
                print(f"  {level} {finding.key!r} ({finding.module}): {finding.reason}")
            if finding.severity == "error":
                rc = 1
    return rc


# ---------------------------------------------------------------------------
# graph plan (design §3.1 debugging story, §4.4 plan_<mode> table)
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    """``salt2 graph plan``: print the ordered §4.4 plan table for one mode.

    Shows each step's binding constraint (the latest predecessor forced by a
    key edge) and the narrowed wildcard results (design §3.1).

    Returns
    -------
    int
        0 on success, 1 on graph errors.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    plan = compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
    print(f"plan [mode={mode.name}] {len(plan.steps)} steps  plan_hash={plan.plan_hash}")
    index = {step.name: i for i, step in enumerate(plan.steps)}
    for i, step in enumerate(plan.steps, start=1):
        binding = [
            (index[edge.producer], edge.key)
            for edge in plan.edges
            if edge.consumer == step.name and edge.producer in index
        ]
        if binding:
            latest, _ = max(binding)
            after = plan.steps[latest].name
        elif any(e.consumer == step.name and e.producer == SOURCES for e in plan.edges):
            after = SOURCES
        else:
            after = "(no inputs)"
        needs = ", ".join(sorted(step.requires)) or "nothing"
        print(f"  {i:2d}. {step.name:<20} after {after:<20} (needs {needs})")
    narrowed_lines = []
    for step in plan.steps:
        declared = step.module.declare_io(mode).produces
        concrete = {key for key in flatten_spec(declared) if not _has_wildcard(key)}
        if extra := sorted(set(step.produces) - concrete):
            narrowed_lines.append(f"  {step.name}: {', '.join(extra)}")
    if narrowed_lines:
        print(f"narrowed wildcards [mode={mode.name}]:")
        print("\n".join(narrowed_lines))
    print(f"sources: {', '.join(sorted(plan.sources)) or '(none)'}")
    return 0


# ---------------------------------------------------------------------------
# graph plot (design §4.3)
# ---------------------------------------------------------------------------


def _cmd_plot(args: argparse.Namespace) -> int:
    """``salt2 graph plot``: render the mode graph (design §4.3).

    Always emits Graphviz DOT next to the requested output; renders the
    requested format via the optional ``graphviz`` package when available,
    else says so and leaves the ``.dot`` for manual rendering.

    Returns
    -------
    int
        0 on success, 1 on graph errors.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    plan = compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
    findings = deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)
    pruned = sorted({finding.module for finding in findings if finding.key == "*"})
    dot_text = _render_dot(plan, cfg.modules, pruned)
    out_path = Path(args.output)
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_path.with_suffix(".dot")
    dot_path.write_text(dot_text)
    print(f"wrote DOT to {dot_path}")
    if out_path.suffix == ".dot":
        return 0
    fmt = out_path.suffix.lstrip(".") or "svg"
    if graphviz is None:
        print(
            f"graphviz is not importable — wrote DOT only; render manually with: "
            f"dot -T{fmt} {dot_path} -o {out_path}"
        )
        return 0
    try:
        payload = graphviz.Source(dot_text).pipe(format=fmt)
    except (OSError, graphviz.ExecutableNotFound, graphviz.CalledProcessError) as err:
        print(
            f"graphviz rendering failed ({err}) — wrote DOT only; render manually with: "
            f"dot -T{fmt} {dot_path} -o {out_path}"
        )
        return 0
    out_path.write_bytes(payload)
    print(f"wrote {fmt.upper()} to {out_path}")
    return 0


def _esc(text: str) -> str:
    """Escape a string for a double-quoted DOT identifier.

    Returns
    -------
    str
        The escaped text (without surrounding quotes).
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _quote(text: str) -> str:
    """Quote a string as a DOT identifier.

    Returns
    -------
    str
        The double-quoted, escaped identifier.
    """
    return f'"{_esc(text)}"'


def _label(*lines: str) -> str:
    r"""Build a quoted multi-line DOT label.

    Returns
    -------
    str
        The quoted label with ``\n`` separators.
    """
    return '"' + "\\n".join(_esc(line) for line in lines) + '"'


def _edge_style(key: str, spec: TensorSpec) -> list[str]:
    """Kind-based DOT edge styling (design §4.3).

    Labels orange, losses red, pad-masks dotted grey, ``preds.*`` blue.

    Returns
    -------
    list[str]
        Extra DOT edge attributes.
    """
    if spec.kind == "label":
        return ["color=orange", "fontcolor=orange"]
    if spec.kind == "loss":
        return ["color=red", "fontcolor=red"]
    if spec.kind == "pad_mask":
        return ["color=grey", "fontcolor=grey", "style=dotted"]
    if spec.kind == "meta":
        return ["color=grey", "fontcolor=grey"]
    if key.partition(KEY_SEP)[0] == "preds":
        return ["color=blue", "fontcolor=blue"]
    return []


def _render_dot(plan: Plan, modules: dict[str, GraphModule], pruned: list[str]) -> str:
    r"""Render a compiled plan as Graphviz DOT text (design §4.3).

    Nodes are module instances (``name\nClassName``); edges are bundle keys
    with kind-based styling; demand-pruned modules render grey-dashed.

    Returns
    -------
    str
        The DOT source.
    """
    lines = [
        f"digraph salt_core_{plan.mode.name.lower()} {{",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Helvetica"];',
    ]
    if any(edge.producer == SOURCES for edge in plan.edges):
        lines.append(f"  {_quote(SOURCES)} [shape=ellipse, style=dashed];")
    if any(edge.consumer == SINKS for edge in plan.edges):
        lines.append(f"  {_quote(SINKS)} [shape=ellipse, style=dashed];")
    lines.extend(
        f"  {_quote(step.name)} [label={_label(step.name, type(step.module).__name__)}];"
        for step in plan.steps
    )
    for name in pruned:
        label = _label(name, type(modules[name]).__name__, "(pruned)")
        lines.append(f"  {_quote(name)} [label={label}, style=dashed, color=grey, fontcolor=grey];")
    for edge in plan.edges:
        if edge.producer == SOURCES:
            spec = plan.sources[edge.key]
        else:
            spec = plan.step(edge.producer).produces[edge.key]
        attrs = [f"label={_quote(edge.key)}", *_edge_style(edge.key, spec)]
        lines.append(f"  {_quote(edge.producer)} -> {_quote(edge.consumer)} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# graph why (design §3.1 debugging story, §4.1 quality bar)
# ---------------------------------------------------------------------------


def _cmd_why(args: argparse.Namespace) -> int:
    """``salt2 graph why``: explain one key's producer/consumers (design §3.1).

    For a key in the plan: prints producer, spec, and consumers. For a key
    absent from the plan: explains *why* (demand-pruned producer, mode-gated
    port, or an undemanded wildcard match) at the §4.1 quality bar. Unknown
    keys exit 1 with nearest-key suggestions.

    Returns
    -------
    int
        0 when the key exists or is explained; 1 when it is unknown.

    Raises
    ------
    ConfigError
        If ``--key`` is not a valid dotted bundle key.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    try:
        key_parts = split_key(args.key)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"invalid --key {args.key!r}: {err}") from err
    key = KEY_SEP.join(key_parts)
    plan = compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
    found = _explain_present(plan, key, mode)
    if found:
        return 0
    return _explain_absent(cfg, plan, key, mode)


def _explain_present(plan: Plan, key: str, mode: Mode) -> bool:
    """Print producer/spec/consumers for a key that is in the plan.

    Returns
    -------
    bool
        True if the key was found (and explained), False otherwise.
    """
    producer: str | None = None
    spec: TensorSpec | None = None
    if key in plan.sources:
        producer, spec = SOURCES, plan.sources[key]
    else:
        for step in plan.steps:
            if key in step.produces:
                producer = f"{step.name} ({type(step.module).__name__})"
                spec = step.produces[key]
                break
    if producer is None or spec is None:
        return False
    consumers = sorted({edge.consumer for edge in plan.edges if edge.key == key})
    print(f"[mode={mode.name}] {key!r}")
    print(f"  producer:  {producer}")
    print(f"  spec:      kind={spec.kind}, shape={spec.shape}, dtype={spec.dtype}")
    if consumers:
        print(f"  consumers: {', '.join(consumers)}")
    else:
        print(
            f"  consumers: none — dead output in mode {mode.name} "
            "(see salt2 graph deadcode, design §4.2)"
        )
    return True


def _explain_absent(cfg: GraphConfig, plan: Plan, key: str, mode: Mode) -> int:
    """Explain why `key` is absent from the mode's plan (design §4.1 quality).

    Returns
    -------
    int
        0 if a definite explanation was printed (pruned / mode-gated /
        undemanded wildcard), 1 if the key is unknown everywhere.
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
    lines = [f"[mode={mode.name}] {key!r} is not in the plan."]
    explained = False
    for name, active_modes in producers_in.items():
        if mode in active_modes and name in pruned:
            fix = (
                f"  fix: add a sink or consumer that demands {key!r} in mode {mode.name}, "
                "or remove the module (design §3.1)"
            )
            lines.extend((
                f"  produced by {name!r}, but {name!r} is demand-pruned: {pruned[name]}",
                fix,
            ))
            explained = True
        elif mode not in active_modes:
            mode_names = "/".join(m.name for m in active_modes)
            lines.append(
                f"  {name!r} produces {key!r} only in mode(s) {mode_names} — it is "
                f"mode-gated out of {mode.name} (design §2.2)"
            )
            explained = True
    for name, pattern in sorted(set(wildcard_hits)):
        lines.append(
            f"  matches wildcard {pattern!r} of {name!r}, but nothing demands {key!r} in "
            f"mode {mode.name} — wildcard producers only materialise demanded keys "
            "(design §2.2)"
        )
        explained = True
    for (name, pattern), active_modes in sorted(gated_wildcards.items()):
        if (name, pattern) in set(wildcard_hits):
            continue  # already explained as undemanded in this mode
        mode_names = "/".join(m.name for m in active_modes)
        lines.append(
            f"  matches wildcard {pattern!r} of {name!r}, which is active only in mode(s) "
            f"{mode_names} — it is mode-gated out of {mode.name} (design §2.2)"
        )
        explained = True
    src_flat = flatten_spec(cfg.sources)
    if key in src_flat and key not in plan.sources:
        active = [m.name for m in PRIMARY_MODES if src_flat[key].active_in(m)]
        lines.append(
            f"  sources provide {key!r} in mode(s) {'/'.join(active)} — gated out of {mode.name}"
        )
        explained = True
    if explained:
        print("\n".join(lines))
        return 0
    universe |= set(cfg.schema or ())
    near = get_close_matches(key, sorted(universe), n=3, cutoff=_SUGGESTION_CUTOFF)
    hint = f"\n  did you mean: {', '.join(repr(k) for k in near)}?" if near else ""
    return _fail(f"[mode={mode.name}] {key!r}: no module or source produces it in any mode.{hint}")


# ---------------------------------------------------------------------------
# schema dump (design §2.6)
# ---------------------------------------------------------------------------


def _cmd_schema_dump(args: argparse.Namespace) -> int:
    """``salt2 schema dump``: scrape an H5 file into a schema artifact (design §2.6).

    Returns
    -------
    int
        0 on success, 1 on unreadable input.
    """
    h5_path = Path(args.file)
    if not h5_path.is_file():
        return _fail(f"input file not found: {h5_path}")
    schema = dump_schema(h5_path)
    if not schema.groups:
        print(f"WARNING: no structured datasets found in {h5_path}", file=sys.stderr)
    save_schema(schema, args.output)
    print(f"wrote schema for {len(schema.groups)} group(s) to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# parser / entry point
# ---------------------------------------------------------------------------


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-c/--config`` and ``--set`` arguments."""
    parser.add_argument(
        "-c", "--config", required=True, help="config YAML (salt2 trainer config or M1 toy graph)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="dotted override forwarded to the trainer parser (repeatable; trainer configs "
        "only) — supplies required init_args data-free, e.g. "
        "model.modules.norm.init_args.norm_dict=unused.yaml",
    )


def _add_mode_arg(parser: argparse.ArgumentParser, default: str | None) -> None:
    """Add the shared ``--mode`` argument (None default = all primary modes)."""
    parser.add_argument(
        "--mode",
        choices=_MODE_CHOICES,
        default=default,
        help="primary mode" + ("" if default else " (default: all modes)"),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``salt2`` argument parser (design §4).

    Returns
    -------
    argparse.ArgumentParser
        The configured parser; each subcommand sets ``func``.
    """
    parser = argparse.ArgumentParser(
        prog="salt2", description="salt v2 static graph tooling (M1 kernel CLI, design §4)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    graph = sub.add_parser("graph", help="static graph tooling (design §4)")
    gsub = graph.add_subparsers(dest="graph_command", required=True)

    validate = gsub.add_parser("validate", help="connectivity validation (design §4.1)")
    _add_config_arg(validate)
    _add_mode_arg(validate, default=None)
    validate.add_argument("--strict", action="store_true", help="promote warnings to errors")
    validate.set_defaults(func=_cmd_validate)

    dead = gsub.add_parser("deadcode", help="mode-aware dead-output report (design §4.2)")
    _add_config_arg(dead)
    _add_mode_arg(dead, default=None)
    dead.set_defaults(func=_cmd_deadcode)

    plan = gsub.add_parser("plan", help="print the ordered plan table (design §4.4)")
    _add_config_arg(plan)
    _add_mode_arg(plan, default="fit")
    plan.set_defaults(func=_cmd_plan)

    plot = gsub.add_parser("plot", help="render the graph via Graphviz DOT (design §4.3)")
    _add_config_arg(plot)
    _add_mode_arg(plot, default="fit")
    plot.add_argument("-o", "--output", required=True, help="output image path (.svg/.png/.dot)")
    plot.set_defaults(func=_cmd_plot)

    why = gsub.add_parser("why", help="explain one key's producer/consumers (design §3.1)")
    _add_config_arg(why)
    _add_mode_arg(why, default="fit")
    why.add_argument("--key", required=True, help="dotted bundle key to explain")
    why.set_defaults(func=_cmd_why)

    schema = sub.add_parser("schema", help="dataset schema artifact tooling (design §2.6)")
    ssub = schema.add_subparsers(dest="schema_command", required=True)
    dump = ssub.add_parser("dump", help="scrape an H5 file into schema.yaml")
    dump.add_argument("file", help="input HDF5 file")
    dump.add_argument("-o", "--output", required=True, help="output schema.yaml path")
    dump.set_defaults(func=_cmd_schema_dump)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """``salt2`` entry point (pyproject ``[project.scripts]``).

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
