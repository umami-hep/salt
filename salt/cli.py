r"""``salt graph``/``salt schema`` — static graph tooling CLI.

Validates, plans, and renders module graphs from either trainer configs
(``model:``/``data:``) or toy configs (``modules:``/``sources:``/``sinks:``);
see `load_config` and `instantiate` for the two formats.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import warnings
import warnings as stdlib_warnings  # stable handle; `warnings` is shadowed by a local list
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any

import yaml

from salt.graph.errors import _SUGGESTION_CUTOFF, ConfigError, GraphError
from salt.graph.planner import SOURCES, Plan, Sinks, compile_plan, deadcode
from salt.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    _has_wildcard,
    _pattern_matches,
    flatten_spec,
    split_key,
    unflatten_spec,
)
from salt.model.bind import resolve_bind_schema
from salt.onnx.config import resolve_export_config
from salt.graph.render import dot_source, plan_table
from salt.schema import dump_schema, load_schema, save_schema

__all__ = ["GraphConfig", "instantiate", "load_config", "main"]

_MODE_CHOICES = ("fit", "val", "test", "onnx")
_SPEC_KEYS = frozenset({"shape", "dtype", "kind", "modes", "optional", "fields"})


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """A loaded graph config: live modules plus planner boundary inputs.

    `reader` is set only for trainer configs (the adapter path) — it carries
    the schema artifact for the validate-time class-names check.
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
    `writers` is always ``None`` (the ``writers:`` block and `WriterCallback`
    were removed; the ``outputs:``/``callbacks:`` sink path drives TEST sinks
    instead). `model_modules` is the model-side subdict kept separate from
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
    writers: Any | None = None
    model_modules: dict[str, GraphModule] | None = None
    mup_cfg: dict[str, Any] | None = None


def instantiate(class_path: str, init_args: Mapping[str, Any] | None = None) -> Any:
    """Instantiate ``pkg.mod.Class(**init_args)`` from a dotted class path.

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


def _is_trainer_format(raw: Mapping[str, Any]) -> bool:
    """Whether a parsed YAML mapping is a trainer config: a top-level
    ``model:``/``data:`` mapping without the toy ``modules:`` key.
    """
    return "modules" not in raw and ("model" in raw or "data" in raw)


def load_config(
    path: str | Path | Sequence[str | Path], set_overrides: Sequence[str] | None = None
) -> GraphConfig:
    """Load and instantiate a graph config (both formats — module docstring).

    A top-level ``model:``/``data:`` mapping is a trainer config and is
    adapted through `SaltCLI` (`_load_fit_config`); a top-level ``modules:``
    mapping is the toy format. Instance names are assigned from the
    module-dict keys (names must match config keys; the planner re-checks
    this invariant).

    Parameters
    ----------
    path : str | Path | Sequence[str | Path]
        The config YAML, or a stack of trainer configs (deep-merged
        left-to-right through the real `SaltCLI` surface — the repeatable
        ``-c`` flag). Toy graphs take exactly one config.
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
        On unreadable files or structurally invalid configs, or a config
        stack with no trainer-format member.
    """
    paths = [Path(p) for p in (path if isinstance(path, (list, tuple)) else [path])]
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
    if len(paths) > 1:
        # override files may carry any subset of keys, but at least one
        # stacked file must be trainer-format
        if not any(_is_trainer_format(raw) for raw in raws):
            raise ConfigError(
                f"repeated -c is supported for salt trainer configs only (deep-merged "
                f"left-to-right, the fit/export stacking semantics) — none of "
                f"{[str(p) for p in paths]} has top-level model:/data: blocks; M1 toy "
                "graph configs take exactly one -c"
            )
        return _load_fit_config(paths, set_overrides)
    raw, path = raws[0], paths[0]
    if _is_trainer_format(raw):
        return _load_fit_config(paths, set_overrides)
    if set_overrides:
        raise ConfigError(
            "--set overrides apply to salt trainer configs only "
            f"({path} is an M1 toy graph config)"
        )
    modules_raw = raw.get("modules")
    if not isinstance(modules_raw, dict) or not modules_raw:
        raise ConfigError(
            f"config file {path} must declare either a salt trainer config "
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


def _load_fit_config(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> GraphConfig:
    """Adapt a trainer config (stack) into one full-pipeline `GraphConfig`:
    parses through `SaltCLI` run-free, combines ``data.modules`` +
    ``model.modules`` into one module dict (the reader is the source node,
    so ``sources`` is empty), and derives per-mode sinks from the model's
    declared anchors plus the callbacks-level sink path (TEST H5/ONNX export
    sinks, FIT/VAL metrics callbacks) — errors go to `GraphConfig.mode_errors`
    rather than raising. Raises `ConfigError` on a trainer-parse failure or a
    module-name collision between ``data.modules`` and ``model.modules``.
    """
    # local import: the trainer surface (lightning/jsonargparse) is heavy
    # and circular with this module (salt.main dispatches to cli.main)
    from salt.data.processors.labels import Labels  # noqa: PLC0415 - heavy/circular (docstring)

    cli = _parse_trainer_cli(paths, set_overrides)
    model, dm = cli.model, cli.datamodule
    # setup-only modules (InputSamples/VDS/ShmStage) are partitioned out of the
    # tensor compile, so the combined full-pipeline graph here uses `batch_modules`,
    # not the union `dm.modules` (a setup-only module in `compile_plan` trips
    # AllModesDeadError). The setup graph is a distinct topology rendered separately.
    data_modules = dm.batch_modules
    reader = dm.reader
    for module in data_modules.values():
        if isinstance(module, Labels):
            module.bind_streams(reader.streams)
    if overlap := sorted(set(data_modules) & set(model._graph_modules)):  # noqa: SLF001 - same-package adapter
        raise ConfigError(
            f"config {' + '.join(str(p) for p in paths)}: module name(s) {overlap} appear "
            "in BOTH data.modules and model.modules — instance names must be unique "
            "across the pipeline graph (design §2.2)"
        )
    modules: dict[str, GraphModule] = {**data_modules, **model._graph_modules}  # noqa: SLF001 - same-package adapter
    writer_cb = None  # writers: block removed; WriterCallback no longer assembled
    writer_sink_cb = _static_writer_sink_callback(cli)
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - same-package adapter
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - same-package adapter
    # fold every callbacks-level renderable sink NODE into the planning module dict
    # so each renders its own card and anchors demand via its declared requires
    # (not the flat <sinks> sentinel). The H5OutputSink is active in TEST only
    # (outputs.*/meta.rows/masks.*); the OnnxExportSink is active in ONNX only (the
    # folded conversion leaves). In other modes a sink node's declare_io is empty
    # so the planner collects it as inactive (no card, no plan_hash perturbation).
    sink_node = _as_sink_node(writer_sink_cb)
    onnx_sink_node = _static_onnx_export_sink(cli)
    if onnx_sink_node is not None and onnx_sink_node.model_name is None:
        # the static render needs a model_name to derive the Athena output names;
        # default it from the export block / sanitised run name exactly as
        # `salt export` does
        onnx_sink_node.model_name = _static_export_model_name(export_cfg, run_name)
    for node in (sink_node, onnx_sink_node):
        if node is None:
            continue
        if node.name in modules:
            raise ConfigError(
                f"sink node name {node.name!r} collides with a pipeline module — instance "
                "names must be unique across the graph (design §2.2); rename the callback key"
            )
        modules[node.name] = node
    fitval_callbacks = _static_fitval_callbacks(cli)
    sinks: dict[Mode, tuple[str, ...]] = {}
    mode_errors: dict[Mode, str] = {}
    mode_warnings: dict[Mode, str] = {}
    sink_origins: dict[Mode, dict[str, str]] = {}
    for mode in PRIMARY_MODES:
        if mode is Mode.TEST and writer_sink_cb is not None:
            # only the callbacks-level sink path (writer_sink_cb / sink_node)
            # drives TEST sinks now.
            try:
                if sink_node is not None:
                    # a renderable sink NODE anchors ALL its demand via its declared
                    # requires (folded into `modules` above) — no flat sinks needed;
                    # its terminal-consumer demand keeps the producers (and
                    # transitively their preds.*) alive.
                    keys = []
                else:
                    keys = list(model._model_sinks(mode))  # noqa: SLF001 - base TEST anchor
                if writer_sink_cb is not None and sink_node is None:
                    # a non-node persistence sink (duck-typed writer_demand
                    # only): fold its writer_demand into the flat
                    # sinks exactly as SaltModule._boundary_demand does at salt
                    # test, so the in-graph conversion producers (outputs.*) stay
                    # alive in the render instead of pruning dead. A renderable
                    # sink NODE is instead folded into `modules` above and anchors
                    # its own demand — no flat sink.
                    sink_demand = writer_sink_cb.writer_demand(
                        model._graph_modules,  # noqa: SLF001 - same-package adapter
                        reader,
                    )
                    keys.extend(key for key in sink_demand if key not in keys)
            except ConfigError as err:
                mode_errors[mode] = str(err)
                keys = list(model._model_sinks(mode))  # noqa: SLF001 - all-preds render fallback
        elif mode is Mode.ONNX and onnx_sink_node is not None:
            # folded ONNX path (the sole ONNX-output authority): the OnnxExportSink
            # node (folded into `modules` above) anchors ALL its conversion-leaf
            # demand via its declared requires — a terminal consumer the planner
            # keeps alive, pulling the folded conversion nodes into the ONNX plan.
            # No flat manifest ports needed (it renders its own card). The
            # export-only half (model_name/inputs) is validated below if an
            # export: block is present.
            keys = []
            if export_cfg is not None:
                try:
                    resolve_export_config(export_cfg, run_name)
                except ConfigError as err:
                    mode_errors[mode] = str(err)
            else:
                mode_warnings[mode] = (
                    "the config has no export: block — the OnnxExportSink names the outputs, "
                    "but export.inputs/model_name were NOT checked; declare the export-only "
                    "half (design §5.1, §7) so `salt graph validate --mode onnx` gates "
                    "everything `salt export` will trace"
                )
        elif mode & Mode.TRAINING and fitval_callbacks:
            # the static half of the FIT/VAL-sink contract: configured metrics
            # callbacks declare plan sinks the same way writers do for TEST, so
            # `salt graph validate --mode fit` sees the same sinks (and the same
            # boundary demand) a real `salt fit` does. Mirror of the TEST branch.
            try:
                keys = list(
                    model._model_sinks(mode, callbacks=fitval_callbacks)  # noqa: SLF001 - same-package adapter
                )
                demand = model._callback_demand(mode, fitval_callbacks)  # noqa: SLF001 - same-package adapter
                # callback-demanded dataset-namespace keys (labels/masks/meta)
                # are FIT/VAL sinks too — their producers stay alive (§3.1)
                keys.extend(key for key in demand if key not in keys)
                sink_origins[mode] = dict(demand)
            except ConfigError as err:
                mode_errors[mode] = str(err)
                keys = list(model._model_sinks(mode))  # noqa: SLF001 - loss-only render fallback
        else:
            if mode is Mode.ONNX:
                mode_warnings[mode] = (
                    "the config declares no OnnxExportSink — the ONNX contract was NOT checked "
                    "(sinks fall back to every preds.* key); since plan-29 W4 the ONNX output "
                    "manifest is declared by an OnnxExportSink (callbacks.onnx_export) naming "
                    "the conversion outputs.* leaves, so `salt graph validate --mode onnx` "
                    "gates what `salt export` will trace"
                )
            keys = list(model._model_sinks(mode))  # noqa: SLF001 - same-package adapter
        # writer row alignment: a flat meta.rows sink — unless a sink NODE was
        # folded, which demands meta.rows itself via a named edge (a flat sink
        # would re-introduce the <sinks> sentinel card).
        if mode is Mode.TEST and sink_node is None and "meta.rows" not in keys:
            keys.append("meta.rows")
        sinks[mode] = tuple(keys)
    return GraphConfig(
        modules=modules,
        sources={},
        sinks=sinks,
        schema=reader.label_universe(),
        reader=reader,
        mode_errors=mode_errors,
        mode_warnings=mode_warnings,
        sink_origins=sink_origins,
        writers=None,  # WriterCallback removed
        model_modules=dict(model._graph_modules),  # noqa: SLF001 - same-package adapter
        mup_cfg=getattr(model, "mup_cfg", None),
    )


def _parse_trainer_cli(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> Any:
    """Parse a trainer config (stack) through the real salt surface, run-free
    (repeated configs deep-merge left-to-right, as `salt fit`/`salt export`
    do). Returns the constructed `SaltCLI` (nothing executed, no data
    touched); raises `ConfigError` on a parse/instantiate failure.
    """
    from salt.config_utils import disable_logger_in_config  # noqa: PLC0415
    from salt.main import SaltCLI  # noqa: PLC0415 - heavy/circular (module docstring)

    args: list[str] = []
    for path in paths:
        # Disable the logger in keyless envs (no COMET_API_KEY) so run-free
        # parsing (graph tools, tests) doesn't fail at instantiate_classes with
        # "Comet.ml requires an API key"
        cfg_no_logger = disable_logger_in_config(str(path))
        args.extend(["--config", cfg_no_logger])
    for entry in set_overrides or []:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        with warnings.catch_warnings():
            # programmatic argv triggers Lightning's 'args parameter is
            # intended...' warning — filtered exactly as salt.main and
            # the salt export run-free parse do (noise on tooling whose
            # output users are told to read)
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            return SaltCLI(args=args, run=False)
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


def _static_writer_sink_callback(cli: Any) -> Any | None:
    """The configured callbacks-level TEST persistence sink (duck-typed on
    ``writer_demand``, e.g. `H5OutputWriter`), or None — the static mirror of
    `SaltModule._attached_writer` so ``salt graph`` resolves the same TEST
    sinks.
    """
    from salt.outputs import is_test_persistence_sink  # noqa: PLC0415 - heavy/circular

    trainer = getattr(cli, "trainer", None)
    callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
    # the SAME selector the runtime uses (`SaltModule._attached_writer`), so the
    # static render and the real run resolve the same sink: an ONNX-only sink
    # (`OnnxExportSink`, empty TEST requires — handled by
    # `_static_onnx_export_sink`) and an auxiliary sink that opts out
    # (`JSONLOutputSink`) are both skipped.
    return next(
        (
            cb
            for cb in callbacks or []
            if callable(getattr(cb, "writer_demand", None)) and is_test_persistence_sink(cb)
        ),
        None,
    )


def _static_onnx_export_sink(cli: Any) -> Any | None:
    """The configured callbacks-level `OnnxExportSink`, or None — the ONNX
    counterpart to `_static_writer_sink_callback`, folded into the planning
    module dict so ``salt graph plot --mode onnx`` renders it and keeps the
    folded conversion nodes alive.
    """
    from salt.outputs import OnnxExportSink  # noqa: PLC0415 - heavy/circular

    trainer = getattr(cli, "trainer", None)
    callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
    return next((cb for cb in callbacks or [] if isinstance(cb, OnnxExportSink)), None)


def _static_export_model_name(export_cfg: Any, run_name: str) -> str:
    """The Athena output prefix for the static folded ONNX render: the export
    block's ``model_name`` if set, else the sanitised run name — matching
    `salt export`'s own default.
    """
    from salt.onnx.config import sanitised_model_name  # noqa: PLC0415 - heavy/circular

    name = getattr(export_cfg, "model_name", None) if export_cfg is not None else None
    return name or sanitised_model_name(run_name)


def _as_sink_node(callback: Any) -> Any | None:
    """`callback` as a renderable sink NODE (a `GraphModule` with
    ``is_sink() -> True``, e.g. `H5OutputSink`), or None for a non-node
    persistence sink (duck-typed ``writer_demand`` only), which keeps the
    legacy flat-``<sinks>`` folding.
    """
    if callback is None:
        return None
    is_sink = getattr(callback, "is_sink", None)
    has_io = callable(getattr(callback, "declare_io", None))
    if has_io and callable(is_sink) and bool(is_sink()):
        return callback
    return None


def _static_fitval_callbacks(cli: Any) -> list[Any]:
    """The configured FIT/VAL-sink callbacks (duck-typed on ``fit_val_demand``,
    e.g. `ConfusionMatrix`) from the run-free CLI — the static-tooling mirror
    of the runtime FIT/VAL-sink contract; config-only, never ``setup``.
    """
    trainer = getattr(cli, "trainer", None)
    callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
    return [cb for cb in callbacks or [] if callable(getattr(cb, "fit_val_demand", None))]


def _parse_mode(name: str) -> Mode:
    """Parse a mode name to a `Mode` flag (may be composite for sinks
    mappings); raises `ConfigError` on an unknown name.
    """
    if isinstance(name, str) and name.upper() in Mode.__members__:
        return Mode[name.upper()]
    valid = "/".join(member.lower() for member in Mode.__members__)
    raise ConfigError(f"unknown mode {name!r}: expected one of {valid}")


def _parse_spec(key: str, node: Mapping[str, Any] | None) -> TensorSpec:
    """Build a `TensorSpec` for source `key` from a plain YAML mapping (empty/
    absent -> the default spec); raises `ConfigError` on an unknown/invalid
    spec key.
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
    """Parse the ``sources:`` section to a `NestedSpec` (empty when absent);
    raises `ConfigError` on malformed entries or clashing dotted keys.
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
    """Parse the ``sinks:`` section to the planner `Sinks` type: None when
    absent, a flat key list, or a ``{Mode: keys}`` per-mode mapping. Raises
    `ConfigError` on malformed entries.
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
    """Parse the ``schema:`` section to a flat ``group.field`` key tuple, or
    None when unconfigured. Raises `ConfigError` on malformed entries (a bad
    schema file raises `SchemaError`).
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


def _modes_for(args: argparse.Namespace) -> tuple[Mode, ...]:
    """The requested primary mode(s) from ``--mode`` (all primary modes if
    unset).
    """
    if getattr(args, "mode", None) is None:
        return PRIMARY_MODES
    return (Mode[args.mode.upper()],)


def _fail(message: str) -> int:
    """Print an error to stderr and return exit code 1."""
    print(message, file=sys.stderr)
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
            "field spellings cannot be checked statically (no 'schema:' in the config); "
            "a misspelled key will fail at the first batch (design §2.6)"
        )
    if cfg.reader is not None:
        # default-on class-names <-> schema-attrs cross-check (set AND order);
        # raises ConfigError -> formatted by main()
        from salt.model.saltmodule import check_class_names  # noqa: PLC0415 - heavy/circular

        checked = check_class_names(cfg.modules, cfg.reader)
        if checked:
            print(f"OK class_names ↔ schema attrs: {checked} list(s) match, set and order (§2.6)")
    if cfg.model_modules is not None:
        # muP routing validator: apply_to naming a module without a mup init_arg
        # errors; a mup:true module outside apply_to warns. The same
        # `validate_mup_routing` SaltModule construction runs, surfaced here as
        # `salt graph validate` findings (warnings promotable under --strict).
        # The hard errors would already abort the parse above; this is the
        # first-class CI check + the warning capture.
        from salt.model.saltmodule import validate_mup_routing  # noqa: PLC0415 - heavy/circular

        with stdlib_warnings.catch_warnings(record=True) as caught:
            stdlib_warnings.simplefilter("always")
            try:
                normalised = validate_mup_routing(cfg.mup_cfg, cfg.model_modules)
            except ConfigError as err:
                errors.append(f"muP routing: {err}")
                normalised = None
        warnings.extend(str(w.message) for w in caught)
        if normalised is not None:
            print(
                f"OK muP routing: apply_to={normalised['apply_to']} — every target has a mup "
                "init_arg, no mup:true module left out (§3.4)"
            )
        # edge bind-time validators: edge-stream-first + EdgeAttention-backend
        # forcing. The same `validate_edge_port` SaltModule construction runs;
        # surfaced here as a first-class CI check (these are hard errors — they
        # would already abort the parse above; this captures the OK line / the
        # error message for the validate report).
        from salt.model.saltmodule import validate_edge_port  # noqa: PLC0415 - heavy/circular

        try:
            n_edge = validate_edge_port(cfg.model_modules)
        except ConfigError as err:
            errors.append(f"edge port: {err}")
            n_edge = 0
        if n_edge:
            print(
                f"OK edge port: {n_edge} edge encoder(s) — edge stream is Concat.streams[0] and "
                "the attention backend is edge-compatible (no silent flash bypass; §6.7)"
            )
    # data-free module preflights: duck-typed `preflight()` checks file-backed
    # materialise sources (e.g. the Normaliser norm dict). Warning-level here —
    # `validate` must stay runnable on data-less machines where the documented
    # `--set ...norm_dict=unused.yaml` override is in play; an actual
    # `salt fit`/`test` run promotes these to hard errors (SaltModule.setup).
    for name, module in cfg.modules.items():
        preflight = getattr(module, "preflight", None)
        if not callable(preflight):
            continue
        try:
            preflight()
        except GraphError as err:
            warnings.append(f"preflight of module {name!r}: {err}")
    for mode in _modes_for(args):
        if stored := cfg.mode_errors.get(mode):
            errors.append(stored)
        if warned := cfg.mode_warnings.get(mode):
            warnings.append(warned)
        try:
            plan = compile_plan(
                cfg.modules,
                mode,
                cfg.sources,
                schema=cfg.schema,
                sinks=cfg.sinks,
                sink_origins=cfg.sink_origins.get(mode),
            )
        except GraphError as err:
            return _fail(_format_graph_error(err))
        print(
            f"OK [mode={mode.name}] {len(plan.steps)} steps, {len(plan.edges)} edges, "
            f"plan_hash={plan.plan_hash[:12]}"
        )
        for finding in deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks):
            line = f"[mode={mode.name}] {finding.module}/{finding.key}: {finding.reason}"
            bucket = {"error": errors, "info": infos}.get(finding.severity, warnings)
            bucket.append(line)
        # cfg.writers is always None (WriterCallback removed); the
        # validate_specs block that used it is removed here.
    for info in infos:
        print(f"info: {info}")
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
            print(f"[deadcode] mode={mode.name}:")
            print(f"  ERROR {stored}")
            rc = 1
            continue
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
# graph plan
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    """``salt graph plan``: print the ordered plan table for one mode — each
    step's binding constraint and narrowed wildcard results.
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    _require_mode_ok(cfg, mode)
    plan = compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )
    print(plan_table(plan))
    _print_onnx_static_caveat(cfg, mode)
    return 0


def _require_mode_ok(cfg: GraphConfig, mode: Mode) -> None:
    """Raise the stored `GraphConfig.mode_errors` entry for `mode` (e.g. a TEST
    dead-preds or ONNX export-block error) when the requested mode is broken.
    """
    if stored := cfg.mode_errors.get(mode):
        raise ConfigError(stored)


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
        print(
            "note: this is the dataset-fed STATIC view of the ONNX graph (reader/features "
            "included, no export reduces). The traced export graph is rendered to "
            "plan_onnx.txt next to network.onnx by `salt export` (design §4.4)."
        )


# ---------------------------------------------------------------------------
# graph plot
# ---------------------------------------------------------------------------


def _cmd_plot(args: argparse.Namespace) -> int:
    """``salt graph plot``: render the mode graph. Emits Graphviz DOT
    (port-card layout) next to the requested output, then shells out to
    ``dot`` (baked into the salt container) to rasterise a PNG + sibling PDF
    — the authoritative graph image, no matplotlib path. A missing/failing
    ``dot`` raises `GraphError` (via `_render_with_dot`).
    """
    cfg = load_config(args.config, args.set)
    mode = Mode[args.mode.upper()]
    _require_mode_ok(cfg, mode)
    plan = compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )
    _print_onnx_static_caveat(cfg, mode)
    findings = deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)
    pruned = sorted({finding.module for finding in findings if finding.key == "*"})
    widths = _resolve_widths(cfg)
    dot_text = dot_source(plan, cfg.modules, pruned, widths=widths)
    out_path = Path(args.output)
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_path.with_suffix(".dot")
    dot_path.write_text(dot_text)
    print(f"wrote DOT to {dot_path}")
    if out_path.suffix == ".dot":
        return 0
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
        if plan_mode in cfg.mode_errors:
            continue
        try:
            plans.append(
                compile_plan(
                    cfg.modules,
                    plan_mode,
                    cfg.sources,
                    schema=cfg.schema,
                    sinks=cfg.sinks,
                    sink_origins=cfg.sink_origins.get(plan_mode),
                )
            )
        except GraphError:
            # a non-requested mode that does not compile is irrelevant to the
            # requested mode's widths — skip it (the requested mode is already
            # compiled by the caller, so it is always represented).
            continue
    return dict(resolve_bind_schema(plans).widths)


def _render_with_dot(dot_path: Path, out_path: Path) -> None:
    """Rasterise the DOT sidecar to a PNG (``out_path``) and a sibling PDF via
    the ``dot`` binary. Raises `GraphError` when ``dot`` is not on PATH, or a
    ``dot`` invocation fails (stderr surfaced in the message).
    """
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise GraphError(
            "the Graphviz `dot` binary was not found on PATH — cannot render the "
            f"graph image. The DOT source was written to {dot_path}; render it "
            "inside the salt container (which bakes in Graphviz), e.g.\n"
            "  apptainer exec .../salt.sif dot -Tpng "
            f"{dot_path} -o {out_path}\n"
            "or rebuild the salt container from repos/salt/container/salt.def."
        )
    png_path = out_path
    pdf_path = out_path.with_suffix(".pdf")
    renders = (
        ([dot_bin, "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(png_path)], png_path),
        ([dot_bin, "-Tpdf", str(dot_path), "-o", str(pdf_path)], pdf_path),
    )
    for cmd, target in renders:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise GraphError(
                f"`dot` failed to render {target} (exit {result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'}"
            )
        print(f"wrote {target.suffix.lstrip('.').upper()} to {target} (graphviz/dot)")


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
    plan = compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )
    found = _explain_present(plan, key, mode)
    if found:
        return 0
    return _explain_absent(cfg, plan, key, mode)


def _explain_present(plan: Plan, key: str, mode: Mode) -> bool:
    """Print producer/spec/consumers for a key that is in the plan; True if
    found (and explained).
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
            "(see salt graph deadcode, design §4.2)"
        )
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
# graph resolve [--annotate]
# ---------------------------------------------------------------------------

def _cmd_resolve(args: argparse.Namespace) -> int:
    """``salt graph resolve``: the writer-derived output manifest, eval + ONNX.
    Prints the assembled manifest; with ``--annotate``, writes it into the
    config as a refreshable comment block. Repeated ``-c`` deep-merges and the
    annotation goes into the last (most specific) file.
    """
    paths = [Path(p) for p in args.config]
    raws = []
    for path in paths:
        if not path.is_file():
            return _fail(f"config file not found: {path}")
        raws.append(yaml.safe_load(path.read_text()))
    if not any(isinstance(raw, dict) and ("model" in raw or "data" in raw) for raw in raws):
        return _fail(
            "salt graph resolve needs a salt trainer config (top-level model:/data: "
            "blocks) — toy graph configs have no writers block (M4.5 unified manifest)"
        )
    # the WriterCallback-based manifest (writers.modules) was removed.
    # `salt graph resolve` no longer has a manifest to derive.
    return _fail(
        "salt graph resolve is no longer supported (W6c removal): the writers.modules "
        "manifest block was removed; the eval columns and ONNX outputs are now declared "
        "by the outputs: section + OnnxExportSink — inspect those directly "
        "(see gn2v2-dummy.yaml for the canonical config pattern)"
    )


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
        "-c",
        "--config",
        required=True,
        action="append",
        help="config YAML (salt trainer config or M1 toy graph). Repeatable: trainer "
        "configs deep-merge left-to-right (the salt fit/export stacking semantics); "
        "toy graphs take exactly one",
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
    """Build the ``salt`` argument parser; each subcommand sets ``func``."""
    parser = argparse.ArgumentParser(
        prog="salt", description="salt v2 static graph tooling (M1 kernel CLI, design §4)"
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

    resolve = gsub.add_parser(
        "resolve",
        help="print the writer-derived output manifest (eval columns + ONNX outputs); "
        "--annotate writes it into the config (design §4.4, M4.5 unified manifest)",
    )
    _add_config_arg(resolve)
    resolve.add_argument(
        "--annotate",
        action="store_true",
        help="rewrite the config file with the manifest comment block (refreshed in "
        "place; with repeated -c the block goes into the LAST config file)",
    )
    resolve.set_defaults(func=_cmd_resolve)

    schema = sub.add_parser("schema", help="dataset schema artifact tooling (design §2.6)")
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
    from salt.model.mup import cmd_mup_coord_check, cmd_mup_shapes  # noqa: PLC0415

    shapes = sub.add_parser(
        "mup-shapes",
        help="generate muP base/delta infshapes for a config (design §3.4, §9.2; the "
        "setup_mup console entry forwards here)",
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
        help="run the muP coordinate-check at several widths; write coord-data CSV + plot "
        "(design §3.4 690-694)",
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
        "protocol that gives a correct width_mult so the MuReadout is damped (the MU-HUMAN fix, "
        "replacing the per-width self-base that forced width_mult==1)",
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
