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
- ``salt2 graph resolve  -c cfg.yaml [--annotate]`` — the writer-derived
  output manifest (eval columns + ONNX outputs, M4.5 unified manifest);
  ``--annotate`` refreshes the §4.4 comment block inside the config
- ``salt2 schema dump <file.h5> -o schema.yaml``

``-c/--config`` is REPEATABLE on every graph subcommand: trainer configs
deep-merge left-to-right exactly as on ``salt2 fit``/``salt2 export`` (the
base + override pattern is statically inspectable without a prior fit;
M1 toy graphs still take exactly one config). With ``--annotate`` and
multiple configs, the comment block is written into the LAST (most
specific) config file.

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

from salt.core.graph.errors import ConfigError, GraphError
from salt.core.graph.planner import SOURCES, Plan, Sinks, compile_plan, deadcode
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
from salt.core.nn.bind import resolve_bind_schema
from salt.core.onnx.config import attach_manifest, manifest_table, resolve_export_config
from salt.core.render import dot_source, plan_table
from salt.core.schema import dump_schema, load_schema, save_schema

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
    `mode_errors` carries per-mode config errors found while deriving sinks
    (the TEST dead-preds / invalid-writers errors from the parsed
    ``writers:`` block, M3-review fix; the ONNX export-block resolution
    errors — bad ``model_name``, malformed entries — M4-review fix):
    ``validate``/``deadcode`` report them as error-level findings for that
    mode, ``plan``/``plot``/``why`` raise them when the broken mode is
    requested — the affected mode's sinks fall back to anchor-on-all-preds
    so the other modes stay inspectable. `mode_warnings` carries per-mode
    warnings (a trainer config without an ``export:`` block leaves the ONNX
    contract unchecked — design §4.1); ``validate`` reports them
    (promotable with ``--strict``). `sink_origins` enriches missing-sink
    planner errors with the demanding config address (e.g.
    ``export.outputs``), per mode. `writers` carries the `WriterCallback`
    built from the parsed ``writers:`` block (set on trainer configs), so
    ``validate`` can run the TEST writer kind/dtype unification
    (`WriterCallback.validate_specs`, design §2.7/§8) statically — the same
    check `SaltModule._validate_writer_specs` runs at ``salt2 test`` setup,
    now also gated by ``salt2 graph validate``. `model_modules` is the
    model-side subdict (the `validate_specs` `model_modules` argument, kept
    separate from the combined `modules` exactly as the runtime path passes
    `SaltModule._graph_modules`).
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


def _is_trainer_format(raw: Mapping[str, Any]) -> bool:
    """Check whether a parsed YAML mapping is a §5.1 trainer config.

    Returns
    -------
    bool
        True for a top-level ``model:``/``data:`` mapping without the M1
        toy ``modules:`` key.
    """
    return "modules" not in raw and ("model" in raw or "data" in raw)


def load_config(
    path: str | Path | Sequence[str | Path], set_overrides: Sequence[str] | None = None
) -> GraphConfig:
    """Load and instantiate a graph config (both formats — module docstring).

    A top-level ``model:``/``data:`` mapping is a §5.1 trainer config and is
    adapted through `Salt2CLI` (`_load_fit_config`); a top-level ``modules:``
    mapping is the M1 toy format. Instance names are assigned from the
    module-dict keys (design §2.2 — names must match config keys; the
    planner re-checks this invariant).

    Parameters
    ----------
    path : str | Path | Sequence[str | Path]
        The config YAML, or a STACK of trainer configs (deep-merged
        left-to-right through the real `Salt2CLI` surface — the repeatable
        ``-c`` flag; the salt2 fit/export stacking semantics). M1 toy
        graphs take exactly one config.
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
        # config stacking is a trainer-surface feature (the salt2 fit /
        # salt2 export deep-merge); override files may carry any subset of
        # keys, but at least one stacked file must be trainer-format
        if not any(_is_trainer_format(raw) for raw in raws):
            raise ConfigError(
                f"repeated -c is supported for salt2 trainer configs only (deep-merged "
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


def _load_fit_config(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> GraphConfig:
    """Adapt a §5.1 trainer config (stack) into one full-pipeline `GraphConfig` (design §4).

    Parses the config(s) through the REAL `Salt2CLI` surface in run-free mode
    (``base2.yaml`` auto-loaded, deep-merge/null-deletion semantics intact —
    repeated configs stack left-to-right exactly as on ``salt2 fit``),
    then builds the combined dataset + model graph: ``data.modules`` and
    ``model.modules`` form one module dict (the reader is the source node,
    so ``sources`` is empty), the per-mode sinks are the model's declared
    anchors (``loss.total`` / ``preds.*`` + TEST ``meta.rows``), and the
    wildcard-narrowing universe comes from the reader's schema artifact.
    Everything stays config-only — no data file is touched (design §2.3).

    The parsed ``writers:`` block enters the TEST sinks exactly as on the
    runtime path (M3-review fix; design §4.2, §8): a `WriterCallback` is
    built from ``writers.modules`` and its declared demand anchors the TEST
    plan, so ``salt2 graph validate``/``deadcode`` fire the dead-preds hard
    error a real ``salt2 test`` would raise, and writer-demanded
    dataset-namespace keys (labels/masks/``meta.rows``) keep their producers
    alive in the static TEST plan. The resulting per-mode errors are carried
    in `GraphConfig.mode_errors` (see its docstring).

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
    # local import: the trainer surface (lightning/jsonargparse) is heavy
    # and circular with this module (salt.core.main dispatches to cli.main)
    from salt.core.data.processors import Labels  # noqa: PLC0415 - heavy/circular (docstring)

    cli = _parse_trainer_cli(paths, set_overrides)
    model, dm = cli.model, cli.datamodule
    # PER-BATCH namespace only (plan-25 §3.6): setup-only modules
    # (InputSamples/VDS/ShmStage) are partitioned out of the tensor compile, so
    # the combined full-pipeline graph here uses `batch_modules`, NOT the union
    # `dm.modules` (a setup-only module in `compile_plan` trips AllModesDeadError).
    # The setup graph is a distinct topology rendered separately.
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
    writer_cb = _static_writer_callback(cli)
    fitval_callbacks = _static_fitval_callbacks(cli)
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - same-package adapter
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - same-package adapter
    sinks: dict[Mode, tuple[str, ...]] = {}
    mode_errors: dict[Mode, str] = {}
    mode_warnings: dict[Mode, str] = {}
    sink_origins: dict[Mode, dict[str, str]] = {}
    for mode in PRIMARY_MODES:
        if mode is Mode.TEST and writer_cb is not None:
            try:
                keys = list(model._model_sinks(mode, writers=writer_cb, reader=reader))  # noqa: SLF001 - same-package adapter
                demand = writer_cb.writer_demand(model._graph_modules, reader)  # noqa: SLF001 - same-package adapter
                # writer-demanded dataset-namespace keys (labels/masks/meta)
                # are TEST sinks too — their producers stay alive (design §8)
                keys.extend(key for key in demand if key not in keys)
            except ConfigError as err:
                mode_errors[mode] = str(err)
                keys = list(model._model_sinks(mode))  # noqa: SLF001 - all-preds render fallback
        elif mode is Mode.ONNX and writer_cb is not None:
            # the static half of the design §3.1/§4.1 export contract,
            # re-sourced at M4.5: ONNX sinks are the union of the writers'
            # declared manifest ports (the unified output manifest —
            # exactly what `salt2 export` will trace), and the export-only
            # half of the export: block is validated as `salt2 export`
            # does (model_name rule, input entries, rename/combine against
            # the manifest, the export.outputs migration error) — a broken
            # manifest fails HERE instead of months later at export time
            try:
                per_writer = writer_cb.per_writer_onnx_manifest(model._graph_modules, reader)  # noqa: SLF001 - same-package adapter
                manifest = writer_cb.onnx_manifest(model._graph_modules, reader)  # noqa: SLF001 - same-package adapter
                if export_cfg is not None and manifest:
                    # validates the export-only half + rename/combine vs the
                    # manifest, incl. the export.outputs migration error
                    attach_manifest(resolve_export_config(export_cfg, run_name), manifest)
                elif export_cfg is None:
                    mode_warnings[mode] = (
                        "the config has no export: block — outputs derive from the writers "
                        "(checked), but export.inputs/model_name were NOT checked; declare "
                        "the export-only half (design §5.1, §7) so `salt2 graph validate "
                        "--mode onnx` gates everything `salt2 export` will trace"
                    )
                if manifest:
                    keys = [out.port for out in manifest]
                    sink_origins[mode] = {
                        out.port: (
                            f"ONNX manifest output {out.port!r} (writer {wname!r}, config: "
                            f"writers.modules.{wname})"
                        )
                        for wname, entries in per_writer.items()
                        for out in entries
                    }
                else:
                    mode_warnings[mode] = (
                        "the configured writers declare no ONNX outputs (all narrowed out?) "
                        "— ONNX sinks fall back to every preds.* key; check TaskWriter "
                        "onnx/onnx_streams/onnx_tasks (M4.5 unified manifest)"
                    )
                    keys = list(model._model_sinks(mode))  # noqa: SLF001 - all-preds render fallback
            except ConfigError as err:
                mode_errors[mode] = str(err)
                keys = list(model._model_sinks(mode))  # noqa: SLF001 - all-preds render fallback
        elif mode & Mode.TRAINING and fitval_callbacks:
            # the static half of the design §3.1/§3.4 FIT/VAL-sink contract
            # (D-prereq): configured metrics callbacks DECLARE plan sinks the
            # same way writers do for TEST, so `salt2 graph validate --mode
            # fit` sees the same sinks (and the same boundary demand) a real
            # `salt2 fit` does — the prereq for the M5 MaskformerMetrics
            # callback to enter graph validation. Mirror of the TEST branch.
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
                    "the config has no writers: block — the ONNX contract was NOT checked "
                    "(sinks fall back to every preds.* key); the ONNX output manifest "
                    "derives from writers.modules (M4.5 unified manifest; base2.yaml ships "
                    "defaults) so `salt2 graph validate --mode onnx` gates what "
                    "`salt2 export` will trace"
                )
            keys = list(model._model_sinks(mode))  # noqa: SLF001 - same-package adapter
        if mode is Mode.TEST and "meta.rows" not in keys:
            keys.append("meta.rows")  # writer row alignment (design §8)
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
        writers=writer_cb,
        model_modules=dict(model._graph_modules),  # noqa: SLF001 - same-package adapter
        mup_cfg=getattr(model, "mup_cfg", None),
    )


def _parse_trainer_cli(paths: Sequence[Path], set_overrides: Sequence[str] | None) -> Any:
    """Parse a §5.1 trainer config (stack) through the REAL salt2 surface, run-free.

    Repeated configs deep-merge left-to-right (the ``salt2 fit`` /
    ``salt2 export`` stacking semantics) — the static tooling accepts the
    same base + override pattern the trainer surface does.

    Returns
    -------
    Salt2CLI
        The run-free CLI (``cli.model``/``cli.datamodule`` constructed,
        nothing executed, no data touched).

    Raises
    ------
    ConfigError
        When the parse fails (with the documented ``--set`` hint).
    """
    from salt.core.main import Salt2CLI  # noqa: PLC0415 - heavy/circular (module docstring)

    args: list[str] = []
    for path in paths:
        args.extend(["--config", str(path)])
    for entry in set_overrides or []:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        with warnings.catch_warnings():
            # programmatic argv triggers Lightning's 'args parameter is
            # intended...' warning — filtered exactly as salt.core.main and
            # the salt2 export run-free parse do (noise on tooling whose
            # output users are told to read)
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            return Salt2CLI(args=args, run=False)
    except SystemExit as err:
        raise ConfigError(
            f"trainer config {' '.join(str(p) for p in paths)} failed to parse through the "
            f"salt2 surface (parser exit {err.code}; the parser error is printed above). "
            "Required init_args left as overrides in the YAML header can be supplied "
            "data-free via --set, e.g. --set model.modules.norm.init_args.norm_dict=unused.yaml"
        ) from err


def _static_writer_callback(cli: Any) -> Any | None:
    """Build a `WriterCallback` from the run-free CLI's parsed ``writers:`` block.

    The static-tooling half of the design §8 writers-are-sinks contract
    (M3-review fix): the same assembly `Salt2CLI.instantiate_trainer`
    performs at runtime, minus the trainer — `_load_fit_config` feeds the
    callback into `SaltModule._model_sinks` so TEST sink derivation (and the
    dead-preds error) match the runtime path exactly.

    Returns
    -------
    Any | None
        The assembled callback, or None when the config carries no writer
        modules (toy/model-only configs keep the M2 anchor-on-all-preds
        TEST sinks).
    """
    from salt.core.writers import WriterCallback  # noqa: PLC0415 - heavy/circular

    writer_modules = {
        name: writer
        for name, writer in (cli._get(cli.config_init, "writers.modules") or {}).items()  # noqa: SLF001 - same-package adapter
        if writer is not None
    }
    if not writer_modules:
        return None
    return WriterCallback(modules=writer_modules)


def _static_fitval_callbacks(cli: Any) -> list[Any]:
    """The configured FIT/VAL-sink callbacks from the run-free CLI (design §3.1).

    The static-tooling half of the §3.1/§3.4 FIT/VAL-sink contract
    (D-prereq, the `_static_writer_callback` sibling): any assembled
    ``trainer.callbacks`` entry exposing a callable ``fit_val_demand`` (the
    metrics family — `ConfusionMatrix`, the M5 `MaskformerMetrics`) is fed
    into `SaltModule._model_sinks` so FIT/VAL sink derivation matches the
    runtime path. Config-only: the callbacks are instantiated by the parse
    but never ``setup``, and `fit_val_demand` resolves from the model module
    dict, not from `setup` state.

    Returns
    -------
    list[Any]
        The FIT/VAL-sink callbacks in trainer order (empty when none).
    """
    trainer = getattr(cli, "trainer", None)
    callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
    return [cb for cb in callbacks or [] if callable(getattr(cb, "fit_val_demand", None))]


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
    unconsumed ``preds.*`` port in TEST, design §4.2) and stored per-mode
    sink errors (`GraphConfig.mode_errors`); warnings (no schema configured,
    warning-level dead outputs) are promoted to errors under ``--strict``
    (design §4). Info-level findings (unconsumed FIT/VAL preds — the normal
    no-metric-callback case, design §3.3) are report-only and NEVER promoted,
    so ``--strict`` stays usable as the CI default on standard tagger
    configs.

    For TEST, when a ``writers:`` block is configured, the writer kind/dtype
    unification (design §2.7/§8) runs statically here too: each writer-declared
    require's kind/dtype is unified against its producing leaf in the
    just-compiled TEST plan (the static equivalent of the runtime
    `SaltModule._validate_writer_specs` producer union), so a writer declaring
    a wrong kind/dtype (e.g. ``preds.jets.classification`` as ``kind=label`` or
    ``dtype=int64`` where the task publishes ``data``/``float32``) is a fatal
    error HERE — at ``salt2 graph validate``, data-free, in CI — instead of
    surfacing only at ``salt2 test`` setup. Skipped when the TEST sinks already
    carry a stored mode error (the writers block already failed at demand
    assembly; `WriterCallback.requires` would re-raise the same root cause).

    Returns
    -------
    int
        0 on success, 1 on error (or warnings under ``--strict``).
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
        # default-on §2.6 class-names ↔ schema-attrs cross-check (set AND
        # order); raises ConfigError -> formatted by main()
        from salt.core.saltmodule import check_class_names  # noqa: PLC0415 - heavy/circular

        checked = check_class_names(cfg.modules, cfg.reader)
        if checked:
            print(f"OK class_names ↔ schema attrs: {checked} list(s) match, set and order (§2.6)")
    if cfg.model_modules is not None:
        # muP routing validator (design §3.4 line 695): apply_to naming a module
        # without a mup init_arg ERRORS; a mup:true module outside apply_to WARNS.
        # The same `validate_mup_routing` SaltModule construction runs, surfaced
        # here as `salt2 graph validate` findings (warnings promotable under
        # --strict). The hard errors would already abort the parse above; this is
        # the first-class CI check + the warning capture.
        from salt.core.saltmodule import validate_mup_routing  # noqa: PLC0415 - heavy/circular

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
        # edge bind-time validators (FD §6.7 1425-1431): edge-stream-first +
        # EdgeAttention-backend forcing. The same `validate_edge_port` SaltModule
        # construction runs; surfaced here as a first-class CI check (these are
        # HARD errors — they would already abort the parse above; this captures
        # the OK line / the error message for the validate report).
        from salt.core.saltmodule import validate_edge_port  # noqa: PLC0415 - heavy/circular

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
    # data-free module preflights (design §2.3): duck-typed `preflight()`
    # checks file-backed materialise sources (e.g. the Normaliser norm dict).
    # WARNING-level here — `validate` must stay runnable on data-less machines
    # where the documented `--set ...norm_dict=unused.yaml` override is in
    # play; an actual `salt2 fit`/`test` run promotes these to hard errors
    # (SaltModule.setup).
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
        if (
            mode is Mode.TEST
            and cfg.writers is not None
            and cfg.model_modules is not None
            and mode not in cfg.mode_errors
        ):
            # writer kind/dtype unification, statically (design §2.7/§8) — the
            # same check `SaltModule._validate_writer_specs` runs at `salt2 test`
            # setup, now also gated by `salt2 graph validate` so a wrong writer
            # kind/dtype fails data-free in CI instead of only at run setup. The
            # producer universe is assembled from the JUST-COMPILED TEST plan:
            # every step's produced leaves unioned with the mode-active boundary
            # sources, the static equivalent of the runtime union of
            # `model_producer_specs` and `GraphDataset.boundary_specs`. An
            # executed step-produced leaf WINS over a same-named boundary source,
            # exactly as runtime (`_validate_writer_specs` does `boundary_specs`
            # then `.update` model_producer_specs); within the steps the first
            # producer wins (plan order), mirroring the runtime `setdefault`.
            producer_specs: dict[str, TensorSpec] = {}
            for step in plan.steps:
                for key, spec in step.produces.items():
                    producer_specs.setdefault(key, spec)
            for key, spec in plan.sources.items():
                producer_specs.setdefault(key, spec)
            try:
                cfg.writers.validate_specs(cfg.model_modules, cfg.reader, producer_specs)
            except GraphError as err:
                errors.append(f"[mode=TEST] writer spec validation: {err}")
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
# graph deadcode (design §4.2)
# ---------------------------------------------------------------------------


def _cmd_deadcode(args: argparse.Namespace) -> int:
    """``salt2 graph deadcode``: mode-aware dead-output report (design §4.2).

    Findings carry severity levels: an unconsumed ``preds.*`` port in TEST is
    an error by default and makes the command exit 1 (design §4.2), as does a
    stored per-mode sink error (the writers-block dead-preds error,
    `GraphConfig.mode_errors`); unconsumed FIT/VAL preds are info (the normal
    no-metric-callback case, design §3.3); the rest are warnings
    (report-only). A per-task ``expose: [fit, val]`` opt-out (design §4.2, M5
    sub-wave D) gates the prediction port out of TEST before this runs, so an
    opted-out task surfaces here as a (warning-level) whole-module pruning
    rather than a TEST preds error.

    Returns
    -------
    int
        0 when no error-level finding exists; 1 on error-level findings or
        if the graph itself fails to resolve.
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
    """Raise the stored per-mode sink error when the requested mode is broken.

    Raises
    ------
    ConfigError
        The `GraphConfig.mode_errors` entry for `mode` (e.g. the TEST
        dead-preds error derived from the parsed ``writers:`` block, or the
        ONNX export-block resolution error).
    """
    if stored := cfg.mode_errors.get(mode):
        raise ConfigError(stored)


def _print_onnx_static_caveat(cfg: GraphConfig, mode: Mode) -> None:
    """Print the dataset-fed-approximation caveat for static ONNX renderings.

    The static ONNX-mode plan/plot of a trainer config includes the dataset
    modules and anchors on dataset sources; the graph `salt2 export`
    actually traces has positional export inputs, no dataset modules, and
    in-graph reduces — its authoritative rendering is the `plan_onnx.txt`
    written next to ``network.onnx`` at export time (design §4.4; the two
    plan hashes legitimately differ).
    """
    if mode is Mode.ONNX and cfg.reader is not None:
        print(
            "note: this is the dataset-fed STATIC view of the ONNX graph (reader/features "
            "included, no export reduces). The traced export graph is rendered to "
            "plan_onnx.txt next to network.onnx by `salt2 export` (design §4.4)."
        )


# ---------------------------------------------------------------------------
# graph plot (design §4.3)
# ---------------------------------------------------------------------------


def _cmd_plot(args: argparse.Namespace) -> int:
    """``salt2 graph plot``: render the mode graph (design §4.3).

    Emits the §4.3 Graphviz DOT (port-card layout, one signature card per
    module) next to the requested output, then shells out to the ``dot``
    binary — baked into the salt container — to rasterise it: a PNG at the
    requested output path and a sibling PDF. This is the authoritative graph
    image; there is no matplotlib path. When ``dot`` is absent, raises a clear
    actionable error rather than silently degrading.

    Returns
    -------
    int
        0 on success, 1 on graph errors. A missing/failing ``dot`` binary
        surfaces as a `GraphError` (raised by `_render_with_dot`), caught at
        the CLI top level and reported as exit 1.
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
    """Resolve concrete feature widths STATICALLY for the plot (design §2.3, §4.3).

    Compiles every inspectable primary-mode plan and feeds them to
    `resolve_bind_schema` — the same static resolution that builds the
    width-dependent nn.Linear layers, with NO data file and NO batch run. The
    resulting per-key last-dim widths let `dot_source` show the concrete
    FEATURE/embedding dim on each port-card row (``encoded.tracks (B, T:tracks,
    16)``) while the data-dependent batch/sequence dims stay symbolic.

    All compilable modes are unified (widths are config-fixed, so resolving
    across modes is sound and gives the requested mode every cross-mode width):
    a mode that legitimately fails to compile (`GraphConfig.mode_errors`, or a
    planner error on a non-requested mode) is skipped rather than aborting the
    plot — the requested mode is guaranteed present (the caller already compiled
    it).

    Returns
    -------
    dict[str, int]
        ``{dotted_key: concrete_last_dim}`` for every statically resolvable key.
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
    """Rasterise the DOT sidecar to PNG (``out_path``) and a sibling PDF via ``dot``.

    Shells out to the Graphviz ``dot`` binary (baked into the salt container):
    ``dot -Tpng -Gdpi=150 <dot> -o <out>.png`` and ``dot -Tpdf <dot> -o
    <out>.pdf``. The PNG goes to the requested ``out_path``; the PDF takes the
    same stem with a ``.pdf`` suffix.

    Raises
    ------
    GraphError
        When ``dot`` is not on PATH (actionable: rebuild/use the salt
        container, which bakes in Graphviz), or when a ``dot`` invocation
        fails (non-zero exit) — the stderr is surfaced in the message.
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
# graph resolve [--annotate] (design §4.4; M4.5 amendment merge condition 8)
# ---------------------------------------------------------------------------

MANIFEST_BEGIN = "# === salt2 output manifest"
"""First line of the generated annotation block (the replace anchor)."""

MANIFEST_END = "# === end salt2 output manifest ==="
"""Last line of the generated annotation block."""


def _cmd_resolve(args: argparse.Namespace) -> int:
    """``salt2 graph resolve``: the writer-derived output manifest, eval + ONNX.

    Prints the assembled manifest (the §4.4-style answer to "what columns
    does eval write / what does Athena see" for the unified-writer config);
    with ``--annotate``, additionally writes it into the config file as a
    refreshable comment block (replaced in place when already present) —
    the amendment merge condition 8 mitigation for the discoverability
    shift from a hand-typed ``export.outputs`` to derived declarations.
    With repeated ``-c`` the configs deep-merge left-to-right and the
    annotation goes into the LAST (most specific) file.

    Returns
    -------
    int
        0 on success; 1 on a non-trainer config or a manifest error.
    """
    paths = [Path(p) for p in args.config]
    raws = []
    for path in paths:
        if not path.is_file():
            return _fail(f"config file not found: {path}")
        raws.append(yaml.safe_load(path.read_text()))
    if not any(isinstance(raw, dict) and ("model" in raw or "data" in raw) for raw in raws):
        return _fail(
            "salt2 graph resolve needs a salt2 trainer config (top-level model:/data: "
            "blocks) — toy graph configs have no writers block (M4.5 unified manifest)"
        )
    cli = _parse_trainer_cli(paths, args.set)
    writer_cb = _static_writer_callback(cli)
    if writer_cb is None:
        return _fail(
            "the config declares no writer modules — the output manifest derives from "
            "writers.modules (M4.5 unified manifest; base2.yaml ships defaults)"
        )
    block = _manifest_block(cli, writer_cb, paths, args.set)
    print(block)
    if args.annotate:
        _write_annotation(paths[-1], block)
        print(f"\nannotated {paths[-1]} (block refreshed in place on the next run)")
    return 0


def _manifest_block(
    cli: Any, writer_cb: Any, paths: Sequence[Path], set_overrides: Sequence[str] | None
) -> str:
    """Render the eval + ONNX manifest as a config comment block.

    The embedded refresh hint reproduces the FULL generating command —
    every ``-c`` file and every ``--set`` override — so the printed
    command re-runs verbatim on configs whose required init_args (e.g.
    ``norm_dict``) are supplied data-free (the merge-condition-8
    self-documenting staleness contract).

    Returns
    -------
    str
        Lines between `MANIFEST_BEGIN` and `MANIFEST_END`, all
        ``#``-prefixed (safe to append to any YAML file).
    """
    model, dm = cli.model, cli.datamodule
    modules = model._graph_modules  # noqa: SLF001 - same-package adapter
    reader = dm.reader
    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001 - same-package adapter
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - same-package adapter
    refresh_args = " ".join([
        *(f"-c {path.name}" for path in paths),
        *(f"--set {entry}" for entry in set_overrides or []),
    ])
    refresh = f"(generated — refresh: salt2 graph resolve {refresh_args} --annotate)"
    lines = [
        f"{MANIFEST_BEGIN} {refresh} ===",
        f"# eval columns (salt2 test; prefix = run name {run_name!r}):",
    ]
    for wname, streams in writer_cb.column_manifests(modules, reader, run_name).items():
        if not streams:
            lines.append(
                f"#   [{wname}] file-dependent or no static columns (e.g. source-file "
                "copies ride along with file dtypes)"
            )
            continue
        lines.extend(
            f"#   [{wname}] {stream}: {' '.join(columns)}" for stream, columns in streams.items()
        )
    manifest = writer_cb.onnx_manifest(modules, reader)
    if not manifest:
        lines.extend((
            "# onnx outputs (salt2 export): NONE — the writers declare no ONNX outputs",
            MANIFEST_END,
        ))
        return "\n".join(lines)
    from dataclasses import replace  # noqa: PLC0415 - stdlib, annotation-path only

    from salt.core.onnx.config import ExportConfig, ExportInput  # noqa: PLC0415 - heavy package

    export_half = export_cfg if export_cfg is not None else ExportConfig()
    if not export_half.inputs:
        # the manifest needs no inputs — satisfy the export-half resolution
        # for this print-only path (the salt2 export --manifest precedent)
        export_half = replace(export_half, inputs=[ExportInput(port="inputs.placeholder")])
    resolved = attach_manifest(resolve_export_config(export_half, run_name), manifest)
    source_note = " — default from run name):" if export_cfg is None else "):"
    lines.append(f"# onnx outputs (salt2 export; model_name {resolved.model_name!r}{source_note}")
    lines.extend(f"#   {line}" for line in manifest_table(resolved).splitlines()[1:])
    lines.append(MANIFEST_END)
    return "\n".join(lines)


def _write_annotation(path: Path, block: str) -> None:
    """Insert or refresh the manifest comment block in a config file.

    A previous generated block (between `MANIFEST_BEGIN` and
    `MANIFEST_END`) is replaced in place; otherwise the block is appended
    at the end of the file — comments are inert YAML, so the config parses
    identically (the §4.4 converter-emitted comment-block contract).
    """
    text = path.read_text()
    new_lines: list[str] = []
    replaced = False
    skipping = False
    for line in text.splitlines():
        if line.startswith(MANIFEST_BEGIN):
            skipping = True
            replaced = True
            new_lines.append(block)
            continue
        if skipping:
            if line.startswith(MANIFEST_END):
                skipping = False
            continue
        new_lines.append(line)
    if not replaced:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(block)
    path.write_text("\n".join(new_lines) + "\n")


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
        "-c",
        "--config",
        required=True,
        action="append",
        help="config YAML (salt2 trainer config or M1 toy graph). Repeatable: trainer "
        "configs deep-merge left-to-right (the salt2 fit/export stacking semantics); "
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
    """Add the ``mup-shapes`` / ``mup-coord-check`` subcommands (design §3.4, §9.2).

    Deferred to `salt.core.mup` handlers (heavy mup/pandas imports stay out of
    the graph-tooling startup path). The casing-fixed ``setup_mup`` console
    entry forwards to ``mup-shapes`` (pyproject.toml).
    """
    from salt.core.mup import cmd_mup_coord_check, cmd_mup_shapes  # noqa: PLC0415

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
        help="SHARED base/delta infshape file applied at EVERY swept width (salt2 mup-shapes "
        "output). Omit to auto-generate one (base=min width, delta=max width) — the shared-base "
        "protocol that gives a correct width_mult so the MuReadout is damped (the MU-HUMAN fix, "
        "replacing the per-width self-base that forced width_mult==1)",
    )
    coord.set_defaults(func=cmd_mup_coord_check)


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
