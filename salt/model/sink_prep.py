"""Trainer-independent sink preparation shared by `SaltModule` (runtime) and `salt graph`
(static): bind manifests, select the TEST/ONNX sinks, fold them as graph nodes, derive
per-mode plan anchors and callback demand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from salt.data.dataset import MODEL_VISIBLE_NAMESPACES
from salt.graph.errors import ConfigError
from salt.graph.planner import Plan
from salt.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    _has_wildcard,
    flatten_spec,
)
from salt.outputs.sinks.sink import Node, is_test_persistence_sink

__all__ = [
    "ModeSinks",
    "PreparedSinks",
    "assert_no_dead_preds",
    "bind_manifests",
    "boundary_demand",
    "callback_demand",
    "dead_preds_message",
    "fold_sink_node",
    "is_section_sink",
    "model_sinks",
    "prepare_sinks",
    "select_fitval_callbacks",
    "select_onnx_sink",
    "select_test_sink",
]


@dataclass(frozen=True)
class ModeSinks:
    """One mode's sink-selection outcome, as gathered by `prepare_sinks`.

    Parameters
    ----------
    anchors : list[str]
        The ``compile_plan(sinks=...)`` argument for this mode.
    anchor_origins : dict[str, str] | None
        Callback-demand map for TRAINING modes with fit/val callbacks
        configured, else None (always None for TEST/ONNX).
    error : str | None
        Collected `ConfigError` text (only set when `prepare_sinks` was
        called with ``collect_errors=True``).
    warning : str | None
        The ONNX export-contract warning, when applicable.
    """

    anchors: list[str]
    anchor_origins: dict[str, str] | None
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class PreparedSinks:
    """The result of `prepare_sinks`: folded modules, selected sinks, per-mode anchors.

    Parameters
    ----------
    modules : dict[str, GraphModule]
        `model_modules` plus the folded TEST node (and the ONNX node iff
        `Mode.ONNX` was requested).
    test_sink : Any | None
        The selected TEST persistence sink node, or None.
    onnx_sink : Any | None
        The selected `OnnxExportSink`, or None.
    writer_demand : dict[str, str] | None
        ``test_sink.writer_demand(model_modules, reader)``, or None.
    by_mode : dict[Mode, ModeSinks]
        Per requested mode, its sink-selection outcome.
    fitval_callbacks : tuple[Any, ...]
        Callbacks declaring FIT/VAL demand; `boundary_demand` derives TRAINING demand from them.
    """

    modules: dict[str, GraphModule]
    test_sink: Any | None
    onnx_sink: Any | None
    writer_demand: dict[str, str] | None
    by_mode: dict[Mode, ModeSinks]
    fitval_callbacks: tuple[Any, ...] = ()


def is_section_sink(entry: Any) -> bool:
    """Whether an ``outputs:`` section entry is a SINK rather than a writer."""
    if isinstance(entry, Node):
        return True
    is_sink = getattr(entry, "is_sink", None)
    return callable(is_sink) and bool(is_sink())


def select_test_sink(sinks: Sequence[Any]) -> Node | None:
    """The TEST persistence sink among `sinks` (first callable ``writer_demand`` with
    `is_test_persistence_sink` true), or None; raises `ConfigError` if the selected
    sink is not a graph node.
    """
    sink = next(
        (
            s
            for s in sinks
            if callable(getattr(s, "writer_demand", None)) and is_test_persistence_sink(s)
        ),
        None,
    )
    if sink is None:
        return None
    is_sink = getattr(sink, "is_sink", None)
    has_io = callable(getattr(sink, "declare_io", None))
    if not (has_io and callable(is_sink) and bool(is_sink())):
        raise ConfigError(
            f"TEST persistence sink {getattr(sink, 'name', type(sink).__name__)!r} "
            f"({type(sink).__name__}) is not a graph node — a persistence sink must provide "
            "declare_io() and is_sink() -> True (subclass salt.outputs.sinks.sink.RuntimeSink); "
            "the non-node writer_demand fallback was removed"
        )
    return sink


def select_onnx_sink(sinks: Sequence[Any]) -> Any | None:
    """The first `OnnxExportSink` among `sinks`, or None."""
    from salt.outputs.sinks.onnx_sink import OnnxExportSink

    return next((sink for sink in sinks if isinstance(sink, OnnxExportSink)), None)


def select_fitval_callbacks(callbacks: Iterable[Any] | None) -> list[Any]:
    """The callbacks among `callbacks` that declare FIT/VAL plan sinks."""
    return [cb for cb in callbacks or [] if callable(getattr(cb, "fit_val_demand", None))]


def bind_manifests(
    sinks: Iterable[Any], model_modules: Mapping[str, Any], output_section: Mapping[str, Any]
) -> None:
    """Bind both manifest sources (outputs: section + model graph modules) to every
    sink before any declare_io/writer_demand resolution; duck-typed, so a sink
    implementing neither bind method is left alone.
    """
    for sink in sinks:
        if callable(getattr(sink, "bind_model_modules", None)):
            sink.bind_model_modules(model_modules)
        if output_section and callable(getattr(sink, "bind_output_section", None)):
            sink.bind_output_section(output_section)


def fold_sink_node(
    modules: Mapping[str, GraphModule], sink_node: GraphModule
) -> dict[str, GraphModule]:
    """Fold a sink NODE into a fresh copy of `modules` under its instance name (the
    planner requires ``module.name == config_key``); raises `ConfigError` on a
    name collision with an existing module.
    """
    name = sink_node.name
    if name in modules:
        raise ConfigError(
            f"sink node name {name!r} collides with a model module — instance names must be "
            "unique across the pipeline graph; rename the callback key"
        )
    folded = dict(modules)
    folded[name] = sink_node
    return folded


def callback_demand(
    model_modules: Mapping[str, Any], mode: Mode, callbacks: Sequence[Any]
) -> dict[str, str]:
    """Merged callback-declared demand for one TRAINING mode; empty outside `Mode.TRAINING`."""
    if not (mode & Mode.TRAINING):
        return {}
    out: dict[str, str] = {}
    for cb in callbacks:
        who = f"callback {type(cb).__name__!r}"
        for key in cb.fit_val_demand(model_modules):
            out.setdefault(key, who)
    return out


def model_sinks(
    model_modules: Mapping[str, Any], mode: Mode, *, callback_keys: Iterable[str] = ()
) -> list[str]:
    """The model-plan sink anchors for one mode: TRAINING is ``loss.total`` plus
    `callback_keys` (demand only ADDS sinks, never narrows), else every declared
    ``preds.*`` key; raises `ConfigError` if the mode's anchor key is unproduced.
    """
    produced: dict[str, str] = {}
    for name, module in model_modules.items():
        for key, spec in flatten_spec(module.declare_io(mode).produces).items():
            if spec.active_in(mode):
                produced.setdefault(key, name)
    if mode & Mode.TRAINING:
        if "loss.total" not in produced:
            raise ConfigError(
                f"no module produces 'loss.total' in mode {mode.name} — training plans "
                "anchor on it; add a LossSum module"
            )
        sinks = ["loss.total"]
        for key in callback_keys:
            if key not in sinks:
                sinks.append(key)
        return sinks
    preds = [key for key in produced if key.split(KEY_SEP, 1)[0] == "preds"]
    # the ONNX output manifest is not writer-derived — the folded OnnxExportSink
    # anchors its conversion-leaf demand itself. When no export sink demand is
    # present it falls back to every preds.* key below.
    if not preds:
        raise ConfigError(
            f"no module produces a 'preds.*' key in mode {mode.name} — evaluation plans "
            "anchor on predictions"
        )
    return preds


def assert_no_dead_preds(model_modules: Mapping[str, Any], plan: Plan) -> None:
    """Hard-error (`ConfigError`) on a produced ``preds.*`` key consumed by no plan
    edge — with a sink NODE folded, the sink demands ``outputs.*`` not ``preds.*``,
    so a genuinely-dead ``preds.*`` would otherwise ship silently.
    """
    produced: dict[str, str] = {}
    for name, module in model_modules.items():
        for key, spec in flatten_spec(module.declare_io(Mode.TEST).produces).items():
            if key.split(KEY_SEP, 1)[0] == "preds" and spec.active_in(Mode.TEST):
                produced.setdefault(key, name)
    # a plan edge exists only between surviving nodes — absent here means no consumer
    consumed = {edge.key for edge in plan.edges}
    if dead := [key for key in produced if key not in consumed]:
        raise ConfigError(dead_preds_message(dead, produced))


def dead_preds_message(dead: list[str], produced: Mapping[str, str]) -> str:
    """Build the TEST dead-preds hard-error text: names each dead key's producing
    module and offers ``expose: [fit, val]`` or ``--model.modules.X=null`` as fixes.
    """
    lines = ["[mode=TEST] prediction keys consumed by NO writer:"]
    for key in dead:
        name = produced.get(key)
        where = f" (produced by {name!r}, config: model.modules.{name})" if name else ""
        lines.append(f"  - {key!r}{where}")
    lines.append(
        "an unconsumed preds.* port in TEST means a computed prediction is never persisted."
    )
    fix = "  fix: widen the writers"
    targets = sorted({produced[key] for key in dead if key in produced})
    if targets:
        expose_form = " / ".join(
            f"--model.modules.{name}.init_args.expose=[fit,val]" for name in targets
        )
        null_form = " / ".join(f"--model.modules.{name}=null" for name in targets)
        # expose first — it keeps the task training and only prunes its TEST/ONNX
        # prediction; --model.modules.X=null deletes the task outright.
        fix += (
            f", or opt the task out of eval with expose: [fit, val] ({expose_form}), "
            f"or remove the task module entirely ({null_form})"
        )
    else:
        fix += ", or opt the task out of eval with expose: [fit, val], or remove the task module"
    lines.append(fix)
    return "\n".join(lines)


def boundary_demand(
    model_modules: Mapping[str, Any], prepared: PreparedSinks
) -> dict[Mode, tuple[list[str], dict[str, str]]]:
    """Per-mode dataset-boundary demand + per-key demander descriptions.

    In TEST, the selected sink's ``writer_demand`` extends the demand so
    writer-demanded dataset keys keep their producers alive. In FIT/VAL, a
    configured callback's dataset-namespace demand does the same.

    Parameters
    ----------
    model_modules : Mapping[str, Any]
        The model-side graph modules to scan for required/produced keys.
    prepared : PreparedSinks
        The result of `prepare_sinks` for the same `model_modules`, with ANY
        `modes` (even ``()``): only its selection, `writer_demand` and
        `fitval_callbacks` are read — per-mode anchors are not needed here.

    Raises
    ------
    ConfigError
        On a wildcard demand key or a demand key outside the dataset-served
        namespaces.
    """
    out: dict[Mode, tuple[list[str], dict[str, str]]] = {}
    # ONNX export feeds the model directly and has no dataset plan
    for mode in (Mode.FIT, Mode.VAL, Mode.TEST):
        required: dict[str, list[str]] = {}
        produced: set[str] = set()
        for name, module in model_modules.items():
            io = module.declare_io(mode)
            for key, spec in flatten_spec(io.requires).items():
                if spec.active_in(mode) and not spec.optional:
                    required.setdefault(key, []).append(name)
            produced.update(
                key for key, spec in flatten_spec(io.produces).items() if spec.active_in(mode)
            )
        demand = [key for key in required if key not in produced]
        for key in demand:
            if _has_wildcard(key):
                raise ConfigError(
                    f"boundary demand key {key!r} (mode {mode.name}) contains a wildcard — "
                    "narrow it before compile"
                )
            if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                consumers = ", ".join(
                    f"{name!r} (config: model.modules.{name})" for name in required[key]
                )
                raise ConfigError(
                    f"[mode={mode.name}] key {key!r} is required by module(s) {consumers} "
                    "but no model module produces it, and it cannot come from the dataset "
                    f"(the dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                    f"keys only).\n  fix: add or restore a module producing "
                    f"{key!r}, or correct the requiring module's config"
                )
        origins = {
            key: f"{required[key][0]!r} (config: model.modules.{required[key][0]})"
            for key in demand
        }
        if mode is Mode.TEST:
            for key, who in (prepared.writer_demand or {}).items():
                if key == "meta.rows" or key in produced or key in required:
                    continue  # appended below / a plan sink / already demanded
                if _has_wildcard(key):
                    raise ConfigError(
                        f"writer demand key {key!r} ({who}) contains a wildcard — "
                        "writer requires are concrete keys"
                    )
                if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                    raise ConfigError(
                        f"[mode=TEST] key {key!r} is required by {who} but no model "
                        "module produces it, and it cannot come from the dataset (the "
                        f"dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                        "keys only).\n  fix: correct the writer's "
                        "requires, or add a module producing the key"
                    )
                demand.append(key)
                origins[key] = who
            demand.append("meta.rows")
        elif mode & Mode.TRAINING:
            # FIT/VAL callback demand (mirrors the TEST writer block above): a
            # callback's dataset-namespace requires that no task already demands
            # extend the boundary so their producers survive. Model-produced
            # (preds.*) callback keys are anchored by `model_sinks`, not here.
            for key, who in callback_demand(model_modules, mode, prepared.fitval_callbacks).items():
                if key in produced or key in required:
                    continue  # a model-plan sink / already task-demanded
                if _has_wildcard(key):
                    raise ConfigError(
                        f"callback demand key {key!r} ({who}) contains a wildcard — "
                        "callback requires are concrete keys"
                    )
                if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                    raise ConfigError(
                        f"[mode={mode.name}] key {key!r} is required by {who} but no "
                        "model module produces it, and it cannot come from the dataset "
                        f"(the dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                        "keys only).\n  fix: correct the callback's "
                        "requires, or add a module producing the key"
                    )
                demand.append(key)
                origins[key] = who
        out[mode] = (demand, origins)
    return out


def prepare_sinks(
    model_modules: Mapping[str, GraphModule],
    sinks: Iterable[Any],
    *,
    output_section: Mapping[str, Any],
    reader: Any,
    callbacks: Iterable[Any] | None,
    run_name: str = "salt",
    modes: Sequence[Mode] = PRIMARY_MODES,
    collect_errors: bool = False,
    anchor_meta_rows: bool = False,
    writer_demand: bool = True,
) -> PreparedSinks:
    """Bind, select, fold and anchor sinks for every requested mode, once.

    Shared by `SaltModule` (runtime, trainer-attached inputs) and
    ``salt graph`` (static, config-parsed inputs): binds both manifest
    sources onto every sink, selects the TEST persistence sink and the ONNX
    export sink, folds them into the planning module dict, and derives each
    requested mode's plan anchors (plus callback-demand origins for
    TRAINING modes).

    Parameters
    ----------
    model_modules : Mapping[str, GraphModule]
        The model-side graph modules (pre-fold).
    sinks : Iterable[Any]
        Every sink reachable by the caller.
    output_section : Mapping[str, Any]
        The ``outputs:`` section writers sinks bind their copy spec from.
    reader : Any
        The attached dataset reader; a TEST sink is only selected (and
        folded) when this is not None.
    callbacks : Iterable[Any] | None
        Candidate FIT/VAL callbacks; filtered to those declaring demand.
    run_name : str
        Defaults an ONNX sink's `model_name` and validates its export
        contract.
    modes : Sequence[Mode]
        The modes to compute `ModeSinks` for.
    collect_errors : bool
        When True, a `ConfigError` raised while deriving one mode's anchors
        is caught onto that mode's `ModeSinks.error` instead of propagating.
    anchor_meta_rows : bool
        When True, append ``"meta.rows"`` to the TEST anchors whenever no
        TEST sink was selected (the CLI's flat-sink convention; the runtime
        gets ``meta.rows`` through the dataset boundary instead).
    writer_demand : bool
        When False, skip the TEST sink's `writer_demand` call — callers that
        consume no boundary demand let `declare_io(TEST)` resolve it at compile.

    Raises
    ------
    ConfigError
        Propagated from sink selection, the TEST sink's writer_demand (when
        writer_demand is True) or anchor derivation whenever collect_errors
        is False (or unconditionally for TEST, which never collects).
    """
    sinks = tuple(sinks)
    bind_manifests(sinks, model_modules, output_section)
    test_sink = select_test_sink(sinks) if reader is not None else None
    test_demand: dict[str, str] | None = None
    if writer_demand and test_sink is not None:
        test_demand = test_sink.writer_demand(model_modules, reader)
    onnx_sink = select_onnx_sink(sinks) if Mode.ONNX in modes else None
    onnx_has_contract = False
    if onnx_sink is not None:
        onnx_has_contract = bool(onnx_sink.inputs) or onnx_sink.model_name is not None
        if onnx_sink.model_name is None:
            from salt.outputs.sinks.onnx.config import sanitised_model_name

            onnx_sink.model_name = sanitised_model_name(run_name)

    modules = dict(model_modules)
    if test_sink is not None:
        modules = fold_sink_node(modules, test_sink)
    if onnx_sink is not None:
        modules = fold_sink_node(modules, onnx_sink)

    fitval = select_fitval_callbacks(callbacks)

    by_mode: dict[Mode, ModeSinks] = {}
    for mode in modes:
        error: str | None = None
        warning: str | None = None
        anchor_origins: dict[str, str] | None = None
        if mode is Mode.TEST:
            anchors = [] if test_sink is not None else model_sinks(model_modules, mode)
            if anchor_meta_rows and test_sink is None and "meta.rows" not in anchors:
                anchors.append("meta.rows")
        elif mode is Mode.ONNX:
            if onnx_sink is not None:
                anchors = []
                if onnx_has_contract:
                    try:
                        onnx_sink.export_config(run_name)
                    except ConfigError as err:
                        if not collect_errors:
                            raise
                        error = str(err)
                else:
                    warning = (
                        "the config declares no export contract — the OnnxExportSink names the "
                        "outputs, but its inputs:/model_name: were NOT checked; declare the "
                        "export-only half on the sink so `salt graph validate --mode onnx` gates "
                        "everything `salt export` will trace"
                    )
            else:
                warning = (
                    "the config declares no OnnxExportSink — the ONNX contract was NOT checked "
                    "(sinks fall back to every preds.* key); the ONNX output manifest is "
                    "declared by an OnnxExportSink (callbacks.onnx_export) naming "
                    "the conversion outputs.* leaves, so `salt graph validate --mode onnx` "
                    "gates what `salt export` will trace"
                )
                anchors = model_sinks(model_modules, mode)
        elif mode & Mode.TRAINING:
            if fitval:
                try:
                    demand = callback_demand(model_modules, mode, fitval)
                    anchors = model_sinks(model_modules, mode, callback_keys=demand)
                    anchor_origins = dict(demand)
                except ConfigError as err:
                    if not collect_errors:
                        raise
                    error = str(err)
                    anchors = model_sinks(model_modules, mode)
                    anchor_origins = None
            else:
                anchors = model_sinks(model_modules, mode)
        by_mode[mode] = ModeSinks(
            anchors=anchors, anchor_origins=anchor_origins, error=error, warning=warning
        )

    return PreparedSinks(
        modules=modules,
        test_sink=test_sink,
        onnx_sink=onnx_sink,
        writer_demand=test_demand,
        by_mode=by_mode,
        fitval_callbacks=tuple(fitval),
    )
