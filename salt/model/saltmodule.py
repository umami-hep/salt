"""`SaltModule` — the LightningModule owning the configured graph-module dict.

Compiles per-mode execution plans against the datamodule's dataset boundary,
binds modules to the resolved schema exactly once, and drives the executor
from the Lightning step hooks.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import lightning
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from salt.data.datamodule import SaltDataModule
from salt.data.dataset import MODEL_VISIBLE_NAMESPACES, SaltDataset
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.executor import Executor
from salt.graph.planner import Plan, compile_plan
from salt.graph.spec import (
    KEY_SEP,
    GraphModule,
    Mode,
    SinkModule,
    TensorSpec,
    _has_wildcard,
    flatten_spec,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.bind import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.model.modules.losses import LossGLS, LossSum
from salt.optim import HybridMuonAdamW, Lion
from salt.outputs.sinks.sink import is_test_persistence_sink
from salt.schedule import (
    EarlyStopTracker,
    LRSchedulerConfig,
    StageConfig,
    TrainingSchedule,
    apply_stage_freeze,
    boundary_record,
    clear_frozen_grads,
    reducer_safe_freeze_required,
    trainable_named_params,
)
from salt.utils.logging import get_logger

try:
    from lion_pytorch import Lion as ReferenceLion

    _lion_available = True
except ImportError:
    _lion_available = False

__all__ = [
    "CKPT_KEY",
    "SaltModule",
    "check_class_names",
    "module_mup_enabled",
    "module_supports_mup",
    "resolve_origin_weighting",
    "safe_pct_start",
    "validate_edge_port",
    "validate_mup_routing",
]

# the only attention backend an EdgeAttention encoder may declare — other
# backends would silently bypass the edge computation.
_EDGE_OK_BACKENDS = frozenset({"torch-math"})

CKPT_KEY = "salt_core"
"""Checkpoint dict key for the schema + plan-hash payload."""

_LOG = get_logger(__name__)

_LRS_REQUIRED = ("initial", "max", "end", "pct_start")
_OPTIMIZERS = ("AdamW", "lion", "lion-pytorch", "HybridMuonAdamW")
_MUP_KEYS = frozenset({"apply_to", "shape_path"})


def safe_pct_start(pct_start: float, total_steps: int) -> float:
    """`pct_start` clamped so neither OneCycleLR phase is empty on a short run.

    OneCycleLR divides by each phase's length, so a small ``pct_start`` on a
    short run (smoke test, ``limit_train_batches``, ``salt profile model``)
    raises ``ZeroDivisionError`` in its constructor (e.g. ``pct_start: 0.01``
    at 100 steps). The clamp keeps the phase boundary strictly inside
    ``(0, total_steps - 1)`` and is a no-op for any run long enough for the
    configured warm-up to span two steps.
    """
    low = 2.0 / total_steps if total_steps > 0 else 1.0
    high = 1.0 - 1.0 / total_steps if total_steps > 0 else 0.0
    if low > high:  # fewer than 3 steps: no schedule is meaningful, just be legal
        return 0.75
    return min(max(pct_start, low), high)


_is_test_persistence_sink = is_test_persistence_sink
"""Local alias for the shared selector (`salt.outputs.is_test_persistence_sink`)."""


def _resolve_lr_scheduler_class(class_path: str) -> type:
    """Import a stage `lr_scheduler.class_path` to its class. Reuses the
    CLI's class-path resolver (local import avoids a load-time cycle).

    Raises
    ------
    ConfigError
        The dotted path is not importable / not a class.
    """
    from salt.main import _resolve_class_path

    try:
        cls = _resolve_class_path(class_path)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ConfigError(
            f"training_schedule lr_scheduler.class_path {class_path!r} is not importable: {exc}"
        ) from exc
    if not isinstance(cls, type):
        raise ConfigError(
            f"training_schedule lr_scheduler.class_path {class_path!r} does not name a class."
        )
    return cls


def _reachable_sinks(model: Any) -> list[Any]:
    """Every sink `model` can reach: trainer registry plus its own section sinks.

    Section sinks are appended (deduplicated by identity) so the programmatic
    path (``SaltModule(outputs=...)``, no trainer to register on) still resolves.
    """
    from salt.outputs.sinks.registry import iter_sinks

    found = list(iter_sinks(getattr(model, "_trainer", None)))
    for sink in getattr(model, "_section_sinks", ()) or ():
        if not any(seen is sink for seen in found):
            found.append(sink)
    return found


def _is_section_sink(entry: Any) -> bool:
    """Whether an ``outputs:`` section entry is a SINK rather than a writer.

    Cannot select on the structural `SinkModule` Protocol — every section
    writer matches it too (`SaltModelModule` defines ``is_sink``). The
    discriminator is the `Node` base plus a duck-typed ``is_sink()``.
    """
    from salt.outputs.sinks.sink import Node

    if isinstance(entry, Node):
        return True
    is_sink = getattr(entry, "is_sink", None)
    return callable(is_sink) and bool(is_sink())


def _is_metric_driven_scheduler(cls: type) -> bool:
    """Whether `cls` is a metric-driven LR scheduler — i.e. a `ReduceLROnPlateau`
    (sub)class whose ``step(metrics)`` needs a monitored value. This is
    the deterministic detection rule for the `monitor`-required check.
    """
    return issubclass(cls, torch.optim.lr_scheduler.ReduceLROnPlateau)


class SaltModule(lightning.LightningModule):
    """Lightning wrapper around a configured graph-module dict.

    Parameters
    ----------
    modules : dict[str, SaltModelModule | None]
        Model-side graph modules by instance name. ``None`` entries are
        dropped (deletion via ``--model.modules.X=null``).
    lrs : Mapping[str, float]
        OneCycleLR schedule config: required keys ``initial``, ``max``,
        ``end``, ``pct_start``; optional ``weight_decay`` (default 1e-5) and
        ``last_epoch`` (default -1).
    optimizer : str, optional
        One of ``"AdamW"`` (default), ``"lion"`` (`salt.optim.Lion`,
        ``_foreach_``-batched, bit-identical to the reference), ``"lion-pytorch"``
        (per-parameter reference, kept for equivalence checks), or
        ``"HybridMuonAdamW"``. With `mup` configured the optimizer is swapped
        to ``mup.optim.MuAdamW`` regardless of this name.
    mup : Mapping[str, Any], optional
        muP routing config; ``None`` = no muP. ``apply_to`` (required) lists
        module instance names carrying ``mup: true`` (each must exist and
        accept a ``mup`` init_arg). ``shape_path`` (optional) is a base-shapes
        file from ``salt mup-shapes``, applied at bind time.
    training_schedule : dict, optional
        Staged-training schedule ``{"stages": {name: {...}}}``. Config home is
        the TOP-LEVEL ``training_schedule:`` key; the CLI injects it into this
        arg and rejects a nested ``model.init_args.training_schedule``. Each
        stage may declare an epoch budget, a `frozen`/`trainable` module list,
        and per-stage `optimizer`/`lrs` overrides. ``None`` desugars to one
        `fit` stage using the top-level ``lrs:``/``optimizer:`` (bitwise-
        identical to unscheduled training). Multi-stage schedules run inside a
        single `trainer.fit()` via the auto-injected `TrainingScheduleCallback`.
    name : str, optional
        Model name (run naming/metadata), by default ``"salt"``.
    debug : bool, optional
        Run the executor in debug mode (read tracking + in-place mutation
        detection), by default False. Slow — tests only.

    Raises
    ------
    ConfigError
        For an empty module dict, a non-`SaltModelModule` module, a bad
        optimizer name, or missing `lrs` keys.
    """

    def __init__(
        self,
        modules: dict[str, SaltModelModule | None],
        lrs: Mapping[str, float],
        optimizer: str = "AdamW",
        mup: Mapping[str, Any] | None = None,
        # BARE `dict` (not Mapping[str, Any]): a subscripted type makes jsonargparse
        # eagerly instantiate nested {class_path, init_args} specs — impossible for
        # an LR scheduler (no optimizer yet). Bare `dict` keeps the specs raw.
        training_schedule: dict | None = None,
        name: str = "salt",
        debug: bool = False,
        outputs: dict[str, SaltModelModule | None] | None = None,
    ) -> None:
        super().__init__()
        # a null entry — from a config-file or CLI override — deletes the module.
        modules = {key: module for key, module in modules.items() if module is not None}
        if not modules:
            raise ConfigError("SaltModule needs a non-empty module dict")
        for key, module in modules.items():
            if not isinstance(module, SaltModelModule):
                raise ConfigError(
                    f"module {key!r} ({type(module).__name__}) is not a SaltModelModule — "
                    "model-graph entries must subclass SaltModelModule; wrap or extend it"
                )
            # instance names come from the config dict key, before any declare_io/compile
            module.name = key
        # narrow the losses.** wildcard before any declare_io — wildcard requires are rejected
        for module in modules.values():
            if isinstance(module, LossSum) and not module.narrowed:
                module.narrow(LossSum.collect_loss_keys(modules, Mode.FIT))
            # GLS does not utilise task weights — fail loudly here, before declare_io
            if isinstance(module, LossGLS):
                LossGLS.check_task_weights(modules)
        if missing := [k for k in _LRS_REQUIRED if k not in lrs]:
            raise ConfigError(
                f"lrs is missing required keys {missing} — the OneCycleLR schema is "
                f"{list(_LRS_REQUIRED)} (+ optional weight_decay, last_epoch)"
            )
        if optimizer not in _OPTIMIZERS:
            raise ConfigError(
                f"optimizer {optimizer!r} is not supported — choose from {list(_OPTIMIZERS)}"
            )
        self.mup_cfg: dict[str, Any] | None = _validate_mup(mup, modules)
        # validated against model-module names NOW, before outputs: writers fold in.
        # The schedule is the single home for optimizer/LR config: no-schedule
        # configs desugar to one `fit` stage so configure_optimizers has ONE path.
        self._schedule: TrainingSchedule = (
            TrainingSchedule.from_config(training_schedule, tuple(modules))
            if training_schedule is not None
            else TrainingSchedule.desugar_legacy(tuple(modules))
        )
        # advanced by TrainingScheduleCallback at each stage boundary
        self._current_stage_index = 0
        # set by _apply_stage_freeze; re-asserted every epoch via `train`
        self._frozen_module_names: set[str] = set()
        # per-stage early-stop runtime state — inert unless some stage declares
        # `early_stop`, so unscheduled runs write no new checkpoint state.
        self._stage_start_epoch = 0
        self._early_stop_tracker: EarlyStopTracker | None = None
        self._boundary_records: list[dict[str, Any]] = []
        self._pending_early_advance = False
        # when True, schedule-managed params keep requires_grad=True at DDP wrap
        # (unfreeze stays rank-synced); "frozen" is enforced by optimizer-exclusion
        # + eval + per-step grad clearing. False = requires_grad-based freeze.
        self._reducer_safe_freeze: bool = False
        _validate_edge_port(modules)
        # resolved config only — the module dict is NOT pickled into hparams
        # (load_from_checkpoint takes modules= explicitly)
        self.save_hyperparameters(logger=False, ignore=["modules"])
        self.name = name
        self.lrs = dict(lrs)
        self.optimizer = optimizer
        self.debug = debug
        self.net = nn.ModuleDict(modules)  # ckpt keys: net.<name>.* (dict order, not topo)
        # model-only by construction: modules + non-manifest-only outputs: writers,
        # never a terminal sink.
        self._graph_modules: dict[str, SaltModelModule] = dict(modules)
        # bind the LIVE dict so section writers folded in below stay visible
        for module in modules.values():
            if callable(getattr(module, "bind_model_modules", None)):
                module.bind_model_modules(self._graph_modules)
        self._output_section: dict[str, SaltModelModule | SinkModule] = {}
        # section sinks, partitioned out: neither graph modules nor manifest
        # entries. On the programmatic path (no trainer) this list IS the registry.
        self._section_sinks: list[Any] = []
        self.plans: dict[Mode, Plan] = {}
        self._executors: dict[Mode, Executor] = {}
        self.schema: ResolvedSchema | None = None
        self._bound = False
        self._materialised = False
        self._loaded_from_checkpoint = False
        self._ckpt_plan_hashes: dict[str, str] = {}
        # --init_from warm start (fresh trainer state; distinct from resume
        # ckpt_path). setup("fit") runs the load after bind; on_fit_start
        # materialises only modules that received no checkpoint weights.
        self._init_from: str | None = None
        self._init_warm_started = False
        self._init_loaded_modules: set[str] = set()
        self._compile_kwargs: dict[str, Any] | None = None
        self._compiled = False
        # CLI path passes outputs=None and composes via instantiate_classes
        if outputs:
            self.compose_output_section({key: w for key, w in outputs.items() if w is not None})

    def compose_output_section(self, section: Mapping[str, SaltModelModule | SinkModule]) -> None:
        """Compose the top-level ``outputs:`` section onto the model.

        WRITERS are folded into the planning module dict (and `net`, params-free);
        their declaration order is the eval-H5 per-group column order. SINKS go
        to `_section_sinks` instead: excluded from the graph dict (folded into
        the per-mode plan at ``compile_mode``) and from the bound section
        manifest (a sink reads that manifest, so it must not contain itself).
        MANIFEST-ONLY writers (`InputCopyWriter`) have no ``produces`` so the
        demand closure would prune them — they stay in the section dict only,
        for the sink to read their copy spec at ``bind_output_section``.

        Raises `ConfigError` on an entry that is neither a `SaltModelModule`
        nor a `SinkModule`, or whose name collides with a model module.
        """
        section = {key: w for key, w in section.items() if w is not None}
        if not section:
            return
        for key, w in section.items():
            if not isinstance(w, (SaltModelModule, SinkModule)):
                raise ConfigError(
                    f"outputs: section writer {key!r} ({type(w).__name__}) is neither a "
                    "SaltModelModule nor a SinkModule — model-graph outputs: entries must "
                    "subclass SaltModelModule; wrap or extend it"
                )
            if key in self._graph_modules:
                raise ConfigError(
                    f"outputs: section writer {key!r} collides with a model module — instance "
                    "names are unique across the pipeline graph"
                )
            w.name = key
        # partition sinks out of BOTH graph_writers (else they collide with the
        # compile-time sink fold) and _output_section (else a sink binds a
        # manifest containing itself).
        sinks = {key: w for key, w in section.items() if _is_section_sink(w)}
        writers = {key: w for key, w in section.items() if key not in sinks}
        # model-side modules before the section folds in — RunTaskOutput
        # resolves its tasks against these.
        model_modules = dict(self._graph_modules)
        graph_writers = {
            key: w
            for key, w in writers.items()
            if not (callable(getattr(w, "is_manifest_only", None)) and w.is_manifest_only())
        }
        for key, w in graph_writers.items():
            self.net[key] = w  # ride the ModuleDict (params-free, state_dict unchanged)
            self._graph_modules[key] = w
        self._output_section = dict(writers)
        self._section_sinks = list(sinks.values())
        for w in writers.values():
            if callable(getattr(w, "bind_model_modules", None)):
                w.bind_model_modules(model_modules)

    # -- lifecycle state (read-only — tests assert on these) --------------------

    @property
    def bound(self) -> bool:
        """Whether the two-phase bind has run (exactly once)."""
        return self._bound

    @property
    def materialised(self) -> bool:
        """Whether this instance ran `materialise_all` (fresh fits only)."""
        return self._materialised

    @property
    def loaded_from_checkpoint(self) -> bool:
        """Whether any checkpoint was loaded into this instance."""
        return self._loaded_from_checkpoint

    # -- static declarations -----------------------------------------------------

    def sink_demand(self) -> dict[Mode, list[str]]:
        """The per-mode dataset-boundary demand, derived from module declarations.

        For each runtime mode, every non-optional required key that no
        sibling module produces must come from the dataset. ``meta.rows`` is
        added in TEST for writer row alignment. `SaltDataModule.setup`
        adopts this automatically when no explicit sinks were configured.

        Returns
        -------
        dict[Mode, list[str]]
            ``{mode: [dotted keys]}`` for FIT/VAL/TEST, in declaration order.
        """
        return {mode: demand for mode, (demand, _) in self._boundary_demand().items()}

    def sink_origins(self) -> dict[Mode, dict[str, str]]:
        """Per-mode demanded-key -> demanding-module description.

        The map is PER MODE: in TEST a label can be demanded by a *writer*
        while the same key is task-demanded in FIT — a merged map would
        misattribute TEST artifacts/errors to the inactive FIT demander.

        Returns
        -------
        dict[Mode, dict[str, str]]
            E.g. ``{Mode.FIT: {"labels.jets.flavour_label":
            "'jets_classification' (config:
            model.modules.jets_classification)"}}``.
        """
        return {mode: origins for mode, (_demand, origins) in self._boundary_demand().items()}

    def _boundary_demand(self) -> dict[Mode, tuple[list[str], dict[str, str]]]:
        """Per-mode boundary demand + per-key demander descriptions; in TEST, an
        attached `WriterCallback`'s requires extend the demand so writer-demanded
        dataset keys keep their producers alive. Raises `ConfigError` on wildcard
        demand keys or a demand key outside the dataset-served namespaces.
        """
        out: dict[Mode, tuple[list[str], dict[str, str]]] = {}
        # ONNX export feeds the model directly and has no dataset plan
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST):
            required: dict[str, list[str]] = {}
            produced: set[str] = set()
            for name, module in self._graph_modules.items():
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
                for key, who in (self._writer_demand() or {}).items():
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
                # metrics callback's dataset-namespace requires (labels/masks/meta)
                # that no task already demands extend the boundary so their
                # producers survive. Model-produced (preds.*) callback keys are
                # anchored by `_model_sinks`, not here (already `in produced`).
                for key, who in self._callback_demand(mode).items():
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

    def _attached_writer(self) -> tuple[Any, Any]:
        """The attached TEST writer/sink node + reader, duck-typed on
        ``writer_demand``. Read from `_sinks` — the trainer's registry plus any
        section-declared sink. Order-independent: an ONNX-only sink's
        ``is_test_sink()`` discriminator skips it so it is never chosen as the
        TEST persistence sink; a plain duck-typed sink without ``is_test_sink``
        counts as one. ``(None, None)`` when none is attached.
        """
        trainer = self._trainer
        callback = next(
            (
                sink
                for sink in _reachable_sinks(self)
                if callable(getattr(sink, "writer_demand", None))
                and _is_test_persistence_sink(sink)
            ),
            None,
        )
        if callback is None:
            return None, None
        reader = getattr(getattr(trainer, "datamodule", None), "reader", None)
        if reader is None:
            return None, None
        return callback, reader

    def _attached_sink_node(self) -> GraphModule | None:
        """The attached TEST sink NODE (a `GraphModule` with ``is_sink() -> True``,
        e.g. `H5OutputSink`), or None. `compile_mode` folds it into the planning
        module dict so it renders its own card and anchors demand via its
        declared requires.
        """
        callback, _reader = self._attached_writer()
        if callback is None:
            return None
        is_sink = getattr(callback, "is_sink", None)
        has_io = callable(getattr(callback, "declare_io", None))
        if has_io and callable(is_sink) and bool(is_sink()):
            return callback
        return None

    def _bind_output_section_to_sink(self) -> None:
        """Bind both manifest sources to every attached sink.

        Sinks resolve their column schema + copy spec + mask streams from the
        outputs: section AND from the model's graph modules (a producer names
        its own leaves), so both must be bound before any
        declare_io/writer_demand resolution. Duck-typed on the two bind
        methods, so a sink implementing neither is left alone.
        """
        for sink in _reachable_sinks(self):
            if callable(getattr(sink, "bind_model_modules", None)):
                sink.bind_model_modules(self._graph_modules)
            if self._output_section and callable(getattr(sink, "bind_output_section", None)):
                sink.bind_output_section(self._output_section)

    @staticmethod
    def _fold_sink_node(
        modules: dict[str, GraphModule], sink_node: GraphModule
    ) -> dict[str, GraphModule]:
        """Add a sink NODE to a planning module dict under its instance name (the
        planner requires ``module.name == config_key``). Returns a fresh dict;
        raises `ConfigError` on a name collision with an existing model module.
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

    def _assert_no_dead_preds(self, plan: Plan) -> None:
        """Hard-error on a produced ``preds.*`` key consumed by no plan edge in the
        compiled TEST plan. With a sink NODE folded (`compile_mode`), the sink
        demands ``outputs.*`` not ``preds.*``, so a genuinely-dead ``preds.*``
        would otherwise ship silently; raises `ConfigError` via
        `_dead_preds_message`.
        """
        produced: dict[str, str] = {}
        for name, module in self._graph_modules.items():
            for key, spec in flatten_spec(module.declare_io(Mode.TEST).produces).items():
                if key.split(KEY_SEP, 1)[0] == "preds" and spec.active_in(Mode.TEST):
                    produced.setdefault(key, name)
        # every key any plan edge carries is consumed by a surviving node — a
        # produced preds.* not here reaches the sink through NO producer (dead).
        consumed = {edge.key for edge in plan.edges}
        if dead := [key for key in produced if key not in consumed]:
            raise ConfigError(_dead_preds_message(dead, produced, writers=None))

    def _writer_demand(self) -> dict[str, str] | None:
        """Merged writer-declared TEST demand from the attached `WriterCallback`,
        or None when none is attached (programmatic ``Trainer.test`` without
        writers keeps the anchor-on-all-preds behaviour).
        """
        callback, reader = self._attached_writer()
        if callback is None:
            return None
        return callback.writer_demand(self._graph_modules, reader)

    def _attached_callbacks(self) -> list[Any]:
        """Attached callbacks declaring FIT/VAL plan sinks, duck-typed on a
        callable ``fit_val_demand`` (e.g. `ConfusionMatrix`). Static/config-only
        so ``salt graph`` sees the same sinks a real ``trainer.fit`` would.
        Empty list when none are attached.
        """
        trainer = self._trainer
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        return [cb for cb in callbacks or [] if callable(getattr(cb, "fit_val_demand", None))]

    def _callback_demand(self, mode: Mode, callbacks: Any = None) -> dict[str, str]:
        """Merged callback-declared FIT/VAL demand — the TRAINING-mode mirror of
        `_writer_demand` (TEST): metrics-family callbacks declare the bundle
        keys they read each VAL epoch, becoming FIT/VAL plan sinks so their
        producers survive demand pruning. Empty outside TRAINING modes.
        ``callbacks=None`` discovers from the attached trainer; explicit is the
        static `salt.cli` path.
        """
        if not (mode & Mode.TRAINING):
            return {}
        if callbacks is None:
            callbacks = self._attached_callbacks()
        out: dict[str, str] = {}
        for cb in callbacks:
            who = f"callback {type(cb).__name__!r}"
            for key in cb.fit_val_demand(self._graph_modules):
                out.setdefault(key, who)
        return out

    def _model_sinks(
        self, mode: Mode, writers: Any = None, reader: Any = None, callbacks: Any = None
    ) -> list[str]:
        """The model-plan sink anchors for one mode.

        FIT/VAL: ``loss.total`` plus callback-declared demand
        (`_callback_demand`) — callback demand only ADDS sinks, never
        narrows: an unconsumed ``preds.*`` in FIT/VAL is not a dead-preds
        error, unlike TEST (writer demand, where it raises `ConfigError`).
        ONNX with no export-sink demand falls through to every declared
        ``preds.*`` key. Explicit writers/reader/callbacks are the static
        `salt.cli` path; default None discovers from the trainer.
        """
        produced: dict[str, str] = {}
        for name, module in self._graph_modules.items():
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
            for key in self._callback_demand(mode, callbacks):
                if key not in sinks:
                    sinks.append(key)
            return sinks
        preds = [key for key in produced if key.split(KEY_SEP, 1)[0] == "preds"]
        if mode is Mode.TEST:
            if writers is None or reader is None:
                writers, reader = self._attached_writer()
            if writers is not None:
                writer_demand = writers.writer_demand(self._graph_modules, reader)
                if dead := [key for key in preds if key not in writer_demand]:
                    raise ConfigError(_dead_preds_message(dead, produced, writers))
                consumed = [key for key in produced if key in writer_demand]
                if not consumed:
                    raise ConfigError(
                        "[mode=TEST] the configured writers consume nothing the model "
                        "produces — check the outputs: section"
                    )
                return consumed
        # the ONNX output manifest is not writer-derived — the folded OnnxExportSink
        # anchors its conversion-leaf demand itself. When no export sink demand is
        # present it falls back to every preds.* key below.
        if not preds:
            raise ConfigError(
                f"no module produces a 'preds.*' key in mode {mode.name} — evaluation plans "
                "anchor on predictions"
            )
        return preds

    # -- compile + two-phase bind -------------------------------------------------

    def compile_mode(self, mode: Mode, boundary: Mapping[str, TensorSpec]) -> Plan:
        """Compile the model-side plan for one mode against a dataset boundary.

        Parameters
        ----------
        mode : Mode
            A primary mode.
        boundary : Mapping[str, TensorSpec]
            Flat ``{dotted key: spec}`` sources — the matching
            `SaltDataset.boundary_specs()`.

        Returns
        -------
        Plan
            The compiled plan (also stored on ``self.plans``); a stashed
            checkpoint plan hash for this mode is verified.

        Raises
        ------
        ConfigError
            If a recompile yields a different plan hash (the boundary or
            module config drifted mid-run), or the checkpoint's FIT hash
            mismatches.
        """
        # fold the discovered TEST sink NODE into the planning module dict so it
        # renders its own card and anchors demand via its declared requires
        # instead of the flat <sinks> sentinel. In FIT/VAL/ONNX the sink's
        # declare_io is empty, so the planner collects it as inactive and
        # plan_hash is unchanged (the trained checkpoint loads unperturbed).
        modules = dict(self._graph_modules)
        sink_node = self._attached_sink_node()
        # bind both manifest sources to the sink so it names the declared
        # outputs.* leaves in declaration order, not producer discovery.
        if sink_node is not None:
            if callable(getattr(sink_node, "bind_model_modules", None)):
                sink_node.bind_model_modules(self._graph_modules)
            if self._output_section and callable(getattr(sink_node, "bind_output_section", None)):
                sink_node.bind_output_section(self._output_section)
        folded_sink = sink_node is not None and mode is Mode.TEST
        if folded_sink:
            # the sink node anchors ALL its demand via its declared requires —
            # flat model sinks are empty. The dead-preds safety net is restored
            # AFTER compile via `_assert_no_dead_preds` below: a `preds.*` the
            # model computes but no producer feeds into a demanded output is
            # still a hard error at `salt test`.
            modules = self._fold_sink_node(modules, sink_node)
            sinks: Any = []
        else:
            sinks = self._model_sinks(mode)
            if sink_node is not None:
                modules = self._fold_sink_node(modules, sink_node)
        plan = compile_plan(
            modules,
            mode,
            sources=unflatten_spec(dict(boundary)),
            sinks=sinks,
        )
        if folded_sink:
            self._assert_no_dead_preds(plan)
        previous = self.plans.get(mode)
        if previous is not None and previous.plan_hash != plan.plan_hash:
            raise ConfigError(
                f"recompiled {mode.name} plan hash changed within one run "
                f"({previous.plan_hash[:16]}… -> {plan.plan_hash[:16]}…) — the dataset "
                "boundary or module config drifted"
            )
        self.plans[mode] = plan
        self._executors[mode] = Executor(plan)
        self._verify_ckpt_hash(mode, plan)
        return plan

    def setup(self, stage: str) -> None:
        """Compile the stage's plans and run the two-phase bind, after the
        datamodule's own ``setup``; binds at most once, before any checkpoint
        state-dict load.
        """
        if stage not in {"fit", "test"}:
            raise ConfigError(
                f"stage {stage!r} is not supported by SaltModule — use trainer.fit or "
                "trainer.test (validate/predict entry points are not supported)"
            )
        # bind the outputs: section to the attached sink BEFORE any boundary/demand
        # resolution — the sink's declare_io/writer_demand resolves its columns
        # from the section manifest, so it must be bound first.
        self._bind_output_section_to_sink()
        dm = self._graph_datamodule()
        if stage == "fit":
            self.compile_mode(Mode.FIT, self._boundary(dm.train_dset, "train"))
            self.compile_mode(Mode.VAL, self._boundary(dm.val_dset, "val"))
            self._assert_fit_val_identical()
            check_class_names(self._graph_modules, dm.train_dset.reader)
            resolve_origin_weighting(self._graph_modules, dm.train_dset.reader)
        else:
            self.compile_mode(Mode.TEST, self._boundary(dm.test_dset, "test"))
            check_class_names(self._graph_modules, dm.test_dset.reader)
            resolve_origin_weighting(self._graph_modules, dm.test_dset.reader)
            self._validate_writer_specs(dm.test_dset)
        self._ensure_bound()
        # --init_from: warm start weights AFTER bind (params now carry their
        # resolved shapes) and BEFORE optimizer construction / the first step
        # (both are strictly later in the Lightning fit sequence). Fit-only —
        # `test` never warm-starts.
        if stage == "fit" and self._init_from is not None:
            self._warm_start_from_checkpoint(self._init_from)
        # training_schedule: reset to stage 0 and apply its freeze mask AFTER any
        # warm start (freeze composes on top of the loaded weights) and BEFORE
        # optimizer construction, so the initial `configure_optimizers` (built by
        # Lightning's strategy.setup, after this) sees stage 0's requires_grad
        # mask. Later stage boundaries are driven by the
        # `TrainingScheduleCallback`. Fit-only.
        if stage == "fit":
            self._apply_training_schedule()
        # --compile LAST: dynamo traces each graph module in its final setup-time
        # state — after bind/materialise have resolved widths, after any
        # --init_from warm start has written weights, and after stage 0's freeze
        # mask is in place.
        self._apply_compile()

    def _apply_training_schedule(self) -> None:
        """Validate the schedule against the attached trainer and apply stage 0's
        freeze mask at fit setup (before the initial optimizer build).
        When any stage declares `early_stop`, also runs the early-stop preflight
        and seeds stage 0's live counters.

        A `ConfigError` propagates from the validators on an over-allocated epoch
        budget, a multi-stage schedule with no finite `trainer.max_epochs` (see
        `TrainingSchedule.validate_epochs`), or an `early_stop` stage under a
        trainer with validation disabled.
        """
        max_epochs = getattr(self._trainer, "max_epochs", None)
        self._schedule.validate_epochs(max_epochs)
        self._preflight_early_stop()
        self._preflight_lr_scheduler()
        # Decide the freeze mode BEFORE applying the stage-0 mask: under a DDP
        # strategy with a freeze set that changes across stages, keep every managed
        # param requires_grad=True at wrap so the reducer manages them across flips
        # (fixes the init-frozen-unfreeze desync). Off DDP / static freeze,
        # stays False → the requires_grad-based freeze (bitwise-parity path).
        self._reducer_safe_freeze = reducer_safe_freeze_required(
            getattr(self._trainer, "strategy", None), self._schedule
        )
        self._current_stage_index = 0
        self._stage_start_epoch = 0
        self._boundary_records = []
        self._pending_early_advance = False
        self._early_stop_tracker = self._make_early_stop_tracker(self._schedule.stages[0])
        self._apply_stage_freeze(self._schedule.stages[0])

    def _preflight_early_stop(self) -> None:
        """Fail fast at fit setup if the schedule declares `early_stop` but the
        trainer cannot ever evaluate it (validation disabled) — an early-stop that
        can never fire would silently wait for the epoch cap. Inert unless the
        schedule declares an `early_stop` (the `has_early_stop` master switch).

        Raises
        ------
        ConfigError
            A stage declares `early_stop` but ``trainer.limit_val_batches == 0``.
        """
        if not self._schedule.has_early_stop:
            return
        if getattr(self._trainer, "limit_val_batches", 1) == 0:
            raise ConfigError(
                "training_schedule declares an 'early_stop' stage but the trainer has validation "
                "disabled (limit_val_batches=0) — early stopping is evaluated on validation-epoch "
                "end and could never fire. Enable validation or remove early_stop."
            )

    def _preflight_lr_scheduler(self) -> None:
        """Fail fast at fit setup for every stage's `lr_scheduler`:
        import its `class_path` (unimportable → ConfigError) and enforce the
        metric-driven-⇒-`monitor` rule (a `ReduceLROnPlateau`-family scheduler needs
        a monitored metric). Inert unless the schedule declares an `lr_scheduler`.
        """
        if not self._schedule.has_lr_scheduler:
            return
        for stage in self._schedule.stages:
            cfg = stage.lr_scheduler
            if cfg is None:
                continue
            cls = _resolve_lr_scheduler_class(cfg.class_path)  # fail-fast on bad path
            if _is_metric_driven_scheduler(cls) and not cfg.monitor:
                raise ConfigError(
                    f"training_schedule stage {stage.name!r} lr_scheduler {cfg.class_path} is "
                    "metric-driven (a ReduceLROnPlateau subclass) and requires a 'monitor' "
                    "(a trainer.callback_metrics key, e.g. 'val/loss')."
                )

    def _make_early_stop_tracker(self, stage: StageConfig) -> EarlyStopTracker | None:
        """A fresh `EarlyStopTracker` for `stage` (reset counters), or ``None`` when
        the stage declares no `early_stop`.
        """
        return EarlyStopTracker(stage.early_stop) if stage.early_stop is not None else None

    def _apply_stage_freeze(self, stage: StageConfig) -> None:
        """Apply `stage`'s freeze mask as a DELTA against the currently-frozen
        set: modules entering the frozen set get ``eval()`` (stops
        dropout + running-stat updates during training); modules leaving it are
        restored to ``train()``. Modules outside both sets are left untouched, so
        the desugared no-freeze path never mutates a param (bitwise parity).

        The requires_grad handling depends on the freeze mode (see
        `apply_stage_freeze`): the default mode toggles ``requires_grad`` so the
        optimizer excludes frozen params by it; reducer-safe mode (under DDP
        with a freeze-flipping schedule) leaves ``requires_grad=True`` on every
        managed param so the reducer keeps managing it across the flip, excluding
        frozen params from the optimizer by membership instead. Records the frozen
        set so `train` re-asserts eval after Lightning's per-epoch ``model.train()``
        and `configure_optimizers`/`on_after_backward` can key off it.
        """
        frozen = self._schedule.frozen_names(stage)
        apply_stage_freeze(
            self.net,
            frozen,
            self._frozen_module_names,
            reducer_safe=self._reducer_safe_freeze,
        )
        self._frozen_module_names = frozen

    # -- stage transitions (driven by TrainingScheduleCallback) -------------------

    def advance_to_stage(self, new_index: int, global_step: int, epoch: int, reason: str) -> None:
        """Enter stage `new_index`: set the active index + apply its freeze mask
        (the delta off the previous stage). Under the `early_stop` master switch,
        also records the boundary, resets the stage's live counters, and marks the
        new stage's start epoch (so the per-stage epoch-cap count is measured from
        here). The optimizer/LR rebuild is done by the caller
        (`TrainingScheduleCallback`) via ``strategy.setup_optimizers`` right after.
        `reason` is ``"epochs"`` or ``"early_stop"`` (recorded only under the
        `early_stop` switch).
        """
        self._current_stage_index = new_index
        stage = self._schedule.stages[new_index]
        self._apply_stage_freeze(stage)
        if self._schedule.has_early_stop:
            self._stage_start_epoch = epoch
            self._pending_early_advance = False
            self._boundary_records.append(
                boundary_record(stage.name, new_index, global_step, epoch, reason)
            )
            self._early_stop_tracker = self._make_early_stop_tracker(stage)

    def next_stage_index_early_stop(self, current_epoch: int) -> int:
        """The stage index to run at `current_epoch` under the `early_stop` switch:
        advance by exactly one when the active stage's early-stop fired (a pending
        advance) OR it reached its epoch cap (``current_epoch - stage_start >=
        epochs``); the final stage never advances by cap (its early-stop ends the
        fit instead). Data-dependent boundaries make this replace the pure
        epoch-arithmetic `stage_index_for_epoch` whenever any stage can early-stop.
        """
        idx = self._current_stage_index
        if idx >= len(self._schedule.stages) - 1:
            return idx
        stage = self._schedule.stages[idx]
        cap_reached = stage.epochs is not None and (current_epoch - self._stage_start_epoch) >= (
            stage.epochs
        )
        return idx + 1 if (self._pending_early_advance or cap_reached) else idx

    def evaluate_early_stop(self, monitored: float | None) -> bool:
        """Fold the active stage's monitored validation value into its early-stop
        tracker and return the LOCAL early-stop decision (this rank). Called by the
        callback at validation-epoch end with the RANK-REDUCED monitor value (or
        ``None`` when the metric is absent — a fail-fast misconfiguration). The
        caller rank-syncs the returned boolean before acting on it; this method only
        advances the stage-local counters. Returns ``False`` (no stop) when the
        active stage declares no `early_stop`; raises `ConfigError` when the
        monitored metric is absent from ``trainer.callback_metrics``.
        """
        stage = self._schedule.stages[self._current_stage_index]
        if stage.early_stop is None:
            return False
        if monitored is None:
            raise ConfigError(
                f"training_schedule stage {stage.name!r} early_stop monitors "
                f"{stage.early_stop.monitor!r} but it is absent from trainer.callback_metrics — "
                "check the metric name (e.g. 'val/loss') or that validation logs it."
            )
        assert self._early_stop_tracker is not None
        return self._early_stop_tracker.check(monitored)

    def mark_pending_early_advance(self) -> None:
        """Flag that the active (non-final) stage's early-stop has fired — consumed
        at the next train-epoch-start transition (`next_stage_index_early_stop`).
        """
        self._pending_early_advance = True

    @property
    def pending_early_advance(self) -> bool:
        """Whether an early-stop trigger is awaiting the next stage transition."""
        return self._pending_early_advance

    # -- torch.compile -------------------------------------------------------------

    def enable_compile(self, **compile_kwargs: Any) -> None:
        """Request `torch.compile` of each graph module; applied at the end of `setup`.

        The compile unit is each graph module (the forward is a `Plan` run by
        `Executor` — there is no single inner nn.Module). Deferred to the end of
        `setup` so dynamo traces each module after bind/materialise have
        resolved widths. `compile_kwargs` are forwarded to `torch.compile`.
        """
        self._compile_kwargs = dict(compile_kwargs)

    def _apply_compile(self) -> None:
        """Compile each graph module's forward, in place.

        Uses ``nn.Module.compile()`` (not the ``OptimizedModule``-returning
        ``torch.compile``) because `Plan` steps and `Executor` hold the module
        INSTANCES — a wrapper would fail the executor's `GraphModule` protocol
        check. In-place compilation leaves ``state_dict`` keys untouched, so a
        checkpoint written under ``--compile`` loads uncompiled with no
        ``_orig_mod.`` rewriting.
        """
        if self._compile_kwargs is None or self._compiled:
            return
        for module in self._graph_modules.values():
            if isinstance(module, nn.Module):
                module.compile(**self._compile_kwargs)
        self._compiled = True

    def _validate_writer_specs(self, test_dset: SaltDataset) -> None:
        """Static writer-input validation on the TEST path.

        Hands the attached `WriterCallback` (if any) the union of the model
        modules' TEST-active produced ports and the test dataset's served
        boundary leaves, so it can prove each writer-declared require exists
        and kind/dtype-unifies against its producer before the first batch.
        Model-produced ports take precedence over a same-named boundary key
        (they are the executed leaf). No writer callback → no-op.
        """
        writers, reader = self._attached_writer()
        if writers is None or not callable(getattr(writers, "validate_specs", None)):
            return
        producer_specs = dict(test_dset.boundary_specs())
        producer_specs.update(writers.model_producer_specs(self._graph_modules))
        writers.validate_specs(self._graph_modules, reader, producer_specs)

    def _run_preflights(self, modules: Mapping[str, SaltModelModule] | None = None) -> None:
        """Fail-fast data-free checks of file-backed `materialise` sources: every
        module exposing a callable ``preflight()`` (e.g. `Normaliser`) is
        checked before any `materialise` writes buffers, so a bad path/content
        raises one `ConfigError` instead of a per-module mid-materialise crash.
        Called from ``on_fit_start`` on fresh fits only. On an ``--init_from``
        warm start only the to-be-materialised (checkpoint-uncovered) modules
        are passed — a retained module's file need not exist on this machine.
        """
        for module in (self._graph_modules if modules is None else modules).values():
            preflight = getattr(module, "preflight", None)
            if callable(preflight):
                preflight()

    def _assert_fit_val_identical(self) -> None:
        """Assert the VAL plan is structurally identical to the FIT plan (a cheap
        hash comparison — the hash is mode-independent). VAL-divergent module
        ports are not yet supported; raises `ConfigError` on mismatch.
        """
        fit_hash = self.plans[Mode.FIT].plan_hash
        val_hash = self.plans[Mode.VAL].plan_hash
        if fit_hash != val_hash:
            raise ConfigError(
                f"the VAL plan ({val_hash[:16]}…) differs structurally from the FIT plan "
                f"({fit_hash[:16]}…) — VAL is contractually the identical plan "
                "(VAL-divergent module ports would need an exemption mechanism)"
            )

    def _graph_datamodule(self) -> SaltDataModule:
        """The attached `SaltDataModule` (model plans need its boundary); raises
        `ConfigError` when none is attached or it is not a `SaltDataModule`.
        """
        dm = getattr(self._trainer, "datamodule", None) if self._trainer is not None else None
        if not isinstance(dm, SaltDataModule):
            raise ConfigError(
                "SaltModule compiles its plans against a SaltDataModule's dataset boundary — "
                f"pass one to trainer.fit/test (got {type(dm).__name__}; raw dataloaders are "
                "not supported)"
            )
        return dm

    @staticmethod
    def _boundary(dset: SaltDataset | None, stage: str) -> dict[str, TensorSpec]:
        """A stage dataset's model-visible boundary specs; raises `ConfigError`
        if the stage dataset was never built.
        """
        if dset is None:
            raise ConfigError(
                f"the datamodule has no {stage} dataset — its setup did not run or the "
                f"{stage}_file is unset"
            )
        return dset.boundary_specs()

    def _ensure_bound(self) -> None:
        """Resolve the bind schema from the compiled plans and bind once; raises
        `ConfigError` if no plan was compiled yet.
        """
        if self._bound:
            return
        if not self.plans:
            raise ConfigError("bind before any compiled plan — call setup/compile_mode first")
        self._bind(resolve_bind_schema(self.plans.values()))

    def _bind(self, schema: ResolvedSchema) -> None:
        """Bind all modules to a resolved schema, exactly once; raises
        `ConfigError` on a second bind (would silently discard loaded values).
        """
        if self._bound:
            raise ConfigError(
                "SaltModule modules are already bound — bind happens exactly once, before any "
                "state-dict load"
            )
        bind_all(self._graph_modules, schema)
        self._apply_mup_shapes()
        self.schema = schema
        self._bound = True

    def _apply_mup_shapes(self) -> None:
        """Apply the muP base shapes from ``mup.shape_path`` over the whole
        ``net`` (after `bind_all`) so `MuReadout.width_mult()`/`MuAdamW` resolve
        against real base widths; ``rescale_params=False`` since params already
        carry their muP init. No-op with no muP / no ``shape_path``; raises
        `ConfigError` if the shape file is missing.
        """
        if self.mup_cfg is None or not self.mup_cfg.get("shape_path"):
            return
        from pathlib import Path as _Path

        from mup import set_base_shapes

        shape_path = _Path(self.mup_cfg["shape_path"])
        if not shape_path.is_file():
            raise ConfigError(
                f"model.init_args.mup.shape_path {str(shape_path)!r} does not exist — generate it "
                "with `salt mup-shapes` (setup_mup) before fit"
            )
        # the file carries infshapes for the whole net (generated from net) — apply
        # it ONCE over self.net so the apply_to MuReadout/linears get their
        # base/delta infshapes; non-apply_to params have fixed dims in the file
        set_base_shapes(self.net, str(shape_path), rescale_params=False)

    # -- materialise (fresh fits only) --------------------------------------------

    def on_fit_start(self) -> None:
        """Materialise file-backed values before the first step of a FRESH fit.

        Runs after checkpoint restore: when a full checkpoint was loaded
        (resume or `load_from_checkpoint`) materialise is skipped entirely and
        the values come from the state_dict — `Normaliser.forward` raises if a
        checkpoint lacked them. On an ``--init_from`` warm start the load is
        PARTIAL, so materialise is SELECTIVE: exactly the config modules that
        received no checkpoint weights are materialised (a newly-added
        `Normaliser` reads its norm_dict; retained modules keep loaded stats).
        """
        if self._materialised or self._loaded_from_checkpoint:
            return
        targets = self._graph_modules
        if self._init_warm_started:
            targets = {
                name: module
                for name, module in self._graph_modules.items()
                if name not in self._init_loaded_modules
            }
        self._run_preflights(targets)
        materialise_all(targets)
        self._materialised = True

    def train(self, mode: bool = True) -> SaltModule:
        """Set training mode, then re-assert ``eval()`` on the schedule's frozen
        modules. Lightning calls ``model.train()`` at every train-epoch start,
        which would otherwise un-eval a frozen module (re-enabling dropout /
        running-stat updates); this override keeps frozen modules in eval across
        epoch boundaries. No-op when nothing is frozen.
        """
        super().train(mode)
        if mode and self._frozen_module_names:
            for name in self._frozen_module_names:
                self.net[name].eval()
        return self

    def on_after_backward(self) -> None:
        """Reducer-safe mode only: clear the frozen modules' gradients each step,
        after the DDP reducer has finished all-reducing them. Frozen params keep
        ``requires_grad=True`` (to stay in the reducer across freeze flips), so
        they accumulate a gradient every backward that is never stepped; clearing
        it here stops a frozen stage's gradient from surviving to the first
        optimizer step after the module is unfrozen. No-op off reducer-safe mode
        (frozen params have ``requires_grad=False`` there and never get a grad).
        """
        if self._reducer_safe_freeze and self._frozen_module_names:
            clear_frozen_grads(self.net, self._frozen_module_names)

    # -- steps ---------------------------------------------------------------------

    def forward(self, batch: Mapping[str, Any] | Bundle, mode: Mode = Mode.TEST) -> Bundle:
        """Run the compiled plan for `mode` over a batch: adopts a dict batch into
        a `Bundle` (or reuses one), and returns the bundle with the mode's
        produces merged in. Raises `ConfigError` if `mode`'s plan wasn't compiled.
        """
        executor = self._executors.get(mode)
        if executor is None:
            raise ConfigError(
                f"no compiled plan for mode {mode.name} — run under trainer.fit/test or call "
                "compile_mode() first"
            )
        bundle = batch if isinstance(batch, Bundle) else Bundle(dict(batch))
        return executor.run(bundle, debug=self.debug)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """One FIT step: run the plan, log losses, return ``{"loss": total,
        "bundle": Bundle}``; raises `RuntimeError` on a NaN total loss.
        """
        del batch_idx
        bundle = self(batch, Mode.FIT)
        total = bundle.get("loss.total")
        if total.isnan():
            raise RuntimeError(
                "Loss is NaN - check dataset for NaNs or infs. "
                "See 'docs/training.md - NaNs' for more info."
            )
        self._log_losses(bundle, stage="train")
        return {"loss": total, "bundle": bundle}

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """One VAL step — mirrors `training_step` with the VAL plan."""
        del batch_idx
        bundle = self(batch, Mode.VAL)
        self._log_losses(bundle, stage="val")
        return {"loss": bundle.get("loss.total"), "bundle": bundle}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Bundle:
        """One TEST step: runs the TEST plan, returning the executed bundle
        (``preds.*`` + ``meta.rows`` for writers).
        """
        del batch_idx
        return self(batch, Mode.TEST)

    def _log_losses(self, bundle: Bundle, stage: str) -> None:
        """Log ``{stage}/loss`` and ``{stage}/{task}_loss``.

        ``prog_bar=True``: without it a fit with the default
        ``logger: false`` shows no loss feedback at all — an added aux task
        would train invisibly.
        """
        sync_dist = self._trainer is not None and len(self.trainer.device_ids) > 1
        self.log(f"{stage}/loss", bundle.get("loss.total"), sync_dist=sync_dist, prog_bar=True)
        for task, value in bundle.subtree("losses").items():
            self.log(f"{stage}/{task}_loss", value, sync_dist=sync_dist, prog_bar=True)

    # -- optimizer -----------------------------------------------------------------

    def _get_optimizer_class(self, optimizer: str | None = None) -> type[Optimizer]:
        """Resolve the optimizer class, with the muP ``MuAdamW`` swap: when `mup`
        is configured the optimizer is always ``mup.optim.MuAdamW`` (coupled to
        the base shapes set at bind via `_apply_mup_shapes`). `optimizer` defaults
        to ``self.optimizer`` (a stage may override it). Raises `ImportError` for
        lion-pytorch without lion-pytorch installed, `ConfigError` for an
        unsupported name.
        """
        optimizer = optimizer or self.optimizer
        if self.mup_cfg is not None:
            from mup.optim import MuAdamW

            return MuAdamW
        if optimizer == "lion":
            return Lion
        if optimizer == "lion-pytorch":
            if not _lion_available:
                raise ImportError(
                    "optimizer: lion-pytorch requested but lion-pytorch is not installed. "
                    "Use optimizer: lion for salt's own (bitwise-identical, foreach) Lion."
                )
            return ReferenceLion
        if optimizer == "AdamW":
            return AdamW
        if optimizer == "HybridMuonAdamW":
            return HybridMuonAdamW
        raise ConfigError(f"Optimizer '{optimizer}' is not supported.")

    def _active_optim_config(self) -> tuple[Mapping[str, float], str]:
        """The ``(lrs, optimizer_name)`` for the currently-active stage. The
        desugared single `fit` stage overrides neither, so this returns the
        top-level pair unchanged (the bitwise-parity path). A stage's `lrs`
        deep-overrides the top-level keys; its `optimizer` replaces the name.
        """
        stage = self._schedule.stages[self._current_stage_index]
        lrs = {**self.lrs, **stage.lrs} if stage.lrs is not None else self.lrs
        return lrs, stage.optimizer or self.optimizer

    def _stage_total_steps(self) -> int:
        """The `OneCycleLR.total_steps` for the active stage. A single-stage
        schedule uses the whole-run `estimated_stepping_batches` exactly (parity);
        a multi-stage schedule uses this stage's proportional per-stage allocation
        of that estimate (never the whole-run figure for a sub-stage).
        Under the `early_stop` switch the envelope is sized from the stage's epoch
        cap measured from its ACTUAL start epoch (see `_early_stop_stage_total_steps`)
        so a stage that starts early — because an earlier stage early-stopped — never
        over-steps its OneCycle.
        """
        total = self.trainer.estimated_stepping_batches
        if not self._schedule.is_multi_stage:
            return total
        if self._schedule.has_early_stop:
            return self._early_stop_stage_total_steps(total)
        allocations = self._schedule.stage_step_allocations(total, self.trainer.max_epochs)
        return allocations[self._current_stage_index]

    def _early_stop_stage_total_steps(self, total: int) -> int:
        """OneCycle `total_steps` for the active stage under the `early_stop` switch:
        a per-epoch step estimate (``total / max_epochs``) times the stage's epoch
        budget — its `epochs` cap for a non-final stage, or ``max_epochs -
        stage_start_epoch`` for the final stage (whose real length depends on when
        earlier stages ended). Sized ``>=`` the steps a stage can actually take, so
        OneCycle never over-steps; equals the arithmetic split when no stage stops
        early. Truncated early, the stage simply under-runs its envelope (D-ES).
        """
        max_epochs = self.trainer.max_epochs
        steps_per_epoch = max(1, round(total / max_epochs))
        index = self._current_stage_index
        is_final = index == len(self._schedule.stages) - 1
        stage = self._schedule.stages[index]
        if is_final:
            budget_epochs = max_epochs - self._stage_start_epoch
        else:
            assert stage.epochs is not None  # non-final stages require epochs
            budget_epochs = stage.epochs
        return max(1, steps_per_epoch * budget_epochs)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        """Build the active stage's optimizer (over TRAINABLE params only — frozen
        modules are excluded entirely) + its LR scheduler. Re-invoked by the
        `TrainingScheduleCallback` at each stage boundary via
        ``trainer.strategy.setup_optimizers``. The scheduler is the default
        step-interval OneCycleLR over the stage's step allocation, unless the stage
        declares an `lr_scheduler` — then that class is instantiated over
        the freshly-built optimizer instead.
        """
        lrs, optimizer_name = self._active_optim_config()
        optimizer_class = self._get_optimizer_class(optimizer_name)
        optimizer_kwargs = {
            "lr": lrs["initial"],
            "weight_decay": lrs.get("weight_decay", 1e-5),
        }
        # Optimizer owns only the active stage's TRAINABLE params: those with
        # requires_grad NOT under a frozen module's prefix. Keying off the frozen
        # SET (not just requires_grad) also excludes a reducer-safe frozen module,
        # whose params keep requires_grad=True — so weight decay never touches a
        # frozen param either. In the default freeze mode this selects the exact
        # same set as the old requires_grad filter (bitwise parity).
        named_trainable = trainable_named_params(self.named_parameters(), self._frozen_module_names)
        if optimizer_class is HybridMuonAdamW:
            params: Any = named_trainable
        else:
            params = [p for _, p in named_trainable]
        opt = optimizer_class(params, **optimizer_kwargs)
        stage = self._schedule.stages[self._current_stage_index]
        if stage.lr_scheduler is not None:
            return [opt], [self._build_stage_lr_scheduler(opt, stage.lr_scheduler)]
        # `safe_pct_start` is clamped against THIS STAGE's step allocation, not the
        # whole-run estimate: a sub-stage of a multi-stage schedule gets a fraction
        # of the run's steps, so it reaches the degenerate-warm-up boundary on runs
        # far longer than a single-stage fit would need to.
        total_steps = int(self._stage_total_steps())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lrs["max"],
            total_steps=total_steps,
            div_factor=lrs["max"] / lrs["initial"],
            final_div_factor=lrs["initial"] / lrs["end"],
            pct_start=safe_pct_start(float(lrs["pct_start"]), total_steps),
            last_epoch=int(lrs.get("last_epoch", -1)),
            cycle_momentum=optimizer_class is not HybridMuonAdamW,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    def _build_stage_lr_scheduler(self, opt: Optimizer, cfg: LRSchedulerConfig) -> dict[str, Any]:
        """Instantiate the stage's chosen LR-scheduler class over the freshly-rebuilt
        `opt` and wrap it in the Lightning scheduler-config dict. The
        optimizer is injected as the first positional argument; a user-supplied
        `init_args.optimizer` was already rejected at parse. A metric-driven scheduler
        (`ReduceLROnPlateau`) is wired through Lightning's monitor mechanics
        (`reduce_on_plateau`/`monitor`); its rank-consistency rides on the synced
        monitor.
        """
        cls = _resolve_lr_scheduler_class(cfg.class_path)
        scheduler = cls(opt, **(dict(cfg.init_args) if cfg.init_args else {}))
        entry: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": cfg.interval,
            "frequency": cfg.frequency,
        }
        if _is_metric_driven_scheduler(cls):
            entry["reduce_on_plateau"] = True
            entry["monitor"] = cfg.monitor
        elif cfg.monitor is not None:
            entry["monitor"] = cfg.monitor
        return entry

    # -- checkpoints: schema + plan hashes, never Plan objects --------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Serialise the resolved schema + per-mode plan hashes + the active
        training-schedule stage under `CKPT_KEY` (Plans aren't picklable; the
        hash is the integrity check). ``schedule.stage_index`` is the stage
        whose optimizer state this checkpoint carries; on resume it is restored
        BEFORE the optimizer rebuild so the state_dict shapes match. Epoch-
        boundary granular: a mid-epoch checkpoint restores the same stage's
        optimizer, with any pending transition re-evaluated at next epoch start.
        """
        if self.schema is None:
            warnings.warn(
                "saving a SaltModule checkpoint before bind — no schema/plan hashes recorded",
                stacklevel=2,
            )
            return
        checkpoint[CKPT_KEY] = {
            "schema": {
                "widths": dict(self.schema.widths),
                "fields": {key: list(val) for key, val in self.schema.fields.items()},
            },
            "plan_hashes": {mode.name: plan.plan_hash for mode, plan in self.plans.items()},
            "schedule": self._schedule_checkpoint_state(),
        }

    def _schedule_checkpoint_state(self) -> dict[str, Any]:
        """The `schedule` sub-payload for the checkpoint. Legacy/no-early-stop
        configs get exactly ``{stage_index, stage_name}`` (byte-identical to the
        legacy payload — a parity guard). Under the `early_stop` master switch it
        additionally carries the active stage's start epoch, the completed-boundary
        records, and the live early-stop counters, so a data-dependent resume
        reconstructs the exact stage position + patience state.
        """
        state: dict[str, Any] = {
            "stage_index": self._current_stage_index,
            "stage_name": self._schedule.stages[self._current_stage_index].name,
        }
        if self._schedule.has_early_stop:
            state["stage_start_epoch"] = self._stage_start_epoch
            state["boundaries"] = list(self._boundary_records)
            if self._early_stop_tracker is not None:
                state["early_stop_state"] = self._early_stop_tracker.state_dict()
        return state

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Runs before the state-dict load on both restore paths: verifies plan
        hashes (fatal on FIT mismatch via `_verify_ckpt_hash`), binds from the
        checkpoint's stored schema if not yet bound, strips a ``--compile``-added
        ``_orig_mod.`` state_dict prefix, rejects the v1 (``ModelWrapper``)
        state-dict layout with `ConfigError`, and (on a fit resume of a multi-stage
        schedule) restores the saved stage index + freeze mask BEFORE the optimizer
        is rebuilt (see `_restore_schedule_stage`). Marks the instance
        checkpoint-loaded (disables `materialise`).
        """
        state_dict = checkpoint.get("state_dict")
        cleaned = self._reject_v1_and_strip_orig_mod(state_dict)
        if cleaned is not state_dict:
            checkpoint["state_dict"] = cleaned

        self._loaded_from_checkpoint = True
        payload = checkpoint.get(CKPT_KEY)
        if payload is None:
            warnings.warn(
                "checkpoint has no 'salt_core' payload (not written by SaltModule?) — "
                "schema/plan-hash verification skipped",
                stacklevel=2,
            )
            return
        self._ckpt_plan_hashes = dict(payload.get("plan_hashes", {}))
        for mode, plan in self.plans.items():
            self._verify_ckpt_hash(mode, plan)
        if not self._bound:
            stored = payload["schema"]
            self._bind(
                ResolvedSchema(
                    widths=dict(stored["widths"]),
                    fields={key: tuple(val) for key, val in stored.get("fields", {}).items()},
                )
            )
        # Re-establish the saved stage + freeze mask while the model is bound but
        # the optimizer is NOT yet built (Lightning restore order: setup("fit") ->
        # on_load_checkpoint -> configure_optimizers -> restore_optimizers), so
        # configure_optimizers builds a stage-k optimizer whose state_dict shape
        # matches the saved stage-k state.
        self._restore_schedule_stage(payload.get("schedule"))

    def _restore_schedule_stage(self, schedule_state: Mapping[str, Any] | None) -> None:
        """On a fit resume, restore the saved stage index + freeze mask
        (multi-stage only; the single-stage path stays bitwise-untouched), then
        the early-stop counters (any-stage, when declared). No-op unless the
        trainer is fitting — `salt test --ckpt_path` never mutates this state.
        A saved stage out of range for, or named differently than, the current
        schedule raises `ConfigError` (schedule changed; resume is undefined).
        """
        if schedule_state is None:
            return
        from lightning.pytorch.trainer.states import TrainerFn

        fn = getattr(getattr(self._trainer, "state", None), "fn", None)
        if fn is not None and fn != TrainerFn.FITTING:
            return
        if self._schedule.is_multi_stage:
            index = schedule_state["stage_index"]
            if not 0 <= index < len(self._schedule.stages):
                raise ConfigError(
                    f"checkpoint records training_schedule stage index {index}, out of range for "
                    f"the current {len(self._schedule.stages)}-stage schedule — the schedule "
                    "changed since the checkpoint was written; resume is not defined."
                )
            saved_name = schedule_state.get("stage_name")
            current_name = self._schedule.stages[index].name
            if saved_name is not None and saved_name != current_name:
                raise ConfigError(
                    f"checkpoint records training_schedule stage {index} as {saved_name!r} but the "
                    f"current schedule names it {current_name!r} — the schedule changed since the "
                    "checkpoint was written; resume is not defined."
                )
            self._current_stage_index = index
            self._apply_stage_freeze(self._schedule.stages[index])
        else:
            index = 0
        if self._schedule.has_early_stop:
            self._restore_early_stop_state(index, schedule_state)

    def _restore_early_stop_state(self, index: int, schedule_state: Mapping[str, Any]) -> None:
        """Restore early-stop resume state (stage start epoch, boundary records,
        live counters) so a mid-stage resume continues patience exactly. No
        `early_stop_state` in the checkpoint resets the counters fresh; a
        criterion-fingerprint mismatch with the current config raises
        `ConfigError` (resume undefined).
        """
        self._stage_start_epoch = schedule_state.get("stage_start_epoch", 0)
        self._boundary_records = list(schedule_state.get("boundaries", []))
        self._pending_early_advance = False
        stage = self._schedule.stages[index]
        es_state = schedule_state.get("early_stop_state")
        if es_state is None or stage.early_stop is None:
            self._early_stop_tracker = self._make_early_stop_tracker(stage)
            return
        saved_fp = es_state.get("fingerprint")
        current_fp = stage.early_stop.fingerprint()
        if saved_fp is not None and saved_fp != current_fp:
            raise ConfigError(
                f"checkpoint records an early_stop criterion {saved_fp} for stage "
                f"{stage.name!r} but the current config declares {current_fp} — the criterion "
                "changed since the checkpoint; patience resume is not defined."
            )
        self._early_stop_tracker = EarlyStopTracker.from_state_dict(stage.early_stop, es_state)

    def _verify_ckpt_hash(self, mode: Mode, plan: Plan) -> None:
        """Compare a compiled plan's hash with the checkpoint's stored hash;
        raises `ConfigError` on a FIT mismatch, other modes warn and continue.
        """
        stored = self._ckpt_plan_hashes.get(mode.name)
        if stored is None or stored == plan.plan_hash:
            return
        msg = (
            f"plan-hash mismatch for mode {mode.name}: checkpoint has {stored[:16]}…, the "
            f"current config compiles {plan.plan_hash[:16]}… — the module graph or dataset "
            "boundary changed since the checkpoint was written"
        )
        if mode is Mode.FIT:
            raise ConfigError(msg)
        warnings.warn(f"{msg} — continuing (non-FIT modes warn only)", stacklevel=2)

    @staticmethod
    def _reject_v1_and_strip_orig_mod(
        state_dict: Mapping[str, Tensor] | None,
    ) -> Mapping[str, Tensor] | None:
        """Reject a v1 (``ModelWrapper``) state-dict and strip a ``--compile``-added
        ``_orig_mod.`` prefix. Shared by the resume (`on_load_checkpoint`) and
        warm-start (`_warm_start_from_checkpoint`) paths.

        Returns the input unchanged when no ``_orig_mod.`` prefix is present
        (so callers can detect a no-op by identity); raises `ConfigError` on a
        v1 layout.
        """
        if state_dict and any(k.startswith("model.pool_net.") for k in state_dict):
            raise ConfigError(
                "this checkpoint has the v1 (ModelWrapper) state-dict layout "
                "('model.pool_net.*' keys) — v1 checkpoints are not supported by "
                "salt. Use them at the v1 pin 29c67a1 (git checkout 29c67a1) "
                "or convert the weights offline (see the parity-closure section "
                "of docs/architecture.md)."
            )
        if state_dict and any("_orig_mod." in k for k in state_dict):
            return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        return state_dict

    # -- init_from: weights-only warm start with module surgery -------------------

    def _warm_start_from_checkpoint(self, path: str) -> None:
        """Warm-start weights from `path` into the (already-bound) model with
        strict per-module accounting — the ``--init_from`` load path.

        Distinct from a resume `ckpt_path`: trainer state stays fresh, the FIT
        plan-hash check is NOT enforced (the checkpoint's hashes are logged for
        information only, since the architecture may have been surgically
        changed), and the state-dict load is prefix-filtered by module name
        rather than strict.

        The load is classified per config-module (``net.<name>.*`` prefix):

        - **loaded** — a module present in both the checkpoint and the current
          config, fully covered (identical key set, shapes, dtypes). Its
          tensors are loaded.
        - **new** — a config module absent from the checkpoint. Left at fresh
          init; `on_fit_start` materialises it.
        - **dropped** — a checkpoint module absent from the current config.
          Skipped and logged.

        A retained module that is only PARTIALLY covered (missing/unexpected
        subkeys, shape/dtype mismatch) is a hard `ConfigError`: that is an
        architecture swap — rename the module so it drops+adds cleanly. Also
        raises when unbound, on a missing ``state_dict``, or a v1 layout.
        """
        if not self._bound:
            raise ConfigError(
                f"--init_from {path!r}: warm start before bind — the model must compile its "
                "plans and bind first. This is an internal ordering error."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
        if not raw_state:
            raise ConfigError(
                f"--init_from {path!r}: the checkpoint carries no 'state_dict' — it is not a "
                "salt/Lightning training checkpoint."
            )
        ckpt_state = self._reject_v1_and_strip_orig_mod(raw_state)
        assert ckpt_state is not None  # non-empty raw_state → non-None

        # plan hashes are informational on a warm start (architecture may differ)
        payload = checkpoint.get(CKPT_KEY) if isinstance(checkpoint, Mapping) else None
        if payload:
            for mode, plan in self.plans.items():
                stored = (payload.get("plan_hashes") or {}).get(mode.name)
                if stored and stored != plan.plan_hash:
                    _LOG.info(
                        "--init_from: %s plan hash differs (checkpoint %s…, current %s…) — "
                        "not enforced on a weights-only warm start.",
                        mode.name,
                        stored[:16],
                        plan.plan_hash[:16],
                    )

        loaded, new, dropped = self._apply_warm_start(ckpt_state, path)
        self._init_warm_started = True
        self._init_loaded_modules = set(loaded)
        _LOG.info(
            "--init_from %s: %d module(s) loaded, %d new (fresh init + materialise), "
            "%d dropped.\n%s",
            path,
            len(loaded),
            len(new),
            len(dropped),
            _warm_start_summary(loaded, new, dropped),
        )

    def _apply_warm_start(
        self, ckpt_state: Mapping[str, Tensor], path: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Classify + load `ckpt_state` by module; returns ``(loaded, new,
        dropped)`` module-name lists. Raises `ConfigError` on partial coverage
        of a retained module (see `_warm_start_from_checkpoint`).
        """
        current_state = self.state_dict()
        current_by_mod = _partition_by_module(current_state)
        ckpt_by_mod = _partition_by_module(ckpt_state)
        config_names = list(self._graph_modules)

        loaded: list[str] = []
        new: list[str] = []
        partial: list[str] = []
        to_load: dict[str, Tensor] = {}
        for name in config_names:
            cur = current_by_mod.get(name, {})
            ckpt = ckpt_by_mod.get(name, {})
            if not ckpt:
                # no checkpoint weights for this config module → new (or a
                # params-free module, e.g. Concat/Split — nothing to load and
                # nothing to materialise, so it does not need reporting).
                if cur:
                    new.append(name)
                continue
            mismatch = _coverage_mismatch(cur, ckpt)
            if mismatch is not None:
                partial.append(f"  - {name}: {mismatch}")
                continue
            loaded.append(name)
            to_load.update({key: ckpt[key] for key in cur})
        if partial:
            raise ConfigError(
                f"--init_from {path!r}: {len(partial)} retained module(s) are only PARTIALLY "
                "covered by the checkpoint — their internal architecture changed. That is a "
                "swap, not a warm start: rename the module so it drops the old weights and "
                "fresh-inits the new ones (rename-with-weights is out of scope). "
                "Offenders:\n" + "\n".join(partial)
            )
        dropped = sorted(
            name for name in ckpt_by_mod if name is not None and name not in self._graph_modules
        )
        # only compatible retained-module tensors are handed to load_state_dict;
        # strict=False tolerates the missing new-module keys (never shape errors,
        # which the coverage preflight above already rejected).
        self.load_state_dict(to_load, strict=False)
        return loaded, new, dropped


def _partition_by_module(state: Mapping[str, Tensor]) -> dict[str | None, dict[str, Tensor]]:
    """Group a ``net.<name>.*`` state dict by module name. Keys outside the
    ``net.<name>.`` layout land under the ``None`` bucket (never a model
    module — informational only).
    """
    grouped: dict[str | None, dict[str, Tensor]] = {}
    for key, value in state.items():
        parts = key.split(".", 2)
        name = parts[1] if len(parts) >= 3 and parts[0] == "net" else None
        grouped.setdefault(name, {})[key] = value
    return grouped


def _coverage_mismatch(current: Mapping[str, Tensor], ckpt: Mapping[str, Tensor]) -> str | None:
    """Return a one-line description of why `ckpt` does not fully cover
    `current` (missing/unexpected keys, or a shape/dtype mismatch on a shared
    key), or ``None`` when coverage is exact. Runs BEFORE `load_state_dict`
    because PyTorch raises on a shape mismatch even under ``strict=False``.
    """
    cur_keys, ckpt_keys = set(current), set(ckpt)
    if missing := cur_keys - ckpt_keys:
        return f"{len(missing)} key(s) missing from checkpoint (e.g. {min(missing)})"
    if unexpected := ckpt_keys - cur_keys:
        return f"{len(unexpected)} extra key(s) in checkpoint (e.g. {min(unexpected)})"
    for key in sorted(cur_keys):
        cval, kval = current[key], ckpt[key]
        if tuple(cval.shape) != tuple(kval.shape):
            return f"shape mismatch at {key}: model {tuple(cval.shape)} vs ckpt {tuple(kval.shape)}"
        if cval.dtype != kval.dtype:
            return f"dtype mismatch at {key}: model {cval.dtype} vs ckpt {kval.dtype}"
    return None


def _warm_start_summary(loaded: list[str], new: list[str], dropped: list[str]) -> str:
    """A per-module warm-start summary table (loaded / new / dropped)."""
    rows = [
        *(f"  loaded   {name}" for name in loaded),
        *(f"  new      {name}  (fresh init + materialise)" for name in new),
        *(f"  dropped  {name}  (in checkpoint, not in config)" for name in dropped),
    ]
    return "\n".join(rows) if rows else "  (no module-level weights)"


def _dead_preds_message(dead: list[str], produced: Mapping[str, str], writers: Any) -> str:
    """Build the TEST dead-preds hard error: names each dead key's producing
    module, attributes the culprit when a writer's ``tasks`` list excludes
    it, and offers ``expose: [fit, val]`` (keep training, prune from
    TEST/ONNX) or ``--model.modules.X=null`` as fixes.
    """
    lines = ["[mode=TEST] prediction keys consumed by NO writer:"]
    for key in dead:
        name = produced.get(key)
        where = f" (produced by {name!r}, config: model.modules.{name})" if name else ""
        lines.append(f"  - {key!r}{where}")
    lines.append(
        "an unconsumed preds.* port in TEST means a computed prediction is never persisted."
    )
    dead_tasks = {
        parts[2]
        for key in dead
        if len(parts := key.split(KEY_SEP)) > 2  # preds.<stream>.<task>
    }
    hints = [
        f"outputs.{wname}.init_args.tasks: {list(tasks)} currently excludes {excluded}"
        for wname, writer in (getattr(writers, "writers", None) or {}).items()
        if (tasks := getattr(writer, "tasks", None)) is not None
        and (excluded := sorted(dead_tasks - set(tasks)))
    ]
    fix = "  fix: widen the writers"
    if hints:
        fix += " — " + "; ".join(hints) + " —"
    targets = sorted({produced[key] for key in dead if key in produced})
    if targets:
        expose_form = " / ".join(
            f"--model.modules.{name}.init_args.expose=[fit,val]" for name in targets
        )
        null_form = " / ".join(f"--model.modules.{name}=null" for name in targets)
        # expose: [fit, val] keeps the task training while pruning its prediction
        # from the TEST/ONNX plans — listed first; --model.modules.X=null (delete
        # the task entirely) is the heavier alternative.
        fix += (
            f", or opt the task out of eval with expose: [fit, val] ({expose_form}), "
            f"or remove the task module entirely ({null_form})"
        )
    else:
        fix += ", or opt the task out of eval with expose: [fit, val], or remove the task module"
    lines.append(fix)
    return "\n".join(lines)


def module_supports_mup(module: Any) -> bool:
    """Whether `module` accepts a ``mup`` init_arg (the ``apply_to`` target test).

    Eligible iff the module carries a ``mup`` attribute (init_arg landed) —
    duck-typed so user modules participate too.
    """
    return hasattr(module, "mup")


def module_mup_enabled(module: Any) -> bool:
    """Whether `module`'s ``mup`` flag is truthy (the validator's mup-on test)."""
    return bool(getattr(module, "mup", False))


def validate_mup_routing(
    mup: Mapping[str, Any] | None, modules: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate the muP routing config against the module dict.

    The single validator behind both `SaltModule.__init__` and
    ``salt graph validate``. ``apply_to`` is an explicit instance-name list.
    Hard `ConfigError`: naming a missing module, naming one without a ``mup``
    init_arg, unknown keys, or a non-list/missing ``apply_to``. Warning only:
    a mup-on module left out of ``apply_to`` (base shapes / MuAdamW grouping
    would be silently inconsistent). Returns the normalised config, or None
    when `mup` is None.
    """
    if mup is None:
        return None
    if not isinstance(mup, Mapping):
        raise ConfigError(
            f"model.init_args.mup must be a mapping with 'apply_to' (and optional 'shape_path'), "
            f"got {type(mup).__name__}"
        )
    if unknown := set(mup) - _MUP_KEYS:
        raise ConfigError(
            f"model.init_args.mup has unknown key(s) {sorted(unknown)} — expected "
            f"{sorted(_MUP_KEYS)}"
        )
    apply_to = mup.get("apply_to")
    if not isinstance(apply_to, (list, tuple)) or not apply_to:
        raise ConfigError(
            "model.init_args.mup needs a non-empty 'apply_to' list of module instance names "
            "(EXPLICIT names, NOT a regex — the v2 design break from v1's apply_to/parameter_name "
            "zip, configuration_muP.py:98)"
        )
    for raw in apply_to:
        if not isinstance(raw, str):
            raise ConfigError(
                f"model.init_args.mup.apply_to entries must be module instance names (str), "
                f"got {raw!r}"
            )
        if raw not in modules:
            raise ConfigError(
                f"model.init_args.mup.apply_to names {raw!r}, which is not a configured module — "
                f"known modules: {sorted(modules)} (apply_to is an explicit "
                "instance-name list)"
            )
        if not module_supports_mup(modules[raw]):
            raise ConfigError(
                f"model.init_args.mup.apply_to names {raw!r} "
                f"({type(modules[raw]).__name__}), which has no 'mup' init_arg — only modules "
                "that accept mup (StreamEmbed, TransformerEncoder) can be muP-routed"
            )
    applied = set(apply_to)
    for name, module in modules.items():
        if name not in applied and module_mup_enabled(module):
            warnings.warn(
                f"module {name!r} has mup: true but is NOT in model.init_args.mup.apply_to — its "
                "base shapes / MuAdamW grouping are skipped by the routing stage, so its training "
                "dynamics silently diverge from the muP intent. Add it to apply_to or "
                "set its mup: false.",
                stacklevel=2,
            )
    return {"apply_to": list(apply_to), "shape_path": mup.get("shape_path")}


def _validate_mup(
    mup: Mapping[str, Any] | None, modules: Mapping[str, Any]
) -> dict[str, Any] | None:
    """`SaltModule.__init__` wrapper around `validate_mup_routing`."""
    return validate_mup_routing(mup, modules)


def _edge_encoders(modules: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The encoder modules that declare an edge port (duck-typed on a non-None
    ``edges_key`` attribute — a `TransformerEncoder` with ``edges:`` configured).
    """
    return [
        (name, module)
        for name, module in modules.items()
        if getattr(module, "edges_key", None) is not None
    ]


def _concat_first_stream(modules: Mapping[str, Any]) -> tuple[str, str] | None:
    """The `Concat`'s first stream (the edge-stream-first reference), identified
    by its produced ``seq.x`` key (not by attribute, since `Normaliser`/`Split`
    also carry ``streams``). The edge tensor's stream must equal this so the
    ``[B, T, T, D_e]`` edge matrix aligns with the encoder's leading sequence
    rows. None when no `Concat` is configured.
    """
    for name, module in modules.items():
        streams = getattr(module, "streams", None)
        if not (isinstance(streams, (list, tuple)) and streams):
            continue
        declare = getattr(module, "declare_io", None)
        if not callable(declare):
            continue
        # the Concat is the module producing seq.x; declare_io is config-only/static,
        # so calling it here is cheap and side-effect-free.
        produces = flatten_spec(declare(Mode.FIT).produces)
        if "seq.x" in produces:
            return name, streams[0]
    return None


def validate_edge_port(modules: Mapping[str, Any]) -> int:
    """Validate the encoder edge-port bind-time constraints.

    Runs at `SaltModule.__init__` and in ``salt graph validate``; no-op when
    no encoder declares an edge port. Rule (a) edge-stream-first: the edge
    tensor's stream must be ``Concat.streams[0]``, because the encoder
    zero-pads the ``[B, T, T, D_e]`` edge matrix assuming the edge stream's
    rows lead the concatenated sequence. Rule (b): an encoder with an edge
    port may not declare a backend outside ``{"torch-math"}`` — EdgeAttention
    only supports raw torch attention. Either violation is a `ConfigError`;
    returns the number of edge-bearing encoders validated.
    """
    encoders = _edge_encoders(modules)
    if not encoders:
        return 0
    concat = _concat_first_stream(modules)
    for name, module in encoders:
        edge_stream = module.edge_stream  # "edges.tracks_emb" -> "tracks"
        # -- rule (a): edge stream must be Concat.streams[0] --------------------
        if concat is None:
            raise ConfigError(
                f"encoder {name!r} declares an edge port (edges={module.edges_key!r}) but no "
                "Concat is configured — the edge tensor's stream must be the FIRST concat stream "
                "so the [B, T, T, D_e] edge matrix aligns with the leading sequence rows "
                "(v1 sort-first hack, saltmodel.py:66-73)"
            )
        concat_name, first_stream = concat
        if edge_stream != first_stream:
            raise ConfigError(
                f"encoder {name!r} edge port stream {edge_stream!r} (from edges="
                f"{module.edges_key!r}) is NOT the first stream of Concat {concat_name!r} "
                f"(streams[0]={first_stream!r}) — the edge tensor must align with the LEADING "
                "rows of the concatenated sequence (the encoder zero-pads it to the "
                "register-augmented length assuming the edge stream is first, "
                f"transformer.py:689-719). fix: put {edge_stream!r} first in {concat_name!r}'s "
                "streams (v1 did this silently via the init-net sort, saltmodel.py:66-73)"
            )
        # -- rule (b): no non-edge attention backend alongside an edge port -----
        # EdgeAttention encoders always run raw torch attention, so any other
        # declared backend is an error rather than a silent no-op.
        attn_type = getattr(getattr(module, "encoder", None), "attn_type", "torch-math")
        if attn_type not in _EDGE_OK_BACKENDS:
            raise ConfigError(
                f"encoder {name!r} declares attention backend {attn_type!r} alongside an edge "
                f"port (edges={module.edges_key!r}), but EdgeAttention supports ONLY raw torch "
                f"attention ({sorted(_EDGE_OK_BACKENDS)}). v1 silently IGNORED the backend when "
                "edge features were on (transformer.py:599-601) — v2 makes that a named error so "
                "a flash-varlen edge config fails loudly instead of running unexpectedly-slow raw "
                "attention. fix: set the encoder's attention.attn_type to 'torch-math' (or drop "
                "the edge port)."
            )
    return len(encoders)


def _validate_edge_port(modules: Mapping[str, Any]) -> int:
    """`SaltModule.__init__` wrapper around `validate_edge_port`."""
    return validate_edge_port(modules)


def check_class_names(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Cross-check configured ``class_names`` against schema label attrs.

    For every module declaring ``class_names`` + ``stream`` + ``label``
    (duck-typed), the configured list must match the schema artifact's label
    attr in set AND order — a reordered ``class_names`` is a silent physics
    mislabeling no shape check can catch. Runs at `SaltModule.setup` and in
    ``salt graph validate``; no-op without a schema artifact. Returns the
    number of lists compared; mismatch raises `ConfigError`.
    """
    schema_group = getattr(reader, "schema_group", None)
    if not callable(schema_group):
        return 0
    checked = 0
    for name, module in modules.items():
        class_names = getattr(module, "class_names", None)
        stream = getattr(module, "stream", None)
        label = getattr(module, "label", None)
        if class_names is None or stream is None or label is None:
            continue
        gschema = schema_group(stream)
        if gschema is None:
            continue
        attr = gschema.attrs.get(label)
        if not (
            isinstance(attr, (list, tuple)) and attr and all(isinstance(item, str) for item in attr)
        ):
            continue  # no class-name attr for this label — nothing to check
        checked += 1
        if list(class_names) != list(attr):
            configured, stored = list(class_names), list(attr)
            diagnosis = (
                "same classes, DIFFERENT ORDER — the model head indices would be "
                "silently mislabelled"
                if sorted(configured) == sorted(stored)
                else "the class sets differ"
            )
            raise ConfigError(
                f"class_names of module {name!r} (config: model.modules.{name}."
                f"init_args.class_names) do not match the {label!r} attr of the "
                f"{stream!r} group in the schema artifact ({diagnosis}).\n"
                f"  configured: {configured}\n"
                f"  schema:     {stored}\n"
                f"  fix: set class_names to the schema order (or re-dump the schema if "
                "the file genuinely changed)"
            )
    return checked


def resolve_origin_weighting(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Resolve name-based ``origin_weighting`` to ids before bind.

    Maps names to integer origin ids against the schema artifact for every
    module exposing ``resolve_origin_names`` (duck-typed). Runs at
    `SaltModule.setup` before the two-phase bind, so resolved ids are in
    place when `VertexingTaskModule.bind` builds the composed head. Integer-id
    weighting and schema-less readers are no-ops; a name-based config that
    resolves nothing fails loudly at bind. Returns the count resolved.
    """
    resolved = 0
    for module in modules.values():
        resolve = getattr(module, "resolve_origin_names", None)
        if callable(resolve) and resolve(reader):
            resolved += 1
    return resolved
