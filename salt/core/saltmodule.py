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

from salt.core.data.datamodule import GraphDataModule
from salt.core.data.dataset import MODEL_VISIBLE_NAMESPACES, GraphDataset
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.executor import Executor
from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import (
    KEY_SEP,
    GraphModule,
    Mode,
    TensorSpec,
    _has_wildcard,
    flatten_spec,
    unflatten_spec,
)
from salt.core.nn.bind import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.losses import LossGLS, LossSum
from salt.core.optim import HybridMuonAdamW

try:
    from lion_pytorch import Lion

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
    "validate_edge_port",
    "validate_mup_routing",
]

# the only attention backend an EdgeAttention encoder may declare — other
# backends would silently bypass the edge computation.
_EDGE_OK_BACKENDS = frozenset({"torch-math"})

CKPT_KEY = "salt_core"
"""Checkpoint dict key for the schema + plan-hash payload."""

_LRS_REQUIRED = ("initial", "max", "end", "pct_start")
_OPTIMIZERS = ("AdamW", "lion", "HybridMuonAdamW")
_MUP_KEYS = frozenset({"apply_to", "shape_path"})
# dataset-boundary demand is declared for the three runtime modes; ONNX
# export feeds the model directly and has no dataset plan.
_DEMAND_MODES = (Mode.FIT, Mode.VAL, Mode.TEST)


def _is_test_persistence_sink(callback: Any) -> bool:
    """Whether a ``writer_demand``-exposing callback is the TEST persistence sink.

    Discriminates a real persistence sink (e.g. `H5OutputSink`) from an
    ONNX-only sink (`OnnxExportSink`), which must never be chosen as the TEST
    sink since its TEST-mode declare_io is empty.
    """
    is_test_sink = getattr(callback, "is_test_sink", None)
    return True if not callable(is_test_sink) else bool(is_test_sink())


class SaltModule(lightning.LightningModule):
    """Lightning wrapper around a configured graph-module dict.

    Parameters
    ----------
    modules : dict[str, GraphModule | None]
        Model-side graph modules by instance name. ``None`` entries are
        dropped (deletion via ``--model.modules.X=null``).
    lrs : Mapping[str, float]
        OneCycleLR schedule config: required keys ``initial``, ``max``,
        ``end``, ``pct_start``; optional ``weight_decay`` (default 1e-5) and
        ``last_epoch`` (default -1).
    optimizer : str, optional
        One of ``"AdamW"`` (default), ``"lion"``, ``"HybridMuonAdamW"``. When
        `mup` is configured the optimizer is swapped to ``mup.optim.MuAdamW``
        regardless of this name.
    mup : Mapping[str, Any], optional
        The muP routing config. ``None`` (default) = no muP. Keys:

        - ``apply_to`` (required): explicit list of module instance names
          that carry ``mup: true``. Each must exist in `modules` and accept
          a ``mup`` init_arg, else `ConfigError`.
        - ``shape_path`` (optional): base-shapes file produced by
          ``salt2 mup-shapes`` / ``setup_mup``, applied at bind time so
          `MuReadout.width_mult()` resolves against real base widths.
    name : str, optional
        Model name (run naming/metadata), by default ``"salt"``.
    debug : bool, optional
        Run the executor in debug mode (read tracking + in-place mutation
        detection), by default False. Slow — tests only.

    Raises
    ------
    ConfigError
        For an empty module dict, a non-``nn.Module`` module, a bad
        optimizer name, or missing `lrs` keys.
    """

    def __init__(
        self,
        modules: dict[str, GraphModule | None],
        lrs: Mapping[str, float],
        optimizer: str = "AdamW",
        mup: Mapping[str, Any] | None = None,
        name: str = "salt",
        debug: bool = False,
        outputs: dict[str, GraphModule | None] | None = None,
    ) -> None:
        super().__init__()
        # a null entry — from a config-file or CLI override — deletes the module.
        modules = {key: module for key, module in modules.items() if module is not None}
        if not modules:
            raise ConfigError("SaltModule needs a non-empty module dict (design §3.4)")
        for key, module in modules.items():
            if not isinstance(module, nn.Module):
                raise ConfigError(
                    f"module {key!r} ({type(module).__name__}) is not an nn.Module — "
                    "model-side graph modules carry parameters/buffers (design §2.5)"
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
                f"{list(_LRS_REQUIRED)} (+ optional weight_decay, last_epoch; design §3.4)"
            )
        if optimizer not in _OPTIMIZERS:
            raise ConfigError(
                f"optimizer {optimizer!r} is not supported — choose from {list(_OPTIMIZERS)}"
            )
        # validates apply_to against the module dict and warns on a mup-on module
        # left out of apply_to — see _validate_mup.
        self.mup_cfg: dict[str, Any] | None = _validate_mup(mup, modules)
        # edge-stream-first + EdgeAttention-backend forcing bind-time validators.
        # No-op without an edge encoder.
        _validate_edge_port(modules)
        # resolved config only — the module dict is NOT pickled into hparams
        # (load_from_checkpoint takes modules= explicitly)
        self.save_hyperparameters(logger=False, ignore=["modules"])
        self.name = name
        self.lrs = dict(lrs)
        self.optimizer = optimizer
        self.debug = debug
        self.net = nn.ModuleDict(modules)  # ckpt keys: net.<name>.* (dict order, not topo)
        self._graph_modules: dict[str, GraphModule] = dict(modules)
        # the top-level outputs: section, composed AFTER the model (empty until
        # compose_output_section runs — either here or from the CLI path).
        self._output_section: dict[str, GraphModule] = {}
        self.plans: dict[Mode, Plan] = {}
        self._executors: dict[Mode, Executor] = {}
        self.schema: ResolvedSchema | None = None
        self._bound = False
        self._materialised = False
        self._loaded_from_checkpoint = False
        self._ckpt_plan_hashes: dict[str, str] = {}
        # programmatic-construction path: compose the outputs: section now (the CLI
        # path passes outputs=None here and composes via instantiate_classes).
        if outputs:
            self.compose_output_section(
                {key: w for key, w in outputs.items() if w is not None}
            )

    def compose_output_section(self, section: Mapping[str, GraphModule]) -> None:
        """Compose the top-level ``outputs:`` section onto the model.

        Section writers (`RunTaskOutput` / `InputCopyWriter` / `PadMaskWriter`) are
        folded into the planning module dict (and `net`, params-free so the
        state_dict is unchanged). Their ``outputs.*`` leaves reach a sink only in
        TEST/ONNX, so FIT/VAL demand-prune them.

        MANIFEST-ONLY writers (`InputCopyWriter` — input copies are re-read from the
        source H5 by the SINK, never flowing through the graph) are NOT folded into
        the graph: they have no ``produces``, so the demand closure would prune
        them. They stay in the section dict for the sink to read their copy spec at
        ``bind_output_section``.

        Parameters
        ----------
        section : Mapping[str, GraphModule]
            The section writers by instance name, in declaration order (the eval-H5
            column-order authority).

        Raises
        ------
        ConfigError
            For a section writer whose name collides with a model module.
        """
        section = {key: w for key, w in section.items() if w is not None}
        if not section:
            return
        for key, w in section.items():
            if key in self._graph_modules:
                raise ConfigError(
                    f"outputs: section writer {key!r} collides with a model module — instance "
                    "names are unique across the pipeline graph (plan 34 W34.2)"
                )
            w.name = key
        # the model-side modules before the section folds in — RunTaskOutput
        # resolves its tasks against these.
        model_modules = dict(self._graph_modules)
        graph_writers = {
            key: w
            for key, w in section.items()
            if not (callable(getattr(w, "is_manifest_only", None)) and w.is_manifest_only())
        }
        for key, w in graph_writers.items():
            self.net[key] = w  # ride the ModuleDict (params-free, state_dict unchanged)
            self._graph_modules[key] = w
        self._output_section = dict(section)
        for w in section.values():
            if callable(getattr(w, "bind_model_modules", None)):
                w.bind_model_modules(model_modules)

    # -- lifecycle state (read-only — gates and tests assert on these) ---------

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
        added in TEST for writer row alignment. `GraphDataModule.setup`
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
        for mode in _DEMAND_MODES:
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
                        "narrow it before compile (design §2.2 rule (d), §3.3)"
                    )
                if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                    consumers = ", ".join(
                        f"{name!r} (config: model.modules.{name})" for name in required[key]
                    )
                    raise ConfigError(
                        f"[mode={mode.name}] key {key!r} is required by module(s) {consumers} "
                        "but no model module produces it, and it cannot come from the dataset "
                        f"(the dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                        f"keys only, design §6.1).\n  fix: add or restore a module producing "
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
                            "writer requires are concrete keys (design §2.2, §8)"
                        )
                    if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                        raise ConfigError(
                            f"[mode=TEST] key {key!r} is required by {who} but no model "
                            "module produces it, and it cannot come from the dataset (the "
                            f"dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                            "keys only, design §6.1, §8).\n  fix: correct the writer's "
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
                            "callback requires are concrete keys (design §2.2, §3.1)"
                        )
                    if key.split(KEY_SEP, 1)[0] not in MODEL_VISIBLE_NAMESPACES:
                        raise ConfigError(
                            f"[mode={mode.name}] key {key!r} is required by {who} but no "
                            "model module produces it, and it cannot come from the dataset "
                            f"(the dataset boundary serves {'/'.join(MODEL_VISIBLE_NAMESPACES)} "
                            "keys only, design §6.1, §3.1).\n  fix: correct the callback's "
                            "requires, or add a module producing the key"
                        )
                    demand.append(key)
                    origins[key] = who
            out[mode] = (demand, origins)
        return out

    def _attached_writer(self) -> tuple[Any, Any]:
        """The attached TEST writer/sink callback + reader, duck-typed on
        ``writer_demand``. Order-independent: an ONNX-only sink's
        ``is_test_sink()`` discriminator skips it so it is never chosen as the
        TEST persistence sink; a plain duck-typed sink without ``is_test_sink``
        counts as one. ``(None, None)`` when none is attached.
        """
        trainer = self._trainer
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        callback = next(
            (
                cb
                for cb in callbacks or []
                if callable(getattr(cb, "writer_demand", None)) and _is_test_persistence_sink(cb)
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
        """Bind the outputs: section to every attached sink callback.

        Sinks resolve their column schema + copy spec + mask streams from the
        section manifest, so the section must be bound before any
        declare_io/writer_demand resolution. Binds to every attached callback
        exposing ``bind_output_section`` (H5 persistence + ONNX export).
        No-op when no outputs: section is configured.
        """
        if not self._output_section:
            return
        trainer = self._trainer
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        for cb in callbacks or []:
            if callable(getattr(cb, "bind_output_section", None)):
                cb.bind_output_section(self._output_section)

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
                "unique across the pipeline graph (design §2.2); rename the callback key"
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
        so ``salt2 graph`` sees the same sinks a real ``trainer.fit`` would.
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
        static `salt.core.cli` path.
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
        `salt.core.cli` path; default None discovers from the trainer.
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
                    "anchor on it; add a LossSum module (design §3.3)"
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
                        "produces — check writers.modules (design §8)"
                    )
                return consumed
        # the ONNX output manifest is not writer-derived — the folded OnnxExportSink
        # anchors its conversion-leaf demand itself. When no export sink demand is
        # present it falls back to every preds.* key below.
        if not preds:
            raise ConfigError(
                f"no module produces a 'preds.*' key in mode {mode.name} — evaluation plans "
                "anchor on predictions (design §3.1, §3.3)"
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
            `GraphDataset.boundary_specs()`.

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
        # bind the outputs: section to the sink so it dumps the section's
        # outputs.* leaves in declaration order, not producer discovery.
        if (
            self._output_section
            and sink_node is not None
            and callable(getattr(sink_node, "bind_output_section", None))
        ):
            sink_node.bind_output_section(self._output_section)
        folded_sink = sink_node is not None and mode is Mode.TEST
        if folded_sink:
            # the sink node anchors ALL its demand via its declared requires —
            # flat model sinks are empty. The dead-preds safety net is restored
            # AFTER compile via `_assert_no_dead_preds` below: a `preds.*` the
            # model computes but no producer feeds into a demanded output is
            # still a hard error at `salt2 test`.
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
                "boundary or module config drifted (design §3.1 determinism)"
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
                f"stage {stage!r} is not supported by SaltModule in M2 — use trainer.fit or "
                "trainer.test (validate/predict entry points are M5+, design §9.5)"
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

    def _validate_writer_specs(self, test_dset: GraphDataset) -> None:
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

    def _run_preflights(self) -> None:
        """Fail-fast data-free checks of file-backed `materialise` sources: every
        module exposing a callable ``preflight()`` (e.g. `Normaliser`) is
        checked before any `materialise` writes buffers, so a bad path/content
        raises one `ConfigError` instead of a per-module mid-materialise crash.
        Called from ``on_fit_start`` on fresh fits only.
        """
        for module in self._graph_modules.values():
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
                f"({fit_hash[:16]}…) — VAL is contractually the identical plan in M2 "
                "(design §3.4; VAL-divergent module ports need the M6 exemption mechanism)"
            )

    def _graph_datamodule(self) -> GraphDataModule:
        """The attached `GraphDataModule` (model plans need its boundary); raises
        `ConfigError` when none is attached or it is not a `GraphDataModule`.
        """
        dm = getattr(self._trainer, "datamodule", None) if self._trainer is not None else None
        if not isinstance(dm, GraphDataModule):
            raise ConfigError(
                "SaltModule compiles its plans against a GraphDataModule's dataset boundary — "
                f"pass one to trainer.fit/test (got {type(dm).__name__}; raw dataloaders are "
                "not supported, design §3.4/§6.1)"
            )
        return dm

    @staticmethod
    def _boundary(dset: GraphDataset | None, stage: str) -> dict[str, TensorSpec]:
        """A stage dataset's model-visible boundary specs; raises `ConfigError`
        if the stage dataset was never built.
        """
        if dset is None:
            raise ConfigError(
                f"the datamodule has no {stage} dataset — its setup did not run or the "
                f"{stage}_file is unset (design §6.1)"
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
                "state-dict load (design §2.3)"
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
        from pathlib import Path as _Path  # noqa: PLC0415 - local, muP-only path

        from mup import set_base_shapes  # noqa: PLC0415 - mup is optional, muP-only path

        shape_path = _Path(self.mup_cfg["shape_path"])
        if not shape_path.is_file():
            raise ConfigError(
                f"model.init_args.mup.shape_path {str(shape_path)!r} does not exist — generate it "
                "with `salt2 mup-shapes` (setup_mup) before fit (design §3.4, §9.2)"
            )
        # the file carries infshapes for the whole net (generated from net) — apply
        # it ONCE over self.net so the apply_to MuReadout/linears get their
        # base/delta infshapes; non-apply_to params have fixed dims in the file
        set_base_shapes(self.net, str(shape_path), rescale_params=False)

    # -- materialise (fresh fits only) --------------------------------------------

    def on_fit_start(self) -> None:
        """Materialise file-backed values before the first step of a FRESH fit.

        Runs after checkpoint restore: when any checkpoint was loaded
        (resume or `load_from_checkpoint`) materialise is skipped and the
        values come from the state_dict — `Normaliser.forward` raises if a
        checkpoint lacked them.
        """
        if self._materialised or self._loaded_from_checkpoint:
            return
        self._run_preflights()
        materialise_all(self._graph_modules)
        self._materialised = True

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
                "compile_mode() first (design §3.4)"
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

    def _get_optimizer_class(self) -> type[Optimizer]:
        """Resolve the optimizer class, with the muP ``MuAdamW`` swap: when `mup`
        is configured the optimizer is always ``mup.optim.MuAdamW`` (coupled to
        the base shapes set at bind via `_apply_mup_shapes`). Raises
        `ImportError` for lion without lion-pytorch, `ConfigError` for an
        unsupported name.
        """
        if self.mup_cfg is not None:
            from mup.optim import MuAdamW  # noqa: PLC0415 - mup is optional, muP-only path

            return MuAdamW
        if self.optimizer == "lion":
            if not _lion_available:
                raise ImportError(
                    "Lion optimizer requested but not available. "
                    "Check installation of lion-pytorch."
                )
            return Lion
        if self.optimizer == "AdamW":
            return AdamW
        if self.optimizer == "HybridMuonAdamW":
            return HybridMuonAdamW
        raise ConfigError(f"Optimizer '{self.optimizer}' is not supported.")

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        """Build the optimizer + a step-interval OneCycleLR scheduler."""
        optimizer_class = self._get_optimizer_class()
        optimizer_kwargs = {
            "lr": self.lrs["initial"],
            "weight_decay": self.lrs.get("weight_decay", 1e-5),
        }
        if optimizer_class is HybridMuonAdamW:
            opt = optimizer_class(self.named_parameters(), **optimizer_kwargs)
        else:
            opt = optimizer_class(self.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs["max"] / self.lrs["initial"],
            final_div_factor=self.lrs["initial"] / self.lrs["end"],
            pct_start=float(self.lrs["pct_start"]),
            last_epoch=int(self.lrs.get("last_epoch", -1)),
            cycle_momentum=optimizer_class is not HybridMuonAdamW,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    # -- checkpoints: schema + plan hashes, never Plan objects --------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Serialise the resolved schema + per-mode plan hashes under `CKPT_KEY`
        (Plans themselves aren't picklable; the hash is the integrity check).
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
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Runs before the state-dict load on both restore paths: verifies plan
        hashes (fatal on FIT mismatch via `_verify_ckpt_hash`), binds from the
        checkpoint's stored schema if not yet bound, strips a ``--compile``-added
        ``_orig_mod.`` state_dict prefix, and rejects the v1 (``ModelWrapper``)
        state-dict layout with `ConfigError`. Marks the instance checkpoint-loaded
        (disables `materialise`).
        """
        state_dict = checkpoint.get("state_dict")
        if state_dict and any(k.startswith("model.pool_net.") for k in state_dict):
            raise ConfigError(
                "this checkpoint has the v1 (ModelWrapper) state-dict layout "
                "('model.pool_net.*' keys) — v1 checkpoints are not supported by "
                "salt.core. Use them at the v1 pin 29c67a1 (git checkout 29c67a1) "
                "or convert the weights offline (see the parity-closure section "
                "of salt/core/README.md)."
            )
        if state_dict and any("_orig_mod." in k for k in state_dict):
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }

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
            "boundary changed since the checkpoint was written (design §2.3, risk 9)"
        )
        if mode is Mode.FIT:
            raise ConfigError(msg)
        warnings.warn(f"{msg} — continuing (non-FIT modes warn only)", stacklevel=2)


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
        "an unconsumed preds.* port in TEST means a computed prediction is never "
        "persisted (design §4.2, §8)."
    )
    dead_tasks = {
        parts[2]
        for key in dead
        if len(parts := key.split(KEY_SEP)) > 2  # preds.<stream>.<task>
    }
    hints = [
        f"writers.modules.{wname}.init_args.tasks: {list(tasks)} currently excludes {excluded}"
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
    """Whether `module` accepts a ``mup`` init_arg (the routing apply_to target test).

    `StreamEmbed`/`TransformerEncoder` carry a ``mup`` init_arg stored as a
    ``mup`` attribute; a module is an eligible ``apply_to`` target iff it
    carries that attribute. Duck-typed so user modules participate too.

    Parameters
    ----------
    module : Any
        The model-side graph module.

    Returns
    -------
    bool
        True if the module has a ``mup`` attribute (the init_arg landed).
    """
    return hasattr(module, "mup")


def module_mup_enabled(module: Any) -> bool:
    """Whether `module`'s ``mup`` flag is truthy (the validator's mup-on test).

    Returns
    -------
    bool
        True if the module both supports muP and has it switched on.
    """
    return bool(getattr(module, "mup", False))


def validate_mup_routing(
    mup: Mapping[str, Any] | None, modules: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate the muP routing config against the module dict.

    The single validator behind both `SaltModule.__init__` and
    ``salt2 graph validate`` so the same rules fire data-free in CI and at
    run construction. ``apply_to`` is an explicit instance-name list, not a
    regex.

    Rules:

    - ``apply_to`` naming a module that does NOT exist → `ConfigError`.
    - ``apply_to`` naming a module that lacks a ``mup`` init_arg
      (`module_supports_mup` False) → `ConfigError`.
    - a module with ``mup`` truthy that is NOT in ``apply_to`` → warning
      (its base shapes / MuAdamW grouping would be silently inconsistent).
    - unknown top-level keys / a non-list ``apply_to`` / a missing
      ``apply_to`` → `ConfigError`.

    Parameters
    ----------
    mup : Mapping[str, Any] | None
        The ``model.init_args.mup`` block (None = no muP).
    modules : Mapping[str, Any]
        The model-side module dict (instance name -> module). For the static
        validator this is the model subdict; for `SaltModule` it is the
        pre-filtered ctor dict.

    Returns
    -------
    dict[str, Any] | None
        The normalised mup config (``{"apply_to": [...], "shape_path": ...}``)
        or None when `mup` is None.

    Raises
    ------
    ConfigError
        On any of the hard-error rules above.
    """
    if mup is None:
        return None
    if not isinstance(mup, Mapping):
        raise ConfigError(
            f"model.init_args.mup must be a mapping with 'apply_to' (and optional 'shape_path'), "
            f"got {type(mup).__name__} (design §3.4 line 685)"
        )
    if unknown := set(mup) - _MUP_KEYS:
        raise ConfigError(
            f"model.init_args.mup has unknown key(s) {sorted(unknown)} — expected "
            f"{sorted(_MUP_KEYS)} (design §3.4 line 685)"
        )
    apply_to = mup.get("apply_to")
    if not isinstance(apply_to, (list, tuple)) or not apply_to:
        raise ConfigError(
            "model.init_args.mup needs a non-empty 'apply_to' list of module instance names "
            "(EXPLICIT names, NOT a regex — the v2 design break from v1's apply_to/parameter_name "
            "zip, configuration_muP.py:98; design §3.4 line 685)"
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
                f"known modules: {sorted(modules)} (design §3.4: apply_to is an explicit "
                "instance-name list)"
            )
        if not module_supports_mup(modules[raw]):
            raise ConfigError(
                f"model.init_args.mup.apply_to names {raw!r} "
                f"({type(modules[raw]).__name__}), which has no 'mup' init_arg — only modules "
                "that accept mup (StreamEmbed, TransformerEncoder) can be muP-routed "
                "(design §3.4 line 695 validator rule 1)"
            )
    applied = set(apply_to)
    for name, module in modules.items():
        if name not in applied and module_mup_enabled(module):
            warnings.warn(
                f"module {name!r} has mup: true but is NOT in model.init_args.mup.apply_to — its "
                "base shapes / MuAdamW grouping are skipped by the routing stage, so its training "
                "dynamics silently diverge from the muP intent (design §3.4 line 695 validator "
                "rule 2). Add it to apply_to or set its mup: false.",
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

    Runs at `SaltModule.__init__` (every fit/test) and in
    ``salt2 graph validate`` so the same rules fire data-free in CI and at
    run construction — the edge analogue of `validate_mup_routing`. No-op
    when no encoder declares an edge port.

    Rules:

    - **edge-stream-first** (rule a): the edge tensor's stream must be the
      first stream of the `Concat` (``Concat.streams[0]``) — required
      because the encoder zero-pads the ``[B, T, T, D_e]`` edge matrix to
      the register-augmented sequence length assuming the edge stream's
      ``T`` rows are the leading rows of the concatenated sequence. A
      mis-ordered concat is a `ConfigError`.
    - **EdgeAttention-backend forcing** (rule b): an encoder with an edge
      port may not declare a non-edge attention backend (any backend
      outside ``{"torch-math"}``), since EdgeAttention only supports raw
      torch attention.

    Parameters
    ----------
    modules : Mapping[str, Any]
        The model-side module dict (instance name -> module). For the static
        validator this is the model subdict; for `SaltModule` it is the
        pre-filtered ctor dict.

    Returns
    -------
    int
        The number of edge-bearing encoders validated (0 = no edge path).

    Raises
    ------
    ConfigError
        On either rule, naming the encoder, the edge stream, and the conflict.
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
                "(v1 sort-first hack, saltmodel.py:66-73; FD §6.7 1425-1431)"
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
                "streams (v1 did this silently via the init-net sort, saltmodel.py:66-73; "
                "FD §6.7 1425-1431 rule a)"
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
                "the edge port). FD §6.7 1425-1431 rule b"
            )
    return len(encoders)


def _validate_edge_port(modules: Mapping[str, Any]) -> int:
    """`SaltModule.__init__` wrapper around `validate_edge_port`."""
    return validate_edge_port(modules)


def check_class_names(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Cross-check configured ``class_names`` against schema label attrs.

    Default-on whenever both sides exist: for every module declaring
    ``class_names`` + ``stream`` + ``label`` (duck-typed so user task
    modules participate too), the reader's schema artifact is consulted —
    if the stream's group carries an attr named like the label whose value
    is a list of strings (e.g. the ``flavour_label`` attr on ``jets``), the
    configured list must match in set **and order**. A reordered
    ``class_names`` is a silent physics mislabeling no shape check can
    catch. Runs at `SaltModule.setup` (every fit/test) and in
    ``salt2 graph validate``. No-op without a schema artifact.

    Parameters
    ----------
    modules : Mapping[str, GraphModule]
        The model-side module dict.
    reader : Any
        The dataset reader; consulted via ``schema_group(stream)``
        (duck-typed — readers without schema support are skipped).

    Returns
    -------
    int
        The number of class-name lists actually compared (0 when no schema
        artifact / no matching attrs exist).

    Raises
    ------
    ConfigError
        Naming the module, its config address, and both orderings, when a
        configured list mismatches the schema attr.
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
                f"{stream!r} group in the schema artifact ({diagnosis}, design §2.6).\n"
                f"  configured: {configured}\n"
                f"  schema:     {stored}\n"
                f"  fix: set class_names to the schema order (or re-dump the schema if "
                "the file genuinely changed)"
            )
    return checked


def resolve_origin_weighting(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Resolve name-based ``origin_weighting`` to ids before bind.

    For every module exposing a callable ``resolve_origin_names`` (duck-typed
    so user vertexing modules participate too), the names are mapped to
    integer origin ids against the schema artifact. Runs at `SaltModule.setup`
    (fit/test) right after `check_class_names`, before the two-phase bind, so
    the resolved ids are in place when `VertexingTaskModule.bind` builds the
    composed head. Integer-id weighting and readers without a schema are
    no-ops here; a name-based config that resolves nothing then fails loudly
    at bind.

    A `ConfigError` from a module's `resolve_origin_names` (an unknown class
    name, or a missing origin class-name attr for a name-based config)
    propagates unchanged.

    Parameters
    ----------
    modules : Mapping[str, GraphModule]
        The model-side module dict.
    reader : Any
        The stage dataset reader (``schema_group`` consulted per module).

    Returns
    -------
    int
        The number of modules whose names were actually resolved here.
    """
    resolved = 0
    for module in modules.values():
        resolve = getattr(module, "resolve_origin_names", None)
        if callable(resolve) and resolve(reader):
            resolved += 1
    return resolved


def bundle_as_v1_outputs(bundle: Bundle) -> dict[str, Any]:
    """Migration shim: expose a bundle as the v1 ``{preds, labels, pad_masks}`` view.

    Deprecated. ``preds`` and ``labels`` keep their nested
    ``{stream: {name: tensor}}`` shape; ``pad_masks`` maps each masked
    stream (``masks.*`` minus the encoder's ``registers`` entry) to its
    True-is-padded mask.

    Returns
    -------
    dict[str, Any]
        ``{"preds": ..., "labels": ..., "pad_masks": ...}`` with ``{}`` for
        absent namespaces.
    """
    masks: dict[str, Tensor] = dict(bundle.subtree("masks")) if "masks" in bundle.data else {}
    masks.pop("registers", None)
    return {
        "preds": bundle.subtree("preds") if "preds" in bundle.data else {},
        "labels": bundle.subtree("labels") if "labels" in bundle.data else {},
        "pad_masks": masks,
    }
