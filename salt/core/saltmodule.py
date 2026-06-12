"""`SaltModule` — the LightningModule that replaces v1 `ModelWrapper` (design §3.4).

The module owns the configured graph-module dict, compiles per-mode plans at
``setup`` against the `GraphDataModule`'s config-derived dataset boundary,
runs the two-phase bind (design §2.3 — config-only, NO file I/O), and drives
the executor from the Lightning step hooks. Checkpoints carry the resolved
schema and per-mode plan hashes — never `Plan` objects, which hold
``MappingProxyType`` views and are not picklable by design (§2.3, risk 9).

Lifecycle (design §2.3, §3.4), in trainer order:

1. ``__init__`` — pure config capture: instance names assigned from the dict
   keys, the ``losses.**`` framework wildcard narrowed (`LossSum.narrow`),
   modules registered in an ``nn.ModuleDict`` (checkpoint keys
   ``net.<name>.*`` — independent of topo order).
2. ``setup(stage)`` — compile the stage's plans (FIT+VAL for fit, TEST for
   test) with sources from the datamodule's ``boundary_specs()``, then
   ``bind(schema)`` over all compiled plans — EXACTLY ONCE, and always
   before any state-dict load (Lightning restores checkpoints after the
   setup hook; rebinding would re-initialise bound layers).
3. checkpoint restore (when resuming/loading) — `on_load_checkpoint` runs
   before the state-dict load: plan hashes are verified (FIT strict, other
   modes warn — design risk 9) and, on the data-less
   ``load_from_checkpoint`` path where ``setup`` never ran, the modules are
   bound from the schema stored in the checkpoint (§2.3: checkpoints load
   on data-less machines).
4. ``on_fit_start`` — `materialise_all` on FRESH fits only (Normaliser
   buffers, ``weight_source`` class weights). Skipped when a checkpoint was
   loaded: the values arrive via the state_dict, reproducing v1's
   bake-on-resume semantics (cli.py:253-267) without re-reading files.
5. steps — assemble a `Bundle` from the batch, run the mode's executor, log
   ``{stage}/loss`` + ``{stage}/{task}_loss`` (v1 monitor convention,
   modelwrapper.py:256-270), return ``{"loss": ..., "bundle": ...}``
   (design §3.4 — the conscious break of hidden contract #8).

M2 scope notes: muP (design §3.4 routing/shape halves) and the
``validate``/``predict`` trainer entry points are M5/M6; compiled plans make
the instance non-picklable, so DDP must use fork-based workers (spawn
strategies are revisited with the M6 cluster work).
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
    flatten_spec,
    unflatten_spec,
)
from salt.core.nn.bind import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.modules import LossSum
from salt.optim import HybridMuonAdamW

try:
    from lion_pytorch import Lion

    _lion_available = True
except ImportError:
    _lion_available = False

__all__ = ["CKPT_KEY", "SaltModule", "check_class_names"]

CKPT_KEY = "salt_core"
"""Checkpoint dict key for the schema + plan-hash payload (design §2.3, §2.6)."""

_LRS_REQUIRED = ("initial", "max", "end", "pct_start")
_OPTIMIZERS = ("AdamW", "lion", "HybridMuonAdamW")
_WILDCARD_PARTS = frozenset({"*", "**"})
# dataset-boundary demand is declared for the three runtime modes; ONNX
# export feeds the model directly (design §7) and has no dataset plan.
_DEMAND_MODES = (Mode.FIT, Mode.VAL, Mode.TEST)


class SaltModule(lightning.LightningModule):
    """Lightning wrapper around a configured graph-module dict (design §3.4).

    Parameters
    ----------
    modules : dict[str, GraphModule | None]
        Model-side graph modules by instance name (the config dict key —
        assigned to ``module.name`` here, before any ``declare_io``).
        Every module must be an ``nn.Module`` implementing the
        `GraphModule` protocol. ``None`` entries are dropped — the
        assembly-time half of the design §5.3 null-deletion semantics
        (``--model.modules.X=null`` parses to None and is filtered here).
        An un-narrowed `LossSum` is narrowed via `LossSum.collect_loss_keys`
        (the ``losses.**`` framework wildcard, design §3.3).
    lrs_config : Mapping[str, float]
        OneCycleLR schedule config, v1 schema kept wholesale
        (modelwrapper.py:340-381): required keys ``initial``, ``max``,
        ``end``, ``pct_start``; optional ``weight_decay`` (default 1e-5)
        and ``last_epoch`` (default -1).
    optimizer : str, optional
        One of ``"AdamW"`` (default), ``"lion"``, ``"HybridMuonAdamW"``
        (v1 surface minus the muP ``MuAdamW`` swap — M6).
    name : str, optional
        Model name (run naming/metadata), by default ``"salt"``.
    debug : bool, optional
        Run the executor in debug mode (read tracking + in-place mutation
        detection, design §3.2), by default False. Slow — tests only.

    Raises
    ------
    ConfigError
        For an empty module dict, a non-``nn.Module`` module, a bad
        optimizer name, or missing `lrs_config` keys.
    """

    def __init__(
        self,
        modules: dict[str, GraphModule | None],
        lrs_config: Mapping[str, float],
        optimizer: str = "AdamW",
        name: str = "salt",
        debug: bool = False,
    ) -> None:
        super().__init__()
        # assembly-time None filtering (design §5.3): a null entry — from a
        # config-file or CLI override — deletes the module.
        modules = {key: module for key, module in modules.items() if module is not None}
        if not modules:
            raise ConfigError("SaltModule needs a non-empty module dict (design §3.4)")
        for key, module in modules.items():
            if not isinstance(module, nn.Module):
                raise ConfigError(
                    f"module {key!r} ({type(module).__name__}) is not an nn.Module — "
                    "model-side graph modules carry parameters/buffers (design §2.5)"
                )
            # instance names come from the config dict key, BEFORE any
            # declare_io/compile (design §2.2)
            module.name = key
        # narrow the losses.** framework wildcard before any declare_io —
        # the M1 kernel rejects wildcard requires (design §3.3)
        for module in modules.values():
            if isinstance(module, LossSum) and not module.narrowed:
                module.narrow(LossSum.collect_loss_keys(modules, Mode.FIT))
        if missing := [k for k in _LRS_REQUIRED if k not in lrs_config]:
            raise ConfigError(
                f"lrs_config is missing required keys {missing} — the OneCycleLR schema is "
                f"{list(_LRS_REQUIRED)} (+ optional weight_decay, last_epoch; design §3.4)"
            )
        if optimizer not in _OPTIMIZERS:
            raise ConfigError(
                f"optimizer {optimizer!r} is not supported — choose from {list(_OPTIMIZERS)}"
            )
        # resolved config only — the module dict is NOT pickled into hparams
        # (design §3.4; load_from_checkpoint takes modules= explicitly)
        self.save_hyperparameters(logger=False, ignore=["modules"])
        self.name = name
        self.lrs_config = dict(lrs_config)
        self.optimizer = optimizer
        self.debug = debug
        self.net = nn.ModuleDict(modules)  # ckpt keys: net.<name>.* (dict order, not topo)
        self._graph_modules: dict[str, GraphModule] = dict(modules)
        self.plans: dict[Mode, Plan] = {}
        self._executors: dict[Mode, Executor] = {}
        self.schema: ResolvedSchema | None = None
        self._bound = False
        self._materialised = False
        self._loaded_from_checkpoint = False
        self._ckpt_plan_hashes: dict[str, str] = {}

    # -- lifecycle state (read-only — gates and tests assert on these) ---------

    @property
    def bound(self) -> bool:
        """Whether the two-phase bind has run (design §2.3 — exactly once).

        Returns
        -------
        bool
            True after `setup`/`on_load_checkpoint` bound the modules.
        """
        return self._bound

    @property
    def materialised(self) -> bool:
        """Whether this instance ran `materialise_all` (fresh fits only).

        Returns
        -------
        bool
            True after a fresh-fit ``on_fit_start``; stays False on
            checkpoint loads (values arrive via the state_dict).
        """
        return self._materialised

    @property
    def loaded_from_checkpoint(self) -> bool:
        """Whether any checkpoint was loaded into this instance.

        Returns
        -------
        bool
            True once `on_load_checkpoint` ran (resume or
            ``load_from_checkpoint``) — disables `materialise`.
        """
        return self._loaded_from_checkpoint

    # -- static declarations (config-only, design §3.1/§3.3) -------------------

    def sink_demand(self) -> dict[Mode, list[str]]:
        """The per-mode dataset-boundary demand, derived from module declarations.

        For each runtime mode, every non-optional required key that no
        sibling module produces must come from the dataset — these keys are
        the `GraphDataset` sinks that drive demand-narrowed reads and label
        narrowing (design §3.3, §6.1). ``meta.rows`` is added in TEST for
        writer row alignment (design §8). `GraphDataModule.setup` adopts
        this automatically when no explicit sinks were configured.

        Returns
        -------
        dict[Mode, list[str]]
            ``{mode: [dotted keys]}`` for FIT/VAL/TEST, in declaration order.
            A `ConfigError` propagates from `_boundary_demand` on wildcard
            demand keys or keys the dataset boundary cannot serve (a
            deleted/missing model-side producer).
        """
        return {mode: demand for mode, (demand, _) in self._boundary_demand().items()}

    def sink_origins(self) -> dict[Mode, dict[str, str]]:
        """Per-mode demanded-key -> demanding-module description.

        `GraphDataModule` threads the matching mode's map into each stage
        dataset plan so dataset-side errors (schema-invalid label narrowing,
        missing sink producers) and the ``plan_<mode>.txt`` artifacts name
        the demander the user actually configured instead of the planner's
        ``'<sinks>'`` placeholder (design §4.1 attribution). The map is
        deliberately PER MODE (M3-review fix): in TEST a label can be
        demanded by a *writer* while the same key is task-demanded in FIT —
        a merged map would attribute TEST artifacts/errors to the inactive
        FIT demander.

        Returns
        -------
        dict[Mode, dict[str, str]]
            E.g. ``{Mode.FIT: {"labels.jets.flavour_label":
            "'jets_classification' (config:
            model.modules.jets_classification)"}}``.
        """
        return {mode: origins for mode, (_demand, origins) in self._boundary_demand().items()}

    def _boundary_demand(self) -> dict[Mode, tuple[list[str], dict[str, str]]]:
        """Compute per-mode boundary demand plus per-key demander descriptions.

        In TEST, an attached `WriterCallback`'s declared requires extend the
        demand (design §8): writer-demanded dataset-namespace keys (labels,
        masks, ``meta.rows``) keep their demand-gated producers alive — the
        TEST-mode `Labels` narrowing serves exactly what writers demand.

        Returns
        -------
        dict[Mode, tuple[list[str], dict[str, str]]]
            ``{mode: (demanded keys, {key: demander description})}``.

        Raises
        ------
        ConfigError
            On wildcard demand keys, or when an unproduced require lies
            outside the dataset-served namespaces (``inputs``/``masks``/
            ``labels``/``meta``) — that means a model-side producer is
            missing (e.g. deleted by a ``null`` override), and the error
            names the consuming modules and their config addresses rather
            than letting the dataset plan fail with a confusing
            missing-sink error (design §4.1).
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
                if _WILDCARD_PARTS & set(key.split(KEY_SEP)):
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
                    if _WILDCARD_PARTS & set(key.split(KEY_SEP)):
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
            out[mode] = (demand, origins)
        return out

    def _attached_writer(self) -> tuple[Any, Any]:
        """The attached writer callback + reader, if both exist.

        Duck-typed (a callback exposing ``writer_demand``) so the model side
        stays free of a writers import — the `sink_demand` symmetry
        precedent (design §3.4, §8).

        Returns
        -------
        tuple[Any, Any]
            ``(callback, reader)`` or ``(None, None)`` when no writer
            callback (or no datamodule boundary) is attached.
        """
        trainer = self._trainer
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        callback = next(
            (cb for cb in callbacks or [] if callable(getattr(cb, "writer_demand", None))),
            None,
        )
        if callback is None:
            return None, None
        reader = getattr(getattr(trainer, "datamodule", None), "reader", None)
        if reader is None:
            return None, None
        return callback, reader

    def _writer_demand(self) -> dict[str, str] | None:
        """Merged writer-declared TEST demand from an attached `WriterCallback`.

        Returns None when no writer callback (or no datamodule boundary) is
        attached: programmatic ``Trainer.test`` without writers keeps the M2
        anchor-on-all-preds behaviour.

        Returns
        -------
        dict[str, str] | None
            ``{demanded key: demander description}``, or None.
        """
        callback, reader = self._attached_writer()
        if callback is None:
            return None
        return callback.writer_demand(self._graph_modules, reader)

    def _model_sinks(self, mode: Mode, writers: Any = None, reader: Any = None) -> list[str]:
        """The model-plan sink anchors for one mode (design §3.1, §8).

        M5 deferral note (M3 review): FIT/VAL sinks are ``loss.total`` only.
        Design §3.1/§3.4 additionally name the declared requires of
        configured training callbacks (e.g. a metrics callback) as FIT/VAL
        sinks — that wiring is deferred to M5 with the MaskFormer metrics
        that first need it. Until then the shipped `ConfusionMatrix` works
        because tasks publish ``preds.*`` in all modes and stay alive via
        their losses; a future callback demanding a key no task keeps alive
        would be demand-pruned (mirror the TEST ``writers=`` mechanism here
        when porting).

        Parameters
        ----------
        mode : Mode
            A primary mode.
        writers : Any, optional
            An explicit writer callback (the `WriterCallback` duck-typed
            surface: ``writer_demand`` + ``writers``), by default None —
            discovered from the attached trainer. The static tooling
            (`salt.core.cli`) passes the callback it builds from the parsed
            ``writers:`` block so ``salt2 graph`` sees the same TEST sinks
            and dead-preds errors as a real ``salt2 test`` run (M3-review
            fix; design §4.2, §8).
        reader : Any, optional
            The reader prototype matching `writers` (static path only), by
            default None — the attached datamodule's reader.

        Returns
        -------
        list[str]
            ``["loss.total"]`` in FIT/VAL. In TEST with a writer callback
            (attached or passed), the writer-demanded model-produced keys —
            demand-gating proper (design §8). In ONNX with a writer
            callback, the union of the writers' declared manifest ports
            (``WriterCallback.onnx_manifest`` — the M4.5 unified manifest:
            ONE demand mechanism in both output modes, amendment §4); the
            export path's `salt.core.onnx.export.compile_onnx_plan` anchors
            on the same manifest. Without writers (programmatic
            ``Trainer.test``, toy configs), every declared ``preds.*`` key
            in declaration order (the M2 fallback). Note the asymmetry, by
            design: TEST narrowing trips the dead-preds hard error (an
            eval column silently dropped is a bug), ONNX narrowing is
            legitimate (the Athena surface is narrower than eval —
            ``onnx_streams``/``onnx_tasks``).

        Raises
        ------
        ConfigError
            When no module produces the mode's anchor keys, or — TEST with
            writers — when a produced ``preds.*`` key is consumed by no
            writer (the design §4.2/§8 dead-preds hard error: a computed
            prediction would never be persisted).
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
            return ["loss.total"]
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
        if mode is Mode.ONNX:
            if writers is None or reader is None:
                writers, reader = self._attached_writer()
            if writers is not None and callable(getattr(writers, "onnx_manifest", None)):
                manifest = writers.onnx_manifest(self._graph_modules, reader)
                if manifest:
                    return [out.port for out in manifest]
        if not preds:
            raise ConfigError(
                f"no module produces a 'preds.*' key in mode {mode.name} — evaluation plans "
                "anchor on predictions (design §3.1, §3.3)"
            )
        return preds

    # -- compile + two-phase bind (design §2.3, §3.4) ---------------------------

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
        plan = compile_plan(
            self._graph_modules,
            mode,
            sources=unflatten_spec(dict(boundary)),
            sinks=self._model_sinks(mode),
        )
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
        """Compile the stage's plans and run the two-phase bind (design §3.4).

        Lightning calls the datamodule's ``setup`` first, so the per-stage
        `GraphDataset` instances (and their config-derived boundary specs)
        already exist. Binding happens at most once per instance and always
        before any checkpoint state-dict load (Lightning restores after the
        setup hook).

        Raises
        ------
        ConfigError
            For unsupported stages (``validate``/``predict`` are M5+), a
            missing/foreign datamodule, or any compile/bind error.
        """
        if stage not in {"fit", "test"}:
            raise ConfigError(
                f"stage {stage!r} is not supported by SaltModule in M2 — use trainer.fit or "
                "trainer.test (validate/predict entry points are M5+, design §9.5)"
            )
        dm = self._graph_datamodule()
        if stage == "fit":
            self.compile_mode(Mode.FIT, self._boundary(dm.train_dset, "train"))
            self.compile_mode(Mode.VAL, self._boundary(dm.val_dset, "val"))
            self._assert_fit_val_identical()
            check_class_names(self._graph_modules, dm.train_dset.reader)
        else:
            self.compile_mode(Mode.TEST, self._boundary(dm.test_dset, "test"))
            check_class_names(self._graph_modules, dm.test_dset.reader)
        self._ensure_bound()

    def _run_preflights(self) -> None:
        """Fail-fast data-free checks of file-backed `materialise` sources.

        Duck-typed: every graph module exposing a callable ``preflight()``
        (e.g. `Normaliser` — norm-dict existence/streams/values) is checked
        BEFORE any `materialise` writes buffers, so a wrong path/content
        fails with one §4.1-quality `ConfigError` covering all modules
        instead of a per-module mid-materialise crash (M3 leftover; design
        §2.3 — preflights do config I/O only, never bind/data I/O). Called
        from ``on_fit_start`` on fresh fits only — on resume the values
        arrive via the state_dict and the files are never read (the
        `materialise_all` contract). A failing module preflight propagates
        as `ConfigError`.
        """
        for module in self._graph_modules.values():
            preflight = getattr(module, "preflight", None)
            if callable(preflight):
                preflight()

    def _assert_fit_val_identical(self) -> None:
        """Assert the VAL plan is structurally identical to the FIT plan (design §3.4).

        The plan hash is deliberately mode-independent (`_plan_hash`), so
        the §3.4 'default-identical plan, asserted' contract is a cheap hash
        comparison. Modules that genuinely declare VAL-divergent ports
        (parameterised training) are an M6 feature — until a named exemption
        mechanism exists, divergence is a config error.

        Raises
        ------
        ConfigError
            When the FIT and VAL plans differ structurally.
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
        """The attached `GraphDataModule` (model plans need its boundary).

        Returns
        -------
        GraphDataModule
            The trainer's datamodule.

        Raises
        ------
        ConfigError
            When no trainer/datamodule is attached, or it is not a
            `GraphDataModule`.
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
        """A stage dataset's model-visible boundary specs.

        Returns
        -------
        dict[str, TensorSpec]
            The flat boundary spec mapping.

        Raises
        ------
        ConfigError
            If the stage dataset was never built.
        """
        if dset is None:
            raise ConfigError(
                f"the datamodule has no {stage} dataset — its setup did not run or the "
                f"{stage}_file is unset (design §6.1)"
            )
        return dset.boundary_specs()

    def _ensure_bound(self) -> None:
        """Resolve the bind schema from the compiled plans and bind once.

        Raises
        ------
        ConfigError
            If no plan was compiled yet.
        """
        if self._bound:
            return
        if not self.plans:
            raise ConfigError("bind before any compiled plan — call setup/compile_mode first")
        self._bind(resolve_bind_schema(self.plans.values()))

    def _bind(self, schema: ResolvedSchema) -> None:
        """Bind all modules to a resolved schema — exactly once (design §2.3).

        Raises
        ------
        ConfigError
            On a second bind: rebinding re-initialises width-dependent
            layers and would silently discard loaded values.
        """
        if self._bound:
            raise ConfigError(
                "SaltModule modules are already bound — bind happens exactly once, before any "
                "state-dict load (design §2.3)"
            )
        bind_all(self._graph_modules, schema)
        self.schema = schema
        self._bound = True

    # -- materialise (fresh fits only, design §2.3) -----------------------------

    def on_fit_start(self) -> None:
        """Materialise file-backed values before the first step of a FRESH fit.

        Runs after checkpoint restore: when any checkpoint was loaded
        (resume or `load_from_checkpoint`) materialise is skipped and the
        values come from the state_dict — `Normaliser.forward` raises if a
        checkpoint lacked them (required-present-on-resume, design §2.3).
        """
        if self._materialised or self._loaded_from_checkpoint:
            return
        self._run_preflights()
        materialise_all(self._graph_modules)
        self._materialised = True

    # -- steps (design §3.4) ----------------------------------------------------

    def forward(self, batch: Mapping[str, Any] | Bundle, mode: Mode = Mode.TEST) -> Bundle:
        """Run the compiled plan for `mode` over a batch.

        Parameters
        ----------
        batch : Mapping[str, Any] | Bundle
            The nested batch dict from `GraphDataset` (adopted into a fresh
            `Bundle`), or an existing bundle.

        Returns
        -------
        Bundle
            The bundle with every step's produces merged in (write-once).

        Raises
        ------
        ConfigError
            If no plan was compiled for `mode`.
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
        """One FIT step: execute the plan, log losses, return loss + bundle.

        Returns
        -------
        dict[str, Any]
            ``{"loss": total, "bundle": Bundle}`` (design §3.4 — v2
            callbacks read the bundle; hidden contract #8 is broken
            consciously).

        Raises
        ------
        RuntimeError
            If the total loss is NaN (v1 guard, modelwrapper.py:291-295).
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
        """One VAL step (mirrors `training_step` with the VAL plan).

        Returns
        -------
        dict[str, Any]
            ``{"loss": total, "bundle": Bundle}``.
        """
        del batch_idx
        bundle = self(batch, Mode.VAL)
        self._log_losses(bundle, stage="val")
        return {"loss": bundle.get("loss.total"), "bundle": bundle}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Bundle:
        """One TEST step: run the TEST plan and return the bundle.

        Returns
        -------
        Bundle
            The executed bundle (``preds.*`` + ``meta.rows`` for writers,
            design §8).
        """
        del batch_idx
        return self(batch, Mode.TEST)

    def _log_losses(self, bundle: Bundle, stage: str) -> None:
        """Log ``{stage}/loss`` and ``{stage}/{task}_loss`` (v1 monitor names).

        ``prog_bar=True`` (v1 parity): without it a fit with the default
        ``logger: false`` shows NO loss feedback at all — an added aux task
        would train invisibly (stage-E ergonomics finding).
        """
        sync_dist = self._trainer is not None and len(self.trainer.device_ids) > 1
        self.log(f"{stage}/loss", bundle.get("loss.total"), sync_dist=sync_dist, prog_bar=True)
        for task, value in bundle.subtree("losses").items():
            self.log(f"{stage}/{task}_loss", value, sync_dist=sync_dist, prog_bar=True)

    # -- optimizer (v1 port, modelwrapper.py:340-381) ----------------------------

    def _get_optimizer_class(self) -> type[Optimizer]:
        """Resolve the optimizer class (v1 surface; muP's MuAdamW swap is M6).

        Returns
        -------
        type[Optimizer]
            The optimizer class to instantiate.

        Raises
        ------
        ImportError
            If lion is requested but lion-pytorch is not installed.
        ConfigError
            For unsupported names (also rejected at construction).
        """
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
        """Build the optimizer + OneCycleLR scheduler (v1 kept wholesale).

        Returns
        -------
        tuple[list[Optimizer], list[dict]]
            One optimizer and one step-interval scheduler config
            (modelwrapper.py:340-381).
        """
        optimizer_class = self._get_optimizer_class()
        optimizer_kwargs = {
            "lr": self.lrs_config["initial"],
            "weight_decay": self.lrs_config.get("weight_decay", 1e-5),
        }
        if optimizer_class is HybridMuonAdamW:
            opt = optimizer_class(self.named_parameters(), **optimizer_kwargs)
        else:
            opt = optimizer_class(self.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
            last_epoch=int(self.lrs_config.get("last_epoch", -1)),
            cycle_momentum=optimizer_class is not HybridMuonAdamW,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    # -- checkpoints: schema + plan hashes, never Plan objects (design §2.3) -----

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Serialise the resolved schema and per-mode plan hashes (design §2.3).

        Plans hold ``MappingProxyType`` views and live module references —
        they are deliberately NOT picklable; the hash is the integrity
        check, and the schema alone suffices to bind on a data-less machine.

        Design-deviation note: §2.3 says "serialised into checkpoint
        hparams"; the payload lives under the TOP-LEVEL checkpoint key
        `CKPT_KEY` (``'salt_core'``) instead, keeping
        ``checkpoint['hyper_parameters']`` purely user config. Functionally
        equivalent — round-trip covered by ``test_saltmodule.py``.
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
        """Verify plan hashes and (if needed) bind from the stored schema.

        Runs BEFORE the state-dict load on both restore paths (trainer
        ``ckpt_path`` and ``load_from_checkpoint``). Hash policy per design
        risk 9: FIT mismatches are fatal, other modes warn. When the modules
        are not yet bound (the data-less ``load_from_checkpoint`` path,
        where ``setup`` never ran) they are bound from the checkpoint's
        schema so the subsequent strict load finds every key (design §2.3).
        Also marks the instance as checkpoint-loaded, which disables
        `materialise` (values arrive via the state_dict). A FIT plan-hash
        mismatch propagates as `ConfigError` from `_verify_ckpt_hash`.
        """
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
        """Compare a compiled plan's hash with the checkpoint's stored hash.

        Raises
        ------
        ConfigError
            On a FIT mismatch (design risk 9); other modes warn and
            continue.
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
    """Build the TEST dead-preds hard error at the design §4.2 quality bar.

    Names, per dead key, the producing task module and its config address;
    attributes the culprit when a configured writer's explicit ``streams``
    list excludes the dead streams (the §4.2 worked-example attribution,
    M3-review fix); and spells out the null-deletion workaround for
    train-only aux tasks (the per-task ``expose:`` opt-out of design §4.2 is
    an M5 deferral and deliberately NOT advertised here).

    Parameters
    ----------
    dead : list[str]
        The produced ``preds.*`` keys no writer consumes.
    produced : Mapping[str, str]
        Produced key -> producing module instance name.
    writers : Any
        The writer callback (duck-typed ``writers`` mapping for the
        streams-narrowing hint; absent attributes degrade gracefully).

    Returns
    -------
    str
        The multi-line error message.
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
    dead_streams = {
        parts[1]
        for key in dead
        if len(parts := key.split(KEY_SEP)) > 2  # preds.<stream>.<task>
    }
    hints = [
        f"writers.modules.{wname}.init_args.streams: {list(streams)} currently excludes {excluded}"
        for wname, writer in (getattr(writers, "writers", None) or {}).items()
        if (streams := getattr(writer, "streams", None)) is not None
        and (excluded := sorted(dead_streams - set(streams)))
    ]
    fix = "  fix: widen the writers"
    if hints:
        fix += " — " + "; ".join(hints) + " —"
    targets = sorted({produced[key] for key in dead if key in produced})
    if targets:
        null_form = " / ".join(f"--model.modules.{name}=null" for name in targets)
        fix += f", or remove the task module ({null_form})"
    else:
        fix += ", or remove the task module"
    lines.append(fix)
    return "\n".join(lines)


def check_class_names(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Cross-check configured ``class_names`` against schema label attrs (design §2.6, §4.1).

    Default-on whenever both sides exist: for every module declaring
    ``class_names`` + ``stream`` + ``label`` (the `ClassificationTaskModule`
    surface, duck-typed so user task modules participate too), the reader's
    schema artifact is consulted — if the stream's group carries an attr
    named like the label whose value is a list of strings (the
    umami-preprocessing convention, e.g. the ``flavour_label`` attr on
    ``jets``), the configured list must match in SET **and ORDER**. A
    reordered ``class_names`` is a silent physics mislabeling no shape check
    can catch — v1 made it impossible by construction (cli.py:418-443), v2
    by this validation. Runs at `SaltModule.setup` (every fit/test) and in
    ``salt2 graph validate`` over §5.1 configs. No-op without a schema
    artifact (the §2.6 opt-out warning already covers that case).

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


def bundle_as_v1_outputs(bundle: Bundle) -> dict[str, Any]:
    """Migration shim: expose a bundle as the v1 ``{preds, labels, pad_masks}`` view.

    Grafted from ordered-pipeline (design §3.4) so unported v1 callbacks
    keep working during migration; deprecated, removed at M7. ``preds`` and
    ``labels`` keep their nested ``{stream: {name: tensor}}`` shape;
    ``pad_masks`` maps each masked stream (``masks.*`` minus the encoder's
    ``registers`` entry) to its True-is-padded mask.

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
