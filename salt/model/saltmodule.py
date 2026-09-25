"""`SaltModule` — the LightningModule owning the configured graph-module dict.

Compiles per-mode execution plans against the datamodule's dataset boundary,
binds modules to the resolved schema exactly once, and drives the executor
from the Lightning step hooks.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import lightning
from torch import nn
from torch.optim import Optimizer

from salt.data.datamodule import SaltDataModule
from salt.data.dataset import SaltDataset
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.executor import Executor
from salt.graph.planner import Plan, compile_plan
from salt.graph.spec import Mode, SinkModule, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.model.bind import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    reader_stream_datasets,
    resolve_bind_schema,
)
from salt.model.checkpoint import reject_v1_and_strip_orig_mod, warm_start_from_checkpoint
from salt.model.modules.losses import LossGLS, LossSum
from salt.model.multistage_training import TrainingController
from salt.model.mup import apply_mup_shapes, validate_mup_routing
from salt.model.sink_prep import (
    PreparedSinks,
    assert_no_dead_preds,
    boundary_demand,
    is_section_sink,
    prepare_sinks,
)
from salt.model.validation import check_class_names, resolve_origin_weighting, validate_edge_port
from salt.optim import (
    build_onecycle_scheduler,
    build_optimizer,
    build_stage_lr_scheduler,
    preflight_lr_schedulers,
    resolve_optimizer_class,
)
from salt.schedule import TrainingSchedule, clear_frozen_grads, trainable_named_params
from salt.utils.logging import get_logger

__all__ = [
    "CKPT_KEY",
    "SaltModule",
]

CKPT_KEY = "salt_core"
"""Checkpoint dict key for the schema + plan-hash payload."""

_LOG = get_logger(__name__)

_LRS_REQUIRED = ("initial", "max", "end", "pct_start")
_OPTIMIZERS = ("AdamW", "lion", "lion-pytorch", "HybridMuonAdamW")


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
        self.mup_cfg: dict[str, Any] | None = validate_mup_routing(mup, modules)
        # validated against model-module names NOW, before outputs: writers fold in.
        # The schedule is the single home for optimizer/LR config: no-schedule
        # configs desugar to one `fit` stage so configure_optimizers has ONE path.
        schedule: TrainingSchedule = (
            TrainingSchedule.from_config(training_schedule, tuple(modules))
            if training_schedule is not None
            else TrainingSchedule.desugar_legacy(tuple(modules))
        )
        validate_edge_port(modules)
        # resolved config only — the module dict is NOT pickled into hparams
        # (load_from_checkpoint takes modules= explicitly)
        self.save_hyperparameters(logger=False, ignore=["modules"])
        self.name = name
        self.lrs = dict(lrs)
        self.optimizer = optimizer
        self.debug = debug
        self.net = nn.ModuleDict(modules)  # ckpt keys: net.<name>.* (dict order, not topo)
        # not an nn.Module: owns no params, so this adds no checkpoint keys
        self.training_controller = TrainingController(schedule, self.net, self.lrs, self.optimizer)
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
        sinks = {key: w for key, w in section.items() if is_section_sink(w)}
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
        return {
            mode: demand
            for mode, (demand, _) in boundary_demand(
                self._graph_modules, self._sink_prep(())
            ).items()
        }

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
        return {
            mode: origins
            for mode, (_demand, origins) in boundary_demand(
                self._graph_modules, self._sink_prep(())
            ).items()
        }

    def _reachable_sinks(self) -> list[Any]:
        """Every sink this model can reach: the trainer registry plus its own section
        sinks (programmatic ``SaltModule(outputs=...)`` has no trainer to register on).
        """
        from salt.outputs.sinks.registry import iter_sinks

        found = list(iter_sinks(self._trainer))
        for sink in self._section_sinks:
            if not any(seen is sink for seen in found):
                found.append(sink)
        return found

    def _sink_prep(
        self,
        modes: Sequence[Mode] = (Mode.FIT, Mode.VAL, Mode.TEST),
        *,
        writer_demand: bool = True,
    ) -> PreparedSinks:
        """Run the shared sink preparation for `modes` with the trainer-side inputs
        (``()`` = selection, binding and boundary inputs only, no anchors).
        """
        trainer = self._trainer
        return prepare_sinks(
            self._graph_modules,
            self._reachable_sinks(),
            output_section=self._output_section,
            reader=getattr(getattr(trainer, "datamodule", None), "reader", None),
            callbacks=getattr(trainer, "callbacks", None) if trainer is not None else None,
            modes=modes,
            writer_demand=writer_demand,
        )

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
            mismatches; ONNX raises `ConfigError` (`compile_mode` prepares
            FIT/VAL/TEST only — `salt export` compiles ONNX separately).
        """
        if mode not in {Mode.FIT, Mode.VAL, Mode.TEST}:
            raise ConfigError(
                "compile_mode prepares FIT/VAL/TEST plans — ONNX plans are compiled by salt export "
                "(compile_onnx_plan)"
            )
        prep = self._sink_prep((mode,), writer_demand=False)
        # the TEST sink node is folded into the planning dict (renders its own card,
        # anchors demand via its declared requires); in FIT/VAL its declare_io is
        # empty, so the planner collects it as inactive and plan_hash is unchanged.
        plan = compile_plan(
            prep.modules,
            mode,
            sources=unflatten_spec(dict(boundary)),
            sinks=prep.by_mode[mode].anchors,
        )
        if mode is Mode.TEST and prep.test_sink is not None:
            assert_no_dead_preds(self._graph_modules, plan)
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
        dm = self._graph_datamodule()
        if stage == "fit":
            self.compile_mode(Mode.FIT, self._boundary(dm.train_dset, "train"))
            self.compile_mode(Mode.VAL, self._boundary(dm.val_dset, "val"))
            self._assert_fit_val_identical()
            reader = dm.train_dset.reader
            check_class_names(self._graph_modules, reader)
            resolve_origin_weighting(self._graph_modules, reader)
        else:
            self.compile_mode(Mode.TEST, self._boundary(dm.test_dset, "test"))
            reader = dm.test_dset.reader
            check_class_names(self._graph_modules, reader)
            resolve_origin_weighting(self._graph_modules, reader)
        # the reader's stream->dataset map rides on the schema so norm/class-dict
        # lookups resolve by dataset name, not the raw config stream name
        self._ensure_bound(reader_stream_datasets(reader))
        # --init_from: warm start weights AFTER bind (params now carry their
        # resolved shapes) and BEFORE optimizer construction / the first step
        # (both are strictly later in the Lightning fit sequence). Fit-only —
        # `test` never warm-starts.
        if stage == "fit" and self._init_from is not None:
            result = warm_start_from_checkpoint(
                self,
                self._init_from,
                config_modules=self._graph_modules,
                plans=self.plans,
                bound=self._bound,
                payload_key=CKPT_KEY,
            )
            self._init_warm_started = True
            self._init_loaded_modules = set(result.loaded)
        # training_schedule: reset to stage 0 and apply its freeze mask AFTER any
        # warm start (freeze composes on top of the loaded weights) and BEFORE
        # optimizer construction, so the initial `configure_optimizers` (built by
        # Lightning's strategy.setup, after this) sees stage 0's requires_grad
        # mask. Later stage boundaries are driven by the
        # `TrainingScheduleCallback`. Fit-only.
        if stage == "fit":
            self.training_controller.preflight(self._trainer)
            preflight_lr_schedulers(self.training_controller.schedule)
            self.training_controller.start_fit(self._trainer)
        # --compile LAST: dynamo traces each graph module in its final setup-time
        # state — after bind/materialise have resolved widths, after any
        # --init_from warm start has written weights, and after stage 0's freeze
        # mask is in place.
        self._apply_compile()

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

    def _run_preflights(self, modules: Mapping[str, SaltModelModule] | None = None) -> None:
        """Fail-fast data-free checks of file-backed `materialise` sources: every
        module exposing a callable ``preflight()`` (e.g. `Normaliser`) is
        checked before any `materialise` writes buffers, so a bad path/content
        raises one `ConfigError` instead of a per-module mid-materialise crash.
        Called from ``on_fit_start`` on fresh fits only. On an ``--init_from``
        warm start only the to-be-materialised (checkpoint-uncovered) modules
        are passed — a retained module's file need not exist on this machine.
        The duck-typed contract is ``preflight(datasets=None)``: bound modules
        resolve dict keys through the map captured at `bind`; ``salt graph
        validate`` passes the reader map explicitly because it never binds.
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

    def _ensure_bound(self, datasets: Mapping[str, str] | None = None) -> None:
        """Resolve the bind schema from the compiled plans and bind once; raises
        `ConfigError` if no plan was compiled yet. `datasets` is the reader's
        stream->dataset map (empty when bound without a reader).
        """
        if self._bound:
            return
        if not self.plans:
            raise ConfigError("bind before any compiled plan — call setup/compile_mode first")
        self._bind(resolve_bind_schema(self.plans.values(), datasets=datasets))

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
        apply_mup_shapes(self.net, self.mup_cfg)
        self.schema = schema
        self._bound = True

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
        frozen = self.training_controller.frozen_module_names
        if mode and frozen:
            for name in frozen:
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
        controller = self.training_controller
        if controller.reducer_safe_freeze and controller.frozen_module_names:
            clear_frozen_grads(self.net, controller.frozen_module_names)

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

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        """Build the active stage's optimizer (over TRAINABLE params only — frozen
        modules are excluded entirely) + its LR scheduler. Re-invoked by the
        `TrainingScheduleCallback` at each stage boundary via
        ``trainer.strategy.setup_optimizers``. The scheduler is the default
        step-interval OneCycleLR over the stage's step allocation, unless the stage
        declares an `lr_scheduler` — then that class is instantiated over
        the freshly-built optimizer instead.
        """
        controller = self.training_controller
        lrs, optimizer_name = controller.active_optim_config()
        optimizer_class = resolve_optimizer_class(optimizer_name, self.mup_cfg)
        # Optimizer owns only the active stage's TRAINABLE params: those with
        # requires_grad NOT under a frozen module's prefix. Keying off the frozen
        # SET (not just requires_grad) also excludes a reducer-safe frozen module,
        # whose params keep requires_grad=True — so weight decay never touches a
        # frozen param either. In the default freeze mode this selects the exact
        # same set as the old requires_grad filter (bitwise parity).
        named_trainable = trainable_named_params(
            self.named_parameters(), controller.frozen_module_names
        )
        opt = build_optimizer(optimizer_class, named_trainable, lrs)
        stage = controller.schedule.stages[controller.current_stage_index]
        if stage.lr_scheduler is not None:
            return [opt], [build_stage_lr_scheduler(opt, stage.lr_scheduler)]
        total_steps = int(controller.stage_total_steps(self.trainer))
        return [opt], [build_onecycle_scheduler(opt, lrs, total_steps)]

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
            # schema.datasets is intentionally NOT serialised — materialise is
            # skipped on checkpoint load, so the stream->dataset map is never
            # needed again once the buffers are restored from the state_dict
            "schema": {
                "widths": dict(self.schema.widths),
                "fields": {key: list(val) for key, val in self.schema.fields.items()},
            },
            "plan_hashes": {mode.name: plan.plan_hash for mode, plan in self.plans.items()},
            "schedule": self.training_controller.checkpoint_state(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Runs before the state-dict load on both restore paths: verifies plan
        hashes (fatal on FIT mismatch via `_verify_ckpt_hash`), binds from the
        checkpoint's stored schema if not yet bound, strips a ``--compile``-added
        ``_orig_mod.`` state_dict prefix, rejects the v1 (``ModelWrapper``)
        state-dict layout with `ConfigError` (via
        `salt.model.checkpoint.reject_v1_and_strip_orig_mod`), and (on a fit resume
        of a multi-stage schedule) restores the saved stage index + freeze mask
        BEFORE the optimizer is rebuilt (see
        `TrainingController.restore_schedule_stage`). Marks the instance
        checkpoint-loaded (disables `materialise`).
        """
        state_dict = checkpoint.get("state_dict")
        cleaned = reject_v1_and_strip_orig_mod(state_dict)
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
        self.training_controller.restore_schedule_stage(payload.get("schedule"), self._trainer)

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
