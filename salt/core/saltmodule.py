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
from salt.core.nn.modules import LossGLS, LossSum
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

# the attention backends an EdgeAttention encoder MAY declare without it being a
# silent bypass: only the deterministic raw-torch backend. v1's EncoderLayer
# skips backend assignment entirely when edge_embed_dim>0 (transformer.py:
# 599-601,616-620), so any flash/varlen/efficient backend declared ALONGSIDE an
# edge port was silently ignored in v1 — ED2 makes that a NAMED error.
_EDGE_OK_BACKENDS = frozenset({"torch-math"})

CKPT_KEY = "salt_core"
"""Checkpoint dict key for the schema + plan-hash payload (design §2.3, §2.6)."""

_LRS_REQUIRED = ("initial", "max", "end", "pct_start")
_OPTIMIZERS = ("AdamW", "lion", "HybridMuonAdamW")
_MUP_KEYS = frozenset({"apply_to", "shape_path"})
_WILDCARD_PARTS = frozenset({"*", "**"})
# dataset-boundary demand is declared for the three runtime modes; ONNX
# export feeds the model directly (design §7) and has no dataset plan.
_DEMAND_MODES = (Mode.FIT, Mode.VAL, Mode.TEST)


def _is_test_persistence_sink(callback: Any) -> bool:
    """Whether a ``writer_demand``-exposing callback is the TEST persistence sink (W2 B2).

    The ORDER-INDEPENDENT discriminator that keeps an ONNX-only sink
    (`OnnxExportSink`, whose ``declare_io(Mode.TEST)`` is empty) from being chosen
    as the TEST writer/sink-node. A `_SinkCallback` advertises ``is_test_sink()``
    (True for `H5OutputSink`, False for `OnnxExportSink`); a plain duck-typed sink
    without that method (e.g. `CollectOutputs`) is treated as a TEST sink so the
    legacy flat-``<sinks>`` path is unchanged.

    Returns
    -------
    bool
        True when the callback should be selected as the TEST persistence sink.
    """
    is_test_sink = getattr(callback, "is_test_sink", None)
    return True if not callable(is_test_sink) else bool(is_test_sink())


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
    lrs : Mapping[str, float]
        OneCycleLR schedule config (design §5.1; the v1 ``lrs_config`` kwarg
        renamed to ``lrs`` — M3 cleanup, M5 sub-wave D): required keys
        ``initial``, ``max``, ``end``, ``pct_start``; optional ``weight_decay``
        (default 1e-5) and ``last_epoch`` (default -1).
    optimizer : str, optional
        One of ``"AdamW"`` (default), ``"lion"``, ``"HybridMuonAdamW"``. When
        `mup` is configured the optimizer is swapped to ``mup.optim.MuAdamW``
        regardless of this name (the muP routing half — design §3.4; v1
        ``MuAdamW`` swap, modelwrapper.py muP path). Other names raise.
    mup : Mapping[str, Any], optional
        The muP ROUTING config (M6 sub-wave B; design §3.4 line 685 —
        KEEP-architecture/BREAK-routing). ``None`` (default) = no muP.
        Keys:

        - ``apply_to`` (required): an EXPLICIT list of module instance names
          (NOT a regex — the design break from v1's ``apply_to``/
          ``parameter_name`` regex zip, configuration_muP.py:98-115) that
          carry ``mup: true``. Each name must exist in `modules` AND its
          module must accept a ``mup`` init_arg (carry a ``mup`` attribute);
          else a `ConfigError`. A module with ``mup`` truthy that is NOT in
          ``apply_to`` is a WARNING (silent muP-off would change training
          dynamics).
        - ``shape_path`` (optional): the base-shapes file produced by
          ``salt2 mup-shapes`` / ``setup_mup``. Applied with
          ``mup.set_base_shapes`` over the ``apply_to`` submodules at bind so
          `MuReadout.width_mult()` resolves against the real base widths (the
          routing-stage shape application). When absent, the architectural
          default (``width_mult == 1.0``, set on each module at construction)
          stands — a width-1 model that still trains correctly under muP.
    name : str, optional
        Model name (run naming/metadata), by default ``"salt"``.
    debug : bool, optional
        Run the executor in debug mode (read tracking + in-place mutation
        detection, design §3.2), by default False. Slow — tests only.

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
            # GLS does not utilise task weights: the v2 home of v1's ctor guard
            # (modelwrapper.py:139-142) — fail loudly here, before declare_io
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
        # the muP ROUTING half (design §3.4): validate apply_to against the
        # module dict (each name exists + carries a `mup` init_arg) and warn
        # on a mup-on module left out of apply_to — see _validate_mup.
        self.mup_cfg: dict[str, Any] | None = _validate_mup(mup, modules)
        # the edge bind-time validators (FD §6.7 1425-1431, M6 sub-wave C):
        # edge-stream-first + EdgeAttention-backend forcing — the named-error
        # replacements for v1's silent sort-first hack (saltmodel.py:66-73) and
        # flash bypass (transformer.py:599-601). No-op without an edge encoder.
        _validate_edge_port(modules)
        # resolved config only — the module dict is NOT pickled into hparams
        # (design §3.4; load_from_checkpoint takes modules= explicitly)
        self.save_hyperparameters(logger=False, ignore=["modules"])
        self.name = name
        self.lrs = dict(lrs)
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
            elif mode & Mode.TRAINING:
                # FIT/VAL callback demand (design §3.1 454-456, §3.4 667-671 —
                # the TRAINING mirror of the TEST writer block above): a metrics
                # callback's dataset-namespace requires (labels/masks/meta) that
                # no task already demands extend the boundary so their producers
                # survive. Model-produced (preds.*) callback keys are anchored by
                # `_model_sinks`, not here (they are already `in produced`).
                for key, who in self._callback_demand(mode).items():
                    if key in produced or key in required:
                        continue  # a model-plan sink / already task-demanded
                    if _WILDCARD_PARTS & set(key.split(KEY_SEP)):
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
        """The attached TEST writer/sink callback + reader, if both exist.

        Duck-typed (a callback exposing ``writer_demand``) so the model side
        stays free of a writers import — the `sink_demand` symmetry
        precedent (design §3.4, §8).

        ORDER-INDEPENDENT TEST-sink selection (plan 29 W2 B2): an ONNX-only sink
        (`OnnxExportSink`) ALSO exposes ``writer_demand`` but its
        ``declare_io(Mode.TEST)`` is EMPTY — it must NEVER be chosen as the TEST
        persistence sink (choosing it would empty the TEST sinks and trip
        ``_assert_no_dead_preds`` on EVERY ``preds.*``, an order-dependent
        ``salt2 test`` crash with ``callbacks: [onnx_export, h5_output]``). The
        ``is_test_sink()`` discriminator (`_SinkCallback`; True for `H5OutputSink`,
        False for `OnnxExportSink`) skips it regardless of callback order —
        symmetric to the static ``cli.py`` ``_static_writer_sink_callback``
        hardening. A plain duck-typed sink without ``is_test_sink`` (e.g.
        `CollectOutputs`) is treated as a TEST sink (legacy behaviour preserved).

        Returns
        -------
        tuple[Any, Any]
            ``(callback, reader)`` or ``(None, None)`` when no TEST writer
            callback (or no datamodule boundary) is attached.
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
        """The attached TEST sink NODE (plan 29 W1), if one is wired as a callback.

        A sink node is a `GraphModule` (``name`` + ``declare_io``) that also marks
        itself a terminal sink via ``is_sink() -> True`` (the `SinkModule` marker,
        e.g. `H5OutputSink`). When wired at ``callbacks:`` (the cutover config),
        `_attached_writer` already finds it by its ``writer_demand`` surface; this
        sibling additionally checks it is a renderable node so `compile_mode` can
        FOLD it into the planning module dict (it then renders its OWN card and
        anchors demand via its declared requires, not the flat ``<sinks>``
        sentinel). The duck-typed ``writer_demand`` discovery keeps working in
        parallel.

        Returns
        -------
        GraphModule | None
            The sink node, or None when no sink-node callback is attached.
        """
        callback, _reader = self._attached_writer()
        if callback is None:
            return None
        is_sink = getattr(callback, "is_sink", None)
        has_io = callable(getattr(callback, "declare_io", None))
        if has_io and callable(is_sink) and bool(is_sink()):
            return callback
        return None

    @staticmethod
    def _fold_sink_node(
        modules: dict[str, GraphModule], sink_node: GraphModule
    ) -> dict[str, GraphModule]:
        """Add a sink NODE to a planning module dict under its instance name (plan 29 W1).

        The planner requires ``module.name == config_key``; the sink's ``name``
        (the config dict key when wired at ``callbacks:``, else its class default)
        is used as the key. A name collision with a model module is a config error
        (instance names are unique across the pipeline graph).

        Returns
        -------
        dict[str, GraphModule]
            A fresh dict including the sink node.

        Raises
        ------
        ConfigError
            If the sink node's name collides with an existing model module.
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
        """Restore the TEST dead-preds hard error on the folded-sink path (plan 29 W1).

        With a sink NODE folded (`compile_mode`), the flat `_model_sinks` writer
        dead-preds gate (`saltmodule.py` TEST branch) is bypassed — the sink
        demands ``outputs.*``, not ``preds.*``. Under the locked design the
        conversion PRODUCERS consume each persisted prediction's ``preds.*``, so
        a genuinely-dead ``preds.*`` (one the model computes every TEST batch but
        NO producer feeds into a demanded ``outputs.*`` leaf) would otherwise
        ship silently — a regression vs the M4.5 `WriterCallback` hard error and
        the eval-safety bug the gate exists to catch (a computed prediction that
        is never persisted).

        This re-establishes that net on the runtime path at `salt2 test` parity
        with M4.5: every ``preds.*`` the model produces (active in TEST) must be
        CONSUMED by some edge in the compiled plan (i.e. by a surviving
        conversion producer that reaches the sink). A produced ``preds.*`` absent
        from the plan's consumed keys is dead -> the existing
        `_dead_preds_message` hard error (the `salt2 graph validate --strict`
        finding, raised here so the eval flow keeps parity with M4.5).

        Parameters
        ----------
        plan : Plan
            The compiled TEST plan (with the sink folded in).

        Raises
        ------
        ConfigError
            When a produced ``preds.*`` key is consumed by no plan edge.
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

    def _attached_callbacks(self) -> list[Any]:
        """Attached callbacks declaring FIT/VAL plan sinks (design §3.1, §3.4).

        Duck-typed (any callback exposing a callable ``fit_val_demand``) so
        the model side stays free of a callbacks import — the SAME demand
        symmetry the writer surface uses (`_attached_writer`), now for the
        TRAINING-mode metrics family (e.g. `ConfusionMatrix`, the M5
        `MaskformerMetrics`). The surface is STATIC (config-only, taking the
        model-module dict): it does not depend on the callback's ``setup``
        having run, so the static `salt2 graph` tooling sees the same FIT/VAL
        sinks a real ``trainer.fit`` does.

        Returns
        -------
        list[Any]
            The attached FIT/VAL-sink callbacks in trainer order (empty when
            none, or when no trainer is attached).
        """
        trainer = self._trainer
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        return [cb for cb in callbacks or [] if callable(getattr(cb, "fit_val_demand", None))]

    def _callback_demand(self, mode: Mode, callbacks: Any = None) -> dict[str, str]:
        """Merged callback-declared FIT/VAL demand (design §3.1 454-456, §3.4 667-671).

        The TRAINING-mode mirror of `_writer_demand` (TEST): configured
        metrics-family callbacks DECLARE the bundle keys they read each VAL
        epoch (e.g. a `preds.<stream>.<task>` no loss anchors), and those
        keys become FIT/VAL plan sinks so their producers survive demand
        pruning. Empty outside TRAINING modes (callbacks declare FIT/VAL
        requires only; TEST/ONNX sinks are the writer manifest).

        Parameters
        ----------
        mode : Mode
            A primary mode. Non-TRAINING modes return ``{}``.
        callbacks : Any, optional
            An explicit iterable of FIT/VAL-sink callbacks (the static
            `salt.core.cli` path passes the callbacks it parses from the
            ``trainer.callbacks`` block so ``salt2 graph`` sees the same
            FIT/VAL sinks as a real ``salt2 fit`` run — the writer
            ``writers=`` precedent). None → discovered from the attached
            trainer.

        Returns
        -------
        dict[str, str]
            ``{demanded key: demander description}`` in callback (config)
            order, the values §4.1-grade attributions.
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
        """The model-plan sink anchors for one mode (design §3.1, §8).

        FIT/VAL sinks (M5, D-prereq): ``loss.total`` PLUS the declared
        requires of any configured training callback (design §3.1 454-456,
        §3.4 667-671) — the TRAINING-mode mirror of the TEST writer-demand
        mechanism. A metrics-family callback DECLARES the bundle keys it
        reads each VAL epoch (`_callback_demand`); those keys anchor the
        FIT/VAL plan so a callback-consumed ``preds.*`` key NO loss keeps
        alive (the M5 `MaskformerMetrics` shape) survives demand pruning.
        The shipped `ConfusionMatrix` already worked because tasks publish
        ``preds.*`` in all modes and stay alive via their losses; this
        wiring extends sink coverage to callbacks demanding a key no task
        anchors. Unlike TEST, an unconsumed ``preds.*`` in FIT/VAL is NOT a
        dead-preds error (the normal no-metric-callback case, design §3.3):
        callback demand only ADDS sinks, never narrows.

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
        callbacks : Any, optional
            An explicit iterable of FIT/VAL-sink callbacks (the static
            `salt.core.cli` path passes the callbacks it parses from the
            ``trainer.callbacks`` block — the ``writers=`` precedent), by
            default None — discovered from the attached trainer. Consulted
            in TRAINING modes only.

        Returns
        -------
        list[str]
            ``["loss.total"]`` plus the callback-declared FIT/VAL demand
            keys in FIT/VAL. In TEST with a writer callback
            (attached or passed), the writer-demanded model-produced keys —
            demand-gating proper (design §8). In ONNX the manifest is NO
            LONGER writer-derived (plan-29 W4): the folded `OnnxExportSink`
            terminal node anchors its own conversion-leaf demand once folded
            into the planning module dict (`cli.py` / `export.py`), so this
            method has no ONNX writer branch — it falls through to every
            declared ``preds.*`` key in declaration order (the M2 fallback)
            when no export-sink demand is present.

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
        # plan-29 W4: the ONNX output manifest is no longer writer-derived — the
        # folded OnnxExportSink (a terminal node in `_graph_modules` once folded in
        # by `cli.py` / `export.py`) anchors its conversion-leaf demand itself, so
        # `_model_sinks(Mode.ONNX)` has no writer-manifest branch any more. When no
        # export sink demand is present it falls back to every preds.* key (the M2
        # all-preds render fallback below).
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
        # plan 29 W1: fold the discovered TEST sink NODE into the planning module
        # dict so it renders its OWN card and anchors demand via its declared
        # requires (outputs.*/meta.rows/masks.*) instead of the flat <sinks>
        # sentinel. In FIT/VAL/ONNX the sink's declare_io is empty -> the planner
        # collects it as inactive (no PlanStep/edge) -> plan_hash byte-UNCHANGED
        # (the trained checkpoint loads unperturbed, design §8 back-compat proof).
        # When a sink node is folded, the flat model sinks for TEST are empty: the
        # node's terminal-consumer demand keeps the producers (and transitively
        # their preds.*) alive (design §4.1).
        modules = dict(self._graph_modules)
        sink_node = self._attached_sink_node()
        # plan 31 W5.1: give an auto-collecting sink node (omitted `outputs:`) the
        # model module dict BEFORE the planner consults its `declare_io`, so it can
        # discover the active conversion producers feeding its collections. Inert
        # for an explicit-`outputs` sink (`bind_model_modules` no-ops there).
        if sink_node is not None and callable(getattr(sink_node, "bind_model_modules", None)):
            sink_node.bind_model_modules(self._graph_modules)
        folded_sink = sink_node is not None and mode is Mode.TEST
        if folded_sink:
            # the sink node anchors ALL its demand via its declared requires
            # (folded below) — flat model sinks are empty. The OLD flat
            # `_model_sinks` writer dead-preds gate does not run on this path
            # (the sink demands outputs.*, not preds.*); the equivalent runtime
            # safety net is restored AFTER compile via `_assert_no_dead_preds`
            # below — a `preds.*` the model computes but no producer feeds into a
            # demanded output is still a hard error at `salt2 test` (design §4.1,
            # parity with the M4.5 WriterCallback dead-preds error).
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
            resolve_origin_weighting(self._graph_modules, dm.train_dset.reader)
        else:
            self.compile_mode(Mode.TEST, self._boundary(dm.test_dset, "test"))
            check_class_names(self._graph_modules, dm.test_dset.reader)
            resolve_origin_weighting(self._graph_modules, dm.test_dset.reader)
            self._validate_writer_specs(dm.test_dset)
        self._ensure_bound()

    def _validate_writer_specs(self, test_dset: GraphDataset) -> None:
        """Static writer-input validation on the TEST path (design §2.7/§8).

        Hands the attached `WriterCallback` (if any) the union of the model
        modules' TEST-active produced ports and the test dataset's served
        boundary leaves, so it can prove each writer-declared require exists
        AND kind/dtype-unifies against its producer before the first batch
        (`WriterCallback.validate_specs`). Model-produced ports take precedence
        over a same-named boundary key (they are the executed leaf). No writer
        callback (programmatic ``Trainer.test`` without writers) → no-op, as the
        M2 anchor-on-all-preds path carries no writer requires to check.
        """
        writers, reader = self._attached_writer()
        if writers is None or not callable(getattr(writers, "validate_specs", None)):
            return
        producer_specs = dict(test_dset.boundary_specs())
        producer_specs.update(writers.model_producer_specs(self._graph_modules))
        writers.validate_specs(self._graph_modules, reader, producer_specs)

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
        self._apply_mup_shapes()
        self.schema = schema
        self._bound = True

    def _apply_mup_shapes(self) -> None:
        """Apply the muP base shapes over the whole ``net`` tree (design §3.4).

        The routing-stage shape application: after `bind_all` builds the
        width-dependent layers (incl. the `MuReadout` out-proj), set the base
        shapes from ``mup.shape_path`` over the WHOLE ``net`` so
        `MuReadout.width_mult()` resolves against the REAL base widths and
        `MuAdamW` sees per-parameter infshapes. The shape file is generated by
        ``salt2 mup-shapes`` over ``base_model.net`` / ``delta_model.net`` (only
        the ``apply_to`` widths differ — every other parameter has a FIXED, i.e.
        non-infinite, dim), so the parameter names match the live ``net`` exactly.
        ``rescale_params=False`` because the parameters already carry their muP
        init from the architectural port (the `Dense`/`MuReadout` resets); this
        call only attaches the infshapes used by `MuReadout.forward`/`MuAdamW`.

        No-op when no muP is configured or no ``shape_path`` was supplied (the
        per-module width-1 base shapes set at construction stand — a self-base
        model that still trains correctly under muP).

        Raises
        ------
        ConfigError
            When ``shape_path`` is set but the file does not exist.
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
        """Resolve the optimizer class (v1 surface + the muP ``MuAdamW`` swap, design §3.4).

        When `mup` is configured the optimizer is ``mup.optim.MuAdamW``
        regardless of the ``optimizer`` name — muP's per-parameter learning
        rates require the muP-aware AdamW (the v1 swap, the
        `_get_optimizer_class` muP branch the M6 TODO named). MuAdamW needs
        the base shapes set on the parameters (done at bind via
        `_apply_mup_shapes`), so the swap and the shape application are
        coupled.

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
        """Build the optimizer + OneCycleLR scheduler (v1 kept wholesale).

        Returns
        -------
        tuple[list[Optimizer], list[dict]]
            One optimizer and one step-interval scheduler config
            (modelwrapper.py:340-381).
        """
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
    attributes the culprit when a configured writer's explicit ``tasks``
    list excludes the dead tasks (the §4.2 worked-example attribution,
    M3-review fix); and offers the real per-task ``expose: [fit, val]`` opt-out
    (design §4.2 — keeps the task training while pruning its prediction from
    the TEST/ONNX plans) as the FIRST fix for train-only aux tasks, with the
    heavier ``--model.modules.X=null`` deletion as the alternative.

    Parameters
    ----------
    dead : list[str]
        The produced ``preds.*`` keys no writer consumes.
    produced : Mapping[str, str]
        Produced key -> producing module instance name.
    writers : Any
        The writer callback (duck-typed ``writers`` mapping for the
        tasks-narrowing hint; absent attributes degrade gracefully).

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
        # the real opt-out: expose: [fit, val] keeps the task training while
        # pruning its prediction from the TEST/ONNX plans (design §4.2). Listed
        # FIRST; --model.modules.X=null (delete the task entirely) is the
        # heavier alternative.
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

    The v2 muP arch port gives `StreamEmbed`/`TransformerEncoder` a ``mup``
    init_arg stored as a ``mup`` attribute (design §3.4 architectural half).
    A module is an eligible ``apply_to`` target iff it carries that attribute.
    Duck-typed so user modules that add a ``mup`` flag participate too.

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
    """Validate the muP routing config against the module dict (design §3.4 line 685).

    The single validator behind both `SaltModule.__init__` and
    ``salt2 graph validate`` (cli.py) so the same rules fire data-free in CI
    and at run construction. ``apply_to`` is an EXPLICIT instance-name list,
    NOT a regex (the design break from v1's ``apply_to`` x ``parameter_name``
    zip, configuration_muP.py:98-115).

    Rules (FD §3.4 line 695):

    - ``apply_to`` naming a module that does NOT exist → `ConfigError`.
    - ``apply_to`` naming a module that lacks a ``mup`` init_arg
      (`module_supports_mup` False) → `ConfigError` (the routing would
      configure a module that can't be parametrised).
    - a module with ``mup`` TRUTHY that is NOT in ``apply_to`` → WARNING
      (the arch flag is on but routing skips it — its base shapes / MuAdamW
      grouping would be silently inconsistent).
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
    """`SaltModule.__init__` wrapper around `validate_mup_routing` (module-private alias).

    Returns
    -------
    dict[str, Any] | None
        The normalised mup config or None.
    """
    return validate_mup_routing(mup, modules)


def _edge_encoders(modules: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The encoder modules that declare an edge port (duck-typed on ``edges_key``).

    A `TransformerEncoder` with ``edges:`` configured carries a non-None
    ``edges_key`` attribute (modules.py); duck-typed so user encoder modules
    that add an edge port participate too.

    Returns
    -------
    list[tuple[str, Any]]
        ``(instance name, module)`` pairs for edge-bearing encoders.
    """
    return [
        (name, module)
        for name, module in modules.items()
        if getattr(module, "edges_key", None) is not None
    ]


def _concat_first_stream(modules: Mapping[str, Any]) -> tuple[str, str] | None:
    """The `Concat`'s first stream (the edge-stream-first reference).

    Identifies the `Concat` PRECISELY by its produced ``seq.x`` key (the concat
    signature, modules.py Concat.declare_io) rather than by an attribute name —
    `Normaliser`/`Split` also carry a ``streams`` attr, so an attribute test would
    be ambiguous. Returns the concat's ``(instance name, streams[0])`` — the edge
    tensor's stream must equal this so the ``[B, T, T, D_e]`` edge matrix aligns
    with the LEADING ``T`` rows/cols of the concatenated ``[B, S, D]`` sequence
    the encoder receives (the v1 sort-first hack put the edge stream first for
    exactly this reason, saltmodel.py:66-73).

    Returns
    -------
    tuple[str, str] | None
        ``(concat name, first stream)``, or None when no `Concat` is configured.
    """
    for name, module in modules.items():
        streams = getattr(module, "streams", None)
        if not (isinstance(streams, (list, tuple)) and streams):
            continue
        declare = getattr(module, "declare_io", None)
        if not callable(declare):
            continue
        # the Concat is the module producing seq.x (its declare_io produces
        # seq.x/seq.mask/seq.layout); declare_io is config-only/static (design
        # §2.2), so calling it here is cheap and side-effect-free.
        produces = flatten_spec(declare(Mode.FIT).produces)
        if "seq.x" in produces:
            return name, streams[0]
    return None


def validate_edge_port(modules: Mapping[str, Any]) -> int:
    """Validate the encoder edge-port bind-time constraints (FD §6.7 1425-1431).

    The two HIDDEN v1 edge mechanisms made into NAMED constraints (ED2; the
    "land in M1 validation" rules). Runs at `SaltModule.__init__` (every
    fit/test) and in ``salt2 graph validate`` so the same rules fire data-free
    in CI and at run construction — the edge analogue of `validate_mup_routing`.
    No-op when no encoder declares an edge port.

    Rules:

    - **edge-stream-first** (rule a): the edge tensor's stream MUST be the FIRST
      stream of the `Concat` (``Concat.streams[0]``). v1 SILENTLY re-sorted the
      init_nets so the edge stream landed first (saltmodel.py:66-73) — required
      because the encoder zero-pads the ``[B, T, T, D_e]`` edge matrix to the
      register-augmented sequence length assuming the edge stream's ``T`` rows are
      the LEADING rows of the concatenated sequence (transformer.py:689-719). v2
      makes the ordering an EXPLICIT config constraint instead of a runtime sort:
      a mis-ordered concat is a named `ConfigError`.
    - **EdgeAttention-backend forcing** (rule b): an encoder with an edge port may
      NOT declare a non-edge attention backend (``flash-varlen`` / any backend
      outside ``{"torch-math"}``). v1 SILENTLY ignored ``attn_type`` whenever
      ``edge_embed_dim>0`` — it skipped the backend assignment
      (transformer.py:599-601,616-620), so a user who set ``flash-varlen`` got
      raw attention with no warning. v2 makes the silent bypass a named error.

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
        # the v2 encoder stores attn_type on its composed v1 Transformer; an
        # EdgeAttention encoder always runs raw torch attention (the v1 silent
        # bypass, transformer.py:599-601), so any other declared backend is a
        # NAMED error instead of a no-op.
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
    """`SaltModule.__init__` wrapper around `validate_edge_port` (module-private alias).

    Returns
    -------
    int
        The number of edge-bearing encoders validated.
    """
    return validate_edge_port(modules)


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


def resolve_origin_weighting(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Resolve name-based ``origin_weighting`` to ids before bind (design §5.1, §2.6).

    For every module exposing a callable ``resolve_origin_names`` (the
    `VertexingTaskModule` surface, duck-typed so user vertexing modules
    participate too), the names are mapped to integer origin ids against the
    schema artifact — the same ``schema_group(stream).attrs`` source the §2.6
    class-names check consults. Runs at `SaltModule.setup` (fit/test) right after
    `check_class_names`, BEFORE the two-phase bind, so the resolved ids are in
    place when `VertexingTaskModule.bind` builds the composed head. Integer-id
    weighting and readers without a schema are no-ops here; a name-based config
    that resolves nothing then fails loudly at bind (the names can't be turned
    into ids without the class-name attr).

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
