"""``salt2`` entry point — the jsonargparse YAML CLI for salt v2.

``salt2 fit``/``test`` go through `Salt2CLI` (`LightningCLI` over `SaltModule`
+ `GraphDataModule`); ``salt2 graph``/``schema``/``export``/muP tooling
dispatch to their own mains.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import time
import warnings
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import comet_ml  # noqa: F401 - must import before lightning (comet import-order contract)
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.trainer import Trainer

from salt.core import cli as graph_cli
from salt.core.data.datamodule import GraphDataModule
from salt.core.graph.errors import ConfigError, GraphError
from salt.core.onnx.config import ExportConfig
from salt.core.outputs.writers import OutputSectionWriter
from salt.core.saltmodule import SaltModule

__all__ = ["CONFIG_DIR", "DeepMergeParser", "Salt2CLI", "main"]


def _patch_jsonargparse_sys_modules_race() -> None:
    """Make jsonargparse's forward-ref ``sys.modules`` walk safe under live threads.

    jsonargparse resolves string forward refs in Protocol method signatures by
    scanning ``sys.modules.values()``. A live background thread that imports
    modules (e.g. Comet's upload threads, started by ``CometLogger.__init__``)
    can mutate ``sys.modules`` mid-walk and raise ``RuntimeError: dictionary
    changed size during iteration``. Retries the enrichment on that race
    signature — additive/idempotent, so re-running after a lost race is safe.

    No-ops gracefully if jsonargparse renames the private helper (pinned to
    ``jsonargparse>=4.49.0``); idempotent across repeat imports via the
    ``_salt_race_safe`` marker.
    """
    from jsonargparse import _postponed_annotations as _pa  # noqa: PLC0415, PLC2701 - patch site

    orig = getattr(_pa, "_enrich_globals_for_string_forward_refs", None)
    if orig is None or getattr(orig, "_salt_race_safe", False):
        return

    @functools.wraps(orig)
    def _race_safe_enrich(global_vars: dict[str, Any]) -> None:
        for _ in range(40):
            try:
                return orig(global_vars)
            except RuntimeError as err:
                if "changed size during iteration" not in str(err):
                    raise
                time.sleep(0.001)  # yield; the import burst that raced us is short-lived
        return orig(global_vars)  # last attempt: let a genuinely stuck race surface loudly

    _race_safe_enrich._salt_race_safe = True  # type: ignore[attr-defined]
    _pa._enrich_globals_for_string_forward_refs = _race_safe_enrich  # noqa: SLF001 - the patch site


_patch_jsonargparse_sys_modules_race()

CONFIG_DIR = Path(__file__).parent / "configs"
"""Directory shipping ``base2.yaml`` and the worked GN2v2 configs."""

_GRAPH_COMMANDS = frozenset({"graph", "schema", "mup-shapes", "mup-coord-check"})
_EXPORT_COMMAND = "export"

# --model.modules.X=null: jsonargparse's SUBCLASS adapter re-emits nested args
# as "--key=value" strings, so a None value arrives at the inner dict typehint
# as a literal string and fails validation. Rewriting to the JSON-block form
# ('--model.modules={"X": null}') goes through the subclass adapter's prev-val
# merge instead — siblings survive and the entry parses to None. Direct dict
# actions (data.modules, callbacks, writers) need NO rewrite: their NestedArg
# path natively merges and admits None — rewriting them would REPLACE the dict.
_MODULE_DICT_NULL = re.compile(
    r"^--(?P<parent>model\.(?:init_args\.)?modules)\.(?P<key>[\w-]+)=(?:null|None)$"
)


class DeepMergeParser(LightningArgumentParser):
    """`LightningArgumentParser` with cross-config-file dict-merge semantics.

    Native jsonargparse merges stacked config files via ``Namespace.update``,
    which treats dict-typed leaves atomically: a later config file REPLACES the
    whole ``modules:`` dict. The `merge_config` override unions dict leaves
    key-by-key instead, so later files add/update entries and earlier entries
    survive.

    ``None`` markers are KEPT through the merge rather than dropped (dropping
    them under subclass-mode ``model`` lets the outer merge resurrect a
    deleted entry from an earlier file). Deletion therefore happens at
    assembly time everywhere: `SaltModule`/`GraphDataModule` filter module
    dicts, `Salt2CLI` filters the callbacks dict.
    """

    def merge_config(self, cfg_from: Any, cfg_to: Any) -> Any:
        """Union dict-typed leaves key-by-key before the standard merge."""
        for key, val_from in list(cfg_from.items()):
            if not isinstance(val_from, dict):
                continue
            val_to = cfg_to.get(key)
            if isinstance(val_to, dict):
                cfg_from[key] = {**val_to, **val_from}
        return super().merge_config(cfg_from, cfg_to)

    def parse_args(self, args: Sequence[str] | None = None, *pargs: Any, **kwargs: Any) -> Any:
        """Parse args with ``--…modules.X=null`` normalised + the class_dict fan-out.

        Two pre-validation steps ride on the standard parse: (1) ``--…modules.X=null``
        is rewritten to the JSON-block form, and (2) ``--class_dict`` is fanned out
        onto the model-side consumers (`_fan_out_artifacts`) — the only point that
        runs AFTER the config-file deep-merge but BEFORE validation and any
        ``--print_config`` dump, so the resolved values are frozen into the saved
        run-dir config. Cannot be a `link_arguments` compute: its targets are dict
        elements of the single ``model.init_args.modules`` action, and a whole-dict
        self-link would destroy that action's deep-merge of ``base2.yaml``.
        """
        if args is None:
            args = sys.argv[1:]
        print_config_flags: str | None = None
        if isinstance(args, Sequence) and not isinstance(args, str):
            normalised: list[Any] = []
            for arg in args:
                if isinstance(arg, str) and (
                    arg == "--print_config" or arg.startswith("--print_config=")
                ):
                    # capture + strip so the deferred print_config dump does not
                    # fire inside super().parse_args before the fan-out lands
                    print_config_flags = arg.split("=", 1)[1] if "=" in arg else ""
                    continue
                normalised.append(_normalise_module_null(arg) if isinstance(arg, str) else arg)
            args = normalised
        # parse without validation so the fan-out lands before validation; honour
        # a caller-requested skip (jsonargparse's internal subcommand re-parse
        # passes _skip_validation) instead of re-validating here
        caller_skips = bool(kwargs.pop("_skip_validation", False))
        cfg = super().parse_args(args, *pargs, _skip_validation=True, **kwargs)
        _fan_out_artifacts(cfg)
        if not caller_skips:
            self.validate(cfg)
        if print_config_flags is not None:
            sys.stdout.write(self.dump(cfg, **_dump_kwargs(print_config_flags)))
            self.exit(0)
        return cfg


def _needs_logger(callback: Any) -> bool:
    """Whether a callback hard-requires an attached experiment logger to run.

    The stock `LearningRateMonitor` raises a ``MisconfigurationException`` on a
    logger-less trainer, so it is dropped from the assembly on a
    ``--trainer.logger false`` run.
    """
    from lightning.pytorch.callbacks import LearningRateMonitor  # noqa: PLC0415 - cheap, local

    return isinstance(callback, LearningRateMonitor)


def _comet_accepts_dict_kwargs() -> bool:
    """Whether this Lightning `CometLogger` still accepts the ``dict_kwargs`` kwarg.

    Introspects the constructor since newer Lightning `CometLogger` versions
    drop the kwarg (the run label flows through ``experiment_name`` instead).
    A bare ``**kwargs`` does NOT count: the modern logger forwards it to a
    Comet ``ExperimentConfig`` that rejects the old ``dict_kwargs`` name, so
    only an explicit parameter is a safe signal.
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "dict_kwargs" in params


def _comet_accepts_experiment_name() -> bool:
    """Whether this Lightning `CometLogger` declares ``experiment_name`` explicitly.

    Newer Lightning `CometLogger` drops ``experiment_name`` from its signature
    (the run label flows through ``**kwargs`` instead); jsonargparse instantiates
    by the DECLARED signature, so setting it as an ``init_arg`` on the newer
    logger crashes with "Option 'experiment_name' is not accepted". When absent,
    the caller routes the name through the ``COMET_EXPERIMENT_NAME`` env var
    instead.
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "experiment_name" in params


def _best_checkpoint(config_path: Path) -> str:
    """Best-epoch selection: lowest ``loss=`` next to the saved config.

    Scans both ``<config dir>/ckpts/*.ckpt`` and ``<config dir>/checkpoints/*.ckpt``
    (Lightning's `ModelCheckpoint` default dirname — keep in sync with
    ``base2.yaml``'s ``epoch={epoch:03d}-loss={val/loss:.5f}.ckpt`` naming) and
    picks the smallest ``loss=<value>`` embedded in any filename.

    Raises
    ------
    ConfigError
        When no ``loss=``-named checkpoint exists next to the config.
    """
    ckpt_dirs = [config_path.parent / name for name in ("ckpts", "checkpoints")]
    print(
        "salt2 test: no --ckpt_path specified, looking for best checkpoint in "
        + " and ".join(str(d) for d in ckpt_dirs)
    )
    scored = [
        (float(found[0]), str(ckpt))
        for ckpt_dir in ckpt_dirs
        for ckpt in sorted(ckpt_dir.glob("*.ckpt"))
        if (found := re.findall(r"(?<=loss=)(?:\d+(?:\.\d*)?|\.\d+)", ckpt.name))
    ]
    if not scored:
        raise ConfigError(
            f"no 'loss='-named checkpoints under {config_path.parent}/{{ckpts,checkpoints}} — "
            "pass --ckpt_path explicitly (v1 best-epoch contract, utils/cli.py:71-78; "
            "base2.yaml names checkpoints 'epoch=NNN-loss=<val/loss>.ckpt' to match)"
        )
    best = min(scored)[1]
    print(f"salt2 test: using checkpoint {best}")
    return best


def _normalise_module_null(arg: str) -> str:
    """Rewrite ``--…modules.X=null`` to the JSON-block form (see `_MODULE_DICT_NULL`).

    Returns `arg` unchanged when it is not a module-dict null deletion.
    """
    match = _MODULE_DICT_NULL.match(arg)
    if match is None:
        return arg
    return f'--{match["parent"]}={{"{match["key"]}": null}}'


_CLASS_DICT_ARG = "class_dict"
"""Top-level convenience flag name (``--class_dict``)."""

_CLASS_DICT_CLASS = "ClassificationTaskModule"
"""Class-name suffix of the only ``class_dict``/``weight_source`` consumer."""

# print_config flag -> ArgumentParser.dump kwarg map (mirrors jsonargparse's
# _ActionPrintConfig flag vocabulary; "skip_null" is "skip_none" on dump)
_PRINT_CONFIG_FLAGS = {"skip_default": "skip_default", "skip_null": "skip_none"}


def _dump_kwargs(flags: str) -> dict[str, bool]:
    """Translate a ``--print_config=<flags>`` value to `ArgumentParser.dump` kwargs.

    Mirrors `jsonargparse`'s ``_ActionPrintConfig`` flag handling so the
    fan-out's manual dump (intercepted in `DeepMergeParser.parse_args`) honours
    the same ``skip_default`` / ``skip_null`` keywords as the native action.

    Raises
    ------
    ConfigError
        On an unrecognised flag (parallels the native action's error).
    """
    kwargs: dict[str, bool] = {}
    for flag in (f for f in flags.split(",") if f):
        if flag not in _PRINT_CONFIG_FLAGS:
            raise ConfigError(
                f"--print_config: invalid flag {flag!r} "
                f"(supported: {', '.join(sorted(_PRINT_CONFIG_FLAGS))})"
            )
        kwargs[_PRINT_CONFIG_FLAGS[flag]] = True
    return kwargs


def _entry_get(entry: Any, key: str) -> Any:
    """Read ``key`` off a config-tree leaf that is a `Namespace` *or* a plain dict.

    Module dict values arrive as `Namespace` from CLI/parse but as plain dicts
    from a deep-merged config file, so the fan-out must read either shape.
    """
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _entry_set(entry: Any, key: str, value: Any) -> None:
    """Write ``key`` onto a config-tree leaf that is a `Namespace` *or* a plain dict."""
    if isinstance(entry, dict):
        entry[key] = value
    else:
        setattr(entry, key, value)


def _resolve_class_path(class_path: str) -> type:
    """Import a dotted ``module.Class`` path and return the class object."""
    module_path, _, attr = class_path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, attr)


def _instantiate_class_config(cfg: Any) -> Any:
    """Instantiate a jsonargparse ``class_path``/``init_args`` config block.

    Mirrors jsonargparse's default subclass instantiation (`class_type(**init_args)`)
    for the deferred fit logger (`Salt2CLI._reattach_fit_logger`) — a leaf ``cfg``
    (already a fully-parsed value, not a subclass block) is returned unchanged, and
    nested ``class_path``/``init_args`` init args are instantiated recursively so an
    arbitrary user logger block round-trips, not just the scalar-arg `CometLogger`.
    """
    if not isinstance(cfg, Namespace):
        return cfg
    class_path = cfg.get("class_path")
    if class_path is None:
        return cfg
    cls = _resolve_class_path(class_path)
    init_args = cfg.get("init_args")
    kwargs: dict[str, Any] = {}
    if init_args is not None:
        kwargs = {key: _instantiate_class_config(val) for key, val in vars(init_args).items()}
    return cls(**kwargs)


def _is_persistence_sink(class_path: str) -> bool:
    """Whether a callback ``class_path`` names a TEST persistence sink.

    A config can wire the persistence sink at the ``callbacks:`` level instead
    of ``writers.modules`` — an `H5OutputWriter` (or a subclass, or any callback
    exposing the duck-typed ``writer_demand`` surface) — so this resolves the
    class at the pre-instantiate point to let the writer-less refusal accept it.
    `OnnxExportSink` also exposes ``writer_demand`` but persists NOTHING in TEST,
    so it is explicitly excluded — a config wiring only an `OnnxExportSink` must
    still fail the writer-less check.

    Returns ``False`` for an unimportable path (treated as not-a-sink), an
    ONNX-only sink, or a plain callback.
    """
    import importlib  # noqa: PLC0415 - local, only on the test path

    from salt.core.outputs import (  # noqa: PLC0415 - avoid import cycle at top
        H5OutputWriter,
        OnnxExportSink,
    )

    module_path, _, attr = class_path.rpartition(".")
    if not module_path:
        return False
    try:
        cls = getattr(importlib.import_module(module_path), attr, None)
    except Exception:  # noqa: BLE001 - an unresolvable class_path is simply not a sink
        return False
    if not isinstance(cls, type):
        return False
    if issubclass(cls, OnnxExportSink):  # ONNX-only sink: no TEST persistence
        return False
    return issubclass(cls, H5OutputWriter) or callable(getattr(cls, "writer_demand", None))


def _has_callback_persistence_sink(callbacks: Any) -> bool:
    """Whether the ``callbacks:`` config carries a TEST persistence sink.

    Inspects the dict-keyed ``callbacks:`` config at the pre-instantiate point
    and returns ``True`` when any non-``None`` entry is an `H5OutputWriter`-style
    persistence sink (`_is_persistence_sink`). Used by the ``salt2 test``
    writer-less check.
    """
    items = (callbacks or {}).items() if hasattr(callbacks, "items") else ()
    for entry in (val for _, val in items if val is not None):
        class_path = _entry_get(entry, "class_path") or ""
        if class_path and _is_persistence_sink(class_path):
            return True
    return False


def _fan_out_artifacts(cfg: Any) -> Any:
    """Fan ``--class_dict`` out onto the model-side consumers.

    The top-level convenience arg reproduces the verbose per-task
    ``weight_source`` override block from one flag: for every module under
    ``model.init_args.modules``, a `ClassificationTaskModule` whose
    ``init_args.weight_source`` is unset/null gets
    ``weight_source := {"from_class_dict": <--class_dict>}`` (validated through
    the same `_checked_weight_source` validator the task's ``__init__`` uses). A
    task that already sets ``weight_source`` (e.g. on resume, or an explicit
    per-task override) is LEFT ALONE.

    The mutation lands on `cfg` in place, BEFORE validation and any
    ``--print_config`` dump, so the resolved values are frozen into the saved
    run-dir config exactly like the verbose form. A no-op when the flag is not
    set.

    Parameters
    ----------
    cfg : Any
        The parsed namespace. For ``salt2 fit``/``test`` the model block lives
        under ``cfg.<subcommand>.model``; on the run-free surface it is
        ``cfg.model``. Both are handled.
    """
    from salt.core.nn.tasks import _checked_weight_source  # noqa: PLC0415 - torch-heavy, CLI-time

    for scope, model in _iter_model_blocks(cfg):
        class_dict = scope.get(_CLASS_DICT_ARG)
        if not class_dict:
            continue
        init_args = getattr(model, "init_args", None)
        modules = getattr(init_args, "modules", None) if init_args is not None else None
        if not isinstance(modules, dict):
            continue
        for entry in modules.values():
            if entry is None:  # null = deleted at assembly
                continue
            class_path = _entry_get(entry, "class_path") or ""
            module_args = _entry_get(entry, "init_args")
            if module_args is None:
                continue
            if (
                class_path.endswith(_CLASS_DICT_CLASS)
                and _entry_get(module_args, "weight_source") is None
            ):
                _entry_set(
                    module_args,
                    "weight_source",
                    _checked_weight_source({"from_class_dict": str(class_dict)}),
                )
    return cfg


def _iter_model_blocks(cfg: Any) -> list[tuple[Any, Any]]:
    """Pair each ``model`` namespace with the scope its ``--class_dict`` lives in.

    The convenience flag is scoped like ``--name``: top-level on the run-free
    surface (``cfg.class_dict`` ↔ ``cfg.model``), subcommand-scoped on a trainer
    run (``cfg.fit.class_dict`` ↔ ``cfg.fit.model``).
    """
    blocks: list[tuple[Any, Any]] = []
    direct = cfg.get("model")
    if direct is not None:
        blocks.append((cfg, direct))
    for sub in Salt2CLI.subcommands():
        sub_cfg = cfg.get(sub)
        sub_model = sub_cfg.get("model") if sub_cfg is not None else None
        if sub_model is not None:
            blocks.append((sub_cfg, sub_model))
    return blocks


class Salt2CLI(LightningCLI):
    """The salt v2 `LightningCLI`.

    Wires `SaltModule` (subclass mode) and `GraphDataModule` through
    `DeepMergeParser`, auto-loads ``configs/base2.yaml``, and adds the salt
    top-level namespaces:

    - ``name:`` — run name, linked to ``model.init_args.name``.
    - ``callbacks:`` — dict-keyed, deep-mergeable; assembled into
      ``trainer.callbacks`` ahead of the stock list entries (``None`` values
      are filtered = deleted).
    - ``writers:`` — ``output`` template + ``half_precision`` + the
      deep-mergeable ``modules`` dict, assembled into ONE `WriterCallback`.

    ``auto_configure_optimizers`` is off (`SaltModule.configure_optimizers`
    owns the OneCycleLR schedule) and Lightning's
    ``load_from_checkpoint_support`` instantiator is disabled so hparams
    never embed the module dict — data-less loads go through
    ``SaltModule.load_from_checkpoint(path, modules=...)`` instead.
    """

    def __init__(self, args: Any = None, run: bool = True, **kwargs: Any) -> None:
        # before super().__init__: add_arguments_to_parser runs inside it and
        # needs to know whether this is the run-free parse surface
        self._run_mode = bool(run)
        default_config = [str(CONFIG_DIR / "base2.yaml")]
        parser_kwargs: dict[str, Any] = {"default_env": True}
        if run:
            parser_kwargs.update({
                sub: {"default_config_files": default_config} for sub in self.subcommands()
            })
        else:
            parser_kwargs["default_config_files"] = default_config
        # overwrite a stale config.yaml from a previous run: until the M6 run
        # dirs land, fit writes into trainer.default_root_dir (cwd by
        # default) and a leftover config.yaml must not abort the retry loop
        # (stage-E ergonomics finding)
        kwargs.setdefault("save_config_kwargs", {"overwrite": True})
        super().__init__(
            model_class=SaltModule,
            datamodule_class=GraphDataModule,
            subclass_mode_model=True,
            parser_class=DeepMergeParser,
            parser_kwargs=parser_kwargs,
            auto_configure_optimizers=False,
            load_from_checkpoint_support=False,
            seed_everything_default=42,
            args=args,
            run=run,
            **kwargs,
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """The trainer entry points exposed.

        ``fit`` and ``test`` only — `SaltModule.setup` rejects ``validate``/``predict``.
        """
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
        }

    def _parse_ckpt_path(self) -> None:
        """No-op override of lightning's checkpoint hyper-parameter re-parse.

        Lightning's ``LightningCLI._parse_ckpt_path`` loads
        ``checkpoint['hyper_parameters']`` whenever ``--ckpt_path`` is set at
        parse time and re-parses them onto the config as ``{model: <hparams>}``.
        `SaltModule` saves its hyper-parameters with ``ignore=["modules"]``
        (modules are runtime graph objects, never reconstructable from
        hparams), so the re-parse presents a modules-less model spec and
        jsonargparse REPLACES the saved config's model block, dying with
        "the following arguments are required: modules".

        The config-driven contract here is: the run's saved ``config.yaml`` is
        the single source of the model architecture; the checkpoint carries
        weights only, loaded unchanged via ``trainer.fit/test(ckpt_path=...)``.
        The hparams re-parse can never contribute information — at best it
        re-applies values already in the config, at worst it wipes the model —
        so it is disabled wholesale, matching the already-disabled
        ``load_from_checkpoint_support=False`` half of the same feature.
        """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add the salt top-level namespaces and the run-name link."""
        parser.add_argument(
            "--name",
            type=str,
            default="salt",
            help="run name (unrestricted; the Athena export name lives at export.model_name, §5)",
        )
        parser.add_argument(
            "--callbacks",
            type=dict[str, Callback | None] | None,
            default={},
            help="dict-keyed callbacks, deep-mergeable; assembled into trainer.callbacks "
            "(design §5.3; an entry set to null is removed)",
        )
        parser.add_argument(
            "--writers.modules",
            type=dict[str, Any] | None,
            default=None,
            help="REMOVED in W6c — migrate to an ``outputs:``/``callbacks:`` sink; "
            "a non-null entry here raises ConfigError at instantiate_classes (see gn2v2-dummy.yaml)",
        )
        parser.add_argument(
            "--outputs",
            type=dict[str, OutputSectionWriter | None] | None,
            default=None,
            help="plan-34 W34.2 top-level outputs: section — dict-keyed GraphModule writers "
            "(RunTaskOutput / InputCopyWriter / PadMaskWriter), deep-mergeable, composed AFTER "
            "the model; the section field order drives the eval-H5 TASK-column order (an entry "
            "set to null is removed). NOT a link_arguments link — the section is composed onto "
            "the model in instantiate_classes (SaltModule.compose_output_section), not wired via "
            "link_arguments.",
        )
        parser.add_argument(
            "--export",
            type=ExportConfig | None,
            default=None,
            help="the export-ONLY half of the ONNX contract, consumed by `salt2 export` "
            "(design §5.1, §7; M4.5): model_name (no '_'/'-', validated ONLY at export "
            "time), inputs (port/name/sequence/dyn_axis/alias) and the rename/combine "
            "manifest post-processing. The OUTPUT manifest derives from writers.modules "
            "(M4.5 unified manifest) — declaring export.outputs is a hard error at export "
            "time. Inert during fit/test; round-trips through saved run configs.",
        )
        parser.add_argument(
            f"--{_CLASS_DICT_ARG}",
            type=str | None,
            default=None,
            help="convenience flag (plan-24 Wave 1): fanned out to weight_source="
            "{from_class_dict: <path>} on EACH ClassificationTaskModule whose weight_source "
            "is unset/null, reproducing the verbose per-task weight_source block from one "
            "flag. Tasks that set weight_source explicitly are left alone; the loss CE-weight "
            "buffer is bitwise-invariant to how the path arrived (design §5.4, R1.3/R1.6).",
        )
        if not self._run_mode:
            # run-free parses must round-trip a saved run config.yaml, which
            # carries the Lightning run-surface key ckpt_path; accept + ignore it.
            parser.add_argument(
                "--ckpt_path",
                type=str | None,
                default=None,
                help="accepted-and-ignored on the run-free parse surface so saved run "
                "configs round-trip into the salt2 graph tooling",
            )
        parser.link_arguments("name", "model.init_args.name")
        # the top-level outputs: section is NOT a link_arguments compute
        # (subclass-mode model targets grab the whole namespace) — it is
        # composed onto the instantiated model in `instantiate_classes` below.

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Assemble ``callbacks:`` dict values and the writers block into the trainer.

        Dict values come first (YAML insertion order), then any stock Lightning
        entries from the raw ``trainer.callbacks`` list; ``None`` values are
        filtered — the assembly-time half of null-deletion.

        The ``base2.yaml`` default ``callbacks.lr_monitor`` LearningRateMonitor
        is dropped when no experiment logger is attached: the stock
        LearningRateMonitor hard-raises a ``MisconfigurationException`` on a
        logger-less trainer, so it is kept only when a logger is attached
        (``--trainer.logger false`` opt-out runs drop it).
        """
        callbacks_dict = self._get(self.config_init, "callbacks") or {}
        has_logger = bool(self._get(self.config_init, "trainer.logger"))
        assembled = [
            cb
            for cb in callbacks_dict.values()
            if cb is not None and (has_logger or not _needs_logger(cb))
        ]
        if assembled:
            stock = self._get(self.config_init, "trainer.callbacks") or []
            kwargs = {**kwargs, "callbacks": [*assembled, *stock]}
        return super().instantiate_trainer(**kwargs)

    def instantiate_classes(self) -> None:
        """Instantiate, then compose the top-level ``outputs:`` section onto the model.

        The top-level ``outputs:`` namespace is instantiated by jsonargparse into
        ``config_init["outputs"]`` (a dict of section writers built from their
        ``class_path``); after the standard instantiation this composes that
        section onto the `SaltModule` (folding the section writers into the
        planning module dict, binding RunTaskOutput's tasks). Done here — NOT via
        ``link_arguments`` — because a subclass-mode model link target grabs the
        whole namespace (jsonargparse pitfall).

        A migration error fires first if the config carries a live ``writers:``
        block (non-null entries under ``writers.modules``): WriterCallback
        assembly was removed — migrate to an ``outputs:``/``callbacks:`` sink
        (see gn2v2-dummy.yaml).
        """
        # writers: block with live modules is no longer supported. Null-delete
        # overrides (writers.modules.X: null) are exempt — they produce an empty
        # dict here.
        live_writer_modules = {
            name: writer
            for name, writer in (self._get(self.config, "writers.modules") or {}).items()
            if writer is not None
        }
        if live_writer_modules:
            raise ConfigError(
                "the `writers:` section was removed; migrate to an `outputs:`/`callbacks:` "
                "sink — see gn2v2-dummy.yaml (W6c removal)"
            )
        # Defer the fit-stage experiment logger past the racy validation pass.
        # jsonargparse's instantiate_classes pass validates the model: block
        # against a runtime_checkable Protocol, which makes jsonargparse walk
        # sys.modules.values() to resolve string forward refs. The default-ON
        # CometLogger is instantiated in that SAME pass, and its __init__ eagerly
        # starts comet's background upload threads, which import modules and
        # mutate sys.modules — the two race into "RuntimeError: dictionary
        # changed size during iteration", surfaced as "model does not validate
        # against any Union subtype". So on `fit` we stash the logger config,
        # null it for the parser pass (trainer built logger-less, no comet
        # threads), then re-instantiate and attach it AFTER validation
        # completes. Test/graph/export never carry a live logger here.
        deferred_logger_cfg = self._detach_fit_logger()
        super().instantiate_classes()
        self._reattach_fit_logger(deferred_logger_cfg)
        section = self._get(self.config_init, "outputs")
        model = getattr(self, "model", None)
        composer = getattr(model, "compose_output_section", None) if model is not None else None
        if section and callable(composer):
            composer({k: w for k, w in section.items() if w is not None})
            # bind the section to the sink callbacks NOW: datamodule setup runs
            # BEFORE model setup and resolves the sink's writer_demand, which
            # needs the section already bound.
            trainer = getattr(self, "trainer", None)
            for cb in (trainer.callbacks if trainer is not None else []):
                if callable(getattr(cb, "bind_output_section", None)):
                    cb.bind_output_section(model._output_section)  # noqa: SLF001 - same-package wiring

    def _detach_fit_logger(self) -> Any:
        """Stash + null the fit-stage ``trainer.logger`` block ahead of the parser pass.

        Returns the un-instantiated logger config and sets ``trainer.logger =
        False`` in the fit config so jsonargparse's ``instantiate_classes`` pass
        builds a logger-less trainer, keeping the eager-comet-thread
        `CometLogger.__init__` out of the racy model-block validation window
        (see ``instantiate_classes``). Returns ``None`` outside ``fit`` or when
        no live logger is configured.
        """
        if getattr(self.config, "subcommand", None) != "fit":
            return None
        cfg = self.config["fit"]
        logger = getattr(cfg.trainer, "logger", None)
        if not logger:  # False / None — nothing to defer
            return None
        cfg.trainer.logger = False
        return logger

    def _reattach_fit_logger(self, logger_cfg: Any) -> None:
        """Instantiate the deferred logger and attach it to the trainer post-validation.

        Called after ``super().instantiate_classes()`` — the racy ``sys.modules``
        walk is done, so building the `CometLogger` (and starting its comet
        threads) is now safe.
        """
        if logger_cfg is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        trainer.logger = _instantiate_class_config(logger_cfg)

    def before_instantiate_classes(self) -> None:
        """Per-stage config patches — fit-stage Comet wiring and test-stage eval ergonomics.

        On ``fit`` a configured experiment logger is wired: the run ``--name``
        drives ``experiment_name`` and ``dict_kwargs: {name}``, ``online`` is
        forced ``false`` when ``COMET_API_KEY`` is absent or under
        ``fast_dev_run``, and ``COMET_OFFLINE_DIRECTORY`` is set + created
        alongside the trainer log dir. ``base2.yaml`` ships ``logger: false``,
        so this is a no-op unless the user opts into a logger.

        On ``test``: no resolved-config dump and no experiment logger on eval
        runs; a missing ``--ckpt_path`` triggers the best-checkpoint glob
        (requires exactly one user ``--config``); multi-device eval is
        rejected/forced to one device; a sink-less eval is refused up front
        (TEST predictions would be computed and never persisted).

        Raises
        ------
        ConfigError
            On a sink-less test config (no callbacks-level persistence sink),
            an ambiguous config list without ``--ckpt_path``, or an explicit
            multi-device list.
        """
        subcommand = getattr(self.config, "subcommand", None)
        if subcommand == "fit":
            self._wire_experiment_logger(self.config["fit"])
            return
        if subcommand != "test":
            return
        cfg = self.config["test"]
        self.save_config_callback = None  # no config.yaml dump on test
        cfg.trainer.logger = False
        # writers.modules no longer constitutes a valid persistence sink; a
        # config with live writers.modules raises the migration error in
        # instantiate_classes. Accept only the callbacks-level sink path.
        has_callback_sink = _has_callback_persistence_sink(cfg.get("callbacks"))
        if not has_callback_sink:
            raise ConfigError(
                "salt2 test needs a persistence sink — a callbacks-level H5OutputWriter "
                "sink; predictions would otherwise be computed and never persisted "
                "(design §4.2, §8). Supply a top-level outputs: section "
                "(InputCopyWriter -> RunTaskOutput -> PadMaskWriter, in v1 H5 column order) "
                "with a callbacks-level salt.core.outputs.H5OutputSink (W6c removal; "
                "the writers: block is gone — see gn2v2-dummy.yaml)"
            )
        if not cfg.get("ckpt_path"):
            configs = cfg.get("config") or []
            if len(configs) != 1:
                raise ConfigError(
                    "salt2 test without --ckpt_path needs exactly one --config (the saved "
                    "run config.yaml next to ckpts/) to glob the best checkpoint — "
                    "v1 contract (utils/cli.py:323-325)"
                )
            cfg.ckpt_path = _best_checkpoint(Path(str(configs[0])))
        devices = cfg.trainer.devices
        if isinstance(devices, str | int):
            try:
                n_devices = int(devices)
            except ValueError:
                n_devices = None  # "auto" — single-device eval contract
            if n_devices is not None and n_devices > 1:
                print("salt2 test: forcing --trainer.devices=1 (single-device eval, design §8)")
                cfg.trainer.devices = "1"
        elif isinstance(devices, list) and len(devices) > 1:
            raise ConfigError("salt2 test requires a single device (design §8, v1 cli.py:330)")

    @staticmethod
    def _wire_experiment_logger(cfg: Any) -> None:
        """Wire a configured fit-stage experiment logger.

        A no-op unless ``trainer.logger`` is a configured logger block. For a
        `CometLogger` block the run ``--name`` is threaded into
        ``experiment_name`` and ``dict_kwargs: {name}``, ``online`` is forced
        ``false`` when ``COMET_API_KEY`` is absent or under ``fast_dev_run`` (so
        a key-less / smoke run never blocks on the Comet API), and
        ``COMET_OFFLINE_DIRECTORY`` is set to — and created at — the trainer log
        dir. Non-Comet loggers are left untouched.
        """
        logger = cfg.trainer.logger
        # a bare False/None (the --trainer.logger false opt-out) means no tracking
        if not logger:
            return
        run_name = cfg.get("name") or "salt"
        init_args = getattr(logger, "init_args", None)
        # class_path check by name keeps this working on the un-instantiated
        # config block (only the Comet path carries this special-casing)
        class_path = getattr(logger, "class_path", "")
        is_comet = class_path.endswith(CometLogger.__name__) or "comet" in class_path.lower()
        if init_args is None or not is_comet:
            return
        # newer Lightning CometLogger drops `experiment_name` from its signature
        # (the label flows through **kwargs instead), and jsonargparse
        # instantiates by the DECLARED signature — so setting it as an init_arg
        # on the newer logger crashes at instantiate_classes. Set it only when
        # the constructor declares it; otherwise route through the env var.
        if _comet_accepts_experiment_name():
            init_args.experiment_name = run_name
        else:
            os.environ.setdefault("COMET_EXPERIMENT_NAME", run_name)
        # dict_kwargs: only inject when the constructor still accepts it
        if _comet_accepts_dict_kwargs():
            dict_kwargs = getattr(init_args, "dict_kwargs", None) or {}
            dict_kwargs["name"] = run_name
            init_args.dict_kwargs = dict_kwargs
        # offline when no API key or smoke run
        if not os.getenv("COMET_API_KEY") or cfg.trainer.fast_dev_run:
            init_args.online = False
        log_dir = cfg.trainer.default_root_dir or "logs"
        os.environ["COMET_OFFLINE_DIRECTORY"] = str(log_dir)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def after_fit(self) -> None:
        """Tell the user where the run artifacts went.

        ``config.yaml`` lands in the trainer log dir (``default_root_dir`` — cwd
        unless overridden) and checkpoints in the checkpoint callback's dirpath;
        neither location is otherwise announced.
        """
        log_dir = self.trainer.log_dir or self.trainer.default_root_dir
        ckpt_dir = getattr(self.trainer.checkpoint_callback, "dirpath", None)
        print(f"salt2 fit artifacts: config.yaml in {log_dir}")
        print(
            f"salt2 fit artifacts: checkpoints in {ckpt_dir}"
            if ckpt_dir
            else "salt2 fit artifacts: no checkpoint callback configured"
        )
        print("(run-directory layout with timestamped names lands in M6 — design §5)")


def main(args: Sequence[str] | None = None) -> int:
    """``salt2`` console entry point.

    ``salt2 graph``/``schema``/``mup-shapes``/``mup-coord-check`` dispatch to
    the static graph + muP tooling (`salt.core.cli.main`) and ``salt2 export``
    to the ONNX exporter; everything else goes to `Salt2CLI` (``salt2
    fit``/``test``). Graph errors (`GraphError`) print as a clean one-block
    form on stderr instead of a Python traceback.

    Raises
    ------
    SystemExit
        Re-raised from the parser (usage errors, ``--help`` — after the
        see-also note when help was requested).
    """
    argv = list(sys.argv[1:] if args is None else args)
    if argv and argv[0] in _GRAPH_COMMANDS:
        return graph_cli.main(argv)
    if argv and argv[0] == _EXPORT_COMMAND:
        # local import: the exporter pulls onnx/onnxruntime — not needed at
        # fit/test/graph startup
        from salt.core.onnx import export as onnx_export  # noqa: PLC0415 - heavy, export-only

        return onnx_export.main(argv[1:])
    help_requested = bool(argv) and argv[0] in {"-h", "--help"}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            Salt2CLI(args=None if args is None else argv)
    except SystemExit:
        if help_requested:
            print(
                "\nsee also: 'salt2 graph --help' (static graph tooling: validate/plan/plot/"
                "why/deadcode/resolve, design §4), 'salt2 schema --help' (schema artifacts, "
                "§2.6), 'salt2 mup-shapes --help' / 'salt2 mup-coord-check --help' (muP base/"
                "delta infshapes + coord-check, design §3.4), 'salt2 export --help' (ONNX "
                "export, §7; --manifest prints the writer-derived output manifest)"
            )
        raise
    except GraphError as err:
        # one-block form — no Python traceback for config errors
        print(f"salt.core.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
