"""``salt2`` entry point — the jsonargparse YAML surface for salt v2 (design §5).

Two families of subcommands share the one console script:

- ``salt2 fit`` / ``salt2 test`` — `LightningCLI` over `SaltModule` +
  `GraphDataModule` (design §3.4, §5.1). ``validate``/``predict`` are M5+
  (`SaltModule.setup` rejects those stages, design §9.5).
- ``salt2 graph ...`` / ``salt2 schema ...`` — the static graph tooling
  (design §4), dispatched unchanged to `salt.core.cli.main`.

The YAML schema follows design §5.1: modules are ``dict[str, Module]``
(never lists), the model block is a ``class_path``/``init_args`` subclass
block, and the salt-added top-level namespaces are ``name:``, ``writers:``
and ``callbacks:``. `DeepMergeParser` restores the §5.3 cross-config-file
dict semantics (add/update/delete-by-null — natively a later file replaces
dict leaves wholesale, see ``salt/core/SPIKE_jsonargparse.md``); ``None``
entries are filtered at assembly time (`SaltModule` / `GraphDataModule` for
modules, `Salt2CLI.instantiate_trainer` for callbacks).

``base2.yaml`` (this package's ``configs/``) is auto-loaded for every
``fit``/``test`` invocation — the v1 ``base.yaml`` mechanism kept wholesale
(design §5: trainer defaults plus the dict-keyed ``callbacks:`` defaults,
including the default ``writers:`` block — design §8).

The ``writers:`` block (M3, design §8) is assembled into ONE `WriterCallback`
appended after the ``callbacks:`` dict entries; ``salt2 test`` keeps the v1
eval ergonomics (single config, best-checkpoint glob without ``--ckpt_path``,
``logger=False``, forced single device — ``utils/cli.py:312-332``).

The Comet logger wiring lands in M6 (sub-wave E, gate CM1; FD §6.5): the
``comet_ml``-before-lightning import order is preserved at the top of this
module (v1 ``main.py:5``), ``--name`` is linked to the configured logger's
``experiment_name`` (v1 glue ``cli.py:101``), and ``before_instantiate_classes``
replicates the v1 fit-stage Comet setup (``cli.py:281-294``): ``dict_kwargs:
{name}``, ``online: false`` when no ``COMET_API_KEY`` / under ``fast_dev_run``,
the ``COMET_OFFLINE_DIRECTORY`` env + its mkdir, and ``logger=False`` on test
(``cli.py:317``). ``base2.yaml`` ships a default-ON ``CometLogger`` block since
the plan-24 Wave 0 flip (v1 parity); local / CI / smoke / gate runs opt OUT with
``--trainer.logger false``. `LearningRateMonitor` is a dict-keyed ``callbacks:
lr_monitor`` entry that only attaches once a logger is present (kept by default
now, dropped on a logger-less run; FD §13 E3). Run-dir timestamping rides with a
later wave.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import comet_ml  # noqa: F401 - import-order contract: comet before lightning (v1 main.py:5, §5)
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.trainer import Trainer

from salt.core import cli as graph_cli
from salt.core.data.datamodule import GraphDataModule
from salt.core.graph.errors import ConfigError, GraphError
from salt.core.onnx.config import ExportConfig
from salt.core.saltmodule import SaltModule
from salt.core.writers import DEFAULT_OUTPUT, Writer, WriterCallback

__all__ = ["CONFIG_DIR", "DeepMergeParser", "Salt2CLI", "main"]

CONFIG_DIR = Path(__file__).parent / "configs"
"""Directory shipping ``base2.yaml`` and the worked GN2v2 configs (design §5.1)."""

_GRAPH_COMMANDS = frozenset({"graph", "schema", "mup-shapes", "mup-coord-check"})
_EXPORT_COMMAND = "export"
_CONVERT_COMMAND = "convert-config"

# --model.modules.X=null (also the explicit --model.init_args.modules.X=null):
# jsonargparse's SUBCLASS adapter re-emits nested args as "--key=value"
# strings, so a None value arrives at the inner dict typehint as a literal
# string and fails validation. Rewriting to the JSON-block form
# ('--model.modules={"X": null}') goes through the subclass adapter's
# prev-val merge instead — siblings survive and the entry parses to None
# (deleted at assembly, design §5.3). Direct dict actions (data.modules,
# callbacks, writers) need NO rewrite: their NestedArg path natively merges
# (_typehints.py:925-933) and admits None — rewriting them would in fact
# REPLACE the dict. Model-side paths only.
_MODULE_DICT_NULL = re.compile(
    r"^--(?P<parent>model\.(?:init_args\.)?modules)\.(?P<key>[\w-]+)=(?:null|None)$"
)


class DeepMergeParser(LightningArgumentParser):
    """`LightningArgumentParser` with the design §5.3 config-file dict semantics.

    Native jsonargparse (4.46.0) merges stacked config files via
    ``Namespace.update``, which treats dict-typed leaves atomically: a later
    config file REPLACES the whole ``modules:`` dict. The `merge_config`
    override unions dict leaves key-by-key, so later files add/update entries
    and earlier entries survive (spike capability 6, SPIKE_jsonargparse.md).

    Unlike the spike's flat-dict recipe, ``None`` markers are KEPT through the
    merge rather than dropped: under subclass-mode ``model`` the inner
    ``init_args`` parser merges first, and dropping the ``None`` there lets
    the outer merge resurrect the deleted entry from the earlier file (found
    in the stage-C nested re-spike). Deletion therefore happens at assembly
    time everywhere — which the CLI-set null path needs anyway (design §5.3):
    `SaltModule`/`GraphDataModule` filter module dicts, `Salt2CLI` filters the
    callbacks dict.
    """

    def merge_config(self, cfg_from: Any, cfg_to: Any) -> Any:
        """Union dict-typed leaves key-by-key before the standard merge.

        Returns
        -------
        Any
            The merged namespace; dict leaves union (later file wins per
            key), ``None`` values are kept as deletion markers for the
            assembly-time filters.
        """
        for key, val_from in list(cfg_from.items()):
            if not isinstance(val_from, dict):
                continue
            val_to = cfg_to.get(key)
            if isinstance(val_to, dict):
                cfg_from[key] = {**val_to, **val_from}
        return super().merge_config(cfg_from, cfg_to)

    def parse_args(self, args: Sequence[str] | None = None, *pargs: Any, **kwargs: Any) -> Any:
        """Parse args with ``--…modules.X=null`` normalised + the Wave-1 artifact fan-out.

        Two pre-validation steps ride on the standard parse:

        1. ``--…modules.X=null`` is rewritten to the JSON-block form (design
           §5.3, module docstring).
        2. ``--norm_dict`` / ``--class_dict`` are fanned out onto the model-side
           consumers (`_fan_out_artifacts`, plan-24 Wave 1) — the only point that
           runs AFTER the config-file deep-merge but BEFORE validation and any
           ``--print_config`` dump, so the resolved ``norm_dict`` / per-task
           ``weight_source`` are validated and frozen into the saved run-dir
           config exactly as the verbose per-module override block produces them.

        The fan-out cannot be a `link_arguments` compute: its targets are dict
        elements of the single ``model.init_args.modules`` action (not registered
        actions), and a whole-dict self-link destroys that action's deep-merge of
        ``base2.yaml``. So the fan-out is applied here, between a
        validation-skipped parse and an explicit `validate`, with
        ``--print_config`` intercepted (a `NonParsingAction` whose dump otherwise
        fires before this) so the dumped config reflects the resolved values.

        Returns
        -------
        Any
            The parsed namespace (`jsonargparse` semantics unchanged outside the
            artifact fan-out).
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
        # parse without validation so the fan-out lands before validation; if the
        # caller already asked to skip validation (jsonargparse's internal
        # subcommand re-parse passes _skip_validation), honour that and do NOT
        # re-validate here — avoids the "multiple values for _skip_validation"
        # clash and respects the caller's lenient intent
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
    logger-less trainer (``lr_monitor.py``). ``base2.yaml`` ships a default-ON
    ``CometLogger`` (plan-24 Wave 0) plus the ``callbacks.lr_monitor``
    LearningRateMonitor entry, so the monitor is KEPT by default; the assembly
    drops such callbacks on a ``--trainer.logger false`` run (M6 sub-wave E) —
    keeping the opt-out local / CI / smoke fit/test path runnable.

    Parameters
    ----------
    callback : Any
        An assembled callback instance.

    Returns
    -------
    bool
        True for a `LearningRateMonitor` (the only logger-hard-required default).
    """
    from lightning.pytorch.callbacks import LearningRateMonitor  # noqa: PLC0415 - cheap, local

    return isinstance(callback, LearningRateMonitor)


def _comet_accepts_dict_kwargs() -> bool:
    """Whether this Lightning `CometLogger` still accepts the v1 ``dict_kwargs`` kwarg.

    The v1 Comet wiring injected ``dict_kwargs={name}`` (the column-prefix
    label, ``cli.py:287``); newer Lightning `CometLogger` drops that kwarg (the
    run label flows through ``experiment_name`` / ``name`` instead). Introspect
    the constructor so the wiring stays compatible across both API generations.

    Returns
    -------
    bool
        True only when ``dict_kwargs`` is an EXPLICIT named parameter of
        `CometLogger.__init__`. A bare ``**kwargs`` does NOT count: the modern
        logger accepts ``**kwargs`` but forwards them to a Comet
        ``ExperimentConfig`` that rejects the v1 ``dict_kwargs`` name, so only an
        explicit parameter is a safe signal.
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "dict_kwargs" in params


def _comet_accepts_experiment_name() -> bool:
    """Whether this Lightning `CometLogger` declares ``experiment_name`` explicitly.

    The v1 wiring set ``init_args.experiment_name`` (``cli.py:101,287``); newer
    Lightning `CometLogger` drops it from the signature (the run label is
    forwarded through ``**kwargs`` to the Comet experiment instead). jsonargparse
    instantiates the logger by its DECLARED signature, so an ``experiment_name``
    ``init_arg`` on the newer logger is rejected at ``instantiate_classes`` with
    "Option 'experiment_name' is not accepted" — crashing the (now default-ON)
    fit. Introspect the constructor so the wiring sets it as an ``init_arg`` only
    when it is an EXPLICIT parameter, and otherwise routes the name through the
    ``COMET_EXPERIMENT_NAME`` env var (the version-robust path). A bare
    ``**kwargs`` does NOT count (jsonargparse validates against named params).

    Returns
    -------
    bool
        True only when ``experiment_name`` is an EXPLICIT named parameter of
        `CometLogger.__init__`.
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "experiment_name" in params


def _best_checkpoint(config_path: Path) -> str:
    """The v1 best-epoch selection: lowest ``loss=`` next to the saved config.

    Port of ``utils/cli.py:55-79``, extended to BOTH checkpoint layouts
    (M3-review fix — the original ``ckpts/``-only glob could never match a
    v2 run): scan ``<config dir>/ckpts/*.ckpt`` (the v1 layout) and
    ``<config dir>/checkpoints/*.ckpt`` (Lightning's `ModelCheckpoint`
    default dirname — ``base2.yaml`` names the files
    ``epoch={epoch:03d}-loss={val/loss:.5f}.ckpt`` so this glob matches by
    construction; keep the two in sync) and pick the smallest
    ``loss=<value>`` embedded in any filename.

    Parameters
    ----------
    config_path : Path
        The single user config (the saved run ``config.yaml``).

    Returns
    -------
    str
        Path to the best checkpoint.

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
    """Rewrite ``--…modules.X=null`` to the JSON-block form (module docstring).

    Returns
    -------
    str
        The rewritten argument, or `arg` unchanged when it is not a
        module-dict null deletion.
    """
    match = _MODULE_DICT_NULL.match(arg)
    if match is None:
        return arg
    return f'--{match["parent"]}={{"{match["key"]}": null}}'


_NORM_DICT_ARG = "norm_dict"
"""Top-level convenience flag name (``--norm_dict``) — Wave 1 fan-out source."""

_CLASS_DICT_ARG = "class_dict"
"""Top-level convenience flag name (``--class_dict``) — Wave 1 fan-out source."""

_NORM_DICT_CLASS = "Normaliser"
"""Class-name suffix of the only ``norm_dict`` consumer (nn/modules.py ~2144)."""

_CLASS_DICT_CLASS = "ClassificationTaskModule"
"""Class-name suffix of the only ``class_dict``/``weight_source`` consumer (nn/tasks.py ~1027)."""

# the print_config flag → ArgumentParser.dump kwarg map (mirrors the
# jsonargparse _ActionPrintConfig flag vocabulary: comma-separated keywords
# under `--print_config[=flag,flag]`; "skip_null" is "skip_none" on dump)
_PRINT_CONFIG_FLAGS = {"skip_default": "skip_default", "skip_null": "skip_none"}


def _dump_kwargs(flags: str) -> dict[str, bool]:
    """Translate a ``--print_config=<flags>`` value to `ArgumentParser.dump` kwargs.

    Mirrors `jsonargparse`'s ``_ActionPrintConfig`` flag handling so the
    fan-out's manual dump (intercepted in `DeepMergeParser.parse_args`) honours
    the same ``skip_default`` / ``skip_null`` keywords as the native action.

    Parameters
    ----------
    flags : str
        The raw flags string (``""`` for a bare ``--print_config``).

    Returns
    -------
    dict[str, bool]
        Keyword arguments for `dump` (empty for the bare form).

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
    from a deep-merged config file (`test_salt2_cli.py` shows both surfaces), so
    the fan-out must read either shape.

    Returns
    -------
    Any
        The value at `key`, or None when absent.
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


def _fan_out_artifacts(cfg: Any) -> Any:
    """Fan ``--norm_dict`` / ``--class_dict`` out onto the model-side consumers (Wave 1).

    Restores v1's one-flag ergonomics (plan-24 §6 Wave 1 + R1.6, §5.4): the two
    top-level convenience args reproduce today's verbose per-module override
    block from two flags. They are *purely* model/CLI-side — no data module, no
    setup graph (R1.1/R1.2): ``norm_dict`` is read only by the `Normaliser`
    (`nn/modules.py` ~2144) and ``class_dict`` only by each
    `ClassificationTaskModule` for its CE-weight buffer (`nn/tasks.py` ~1027).

    For every module under ``model.init_args.modules`` (the subclass-mode block):

    - a `Normaliser` gets ``init_args.norm_dict := <--norm_dict>`` (its sole
      consumer);
    - a `ClassificationTaskModule` whose ``init_args.weight_source`` is
      unset/null gets ``weight_source := {"from_class_dict": <--class_dict>}``,
      validated through the same `_checked_weight_source` validator the task's
      ``__init__`` uses, so ``bind()``/``materialise()`` behave identically.
      A task that already sets ``weight_source`` (e.g. the saved run-dir config
      on resume, or an explicit per-task override) is LEFT ALONE — this is what
      keeps resume unaffected and makes the resolved namespace byte-equal to the
      verbose form (plan-24 §5.4 retirement-target spelling; gate R3).

    The mutation lands on `cfg` in place (and is returned), BEFORE validation +
    any ``--print_config`` dump (`DeepMergeParser.parse_args`), so the resolved
    values are frozen into the saved run-dir config exactly like the verbose
    form. A no-op when neither flag is set (the args are absent on the run-free
    graph-tooling parser, where ``cfg.get`` returns None).

    Parameters
    ----------
    cfg : Any
        The parsed namespace. For ``salt2 fit``/``test`` the model block lives
        under ``cfg.<subcommand>.model``; on the run-free surface it is
        ``cfg.model``. Both are handled.

    The resolved ``weight_source`` is built through the same
    `_checked_weight_source` validator the task's ``__init__`` uses (it cannot
    fail for the literal ``{from_class_dict}`` the fan-out constructs, but the
    validator stays the single source of truth for the canonical spelling).

    Returns
    -------
    Any
        `cfg`, mutated in place.
    """
    from salt.core.nn.tasks import _checked_weight_source  # noqa: PLC0415 - torch-heavy, CLI-time

    for scope, model in _iter_model_blocks(cfg):
        norm_dict = scope.get(_NORM_DICT_ARG)
        class_dict = scope.get(_CLASS_DICT_ARG)
        if not norm_dict and not class_dict:
            continue
        init_args = getattr(model, "init_args", None)
        modules = getattr(init_args, "modules", None) if init_args is not None else None
        if not isinstance(modules, dict):
            continue
        for entry in modules.values():
            if entry is None:  # null = deleted at assembly (design §5.3)
                continue
            class_path = _entry_get(entry, "class_path") or ""
            module_args = _entry_get(entry, "init_args")
            if module_args is None:
                continue
            if norm_dict and class_path.endswith(_NORM_DICT_CLASS):
                _entry_set(module_args, "norm_dict", str(norm_dict))
            if (
                class_dict
                and class_path.endswith(_CLASS_DICT_CLASS)
                and _entry_get(module_args, "weight_source") is None
            ):
                _entry_set(
                    module_args,
                    "weight_source",
                    _checked_weight_source({"from_class_dict": str(class_dict)}),
                )
    return cfg


def _iter_model_blocks(cfg: Any) -> list[tuple[Any, Any]]:
    """Pair each ``model`` namespace with the scope its ``--norm_dict``/``--class_dict`` live in.

    The two convenience flags are scoped exactly like the ``--name`` arg the
    existing CLI glue links into the model: top-level on the run-free surface
    (``cfg.norm_dict`` ↔ ``cfg.model``), and subcommand-scoped on a trainer run
    (``cfg.fit.norm_dict`` ↔ ``cfg.fit.model``). Pairing the flag scope with its
    model block keeps the fan-out reading the flags from the right level.

    Returns
    -------
    list[tuple[Any, Any]]
        ``(scope, model)`` pairs: ``(cfg, cfg.model)`` on the run-free surface,
        ``(cfg.<sub>, cfg.<sub>.model)`` per present trainer subcommand, or
        ``[]`` when no model block is present.
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
    """The salt v2 `LightningCLI` (design §5, §5.3).

    Wires `SaltModule` (subclass mode — the §5.1 ``class_path``/``init_args``
    model block) and `GraphDataModule` (plain ``data:`` block) through
    `DeepMergeParser`, auto-loads ``configs/base2.yaml``, and adds the
    salt top-level namespaces:

    - ``name:`` — run name, linked to ``model.init_args.name`` (the single
      surviving link of v1's CLI glue, design §5).
    - ``callbacks:`` — dict-keyed, deep-mergeable; values are assembled into
      ``trainer.callbacks`` ahead of the stock list entries (design §5.3;
      ``None`` values are filtered = deleted).
    - ``writers:`` — ``output`` template + ``half_precision`` + the
      deep-mergeable ``modules`` dict, assembled into ONE `WriterCallback`
      (design §8; defaults ship in ``base2.yaml`` in the v1 column order
      ``inputs_copy -> tasks -> pad_mask``).

    ``auto_configure_optimizers`` is off (`SaltModule.configure_optimizers`
    owns the v1 OneCycleLR schedule, design §3.4) and Lightning's
    ``load_from_checkpoint_support`` instantiator is disabled so hparams
    never embed the module dict (design §3.4 — data-less loads go through
    ``SaltModule.load_from_checkpoint(path, modules=...)`` instead).
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
        """The trainer entry points exposed in M2 (design §9.5).

        Returns
        -------
        dict[str, set[str]]
            ``fit`` and ``test`` only — `SaltModule.setup` rejects
            ``validate``/``predict`` until M5+.
        """
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
        }

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add the salt top-level namespaces and the run-name link (design §5)."""
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
            type=dict[str, Writer | None] | None,
            default=None,
            help="dict-keyed prediction-writer modules, deep-mergeable; assembled into one "
            "WriterCallback (design §8; an entry set to null is removed; dict order is the "
            "per-group column order)",
        )
        parser.add_argument(
            "--writers.output",
            type=str,
            default=DEFAULT_OUTPUT,
            help="eval-file path template; keys: ckpt_dir, ckpt_stem, sample (design §5.1)",
        )
        parser.add_argument(
            "--writers.half_precision",
            type=bool,
            default=False,
            help="write float columns at half precision (v1 PredictionWriter flag)",
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
            f"--{_NORM_DICT_ARG}",
            type=str | None,
            default=None,
            help="convenience flag (plan-24 Wave 1): fanned out to the Normaliser module's "
            "norm_dict init_arg (its only consumer), reproducing "
            "--model.modules.<norm>.init_args.norm_dict from one flag. Purely model-side — "
            "no data module reads it (design §5.4, R1.1/R1.6).",
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
            # run-free parses must round-trip a SAVED run config.yaml, which
            # carries the Lightning run-surface key ckpt_path (M3-review fix:
            # `salt2 graph ... -c <run_dir>/config.yaml` used to die with
            # "Option 'ckpt_path' is not accepted"); accepted and ignored.
            parser.add_argument(
                "--ckpt_path",
                type=str | None,
                default=None,
                help="accepted-and-ignored on the run-free parse surface so saved run "
                "configs round-trip into the salt2 graph tooling",
            )
        parser.link_arguments("name", "model.init_args.name")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Assemble ``callbacks:`` dict values and the writers block into the trainer.

        Dict values come first (YAML insertion order), then the
        `WriterCallback` built from ``writers:`` (design §8 — inert outside
        the test stage), then any stock Lightning entries from the raw
        ``trainer.callbacks`` list (design §5.3 assembly order); ``None``
        values are filtered — the assembly-time half of null-deletion.

        The ``base2.yaml`` default ``callbacks.lr_monitor`` LearningRateMonitor
        is dropped when no experiment logger is attached (M6 sub-wave E): the
        stock LearningRateMonitor hard-raises a ``MisconfigurationException`` on
        a logger-less trainer. Since the plan-24 Wave 0 flip ``base2.yaml`` ships
        a default-ON CometLogger, so the callback is kept by default and only
        drops on a ``--trainer.logger false`` opt-out run (CI/smoke/gate fixtures)
        — otherwise it would break those logger-less runs. Mirrors the v1 intent
        (LR monitoring is meaningful only with a logger; base.yaml:37 pairs it
        with the CometLogger).

        Returns
        -------
        Trainer
            The instantiated trainer.
        """
        callbacks_dict = self._get(self.config_init, "callbacks") or {}
        has_logger = bool(self._get(self.config_init, "trainer.logger"))
        assembled = [
            cb
            for cb in callbacks_dict.values()
            if cb is not None and (has_logger or not _needs_logger(cb))
        ]
        writer_modules = {
            name: writer
            for name, writer in (self._get(self.config_init, "writers.modules") or {}).items()
            if writer is not None
        }
        if writer_modules:
            assembled.append(
                WriterCallback(
                    modules=writer_modules,
                    output=self._get(self.config_init, "writers.output") or DEFAULT_OUTPUT,
                    half_precision=bool(self._get(self.config_init, "writers.half_precision")),
                )
            )
        if assembled:
            stock = self._get(self.config_init, "trainer.callbacks") or []
            kwargs = {**kwargs, "callbacks": [*assembled, *stock]}
        return super().instantiate_trainer(**kwargs)

    def before_instantiate_classes(self) -> None:
        """Per-stage config patches — the v1 eval + Comet surface kept (utils/cli.py:281-332).

        On ``fit`` a configured experiment logger is wired the v1 way (the M6
        CometLogger half, sub-wave E / gate CM1, ``cli.py:281-294``): the run
        ``--name`` drives ``experiment_name`` and ``dict_kwargs: {name}``,
        ``online`` is forced ``false`` when ``COMET_API_KEY`` is absent or under
        ``fast_dev_run``, and ``COMET_OFFLINE_DIRECTORY`` is set + created
        alongside the trainer log dir. ``base2.yaml`` ships ``logger: false``,
        so this is a no-op on the default local run — it only fires once the
        user opts into a logger.

        On ``test`` the v1 eval surface is kept (``cli.py:312-332``): no
        resolved-config dump and no experiment logger on eval runs; a missing
        ``--ckpt_path`` triggers the v1 best-checkpoint glob (which requires
        exactly ONE user ``--config``, the saved run config next to ``ckpts/``
        or ``checkpoints/``); multi-device eval is rejected/forced to one
        device; a writer-less eval is refused up front (TEST predictions would
        be computed and never persisted — design §4.2, §8).

        Raises
        ------
        ConfigError
            On a writer-less test config, an ambiguous config list without
            ``--ckpt_path``, or an explicit multi-device list.
        """
        subcommand = getattr(self.config, "subcommand", None)
        if subcommand == "fit":
            self._wire_experiment_logger(self.config["fit"])
            return
        if subcommand != "test":
            return
        cfg = self.config["test"]
        self.save_config_callback = None  # v1: no config.yaml dump on test (cli.py:312-316)
        cfg.trainer.logger = False
        writer_modules = cfg.get("writers.modules") or {}
        if not any(writer is not None for writer in writer_modules.values()):
            raise ConfigError(
                "salt2 test needs at least one writer under writers.modules — predictions "
                "would be computed and never persisted (design §4.2, §8; base2.yaml ships "
                "inputs_copy/tasks/pad_mask defaults)"
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
                n_devices = None  # "auto" — the WriterCallback single-device assert covers it
            if n_devices is not None and n_devices > 1:
                print("salt2 test: forcing --trainer.devices=1 (single-device eval, design §8)")
                cfg.trainer.devices = "1"
        elif isinstance(devices, list) and len(devices) > 1:
            raise ConfigError("salt2 test requires a single device (design §8, v1 cli.py:330)")

    @staticmethod
    def _wire_experiment_logger(cfg: Any) -> None:
        """Wire a configured fit-stage experiment logger the v1 way (cli.py:281-294).

        A no-op unless ``trainer.logger`` is a configured logger block (the
        ``base2.yaml`` default ``logger: false`` skips this entirely). For a
        `CometLogger` block the run ``--name`` is threaded into
        ``experiment_name`` and ``dict_kwargs: {name}`` (the run label Comet
        shows + the column-prefix source), ``online`` is forced ``false`` when
        ``COMET_API_KEY`` is absent or under ``fast_dev_run`` (the v1 offline
        fallback, so a key-less / smoke run never blocks on the Comet API), and
        ``COMET_OFFLINE_DIRECTORY`` is set to — and created at — the trainer log
        dir so offline runs have somewhere to write. Non-Comet loggers are left
        untouched (only the Comet path carries the v1 special-casing,
        ``CLAUDE.md`` logger convention).

        Parameters
        ----------
        cfg : Any
            The ``fit`` subcommand config namespace.
        """
        logger = cfg.trainer.logger
        # base2 now ships a default-ON CometLogger (plan-24 Wave 0); a bare
        # False/None (the --trainer.logger false opt-out) means no tracking
        if not logger:
            return
        run_name = cfg.get("name") or "salt"
        init_args = getattr(logger, "init_args", None)
        # only the CometLogger block carries the v1 special-casing (class_path
        # check by name keeps this working on the un-instantiated config block)
        class_path = getattr(logger, "class_path", "")
        is_comet = class_path.endswith(CometLogger.__name__) or "comet" in class_path.lower()
        if init_args is None or not is_comet:
            return
        # the run name drives experiment_name (v1 link_arguments, cli.py:101,287).
        # Newer Lightning CometLogger drops `experiment_name` from its signature
        # (the label flows through **kwargs to the Comet experiment), and
        # jsonargparse instantiates by the DECLARED signature — so setting it as an
        # init_arg on the newer logger is rejected at instantiate_classes and
        # crashes the (now default-ON) fit. Set it as an init_arg only when the
        # constructor declares it; otherwise route the name through the
        # COMET_EXPERIMENT_NAME env var (version-robust, parallels dict_kwargs).
        if _comet_accepts_experiment_name():
            init_args.experiment_name = run_name
        else:
            os.environ.setdefault("COMET_EXPERIMENT_NAME", run_name)
        # dict_kwargs was the v1 column-prefix mechanism; newer Lightning
        # CometLogger drops it (the name now flows through experiment_name), so
        # only inject it when the constructor still accepts it — version-robust
        if _comet_accepts_dict_kwargs():
            dict_kwargs = getattr(init_args, "dict_kwargs", None) or {}
            dict_kwargs["name"] = run_name
            init_args.dict_kwargs = dict_kwargs
        # offline when no API key or smoke run (v1 cli.py:289-290)
        if not os.getenv("COMET_API_KEY") or cfg.trainer.fast_dev_run:
            init_args.online = False
        # the offline output dir (v1 cli.py:293-294)
        log_dir = cfg.trainer.default_root_dir or "logs"
        os.environ["COMET_OFFLINE_DIRECTORY"] = str(log_dir)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def after_fit(self) -> None:
        """Tell the user where the run artifacts went (M6 run dirs pending).

        Without run dirs, ``config.yaml`` lands in the trainer log dir
        (``default_root_dir`` — cwd unless overridden) and checkpoints in
        the checkpoint callback's dirpath; neither location is otherwise
        announced (stage-E ergonomics finding).
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
    """``salt2`` console entry point (pyproject ``[project.scripts]``).

    ``salt2 graph …`` / ``salt2 schema …`` / ``salt2 mup-shapes`` /
    ``salt2 mup-coord-check`` dispatch to the static graph + muP tooling
    (`salt.core.cli.main`, design §4, §3.4) and ``salt2 export`` to the
    ONNX exporter (`salt.core.onnx.export.main`, design §7); everything
    else goes to `Salt2CLI` (``salt2 fit`` / ``salt2 test``, design §5).
    Console use (``args is None``) passes ``args=None`` through so
    Lightning reads ``sys.argv`` natively (no spurious "args parameter is
    intended..." warning); programmatic argv is filtered for the same
    warning. Graph errors (`GraphError`) print as the clean §4.1 one-block
    form on stderr instead of a Python traceback. Top-level
    ``-h``/``--help`` gains a see-also note for the graph/schema/export
    subcommand family.

    Returns
    -------
    int
        Process exit code (graph tooling semantics for graph/schema/export;
        0 when a trainer subcommand completes).

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
    if argv and argv[0] == _CONVERT_COMMAND:
        # local import: the v1->v2 config converter (M7 W1). YAML-only, but it
        # runs `salt2 graph validate` on its own output, so import at use time.
        from salt.core import convert  # noqa: PLC0415 - converter, CLI-time only

        return convert.main(argv[1:])
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
                "export, §7; --manifest prints the writer-derived output manifest) and "
                "'salt2 convert-config --help' (v1 YAML -> v2 schema translator, §9.3)"
            )
        raise
    except GraphError as err:
        # the §4.1 one-block form — no Python traceback for config errors
        print(f"salt.core.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
