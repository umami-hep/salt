"""``salt`` entry point — the jsonargparse YAML CLI for salt v2.

``salt fit``/``test`` go through `SaltCLI` (`LightningCLI` over `SaltModule`
+ `GraphDataModule`); ``salt graph``/``schema``/``export``/``inference``/muP
tooling dispatch to their own mains.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import comet_ml  # noqa: F401 - must import before lightning (comet import-order contract)
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.trainer import Trainer

from salt import cli as graph_cli
from salt.data.datamodule import GraphDataModule
from salt.graph.errors import ConfigError, GraphError
from salt.onnx.config import ExportConfig
from salt.outputs.run_task_output import OutputSectionWriter
from salt.parser import DeepMergeParser
from salt.model.saltmodule import SaltModule

__all__ = ["CONFIG_DIR", "SaltCLI", "main"]


def _patch_jsonargparse_sys_modules_race() -> None:
    """Make jsonargparse's forward-ref ``sys.modules`` walk safe under live
    threads (e.g. Comet's upload threads mutating ``sys.modules`` mid-walk
    raises ``RuntimeError: ... changed size during iteration``): retries the
    enrichment on that race signature. No-op if jsonargparse renames the
    private helper; idempotent via the ``_salt_race_safe`` marker.
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


# --- pre-de-core checkpoint/config compatibility (Plan 61) -------------------
# v2 checkpoints and their saved ``config.yaml`` written BEFORE the de-core
# rename carry ``salt.core.*`` class_paths. This remapper resolves them to their
# flat ``salt.*`` homes at load time so old artifacts stay loadable with no edit.
# Ordered longest-prefix-first so the ``salt.core.nn.*`` / ``salt.core.data.*``
# fan-out (which splits across three destination packages) is resolved before
# the bare package facades. The ONLY sanctioned ``salt.core.*`` strings in the
# codebase live here + the compat tests (see docs/architecture.md §Checkpoints).
_CLASS_PATH_REMAP: dict[str, str] = {
    # data -> readers
    "salt.core.data.reader": "salt.data.readers.reader",
    "salt.core.data.uproot_reader": "salt.data.readers.uproot_reader",
    "salt.core.data.multisample_reader": "salt.data.readers.multisample_reader",
    "salt.core.data.stream": "salt.data.readers.stream",
    "salt.core.data.cuts": "salt.data.readers.cuts",
    "salt.core.data.vds_module": "salt.data.readers.vds",
    "salt.core.data.vds": "salt.data.readers.vds",
    # data -> processors
    "salt.core.data.features": "salt.data.processors.features",
    "salt.core.data.labels": "salt.data.processors.labels",
    "salt.core.data.maskformer_targets": "salt.data.processors.maskformer_targets",
    "salt.core.data.multi_target": "salt.data.processors.multi_target",
    "salt.core.data.ftag_labeller": "salt.data.processors.ftag_labeller",
    # data top-level + facade
    "salt.core.data.datamodule": "salt.data.datamodule",
    "salt.core.data.dataset": "salt.data.dataset",
    "salt.core.data.base": "salt.data.base",
    "salt.core.data.dtypes": "salt.data.dtypes",
    "salt.core.data.samplers": "salt.data.samplers",
    "salt.core.data.input_samples": "salt.data.input_samples",
    "salt.core.data": "salt.data",
    # nn -> model / model.modules / model.nn
    "salt.core.nn.tasks": "salt.model.modules.tasks",
    "salt.core.nn.base": "salt.model.base",
    "salt.core.nn.bind": "salt.model.bind",
    "salt.core.nn.stream_embed": "salt.model.modules.stream_embed",
    "salt.core.nn.transformer_encoder": "salt.model.modules.transformer_encoder",
    "salt.core.nn.pooling": "salt.model.modules.pooling",
    "salt.core.nn.norm": "salt.model.modules.norm",
    "salt.core.nn.plumbing": "salt.model.modules.plumbing",
    "salt.core.nn.losses": "salt.model.modules.losses",
    "salt.core.nn.maskdecoder": "salt.model.modules.maskdecoder",
    "salt.core.nn.maskformer_matched_loss": "salt.model.modules.maskformer_matched_loss",
    "salt.core.nn.edge_embed": "salt.model.modules.edge_embed",
    "salt.core.nn.transformer": "salt.model.nn.transformer",
    "salt.core.nn.attention": "salt.model.nn.attention",
    "salt.core.nn.dense": "salt.model.nn.dense",
    "salt.core.nn.layernorm": "salt.model.nn.layernorm",
    "salt.core.nn.posenc": "salt.model.nn.posenc",
    "salt.core.nn.featurewise": "salt.model.nn.featurewise",
    "salt.core.nn.matcher": "salt.model.nn.matcher",
    "salt.core.nn.maskformer_loss": "salt.model.nn.maskformer_loss",
    "salt.core.nn": "salt.model.modules",
    # top-level singletons
    "salt.core.saltmodule": "salt.model.saltmodule",
    "salt.core.SaltModule": "salt.model.SaltModule",
    "salt.core.mup": "salt.model.mup",
    "salt.core.render": "salt.graph.render",
    "salt.core.loss_history": "salt.utils.loss_history",
    "salt.core.outputs": "salt.outputs",
    "salt.core.graph": "salt.graph",
    "salt.core.callbacks": "salt.callbacks",
    "salt.core.onnx": "salt.onnx",
    "salt.core.utils": "salt.utils",
    "salt.core.testing": "salt.testing",
    "salt.core.optim": "salt.optim",
    "salt.core.inference": "salt.inference",
    "salt.core.config_utils": "salt.config_utils",
    "salt.core.schema": "salt.schema",
    "salt.core.parser": "salt.parser",
    "salt.core.main": "salt.main",
    "salt.core.cli": "salt.cli",
}
_CLASS_PATH_REMAP_KEYS = sorted(_CLASS_PATH_REMAP, key=len, reverse=True)


def _remap_class_path(name: str) -> str:
    """Remap a pre-de-core ``salt.core.*`` dotted path to its ``salt.*`` home.

    No-op for any non-``salt.core`` path (longest matching prefix wins). Lets
    checkpoints/configs authored before the Plan-61 rename load unmodified.
    """
    if not isinstance(name, str) or not name.startswith("salt.core"):
        return name
    for old in _CLASS_PATH_REMAP_KEYS:
        if name == old or name.startswith(old + "."):
            return _CLASS_PATH_REMAP[old] + name[len(old):]
    return name


def _patch_jsonargparse_class_path_remap() -> None:
    """Wrap jsonargparse's class-path importer so a saved config/ckpt naming
    ``salt.core.*`` resolves to its ``salt.*`` home — covers the top-level model
    subclass resolution during config parse (jsonargparse ``_typehints`` binds
    ``import_object`` by value at import, so the wrapper is installed on every
    loaded jsonargparse module holding that reference). Idempotent; no-op if
    jsonargparse renames the helper (falls back to salt-owned resolution +
    the documented ckpt-config rewrite).
    """
    import sys  # noqa: PLC0415

    from jsonargparse import _typehints as _th  # noqa: F401, PLC0415, PLC2701 - force-load the binder
    from jsonargparse import _util as _ju  # noqa: PLC0415, PLC2701 - patch site

    orig = getattr(_ju, "import_object", None)
    if orig is None or getattr(orig, "_salt_decore_remap", False):
        return

    @functools.wraps(orig)
    def _remapped_import_object(name: str) -> Any:
        return orig(_remap_class_path(name))

    _remapped_import_object._salt_decore_remap = True  # type: ignore[attr-defined]
    for _mod in list(sys.modules.values()):
        if getattr(_mod, "__name__", "").startswith("jsonargparse") and \
                getattr(_mod, "import_object", None) is orig:
            _mod.import_object = _remapped_import_object  # noqa: SLF001 - the patch site


_patch_jsonargparse_class_path_remap()

CONFIG_DIR = Path(__file__).parent / "configs"
"""Directory shipping ``base2.yaml`` and the worked GN2v2 configs."""

_GRAPH_COMMANDS = frozenset({"graph", "schema", "mup-shapes", "mup-coord-check"})
_EXPORT_COMMAND = "export"
_INFERENCE_COMMAND = "inference"


def _needs_logger(callback: Any) -> bool:
    """Whether a callback hard-requires an attached experiment logger to run.

    The stock `LearningRateMonitor` raises a ``MisconfigurationException`` on a
    logger-less trainer, so it is dropped from the assembly on a
    ``--trainer.logger false`` run.
    """
    from lightning.pytorch.callbacks import LearningRateMonitor  # noqa: PLC0415 - cheap, local

    return isinstance(callback, LearningRateMonitor)


def _comet_accepts_dict_kwargs() -> bool:
    """Whether this Lightning `CometLogger` still declares ``dict_kwargs``
    explicitly (newer versions drop it; a bare ``**kwargs`` does NOT count —
    the modern logger forwards it to a Comet ``ExperimentConfig`` that rejects
    the old name).
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "dict_kwargs" in params


def _comet_accepts_experiment_name() -> bool:
    """Whether this Lightning `CometLogger` declares ``experiment_name``
    explicitly (jsonargparse instantiates by the declared signature, so
    setting it when absent crashes with "Option not accepted"). When absent,
    the caller routes the name through ``COMET_EXPERIMENT_NAME`` instead.
    """
    import inspect  # noqa: PLC0415 - one-shot introspection, wiring-only

    try:
        params = inspect.signature(CometLogger.__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtin/uninspectable
        return False
    return "experiment_name" in params


def _best_checkpoint(config_path: Path) -> str:
    """Best-epoch selection: lowest ``loss=`` next to the saved config. Scans
    both ``ckpts/*.ckpt`` and ``checkpoints/*.ckpt`` (Lightning's
    `ModelCheckpoint` default dirname) and picks the smallest embedded
    ``loss=<value>``. Raises `ConfigError` when none exist.
    """
    ckpt_dirs = [config_path.parent / name for name in ("ckpts", "checkpoints")]
    print(
        "salt test: no --ckpt_path specified, looking for best checkpoint in "
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
    print(f"salt test: using checkpoint {best}")
    return best


_CLASS_DICT_ARG = "class_dict"
"""Top-level convenience flag name (``--class_dict``)."""

_INIT_FROM_ARG = "init_from"
"""Top-level warm-start flag name (``--init_from``)."""

_CLASS_DICT_CLASS = "ClassificationTaskModule"
"""Class-name suffix of the only ``class_dict``/``weight_source`` consumer."""


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
    """Import a dotted ``module.Class`` path and return the class object.

    Pre-de-core ``salt.core.*`` paths (old checkpoints/configs) are remapped to
    their ``salt.*`` homes first (see ``_remap_class_path``).
    """
    module_path, _, attr = _remap_class_path(class_path).rpartition(".")
    module = import_module(module_path)
    return getattr(module, attr)


def _instantiate_class_config(cfg: Any) -> Any:
    """Instantiate a jsonargparse ``class_path``/``init_args`` config block.

    Mirrors jsonargparse's default subclass instantiation (`class_type(**init_args)`)
    for the deferred fit logger (`SaltCLI._reattach_fit_logger`) — a leaf ``cfg``
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
    """Whether a callback ``class_path`` names a TEST persistence sink: an
    `H5OutputWriter` (or subclass, or any callback exposing duck-typed
    ``writer_demand``), EXCLUDING `OnnxExportSink` (ONNX-only, persists
    nothing in TEST). False for an unimportable path or a plain callback.
    """
    import importlib  # noqa: PLC0415 - local, only on the test path

    from salt.outputs import (  # noqa: PLC0415 - avoid import cycle at top
        H5OutputWriter,
        OnnxExportSink,
    )

    module_path, _, attr = _remap_class_path(class_path).rpartition(".")
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
    """Whether the pre-instantiate ``callbacks:`` config carries a TEST
    persistence sink (any non-None entry passing `_is_persistence_sink`); used
    by the ``salt test`` writer-less check.
    """
    items = (callbacks or {}).items() if hasattr(callbacks, "items") else ()
    for entry in (val for _, val in items if val is not None):
        class_path = _entry_get(entry, "class_path") or ""
        if class_path and _is_persistence_sink(class_path):
            return True
    return False


def _fan_out_artifacts(cfg: Any) -> Any:
    """Fan ``--class_dict`` out onto the model-side consumers: for every module
    under ``model.init_args.modules``, a `ClassificationTaskModule` whose
    ``weight_source`` is unset/null gets
    ``weight_source := {"from_class_dict": <--class_dict>}`` (validated via
    `_checked_weight_source`); a task that already sets ``weight_source`` is
    left alone. Mutates `cfg` in place before validation/``--print_config``, so
    resolved values land in the saved run-dir config. No-op when unset.
    """
    from salt.model.modules.tasks import _checked_weight_source  # noqa: PLC0415 - torch-heavy, CLI-time

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


def _section_runs_mode(section: Mapping[str, Any], mode: Any) -> bool:
    """Whether any composed ``outputs:`` section writer runs in `mode`.

    Drives the implicit H5-sink wiring — a section with at least one
    TEST-mode writer needs the H5 persistence sink.
    """  # noqa: DOC201 - private helper, no Returns block per docstring policy
    return any(
        callable(getattr(writer, "runs_in_mode", None)) and writer.runs_in_mode(mode)
        for writer in section.values()
    )


def _section_produces_onnx(section: Mapping[str, Any]) -> bool:
    """Whether any ``RunTaskOutput`` in the section opts into ``export`` (ONNX).

    Gates the implicit ONNX-sink wiring: only a `RunTaskOutput` mints ONNX
    leaves (the manifest-only writers — input copies, pad masks — do not), so a
    section whose RunTaskOutputs are all ``modes: [test]`` assembles no ONNX
    tuple (matching a config that historically wired no OnnxExportSink).
    """  # noqa: DOC201 - private helper, no Returns block per docstring policy
    from salt.graph.spec import Mode  # noqa: PLC0415 - avoid import cycle at top

    for writer in section.values():
        is_rto = getattr(writer, "is_run_task_output", None)
        if callable(is_rto) and is_rto() and writer.runs_in_mode(Mode.ONNX):
            return True
    return False


def _iter_model_blocks(cfg: Any) -> list[tuple[Any, Any]]:
    """Pair each ``model`` namespace with the scope its ``--class_dict`` lives in
    — top-level on the run-free surface, subcommand-scoped on a trainer run.
    """
    blocks: list[tuple[Any, Any]] = []
    direct = cfg.get("model")
    if direct is not None:
        blocks.append((cfg, direct))
    for sub in SaltCLI.subcommands():
        sub_cfg = cfg.get(sub)
        sub_model = sub_cfg.get("model") if sub_cfg is not None else None
        if sub_model is not None:
            blocks.append((sub_cfg, sub_model))
    return blocks


class SaltCLI(LightningCLI):
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
        """No-op override disabling Lightning's ckpt-hparams re-parse:
        `SaltModule` saves hparams with ``ignore=["modules"]``, so re-parsing
        them would replace the saved config's modules-bearing model block and
        crash. The saved ``config.yaml`` is the sole source of model
        architecture; the checkpoint carries weights only.
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
            help="the export-ONLY half of the ONNX contract, consumed by `salt export` "
            "(design §5.1, §7): model_name (no '_'/'-', validated ONLY at export "
            "time), inputs (port/name/sequence/dyn_axis/alias) and the rename/combine "
            "manifest post-processing. The OUTPUT manifest derives from the outputs: "
            "section's export-mode selection — declaring export.outputs is a hard error "
            "at export time. Inert during fit/test; round-trips through saved run configs.",
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
        parser.add_argument(
            f"--{_INIT_FROM_ARG}",
            type=str | None,
            default=None,
            help="warm-start a fit from a pretrained checkpoint's weights (plan 01 W1). "
            "Unlike a resume --ckpt_path, trainer state stays fresh and the state dict is "
            "loaded prefix-filtered per module with strict per-module accounting: retained "
            "modules must be fully covered, config modules absent from the checkpoint are "
            "fresh-inited + materialised, and checkpoint modules absent from the config are "
            "dropped (logged). Mutually exclusive with --ckpt_path (fit subcommand only).",
        )
        if not self._run_mode:
            # run-free parses must round-trip a saved run config.yaml, which
            # carries the Lightning run-surface key ckpt_path; accept + ignore it.
            parser.add_argument(
                "--ckpt_path",
                type=str | None,
                default=None,
                help="accepted-and-ignored on the run-free parse surface so saved run "
                "configs round-trip into the salt graph tooling",
            )
        parser.link_arguments("name", "model.init_args.name")
        # the top-level outputs: section is NOT a link_arguments compute
        # (subclass-mode model targets grab the whole namespace) — it is
        # composed onto the instantiated model in `instantiate_classes` below.

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Assemble ``callbacks:`` dict values (YAML order, ``None`` filtered)
        ahead of the stock ``trainer.callbacks`` list. Drops the default
        ``lr_monitor`` LearningRateMonitor when no experiment logger is
        attached (it hard-raises on a logger-less trainer).
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
        """Instantiate, then compose the top-level ``outputs:`` section onto the
        model (folding section writers into the planning module dict) — done
        here rather than via ``link_arguments`` since a subclass-mode model
        link target grabs the whole namespace. Raises `ConfigError` if a live
        ``writers:`` block remains (removed; migrate to
        ``outputs:``/``callbacks:``).
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
        # hand the --init_from path to the instantiated model — its setup("fit")
        # runs the weights-only warm-start load after bind (plan 01 W1, design D4).
        init_from = self._get(self.config_init, _INIT_FROM_ARG)
        if init_from and model is not None and getattr(self.config, "subcommand", None) == "fit":
            model._init_from = str(init_from)  # noqa: SLF001 - same-package wiring
        composer = getattr(model, "compose_output_section", None) if model is not None else None
        if section and callable(composer):
            live_section = {k: w for k, w in section.items() if w is not None}
            composer(live_section)
            # plan 50 Phase B: the command wires the implicit per-command sinks
            # (test -> H5, export/graph -> ONNX) over the composed section, before
            # the bind loop — so a config declaring only WHAT (modules + modes)
            # gets the right sink without ever naming H5OutputSink/OnnxExportSink.
            self._inject_command_sinks(live_section)
            # bind the section to the sink callbacks NOW: datamodule setup runs
            # BEFORE model setup and resolves the sink's writer_demand, which
            # needs the section already bound.
            trainer = getattr(self, "trainer", None)
            for cb in (trainer.callbacks if trainer is not None else []):
                if callable(getattr(cb, "bind_output_section", None)):
                    cb.bind_output_section(model._output_section)  # noqa: SLF001 - same-package wiring

    def _inject_command_sinks(self, section: Mapping[str, Any]) -> None:
        """Wire the implicit per-command output sinks over the composed section.

        Plan 50 Phase B — the config declares WHAT (section modules + their
        ``modes:``); the command picks the sink:

        - ``salt test`` (subcommand ``test``): the H5 persistence sink, if any
          section writer runs in TEST.
        - The run-free parses (``salt graph``/``schema``/``export`` — all
          ``subcommand is None``, the caveat from plan 50a): BOTH the H5 sink
          (TEST section) and the ONNX sink (any RunTaskOutput running in
          ``export``), so the static tooling and the exporter see the same
          implicit sinks a real run would.
        - ``salt fit`` (subcommand ``fit``): no output sinks.

        A sink already present in ``trainer.callbacks`` (a programmatic build,
        or the MaskFormer ONNX escape hatch) is left alone — never double-wired.
        """
        from salt.graph.spec import Mode  # noqa: PLC0415 - avoid import cycle at top
        from salt.outputs import (  # noqa: PLC0415 - avoid import cycle at top
            H5OutputSink,
            OnnxExportSink,
        )

        subcommand = getattr(self.config, "subcommand", None)
        if subcommand == "fit":
            return
        trainer = getattr(self, "trainer", None)
        callbacks = getattr(trainer, "callbacks", None) if trainer is not None else None
        if callbacks is None:
            return

        def _present(cls: type) -> bool:
            return any(isinstance(cb, cls) for cb in callbacks)

        # H5 sink: any command that persists TEST predictions (test or run-free)
        if _section_runs_mode(section, Mode.TEST) and not _present(H5OutputSink):
            h5 = H5OutputSink()
            h5.name = "h5_output"
            callbacks.append(h5)
        # ONNX sink: only the run-free parses assemble the ONNX tuple, and only
        # when a RunTaskOutput opts into export (a test-only section mints none).
        if (
            subcommand is None
            and _section_produces_onnx(section)
            and not _present(OnnxExportSink)
        ):
            onnx = OnnxExportSink()
            onnx.name = "onnx_export"
            callbacks.append(onnx)

    def _detach_fit_logger(self) -> Any:
        """Stash + null the fit-stage ``trainer.logger`` ahead of the parser pass,
        so `instantiate_classes` builds a logger-less trainer and keeps the
        eager comet-thread `CometLogger.__init__` out of the racy validation
        window. Returns the un-instantiated logger config, or None outside
        ``fit``/with no logger configured.
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
        """Instantiate the deferred logger and attach it to the trainer, after
        the racy ``sys.modules`` walk in ``super().instantiate_classes()`` is
        done.
        """
        if logger_cfg is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        trainer.logger = _instantiate_class_config(logger_cfg)

    def before_instantiate_classes(self) -> None:
        """Per-stage config patches. On ``fit``: wires a configured experiment
        logger (name, offline mode, log dir); a no-op unless the user opts
        into a logger. On ``test``: disables the config dump + logger, globs
        the best checkpoint when ``--ckpt_path`` is unset, forces
        single-device eval, and refuses a sink-less config (TEST predictions
        would never be persisted). Raises `ConfigError` on a sink-less test
        config, an ambiguous config list without ``--ckpt_path``, or an
        explicit multi-device list.
        """
        subcommand = getattr(self.config, "subcommand", None)
        if subcommand == "fit":
            fit_cfg = self.config["fit"]
            if fit_cfg.get(_INIT_FROM_ARG) and fit_cfg.get("ckpt_path"):
                raise ConfigError(
                    "--init_from and --ckpt_path are mutually exclusive: --ckpt_path RESUMES "
                    "(restores trainer state, strict weight load, plan-hash gate enforced) "
                    "while --init_from WARM-STARTS a fresh run from a possibly surgically-"
                    "changed architecture (weights-only, per-module accounting). Pick one "
                    "(plan 01 W1, design D4)."
                )
            self._wire_experiment_logger(fit_cfg)
            return
        if subcommand != "test":
            return
        cfg = self.config["test"]
        self.save_config_callback = None  # no config.yaml dump on test
        cfg.trainer.logger = False
        # plan 50 Phase B: the H5 persistence sink is now IMPLICIT — the command
        # wires it in instantiate_classes over the top-level outputs: section. So
        # the writer-less guard checks for the section (the WHAT), not a
        # declared callbacks-level sink; a config still MAY declare its own sink
        # (programmatic / MaskFormer), which _inject_command_sinks leaves alone.
        has_callback_sink = _has_callback_persistence_sink(cfg.get("callbacks"))
        has_outputs_section = bool(cfg.get("outputs"))
        if not has_callback_sink and not has_outputs_section:
            raise ConfigError(
                "salt test needs an `outputs:` section to persist predictions — the "
                "command wires the H5 sink over it (plan 50 Phase B). Supply a top-level "
                "outputs: section (InputCopyWriter -> RunTaskOutput -> PadMaskWriter, in v1 "
                "H5 column order); use each RunTaskOutput's `modes:` list to control "
                "test-vs-export participation. See gn2v2-opendata.yaml."
            )
        if not cfg.get("ckpt_path"):
            configs = cfg.get("config") or []
            if len(configs) != 1:
                raise ConfigError(
                    "salt test without --ckpt_path needs exactly one --config (the saved "
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
                print("salt test: forcing --trainer.devices=1 (single-device eval, design §8)")
                cfg.trainer.devices = "1"
        elif isinstance(devices, list) and len(devices) > 1:
            raise ConfigError("salt test requires a single device (design §8, v1 cli.py:330)")

    @staticmethod
    def _wire_experiment_logger(cfg: Any) -> None:
        """Wire a configured fit-stage `CometLogger`: threads ``--name`` into
        ``experiment_name``/``dict_kwargs``, forces ``online=false`` with no
        ``COMET_API_KEY`` or under ``fast_dev_run``, and sets/creates
        ``COMET_OFFLINE_DIRECTORY`` at the trainer log dir. No-op for a
        non-Comet or unconfigured logger.
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
        print(f"salt fit artifacts: config.yaml in {log_dir}")
        print(
            f"salt fit artifacts: checkpoints in {ckpt_dir}"
            if ckpt_dir
            else "salt fit artifacts: no checkpoint callback configured"
        )
        print("(run-directory layout with timestamped names lands in M6 — design §5)")


def main(args: Sequence[str] | None = None) -> int:
    """``salt`` console entry point.

    ``salt graph``/``schema``/``mup-shapes``/``mup-coord-check`` dispatch to
    the static graph + muP tooling (`salt.cli.main`), ``salt export``
    to the ONNX exporter, and ``salt inference`` to the eager export-set
    runner; everything else goes to `SaltCLI` (``salt fit``/``test``).
    Graph errors (`GraphError`) print as a clean one-block form on stderr
    instead of a Python traceback.

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
        from salt.onnx import export as onnx_export  # noqa: PLC0415 - heavy, export-only

        return onnx_export.main(argv[1:])
    if argv and argv[0] == _INFERENCE_COMMAND:
        # trainer-free like export: inference executes the export-mode plan
        # eagerly per jet (plan 50 Phase D), so it dispatches to its own main
        # rather than a Lightning Trainer subcommand.
        from salt import inference as inference_cli  # noqa: PLC0415 - heavy, eager-only

        return inference_cli.main(argv[1:])
    help_requested = bool(argv) and argv[0] in {"-h", "--help"}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            SaltCLI(args=None if args is None else argv)
    except SystemExit:
        if help_requested:
            print(
                "\nsee also: 'salt graph --help' (static graph tooling: validate/plan/plot/"
                "why/deadcode/resolve, design §4), 'salt schema --help' (schema artifacts, "
                "§2.6), 'salt mup-shapes --help' / 'salt mup-coord-check --help' (muP base/"
                "delta infshapes + coord-check, design §3.4), 'salt export --help' (ONNX "
                "export, §7; --manifest prints the writer-derived output manifest), "
                "'salt inference --help' (label-free eager inference: the export output "
                "set written to H5, plan 50)"
            )
        raise
    except GraphError as err:
        # one-block form — no Python traceback for config errors
        print(f"salt.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
