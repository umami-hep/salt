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
Comet logger wiring and run-dir timestamping are M6; ``base2.yaml`` ships
``logger: false``.
"""

from __future__ import annotations

import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import comet_ml  # noqa: F401 - import-order contract: comet before lightning (v1 main.py:5, §5)
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
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

_GRAPH_COMMANDS = frozenset({"graph", "schema"})
_EXPORT_COMMAND = "export"

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
        """Parse args with ``--<modules-dict>.X=null`` normalised (design §5.3).

        Returns
        -------
        Any
            The parsed namespace (`jsonargparse` semantics unchanged).
        """
        if args is None:
            args = sys.argv[1:]
        if isinstance(args, Sequence) and not isinstance(args, str):
            args = [_normalise_module_null(arg) if isinstance(arg, str) else arg for arg in args]
        return super().parse_args(args, *pargs, **kwargs)


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
            help="the declarative ONNX export block consumed by `salt2 export` (design §5.1, "
            "§7): model_name (no '_'/'-', validated ONLY at export time), inputs "
            "(port/name/sequence/dyn_axis/alias) and outputs (port/name|names/dtype/reduce). "
            "Inert during fit/test; round-trips through saved run configs.",
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

        Returns
        -------
        Trainer
            The instantiated trainer.
        """
        callbacks_dict = self._get(self.config_init, "callbacks") or {}
        assembled = [cb for cb in callbacks_dict.values() if cb is not None]
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
        """``salt2 test`` ergonomics — the v1 eval surface kept (utils/cli.py:312-332).

        No resolved-config dump and no experiment logger on eval runs; a
        missing ``--ckpt_path`` triggers the v1 best-checkpoint glob (which
        requires exactly ONE user ``--config``, the saved run config next to
        ``ckpts/`` or ``checkpoints/``); multi-device eval is rejected/forced
        to one device; a
        writer-less eval is refused up front (TEST predictions would be
        computed and never persisted — design §4.2, §8).

        Raises
        ------
        ConfigError
            On a writer-less test config, an ambiguous config list without
            ``--ckpt_path``, or an explicit multi-device list.
        """
        if getattr(self.config, "subcommand", None) != "test":
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

    ``salt2 graph …`` / ``salt2 schema …`` dispatch to the static graph
    tooling (`salt.core.cli.main`, design §4) and ``salt2 export`` to the
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
                "why/deadcode, design §4), 'salt2 schema --help' (schema artifacts, §2.6) and "
                "'salt2 export --help' (ONNX export, §7)"
            )
        raise
    except GraphError as err:
        # the §4.1 one-block form — no Python traceback for config errors
        print(f"salt.core.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
