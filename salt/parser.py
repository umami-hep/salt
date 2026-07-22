"""`DeepMergeParser` — the salt argument parser with cross-config dict-merge semantics."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from typing import Any

from lightning.pytorch.cli import LightningArgumentParser

from salt.graph.errors import ConfigError

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
    dicts, `SaltCLI` filters the callbacks dict.
    """

    def merge_config(self, cfg_from: Any, cfg_to: Any) -> Any:
        """Union dict-typed leaves key-by-key before the standard merge."""
        print("=DBG merge_config CALLED type=", type(self).__name__, flush=True)  # noqa: T201
        for _k, _v in list(cfg_from.items()):
            print("=DBG from", repr(_k), type(_v).__name__, flush=True)  # noqa: T201
        for key, val_from in list(cfg_from.items()):
            if not isinstance(val_from, dict):
                continue
            val_to = cfg_to.get(key)
            print("=DBG dict-key", repr(key), "val_to", type(val_to).__name__, flush=True)  # noqa: T201
            if isinstance(val_to, dict):
                cfg_from[key] = {**val_to, **val_from}
        return super().merge_config(cfg_from, cfg_to)

    def parse_args(self, args: Sequence[str] | None = None, *pargs: Any, **kwargs: Any) -> Any:
        """Parse args with ``--…modules.X=null`` normalised to the JSON-block form,
        then fan out ``--class_dict`` onto model-side consumers
        (`_fan_out_artifacts`) after the deep-merge but before validation/
        ``--print_config``, so resolved values freeze into the saved run config.
        """
        # W45.2c import-placement fix: _fan_out_artifacts stays in salt.main
        # (it resolves SaltCLI subcommand scopes) and main imports this parser,
        # so a module-top import here would be a parser<->main cycle.
        from salt.main import _fan_out_artifacts  # noqa: PLC0415

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


def _normalise_module_null(arg: str) -> str:
    """Rewrite ``--…modules.X=null`` to the JSON-block form (see
    `_MODULE_DICT_NULL`); returns `arg` unchanged otherwise.
    """
    match = _MODULE_DICT_NULL.match(arg)
    if match is None:
        return arg
    return f'--{match["parent"]}={{"{match["key"]}": null}}'


# print_config flag -> ArgumentParser.dump kwarg map (mirrors jsonargparse's
# _ActionPrintConfig flag vocabulary; "skip_null" is "skip_none" on dump)
_PRINT_CONFIG_FLAGS = {"skip_default": "skip_default", "skip_null": "skip_none"}


def _dump_kwargs(flags: str) -> dict[str, bool]:
    """Translate a ``--print_config=<flags>`` value to `ArgumentParser.dump`
    kwargs, mirroring jsonargparse's ``_ActionPrintConfig`` flag handling.
    Raises `ConfigError` on an unrecognised flag.
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
