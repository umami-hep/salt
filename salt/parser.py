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

# the top-level staged-training schedule key (kept in sync with
# salt.main._TRAINING_SCHEDULE_ARG; duplicated here to avoid a parser<->main
# import cycle at module load). Its nested {stages: {name: {...}}} content needs a
# RECURSIVE deep-merge across stacked config files — the shallow dict-leaf union
# below would replace the whole `stages` dict wholesale (plain dict[str, Any] has
# no per-entry subclass adapter), losing D1's per-stage-by-name merge.
_TRAINING_SCHEDULE_KEY = "training_schedule"


def _extract_schedule_cli_overrides(
    args: list[Any],
) -> tuple[list[Any], list[tuple[str, Any]]]:
    """Split ``--training_schedule.<dotted.path>[=<value>]`` (or the two-token
    ``--training_schedule.<path> <value>``) CLI args out of `args`.

    jsonargparse does not split multi-level dotted keys into a plain
    ``dict[str, Any]`` (it keeps the tail as a literal flat key and replaces the
    base value wholesale), so these deep overrides are pulled out here and applied
    as a controlled deep-merge in `_relocate_training_schedule` instead. The
    whole-value forms (``--training_schedule=<json>`` / ``--training_schedule
    <json>``) do NOT start with the ``--training_schedule.`` prefix and pass
    through untouched. Returns the remaining args plus ``(dotted_path, raw_value)``
    pairs.
    """
    prefix = f"--{_TRAINING_SCHEDULE_KEY}."
    kept: list[Any] = []
    overrides: list[tuple[str, Any]] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if isinstance(arg, str) and arg.startswith(prefix):
            body = arg[len(prefix) :]
            if "=" in body:
                path, value = body.split("=", 1)
                overrides.append((path, value))
                i += 1
            elif i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                overrides.append((body, args[i + 1]))
                i += 2
            else:  # a bare --training_schedule.key with no value: treat as null
                overrides.append((body, None))
                i += 1
        else:
            kept.append(arg)
            i += 1
    return kept, overrides


def _deep_merge_dicts(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base``: nested dicts merge key-by-key;
    any scalar / list / ``None`` (the stage-name null-delete idiom, filtered at
    assembly by `TrainingSchedule.from_config`) replaces.
    """
    merged = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


_CONFIG_FLAGS = ("--config", "-c")


def _expand_config_includes(args: list[Any]) -> list[Any]:
    """Rewrite every ``--config``/``-c`` value to its include-expanded form.

    ``include:`` is a salt-level key, so it has to be resolved and stripped
    before jsonargparse reads the file. Configs without includes are passed
    through untouched.
    """
    from salt.config_utils import (
        expand_includes,
    )  # local import: avoids a parser<->config_utils cycle

    out: list[Any] = []
    expect_value = False
    for arg in args:
        if expect_value and isinstance(arg, str):
            out.append(expand_includes(arg))
            expect_value = False
            continue
        expect_value = False
        if isinstance(arg, str):
            if arg in _CONFIG_FLAGS:
                expect_value = True
            elif arg.startswith("--config="):
                out.append("--config=" + expand_includes(arg.split("=", 1)[1]))
                continue
        out.append(arg)
    return out


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
    assembly time everywhere: `SaltModule`/`SaltDataModule` filter module
    dicts, `SaltCLI` filters the callbacks dict.
    """

    def merge_config(self, cfg_from: Any, cfg_to: Any) -> Any:
        """Union dict-typed leaves key-by-key before the standard merge.

        The top-level ``training_schedule`` leaf is RECURSIVELY deep-merged instead
        (its stage names live one level down under ``stages:``), so stacked configs
        override per-stage-by-name rather than replacing the whole schedule.
        """
        for key, val_from in list(cfg_from.items()):
            if not isinstance(val_from, dict):
                continue
            val_to = cfg_to.get(key)
            if isinstance(val_to, dict):
                cfg_from[key] = (
                    _deep_merge_dicts(val_to, val_from)
                    if key == _TRAINING_SCHEDULE_KEY
                    else {**val_to, **val_from}
                )
        return super().merge_config(cfg_from, cfg_to)

    def parse_args(self, args: Sequence[str] | None = None, *pargs: Any, **kwargs: Any) -> Any:
        """Parse args with ``--…modules.X=null`` normalised to the JSON-block form,
        then fan out ``--class_dict`` onto model-side consumers
        (`_fan_out_artifacts`) and relocate the top-level ``training_schedule:``
        into the `SaltModule` constructor arg (`_relocate_training_schedule`,
        rejecting the retired nested home) — both after the deep-merge but before
        validation/``--print_config``, so resolved values freeze into the saved
        run config.
        """
        # Import placement: _fan_out_artifacts stays in salt.main
        # (it resolves SaltCLI subcommand scopes) and main imports this parser,
        # so a module-top import here would be a parser<->main cycle.
        from salt.main import (
            _fan_out_artifacts,
            _relocate_training_schedule,
        )

        if args is None:
            args = sys.argv[1:]
        print_config_flags: str | None = None
        schedule_overrides: list[tuple[str, Any]] = []
        if isinstance(args, Sequence) and not isinstance(args, str):
            args = _expand_config_includes(list(args))
            args, schedule_overrides = _extract_schedule_cli_overrides(list(args))
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
        _relocate_training_schedule(cfg, schedule_overrides)
        if not caller_skips:
            self.validate(cfg)
        if print_config_flags is not None:
            sys.stdout.write(self.dump(cfg, **_dump_kwargs(print_config_flags)))
            self.exit(0)
        return cfg

    def dump(self, *args: Any, **kwargs: Any) -> str:
        """Serialise via the base parser, then normalise key order so every
        ``class_path``/``init_args`` mapping lists ``class_path`` first.

        A stacked config that overrides only a subclass's ``init_args`` makes
        jsonargparse emit that mapping ``init_args``-first; this is the single
        serialization seam shared by ``--print_config``, ``salt merge-config``,
        and the ``config.yaml`` a fit run saves, so fixing it here fixes all
        three. Serialization order only — the reparsed object is unchanged.
        """
        text = super().dump(*args, **kwargs)
        return _class_path_before_init_args(text) if isinstance(text, str) else text


# jsonargparse dumps block mappings as ``<indent><key>:`` — a scalar carries its
# value after the colon, a nested block ends the line. List items (``- ...``) and
# comment lines never match, so they are skipped by the reorder pass.
_YAML_KEY = re.compile(r"^(?P<indent> *)(?P<key>[\w-]+):(?:\s|$)")


def _class_path_before_init_args(text: str) -> str:
    """Hoist each block mapping's ``class_path:`` scalar line above its sibling
    ``init_args:`` line in a dumped YAML — a pure serialization-order fix.

    jsonargparse emits an override that touches only a subclass's ``init_args``
    with ``init_args`` first; this moves the one-line ``class_path`` scalar above
    it so every pair reads class-path-first (remaining keys keep their relative
    order). Idempotent; only block-mapping pairs (the salt module/callback/output/
    logger form) are touched — list-item subclasses are left as-is.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = _YAML_KEY.match(lines[i])
        if match is not None and match.group("key") == "class_path":
            target = _sibling_init_args_before(lines, i, len(match.group("indent")))
            if target is not None:
                lines.insert(target, lines.pop(i))
        i += 1
    return "\n".join(lines)


def _sibling_init_args_before(lines: list[str], idx: int, indent: int) -> int | None:
    """Index of the same-``indent`` sibling ``init_args:`` line preceding the
    ``class_path:`` at `idx` in the same parent mapping, or None when
    ``class_path`` already precedes ``init_args`` (scan stops at the first
    shallower key — the parent — so an ancestor ``init_args`` is never matched).
    """
    j = idx - 1
    while j >= 0:
        match = _YAML_KEY.match(lines[j])
        if match is not None:
            here = len(match.group("indent"))
            if here < indent:
                return None
            if here == indent and match.group("key") == "init_args":
                return j
        j -= 1
    return None


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
