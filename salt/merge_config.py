"""``salt merge-config``: take the same arguments as ``salt fit`` (stacked
``--config`` files + deep CLI overrides), emit the fully-merged config YAML,
and render one static model-graph plot per ``training_schedule`` stage with the
stage's frozen modules visually distinguished.

Trainer-free and data-free: the merged YAML is the ``salt fit --print_config``
dump (same `DeepMergeParser` surface, `base2.yaml` defaults, fan-out + schedule
relocation), and the per-stage graphs are the FIT-mode plan overlaid with each
stage's freeze mask. ``--init_from``/``ckpt_path`` are accepted for fit-parity
but only echoed into the merged YAML — no checkpoint is read.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from salt.graph.errors import ConfigError
from salt.graph.planner import compile_plan, deadcode
from salt.graph.render import dot_source
from salt.graph.spec import Mode
from salt.schedule import TrainingSchedule

__all__ = ["main"]

_HELP = """usage: salt merge-config [salt fit args ...] --merged.output <path>
                         [--merged.plots true|false]

Merge the given salt-fit config stack (repeated --config + deep CLI overrides)
and write:
  * <path>                              the fully-merged config YAML
  * <stem>_stage{NN}_{name}.dot / .png  one graph per training_schedule stage
                                        (NN = 0-based execution order), with the
                                        stage's frozen modules greyed + badged

Options:
  --merged.output <path>   (required) merged-YAML path; plots are its siblings
  --merged.plots <bool>    rasterise the per-stage .dot to .png/.pdf via
                           Graphviz (default: true). false = write .dot only
                           (no `dot` binary needed).

Everything else is passed through verbatim to the salt fit parser, so the
merged YAML is byte-identical to `salt fit [same args] --print_config`.
"""


def main(args: Sequence[str] | None = None) -> int:
    """``salt merge-config`` entry point (dispatched from `salt.main.main`).

    A `ConfigError` (bad ``--merged.*`` option / fit-parse failure) or a
    `GraphError` (e.g. a missing ``dot`` binary) propagates to `salt.main.main`,
    which prints it as a clean one-block message.

    Returns
    -------
    int
        ``0`` on success.
    """
    argv = list(sys.argv[1:] if args is None else args)
    if any(a in {"-h", "--help"} for a in argv):
        print(_HELP)
        return 0

    output_path, do_plots, fit_args = _split_merged_args(argv)

    merged_text = _dump_merged_config(fit_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged_text)
    print(f"wrote merged config to {output_path}")

    _write_stage_plots(output_path, merged_text, do_plots=do_plots)
    return 0


def _split_merged_args(argv: list[str]) -> tuple[Path, bool, list[str]]:
    """Pull ``--merged.output`` (required) and ``--merged.plots`` (default true)
    out of `argv`, returning ``(output_path, do_plots, remaining_fit_args)``.

    Both the ``--flag=value`` and ``--flag value`` forms are accepted; the
    remaining args are forwarded verbatim to the salt fit parser.
    """  # noqa: DOC201, DOC501
    output: str | None = None
    plots = True
    fit_args: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        key, sep, inline = arg.partition("=")
        if key == "--merged.output":
            output, i = _consume_value(argv, i, inline if sep else None, "--merged.output")
        elif key == "--merged.plots":
            raw, i = _consume_value(argv, i, inline if sep else None, "--merged.plots")
            plots = _as_bool(raw)
        else:
            fit_args.append(arg)
            i += 1
    if output is None:
        raise ConfigError(
            "salt merge-config requires --merged.output <path> (the merged-YAML path)"
        )
    return Path(output), plots, fit_args


def _consume_value(
    argv: list[str], i: int, inline: str | None, flag: str
) -> tuple[str, int]:
    """Resolve `flag`'s value from the inline ``=value`` or the next token,
    returning ``(value, next_index)``.
    """  # noqa: DOC201, DOC501
    if inline is not None:
        return inline, i + 1
    if i + 1 >= len(argv):
        raise ConfigError(f"{flag} expects a value")
    return argv[i + 1], i + 2


def _as_bool(raw: str) -> bool:
    """Parse a ``--merged.plots`` boolean (true/false, 1/0, yes/no)."""  # noqa: DOC201, DOC501
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ConfigError(f"--merged.plots expects a boolean (true/false), got {raw!r}")


def _dump_merged_config(fit_args: list[str]) -> str:
    """The fully-merged config YAML — the ``salt fit --print_config`` dump for
    `fit_args`, produced through the real `SaltCLI` parser (run-free, trainer- and
    data-free: ``--print_config`` dumps and exits before any instantiation).
    """  # noqa: DOC201, DOC501
    import warnings  # noqa: PLC0415

    from salt.main import SaltCLI  # noqa: PLC0415 - heavy/circular (module docstring)

    buffer = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(buffer),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            SaltCLI(args=[*fit_args, "--print_config"], run=False)
    except SystemExit as err:
        if err.code not in (0, None):
            raise ConfigError(
                "salt merge-config: the fit config stack failed to parse "
                f"(parser exit {err.code}; the error is printed above)"
            ) from err
    text = buffer.getvalue()
    if not text.strip():
        raise ConfigError("salt merge-config: the fit parser produced no merged config")
    return text


def _write_stage_plots(output_path: Path, merged_text: str, *, do_plots: bool) -> None:
    """Render one graph per `training_schedule` stage next to `output_path`.

    The merged config re-loads as a full-pipeline `GraphConfig` (data + model
    modules + sinks); the FIT-mode plan is compiled once and re-styled per stage
    with the stage's frozen mask. A ``.dot`` is always written; each is
    rasterised to ``.png``/``.pdf`` only when `do_plots` is set.
    """  # noqa: DOC501
    from salt.cli import _resolve_widths, load_config  # noqa: PLC0415 - heavy/circular

    merged = yaml.safe_load(merged_text)
    schedule = _schedule_from_merged(merged)

    cfg = load_config([str(output_path)])
    if err := cfg.mode_errors.get(Mode.FIT):
        raise ConfigError(f"salt merge-config: the FIT-mode plan does not compile: {err}")
    plan = compile_plan(
        cfg.modules,
        Mode.FIT,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(Mode.FIT),
    )
    findings = deadcode(cfg.modules, Mode.FIT, cfg.sources, cfg.schema, cfg.sinks)
    pruned = sorted({finding.module for finding in findings if finding.key == "*"})
    widths = _resolve_widths(cfg)

    stem = output_path.with_suffix("")
    n_stages = len(schedule.stages)
    for index, stage in enumerate(schedule.stages):
        frozen = frozenset(schedule.frozen_names(stage))
        title = _stage_title(index, n_stages, stage.name, frozen)
        dot_text = dot_source(plan, cfg.modules, pruned, widths=widths, frozen=frozen, title=title)
        dot_path = Path(f"{stem}_stage{index:02d}_{stage.name}.dot")
        dot_path.write_text(dot_text)
        print(f"wrote DOT to {dot_path}")
        if do_plots:
            from salt.cli import _render_with_dot  # noqa: PLC0415 - heavy/circular

            _render_with_dot(dot_path, dot_path.with_suffix(".png"))


def _schedule_from_merged(merged: Any) -> TrainingSchedule:
    """Build the `TrainingSchedule` the merged config describes.

    Freeze specs range over the model's ``modules`` names (in declaration order,
    the exact set `SaltModule` validates against); a config with no top-level
    ``training_schedule`` desugars to the single ``fit`` stage.
    """  # noqa: DOC201, DOC501
    try:
        module_names = list(merged["model"]["init_args"]["modules"].keys())
    except (TypeError, KeyError, AttributeError) as err:
        raise ConfigError(
            "salt merge-config: merged config has no model.init_args.modules block"
        ) from err
    raw_schedule = merged.get("training_schedule")
    if raw_schedule is None:
        return TrainingSchedule.desugar_legacy(module_names)
    return TrainingSchedule.from_config(raw_schedule, module_names)


def _stage_title(index: int, total: int, name: str, frozen: frozenset[str]) -> str:
    """The caption for a stage graph: position, name, and its frozen module set."""  # noqa: DOC201
    frozen_str = ", ".join(sorted(frozen)) if frozen else "(none)"
    return f"stage {index + 1}/{total}: {name} — frozen: {frozen_str}"
