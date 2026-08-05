"""Profiling harnesses: a dataset-side ``line_profiler`` CLI and a model-side
`torch.profiler` callback, each with a ``salt profile`` subcommand.

``salt profile dataset`` runs the dataset plan in-process and prints the classic
annotated per-line report; ``salt profile model`` runs a short capped fit with
`TorchProfilerCallback` attached, which records a steady-state window of
training steps and writes per-op tables, source stacks and a Chrome trace. Both
subcommands take ``--steps``. See ``docs/profiling.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import operator
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch.callbacks import Callback

from salt.graph.executor import STEP_SCOPE_PREFIX, record_steps
from salt.logging import console

__all__ = [
    "DEFAULT_DATASET_FUNCTIONS",
    "DEFAULT_STEPS",
    "PlanStepScopes",
    "TorchProfilerCallback",
    "main",
    "profile_dataset",
    "profile_model",
    "resolve_schedule",
]

DEFAULT_STEPS = 100
"""Train batches both ``salt profile`` subcommands consume unless ``--steps`` says otherwise."""

# The capture window `resolve_schedule` aims for; everything left over becomes
# `wait`, so the recorded steps are the LAST ones of the run — the most
# steady-state part of the budget the caller paid for.
_DEFAULT_ACTIVE = 20
_DEFAULT_WARMUP = 5

DEFAULT_DATASET_FUNCTIONS: tuple[str, ...] = (
    "salt.data.dataset.GraphDataset.__getitem__",
    "salt.data.dataset.GraphDataset._to_torch",
    "salt.data.readers.reader.H5StructuredReader.read",
    "salt.data.readers.reader.H5StructuredReader._read_kept",
    "salt.data.readers.cuts.ConstituentCuts.apply",
    "salt.data.processors.features.Features.process",
    "salt.data.processors.labels.Labels.process",
    "salt.graph.bundle.Bundle.merge",
)
"""The read path profiled by ``salt profile dataset`` unless ``--functions`` overrides it."""

_BATCH_SCOPE = "salt.batch"
_BACKWARD_SCOPE = "salt.backward"
_AUTOGRAD_PREFIX = "autograd::engine::evaluate_function"
_OPTIMIZER_PREFIX = "Optimizer.step#"

_LINE_PROFILER_HINT = (
    "line_profiler is not installed — it is an optional dependency: "
    "pip install 'salt-ml[profile]' (or pip install line-profiler)"
)


# --------------------------------------------------------------------------- #
# model side
# --------------------------------------------------------------------------- #


def _time_us(event: Any, *names: str) -> float:
    """First non-None timing attribute among `names` (torch renamed cuda_* to device_*)."""
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _device_total(event: Any) -> float:
    """Inclusive device (GPU) time of one averaged event, in microseconds."""
    return _time_us(event, "device_time_total", "cuda_time_total")


def _self_device_total(event: Any) -> float:
    """Self (exclusive) device time of one averaged event, in microseconds."""
    return _time_us(event, "self_device_time_total", "self_cuda_time_total")


def _first_supported_sort_key(events: Any, candidates: Sequence[str]) -> str | None:
    """First `candidates` entry ``EventList.table`` accepts for these events."""
    for key in candidates:
        with contextlib.suppress(Exception):
            events.table(sort_by=key, row_limit=1)
            return key
    return None


class PlanStepScopes(Callback):
    """Wrap each plan step in a ``salt.step/<name>`` profiler scope for the whole run.

    Enables `salt.graph.executor.record_steps` from ``on_fit_start`` to
    ``on_fit_end``, so ANY active `torch.profiler` capture — Lightning's
    ``PyTorchProfiler`` or `TorchProfilerCallback` — attributes time to the
    configured module instances. The scope is outside the module call, so
    ``--compile`` is unaffected.

    Use it only while profiling: it is a no-op without a profiler attached,
    but it is not part of a production training config.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scope: Any = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Enter the step-scope context for the fit run."""
        del trainer, pl_module
        if self._scope is None:
            self._scope = record_steps()
            self._scope.__enter__()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Leave the step-scope context."""
        del trainer, pl_module
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None


class TorchProfilerCallback(Callback):
    """Capture a steady-state window of training steps with `torch.profiler`.

    Wraps `torch.profiler.profile` with a ``wait``/``warmup``/``active``
    schedule so the capture skips startup (and, under ``--compile``, the
    compilation step), then writes four artifacts under `dirpath` with the
    `tag` prefix:

    - ``<tag>_key_averages.txt`` — per-op table sorted by device time.
    - ``<tag>_stacks.txt`` + ``<tag>_stacks.flame`` — source attribution
      (``with_stack`` only).
    - ``<tag>_trace.json.gz`` — the Chrome trace.
    - ``<tag>_summary.json`` — machine-readable buckets: per-plan-step forward
      time (from `PlanStepScopes`), backward, optimizer, top ops, and the
      it/s measured *inside* the profiled window (divide the unprofiled rate
      by this to get the profiler overhead).

    Enables the plan-step scopes itself, so it needs no companion callback.
    Profiled rates are for attribution only — never quote them as throughput.

    Parameters
    ----------
    dirpath : str | Path
        Output directory (created if missing).
    tag : str, optional
        Filename prefix identifying the capture, by default ``"profile"``.
    wait : int, optional
        Steps skipped before the capture, by default 5.
    warmup : int, optional
        Steps traced but discarded (CUPTI warm-up), by default 5.
    active : int, optional
        Steps recorded, by default 20.
    row_limit : int, optional
        Rows in the per-op table, by default 30.
    with_stack : bool, optional
        Record Python/C++ source stacks (source attribution, higher
        overhead), by default True.
    profile_memory : bool, optional
        Record allocator events, by default True.
    record_shapes : bool, optional
        Record operator input shapes, by default False.
    stack_depth : int, optional
        Stack frames kept when grouping the stack table, by default 5.
    """

    def __init__(
        self,
        dirpath: str | Path,
        tag: str = "profile",
        wait: int = 5,
        warmup: int = 5,
        active: int = 20,
        row_limit: int = 30,
        with_stack: bool = True,
        profile_memory: bool = True,
        record_shapes: bool = False,
        stack_depth: int = 5,
    ) -> None:
        super().__init__()
        if active < 1:
            raise ValueError(f"TorchProfilerCallback needs active >= 1, got {active}")
        self.dirpath = Path(dirpath)
        self.tag = tag
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.row_limit = row_limit
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.stack_depth = stack_depth
        self._prof: Any = None
        self._scope: Any = None
        self._batch_scope: Any = None
        self._backward_scope: Any = None
        self._seen = 0
        self._window_start: float | None = None
        self._window_wall: float | None = None
        self._written = False

    @property
    def total_steps(self) -> int:
        """Steps consumed by the schedule before the capture closes."""
        return self.wait + self.warmup + self.active

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Start the profiler and the plan-step scopes."""
        del trainer, pl_module
        if self._prof is not None:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait, warmup=self.warmup, active=self.active, repeat=1
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._scope = record_steps()
        self._scope.__enter__()
        self._prof.start()

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
    ) -> None:
        """Open the per-batch scope and start the window clock on the first active step."""
        del trainer, pl_module, batch, batch_idx
        if self._prof is None:
            return
        if self._seen == self.wait + self.warmup:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._window_start = time.perf_counter()
        self._batch_scope = torch.profiler.record_function(_BATCH_SCOPE)
        self._batch_scope.__enter__()

    def on_before_backward(self, trainer: Any, pl_module: Any, loss: Any) -> None:
        """Open the backward scope (paired with `on_after_backward`)."""
        del trainer, pl_module, loss
        if self._prof is None or self._backward_scope is not None:
            return
        self._backward_scope = torch.profiler.record_function(_BACKWARD_SCOPE)
        self._backward_scope.__enter__()

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Close the backward scope."""
        del trainer, pl_module
        if self._backward_scope is not None:
            self._backward_scope.__exit__(None, None, None)
            self._backward_scope = None

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Close the per-batch scope, advance the schedule and stop once the window closes."""
        del trainer, pl_module, outputs, batch, batch_idx
        if self._prof is None:
            return
        if self._backward_scope is not None:  # unpaired hook (defensive)
            self._backward_scope.__exit__(None, None, None)
            self._backward_scope = None
        if self._batch_scope is not None:
            self._batch_scope.__exit__(None, None, None)
            self._batch_scope = None
        self._seen += 1
        if self._seen == self.total_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._window_start is not None:
                self._window_wall = time.perf_counter() - self._window_start
        self._prof.step()  # fires _on_trace_ready at the end of the active window
        if self._seen >= self.total_steps:
            self._teardown()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Stop the profiler if the run ended before the window closed."""
        del trainer, pl_module
        self._teardown()

    def _teardown(self) -> None:
        """Stop the profiler and leave the step scopes (idempotent)."""
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None
        if self._prof is not None:
            with contextlib.suppress(Exception):
                self._prof.stop()
            self._prof = None

    def _on_trace_ready(self, prof: Any) -> None:
        """Write the tables, stacks, trace and summary for the completed window."""
        if self._written:
            return
        self._written = True
        self.dirpath.mkdir(parents=True, exist_ok=True)
        events = prof.key_averages()
        sort_key = _first_supported_sort_key(
            events, ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total")
        )
        table_kwargs: dict[str, Any] = {"row_limit": self.row_limit}
        if sort_key is not None:
            table_kwargs["sort_by"] = sort_key
        (self.dirpath / f"{self.tag}_key_averages.txt").write_text(events.table(**table_kwargs))
        stacks = self._write_stacks(prof, events, sort_key) if self.with_stack else False
        self._write_trace(prof)
        summary = self._summarise(events, sort_key)
        summary["stacks_available"] = stacks
        (self.dirpath / f"{self.tag}_summary.json").write_text(json.dumps(summary, indent=2))
        console(f"[TorchProfilerCallback] wrote {self.tag}_* artifacts to {self.dirpath}")

    def _write_stacks(self, prof: Any, events: Any, sort_key: str | None) -> bool:
        """Write the stack-grouped table and flamegraph file; False when the build
        recorded no stacks.

        ``with_stack=True`` is a request, not a guarantee — some torch builds
        return events with empty ``stack`` lists, in which case
        ``group_by_stack_n`` degenerates to the plain per-op table and
        ``export_stacks`` writes an empty file. Say so in a note rather than
        leaving a duplicate table and a 0-byte flamegraph to be misread as
        source attribution.
        """
        if not any(getattr(event, "stack", None) for event in events):
            (self.dirpath / f"{self.tag}_stacks.txt").write_text(
                "torch recorded no python stacks for this run despite with_stack=True — "
                "this build does not populate FunctionEvent.stack, so there is no source "
                "attribution to report. The salt.step/<module> rows in "
                f"{self.tag}_key_averages.txt carry the per-module attribution instead.\n"
            )
            return False
        with contextlib.suppress(Exception):
            grouped = prof.key_averages(group_by_stack_n=self.stack_depth)
            kwargs: dict[str, Any] = {"row_limit": self.row_limit}
            if sort_key is not None:
                kwargs["sort_by"] = sort_key
            (self.dirpath / f"{self.tag}_stacks.txt").write_text(grouped.table(**kwargs))
        # torch renamed the CUDA metrics to device_*; try the modern spelling
        # first and keep going until one writes something (an accepted-but-empty
        # metric name silently produces a 0-byte file).
        flame = self.dirpath / f"{self.tag}_stacks.flame"
        for metric in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
            with contextlib.suppress(Exception):
                prof.export_stacks(str(flame), metric)
                if flame.stat().st_size > 0:
                    break
        return True

    def _write_trace(self, prof: Any) -> None:
        """Export the Chrome trace, gzipped (traces with stacks are large)."""
        raw = self.dirpath / f"{self.tag}_trace.json"
        with contextlib.suppress(Exception):
            prof.export_chrome_trace(str(raw))
            with raw.open("rb") as src, gzip.open(f"{raw}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()

    def _summarise(self, events: Any, sort_key: str | None) -> dict[str, Any]:
        """Aggregate the window into forward-per-step / backward / optimizer buckets."""
        steps: dict[str, dict[str, float]] = {}
        backward_us = optimizer_us = batch_us = autograd_us = 0.0
        total_self_device = 0.0
        for event in events:
            key = str(event.key)
            device_total = _device_total(event)
            total_self_device += _self_device_total(event)
            if key.startswith(STEP_SCOPE_PREFIX):
                steps[key[len(STEP_SCOPE_PREFIX) :]] = {
                    "device_us": device_total,
                    "self_cpu_us": _time_us(event, "self_cpu_time_total"),
                    "cpu_us": _time_us(event, "cpu_time_total"),
                    "count": float(getattr(event, "count", 0)),
                }
            elif key == _BACKWARD_SCOPE:
                backward_us = device_total
            elif key == _BATCH_SCOPE:
                batch_us = device_total
            elif key.startswith(_AUTOGRAD_PREFIX):
                autograd_us += device_total
            elif key.startswith(_OPTIMIZER_PREFIX):
                optimizer_us += device_total
        ranked = sorted(events, key=_self_device_total, reverse=True)[: self.row_limit]
        forward_us = sum(entry["device_us"] for entry in steps.values())
        return {
            "tag": self.tag,
            "schedule": {"wait": self.wait, "warmup": self.warmup, "active": self.active},
            "sort_key": sort_key,
            "window_wall_s": self._window_wall,
            "profiled_it_s": (
                self.active / self._window_wall
                if self._window_wall and self._window_wall > 0
                else None
            ),
            "totals_us": {
                "batch_scope_device": batch_us,
                "forward_steps_device": forward_us,
                "backward_scope_device": backward_us,
                "autograd_engine_device": autograd_us,
                "optimizer_device": optimizer_us,
                "all_ops_self_device": total_self_device,
            },
            "forward_steps_us": steps,
            "top_ops": [
                {
                    "op": str(event.key),
                    "self_device_us": _self_device_total(event),
                    "device_us": _device_total(event),
                    "self_cpu_us": _time_us(event, "self_cpu_time_total"),
                    "count": getattr(event, "count", 0),
                }
                for event in ranked
            ],
        }


def resolve_schedule(
    steps: int,
    wait: int | None = None,
    warmup: int | None = None,
    active: int | None = None,
) -> dict[str, int]:
    """Fit a ``wait``/``warmup``/``active`` capture window inside `steps` batches.

    A schedule whose phases outlast the run records nothing at all — the window
    never closes, ``on_trace_ready`` never fires, and the command exits having
    written no artifacts. So the schedule is derived from `steps` rather than
    defaulted independently of it, and an explicit schedule that does not fit is
    a hard error instead of an empty trace.

    Derivation: `active` takes up to `_DEFAULT_ACTIVE` steps, `warmup` up to
    `_DEFAULT_WARMUP` of what remains, and **everything left over becomes**
    ``wait`` — so the capture is the tail of the run, which is the most
    steady-state part of it. Any of the three may be pinned explicitly; the
    others still fill in around it.

    Parameters
    ----------
    steps : int
        Train batches the run will do. Must be >= 3 (one step per phase).
    wait, warmup, active : int | None, optional
        Explicit phase lengths. ``None`` (default) derives them.

    Returns
    -------
    dict[str, int]
        ``{"wait": ..., "warmup": ..., "active": ...}``, summing to <= `steps`.

    Raises
    ------
    ValueError
        `steps` below 3, a negative/zero phase, or an explicit schedule whose
        total exceeds `steps`.
    """
    if steps < 3:
        raise ValueError(
            f"--steps must be at least 3 (one wait + one warmup + one active batch), got {steps}"
        )
    if any(value is not None and value < 1 for value in (wait, warmup, active)):
        raise ValueError(
            f"wait/warmup/active must each be >= 1 when given, got "
            f"wait={wait}, warmup={warmup}, active={active}"
        )
    given = [value for value in (wait, warmup, active) if value is not None]
    unset = 3 - len(given)
    if sum(given) > steps - unset:
        raise ValueError(
            f"the profiler schedule wait={wait}, warmup={warmup}, active={active} does not fit "
            f"in --steps {steps} — the capture window would never close and no trace would be "
            f"written. Raise --steps to at least {sum(given) + unset}, or shorten the schedule."
        )

    # active first (it is the payload), then warmup, then wait absorbs the rest,
    # always leaving at least one batch for each phase still unassigned
    remaining = steps - sum(given)
    if active is None:
        active = max(1, min(_DEFAULT_ACTIVE, remaining - (wait is None) - (warmup is None)))
        remaining -= active
    if warmup is None:
        warmup = max(1, min(_DEFAULT_WARMUP, remaining - (wait is None)))
        remaining -= warmup
    if wait is None:
        wait = max(1, remaining)
    return {"wait": wait, "warmup": warmup, "active": active}


def _model_overlay(steps: int) -> str:
    """Write the fit-shaping overlay config (capped, logger-less, artifact-free).

    Stacked LAST of the salt configs so it wins the deep merge, but BEFORE the
    caller's ``--set`` overrides so they still win over it. Profiling a model is
    not training it: no checkpoints, no graph artifacts, no LR logging, no
    validation, exactly `steps` train batches.
    """
    overlay = {
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": steps,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "enable_checkpointing": False,
            "logger": False,
        },
        "callbacks": {"checkpoint": None, "lr_monitor": None, "artifacts": None},
    }
    handle = tempfile.NamedTemporaryFile("w", suffix="_salt_profile_model.yaml", delete=False)
    with handle:
        yaml.dump(overlay, handle, sort_keys=False)
    return handle.name


def profile_model(
    configs: Sequence[Path],
    out_dir: Path,
    steps: int = DEFAULT_STEPS,
    overrides: Sequence[str] = (),
    tag: str = "model",
    wait: int | None = None,
    warmup: int | None = None,
    active: int | None = None,
    row_limit: int = 30,
    with_stack: bool = False,
    profile_memory: bool = False,
    record_shapes: bool = False,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Run a short capped fit under `TorchProfilerCallback` and report the split.

    The symmetric counterpart of `profile_dataset`: same ``--config`` stacking,
    same ``--set`` overrides, same ``--steps``. This is plumbing over the
    callback, not a second capture implementation — attaching the callback to a
    real `salt fit` is still the supported route for profiling a training run
    you were going to do anyway.

    Parameters
    ----------
    configs : Sequence[Path]
        Config stack, deep-merged left-to-right exactly as ``salt fit`` does.
    out_dir : Path
        Output directory (created if missing).
    steps : int, optional
        Train batches to run, by default `DEFAULT_STEPS`. The capture window is
        the tail of these — see `resolve_schedule`.
    overrides : Sequence[str], optional
        ``KEY=VALUE`` config overrides, by default none.
    tag : str, optional
        Artifact filename prefix, by default ``"model"``.
    wait, warmup, active : int | None, optional
        Explicit profiler schedule; derived from `steps` when unset.
    row_limit : int, optional
        Rows in the printed/written per-op tables, by default 30.
    with_stack, profile_memory, record_shapes : bool, optional
        Passed to the callback. All default to **False** here (the callback's
        own defaults are tuned for a hand-configured capture): stacks are empty
        on several torch builds and both stacks and allocator events cost host
        RAM, which is what kills long windows at large batch.
    compile_model : bool, optional
        Pass ``--compile`` to the fit, by default False.

    Returns
    -------
    dict[str, Any]
        The summary the callback wrote to ``<tag>_summary.json``.

    Raises
    ------
    ValueError
        A malformed ``--set`` entry, or a schedule that will not fit `steps`.
    RuntimeError
        The fit finished without the capture window closing (no summary).
    """
    from salt.config_utils import disable_logger_in_config
    from salt.main import SaltCLI

    # everything that can be rejected without touching the filesystem, first —
    # a bad flag should not cost a config parse (or a confusing FileNotFoundError)
    schedule = resolve_schedule(steps, wait=wait, warmup=warmup, active=active)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"--set entries must be KEY=VALUE, got {entry!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["fit"]
    for path in configs:
        args.extend(["--config", disable_logger_in_config(str(path))])
    args.extend(["--config", _model_overlay(steps)])
    args.extend(f"--{entry}" for entry in overrides)
    if compile_model:
        args.append("--compile")
    args.extend([
        "--trainer.callbacks+=salt.profiling.TorchProfilerCallback",
        f"--trainer.callbacks.dirpath={out_dir}",
        f"--trainer.callbacks.tag={tag}",
        f"--trainer.callbacks.wait={schedule['wait']}",
        f"--trainer.callbacks.warmup={schedule['warmup']}",
        f"--trainer.callbacks.active={schedule['active']}",
        f"--trainer.callbacks.row_limit={row_limit}",
        f"--trainer.callbacks.with_stack={str(with_stack).lower()}",
        f"--trainer.callbacks.profile_memory={str(profile_memory).lower()}",
        f"--trainer.callbacks.record_shapes={str(record_shapes).lower()}",
    ])

    console(
        f"[profile model] {steps} train batches, capture = last "
        f"{schedule['active']} (wait {schedule['wait']}, warmup {schedule['warmup']})"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*args parameter is intended to run from within Python.*"
        )
        SaltCLI(args=args)

    summary_path = out_dir / f"{tag}_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"the fit finished without the profiler window closing — no {summary_path.name} "
            f"was written. The run did fewer than {sum(schedule.values())} train batches "
            "(a dataset smaller than --steps, or an override capping limit_train_batches)."
        )
    summary = json.loads(summary_path.read_text())
    _print_model_report(summary, out_dir, tag, row_limit)
    return summary


def _print_model_report(
    summary: Mapping[str, Any], out_dir: Path, tag: str, row_limit: int
) -> None:
    """Print the per-op table plus the per-module / bucket split for the window."""
    table = out_dir / f"{tag}_key_averages.txt"
    if table.exists():
        console(f"\n=== {tag}: torch.profiler key_averages ===")
        console(table.read_text())

    totals = dict(summary.get("totals_us") or {})
    steps = dict(summary.get("forward_steps_us") or {})
    active = int((summary.get("schedule") or {}).get("active") or 0) or 1
    denominator = sum(
        totals.get(key, 0.0)
        for key in ("forward_steps_device", "autograd_engine_device", "optimizer_device")
    )
    # A CPU-only run records no device time at all, and printing a table of
    # zeroes is worse than printing nothing. Fall back to the wall time of the
    # same scopes and SAY which one is on screen.
    on_device = denominator > 0
    axis = "device" if on_device else "CPU wall"
    step_key, bucket_suffix = ("device_us", "device") if on_device else ("cpu_us", "cpu")
    if not on_device:
        denominator = sum(entry.get("cpu_us", 0.0) for entry in steps.values())

    console(f"=== {tag}: per-plan-step {axis} time (ms/step over {active} steps) ===")
    ranked = sorted(steps.items(), key=lambda item: item[1].get(step_key, 0.0), reverse=True)
    console(f"{'module':<32}{'ms/step':>12}{'% of fwd':>12}")
    for name, entry in ranked[:row_limit]:
        value = entry.get(step_key, 0.0)
        share = 100.0 * value / denominator if denominator else 0.0
        console(f"{name:<32}{value / active / 1e3:>12.3f}{share:>11.1f}%")

    if not on_device:
        console(
            "\nno device time was recorded (CPU-only run) — the rows above are CPU wall time "
            "and are INCLUSIVE, so they do not partition the step. Run on a GPU for the "
            "forward/backward/optimizer split."
        )
        console(f"\nartifacts: {out_dir}/{tag}_*")
        return

    console(f"\n=== {tag}: step buckets (ms/step) ===")
    buckets = (
        ("forward (sum of plan steps)", f"forward_steps_{bucket_suffix}"),
        ("backward (autograd engine)", f"autograd_engine_{bucket_suffix}"),
        ("optimizer", f"optimizer_{bucket_suffix}"),
    )
    for label, key in buckets:
        value = totals.get(key, 0.0)
        share = 100.0 * value / denominator if denominator else 0.0
        console(f"{label:<32}{value / active / 1e3:>12.3f}{share:>11.1f}%")
    profiled = summary.get("profiled_it_s")
    if profiled:
        console(
            f"\nprofiled rate {profiled:.3f} it/s — ATTRIBUTION ONLY. Divide an unprofiled "
            "run's rate by this to get the profiler overhead; never quote it as throughput."
        )
    console(f"\nartifacts: {out_dir}/{tag}_*")


# --------------------------------------------------------------------------- #
# dataset side
# --------------------------------------------------------------------------- #


def _resolve_target(dotted: str) -> tuple[Any, str, Any]:
    """Resolve ``pkg.mod.Class.method`` / ``pkg.mod.func`` to ``(owner, attr, function)``.

    Walks the longest importable module prefix, then attribute-chains the
    remainder; raises `ValueError` when the path does not resolve to a
    callable.
    """
    parts = dotted.split(".")
    module = None
    index = len(parts)
    while index > 0:
        with contextlib.suppress(ImportError):
            module = import_module(".".join(parts[:index]))
            break
        index -= 1
    if module is None:
        raise ValueError(f"cannot import any module prefix of {dotted!r}")
    owner: Any = module
    for attr in parts[index:-1]:
        owner = getattr(owner, attr)
    attr = parts[-1]
    function = getattr(owner, attr, None)
    if function is None:
        raise ValueError(f"{dotted!r} does not exist")
    if not callable(function):
        # ValueError, and salt/tests/unit/test_profiling.py pins that contract.
        raise ValueError(f"{dotted!r} is not callable")  # noqa: TRY004
    return owner, attr, function


def _structure(value: Any) -> Any:
    """Nested key -> shape/dtype description of one batch (the sanity-check fingerprint)."""
    if isinstance(value, Mapping):
        return {key: _structure(item) for key, item in sorted(value.items())}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return f"{type(value).__name__}{tuple(shape) if shape is not None else ''}:{dtype}"


def _build_datamodule(configs: Sequence[Path], overrides: Sequence[str]) -> tuple[Any, Any]:
    """Parse the config stack run-free and return ``(datamodule, model)`` with workers off.

    The run-free parse still constructs a `Trainer`, so a GPU training config
    would refuse to instantiate on a machine with no CUDA — and profiling the
    read path is exactly the thing you want to do on a login node. The
    accelerator is therefore forced to CPU FIRST, before the caller's
    overrides, so an explicit ``--set trainer.accelerator=gpu`` still wins. No
    trainer is ever run.
    """
    from salt.cli import _parse_trainer_cli

    forced = ["trainer.accelerator=cpu", "trainer.devices=1", "trainer.precision=32-true"]
    cli = _parse_trainer_cli(list(configs), [*forced, *overrides])
    datamodule = cli.datamodule
    model = cli.model
    demand = getattr(model, "sink_demand", None)
    if callable(demand):
        datamodule.set_sinks(demand())
    # the profiled code must run in THIS process: no worker forks, no prefetch
    datamodule.num_workers = 0
    datamodule.persistent_workers = False
    datamodule.pin_memory = False
    return datamodule, model


def profile_dataset(
    configs: Sequence[Path],
    out_dir: Path,
    steps: int = DEFAULT_STEPS,
    overrides: Sequence[str] = (),
    functions: Sequence[str] = DEFAULT_DATASET_FUNCTIONS,
    tag: str = "dataset",
) -> dict[str, Any]:
    """Line-profile the dataset read path over `steps` in-process batches.

    Builds the datamodule from the config stack (run-free, ``num_workers=0``),
    wraps `functions` in a ``line_profiler.LineProfiler`` by patching the
    owning class attribute, iterates the training dataloader, and writes the
    annotated per-line report plus a machine-readable summary.

    A structural sanity check runs first: one batch is drawn through the
    unprofiled dataloader and its nested keys/shapes/dtypes are compared to a
    profiled batch, so the report cannot silently miss part of the pipeline.

    Parameters
    ----------
    configs : Sequence[Path]
        Config stack, deep-merged left-to-right exactly as ``salt fit`` does.
    out_dir : Path
        Output directory (created if missing).
    steps : int, optional
        Batches to iterate under the profiler, by default `DEFAULT_STEPS`.
    overrides : Sequence[str], optional
        ``KEY=VALUE`` config overrides, by default none.
    functions : Sequence[str], optional
        Dotted paths to profile; entries that do not resolve are skipped and
        reported. Defaults to `DEFAULT_DATASET_FUNCTIONS`.
    tag : str, optional
        Filename prefix, by default ``"dataset"``.

    Returns
    -------
    dict[str, Any]
        The summary written to ``<tag>_summary.json``.

    Raises
    ------
    ImportError
        When ``line_profiler`` is not installed.
    RuntimeError
        When the profiled batch structure differs from the unprofiled one.
    """
    try:
        from line_profiler import LineProfiler
    except ImportError as err:  # pragma: no cover - exercised by the CLI path
        raise ImportError(_LINE_PROFILER_HINT) from err

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datamodule, _ = _build_datamodule(configs, overrides)
    datamodule.setup("fit")

    # 1) the unprofiled reference batch (structure oracle)
    reference = _structure(next(iter(datamodule.train_dataloader())))

    # 2) patch the hot path onto a LineProfiler
    profiler = LineProfiler()
    wrapped: list[str] = []
    skipped: dict[str, str] = {}
    patched: list[tuple[Any, str, Any]] = []
    for dotted in functions:
        try:
            owner, attr, function = _resolve_target(dotted)
        except (ValueError, AttributeError) as err:
            skipped[dotted] = str(err)
            continue
        target = function.__func__ if hasattr(function, "__func__") else function
        setattr(owner, attr, profiler(target))
        patched.append((owner, attr, function))
        wrapped.append(dotted)

    started = time.perf_counter()
    seen = 0
    passes = 0
    first_profiled: Any = None
    try:
        loader = datamodule.train_dataloader()
        profiler.enable_by_count()
        # A file smaller than `steps` batches is re-iterated rather than
        # silently short-changing the sample; `passes` records how often, so a
        # page-cache-warm result is visible in the summary rather than implied.
        while seen < steps:
            passes += 1
            drawn = 0
            for batch in loader:
                if first_profiled is None:
                    first_profiled = _structure(batch)
                drawn += 1
                seen += 1
                if seen >= steps:
                    break
            if drawn == 0:
                break
        profiler.disable_by_count()
    finally:
        for owner, attr, function in patched:
            setattr(owner, attr, function)
    elapsed = time.perf_counter() - started

    if first_profiled != reference:
        raise RuntimeError(
            "profiled batch structure differs from the unprofiled dataloader batch — the "
            "wrapped function list does not cover the same pipeline; compare "
            f"{first_profiled} with {reference}"
        )

    report = out_dir / f"{tag}_profile.txt"
    with report.open("w") as stream:
        profiler.print_stats(stream=stream, output_unit=1e-6)
    with contextlib.suppress(Exception):
        profiler.dump_stats(str(out_dir / f"{tag}_profile.lprof"))

    summary = _dataset_summary(profiler.get_stats(), wrapped, skipped, seen, elapsed)
    summary["reference_structure_matches"] = True
    summary["dataloader_passes"] = passes
    summary["batches_per_pass"] = len(loader)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    note = ""
    if passes > 1:
        note = f" ({passes} passes over {len(loader)} batches — page cache is warm)"
    console(f"[profile dataset] {seen} batches in {elapsed:.2f}s{note} -> {report}")
    return summary


def _dataset_summary(
    stats: Any,
    wrapped: Sequence[str],
    skipped: Mapping[str, str],
    batches: int,
    elapsed: float,
) -> dict[str, Any]:
    """Fold `line_profiler` stats into per-function totals and a ranked hot-line list."""
    unit = float(getattr(stats, "unit", 1e-6))
    functions: list[dict[str, Any]] = []
    hot: list[dict[str, Any]] = []
    for (filename, start_line, name), timings in stats.timings.items():
        total_s = sum(entry[2] for entry in timings) * unit
        functions.append({
            "function": name,
            "file": filename,
            "line": start_line,
            "total_s": total_s,
            "per_batch_ms": (total_s / batches * 1e3) if batches else None,
        })
        hot.extend(
            {
                "function": name,
                "file": filename,
                "line": lineno,
                "hits": hits,
                "total_s": duration * unit,
                "per_batch_ms": (duration * unit / batches * 1e3) if batches else None,
            }
            for lineno, hits, duration in timings
        )
    functions.sort(key=operator.itemgetter("total_s"), reverse=True)
    hot.sort(key=operator.itemgetter("total_s"), reverse=True)
    return {
        "batches": batches,
        "wall_s": elapsed,
        "s_per_batch": elapsed / batches if batches else None,
        "wrapped_functions": list(wrapped),
        "skipped_functions": dict(skipped),
        "functions": functions,
        "hot_lines": hot[:40],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _dataset_parser() -> argparse.ArgumentParser:
    """Build the ``salt profile dataset`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="salt profile dataset",
        description=(
            "line_profiler over the dataset read path: builds the datamodule from a config "
            "stack, forces num_workers=0 so the profiled code runs in-process, and prints "
            "the annotated per-line report."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        required=True,
        help="training config; repeat to stack (deep-merged left-to-right, as salt fit)",
    )
    parser.add_argument("--set", action="append", default=[], help="KEY=VALUE config override")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"batches to profile (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --steps
    )
    parser.add_argument("--out", type=Path, default=Path("profile"), help="output directory")
    parser.add_argument("--tag", default="dataset", help="output filename prefix")
    parser.add_argument(
        "--functions",
        default=None,
        help="comma-separated dotted paths to profile (default: the salt read path)",
    )
    parser.add_argument(
        "--extra-functions",
        default=None,
        help="comma-separated dotted paths to profile IN ADDITION to the defaults",
    )
    return parser


def _model_parser() -> argparse.ArgumentParser:
    """Build the ``salt profile model`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="salt profile model",
        description=(
            "torch.profiler over the model: runs a short capped fit (no logger, no "
            "checkpoints, no validation) with TorchProfilerCallback attached, then prints "
            "the per-op table and the per-plan-step / backward / optimizer split."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        required=True,
        help="training config; repeat to stack (deep-merged left-to-right, as salt fit)",
    )
    parser.add_argument("--set", action="append", default=[], help="KEY=VALUE config override")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"train batches to run (default {DEFAULT_STEPS}); the capture is their tail",
    )
    parser.add_argument("--out", type=Path, default=Path("profile"), help="output directory")
    parser.add_argument("--tag", default="model", help="output filename prefix")
    parser.add_argument("--wait", type=int, default=None, help="steps skipped (default: derived)")
    parser.add_argument(
        "--warmup", type=int, default=None, help="steps traced then discarded (default: derived)"
    )
    parser.add_argument(
        "--active", type=int, default=None, help="steps recorded (default: derived, <= 20)"
    )
    parser.add_argument("--row-limit", type=int, default=30, help="rows per printed table")
    parser.add_argument(
        "--with-stack", action="store_true", help="record source stacks (costly, often empty)"
    )
    parser.add_argument(
        "--profile-memory", action="store_true", help="record allocator events (costs host RAM)"
    )
    parser.add_argument("--record-shapes", action="store_true", help="record operator input shapes")
    parser.add_argument("--compile", action="store_true", help="pass --compile to the fit")
    return parser


def _split(value: str | None) -> tuple[str, ...]:
    """Comma-separated CLI list -> tuple of stripped, non-empty entries."""
    return tuple(entry.strip() for entry in (value or "").split(",") if entry.strip())


_USAGE = (
    "usage: salt profile {dataset,model} --config <yaml> [--config <yaml>] "
    "[--steps N] [--out DIR]\n\n"
    "  dataset   line_profiler over the read path (needs salt-ml[profile])\n"
    "  model     torch.profiler over a short capped fit\n\n"
    "Both default to --steps 100. `salt profile <subcommand> --help` for the full "
    "flag list; see docs/profiling.md. To profile a training run you were going to do "
    "anyway, attach the callback directly instead:\n"
    "  salt fit ... --trainer.callbacks+=salt.profiling.TorchProfilerCallback \\\n"
    "               --trainer.callbacks.dirpath <dir>"
)


def _run_dataset(args: Sequence[str]) -> int:
    """``salt profile dataset``."""
    parsed = _dataset_parser().parse_args(args)
    if parsed.batches is not None and parsed.steps is not None:
        console("salt profile dataset: pass --steps or --batches, not both", file=sys.stderr)
        return 1
    steps = parsed.steps
    if steps is None and parsed.batches is not None:
        console(
            "salt profile dataset: --batches is a deprecated alias for --steps", file=sys.stderr
        )
        steps = parsed.batches
    if steps is None:
        steps = DEFAULT_STEPS
    functions: Iterable[str] = _split(parsed.functions) or DEFAULT_DATASET_FUNCTIONS
    functions = (*functions, *_split(parsed.extra_functions))
    try:
        profile_dataset(
            configs=parsed.config,
            out_dir=parsed.out,
            steps=steps,
            overrides=parsed.set,
            functions=tuple(functions),
            tag=parsed.tag,
        )
    except ImportError as err:
        console(f"salt profile dataset: {err}", file=sys.stderr)
        return 1
    return 0


def _run_model(args: Sequence[str]) -> int:
    """``salt profile model``."""
    parsed = _model_parser().parse_args(args)
    try:
        profile_model(
            configs=parsed.config,
            out_dir=parsed.out,
            steps=parsed.steps,
            overrides=parsed.set,
            tag=parsed.tag,
            wait=parsed.wait,
            warmup=parsed.warmup,
            active=parsed.active,
            row_limit=parsed.row_limit,
            with_stack=parsed.with_stack,
            profile_memory=parsed.profile_memory,
            record_shapes=parsed.record_shapes,
            compile_model=parsed.compile,
        )
    except (ValueError, RuntimeError) as err:
        console(f"salt profile model: {err}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``salt profile`` entry point (dispatched from `salt.main.main`).

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Arguments after ``salt profile``; ``sys.argv[2:]`` when None.

    Returns
    -------
    int
        Process exit code.
    """
    args = list(sys.argv[2:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        console(_USAGE)
        return 0 if args else 1
    if args[0] == "dataset":
        return _run_dataset(args[1:])
    if args[0] == "model":
        return _run_model(args[1:])
    console(
        f"salt profile: unknown subcommand {args[0]!r} (expected 'dataset' or 'model')",
        file=sys.stderr,
    )
    return 1
