"""Shipped writer modules: task columns, input copies, pad masks (design §8).

Together (in the ``base2.yaml`` order ``inputs_copy -> tasks -> pad_mask``)
these reproduce the v1 `PredictionWriter` per-group column layout byte for
byte: input copies first (full source dtypes, ``predictionwriter.py:186-196``),
task columns next in model declaration order (``:201-205, 255-261``), the
boolean ``mask`` column last (``:208-211``).

M4.5 unified manifest: `TaskWriter` additionally declares its
ONNX-manifest entries (`TaskWriter.onnx_outputs`) from the SAME tasks that
name the TEST columns — one owner per family, both representations rendered
by the task and gated together (amendment §2.2/§3).
`InputCopyWriter` and `PadMaskWriter` are eval-only by design: input
copies are meaningless in ONNX (Athena feeds the inputs) and a pad-mask
output has no Athena consumer (the adapter constructs all-valid masks) —
they keep the base ``onnx_outputs() == []``.

**Where the per-family knowledge lives** (the M-modular refactor): the TEST
column schema/values and the ONNX `ExportOutput` entry for each family are
rendered by the TASK (`salt.core.nn.tasks` ``output_names`` / ``get_h5`` /
``onnx_outputs``, mirroring v1's `task.py` placement). `TaskWriter` is pure
ORCHESTRATION — it selects tasks by instance name, groups their columns by
stream, prefixes with the run name, pads per-token fragments, applies the
``onnx_names`` overrides, orders ONNX entries global-before-sequence, and
owns the file I/O — but it never branches on the task family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import h5py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import GraphModule, Mode, TensorSpec
from salt.core.onnx.config import ExportOutput
from salt.core.outputs.writer_base import WriteCtx, Writer, WriterDeclareCtx, task_modules
from salt.core.utils.array_utils import join_structured_arrays

__all__ = ["InputCopyWriter", "PadMaskWriter", "TaskWriter"]


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad a per-token array along axis 1 to the file sequence length.

    The v1 ``maybe_pad`` truncation re-expansion (``array_utils.py:62-90``):
    positions beyond the model's (possibly truncated) sequence read as
    zeros — including the ``mask`` column's documented quirk (padded-away
    positions read ``mask=False``), preserved for byte parity.

    Returns
    -------
    np.ndarray
        `arr` unchanged when already long enough, else a zero-padded copy.
    """
    if arr.ndim < 2 or arr.shape[1] >= length:
        return arr
    out = np.zeros((arr.shape[0], length, *arr.shape[2:]), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def _render_method(module: GraphModule, attr: str, what: str) -> Any:
    """The task's render method (``output_names``/``get_h5``/``onnx_outputs``), or raise.

    The per-family rendering lives on the TASK now (mirroring v1's `task.py`
    placement); `task_modules` discovers task modules by DUCK TYPING (string
    ``pred_key`` + ``stream``), so a user task that does not subclass
    `_TaskModuleBase` may be selected without shipping a rendering. A
    `_TaskModuleBase` subclass raises the unsupported-family `ConfigError`
    from its own base guard; a duck-typed task missing the method entirely
    would otherwise surface a cryptic ``AttributeError`` — restore the SAME
    loud `ConfigError` here (design §8, the v1 "write a custom Writer"
    journey), so the error is identical regardless of how the task is
    discovered.

    Returns
    -------
    Any
        The bound render method.

    Raises
    ------
    ConfigError
        For a discovered task that ships no ``attr`` rendering.
    """
    method = getattr(module, attr, None)
    if not callable(method):
        name = getattr(module, "name", "unnamed")
        raise ConfigError(
            f"task {name!r} ({type(module).__name__}) ships no {what} rendering — "
            "supported families are ClassificationTaskModule, VertexingTaskModule and "
            "RegressionTaskModule; give a custom task module output_names/get_h5/onnx_outputs "
            "methods, or write a custom Writer for its outputs (design §8)"
        )
    return method


class TaskWriter(Writer):
    """Orchestrate task-rendered prediction columns + ONNX entries (design §8).

    Pure ORCHESTRATION: the per-family rendering (column names/dtypes/values
    and the ONNX `ExportOutput`) lives on the TASK modules
    (`salt.core.nn.tasks` ``output_names`` / ``get_h5`` / ``onnx_outputs``,
    mirroring v1's `task.py` placement). This writer selects which tasks to
    persist, groups their columns by stream, prefixes with the run name, pads
    per-token fragments to the file sequence length, and owns the file I/O —
    it never branches on the task family.

    - **TEST columns/values**: each selected task renders its own columns
      (``task.output_names(run_name)``) and structured values
      (``task.get_h5(bundle)``); the writer concatenates same-stream
      fragments, checks for duplicate column names, and pads. Classification
      -> ``f4`` per class, vertexing -> a single ``i8`` ``VertexIndex``
      column (bare by default, ``VertexingTaskModule.prefix_vertex_column``
      flips on the run-name prefix), regression -> ``f4`` per target.
    - **ONNX manifest** (M4.5): with ``onnx: true`` (default) the writer
      collects each selected task's ``onnx_outputs()`` entry (global
      classification -> per-class ``split_scalars``; sequence classification
      -> ``argmax`` int8; vertexing -> ``vertex_union_find`` int8;
      regression -> per-target ``split_scalars``), applies the per-task
      ``onnx_names`` overrides, and emits global-stream entries before
      sequence-stream entries (the v1 ``output_names`` order,
      ``to_onnx.py:258-292`` — O2/O5 byte parity). Adding an aux task lands
      in eval AND export with zero extra config; narrow with
      ``onnx``/``onnx_streams``/``onnx_tasks``.

    Parameters
    ----------
    tasks : Sequence[str] | None, optional
        Task INSTANCE names to persist. ``None`` (default) means EVERY
        configured task — v1's derive-from-the-model behaviour, so
        ``base2.yaml`` ships one generic writer and a new aux task is
        persisted automatically (design §8). An explicit list narrows it;
        the TEST dead-preds error catches a narrowed-list-forgot-a-task
        mistake.
    onnx : bool, optional
        Participate in the ONNX manifest, by default True (the M4.5
        polarity adjudication: greenfield defaults to full participation;
        the M7 converter emits explicit knobs — amendment addendum).
        ``False`` makes this writer eval-only.
    onnx_streams : Sequence[str] | None, optional
        Narrow ONNX participation to these task streams (a subset of the
        selected tasks' streams), by default None — every selected task's
        stream exports.
    onnx_tasks : Sequence[str] | None, optional
        Task-grained narrowing by task INSTANCE name (amendment merge
        condition 1 — v1's ``tasks_to_output`` expressiveness: "export
        ``track_vertexing`` but not ``track_origin``", both on one
        stream), by default None. Combines with ``onnx_streams`` as an
        intersection.
    onnx_names : Mapping[str, str | list[str]] | None, optional
        Per-task suffix override (amendment merge condition 6), by default
        None. A ``str`` renames a single-output (argmax) entry; a ``list``
        replaces a classification entry's per-class suffixes (length
        validated against the entry's suffix count — the collision-fix path
        for two classification tasks with overlapping class names).
        Vertexing/regression entries are NOT renameable here (the task owns
        their names: vertexing's shared `VERTEX_INDEX` constant, regression's
        ``custom_output_names``; ``Task.onnx_renameable``, amendment §2.2).
    """

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        onnx: bool = True,
        onnx_streams: Sequence[str] | None = None,
        onnx_tasks: Sequence[str] | None = None,
        onnx_names: Mapping[str, str | list[str]] | None = None,
    ) -> None:
        self.tasks = tuple(tasks) if tasks is not None else None
        self.onnx = onnx
        self.onnx_streams = tuple(onnx_streams) if onnx_streams is not None else None
        self.onnx_tasks = tuple(onnx_tasks) if onnx_tasks is not None else None
        self.onnx_names = dict(onnx_names) if onnx_names is not None else {}
        if not onnx and (self.onnx_streams is not None or self.onnx_tasks is not None):
            raise ConfigError(
                f"TaskWriter: 'onnx: false' disables ONNX participation entirely — drop the "
                f"contradictory onnx_streams={list(self.onnx_streams or [])} / "
                f"onnx_tasks={list(self.onnx_tasks or [])} narrowing (M4.5 amendment §2.2)"
            )

    def _selected(self, model_modules: Mapping[str, GraphModule]) -> dict[str, GraphModule]:
        """The task modules this writer persists, in declaration order.

        Returns
        -------
        dict[str, GraphModule]
            ``{instance name: task module}``.

        Raises
        ------
        ConfigError
            When an explicit ``tasks`` entry matches no configured task.
        """
        tasks = task_modules(model_modules)
        if self.tasks is None:
            return tasks
        if unknown := sorted(set(self.tasks) - set(tasks)):
            raise ConfigError(
                f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): tasks "
                f"{unknown} name no configured task — task instances are {sorted(tasks)} "
                "(design §8)"
            )
        return {name: m for name, m in tasks.items() if name in self.tasks}

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the consumed ``preds.*`` leaves (design §8).

        Returns
        -------
        dict[str, TensorSpec]
            One unconstrained spec per selected task prediction.
        """
        return {
            module.pred_key: TensorSpec(shape=None, dtype=None)
            for module in self._selected(ctx.model_modules).values()
        }

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the per-stream task columns, in task declaration order.

        Delegates the per-family naming/dtype to each task's
        ``output_names(run_name)`` (the task owns its column schema); the
        writer only groups by stream and rejects colliding column names.

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}``.

        Raises
        ------
        ConfigError
            For a task that ships no TEST rendering (raised by the task), or
            colliding column names between two tasks on one stream.
        """
        descrs: dict[str, list[tuple[str, str]]] = {}
        for module in self._selected(ctx.model_modules).values():
            output_names = _render_method(module, "output_names", "TEST columns")
            descrs.setdefault(module.stream, []).extend(output_names(ctx.run_name))
        out: dict[str, np.dtype] = {}
        for stream, descr in descrs.items():
            if len({field for field, _ in descr}) != len(descr):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: duplicate column names on stream {stream!r}: "
                    f"{[field for field, _ in descr]} — two tasks declare the same output "
                    "columns (design §8)"
                )
            out[stream] = np.dtype(descr)
        return out

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """The statically-derivable eval columns (design §4.4 annotation surface).

        Returns
        -------
        dict[str, list[str]]
            ``{stream: [column names]}`` — the same names `columns`
            declares at run time (the task's ``output_names``).
        """
        out: dict[str, list[str]] = {}
        for module in self._selected(ctx.model_modules).values():
            output_names = _render_method(module, "output_names", "TEST columns")
            cols = [column for column, _ in output_names(run_name)]
            out.setdefault(module.stream, []).extend(cols)
        return out

    # -- ONNX role: the manifest declarations (M4.5 amendment §2.2) -------------

    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The selected tasks' export entries, each rendered by its task.

        The task owns its per-family `ExportOutput` (``task.onnx_outputs()``);
        the writer collects the entries, applies the ``onnx_names`` overrides,
        and emits them in v1's ``output_names`` order (``to_onnx.py:258-292``):
        global-stream entries first, then sequence-stream aux entries —
        regardless of module declaration order, so converted goldens
        byte-match O2/O5 (amendment §5 rule 5). Unknown
        ``onnx_streams``/``onnx_tasks``/``onnx_names`` entries, a malformed
        ``onnx_names`` value, or a task that ships no export representation
        propagate a `ConfigError` from `_onnx_selected` / `_check_onnx_names`
        / the task's own ``onnx_outputs``.

        Returns
        -------
        list[ExportOutput]
            One entry per ONNX-selected task (empty with ``onnx: false``).
        """
        if not self.onnx:
            return []
        selected = self._onnx_selected(ctx)
        self._check_onnx_names(selected)
        ordered = [it for it in selected.items() if it[1].stream not in ctx.sequence_streams]
        ordered += [it for it in selected.items() if it[1].stream in ctx.sequence_streams]
        out: list[ExportOutput] = []
        for name, module in ordered:
            onnx_outputs = _render_method(module, "onnx_outputs", "ONNX output")
            out.extend(self._apply_onnx_name(name, entry) for entry in onnx_outputs())
        return out

    def _onnx_selected(self, ctx: WriterDeclareCtx) -> dict[str, GraphModule]:
        """The ONNX-participating tasks: TEST selection ∩ onnx_streams ∩ onnx_tasks.

        Returns
        -------
        dict[str, GraphModule]
            ``{instance name: task module}`` in declaration order.

        Raises
        ------
        ConfigError
            For ``onnx_streams`` entries outside the selected tasks' streams
            or ``onnx_tasks`` entries naming no selected task.
        """
        tasks = self._selected(ctx.model_modules)
        if self.onnx_streams is not None:
            known = {module.stream for module in tasks.values()}
            if unknown := sorted(set(self.onnx_streams) - known):
                raise ConfigError(
                    f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"onnx_streams {unknown} match no selected task stream — selected "
                    f"streams are {sorted(known)} (onnx_streams narrows WITHIN the selected "
                    "'tasks', M4.5 amendment §2.2)"
                )
            tasks = {n: m for n, m in tasks.items() if m.stream in self.onnx_streams}
        if self.onnx_tasks is not None:
            if unknown := sorted(set(self.onnx_tasks) - set(tasks)):
                raise ConfigError(
                    f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"onnx_tasks {unknown} name no selected task — candidates are "
                    f"{sorted(tasks)} (task INSTANCE names; amendment merge condition 1)"
                )
            tasks = {n: m for n, m in tasks.items() if n in self.onnx_tasks}
        return tasks

    def _check_onnx_names(self, selected: Mapping[str, GraphModule]) -> None:
        """Validate the ``onnx_names`` mapping against the selected tasks (condition 6).

        The task decides whether its ONNX suffix is renameable
        (``Task.onnx_renameable`` — classification yes, vertexing/regression
        no, since the task owns those names). For a renameable task the value
        shape must match the rendered entry: a per-class ``names`` entry takes
        a LIST (length == the entry's suffix count), a single-``name`` entry a
        string.

        Raises
        ------
        ConfigError
            For an entry naming no ONNX-selected task, a non-renameable task,
            or a value of the wrong shape for the rendered entry.
        """
        for task_name, value in self.onnx_names.items():
            module = selected.get(task_name)
            if module is None:
                raise ConfigError(
                    f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"onnx_names entry {task_name!r} names no ONNX-selected task — "
                    f"candidates are {sorted(selected)}"
                )
            if not getattr(module, "onnx_renameable", False):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: onnx_names cannot rename task {task_name!r} "
                    f"({type(module).__name__}) — the task owns its ONNX suffix (vertexing's "
                    "shared VertexIndex constant / regression's custom_output_names; rename "
                    "those via the task, amendment §2.2 single ownership)"
                )
            entry = self._single_onnx_entry(task_name, module)
            if entry.names is not None:
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ConfigError(
                        f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] must be a LIST "
                        f"of per-class suffixes for a global classification task, got "
                        f"{value!r} (amendment merge condition 6)"
                    )
                if len(value) != len(entry.names):
                    raise ConfigError(
                        f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] lists "
                        f"{len(value)} suffixes but the task has {len(entry.names)} classes "
                        f"({list(entry.names)}) — one suffix per class, in class order "
                        "(amendment merge condition 6)"
                    )
            elif not isinstance(value, str):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] must be a single "
                    f"string for a sequence (argmax) task, got {value!r} "
                    "(amendment merge condition 6)"
                )

    @staticmethod
    def _single_onnx_entry(name: str, module: GraphModule) -> ExportOutput:
        """The task's single ONNX entry (the renameable families produce exactly one).

        Returns
        -------
        ExportOutput
            The task's lone export entry.

        Raises
        ------
        ConfigError
            If a renameable task does not render exactly one entry (``onnx_names``
            targets one entry by task name — a multi-entry renameable task has
            no unambiguous target).
        """
        entries = _render_method(module, "onnx_outputs", "ONNX output")()
        if len(entries) != 1:
            raise ConfigError(
                f"TaskWriter: onnx_names targets task {name!r} but it renders {len(entries)} "
                "ONNX entries — onnx_names renames a single-entry task (amendment condition 6)"
            )
        return entries[0]

    def _apply_onnx_name(self, name: str, entry: ExportOutput) -> ExportOutput:
        """Apply an ``onnx_names`` override to one rendered entry (else pass through).

        Returns
        -------
        ExportOutput
            The entry with its suffix(es) replaced by the ``onnx_names``
            value when one is configured (validated in `_check_onnx_names`),
            else the task-rendered entry unchanged.
        """
        override = self.onnx_names.get(name)
        if override is None:
            return entry
        if entry.names is not None:
            return replace(entry, names=list(override))
        return replace(entry, name=str(override))

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Convert this batch's predictions to structured arrays.

        Delegates the per-family value formatting to each task's
        ``get_h5(bundle, run_name)`` (the task reads its own ``preds.*`` leaf
        and renders the run-name-prefixed structured array); the writer pads
        per-token fragments to the file sequence length and concatenates
        same-stream fragments.

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}``; per-token arrays padded to the
            file sequence length.
        """
        del rows
        ctx = self.ctx
        frags: dict[str, list[np.ndarray]] = {}
        for module in self._selected(ctx.model_modules).values():
            get_h5 = _render_method(module, "get_h5", "TEST values")
            arr = get_h5(bundle, ctx.run_name)
            if arr.ndim == 2:
                arr = _pad_to(arr, ctx.seq_lengths[module.stream])
            frags.setdefault(module.stream, []).append(arr)
        return {
            stream: arrays[0] if len(arrays) == 1 else join_structured_arrays(arrays)
            for stream, arrays in frags.items()
        }


class InputCopyWriter(Writer):
    """Copy source-file variables into the output (design §8).

    Re-reads the ORIGINAL test file by absolute rows (``meta.rows``)
    through one cached handle — fixing v1's per-batch open/close
    (``predictionwriter.py:160-162``) while keeping its semantics: by
    default ALL dtype fields of each stream's source dataset ride along
    (labels/truth columns included) with SOURCE dtypes and FILE field order
    (``predictionwriter.py:186-196``); the shared H5 sink's ``full``
    precision policy still downcasts floats to f4 exactly as v1.

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to copy. ``None`` (default) = every stream with a
        configured task (the v1 output-group set,
        ``predictionwriter.py:112``); an explicit list overrides — the
        ``extra_vars``-creates-a-group behaviour (``:233-235``).
    variables : Mapping[str, Sequence[str]] | None, optional
        Per-stream narrowing of the copied fields (v1 ``extra_vars``);
        streams not listed copy all source fields, by default None.
    """

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        variables: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.streams = tuple(streams) if streams is not None else None
        self.variables = {key: list(val) for key, val in (variables or {}).items()}
        self._file: h5py.File | None = None
        self._reads: dict[str, tuple[h5py.Dataset, list[str]]] = {}

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the row-alignment dependency (``meta.rows``, design §8).

        Returns
        -------
        dict[str, TensorSpec]
            The TEST-only ``meta.rows`` spec (input copies come from the
            file, not the bundle).
        """
        del ctx
        return {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)}

    def _selected_streams(self, ctx: WriteCtx | WriterDeclareCtx) -> tuple[str, ...]:
        """Resolve the copied streams (explicit list or v1 task-stream set).

        Returns
        -------
        tuple[str, ...]
            Stream names in reader config order.

        Raises
        ------
        ConfigError
            When an explicit entry is not a reader stream.
        """
        if self.streams is not None:
            if unknown := sorted(set(self.streams) - set(ctx.streams)):
                raise ConfigError(
                    f"InputCopyWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"unknown streams {unknown} — reader streams are {list(ctx.streams)}"
                )
            return tuple(s for s in ctx.streams if s in self.streams)
        tasked = {module.stream for module in task_modules(ctx.model_modules).values()}
        return tuple(s for s in ctx.streams if s in tasked)

    def setup(self, ctx: WriteCtx) -> None:
        """Open the cached source handle and resolve per-stream field lists.

        Raises
        ------
        ConfigError
            When configured ``variables`` are missing from the source file
            (v1 ``predictionwriter.py:88-100`` validation).
        """
        super().setup(ctx)
        self.finalize()  # re-entrant across test runs
        # SAME open flags as H5StructuredReader._ensure_open: HDF5 refuses to
        # open one file with mixed SWMR flags in one process, and the reader's
        # swmr handle to this very file opens lazily at the first batch —
        # a plain handle here would make that open fail (order-dependent!)
        self._file = h5py.File(ctx.source_path, "r", swmr=True, libver="latest")
        self._reads = {}
        for stream in self._selected_streams(ctx):
            ds = self._file[ctx.group_datasets[stream]]
            file_fields = list(ds.dtype.names or ())
            fields = self.variables.get(stream) or file_fields
            if missing := sorted(set(fields) - set(file_fields)):
                raise ConfigError(
                    f"InputCopyWriter {self.name!r}: variables {missing} missing for stream "
                    f"{stream!r} in {ctx.source_path.name!r} "
                    "(v1 extra_vars contract, predictionwriter.py:94-98)"
                )
            self._reads[stream] = (ds, list(fields))

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the copied columns with SOURCE dtypes and order.

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}``.
        """
        del ctx
        return {
            stream: np.dtype([(field, ds.dtype[field]) for field in fields])
            for stream, (ds, fields) in self._reads.items()
        }

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Re-read this batch's absolute rows from the source file.

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` at full file sequence length.
        """
        del bundle
        return {
            stream: ds.fields(fields)[rows.start : rows.stop]
            for stream, (ds, fields) in self._reads.items()
        }

    def finalize(self) -> None:
        """Close the cached source handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._reads = {}


class PadMaskWriter(Writer):
    """Boolean ``mask`` column (True = padded) per requested stream (design §8).

    Fixes v1's only-first-task's-stream ``add_mask`` behaviour and the
    internal/external name confusion (``predictionwriter.py:208, 263-265``);
    for single-sequence-stream models (GN2) the default output is
    v1-identical. The v1 ``maybe_pad`` zero-fill quirk is preserved:
    truncated-away positions read ``mask=False``.

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Sequence streams to write a mask for. ``None`` (default) = every
        sequence stream consumed by a configured task.
    """

    def __init__(self, streams: Sequence[str] | None = None) -> None:
        self.streams = tuple(streams) if streams is not None else None

    def _selected(self, ctx: WriteCtx | WriterDeclareCtx) -> tuple[str, ...]:
        """Resolve the masked streams.

        Returns
        -------
        tuple[str, ...]
            Stream names in reader config order.

        Raises
        ------
        ConfigError
            When an explicit entry is not a sequence stream (vector streams
            carry no pad mask, design §6.1).
        """
        if self.streams is not None:
            if unknown := sorted(set(self.streams) - set(ctx.sequence_streams)):
                raise ConfigError(
                    f"PadMaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"streams {unknown} are not sequence streams — pad masks exist for "
                    f"{list(ctx.sequence_streams)} only (design §6.1)"
                )
            return tuple(s for s in ctx.sequence_streams if s in self.streams)
        tasked = {module.stream for module in task_modules(ctx.model_modules).values()}
        return tuple(s for s in ctx.sequence_streams if s in tasked)

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the consumed pad masks.

        Returns
        -------
        dict[str, TensorSpec]
            One pad-mask spec per selected stream.
        """
        return {
            f"masks.{stream}": TensorSpec(shape=None, dtype="bool", kind="pad_mask")
            for stream in self._selected(ctx)
        }

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the ``('mask', '?')`` column per selected stream.

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}``.
        """
        return {stream: np.dtype([("mask", "?")]) for stream in self._selected(ctx)}

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """The statically-derivable eval columns (design §4.4 annotation surface).

        Returns
        -------
        dict[str, list[str]]
            ``{stream: ["mask"]}`` per selected sequence stream.
        """
        del run_name
        return {stream: ["mask"] for stream in self._selected(ctx)}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Format this batch's pad masks (v1 ``predictionwriter.py:208-211``).

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` padded to file length with
            ``mask=False`` (the preserved v1 quirk).
        """
        del rows
        out: dict[str, np.ndarray] = {}
        for stream in self._selected(self.ctx):
            mask = bundle.get(f"masks.{stream}").cpu().numpy()
            arr = u2s(np.expand_dims(mask, -1), dtype=np.dtype([("mask", "?")]))
            out[stream] = _pad_to(arr, self.ctx.seq_lengths[stream])
        return out
