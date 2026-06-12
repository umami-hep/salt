"""Shipped writer modules: task columns, input copies, pad masks (design §8).

Together (in the ``base2.yaml`` order ``inputs_copy -> tasks -> pad_mask``)
these reproduce the v1 `PredictionWriter` per-group column layout byte for
byte: input copies first (full source dtypes, ``predictionwriter.py:186-196``),
task columns next in model declaration order (``:201-205, 255-261``), the
boolean ``mask`` column last (``:208-211``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import h5py
import numpy as np
from ftag import Flavours
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import GraphModule, Mode, TensorSpec
from salt.core.nn.tasks import VertexingTaskModule
from salt.core.writers.base import WriteCtx, Writer, WriterDeclareCtx, task_modules
from salt.utils.array_utils import join_structured_arrays

__all__ = ["InputCopyWriter", "PadMaskWriter", "TaskWriter"]

_VERTEX_COLUMN = "VertexIndex"


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


class TaskWriter(Writer):
    """Persist task predictions with v1 column naming/dtypes (design §8).

    Consumes the inference-ready ``preds.<stream>.<task>`` leaves (tasks
    publish converted values in TEST, design §3.3) and formats them exactly
    as v1's writer-side ``get_h5`` chain:

    - classification: one ``f4`` column per class named
      ``{run_name}_{Flavours[c].px or 'p'+c}`` (``task.py:140-151``), from
      the already-softmaxed probabilities (padded positions read 0.0);
    - vertexing: a single ``('VertexIndex', 'i8')`` column from the
      TEST-mode node assignments via the exact v1 op chain
      ``preds.int().cpu()`` then u2s (``task.py:988-1005``) — padded
      positions read the int32 cast of ``-inf`` (-2147483648).

    **v1-compat decision (M3)**: the vertexing column is BARE
    ``VertexIndex`` by default — no run-name prefix — because the W1 gate's
    byte-schema bar is v1's output (``task.py:1003``). The design §8
    run-name-prefix fix is available behind ``prefix_vertex_column: true``
    (the config converter will set the compat flag, design §8).

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to persist. ``None`` (default) means EVERY stream with a
        configured task — v1's derive-from-the-model behaviour, so
        ``base2.yaml`` ships one generic writer and a new aux task is
        persisted automatically (design §8). An explicit list narrows it;
        the TEST dead-preds error catches a narrowed-list-forgot-a-stream
        mistake.
    prefix_vertex_column : bool, optional
        Name the vertexing column ``{run_name}_VertexIndex`` instead of the
        v1-compatible bare ``VertexIndex``, by default False.
    """

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        prefix_vertex_column: bool = False,
    ) -> None:
        self.streams = tuple(streams) if streams is not None else None
        self.prefix_vertex_column = prefix_vertex_column

    def _selected(self, model_modules: Mapping[str, GraphModule]) -> dict[str, GraphModule]:
        """The task modules this writer persists, in declaration order.

        Returns
        -------
        dict[str, GraphModule]
            ``{instance name: task module}``.

        Raises
        ------
        ConfigError
            When an explicit ``streams`` entry matches no configured task.
        """
        tasks = task_modules(model_modules)
        if self.streams is None:
            return tasks
        known = {module.stream for module in tasks.values()}
        if unknown := sorted(set(self.streams) - known):
            raise ConfigError(
                f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): streams "
                f"{unknown} have no configured task — task streams are {sorted(known)} "
                "(design §8)"
            )
        return {name: m for name, m in tasks.items() if m.stream in self.streams}

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

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}``.

        Raises
        ------
        ConfigError
            For a task family this writer cannot format, or colliding
            column names between two tasks on one stream.
        """
        descrs: dict[str, list[tuple[str, str]]] = {}
        for name, module in self._selected(ctx.model_modules).items():
            descrs.setdefault(module.stream, []).extend(
                self._task_descr(name, module, ctx.run_name)
            )
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

    def _task_descr(self, name: str, module: GraphModule, run_name: str) -> list[tuple[str, str]]:
        """One task's output dtype descr (v1 naming contract).

        Returns
        -------
        list[tuple[str, str]]
            ``(column, format)`` pairs.

        Raises
        ------
        ConfigError
            For an unsupported task family.
        """
        class_names = getattr(module, "class_names", None)
        if class_names is not None:
            # v1 ClassificationTask.output_names (task.py:140-151)
            pxs = [Flavours[c].px if c in Flavours else f"p{c}" for c in class_names]
            return [(f"{run_name}_{px}", "f4") for px in pxs]
        if isinstance(module, VertexingTaskModule):
            column = f"{run_name}_{_VERTEX_COLUMN}" if self.prefix_vertex_column else _VERTEX_COLUMN
            return [(column, "i8")]  # v1 task.py:1003
        raise ConfigError(
            f"TaskWriter {self.name!r} cannot format predictions of module {name!r} "
            f"({type(module).__name__}) — supported families are classification "
            "(class_names attr) and VertexingTaskModule; write a custom Writer for other "
            "outputs (design §8), or narrow this writer's streams"
        )

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Convert this batch's predictions to structured arrays.

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}``; per-token arrays padded to the
            file sequence length.
        """
        del rows
        ctx = self.ctx
        frags: dict[str, list[np.ndarray]] = {}
        for name, module in self._selected(ctx.model_modules).items():
            preds = bundle.get(module.pred_key)
            dtype = np.dtype(self._task_descr(name, module, ctx.run_name))
            arr = self._convert(module, preds, dtype)
            if arr.ndim == 2:
                arr = _pad_to(arr, ctx.seq_lengths[module.stream])
            frags.setdefault(module.stream, []).append(arr)
        return {
            stream: arrays[0] if len(arrays) == 1 else join_structured_arrays(arrays)
            for stream, arrays in frags.items()
        }

    @staticmethod
    def _convert(module: GraphModule, preds: Tensor, dtype: np.dtype) -> np.ndarray:
        """Apply the v1 writer-side tensor->structured conversion per family.

        Returns
        -------
        np.ndarray
            The structured array (``[B]`` or ``[B, L]``).
        """
        if getattr(module, "class_names", None) is not None:
            # v1 ClassificationTask.get_h5 (task.py:266-283): probs -> f4 u2s
            return u2s(preds.float().cpu().numpy(), dtype)
        # v1 VertexingTask.get_h5 (task.py:988-1005): the EXACT op chain —
        # float assignments (-inf padded) -> .int() (int32 cast) -> u2s i8
        return u2s(preds.int().cpu().numpy(), dtype)


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
