"""Shipped writer modules: task columns, input copies, pad masks (design §8).

Together (in the ``base2.yaml`` order ``inputs_copy -> tasks -> pad_mask``)
these reproduce the v1 `PredictionWriter` per-group column layout byte for
byte: input copies first (full source dtypes, ``predictionwriter.py:186-196``),
task columns next in model declaration order (``:201-205, 255-261``), the
boolean ``mask`` column last (``:208-211``).

M4.5 unified manifest: `TaskWriter` additionally declares its
ONNX-manifest entries (`TaskWriter.onnx_outputs`) from the SAME per-family
suffix helpers that name the TEST columns — one owner per family, both
representations co-located and gated together (amendment §2.2/§3).
`InputCopyWriter` and `PadMaskWriter` are eval-only by design: input
copies are meaningless in ONNX (Athena feeds the inputs) and a pad-mask
output has no Athena consumer (the adapter constructs all-valid masks) —
they keep the base ``onnx_outputs() == []``.
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
from salt.core.nn.tasks import RegressionTaskModule, VertexingTaskModule
from salt.core.onnx.config import ExportOutput
from salt.core.writers.base import WriteCtx, Writer, WriterDeclareCtx, task_modules
from salt.core.writers.names import VERTEX_INDEX, pascal_case
from salt.utils.array_utils import join_structured_arrays

__all__ = ["InputCopyWriter", "PadMaskWriter", "TaskWriter"]

_VERTEX_COLUMN = VERTEX_INDEX
"""Back-compat alias — the constant moved to `salt.core.writers.names`
(M4.5 merge condition 4: one shared suffix module for both modes)."""


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
      positions read the int32 cast of ``-inf`` (-2147483648);
    - regression (M5 sub-wave A): one ``f4`` column per target named
      ``{run_name}_{suffix}`` where the suffix is ``custom_output_names``
      when set else the target (v1 ``RegressionTask.output_names`` +
      ``get_h5``, ``task.py:512-518,604``), from the de-scaled values the
      task publishes in TEST (mode-split de-scaling, design §3.3).

    **v1-compat decision (M3, re-recorded at M4.5)**: the vertexing column
    is BARE ``VertexIndex`` by default — no run-name prefix — because the
    W1 gate's byte-schema bar is v1's output (``task.py:1003``). The design
    §8 run-name-prefix fix is available behind ``prefix_vertex_column:
    true`` (the config converter will set the compat flag, design §8).
    Under the unified manifest the asymmetry is a declared, single-file
    property (amendment §5 rule 6): ONE suffix constant
    (`salt.core.writers.names.VERTEX_INDEX`) feeds BOTH modes, TEST
    prefixes it only when ``prefix_vertex_column`` is set, ONNX always
    prefixes with ``export.model_name`` (v1 parity, ``to_onnx.py:287``).
    Flipping the flag at the M7 adjudication aligns the two names up to the
    prefix value by construction.

    **ONNX manifest** (M4.5): with ``onnx: true`` (default) the writer
    declares one `ExportOutput` per selected task from the SAME suffix
    helpers as the TEST columns — global classification ->
    ``split_scalars`` per-class suffixes, sequence classification -> one
    ``argmax`` int8 entry (Pascal-case of the task instance name,
    ``onnx_names:`` to override), vertexing -> ``vertex_union_find`` int8
    on `VERTEX_INDEX`, regression -> ``split_scalars`` per-target suffixes
    (rename via the task's ``custom_output_names``). Global-stream entries
    are emitted before
    sequence-stream entries regardless of module declaration order (the v1
    ``output_names`` order, ``to_onnx.py:258-292`` — O2/O5 byte parity).
    Adding an aux task therefore lands in eval AND export with zero extra
    config; narrow explicitly with ``onnx_streams``/``onnx_tasks``.

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
    onnx : bool, optional
        Participate in the ONNX manifest, by default True (the M4.5
        polarity adjudication: greenfield defaults to full participation;
        the M7 converter emits explicit knobs — amendment addendum).
        ``False`` makes this writer eval-only.
    onnx_streams : Sequence[str] | None, optional
        Narrow ONNX participation to these task streams (a subset of the
        TEST ``streams`` selection), by default None — every TEST-selected
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
        validated against ``class_names`` — the collision-fix path for two
        classification tasks with overlapping class names). Vertexing
        entries are NOT renameable here: their suffix is the shared
        `VERTEX_INDEX` constant (amendment §2.2).
    """

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        prefix_vertex_column: bool = False,
        onnx: bool = True,
        onnx_streams: Sequence[str] | None = None,
        onnx_tasks: Sequence[str] | None = None,
        onnx_names: Mapping[str, str | list[str]] | None = None,
    ) -> None:
        self.streams = tuple(streams) if streams is not None else None
        self.prefix_vertex_column = prefix_vertex_column
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

    @staticmethod
    def _class_suffixes(module: GraphModule) -> list[str]:
        """A classification task's per-class logical suffixes — ONE owner, BOTH modes.

        The v1 ``ClassificationTask.output_names`` derivation
        (``task.py:140-151``): ``Flavours[c].px`` when the class is a known
        flavour, else ``p{c}``. TEST columns prefix these with the run name
        (`_task_descr`), the ONNX manifest carries them bare for the
        exporter's ``{model_name}_`` prefix (`onnx_outputs`) — the M4.5
        single-ownership refactor (amendment §2.2; byte parity re-gated by
        W1-W5).

        Returns
        -------
        list[str]
            One suffix per ``class_names`` entry, in class order.
        """
        class_names = module.class_names
        return [Flavours[c].px if c in Flavours else f"p{c}" for c in class_names]

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
        if getattr(module, "class_names", None) is not None:
            # v1 ClassificationTask.output_names (task.py:140-151)
            return [(f"{run_name}_{px}", "f4") for px in self._class_suffixes(module)]
        if isinstance(module, VertexingTaskModule):
            column = f"{run_name}_{VERTEX_INDEX}" if self.prefix_vertex_column else VERTEX_INDEX
            return [(column, "i8")]  # v1 task.py:1003
        if isinstance(module, RegressionTaskModule):
            # M5 sub-wave A: regression is a supported family — one
            # `{run_name}_{suffix}` f4 column per target, the suffix being the
            # custom output name when set else the target (v1
            # RegressionTask.output_names + get_h5, task.py:512-518,604). The
            # SAME `output_suffixes` ownership feeds `onnx_outputs` (amendment
            # §2.2 single ownership; M5 decision in callback.
            # _validate_writer_roles + README M4.5 addendum).
            return [(f"{run_name}_{suffix}", "f4") for suffix in module.output_suffixes]
        # The raise stays as the guard for families with no representation at
        # all (e.g. a genuinely custom writer-role task).
        raise ConfigError(
            f"TaskWriter {self.name!r} cannot format predictions of module {name!r} "
            f"({type(module).__name__}) — supported families are classification "
            "(class_names attr) and VertexingTaskModule; write a custom Writer for other "
            "outputs (design §8), or narrow this writer's streams"
        )

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """The statically-derivable eval columns (design §4.4 annotation surface).

        Returns
        -------
        dict[str, list[str]]
            ``{stream: [column names]}`` — the same names `columns`
            declares at run time (same `_task_descr` helper).
        """
        out: dict[str, list[str]] = {}
        for name, module in self._selected(ctx.model_modules).items():
            cols = [column for column, _ in self._task_descr(name, module, run_name)]
            out.setdefault(module.stream, []).extend(cols)
        return out

    # -- ONNX role: the manifest declarations (M4.5 amendment §2.2) -------------

    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The selected tasks' export entries, from the TEST suffix helpers.

        Emission order is v1's ``output_names`` order (``to_onnx.py:
        258-292``): global-stream entries first, then sequence-stream aux
        entries — regardless of module declaration order, so converted
        goldens byte-match O2/O5 (amendment §5 rule 5).

        Returns
        -------
        list[ExportOutput]
            One entry per ONNX-selected task (empty with ``onnx: false``).

        Raises
        ------
        ConfigError
            For unknown ``onnx_streams``/``onnx_tasks``/``onnx_names``
            entries, a malformed ``onnx_names`` value (condition 6), or a
            task family without an export representation.
        """
        if not self.onnx:
            return []
        selected = self._onnx_selected(ctx)
        self._check_onnx_names(selected, ctx)
        ordered = [it for it in selected.items() if it[1].stream not in ctx.sequence_streams]
        ordered += [it for it in selected.items() if it[1].stream in ctx.sequence_streams]
        out: list[ExportOutput] = []
        for name, module in ordered:
            class_names = getattr(module, "class_names", None)
            if class_names is not None and module.stream not in ctx.sequence_streams:
                suffixes = self._onnx_class_suffixes(name, module)
                out.append(ExportOutput(port=module.pred_key, names=suffixes))
            elif class_names is not None:
                out.append(
                    ExportOutput(
                        port=module.pred_key,
                        name=self._onnx_aux_name(name),
                        reduce="argmax",
                        dtype="int8",
                    )
                )
            elif isinstance(module, VertexingTaskModule):
                # the SAME constant as the TEST column — that is the point
                # (amendment §2.2; v1 to_onnx.py:286-288)
                out.append(
                    ExportOutput(
                        port=module.pred_key,
                        name=VERTEX_INDEX,
                        reduce="vertex_union_find",
                        dtype="int8",
                    )
                )
            elif isinstance(module, RegressionTaskModule):
                # M5 sub-wave A: regression is export-representable — one
                # `split_scalars` scalar per target (v1 get_onnx splits the
                # de-scaled `[B, R]` preds into R squeezed scalars,
                # task.py:625-642). The suffixes ARE the TEST `output_suffixes`
                # (custom_output_names else targets) minus the run-name prefix;
                # the exporter prepends `{model_name}_` (amendment §2.2 single
                # ownership). A regression task an author declines to export
                # uses onnx/onnx_tasks/onnx_streams narrowing (handled above).
                out.append(
                    ExportOutput(
                        port=module.pred_key,
                        names=list(module.output_suffixes),
                        reduce="split_scalars",
                    )
                )
            else:
                # The raise stays for a genuinely unrepresentable family (e.g.
                # a custom writer-role task with no shipped reduce).
                raise ConfigError(
                    f"TaskWriter {self.name!r} cannot derive an ONNX output for module "
                    f"{name!r} ({type(module).__name__}) — supported families are "
                    "classification (class_names attr) and VertexingTaskModule; exclude it "
                    f"(onnx_tasks/onnx_streams) or declare it from a custom writer's "
                    "onnx_outputs with a shipped reduce (salt.core.onnx.config."
                    "KNOWN_REDUCES; design §8, M4.5 amendment §3)"
                )
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
            For ``onnx_streams`` entries outside the TEST stream selection
            or ``onnx_tasks`` entries naming no selected task.
        """
        tasks = self._selected(ctx.model_modules)
        if self.onnx_streams is not None:
            known = {module.stream for module in tasks.values()}
            if unknown := sorted(set(self.onnx_streams) - known):
                raise ConfigError(
                    f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"onnx_streams {unknown} match no TEST-selected task stream — selected "
                    f"streams are {sorted(known)} (onnx_streams narrows WITHIN the TEST "
                    "'streams' selection, M4.5 amendment §2.2)"
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

    def _check_onnx_names(self, selected: Mapping[str, GraphModule], ctx: WriterDeclareCtx) -> None:
        """Validate the ``onnx_names`` mapping against the selected tasks (condition 6).

        Raises
        ------
        ConfigError
            For an entry naming no ONNX-selected task, a vertexing entry
            (the suffix is the shared `VERTEX_INDEX` constant), or a value
            of the wrong shape for the task's family: classification
            entries take a per-class LIST (length == ``class_names``),
            sequence (argmax) entries a single string.
        """
        for task_name, value in self.onnx_names.items():
            module = selected.get(task_name)
            if module is None:
                raise ConfigError(
                    f"TaskWriter {self.name!r} (config: writers.modules.{self.name}): "
                    f"onnx_names entry {task_name!r} names no ONNX-selected task — "
                    f"candidates are {sorted(selected)}"
                )
            if isinstance(module, VertexingTaskModule):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: onnx_names cannot rename vertexing task "
                    f"{task_name!r} — its suffix is the shared cross-mode constant "
                    f"{VERTEX_INDEX!r} (salt.core.writers.names; amendment §2.2)"
                )
            if isinstance(module, RegressionTaskModule):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: onnx_names cannot rename regression task "
                    f"{task_name!r} — its per-target suffixes ARE the cross-mode "
                    f"output_suffixes; rename via the task's custom_output_names instead "
                    "(amendment §2.2 single ownership)"
                )
            class_names = getattr(module, "class_names", None)
            if class_names is not None and module.stream not in ctx.sequence_streams:
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ConfigError(
                        f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] must be a LIST "
                        f"of per-class suffixes for a global classification task, got "
                        f"{value!r} (amendment merge condition 6)"
                    )
                if len(value) != len(class_names):
                    raise ConfigError(
                        f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] lists "
                        f"{len(value)} suffixes but the task has {len(class_names)} classes "
                        f"({list(class_names)}) — one suffix per class, in class order "
                        "(amendment merge condition 6)"
                    )
            elif not isinstance(value, str):
                raise ConfigError(
                    f"TaskWriter {self.name!r}: onnx_names[{task_name!r}] must be a single "
                    f"string for a sequence (argmax) task, got {value!r} "
                    "(amendment merge condition 6)"
                )

    def _onnx_class_suffixes(self, name: str, module: GraphModule) -> list[str]:
        """A global classification task's ONNX suffixes (override or shared helper).

        Returns
        -------
        list[str]
            The ``onnx_names`` override when given (length pre-validated),
            else the SAME `_class_suffixes` list the TEST columns use.
        """
        override = self.onnx_names.get(name)
        if override is not None:
            return list(override)
        return self._class_suffixes(module)

    def _onnx_aux_name(self, name: str) -> str:
        """A sequence task's single ONNX suffix (override or Pascal-case default).

        Returns
        -------
        str
            ``onnx_names[name]`` when given, else `pascal_case` of the
            task instance name — reproducing v1's hand-built strings
            (``track_origin -> TrackOrigin``, ``to_onnx.py:283-292``).
        """
        override = self.onnx_names.get(name)
        if override is not None:
            return str(override)
        return pascal_case(name)

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
        if isinstance(module, RegressionTaskModule):
            # v1 RegressionTask.get_h5 (task.py:604-623): de-scaled values
            # (the forward already inverted scaling in TEST) -> f4 u2s
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
