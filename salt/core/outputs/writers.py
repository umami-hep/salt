"""Output-section writers — the ``outputs:`` section's `GraphModule` writers.

Three section writers ship here:

- `RunTaskOutput([tasks])` — calls each listed task's ``get_output`` and writes
  the minted ``OutputField`` torch values into ``outputs.<stream>.<leaf>``.
- `InputCopyWriter` — declares ``outputs.<stream>.<var>`` input-copy columns
  (re-read from the source H5 by absolute rows by the sink; this writer carries
  only the manifest + the ``meta.rows`` row anchor demand, not the data).
- `PadMaskWriter` — declares ``outputs.<stream>.mask`` and produces the bool
  pad-mask leaf (read from ``masks.<stream>``); the per-token file-length
  re-expansion (incl. the ``mask=False`` truncation quirk) stays in the sink.

Column order: the dumb H5 sink's ``_merge_columns`` enforces input-copies ->
task outputs -> mask block order. Within the task block, `RunTaskOutput`'s
field order (the v1 model-declaration order, NOT executor topo order) drives
the H5 task-column order, via ``manifest_fields(mode)``.

All section writers are ``GraphModule``s with ``modes=ALL`` ports, demand-
gated: FIT/VAL never demand ``outputs.*`` so the planner drops them; TEST/ONNX
sinks demand the leaves, pulling the writer (and its sources) into the plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.outputs.producers import OutputField
from salt.core.outputs.maskformer import MaskFormerObjectWriter
from salt.core.outputs.writer_base import WriterDeclareCtx

__all__ = [
    "InputCopyWriter",
    "MaskFormerObjectsSink",
    "OutputSectionWriter",
    "PadMaskWriter",
    "RunTaskOutput",
]


class OutputSectionWriter(nn.Module):
    """Shared concrete base for the ``outputs:`` section writers.

    Concrete (instantiable) so the top-level ``outputs:`` CLI namespace can be
    typed ``dict[str, OutputSectionWriter | None]`` and jsonargparse builds each
    writer from its ``class_path``. Carries no behaviour — the three section
    writers supply their own ``declare_io`` / manifest surface.
    """


_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned at assembly."""

# the bundle modes that run get_output (everything but pure FIT/VAL training):
# get_output mints serialisation leaves only for TEST + ONNX. (FIT/VAL prune the
# whole section by demand, so this is a belt-and-braces selector for the
# manifest helpers that resolve fields without a live bundle.)
_OUTPUT_MODES = Mode.TEST | Mode.ONNX


class RunTaskOutput(OutputSectionWriter):
    """The ``outputs:`` section's per-task serialisation orchestrator.

    A `GraphModule` that, for each listed task instance, reads the task's raw
    ``preds.<stream>.<task>`` leaf (forward is loss-space) plus that task's
    output-time deps (``task.output_time_requires(mode)`` — at minimum the
    stream pad mask for a padded sequence head) and calls
    ``task.get_output(b, mode, run_name)``. Each returned `OutputField` carries
    the converted torch ``value`` (softmax / masked-softmax / argmax —
    traceable ops, so ONNX sees them in-graph); ``forward`` writes each value
    into its own ``outputs.<stream>.<task>.<col>`` leaf.

    Leaf naming: a field's H5/ONNX representations can diverge (a per-token
    classification head's H5 columns are the per-class probs, its ONNX output
    the argmax index). Because ``get_output`` mode-keys, a single
    ``outputs.<stream>.<task>.<col>`` leaf carries the probs in H5 modes and the
    argmax index in ONNX — different modes mint different leaf sets, so there
    is no collision.

    ``modes=ALL``, demand-pruned (inert in FIT/VAL, kept alive in TEST/ONNX by
    sink demand). The per-class field split (one field per class) is a sink
    concern (the H5 sink packs ``[B, C]`` into per-class columns; the ONNX sink
    names the per-class scalars) — this writer writes one leaf per field, not
    one stacked leaf, so no sink ever re-splits or re-squeezes a value.

    Parameters
    ----------
    tasks : Sequence[str]
        The task instance names to serialise, in the order their columns appear
        in the eval H5 (the v1 model-declaration order). Each must resolve to a
        task carrying ``get_output`` / ``output_time_requires`` / ``pred_key`` /
        ``stream`` at compile time.

    Raises
    ------
    ConfigError
        For an empty task list or a duplicate task name.
    """

    def __init__(self, tasks: Sequence[str]) -> None:
        super().__init__()
        self.name = _UNNAMED
        names = list(tasks or [])
        if not names:
            raise ConfigError(
                "RunTaskOutput needs a non-empty 'tasks' list — name the task instances whose "
                "get_output() fields this writer serialises (plan 34 W34.2)"
            )
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(
                f"RunTaskOutput: duplicate task name(s) {dup} — one entry per task (plan 34 W34.2)"
            )
        self.tasks = tuple(names)
        # the model module dict, captured at fold/compile so the writer can resolve
        # the tasks it orchestrates (declare_io needs each task's pred_key + stream
        # + output_time_requires + the leaf names get_output mints).
        self._model_modules: Mapping[str, Any] | None = None

    # -- section wiring (bound by SaltModule / cli.py before declare_io) ---------

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model module dict so the writer can resolve its tasks.

        Called before the planner consults ``declare_io``. The writer holds
        only task instance names; it resolves the live task objects from this
        dict to read their ``pred_key`` / ``stream`` / ``output_time_requires``
        / ``get_output``.
        """
        self._model_modules = model_modules

    def _resolved_tasks(self) -> dict[str, Any]:
        """The live task objects this writer orchestrates, in declaration order.

        Raises
        ------
        ConfigError
            When the module dict was not bound, or a named task is absent / does
            not expose the ``get_output`` surface.
        """
        if self._model_modules is None:
            raise ConfigError(
                f"RunTaskOutput {self.name!r} has no model modules bound — it resolves the tasks "
                "it orchestrates from the model (plan 34 W34.2); ensure the outputs: section is "
                "composed after the model (bind_model_modules is called at compile)"
            )
        out: dict[str, Any] = {}
        for task_name in self.tasks:
            task = self._model_modules.get(task_name)
            if task is None:
                raise ConfigError(
                    f"RunTaskOutput {self.name!r}: task {task_name!r} is not a model module — "
                    f"candidates are {sorted(self._model_modules)} (plan 34 W34.2)"
                )
            for attr in ("get_output", "output_time_requires", "pred_key", "stream"):
                if not hasattr(task, attr):
                    raise ConfigError(
                        f"RunTaskOutput {self.name!r}: task {task_name!r} "
                        f"({type(task).__name__}) does not expose {attr!r} — RunTaskOutput "
                        "orchestrates _TaskModuleBase tasks (plan 34 W34.2)"
                    )
            out[task_name] = task
        return out

    def field_leaf_key(self, task: Any, field: OutputField) -> str:
        """The ``outputs.<stream>.<task>.<col>`` leaf one get_output field writes under.

        Per-field leaves (locked, no double-split): each serialisation column
        is its own ``outputs.*`` leaf, so the dumb sinks consume each leaf
        directly with no re-split or re-squeeze. The leaf's last component is
        the field's logical column name (``h5_name`` for an H5 field,
        ``resolved_onnx_name`` for an ONNX-only field) so probs (H5) and the
        argmax index (ONNX) never collide.
        """
        col = field.h5_name if field.h5_name is not None else field.resolved_onnx_name
        return f"outputs.{task.stream}.{task.name}.{col}"

    # -- graph node surface -------------------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """Declare each task's ``preds.*`` (+ deps) -> per-field ``outputs.*`` leaves.

        Requires, per orchestrated task: its raw ``preds.<stream>.<task>`` leaf
        (``kind=data``) plus each ``task.output_time_requires(mode)`` key (e.g.
        ``masks.<stream>`` for a padded seq head, ``kind=pad_mask``). Produces
        one ``outputs.<stream>.<task>.<col>`` leaf per field the task's
        ``get_output_manifest(mode, ...)`` declares. All ports ``modes=ALL``,
        demand-gated. Shapes are ``None`` (rank-agnostic).
        """
        requires: dict[str, TensorSpec] = {}
        produces: dict[str, TensorSpec] = {}
        for task in self._resolved_tasks().values():
            requires[task.pred_key] = TensorSpec(shape=None, dtype=None, kind="data")
            for dep in task.output_time_requires(mode):
                requires.setdefault(dep, _dep_spec(dep))
            for field in _task_manifest(task, mode):
                produces[self.field_leaf_key(task, field)] = TensorSpec(
                    shape=None, dtype=None, kind="data"
                )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run each task's ``get_output`` and write one leaf per field.

        The mode split (probs vs argmax index) is owned by ``get_output``, so
        the produced leaf set matches ``declare_io(mode)``.

        Raises
        ------
        ConfigError
            When a task's ``get_output`` field carries no torch ``value``.
        """
        produced: dict[str, Tensor] = {}
        for task in self._resolved_tasks().values():
            for field in task.get_output(b, mode, self._run_name()):
                if field.value is None:
                    raise ConfigError(
                        f"task {task.name!r}.get_output field carries no value — RunTaskOutput "
                        "writes torch values into the graph (plan 34 W34.2)"
                    )
                produced[self.field_leaf_key(task, field)] = field.value
        return produced

    def _run_name(self) -> str:
        """The run name passed to ``get_output`` (the sink owns the actual prefix)."""
        return "salt"

    # -- section manifest (consumed by the dumb sinks for names/dtypes/order) ----

    def manifest_fields(self, mode: Mode) -> list[tuple[str, OutputField]]:
        """The bundle-free ``(leaf_key, OutputField)`` manifest for the column schema.

        The dumb sinks need the column names/dtypes/order at declare/open time,
        before any batch runs. ``get_output`` reads ``preds.*`` so it cannot run
        without a bundle; instead the task exposes its serialisation-leaf
        metadata through ``get_output_manifest(mode, run_name)`` — the value-
        free twin of ``get_output``. Returns fields tagged with their per-field
        leaf key, in task then field order (the H5/ONNX column-order authority).
        """
        out: list[tuple[str, OutputField]] = []
        for task in self._resolved_tasks().values():
            out.extend(
                (self.field_leaf_key(task, field), field)
                for field in _task_manifest(task, mode)
            )
        return out

    # marker for the dumb-sink discovery (a RunTaskOutput section writer)
    def is_run_task_output(self) -> bool:
        """Mark this as a `RunTaskOutput` section writer (the sink manifest source)."""
        return True


def _dep_spec(dep: str) -> TensorSpec:
    """The require `TensorSpec` for an output-time dep, keyed on its namespace.

    A task's ``output_time_requires`` mixes namespaces: the stream pad mask
    (``masks.<stream>`` -> ``kind=pad_mask`` bool), a regression ratio-
    denominator label (``labels.<stream>.<denom>`` -> ``kind=label`` float, the
    TEST source), and the raw input feature (``inputs.<stream>`` -> ``kind=data``
    float, the ONNX source). The require kind must match the dataset-source kind
    or the planner's kind-unify raises.
    """
    namespace = dep.split(".", 1)[0]
    if namespace == "masks":
        return TensorSpec(shape=None, dtype="bool", kind="pad_mask")
    if namespace == "labels":
        return TensorSpec(shape=None, dtype="float32", kind="label")
    # inputs.* (the ONNX ratio-denominator Feature) — a raw data tensor
    return TensorSpec(shape=None, dtype="float32", kind="data")


def _task_manifest(task: Any, mode: Mode) -> list[OutputField]:
    """The value-free serialisation-leaf metadata for a task in `mode`.

    Prefers the task's own ``get_output_manifest(mode, run_name)`` (the value-
    free twin of ``get_output``); the run name is cosmetic (the sink prefixes).

    Raises
    ------
    ConfigError
        When the task exposes no manifest surface.
    """
    manifest = getattr(task, "get_output_manifest", None)
    if not callable(manifest):
        raise ConfigError(
            f"task {task.name!r} ({type(task).__name__}) ships no get_output_manifest — the dumb "
            "sinks need the column NAMES/DTYPES/ORDER before any batch runs (plan 34 W34.2)"
        )
    return list(manifest(mode, "salt"))


class InputCopyWriter(OutputSectionWriter):
    """The ``outputs:`` section input-copy writer.

    Declares the ``outputs.<stream>.<var>`` input-copy columns the dumb H5 sink
    re-reads from the source H5 by absolute rows (``meta.rows``). The copy is a
    serialisation concern (re-read by the sink through one cached handle, with
    source dtypes/order), so this writer carries only:

    - the column manifest (``section_fields`` / a copy spec the sink consumes), and
    - the ``meta.rows`` row anchor (a TEST require), so the demand closure keeps the
      sink's copy machinery anchored.

    It produces no graph leaf (the copy data never flows through the bundle —
    the sink reads it from the file). It is a manifest-only section writer.
    Matches the legacy ``salt.core.writers.InputCopyWriter`` semantics byte-for-
    byte (copies re-read from the source H5 by absolute rows, all source fields
    by default).

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to copy. ``None`` (default) = every stream the sink resolves with a
        configured task. An explicit list overrides.
    variables : Mapping[str, Sequence[str]] | None, optional
        Per-stream narrowing of the copied fields (v1 ``extra_vars``); a stream not
        listed copies all source fields, by default None.
    """

    name = "inputs_copy"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        variables: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        self.name = type(self).name
        self.streams = tuple(streams) if streams is not None else None
        self.variables = {key: list(val) for key, val in (variables or {}).items()}

    def is_manifest_only(self) -> bool:
        """Mark this writer as a manifest-only section node (no graph leaf).

        The dumb H5 sink reads the copy data from the file (not the bundle), so
        this writer produces nothing — the planner must keep it alive purely as
        a manifest contributor, anchored by its ``meta.rows`` require + the
        sink's demand. The executor never invokes it (no forward).
        """
        return True

    def declare_io(self, mode: Mode) -> IO:
        """Declare the ``meta.rows`` row anchor (TEST); produce nothing.

        Input copies are re-read from the source H5 by the sink (by absolute
        rows from ``meta.rows``), not the graph — so this writer requires only
        the row anchor in TEST and produces no graph leaf. Demand-gated:
        FIT/VAL/ONNX prune it (input copies are eval-H5-only, v1).
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)}
        return IO(requires=unflatten_spec(req), produces={})

    def copy_spec(self) -> dict[str, Any]:
        """The input-copy intent the dumb H5 sink consumes.

        Returns ``{"streams": <list|None>, "variables": {stream: [vars]}}`` —
        the same knobs the legacy InputCopyWriter took; the sink resolves the
        file read.
        """
        return {"streams": list(self.streams) if self.streams is not None else None,
                "variables": {k: list(v) for k, v in self.variables.items()}}


class PadMaskWriter(OutputSectionWriter):
    """The ``outputs:`` section pad-mask writer.

    Declares ``outputs.<stream>.mask`` and produces the bool pad-mask leaf (read
    verbatim from ``masks.<stream>`` — True = padded). The per-token file-length
    re-expansion (incl. the v1 ``mask=False`` truncation quirk) stays in the dumb
    H5 sink. Matches the legacy ``salt.core.writers.PadMaskWriter`` semantics.

    Unlike `InputCopyWriter`, the pad mask IS a bundle leaf (``masks.<stream>``),
    so this writer is a real graph node: it requires ``masks.<stream>`` and
    produces ``outputs.<stream>.mask`` so the leaf flows through the graph to
    the sink, which then packs it as a ``('mask', '?')`` column and pads to the
    file length.

    Parameters
    ----------
    streams : Sequence[str]
        The sequence streams to write a mask column for. Each must be a padded
        sequence stream (validated by the sink against the reader). Required and
        explicit (the v1 default — every sequence stream with a configured task —
        is resolved by the sink, which knows the reader).
    """

    name = "pad_mask"
    """The section instance name (overridable by the config dict key)."""

    def __init__(self, streams: Sequence[str]) -> None:
        super().__init__()
        self.name = type(self).name
        names = list(streams or [])
        if not names:
            raise ConfigError(
                "PadMaskWriter needs a non-empty 'streams' list — name the sequence streams whose "
                "boolean pad-mask column to write (plan 34 W34.2)"
            )
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(
                f"PadMaskWriter: duplicate stream(s) {dup} — one entry per stream (plan 34 W34.2)"
            )
        self.streams = tuple(names)

    def output_key(self, stream: str) -> str:
        """The ``outputs.<stream>.mask`` leaf for a stream."""
        return f"outputs.{stream}.mask"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``masks.<stream>`` -> ``outputs.<stream>.mask`` per stream.

        Demand-gated: FIT/VAL prune it (the mask column is eval-H5-only). The
        require is ``kind=pad_mask`` (unifies with the stream mask producer);
        the produced leaf is ``kind=data`` (a serialisation column).
        """
        del mode
        requires: dict[str, TensorSpec] = {}
        produces: dict[str, TensorSpec] = {}
        for stream in self.streams:
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
            produces[self.output_key(stream)] = TensorSpec(shape=None, dtype="bool", kind="data")
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pass each stream's bool pad mask through to ``outputs.<stream>.mask``.

        The sink owns the ``u2s`` pack + the file-length re-expansion. A fresh
        clone avoids aliasing the bundle's ``masks.*`` leaf (write-once).
        """
        del mode
        return {
            self.output_key(stream): b.get(f"masks.{stream}").clone()
            for stream in self.streams
        }

    def mask_streams(self) -> tuple[str, ...]:
        """The streams a mask column is written for (the sink reads this)."""
        return self.streams


class _MFWriteCtxShim:
    """Minimal stand-in for the legacy `WriteCtx`, carrying only the two fields
    `MaskFormerObjectWriter.write` reads — ``run_name`` + ``precision``.

    The host `H5OutputSink` owns the run name and the f4/f2 float policy, so the
    relocated writer needs nothing else; the object axis ``M`` rides the node
    (``bind_model_modules``), the constituent token length rides the sink-
    supplied `_ExtraGroupCtx`.
    """

    __slots__ = ("precision", "run_name")

    def __init__(self, run_name: str, precision: str) -> None:
        self.run_name = run_name
        self.precision = precision


class MaskFormerObjectsSink(OutputSectionWriter):
    """Sink-hosted MaskFormer object writer.

    Relocates the legacy `salt.core.writers.MaskFormerObjectWriter` TEST role
    onto the `H5OutputSink` ``extra_groups`` seam so MaskFormer's eval-H5 object
    columns are produced on the sink path, not the legacy `WriterCallback`.

    Like `InputCopyWriter` this is a manifest-only section node
    (``is_manifest_only`` -> not graph-folded; it mints no ``outputs.*`` leaf).
    Instead the host `H5OutputSink`:

    1. folds this node's `sink_requires` (the MaskDecoder products
       ``objects.{class_probs,masks}``, the `MaskFormerTargets` labels
       ``labels.objects.{object_class,masks}``, and the constituent pad mask
       ``masks.<constituent_stream>``) into its own TEST demand — anchoring the
       decoder + targets in the TEST plan and threading those leaves into the
       consume bundle; and
    2. calls this node's `write` per batch and merges the returned structured
       arrays into the eval H5 (the ``objects`` / ``object_masks`` non-reader
       extra groups + the ``MaskIndex`` column on the constituent reader stream).

    The object-prediction group is emitted as the v2-native ``objects`` group
    (named by ``object_stream``), a non-reader extra group. It is NOT merged
    into the ``truth_hadrons`` stream group — that merge, and the per-object
    regression eval columns, stay deferred.

    Byte parity: an internal `MaskFormerObjectWriter` (``onnx=False`` — ONNX is
    wired separately via the `MaskFormerObjects` conversion node +
    `OnnxExportSink`) owns ``columns`` + ``write``, so the emitted column set,
    dtypes, the ``[B, M]`` compactness and the ``-2``/``-1`` ``MaskIndex``
    sentinels are produced by the same code as the legacy writer — the sink
    path can never drift from it.

    Parameters
    ----------
    object_classes : Sequence[str]
        All object class names including the trailing ``null`` (the per-object
        probability columns ``{run_name}_p{name}``; v1 ``object.class_names``).
    object_stream : str, optional
        The decoder object stream (the ``objects`` / ``object_masks`` H5 groups
        derive from it), by default ``objects``.
    constituent_stream : str, optional
        The constituent reader stream the masks span; the ``MaskIndex`` column
        lands here, by default ``tracks``.
    regression_task : str, optional
        The object-regression task instance name (carried for symmetry with the
        legacy writer; inert on the eval-only sink path — ``onnx=False``), by
        default ``regression``.
    """

    name = "maskformer_objects"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        object_classes: Sequence[str],
        object_stream: str = "objects",
        constituent_stream: str = "tracks",
        regression_task: str = "regression",
    ) -> None:
        super().__init__()
        self.name = type(self).name
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        # the legacy writer owns the byte-identical columns()/write() ops. onnx=False:
        # ONNX participation is wired separately (the MaskFormerObjects conversion node
        # + OnnxExportSink), so this eval-only fold never declares an export manifest.
        self._writer = MaskFormerObjectWriter(
            object_classes=object_classes,
            object_stream=object_stream,
            constituent_stream=constituent_stream,
            regression_task=regression_task,
            onnx=False,
        )
        self._writer.name = self.name
        # resolved at bind (the object axis M comes from the bound MaskDecoder).
        self._model_modules: Mapping[str, Any] | None = None
        self._num_objects: int | None = None

    def is_manifest_only(self) -> bool:
        """Not graph-folded: the sink owns this node's demand + per-batch write.

        Like `InputCopyWriter`, the data never flows through the graph; the
        host `H5OutputSink` folds `sink_requires` into its demand and calls
        `write` itself.
        """
        return True

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model modules and resolve the object axis ``M``.

        Called by `SaltModule.compose_output_section` before any plan compiles.
        The object-query count ``M`` is read from the bound `MaskDecoder` (the
        ``num_objects`` attribute of the module whose ``out_stream`` is this
        node's ``object_stream``) — the `_ExtraGroupCtx` the sink later threads
        carries no model modules, so ``M`` must be resolved here.
        """
        self._model_modules = dict(model_modules)
        self._num_objects = self._writer._num_objects(self._model_modules)  # noqa: SLF001

    # -- schema (consumed by H5OutputSink at open_schema) -----------------------

    def extra_groups(self, ctx: Any) -> dict[str, tuple[int, ...]]:
        """The two non-reader output groups ``objects`` ``(M,)`` / ``object_masks`` ``(M, T)``.

        Mirrors `MaskFormerObjectWriter.extra_groups`, but reads ``M`` from the
        bound decoder (`bind_model_modules`) rather than ``ctx.model_modules``
        (the sink's `_ExtraGroupCtx` carries only file geometry).

        Raises
        ------
        ConfigError
            When `bind_model_modules` has not run, or the constituent stream has no
            file token length (it must be a sequence stream).
        """
        if self._num_objects is None:
            raise ConfigError(
                f"MaskFormerObjectsSink {self.name!r}: bind_model_modules must run before "
                "extra_groups — the object axis M comes from the bound MaskDecoder (W6b)"
            )
        if self.constituent_stream not in ctx.seq_lengths:
            raise ConfigError(
                f"MaskFormerObjectsSink {self.name!r}: constituent stream "
                f"{self.constituent_stream!r} is not a sequence stream (no file token length) — "
                f"the object masks span its constituents (sequence streams: "
                f"{sorted(ctx.seq_lengths)})"
            )
        tok = ctx.seq_lengths[self.constituent_stream]
        return {self.object_stream: (self._num_objects,), "object_masks": (self._num_objects, tok)}

    def columns(self, ctx: Any) -> dict[str, np.dtype]:
        """The per-group object columns — delegated verbatim to the legacy writer.

        `MaskFormerObjectWriter.columns` reads only ``ctx.run_name`` +
        ``ctx.precision`` (both carried by the sink's `_ExtraGroupCtx`), so the
        byte-identical column schema (``{run_name}_p{class}`` f4 + ``class_label``
        i8 on ``objects``; the ``MaskIndex`` i8 on the constituent stream;
        ``truth_mask`` i8 + ``mask_logits`` f4 on ``object_masks``) is produced
        by the same code the legacy writer ships.
        """
        return self._writer.columns(ctx)

    # -- demand (folded by H5OutputSink into its TEST declare_io requires) -------

    def sink_requires(self) -> dict[str, TensorSpec]:
        """The decoder/truth/pad-mask leaves the host sink must demand on this node's behalf.

        A manifest-only node never anchors its own demand (it is not graph-
        folded); the host `H5OutputSink` folds these into its TEST
        ``declare_io`` requires so the MaskDecoder + `MaskFormerTargets` stay
        alive in the plan and the consume bundle carries the leaves `write`
        reads. Delegated to `MaskFormerObjectWriter.requires` (its ``requires``
        ignores the ctx).
        """
        ctx = WriterDeclareCtx(
            model_modules=self._model_modules or {}, streams=(), sequence_streams=()
        )
        return self._writer.requires(ctx)

    # -- per-batch data (called by H5OutputSink.consume) ------------------------

    def write(
        self, bundle: Bundle, rows: slice, run_name: str, precision: str
    ) -> dict[str, np.ndarray]:
        """One batch of structured arrays — delegated verbatim to the legacy writer.

        The host sink supplies ``run_name`` + ``precision`` (it owns the column
        prefix + float policy); a `_MFWriteCtxShim` carries them to
        `MaskFormerObjectWriter.write`, which packs the exact legacy columns.
        The sink then re-expands the per-token ``MaskIndex`` to the file token
        length and merges the ``objects`` / ``object_masks`` extra groups.
        """
        self._writer.ctx = _MFWriteCtxShim(run_name, precision)
        return self._writer.write(bundle, rows)
