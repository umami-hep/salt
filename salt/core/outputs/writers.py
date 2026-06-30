"""Output-section writers — the ``outputs:`` section's `GraphModule` writers (plan 34 W34.2).

The plan-34 ``outputs:`` section is the layer between ``model.modules`` (which
now publishes RAW loss-space ``preds.*`` and the loss) and the dumb sinks (which
serialise ALL active ``outputs.*`` leaves). Instead of the plan-29/31 standalone
conversion PRODUCER nodes (``ClassProbs`` / ``SeqClassProbs`` / ...), the 1:1
conversion math FOLDS back onto the task as ``task.get_output(b, mode, run_name)``
(W34.1), orchestrated by a single `RunTaskOutput([tasks])` writer here.

Three section writers ship here (plan 34 §2):

- `RunTaskOutput([tasks])` — calls each listed task's ``get_output`` and writes
  the minted ``OutputField`` torch values into ``outputs.<stream>.<leaf>``. The
  H5/ONNX representation split is keyed on ``mode`` INSIDE ``get_output`` (probs
  for H5 modes, per-class scalars / argmax index for ONNX), so the write-once
  ``outputs.*`` constraint holds (distinct probs vs index leaf names, W34.1).
- `InputCopyWriter` — declares ``outputs.<stream>.<var>`` input-copy columns
  (re-read from the source H5 by absolute rows by the SINK; this writer carries
  only the manifest + the ``meta.rows`` row anchor demand, not the data — the
  copy is a serialisation concern, plan §5).
- `PadMaskWriter` — declares ``outputs.<stream>.mask`` and produces the bool
  pad-mask leaf (read from ``masks.<stream>``); the per-token file-length
  re-expansion (incl. the ``mask=False`` truncation quirk) stays in the SINK.

**Column ORDER ownership** (plan §4 W34.2 / §7 risk 4): the copies -> tasks ->
mask BLOCK order is enforced by the dumb H5 sink's ``_merge_columns`` (input copies
first, then the task output columns, then the pad mask). WITHIN the task block, the
section's ``RunTaskOutput`` field order — the v1 model-declaration order — drives the
H5 TASK-column order, NOT the executor topo order, so reordering for memory tuning
never reshuffles the eval H5 task columns. Each ``RunTaskOutput`` exposes
``manifest_fields(mode) -> list[(output_key, OutputField)]`` so the dumb sinks collect
the task-column names/dtypes/order from the SECTION (not from producer discovery) in
declaration order.

All section writers are ``GraphModule``s with ``modes=ALL`` ports, gated by
DEMAND: in FIT/VAL nothing demands ``outputs.*`` (losses read ``preds.*``), so
the planner's demand closure drops them and the FIT/VAL plan_hash is byte-
unchanged (plan §6 gate 4). In TEST/ONNX a sink demands the ``outputs.*`` leaf,
pulling the writer (and transitively its ``preds.*`` / pad-mask sources) into the
plan.
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
    """Shared concrete base for the ``outputs:`` section writers (plan 34 W34.2).

    A concrete (instantiable) base so the top-level ``outputs:`` CLI namespace can
    be typed ``dict[str, OutputSectionWriter | None]`` and jsonargparse builds each
    writer from its ``class_path`` (the same surface ``writers.modules`` /
    ``callbacks`` use). Carries no behaviour — the three section writers
    (`RunTaskOutput`, `InputCopyWriter`, `PadMaskWriter`) supply their own
    ``declare_io`` / manifest surface.
    """


_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned at assembly."""

# the bundle modes that run get_output (everything but pure FIT/VAL training):
# get_output mints serialisation leaves only for TEST + ONNX. (FIT/VAL prune the
# whole section by demand, so this is a belt-and-braces selector for the
# manifest helpers that resolve fields without a live bundle.)
_OUTPUT_MODES = Mode.TEST | Mode.ONNX


class RunTaskOutput(OutputSectionWriter):
    """The ``outputs:`` section's per-task serialisation orchestrator (plan 34 W34.2).

    A `GraphModule` that, for each listed task instance, reads the task's RAW
    ``preds.<stream>.<task>`` leaf (forward is loss-space since W34.1/W34.3) plus
    that task's output-time deps (``task.output_time_requires(mode)`` — at minimum
    the stream pad mask for a padded sequence head) and calls
    ``task.get_output(b, mode, run_name)``. Each returned `OutputField` carries the
    converted torch ``value`` (softmax / masked-softmax / argmax — traceable ops,
    so ONNX sees them in-graph); ``forward`` writes each value into
    ``outputs.<stream>.<leafname>``.

    Leaf naming (write-once ``outputs.*`` constraint, plan §4 / W34.1): a field's
    H5/ONNX representations can DIVERGE (a per-token classification head's H5
    columns are the per-class probs, its ONNX output the argmax index). The
    distinct ``outputs.*`` LEAF the field is written under is derived from the
    field's logical role so probs and index never collide:

    - a global head's per-class probs -> ``outputs.<stream>.<task>`` (one leaf,
      the per-class ``[B, C]`` value stacked back — see below).
    - a seq head's H5 probs -> ``outputs.<stream>.<task>`` (the ``[B, L, C]``
      probs leaf).
    - a seq head's ONNX argmax -> ``outputs.<stream>.<task>`` carries the int8
      index in ONNX mode (the H5 probs leaf is mode-pruned in ONNX, so there is
      no collision — they are DIFFERENT modes of the same logical leaf).

    Because ``get_output`` mode-keys (W34.1), a single ``outputs.<stream>.<task>``
    leaf carries the probs in H5 modes and the argmax index in ONNX — the sink
    consumes the mode-appropriate one. ``declare_io`` therefore declares ONE leaf
    per ``(task, leaf-name)`` pair, active in every mode (demand-gated), and the
    forward writes the per-mode value.

    ``modes=ALL``, demand-pruned (inert in FIT/VAL, kept alive in TEST/ONNX by
    sink demand). The per-class field SPLIT (one field per class, each carrying
    that class's column) is a SINK concern (the H5 sink packs ``[B, C]`` into per-
    class columns; the ONNX sink names the per-class scalars) — so this writer
    writes ONE multi-channel torch leaf per ``get_output`` group, reassembled from
    the per-class field values, NOT one leaf per class (the leaf is the graph
    edge; the columns are the serialisation expansion the sink owns).

    Parameters
    ----------
    tasks : Sequence[str]
        The task instance names to serialise, in the order their columns appear
        in the eval H5 (the v1 model-declaration order). Each must resolve to a
        task carrying ``get_output`` / ``output_time_requires`` / ``pred_key`` /
        ``stream`` at compile time (validated when the model module dict is
        bound, since the writer holds only names).

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
        """Capture the model module dict so the writer can resolve its tasks (plan 34 W34.2).

        Called before the planner consults ``declare_io`` (mirroring the sink's
        ``bind_model_modules``). The writer holds only task INSTANCE NAMES; it
        resolves the live task objects from this dict to read their
        ``pred_key`` / ``stream`` / ``output_time_requires`` / ``get_output``.
        """
        self._model_modules = model_modules

    def _resolved_tasks(self) -> dict[str, Any]:
        """The live task objects this writer orchestrates, in declaration order.

        Returns
        -------
        dict[str, Any]
            ``{instance name: task module}``.

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
        """The ``outputs.<stream>.<task>.<col>`` leaf ONE get_output field writes under.

        PER-FIELD leaves (plan §4 W34.2 LOCKED no-double-split decision): each
        serialisation COLUMN is its own ``outputs.*`` leaf — a global head's per-
        class probs become C separate leaves, a seq argmax becomes one leaf. The
        dumb sinks then consume each leaf DIRECTLY (the H5 sink packs each into its
        single column; the ONNX sink NAMES each already-scalar value — NO re-split,
        NO re-squeeze, honouring the W34.1 in-``get_output`` squeeze). The leaf's
        last component is the field's logical column name (``h5_name`` for an H5
        field, ``resolved_onnx_name`` for an ONNX-only field) so probs (H5) and the
        argmax index (ONNX) never collide (different modes mint different leaf sets).

        Returns
        -------
        str
            ``outputs.<stream>.<task instance name>.<field column>``.
        """
        col = field.h5_name if field.h5_name is not None else field.resolved_onnx_name
        return f"outputs.{task.stream}.{task.name}.{col}"

    # -- graph node surface (design §2.5) ---------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """Declare each task's ``preds.*`` (+ deps) -> per-field ``outputs.*`` leaves (W34.2).

        Requires, per orchestrated task: its raw ``preds.<stream>.<task>`` leaf
        (``kind=data``) PLUS each ``task.output_time_requires(mode)`` key (e.g.
        ``masks.<stream>`` for a padded seq head, ``kind=pad_mask``). Produces ONE
        ``outputs.<stream>.<task>.<col>`` leaf PER serialisation field the task's
        ``get_output_manifest(mode, ...)`` declares (the PER-FIELD leaf model — the
        LOCKED no-double-split decision, plan §4 W34.2). All ports ``modes=ALL``:
        demand-gated (FIT/VAL prune the whole section — plan §6 gate 4). Shapes are
        ``None`` (rank-agnostic).

        Returns
        -------
        IO
            The declared requires/produces for this writer in `mode`.
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
        """Run each task's ``get_output`` and write one leaf PER field (plan 34 W34.2).

        For each orchestrated task, calls ``task.get_output(b, mode, run_name)`` and
        writes each ``OutputField``'s torch ``value`` into its own
        ``outputs.<stream>.<task>.<col>`` leaf (the per-field-leaf model — NO
        stacking, so the ONNX sink names already-scalar values without re-splitting,
        plan §4 W34.2 LOCKED). The mode split (probs vs argmax index) is owned by
        ``get_output`` (W34.1), so the produced leaf set matches ``declare_io(mode)``.

        Returns
        -------
        dict[str, Tensor]
            ``{outputs.<stream>.<task>.<col>: field value}`` per field per task.

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
        """The run name passed to ``get_output`` (the sink owns the prefix, so this is cosmetic).

        Returns
        -------
        str
            A constant placeholder — ``get_output`` does not bake the run name in.
        """
        return "salt"

    # -- section manifest (consumed by the dumb sinks for names/dtypes/order) ----

    def manifest_fields(self, mode: Mode) -> list[tuple[str, OutputField]]:
        """The bundle-free ``(leaf_key, OutputField)`` manifest for the column schema (W34.2).

        The dumb sinks need the column NAMES / DTYPES / ORDER at declare/open time,
        before any batch runs. ``get_output`` reads ``preds.*`` so it cannot run
        without a bundle; instead the task exposes its serialisation-leaf metadata
        through ``get_output_manifest(mode, run_name)`` — the value-free twin of
        ``get_output``. This returns those metadata fields tagged with their PER-
        FIELD leaf key (`field_leaf_key`), in task then field order (the H5/ONNX
        column-order authority, plan §4 / §7 risk 4).

        Returns
        -------
        list[tuple[str, OutputField]]
            ``(leaf_key, field)`` per serialisation leaf (``value`` is None), in
            task then field order.
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
        """Mark this as a `RunTaskOutput` section writer (the sink manifest source).

        Returns
        -------
        bool
            Always True.
        """
        return True


def _dep_spec(dep: str) -> TensorSpec:
    """The require `TensorSpec` for an output-time dep, keyed on its namespace (plan 34 W34.3).

    A task's ``output_time_requires`` mixes namespaces (plan §4): the stream pad mask
    (``masks.<stream>`` -> ``kind=pad_mask`` bool), a regression ratio-denominator
    LABEL (``labels.<stream>.<denom>`` -> ``kind=label`` float, the TEST source), and
    the raw input Feature (``inputs.<stream>`` -> ``kind=data`` float, the ONNX source).
    The require kind MUST match the dataset-source kind or the planner's kind-unify
    raises (a label dep declared ``pad_mask`` is the W34.3 regression-cutover bug).

    Returns
    -------
    TensorSpec
        The namespace-appropriate require spec (shape-agnostic; the source/producer
        carries the concrete shape).
    """
    namespace = dep.split(".", 1)[0]
    if namespace == "masks":
        return TensorSpec(shape=None, dtype="bool", kind="pad_mask")
    if namespace == "labels":
        return TensorSpec(shape=None, dtype="float32", kind="label")
    # inputs.* (the ONNX ratio-denominator Feature) — a raw data tensor
    return TensorSpec(shape=None, dtype="float32", kind="data")


def _task_manifest(task: Any, mode: Mode) -> list[OutputField]:
    """The value-free serialisation-leaf metadata for a task in `mode` (plan 34 W34.2).

    Prefers the task's own ``get_output_manifest(mode, run_name)`` (the value-free
    twin of ``get_output``); the run name is cosmetic (the sink prefixes). The base
    classification family ships this twin in tasks.py.

    Returns
    -------
    list[OutputField]
        The serialisation fields (``value`` None), in field order.

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
    """The ``outputs:`` section input-copy writer (plan 34 W34.2; folds the legacy InputCopyWriter).

    Declares the ``outputs.<stream>.<var>`` input-copy columns the dumb H5 sink
    re-reads from the SOURCE H5 by absolute rows (``meta.rows``). The COPY is a
    serialisation concern (re-read by the sink through one cached handle, with
    SOURCE dtypes/order — the v1 input-copy contract), so this writer carries only:

    - the column MANIFEST (``section_fields`` / a copy spec the sink consumes), and
    - the ``meta.rows`` row anchor (a TEST require), so the demand closure keeps the
      sink's copy machinery anchored.

    It produces NO graph leaf (the copy data never flows through the bundle — the
    sink reads it from the file). It is a manifest-only section writer: it declares
    its copy intent to the sink, and the sink owns the file read + pack. Matches the
    legacy ``salt.core.writers.InputCopyWriter`` semantics byte-for-byte (copies
    re-read from the source H5 by absolute rows, all source fields by default).

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to copy. ``None`` (default) = every stream the sink resolves with a
        configured task (the v1 output-group set). An explicit list overrides.
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
        this writer produces nothing — the planner must keep it alive purely as a
        manifest contributor, anchored by its ``meta.rows`` require + the sink's
        demand. The executor never invokes it (no forward); it carries only the
        copy spec the sink reads.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def declare_io(self, mode: Mode) -> IO:
        """Declare the ``meta.rows`` row anchor (TEST); produce nothing (plan 34 W34.2).

        Input copies are re-read from the source H5 by the SINK (by absolute rows
        from ``meta.rows``), not the graph — so this writer requires only the row
        anchor in TEST and produces no graph leaf. ``modes=ALL`` on the require,
        demand-gated: FIT/VAL/ONNX prune it (input copies are eval-H5-only, v1).

        Returns
        -------
        IO
            ``meta.rows`` require in TEST, empty produces; empty otherwise.
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)}
        return IO(requires=unflatten_spec(req), produces={})

    def copy_spec(self) -> dict[str, Any]:
        """The input-copy intent the dumb H5 sink consumes (plan 34 W34.2).

        Returns
        -------
        dict[str, Any]
            ``{"streams": <list|None>, "variables": {stream: [vars]}}`` — the same
            knobs the legacy InputCopyWriter took; the sink resolves the file read.
        """
        return {"streams": list(self.streams) if self.streams is not None else None,
                "variables": {k: list(v) for k, v in self.variables.items()}}


class PadMaskWriter(OutputSectionWriter):
    """The ``outputs:`` section pad-mask writer (plan 34 W34.2; folds the legacy PadMaskWriter).

    Declares ``outputs.<stream>.mask`` and produces the bool pad-mask leaf (read
    verbatim from ``masks.<stream>`` — True = padded). The per-token file-length
    re-expansion (incl. the v1 ``mask=False`` truncation quirk) stays in the dumb
    H5 sink (a serialisation concern, plan §5). Matches the legacy
    ``salt.core.writers.PadMaskWriter`` semantics.

    Unlike `InputCopyWriter`, the pad mask IS a bundle leaf (``masks.<stream>``), so
    this writer is a REAL graph node: it requires ``masks.<stream>`` and produces
    ``outputs.<stream>.mask`` so the leaf flows through the graph to the sink. The
    sink then packs it as a ``('mask', '?')`` column and pads to the file length.

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
        """The ``outputs.<stream>.mask`` leaf for a stream.

        Returns
        -------
        str
            ``outputs.<stream>.mask``.
        """
        return f"outputs.{stream}.mask"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``masks.<stream>`` -> ``outputs.<stream>.mask`` per stream (plan 34 W34.2).

        ``modes=ALL``, demand-gated: FIT/VAL prune it (the mask column is eval-H5-
        only). The require is ``kind=pad_mask`` (unifies with the stream mask
        producer); the produced leaf is ``kind=data`` (a serialisation column).

        Returns
        -------
        IO
            One ``masks.<stream>`` -> ``outputs.<stream>.mask`` edge per stream.
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

        The sink owns the ``u2s`` pack + the file-length re-expansion (the
        ``mask=False`` truncation quirk). A fresh clone avoids aliasing the bundle's
        ``masks.*`` leaf (write-once).

        Returns
        -------
        dict[str, Tensor]
            ``{outputs.<stream>.mask: bool tensor}`` per stream.
        """
        del mode
        return {
            self.output_key(stream): b.get(f"masks.{stream}").clone()
            for stream in self.streams
        }

    def mask_streams(self) -> tuple[str, ...]:
        """The streams a mask column is written for (the sink reads this).

        Returns
        -------
        tuple[str, ...]
            The configured streams, in declaration order.
        """
        return self.streams


class _MFWriteCtxShim:
    """Minimal stand-in for the legacy `WriteCtx`, carrying ONLY the two fields
    `MaskFormerObjectWriter.write` reads — ``run_name`` + ``precision`` (W6b).

    The host `H5OutputSink` owns the run name and the f4/f2 float policy, so the
    relocated writer needs nothing else from the legacy per-test-run ctx; the
    object axis ``M`` rides the node (``bind_model_modules``), the constituent
    token length rides the sink-supplied `_ExtraGroupCtx`.
    """

    __slots__ = ("precision", "run_name")

    def __init__(self, run_name: str, precision: str) -> None:
        self.run_name = run_name
        self.precision = precision


class MaskFormerObjectsSink(OutputSectionWriter):
    """Sink-hosted MaskFormer object writer — the W6b fold (USER DECISION 2026-06-30).

    Relocates the legacy `salt.core.writers.MaskFormerObjectWriter` TEST role onto
    the `H5OutputSink` ``extra_groups`` seam (W6a) so MaskFormer's eval-H5 object
    columns are produced on the SINK path, NOT the legacy `WriterCallback` — the
    last writer family standing between MaskFormer and a fully sink-native eval H5.

    Like `InputCopyWriter` this is a MANIFEST-ONLY section node (``is_manifest_only``
    -> not graph-folded; it mints no ``outputs.*`` leaf). Instead the host
    `H5OutputSink`:

    1. folds this node's `sink_requires` (the MaskDecoder products
       ``objects.{class_probs,masks}``, the `MaskFormerTargets` labels
       ``labels.objects.{object_class,masks}``, and the constituent pad mask
       ``masks.<constituent_stream>``) into its OWN TEST demand — anchoring the
       decoder + targets in the TEST plan and threading those leaves into the
       consume bundle; and
    2. calls this node's `write` per batch and merges the returned structured
       arrays into the eval H5 (the ``objects`` / ``object_masks`` NON-reader extra
       groups + the ``MaskIndex`` column on the constituent reader stream).

    The object-prediction group is emitted as the v2-native **``objects``** group
    (named by ``object_stream``), a NON-reader extra group that passes the W6a
    shadow check cleanly. It is NOT merged into the ``truth_hadrons`` stream group
    — the full upstream truth_hadrons eval-H5 merge is DEFERRED, and the per-object
    regression eval columns stay DEFERRED too (USER DECISION 2026-06-30).

    Byte parity (drift-proof by construction): an INTERNAL `MaskFormerObjectWriter`
    (``onnx=False`` — ONNX is wired separately via the `MaskFormerObjects`
    conversion node + `OnnxExportSink`) owns ``columns`` + ``write``, so the emitted
    column set, dtypes, the ``[B, M]`` compactness and the ``-2``/``-1`` ``MaskIndex``
    sentinels are produced by the SAME code as MFU-6's writer — the sink path can
    never drift from the legacy writer.

    Parameters
    ----------
    object_classes : Sequence[str]
        ALL object class names INCLUDING the trailing ``null`` (the per-object
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
        """Not graph-folded: the sink owns this node's demand + per-batch write (W6b).

        Returns
        -------
        bool
            Always True — like `InputCopyWriter`, the data never flows through the
            graph; the host `H5OutputSink` folds `sink_requires` into its demand and
            calls `write` itself.
        """
        return True

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model modules and resolve the object axis ``M`` (W6b).

        Called by `SaltModule.compose_output_section` before any plan compiles.
        The object-query count ``M`` is read from the bound `MaskDecoder` (the
        ``num_objects`` attribute of the module whose ``out_stream`` is this node's
        ``object_stream``) — the `_ExtraGroupCtx` the sink later threads carries NO
        model modules (W6a), so ``M`` must be resolved here.
        """
        self._model_modules = dict(model_modules)
        self._num_objects = self._writer._num_objects(self._model_modules)  # noqa: SLF001

    # -- schema (consumed by H5OutputSink at open_schema) -----------------------

    def extra_groups(self, ctx: Any) -> dict[str, tuple[int, ...]]:
        """The two NON-reader output groups ``objects`` ``(M,)`` / ``object_masks`` ``(M, T)``.

        Mirrors `MaskFormerObjectWriter.extra_groups`, but reads ``M`` from the
        bound decoder (`bind_model_modules`) rather than ``ctx.model_modules`` (the
        sink's `_ExtraGroupCtx` carries only file geometry).

        Returns
        -------
        dict[str, tuple[int, ...]]
            ``{object_stream: (M,), "object_masks": (M, T)}``.

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
        """The per-group object columns — delegated VERBATIM to the legacy writer (W6b).

        `MaskFormerObjectWriter.columns` reads only ``ctx.run_name`` + ``ctx.precision``
        (both carried by the sink's `_ExtraGroupCtx`), so the byte-identical column
        schema (``{run_name}_p{class}`` f4 + ``class_label`` i8 on ``objects``; the
        ``MaskIndex`` i8 on the constituent stream; ``truth_mask`` i8 + ``mask_logits``
        f4 on ``object_masks``) is produced by the SAME code MFU-6 ships.

        Returns
        -------
        dict[str, np.dtype]
            ``{object_stream: dtype, constituent_stream: dtype, "object_masks": dtype}``.
        """
        return self._writer.columns(ctx)

    # -- demand (folded by H5OutputSink into its TEST declare_io requires) -------

    def sink_requires(self) -> dict[str, TensorSpec]:
        """The decoder/truth/pad-mask leaves the host sink must demand on this node's behalf.

        A manifest-only node never anchors its own demand (it is not graph-folded);
        the host `H5OutputSink` folds these into its TEST ``declare_io`` requires so
        the MaskDecoder + `MaskFormerTargets` stay alive in the plan and the consume
        bundle carries the leaves `write` reads. Delegated to
        `MaskFormerObjectWriter.requires` (its ``requires`` ignores the ctx).

        Returns
        -------
        dict[str, TensorSpec]
            ``objects.{class_probs,masks}`` + ``labels.objects.{object_class,masks}``
            + ``masks.<constituent_stream>``.
        """
        ctx = WriterDeclareCtx(
            model_modules=self._model_modules or {}, streams=(), sequence_streams=()
        )
        return self._writer.requires(ctx)

    # -- per-batch data (called by H5OutputSink.consume) ------------------------

    def write(
        self, bundle: Bundle, rows: slice, run_name: str, precision: str
    ) -> dict[str, np.ndarray]:
        """One batch of structured arrays — delegated VERBATIM to the legacy writer (W6b).

        The host sink supplies ``run_name`` + ``precision`` (it owns the column
        prefix + float policy); a `_MFWriteCtxShim` carries them to
        `MaskFormerObjectWriter.write`, which packs the exact MFU-6 columns. The
        sink then re-expands the per-token ``MaskIndex`` to the file token length
        and merges the ``objects`` / ``object_masks`` extra groups.

        Returns
        -------
        dict[str, np.ndarray]
            ``{object_stream: arr, constituent_stream: arr, "object_masks": arr}``.
        """
        self._writer.ctx = _MFWriteCtxShim(run_name, precision)
        return self._writer.write(bundle, rows)
