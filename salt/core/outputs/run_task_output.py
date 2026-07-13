"""`OutputSectionWriter` base + `RunTaskOutput` — the ``outputs:`` section's per-task writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.core.outputs.output_field import OutputField


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
