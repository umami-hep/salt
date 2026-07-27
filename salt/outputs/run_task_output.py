"""`OutputSectionWriter` base + `RunTaskOutput` — the ``outputs:`` section's per-task writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.outputs.output_schema import OutputField

_OUTPUT_MODE_NAMES = {"test": Mode.TEST, "export": Mode.ONNX}
"""The YAML ``modes:`` vocabulary — ``test`` -> Mode.TEST, ``export`` -> Mode.ONNX."""


def parse_output_modes(modes: Any, who: str) -> Mode:
    """Map a YAML ``modes:`` list onto the Mode flag the section writer runs in.

    ``None`` (omitted) -> ``Mode.TEST | Mode.ONNX`` (both, the pre-plan-50
    default). A non-empty list of ``test``/``export`` names ORs into the flag.
    Raises `ConfigError` on an empty list or an unknown mode name (naming the key).
    """  # noqa: DOC201, DOC501 - internal helper, no Returns/Raises blocks per docstring policy
    if modes is None:
        return Mode.TEST | Mode.ONNX
    names = [modes] if isinstance(modes, str) else list(modes)
    if not names:
        raise ConfigError(
            f"{who}: 'modes' is an empty list — name at least one of "
            f"{sorted(_OUTPUT_MODE_NAMES)} (or omit 'modes' entirely for both)"
        )
    out = Mode(0)
    for name in names:
        flag = _OUTPUT_MODE_NAMES.get(str(name).lower())
        if flag is None:
            raise ConfigError(
                f"{who}: unknown output mode {name!r} — valid names are "
                f"{sorted(_OUTPUT_MODE_NAMES)} (test -> Mode.TEST eval H5, export -> Mode.ONNX)"
            )
        out |= flag
    return out


class OutputSectionWriter(SaltModelModule):
    """Shared base for the ``outputs:`` section writers.

    Named (not just `SaltModelModule` directly) so the top-level ``outputs:``
    CLI namespace can be typed ``dict[str, OutputSectionWriter | None]`` and
    jsonargparse builds each writer from its ``class_path``. Beyond the
    `SaltModelModule` contract it owns the ``modes:`` surface (plan 50): each
    section writer declares the modes it runs in (``test``/``export``), which
    the command uses to pick the implicit per-command sink and which gates the
    writer's own `declare_io` / manifest so an ``export``-omitted writer mints
    no ONNX leaves.

    Parameters
    ----------
    modes : Sequence[str] | None, optional
        The modes this writer participates in — a subset of ``["test",
        "export"]``. ``None`` (default) = both (the pre-plan-50 behaviour).
    """

    def __init__(self, modes: Sequence[str] | None = None) -> None:
        super().__init__()
        self._section_modes = parse_output_modes(modes, type(self).__name__)

    def section_modes(self) -> Mode:
        """The Mode flag this writer runs in."""  # noqa: DOC201 - getter, one-line
        return self._section_modes

    def runs_in_mode(self, mode: Mode) -> bool:
        """Whether this writer runs in `mode`."""  # noqa: DOC201 - getter, one-line
        return bool(mode & self._section_modes)


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
        The task instance names to serialise, in the order their columns
        appear in the eval H5 (the model-declaration order). Each must
        resolve to a task carrying ``get_output`` / ``output_time_requires``
        / ``pred_key`` / ``stream`` at compile time.
    modes : Sequence[str] | None, optional
        The modes this writer serialises in (``["test", "export"]`` subset;
        ``None`` = both). A ``test``-only writer mints no ONNX leaves (so the
        implicit ONNX sink names none of its fields); an ``export``-only
        writer contributes no eval-H5 columns.

    Raises
    ------
    ConfigError
        For an empty task list, a duplicate task name, or an unknown mode name.
    """

    def __init__(self, tasks: Sequence[str], modes: Sequence[str] | None = None) -> None:
        super().__init__(modes=modes)
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
        """The live task objects this writer orchestrates, in declaration order;
        raises `ConfigError` when unbound or a named task lacks the
        ``get_output`` surface.
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
        """Per task: requires its raw preds + output-time deps; produces its output fields.

        Gated by the writer's ``modes:`` list — a mode the writer opts out of
        (e.g. ``export`` on a ``modes: [test]`` writer) declares nothing, so the
        planner prunes it and the mode's implicit sink names none of its leaves.
        """
        if not self.runs_in_mode(mode):
            return IO(requires={}, produces={})
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
        """Run each task's ``get_output`` and write one leaf per field (the mode
        split is owned by ``get_output``, matching ``declare_io(mode)``); raises
        `ConfigError` if a field carries no torch value.
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
        Empty when the writer opts out of `mode` (its ``modes:`` list).
        """
        if not self.runs_in_mode(mode):
            return []
        out: list[tuple[str, OutputField]] = []
        for task in self._resolved_tasks().values():
            out.extend(
                (self.field_leaf_key(task, field), field) for field in _task_manifest(task, mode)
            )
        return out

    # marker for the dumb-sink discovery (a RunTaskOutput section writer)
    def is_run_task_output(self) -> bool:
        """Mark this as a `RunTaskOutput` section writer (the sink manifest source)."""
        return True


def _dep_spec(dep: str) -> TensorSpec:
    """The require `TensorSpec` for an output-time dep, keyed on its namespace.

    A task's ``output_time_requires`` mixes namespaces: the stream pad mask
    (``masks.<stream>`` -> ``kind=pad_mask`` bool), a label (``labels.<stream>.
    <var>`` -> ``kind=label``, the TEST source for ratio denominators AND the
    Phase-C target-label columns — dtype unconstrained since label dtypes vary
    per label: int64 class/vertex labels vs float32 regression targets), and
    the raw input feature (``inputs.<stream>`` -> ``kind=data`` float, the
    ONNX source). The require kind must match the dataset-source kind or the
    planner's kind-unify raises.
    """
    namespace = dep.split(".", 1)[0]
    if namespace == "masks":
        return TensorSpec(shape=None, dtype="bool", kind="pad_mask")
    if namespace == "labels":
        return TensorSpec(shape=None, dtype=None, kind="label")
    # inputs.* (the ONNX ratio-denominator Feature) — a raw data tensor
    return TensorSpec(shape=None, dtype="float32", kind="data")


def _task_manifest(task: Any, mode: Mode) -> list[OutputField]:
    """The value-free serialisation-leaf metadata for a task in `mode`.

    Uses ``get_output_manifest(mode, run_name)`` (the value-free twin of
    ``get_output``; the run name is cosmetic, the sink prefixes). Raises
    `ConfigError` when the task exposes no manifest surface.
    """
    manifest = getattr(task, "get_output_manifest", None)
    if not callable(manifest):
        raise ConfigError(
            f"task {task.name!r} ({type(task).__name__}) ships no get_output_manifest — the dumb "
            "sinks need the column NAMES/DTYPES/ORDER before any batch runs (plan 34 W34.2)"
        )
    return list(manifest(mode, "salt"))
