"""`InputCopyWriter` — the ``outputs:`` section input-copy manifest writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.outputs.run_task_output import OutputSectionWriter


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

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to copy. ``None`` (default) = every stream the sink resolves with a
        configured task. An explicit list overrides.
    variables : Mapping[str, Sequence[str]] | None, optional
        Per-stream narrowing of the copied fields; a stream not listed
        copies all source fields, by default None.
    modes : Sequence[str] | None, optional
        The modes this writer runs in (``["test", "export"]`` subset; ``None``
        = both). Input copies are eval-H5 only, so ``export`` is inert here.
    """

    name = "inputs_copy"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        variables: Mapping[str, Sequence[str]] | None = None,
        modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(modes=modes)
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
        """Requires the ``meta.rows`` row anchor in TEST (input copies are
        re-read by the sink, not the graph); produces nothing. FIT/VAL/ONNX
        prune it (input copies are eval-H5-only).
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)}
        return IO(requires=unflatten_spec(req), produces={})

    def copy_spec(self) -> dict[str, Any]:
        """The input-copy intent the dumb H5 sink consumes: ``{"streams":
        <list|None>, "variables": {stream: [vars]}}``; the sink resolves the
        file read.
        """
        return {
            "streams": list(self.streams) if self.streams is not None else None,
            "variables": {k: list(v) for k, v in self.variables.items()},
        }
