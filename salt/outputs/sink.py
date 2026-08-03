"""The output-sink base classes: `Node` (declare-only) and `RuntimeSink`."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, flatten_spec

__all__ = [
    "Node",
    "OutputSink",
    "RuntimeSink",
    "SinkContext",
    "is_test_persistence_sink",
]

ALL_MODES: frozenset[Mode] = frozenset({Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX})
"""The permissive `Node.allowed_modes` default; concrete sinks narrow it."""


@dataclass(frozen=True)
class SinkContext:
    """The run facts a sink needs when it opens its output.

    A sink is driven by whichever command is running — the Lightning test
    loop, or `salt inference`, which runs no Lightning at all. `SinkContext`
    is the driver-independent handle that carries the few facts a sink
    actually reads at open time, so a sink never touches a `Trainer` and
    every driver can supply the context honestly.

    Parameters
    ----------
    run_name : str
        The model's run name, used to prefix output column names.
    datamodule : Any
        The datamodule whose ``test_dset`` carries the reader — the source
        schema, sequence lengths for pad-back, and the source filename the
        output is named after.
    ckpt_path : str | None
        The checkpoint being evaluated. Output paths are templated on it, so
        a sink that names files rejects None.
    num_test_batches : Any, optional
        Lightning's per-dataloader batch counts, used to size a
        ``limit_test_batches``-capped run. None (the default, and what a
        non-Lightning driver supplies) means "the whole dataset".
    world_size : int, optional
        Devices taking part, by default 1. Multi-device test writing is out
        of scope, so anything else is refused at wiring time.
    """

    run_name: str
    datamodule: Any
    ckpt_path: str | None
    num_test_batches: Any = None
    world_size: int = 1

    @classmethod
    def from_trainer(cls, trainer: Any) -> SinkContext:
        """Build the context from a live Lightning `Trainer`.

        Duck-typed on purpose: this reads only the handful of attributes
        above, which is what lets the same sink run under a driver that has
        no trainer at all — and what keeps this module free of any Lightning
        import.
        """  # noqa: DOC201 - one-line constructor contract
        module = getattr(trainer, "lightning_module", None)
        ckpt_path = getattr(trainer, "ckpt_path", None)
        return cls(
            run_name=getattr(module, "name", None) or "salt",
            datamodule=getattr(trainer, "datamodule", None),
            ckpt_path=None if ckpt_path is None else str(ckpt_path),
            num_test_batches=getattr(trainer, "num_test_batches", None),
            world_size=int(getattr(trainer, "world_size", 1) or 1),
        )

    @property
    def reader(self) -> Any:
        """The test dataset's reader, or None when no dataset is built."""
        return getattr(getattr(self.datamodule, "test_dset", None), "reader", None)


def is_test_persistence_sink(sink: Any) -> bool:
    """Whether a ``writer_demand``-exposing sink is THE TEST persistence sink.

    The one selector both the runtime (`SaltModule._attached_writer`) and the
    static graph tooling (`salt.cli`) use, so a config resolves the same sink
    either way. True for a primary sink like `H5OutputSink`; False for an
    ONNX-only sink (`OnnxExportSink`, empty TEST requires) and for an
    auxiliary sink that opts out (`JSONLOutputSink`). A duck-typed object
    without ``is_test_sink`` counts as one.

    Parameters
    ----------
    sink : Any
        A sink node, typically one exposing ``writer_demand``.

    Returns
    -------
    bool
        Whether `sink` should anchor the TEST boundary demand.
    """
    is_test_sink = getattr(sink, "is_test_sink", None)
    return True if not callable(is_test_sink) else bool(is_test_sink())


class Node:
    """Base class for a terminal output node: DECLARE-ONLY, no lifecycle.

    A node is the last thing in a mode's graph. It declares which
    ``outputs.*`` leaves it consumes and nothing else — that single
    declaration drives both the planner (demand-gating keeps exactly the
    required producers alive) and `writer_demand`, so the two can never
    disagree.

    Subclass `Node` directly when the node has no run-time work at all —
    `OnnxExportSink` is the case in the tree: export never executes a test
    loop, so naming the leaves at compile time is its whole job. Subclass
    `RuntimeSink` when the node consumes batches as they are produced.

    A node is NOT a Lightning callback. The Lightning test loop drives a
    `RuntimeSink` through a generated adapter (`salt.callbacks.SinkAdapter`),
    and `salt inference` drives the same methods directly — so a sink author
    never writes a Lightning hook and no driver is privileged.

    What a subclass provides:

    ``name``
        The graph-node instance name, unique across the graph.
    ``allowed_modes``
        The modes this class may be configured to run in. A config `modes:`
        list must be a subset; the default is permissive and concrete
        classes narrow it.
    ``declare_io(mode)``
        The node's requires/produces for `mode`. A sink requires the
        ``outputs.*`` leaves it serialises (plus anything else it needs, e.g.
        ``meta.rows``) and produces nothing.

    Two predicates control how the graph machinery treats the node:

    - `is_sink` (True here, rarely overridden) marks the node terminal, so
      the executor excludes it from the per-batch forward loop and the
      planner renders it as a sink card.
    - `is_test_sink` marks the node as *the* TEST persistence sink. Exactly
      one registered sink holds that role: it anchors the TEST boundary
      demand (`SaltModule._attached_writer` picks the first one). The base
      implementation answers True whenever ``declare_io(Mode.TEST).requires``
      is non-empty, which is right for a primary sink and for an ONNX-only
      node (empty TEST requires -> False). An AUXILIARY sink that rides
      alongside the primary H5 sink must override it to return False — see
      `JSONLOutputSink`.

    Attributes
    ----------
    name : str
        The graph-node instance name, unique across the graph.
    allowed_modes : ClassVar[frozenset[Mode]]
        The modes this class may be configured to run in.
    """

    name: str
    """The graph-node instance name, unique across the graph."""

    allowed_modes: ClassVar[frozenset[Mode]] = ALL_MODES
    """The modes a config may select for this class (concrete classes narrow it)."""

    def is_sink(self) -> bool:
        """Mark this module a terminal sink, excluded from the executor forward loop."""  # noqa: DOC201 - one-line predicate
        return True

    def is_test_sink(self) -> bool:
        """Whether this node is the TEST persistence sink.

        Discriminator among ``writer_demand``-exposing sinks: True when
        ``declare_io(Mode.TEST).requires`` is non-empty (e.g. `H5OutputSink`);
        an ONNX-only node (`OnnxExportSink`) returns False regardless of
        declaration order.
        """  # noqa: DOC201 - predicate, no Returns block per docstring policy
        return bool(flatten_spec(self.declare_io(Mode.TEST).requires))

    def declare_io(self, mode: Mode) -> IO:  # pragma: no cover - overridden by subclasses
        """Declare the node's requires/produces for `mode` (subclass override)."""  # noqa: DOC201 - contract documented on the class
        del mode
        return IO(requires={}, produces={})

    def writer_demand(
        self, model_modules: Mapping[str, Any], reader: Any
    ) -> dict[str, str]:  # pragma: no cover - overridden
        """The TEST demand this node anchors, GENERATED from `declare_io` (subclass override)."""  # noqa: DOC201 - contract documented on the class
        del model_modules, reader
        return {}


class RuntimeSink(Node):
    """A terminal node that also consumes batches: the lifecycle lives here.

    Adds the four driver-called methods to `Node`. `H5OutputSink` writes the
    eval H5, and `JSONLOutputSink` (in ``salt/outputs/jsonl_sink.py``) is the
    worked example of a third format. Subclass this to add a format of your
    own; see ``docs/outputs.md``.

    The lifecycle is driver-independent — the Lightning test loop reaches it
    through a generated `salt.callbacks.SinkAdapter`, `salt inference` calls
    it directly:

    ``open_schema(ctx)``
        Called once before the first batch with a `SinkContext` — open the
        file, write the header, resolve the column schema. The context, not
        a `Trainer`, is what carries the run facts, so the same sink works
        under any driver.
    ``consume(bundle)``
        Called once per batch with the executed `Bundle`. Read each required
        leaf with ``bundle.get(key)`` and append it.
    ``flush()``
        Called after the last batch — close the handle, report.
    ``close_if_open()``
        Called on any exit path, including an exception mid-run. Must be
        idempotent: it is what guarantees a crashed test still closes the
        file.

    Notes
    -----
    Multi-device TEST is out of scope: the adapter raises `ConfigError` when
    ``world_size != 1``, so every sink writes from a single process and no
    rank-zero guard is needed in `consume` / `flush`.

    Examples
    --------
    A minimal sink counting the rows it saw::

        from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
        from salt.outputs import RuntimeSink


        class RowCountSink(RuntimeSink):
            name = "row_count"

            def declare_io(self, mode: Mode) -> IO:
                if not (mode & Mode.TEST):
                    return IO(requires={}, produces={})
                req = {"meta.rows": TensorSpec(shape=None, dtype="int64", kind="meta")}
                return IO(requires=unflatten_spec(req), produces={})

            def is_test_sink(self) -> bool:
                return False  # auxiliary: the H5 sink stays the demand anchor

            def open_schema(self, ctx):
                self.rows = 0

            def consume(self, bundle):
                start, stop = bundle.get("meta.rows")
                self.rows += int(stop) - int(start)

            def flush(self):
                print(f"saw {self.rows} rows")
    """

    allowed_modes: ClassVar[frozenset[Mode]] = frozenset({Mode.TEST})
    """A runtime sink runs in the TEST loop; export/fit never drive one."""

    def open_schema(self, ctx: SinkContext) -> None:  # pragma: no cover - overridden
        """Open the sink's output schema before the first batch."""

    def consume(self, bundle: Bundle) -> None:  # pragma: no cover - overridden
        """Consume one executed bundle."""

    def flush(self) -> None:  # pragma: no cover - overridden
        """Finalise the sink."""

    def close_if_open(self) -> None:  # pragma: no cover - overridden
        """Idempotently close any open handle (failure-cleanup)."""


class OutputSink(RuntimeSink):
    """Deprecated alias of `RuntimeSink`, kept for third-party subclasses.

    The sink base used to BE a `lightning.Callback`; it was split into
    `Node` (declare-only) and `RuntimeSink` (lifecycle) so a sink stops
    privileging one driver. An existing ``class MySink(OutputSink)`` keeps
    working unchanged — the lifecycle methods and their contracts are
    identical — but subclassing warns, and the name is removed after the
    deprecation window. Subclass `RuntimeSink` instead.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Warn once per subclass that `OutputSink` is deprecated."""
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} subclasses OutputSink, which is a deprecated alias of "
            "RuntimeSink — subclass salt.outputs.RuntimeSink instead (a sink is no "
            "longer a lightning Callback; the test loop drives it through a generated "
            "adapter).",
            DeprecationWarning,
            stacklevel=2,
        )
