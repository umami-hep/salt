"""`OutputSink` — the public base class every output sink subclasses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lightning import Callback, LightningModule, Trainer

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, flatten_spec

__all__ = ["OutputSink", "is_test_persistence_sink"]


def is_test_persistence_sink(callback: Any) -> bool:
    """Whether a ``writer_demand``-exposing callback is THE TEST persistence sink.

    The one selector both the runtime (`SaltModule._attached_writer`) and the
    static graph tooling (`salt.cli`) use, so a config resolves the same sink
    either way. True for a primary sink like `H5OutputSink`; False for an
    ONNX-only sink (`OnnxExportSink`, empty TEST requires) and for an
    auxiliary sink that opts out (`JSONLOutputSink`). A duck-typed callback
    without ``is_test_sink`` counts as one.

    Parameters
    ----------
    callback : Any
        A Lightning callback, typically one exposing ``writer_demand``.

    Returns
    -------
    bool
        Whether `callback` should anchor the TEST boundary demand.
    """
    is_test_sink = getattr(callback, "is_test_sink", None)
    return True if not callable(is_test_sink) else bool(is_test_sink())


class OutputSink(Callback):
    """Base class for a terminal output sink: the node IS the Lightning callback.

    A sink is the last node in the TEST graph. It consumes ``outputs.*``
    leaves that the ``outputs:`` section's writers produced and serialises
    them somewhere — `H5OutputSink` writes the eval H5, `OnnxExportSink`
    names the export tuple, and `JSONLOutputSink` (in
    ``salt/outputs/jsonl_sink.py``) is the worked example of a third format.
    Subclass this to add a format of your own; see ``docs/outputs.md``.

    The class is simultaneously a graph node and a `lightning.Callback`. The
    Lightning hooks implemented here are a thin bridge: they forward to the
    node's own named lifecycle methods, so a subclass never writes a
    Lightning hook. What a subclass provides:

    ``name``
        The graph-node instance name, unique across the graph. A ``callbacks:``
        dict key overrides it.
    ``declare_io(mode)``
        The node's requires/produces for `mode`. A sink requires the
        ``outputs.*`` leaves it serialises (plus anything else it needs, e.g.
        ``meta.rows``) and produces nothing. This single declaration drives
        both the planner (demand-gating keeps exactly the required producers
        alive) and `writer_demand`, so the two can never disagree.
    ``open_schema(trainer)``
        Called once before the first test batch — open the file, write the
        header, resolve the column schema.
    ``consume(bundle)``
        Called once per test batch with the executed `Bundle`. Read each
        required leaf with ``bundle.get(key)`` and append it.
    ``flush()``
        Called after the last batch — close the handle, report.
    ``close_if_open()``
        Called from ``teardown`` on any exit path, including an exception
        mid-test. Must be idempotent.

    Two predicates control how the graph machinery treats the sink:

    - `is_sink` (True here, rarely overridden) marks the node terminal, so
      the executor excludes it from the per-batch forward loop and the
      planner renders it as a sink card.
    - `is_test_sink` marks the node as *the* TEST persistence sink. Exactly
      one attached callback holds that role: it anchors the TEST boundary
      demand (`SaltModule._attached_writer` picks the first one). The base
      implementation answers True whenever ``declare_io(Mode.TEST).requires``
      is non-empty, which is right for a primary sink and for an ONNX-only
      sink (empty TEST requires -> False). An AUXILIARY sink that rides
      alongside the primary H5 sink must override it to return False — see
      `JSONLOutputSink`.

    Notes
    -----
    Multi-device TEST is out of scope: `setup` raises `ConfigError` when
    ``trainer.world_size != 1``, so every sink writes from a single process
    and no rank-zero guard is needed in `consume` / `flush`.

    Examples
    --------
    A minimal sink counting the rows it saw::

        from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
        from salt.outputs import OutputSink


        class RowCountSink(OutputSink):
            name = "row_count"

            def declare_io(self, mode: Mode) -> IO:
                if not (mode & Mode.TEST):
                    return IO(requires={}, produces={})
                req = {"meta.rows": TensorSpec(shape=None, dtype="int64", kind="meta")}
                return IO(requires=unflatten_spec(req), produces={})

            def is_test_sink(self) -> bool:
                return False  # auxiliary: the H5 sink stays the demand anchor

            def open_schema(self, trainer):
                self.rows = 0

            def consume(self, bundle):
                start, stop = bundle.get("meta.rows")
                self.rows += int(stop) - int(start)

            def flush(self):
                print(f"saw {self.rows} rows")
    """

    name: str
    """The graph-node instance name (overridable by the ``callbacks:`` dict key)."""

    def is_sink(self) -> bool:
        """Mark this module a terminal sink, excluded from the executor forward loop."""
        return True

    def is_test_sink(self) -> bool:
        """Whether this sink is the TEST persistence sink.

        Discriminator among ``writer_demand``-exposing callbacks: True when
        ``declare_io(Mode.TEST).requires`` is non-empty (e.g. `H5OutputSink`);
        an ONNX-only sink (`OnnxExportSink`) returns False regardless of
        ``callbacks:`` list order.
        """
        return bool(flatten_spec(self.declare_io(Mode.TEST).requires))

    def declare_io(self, mode: Mode) -> IO:  # pragma: no cover - overridden by subclasses
        """Declare the sink's requires/produces for `mode` (subclass override)."""
        del mode
        return IO(requires={}, produces={})

    def open_schema(self, trainer: Trainer) -> None:  # pragma: no cover - overridden
        """Open the sink's output schema before the first batch."""

    def consume(self, bundle: Bundle) -> None:  # pragma: no cover - overridden
        """Consume one executed bundle."""

    def flush(self) -> None:  # pragma: no cover - overridden
        """Finalise the sink."""

    def close_if_open(self) -> None:  # pragma: no cover - overridden
        """Idempotently close any open handle (failure-cleanup)."""

    def writer_demand(
        self, model_modules: Mapping[str, Any], reader: Any
    ) -> dict[str, str]:  # pragma: no cover - overridden
        """The TEST demand this sink anchors, GENERATED from `declare_io` (subclass override)."""
        del model_modules, reader
        return {}

    # -- lightning hooks: forward to the node's named methods (the bridge) -------

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Enforce single-device TEST; raises `ConfigError` otherwise (multi-device
        out of scope).
        """
        del pl_module
        if stage == "test" and trainer.world_size != 1:
            raise ConfigError(
                f"{type(self).__name__} requires a single device, got "
                f"world_size={trainer.world_size} — multi-device test writing is out of scope "
                "(design §5.3, v1 contract)"
            )

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Open the sink schema before the first batch."""
        del pl_module
        self.open_schema(trainer)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Bundle,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Consume one batch's executed bundle (Lightning threads the ``test_step`` return)."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.consume(outputs)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Finalise the sink at test end."""
        del trainer, pl_module
        self.flush()

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Idempotent cleanup: close any leaked handle on an interrupted test."""
        del trainer, pl_module, stage
        self.close_if_open()
