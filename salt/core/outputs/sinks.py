"""Output sinks — terminal consumers of the ``outputs.*`` dict (design §2 layer 2).

A sink consumes the producers' ``outputs.*`` leaves and does something terminal
with them. There are two shapes here:

- `H5OutputSink` (plan 29 W1) is a terminal graph NODE (`SinkModule`): it
  ``declare_io``-requires ``outputs.*``/``meta.rows``/``masks.*`` in TEST and
  produces nothing, so the planner keeps it in the TEST plan (rendering its own
  card and anchoring demand) and prunes it from FIT/VAL/ONNX. Its Lightning
  lifecycle rides the generated `_SinkCallback` bridge (the "node IS the
  callback" adapter, design §5.2) rather than being a hand-coded callback.
- `CollectOutputs` is the original duck-typed `lightning.Callback` P0 sink: it
  anchors demand only through ``writer_demand`` (the legacy flat-``<sinks>``
  path), is NOT a graph node, and is preserved for that path.

Sinks are independent and composable: each takes, as config, the list of output
names it wants.

The load-bearing mechanism is **demand**: a sink declares the ``outputs.*``
keys it needs via `writer_demand`, the duck-typed surface `SaltModule`
already consumes for the M4.5 `WriterCallback` (saltmodule.py
``_attached_writer`` / ``_boundary_demand``; design §8). Those keys become the
TEST plan's sinks, so the planner keeps the producers (and transitively the
``preds.*`` they read) alive in TEST while FIT/VAL prune them (design §2, §4
risk 4 — the demand-gating keystone). The sink need not serialise anything to
anchor demand; this P0 sink is a no-op collector that records the keys it saw
per batch, proving the demand path without pulling in the H5 serialisation
concerns deferred to the P1 `H5OutputWriter`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag.hdf5 import H5Writer
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    KEY_SEP,
    Mode,
    TensorSpec,
    flatten_spec,
    sym_dim,
    unflatten_spec,
)
from salt.core.utils.array_utils import join_structured_arrays

__all__ = ["CollectOutputs", "H5OutputSink", "H5OutputWriter", "OutputColumn"]

_OUTPUTS_NAMESPACE = "outputs"
"""The bundle namespace this sink demands from (design §2.1)."""

DEFAULT_OUTPUT = "{ckpt_dir}/{ckpt_stem}__test_{sample}.h5"
"""The v1-compatible output template (design §5.1; mirrors the M4.5 sink)."""


class CollectOutputs(Callback):
    """Minimal TEST-only sink: anchor demand on named ``outputs.*`` leaves.

    The smallest sink that proves the producer -> ``outputs.*`` -> sink demand
    path (design §4b P0). It declares the configured ``outputs.*`` keys as its
    demand (`writer_demand`), which `SaltModule` folds into the TEST plan's
    sinks; at test time it collects the demanded leaves per batch into
    `collected` (a no-op stand-in for the P1 `H5OutputWriter`'s serialisation).

    Parameters
    ----------
    outputs : Sequence[str]
        The ``outputs.<stream>.<name>`` keys this sink consumes — each must be
        a concrete (wildcard-free) key under the ``outputs`` namespace.

    Raises
    ------
    ConfigError
        For an empty list, a wildcard key, or a key outside the ``outputs``
        namespace (the sink consumes producer leaves, not raw predictions).
    """

    def __init__(self, outputs: Sequence[str]) -> None:
        super().__init__()
        keys = list(outputs or [])
        if not keys:
            raise ConfigError(
                "CollectOutputs needs a non-empty outputs list — name the "
                "outputs.<stream>.<name> leaves to consume (design §2 layer 2)"
            )
        for key in keys:
            parts = key.split(KEY_SEP)
            if any(part in {"*", "**"} for part in parts):
                raise ConfigError(
                    f"CollectOutputs output key {key!r} contains a wildcard — sink "
                    "demand keys are concrete (design §2.2)"
                )
            if parts[0] != _OUTPUTS_NAMESPACE:
                raise ConfigError(
                    f"CollectOutputs output key {key!r} is not under the "
                    f"{_OUTPUTS_NAMESPACE!r} namespace — sinks consume producer "
                    "outputs.* leaves, not raw predictions (design §2)"
                )
        self._outputs = tuple(keys)
        #: Per-batch list of the demanded leaves seen at test time (no-op
        #: collector; replaced by real serialisation in the P1 H5OutputWriter).
        self.collected: list[dict[str, Any]] = []

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` keys (read-only view).

        Returns
        -------
        tuple[str, ...]
            The configured keys, in declaration order.
        """
        return self._outputs

    def writer_demand(self, model_modules: Any, reader: Any) -> dict[str, str]:
        """The TEST demand this sink anchors (duck-typed `SaltModule` surface, design §8).

        Returns the configured ``outputs.*`` keys mapped to a §4.1-grade
        demander description, exactly the shape `SaltModule._boundary_demand`
        consumes from the M4.5 `WriterCallback`. The arguments mirror that
        contract and are unused here (this sink demands fixed model-produced
        keys, not reader-derived ones).

        Returns
        -------
        dict[str, str]
            ``{outputs key: "sink 'CollectOutputs' demanding <key>"}`` in
            declaration order.
        """
        del model_modules, reader
        return {key: f"sink 'CollectOutputs' demanding {key}" for key in self._outputs}

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Bundle,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect the demanded ``outputs.*`` leaves for one batch (no-op sink).

        The executed bundle carries the producers' ``outputs.*`` leaves
        (the demand kept them alive); this records them for the P0 collector
        contract. Replaced by real H5 serialisation in P1.
        """
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.collected.append({key: outputs.get(key) for key in self._outputs})


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad a per-token array along axis 1 to the file sequence length.

    The v1 ``maybe_pad`` truncation re-expansion (``array_utils.py:62-90``,
    mirrored verbatim from the M4.5 sink ``writers/modules.py:48-65``):
    positions beyond the model's (possibly truncated) sequence read as zeros —
    including the ``mask`` column's documented quirk (truncated-away positions
    read ``mask=False``), preserved for byte parity. Keeping this in the SINK is
    deliberate (design §2 layer 2, §4 risk 7): per-token re-expansion to the
    FILE sequence length is a serialisation concern the producers must not know
    about.

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


@dataclass(frozen=True)
class OutputColumn:
    """One ``outputs.*`` leaf's declarative H5 column schema (design §1, §4 P4).

    The producer emits a BARE ``outputs.<stream>.<name>`` tensor; the sink owns
    the serialisation schema the trace cannot recover — the per-column SUFFIXES,
    the H5 dtype, and the run-name prefix (design §1: dtype + split-count are the
    facts re-homed into a thin declarative table, not auto-derived). This is the
    P1 spelling of that table: each consumed output leaf carries the column
    name(s) it expands into, exactly reproducing the M4.5 ``task.output_names``
    schema (``tasks.py:1247-1255`` for classification, ``:1638-1650`` for the
    bare-``VertexIndex`` family) without the sink reaching into the task.

    Parameters
    ----------
    key : str
        The ``outputs.<stream>.<name>`` leaf this column declares. Concrete
        (wildcard-free) and under the ``outputs`` namespace.
    suffixes : Sequence[str]
        The per-channel logical suffixes the leaf's last dim expands into, in
        last-dim order (e.g. ``["pb", "pc", "pu"]`` for a 3-class head, or
        ``["VertexIndex"]`` for a single-column index). The H5 column is
        ``{run_name}_{suffix}`` when `prefix` (default), else the bare
        `suffix` — the M4.5 column-naming contract (``task.py:140-151``).
    dtype : str, optional
        The H5 column dtype (numpy descriptor), by default ``"f4"`` (the v1
        classification/regression float policy). ``"i8"`` for integer index
        columns (the bare-``VertexIndex`` family, deferred to P2).
    prefix : bool, optional
        Whether to prefix each suffix with ``{run_name}_`` (the M4.5 default,
        ``True``); ``False`` reproduces the bare-column families (the v1
        ``VertexIndex`` byte-parity column), by default True.

    Raises
    ------
    ConfigError
        For a non-``outputs`` key, a wildcard key, or an empty suffix list.
    """

    key: str
    suffixes: Sequence[str]
    dtype: str = "f4"
    prefix: bool = True

    def __post_init__(self) -> None:
        parts = self.key.split(KEY_SEP)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"OutputColumn key {self.key!r} contains a wildcard — sink demand keys "
                "are concrete (design §2.2)"
            )
        if parts[0] != _OUTPUTS_NAMESPACE:
            raise ConfigError(
                f"OutputColumn key {self.key!r} is not under the {_OUTPUTS_NAMESPACE!r} "
                "namespace — sinks consume producer outputs.* leaves, not raw predictions "
                "(design §2)"
            )
        if not list(self.suffixes):
            raise ConfigError(
                f"OutputColumn {self.key!r} needs a non-empty suffix list — name the "
                "per-channel columns the leaf expands into (design §1 declarative table)"
            )

    @property
    def stream(self) -> str:
        """The leaf's stream (``outputs.<stream>.<name>`` middle segment).

        Returns
        -------
        str
            The stream name.
        """
        return self.key.split(KEY_SEP)[1]

    def column_names(self, run_name: str) -> list[str]:
        """The H5 column names for this leaf (run-name prefixed unless bare).

        Returns
        -------
        list[str]
            ``[{run_name}_{suffix}]`` per suffix (or bare suffixes when
            ``prefix=False``), in last-dim order.
        """
        return [f"{run_name}_{s}" if self.prefix else str(s) for s in self.suffixes]

    def np_dtype(self, run_name: str) -> np.dtype:
        """The structured numpy dtype this leaf contributes to its group.

        Returns
        -------
        np.dtype
            One ``(column, dtype)`` field per suffix, in last-dim order.
        """
        return np.dtype([(col, self.dtype) for col in self.column_names(run_name)])


class _SinkCallback(Callback):
    """The generated thin Lightning bridge for a terminal sink NODE (design §5.2, D3).

    "The node IS the callback": a sink node owns its Lightning hooks through this
    shared base, eliminating the parallel duck-typed callback. The single node
    declaration (`declare_io`) drives BOTH the planner demand (the H5 sink's
    ``writer_demand`` is GENERATED from ``declare_io(Mode.TEST).requires``) and
    the lifecycle (the hooks below forward to the node's named methods).

    The discovery seam is unchanged: `SaltModule._attached_writer`
    (saltmodule.py) still scans for ``callable(getattr(cb, "writer_demand",
    None))`` and finds this adapter; the node also registers in the TEST plan as
    a real `PlanStep` (so it renders its own card), and the executor partitions
    it OUT of the per-batch forward loop via `is_sink()` (Q5).

    A subclass provides the node surface: ``name``, ``declare_io(mode)`` (TEST
    requires, empty produces), ``open_schema(trainer)`` / ``consume(bundle)`` /
    ``flush()`` / ``close_if_open()``, and ``_pad_mask_streams()`` for the
    GENERATED ``writer_demand``.
    """

    name: str

    def is_sink(self) -> bool:
        """Mark this module a terminal sink (excluded from the executor forward loop, Q5).

        Returns
        -------
        bool
            Always True — a sink produces no tensor and is never invoked as
            ``module(bundle, mode)`` by the executor.
        """
        return True

    def declare_io(self, mode: Mode) -> IO:  # pragma: no cover - overridden by subclasses
        """Declare the sink's requires/produces for `mode` (subclass override).

        Returns
        -------
        IO
            The terminal node's IO: mode-gated requires, empty produces.
        """
        del mode
        return IO(requires={}, produces={})

    def open_schema(self, trainer: Trainer) -> None:  # pragma: no cover - overridden
        """Open the sink's output schema before the first batch (was ``on_test_start``)."""

    def consume(self, bundle: Bundle) -> None:  # pragma: no cover - overridden
        """Consume one executed bundle (was ``on_test_batch_end``)."""

    def flush(self) -> None:  # pragma: no cover - overridden
        """Finalise the sink (was ``on_test_end``)."""

    def close_if_open(self) -> None:  # pragma: no cover - overridden
        """Idempotently close any open handle (failure-cleanup, design §5.3)."""

    def writer_demand(
        self, model_modules: Mapping[str, Any], reader: Any
    ) -> dict[str, str]:  # pragma: no cover - overridden
        """The TEST demand this sink anchors, GENERATED from `declare_io` (subclass override).

        Returns
        -------
        dict[str, str]
            ``{demanded key: demander description}``.
        """
        del model_modules, reader
        return {}

    # -- lightning hooks: forward to the node's named methods (the bridge) -------

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Single-device assertion at test setup (multi-device test writing out of scope).

        Raises
        ------
        ConfigError
            When testing on more than one device.
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
        """Idempotent cleanup: close any leaked handle on an interrupted test (design §5.3)."""
        del trainer, pl_module, stage
        self.close_if_open()


class H5OutputSink(_SinkCallback):
    """The P1 H5 sink: a terminal graph NODE serialising ``outputs.*`` to the eval H5 (design §4.1).

    Promoted from ``H5OutputWriter`` (Q4): a `GraphModule` terminal node whose
    ``declare_io`` requires the demanded ``outputs.*`` leaves (+ ``meta.rows`` +
    each pad-mask stream) in TEST and produces nothing — so the planner keeps it
    in TEST (rendering its own card) while FIT/VAL/ONNX prune it. Its Lightning
    lifecycle (the relocated serialisation) rides the `_SinkCallback` bridge.

    It accumulates the named ``outputs.*`` producer leaves per batch and writes
    them through ONE ftag `H5Writer`
    (FIXED mode, ``num_jets`` known up front — a valid empty file for an empty
    test set), matching the M4.5 `WriterCallback` H5 contract (recon PR1.1).
    Replaces `TaskWriter` + `InputCopyWriter` + `PadMaskWriter` +
    `WriterCallback`'s H5 sink for the classification + regression families.

    Responsibilities the SINK owns (design §2 layer 2, §4 risk 7) — the producers
    must NOT know about these:

    - **structured-array packing**: each producer leaf is a plain ``[B, C]`` /
      ``[B, L, C]`` tensor; the sink ``u2s``-packs it into the declared
      ``{run_name}_{suffix}`` columns (`OutputColumn`).
    - **per-token pad re-expansion**: per-token leaves and the pad-mask column
      are zero-padded to the FILE sequence length (the v1 ``maybe_pad``,
      ``_pad_to``), the truncation re-expansion the model never sees.
    - **input-variable copies**: optional source-file columns re-read by
      absolute rows (``meta.rows``) through one cached handle, with SOURCE
      dtypes/order (the v1 input-copy contract, ``predictionwriter.py:186-196``).
    - **pad-mask columns**: an optional boolean ``mask`` column per sequence
      stream (True = padded), the v1 ``add_mask`` quirk (truncated-away
      positions read ``mask=False``).
    - **output-group naming**: reader streams map to their FILE dataset name
      (the v1 group naming), so the eval H5 groups match v1.

    Demand (design §2 §8): `writer_demand` is the duck-typed surface
    `SaltModule` folds into the TEST plan sinks — exactly the shape the M4.5
    `WriterCallback` exposes. It returns the consumed ``outputs.*`` leaves PLUS
    the source ``preds.*`` leaf of each producer feeding them (so the TEST
    dead-preds gate — ``saltmodule.py:613-627`` — sees those predictions as
    consumed, since under this design preds.* are consumed by producers, not by
    the sink), the ``meta.rows`` row anchor, and each pad-mask stream's mask.

    Honest coupling note (design §4 risk 7): keeping input copies / masks / pad
    re-expansion here means `H5OutputWriter` INHERITS the full
    DataModule/reader/source-file coupling of the M4.5 `WriterCallback` (it
    reads the file's sequence length from ``reader.source_path``). The win is
    COMPOSABLE additional sinks (a second `PlotOutputWriter` on the same
    ``outputs.*``), not decoupling of the H5 sink itself.

    Parameters
    ----------
    outputs : Sequence[OutputColumn | Mapping[str, Any]]
        The output columns to serialise, in H5 column order within each group.
        Each entry is an `OutputColumn` (or a mapping jsonargparse builds into
        one) naming the ``outputs.*`` leaf and its per-channel suffixes/dtype.
    copy_inputs : Mapping[str, Sequence[str]] | None, optional
        Per-stream source-file variables to copy into the eval H5 (the v1 input
        copies), in column order, by default None (no copies). A stream maps to
        the list of source fields to copy.
    write_pad_mask : bool | Sequence[str], optional
        Boolean ``mask`` column per sequence stream: ``True`` writes one for
        every demanded output's sequence stream; a list names the streams; by
        default False. Mirrors the M4.5 `PadMaskWriter`.
    output : str, optional
        Output path template (``{ckpt_dir}``/``{ckpt_stem}``/``{sample}``), by
        default `DEFAULT_OUTPUT` (the v1 path contract).
    half_precision : bool, optional
        Write float columns at f2 instead of f4 (the v1 flag), by default False.

    Raises
    ------
    ConfigError
        For an empty outputs list, a duplicate output key, or (at run setup) a
        column-name collision / unknown stream / missing source variable.
    """

    name = "h5_output"
    """The graph-node instance name (overridable by the config dict key)."""

    def __init__(
        self,
        outputs: Sequence[OutputColumn | Mapping[str, Any]],
        copy_inputs: Mapping[str, Sequence[str]] | None = None,
        write_pad_mask: bool | Sequence[str] = False,
        output: str = DEFAULT_OUTPUT,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        cols = [
            c if isinstance(c, OutputColumn) else OutputColumn(**dict(c)) for c in outputs or []
        ]
        if not cols:
            raise ConfigError(
                "H5OutputSink needs a non-empty outputs list — name the outputs.* leaves "
                "(OutputColumn) to serialise (design §4.1)"
            )
        seen: set[str] = set()
        for col in cols:
            if col.key in seen:
                raise ConfigError(
                    f"H5OutputSink: duplicate output key {col.key!r} — one OutputColumn per "
                    "outputs.* leaf (design §2.2)"
                )
            seen.add(col.key)
        self._columns: tuple[OutputColumn, ...] = tuple(cols)
        self.copy_inputs = {s: list(v) for s, v in (copy_inputs or {}).items()}
        self.write_pad_mask = write_pad_mask
        self.output = output
        self.half_precision = half_precision
        # per-test-run state (reset at open_schema)
        self._h5: H5Writer | None = None
        self._rows_written = 0
        self._expected = 0
        self._run_name = "salt"
        self._group_of: dict[str, str] = {}
        self._seq_lengths: dict[str, int] = {}
        self._mask_streams: tuple[str, ...] = ()
        self._copy_handle: h5py.File | None = None
        self._copy_reads: dict[str, tuple[h5py.Dataset, list[str]]] = {}
        self.output_path: Path | None = None

    @property
    def columns(self) -> tuple[OutputColumn, ...]:
        """The configured output columns (read-only view).

        Returns
        -------
        tuple[OutputColumn, ...]
            The columns in declaration (H5 column) order.
        """
        return self._columns

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` leaf keys, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The configured output keys.
        """
        return tuple(col.key for col in self._columns)

    # -- graph node surface (design §4.1) ---------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """The terminal node's IO: TEST requires ``outputs.*``/``meta.rows``/``masks.*``; empty out.

        In TEST the sink requires the demanded ``outputs.*`` leaves (``kind=data``),
        the ``meta.rows`` row anchor (``kind=meta``), and each pad-mask stream's
        ``masks.<stream>`` (``kind=pad_mask``); it produces nothing — a terminal
        node the planner keeps via `_is_terminal_consumer`/`_demand_closure`
        (rendering its own card). In FIT/VAL/ONNX it declares empty requires AND
        empty produces, so the demand-closure prunes it — and the FIT/VAL
        ``plan_hash`` is byte-unchanged (the trained checkpoint loads unperturbed,
        design §8 back-compat proof). The ``preds.*`` back-discovery the old
        ``writer_demand`` did DISAPPEARS: the conversion PRODUCERS declare their
        own ``preds.*`` requires, so the dead-preds gate is satisfied
        transitively via the producer node (design §4.1).

        Returns
        -------
        IO
            The declared interface for `mode`.
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        # dtype is None on the require: ``OutputColumn.dtype`` is the H5 NUMPY
        # descriptor (``"f4"``/``"i8"``) — a serialisation concern — not the
        # producer's torch dtype (``"float32"``). Constraining it would conflict
        # with the producer's declared dtype; the sink consumes whatever leaf the
        # producer emits and casts at write time (design §4.1, kind unifies).
        req: dict[str, TensorSpec] = {
            col.key: TensorSpec(shape=None, dtype=None, kind="data") for col in self._columns
        }
        req["meta.rows"] = TensorSpec(shape=None, dtype="int64", kind="meta")
        for stream in self._pad_mask_streams():
            req[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        return IO(requires=unflatten_spec(req), produces={})

    # -- static demand (consumed by SaltModule, design §8) ----------------------

    def writer_demand(self, model_modules: Mapping[str, Any], reader: Any) -> dict[str, str]:
        """The TEST demand this sink anchors — GENERATED from `declare_io` (design §4.1, §5.2).

        Returns the sink's TEST-mode ``declare_io`` requires (the consumed
        ``outputs.*`` leaves, the ``meta.rows`` row anchor, and each pad-mask
        stream's ``masks.<stream>``), each mapped to a §4.1-grade demander
        description. `SaltModule._attached_writer` (saltmodule.py) discovers this
        via ``callable(getattr(cb, "writer_demand", None))`` and folds the keys
        into the TEST plan sinks — keeping that seam intact while the keys are now
        DERIVED from the single node declaration rather than hand-coded.

        The old ``preds.*`` back-discovery is GONE (design §4.1): the conversion
        producers declare their own ``preds.*`` requires, so the dead-preds gate
        is satisfied transitively. This sink demands ONLY
        ``outputs.*``/``meta.rows``/``masks.*``.

        Parameters
        ----------
        model_modules : Mapping[str, Any]
            The model-side module dict (unused — demand is generated from the
            sink's own declaration).
        reader : Any
            The configured reader prototype (unused — this sink demands fixed
            model-produced keys).

        Returns
        -------
        dict[str, str]
            ``{dotted key: "sink 'H5OutputSink' demanding <key>"}`` in
            declaration order (outputs, then meta.rows, then masks).
        """
        del model_modules, reader
        who = "sink 'H5OutputSink' demanding"
        return {key: f"{who} {key}" for key in flatten_spec(self.declare_io(Mode.TEST).requires)}

    def _pad_mask_streams(self) -> tuple[str, ...]:
        """The sequence streams a pad-mask column is requested for.

        Returns
        -------
        tuple[str, ...]
            The configured streams (``True`` → every demanded output's stream),
            de-duplicated in first-seen order.
        """
        if self.write_pad_mask is False:
            return ()
        if self.write_pad_mask is True:
            seen: dict[str, None] = {}
            for col in self._columns:
                seen.setdefault(col.stream, None)
            return tuple(seen)
        return tuple(dict.fromkeys(self.write_pad_mask))

    # -- node lifecycle (relocated VERBATIM from on_test_* — design §5.1) --------

    def open_schema(self, trainer: Trainer) -> None:
        """Create the eval H5 with the full schema BEFORE the first batch (was ``on_test_start``).

        Resolves the output path + total rows from the trainer/datamodule, opens
        the source handle for input copies, merges the output / input-copy /
        pad-mask columns into per-group dtypes/shapes, and creates the FIXED-mode
        `H5Writer` (recon PR1.1). Driven by the `_SinkCallback` bridge's
        ``on_test_start`` hook.

        Raises
        ------
        ConfigError
            For a missing ``ckpt_path``, a foreign datamodule, an unknown output
            template key, a non-sequence pad-mask stream, a column collision, or
            a missing input-copy source variable.
        """
        pl_module = trainer.lightning_module
        dm = getattr(trainer, "datamodule", None)
        dset = getattr(dm, "test_dset", None)
        if dset is None:
            raise ConfigError(
                "H5OutputSink needs a GraphDataModule with a built test dataset — "
                f"got {type(dm).__name__} (design §5.1)"
            )
        reader = dset.reader
        self._run_name = getattr(pl_module, "name", "salt")
        streams = tuple(getattr(reader, "streams", ()) or ())
        groups = getattr(reader, "groups", None)
        if not streams or groups is None:
            raise ConfigError(
                "H5OutputSink needs an H5StructuredReader-style reader exposing "
                f"streams/groups — got {type(reader).__name__} (design §5.1)"
            )
        sequence_streams = tuple(
            s for s in streams if not getattr(groups[s], "global_object", False)
        )
        group_datasets = {stream: groups[stream].dataset for stream in streams}
        source_path = Path(reader.source_path)
        # reader-matching open flags (HDF5 rejects mixed SWMR flags on one file
        # within a process — the M4.5 InputCopyWriter.setup note)
        with h5py.File(source_path, "r", swmr=True, libver="latest") as f:
            self._seq_lengths = {
                stream: int(f[group_datasets[stream]].shape[1]) for stream in sequence_streams
            }
        self._mask_streams = self._pad_mask_streams()
        for stream in self._mask_streams:
            if stream not in sequence_streams:
                raise ConfigError(
                    f"H5OutputSink: pad-mask stream {stream!r} is not a sequence stream — "
                    f"pad masks exist for {list(sequence_streams)} only (design §6.1)"
                )
        total = self._expected_rows(trainer, len(dset), dm.batch_size)
        self._open_copies(source_path, group_datasets, streams)
        dtypes, shapes = self._merge_columns(streams, sequence_streams, group_datasets, total)
        self.output_path = self._output_path(trainer, dm, reader)
        self._h5 = H5Writer(
            dst=self.output_path,
            dtypes=dtypes,
            shapes=shapes,
            shuffle=False,
            jets_name=next(iter(dtypes)),  # batch-sizing group (v1 jets_name semantics)
            precision="half" if self.half_precision else "full",
            num_jets=total,  # FIXED mode: valid empty file for an empty test set
        )
        self._rows_written = 0
        self._expected = total

    def consume(self, bundle: Bundle) -> None:
        """Serialise one batch of ``outputs.*`` (+ input copies + masks); was ``on_test_batch_end``.

        Reads ``meta.rows`` to validate row alignment against the running
        counter (sharded/uneven-batch loaders fail loudly, the M4.5 contract),
        packs each demanded leaf into its declared columns, re-reads the
        input-copy columns by absolute rows, builds the pad-mask columns, merges
        same-group fragments, and streams one `H5Writer.write`. Driven by the
        `_SinkCallback` bridge's ``on_test_batch_end`` hook (Lightning threads
        the ``test_step`` return — the executed `Bundle` — as the ``outputs`` arg).

        Raises
        ------
        ConfigError
            On a row-alignment break or a fragment with the wrong row count.
        """
        assert self._h5 is not None, "consume before open_schema"
        rows_t = bundle.get("meta.rows")
        start, stop = int(rows_t[0]), int(rows_t[1])
        if start != self._rows_written:
            raise ConfigError(
                f"H5OutputSink row alignment broke: batch rows [{start}, {stop}) but "
                f"{self._rows_written} rows written so far — sharded or uneven-batch test "
                "loaders are not supported (design §5 row-alignment contract)"
            )
        rows = slice(start, stop)
        n = stop - start
        fragments: dict[str, list[np.ndarray]] = {}
        # input copies first (the v1 inputs_copy -> tasks -> pad_mask column order)
        for stream, arr in self._copy_fragments(rows).items():
            fragments.setdefault(stream, []).append(arr)
        # output columns next
        for arr_stream, arr in self._output_fragments(bundle).items():
            fragments.setdefault(arr_stream, []).append(arr)
        # pad masks last
        for stream, arr in self._mask_fragments(bundle).items():
            fragments.setdefault(stream, []).append(arr)
        for stream, arrs in fragments.items():
            for arr in arrs:
                if len(arr) != n:
                    raise ConfigError(
                        f"H5OutputSink: fragment for group {stream!r} has {len(arr)} rows, "
                        f"expected {n} (rows [{start}, {stop}))"
                    )
        data = {
            self._group_of[stream]: (arrs[0] if len(arrs) == 1 else join_structured_arrays(arrs))
            for stream, arrs in fragments.items()
        }
        self._h5.write(data)
        self._rows_written = stop

    def flush(self) -> None:
        """Close the source handle and the sink, warning on a truncated loop (was ``on_test_end``).

        Driven by the `_SinkCallback` bridge's ``on_test_end`` hook.
        """
        self._close_copies()
        if self._h5 is None:
            return
        if self._rows_written == self._expected:
            self._h5.close()
        else:
            print(
                f"WARNING: wrote {self._rows_written:,} of {self._expected:,} expected rows "
                f"to {self.output_path} — trailing rows are zero-filled"
            )
            self._h5.file.close()
        self._h5 = None
        print("-" * 100)
        print(f"Wrote eval file {self.output_path}")
        print("-" * 100)

    def close_if_open(self) -> None:
        """Idempotently close any open handle on an interrupted test (design §5.3).

        Wired to the `_SinkCallback` bridge's ``teardown`` hook: if ``consume``
        raised mid-test, Lightning's ``on_test_end`` may not run, leaking the
        FIXED-mode handle and leaving a half-written file open. This closes the
        raw `H5Writer.file` handle (and the cached source handle) WITHOUT the
        full-count assertion — matching `flush`'s truncated branch — and is a
        no-op once `flush`/`close_if_open` already ran (``self._h5 is None``).
        """
        self._close_copies()
        if self._h5 is None:
            return
        self._h5.file.close()
        self._h5 = None

    # -- per-batch fragment builders --------------------------------------------

    def _output_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Pack each demanded ``outputs.*`` leaf into its declared columns.

        Per-token (sequence-stream) leaves are re-expanded to the file sequence
        length (the v1 ``maybe_pad``); a 2-D ``[B, C]`` global leaf is packed
        directly. The ``u2s`` cast applies the column dtype (the ftag H5Writer's
        ``full``/``half`` precision then handles f4/f2 downcast for floats).

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` per demanded output's stream.
        """
        out: dict[str, np.ndarray] = {}
        frags: dict[str, list[np.ndarray]] = {}
        for col in self._columns:
            tensor = bundle.get(col.key)
            values = tensor.detach().cpu().numpy()
            # the producer leaf may be [B] / [B, L] (collapsed index columns) or
            # [B, C] / [B, L, C] — give u2s an explicit trailing channel axis so a
            # single-suffix column packs the same as the M4.5 task.get_h5 path
            if values.ndim == 1 or (values.ndim == 2 and col.stream in self._seq_lengths):
                values = values[..., np.newaxis]
            arr = u2s(np.ascontiguousarray(values), col.np_dtype(self._run_name))
            if arr.ndim == 2:
                arr = _pad_to(arr, self._seq_lengths[col.stream])
            frags.setdefault(col.stream, []).append(arr)
        for stream, arrs in frags.items():
            out[stream] = arrs[0] if len(arrs) == 1 else join_structured_arrays(arrs)
        return out

    def _mask_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Build the boolean ``mask`` column per requested sequence stream.

        Mirrors the M4.5 `PadMaskWriter`: ``True`` = padded, padded to the file
        sequence length with ``mask=False`` (the preserved v1 quirk).

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` with one ``('mask', '?')`` field.
        """
        out: dict[str, np.ndarray] = {}
        for stream in self._mask_streams:
            mask = bundle.get(f"masks.{stream}").detach().cpu().numpy()
            arr = u2s(np.expand_dims(mask, -1), dtype=np.dtype([("mask", "?")]))
            out[stream] = _pad_to(arr, self._seq_lengths[stream])
        return out

    def _copy_fragments(self, rows: slice) -> dict[str, np.ndarray]:
        """Re-read this batch's input-copy columns by absolute rows.

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` at full file sequence length, SOURCE
            dtypes/order (the v1 input-copy contract).
        """
        return {
            stream: ds.fields(fields)[rows.start : rows.stop]
            for stream, (ds, fields) in self._copy_reads.items()
        }

    # -- setup plumbing ----------------------------------------------------------

    def _open_copies(
        self, source_path: Path, group_datasets: Mapping[str, str], streams: tuple[str, ...]
    ) -> None:
        """Open the cached source handle and resolve per-stream copy field lists.

        Raises
        ------
        ConfigError
            For an unknown copy stream or a copy variable missing from the file
            (the v1 ``extra_vars`` validation, ``predictionwriter.py:88-100``).
        """
        self._close_copies()
        if not self.copy_inputs:
            return
        if unknown := sorted(set(self.copy_inputs) - set(streams)):
            raise ConfigError(
                f"H5OutputSink: copy_inputs streams {unknown} are not reader streams — "
                f"streams are {list(streams)}"
            )
        self._copy_handle = h5py.File(source_path, "r", swmr=True, libver="latest")
        for stream in streams:  # reader config order
            if stream not in self.copy_inputs:
                continue
            ds = self._copy_handle[group_datasets[stream]]
            file_fields = list(ds.dtype.names or ())
            fields = self.copy_inputs[stream] or file_fields
            if missing := sorted(set(fields) - set(file_fields)):
                raise ConfigError(
                    f"H5OutputSink: copy_inputs variables {missing} missing for stream "
                    f"{stream!r} in {source_path.name!r} (v1 extra_vars contract)"
                )
            self._copy_reads[stream] = (ds, list(fields))

    def _close_copies(self) -> None:
        """Close the cached source handle (re-entrant across test runs)."""
        if self._copy_handle is not None:
            self._copy_handle.close()
            self._copy_handle = None
        self._copy_reads = {}

    def _merge_columns(
        self,
        streams: tuple[str, ...],
        sequence_streams: tuple[str, ...],
        group_datasets: Mapping[str, str],
        total: int,
    ) -> tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]:
        """Merge copy / output / pad-mask columns into per-group dtypes/shapes.

        Column order = input copies, then output columns (declaration order),
        then the pad mask — the M4.5 ``inputs_copy -> tasks -> pad_mask`` group
        layout (recon PR1.1). Output groups are named after the FILE dataset (v1
        group naming). A column-name collision is a `ConfigError` naming both
        contributors.

        Returns
        -------
        tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]
            ``(dtypes, shapes)`` keyed by the H5 dataset name.

        Raises
        ------
        ConfigError
            On a column collision or an output leaf for an unknown stream.
        """
        descrs: dict[str, list] = {}
        owners: dict[tuple[str, str], str] = {}

        def _add(stream: str, dtype: np.dtype, who: str) -> None:
            if stream not in streams:
                raise ConfigError(
                    f"H5OutputSink: {who} targets unknown group {stream!r} — reader streams "
                    f"are {list(streams)}"
                )
            for descr in dtype.descr:
                field_name = descr[0]
                if (other := owners.get((stream, field_name))) is not None:
                    raise ConfigError(
                        f"column {field_name!r} in group {stream!r} is declared by {other} AND "
                        f"{who} — rename one output (collision check, design §8)"
                    )
                owners[stream, field_name] = who
                descrs.setdefault(stream, []).append(descr)

        for stream, (ds, fields) in self._copy_reads.items():
            _add(
                stream,
                np.dtype([(fname, ds.dtype[fname]) for fname in fields]),
                f"copy_inputs[{stream!r}]",
            )
        for col in self._columns:
            _add(col.stream, col.np_dtype(self._run_name), f"output {col.key!r}")
        for stream in self._mask_streams:
            _add(stream, np.dtype([("mask", "?")]), f"pad mask[{stream!r}]")
        if not descrs:
            raise ConfigError("H5OutputSink declares no output columns at all (design §8)")
        self._group_of = {stream: group_datasets.get(stream, stream) for stream in descrs}
        dtypes = {self._group_of[stream]: np.dtype(descr) for stream, descr in descrs.items()}
        shapes = {
            self._group_of[stream]: (
                (total, self._seq_lengths[stream]) if stream in sequence_streams else (total,)
            )
            for stream in descrs
        }
        return dtypes, shapes

    @staticmethod
    def _expected_rows(trainer: Trainer, total: int, batch_size: int) -> int:
        """Rows the (possibly ``limit_test_batches``-capped) loop will write.

        Returns
        -------
        int
            ``min(total, num_batches * batch_size)`` (the M4.5 sink contract:
            with the sequential no-drop sampler only the last batch is partial).
        """
        num_batches = getattr(trainer, "num_test_batches", None)
        if not num_batches:
            return total
        limit = num_batches[0]
        if isinstance(limit, (int, float)) and math.isfinite(limit):
            return min(total, int(limit) * batch_size)
        return total

    def _output_path(self, trainer: Trainer, dm: Any, reader: Any) -> Path:
        """Render the output template (the v1 ``predictionwriter.py:164-171`` contract).

        Returns
        -------
        Path
            The resolved output path.

        Raises
        ------
        ConfigError
            When ``ckpt_path`` is unset or the template names an unknown key.
        """
        ckpt_path = trainer.ckpt_path
        if ckpt_path is None:
            raise ConfigError(
                "H5OutputSink needs trainer.ckpt_path — run salt2 test with --ckpt_path "
                "<ckpt> (the output file is named after the checkpoint, v1 contract)"
            )
        stem = Path(getattr(reader, "filename", None) or reader.source_path).stem
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        test_suff = getattr(dm, "test_suff", None)
        if test_suff:
            sample = f"{sample}_{test_suff}"
        keys = {
            "ckpt_dir": str(Path(ckpt_path).parent),
            "ckpt_stem": Path(ckpt_path).stem,
            "sample": sample,
        }
        try:
            return Path(self.output.format(**keys))
        except KeyError as err:
            raise ConfigError(
                f"unknown H5OutputSink output template key {err} — available: {sorted(keys)}"
            ) from None


# DEPRECATED one-window alias (design Q4): the node-shaped sink was renamed
# H5OutputWriter -> H5OutputSink (plan 29 W1). Downstream configs that wire
# `salt.core.outputs.H5OutputWriter` (incl. gn2v2-dummy-cutover.yaml) keep
# working — the alias resolves to the promoted node. Remove after the migration
# window (mirrors plan-25's one-window alias policy).
H5OutputWriter = H5OutputSink
"""Deprecated alias for `H5OutputSink` (Q4 one-window migration; plan 29 W1)."""
