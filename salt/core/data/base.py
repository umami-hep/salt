"""Dataset-side base classes for the salt v2 pipeline (design §2.4).

Dataset modules operate on **batches of B elements** as nested dicts of
*numpy* arrays — preserving the batched contiguous H5 read design
(``samplers.py:41-55``, ``datasets.py:459-462``). Two roles exist:

- `Reader` — disk -> nested dict of numpy arrays for a contiguous batch
  slice. Source node (``requires={}``). Readers OWN the read-time mutation
  stages (selections, augmentation transforms): they run inside ``read()``
  on the structured array BEFORE any bundle key exists, so cuts flow into
  features, masks AND labels exactly as v1 (``datasets.py:464-466`` runs
  before ``:519-524`` and ``:546``) — the one sanctioned mutation point.
- `Processor` — pure batch transform on the numpy bundle, returning ONLY
  newly produced keys (write-once applies, design §2.1).

Both are `GraphModule`s: plans are compiled by the same kernel planner used
on the model side (design §3.1); only the call convention differs — the
dataset runner (`salt.core.data.dataset.GraphDataset`) passes the batch row
slice explicitly, because dataset modules are functions of *which rows* are
being read, which model modules never are.

Lifecycle (design §2.3): ``__init__`` is pure config capture (reading the
small schema YAML artifact is config I/O, sanctioned by §2.6); per-worker
file handles and reusable buffers are created in ``bind(ctx)``; main-process
file probing (VDS resolution, row counts) lives in `Reader.prepare`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, KEY_SEP, Mode
from salt.core.schema import GroupSchema, Schema

__all__ = ["DatasetModule", "Processor", "Reader", "WorkerCtx"]

RAW_NAMESPACE = "raw"
"""Bundle namespace for post-selection structured arrays (design §2.1)."""

_UNNAMED = "unnamed"  # instance names are assigned from the config dict key (design §2.2)


@dataclass(frozen=True)
class WorkerCtx:
    """Per-worker binding context handed to `DatasetModule.bind` (design §2.4).

    Built by `GraphDataset` once per (worker, plan): `read_fields` is the
    demand-narrowed per-stream read set computed **from the compiled plan**
    (design §6.1 — strictly less I/O than v1's read-everything amplification,
    ``datasets.py:395-396``), mapping ``stream -> {field: demanding module}``
    in demand order. `step` is the receiving module's own plan step, so
    wildcard producers (`Labels`) can learn their narrowed key set. `seed` is
    the per-worker seed (torch's per-worker dataloader seed when running in a
    worker) for read-time augmentations — fixing v1's unseeded transforms
    (``transforms.py:58``, design §6.1).
    """

    mode: Mode
    read_fields: Mapping[str, Mapping[str, str]]
    seed: int
    worker_id: int = 0
    num_workers: int = 0
    step: PlanStep | None = None


class DatasetModule(ABC):
    """Base class for dataset-side graph participants (design §2.4).

    Subclasses implement the `GraphModule` protocol (``name`` +
    ``declare_io``). ``declare_io`` is a function of the module's own config
    only — no data files, no tensors (design §2.2/§2.3); the schema artifact
    consumed at construction time is config I/O (design §2.6).
    """

    def __init__(self) -> None:
        """Initialise the instance name placeholder (assigned from the config key)."""
        self.name: str = _UNNAMED

    @abstractmethod
    def declare_io(self, mode: Mode) -> IO:
        """Return the declared requires/produces for the given mode."""

    def bind(self, ctx: WorkerCtx) -> None:  # noqa: B027 - optional hook, deliberately concrete
        """Per-worker lazy setup (open handles, allocate buffers); default no-op.

        Design §2.3: this is the only place dataset modules may touch data
        files. Called once per (worker process, plan) by `GraphDataset`.
        """

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Per-stream raw fields this module demands from the reader (design §6.1).

        Default: the ``fields`` metadata of the module's bound
        ``raw.<stream>`` requires. Wildcard producers whose demands are only
        known after narrowing (`Labels`) override this using `step.produces`.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {field: demanding module name}}``.
        """
        out: dict[str, dict[str, str]] = {}
        for key, spec in step.requires.items():
            parts = key.split(KEY_SEP)
            if parts[0] != RAW_NAMESPACE or len(parts) != 2 or spec.fields is None:
                continue
            stream = parts[1]
            for field in spec.fields:
                out.setdefault(stream, {}).setdefault(field, self.name)
        return out


class Reader(DatasetModule):
    """Disk -> flat dotted dict of numpy arrays for a contiguous batch slice (design §2.4).

    Source node: ``requires={}``. The arrays returned by `read` may alias the
    reader's REUSABLE per-worker buffers — that is the documented contract #9
    aliasing boundary: exactly one mandatory copy per batch (the `Features` /
    `Labels` materialisation) separates reader buffers from anything handed
    to the trainer; ``raw.*`` never crosses the torch boundary (design §2.4).

    Beyond ``read``, readers carry the framework-facing config surface the
    runtime builds on: the served `streams` (demand-driven producers adopt
    them), the optional `schema` artifact + `schema_group`/`label_universe`
    views (static validation and wildcard narrowing, design §2.6), and
    `with_source` (the datamodule's per-stage cloning, design §6.1).
    """

    schema: Schema | None = None
    """The dataset schema artifact, when configured (design §2.6)."""

    @property
    @abstractmethod
    def streams(self) -> tuple[str, ...]:
        """The stream names this reader serves, in config order."""

    def prepare(self) -> None:
        """Main-process file probing (VDS resolution, row counts); default no-op.

        Idempotent; called lazily by ``__len__`` and eagerly by the
        datamodule's rank-0 VDS pre-creation (design §6.1).
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of rows this reader serves."""

    @abstractmethod
    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous batch and return the produced keys (flat dotted dict)."""

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The schema for one served stream, when a schema artifact is configured.

        Used by `GraphDataset` to statically validate demanded raw fields
        (design §2.6). Default: None (no static validation possible).

        Returns
        -------
        GroupSchema | None
            The stream's group schema, or None.
        """
        del stream
        return None

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing.

        Design §2.2 rule (d): when not None, every narrowed ``labels.**`` key
        is validated against this set at plan-compile time. Default: None
        (no schema — narrowing is unvalidated, bind-time checks remain).

        Returns
        -------
        tuple[str, ...] | None
            The label-key universe, or None.
        """
        return None

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,
    ) -> Reader:
        """Clone this reader onto another source file (config-only, no file I/O).

        `GraphDataModule` uses this to derive the per-stage readers from the
        single configured prototype (design §6.1).

        Returns
        -------
        Reader
            A fresh, unbound reader instance.

        Raises
        ------
        NotImplementedError
            If the reader does not support re-sourcing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_source(); it cannot be used "
            "as a GraphDataModule reader prototype (design §6.1)"
        )

    def aliases(self, array: np.ndarray) -> bool:
        """Check whether `array` shares memory with a reusable reader buffer.

        Used by the ``debug`` boundary check (design §2.4: leaves surviving
        to the torch boundary must not alias a reusable buffer). Default:
        False (no registered buffers).

        Returns
        -------
        bool
            True if `array` may share memory with a reader buffer.
        """
        del array
        return False


class Processor(DatasetModule):
    """Pure batch transform on the numpy bundle (design §2.4).

    `process` reads its declared requires from the bundle and returns ONLY
    newly produced keys, as a flat dotted dict (or nested — both spellings
    are canonicalised by the runner, design §2.5/§3.2). Processors may use
    private scratch buffers, but any leaf that survives to the torch
    boundary must not alias a reusable reader buffer (design §2.4).
    """

    @abstractmethod
    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Transform one batch.

        Parameters
        ----------
        batch : Bundle
            The run bundle; read declared requires via ``batch.get(...)``.
        rows : slice
            The batch's row range in the source file.
        mode : Mode
            The plan's primary mode.

        Returns
        -------
        dict[str, np.ndarray]
            Newly produced keys only.
        """
