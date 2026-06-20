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

from salt.core.data.stream import OffsetIndex, StreamConfig, _cut_sort_truncate_pad
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, KEY_SEP, Mode
from salt.core.schema import GroupSchema, Schema

__all__ = ["DatasetModule", "OffsetIndex", "Processor", "Reader", "StreamConfig", "WorkerCtx"]

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
        stage: str | None = None,
    ) -> Reader:
        """Clone this reader onto another source file (config-only, no file I/O).

        `GraphDataModule` uses this to derive the per-stage readers from the
        single configured prototype (design §6.1).

        `stage` (``"train"``/``"val"``/``"test"``) is the OPTIONAL per-reader
        stage-sourcing hook (plan 02): single-source readers (`H5StructuredReader`,
        `EasyjetReader`) IGNORE it — their one ``filename`` per stage IS the data,
        so adding the kwarg is byte-for-byte backward compatible. Multi-source
        readers (`MultiSampleReader`, a future cut-based reader) use it to select
        each sub-source's per-stage data. The datamodule passes the stage it
        already knows; readers that don't need it never look at it.

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

    def sources(self) -> list[Path]:
        """The concrete on-disk file(s) this reader will read (the staging surface).

        Each reader is the file-authority: it declares which files it reads so
        the framework can relocate them (M8 reader-owned staging — the datamodule
        no longer special-cases ``train_file``/``val_file``, which silently missed
        multi-file / multi-sample readers). The base default introspects a
        ``filename`` or ``files`` attribute, returning its `Path`(s); readers whose
        sources are not a single such attribute (`MultiSampleReader`) override.

        Wildcard / glob filenames are returned VERBATIM (a literal pattern, not its
        expansion) — staging a wildcard reader is the caller's responsibility (the
        shipped staging path stages already-resolved single files). A reader with no
        bound source returns an empty list.

        Returns
        -------
        list[Path]
            The file(s) backing this reader, in read order.
        """
        files = getattr(self, "files", None)
        if files is not None:
            return [Path(f) for f in files]
        filename = getattr(self, "filename", None)
        return [Path(filename)] if filename is not None else []

    def restage(self, root: str | Path) -> Reader:
        """Return a CLONE of this reader whose `sources` point at copies under `root`.

        The reader-owned twin of `with_source` (the per-stage cloning hook above):
        rather than re-source onto a DIFFERENT file, `restage` copies THIS reader's
        own source file(s) into ``root`` (typically a RAM disk like ``/dev/shm``) and
        returns a clone that reads the copies. This is the M8 replacement for the
        datamodule's fat ``move_files_temp`` ``prepare_data``/``setup`` block: the
        datamodule's only job becomes ``reader = reader.restage(root)`` before VDS
        precreation, so VDS + datasets build against the staged copies, and multi-file
        / multi-sample readers stage ALL their files (the old code staged only
        ``train_file``/``val_file``).

        Base default: copy each `sources` file to ``root`` via the FileLock-coordinated
        `salt.core.data.vds.stage_file` (reuses the existing copy + ``.done`` marker
        machinery so a DDP / worker stampede copies each file exactly once, no trainer
        handle needed), then clone with the new path via `with_source`. A reader with a
        single source uses this directly; multi-source readers (`MultiSampleReader`)
        override to restage each sub-reader recursively. A reader with NO source clones
        unchanged (nothing to stage).

        Parameters
        ----------
        root : str | Path
            The staging root directory (created if missing). Each source file is
            copied to ``root / <source name>``.

        Returns
        -------
        Reader
            A fresh, unbound reader reading the staged copies (same instance
            ``name``). Identity-stable when there is nothing to stage.

        Raises
        ------
        NotImplementedError
            If the reader has >1 source but does not override `restage` (the base
            default only knows how to re-point a single-source reader via
            `with_source`).
        """
        from salt.core.data.vds import stage_file  # noqa: PLC0415 - opt-in staging path only

        root = Path(root)
        srcs = self.sources()
        if not srcs:
            return self
        if len(srcs) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} has {len(srcs)} sources; the base restage() can only "
                "re-point a single-source reader via with_source(). Override restage() to stage "
                "each source (see MultiSampleReader)."
            )
        staged = stage_file(srcs[0], root / srcs[0].name)
        return self.with_source(filename=staged)

    def assemble_jagged(
        self,
        cols: dict[str, object],
        fields: list[str],
        stream_cfg: StreamConfig,
        b: int,
        gschema: GroupSchema | None = None,
        labels: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cut → sort → truncate → pad jagged columns into a structured ``(B, T)`` array.

        The shared `Reader`-base assembly (plan 24, Wave 2): delegates to
        `salt.core.data.stream._cut_sort_truncate_pad`. Every jagged-stream reader
        (easyjet, ftag1lite, the jagged-combine path of multisample, a future
        jagged-H5 reader) calls THIS instead of re-implementing pad/sentinel logic.

        **PARITY.** With ``stream_cfg`` carrying no cuts and no sort (the default)
        this reproduces the readers' previous contiguous truncate+pad+valid path
        byte-for-byte; the drop-then-pad / sort machinery engages only when cuts/sort
        are configured.

        Parameters
        ----------
        cols : dict[str, object]
            ``{field: jagged awkward array}`` of length ``b``.
        fields : list[str]
            Served field names in config order (the structured field order).
        stream_cfg : StreamConfig
            The cut/sort/pad spec (``pad_max`` resolved).
        b : int
            The number of rows.
        gschema : GroupSchema | None, optional
            The stream's schema group for per-field dtype casting.
        labels : dict[str, object] | None, optional
            Aligned columns permuted/cut in lockstep but not emitted as fields.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(structured (B, T) array, valid (B, T) bool)``.
        """
        return _cut_sort_truncate_pad(cols, fields, stream_cfg, b, gschema, labels)

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
