"""Dataset-side base classes for the v2 pipeline: `Reader` (disk -> numpy batch
dicts, owns read-time selections/transforms) and `Processor` (pure batch
transform). Both are `GraphModule`s compiled by the same kernel planner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from salt.data.readers.stream import OffsetIndex, StreamConfig, _cut_sort_truncate_pad
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.planner import PlanStep
from salt.graph.setup_spec import SetupIO, SetupStage
from salt.graph.spec import _UNNAMED, IO, KEY_SEP, Mode
from salt.schema import GroupSchema, Schema

if TYPE_CHECKING:
    from salt.data.readers.cuts import ConstituentCuts, GlobalObjectCuts

__all__ = [
    "OffsetIndex",
    "Processor",
    "Reader",
    "SaltDatasetModule",
    "SetupBundle",
    "StreamConfig",
    "WorkerCtx",
]

# The setup-time carrier: the run `Bundle`'s write-once, dotted-key machinery
# is leaf-type-agnostic, so it is reused as-is as the setup bundle — its
# leaves are path strings / scalar artifacts instead of tensors. `SetupBundle`
# is an alias (not a subclass) to keep the carrier a single implementation.
SetupBundle = Bundle

RAW_NAMESPACE = "raw"
"""Bundle namespace for post-selection structured arrays."""


def _require_root_deps(who: str, extra: str) -> None:
    """Import-time guard for the optional ROOT reader extras.

    Raises a clear, actionable error pointing at the correct install command
    instead of a bare ``ModuleNotFoundError`` from deep inside an array method.
    Cheap when the deps are present (cached imports).

    Raises
    ------
    ImportError
        When uproot / awkward are missing — names `who` and the pip extra.
    """
    try:
        import awkward  # noqa: F401
        import uproot  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{who} requires uproot + awkward — install with:\n"
            f"  pip install 'salt[{extra}]'\n"
            "or directly:\n"
            "  pip install uproot awkward"
        ) from exc


@dataclass(frozen=True)
class WorkerCtx:
    """Per-worker binding context handed to `SaltDatasetModule.bind`.

    Built by `GraphDataset` once per (worker, plan): `read_fields` is the
    demand-narrowed per-stream read set computed from the compiled plan,
    mapping ``stream -> {field: demanding module}`` in demand order. `step` is
    the receiving module's own plan step, so wildcard producers (`Labels`)
    can learn their narrowed key set. `seed` is the per-worker seed (torch's
    per-worker dataloader seed when running in a worker) for read-time
    augmentations.
    """

    mode: Mode
    read_fields: Mapping[str, Mapping[str, str]]
    seed: int
    worker_id: int = 0
    num_workers: int = 0
    step: PlanStep | None = None


class SaltDatasetModule(ABC):
    """Base class for dataset-side graph participants.

    Subclasses implement the `GraphModule` protocol (``name`` +
    ``declare_io``). ``declare_io`` is a function of the module's own config
    only — no data files, no tensors; the schema artifact consumed at
    construction time is config I/O.
    """

    incompatible_with: tuple[str, ...] = ()
    """Class names of setup modules this module must NOT coexist with.

    A reusable declarative mutual-exclusion pattern: a module names the class
    names (strings, not types — the named class may not exist yet) it is
    structurally incompatible with, and the setup-plan compiler — the only
    thing that sees the full module dict — enforces it
    (`_check_incompatibilities`). Default `()` (no exclusions). E.g. `VDS` sets
    ``("ShmStage",)``: staging a VDS would copy h5py pointers, not data.
    """

    def __init__(self) -> None:
        """Initialise the instance name placeholder (assigned from the config key)."""
        self.name: str = _UNNAMED

    @abstractmethod
    def declare_io(self, mode: Mode) -> IO:
        """Return the declared requires/produces for the given mode."""

    def bind(self, ctx: WorkerCtx) -> None:  # noqa: B027 - optional hook, deliberately concrete
        """Per-worker lazy setup (open handles, allocate buffers); default no-op.

        This is the only place dataset modules may touch data files. Called
        once per (worker process, plan) by `GraphDataset`.
        """

    # -- setup-time face (once per stage, not per batch) ----------------------
    # All three default to no-ops: a pure-source module (InputSamples/VDS/
    # ShmStage) overrides only declare_setup_io/setup and inherits an empty
    # declare_io; a pure processor inherits these no-ops; a dual-face reader
    # overrides both.

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Return the module's setup-time interface for `stage`; default empty.

        The setup-time analogue of `declare_io`. A function of the module's own
        config only — no data files, no tensors. A non-empty return for some
        stage is what marks a module as setup-participating; the per-batch
        `declare_io` face is unaffected.
        """
        del stage
        return SetupIO()

    def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
        """Run this module's setup-time side-effect for `stage`; default identity.

        The sole sanctioned setup-time ctx-mutation point (the setup analogue
        of per-batch `read`). Runs once per stage inside
        ``datamodule.setup(stage)``, reads its declared setup-`requires` off
        `ctx`, may touch the filesystem (glob, build a VDS, copy to
        ``/dev/shm``), and merges back only its declared setup-`produces`
        (write-once). Same code path on every DDP rank.

        The base default returns `ctx` unchanged — a no-op for per-batch-only
        modules (processors) whose `declare_setup_io` is empty.
        """
        del stage
        return ctx

    def teardown(self, ctx: SetupBundle, stage: SetupStage) -> None:
        """Reverse a setup-time side-effect for `stage`; default no-op.

        The symmetric cleanup hook (e.g. `ShmStage` rmtree-ing its
        ``/dev/shm`` root). Called from ``datamodule.teardown(stage)``, guarded
        so it fires only for the stage(s) the module actually set up.
        """
        del ctx, stage

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Per-stream raw fields this module demands from the reader.

        Default: the ``fields`` metadata of the module's bound
        ``raw.<stream>`` requires. Wildcard producers whose demands are only
        known after narrowing (`Labels`) override this using `step.produces`.
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


class Reader(SaltDatasetModule):
    """Disk -> flat dotted dict of numpy arrays for a contiguous batch slice.

    Source node: ``requires={}``. The arrays returned by `read` may alias the
    reader's reusable per-worker buffers — exactly one mandatory copy per
    batch (the `Features` / `Labels` materialisation) separates reader
    buffers from anything handed to the trainer; ``raw.*`` never crosses the
    torch boundary.

    Beyond ``read``, readers carry the framework-facing config surface the
    runtime builds on: the served `streams` (demand-driven producers adopt
    them), the optional `schema` artifact + `schema_group`/`label_universe`
    views (static validation and wildcard narrowing), and `with_source` (the
    datamodule's per-stage cloning).
    """

    schema: Schema | None = None
    """The dataset schema artifact, when configured."""

    cuts: GlobalObjectCuts | None = None
    """Sample-axis row eligibility (index-build kept-index), when configured.

    The uniform row-cut surface across readers: a `GlobalObjectCuts` evaluated once in
    `prepare` over the reader's sample-axis scalar record, selecting which rows
    enter the index. Changes `__len__`, and the served rows never read the dropped
    ones. SAMPLE-AXIS ONLY (jets for jet readers, events for event readers) —
    constituent (per-track) filtering is a separate concern and must NEVER route
    through this engine. Default `None` (identity: every row eligible).
    """

    constituent_cuts: dict[str, ConstituentCuts]
    """Per-stream constituent cuts (``on_fail: mask | drop``), when configured.

    The counterpart of `cuts`: these act WITHIN a row and never change `__len__`.
    Normalised from the reader's config surface by `_parse_constituent_cuts`.
    """

    stage: str | None = None
    """The bound stage (``"train"``/``"val"``/``"test"``) selecting per-split cuts."""

    vds_capable: bool = False
    """Whether this reader builds an h5py virtual dataset for wildcard sources.

    The `VDS` setup module gates build-vs-identity on this flag (not an
    `isinstance` check). Default `False` on the `Reader` base — a
    non-`vds_capable` reader (the ROOT `UprootReader`) keeps
    its own native glob, and the `VDS` module is an identity edge for it
    (``vds_path == pattern``, never calling `create_vds` on a ROOT glob, which
    would crash). `H5StructuredReader` overrides it to `True`.
    """

    @property
    @abstractmethod
    def streams(self) -> tuple[str, ...]:
        """The stream names this reader serves, in config order."""

    def prepare(self) -> None:
        """Main-process file probing (VDS resolution, row counts); default no-op.

        Idempotent; called lazily by ``__len__`` and eagerly by the
        datamodule's rank-0 VDS pre-creation.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of rows this reader serves."""

    @abstractmethod
    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous batch and return the produced keys (flat dotted dict)."""

    # -- shared row-cut engine (sample-axis, index-build only) ----------------

    @staticmethod
    def _row_record(row_scalars: Mapping[str, np.ndarray], n_rows: int) -> np.ndarray:
        """Pack sample-axis scalar columns into a structured ``(n_rows,)`` cut record."""
        names = list(row_scalars)
        dtype = np.dtype([(nm, row_scalars[nm].dtype) for nm in names])
        rec = np.empty(n_rows, dtype=dtype)
        for nm in names:
            rec[nm] = row_scalars[nm]
        return rec

    def _apply_row_cuts(self, rows: np.ndarray, split: str | None) -> np.ndarray:
        """Sample-axis keep mask over a structured row-scalar record — the shared engine.

        Delegates to `GlobalObjectCuts.eligible`; all-True when no cuts are configured or the
        split has none. Called at index-build (`prepare`), NEVER per batch: a dropped
        row must never be read. SAMPLE-AXIS ONLY — row cuts change `__len__`, so
        constituent filtering must not route through here.
        """
        if self.cuts is None or not self.cuts.for_split(split):
            return np.ones(len(rows), dtype=bool)
        return self.cuts.eligible(rows, split)

    @staticmethod
    def _parse_constituent_cuts(
        spec: Mapping[str, Any] | None, streams: Sequence[str]
    ) -> dict[str, ConstituentCuts]:
        """Normalise the per-stream constituent-cut config surface to `ConstituentCuts`.

        Accepts a `ConstituentCuts` or a plain mapping (``{cuts: [...], on_fail: ...}``)
        per stream; validates the stream names against `streams`.

        Returns
        -------
        dict[str, ConstituentCuts]
            The per-stream containers (streams configured ``null`` are dropped).

        Raises
        ------
        ConfigError
            On an unknown stream name or a value that is not a cut container.
        """
        from salt.data.readers.cuts import ConstituentCuts  # noqa: PLC0415 - avoid import cycle

        out: dict[str, ConstituentCuts] = {}
        for stream, cfg in (spec or {}).items():
            if stream not in streams:
                raise ConfigError(
                    f"constituent_cuts configured for unknown stream {stream!r} — known "
                    f"streams: {sorted(streams)}"
                )
            if cfg is None:
                continue
            if isinstance(cfg, ConstituentCuts):
                out[stream] = cfg
            elif isinstance(cfg, Mapping):
                try:
                    out[stream] = ConstituentCuts(**dict(cfg))
                except TypeError as exc:
                    raise ConfigError(
                        f"constituent_cuts[{stream!r}]: expected keys cuts/on_fail, got "
                        f"{sorted(cfg)}"
                    ) from exc
            else:
                raise ConfigError(
                    f"constituent_cuts[{stream!r}] must be a ConstituentCuts or a mapping "
                    f"of cuts/on_fail, got {cfg!r}"
                )
        return out

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The schema for one served stream, when a schema artifact is configured.

        Used by `GraphDataset` to statically validate demanded raw fields.
        Default: None (no static validation possible).
        """
        del stream
        return None

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing.

        When not None, every narrowed ``labels.**`` key is validated against
        this set at plan-compile time. Default: None (no schema — narrowing is
        unvalidated, bind-time checks remain).
        """
        return None

    @property
    def h5_source(self) -> Path | None:
        """The h5py-openable structured source file, when this reader has one.

        A capability advertisement (not a type tag): `H5OutputSink` opens this
        file to probe per-stream sequence lengths (pad-mask columns) and to copy
        input fields. A reader without a structured HDF5 source — the ROOT/uproot
        `UprootReader` (whose source is a ROOT file and whose ``groups`` are a
        different, non-H5 config shape), a `MultiSampleReader` (N sources, no
        single file), or a
        global-only custom reader — returns None, and the sink takes its
        no-source path (task-outputs only; pad-mask columns / input-copying then
        raise a clear ConfigError). Default None; `H5StructuredReader` overrides.
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
        single configured prototype.

        `stage` (``"train"``/``"val"``/``"test"``) is the optional per-reader
        stage-sourcing hook: single-source readers (`H5StructuredReader`,
        `UprootReader`) ignore it — their one ``filename`` per stage is the
        data. Multi-source readers (`MultiSampleReader`, a future cut-based
        reader) use it to select each sub-source's per-stage data. The
        datamodule passes the stage it already knows; readers that don't need
        it never look at it.

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
        the framework can relocate them. The base default introspects a
        ``filename`` or ``files`` attribute, returning its `Path`(s); readers
        whose sources are not a single such attribute (`MultiSampleReader`)
        override.

        Wildcard / glob filenames are returned verbatim (a literal pattern, not
        its expansion) — staging a wildcard reader is the caller's
        responsibility (the shipped staging path stages already-resolved
        single files). A reader with no bound source returns an empty list.
        """
        files = getattr(self, "files", None)
        if files is not None:
            return [Path(f) for f in files]
        filename = getattr(self, "filename", None)
        return [Path(filename)] if filename is not None else []

    def restage(self, root: str | Path) -> Reader:
        """Return a clone of this reader whose `sources` point at copies under `root`.

        The reader-owned twin of `with_source`: rather than re-source onto a
        different file, `restage` copies this reader's own source file(s) into
        ``root`` (typically a RAM disk like ``/dev/shm``) and returns a clone
        that reads the copies, so multi-file / multi-sample readers stage all
        their files.

        Base default: copy each `sources` file to ``root`` via the
        FileLock-coordinated `salt.data.readers.vds.stage_file` (a DDP / worker
        stampede copies each file exactly once), then clone with the new path
        via `with_source`. A reader with a single source uses this directly;
        multi-source readers (`MultiSampleReader`) override to restage each
        sub-reader recursively. A reader with no source clones unchanged.

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
        from salt.data.readers.vds import stage_file  # noqa: PLC0415 - opt-in staging path only

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
        """Cut -> sort -> truncate -> pad jagged columns into a structured ``(B, T)`` array.

        Delegates to `salt.data.readers.stream._cut_sort_truncate_pad`. Every
        jagged-stream reader (easyjet, ftag1lite, the jagged-combine path of
        multisample) calls this instead of re-implementing pad/sentinel logic.

        With ``stream_cfg`` carrying no cuts and no sort (the default), only
        truncate+pad+valid run; the drop-then-pad / sort machinery engages
        only when cuts/sort are configured.

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

        Used by the ``debug`` boundary check (leaves surviving to the torch
        boundary must not alias a reusable buffer). Default: False (no
        registered buffers).
        """
        del array
        return False


class Processor(SaltDatasetModule):
    """Pure batch transform on the numpy bundle.

    `process` reads its declared requires from the bundle and returns only
    newly produced keys, as a flat dotted dict (or nested — both spellings
    are canonicalised by the runner). Processors may use private scratch
    buffers, but any leaf that survives to the torch boundary must not alias
    a reusable reader buffer.
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
