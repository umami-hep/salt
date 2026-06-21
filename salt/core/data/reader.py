"""`H5StructuredReader` — the throughput-preserving H5 reader (design §6.1).

A port of the fast path of v1 ``SaltDataset`` (hidden contract #9, kept
wholesale — ``datasets.py:367-405, 448-524``):

- contiguous B-element slab reads from structured arrays into REUSABLE
  per-worker numpy buffers (``ndarray.resize(refcheck=False)`` +
  ``ds.read_direct``, ``datasets.py:459-462``);
- lazy pid-guarded ``h5py.File(swmr=True, libver="latest")`` handles, one per
  (worker, file), shared across groups (``datasets.py:367-381``);
- VDS creation for wildcard filenames (`salt.core.data.vds`, FileLock + done
  marker + the design's staleness check);
- per-group ``truncate`` keeping the LEADING constituents (pt-sorted dumps,
  ``datasets.py:469-471``);
- read-time mutation stages in v1 order: per-stream ``selections`` (ftag
  `Cuts` via `TrackSelector` — NaN floats, -1 ints, ``valid=False`` for
  failing constituents) immediately after the read, then ``transforms``
  (FIT-only by mode flag, per-worker seeded — fixing ``transforms.py:58``).
  Because they run before ``raw.*`` / ``masks.*`` exist, cuts flow into
  features, masks AND labels exactly as v1 (design §2.4).
  **Documented deviation — transform scope is BROADER than v1**: transforms
  receive the FULL post-selection structured batch, so label fields are in
  scope; v1 transformed the input-variables subset only
  (``datasets.py:507-510``), so labels could never be augmented there. The
  design (§2.4) sanctions the reader-stage placement; transform authors
  must keep ``used_fields`` to input variables unless changing training
  labels is intended;
- demand-narrowed read columns: the per-mode read set is
  ``(demanded union selection/transform fields) intersect schema``, computed from the
  compiled plan and delivered via `WorkerCtx.read_fields` — strictly less
  I/O than v1's read-all-jets-columns amplification (``datasets.py:395-396``).
  ``get_dtype`` semantics are kept (FILE field order, on-disk ``f2``
  preserved via ``as_half``, ``valid`` auto-appended, ``datasets.py:742-776``).

Produces ``raw.<stream>`` (structured, post-selection — may alias the
reusable buffer; never crosses the torch boundary), ``masks.<stream>``
(``~valid``, True = padded, ``datasets.py:523``) for non-``global_object``
streams, and ``meta.rows`` (TEST only). The v1 magic exclusion set
``{parameters, global_object, "global"}`` becomes the explicit per-group
``global_object: true`` config (design §6.1), inferred from the schema
artifact's ``valid`` field when not given.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag import Cuts
from ftag.track_selector import TrackSelector

from salt.core.data.base import Reader, WorkerCtx
from salt.core.data.stream import StreamConfig
from salt.core.data.vds import create_vds, has_wildcard
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema, Schema, load_schema
from salt.core.data.dtypes import get_dtype

__all__ = ["GroupConfig", "H5StructuredReader"]

_SUGGESTION_CUTOFF = 0.5


@dataclass(frozen=True)
class GroupConfig:
    """Per-stream reader configuration (design §2.6 / §6.1).

    `dataset` is the H5 dataset name the stream maps to (v1 ``input_map``;
    ``None`` — the default, so empty YAML group blocks parse through
    jsonargparse (design §5.3) — resolves to the stream name in
    `H5StructuredReader._parse_group`). `truncate` keeps the leading N
    constituents (v1 ``num_inputs``). `global_object` declares a ``[B, F]``
    stream carrying no pad mask (the explicit replacement for v1's magic
    exclusion set, design §6.1; aligns with `Normaliser(global_object=...)`);
    None infers it from the schema artifact (no ``valid`` field =>
    global_object).
    """

    dataset: str | None = None
    truncate: int | None = None
    global_object: bool | None = None

    def __post_init__(self) -> None:
        if self.truncate is not None and self.truncate < 1:
            raise ConfigError(f"group truncate must be >= 1, got {self.truncate}")


class H5StructuredReader(Reader):
    """Structured-array H5 reader with reusable per-worker buffers (design §6.1).

    Parameters
    ----------
    groups : Mapping[str, GroupConfig | Mapping | None]
        Stream name -> group config (``{dataset:, truncate:, global_object:}``).
        A None / empty value defaults the dataset name to the stream name.
        The FIRST group defines the reader length (rows on axis 0 are aligned
        across groups by the file format).
    schema : Schema | str | Path | None, optional
        The dataset schema artifact (design §2.6) or its YAML path. Reading
        it here is config I/O, not data I/O (§2.6). When given,
        ``global_object`` is inferred for unset groups and selection/transform
        fields are validated statically; when None (explicit opt-out) every
        group must set ``global_object`` and field typos surface at worker
        bind (design §2.6).
    filename : str | Path | None, optional
        Input H5 file (wildcards trigger VDS creation in `prepare`). May be
        omitted at construction and supplied via `with_source` — the
        datamodule pattern (design §6.1).
    num : int, optional
        Number of rows to serve; ``-1`` = all (v1 ``get_num`` semantics).
    selections : Mapping[str, Sequence[str]] | None, optional
        Per-stream ftag cut lists, applied to the structured array
        immediately after the read (v1 ``datasets.py:464-466`` semantics).
    transforms : Sequence[Callable] | None, optional
        Read-time augmentations with the v1 protocol
        ``transform(struct_array, stream) -> struct_array``, applied FIT-only
        (design §6.1, fixing v1's every-stage application). A transform
        accepting an ``rng`` keyword receives the per-worker seeded
        ``np.random.Generator``; one exposing ``used_fields(stream)`` gets
        those fields added to the read set. NOTE: the array handed over is
        the FULL post-selection structured batch — label fields are in scope
        (broader than v1, see the module docstring); keep ``used_fields`` to
        input variables unless label augmentation is intended.
    vds_path : str | Path | None, optional
        Explicit VDS output path for wildcard filenames.

    Raises
    ------
    ConfigError
        On malformed group configs, unknown selection/transform streams, or
        an unset ``global_object`` flag with no schema to infer from.
    SchemaError
        When the schema artifact contradicts the config (missing groups or
        fields, a non-global_object group without ``valid``).
    """

    vds_capable: bool = True
    """H5 reader builds a virtual dataset for wildcard sources (plan-25 O-VDS-CAP).

    Overrides the `Reader` base default (`False`): the `VDS` setup module builds
    a real VDS (via `create_vds`) for a wildcard source of an `H5StructuredReader`,
    rather than passing the glob through as an identity edge.
    """

    def __init__(
        self,
        groups: Mapping[str, GroupConfig | Mapping[str, Any] | None],
        schema: Schema | str | Path | None = None,
        filename: str | Path | None = None,
        num: int = -1,
        selections: Mapping[str, Sequence[str]] | None = None,
        transforms: Sequence[Callable] | None = None,
        vds_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError("H5StructuredReader needs at least one group (design §6.1)")
        # config capture only (design §2.3); schema load is config I/O (§2.6)
        self.schema = load_schema(schema) if isinstance(schema, str | Path) else schema
        self.filename = Path(filename) if filename is not None else None
        self.num = num
        self.vds_path = Path(vds_path) if vds_path is not None else None
        self.groups: dict[str, GroupConfig] = {
            stream: self._parse_group(stream, cfg) for stream, cfg in groups.items()
        }
        self.selections = {k: list(v) for k, v in (selections or {}).items()}
        self.transforms = list(transforms or [])
        self._selectors: dict[str, TrackSelector] = {}
        self._selection_fields: dict[str, list[str]] = {}
        for stream, cut_list in self.selections.items():
            if stream not in self.groups:
                raise ConfigError(
                    f"selections configured for unknown stream {stream!r} — known streams: "
                    f"{sorted(self.groups)} (design §6.1)"
                )
            cuts = Cuts.from_list(list(cut_list))
            self._selectors[stream] = TrackSelector(cuts)
            self._selection_fields[stream] = list(cuts.variables)
        self._transform_wants_rng = [
            "rng" in inspect.signature(transform).parameters for transform in self.transforms
        ]
        self._transform_fields: dict[str, list[str]] = {}
        for transform in self.transforms:
            used = getattr(transform, "used_fields", None)
            if used is None:
                continue
            for stream in self.groups:
                self._transform_fields.setdefault(stream, []).extend(used(stream))
        self._validate_against_schema()
        # transient per-process state (never pickled, see __getstate__)
        self._resolved: Path | None = None
        self._num_rows: int | None = None
        self._h5: h5py.File | None = None
        self._pid: int | None = None
        self._dss: dict[str, h5py.Dataset] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._rng: np.random.Generator | None = None

    # -- config helpers ------------------------------------------------------

    @staticmethod
    def _parse_group(stream: str, cfg: GroupConfig | Mapping[str, Any] | None) -> GroupConfig:
        """Normalise one group config entry to a `GroupConfig`.

        Returns
        -------
        GroupConfig
            The parsed config (dataset defaults to the stream name).

        Raises
        ------
        ConfigError
            On unknown keys or a non-mapping entry.
        """
        if isinstance(cfg, GroupConfig):
            if cfg.dataset is None:
                return GroupConfig(
                    dataset=stream, truncate=cfg.truncate, global_object=cfg.global_object
                )
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - {"dataset", "truncate", "global_object"}
        if unknown:
            raise ConfigError(
                f"group {stream!r}: unknown config keys {sorted(unknown)} — expected "
                "dataset/truncate/global_object (design §2.6)"
            )
        return GroupConfig(
            dataset=str(cfg.get("dataset", stream)),
            truncate=cfg.get("truncate"),
            global_object=cfg.get("global_object"),
        )

    def _validate_against_schema(self) -> None:
        """Resolve per-group ``global_object`` flags and statically validate config fields.

        Raises
        ------
        ConfigError
            If ``global_object`` is unset for a group and no schema is available.
        SchemaError
            If the schema lacks a configured group/field, or a group is
            declared (or inferred) non-global_object without a ``valid`` field.
        """
        resolved: dict[str, GroupConfig] = {}
        for stream, cfg in self.groups.items():
            gschema = self.schema.groups.get(cfg.dataset) if self.schema is not None else None
            if self.schema is not None and gschema is None:
                near = get_close_matches(
                    cfg.dataset, sorted(self.schema.groups), n=3, cutoff=_SUGGESTION_CUTOFF
                )
                hint = f"; nearest: {', '.join(near)}" if near else ""
                raise SchemaError(
                    f"group {stream!r}: dataset {cfg.dataset!r} not in the schema artifact"
                    f"{hint} (design §2.6)"
                )
            global_object = cfg.global_object
            if global_object is None:
                if gschema is None:
                    raise ConfigError(
                        f"group {stream!r}: 'global_object' is unset and no schema artifact is "
                        "available to infer it — set global_object: true/false explicitly or "
                        "provide schema: (design §2.6, §6.1)"
                    )
                global_object = not gschema.has_valid
            if gschema is not None and not global_object and not gschema.has_valid:
                raise SchemaError(
                    f"group {stream!r} (dataset {cfg.dataset!r}) is a sequence stream "
                    "(global_object: false) but the schema has no 'valid' field — pad masks "
                    "cannot be derived (design §6.1)"
                )
            resolved[stream] = GroupConfig(cfg.dataset, cfg.truncate, global_object)
            if gschema is not None:
                config_fields = self._selection_fields.get(stream, []) + self._transform_fields.get(
                    stream, []
                )
                for field in config_fields:
                    if field not in gschema.fields:
                        near = get_close_matches(
                            field, sorted(gschema.fields), n=3, cutoff=_SUGGESTION_CUTOFF
                        )
                        hint = f"; nearest: {', '.join(near)}" if near else ""
                        raise SchemaError(
                            f"selection/transform field {field!r} for stream {stream!r} not "
                            f"present in schema group {cfg.dataset!r}{hint} (design §6.1)"
                        )
        self.groups = resolved

    @property
    def streams(self) -> tuple[str, ...]:
        """The configured stream names, in config order.

        Returns
        -------
        tuple[str, ...]
            Stream names.
        """
        return tuple(self.groups)

    def sources(self) -> list[Path]:
        """The single H5 source file (the M8 staging surface, design §6.1).

        One configured ``filename`` per stage IS the H5 reader's data (the
        single-source contract `with_source` and `restage` both rely on). A wildcard
        ``filename`` is returned VERBATIM — the literal pattern, not the matched
        members — so the base `Reader.restage` (which copies one file) is only used on
        the resolved single-file staging path the datamodule drives; the empty list is
        returned when no source is bound yet.

        Returns
        -------
        list[Path]
            ``[self.filename]`` (literal), or ``[]`` when unbound.
        """
        return [self.filename] if self.filename is not None else []

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The schema artifact's group for one served stream (design §2.6).

        Returns
        -------
        GroupSchema | None
            The group schema (looked up via the stream's ``dataset`` name),
            or None when no schema artifact is configured or the stream is
            unknown.
        """
        if self.schema is None:
            return None
        cfg = self.groups.get(stream)
        if cfg is None:
            return None
        return self.schema.groups.get(cfg.dataset)

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing (design §2.2 rule d).

        Returns
        -------
        tuple[str, ...] | None
            All schema-backed label keys, or None when no schema artifact is
            configured (validation falls back to the bind-time check, §2.6).
        """
        if self.schema is None:
            return None
        return tuple(
            f"labels.{stream}.{field}"
            for stream, cfg in self.groups.items()
            for field in self.schema.groups[cfg.dataset].fields
        )

    def _stream_config(self, stream: str) -> StreamConfig | None:
        """The `StreamConfig` for a sequence stream (``truncate`` → ``pad_max``), else None.

        The shared base vocabulary (plan 24, Wave 2): the H5 reader reads dense,
        already-padded structured arrays, so its only `StreamConfig` knob is the
        ``truncate`` leading-keep (``pad_max``). Streams with no ``truncate`` return
        ``None``; a configured ``truncate`` applies to the leading axis whether or
        not the stream is ``global_object`` (mirrors the pre-Wave-2 leading slice).
        The H5 read path has NO per-constituent cuts/sort surface, so this config
        never carries cuts/sort — the drop-then-pad / sort machinery is never engaged
        on the H5 path (it serves dense contiguous slabs), preserving byte-identity
        (plan 24 §6/§7).

        Returns
        -------
        StreamConfig | None
            The per-stream pad config, or None for scalar / untruncated streams.
        """
        cfg = self.groups[stream]
        if cfg.truncate is None:
            return None
        # truncate applies on the leading axis whether or not the stream is a
        # global_object — mirrors the pre-Wave-2 `batch[:, :cfg.truncate]` exactly
        # (byte-identical; the global+truncate combo is configurable, reader.py:91-93).
        return StreamConfig(pad_max=cfg.truncate, jagged=not cfg.global_object)

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,
        stage: str | None = None,
    ) -> H5StructuredReader:
        """Clone this reader for another source file (the datamodule pattern, design §6.1).

        Config-only (no file I/O): the loaded schema artifact, group configs,
        selections and transforms are shared; only the file binding changes.
        ``stage`` is accepted for the plan-02 stage-sourcing contract and IGNORED
        here (a single-source reader's one ``filename`` per stage IS its data).

        Returns
        -------
        H5StructuredReader
            A fresh, unbound reader instance (same instance ``name``).
        """
        del stage  # single-source reader: the per-stage filename is the data
        clone = H5StructuredReader(
            groups=self.groups,
            schema=self.schema,
            filename=filename,
            num=num,
            selections=self.selections,
            transforms=self.transforms,
            vds_path=vds_path,
        )
        clone.name = self.name
        return clone

    # -- GraphModule declaration (config-only, design §2.2/§2.3) -------------

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.* / masks.* / meta.rows`` produces (source node, requires={}).

        Sequence dims are concrete when ``truncate`` is set, else symbolic
        ``T:<stream>``; ``meta.rows`` is TEST-only (design §2.1).

        Returns
        -------
        IO
            The declared interface.
        """
        del mode
        flat: dict[str, TensorSpec] = {}
        for stream, cfg in self.groups.items():
            t_dim: int | str = cfg.truncate if cfg.truncate is not None else sym_dim("T", stream)
            shape = ("B",) if cfg.global_object else ("B", t_dim)
            flat[f"raw.{stream}"] = TensorSpec(shape=shape, kind="data")
            if not cfg.global_object:
                flat[f"masks.{stream}"] = TensorSpec(shape=shape, dtype="bool", kind="pad_mask")
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    # -- file-touching lifecycle hooks (design §2.3) --------------------------

    def prepare(self) -> None:
        """Resolve the source file (VDS for wildcards) and probe row counts.

        Main-process hook (called lazily by ``__len__``, eagerly by the
        datamodule's rank-0 VDS pre-creation). Idempotent.

        Raises
        ------
        ConfigError
            If no filename is bound, or ``truncate`` exceeds the file's
            constituent dimension (v1's per-batch assert, ``datasets.py:470``,
            moved up front).
        ValueError
            If ``num`` requests more rows than available (v1 ``get_num``).
        SchemaError
            If a configured dataset is missing or not a structured array.
        """
        if self._resolved is not None:
            return
        if self.filename is None:
            raise ConfigError(
                f"reader {self.name!r} has no source file — pass filename= or use "
                "with_source() (design §6.1)"
            )
        path = self.filename
        if has_wildcard(path):
            path = create_vds(path, self.vds_path)
        with h5py.File(path, "r") as f:
            for stream, cfg in self.groups.items():
                node = f.get(cfg.dataset)
                if not isinstance(node, h5py.Dataset) or node.dtype.names is None:
                    raise SchemaError(
                        f"group {stream!r}: {cfg.dataset!r} in {path.name!r} is missing or "
                        f"not a structured dataset; available: {sorted(f.keys())}"
                    )
                if cfg.truncate is not None and (node.ndim < 2 or cfg.truncate > node.shape[1]):
                    raise ConfigError(
                        f"group {stream!r}: truncate={cfg.truncate} exceeds the file's "
                        f"constituent dimension {node.shape[1:]} (datasets.py:470 semantics)"
                    )
            first = next(iter(self.groups.values()))
            num_available = len(f[first.dataset])
        if self.num > num_available:
            raise ValueError(
                f"Requested {self.num:,} rows, but only {num_available:,} are available "
                f"in {path.name!r}."
            )
        self._num_rows = num_available if self.num < 0 else self.num
        self._resolved = path

    def __len__(self) -> int:
        """Return the number of rows served (resolving the source on first call).

        Returns
        -------
        int
            The row count.
        """
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    @property
    def source_path(self) -> Path:
        """The resolved source file (the VDS for wildcard filenames).

        Resolves on first access (`prepare` is idempotent). Writers re-read
        input copies from this path by absolute rows (design §8).

        Returns
        -------
        Path
            The concrete H5 file backing this reader.
        """
        self.prepare()
        assert self._resolved is not None
        return self._resolved

    def _ensure_open(self) -> None:
        """Open (or re-open) the per-process H5 handle, pid-guarded.

        Port of ``SaltDataset._ensure_open`` (``datasets.py:367-381``): guards
        against handle inheritance across fork.
        """
        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            if self._h5 is not None and self._h5.id.valid:
                self._h5.close()
            assert self._resolved is not None
            self._h5 = h5py.File(self._resolved, "r", swmr=True, libver="latest")
            self._pid = pid

    def bind(self, ctx: WorkerCtx) -> None:
        """Per-worker setup: open the handle, allocate demand-narrowed buffers.

        Port of ``SaltDataset._setup`` (``datasets.py:383-405``) with the
        design's read-set narrowing: each group's buffer dtype covers
        ``demanded union selection/transform fields`` only (FILE field order,
        ``as_half``, ``valid`` auto-appended — ``get_dtype``,
        ``datasets.py:742-776``). Demanded fields absent from the live file
        raise before any training step (design §2.6).

        Raises
        ------
        SchemaError
            Naming the demanding module and the nearest field, when a
            demanded field is not in the H5 group.
        """
        self.prepare()
        self._ensure_open()
        assert self._h5 is not None
        self._dss = {}
        self._buffers = {}
        for stream, cfg in self.groups.items():
            ds = self._h5[cfg.dataset]
            demanded = dict(ctx.read_fields.get(stream, {}))
            for field in self._selection_fields.get(stream, []):
                demanded.setdefault(field, f"{self.name} (selections)")
            for field in self._transform_fields.get(stream, []):
                demanded.setdefault(field, f"{self.name} (transforms)")
            file_fields = ds.dtype.names or ()
            for field, who in demanded.items():
                if field not in file_fields:
                    near = get_close_matches(
                        field, sorted(file_fields), n=1, cutoff=_SUGGESTION_CUTOFF
                    )
                    hint = f"; nearest: {near[0]}" if near else ""
                    raise SchemaError(
                        f"field {field!r} demanded by {who!r} not present in h5 "
                        f"group {cfg.dataset!r}{hint} (design §2.6)"
                    )
            if not cfg.global_object and "valid" not in file_fields:
                raise SchemaError(
                    f"group {stream!r} (dataset {cfg.dataset!r}) is a sequence stream but "
                    f"the file has no 'valid' field — pad masks cannot be derived "
                    "(design §6.1)"
                )
            dtype = get_dtype(ds, list(demanded))
            self._dss[stream] = ds
            self._buffers[stream] = np.array(0, dtype=dtype)
        self._rng = np.random.default_rng(ctx.seed)

    # -- the per-batch read (design §6.1, v1 order kept) -----------------------

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous batch slab (v1 ``__getitem__`` fast path order).

        Per group: resize the reusable buffer + ``read_direct``
        (``datasets.py:459-462``), apply selections (``:464-466``), truncate
        (``:469-471``), apply transforms (FIT-only, seeded — design §6.1),
        then derive ``masks.<stream> = ~valid`` (``:518-523``). ``raw.*``
        values may alias the reusable buffers (contract #9); masks are fresh
        arrays. ``meta.rows`` is produced in TEST.

        Returns
        -------
        dict[str, np.ndarray]
            The produced keys, as a flat dotted dict.
        """
        out: dict[str, np.ndarray] = {}
        for stream, cfg in self.groups.items():
            ds = self._dss[stream]
            buf = self._buffers[stream]
            shape = (rows.stop - rows.start, *ds.shape[1:])
            buf.resize(shape, refcheck=False)
            if buf.dtype.names:
                ds.read_direct(buf, rows)
            batch = buf
            if (selector := self._selectors.get(stream)) is not None:
                batch = selector(batch)
            # truncate via the shared StreamConfig leading-keep (pad_max == truncate);
            # None for scalar/untruncated streams → no-op (byte-identical, plan 24 W2)
            stream_cfg = self._stream_config(stream)
            if stream_cfg is not None:
                batch = batch[:, : stream_cfg.pad_max]
            if mode == Mode.FIT:
                for transform, wants_rng in zip(
                    self.transforms, self._transform_wants_rng, strict=True
                ):
                    batch = (
                        transform(batch, stream, rng=self._rng)
                        if wants_rng
                        else transform(batch, stream)
                    )
            out[f"raw.{stream}"] = batch
            if not cfg.global_object:
                out[f"masks.{stream}"] = ~batch["valid"]  # True = padded (datasets.py:523)
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out

    def aliases(self, array: np.ndarray) -> bool:
        """Check whether `array` shares memory with a reusable read buffer.

        The debug-boundary non-aliasing assertion of design §2.4.

        Returns
        -------
        bool
            True if `array` may share memory with any group buffer.
        """
        return any(np.may_share_memory(array, buf) for buf in self._buffers.values())

    # -- pickling (fork is free; spawn re-binds in the worker) ----------------

    def __getstate__(self) -> dict[str, Any]:
        """Drop live H5 state so the reader pickles under spawn contexts.

        Returns
        -------
        dict[str, Any]
            The picklable state (handles/buffers reset; re-created at bind).
        """
        state = self.__dict__.copy()
        state.update({"_h5": None, "_pid": None, "_dss": {}, "_buffers": {}, "_rng": None})
        return state
