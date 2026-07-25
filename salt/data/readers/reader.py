"""`H5StructuredReader` — the throughput-preserving H5 reader.

Contiguous slab reads into reusable per-worker buffers, lazy SWMR handles,
demand-narrowed columns; produces ``raw.<stream>``, ``masks.<stream>``, ``meta.rows``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from salt.data.base import Reader, WorkerCtx
from salt.data.dtypes import get_dtype
from salt.data.readers.stream import StreamConfig
from salt.data.readers.vds import create_vds, has_wildcard
from salt.graph.errors import _SUGGESTION_CUTOFF, ConfigError, SchemaError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.schema import GroupSchema, Schema, load_schema

if TYPE_CHECKING:
    from salt.data.readers.cuts import GlobalObjectCuts

__all__ = ["GroupConfig", "H5StructuredReader"]


@dataclass(frozen=True)
class GroupConfig:
    """Per-stream reader configuration.

    `dataset` is the H5 dataset name the stream maps to (``None`` — the
    default, so empty YAML group blocks parse through jsonargparse — resolves
    to the stream name in `H5StructuredReader._parse_group`). `truncate` keeps
    the leading N constituents. `global_object` declares a ``[B, F]`` stream
    carrying no pad mask; None infers it from the schema artifact (no
    ``valid`` field => global_object).
    """

    dataset: str | None = None
    truncate: int | None = None
    global_object: bool | None = None

    def __post_init__(self) -> None:
        if self.truncate is not None and self.truncate < 1:
            raise ConfigError(f"group truncate must be >= 1, got {self.truncate}")


class H5StructuredReader(Reader):
    """Structured-array H5 reader with reusable per-worker buffers.

    Parameters
    ----------
    groups : Mapping[str, GroupConfig | Mapping | None]
        Stream name -> group config (``{dataset:, truncate:, global_object:}``).
        A None / empty value defaults the dataset name to the stream name.
        The first group defines the reader length (rows on axis 0 are aligned
        across groups by the file format).
    schema : Schema | str | Path | None, optional
        The dataset schema artifact or its YAML path. When given,
        ``global_object`` is inferred for unset groups and selection/transform
        fields are validated statically; when None (explicit opt-out) every
        group must set ``global_object`` and field typos surface at worker
        bind.
    filename : str | Path | None, optional
        Input H5 file (wildcards trigger VDS creation in `prepare`). May be
        omitted at construction and supplied via `with_source`.
    num : int, optional
        Number of rows to serve; ``-1`` = all.
    constituent_cuts : Mapping[str, Any] | None, optional
        Per-stream `ConstituentCuts` (or the equivalent mapping
        ``{cuts: [...], on_fail: mask|drop}``), applied to the structured array
        immediately after the read. Distinct from `cuts` (below): constituent cuts
        filter constituents WITHIN a row (``mask`` blanks them in place, ``drop``
        removes and re-pads); `cuts` drop whole rows.
    cuts : GlobalObjectCuts | None, optional
        Sample-axis (per-jet) row eligibility evaluated once in `prepare` over the
        first group's scalar record; only passing rows enter the index (kept-index +
        filtered read). Changes `__len__`; the served rows never read the dropped
        ones. ``None`` (default) is the identity path — byte-identical to a no-cut
        contiguous read.
    stage : str | None, optional
        The bound stage (``"train"``/``"val"``/``"test"``) selecting per-split cuts;
        threaded by the datamodule via `with_source`.
    transforms : Sequence[Callable] | None, optional
        Read-time augmentations with the protocol
        ``transform(struct_array, stream) -> struct_array``, applied FIT-only.
        A transform accepting an ``rng`` keyword receives the per-worker seeded
        ``np.random.Generator``; one exposing ``used_fields(stream)`` gets
        those fields added to the read set. The array handed over is the full
        post-selection structured batch — label fields are in scope; keep
        ``used_fields`` to input variables unless label augmentation is
        intended.
    vds_path : str | Path | None, optional
        Explicit VDS output path for wildcard filenames.

    Raises
    ------
    ConfigError
        On malformed group configs, unknown constituent-cut/transform streams,
        constituent cuts on a ``global_object`` stream, or an unset
        ``global_object`` flag with no schema to infer from.
    SchemaError
        When the schema artifact contradicts the config (missing groups or
        fields, a non-global_object group without ``valid``).
    """

    vds_capable: bool = True
    """H5 reader builds a virtual dataset for wildcard sources.

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
        constituent_cuts: Mapping[str, Any] | None = None,
        cuts: GlobalObjectCuts | None = None,
        stage: str | None = None,
        transforms: Sequence[Callable] | None = None,
        vds_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError("H5StructuredReader needs at least one group (design §6.1)")
        self.schema = load_schema(schema) if isinstance(schema, str | Path) else schema
        self.filename = Path(filename) if filename is not None else None
        self.num = num
        self.cuts = cuts
        self.stage = stage
        self.vds_path = Path(vds_path) if vds_path is not None else None
        self.groups: dict[str, GroupConfig] = {
            stream: self._parse_group(stream, cfg) for stream, cfg in groups.items()
        }
        self.constituent_cuts = self._parse_constituent_cuts(constituent_cuts, tuple(self.groups))
        self.transforms = list(transforms or [])
        self._constituent_fields: dict[str, list[str]] = {
            stream: list(cc.fields) for stream, cc in self.constituent_cuts.items()
        }
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
        self._kept: np.ndarray | None = None  # sample-axis kept file-rows; None = identity
        self._h5: h5py.File | None = None
        self._pid: int | None = None
        self._dss: dict[str, h5py.Dataset] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._rng: np.random.Generator | None = None

    @staticmethod
    def _parse_group(stream: str, cfg: GroupConfig | Mapping[str, Any] | None) -> GroupConfig:
        """Normalise one group config entry to a `GroupConfig` (dataset defaults to stream name)."""
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
        """Resolve per-group ``global_object`` flags and statically validate config fields."""
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
            if global_object and stream in self.constituent_cuts:
                raise ConfigError(
                    f"constituent_cuts on stream {stream!r}, which is a global_object "
                    "(scalar) stream — constituent cuts act within a sequence row; use "
                    "the reader's row-level cuts: instead"
                )
            if gschema is not None:
                config_fields = self._constituent_fields.get(
                    stream, []
                ) + self._transform_fields.get(stream, [])
                for field in config_fields:
                    if field not in gschema.fields:
                        near = get_close_matches(
                            field, sorted(gschema.fields), n=3, cutoff=_SUGGESTION_CUTOFF
                        )
                        hint = f"; nearest: {', '.join(near)}" if near else ""
                        raise SchemaError(
                            f"constituent-cut/transform field {field!r} for stream "
                            f"{stream!r} not present in schema group {cfg.dataset!r}"
                            f"{hint} (design §6.1)"
                        )
        self.groups = resolved

    @property
    def streams(self) -> tuple[str, ...]:
        """The configured stream names, in config order."""
        return tuple(self.groups)

    def sources(self) -> list[Path]:
        """The single H5 source file (the multi-file staging surface); a wildcard
        `filename` is returned verbatim (not its matched members). Empty when unbound.
        """
        return [self.filename] if self.filename is not None else []

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The schema artifact's group for one served stream, looked up via its ``dataset``
        name; None when no schema is configured or the stream is unknown.
        """
        if self.schema is None:
            return None
        cfg = self.groups.get(stream)
        if cfg is None:
            return None
        return self.schema.groups.get(cfg.dataset)

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing; None
        when no schema is configured.
        """
        if self.schema is None:
            return None
        return tuple(
            f"labels.{stream}.{field}"
            for stream, cfg in self.groups.items()
            for field in self.schema.groups[cfg.dataset].fields
        )

    def _stream_config(self, stream: str) -> StreamConfig | None:
        """The `StreamConfig` for a sequence stream (``truncate`` -> ``pad_max``), else None.

        Carries no cuts: the structured H5 slab applies `ConstituentCuts` directly in
        `read` (both ``mask`` and ``drop``), never through the jagged awkward pipeline.
        """
        cfg = self.groups[stream]
        if cfg.truncate is None:
            return None
        return StreamConfig(pad_max=cfg.truncate, jagged=not cfg.global_object)

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,
        stage: str | None = None,
    ) -> H5StructuredReader:
        """Clone onto another source file (config-only: schema, group configs, constituent
        cuts, row cuts, and transforms are shared); `stage` selects the clone's per-split
        row cuts.
        """
        clone = H5StructuredReader(
            groups=self.groups,
            schema=self.schema,
            filename=filename,
            num=num,
            constituent_cuts=self.constituent_cuts,
            cuts=self.cuts,
            stage=stage if stage is not None else self.stage,
            transforms=self.transforms,
            vds_path=vds_path,
        )
        clone.name = self.name
        return clone

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>``/``masks.<stream>``/``meta.rows`` (source node,
        requires={}); sequence dims concrete when `truncate` is set, else symbolic.
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

    def prepare(self) -> None:
        """Resolve the source file (VDS for wildcards), probe row counts, and build the
        sample-axis kept-index for the (per-stage) `CutSpec` (idempotent).

        With no cuts the kept-index is `None` — the identity sentinel that preserves
        the byte-identical contiguous read path (Wave-3c gate).
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
            file_rows = len(f[first.dataset])
            kept = self._build_kept_index(f, first)
        num_available = file_rows if kept is None else int(kept.size)
        if self.num > num_available:
            raise ValueError(
                f"Requested {self.num:,} rows, but only {num_available:,} are available "
                f"in {path.name!r}."
            )
        self._kept = kept
        self._num_rows = num_available if self.num < 0 else self.num
        self._resolved = path

    def _build_kept_index(self, f: h5py.File, first: GroupConfig) -> np.ndarray | None:
        """Ascending kept file-row indices from the (per-stage) row `CutSpec`; None = identity.

        Reads the cut fields off the first (sample-axis) group and evaluates the shared
        `_apply_row_cuts` engine. Returns `None` when no cuts engage — the sentinel that
        keeps the contiguous read path byte-identical.
        """
        if self.cuts is None or not self.cuts.for_split(self.stage):
            return None
        ds = f[first.dataset]
        if ds.ndim != 1:
            raise ConfigError(
                f"row cuts require the sample-axis group {self.streams[0]!r} (dataset "
                f"{first.dataset!r}) to be a 1-D scalar stream, but it has shape {ds.shape} — "
                "cut fields must be per-row scalars"
            )
        file_fields = ds.dtype.names or ()
        cut_fields = self.cuts.fields(self.stage)
        missing = [cf for cf in cut_fields if cf not in file_fields]
        if missing:
            raise SchemaError(
                f"row-cut field(s) {missing} not present in the sample-axis h5 group "
                f"{first.dataset!r} (available: {sorted(file_fields)}); cut variables must be "
                "row-axis scalar fields read at index-build"
            )
        rec = ds.fields(list(cut_fields))[:]  # (file_rows,) structured
        keep = self._apply_row_cuts(rec, self.stage)
        return np.flatnonzero(keep).astype(np.int64)

    def __len__(self) -> int:
        """Return the number of rows served (resolving the source on first call)."""
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    @property
    def source_path(self) -> Path:
        """The resolved source file (the VDS for wildcard filenames); resolves on first access."""
        self.prepare()
        assert self._resolved is not None
        return self._resolved

    @property
    def h5_source(self) -> Path:
        """This reader's h5py-openable structured source (the resolved `source_path`)."""
        return self.source_path

    def _ensure_open(self) -> None:
        """Open (or re-open) the per-process H5 handle, pid-guarded against fork."""
        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            if self._h5 is not None and self._h5.id.valid:
                self._h5.close()
            assert self._resolved is not None
            self._h5 = h5py.File(self._resolved, "r", swmr=True, libver="latest")
            self._pid = pid

    def bind(self, ctx: WorkerCtx) -> None:
        """Open the handle and allocate demand-narrowed buffers per group (dtype covers
        demanded + constituent-cut/transform fields via `get_dtype`); raises `SchemaError`
        naming the demanding module for a field absent from the live file.
        """
        self.prepare()
        self._ensure_open()
        assert self._h5 is not None
        self._dss = {}
        self._buffers = {}
        for stream, cfg in self.groups.items():
            ds = self._h5[cfg.dataset]
            demanded = dict(ctx.read_fields.get(stream, {}))
            for field in self._constituent_fields.get(stream, []):
                demanded.setdefault(field, f"{self.name} (constituent cuts)")
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

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one batch: contiguous ``read_direct`` slab (no cuts) or a filtered read
        of the kept file rows (cuts), then apply constituent cuts, truncate, transforms
        (FIT-only, seeded), and derive ``masks.<stream> = ~valid``.

        With no cuts (`_kept is None`) the path is byte-identical to a contiguous read
        and ``raw.*`` may alias the reusable buffers; with cuts the served rows map
        through the kept-index and the filtered read returns fresh (non-aliasing) arrays.
        """
        out: dict[str, np.ndarray] = {}
        file_rows = None if self._kept is None else self._kept[rows]
        for stream, cfg in self.groups.items():
            ds = self._dss[stream]
            buf = self._buffers[stream]
            if file_rows is None:
                shape = (rows.stop - rows.start, *ds.shape[1:])
                buf.resize(shape, refcheck=False)
                if buf.dtype.names:
                    ds.read_direct(buf, rows)
                batch = buf
            else:
                batch = self._read_kept(ds, buf.dtype, file_rows)
            if (cc := self.constituent_cuts.get(stream)) is not None:
                batch = cc.apply(batch)
            # None for scalar/untruncated streams -> no-op
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
                out[f"masks.{stream}"] = ~batch["valid"]  # True = padded
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out

    def _read_kept(
        self, ds: h5py.Dataset, dtype: np.dtype, file_rows: np.ndarray
    ) -> np.ndarray:
        """Fancy-read the ascending kept file rows into a fresh demand-narrowed array.

        The row-cut read path (non-contiguous): reads only the demanded fields for the
        `file_rows` selection and casts to the buffer dtype (`get_dtype` half-casts).
        Unlike the identity path's reusable-buffer ``read_direct``, this does not alias.
        """
        names = list(dtype.names or ())
        if not names:
            return np.empty((len(file_rows), *ds.shape[1:]), dtype=dtype)
        return ds.fields(names)[file_rows].astype(dtype, copy=False)

    def aliases(self, array: np.ndarray) -> bool:
        """Check whether `array` shares memory with a reusable read buffer (the
        debug-boundary check).
        """
        return any(np.may_share_memory(array, buf) for buf in self._buffers.values())

    def __getstate__(self) -> dict[str, Any]:
        """Drop live H5 state so the reader pickles under spawn contexts."""
        state = self.__dict__.copy()
        state.update({"_h5": None, "_pid": None, "_dss": {}, "_buffers": {}, "_rng": None})
        return state
