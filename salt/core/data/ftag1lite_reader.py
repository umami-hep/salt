"""`FTAG1LiteReader` — a config-driven Reader for xAOD DAOD_FTAG1LITE POOL files.

Reads jet-keyed slices out of event-keyed CollectionTree branches via uproot,
emitting the same ``raw.<stream>`` / ``masks.<stream>`` arrays as the H5 readers.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx, _require_root_deps
from salt.core.data.cuts import CutSpec
from salt.core.data.stream import OffsetIndex, StreamConfig
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema, Schema

__all__ = ["FTAG1LiteGroupConfig", "FTAG1LiteReader"]


@dataclass(frozen=True)
class FTAG1LiteGroupConfig:
    """Per-stream reader configuration for `FTAG1LiteReader`.

    `branches` maps the v2 field name (``pt``, ``flavour_label``, ``d0``) to the
    bare aux-store branch name; the reader prepends the aux prefix
    (``<jet_collection>AuxDyn.``). Field order = config dict order.

    `jagged=False` is a jet-level scalar stream (``[event][jet]`` -> ``(B,)``, no
    pad mask). `jagged=True` is a constituent stream (``[event][jet][track]`` ->
    ``(B, pad_max)`` padded, with a ``valid`` field). `pad_max` caps the per-jet
    constituent multiplicity; None auto-resolves the file-wide max in `prepare`.
    """

    branches: dict[str, str]
    jagged: bool = True
    pad_max: int | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ConfigError("FTAG1LiteGroupConfig: 'branches' must be a non-empty mapping")
        if self.pad_max is not None and self.pad_max < 1:
            raise ConfigError(f"group pad_max must be >= 1, got {self.pad_max}")
        if not self.jagged and self.pad_max is not None:
            raise ConfigError(
                "FTAG1LiteGroupConfig: 'pad_max' is only valid for jagged (constituent) streams"
            )

    @property
    def global_object(self) -> bool:
        """Whether this is a jet-level (global, non-sequence) stream."""
        return not self.jagged


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + per-event jet structure.

    `event_start` is the global offset of this file's first event; `jet_start` is
    the global offset of this file's first kept jet. `kept_jets` are the local
    (per-file) flat-jet indices that survive the cuts, in flat order; `njets` is
    the per-event count of kept jets (length = n_events), used to map a
    jet-slice back to the covering event range.
    """

    path: Path
    n_events: int
    event_start: int
    jet_start: int
    kept_jets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    njets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    orig_njets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fields: dict[str, dict[str, str]] = field(default_factory=dict)  # stream -> {field: dtype}


class FTAG1LiteReader(Reader):
    """Config-driven Reader for xAOD DAOD_FTAG1LITE POOL files.

    Each small-R jet (``AntiKt4EMPFlowJets`` by default) is one sample. Jet-level
    ``std::vector<T>`` scalars and ``std::vector<std::vector<T>>`` constituent
    vectors are read via uproot from ``CollectionTree``, flattened
    ``[event][jet] -> jet``, and emitted as the standard ``raw.<stream>`` /
    ``masks.<stream>`` structured arrays.

    Parameters
    ----------
    groups : Mapping[str, FTAG1LiteGroupConfig | Mapping | ...]
        Stream name -> group config (``{branches:, jagged:, pad_max:}``). At least
        one non-jagged (jet-level) stream is required — it carries the served
        length axis and the cut variables. By convention it's named ``jets``.
    jet_collection : str, optional
        The jet collection whose aux store is read, by default
        ``"AntiKt4EMPFlowJets"``. The aux prefix is ``<jet_collection>AuxDyn.``.
    filename : str | Path | None, optional
        The source: a single ``.root`` (or ``.pool.root``) file, a directory of
        them, or a glob. May be omitted and supplied via `with_source`.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    num : int, optional
        Number of jets to serve (post-cut); ``-1`` = all.
    cuts : CutSpec | None, optional
        Index-build-time jet eligibility. Evaluated in `prepare` over the
        jet-level scalars; only passing jets enter the index.
    stage : str | None, optional
        The bound stage (``"train"``/``"val"``/``"test"``); selects per-split cuts.
        Set by `with_source(stage=...)`.

    Raises
    ------
    ConfigError
        On an empty / malformed group config, no jet-level stream, or no source.
    SchemaError
        When a configured branch is missing, or a stream's jaggedness disagrees
        with the config.
    """

    def __init__(
        self,
        groups: Mapping[str, FTAG1LiteGroupConfig | Mapping[str, Any] | Any],
        jet_collection: str = "AntiKt4EMPFlowJets",
        filename: str | Path | None = None,
        tree: str = "CollectionTree",
        num: int = -1,
        cuts: CutSpec | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError("FTAG1LiteReader needs at least one group (plan 19)")
        self.jet_collection = str(jet_collection)
        self.aux_prefix = f"{self.jet_collection}AuxDyn."
        self.filename = str(filename) if filename is not None else None
        self.tree = str(tree)
        self.num = num
        self.cuts = cuts
        self.stage = stage
        self.groups: dict[str, FTAG1LiteGroupConfig] = {
            stream: self._parse_group(stream, cfg) for stream, cfg in groups.items()
        }
        scalar_streams = [s for s, c in self.groups.items() if not c.jagged]
        if not scalar_streams:
            raise ConfigError(
                "FTAG1LiteReader needs at least one jet-level (jagged=False) stream to carry "
                "the served jet axis + cut variables (plan 19)"
            )
        # the first scalar stream defines the jet axis / cut scalars
        self.jet_stream = scalar_streams[0]
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see __getstate__)
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}  # stream -> served pad_max
        self._read_fields: dict[str, dict[str, str]] = {}

    @staticmethod
    def _parse_group(
        stream: str, cfg: FTAG1LiteGroupConfig | Mapping[str, Any] | Any
    ) -> FTAG1LiteGroupConfig:
        """Normalise one group config entry to a `FTAG1LiteGroupConfig`."""
        if isinstance(cfg, FTAG1LiteGroupConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - {"branches", "jagged", "pad_max"}
        if unknown:
            raise ConfigError(
                f"group {stream!r}: unknown config keys {sorted(unknown)} — expected "
                "branches/jagged/pad_max (plan 19)"
            )
        if "branches" not in cfg:
            raise ConfigError(
                f"group {stream!r}: 'branches' mapping is required (field -> bare branch name)"
            )
        return FTAG1LiteGroupConfig(
            branches={str(k): str(v) for k, v in dict(cfg["branches"]).items()},
            jagged=bool(cfg.get("jagged", True)),
            pad_max=cfg.get("pad_max"),
        )

    def _branch(self, bare: str) -> str:
        """The full aux-store branch name for a bare field branch."""
        return f"{self.aux_prefix}{bare}"

    @property
    def streams(self) -> tuple[str, ...]:
        """The configured stream names, in config order."""
        return tuple(self.groups)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>``/``masks.<stream>``/``meta.rows`` (source node,
        requires={}); sequence dims concrete when `pad_max` is set, else symbolic.
        """
        del mode
        flat: dict[str, TensorSpec] = {}
        for stream, cfg in self.groups.items():
            if cfg.jagged:
                t_dim: int | str = cfg.pad_max if cfg.pad_max is not None else sym_dim("T", stream)
                flat[f"raw.{stream}"] = TensorSpec(shape=("B", t_dim), kind="data")
                flat[f"masks.{stream}"] = TensorSpec(
                    shape=("B", t_dim), dtype="bool", kind="pad_mask"
                )
            else:
                flat[f"raw.{stream}"] = TensorSpec(shape=("B",), kind="data")
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic, sorted list of ROOT files (matches
        ``*.root`` and ``*.pool.root`` for a directory source).
        """
        if self.filename is None:
            raise ConfigError(
                f"reader {self.name!r} has no source file — pass filename= or use "
                "with_source() (plan 19)"
            )
        src = Path(self.filename)
        if src.is_dir():
            matches = sorted(set(src.glob("*.root")) | set(src.glob("*.pool.root")))
        elif any(ch in self.filename for ch in "*?["):
            matches = sorted(Path(p) for p in _glob.glob(self.filename))
        else:
            matches = [src]
        if not matches:
            raise ConfigError(
                f"reader {self.name!r}: source {self.filename!r} matched no .root files"
            )
        return matches

    def prepare(self) -> None:
        """Resolve files, build the event->jet offset index (post-cut), and the schema
        (idempotent); evaluates the (per-stage) `CutSpec` over jet-level scalars to keep
        only passing jets, then resolves each jagged stream's served ``pad_max``.
        """
        if self._table is not None:
            return
        _require_root_deps("FTAG1LiteReader", "root")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        files = self._resolve_files()
        cut_fields = self.cuts.fields() if self.cuts is not None else ()
        table: list[_FileEntry] = []
        event_offset = 0
        jet_offset = 0
        max_mult: dict[str, int] = {s: 0 for s, c in self.groups.items() if c.jagged}
        schema_groups: dict[str, GroupSchema] | None = None

        for path in files:
            with uproot.open(f"{path}:{self.tree}") as t:
                n_events = int(t.num_entries)
                avail = set(t.keys())
                entry = _FileEntry(
                    path=path, n_events=n_events, event_start=event_offset, jet_start=jet_offset
                )
                # validate branches + capture field dtypes per stream
                jet_scalar_cols: dict[str, np.ndarray] = {}
                jet_counts: np.ndarray | None = None
                for stream, cfg in self.groups.items():
                    fdtypes: dict[str, str] = {}
                    for fieldname, bare in cfg.branches.items():
                        branch = self._branch(bare)
                        if branch not in avail:
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                                f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                            )
                        arr = t[branch].array(library="ak")
                        # jet-level scalar: ndim==2 ([event][jet]); constituent: ndim==3
                        is_jagged = arr.ndim >= 3
                        if is_jagged != cfg.jagged:
                            kind = (
                                "constituent (double-jagged)" if is_jagged else "jet-level scalar"
                            )
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} reads as {kind} (ndim="
                                f"{arr.ndim}) but config says jagged={cfg.jagged} (plan 19)"
                            )
                        fdtypes[fieldname] = self._array_dtype_name(arr, is_jagged)
                        if not cfg.jagged and stream == self.jet_stream:
                            jet_scalar_cols[fieldname] = ak.to_numpy(ak.flatten(arr, axis=1))
                            if jet_counts is None:
                                jet_counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
                    if cfg.jagged:
                        fdtypes["valid"] = "bool"
                    entry.fields[stream] = fdtypes

                # also read any cut fields not already in jet_scalar_cols (they
                # MUST be jet-level branches — bare or aux-prefixed)
                for cf in cut_fields:
                    if cf in jet_scalar_cols:
                        continue
                    branch = self._branch(cf)
                    if branch not in avail:
                        raise SchemaError(
                            f"CutSpec field {cf!r} -> branch {branch!r} not in {path.name!r}; "
                            "cut variables must be jet-level scalar branches (plan 19)"
                        )
                    arr = t[branch].array(library="ak")
                    jet_scalar_cols[cf] = ak.to_numpy(ak.flatten(arr, axis=1))

            assert jet_counts is not None
            kept, per_event_kept = self._apply_cuts(jet_scalar_cols, jet_counts)
            entry.kept_jets = kept
            entry.njets = per_event_kept
            entry.orig_njets = jet_counts
            n_kept = int(kept.size)
            # resolve pad_max over KEPT jets per jagged stream
            if n_kept > 0:
                for stream, cfg in self.groups.items():
                    if cfg.jagged and cfg.pad_max is None:
                        m = self._stream_max_mult(path, stream, kept)
                        max_mult[stream] = max(max_mult[stream], m)
            if schema_groups is None:
                schema_groups = {s: GroupSchema(fields=dict(entry.fields[s])) for s in self.groups}
            entry.jet_start = jet_offset
            table.append(entry)
            event_offset += n_events
            jet_offset += n_kept

        del ak
        num_available = jet_offset
        if self.num > num_available:
            raise ValueError(
                f"Requested {self.num:,} jets, but only {num_available:,} are available "
                f"(post-cut) across {len(files)} file(s)."
            )
        for stream, cfg in self.groups.items():
            if cfg.jagged:
                self._mult[stream] = (
                    cfg.pad_max if cfg.pad_max is not None else max(1, max_mult[stream])
                )
        assert schema_groups is not None
        self.schema = Schema(groups=schema_groups)
        self._table = table
        self._num_rows = num_available if self.num < 0 else self.num

    def _apply_cuts(
        self, jet_scalar_cols: dict[str, np.ndarray], jet_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the CutSpec -> kept flat-jet indices + per-event kept counts.

        With no cuts every jet is kept; otherwise `CutSpec.eligible` gives the
        keep mask over a structured jet-scalar record built from the flat columns.
        """
        n_jets_total = int(jet_counts.sum())
        if self.cuts is None or not self.cuts.for_split(self.stage):
            return np.arange(n_jets_total, dtype=np.int64), jet_counts.copy()
        # build a structured record from the flat columns for CutSpec.eligible
        names = list(jet_scalar_cols)
        dtype = np.dtype([(nm, jet_scalar_cols[nm].dtype) for nm in names])
        rec = np.empty(n_jets_total, dtype=dtype)
        for nm in names:
            rec[nm] = jet_scalar_cols[nm]
        keep = self.cuts.eligible(rec, self.stage)  # (n_jets_total,) bool
        kept = np.flatnonzero(keep).astype(np.int64)
        # per-event kept counts: split the flat keep mask at event boundaries
        bounds = np.concatenate([[0], np.cumsum(jet_counts)])
        per_event_kept = np.add.reduceat(keep.astype(np.int64), bounds[:-1])
        # reduceat on an empty slice misbehaves for trailing zero-length events;
        # guard by zeroing events with zero original jets
        per_event_kept = np.where(jet_counts == 0, 0, per_event_kept).astype(np.int64)
        return kept, per_event_kept

    def _stream_max_mult(self, path: Path, stream: str, kept: np.ndarray) -> int:
        """Max per-jet constituent multiplicity over kept jets, from the stream's first branch."""
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        cfg = self.groups[stream]
        first_bare = next(iter(cfg.branches.values()))
        with __import__("uproot").open(f"{path}:{self.tree}") as t:
            arr = t[self._branch(first_bare)].array(library="ak")
        flat = ak.flatten(arr, axis=1)  # [jet][track]
        counts = ak.to_numpy(ak.num(flat, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

    @staticmethod
    def _array_dtype_name(arr: Any, is_jagged: bool) -> str:
        """Native-endian numpy dtype name for a (single- or double-jagged) awkward array
        (both are ragged, so fully flattened to the innermost content dtype; ROOT is
        big-endian).
        """
        _require_root_deps("FTAG1LiteReader", "root")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        del is_jagged  # both jet-level and constituent arrays are ragged here
        flat = ak.flatten(arr, axis=None)
        dtype = np.asarray(ak.to_numpy(flat)).dtype
        return np.dtype(dtype.newbyteorder("=")).name

    def __len__(self) -> int:
        """Return the number of jets served (post-cut), resolving on first call."""
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    @property
    def source_path(self) -> Path:
        """The first resolved source file."""
        self.prepare()
        assert self._table is not None
        return self._table[0].path

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The built schema's group for one served stream."""
        if self.schema is None:
            self.prepare()
        if self.schema is None or stream not in self.groups:
            return None
        return self.schema.groups.get(stream)

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing."""
        if self.schema is None:
            self.prepare()
        if self.schema is None:
            return None
        return tuple(
            f"labels.{stream}.{fieldname}"
            for stream in self.groups
            for fieldname in self.schema.groups[stream].fields
            if fieldname != "valid"
        )

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,  # accepted for API parity; ROOT has no VDS
        stage: str | None = None,
    ) -> FTAG1LiteReader:
        """Clone onto another source + stage (config-only, group/collection/cuts shared);
        `stage` selects per-split cuts at the next `prepare`; `vds_path` is API parity only.
        """
        del vds_path
        clone = FTAG1LiteReader(
            groups=self.groups,
            jet_collection=self.jet_collection,
            filename=filename,
            tree=self.tree,
            num=num,
            cuts=self.cuts,
            stage=stage,
        )
        clone.name = self.name
        return clone

    def bind(self, ctx: WorkerCtx) -> None:
        """Resolve files and record the demand-narrowed read set per stream (demanded
        fields intersected with configured branches; empty demand -> all configured).
        """
        self.prepare()
        assert self._table is not None
        self._read_fields = {}
        for stream, cfg in self.groups.items():
            demanded = dict(ctx.read_fields.get(stream, {}))
            for fieldname, who in demanded.items():
                if fieldname not in cfg.branches:
                    raise SchemaError(
                        f"field {fieldname!r} demanded by {who!r} not a configured branch in "
                        f"group {stream!r} (configured: {sorted(cfg.branches)}) (plan 19)"
                    )
            self._read_fields[stream] = demanded

    def _covering_events(self, entry: _FileEntry, jlo: int, jhi: int) -> tuple[int, int, int]:
        """Map a local kept-jet range ``[jlo, jhi)`` to its covering local event range
        ``(e0, e1)``, plus the kept-jet offset of ``e0`` for exact slicing.
        """
        cum = np.concatenate([[0], np.cumsum(entry.njets)])  # (n_events+1,)
        return OffsetIndex.covering_range(cum, jlo, jhi)

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous jet-slice, translating jets -> covering events per file
        (events flattened, kept-jet mask applied, block sliced to the requested jets).
        """
        if self._table is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))  # standalone (tests)
        start, stop = rows.start, rows.stop
        b = stop - start
        out: dict[str, np.ndarray] = {}
        for stream, cfg in self.groups.items():
            fields = self._served_fields(stream)
            # cols maps each field to a flat awkward array, depth-1 for jet-level
            # scalars and depth-2 (per-jet track lists) for constituent streams
            cols = self._read_stream_columns(stream, fields, start, stop)
            if cfg.jagged:
                raw, valid = self._assemble_jagged(stream, fields, cols, b)
                out[f"raw.{stream}"] = raw
                out[f"masks.{stream}"] = ~valid
            else:
                out[f"raw.{stream}"] = self._assemble_scalar(stream, fields, cols, b)
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([start, stop], dtype=np.int64)
        return out

    def _served_fields(self, stream: str) -> list[str]:
        """The field names served for a stream (demanded subset or all configured)."""
        cfg = self.groups[stream]
        demanded = self._read_fields.get(stream, {})
        names = set(demanded) if demanded else set(cfg.branches)
        return [f for f in cfg.branches if f in names]

    def _read_stream_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read a stream's demanded branches over a global jet range (multi-file): per
        file, reads the covering event range, flattens, applies the kept-jet mask, and
        slices to the file's contribution; blocks are concatenated in jet order.
        """
        _require_root_deps("FTAG1LiteReader", "root")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        assert self._table is not None
        cfg = self.groups[stream]
        branch_of = cfg.branches
        per_field_chunks: dict[str, list[Any]] = {f: [] for f in fields}

        for entry in self._table:
            file_jlo = entry.jet_start
            file_jhi = entry.jet_start + int(entry.kept_jets.size)
            lo = max(start, file_jlo)
            hi = min(stop, file_jhi)
            if lo >= hi:
                continue
            # local KEPT-jet range within this file
            local_jlo = lo - file_jlo
            local_jhi = hi - file_jlo
            e0, e1, jet_off = self._covering_events(entry, local_jlo, local_jhi)
            # which of THIS file's flat (pre-cut) jets are kept AND fall inside the
            # covering [e0, e1) event block, expressed relative to that block:
            orig_cum = np.concatenate([[0], np.cumsum(entry.orig_njets)])
            block_flat_start = int(orig_cum[e0])
            block_flat_stop = int(orig_cum[e1])
            kept = entry.kept_jets
            in_block = (
                kept[(kept >= block_flat_start) & (kept < block_flat_stop)] - block_flat_start
            )
            # restrict to the requested kept-jet window inside the block
            sel = in_block[jet_off : jet_off + (hi - lo)]
            with uproot.open(f"{entry.path}:{self.tree}") as t:
                for f in fields:
                    arr = t[self._branch(branch_of[f])].array(
                        entry_start=e0, entry_stop=e1, library="ak"
                    )
                    flat = ak.flatten(arr, axis=1)  # [jet] or [jet][track]
                    per_field_chunks[f].append(flat[sel])
        cols: dict[str, Any] = {}
        for f in fields:
            chunks = per_field_chunks[f]
            cols[f] = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return cols

    def _stream_config(self, stream: str) -> StreamConfig:
        """The `StreamConfig` for a constituent stream (resolved ``pad_max``; no cuts/sort —
        FTAG1LITE cuts are jet-level, evaluated at index-build).
        """
        return StreamConfig(pad_max=self._mult[stream], jagged=True)

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad constituent columns to ``pad_max`` via `Reader.assemble_jagged` (no
        per-constituent cuts/sort, so the contiguous truncate+pad+valid path).
        """
        gschema = self.schema.groups[stream] if self.schema is not None else None
        return self.assemble_jagged(cols, fields, self._stream_config(stream), b, gschema)

    def _assemble_scalar(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> np.ndarray:
        """Assemble a structured ``(B,)`` array from jet-level scalar columns."""
        gschema = self.schema.groups[stream] if self.schema is not None else None
        dtype_fields: list[tuple[str, np.dtype]] = []
        blocks: dict[str, np.ndarray] = {}
        for f in fields:
            block = np.asarray(cols[f])
            dt = np.dtype(gschema.fields[f]) if gschema is not None else block.dtype
            block = block.astype(dt, copy=False)
            blocks[f] = block
            dtype_fields.append((f, block.dtype))
        raw = np.empty((b,), dtype=np.dtype(dtype_fields))
        for f in fields:
            raw[f] = blocks[f]
        return raw

    # -- pickling (fork is free; spawn re-binds in the worker) ----------------

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient probe state so the reader pickles under spawn contexts."""
        state = self.__dict__.copy()
        state.update({
            "_table": None,
            "_num_rows": None,
            "_mult": {},
            "_read_fields": {},
            "schema": None,
        })
        return state
