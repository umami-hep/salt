"""`EasyjetReader` — a config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples.

Pads jagged per-event branches into the same ``raw.<stream>`` / ``masks.<stream>``
structured arrays the H5 reader produces; branch names come from config.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx, _require_root_deps
from salt.core.data.stream import OffsetIndex, StreamConfig
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema, Schema

__all__ = ["EasyjetGroupConfig", "EasyjetReader"]


@dataclass(frozen=True)
class EasyjetGroupConfig:
    """Per-stream reader configuration for `EasyjetReader`.

    `branches` maps the v2 field name (the name the rest of the pipeline sees,
    e.g. ``pt``) to the ROOT branch name in the file (e.g.
    ``recojet_antikt4PFlow_pt_NOSYS``). This is where the ``_NOSYS`` suffix and
    collection-prefix variation lives — HH4b vs flavtag is a config change, not a
    code change. Field order = the config dict order.

    `jagged` declares a per-event variable-length sequence stream (``jets``,
    padded to a fixed ``T`` with a ``valid`` field + pad mask). ``jagged: false``
    is a scalar event-level stream (``[B]``, no pad mask). `truncate` keeps the
    leading N constituents of a jagged stream; None auto-resolves the file's max
    multiplicity in `prepare`.
    """

    branches: dict[str, str]
    jagged: bool = True
    truncate: int | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ConfigError("EasyjetGroupConfig: 'branches' must be a non-empty mapping")
        if self.truncate is not None and self.truncate < 1:
            raise ConfigError(f"group truncate must be >= 1, got {self.truncate}")
        if not self.jagged and self.truncate is not None:
            raise ConfigError("EasyjetGroupConfig: 'truncate' is only valid for jagged streams")


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + cumulative offsets."""

    path: Path
    n: int
    start: int  # global offset of this file's first entry
    fields: dict[str, dict[str, str]] = field(default_factory=dict)  # stream -> {field: dtype}


class EasyjetReader(Reader):
    """Config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples.

    Parameters
    ----------
    groups : Mapping[str, EasyjetGroupConfig | Mapping | ...]
        Stream name -> group config (``{branches:, jagged:, truncate:}``). The
        first jagged stream defines the served length axis. At least one group
        required.
    filename : str | Path | None, optional
        The source: a single ``.root`` file, a directory of them, or a glob. A
        directory globs to ``*.root``. May be omitted at construction and supplied
        via `with_source`.
    tree : str, optional
        The TTree name, by default ``"AnalysisMiniTree"``.
    num : int, optional
        Number of rows to serve; ``-1`` = all.

    Raises
    ------
    ConfigError
        On an empty / malformed group config, or no source file.
    SchemaError
        When a configured branch is missing from the file, or a stream has
        inconsistent branch shapes (jagged vs scalar).
    """

    def __init__(
        self,
        groups: Mapping[str, EasyjetGroupConfig | Mapping[str, Any] | Any],
        filename: str | Path | None = None,
        tree: str = "AnalysisMiniTree",
        num: int = -1,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError("EasyjetReader needs at least one group (design §6.1)")
        self.filename = str(filename) if filename is not None else None
        self.tree = str(tree)
        self.num = num
        self.groups: dict[str, EasyjetGroupConfig] = {
            stream: self._parse_group(stream, cfg) for stream, cfg in groups.items()
        }
        # no on-disk schema artifact; built in prepare() from the resolved files.
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see __getstate__)
        self._table: list[_FileEntry] | None = None
        self._offsets: OffsetIndex | None = None  # cumulative per-file row offsets
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}  # stream -> served multiplicity T
        self._read_fields: dict[str, dict[str, str]] = {}

    @staticmethod
    def _parse_group(
        stream: str, cfg: EasyjetGroupConfig | Mapping[str, Any] | Any
    ) -> EasyjetGroupConfig:
        """Normalise one group config entry to an `EasyjetGroupConfig`."""
        if isinstance(cfg, EasyjetGroupConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - {"branches", "jagged", "truncate"}
        if unknown:
            raise ConfigError(
                f"group {stream!r}: unknown config keys {sorted(unknown)} — expected "
                "branches/jagged/truncate (design §2.6)"
            )
        if "branches" not in cfg:
            raise ConfigError(
                f"group {stream!r}: 'branches' mapping is required (field -> ROOT branch name)"
            )
        return EasyjetGroupConfig(
            branches={str(k): str(v) for k, v in dict(cfg["branches"]).items()},
            jagged=bool(cfg.get("jagged", True)),
            truncate=cfg.get("truncate"),
        )

    @property
    def streams(self) -> tuple[str, ...]:
        """The configured stream names, in config order."""
        return tuple(self.groups)

    def sources(self) -> list[Path]:
        """The resolved ROOT file list (the multi-file staging surface).

        Unlike the H5 reader, an easyjet source can be a directory or a glob
        matching many ``.root`` files, so `sources` returns the full resolved
        member list. Returns ``[]`` when no source is bound or the glob is empty
        (no error here — staging an unbound reader is a no-op).
        """
        if self.filename is None:
            return []
        try:
            return self._resolve_files()
        except ConfigError:
            return []

    def restage(self, root: str | Path) -> EasyjetReader:
        """Stage all resolved ROOT files into `root`, keyed by a digest of this reader's
        source paths (isolates sibling sub-readers sharing one root from globbing each
        other's members); re-sources onto the staged file (single) or subdirectory (many).
        """
        import hashlib  # noqa: PLC0415 - opt-in staging path only

        from salt.core.data.vds import stage_file  # noqa: PLC0415 - opt-in staging path only

        root = Path(root)
        srcs = self.sources()
        if not srcs:
            return self
        # per-reader subdir keyed by the absolute source set: isolates this reader's
        # staged members from any sibling sub-reader sharing the same root.
        digest = hashlib.sha1(  # noqa: S324 - non-crypto path key, collision-safe enough
            "\n".join(sorted(str(s.resolve()) for s in srcs)).encode()
        ).hexdigest()[:16]
        dest_dir = root / digest
        staged = [stage_file(src, dest_dir / src.name) for src in srcs]
        # one file -> re-source onto the file; many -> onto the subdir (globs *.root)
        new_src = staged[0] if len(staged) == 1 else dest_dir
        return self.with_source(filename=new_src)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>``/``masks.<stream>``/``meta.rows`` (source node,
        requires={}); sequence dims concrete when `truncate` is set, else symbolic.
        """
        del mode
        flat: dict[str, TensorSpec] = {}
        for stream, cfg in self.groups.items():
            if cfg.jagged:
                t_dim: int | str = (
                    cfg.truncate if cfg.truncate is not None else sym_dim("T", stream)
                )
                flat[f"raw.{stream}"] = TensorSpec(shape=("B", t_dim), kind="data")
                flat[f"masks.{stream}"] = TensorSpec(
                    shape=("B", t_dim), dtype="bool", kind="pad_mask"
                )
            else:
                flat[f"raw.{stream}"] = TensorSpec(shape=("B",), kind="data")
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic, sorted list of ROOT files."""
        if self.filename is None:
            raise ConfigError(
                f"reader {self.name!r} has no source file — pass filename= or use "
                "with_source() (design §6.1)"
            )
        src = Path(self.filename)
        if src.is_dir():
            matches = sorted(src.glob("*.root"))
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
        """Resolve source files, probe entry counts, and build the schema (idempotent).

        Resolves each jagged stream's served multiplicity ``T`` (`truncate` or the
        file-wide max) and builds the `Schema` from the first file's branch dtypes.
        """
        if self._table is not None:
            return
        _require_root_deps("EasyjetReader", "easyjet")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        files = self._resolve_files()
        table: list[_FileEntry] = []
        offset = 0
        max_mult: dict[str, int] = {s: 0 for s, c in self.groups.items() if c.jagged}
        schema_groups: dict[str, GroupSchema] | None = None
        for path in files:
            with uproot.open(f"{path}:{self.tree}") as t:
                n = int(t.num_entries)
                avail = set(t.keys())
                entry = _FileEntry(path=path, n=n, start=offset)
                # validate branches + capture field dtypes (per file; the schema
                # is built from the FIRST file, but every file is validated).
                # Jaggedness + dtype are probed by reading the actual array (works
                # for both TTree branches and RNTuple fields — no reliance on the
                # uproot .interpretation attribute, which RField lacks).
                for stream, cfg in self.groups.items():
                    fdtypes: dict[str, str] = {}
                    for fieldname, branch in cfg.branches.items():
                        if branch not in avail:
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                                f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                            )
                        arr = t[branch].array(library="ak")
                        is_jagged = arr.ndim >= 2
                        if is_jagged != cfg.jagged:
                            kind = "jagged" if is_jagged else "scalar"
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} reads as {kind} but config "
                                f"says jagged={cfg.jagged} (group {stream!r})"
                            )
                        fdtypes[fieldname] = self._array_dtype_name(arr, is_jagged)
                        if cfg.jagged and cfg.truncate is None and n > 0:
                            max_mult[stream] = max(max_mult[stream], int(_max_count(arr)))
                    if cfg.jagged:
                        fdtypes["valid"] = "bool"
                    entry.fields[stream] = fdtypes
                if schema_groups is None:
                    schema_groups = {
                        cfg_stream: GroupSchema(fields=dict(entry.fields[cfg_stream]))
                        for cfg_stream in self.groups
                    }
            table.append(entry)
            offset += n
        del ak
        num_available = offset
        if self.num > num_available:
            raise ValueError(
                f"Requested {self.num:,} rows, but only {num_available:,} are available "
                f"across {len(files)} file(s)."
            )
        # served multiplicity per jagged stream
        for stream, cfg in self.groups.items():
            if not cfg.jagged:
                continue
            self._mult[stream] = (
                cfg.truncate if cfg.truncate is not None else max(1, max_mult[stream])
            )
        assert schema_groups is not None
        self.schema = Schema(groups=schema_groups)
        self._table = table
        self._offsets = OffsetIndex([e.n for e in table])
        self._num_rows = num_available if self.num < 0 else self.num

    @staticmethod
    def _array_dtype_name(arr: Any, is_jagged: bool) -> str:
        """Native-endian numpy dtype name for a (possibly jagged) awkward array (ROOT is
        big-endian; a jagged field uses its inner content dtype).
        """
        _require_root_deps("EasyjetReader", "easyjet")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        flat = ak.flatten(arr, axis=None) if is_jagged else arr
        dtype = np.asarray(ak.to_numpy(flat)).dtype
        return np.dtype(dtype.newbyteorder("=")).name

    def __len__(self) -> int:
        """Return the number of rows served (resolving the source on first call)."""
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
    ) -> EasyjetReader:
        """Clone onto another source (config-only, group configs shared); `vds_path` is
        accepted for API parity only (ROOT has no VDS), `stage` is ignored.
        """
        del vds_path, stage
        clone = EasyjetReader(
            groups=self.groups, filename=filename, tree=self.tree, num=num
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
                        f"group {stream!r} (configured: {sorted(cfg.branches)}) (design §2.6)"
                    )
            self._read_fields[stream] = demanded

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one batch slab, translating the global slice into per-file reads.

        A jagged stream is padded to ``T`` (structured ``(B, T)`` +
        ``masks.<stream> = ~valid``); a scalar stream is ``(B,)``.
        """
        if self._table is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))  # standalone (tests)
        start, stop = rows.start, rows.stop
        b = stop - start
        out: dict[str, np.ndarray] = {}
        for stream, cfg in self.groups.items():
            fields = self._served_fields(stream)
            cols = self._read_columns(stream, fields, start, stop)
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

    def _read_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read the demanded branches over a global row range (multi-file stitched)."""
        _require_root_deps("EasyjetReader", "easyjet")
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        assert self._table is not None
        assert self._offsets is not None
        cfg = self.groups[stream]
        branch_of = cfg.branches
        per_field_chunks: dict[str, list[Any]] = {f: [] for f in fields}
        # decompose the global slice into per-file (entry_start, entry_stop) runs
        for fidx, local_lo, local_hi in self._offsets.runs(slice(start, stop)):
            entry = self._table[fidx]
            with uproot.open(f"{entry.path}:{self.tree}") as t:
                for f in fields:
                    arr = t[branch_of[f]].array(
                        entry_start=local_lo,
                        entry_stop=local_hi,
                        library="ak",
                    )
                    per_field_chunks[f].append(arr)
        cols: dict[str, Any] = {}
        for f in fields:
            chunks = per_field_chunks[f]
            cols[f] = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return cols

    def _stream_config(self, stream: str) -> StreamConfig:
        """The `StreamConfig` for a jagged stream (resolved ``pad_max``; no cuts/sort —
        easyjet cuts are jet-level).
        """
        return StreamConfig(pad_max=self._mult[stream], jagged=True)

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad jagged columns to ``T`` via `Reader.assemble_jagged` (no cuts/sort
        configured, so the contiguous truncate+pad+valid path).
        """
        gschema = self.schema.groups[stream] if self.schema is not None else None
        return self.assemble_jagged(cols, fields, self._stream_config(stream), b, gschema)

    def _assemble_scalar(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> np.ndarray:
        """Assemble a structured ``(B,)`` array from scalar event-level columns."""
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
        state.update(
            {
                "_table": None,
                "_offsets": None,
                "_num_rows": None,
                "_mult": {},
                "_read_fields": {},
                "schema": None,
            }
        )
        return state


def _max_count(jagged: Any) -> int:
    """Max per-event multiplicity of an awkward jagged array (0 for empty)."""
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    counts = np.asarray(ak.num(jagged, axis=1))
    return int(counts.max()) if counts.size else 0
