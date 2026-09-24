"""`UprootReader` — one config-driven uproot `Reader` for generic ROOT ntuples.

ElementLink/PHYSLITE dereference lives in `xaod_reader.xAODReader`, a subclass.
"""

from __future__ import annotations

import dataclasses
import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from salt.data.base import Reader, RowBlock, WorkerCtx, _require_root_deps
from salt.data.readers.cuts import VALID_FIELD, GlobalObjectCuts
from salt.data.readers.stream import OffsetIndex, StreamConfig
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.schema import GroupSchema, Schema

__all__ = ["UprootGroupConfig", "UprootReader"]

GROUP_KEYS = frozenset({"branches", "prefix", "jagged", "pad_max"})
XAOD_GROUP_KEYS = frozenset({
    "link_branch",
    "target_prefix",
    "join_branch",
    "join_prefix",
    "join_branches",
})
"""ElementLink group keys the base reader REJECTS, redirecting to `xAODReader`."""


def _interpretation_type(interp: Any) -> tuple[np.dtype, int] | None:
    """``(leaf dtype, ndim)`` for a flat or single-level-jagged interpretation, else None."""
    if interp is None:
        return None
    from uproot.interpretation.jagged import AsJagged
    from uproot.interpretation.numerical import AsDtype

    if isinstance(interp, AsJagged):
        content = interp.content
        if isinstance(content, AsDtype) and not content.to_dtype.shape:
            return np.dtype(content.to_dtype), 2
        return None
    if isinstance(interp, AsDtype) and not interp.to_dtype.shape:
        return np.dtype(interp.to_dtype), 1
    return None


@dataclass(frozen=True)
class UprootGroupConfig:
    """Per-stream reader configuration for a `UprootReader`.

    Parameters
    ----------
    branches : dict[str, str]
        Field name -> bare on-disk branch name, in config order.
    prefix : str, optional
        Prepended to every branch (empty = flat ntuple; else an xAOD aux-store prefix).
    jagged : bool, optional
        True (default): padded variable-length sequence + ``valid``/pad mask. False: scalar.
    pad_max : int | None, optional
        Keep the leading N constituents (pad/truncate); None (default) auto-resolves
        the file's max multiplicity in `prepare`.
    """

    branches: dict[str, str]
    prefix: str = ""
    jagged: bool = True
    pad_max: int | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ConfigError("group config: 'branches' must be a non-empty mapping")
        if self.pad_max is not None and self.pad_max < 1:
            raise ConfigError(f"group pad_max must be >= 1, got {self.pad_max}")
        if not self.jagged and self.pad_max is not None:
            raise ConfigError("group config: 'pad_max' is only valid for jagged streams")

    @property
    def served_branches(self) -> dict[str, str]:
        """Every field this group serves -> bare branch; subclasses may append."""
        return dict(self.branches)


@dataclass
class _FileEntry:
    """One file's row-index contribution: post-cut rows + their on-disk mapping."""

    path: Path
    row_start: int
    kept: np.ndarray
    per_row_kept: np.ndarray
    orig_counts: np.ndarray


class UprootReader(Reader):
    """uproot-backed `Reader` — one ROOT ntuple format to ``raw.<stream>`` rows.

    A row is a tree entry (``unroll=None``) or an element of one jagged group
    (``unroll=<group>``). ElementLink/PHYSLITE dereference is the `xAODReader` subclass.

    Parameters
    ----------
    groups : Mapping[str, UprootGroupConfig | Mapping]
        Stream name -> group config; at least one required.
    filename : str | Path | None, optional
        A ``.root``/``.pool.root`` file, a directory, or a glob; may arrive later via
        `with_source`.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    unroll : str | None, optional
        None (default) serves tree entries; a group name serves that ``jagged=False``
        group's elements.
    num : int, optional
        Rows to serve (post-cut); -1 = all.
    cuts : GlobalObjectCuts | None, optional
        Row eligibility evaluated in `prepare`, including a constituent reduction like
        ``sum(jets.valid) >= 4`` (clipped to ``pad_max``).
    stage : str | None, optional
        The bound stage (``"train"``/``"val"``/``"test"``).

    Raises
    ------
    ConfigError
        Empty/malformed group config, or a bad ``unroll``.
    SchemaError
        A configured branch missing from the FIRST file.
    """

    _group_keys: ClassVar[frozenset[str]] = GROUP_KEYS
    _group_config_cls: ClassVar[type[UprootGroupConfig]] = UprootGroupConfig

    def __init__(
        self,
        groups: Mapping[str, UprootGroupConfig | Mapping[str, Any] | Any],
        filename: str | Path | None = None,
        tree: str = "CollectionTree",
        unroll: str | None = None,
        num: int = -1,
        cuts: GlobalObjectCuts | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError("UprootReader needs at least one group")
        self.filename = str(filename) if filename is not None else None
        self.tree = str(tree)
        self.unroll = str(unroll) if unroll is not None else None
        self.num = num
        self.cuts = cuts
        self.stage = stage
        self.groups: Mapping[str, UprootGroupConfig] = {
            s: self._parse_group(s, cfg) for s, cfg in groups.items()
        }
        if self.unroll is not None:
            if self.unroll not in self.groups:
                raise ConfigError(f"unroll={self.unroll!r} names no configured group")
            if self.groups[self.unroll].jagged:
                raise ConfigError(
                    f"unroll group {self.unroll!r} must be declared jagged=False — its "
                    "on-disk jaggedness is consumed by the unroll"
                )
        self._validate_links()
        self.schema: Schema | None = None
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}
        self._read_fields: dict[str, dict[str, str]] = {}
        self._open_trees: dict[str, Any] = {}

    def _validate_links(self) -> None:
        return

    @classmethod
    def _parse_group(
        cls, stream: str, cfg: UprootGroupConfig | Mapping[str, Any] | Any
    ) -> UprootGroupConfig:
        """Normalise a group config entry (a subclass promotes a plain base instance)."""
        if isinstance(cfg, UprootGroupConfig):
            if type(cfg) is cls._group_config_cls:
                return cfg
            if issubclass(cls._group_config_cls, type(cfg)):
                return cls._group_config_cls(**dataclasses.asdict(cfg))
            raise ConfigError(
                f"group {stream!r}: {type(cfg).__name__} is not a "
                f"{cls._group_config_cls.__name__} — use class_path: salt.data.xAODReader"
            )
        cfg = dict(cfg or {})
        unknown = set(cfg) - cls._group_keys
        xaod = sorted(unknown & XAOD_GROUP_KEYS)
        if xaod:
            raise ConfigError(
                f"group {stream!r}: {xaod} are ElementLink keys — use "
                "class_path: salt.data.xAODReader for ElementLink/PHYSLITE groups"
            )
        if unknown:
            raise ConfigError(f"group {stream!r}: unknown config keys {sorted(unknown)}")
        if "branches" not in cfg:
            raise ConfigError(f"group {stream!r}: 'branches' mapping is required")
        return cls._group_config_cls(**cls._group_kwargs(cfg))

    @classmethod
    def _group_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "branches": {str(k): str(v) for k, v in dict(cfg["branches"]).items()},
            "prefix": str(cfg.get("prefix", "")),
            "jagged": bool(cfg.get("jagged", True)),
            "pad_max": cfg.get("pad_max"),
        }

    @staticmethod
    def _on_disk(cfg: UprootGroupConfig, bare: str) -> str:
        return f"{cfg.prefix}{bare}"

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(self.groups)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat: dict[str, TensorSpec] = {}
        for stream, cfg in self.groups.items():
            if not cfg.jagged:
                flat[f"raw.{stream}"] = TensorSpec(shape=("B",), kind="data")
                continue
            t_dim: int | str = cfg.pad_max if cfg.pad_max is not None else sym_dim("T", stream)
            flat[f"raw.{stream}"] = TensorSpec(shape=("B", t_dim), kind="data")
            flat[f"masks.{stream}"] = TensorSpec(shape=("B", t_dim), dtype="bool", kind="pad_mask")
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    def sources(self) -> list[Path]:
        if self.filename is None:
            return []
        try:
            return self._resolve_files()
        except ConfigError:
            return []

    def restage(self, root: str | Path) -> Self:
        """Stage all resolved ROOT files into `root` (keyed by a source-path digest)."""
        import hashlib

        from salt.data.readers.stream import stage_file

        root, srcs = Path(root), self.sources()
        if not srcs:
            return self
        digest = hashlib.sha1(  # noqa: S324 - non-crypto path key, collision-safe enough
            "\n".join(sorted(str(s.resolve()) for s in srcs)).encode()
        ).hexdigest()[:16]
        dest_dir = root / digest
        staged = [stage_file(src, dest_dir / src.name) for src in srcs]
        return self.with_source(filename=staged[0] if len(staged) == 1 else dest_dir)

    def _resolve_files(self) -> list[Path]:
        if self.filename is None:
            raise ConfigError(f"reader {self.name!r} has no source file — pass filename=")
        src = Path(self.filename)
        if src.is_dir():
            matches = sorted(set(src.glob("*.root")) | set(src.glob("*.pool.root")))
        elif any(ch in self.filename for ch in "*?["):
            matches = sorted(Path(p) for p in _glob.glob(self.filename))
        else:
            matches = [src]
        if not matches:
            raise ConfigError(f"reader {self.name!r}: source {self.filename!r} matched no files")
        return matches

    def __len__(self) -> int:
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    @property
    def source_path(self) -> Path:
        self.prepare()
        assert self._table is not None
        return self._table[0].path

    def schema_group(self, stream: str) -> GroupSchema | None:
        if self.schema is None:
            self.prepare()
        if self.schema is None or stream not in self.groups:
            return None
        return self.schema.groups.get(stream)

    def label_universe(self) -> tuple[str, ...] | None:
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

    def with_source(self, filename: str | Path, num: int = -1, stage: str | None = None) -> Self:
        """Clone onto another source (config-only)."""
        clone = type(self)(self.groups, filename, self.tree, self.unroll, num, self.cuts, stage)
        clone.name = self.name
        return clone

    def config_fingerprint(self) -> dict[str, Any]:
        return {
            "reader": f"{type(self).__module__}.{type(self).__qualname__}",
            "tree": self.tree,
            "unroll": self.unroll,
            "num": self.num,
            "stage": self.stage,
            "cuts": repr(self.cuts),
            "groups": {s: dataclasses.asdict(cfg) for s, cfg in self.groups.items()},
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update({
            "_table": None,
            "_num_rows": None,
            "_mult": {},
            "_read_fields": {},
            "_open_trees": {},
            "schema": None,
        })
        return state

    def prepare(self) -> None:
        """Build the entry -> row-offset index (post-cut) and the schema. Idempotent."""
        if self._table is not None:
            return
        _require_root_deps(type(self).__name__, "root")
        import uproot

        files = self._resolve_files()
        cut_fields = self.cuts.fields() if self.cuts is not None else ()
        cut_aggs = self.cuts.aggregations() if self.cuts is not None else ()
        table: list[_FileEntry] = []
        row_offset = 0
        max_mult = {s: 0 for s, c in self.groups.items() if c.jagged}
        schema_groups: dict[str, GroupSchema] = {}
        for path in files:
            with uproot.open(f"{path}:{self.tree}") as t:
                n, avail = int(t.num_entries), set(t.keys())
                if not schema_groups:
                    schema_groups = {
                        s: GroupSchema(fields=self._stream_dtypes(t, avail, path, s, cfg, n))
                        for s, cfg in self.groups.items()
                    }
                kept, per_row_kept, orig_counts = self._index_file(
                    t, avail, path, cut_fields, cut_aggs
                )
                if kept.size > 0:
                    for s, cfg in self.groups.items():
                        if cfg.jagged and cfg.pad_max is None:
                            max_mult[s] = max(max_mult[s], self._stream_max_mult(t, s, kept))
                table.append(_FileEntry(path, row_offset, kept, per_row_kept, orig_counts))
                row_offset += int(kept.size)
        available = row_offset
        if self.num > available:
            raise ValueError(
                f"Requested {self.num:,} rows, but only {available:,} are available "
                f"(post-cut) across {len(files)} file(s)."
            )
        for s, cfg in self.groups.items():
            if cfg.jagged:
                self._mult[s] = cfg.pad_max if cfg.pad_max is not None else max(1, max_mult[s])
        self.schema = Schema(groups=schema_groups)
        self._table = table
        self._num_rows = available if self.num < 0 else self.num

    def _stream_dtypes(
        self, t: Any, avail: set[str], path: Path, stream: str, cfg: Any, n: int
    ) -> dict[str, str]:
        """Presence-check a stream's branches + capture per-field dtypes; ``valid`` for jagged."""
        fdtypes: dict[str, str] = {}
        for fieldname, bare in cfg.branches.items():
            branch = self._on_disk(cfg, bare)
            if branch not in avail:
                raise SchemaError(
                    f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                    f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                )
            fdtypes[fieldname] = self._branch_dtype(t, branch, n)
        if cfg.jagged:
            fdtypes["valid"] = "bool"
        return fdtypes

    def _branch_dtype(self, t: Any, branch: str, n_events: int) -> str:
        """A branch's native-endian numpy dtype, from metadata or a bounded one-entry read."""
        from_meta = _interpretation_type(getattr(t[branch], "interpretation", None))
        if from_meta is not None:
            dtype = from_meta[0]
        else:
            import awkward as ak

            arr = t[branch].array(entry_stop=min(1, n_events), library="ak")
            if "unknown" in str(arr.type):
                arr = t[branch].array(library="ak")
            dtype = np.asarray(ak.to_numpy(ak.flatten(arr, axis=None))).dtype
        return np.dtype(dtype.newbyteorder("=")).name

    def _index_file(
        self, t: Any, avail: set[str], path: Path, cut_fields: tuple, cut_aggs: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-file row index: kept flat-row indices, per-entry kept + orig row counts."""
        import awkward as ak

        row_scalars: dict[str, np.ndarray] = {}
        need = self.cuts is not None
        if self.unroll is None:
            orig_counts = np.ones(int(t.num_entries), dtype=np.int64)
            carrier = next((c for c in self.groups.values() if not c.jagged), None)
            carriers = [c for c in self.groups.values() if not c.jagged] if need else []
        else:
            carrier = self.groups[self.unroll]
            first = next(iter(carrier.branches.values()))
            counts_arr = t[self._on_disk(carrier, first)].array(library="ak")
            orig_counts = ak.to_numpy(ak.num(counts_arr, axis=1)).astype(np.int64)
            carriers = [carrier] if need else []
        for cfg in carriers:
            for fieldname, bare in cfg.branches.items():
                arr = t[self._on_disk(cfg, bare)].array(library="ak")
                arr = ak.flatten(arr, axis=1) if self.unroll is not None else arr
                row_scalars.setdefault(fieldname, ak.to_numpy(arr))
        for cf in cut_fields:
            if cf in row_scalars:
                continue
            if carrier is None:
                raise SchemaError(f"row-cut field {cf!r} cannot resolve — no row-scalar carrier")
            branch = self._on_disk(carrier, cf)
            if branch not in avail:
                raise SchemaError(f"row-cut field {cf!r} -> branch {branch!r} not in {path.name!r}")
            arr = t[branch].array(library="ak")
            row_scalars[cf] = (
                ak.to_numpy(arr) if self.unroll is None else ak.to_numpy(ak.flatten(arr, axis=1))
            )
        for agg in cut_aggs:
            cfg = self.groups[agg.stream]
            needed = list(dict.fromkeys(f for f in agg.fields if f != VALID_FIELD)) or [
                next(iter(cfg.branches))
            ]
            n = int(t.num_entries)
            raw = self._read_branches(t, list(self._direct_branches(cfg, needed).values()), 0, n)
            shared = {
                b: (ak.flatten(a, axis=1) if self.unroll is not None else a) for b, a in raw.items()
            }
            cols = self._group_block(t, cfg, needed, 0, n, shared)
            if cfg.pad_max is not None:
                cols = {f: c[:, : cfg.pad_max] for f, c in cols.items()}
            if VALID_FIELD in agg.fields:
                cols[VALID_FIELD] = ak.full_like(cols[needed[0]], True, dtype=bool)
            row_scalars[agg.key] = agg.evaluate(cols)
        n_rows = int(orig_counts.sum())
        if self.cuts is None or not self.cuts.global_cuts:
            return np.arange(n_rows, dtype=np.int64), orig_counts.copy(), orig_counts
        rec = self._row_record(row_scalars, n_rows)
        keep = self._apply_row_cuts(rec)
        kept = np.flatnonzero(keep).astype(np.int64)
        bounds = np.concatenate([[0], np.cumsum(orig_counts)])
        per_row_kept = np.add.reduceat(keep.astype(np.int64), bounds[:-1])
        per_row_kept = np.where(orig_counts == 0, 0, per_row_kept).astype(np.int64)
        return kept, per_row_kept, orig_counts

    def _stream_max_mult(self, t: Any, stream: str, kept: np.ndarray) -> int:
        """Max per-row constituent multiplicity over kept rows, off the OPEN tree."""
        import awkward as ak

        cfg = self.groups[stream]
        branch = self._on_disk(cfg, next(iter(cfg.branches.values())))
        arr = t[branch].array(library="ak")
        if self.unroll is not None:
            arr = ak.flatten(arr, axis=1)
        counts = ak.to_numpy(ak.num(arr, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

    def bind(self, ctx: WorkerCtx) -> None:
        """Resolve files and record the demand-narrowed read set per stream."""
        self.prepare()
        self._read_fields = {}
        for stream, cfg in self.groups.items():
            demanded = dict(ctx.read_fields.get(stream, {}))
            for fieldname, who in demanded.items():
                if fieldname not in cfg.served_branches:
                    raise SchemaError(
                        f"field {fieldname!r} demanded by {who!r} not a configured branch in "
                        f"group {stream!r} (configured: {sorted(cfg.served_branches)})"
                    )
            self._read_fields[stream] = demanded

    def _served_fields(self, stream: str) -> list[str]:
        cfg = self.groups[stream]
        demanded = self._read_fields.get(stream, {})
        names = set(demanded) if demanded else set(cfg.served_branches)
        return [f for f in cfg.served_branches if f in names]

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Jagged streams pad to ``T`` (+ ``masks.<stream> = ~valid``); scalars are ``(B,)``."""
        import awkward as ak

        if self._table is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))
        assert self._table is not None  # bind() -> prepare() fills _table or raises
        start, stop = rows.start, rows.stop
        b = stop - start
        wanted = {s: self._served_fields(s) for s in self.groups}
        direct = {s: self._direct_branches(self.groups[s], fs) for s, fs in wanted.items()}
        union = [br for names in direct.values() for br in names.values()]
        chunks: dict[str, dict[str, list[Any]]] = {
            s: {f: [] for f in fs} for s, fs in wanted.items()
        }
        for entry in self._table:
            file_lo, file_hi = entry.row_start, entry.row_start + int(entry.kept.size)
            lo, hi = max(start, file_lo), min(stop, file_hi)
            if lo >= hi:
                continue
            cum = np.concatenate([[0], np.cumsum(entry.per_row_kept)])
            e0, e1, row_off = OffsetIndex.covering_range(cum, lo - file_lo, hi - file_lo)
            orig_cum = np.concatenate([[0], np.cumsum(entry.orig_counts)])
            in_block = (
                entry.kept[(entry.kept >= orig_cum[e0]) & (entry.kept < orig_cum[e1])]
                - orig_cum[e0]
            )
            sel = in_block[row_off : row_off + (hi - lo)]
            t = self._tree(entry.path)
            raw = self._read_branches(t, union, e0, e1) if union else {}
            shared = {
                br: (ak.flatten(a, axis=1) if self.unroll is not None else a)
                for br, a in raw.items()
            }
            for s, fields in wanted.items():
                block = self._group_block(t, self.groups[s], fields, e0, e1, shared)
                for f in fields:
                    chunks[s][f].append(block[f][sel])
        out: dict[str, np.ndarray] = {}
        for stream, cfg in self.groups.items():
            fields = wanted[stream]
            cols = {
                f: (
                    ak.concatenate(chunks[stream][f])
                    if len(chunks[stream][f]) > 1
                    else chunks[stream][f][0]
                )
                for f in fields
            }
            gschema = self.schema.groups[stream] if self.schema is not None else None
            if cfg.jagged:
                stream_cfg = StreamConfig(pad_max=self._mult[stream], jagged=True)
                raw_arr, valid = self.assemble_jagged(cols, fields, stream_cfg, b, gschema)
                out[f"raw.{stream}"] = raw_arr
                out[f"masks.{stream}"] = ~valid
            else:
                sdt = None if gschema is None else gschema.fields
                dtf = [
                    (f, np.dtype(sdt[f]) if sdt is not None else np.asarray(cols[f]).dtype)
                    for f in fields
                ]
                raw = np.empty((b,), dtype=np.dtype(dtf))
                for f, dt in dtf:
                    raw[f] = np.asarray(cols[f]).astype(dt, copy=False)
                out[f"raw.{stream}"] = raw
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([start, stop], dtype=np.int64)
        return out

    def row_blocks(self) -> list[RowBlock]:
        """One block per file (files fully cut are omitted); tiles ``[0, len(self))``."""
        n_rows = len(self)
        assert self._table is not None
        blocks = []
        for entry in self._table:
            start = entry.row_start
            stop = min(start + int(entry.kept.size), n_rows)
            if stop > start:
                blocks.append(RowBlock(group=0, start=start, stop=stop))
            if stop >= n_rows:
                break
        return blocks

    def _tree(self, path: str | Path) -> Any:
        """The open TTree for `path`, cached per process — the HOT read path only."""
        import uproot

        key = str(path)
        tree = self._open_trees.get(key)
        if tree is None:
            tree = uproot.open(f"{key}:{self.tree}")
            self._open_trees[key] = tree
        return tree

    def _direct_branches(self, cfg: Any, fields: list[str]) -> dict[str, str]:
        return {f: self._on_disk(cfg, cfg.branches[f]) for f in fields}

    def _group_block(
        self, t: Any, cfg: Any, fields: list[str], e0: int, e1: int, shared: dict[str, Any]
    ) -> dict[str, Any]:
        """Base: pure selection out of `shared` (on-disk-branch -> unrolled array)."""
        del t, e0, e1
        return {f: shared[self._on_disk(cfg, cfg.branches[f])] for f in fields}

    def _read_branches(self, t: Any, branches: list[str], e0: int, e1: int) -> dict[str, Any]:
        """Read several branches over entry block ``[e0, e1)`` in ONE grouped call."""
        if not branches:
            return {}
        unique = list(dict.fromkeys(branches))
        got = t.arrays(unique, entry_start=e0, entry_stop=e1, library="ak", how=dict)
        missing = [b for b in unique if b not in got]
        if missing:
            raise SchemaError(
                f"grouped read of entries [{e0}, {e1}) did not return branch(es) {missing} "
                f"— asked for {len(unique)}, got {sorted(got)[:8]}..."
            )
        return {b: got[b] for b in unique}
