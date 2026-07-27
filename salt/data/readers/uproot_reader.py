"""`UprootReader` — one config-driven uproot `Reader` for every ROOT ntuple.

A TTree is entries x branches; a group's `prefix`/`branches` name the on-disk
branches, and `unroll` picks the sample axis (tree entries, or the elements of
one jagged group). xAOD POOL, easyjet ntuples and pre-flattened per-jet ntuples
differ only in YAML.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.data.base import Reader, WorkerCtx, _require_root_deps
from salt.data.readers.cuts import GlobalObjectCuts
from salt.data.readers.stream import OffsetIndex, StreamConfig
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.schema import GroupSchema, Schema

__all__ = ["UprootGroupConfig", "UprootReader"]

_GROUP_KEYS = {
    "branches",
    "prefix",
    "jagged",
    "pad_max",
    "truncate",
    "link_branch",
    "target_prefix",
    "target_collection",
}


@dataclass(frozen=True)
class UprootGroupConfig:
    """Per-stream reader configuration for a `UprootReader`.

    Parameters
    ----------
    branches : dict[str, str]
        Maps the v2 field name (``pt``) to the *bare* on-disk branch name. Field
        order = the config dict order.
    prefix : str, optional
        Prepended to every branch of this group. Empty (default) reads branches
        verbatim (flat ntuples); an aux-store prefix (``AntiKt4EMPFlowJetsAuxDyn.``)
        addresses an xAOD collection's decorations.
    jagged : bool, optional
        ``True`` (default) is a per-row variable-length sequence stream (padded to
        a fixed ``T`` with a ``valid`` field + pad mask); ``False`` is a scalar
        per-row stream. The row axis is set by the reader's ``unroll``.
    pad_max : int | None, optional
        Keeps the leading N constituents of a jagged stream; ``None`` (default)
        auto-resolves the file's max multiplicity in `prepare`. (``truncate`` is
        accepted as an alias.)
    link_branch : str | None, optional
        Turns a jagged stream into an ElementLink-dereferenced constituent stream
        (PHYSLITE ``GhostTrack``): the per-row link vector
        ``<unroll_prefix><link_branch>`` carries ``m_persIndex`` into the target
        container. Requires the reader's ``unroll`` to name a group.
    target_prefix : str | None, optional
        The aux-store prefix of the ElementLink target container
        (``InDetTrackParticlesAuxDyn.``); this group's ``branches`` map onto it.
        Set together with ``link_branch``. (``target_collection`` is a deprecated
        alias meaning ``f"{target_collection}AuxDyn."``.)
    """

    branches: dict[str, str]
    prefix: str = ""
    jagged: bool = True
    pad_max: int | None = None
    link_branch: str | None = None
    target_prefix: str | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ConfigError("group config: 'branches' must be a non-empty mapping")
        if self.pad_max is not None and self.pad_max < 1:
            raise ConfigError(f"group pad_max must be >= 1, got {self.pad_max}")
        if not self.jagged and self.pad_max is not None:
            raise ConfigError("group config: 'pad_max' is only valid for jagged streams")
        if (self.link_branch is None) != (self.target_prefix is None):
            raise ConfigError(
                "group config: 'link_branch' and 'target_prefix' must be set together "
                "(ElementLink dereference needs both the row link vector and the target prefix)"
            )
        if self.link_branch is not None and not self.jagged:
            raise ConfigError(
                "group config: 'link_branch'/'target_prefix' are only valid for jagged "
                "(constituent) streams"
            )

    @property
    def is_linked(self) -> bool:
        """Whether this stream reads constituents via an ElementLink dereference."""
        return self.link_branch is not None


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + per-entry row structure.

    `event_start` is the global entry offset of this file; `row_start` the global
    offset of its first served row. `kept` are the local flat-row indices that
    survive the cuts (identity ``arange`` when no cuts / no unroll); `per_row_kept`
    the per-entry kept-row count (maps a row slice to its covering entry range);
    `orig_counts` the per-entry pre-cut row count (``ones`` when ``unroll is None``,
    the per-entry object count when unrolling).
    """

    path: Path
    n_events: int
    event_start: int
    row_start: int
    kept: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    per_row_kept: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    orig_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fields: dict[str, dict[str, str]] = field(default_factory=dict)  # stream -> {field: dtype}


class UprootReader(Reader):
    """uproot-backed `Reader` — one ROOT ntuple format to ``raw.<stream>`` rows.

    A row is a tree entry (``unroll=None``) or an element of one jagged group
    (``unroll=<group>``, which flattens that group's outer level and serves its
    elements). The XAOD index machinery (per-file kept mask, cumulative row
    offsets, covering-entry ranges) is the general case; ``unroll=None`` is its
    degenerate case (kept = all, covering = identity). Branch naming
    (``prefix``/``target_prefix``) and ElementLink dereference are per-group config.

    Parameters
    ----------
    groups : Mapping[str, UprootGroupConfig | Mapping]
        Stream name -> group config. At least one group is required.
    filename : str | Path | None, optional
        A single ``.root`` / ``.pool.root`` file, a directory of them, or a glob.
        May be supplied later via `with_source`.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    unroll : str | None, optional
        ``None`` (default) serves tree entries as rows; a group name (e.g.
        ``"jets"``) serves that group's collection elements as rows. The named
        group must exist and be declared ``jagged=False`` (its on-disk jaggedness
        is consumed by the unroll). Group insertion order is never semantic.
    num : int, optional
        Number of rows to serve (post-cut); ``-1`` = all.
    cuts : GlobalObjectCuts | None, optional
        Row-level eligibility evaluated in `prepare` over the row-axis scalars;
        only passing rows enter the index (available on either axis).
    constituent_cuts : Mapping[str, Any] | None, optional
        Per-stream `ConstituentCuts` (or the equivalent mapping
        ``{cuts: [...], on_fail: drop}``) applied WITHIN each jagged row by the
        shared drop-then-pad pipeline. Only ``on_fail: drop`` is implemented on
        this reader — ``mask`` is H5-first (a follow-up on the awkward path).
    stage : str | None, optional
        The bound stage (``"train"``/``"val"``/``"test"``); selects per-split cuts.

    Raises
    ------
    ConfigError
        On an empty/malformed group config; an ``unroll`` that names no group or
        names a ``jagged=True`` group; a linked group while ``unroll is None``;
        constituent cuts on a scalar stream, referencing an unconfigured branch,
        or requesting ``on_fail: mask``.
    SchemaError
        When a configured branch is missing, or a stream's jaggedness disagrees
        with the config.
    """

    def __init__(
        self,
        groups: Mapping[str, UprootGroupConfig | Mapping[str, Any] | Any],
        filename: str | Path | None = None,
        tree: str = "CollectionTree",
        unroll: str | None = None,
        num: int = -1,
        cuts: GlobalObjectCuts | None = None,
        constituent_cuts: Mapping[str, Any] | None = None,
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
        self.groups: dict[str, UprootGroupConfig] = {
            stream: self._parse_group(stream, cfg) for stream, cfg in groups.items()
        }
        self._validate_unroll()
        self.constituent_cuts = self._parse_constituent_cuts(constituent_cuts, tuple(self.groups))
        self._validate_constituent_cuts()
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see base __getstate__)
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}  # stream -> served pad_max
        self._read_fields: dict[str, dict[str, str]] = {}

    def _validate_unroll(self) -> None:
        """Enforce the unroll<->group invariants (row group jagged=False; links need unroll)."""
        if self.unroll is not None:
            if self.unroll not in self.groups:
                raise ConfigError(
                    f"unroll={self.unroll!r} names no configured group (have {sorted(self.groups)})"
                )
            if self.groups[self.unroll].jagged:
                raise ConfigError(
                    f"unroll group {self.unroll!r} must be declared jagged=False — it carries the "
                    "row axis + cut scalars; its on-disk jaggedness is consumed by the unroll"
                )
        linked = [s for s, c in self.groups.items() if c.is_linked]
        if linked and self.unroll is None:
            raise ConfigError(
                f"linked groups {linked} (link_branch/target_prefix) require unroll to name a "
                "group — ElementLink dereference is per-row-object, no meaning on the entry axis"
            )

    def _validate_constituent_cuts(self) -> None:
        """Enforce jagged-only, drop-only, configured-branch invariants for constituent cuts.

        Raises
        ------
        ConfigError
            On a scalar stream, ``on_fail: mask``, or an unconfigured cut branch.
        """
        for stream, cc in self.constituent_cuts.items():
            if not self.groups[stream].jagged:
                raise ConfigError(
                    f"constituent_cuts on stream {stream!r}, which is declared jagged=False "
                    "— constituent cuts act within a sequence row; use the reader's "
                    "row-level cuts: instead"
                )
            if cc.on_fail != "drop":
                raise ConfigError(
                    f"constituent_cuts[{stream!r}]: UprootReader implements on_fail: drop "
                    f"only, got {cc.on_fail!r} — in-place masking is H5-first"
                )
            branches = self.groups[stream].branches
            missing = [f for f in cc.fields if f not in branches]
            if missing:
                raise ConfigError(
                    f"constituent_cuts[{stream!r}]: cut field(s) {missing} are not configured "
                    f"branches of that group (configured: {sorted(branches)})"
                )

    @staticmethod
    def _parse_group(
        stream: str, cfg: UprootGroupConfig | Mapping[str, Any] | Any
    ) -> UprootGroupConfig:
        """Normalise a group config entry (``truncate``->``pad_max``,
        ``target_collection``->``target_prefix``).
        """
        if isinstance(cfg, UprootGroupConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - _GROUP_KEYS
        if unknown:
            raise ConfigError(
                f"group {stream!r}: unknown config keys {sorted(unknown)} — expected "
                "branches/prefix/jagged/pad_max/link_branch/target_prefix (truncate, "
                "target_collection accepted as aliases)"
            )
        if "branches" not in cfg:
            raise ConfigError(
                f"group {stream!r}: 'branches' mapping is required (field -> branch name)"
            )
        if "pad_max" in cfg and "truncate" in cfg:
            raise ConfigError(
                f"group {stream!r}: give either 'pad_max' or 'truncate' (its alias), not both"
            )
        if "target_prefix" in cfg and "target_collection" in cfg:
            raise ConfigError(
                f"group {stream!r}: give either 'target_prefix' or 'target_collection' (its "
                "alias), not both"
            )
        target_prefix = cfg.get("target_prefix")
        if target_prefix is None and cfg.get("target_collection") is not None:
            target_prefix = f"{cfg['target_collection']}AuxDyn."
        return UprootGroupConfig(
            branches={str(k): str(v) for k, v in dict(cfg["branches"]).items()},
            prefix=str(cfg.get("prefix", "")),
            jagged=bool(cfg.get("jagged", True)),
            pad_max=cfg.get("pad_max", cfg.get("truncate")),
            link_branch=cfg.get("link_branch"),
            target_prefix=target_prefix,
        )

    # -- branch-name resolution ----------------------------------------------

    @staticmethod
    def _on_disk(cfg: UprootGroupConfig, bare: str) -> str:
        """The on-disk branch name for a normal group field: ``prefix + bare``."""
        return f"{cfg.prefix}{bare}"

    def _link_branch(self, cfg: UprootGroupConfig) -> str:
        """The on-disk link vector of a linked group (a decoration on the unroll group)."""
        assert self.unroll is not None
        return f"{self.groups[self.unroll].prefix}{cfg.link_branch}"

    @staticmethod
    def _target_branch(cfg: UprootGroupConfig, bare: str) -> str:
        """The on-disk target-container branch for an ElementLink constituent field."""
        return f"{cfg.target_prefix}{bare}"

    # -- config-surface views -------------------------------------------------

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

    def sources(self) -> list[Path]:
        """The resolved ROOT file list (empty when unbound / the glob is empty)."""
        if self.filename is None:
            return []
        try:
            return self._resolve_files()
        except ConfigError:
            return []

    def restage(self, root: str | Path) -> UprootReader:
        """Stage all resolved ROOT files into `root` (keyed by a source-path digest) and
        re-source onto the staged file (single) or subdirectory (many).
        """
        import hashlib  # noqa: PLC0415 - opt-in staging path only

        from salt.data.readers.vds import stage_file  # noqa: PLC0415 - opt-in staging path only

        root = Path(root)
        srcs = self.sources()
        if not srcs:
            return self
        digest = hashlib.sha1(  # noqa: S324 - non-crypto path key, collision-safe enough
            "\n".join(sorted(str(s.resolve()) for s in srcs)).encode()
        ).hexdigest()[:16]
        dest_dir = root / digest
        staged = [stage_file(src, dest_dir / src.name) for src in srcs]
        new_src = staged[0] if len(staged) == 1 else dest_dir
        return self.with_source(filename=new_src)

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic sorted list of ROOT files."""
        if self.filename is None:
            raise ConfigError(
                f"reader {self.name!r} has no source file — pass filename= or use with_source()"
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

    def _array_dtype_name(self, arr: Any) -> str:
        """Native-endian numpy dtype name for a (possibly ragged) awkward array."""
        self._require_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        flat = ak.flatten(arr, axis=None)
        dtype = np.asarray(ak.to_numpy(flat)).dtype
        return np.dtype(dtype.newbyteorder("=")).name

    def _require_deps(self) -> None:
        """Raise a helpful ImportError naming the ``root`` pip extra if deps are missing."""
        _require_root_deps("UprootReader", "root")

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

    def _clone(self, filename: str | Path, num: int, stage: str | None) -> UprootReader:
        """Construct a config-identical reader onto another source."""
        return UprootReader(
            groups=self.groups,
            filename=filename,
            tree=self.tree,
            unroll=self.unroll,
            num=num,
            cuts=self.cuts,
            constituent_cuts=self.constituent_cuts,
            stage=stage,
        )

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,  # accepted for API parity; ROOT has no VDS
        stage: str | None = None,
    ) -> UprootReader:
        """Clone onto another source (config-only); `vds_path` is accepted for API parity only."""
        del vds_path
        clone = self._clone(filename=filename, num=num, stage=stage)
        clone.name = self.name
        return clone

    # -- index build (prepare) ------------------------------------------------

    def prepare(self) -> None:
        """Resolve files, build the entry->row offset index (post-cut) and the schema
        (idempotent); evaluates the (per-stage) `GlobalObjectCuts` over row-axis scalars to keep
        only passing rows, then resolves each jagged stream's served ``pad_max``.
        """
        if self._table is not None:
            return
        self._require_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        files = self._resolve_files()
        cut_fields = self.cuts.fields() if self.cuts is not None else ()
        table: list[_FileEntry] = []
        event_offset = 0
        row_offset = 0
        max_mult: dict[str, int] = {s: 0 for s, c in self.groups.items() if c.jagged}
        schema_groups: dict[str, GroupSchema] | None = None

        for path in files:
            with uproot.open(f"{path}:{self.tree}") as t:
                n_events = int(t.num_entries)
                avail = set(t.keys())
                entry = _FileEntry(
                    path=path, n_events=n_events, event_start=event_offset, row_start=row_offset
                )
                for stream, cfg in self.groups.items():
                    entry.fields[stream] = self._probe_stream_fields(t, avail, path, stream, cfg)
                row_scalars, orig_counts = self._read_index_scalars(t, avail, path, cut_fields)

            kept, per_row_kept = self._apply_cuts(row_scalars, orig_counts)
            entry.kept = kept
            entry.per_row_kept = per_row_kept
            entry.orig_counts = orig_counts
            n_kept = int(kept.size)
            if n_kept > 0:
                for stream, cfg in self.groups.items():
                    if cfg.jagged and cfg.pad_max is None:
                        m = self._stream_max_mult(path, stream, kept)
                        max_mult[stream] = max(max_mult[stream], m)
            if schema_groups is None:
                schema_groups = {s: GroupSchema(fields=dict(entry.fields[s])) for s in self.groups}
            table.append(entry)
            event_offset += n_events
            row_offset += n_kept

        del ak
        num_available = row_offset
        if self.num > num_available:
            raise ValueError(
                f"Requested {self.num:,} rows, but only {num_available:,} are available "
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

    def _probe_stream_fields(
        self, t: Any, avail: set[str], path: Path, stream: str, cfg: UprootGroupConfig
    ) -> dict[str, str]:
        """Validate a stream's branches + capture per-field dtypes (linked or direct).

        The jagged-ndim threshold is ``2 + (1 if unroll else 0)`` — an unroll adds
        one outer collection level; linked-group target columns are always
        ``[entry][obj]`` (no ndim check, jaggedness is consumed by the deref).
        """
        fdtypes: dict[str, str] = {}
        if cfg.is_linked:
            link = self._link_branch(cfg)
            if link not in avail:
                raise SchemaError(
                    f"group {stream!r}: link branch {link!r} not in {path.name!r} "
                    "(ElementLink dereference needs the row link vector)"
                )
            for fieldname, bare in cfg.branches.items():
                branch = self._target_branch(cfg, bare)
                if branch not in avail:
                    raise SchemaError(
                        f"group {stream!r}: target branch {branch!r} (field {fieldname!r}) not "
                        f"in {path.name!r}; ElementLink target fields live under "
                        f"{cfg.target_prefix!r}"
                    )
                arr = t[branch].array(library="ak")  # [entry][obj]
                fdtypes[fieldname] = self._array_dtype_name(arr)
            fdtypes["valid"] = "bool"
            return fdtypes
        threshold = 2 + (1 if self.unroll is not None else 0)
        for fieldname, bare in cfg.branches.items():
            branch = self._on_disk(cfg, bare)
            if branch not in avail:
                raise SchemaError(
                    f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                    f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                )
            arr = t[branch].array(library="ak")
            is_jagged = arr.ndim >= threshold
            if is_jagged != cfg.jagged:
                kind = "sequence (jagged)" if is_jagged else "scalar"
                raise SchemaError(
                    f"group {stream!r}: branch {branch!r} reads as {kind} (ndim="
                    f"{arr.ndim}) but config says jagged={cfg.jagged}"
                )
            fdtypes[fieldname] = self._array_dtype_name(arr)
        if cfg.jagged:
            fdtypes["valid"] = "bool"
        return fdtypes

    def _read_index_scalars(
        self, t: Any, avail: set[str], path: Path, cut_fields: tuple[str, ...]
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Read the row-axis scalar columns (for cuts) + the per-entry pre-cut row count.

        ``unroll=None``: rows are entries, so ``orig_counts`` is ``ones`` and cut
        scalars come from ``jagged=False`` streams read per entry. ``unroll=group``:
        ``orig_counts`` is the unroll group's per-entry object count, and the row
        scalars are that group's branches flattened ``[entry][obj] -> [obj]``.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        row_scalars: dict[str, np.ndarray] = {}
        need_scalars = self.cuts is not None
        if self.unroll is None:
            orig_counts = np.ones(int(t.num_entries), dtype=np.int64)
            carrier = next((c for c in self.groups.values() if not c.jagged), None)
            if need_scalars:
                for cfg in self.groups.values():
                    if cfg.jagged:
                        continue
                    for fieldname, bare in cfg.branches.items():
                        arr = t[self._on_disk(cfg, bare)].array(library="ak")
                        row_scalars.setdefault(fieldname, ak.to_numpy(arr))
        else:
            carrier = self.groups[self.unroll]
            first_bare = next(iter(carrier.branches.values()))
            counts_arr = t[self._on_disk(carrier, first_bare)].array(library="ak")
            orig_counts = ak.to_numpy(ak.num(counts_arr, axis=1)).astype(np.int64)
            if need_scalars:
                for fieldname, bare in carrier.branches.items():
                    arr = t[self._on_disk(carrier, bare)].array(library="ak")
                    row_scalars[fieldname] = ak.to_numpy(ak.flatten(arr, axis=1))

        for cf in cut_fields:
            if cf in row_scalars:
                continue
            if carrier is None:
                raise SchemaError(
                    f"row-cut field {cf!r} cannot resolve — unroll=None with no jagged=False "
                    "(row-scalar) stream to carry cut variables"
                )
            branch = self._on_disk(carrier, cf)
            if branch not in avail:
                raise SchemaError(
                    f"row-cut field {cf!r} -> branch {branch!r} not in {path.name!r}; "
                    "cut variables must be row-axis scalar branches"
                )
            arr = t[branch].array(library="ak")
            row_scalars[cf] = (
                ak.to_numpy(arr) if self.unroll is None else ak.to_numpy(ak.flatten(arr, axis=1))
            )
        return row_scalars, orig_counts

    def _apply_cuts(
        self, row_scalars: dict[str, np.ndarray], orig_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the row cuts -> kept flat-row indices + per-entry kept-row counts.

        With no cuts every row is kept; otherwise `GlobalObjectCuts.eligible` gives the keep
        mask over a structured row-scalar record built from the flat columns.
        """
        n_rows_total = int(orig_counts.sum())
        if self.cuts is None or not self.cuts.for_split(self.stage):
            return np.arange(n_rows_total, dtype=np.int64), orig_counts.copy()
        rec = self._row_record(row_scalars, n_rows_total)
        keep = self._apply_row_cuts(rec, self.stage)  # (n_rows_total,) bool
        kept = np.flatnonzero(keep).astype(np.int64)
        bounds = np.concatenate([[0], np.cumsum(orig_counts)])
        per_row_kept = np.add.reduceat(keep.astype(np.int64), bounds[:-1])
        per_row_kept = np.where(orig_counts == 0, 0, per_row_kept).astype(np.int64)
        return kept, per_row_kept

    def _stream_max_mult(self, path: Path, stream: str, kept: np.ndarray) -> int:
        """Max per-row constituent multiplicity over kept rows.

        Direct stream: per-row constituent count of the stream's first branch;
        linked stream: per-row NON-NULL link count. ``unroll`` flattens the outer
        collection level away first (``[entry][obj][const] -> [row][const]``);
        ``unroll=None`` reads per-entry constituents directly.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        cfg = self.groups[stream]
        branch = (
            self._link_branch(cfg)
            if cfg.is_linked
            else self._on_disk(cfg, next(iter(cfg.branches.values())))
        )
        with uproot.open(f"{path}:{self.tree}") as t:
            arr = t[branch].array(library="ak")
        if self.unroll is not None:
            arr = ak.flatten(arr, axis=1)  # [entry][obj][...] -> [row][...]
        if cfg.is_linked:
            pkey = _pers_key(arr)
            arr = arr if pkey is None else _pers_index(arr)[pkey != 0]
        counts = ak.to_numpy(ak.num(arr, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

    # -- read -----------------------------------------------------------------

    def bind(self, ctx: WorkerCtx) -> None:
        """Resolve files and record the demand-narrowed read set per stream (demanded
        fields intersected with configured branches; empty demand -> all configured).

        Constituent-cut fields join the demand so a cut variable is always read.
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
                        f"group {stream!r} (configured: {sorted(cfg.branches)})"
                    )
            if demanded and (cc := self.constituent_cuts.get(stream)) is not None:
                for fieldname in cc.fields:
                    demanded.setdefault(fieldname, f"{self.name} (constituent cuts)")
            self._read_fields[stream] = demanded

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous batch slab: jagged streams pad to ``T`` (``(B, T)`` +
        ``masks.<stream> = ~valid``); scalar streams are ``(B,)``.
        """
        if self._table is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))  # standalone (tests)
        start, stop = rows.start, rows.stop
        b = stop - start
        out: dict[str, np.ndarray] = {}
        for stream, cfg in self.groups.items():
            fields = self._served_fields(stream)
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
        """Read a stream's demanded branches over a global row range (multi-file): per
        file, reads the covering entry range, flattens (when unrolling), applies the
        kept-row mask and slices to the file's contribution; blocks concat in row order.
        Linked streams dereference the target container by ``m_persIndex`` per row.
        """
        self._require_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        assert self._table is not None
        cfg = self.groups[stream]
        per_field_chunks: dict[str, list[Any]] = {f: [] for f in fields}

        for entry in self._table:
            file_rlo = entry.row_start
            file_rhi = entry.row_start + int(entry.kept.size)
            lo = max(start, file_rlo)
            hi = min(stop, file_rhi)
            if lo >= hi:
                continue
            local_lo = lo - file_rlo
            local_hi = hi - file_rlo
            cum = np.concatenate([[0], np.cumsum(entry.per_row_kept)])
            e0, e1, row_off = OffsetIndex.covering_range(cum, local_lo, local_hi)
            orig_cum = np.concatenate([[0], np.cumsum(entry.orig_counts)])
            block_flat_start = int(orig_cum[e0])
            block_flat_stop = int(orig_cum[e1])
            kept = entry.kept
            in_block = (
                kept[(kept >= block_flat_start) & (kept < block_flat_stop)] - block_flat_start
            )
            sel = in_block[row_off : row_off + (hi - lo)]
            with uproot.open(f"{entry.path}:{self.tree}") as t:
                if cfg.is_linked:
                    block = self._read_linked_block(t, cfg, fields, e0, e1)
                else:
                    block = {f: self._read_col(t, cfg, f, e0, e1) for f in fields}
            for f in fields:
                per_field_chunks[f].append(block[f][sel])
        cols: dict[str, Any] = {}
        for f in fields:
            chunks = per_field_chunks[f]
            cols[f] = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return cols

    def _read_col(self, t: Any, cfg: UprootGroupConfig, f: str, e0: int, e1: int) -> Any:
        """Read one normal branch over entry block ``[e0, e1)`` as ``[row]`` / ``[row][const]``
        (flattening the outer collection level away when unrolling).
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        arr = t[self._on_disk(cfg, cfg.branches[f])].array(
            entry_start=e0, entry_stop=e1, library="ak"
        )
        return ak.flatten(arr, axis=1) if self.unroll is not None else arr

    def _read_linked_block(
        self, t: Any, cfg: UprootGroupConfig, fields: list[str], e0: int, e1: int
    ) -> dict[str, Any]:
        """Dereference an ElementLink constituent stream over entry block ``[e0, e1)``.

        Reads the per-row link vector (``m_persIndex`` into the target container),
        gathers each demanded target column by index, and returns ``[row][const]``
        awkward arrays (entries flattened away) ready for the kept-row ``sel``.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        links = t[self._link_branch(cfg)].array(
            entry_start=e0, entry_stop=e1, library="ak"
        )  # [event][obj][link] (struct m_persKey/m_persIndex, or plain int for synthetic)
        pidx = _pers_index(links)  # [event][obj][link] int (m_persIndex into the target)
        pkey = _pers_key(links)  # [event][obj][link] uint | None
        _check_single_pers_key(pkey, cfg.link_branch)  # type: ignore[arg-type]
        # drop null ElementLinks (m_persKey==0): PHYSLITE thins the target container, so
        # ghost links into thinned tracks are null and are not real constituents.
        if pkey is not None:
            pidx = pidx[pkey != 0]  # [event][obj][valid_link]
        # shift each event's local indices onto the flattened target axis (a per-event
        # scalar offset broadcast over [obj][link]), flatten events away, then
        # numpy-gather + re-impose the [obj][link] grouping (avoids nested ak fancy-index).
        block: dict[str, Any] = {}
        for f in fields:
            tgt = t[self._target_branch(cfg, cfg.branches[f])].array(
                entry_start=e0, entry_stop=e1, library="ak"
            )  # [event][track]
            counts = ak.to_numpy(ak.num(tgt, axis=1))
            offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]  # (n_events,) event starts
            global_idx = pidx + ak.Array(offsets)  # broadcast [event] over [event][obj][link]
            gidx_rows = ak.flatten(global_idx, axis=1)  # [row][link] in flat row order
            links_per_row = ak.to_numpy(ak.num(gidx_rows, axis=1))  # [row] -> #links
            gidx_1d = ak.to_numpy(ak.flatten(gidx_rows, axis=None)).astype(np.int64)
            vals_1d = np.asarray(ak.flatten(tgt, axis=1))[gidx_1d]  # 1-D gathered
            block[f] = ak.unflatten(vals_1d, links_per_row)  # [row][link]
        return block

    # -- assembly -------------------------------------------------------------

    def _stream_config(self, stream: str) -> StreamConfig:
        """The `StreamConfig` for a jagged stream: resolved ``pad_max`` + this stream's
        `ConstituentCuts` (drop-then-pad); no sort.
        """
        return StreamConfig(
            pad_max=self._mult[stream],
            jagged=True,
            cuts=self.constituent_cuts.get(stream),
        )

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad jagged columns to ``T`` via `Reader.assemble_jagged` (contiguous
        truncate+pad+valid path).
        """
        gschema = self.schema.groups[stream] if self.schema is not None else None
        return self.assemble_jagged(cols, fields, self._stream_config(stream), b, gschema)

    def _assemble_scalar(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> np.ndarray:
        """Assemble a structured ``(B,)`` array from scalar (per-row) columns."""
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


def _pers_index(links: Any) -> Any:
    """The ``m_persIndex`` (target-container index) of an ElementLink array, or the array
    itself when it is already a plain integer index (synthetic fixtures).
    """
    fields = getattr(links, "fields", []) or []
    if "m_persIndex" in fields:
        return links["m_persIndex"]
    return links


def _pers_key(links: Any) -> Any | None:
    """The ``m_persKey`` (target-container key) of an ElementLink array, or None when the
    array is a plain integer index (synthetic fixtures — no key to validate).
    """
    fields = getattr(links, "fields", []) or []
    return links["m_persKey"] if "m_persKey" in fields else None


def _check_single_pers_key(pers_key: Any | None, link_branch: str) -> None:
    """Guard: all non-null ElementLinks must point at a single target container key.

    ``m_persKey==0`` is the null-link sentinel (a thinned target) and is ignored;
    more than one distinct NON-ZERO key means the link vector spans several
    containers, which the by-index gather cannot honor — raise rather than read the
    wrong tracks.
    """
    if pers_key is None:
        return
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    flat = ak.to_numpy(ak.flatten(pers_key, axis=None))
    nonzero = np.unique(flat[flat != 0]) if flat.size else np.empty(0)
    if nonzero.size > 1:
        raise SchemaError(
            f"ElementLink branch {link_branch!r} spans multiple target containers "
            f"(m_persKey values {nonzero.tolist()[:8]}); the by-index dereference requires a "
            "single target container per file"
        )
