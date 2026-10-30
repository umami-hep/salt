"""`FTAG1LITEReader` (DAOD_FTAG1LITE small-R and/or large-R jets on plain `UprootReader`s) and
its container streaming: ``per_rank`` / ``owner`` scopes, `PoolSchedule` / `WindowSchedule`.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.iterable_dataset import IterableSaltDataset, _GroupCursor
from salt.data.readers import ftag1lite_categories as categories
from salt.data.readers.cuts import GlobalObjectCuts
from salt.data.readers.stream import covering_range, stage_file
from salt.data.readers.uproot_reader import UprootReader
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.planner import Sinks
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema
from salt.utils.logging import get_logger

__all__ = [
    "FTAG1LITEReader",
    "FTAG1LITEStreamingDataset",
    "PoolSchedule",
    "WindowSchedule",
    "shard_assignment",
]

JETS = ("small_r", "large_r", "both")
_SIDES_OF = {"small_r": ("small",), "large_r": ("large",), "both": ("small", "large")}
_SIDE_KEYS = frozenset({"groups", "cuts"})
_READER_OWNED = ("filename", "tree", "unroll", "num", "stage")
_STREAMING = ("shard_scope", "shuffle_pool_rows")
_SETTINGS = ("jets", "tree", "unroll", "jet_type_field", "small_type", "large_type", *_STREAMING)
_FINGERPRINTED = ("jets", "jet_type_field", "small_type", "large_type", "num", "stage")
SCOPES = ("per_rank", "owner")
_CODES = frozenset(categories.CATEGORY_FIELDS)
_Parts = dict[str, dict[str, np.ndarray]]
_LOG = get_logger(__name__)


@dataclass
class _FileEntry:
    """One file; per side, cumulative kept rows per tree entry and its own first-row index."""

    path: Path
    row_start: int
    n_rows: int
    cum: dict[str, np.ndarray] = field(default_factory=dict)
    side_start: dict[str, int] = field(default_factory=dict)
    codes: dict[str, int] = field(default_factory=dict)


class FTAG1LITEReader(Reader):
    """DAOD_FTAG1LITE reader over small-R (AntiKt4EMPFlow) and/or large-R (AntiKt10UFO…) jets.

    `jets` selects the collection(s); each active side is a plain `UprootReader` with its
    own ``groups``/``cuts`` over the SAME files. ``both`` serves, per file and per tree
    entry (event) in order, that event's kept small-R rows then its kept large-R rows,
    with an int64 `jet_type_field` (`small_type`/`large_type`) appended to the `unroll`
    stream. Every mode serves the int64 container codes ``subcampaign``/``shower``/``sim_type``
    on `unroll` when demanded (parsed at `bind`, per file, from its directory name).
    Union contract (``both``): scalar streams serve the union of both sides'
    fields; a one-sided field is filled on the other side's rows (floating -> NaN, signed
    integer -> -1, any other dtype is a `ConfigError`); a dtype disagreement is promoted
    (`numpy.promote_types`). A jagged stream on both sides must agree on ``pad_max``,
    field names and on-disk dtypes; a one-sided jagged stream is served all-padding
    (zeros, ``masks`` True) on the other side's rows. Every jagged stream must set
    ``pad_max`` (all modes: batches are fixed ``(B, pad_max)``); `num` caps served rows in
    every mode (``len == min(total, num)``).

    Parameters
    ----------
    jets : str, optional
        ``"small_r"`` (default), ``"large_r"`` or ``"both"``.
    small, large : Mapping[str, Any] | None, optional
        Per ACTIVE collection, only ``groups`` and ``cuts`` (a `GlobalObjectCuts` or its
        mapping form); an inactive collection takes None.
    filename : str | Path | None, optional
        The FTAG1LITE source (file, directory or glob), shared by every side.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    unroll : str | None, optional
        The row-carrying scalar group, by default ``"jets"`` (required in ``both``).
    num : int, optional
        Rows to serve; -1 (default) = all. The sides read uncapped.
    stage : str | None, optional
        The bound stage; selects every side's ``per_split`` cuts.
    jet_type_field : str, optional
        The int64 jet-type column injected in ``both``, by default ``"jet_type"``.
    small_type, large_type : int, optional
        The `jet_type_field` values of small-R / large-R rows, by default 0 / 1.
    shard_scope : str, optional
        Streaming FIT scope over the containers (file parent dirs): ``"per_rank"`` or ``"owner"``.
    shuffle_pool_rows : int | None, optional
        Streaming shuffle-pool rows, by default 100,000; null = one batch (``owner``: null).

    Raises
    ------
    ConfigError
        A bad `jets`, side mapping or side key, a broken stream contract, or a bad container.
    """

    def __init__(
        self,
        jets: str = "small_r",
        small: Mapping[str, Any] | None = None,
        large: Mapping[str, Any] | None = None,
        filename: str | Path | None = None,
        tree: str = "CollectionTree",
        unroll: str | None = "jets",
        num: int = -1,
        stage: str | None = None,
        jet_type_field: str = "jet_type",
        small_type: int = 0,
        large_type: int = 1,
        shard_scope: str = "per_rank",
        shuffle_pool_rows: int | None = 100_000,
    ) -> None:
        super().__init__()
        if jets not in JETS:
            raise ConfigError(f"FTAG1LITEReader: jets={jets!r} — expected one of {list(JETS)}")
        if shard_scope not in {"per_rank", "owner"}:
            raise ConfigError(f"FTAG1LITEReader: shard_scope={shard_scope!r} — per_rank or owner")
        if shuffle_pool_rows is not None and shuffle_pool_rows < 1:
            raise ConfigError(f"shuffle_pool_rows must be null or >= 1, got {shuffle_pool_rows}")
        self._cfgs: dict[str, dict[str, Any]] = {}
        for side, cfg in (("small", small), ("large", large)):
            if side in _SIDES_OF[jets]:
                self._cfgs[side] = self._check_side_cfg(jets, side, cfg)
            elif cfg is not None:
                raise ConfigError(f"FTAG1LITEReader: jets={jets!r} takes no {side!r} mapping")
        self.jets = jets
        self.filename = str(filename) if filename is not None else None
        self.tree = str(tree)
        self.unroll = str(unroll) if unroll is not None else None
        self.num = int(num)
        self.stage = stage
        self.jet_type_field = str(jet_type_field)
        self.small_type = int(small_type)
        self.large_type = int(large_type)
        self.shard_scope = shard_scope
        self.shuffle_pool_rows = None if shuffle_pool_rows is None else int(shuffle_pool_rows)
        self.cuts = None
        self.sides: dict[str, UprootReader] = {
            side: self._make_side(side, cfg) for side, cfg in self._cfgs.items()
        }
        self._both = len(self.sides) == 2
        self._validate_config()
        self.schema: Schema | None = None
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._fill: dict[str, dict[str, Any]] = {}
        self._demand: dict[str, dict[str, str]] | None = None
        self._served: dict[str, list[str]] | None = None
        self._dtypes: dict[str, np.dtype] = {}

    @staticmethod
    def _check_side_cfg(jets: str, side: str, cfg: Mapping[str, Any] | None) -> dict[str, Any]:
        if cfg is None:
            raise ConfigError(f"FTAG1LITEReader: jets={jets!r} needs a {side!r} mapping")
        if owned := sorted(k for k in cfg if k in _READER_OWNED):
            raise ConfigError(
                f"FTAG1LITEReader: {side!r} sets reader-owned key(s) {owned} — set them once on "
                f"the reader; a side mapping carries only {sorted(_SIDE_KEYS)}"
            )
        if unknown := sorted(set(cfg) - _SIDE_KEYS):
            raise ConfigError(f"FTAG1LITEReader: {side!r} has unknown key(s) {unknown}")
        if "groups" not in cfg:
            raise ConfigError(f"FTAG1LITEReader: {side!r} needs a 'groups' mapping")
        return dict(cfg)

    def _make_side(self, side: str, cfg: Mapping[str, Any]) -> UprootReader:
        """One side's uncapped `UprootReader` with the reader-owned settings applied."""
        cuts = cfg.get("cuts")
        if isinstance(cuts, Mapping):
            cuts = GlobalObjectCuts(**dict(cuts))
        r = UprootReader(cfg["groups"], self.filename, self.tree, self.unroll, -1, cuts, self.stage)
        r.name = side
        return r

    def _validate_config(self) -> None:
        for side, reader in self.sides.items():
            for stream, cfg in reader.groups.items():
                if cfg.jagged and cfg.pad_max is None:
                    raise ConfigError(f"jagged stream {stream!r} of {side!r} must set pad_max")
                if hit := sorted(self._special(stream) & set(cfg.served_branches)):
                    raise ConfigError(
                        f"FTAG1LITEReader: injected field(s) {hit} collide with a configured "
                        f"field of {side!r} stream {stream!r}"
                    )
        if self._both and self.jet_type_field in _CODES:
            raise ConfigError(f"jet_type_field={self.jet_type_field!r} is a container code")
        if not self._both:
            return
        if self.unroll is None:
            raise ConfigError("FTAG1LITEReader: jets='both' needs unroll to carry the jet type")
        if self.small_type == self.large_type:
            raise ConfigError(f"small_type and large_type must differ (both {self.small_type})")
        small, large = self.sides["small"].groups, self.sides["large"].groups
        for stream in (s for s in small if s in large):
            s_cfg, l_cfg = small[stream], large[stream]
            if s_cfg.jagged != l_cfg.jagged:
                raise ConfigError(f"FTAG1LITEReader: stream {stream!r} jaggedness differs by side")
            if s_cfg.jagged and s_cfg.pad_max != l_cfg.pad_max:
                raise ConfigError(
                    f"jagged stream {stream!r} pad_max differs: {s_cfg.pad_max} vs {l_cfg.pad_max}"
                )
            s_f, l_f = set(s_cfg.served_branches), set(l_cfg.served_branches)
            if s_cfg.jagged and s_f != l_f:
                raise ConfigError(
                    f"FTAG1LITEReader: jagged stream {stream!r} serves different fields — "
                    f"small-only {sorted(s_f - l_f)}, large-only {sorted(l_f - s_f)}"
                )

    @property
    def streams(self) -> tuple[str, ...]:
        """The first side's streams in config order, then the other side's extra streams."""
        return tuple(dict.fromkeys(s for r in self.sides.values() for s in r.groups))

    def _group_cfg(self, stream: str) -> Any:
        return next(r.groups[stream] for r in self.sides.values() if stream in r.groups)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat: dict[str, TensorSpec] = {}
        for stream in self.streams:
            cfg = self._group_cfg(stream)
            if not cfg.jagged:
                flat[f"raw.{stream}"] = TensorSpec(shape=("B",), kind="data")
                continue
            t_dim = cfg.pad_max
            flat[f"raw.{stream}"] = TensorSpec(shape=("B", t_dim), kind="data")
            flat[f"masks.{stream}"] = TensorSpec(shape=("B", t_dim), dtype="bool", kind="pad_mask")
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    def sources(self) -> list[Path]:
        return next(iter(self.sides.values())).sources()

    @property
    def group_names(self) -> list[str]:
        """Container names: each source file's parent directory, ids by first appearance."""
        return list(dict.fromkeys(p.parent.name for p in self.sources()))

    @property
    def iterable_dataset_class(self) -> type:
        """The streaming dataset `SaltDataModule` builds over this reader."""
        return FTAG1LITEStreamingDataset

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
        return self.schema.groups.get(stream) if self.schema is not None else None

    def label_universe(self) -> tuple[str, ...] | None:
        if self.schema is None:
            self.prepare()
        assert self.schema is not None
        return tuple(
            f"labels.{stream}.{fieldname}"
            for stream in self.streams
            if stream in self.schema.groups
            for fieldname in self.schema.groups[stream].fields
            if fieldname != "valid"
        )

    def with_source(self, filename: str | Path, num: int = -1, stage: str | None = None) -> Self:
        """Clone onto another source (config-only); the sides are rebuilt."""
        kwargs: dict[str, Any] = {k: getattr(self, k) for k in _SETTINGS} | self._cfgs
        clone = type(self)(**kwargs, filename=filename, num=num, stage=stage)
        clone.name = self.name
        return clone

    def restage(self, root: str | Path) -> Self:
        """Stage file ``i`` as ``<root>/<digest>/<i>/<parent>/data.root``: order + ids kept."""
        if not self.sources():
            return self
        srcs = self.sources()
        digest = hashlib.sha1(  # noqa: S324 - non-crypto path key
            "\n".join(sorted(str(s.resolve()) for s in srcs)).encode()
        ).hexdigest()[:16]
        dest, width = Path(root) / digest, len(str(len(srcs)))
        for i, src in enumerate(srcs):
            stage_file(src, dest / f"{i:0{width}d}" / src.parent.name / "data.root")
        staged = str(dest / "*" / "*" / "data.root")
        return self.with_source(filename=staged, num=self.num, stage=self.stage)

    def config_fingerprint(self) -> dict[str, Any]:
        return {
            "reader": f"{type(self).__module__}.{type(self).__qualname__}",
            **{k: getattr(self, k) for k in _FINGERPRINTED},
            "row_blocks": "per_container",
            "sides": {side: r.config_fingerprint() for side, r in self.sides.items()},
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_table=None, _num_rows=None, _fill={}, schema=None, _demand=None)
        state.update(_served=None, _dtypes={})
        return state

    def prepare(self) -> None:
        """Prepare every side, check they agree, build the file table + union schema."""
        if self._table is not None:
            return
        if self.filename is None:
            raise ConfigError(f"reader {self.name!r} has no source file — pass filename=")
        for r in self.sides.values():
            r.prepare()
        rows = {side: r.entry_rows() for side, r in self.sides.items()}
        if self._both:
            small, large = ([(p, len(a)) for p, a in rows[s]] for s in ("small", "large"))
            if small != large:
                first = next((a for a, b in zip(small, large, strict=False) if a != b), None)
                raise ConfigError(
                    "FTAG1LITEReader: the small-R and large-R sides resolved different file lists"
                    f" or entry counts ({len(small)} vs {len(large)} files; first mismatch {first})"
                )
        table: list[_FileEntry] = []
        row_start = 0
        side_start = dict.fromkeys(self.sides, 0)
        for per_file in zip(*rows.values(), strict=True):
            cum = {
                side: np.concatenate([[0], np.cumsum(arr)]).astype(np.int64)
                for side, (_, arr) in zip(self.sides, per_file, strict=True)
            }
            n_rows = sum(int(c[-1]) for c in cum.values())
            table.append(_FileEntry(per_file[0][0], row_start, n_rows, cum, dict(side_start)))
            row_start += n_rows
            for side, c in cum.items():
                side_start[side] += int(c[-1])
        self.schema, self._fill = self._union_schema()
        self._num_rows = row_start if self.num < 0 else min(row_start, self.num)
        self._table = table
        _LOG.info(f"{self.name!r}: {len(self.group_names)} FTAG1LITE containers")

    def _union_schema(self) -> tuple[Schema, dict[str, dict[str, Any]]]:
        """Union schema (side order, + ``valid`` / jet type) + each one-sided field's fill."""
        groups: dict[str, GroupSchema] = {}
        fills: dict[str, dict[str, Any]] = {stream: {} for stream in self.streams}
        for stream in self.streams:
            have = {s: getattr(r.schema, "groups", {}).get(stream) for s, r in self.sides.items()}
            jagged = self._group_cfg(stream).jagged
            fields: dict[str, str] = {}
            for side, g in have.items():
                for name, dt in (g.fields if g is not None else {}).items():
                    if jagged and name == "valid":
                        continue
                    if jagged and fields.get(name, dt) != dt:
                        raise ConfigError(
                            f"FTAG1LITEReader: jagged stream {stream!r} field {name!r} reads as "
                            f"{fields[name]} on 'small' but {dt} on {side!r} — a shared "
                            "constituent stream needs one dtype"
                        )
                    fields[name] = np.promote_types(fields.get(name, dt), dt).name
            if jagged:
                fields["valid"] = "bool"
            else:
                for name, dt in fields.items():
                    lacking = [s for s, g in have.items() if g is None or name not in g.fields]
                    if lacking:
                        fills[stream][name] = self._fill_value(stream, name, dt, lacking)
                fields.update(dict.fromkeys(self._special(stream) - _CODES, "int64"))
            groups[stream] = GroupSchema(fields=fields)
        return Schema(groups=groups), fills

    @staticmethod
    def _fill_value(stream: str, name: str, dtype: str, lacking: list[str]) -> Any:
        kind = np.dtype(dtype).kind
        if kind in {"f", "i"}:
            return np.nan if kind == "f" else -1
        raise ConfigError(
            f"FTAG1LITEReader: scalar field {stream}.{name} ({dtype}) is missing on {lacking} "
            "and has no fill value (only floating -> NaN and signed integer -> -1)"
        )

    def _special(self, stream: str) -> set[str]:
        """Reader-injected int64 fields: on `unroll` the codes (on demand) + ``both``'s jet type."""
        jet_type = {self.jet_type_field} if self._both else set()
        return {*_CODES, *jet_type} if stream == self.unroll else set()

    def _resolve_codes(self, demand: dict[str, dict[str, str]]) -> None:
        """Parse every container name into its codes — only when a code column is demanded."""
        if self.unroll is None or not (set(demand.get(self.unroll, {})) & _CODES):
            return
        bad: dict[str, str] = {}  # container name -> its parse error, in group_names order
        for entry in self._table or ():
            name = entry.path.parent.name
            try:
                entry.codes = categories.category_codes(categories.parse_container_categories(name))
            except ConfigError as exc:
                bad[name] = str(exc)
        if bad:
            msg = f"FTAG1LITEReader {self.name!r}: {len(bad)} container(s) cannot be categorised:"
            raise ConfigError("\n  ".join([msg, *bad.values()]))

    def bind(self, ctx: WorkerCtx) -> None:
        """Resolve the served fields per stream; bind each side to the share it configures."""
        self.prepare()
        assert self.schema is not None
        demand = {s: dict(f) for s, f in ctx.read_fields.items()}
        self._resolve_codes(demand)
        served: dict[str, list[str]] = {}
        for stream in self.streams:
            special, fields = self._special(stream), self.schema.groups[stream].fields
            jagged = self._group_cfg(stream).jagged
            union = [f for f in fields if f not in special and (f != "valid" or not jagged)]
            demanded = demand.get(stream, {})
            for name, who in demanded.items():
                if name not in special and name not in union:
                    raise SchemaError(
                        f"field {name!r} demanded by {who!r} is not served on stream "
                        f"{stream!r} (served: {sorted(union)})"
                    )
            served[stream] = [f for f in union if f in demanded] if demanded else union
        for reader in self.sides.values():
            filtered: dict[str, dict[str, str]] = {}
            for stream, cfg in reader.groups.items():
                demanded = demand.get(stream, {})
                filtered[stream] = {
                    f: demanded.get(f, f"{self.name} (union)")
                    for f in served[stream]
                    if f in cfg.served_branches
                }
                if not filtered[stream] and demanded:
                    filtered[stream] = {next(iter(cfg.served_branches)): self.name}
            reader.bind(replace(ctx, read_fields=filtered))
        self._demand, self._served = demand, served
        self._dtypes = {stream: self._joint_dtype(stream, served[stream]) for stream in served}

    def _joint_dtype(self, stream: str, fields: list[str]) -> np.dtype:
        """Served fields in union order, + ``valid`` (jagged) / the jet type (``both``)."""
        assert self.schema is not None
        schema = self.schema.groups[stream].fields
        names = [*fields, *(["valid"] if self._group_cfg(stream).jagged else [])]
        demanded = (self._demand or {}).get(stream, {})
        names += sorted(f for f in self._special(stream) if f not in _CODES or f in demanded)
        return np.dtype([(name, np.dtype(schema.get(name, "int64"))) for name in names])

    def _ensure_bound(self, mode: Mode) -> None:
        if self._served is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Jagged streams are ``(B, pad_max)`` (+ ``masks.<stream>``); scalars ``(B,)``."""
        self._ensure_bound(mode)
        return self._read_global(int(rows.start), int(rows.stop), mode)

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        self._ensure_bound(mode)
        return self._read_global(block.start, block.stop, mode)

    def row_blocks(self) -> list[RowBlock]:
        """One block per non-empty file, clipped to the `num` cap; tiles ``[0, len(self))``."""
        n = len(self)
        assert self._table is not None
        ids = {name: i for i, name in enumerate(self.group_names)}
        blocks = []
        for entry in self._table:
            stop = min(entry.row_start + entry.n_rows, n)
            if stop > entry.row_start:
                blocks.append(RowBlock(ids[entry.path.parent.name], entry.row_start, stop))
            if stop >= n:
                break
        return blocks

    def file_index(self, block: RowBlock) -> int:
        """Index of the `_table` file holding `block` (a block never spans two files)."""
        assert self._table is not None, "prepare() first"
        i = int(np.searchsorted([e.row_start for e in self._table], block.start, "right")) - 1
        if i < 0 or block.stop > self._table[i].row_start + self._table[i].n_rows:
            raise ConfigError(f"FTAG1LITEReader: block {block} is not within one file")
        return i

    def close_file(self, index: int) -> None:
        """Close file `index`'s handle on every side (`UprootReader.close_tree`)."""
        assert self._table is not None
        for reader in self.sides.values():
            reader.close_tree(self._table[index].path)

    def close_all(self) -> None:
        """Close every open handle on every side."""
        for reader in self.sides.values():
            reader.close_all()

    def _read_global(self, start: int, stop: int, mode: Mode) -> dict[str, np.ndarray]:
        assert self._table is not None
        pieces = [
            self._read_file_range(i, lo - e.row_start, hi - e.row_start, mode)
            for i, e in enumerate(self._table)
            if (lo := max(start, e.row_start)) < (hi := min(stop, e.row_start + e.n_rows))
        ]
        if not pieces:
            out = self._assemble({}, dict.fromkeys(self.sides, 0), slice(None), {})
        elif len(pieces) == 1:
            out = pieces[0]
        else:
            out = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([start, stop], dtype=np.int64)
        return out

    def _read_file_range(
        self, index: int, local_lo: int, local_hi: int, mode: Mode
    ) -> dict[str, np.ndarray]:
        """File rows ``[local_lo, local_hi)``: side `read`s over the covering entries, per entry."""
        assert self._table is not None
        entry = self._table[index]
        cum_joint = np.stack(list(entry.cum.values())).sum(axis=0)
        e0, e1, row_off = covering_range(cum_joint, local_lo, local_hi)
        parts: dict[str, dict[str, np.ndarray]] = {}
        n: dict[str, int] = {}
        for side, reader in self.sides.items():
            lo, hi = int(entry.cum[side][e0]), int(entry.cum[side][e1])
            n[side] = hi - lo
            if hi > lo:
                first = entry.side_start[side]
                parts[side] = reader.read(slice(first + lo, first + hi), mode)
        events = np.arange(e0, e1, dtype=np.int64)
        key = np.concatenate([
            np.repeat(events, np.diff(entry.cum[side][e0 : e1 + 1])) * len(self.sides) + i
            for i, side in enumerate(self.sides)
        ])
        take = np.argsort(key, kind="stable")[row_off : row_off + (local_hi - local_lo)]
        return self._assemble(parts, n, take, entry.codes)

    def _assemble(
        self, parts: _Parts, n: dict[str, int], take: np.ndarray | slice, codes: dict[str, int]
    ) -> dict[str, np.ndarray]:
        """The served ``raw.*`` / ``masks.*`` of the side-ordered rows, indexed by `take`."""
        assert self._served is not None
        out: dict[str, np.ndarray] = {}
        for stream in self._served:
            out[f"raw.{stream}"] = raw = self._stack(stream, parts, n, codes)[take]
            if self._group_cfg(stream).jagged:
                out[f"masks.{stream}"] = ~raw["valid"]
        return out

    def _stack(
        self, stream: str, parts: _Parts, n: dict[str, int], codes: dict[str, int]
    ) -> np.ndarray:
        """Side-ordered rows of one stream: all-padding / fill value / injected value if lacking."""
        dt, cfg = self._dtypes[stream], self._group_cfg(stream)
        total = sum(n.values())
        out = np.zeros((total, cfg.pad_max) if cfg.jagged else (total,), dtype=dt)
        types = {"small": self.small_type, "large": self.large_type}
        special = self._special(stream)
        lo = 0
        for side in self.sides:
            hi = lo + n[side]
            src = parts[side].get(f"raw.{stream}") if hi > lo else None
            if cfg.jagged and src is not None:
                if src.shape[1] != cfg.pad_max:
                    raise SchemaError(
                        f"FTAG1LITEReader: {side!r} served {stream!r} with T={src.shape[1]}, "
                        f"expected pad_max={cfg.pad_max}"
                    )
                for name in dt.names or ():
                    out[name][lo:hi] = src[name]
            elif not cfg.jagged and hi > lo:
                have = (src.dtype.names or ()) if src is not None else ()
                for name in dt.names or ():
                    if name in special:
                        out[name][lo:hi] = codes[name] if name in _CODES else types[side]
                    elif src is not None and name in have:
                        out[name][lo:hi] = src[name]
                    else:
                        out[name][lo:hi] = self._fill[stream][name]
            lo = hi
        return out


def _interval(index: int, n_shards: int, n_rows: int) -> tuple[int, int]:
    """Shard `index`'s ``[floor(i*N/S), floor((i+1)*N/S))``; the intervals tile ``[0, N)``."""
    return (index * n_rows) // n_shards, ((index + 1) * n_rows) // n_shards


def _global_cut(rows: list[int], n_shards: int, seed: int) -> list[dict[int, tuple[int, int]]]:
    """``owner`` rank cut: `rows` in a `seed`-only permutation, cut into `n_shards` intervals."""
    n_groups = len(rows)
    order = np.random.default_rng(seed).permutation(n_groups).tolist()
    group_offsets = [0] * (n_groups + 1)
    for i, g in enumerate(order):
        group_offsets[i + 1] = group_offsets[i] + rows[g]
    total = group_offsets[-1]

    assignment: list[dict[int, tuple[int, int]]] = [{} for _ in range(n_shards)]
    gi = 0
    for shard in range(n_shards):
        lo, hi = _interval(shard, n_shards, total)
        if hi <= lo:
            continue
        while gi < n_groups and group_offsets[gi + 1] <= lo:
            gi += 1
        j = gi
        while j < n_groups and group_offsets[j] < hi:
            g_lo, g_hi = group_offsets[j], group_offsets[j + 1]
            seg_lo, seg_hi = max(lo, g_lo) - g_lo, min(hi, g_hi) - g_lo
            if seg_hi > seg_lo:
                assignment[shard][order[j]] = (seg_lo, seg_hi)
            j += 1
    return assignment


def shard_assignment(
    group_rows: Sequence[int],
    n_shards: int,
    scope: str = "per_rank",
    seed: int = 42,
    *,
    n_workers: int = 1,
) -> list[dict[int, tuple[int, int]]]:
    """One ``{group: (lo, hi)}`` per shard (`n_shards`): GROUP-LOCAL row intervals, empty omitted.

    Parameters
    ----------
    group_rows : Sequence[int]
        Row count per group, indexed by group id.
    n_shards : int
        Total shards (``world_size * num_workers``).
    scope : str, optional
        ``"per_rank"`` (default): shard `s` takes ``_interval(s, n_shards, n_g)`` of EVERY
        group. ``"owner"``: `_global_cut` into ``n_shards // n_workers`` rank intervals
        (each rank owns whole containers plus at most a partial one at each end), then
        worker `w` takes ``_interval(w, n_workers, hi - lo)`` of every atom its rank owns.
    seed : int, optional
        Seeds ``owner``'s group permutation; unused by ``per_rank``.
    n_workers : int, optional
        DataLoader workers per rank, by default 1 (``owner`` only).

    Raises
    ------
    ConfigError
        Non-positive `n_shards`/`n_workers`, unknown `scope`, or `n_shards % n_workers` (owner).
    """
    if n_shards < 1:
        raise ConfigError(f"n_shards must be >= 1, got {n_shards}")
    if scope not in SCOPES:
        raise ConfigError(f"scope must be one of {SCOPES}, got {scope!r}")
    if n_workers < 1:
        raise ConfigError(f"n_workers must be >= 1, got {n_workers}")
    rows = [int(n) for n in group_rows]
    if scope == "per_rank":
        cuts = [[_interval(s, n_shards, n_g) for n_g in rows] for s in range(n_shards)]
        return [{g: (lo, hi) for g, (lo, hi) in enumerate(ivs) if hi > lo} for ivs in cuts]
    if n_shards % n_workers != 0:
        raise ConfigError(
            f"scope='owner' needs n_shards ({n_shards}) divisible by n_workers ({n_workers})"
        )
    rank_assignment = _global_cut(rows, n_shards // n_workers, seed)
    owner: list[dict[int, tuple[int, int]]] = [{} for _ in range(n_shards)]
    for r, atoms in enumerate(rank_assignment):
        for g, (lo, hi) in atoms.items():
            for w in range(n_workers):
                a, b = _interval(w, n_workers, hi - lo)
                if b > a:
                    owner[r * n_workers + w][g] = (lo + a, lo + b)
    return owner


def shard_row_counts(assignment: Sequence[Mapping[int, tuple[int, int]]]) -> list[int]:
    """Rows every shard receives, summed from a `shard_assignment` result."""
    return [sum(hi - lo for lo, hi in mapping.values()) for mapping in assignment]


def _partition(
    blocks: Iterable[RowBlock],
    assignment: Mapping[int, tuple[int, int]],
    block_rows: int | None,
) -> dict[int, list[RowBlock]]:
    """``{group: blocks}`` of one shard: each group-local ``[lo, hi)`` is mapped through the
    group's start-sorted blocks cumulatively (any coordinates), then chopped at `block_rows`.
    """
    if block_rows is not None and block_rows < 1:
        raise ConfigError(f"block_rows must be >= 1, got {block_rows}")
    by_group: dict[int, list[RowBlock]] = {}
    for block in blocks:
        by_group.setdefault(block.group, []).append(block)
    out: dict[int, list[RowBlock]] = {}
    for group, (lo, hi) in sorted(assignment.items()):
        pieces: list[RowBlock] = []
        offset = 0
        for block in sorted(by_group.get(group, []), key=lambda b: b.start):
            a, z = max(lo, offset), min(hi, offset + block.n_rows)
            if z > a:
                start, stop = block.start + a - offset, block.start + z - offset
                step = stop - start if block_rows is None else block_rows
                pieces += [block.subrange(s, min(s + step, stop)) for s in range(start, stop, step)]
            offset += block.n_rows
            if offset >= hi:
                break
        if pieces:
            out[group] = pieces
    return out


class PoolSchedule:
    """Pool-size-invariant cumulative allocation, for the ``per_rank`` stream.

    Group `i`'s row `m` has key ``(m + 0.5) * N / n_i``; rows go in key order, ties by group id.
    ``Q_i(h)`` (group-``i`` keys among the `h` smallest) depends on `h` only, so pool counts
    ``Q(stop) - Q(start)`` telescope over any pooling; ``sum_i Q_i(h) == h``, `Q_i` is monotone,
    ``|Q_i(h) - h*p_i| <= 0.5*(1 + G*p_i)``. `cumulative` bisects the key and fills the residual
    from the groups stepping inside the bracket via `_quota_at` alone (monotone under key ties).

    Parameters
    ----------
    counts : Mapping[int, int]
        Rows available per group.
    n_rows : int | None, optional
        Schedule only this prefix of rows (`Q` is monotone); ``None`` = every row.
    """

    def __init__(self, counts: Mapping[int, int], n_rows: int | None = None) -> None:
        self.groups: np.ndarray = np.array(sorted(counts), dtype=np.int64)
        self.lens: np.ndarray = np.array([int(counts[int(g)]) for g in self.groups], dtype=np.int64)
        self.total: int = int(self.lens.sum())
        self.n_rows: int = self.total if n_rows is None else min(int(n_rows), self.total)

    def _quota_at(self, x: float) -> np.ndarray:
        """Per-group quota at virtual time `x`."""
        q = np.floor(x * self.lens / self.total + 0.5)
        return np.clip(q, 0, self.lens).astype(np.int64)

    def cumulative(self, h: int) -> np.ndarray:
        """``Q_i(h)``, the cumulative allocation per group at house size `h`."""
        g = len(self.groups)
        if g == 0:
            return np.zeros(0, dtype=np.int64)
        h = int(h)
        if h <= 0:
            return np.zeros(g, dtype=np.int64)
        if h >= self.total:
            return self.lens.copy()
        lo, hi = 0.0, float(self.total)
        q_lo = np.zeros(g, dtype=np.int64)
        q_hi = self.lens.copy()
        for _ in range(60):
            mid = (lo + hi) / 2.0
            q_mid = self._quota_at(mid)
            if int(q_mid.sum()) <= h:
                lo, q_lo = mid, q_mid
            else:
                hi, q_hi = mid, q_mid
        d = h - int(q_lo.sum())
        stepped = q_hi - q_lo
        cs = np.cumsum(stepped)
        take = np.clip(d - (cs - stepped), 0, stepped)
        return (q_lo + take).astype(np.int64)

    def pool_counts(self, start: int, stop: int) -> np.ndarray:
        """Per-group row counts of the half-open pool ``[start, stop)``."""
        return self.cumulative(stop) - self.cumulative(start)

    def _materialize(self, q_start: np.ndarray, q_stop: np.ndarray) -> np.ndarray:
        """Group ids of the pool between two cumulative allocations, in key order."""
        counts = q_stop - q_start
        if (counts < 0).any():
            bad = {int(g): int(c) for g, c in zip(self.groups, counts, strict=True) if c < 0}
            raise ValueError(f"PoolSchedule.cumulative() lost monotonicity for groups {bad}")
        n = int(counts.sum())
        keys = np.empty(n, dtype=np.float64)
        group_ids = np.empty(n, dtype=np.int64)
        pos = 0
        for i, g in enumerate(self.groups):
            d = int(counts[i])
            if d == 0:
                continue
            m = np.arange(int(q_start[i]), int(q_start[i]) + d, dtype=np.float64)
            keys[pos : pos + d] = (m + 0.5) * self.total / self.lens[i]
            group_ids[pos : pos + d] = g
            pos += d
        return group_ids[np.lexsort((group_ids, keys))]

    def pool(self, start: int, stop: int) -> np.ndarray:
        """Group id per row of the pool ``[start, stop)``; materialises only this pool."""
        if stop <= start:
            return np.empty(0, dtype=np.int64)
        return self._materialize(self.cumulative(start), self.cumulative(stop))

    def pools(self, pool_rows: int) -> Iterator[np.ndarray]:
        """Yield ``pool(k, min(k + pool_rows, n_rows))`` for ``k = 0, pool_rows, ...``."""
        if pool_rows < 1:
            raise ConfigError(f"pool_rows must be >= 1, got {pool_rows}")
        return self._pools_iter(pool_rows)

    def _pools_iter(self, pool_rows: int) -> Iterator[np.ndarray]:
        if self.n_rows == 0 or len(self.groups) == 0:
            return
        q_prev = self.cumulative(0)
        start = 0
        while start < self.n_rows:
            stop = min(start + pool_rows, self.n_rows)
            q_stop = self.cumulative(stop)
            yield self._materialize(q_prev, q_stop)
            q_prev = q_stop
            start = stop


def _largest_remainder(lens: np.ndarray, n_rows: int, total: int) -> np.ndarray:
    """Hamilton apportionment of `n_rows` proportional to `lens`, remainder ties -> lower
    id: ``sum == n_rows``, ``<= lens`` (``n_rows <= total``), ``|q - n_rows*lens/total| < 1``.
    """
    frac = lens.astype(np.float64) * n_rows / total
    floor = np.floor(frac).astype(np.int64)
    remainder = int(n_rows) - int(floor.sum())
    if remainder > 0:
        order = np.argsort(-(frac - floor), kind="stable")
        floor[order[:remainder]] += 1
    return floor


def _pf_successor_cmp(a: int, c: int, extra: np.ndarray, ceils: np.ndarray, n_win: int) -> int:
    """PF tie-break (Baruah et al.) at equal pseudo-deadline and successor bit: recurse on the
    successor subtasks while both bits are 1 and deadlines tie, then lower id; ``< 0``: `a` first.
    """
    j_a, j_c = int(ceils[a]) + 1, int(ceils[c]) + 1
    m_a, m_c = int(extra[a]), int(extra[c])
    while True:
        d_a = (j_a * n_win + m_a - 1) // m_a
        d_c = (j_c * n_win + m_c - 1) // m_c
        if d_a != d_c:
            return -1 if d_a < d_c else 1
        b_a = 1 if (j_a * n_win) % m_a else 0
        b_c = 1 if (j_c * n_win) % m_c else 0
        if b_a != b_c:
            return -1 if b_a > b_c else 1
        if b_a == 0:
            break
        j_a += 1
        j_c += 1
    return -1 if a < c else (1 if a > c else 0)


def _pf_tie_order(groups: list[int], extra: np.ndarray, ceils: np.ndarray, n_win: int) -> list[int]:
    """`groups` sorted into PF priority order by `_pf_successor_cmp`."""

    def cmp(a: int, c: int) -> int:
        return _pf_successor_cmp(a, c, extra, ceils, n_win)

    return sorted(groups, key=functools.cmp_to_key(cmp))


class WindowSchedule:
    """Exact per-window stratified allocation, for ``owner``.

    ``T = n_rows // window`` windows (rank batches); each sums to `window`, each group's
    count is ``< 1`` from ``quotas[c] / T`` (largest-remainder `quotas`) and exact over
    the epoch. PF (Baruah et al., *Algorithmica* 15(6), 1996): each window's bonus
    ``k = window - sum(base)`` rows go to the released subtasks (``t >=
    floor(ceils*T/extra)``) of earliest pseudo-deadline ``ceil((ceils+1)*T/extra)``,
    successor bit 1 first, then `_pf_tie_order`, then lower group id; Pfair-optimal, so
    every prefix stays ``< 1`` from ideal too. `verify` ``>= 1.0`` is a bug.

    Parameters
    ----------
    counts : Mapping[int, int]
        Rows available per group (this shard's atoms).
    n_rows : int
        Rows to schedule: a non-negative multiple of `window`, at most ``sum(counts)``.
    window : int
        Rows per window (the batch size), >= 1; a violated constraint is a `ConfigError`.
    """

    def __init__(self, counts: Mapping[int, int], n_rows: int, window: int) -> None:
        if window <= 0:
            raise ConfigError(f"window must be >= 1, got {window}")
        if n_rows < 0:
            raise ConfigError(f"n_rows must be >= 0, got {n_rows}")
        if n_rows % window != 0:
            raise ConfigError(f"n_rows ({n_rows}) must be a multiple of window ({window})")
        groups = np.array(sorted(counts), dtype=np.int64)
        lens = np.array([int(counts[int(g)]) for g in groups], dtype=np.int64)
        total = int(lens.sum())
        if n_rows > total:
            raise ConfigError(f"n_rows ({n_rows}) exceeds the corpus total ({total})")
        self.groups: np.ndarray = groups
        self.lens: np.ndarray = lens
        self.total: int = total
        self.window: int = int(window)
        self.n_rows: int = int(n_rows)
        self.n_windows: int = self.n_rows // self.window
        g = len(groups)
        self.quotas: np.ndarray = (
            _largest_remainder(lens, self.n_rows, total) if g and n_rows else np.zeros(g, np.int64)
        )
        if self.n_windows > 0:
            self.base: np.ndarray = self.quotas // self.n_windows
            self.extra: np.ndarray = self.quotas - self.n_windows * self.base
        else:
            self.base = np.zeros(g, dtype=np.int64)
            self.extra = np.zeros(g, dtype=np.int64)
        self.k: int = int(self.window - int(self.base.sum()))

    def counts(self) -> Iterator[np.ndarray]:
        """Yield `n_windows` ``(G,)`` int64 per-window count arrays (`groups` order)."""
        n_win = self.n_windows
        if n_win == 0:
            return
        base, extra, k = self.base, self.extra, self.k
        if k == 0:
            for _ in range(n_win):
                yield base.copy()
            return
        needy = np.nonzero(extra > 0)[0]
        ceils = np.zeros(len(base), dtype=np.int64)
        for t in range(n_win):
            not_done = needy[ceils[needy] < extra[needy]]
            m_nd, c_nd = extra[not_done], ceils[not_done]
            eligible = not_done[t >= (c_nd * n_win) // m_nd]
            if len(eligible) < k:
                raise AssertionError(
                    f"WindowSchedule: {len(eligible)} released bonus subtasks < k={k} at "
                    f"window {t} — Pfair invariant broken"
                )
            if len(eligible) == k:
                chosen = eligible
            else:
                m, j = extra[eligible], ceils[eligible] + 1
                d = (j * n_win + m - 1) // m  # pseudo-deadline ceil(j*T/m)
                b = ((j * n_win) % m != 0).astype(np.int64)  # successor bit
                order = np.lexsort((eligible, -b, d))
                priority = eligible[order]
                d_sorted, b_sorted = d[order], b[order]
                if d_sorted[k - 1] == d_sorted[k] and b_sorted[k - 1] == b_sorted[k] == 1:
                    tie_pos = np.nonzero((d_sorted == d_sorted[k - 1]) & (b_sorted == 1))[0]
                    resolved = _pf_tie_order(
                        [int(g) for g in priority[tie_pos]], extra, ceils, n_win
                    )
                    priority = priority.copy()
                    priority[tie_pos] = np.array(resolved, dtype=np.int64)
                chosen = priority[:k]
            row = base.copy()
            row[chosen] += 1
            ceils[chosen] += 1
            yield row

    def counts_at(self, t: int) -> np.ndarray:
        """The count array of window `t`, by replaying `counts` (tests, not a hot path)."""
        if not 0 <= t < self.n_windows:
            raise ConfigError(f"window index {t} out of range [0, {self.n_windows})")
        for i, row in enumerate(self.counts()):
            if i == t:
                return row
        raise AssertionError("unreachable: n_windows/counts() disagree")  # pragma: no cover

    def windows(self) -> Iterator[np.ndarray]:
        """Yield `n_windows` length-`window` group-id arrays, ids ascending per window."""
        for row in self.counts():
            yield np.repeat(self.groups, row)

    def verify(self) -> float:
        """Observed max ``|count - quotas/T|`` over every window and group."""
        if self.n_windows == 0 or len(self.groups) == 0:
            return 0.0
        x = self.quotas.astype(np.float64) / self.n_windows
        return max(float(np.max(np.abs(row.astype(np.float64) - x))) for row in self.counts())


class FTAG1LITEStreamingDataset(IterableSaltDataset):
    """`IterableSaltDataset` over `FTAG1LITEReader` containers, one group per container.

    The reader's `shard_scope` / `shuffle_pool_rows` pick the plan (VAL/TEST: always ``per_rank``).
    ``per_rank``: each shard takes an interval of every container, batches dealt from `PoolSchedule`
    pools. ``owner`` (FIT): ranks own disjoint seeded container slices, split across their workers;
    each batch is a `WindowSchedule` window, so the super batch (all ranks, one step) is stratified
    by container. One resident block per container cursor, no eviction; ROOT files close after
    their last block; `max_live_streams` asserts.

    Effective epoch ``E = epoch + k_w``: ``k_w`` counts worker slot ``w``'s (worker id, 0 in the
    main process) earlier `__iter__` calls since `epoch` last changed, kept in shared-memory
    ``(epoch seen, k)`` slots made at construction. Persistent, re-forked and ``num_workers=0``
    workers all advance once per pass; a new `epoch` (a later trainer epoch's loader, a resume at
    ``e``) resets them. Shard rng ``default_rng([seed, E, shard_id])``; coverage/ownership ignore E.

    Parameters
    ----------
    modules, mode, sinks, batch_size
        As `IterableSaltDataset`; every other keyword is forwarded to it unchanged.

    Raises
    ------
    ConfigError
        An unknown scope, ``interleave_block != 1``, a pool below `batch_size`, or for
        FIT ``owner`` a non-null `shuffle_pool_rows` or ``drop_last=False``.
    """

    def __init__(
        self, modules: dict[str, Any], mode: Mode, sinks: Sinks, batch_size: int, **kwargs: Any
    ) -> None:
        super().__init__(modules, mode, sinks, batch_size, **kwargs)
        scope = str(getattr(self._reader, "shard_scope", "per_rank"))
        pool = getattr(self._reader, "shuffle_pool_rows", 100_000)
        if scope not in SCOPES:
            raise ConfigError(f"shard_scope must be one of {SCOPES}, got {scope!r}")
        if self.interleave_block != 1:
            raise ConfigError(
                f"FTAG1LITEStreamingDataset needs interleave_block 1, got {self.interleave_block}"
            )
        fit = self._mode == Mode.FIT
        if fit and scope == "owner" and pool is not None:
            raise ConfigError(
                f"owner scope needs shuffle_pool_rows null — the window is the batch (got {pool})"
            )
        if fit and scope == "owner" and not self.drop_last:
            raise ConfigError("shard_scope='owner' requires drop_last=True (whole windows)")
        self.shard_scope = scope if fit else "per_rank"
        self.shuffle_pool_rows: int | None = None if pool is None else int(pool)
        if self.shuffle_pool_rows is not None and self.pool_rows() != self.shuffle_pool_rows:
            _LOG.info(f"shuffle_pool_rows {self.shuffle_pool_rows} floored to {self.pool_rows()}")
        n_slots = max(1, int(self._num_workers or 1))
        self._epoch_slots = torch.zeros((n_slots, 2), dtype=torch.int64).share_memory_()

    def pool_rows(self) -> int:
        """Rows per pool: `batch_size` if unset, else floored to whole batches (exact `__len__`)."""
        if self.shuffle_pool_rows is None:
            return self.batch_size
        if self.shuffle_pool_rows < self.batch_size:
            raise ConfigError(
                f"shuffle_pool_rows ({self.shuffle_pool_rows}) must be >= batch_size "
                f"({self.batch_size}), or null for one batch per pool"
            )
        return (self.shuffle_pool_rows // self.batch_size) * self.batch_size

    def _effective_epoch(self, slot: int) -> int:
        """``epoch + k`` for worker `slot`, then count this pass (see the class docstring)."""
        slots = self._epoch_slots
        if slot >= slots.shape[0]:
            raise ConfigError(
                f"worker {slot} has no epoch slot: construct with the DataLoader's num_workers"
            )
        if int(slots[slot, 0]) != self.epoch:
            slots[slot, 0], slots[slot, 1] = self.epoch, 0
        k = int(slots[slot, 1])
        slots[slot, 1] = k + 1
        return self.epoch + k

    def _assignment(self, n_shards: int, n_workers: int) -> list[dict[int, tuple[int, int]]]:
        _blocks, group_rows = self._resolve_layout()
        return shard_assignment(
            group_rows, n_shards, self.shard_scope, seed=self._seed, n_workers=n_workers
        )

    def _check_live_streams(
        self, n_groups_here: int, blocks: Iterable[RowBlock], n_shards: int, shard_id: int
    ) -> None:
        """Residency assertion: a shard may hold at most `max_live_streams` resident blocks."""
        cap = self.max_live_streams
        if cap is None or cap >= n_groups_here:
            return
        rows = self.block_rows if self.block_rows is not None else max(b.n_rows for b in blocks)
        amp = rows * n_groups_here / self.batch_size
        raise ConfigError(
            f"max_live_streams={cap} < G_per_shard={n_groups_here} (shard_scope="
            f"{self.shard_scope!r}, shard {shard_id}/{n_shards}): evicting would re-read "
            f"blocks of block_rows={rows}, read amplification ~ {amp:.0f}x. Set "
            f"max_live_streams to null or >= {n_groups_here} ({n_groups_here}*{rows} rows)."
        )

    def _shard_batches(self, n_shards: int, shard_id: int) -> int:
        """Exact batches of shard `shard_id`; under `drop_last` the minimum over all shards."""
        blocks, _rows = self._resolve_layout()
        assignment = self._assignment(n_shards, self._rank_worker()[2])
        self._check_live_streams(len(assignment[shard_id]), blocks, n_shards, shard_id)
        per_shard = shard_row_counts(assignment)
        if self.drop_last:
            return min(per_shard) // self.batch_size
        return -(-per_shard[shard_id] // self.batch_size)  # ceil

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Stream this shard's batches for ``(seed, effective epoch)``; files open post-fork.
        Handles are released when the pass ends, the generator is closed, or the iterator is
        dropped (a persistent worker keeps a partial pass's handles until its iterator is replaced).
        """
        self._maybe_bind()
        blocks, _rows = self._resolve_layout()
        world, rank, n_workers, worker_id = self._rank_worker()
        if worker_id is None:
            raise ConfigError(
                f"main-process handle of a {n_workers}-worker rank: not a shard, use its DataLoader"
            )
        n_shards, shard_id = world * n_workers, rank * n_workers + worker_id
        epoch = self._effective_epoch(worker_id)
        assignment = self._assignment(n_shards, n_workers)
        mine = _partition(blocks, assignment[shard_id], self.block_rows)
        if not mine:
            return
        n_batches = self._shard_batches(n_shards, shard_id)  # also the live-stream check
        _LOG.info(
            f"ftag1lite streaming: world={world} rank={rank} workers={n_workers} "
            f"worker={worker_id} -> shard {shard_id}/{n_shards} scope={self.shard_scope} "
            f"epoch={epoch} batches={n_batches} groups={len(mine)}"
        )
        rng = np.random.default_rng([self._seed, epoch, shard_id])
        if self.shuffle:
            for group_blocks in mine.values():
                rng.shuffle(group_blocks)  # type: ignore[arg-type]
        counts = {group: sum(b.n_rows for b in bs) for group, bs in mine.items()}
        pools: Iterable[np.ndarray]
        if self.shard_scope == "owner":
            n_rows = n_batches * self.batch_size
            pools = WindowSchedule(counts, n_rows=n_rows, window=self.batch_size).windows()
        else:
            n_rows = n_batches * self.batch_size if self.drop_last else sum(counts.values())
            pools = PoolSchedule(counts, n_rows=n_rows).pools(self.pool_rows())
        try:
            yield from self._stream_pools(mine, pools, rng)
        finally:
            getattr(self._reader, "close_all", lambda: None)()

    def _stream_pools(
        self, mine: dict[int, list[RowBlock]], pools: Iterable[np.ndarray], rng: np.random.Generator
    ) -> Iterator[dict[str, Any]]:
        """Permute each pool (when `shuffle`) and deal it in whole `batch_size` batches.
        Each file closes right after its last block in this shard's order (so at most the files the
        slice spans per container stay open; 2 on a ~3-file corpus); `__iter__`'s `finally` closes
        the rest.
        """
        file_index = getattr(self._reader, "file_index", None)
        close_file = getattr(self._reader, "close_file", lambda _: None)
        last = {int(file_index(b)): b for g in mine.values() for b in g} if file_index else {}
        last_of_file = {b: i for i, b in last.items()}  # each file's last block -> file index

        def read(block: RowBlock) -> dict[str, np.ndarray]:
            out = self._read_block(block)
            if block in last_of_file:
                close_file(last_of_file[block])
            return out

        cursors = {group: _GroupCursor(blocks) for group, blocks in mine.items()}
        for pool in pools:
            if pool.size == 0:
                break
            pooled = pool[rng.permutation(pool.size)] if self.shuffle else pool
            for start in range(0, pooled.size, self.batch_size):
                wanted = pooled[start : start + self.batch_size]
                chunks: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = []
                for group in np.unique(wanted):
                    positions = np.flatnonzero(wanted == group)
                    cursor, filled = cursors[int(group)], 0
                    for produced, rows in cursor.take(positions.size, read):
                        chunks.append((positions[filled : filled + rows.size], produced, rows))
                        filled += rows.size
                n = int(wanted.size)
                yield self._run_plan(slice(0, n), raw=self._assemble(chunks, n, rng))
