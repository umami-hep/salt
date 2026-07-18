"""`UprootReader` — the shared uproot-backed `Reader` base for ROOT ntuples.

Owns the config surface (`UprootGroupConfig`, `declare_io`, `with_source`), the
file-table probing skeleton and the cut->sort->truncate->pad assembly common to
every ROOT reader; concrete readers (`EasyjetReader`, `XAODReader`) supply only
the branch-name mapping, the jaggedness ndim threshold, the per-file index build
and the per-stream column read.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx, _require_root_deps
from salt.core.data.stream import StreamConfig
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema

__all__ = ["UprootGroupConfig", "UprootReader"]

_GROUP_KEYS = {"branches", "jagged", "pad_max", "truncate", "link_branch", "target_collection"}


@dataclass(frozen=True)
class UprootGroupConfig:
    """Per-stream reader configuration for a `UprootReader`.

    `branches` maps the v2 field name (what the rest of the pipeline sees, e.g.
    ``pt``) to the source branch name; how that maps onto the on-disk branch is
    the concrete reader's business (identity for flat ntuples, an aux-store
    prefix for xAOD). Field order = the config dict order.

    `jagged=True` declares a per-row variable-length sequence stream (padded to a
    fixed ``T`` with a ``valid`` field + pad mask). `jagged=False` is a scalar
    per-row stream (``[B]``, no pad mask). `pad_max` keeps the leading N
    constituents of a jagged stream; None auto-resolves the file's max
    multiplicity in `prepare`. (The legacy easyjet spelling ``truncate`` is
    accepted as an alias for `pad_max`.)

    `link_branch` + `target_collection` turn a jagged stream into an
    ElementLink-dereferenced constituent stream (PHYSLITE): the per-jet link
    vector ``<jet_collection>AuxDyn.<link_branch>`` (e.g. ``GhostTrack``) carries
    ``m_persIndex`` into the ``<target_collection>`` container (e.g.
    ``InDetTrackParticles``), whose ``branches`` are gathered per jet. When unset
    (default) a jagged stream is a direct double-jagged decoration.
    """

    branches: dict[str, str]
    jagged: bool = True
    pad_max: int | None = None
    link_branch: str | None = None
    target_collection: str | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ConfigError("group config: 'branches' must be a non-empty mapping")
        if self.pad_max is not None and self.pad_max < 1:
            raise ConfigError(f"group pad_max must be >= 1, got {self.pad_max}")
        if not self.jagged and self.pad_max is not None:
            raise ConfigError("group config: 'pad_max' is only valid for jagged streams")
        if (self.link_branch is None) != (self.target_collection is None):
            raise ConfigError(
                "group config: 'link_branch' and 'target_collection' must be set together "
                "(ElementLink dereference needs both the jet link vector and the target container)"
            )
        if self.link_branch is not None and not self.jagged:
            raise ConfigError(
                "group config: 'link_branch'/'target_collection' are only valid for jagged "
                "(constituent) streams"
            )

    @property
    def is_linked(self) -> bool:
        """Whether this stream reads constituents via an ElementLink dereference."""
        return self.link_branch is not None


class UprootReader(Reader):
    """uproot-backed `Reader` base — disk -> ``raw.<stream>`` / ``masks.<stream>``.

    Concrete readers set the class hooks (`_branch`, `_JAGGED_NDIM`, the
    dependency-extra names) and implement `prepare` (build the per-file index +
    schema), `_read_stream_columns` (the per-stream column read) and `_clone`
    (constructor for `with_source`). Everything else — config parsing, the served
    length axis, padding/assembly, demand narrowing, pickling — lives here.
    """

    #: ndim at/above which a probed branch counts as a jagged (sequence) stream.
    _JAGGED_NDIM: int = 2
    #: name + pip extra used in the missing-dependency error (per reader).
    _DEP_NAME: str = "UprootReader"
    _DEP_EXTRA: str = "root"

    def _require_deps(self) -> None:
        """Raise a helpful ImportError naming this reader's pip extra if deps are missing."""
        _require_root_deps(self._DEP_NAME, self._DEP_EXTRA)

    def _branch(self, bare: str) -> str:
        """Map a bare (config) branch name to the on-disk branch name; identity by default."""
        return bare

    @staticmethod
    def _parse_group(
        stream: str, cfg: UprootGroupConfig | Mapping[str, Any] | Any
    ) -> UprootGroupConfig:
        """Normalise a group config entry to a `UprootGroupConfig` (``truncate`` -> ``pad_max``)."""
        if isinstance(cfg, UprootGroupConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - _GROUP_KEYS
        if unknown:
            raise ConfigError(
                f"group {stream!r}: unknown config keys {sorted(unknown)} — expected "
                "branches/jagged/pad_max (truncate accepted as a pad_max alias)"
            )
        if "branches" not in cfg:
            raise ConfigError(
                f"group {stream!r}: 'branches' mapping is required (field -> branch name)"
            )
        if "pad_max" in cfg and "truncate" in cfg:
            raise ConfigError(
                f"group {stream!r}: give either 'pad_max' or 'truncate' (its alias), not both"
            )
        pad_max = cfg.get("pad_max", cfg.get("truncate"))
        return UprootGroupConfig(
            branches={str(k): str(v) for k, v in dict(cfg["branches"]).items()},
            jagged=bool(cfg.get("jagged", True)),
            pad_max=pad_max,
            link_branch=cfg.get("link_branch"),
            target_collection=cfg.get("target_collection"),
        )

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
        """The resolved ROOT file list (the multi-file staging surface).

        A source can be a directory or a glob matching many files, so `sources`
        returns the full resolved member list. Returns ``[]`` when no source is
        bound or the glob is empty (staging an unbound reader is a no-op).
        """
        if self.filename is None:
            return []
        try:
            return self._resolve_files()
        except ConfigError:
            return []

    def restage(self, root: str | Path) -> UprootReader:
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
        digest = hashlib.sha1(  # noqa: S324 - non-crypto path key, collision-safe enough
            "\n".join(sorted(str(s.resolve()) for s in srcs)).encode()
        ).hexdigest()[:16]
        dest_dir = root / digest
        staged = [stage_file(src, dest_dir / src.name) for src in srcs]
        new_src = staged[0] if len(staged) == 1 else dest_dir
        return self.with_source(filename=new_src)

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic, sorted list of ROOT files (``*.root`` and
        ``*.pool.root`` for a directory source).
        """
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
        """Native-endian numpy dtype name for a (possibly ragged) awkward array (ROOT is
        big-endian; a ragged field uses its innermost content dtype).
        """
        self._require_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        flat = ak.flatten(arr, axis=None)
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

    def _clone(self, filename: str | Path, num: int, stage: str | None) -> UprootReader:
        """Construct a config-identical reader onto another source (subclass constructor)."""
        raise NotImplementedError

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,  # accepted for API parity; ROOT has no VDS
        stage: str | None = None,
    ) -> UprootReader:
        """Clone onto another source (config-only, group configs shared); `vds_path` is
        accepted for API parity only (ROOT has no VDS).
        """
        del vds_path
        clone = self._clone(filename=filename, num=num, stage=stage)
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
                        f"group {stream!r} (configured: {sorted(cfg.branches)})"
                    )
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
        """Read a stream's demanded branches over a global row range (subclass hook)."""
        raise NotImplementedError

    def _stream_config(self, stream: str) -> StreamConfig:
        """The `StreamConfig` for a jagged stream (resolved ``pad_max``; no cuts/sort —
        constituent selection, if any, is a subclass concern).
        """
        return StreamConfig(pad_max=self._mult[stream], jagged=True)

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad jagged columns to ``T`` via `Reader.assemble_jagged` (contiguous
        truncate+pad+valid path — no per-constituent cuts/sort configured here).
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
        state.update(
            {
                "_table": None,
                "_num_rows": None,
                "_mult": {},
                "_read_fields": {},
                "schema": None,
            }
        )
        return state
