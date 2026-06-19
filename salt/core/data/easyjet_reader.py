"""`EasyjetReader` — a config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples.

The v2 modular-Reader boundary in action (design §2.4, §6.1): a NEW file format
is a NEW `Reader` subclass — `Features` / `Labels` / `Normaliser` / the model /
``salt2`` are all UNCHANGED. This reads easyjet output-tree ROOT files (the
DAOD_PHYS → ntuple step's product) and emits the same ``raw.<stream>`` structured
numpy arrays + ``masks.<stream>`` pad masks the `H5StructuredReader` produces, so
the rest of the pipeline cannot tell the two apart.

Two differences from the H5 path drive the design:

- **The file format is jagged, not pre-padded.** easyjet jets are a per-event
  variable-length list (``recojet_antikt4PFlow_*`` branches are ``AsJagged``). The
  reader pads each jagged stream to a fixed multiplicity ``T`` per ``read``,
  computing the ``valid`` length FIRST (the true per-event jet count), then
  ``ak.pad_none`` + ``ak.fill_none`` per dtype — **float features → 0.0, integer
  LABELS → sentinel -1** (NOT 0, so a padded label can never be consumed as a real
  class; the sequence tasks fold ``masks.<stream>`` to ``ignore_index=-1``). The
  result is a structured ``(B, T)`` numpy array with a ``valid`` bool field, and
  ``masks.<stream> = ~valid`` (True = padded) — exactly the H5 contract.

- **Branch names are config, not hardcoded.** The ``_NOSYS`` suffix and the jet
  collection prefix vary between samples (HH4b vs flavtag), so the reader takes a
  ``groups`` mapping of ``stream -> {branches: {field: branch_name}, jagged, truncate}``.
  The HH4b and flavtag files become two CONFIGS sharing one reader.

``uproot`` and ``awkward`` are imported LAZILY inside the methods that touch the
file — ``salt.core`` imports cleanly without them (they are optional reader extras,
NOT a salt package dependency; only configs that use this reader need them at
runtime).

Multi-file: ``prepare`` globs the source (a directory, a glob, or a single file),
builds a DETERMINISTIC (sorted) file table with cumulative entry offsets, and
``read(rows, mode)`` translates a global row slice into per-file uproot
``entry_start``/``entry_stop`` reads of the DEMANDED branches only (honouring
`WorkerCtx.read_fields`). A slice crossing a file boundary is stitched together.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema, Schema

__all__ = ["EasyjetGroupConfig", "EasyjetReader"]


def _require_root_deps() -> None:
    """Import-time guard for the optional ROOT reader extra.

    Called once at the first file-touching operation (``prepare`` /
    ``_read_columns`` / ``_array_dtype_name`` / ``_max_count``).  If uproot or
    awkward are absent the user gets a clear, actionable error pointing at the
    correct install command instead of a bare ``ModuleNotFoundError`` from deep
    inside an array method.

    This function is intentionally cheap when the deps ARE present (two imports
    that are already cached in ``sys.modules`` after the first call).
    """
    try:
        import awkward  # noqa: F401
        import uproot  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "EasyjetReader requires the 'easyjet' extra — install with:\n"
            "  pip install 'salt[easyjet]'\n"
            "or directly:\n"
            "  pip install uproot awkward"
        ) from exc


# Default pad fills, by dtype kind (numpy kind code). Floats pad to 0.0 (zeroed
# again downstream after masking in Features); SIGNED ints (labels) pad to the
# -1 sentinel so a padded label is never a real class; unsigned ints pad to 0
# (they are not labels — counts/masks/ids — and -1 is unrepresentable). bool
# pads to False. The 'valid' field is set explicitly, never via these fills.
_INT_PAD_SENTINEL = -1


@dataclass(frozen=True)
class EasyjetGroupConfig:
    """Per-stream reader configuration for `EasyjetReader` (design §2.6 / §6.1).

    `branches` maps the v2 FIELD name (the name the rest of the pipeline sees,
    e.g. ``pt``) to the ROOT BRANCH name in the file (e.g.
    ``recojet_antikt4PFlow_pt_NOSYS``). This is where the ``_NOSYS`` suffix and
    collection-prefix variation lives — HH4b vs flavtag is a config change, not a
    code change. Field order = the config dict order (the structured-array field
    order, mirroring the H5 reader's FILE field order contract).

    `jagged` declares a per-event variable-length sequence stream (``jets``,
    padded to a fixed ``T`` with a ``valid`` field + pad mask). ``jagged: false``
    is a scalar event-level stream (``[B]``, no pad mask — the
    ``global_object: true`` analogue). `truncate` keeps the leading N constituents
    of a jagged stream (v1 ``num_inputs`` / H5 ``truncate``); None auto-resolves
    the file's max multiplicity in `prepare`.
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
    """Config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples (design §2.4, §6.1).

    Parameters
    ----------
    groups : Mapping[str, EasyjetGroupConfig | Mapping | ...]
        Stream name -> group config (``{branches:, jagged:, truncate:}``). The
        FIRST jagged stream defines the served length axis (rows are aligned
        across streams by the per-event structure). At least one group required.
    filename : str | Path | None, optional
        The source: a single ``.root`` file, a directory of them, or a glob. A
        directory globs to ``*.root``. May be omitted at construction and supplied
        via `with_source` (the datamodule pattern, design §6.1).
    tree : str, optional
        The TTree name, by default ``"AnalysisMiniTree"``.
    num : int, optional
        Number of rows to serve; ``-1`` = all (v1 ``get_num`` semantics).

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
        # No on-disk schema artifact (ROOT files are not the §2.6 H5 layout); the
        # schema is BUILT in prepare() from the resolved files (config I/O over the
        # file structure, not data I/O). schema stays None until prepare().
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see __getstate__)
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}  # stream -> served multiplicity T
        self._read_fields: dict[str, dict[str, str]] = {}

    # -- config helpers ------------------------------------------------------

    @staticmethod
    def _parse_group(
        stream: str, cfg: EasyjetGroupConfig | Mapping[str, Any] | Any
    ) -> EasyjetGroupConfig:
        """Normalise one group config entry to an `EasyjetGroupConfig`.

        Returns
        -------
        EasyjetGroupConfig
            The parsed config.

        Raises
        ------
        ConfigError
            On unknown keys or a missing/empty ``branches`` mapping.
        """
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
        """The configured stream names, in config order.

        Returns
        -------
        tuple[str, ...]
            Stream names.
        """
        return tuple(self.groups)

    # -- GraphModule declaration (config-only, design §2.2/§2.3) -------------

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.* / masks.* / meta.rows`` produces (source node, requires={}).

        Sequence dims are concrete when ``truncate`` is set, else symbolic
        ``T:<stream>``; ``meta.rows`` is TEST-only (mirrors `H5StructuredReader`).

        Returns
        -------
        IO
            The declared interface.
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

    # -- file-touching lifecycle hooks (design §2.3) --------------------------

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic, sorted list of ROOT files.

        Returns
        -------
        list[Path]
            Sorted file paths.

        Raises
        ------
        ConfigError
            If no source is bound or the glob matches nothing.
        """
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
        """Resolve the source files, probe entry counts, and build the schema.

        Main-process hook (called lazily by ``__len__``, eagerly by the
        datamodule's rank-0 pre-creation). Idempotent. Builds the deterministic
        file table (cumulative offsets), resolves each jagged stream's served
        multiplicity ``T`` (``truncate`` or the file-wide max), and constructs the
        in-memory `Schema` from the first file's branch dtypes (+ the auto
        ``valid`` field on jagged streams). Propagates `ConfigError` from
        `_resolve_files` when no source is bound or the glob is empty.

        Raises
        ------
        ValueError
            If ``num`` requests more rows than available.
        SchemaError
            If a configured branch is missing, or a jagged branch reads as scalar
            (or vice versa).
        """
        if self._table is not None:
            return
        _require_root_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy; design §6.1)

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
        self._num_rows = num_available if self.num < 0 else self.num

    @staticmethod
    def _array_dtype_name(arr: Any, is_jagged: bool) -> str:
        """Native-endian numpy dtype name for a (possibly jagged) awkward array.

        ROOT stores big-endian; the structured array uses NATIVE byte order
        (``>f4`` -> ``float32``) so the rest of the pipeline sees ordinary native
        arrays. For a jagged field the inner content dtype is taken (the per-jet
        scalar type); a float field stays float (scores/kinematics), an integer
        field stays integer (the flavour label).

        Returns
        -------
        str
            A ``np.dtype(name)``-constructible name.
        """
        _require_root_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        flat = ak.flatten(arr, axis=None) if is_jagged else arr
        dtype = np.asarray(ak.to_numpy(flat)).dtype
        return np.dtype(dtype.newbyteorder("=")).name

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
        """The first resolved source file (writers re-read by absolute rows, design §8).

        Returns
        -------
        Path
            The first concrete ROOT file backing this reader.
        """
        self.prepare()
        assert self._table is not None
        return self._table[0].path

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The built schema's group for one served stream (design §2.6).

        Returns
        -------
        GroupSchema | None
            The group schema (built in `prepare`), or None before prepare /
            for an unknown stream.
        """
        if self.schema is None:
            self.prepare()
        if self.schema is None or stream not in self.groups:
            return None
        return self.schema.groups.get(stream)

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe for wildcard narrowing (§2.2 rule d).

        Returns
        -------
        tuple[str, ...] | None
            All schema-backed label keys (built in `prepare`).
        """
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
        """Clone this reader for another source (the datamodule pattern, design §6.1).

        Config-only (no file I/O): the group configs are shared; only the source
        binding and ``num`` change. ``vds_path`` is accepted for `Reader` API
        parity but unused (ROOT files need no VDS). ``stage`` is accepted for the
        plan-02 stage-sourcing contract and IGNORED here (a single-source reader's
        one ``filename`` per stage IS its data).

        Returns
        -------
        EasyjetReader
            A fresh, unbound reader instance (same instance ``name``).
        """
        del vds_path, stage
        clone = EasyjetReader(
            groups=self.groups, filename=filename, tree=self.tree, num=num
        )
        clone.name = self.name
        return clone

    # -- per-worker binding (design §2.3) -------------------------------------

    def bind(self, ctx: WorkerCtx) -> None:
        """Per-worker setup: resolve files + record the demand-narrowed read set.

        The per-stream read set is ``demanded fields`` (from `WorkerCtx.
        read_fields`) intersected with the configured branches; an empty demand
        falls back to ALL configured branches (so a bare round-trip read still
        works). Demanded fields absent from the config raise before any step.

        Raises
        ------
        SchemaError
            When a demanded field is not in the group's configured branches.
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

    # -- the per-batch read (design §6.1) -------------------------------------

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous batch slab, translating the global slice per file.

        For each stream the demanded branches are read over the global row range
        (split into per-file ``entry_start``/``entry_stop`` reads, stitched in
        file order). A JAGGED stream is padded to the served multiplicity ``T``:
        the ``valid`` length is computed FIRST, then per dtype fills (float -> 0.0,
        signed int label -> -1 sentinel, unsigned int -> 0, bool -> False), and the
        result is a structured ``(B, T)`` array with named fields + a ``valid``
        field; ``masks.<stream> = ~valid`` (True = padded). A SCALAR stream is a
        structured ``(B,)`` array. ``meta.rows`` is produced in TEST.

        Returns
        -------
        dict[str, np.ndarray]
            The produced keys, as a flat dotted dict.
        """
        if self._table is None:
            self.bind(
                WorkerCtx(mode=mode, read_fields={}, seed=0)
            )  # standalone use (tests) — full read set
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
        """The field names served for a stream (demanded subset or all configured).

        Returns
        -------
        list[str]
            Field names in config order; falls back to all branches when no
            demand was narrowed for this stream.
        """
        cfg = self.groups[stream]
        demanded = self._read_fields.get(stream, {})
        names = set(demanded) if demanded else set(cfg.branches)
        # keep config order, narrow to the demanded subset
        return [f for f in cfg.branches if f in names]

    def _read_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read the demanded branches over a global row range (multi-file stitched).

        Returns
        -------
        dict[str, Any]
            ``{field: awkward/numpy array}`` of length ``stop - start``, in global
            row order. Jagged fields are awkward arrays; scalar fields are numpy.
        """
        _require_root_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        assert self._table is not None
        cfg = self.groups[stream]
        branch_of = cfg.branches
        per_field_chunks: dict[str, list[Any]] = {f: [] for f in fields}
        for entry in self._table:
            lo = max(start, entry.start)
            hi = min(stop, entry.start + entry.n)
            if lo >= hi:
                continue
            with uproot.open(f"{entry.path}:{self.tree}") as t:
                for f in fields:
                    arr = t[branch_of[f]].array(
                        entry_start=lo - entry.start,
                        entry_stop=hi - entry.start,
                        library="ak",
                    )
                    per_field_chunks[f].append(arr)
        cols: dict[str, Any] = {}
        for f in fields:
            chunks = per_field_chunks[f]
            cols[f] = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return cols

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad jagged columns to ``T`` and assemble a structured ``(B, T)`` array.

        The ``valid`` length is computed FIRST from the per-event counts, THEN each
        field is truncated/padded with ``ak.pad_none`` + ``ak.fill_none`` per dtype
        and converted to a dense numpy ``(B, T)`` block. The structured array
        carries the fields in config order plus a ``valid`` bool field.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(structured (B, T) array, valid (B, T) bool)``.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        t_dim = self._mult[stream]
        # valid length FIRST: true per-event count, then truncate to T
        first = cols[fields[0]]
        counts = np.asarray(ak.num(first, axis=1))
        valid = np.arange(t_dim)[None, :] < np.minimum(counts, t_dim)[:, None]  # (B, T) bool

        gschema = self.schema.groups[stream] if self.schema is not None else None
        dtype_fields: list[tuple[str, np.dtype]] = []
        blocks: dict[str, np.ndarray] = {}
        for f in fields:
            arr = cols[f][:, :t_dim]  # truncate to served multiplicity (leading)
            padded = ak.pad_none(arr, t_dim, axis=1, clip=True)
            dt = np.dtype(gschema.fields[f]) if gschema is not None else None
            fill = self._pad_fill(dt, arr)
            dense = ak.to_numpy(ak.fill_none(padded, fill, axis=1))
            block = np.asarray(dense)
            if dt is not None:
                block = block.astype(dt, copy=False)
            blocks[f] = block
            dtype_fields.append((f, block.dtype))
        dtype_fields.append(("valid", np.dtype("bool")))
        raw = np.empty((b, t_dim), dtype=np.dtype(dtype_fields))
        for f in fields:
            raw[f] = blocks[f]
        raw["valid"] = valid
        return raw, valid

    def _assemble_scalar(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> np.ndarray:
        """Assemble a structured ``(B,)`` array from scalar event-level columns.

        Returns
        -------
        np.ndarray
            The structured ``(B,)`` array, fields in config order.
        """
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

    @staticmethod
    def _pad_fill(dt: np.dtype | None, arr: Any) -> Any:
        """The pad fill value for a field, by dtype kind (design risk note §11).

        float -> 0.0 (zeroed again after masking downstream); SIGNED int (labels)
        -> -1 sentinel (never a real class; folded to ``ignore_index=-1``);
        unsigned int -> 0; bool -> False.

        Returns
        -------
        Any
            The scalar fill value.
        """
        kind = dt.kind if dt is not None else np.asarray(arr.layout.content).dtype.kind
        if kind == "f":
            return 0.0
        if kind == "i":
            return _INT_PAD_SENTINEL
        if kind == "b":
            return False
        return 0  # unsigned ints / other: 0

    # -- pickling (fork is free; spawn re-binds in the worker) ----------------

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient probe state so the reader pickles under spawn contexts.

        Returns
        -------
        dict[str, Any]
            The picklable state (file table / read set reset; re-created at
            prepare/bind).
        """
        state = self.__dict__.copy()
        state.update(
            {"_table": None, "_num_rows": None, "_mult": {}, "_read_fields": {}, "schema": None}
        )
        return state


def _max_count(jagged: Any) -> int:
    """Max per-event multiplicity of an awkward jagged array (0 for empty).

    Returns
    -------
    int
        The largest ``len(event)`` across the array, or 0 if empty.
    """
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    counts = np.asarray(ak.num(jagged, axis=1))
    return int(counts.max()) if counts.size else 0
