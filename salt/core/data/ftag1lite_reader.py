"""`FTAG1LiteReader` — a config-driven Reader for xAOD DAOD_FTAG1LITE POOL files.

The v2 modular-Reader boundary on a THIRD file type (design §2.4, §6.1): a new
format is a new `Reader` subclass — `Features` / `Labels` / `Normaliser` / the
model / ``salt2`` are all UNCHANGED. This reads ATLAS xAOD DAOD POOL files
(``CollectionTree`` with aux-store branches) via ``uproot`` and emits the same
``raw.<stream>`` structured numpy arrays + ``masks.<stream>`` pad masks the
`H5StructuredReader` / `EasyjetReader` produce, so the rest of the pipeline cannot
tell the three apart.

Three differences from the easyjet path drive the design:

- **The sample unit is the JET, but the file is keyed by EVENT.** FTAG1LITE stores
  ``std::vector<T>`` jet-level scalars (``[event][jet]``) and
  ``std::vector<std::vector<T>>`` constituent vectors (``[event][jet][track]``).
  Each small-R jet is one training sample (an H5 row). `prepare` builds an
  event→jet OFFSET INDEX (cumulative per-event jet counts) so a contiguous
  jet-slice maps to a covering EVENT range; `read` reads whole events, flattens
  ``[event][jet] → jet``, and slices to exactly the requested jets.

- **Constituents are DOUBLE-jagged.** A track stream is ``[event][jet][track]``;
  after flattening events away it is ``[jet][track]`` (depth-2 jagged), padded to
  ``pad_max`` per jet — ``ak.pad_none`` + ``ak.fill_none`` per dtype (float → 0.0,
  signed-int LABELS → ``-1`` sentinel, unsigned/counts → 0). The result is a
  structured ``(B, pad_max)`` array with a ``valid`` field + ``masks.<stream> =
  ~valid`` — exactly the H5 contract.

- **Branch names are config, not hardcoded, and prefixed by the aux store.** The
  reader takes a ``jet_collection`` (e.g. ``AntiKt4EMPFlowJets``) → the aux
  prefix ``<collection>AuxDyn.``; per-stream ``groups`` map the v2 FIELD name to
  the *bare* branch (``pt`` → ``AntiKt4EMPFlowJetsAuxDyn.pt``; ``d0`` →
  ``...AuxDyn.ft1l_trk_d0_bf16``). ``_bf16`` branches are AUTO-decoded to float32
  by uproot — no manual decode.

Cuts (plan 19): an optional `CutSpec` is evaluated at INDEX-BUILD over the
jet-level scalars, keeping only passing jets in the offset index — ``__len__`` and
every ``read`` slice are over the FILTERED set. Per-split cuts ride
``with_source(stage=...)``.

``uproot`` and ``awkward`` are imported LAZILY (the optional ``root`` reader
extra, reuse `_require_root_deps`); ``salt.core`` imports without them.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx
from salt.core.data.cuts import CutSpec
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.schema import GroupSchema, Schema

__all__ = ["FTAG1LiteGroupConfig", "FTAG1LiteReader"]


def _require_root_deps() -> None:
    """Import-time guard for the optional ROOT reader extra (reuse easyjet pattern).

    Raises a clear, actionable `ImportError` (pointing at ``salt[root]`` /
    ``salt[easyjet]``) instead of a bare ``ModuleNotFoundError`` from deep inside
    an array method. Cheap when the deps ARE present (cached imports).
    """
    try:
        import awkward  # noqa: F401
        import uproot  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FTAG1LiteReader requires uproot + awkward — install with:\n"
            "  pip install 'salt[root]'\n"
            "or directly:\n"
            "  pip install uproot awkward"
        ) from exc


# Padding sentinel for SIGNED-int (label) fields: never a real class, folded to
# ignore_index=-1 downstream. Floats pad to 0.0; unsigned/counts pad to 0; bool
# pads to False. The 'valid' field is set explicitly, never via these fills.
_INT_PAD_SENTINEL = -1


@dataclass(frozen=True)
class FTAG1LiteGroupConfig:
    """Per-stream reader configuration for `FTAG1LiteReader` (plan 19).

    `branches` maps the v2 FIELD name (``pt``, ``flavour_label``, ``d0``) to the
    BARE aux-store branch name (``pt``, ``HadronConeExclTruthLabelID``,
    ``ft1l_trk_d0_bf16``). The reader prepends the aux prefix
    (``<jet_collection>AuxDyn.``). Field order = config dict order (the
    structured-array field order).

    `jagged=False` is a jet-level SCALAR stream (``[event][jet]`` → ``(B,)``
    structured, no pad mask — the ``global_object`` analogue). `jagged=True` is a
    constituent stream (``[event][jet][track]`` → ``(B, pad_max)`` padded, with a
    ``valid`` field + pad mask). `pad_max` caps the per-jet constituent
    multiplicity (the leading N constituents); None auto-resolves the file-wide
    max in `prepare`.
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
        """Whether this is a jet-level (global, non-sequence) stream.

        The writers callback (design §8) introspects ``groups[stream].
        global_object`` to split global (no pad mask) vs sequence (padded,
        masked) streams. A non-jagged FTAG1LITE stream IS the global object
        (one value per jet), so ``global_object == not jagged``.

        Returns
        -------
        bool
            True for jet-level scalar streams, False for constituent streams.
        """
        return not self.jagged


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + per-event jet structure.

    `event_start` is the global offset of this file's first EVENT; `jet_start` is
    the global offset of this file's first kept JET. `kept_jets` are the local
    (per-file) flat-jet indices that survive the cuts, in flat order; `njets` is
    the per-event count of KEPT jets (length = n_events), used to map a
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
    """Config-driven Reader for xAOD DAOD_FTAG1LITE POOL files (plan 19, Track C).

    Each small-R jet (``AntiKt4EMPFlowJets`` by default) is one sample. Jet-level
    ``std::vector<T>`` scalars and ``std::vector<std::vector<T>>`` constituent
    vectors are read via uproot from ``CollectionTree``, flattened
    ``[event][jet] → jet``, and emitted as the standard ``raw.<stream>`` /
    ``masks.<stream>`` structured arrays.

    Parameters
    ----------
    groups : Mapping[str, FTAG1LiteGroupConfig | Mapping | ...]
        Stream name → group config (``{branches:, jagged:, pad_max:}``). At least
        ONE non-jagged (jet-level) stream is required — it carries the served
        length axis (the flattened jet count) and the cut variables. By convention
        the jet-level stream is named ``jets``.
    jet_collection : str, optional
        The jet collection whose aux store is read, by default
        ``"AntiKt4EMPFlowJets"``. The aux prefix is ``<jet_collection>AuxDyn.``.
    filename : str | Path | None, optional
        The source: a single ``.root`` (or ``.pool.root``) file, a directory of
        them, or a glob. May be omitted and supplied via `with_source`.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    num : int, optional
        Number of JETS to serve (post-cut); ``-1`` = all.
    cuts : CutSpec | None, optional
        Index-build-time jet eligibility (plan 19). Evaluated in `prepare` over
        the jet-level scalars; only passing jets enter the index.
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

    # -- config helpers ------------------------------------------------------

    @staticmethod
    def _parse_group(
        stream: str, cfg: FTAG1LiteGroupConfig | Mapping[str, Any] | Any
    ) -> FTAG1LiteGroupConfig:
        """Normalise one group config entry to a `FTAG1LiteGroupConfig`.

        Returns
        -------
        FTAG1LiteGroupConfig
            The parsed config.

        Raises
        ------
        ConfigError
            On unknown keys or a missing/empty ``branches`` mapping.
        """
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
        """The full aux-store branch name for a bare field branch.

        Returns
        -------
        str
            ``<jet_collection>AuxDyn.<bare>``.
        """
        return f"{self.aux_prefix}{bare}"

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

        Sequence dims are concrete when ``pad_max`` is set, else symbolic
        ``T:<stream>``; ``meta.rows`` is TEST-only (mirrors `EasyjetReader`).

        Returns
        -------
        IO
            The declared interface.
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

    # -- file-touching lifecycle hooks (design §2.3) --------------------------

    def _resolve_files(self) -> list[Path]:
        """Glob the source into a deterministic, sorted list of ROOT files.

        Matches ``*.root`` AND ``*.pool.root`` (the FTAG1LITE convention) when a
        directory is given.

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
        """Resolve files, build the event→jet offset index (post-cut), and the schema.

        Main-process hook (called lazily by ``__len__``, eagerly by the
        datamodule). Idempotent. For each file: probe the jet-level branches,
        evaluate the (per-stage) `CutSpec` over a structured jet-scalar record to
        get the kept-jet mask, store per-event kept-jet counts + the kept flat-jet
        indices, and accumulate cumulative jet offsets. Resolves each jagged
        stream's served ``pad_max`` (config or file-wide max over KEPT jets) and
        builds the `Schema` from the first file's branch dtypes (+ the auto
        ``valid`` field on jagged streams).

        Raises
        ------
        ValueError
            If ``num`` requests more jets than available (post-cut).
        SchemaError
            If a configured branch is missing, or a jagged branch reads as scalar
            (or vice versa).
        KeyError
            If a `CutSpec` references a field absent from the jet-level scalars.
        """
        if self._table is not None:
            return
        _require_root_deps()
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
        """Evaluate the CutSpec → kept flat-jet indices + per-event kept counts.

        With no cuts every jet is kept. Otherwise a structured jet-scalar record
        is built from the flat per-jet columns and `CutSpec.eligible` gives the
        keep mask; the per-event kept count is recomputed from the original
        per-event jet boundaries.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(kept_flat_jet_indices (M,), per_event_kept_counts (n_events,))``.
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
        """Max per-jet constituent multiplicity over KEPT jets for one stream.

        Reads the FIRST branch of the stream (its multiplicity defines the
        stream's per-jet count), flattens ``[event][jet] → [jet]``, takes the
        kept jets, and returns the largest track count.

        Returns
        -------
        int
            The max per-jet constituent count over kept jets (0 if none).
        """
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
        """Native-endian numpy dtype name for a (single- or double-jagged) awkward array.

        Both jet-level scalars (``[event][jet]``, ndim=2) and constituents
        (``[event][jet][track]``, ndim=3) are ragged, so the array is FULLY
        flattened (``axis=None``) to reach the innermost content dtype (the
        per-jet / per-track scalar type) before reading its numpy dtype. ROOT
        stores big-endian; the structured array uses NATIVE byte order so the
        rest of the pipeline sees ordinary native arrays.

        Returns
        -------
        str
            A ``np.dtype(name)``-constructible name.
        """
        _require_root_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        del is_jagged  # both jet-level and constituent arrays are ragged here
        flat = ak.flatten(arr, axis=None)
        dtype = np.asarray(ak.to_numpy(flat)).dtype
        return np.dtype(dtype.newbyteorder("=")).name

    def __len__(self) -> int:
        """Return the number of JETS served (post-cut), resolving on first call.

        Returns
        -------
        int
            The jet (row) count.
        """
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    @property
    def source_path(self) -> Path:
        """The first resolved source file.

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
    ) -> FTAG1LiteReader:
        """Clone this reader for another source + stage (datamodule pattern, design §6.1).

        Config-only (no file I/O): group configs / collection / cuts are shared;
        the source binding, ``num``, and ``stage`` change. ``stage`` selects the
        per-split cuts at the next `prepare` (the index is rebuilt for that
        split). ``vds_path`` is accepted for `Reader` API parity but unused.

        Returns
        -------
        FTAG1LiteReader
            A fresh, unbound reader instance (same instance ``name``).
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

    # -- per-worker binding (design §2.3) -------------------------------------

    def bind(self, ctx: WorkerCtx) -> None:
        """Per-worker setup: resolve files + record the demand-narrowed read set.

        The per-stream read set is ``demanded fields`` (from `WorkerCtx.
        read_fields`) intersected with the configured branches; an empty demand
        falls back to ALL configured branches. Demanded fields absent from the
        config raise before any step.

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
                        f"group {stream!r} (configured: {sorted(cfg.branches)}) (plan 19)"
                    )
            self._read_fields[stream] = demanded

    # -- jet-slice -> event-range mapping -------------------------------------

    def _covering_events(self, entry: _FileEntry, jlo: int, jhi: int) -> tuple[int, int, int]:
        """Map a local KEPT-jet range ``[jlo, jhi)`` to a covering local event range.

        Uses the per-event KEPT-jet counts: the cumulative sum gives each event's
        kept-jet span; the covering event range is the smallest ``[e0, e1)`` whose
        kept jets include ``[jlo, jhi)``. Also returns the offset of the first
        kept jet of ``e0`` within the flattened-event block, so the caller can
        slice the read block to exactly ``[jlo, jhi)``.

        Returns
        -------
        tuple[int, int, int]
            ``(e0, e1, jet_offset_in_block)``: local event start/stop and the
            local jet-offset of ``jlo`` within the ``[e0, e1)`` flattened block.
        """
        cum = np.concatenate([[0], np.cumsum(entry.njets)])  # (n_events+1,)
        # first event whose cumulative END > jlo
        e0 = int(np.searchsorted(cum, jlo, side="right") - 1)
        # last event whose cumulative START < jhi
        e1 = int(np.searchsorted(cum, jhi, side="left"))
        jet_offset_in_block = jlo - int(cum[e0])
        return e0, e1, jet_offset_in_block

    # -- the per-batch read (design §6.1) -------------------------------------

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one contiguous JET-slice, translating jets→covering events per file.

        ``rows`` is a contiguous slice over the FILTERED jet index. The slice is
        split per file; within each file the covering local event range is read,
        events flattened ``[event][jet] → jet``, the kept-jet mask applied, and the
        block sliced to exactly the requested jets. Jagged streams are padded to
        ``pad_max`` (``valid`` length first, then per-dtype fills) → structured
        ``(B, pad_max)`` + ``masks.<stream> = ~valid``. Scalar streams →
        structured ``(B,)``. ``meta.rows`` is produced in TEST.

        Returns
        -------
        dict[str, np.ndarray]
            The produced keys, as a flat dotted dict.
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
        return [f for f in cfg.branches if f in names]

    def _read_stream_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read a stream's demanded branches over a global JET range (multi-file).

        For each file overlapping ``[start, stop)`` the covering local event range
        is read, events flattened away, the per-file kept-jet mask applied, and the
        block sliced to the file's contribution. Per-file blocks are concatenated
        in global jet order. Jet-level fields return a flat awkward/numpy array of
        length ``stop-start``; constituent fields return a depth-1 jagged
        ``[jet][track]`` array.

        Returns
        -------
        dict[str, Any]
            ``{field: array}`` of total length ``stop - start``, in global jet order.
        """
        _require_root_deps()
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

    def _assemble_jagged(
        self, stream: str, fields: list[str], cols: dict[str, Any], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad constituent columns to ``pad_max`` and assemble a structured ``(B, T)`` array.

        The ``valid`` length is computed FIRST from the per-jet track counts, THEN
        each field is truncated/padded with ``ak.pad_none`` + ``ak.fill_none`` per
        dtype and converted to dense numpy ``(B, T)``. The structured array carries
        the fields in config order plus a ``valid`` bool field.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(structured (B, T) array, valid (B, T) bool)``.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        t_dim = self._mult[stream]
        first = cols[fields[0]]
        counts = np.asarray(ak.num(first, axis=1)) if b > 0 else np.zeros(0, dtype=np.int64)
        valid = np.arange(t_dim)[None, :] < np.minimum(counts, t_dim)[:, None]  # (B, T) bool

        gschema = self.schema.groups[stream] if self.schema is not None else None
        dtype_fields: list[tuple[str, np.dtype]] = []
        blocks: dict[str, np.ndarray] = {}
        for f in fields:
            arr = cols[f][:, :t_dim]  # truncate to served pad_max (leading)
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
        """Assemble a structured ``(B,)`` array from jet-level scalar columns.

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
        """The pad fill value for a field, by dtype kind (plan 19 sentinel rule).

        float → 0.0 (zeroed again after masking downstream); SIGNED int (labels)
        → -1 sentinel (never a real class; folded to ``ignore_index=-1``);
        unsigned int / counts → 0; bool → False.

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
        return 0  # unsigned ints / counts: 0

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
        state.update({
            "_table": None,
            "_num_rows": None,
            "_mult": {},
            "_read_fields": {},
            "schema": None,
        })
        return state
