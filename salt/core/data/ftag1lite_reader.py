"""`FTAG1LiteReader` — a config-driven Reader for xAOD DAOD_FTAG1LITE POOL files.

Reads jet-keyed slices out of event-keyed ``CollectionTree`` aux-store branches via
uproot (one row = one jet: event->jet flattening + optional jet-level cuts), emitting
the same ``raw.<stream>`` / ``masks.<stream>`` arrays as the H5 / easyjet readers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.cuts import CutSpec
from salt.core.data.stream import OffsetIndex
from salt.core.data.uproot_reader import UprootGroupConfig, UprootReader
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.schema import GroupSchema, Schema

__all__ = ["FTAG1LiteGroupConfig", "FTAG1LiteReader"]


@dataclass(frozen=True)
class FTAG1LiteGroupConfig(UprootGroupConfig):
    """Per-stream reader configuration for `FTAG1LiteReader`.

    `branches` maps the v2 field name (``pt``, ``flavour_label``, ``d0``) to the
    bare aux-store branch name; the reader prepends ``<jet_collection>AuxDyn.``.
    Field order = config dict order. `jagged=False` is a jet-level scalar stream
    (``[event][jet]`` -> ``(B,)``); `jagged=True` is a constituent stream
    (``[event][jet][track]`` -> ``(B, pad_max)`` padded). `pad_max` caps the
    per-jet constituent multiplicity; None auto-resolves the file-wide max.
    """


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + per-event jet structure.

    `event_start` is the global offset of this file's first event; `jet_start` is
    the global offset of this file's first kept jet. `kept_jets` are the local
    (per-file) flat-jet indices that survive the cuts, in flat order; `njets` is
    the per-event count of kept jets (used to map a jet-slice back to the covering
    event range); `orig_njets` is the per-event pre-cut jet count.
    """

    path: Path
    n_events: int
    event_start: int
    jet_start: int
    kept_jets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    njets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    orig_njets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fields: dict[str, dict[str, str]] = field(default_factory=dict)  # stream -> {field: dtype}


class FTAG1LiteReader(UprootReader):
    """Config-driven Reader for xAOD DAOD_FTAG1LITE POOL files.

    Each small-R jet (``AntiKt4EMPFlowJets`` by default) is one sample. Jet-level
    ``std::vector<T>`` scalars and ``std::vector<std::vector<T>>`` constituent
    vectors are read from ``CollectionTree``, flattened ``[event][jet] -> jet``,
    and emitted as the standard ``raw.<stream>`` / ``masks.<stream>`` arrays.

    Parameters
    ----------
    groups : Mapping[str, FTAG1LiteGroupConfig | Mapping | ...]
        Stream name -> group config (``{branches:, jagged:, pad_max:}``). At least
        one non-jagged (jet-level) stream is required — it carries the served
        length axis and cut variables. By convention it's named ``jets``.
    jet_collection : str, optional
        The jet collection whose aux store is read, by default
        ``"AntiKt4EMPFlowJets"``. The aux prefix is ``<jet_collection>AuxDyn.``.
    filename : str | Path | None, optional
        A single ``.root`` / ``.pool.root`` file, a directory of them, or a glob.
    tree : str, optional
        The TTree name, by default ``"CollectionTree"``.
    num : int, optional
        Number of jets to serve (post-cut); ``-1`` = all.
    cuts : CutSpec | None, optional
        Index-build-time jet eligibility, evaluated in `prepare` over the jet-level
        scalars; only passing jets enter the index.
    stage : str | None, optional
        The bound stage (``"train"``/``"val"``/``"test"``); selects per-split cuts.

    Raises
    ------
    ConfigError
        On an empty / malformed group config, no jet-level stream, or no source.
    SchemaError
        When a configured branch is missing, or a stream's jaggedness disagrees
        with the config.
    """

    _JAGGED_NDIM = 3
    _DEP_NAME = "FTAG1LiteReader"
    _DEP_EXTRA = "root"

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
        self.groups: dict[str, UprootGroupConfig] = {
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

    def _branch(self, bare: str) -> str:
        """The full aux-store branch name for a bare field branch."""
        return f"{self.aux_prefix}{bare}"

    def prepare(self) -> None:
        """Resolve files, build the event->jet offset index (post-cut), and the schema
        (idempotent); evaluates the (per-stage) `CutSpec` over jet-level scalars to keep
        only passing jets, then resolves each jagged stream's served ``pad_max``.
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
                        is_jagged = arr.ndim >= self._JAGGED_NDIM
                        if is_jagged != cfg.jagged:
                            kind = (
                                "constituent (double-jagged)" if is_jagged else "jet-level scalar"
                            )
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} reads as {kind} (ndim="
                                f"{arr.ndim}) but config says jagged={cfg.jagged} (plan 19)"
                            )
                        fdtypes[fieldname] = self._array_dtype_name(arr)
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

        With no cuts every jet is kept; otherwise `CutSpec.eligible` gives the keep
        mask over a structured jet-scalar record built from the flat columns.
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
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        cfg = self.groups[stream]
        first_bare = next(iter(cfg.branches.values()))
        with uproot.open(f"{path}:{self.tree}") as t:
            arr = t[self._branch(first_bare)].array(library="ak")
        flat = ak.flatten(arr, axis=1)  # [jet][track]
        counts = ak.to_numpy(ak.num(flat, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

    def _clone(self, filename: str | Path, num: int, stage: str | None) -> FTAG1LiteReader:
        return FTAG1LiteReader(
            groups=self.groups,
            jet_collection=self.jet_collection,
            filename=filename,
            tree=self.tree,
            num=num,
            cuts=self.cuts,
            stage=stage,
        )

    def _covering_events(self, entry: _FileEntry, jlo: int, jhi: int) -> tuple[int, int, int]:
        """Map a local kept-jet range ``[jlo, jhi)`` to its covering local event range
        ``(e0, e1)``, plus the kept-jet offset of ``e0`` for exact slicing.
        """
        cum = np.concatenate([[0], np.cumsum(entry.njets)])  # (n_events+1,)
        return OffsetIndex.covering_range(cum, jlo, jhi)

    def _read_stream_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read a stream's demanded branches over a global jet range (multi-file): per
        file, reads the covering event range, flattens, applies the kept-jet mask, and
        slices to the file's contribution; blocks are concatenated in jet order.
        """
        self._require_deps()
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
