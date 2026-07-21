"""`XAODReader` — the shared xAOD (POOL ``CollectionTree``) `UprootReader` base.

Reads jet-keyed slices out of event-keyed aux-store branches: jet-level
``std::vector<T>`` scalars and constituent streams (either direct
``std::vector<std::vector<T>>`` decorations, or ElementLink-dereferenced target
containers), flattening ``[event][jet] -> jet`` with optional jet-level cuts.
Concrete DAOD flavors (`FTAG1LiteReader`, `PhysliteReader`) are thin presets that
only set the default jet collection.
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

__all__ = ["XAODReader"]


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


class XAODReader(UprootReader):
    """Config-driven `UprootReader` for xAOD POOL (``CollectionTree``) files.

    Each small-R jet (``<jet_collection>``) is one sample. Jet-level scalars are
    read from ``<jet_collection>AuxDyn.<branch>`` and flattened ``[event][jet] ->
    jet``. Constituent streams are either direct double-jagged aux decorations, or
    ElementLink-dereferenced (``link_branch`` + ``target_collection`` on the group
    config): the per-jet link vector's ``m_persIndex`` gathers the target
    container's columns.

    Parameters
    ----------
    groups : Mapping[str, UprootGroupConfig | Mapping | ...]
        Stream name -> group config. At least one non-jagged (jet-level) stream is
        required — it carries the served length axis and cut variables (by
        convention named ``jets``).
    jet_collection : str, optional
        The jet collection whose aux store is read. The aux prefix is
        ``<jet_collection>AuxDyn.``.
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
    _DEP_NAME = "XAODReader"
    _DEP_EXTRA = "root"
    #: default jet collection; presets override.
    JET_COLLECTION = "AntiKt4EMPFlowJets"

    def __init__(
        self,
        groups: Mapping[str, UprootGroupConfig | Mapping[str, Any] | Any],
        jet_collection: str | None = None,
        filename: str | Path | None = None,
        tree: str = "CollectionTree",
        num: int = -1,
        cuts: CutSpec | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__()
        if not groups:
            raise ConfigError(f"{type(self).__name__} needs at least one group (plan 19)")
        self.jet_collection = (
            str(jet_collection) if jet_collection is not None else self.JET_COLLECTION
        )
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
                f"{type(self).__name__} needs at least one jet-level (jagged=False) stream to "
                "carry the served jet axis + cut variables (plan 19)"
            )
        # the first scalar stream defines the jet axis / cut scalars
        self.jet_stream = scalar_streams[0]
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see base __getstate__)
        self._table: list[_FileEntry] | None = None
        self._num_rows: int | None = None
        self._mult: dict[str, int] = {}  # stream -> served pad_max
        self._read_fields: dict[str, dict[str, str]] = {}

    def _branch(self, bare: str) -> str:
        """The full jet-collection aux-store branch name for a bare field branch."""
        return f"{self.aux_prefix}{bare}"

    @staticmethod
    def _target_branch(cfg: UprootGroupConfig, bare: str) -> str:
        """The full target-container aux-store branch for an ElementLink constituent field."""
        return f"{cfg.target_collection}AuxDyn.{bare}"

    def _clone(self, filename: str | Path, num: int, stage: str | None) -> XAODReader:
        return type(self)(
            groups=self.groups,
            jet_collection=self.jet_collection,
            filename=filename,
            tree=self.tree,
            num=num,
            cuts=self.cuts,
            stage=stage,
        )

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
                jet_scalar_cols: dict[str, np.ndarray] = {}
                jet_counts: np.ndarray | None = None
                for stream, cfg in self.groups.items():
                    fdtypes = self._probe_stream_fields(t, avail, path, stream, cfg)
                    if not cfg.jagged and stream == self.jet_stream:
                        for fieldname, bare in cfg.branches.items():
                            arr = t[self._branch(bare)].array(library="ak")
                            jet_scalar_cols[fieldname] = ak.to_numpy(ak.flatten(arr, axis=1))
                            if jet_counts is None:
                                jet_counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
                    entry.fields[stream] = fdtypes

                # also read any cut fields not already in jet_scalar_cols
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

    def _probe_stream_fields(
        self, t: Any, avail: set[str], path: Path, stream: str, cfg: UprootGroupConfig
    ) -> dict[str, str]:
        """Validate a stream's branches + capture per-field dtypes (linked or direct)."""
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        del ak  # imported for parity with the read path; dtype probing via base helper
        fdtypes: dict[str, str] = {}
        if cfg.is_linked:
            link = self._branch(cfg.link_branch)  # type: ignore[arg-type]
            if link not in avail:
                raise SchemaError(
                    f"group {stream!r}: link branch {link!r} not in {path.name!r} "
                    f"(ElementLink dereference needs the jet link vector) (plan 54)"
                )
            for fieldname, bare in cfg.branches.items():
                branch = self._target_branch(cfg, bare)
                if branch not in avail:
                    raise SchemaError(
                        f"group {stream!r}: target branch {branch!r} (field {fieldname!r}) not "
                        f"in {path.name!r}; ElementLink target fields live in "
                        f"{cfg.target_collection!r} (plan 54)"
                    )
                arr = t[branch].array(library="ak")  # [event][track]
                fdtypes[fieldname] = self._array_dtype_name(arr)
            fdtypes["valid"] = "bool"
            return fdtypes
        for fieldname, bare in cfg.branches.items():
            branch = self._branch(bare)
            if branch not in avail:
                raise SchemaError(
                    f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                    f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                )
            arr = t[branch].array(library="ak")
            is_jagged = arr.ndim >= self._JAGGED_NDIM
            if is_jagged != cfg.jagged:
                kind = "constituent (double-jagged)" if is_jagged else "jet-level scalar"
                raise SchemaError(
                    f"group {stream!r}: branch {branch!r} reads as {kind} (ndim="
                    f"{arr.ndim}) but config says jagged={cfg.jagged} (plan 19)"
                )
            fdtypes[fieldname] = self._array_dtype_name(arr)
        if cfg.jagged:
            fdtypes["valid"] = "bool"
        return fdtypes

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
        names = list(jet_scalar_cols)
        dtype = np.dtype([(nm, jet_scalar_cols[nm].dtype) for nm in names])
        rec = np.empty(n_jets_total, dtype=dtype)
        for nm in names:
            rec[nm] = jet_scalar_cols[nm]
        keep = self.cuts.eligible(rec, self.stage)  # (n_jets_total,) bool
        kept = np.flatnonzero(keep).astype(np.int64)
        bounds = np.concatenate([[0], np.cumsum(jet_counts)])
        per_event_kept = np.add.reduceat(keep.astype(np.int64), bounds[:-1])
        per_event_kept = np.where(jet_counts == 0, 0, per_event_kept).astype(np.int64)
        return kept, per_event_kept

    def _stream_max_mult(self, path: Path, stream: str, kept: np.ndarray) -> int:
        """Max per-jet constituent multiplicity over kept jets.

        For a direct stream this is the per-jet track count of the stream's first
        branch; for an ElementLink stream it is the per-jet link count.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        cfg = self.groups[stream]
        branch = (
            self._branch(cfg.link_branch)  # type: ignore[arg-type]
            if cfg.is_linked
            else self._branch(next(iter(cfg.branches.values())))
        )
        with uproot.open(f"{path}:{self.tree}") as t:
            arr = t[branch].array(library="ak")
        flat = ak.flatten(arr, axis=1)  # [jet][track] or [jet][link]
        if cfg.is_linked:
            # served multiplicity is the NON-NULL link count per jet (nulls are dropped)
            pkey = _pers_key(flat)
            flat = flat if pkey is None else _pers_index(flat)[pkey != 0]
        counts = ak.to_numpy(ak.num(flat, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

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
        ElementLink streams dereference the target container by ``m_persIndex`` per jet.
        """
        self._require_deps()
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)
        import uproot  # noqa: PLC0415 - optional reader extra (lazy)

        assert self._table is not None
        cfg = self.groups[stream]
        per_field_chunks: dict[str, list[Any]] = {f: [] for f in fields}

        for entry in self._table:
            file_jlo = entry.jet_start
            file_jhi = entry.jet_start + int(entry.kept_jets.size)
            lo = max(start, file_jlo)
            hi = min(stop, file_jhi)
            if lo >= hi:
                continue
            local_jlo = lo - file_jlo
            local_jhi = hi - file_jlo
            e0, e1, jet_off = self._covering_events(entry, local_jlo, local_jhi)
            orig_cum = np.concatenate([[0], np.cumsum(entry.orig_njets)])
            block_flat_start = int(orig_cum[e0])
            block_flat_stop = int(orig_cum[e1])
            kept = entry.kept_jets
            in_block = (
                kept[(kept >= block_flat_start) & (kept < block_flat_stop)] - block_flat_start
            )
            sel = in_block[jet_off : jet_off + (hi - lo)]
            with uproot.open(f"{entry.path}:{self.tree}") as t:
                if cfg.is_linked:
                    block = self._read_linked_block(t, cfg, fields, e0, e1)
                else:
                    block = {
                        f: ak.flatten(
                            t[self._branch(cfg.branches[f])].array(
                                entry_start=e0, entry_stop=e1, library="ak"
                            ),
                            axis=1,
                        )
                        for f in fields
                    }
            for f in fields:
                per_field_chunks[f].append(block[f][sel])
        cols: dict[str, Any] = {}
        for f in fields:
            chunks = per_field_chunks[f]
            cols[f] = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return cols

    def _read_linked_block(
        self, t: Any, cfg: UprootGroupConfig, fields: list[str], e0: int, e1: int
    ) -> dict[str, Any]:
        """Dereference an ElementLink constituent stream over event block ``[e0, e1)``.

        Reads the per-jet link vector (``m_persIndex`` into ``target_collection``),
        gathers each demanded target column by index, and returns ``[jet][track]``
        awkward arrays (events flattened away) ready for the kept-jet ``sel``.
        """
        import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

        links = t[self._branch(cfg.link_branch)].array(  # type: ignore[arg-type]
            entry_start=e0, entry_stop=e1, library="ak"
        )  # [event][jet][link] (struct m_persKey/m_persIndex, or plain int for synthetic)
        pidx = _pers_index(links)  # [event][jet][link] int (m_persIndex into the target)
        pkey = _pers_key(links)  # [event][jet][link] uint | None
        _check_single_pers_key(pkey, cfg.link_branch)  # type: ignore[arg-type]
        # drop null ElementLinks (m_persKey==0): PHYSLITE thins the target container, so
        # ghost links into thinned tracks are null and are not real constituents.
        if pkey is not None:
            pidx = pidx[pkey != 0]  # [event][jet][valid_link]
        # shift each event's local indices onto the flattened target track axis (a
        # per-event scalar offset broadcast over [jet][link]), flatten events away, then
        # numpy-gather + re-impose the [jet][link] grouping (avoids nested ak fancy-index).
        block: dict[str, Any] = {}
        for f in fields:
            tgt = t[self._target_branch(cfg, cfg.branches[f])].array(
                entry_start=e0, entry_stop=e1, library="ak"
            )  # [event][track]
            counts = ak.to_numpy(ak.num(tgt, axis=1))
            offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]  # (n_events,) event starts
            global_idx = pidx + ak.Array(offsets)  # broadcast [event] over [event][jet][link]
            gidx_jets = ak.flatten(global_idx, axis=1)  # [jet][link] in flat jet order
            links_per_jet = ak.to_numpy(ak.num(gidx_jets, axis=1))  # [jet] -> #links
            gidx_1d = ak.to_numpy(ak.flatten(gidx_jets, axis=None)).astype(np.int64)
            vals_1d = np.asarray(ak.flatten(tgt, axis=1))[gidx_1d]  # 1-D gathered
            block[f] = ak.unflatten(vals_1d, links_per_jet)  # [jet][link]
        return block


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
            "single target container per file (plan 54)"
        )
