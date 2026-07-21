"""`EasyjetReader` — a config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples.

A flat ntuple (one row = one event, single-jagged constituents); branch names come
from config verbatim (no aux-store prefix). Everything but the per-file offset index
and the per-file column read is inherited from `UprootReader`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.stream import OffsetIndex
from salt.core.data.uproot_reader import UprootGroupConfig, UprootReader
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.schema import GroupSchema, Schema

__all__ = ["EasyjetGroupConfig", "EasyjetReader"]


@dataclass(frozen=True)
class EasyjetGroupConfig(UprootGroupConfig):
    """Per-stream reader configuration for `EasyjetReader`.

    `branches` maps the v2 field name (``pt``) to the ROOT branch name in the file
    (``recojet_antikt4PFlow_pt_NOSYS``) — the ``_NOSYS`` suffix and collection
    prefix live here, so HH4b vs flavtag is a config change, not code. Field order
    = config dict order. ``truncate`` is the easyjet spelling of `pad_max` (leading
    N constituents of a jagged stream; None auto-resolves the file-wide max).
    """

    truncate: InitVar[int | None] = None

    def __post_init__(self, truncate: int | None) -> None:  # type: ignore[override]
        if truncate is not None:
            if self.pad_max is not None:
                raise ConfigError(
                    "EasyjetGroupConfig: give either 'pad_max' or 'truncate' (its alias), not both"
                )
            object.__setattr__(self, "pad_max", truncate)
        super().__post_init__()


@dataclass
class _FileEntry:
    """One file in the deterministic file table: path + cumulative offsets."""

    path: Path
    n: int
    start: int  # global offset of this file's first entry
    fields: dict[str, dict[str, str]] = field(default_factory=dict)  # stream -> {field: dtype}


class EasyjetReader(UprootReader):
    """Config-driven Reader for easyjet ``AnalysisMiniTree`` ROOT ntuples.

    Parameters
    ----------
    groups : Mapping[str, EasyjetGroupConfig | Mapping | ...]
        Stream name -> group config (``{branches:, jagged:, truncate:}``). The
        first jagged stream defines the served length axis. At least one group
        required.
    filename : str | Path | None, optional
        The source: a single ``.root`` file, a directory of them, or a glob. A
        directory globs to ``*.root``. May be supplied later via `with_source`.
    tree : str, optional
        The TTree name, by default ``"AnalysisMiniTree"``.
    num : int, optional
        Number of rows to serve; ``-1`` = all.

    Raises
    ------
    ConfigError
        On an empty / malformed group config, or no source file.
    SchemaError
        When a configured branch is missing, or a stream has inconsistent branch
        shapes (jagged vs scalar).
    """

    _JAGGED_NDIM = 2
    _DEP_NAME = "EasyjetReader"
    _DEP_EXTRA = "easyjet"

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
        self.groups: dict[str, UprootGroupConfig] = {
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

    def prepare(self) -> None:
        """Resolve source files, probe entry counts, and build the schema (idempotent).

        Resolves each jagged stream's served multiplicity ``T`` (`truncate` or the
        file-wide max) and builds the `Schema` from the first file's branch dtypes.
        """
        if self._table is not None:
            return
        self._require_deps()
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
                # for both TTree branches and RNTuple fields).
                for stream, cfg in self.groups.items():
                    fdtypes: dict[str, str] = {}
                    for fieldname, branch in cfg.branches.items():
                        if branch not in avail:
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} (field {fieldname!r}) not "
                                f"in {path.name!r}; tree {self.tree!r} has {len(avail)} branches"
                            )
                        arr = t[branch].array(library="ak")
                        is_jagged = arr.ndim >= self._JAGGED_NDIM
                        if is_jagged != cfg.jagged:
                            kind = "jagged" if is_jagged else "scalar"
                            raise SchemaError(
                                f"group {stream!r}: branch {branch!r} reads as {kind} but config "
                                f"says jagged={cfg.jagged} (group {stream!r})"
                            )
                        fdtypes[fieldname] = self._array_dtype_name(arr)
                        if cfg.jagged and cfg.pad_max is None and n > 0:
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
                cfg.pad_max if cfg.pad_max is not None else max(1, max_mult[stream])
            )
        assert schema_groups is not None
        self.schema = Schema(groups=schema_groups)
        self._table = table
        self._offsets = OffsetIndex([e.n for e in table])
        self._num_rows = num_available if self.num < 0 else self.num

    def _clone(self, filename: str | Path, num: int, stage: str | None) -> EasyjetReader:
        del stage  # easyjet has no per-stage cuts
        return EasyjetReader(groups=self.groups, filename=filename, tree=self.tree, num=num)

    def _read_stream_columns(
        self, stream: str, fields: list[str], start: int, stop: int
    ) -> dict[str, Any]:
        """Read the demanded branches over a global row range (multi-file stitched)."""
        self._require_deps()
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

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient probe state (incl. the offset index) so the reader pickles."""
        state = super().__getstate__()
        state["_offsets"] = None
        return state


def _max_count(jagged: Any) -> int:
    """Max per-event multiplicity of an awkward jagged array (0 for empty)."""
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    counts = np.asarray(ak.num(jagged, axis=1))
    return int(counts.max()) if counts.size else 0
