"""Deterministic two-collection (small-R + large-R) FTAG1LITE-like ROOT corpus for the
`FTAG1LITEReader` tests: 2 container dirs x N ``DAOD_FTAG1LITE.*.pool.root`` files.

Real FTAG1LITE stores per-jet constituents as ``vector<vector<T>>`` (``[event][jet][const]``).
uproot 5.7.6's TTree writer cannot produce that: it rejects ``var * var * T`` outright and
writes ``var * K * T`` in a form uproot itself cannot read back. So every constituent branch
(``ft1l_trk_*``, ``ft1l_tower_*``, ``ft1lR_trk_*``) is written as ``var * T`` per event (that
event's jets' constituents concatenated jet by jet), plus ONE per-jet ``int32`` count branch
per constituent group (``ft1l_trk_n``, ``ft1l_tower_n``, ``ft1lR_trk_n``).

`_NestedTree` / `_NestedBranch` reshape those two real uproot reads (flat values + per-jet
counts, same entry window) into ``[event][jet][const]`` below the `UprootReader` tree API.
`SeamUprootReader` installs the adaptor at `UprootReader._tree` (the one cached open on the
read path) and at the three tree-receiving helpers of `prepare`; `SeamFTAG1LITEReader`
re-classes its sides to it. prepare / probe / read are the real reader code. Every seam class
is module-level, so spawned ranks and forkserver workers unpickle it by name.

The corpus is written as a TTree via ``mktree``: uproot >= 5.7 dict assignment writes an
RNTuple, which parses the dotted aux branch names as sub-fields.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from salt.data.readers.cuts import GlobalObjectCuts
from salt.data.readers.ftag1lite_reader import FTAG1LITEReader
from salt.data.readers.uproot_reader import UprootReader

SMALL_PREFIX = "AntiKt4EMPFlowJetsAuxDyn."
LARGE_PREFIX = "AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsAuxDyn."
EVENT_PREFIX = "EventInfoAuxDyn."
TREE = "CollectionTree"

TRACK_PAD = 6
TOWER_PAD = 4

SMALL_TYPE = 0
LARGE_TYPE = 1

CONTAINERS = (
    "mc23_13p6TeV_mc23_13p6TeV.601190.PhPy8EG_AZNLO_Zmumu.deriv.DAOD_FTAG1LITE."
    "e8514_s4159_r15530_p7356",
    "mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_FTAG1LITE."
    "e8514_s4162_r15540_p7356",
)
"""Two real-style FTAG1LITE container directory names, two distinct DSIDs."""

_TRK_FLOAT = ("d0", "z0SinTheta", "qOverP")
_TRK_INT = ("ftagTruthOriginLabel", "ftagTruthVertexIndex", "ftagTruthTypeLabel")
TRACK_FIELDS = _TRK_FLOAT + _TRK_INT

SMALL_GROUPS: dict[str, Any] = {
    "jets": {
        "prefix": SMALL_PREFIX,
        "jagged": False,
        "branches": {
            "pt": "pt",
            "eta": "eta",
            "HadronGhostTruthLabelID": "HadronGhostTruthLabelID",
            "PartonTruthLabelID": "PartonTruthLabelID",
            "ptFromTruthDressedWZJet": "ptFromTruthDressedWZJet",
            "matchedToTruthDressedWZJet": "matchedToTruthDressedWZJet",
        },
    },
    "tracks": {
        "prefix": SMALL_PREFIX,
        "jagged": True,
        "pad_max": TRACK_PAD,
        "branches": {f: f"ft1l_trk_{f}" for f in TRACK_FIELDS},
    },
    "towers": {
        "prefix": SMALL_PREFIX,
        "jagged": True,
        "pad_max": TOWER_PAD,
        "branches": {"pt": "ft1l_tower_pt", "deta": "ft1l_tower_deta"},
    },
    "event": {
        "prefix": EVENT_PREFIX,
        "jagged": False,
        "broadcast": True,
        "branches": {"eventNumber": "eventNumber"},
    },
}

LARGE_GROUPS: dict[str, Any] = {
    "jets": {
        "prefix": LARGE_PREFIX,
        "jagged": False,
        "branches": {
            "pt": "pt",
            "eta": "eta",
            "m": "m",
            "R10TruthLabel_R22v1": "R10TruthLabel_R22v1",
            "GhostBHadronsFinalCount": "GhostBHadronsFinalCount",
            "GhostCHadronsFinalCount": "GhostCHadronsFinalCount",
        },
    },
    "tracks": {
        "prefix": LARGE_PREFIX,
        "jagged": True,
        "pad_max": TRACK_PAD,
        "branches": {f: f"ft1lR_trk_{f}" for f in TRACK_FIELDS},
    },
    "event": {
        "prefix": EVENT_PREFIX,
        "jagged": False,
        "broadcast": True,
        "branches": {"eventNumber": "eventNumber"},
    },
}

_PER_SPLIT = {
    "train": ["eventNumber % 10 < 8"],
    "val": ["eventNumber % 10 == 8"],
    "test": ["eventNumber % 10 == 9"],
}

SMALL_CUTS: dict[str, Any] = {
    "global_cuts": ["HadronGhostTruthLabelID * 100 + abs(PartonTruthLabelID - 12) >= 9"],
    "per_split": _PER_SPLIT,
}
"""Six-class ghost-flavour eligibility (orphans ``ghost == 0 & parton in [4, 20]`` cut)."""

LARGE_CUTS: dict[str, Any] = {
    "global_cuts": ["R10TruthLabel_R22v1 != 3"],
    "per_split": _PER_SPLIT,
}
"""Large-R eligibility: ``R10TruthLabel_R22v1 == 3`` is cut."""

SPLIT_OF_RESIDUE = {**dict.fromkeys(range(8), "train"), 8: "val", 9: "test"}


@dataclass(frozen=True)
class JointRow:
    """One expected served row: where it comes from and its identity."""

    file: int
    entry: int
    jet_type: int
    jet: int
    event_number: int
    pt: float

    @property
    def split(self) -> str:
        """The stage this row's eventNumber selects."""
        return SPLIT_OF_RESIDUE[self.event_number % 10]

    @property
    def identity(self) -> tuple[int, int, float]:
        """``(eventNumber, jet_type, pt)`` — unique per jet in this corpus."""
        return (self.event_number, self.jet_type, self.pt)


@dataclass
class JointCorpusFacts:
    """Ground truth for a built corpus (files in reader order: sorted by (dir, path))."""

    root: Path
    glob: str
    files: list[Path]
    n_events: list[int]
    small_kept: list[np.ndarray]
    large_kept: list[np.ndarray]
    rows: list[JointRow]
    ineligible_small_pt: set[float] = field(default_factory=set)
    ineligible_large_pt: set[float] = field(default_factory=set)

    def rows_for(self, stage: str | None = None) -> list[JointRow]:
        """Expected joint rows (reader order) under `stage`'s per-split cut (None: all)."""
        return [r for r in self.rows if stage is None or r.split == stage]

    def rows_in_file(self, file: int, stage: str | None = None) -> list[JointRow]:
        """Expected joint rows of one file, in order."""
        return [r for r in self.rows_for(stage) if r.file == file]


def corpus_glob(root: Path) -> str:
    """The reader source pattern covering every container dir."""
    return f"{Path(root)}/*/*.pool.root"


def _jagged(values: list[np.ndarray], dtype: Any) -> Any:
    """``var * dtype`` from per-event 1-D arrays."""
    import awkward as ak

    counts = np.array([len(v) for v in values], dtype=np.int64)
    flat = np.concatenate(values).astype(dtype) if counts.sum() > 0 else np.zeros(0, dtype=dtype)
    return ak.unflatten(flat, counts)


def _write_file(path: Path, ev: dict[str, Any]) -> None:
    """Write one file's per-event lists as an FTAG1LITE-style ``CollectionTree``."""
    import uproot

    branches: dict[str, Any] = {
        f"{EVENT_PREFIX}eventNumber": np.asarray(ev["eventNumber"], dtype=np.uint64)
    }
    for name, dtype in (
        ("pt", np.float32),
        ("eta", np.float32),
        ("HadronGhostTruthLabelID", np.int32),
        ("PartonTruthLabelID", np.int32),
        ("ptFromTruthDressedWZJet", np.float32),
        ("matchedToTruthDressedWZJet", np.int32),
    ):
        branches[f"{SMALL_PREFIX}{name}"] = _jagged(ev[f"s_{name}"], dtype)
    for name in TRACK_FIELDS:
        dtype = np.float32 if name in _TRK_FLOAT else np.int32
        branches[f"{SMALL_PREFIX}ft1l_trk_{name}"] = _jagged(ev[f"s_trk_{name}"], dtype)
        branches[f"{LARGE_PREFIX}ft1lR_trk_{name}"] = _jagged(ev[f"l_trk_{name}"], dtype)
    for name in ("pt", "deta"):
        branches[f"{SMALL_PREFIX}ft1l_tower_{name}"] = _jagged(ev[f"s_tower_{name}"], np.float32)
    # per-jet constituent counts: `_NestedBranch` nests the flat values above with these
    branches[f"{SMALL_PREFIX}ft1l_trk_n"] = _jagged(ev["s_trk_n"], np.int32)
    branches[f"{SMALL_PREFIX}ft1l_tower_n"] = _jagged(ev["s_tower_n"], np.int32)
    branches[f"{LARGE_PREFIX}ft1lR_trk_n"] = _jagged(ev["l_trk_n"], np.int32)
    for name, dtype in (
        ("pt", np.float32),
        ("eta", np.float32),
        ("m", np.float32),
        ("R10TruthLabel_R22v1", np.int32),
        ("GhostBHadronsFinalCount", np.int32),
        ("GhostCHadronsFinalCount", np.int32),
    ):
        branches[f"{LARGE_PREFIX}{name}"] = _jagged(ev[f"l_{name}"], dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        # uproot>=5.7 dict-assignment writes an RNTuple; FTAG1LITE is a TTree with dotted
        # aux branch names
        f.mktree(TREE, branches)


def _multiplicities(g_event: int, rng: np.random.Generator) -> tuple[int, int]:
    """Per-event (n_small, n_large): empty / large-only / small-only patterns, else random."""
    pattern = g_event % 7
    if pattern == 0:
        return 0, 0
    if pattern == 1:
        return 0, int(rng.integers(1, 3))
    if pattern == 2:
        return int(rng.integers(1, 6)), 0
    return int(rng.integers(0, 6)), int(rng.integers(0, 3))


def build_joint_corpus(
    root: Path,
    *,
    seed: int = 0,
    n_files_per_container: int = 2,
    n_events: int = 30,
    containers: Sequence[str] = CONTAINERS,
) -> JointCorpusFacts:
    """Write the corpus under `root` and return its ground truth.

    eventNumber is ``1000 + global event index`` (unique, every residue mod 10, exact
    in float32); every jet's ``pt`` is a distinct integer (small ``20000 + 10 k``, large
    ``250000 + 10 k``), so ``(eventNumber, jet_type, pt)`` identifies one jet.
    Ineligible jets: small ``ghost == 0 & parton == 9``; large ``R10 == 3``.
    `containers` are the container directory names, written in the given order.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    g_event = 0
    small_counter = 0
    large_counter = 0
    written: list[tuple[Path, dict[str, Any], list[np.ndarray], list[np.ndarray]]] = []
    inel_small: set[float] = set()
    inel_large: set[float] = set()
    for c, cname in enumerate(containers):
        for f in range(n_files_per_container):
            path = root / cname / f"DAOD_FTAG1LITE.{51281801 + c}._{f + 1:06d}.pool.root"
            ev: dict[str, Any] = defaultdict(list)  # per-event lists, one per branch
            s_kept: list[np.ndarray] = []
            l_kept: list[np.ndarray] = []
            for _ in range(n_events):
                ns, nl = _multiplicities(g_event, rng)
                ev["eventNumber"].append(1000 + g_event)
                # small-R jets
                pt = (20_000 + 10 * (small_counter + np.arange(ns))).astype(np.float32)
                small_counter += ns
                ghost = rng.choice(np.array([0, 4, 5, 15], dtype=np.int32), size=ns)
                parton = rng.choice(np.array([1, 2, 3, 21, 9], dtype=np.int32), size=ns)
                parton = np.where(ghost == 0, parton, 5).astype(np.int32)
                if ns > 0 and g_event % 11 == 5:
                    ghost[0], parton[0] = 0, 9  # a guaranteed orphan
                eligible = ~((ghost == 0) & (parton >= 4) & (parton <= 20))
                inel_small.update(float(p) for p in pt[~eligible])
                s_kept.append(np.flatnonzero(eligible))
                truth = (pt * rng.uniform(0.8, 1.2, size=ns)).astype(np.float32)
                unmatched = rng.uniform(size=ns) < 0.3
                truth[unmatched] = np.nan
                ev["s_pt"].append(pt)
                ev["s_eta"].append(rng.uniform(-2.5, 2.5, size=ns).astype(np.float32))
                ev["s_HadronGhostTruthLabelID"].append(ghost)
                ev["s_PartonTruthLabelID"].append(parton)
                ev["s_ptFromTruthDressedWZJet"].append(truth)
                ev["s_matchedToTruthDressedWZJet"].append((~unmatched).astype(np.int32))
                # large-R jets
                lpt = (250_000 + 10 * (large_counter + np.arange(nl))).astype(np.float32)
                large_counter += nl
                r10 = rng.choice(np.array([1, 2, 3, 10, 11], dtype=np.int32), size=nl)
                if nl > 0 and g_event % 13 == 4:
                    r10[0] = 3  # a guaranteed ineligible large-R jet
                l_ok = r10 != 3
                inel_large.update(float(p) for p in lpt[~l_ok])
                l_kept.append(np.flatnonzero(l_ok))
                ev["l_pt"].append(lpt)
                ev["l_eta"].append(rng.uniform(-2.0, 2.0, size=nl).astype(np.float32))
                ev["l_m"].append(rng.uniform(50e3, 200e3, size=nl).astype(np.float32))
                ev["l_R10TruthLabel_R22v1"].append(r10.astype(np.int32))
                ev["l_GhostBHadronsFinalCount"].append(rng.integers(0, 3, nl).astype(np.int32))
                ev["l_GhostCHadronsFinalCount"].append(rng.integers(0, 3, nl).astype(np.int32))
                # constituents: per-jet counts; values flat per event, jet by jet
                s_trk_n = rng.integers(1, TRACK_PAD + 1, size=ns).astype(np.int32)
                s_tower_n = rng.integers(0, TOWER_PAD + 1, size=ns).astype(np.int32)
                l_trk_n = rng.integers(1, TRACK_PAD + 1, size=nl).astype(np.int32)
                ev["s_trk_n"].append(s_trk_n)
                ev["s_tower_n"].append(s_tower_n)
                ev["l_trk_n"].append(l_trk_n)
                for prefix, counts in (("s_trk_", s_trk_n), ("l_trk_", l_trk_n)):
                    total = int(counts.sum())
                    for name in _TRK_FLOAT:
                        ev[f"{prefix}{name}"].append(rng.normal(size=total).astype(np.float32))
                    for name in _TRK_INT:
                        ev[f"{prefix}{name}"].append(
                            rng.integers(-1, 8, size=total).astype(np.int32)
                        )
                n_towers = int(s_tower_n.sum())
                ev["s_tower_pt"].append(rng.uniform(1e3, 5e4, size=n_towers).astype(np.float32))
                ev["s_tower_deta"].append(rng.uniform(-0.4, 0.4, size=n_towers).astype(np.float32))
                g_event += 1
            _write_file(path, ev)
            written.append((path, ev, s_kept, l_kept))

    # reader order: files sorted by (parent dir, path) == `UprootReader._resolve_files` glob order
    written.sort(key=lambda w: (str(w[0].parent), str(w[0])))
    rows: list[JointRow] = []
    for fi, (_, ev, s_kept, l_kept) in enumerate(written):
        for entry, evn in enumerate(ev["eventNumber"]):
            for jet in s_kept[entry]:
                pt = float(ev["s_pt"][entry][jet])
                rows.append(JointRow(fi, entry, SMALL_TYPE, int(jet), int(evn), pt))
            for jet in l_kept[entry]:
                pt = float(ev["l_pt"][entry][jet])
                rows.append(JointRow(fi, entry, LARGE_TYPE, int(jet), int(evn), pt))
    return JointCorpusFacts(
        root=root,
        glob=corpus_glob(root),
        files=[w[0] for w in written],
        n_events=[len(w[1]["eventNumber"]) for w in written],
        small_kept=[np.array([len(k) for k in w[2]], dtype=np.int64) for w in written],
        large_kept=[np.array([len(k) for k in w[3]], dtype=np.int64) for w in written],
        rows=rows,
        ineligible_small_pt=inel_small,
        ineligible_large_pt=inel_large,
    )


# --------------------------------------------------------------------------- #
# the nested-tree seam: two real ``var * T`` reads -> ``[event][jet][const]``
# --------------------------------------------------------------------------- #

_NESTED_COUNTS: dict[str, str] = {
    f"{SMALL_PREFIX}ft1l_trk_": f"{SMALL_PREFIX}ft1l_trk_n",
    f"{SMALL_PREFIX}ft1l_tower_": f"{SMALL_PREFIX}ft1l_tower_n",
    f"{LARGE_PREFIX}ft1lR_trk_": f"{LARGE_PREFIX}ft1lR_trk_n",
}
"""``{constituent-branch prefix: its per-jet count branch}`` — the three nested groups."""

_CTYPES: dict[str, str] = {
    "float32": "float",
    "float64": "double",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "bool": "bool",
}
"""numpy dtype name -> C++ leaf type, for the ``vector<vector<T>>`` typename."""


def _count_branch_for(name: str) -> str | None:
    """The count branch that reshapes constituent branch `name`, else None.

    A count branch itself resolves to None: it is read as the real ``var * int32``.
    """
    for prefix, count in _NESTED_COUNTS.items():
        if name.startswith(prefix) and name != count:
            return count
    return None


class _NestedBranch:
    """``[event][jet][const]`` view of one flat ``var * T`` branch and its count branch."""

    interpretation = None
    """No uproot interpretation, so `UprootReader._branch_dtype` falls back to a one-entry read."""

    def __init__(self, real_tree: Any, flat_name: str, count_name: str) -> None:
        self._real = real_tree
        self.flat_name = flat_name
        self.count_name = count_name

    @property
    def typename(self) -> str:
        """``std::vector<std::vector<T>>`` with ``T`` from THIS file's flat on-disk branch."""
        from uproot.interpretation.jagged import AsJagged
        from uproot.interpretation.numerical import AsDtype

        flat = self._real[self.flat_name]
        interp = flat.interpretation
        if isinstance(interp, AsJagged) and isinstance(interp.content, AsDtype):
            dtype = np.dtype(interp.content.to_dtype).newbyteorder("=")
            return f"std::vector<std::vector<{_CTYPES.get(dtype.name, dtype.name)}>>"
        return f"std::vector<{flat.typename}>"

    def array(
        self,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
        library: str = "ak",
        **kw: Any,
    ) -> Any:
        """Read the flat values and the per-jet counts over ONE entry window and nest them.

        Raises
        ------
        NotImplementedError
            For any `library` but ``"ak"``.
        ValueError
            When the counts do not tile the flat values (a fixture self-consistency bug).
        """
        del kw  # the reader passes no other read option
        if library != "ak":
            raise NotImplementedError(f"_NestedBranch.array: library={library!r} (only 'ak')")
        import awkward as ak

        window = {"entry_start": entry_start, "entry_stop": entry_stop, "library": "ak"}
        flat = self._real[self.flat_name].array(**window)
        counts = self._real[self.count_name].array(**window)
        per_jet = ak.to_numpy(ak.flatten(counts, axis=1)).astype(np.int64)
        per_event = ak.to_numpy(ak.num(counts, axis=1)).astype(np.int64)
        values = ak.flatten(flat, axis=1)
        jet_sums = ak.to_numpy(ak.sum(counts, axis=1)).astype(np.int64)
        flat_lens = ak.to_numpy(ak.num(flat, axis=1)).astype(np.int64)
        if int(per_jet.sum()) != len(values) or not np.array_equal(jet_sums, flat_lens):
            raise ValueError(
                f"_NestedBranch {self.flat_name!r}: count branch {self.count_name!r} does not "
                "tile the flat values (fixture self-consistency)"
            )
        return ak.unflatten(ak.unflatten(values, per_jet), per_event)


class _NestedTree:
    """A real uproot TTree whose constituent branches read as ``[event][jet][const]``.

    `__getitem__` and `arrays` route the three nested groups through `_NestedBranch`;
    every other branch, and every other attribute (``keys``, ``num_entries``, ``close``),
    is the real tree's.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getitem__(self, name: str) -> Any:
        count = _count_branch_for(name)
        if count is None:
            return self._real[name]
        return _NestedBranch(self._real, name, count)

    def __contains__(self, name: str) -> bool:
        return name in self._real.keys()

    def arrays(
        self,
        names: Any,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
        library: str = "ak",
        how: Any = None,
        **kw: Any,
    ) -> Any:
        """One real ``arrays`` read for the plain branches + one nested read per constituent
        branch, merged (``how=dict``, ``library="ak"`` only when any branch is nested).
        """
        names = [names] if isinstance(names, str) else list(names)
        nested = [n for n in names if _count_branch_for(n) is not None]
        if not nested:
            return self._real.arrays(
                names,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library=library,
                how=how,
                **kw,
            )
        if how is not dict or library != "ak":
            raise NotImplementedError(
                "_NestedTree.arrays over nested branches needs how=dict, library='ak' "
                f"(got how={how!r}, library={library!r})"
            )
        real = [n for n in names if _count_branch_for(n) is None]
        out: dict[str, Any] = {}
        if real:
            out = dict(
                self._real.arrays(
                    real,
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    library="ak",
                    how=dict,
                    **kw,
                )
            )
        for n in nested:
            out[n] = self[n].array(entry_start=entry_start, entry_stop=entry_stop)
        return out

    def __getattr__(self, name: str) -> Any:
        if name == "_real":  # not yet set (e.g. mid-unpickle): never recurse
            raise AttributeError(name)
        return getattr(self._real, name)


def _nested(t: Any) -> _NestedTree:
    """`t` wrapped in `_NestedTree` (unchanged when it already is one)."""
    return t if isinstance(t, _NestedTree) else _NestedTree(t)


class SeamUprootReader(UprootReader):
    """`UprootReader` whose tree handles are `_NestedTree`-wrapped — nothing else changes.

    The seam wraps the tree handle only: `_tree` (the cached open on the read path; the real
    tree stays in `_open_trees`) and the three `prepare` helpers that receive prepare's own
    ``uproot.open(...)`` handle. Every line of prepare / probe / read is the real code.
    """

    def _tree(self, path: str | Path) -> Any:
        return _nested(super()._tree(path))

    def _stream_dtypes(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        return super()._stream_dtypes(_nested(t), *args, **kwargs)

    def _index_file(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        return super()._index_file(_nested(t), *args, **kwargs)

    def _stream_max_mult(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        return super()._stream_max_mult(_nested(t), *args, **kwargs)


class SeamFTAG1LITEReader(FTAG1LITEReader):
    """`FTAG1LITEReader` whose sides are `SeamUprootReader`s.

    `_make_side` re-classes the real side reader (same config, no logic duplicated), so
    `with_source` (``type(self)(...)``) and pickling carry the seam by name.
    """

    def _make_side(self, side: str, cfg: Mapping[str, Any]) -> UprootReader:
        reader = super()._make_side(side, cfg)
        reader.__class__ = SeamUprootReader
        return reader


def side_config(side: str) -> dict[str, Any]:
    """One side's sub-mapping (``groups`` + ``cuts``) for the reader."""
    if side == "small":
        return {"groups": SMALL_GROUPS, "cuts": SMALL_CUTS}
    return {"groups": LARGE_GROUPS, "cuts": LARGE_CUTS}


def make_reader(root: Path, jets: str = "both", **kw: Any) -> Any:
    """A seam `FTAG1LITEReader` over the corpus at `root` (kwargs override the defaults)."""
    sides = {"small_r": ("small",), "large_r": ("large",), "both": ("small", "large")}[jets]
    defaults: dict[str, Any] = {side: side_config(side) for side in sides}
    return SeamFTAG1LITEReader(jets=jets, **{"filename": corpus_glob(root), **defaults, **kw})


def make_side_reader(root: Path, side: str, stage: str | None = None, **kw: Any) -> Any:
    """The single-collection (``jets: small_r | large_r``) seam reader for `side`."""
    return make_reader(root, jets=f"{side}_r", stage=stage, **kw)


def make_plain_reader(root: Path, side: str, stage: str | None = None, **kw: Any) -> Any:
    """An independently configured plain seam `UprootReader` for `side` — the oracle for
    the single modes.
    """
    cfg = side_config(side)
    return SeamUprootReader(
        groups=cfg["groups"],
        cuts=GlobalObjectCuts(**cfg["cuts"]),
        filename=corpus_glob(root),
        tree=TREE,
        unroll="jets",
        stage=stage,
        **kw,
    )
