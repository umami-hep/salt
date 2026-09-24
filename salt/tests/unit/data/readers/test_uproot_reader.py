"""Behavioural tests for the unified `salt.data.UprootReader`.

Format is config, not class: the same reader serves tree entries (``unroll=None``)
or the elements of one jagged group (``unroll=<group>``). Format-parity vs the
legacy presets lives in `test_uproot_equivalence.py`; here we exercise the
capabilities (easyjet jet-rows, cuts on either axis) and the config-validation
error surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from salt.data import UprootGroupConfig, UprootReader
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import Mode

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.tests._fixtures.easyjet_minitree import (  # noqa: E402
    build_fixture_arrays,
    write_minitree,
)

_JET = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "label": "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
}
_EVENT = {"eventNumber": "eventNumber", "mcChannelNumber": "mcChannelNumber"}


@pytest.fixture
def ej_file(tmp_path: Path) -> tuple[Path, dict]:
    arrays = build_fixture_arrays(seed=99)
    return write_minitree(tmp_path / "ej.root", arrays), arrays


# --------------------------------------------------------------------------- #
# 1. easyjet event-rows (unroll=None) — rows are tree entries
# --------------------------------------------------------------------------- #


def test_event_rows_len_and_streams(ej_file) -> None:
    path, arrays = ej_file
    reader = UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
    )
    assert len(reader) == arrays["n_events"]
    assert reader.streams == ("jets", "event")


def test_event_rows_jagged_and_scalar(ej_file) -> None:
    path, arrays = ej_file
    reader = UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
    )
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    njets = np.array(arrays["njets"])
    assert out["raw.jets"].shape == (n, 8)
    assert np.array_equal(out["raw.jets"]["valid"].sum(axis=1), np.minimum(njets, 8))
    assert np.array_equal(out["masks.jets"], ~out["raw.jets"]["valid"])
    assert out["raw.event"].shape == (n,)
    np.testing.assert_array_equal(out["raw.event"]["eventNumber"], arrays["eventNumber"])


# --------------------------------------------------------------------------- #
# 2. easyjet jet-rows (unroll="jets") — NEW capability: rows are jets
# --------------------------------------------------------------------------- #


def test_jet_rows_flattens_events_to_jets(ej_file) -> None:
    path, arrays = ej_file
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": False}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll="jets",
    )
    njets = np.array(arrays["njets"])
    assert len(reader) == int(njets.sum())  # one row per jet, events flattened away
    out = reader.read(slice(0, len(reader)), Mode.FIT)
    raw = out["raw.jets"]
    assert raw.shape == (int(njets.sum()),)
    assert "masks.jets" not in out  # jagged=False row-carrier -> no pad mask
    # values == the flattened per-event fixture arrays
    expected_pt = np.concatenate([
        arrays["recojet_antikt4PFlow_pt_NOSYS"][ev] for ev in range(len(njets))
    ]).astype(raw["pt"].dtype)
    np.testing.assert_array_equal(raw["pt"], expected_pt)


def test_jet_rows_cut_on_pt(ej_file) -> None:
    from salt.data import GlobalObjectCuts

    path, arrays = ej_file
    thresh = 100_000.0
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": False}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll="jets",
        cuts=GlobalObjectCuts(global_cuts=(f"pt > {thresh}",)),
    )
    all_pt = np.concatenate([
        arrays["recojet_antikt4PFlow_pt_NOSYS"][ev] for ev in range(arrays["n_events"])
    ])
    assert len(reader) == int((all_pt > thresh).sum())
    served = reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]
    assert np.all(served["pt"] > thresh)


def test_pre_flattened_entry_is_jet(ej_file) -> None:
    """A per-entry scalar stream with unroll=None serves each entry directly (pre-flattened)."""
    path, arrays = ej_file
    reader = UprootReader(
        groups={"event": {"branches": dict(_EVENT), "jagged": False}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
    )
    assert len(reader) == arrays["n_events"]
    raw = reader.read(slice(0, len(reader)), Mode.FIT)["raw.event"]
    np.testing.assert_array_equal(raw["eventNumber"], arrays["eventNumber"])


# --------------------------------------------------------------------------- #
# 3. group insertion order is never semantic
# --------------------------------------------------------------------------- #


def test_group_order_independent(ej_file) -> None:
    path, _ = ej_file
    a = UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
    )
    b = UprootReader(
        groups={
            "event": {"branches": dict(_EVENT), "jagged": False},
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
        },
        filename=path,
        tree="AnalysisMiniTree",
    )
    assert len(a) == len(b)
    ra = a.read(slice(0, len(a)), Mode.FIT)
    rb = b.read(slice(0, len(b)), Mode.FIT)
    np.testing.assert_array_equal(ra["raw.jets"]["pt"], rb["raw.jets"]["pt"])
    np.testing.assert_array_equal(ra["raw.event"]["eventNumber"], rb["raw.event"]["eventNumber"])


# --------------------------------------------------------------------------- #
# 4. config validation error surface (no file I/O)
# --------------------------------------------------------------------------- #


def test_unknown_unroll_group_raises() -> None:
    with pytest.raises(ConfigError, match="names no configured group"):
        UprootReader(groups={"jets": {"branches": {"pt": "pt"}, "jagged": False}}, unroll="tracks")


def test_unroll_group_must_be_jagged_false() -> None:
    with pytest.raises(ConfigError, match="must be declared jagged=False"):
        UprootReader(groups={"jets": {"branches": {"pt": "pt"}, "jagged": True}}, unroll="jets")


def test_empty_groups_raises() -> None:
    with pytest.raises(ConfigError, match="at least one group"):
        UprootReader(groups={})


def test_missing_branch_raises(ej_file) -> None:
    path, _ = ej_file
    reader = UprootReader(
        groups={"jets": {"branches": {"pt": "NOT_A_BRANCH"}, "jagged": True, "pad_max": 4}},
        filename=path,
        tree="AnalysisMiniTree",
    )
    with pytest.raises(SchemaError):
        reader.prepare()


# --------------------------------------------------------------------------- #
# 5. base reader config surface: no ElementLink/join vocabulary
# --------------------------------------------------------------------------- #


class _FakeBranch:
    def __init__(self, arr):
        self._arr = arr

    def array(self, entry_start=None, entry_stop=None, library="ak"):  # noqa: ARG002
        return self._arr[entry_start:entry_stop]


class _FakeTree:
    def __init__(self, branches):
        self._b = {name: _FakeBranch(arr) for name, arr in branches.items()}

    def __getitem__(self, name):
        return self._b[name]

    def arrays(self, expressions, entry_start=None, entry_stop=None, library="ak", how=None):  # noqa: ARG002
        """The grouped read the reader uses: same values as N per-branch `array()` calls.

        Only ``how=dict`` is modelled, because that is the only form the reader
        asks for — anything else would be a silent behaviour change rather than a
        double that has fallen behind its subject.
        """
        if how is not dict:
            raise NotImplementedError(f"_FakeTree.arrays only models how=dict, got {how!r}")
        return {
            name: self._b[name].array(entry_start=entry_start, entry_stop=entry_stop)
            for name in expressions
        }


def _direct_reader():
    return UprootReader(
        groups={
            "jets": {"branches": {"pt": "pt"}, "prefix": "AnalysisJetsAuxDyn.", "jagged": False},
            "tracks": {
                "branches": {"d0": "d0", "z0": "z0"},
                "prefix": "InDetTrackParticlesAuxDyn.",
                "jagged": True,
                "pad_max": 4,
            },
        },
        tree="CollectionTree",
        unroll="jets",
    )


def test_base_reader_rejects_elementlink_keys() -> None:
    """Link/join keys on a plain group redirect the config error to `xAODReader`."""
    with pytest.raises(ConfigError, match="salt.data.xAODReader"):
        UprootReader(
            groups={
                "tracks": {
                    "branches": {"d0": "d0"},
                    "jagged": True,
                    "link_branch": "GhostTrack",
                    "target_prefix": "InDetTrackParticlesAuxDyn.",
                }
            },
            unroll=None,
        )
    with pytest.raises(ConfigError, match="salt.data.xAODReader"):
        UprootReader(
            groups={
                "jets": {
                    "branches": {"pt": "pt"},
                    "jagged": True,
                    "join_branch": "btaggingLink",
                    "join_prefix": "BTagging_AntiKt4EMPFlowAuxDyn.",
                    "join_branches": {"GN2v01_pb": "GN2v01_pb"},
                }
            },
            unroll=None,
        )
    with pytest.raises(TypeError):
        UprootGroupConfig(branches={"d0": "d0"}, link_branch="GhostTrack")


# --------------------------------------------------------------------------- #
# 6. lazy import contract
# --------------------------------------------------------------------------- #


def test_no_top_level_uproot_awkward_import() -> None:
    import salt.data.readers.uproot_reader as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert "uproot" not in top
    assert "awkward" not in top


# --------------------------------------------------------------------------- #
# 7. row cuts that reduce a constituent axis (sum/count)
# --------------------------------------------------------------------------- #


def _agg_reader(path, cut: str, pad_max: int = 8):
    from salt.data import GlobalObjectCuts

    return UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": pad_max},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
        cuts=GlobalObjectCuts(global_cuts=(cut,)),
    )


def test_sum_valid_cuts_events_on_jet_multiplicity(ej_file) -> None:
    path, arrays = ej_file
    njets = np.array(arrays["njets"])  # [3, 1, 5, 0, 2, 4]
    reader = _agg_reader(path, "sum(jets.valid) >= 3")
    assert len(reader) == int((njets >= 3).sum())
    served = reader.read(slice(0, len(reader)), Mode.FIT)
    assert np.all(served["raw.jets"]["valid"].sum(axis=1) >= 3)
    np.testing.assert_array_equal(
        served["raw.event"]["eventNumber"], np.asarray(arrays["eventNumber"])[njets >= 3]
    )


def test_sum_valid_counts_what_is_served_not_what_is_on_disk(ej_file) -> None:
    """`pad_max` caps the served multiplicity, so the reduction has to see the cap."""
    path, arrays = ej_file
    njets = np.array(arrays["njets"])
    reader = _agg_reader(path, "sum(jets.valid) >= 2", pad_max=2)
    assert len(reader) == int((np.minimum(njets, 2) >= 2).sum())  # 4, not 3


def test_a_reduction_counts_on_disk_jets_not_post_cut(ej_file) -> None:
    """No constituent cuts configured: the reduction counts what is on disk, clipped to pad_max."""
    from salt.data import GlobalObjectCuts

    path, arrays = ej_file
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
        cuts=GlobalObjectCuts(global_cuts=("sum(jets.pt > 100000.0) >= 2",)),
    )
    n_pass = np.array([
        int((np.asarray(ev) > 100_000.0).sum()) for ev in arrays["recojet_antikt4PFlow_pt_NOSYS"]
    ])
    assert len(reader) == int((n_pass >= 2).sum())
    served = reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]
    njets = np.array(arrays["njets"])[n_pass >= 2]
    # served multiplicity == ON-DISK jet count clipped to pad_max — NOT the post-cut count
    np.testing.assert_array_equal(served["valid"].sum(axis=1), np.minimum(njets, 8))


def test_a_predicate_reduction_counts_passing_constituents(ej_file) -> None:
    path, arrays = ej_file
    reader = _agg_reader(path, "sum(jets.pt > 100000.0) >= 2")
    n_pass = np.array([
        int((np.asarray(ev) > 100_000.0).sum()) for ev in arrays["recojet_antikt4PFlow_pt_NOSYS"]
    ])
    assert len(reader) == int((n_pass >= 2).sum())


def test_reduction_over_unknown_stream_fails_at_prepare(ej_file) -> None:
    """A row-cut reduction over an unconfigured stream constructs fine and fails at prepare()."""
    from salt.data import GlobalObjectCuts

    path, _ = ej_file
    reader = UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
        cuts=GlobalObjectCuts(global_cuts=("sum(tracks.valid) >= 4",)),
    )
    with pytest.raises((KeyError, SchemaError)):
        reader.prepare()


# --------------------------------------------------------------------------- #
# 8. the hot read path opens each file ONCE per process
# --------------------------------------------------------------------------- #


def _cache_reader(path: Path) -> UprootReader:
    return UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "pad_max": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
    )


def test_read_reuses_one_open_tree_per_file(ej_file, monkeypatch) -> None:
    """Many small reads must not re-open (and re-parse) the file each time.

    `MultiSampleReader` decomposes an interleaved window into one contiguous run
    per few rows, so an uncached `uproot.open` ran once per run per group. Count
    the opens rather than time them.
    """
    path, _ = ej_file
    reader = _cache_reader(path)
    reader.prepare()

    opened: list[str] = []
    real_open = uproot.open

    def counting_open(spec, *args, **kwargs):
        opened.append(str(spec))
        return real_open(spec, *args, **kwargs)

    # the reader imports uproot inside its methods, so patching the module
    # attribute is what the hot path resolves
    monkeypatch.setattr(uproot, "open", counting_open)

    n = len(reader)
    for lo in range(0, n, 2):
        reader.read(slice(lo, min(lo + 2, n)), Mode.FIT)

    assert len(opened) == 1, f"expected one open for the whole read sequence, got {opened}"
    assert len(reader._open_trees) == 1


def test_read_matches_an_uncached_reader(ej_file) -> None:
    """The cache is a pure optimisation: same bytes out, read at any granularity."""
    path, _ = ej_file
    whole = _cache_reader(path)
    n = len(whole)
    reference = whole.read(slice(0, n), Mode.FIT)

    piecewise = _cache_reader(path)
    chunks = [piecewise.read(slice(lo, min(lo + 3, n)), Mode.FIT) for lo in range(0, n, 3)]
    for key in ("raw.event",):
        joined = np.concatenate([c[key] for c in chunks])
        np.testing.assert_array_equal(joined["eventNumber"], reference[key]["eventNumber"])


def test_open_trees_are_dropped_on_pickle(ej_file) -> None:
    """An open uproot file is not picklable — spawn workers must get a clean reader."""
    import pickle

    path, _ = ej_file
    reader = _cache_reader(path)
    reader.prepare()
    reader.read(slice(0, 2), Mode.FIT)
    assert reader._open_trees

    revived = pickle.loads(pickle.dumps(reader))
    assert revived._open_trees == {}


# --------------------------------------------------------------------------- #
# 9. prepare() probes the schema from metadata, once, on the first file
# --------------------------------------------------------------------------- #

_TREE = "AnalysisMiniTree"


class _RecordingBranch:
    """A branch proxy that records every ``array()`` call's entry bound."""

    def __init__(self, branch, calls: list, tag: int, name: str) -> None:
        self._branch, self._calls, self._tag, self._name = branch, calls, tag, name

    def __getattr__(self, attr):
        return getattr(self._branch, attr)

    def array(self, *args, **kwargs):
        self._calls.append({
            "file": self._tag,
            "branch": self._name,
            "entry_stop": kwargs.get("entry_stop"),
        })
        return self._branch.array(*args, **kwargs)


class _RecordingTree:
    """A TTree proxy handing out `_RecordingBranch`es (dunders must be explicit)."""

    def __init__(self, tree, calls: list, tag: int) -> None:
        self._tree, self._calls, self._tag = tree, calls, tag

    def __getattr__(self, attr):
        return getattr(self._tree, attr)

    def __getitem__(self, name):
        return _RecordingBranch(self._tree[name], self._calls, self._tag, name)

    def __enter__(self):
        self._tree.__enter__()
        return self

    def __exit__(self, *exc):
        return self._tree.__exit__(*exc)


def _record_prepare(reader: UprootReader, monkeypatch) -> tuple[list, list]:
    """Run `prepare` with every `uproot.open` and branch read recorded."""
    calls: list = []
    opens: list = []
    real_open = uproot.open

    def recording_open(spec, *args, **kwargs):
        opens.append(str(spec))
        return _RecordingTree(real_open(spec, *args, **kwargs), calls, len(opens) - 1)

    monkeypatch.setattr(uproot, "open", recording_open)
    reader.prepare()
    return calls, opens


def _probe_reader(source, pad_max: int | None = 8, **kwargs) -> UprootReader:
    jets: dict = {"branches": dict(_JET), "jagged": True}
    if pad_max is not None:
        jets["pad_max"] = pad_max
    return UprootReader(
        groups={"jets": jets, "event": {"branches": dict(_EVENT), "jagged": False}},
        filename=source,
        tree=_TREE,
        **kwargs,
    )


def _two_files(tmp_path: Path, second: dict | None = None) -> Path:
    """A directory of two minitrees (the second optionally re-shaped)."""
    d = tmp_path / "pair"
    d.mkdir()
    write_minitree(d / "a.root", build_fixture_arrays(seed=99))
    write_minitree(d / "b.root", second if second is not None else build_fixture_arrays(seed=7))
    return d


def test_prepare_makes_no_unbounded_reads_when_pad_max_is_set(ej_file, monkeypatch) -> None:
    """The schema probe must not decompress whole branches to learn their type."""
    path, _ = ej_file
    calls, opens = _record_prepare(_probe_reader(path), monkeypatch)
    unbounded = [c for c in calls if c["entry_stop"] is None]
    assert unbounded == [], f"prepare read whole branches: {unbounded}"
    assert len(opens) == 1


def test_prepare_opens_each_file_once(tmp_path, monkeypatch) -> None:
    """One open per file — the max-multiplicity scan must not re-open."""
    _, opens = _record_prepare(_probe_reader(_two_files(tmp_path), pad_max=None), monkeypatch)
    assert len(opens) == 2


def test_prepare_reads_one_branch_for_an_unresolved_pad_max(ej_file, monkeypatch) -> None:
    """An unset ``pad_max`` is the one legitimate bulk read: a single branch, once."""
    path, _ = ej_file
    calls, opens = _record_prepare(_probe_reader(path, pad_max=None), monkeypatch)
    unbounded = [c for c in calls if c["entry_stop"] is None]
    assert [c["branch"] for c in unbounded] == [_JET["pt"]]
    assert len(opens) == 1


def test_prepare_probes_the_schema_only_on_the_first_file(tmp_path, monkeypatch) -> None:
    """With the metadata fast path disabled, only file 0 pays a (bounded) probe read."""
    monkeypatch.setattr(
        "salt.data.readers.uproot_reader._interpretation_type", lambda _interp: None
    )
    calls, opens = _record_prepare(_probe_reader(_two_files(tmp_path)), monkeypatch)
    assert len(opens) == 2
    assert {c["file"] for c in calls} == {0}
    assert all(c["entry_stop"] == 1 for c in calls)
    assert {c["branch"] for c in calls} == set(_JET.values()) | set(_EVENT.values())


def test_prepare_schema_dtypes_match_a_full_read_oracle(ej_file) -> None:
    """The probed dtypes are exactly what a whole-branch read would have reported."""
    import awkward as ak

    path, _ = ej_file
    reader = _probe_reader(path)
    reader.prepare()
    with uproot.open(f"{path}:{_TREE}") as tree:
        for stream, branches in (("jets", _JET), ("event", _EVENT)):
            for fieldname, bare in branches.items():
                flat = ak.flatten(tree[bare].array(library="ak"), axis=None)
                want = np.dtype(np.asarray(ak.to_numpy(flat)).dtype.newbyteorder("=")).name
                assert reader.schema.groups[stream].fields[fieldname] == want


def test_metadata_and_bounded_probe_agree_field_by_field(ej_file, monkeypatch) -> None:
    """The interpretation fast path and the one-entry fallback are interchangeable."""
    path, _ = ej_file
    reader = _probe_reader(path)
    names = list(_JET.values()) + list(_EVENT.values())
    with uproot.open(f"{path}:{_TREE}") as tree:
        n = int(tree.num_entries)
        via_meta = {b: reader._branch_dtype(tree, b, n) for b in names}
        monkeypatch.setattr(
            "salt.data.readers.uproot_reader._interpretation_type", lambda _interp: None
        )
        via_read = {b: reader._branch_dtype(tree, b, n) for b in names}
    assert via_meta == via_read


def test_prepare_bookkeeping_is_unchanged_across_files(tmp_path) -> None:
    """Row index, per-file kept counts and resolved multiplicity survive the probe change."""
    from salt.data import GlobalObjectCuts

    directory = _two_files(tmp_path)
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": False}},
        filename=directory,
        tree=_TREE,
        unroll="jets",
        cuts=GlobalObjectCuts(global_cuts=("pt > 100000.0",)),
    )
    reader.prepare()
    total = 0
    for entry, seed in zip(reader._table, (99, 7), strict=True):
        arrays = build_fixture_arrays(seed=seed)
        pt = [np.asarray(v) for v in arrays["recojet_antikt4PFlow_pt_NOSYS"]]
        expected_kept = np.flatnonzero(np.concatenate(pt) > 100_000.0)
        np.testing.assert_array_equal(entry.kept, expected_kept)
        np.testing.assert_array_equal(entry.orig_counts, [len(v) for v in pt])
        np.testing.assert_array_equal(entry.per_row_kept, [int((v > 100_000.0).sum()) for v in pt])
        total += int(expected_kept.size)
    assert len(reader) == total


# --------------------------------------------------------------------------- #
# 10. grouped-read primitive (_read_branches): one shared request per file
# --------------------------------------------------------------------------- #


class _CountingTree(_FakeTree):
    """A `_FakeTree` that records how many grouped reads it was asked for."""

    def __init__(self, branches):
        super().__init__(branches)
        self.calls: list[list[str]] = []

    def arrays(self, expressions, entry_start=None, entry_stop=None, library="ak", how=None):
        self.calls.append(list(expressions))
        return super().arrays(
            expressions, entry_start=entry_start, entry_stop=entry_stop, library=library, how=how
        )


def _grouped_read_tree():
    import awkward as ak

    return _CountingTree({
        f"AnalysisJetsAuxDyn.f{i}": ak.Array([[float(i), float(i) + 1.0], [float(i) + 2.0]])
        for i in range(5)
    })


def test_grouped_read_returns_every_requested_branch():
    """One grouped call serves the same values a per-branch read would."""
    reader = _direct_reader()
    tree = _grouped_read_tree()
    names = [f"AnalysisJetsAuxDyn.f{i}" for i in range(5)]

    got = reader._read_branches(tree, names, 0, 2)

    assert sorted(got) == sorted(names)
    assert len(tree.calls) == 1  # 5 branches -> one grouped call
    for name in names:
        expected = tree[name].array(entry_start=0, entry_stop=2)
        assert got[name].to_list() == expected.to_list()


def test_grouped_read_deduplicates_repeated_branches():
    """A branch named twice is read once, and still served under its name."""
    reader = _direct_reader()
    tree = _grouped_read_tree()
    name = "AnalysisJetsAuxDyn.f0"

    got = reader._read_branches(tree, [name, name, name], 0, 2)

    assert tree.calls == [[name]]
    assert list(got) == [name]


def test_grouped_read_raises_when_a_branch_is_not_returned():
    """A silently dropped/renamed expression must fail loudly, not serve wrong values."""

    class _DroppingTree(_FakeTree):
        def arrays(self, expressions, entry_start=None, entry_stop=None, library="ak", how=None):
            out = super().arrays(
                expressions,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library=library,
                how=how,
            )
            out.pop("AnalysisJetsAuxDyn.f1", None)
            return out

    import awkward as ak

    reader = _direct_reader()
    tree = _DroppingTree({
        f"AnalysisJetsAuxDyn.f{i}": ak.Array([[1.0, 2.0], [3.0]]) for i in range(3)
    })

    with pytest.raises(SchemaError, match="did not return branch"):
        reader._read_branches(tree, [f"AnalysisJetsAuxDyn.f{i}" for i in range(3)], 0, 2)


def test_read_issues_one_grouped_request_per_file_for_all_direct_streams(
    ej_file, monkeypatch
) -> None:
    """One `read()` over N non-linked streams in one file issues ONE `arrays()` call."""
    path, _ = ej_file
    reader = _cache_reader(path)
    reference = _cache_reader(path)
    reader.prepare()

    real_tree = reader._tree
    calls: list[list[str]] = []

    class _CountingRealTree:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, attr):
            return getattr(self._wrapped, attr)

        def __getitem__(self, name):
            return self._wrapped[name]

        def arrays(self, expressions, *args, **kwargs):
            calls.append(list(expressions))
            return self._wrapped.arrays(expressions, *args, **kwargs)

    monkeypatch.setattr(reader, "_tree", lambda p: _CountingRealTree(real_tree(p)))

    out = reader.read(slice(0, 5), Mode.FIT)

    assert len(calls) == 1
    expected = set(_JET.values()) | set(_EVENT.values())
    assert set(calls[0]) == expected

    ref_out = reference.read(slice(0, 5), Mode.FIT)
    for key in ("raw.jets", "raw.event"):
        for field in out[key].dtype.names:
            np.testing.assert_array_equal(out[key][field], ref_out[key][field])


# --------------------------------------------------------------------------- #
# clone/restage preserve the reader subclass
# --------------------------------------------------------------------------- #


def test_clone_keeps_the_base_class(tmp_path, ej_file) -> None:
    """`with_source`/`restage` must build `type(self)`, not a hardcoded `UprootReader`."""
    path, _ = ej_file
    assert _cache_reader(path).with_source(path, num=3, stage="val").__class__ is UprootReader
    assert _cache_reader(path).restage(tmp_path).__class__ is UprootReader
