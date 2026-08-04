"""Behavioural tests for the unified `salt.data.UprootReader`.

Format is config, not class: the same reader serves tree entries (``unroll=None``)
or the elements of one jagged group (``unroll=<group>``). Format-parity vs the
legacy presets lives in `test_uproot_equivalence.py`; here we exercise the
capabilities (easyjet jet-rows, cuts on either axis, config aliases) and the
config-validation error surface.
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
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
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
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
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
    from salt.data import Cut, CutSpec

    path, arrays = ej_file
    thresh = 100_000.0
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": False}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll="jets",
        cuts=CutSpec(global_cuts=(Cut("pt", ">", thresh),)),
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
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
            "event": {"branches": dict(_EVENT), "jagged": False},
        },
        filename=path,
        tree="AnalysisMiniTree",
    )
    b = UprootReader(
        groups={
            "event": {"branches": dict(_EVENT), "jagged": False},
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
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
# 4. config aliases + validation error surface (no file I/O)
# --------------------------------------------------------------------------- #


def test_truncate_alias_maps_to_pad_max() -> None:
    g = UprootReader._parse_group("jets", {"branches": {"pt": "pt"}, "truncate": 12})
    assert g.pad_max == 12


def test_target_collection_alias_maps_to_target_prefix() -> None:
    g = UprootReader._parse_group(
        "tracks",
        {
            "branches": {"d0": "d0"},
            "jagged": True,
            "link_branch": "GhostTrack",
            "target_collection": "InDetTrackParticles",
        },
    )
    assert g.target_prefix == "InDetTrackParticlesAuxDyn."


def test_unknown_unroll_group_raises() -> None:
    with pytest.raises(ConfigError, match="names no configured group"):
        UprootReader(groups={"jets": {"branches": {"pt": "pt"}, "jagged": False}}, unroll="tracks")


def test_unroll_group_must_be_jagged_false() -> None:
    with pytest.raises(ConfigError, match="must be declared jagged=False"):
        UprootReader(groups={"jets": {"branches": {"pt": "pt"}, "jagged": True}}, unroll="jets")


def test_link_without_unroll_raises() -> None:
    with pytest.raises(ConfigError, match="require unroll"):
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


def test_group_link_target_must_be_set_together() -> None:
    with pytest.raises(ConfigError, match="must be set together"):
        UprootGroupConfig(branches={"d0": "d0"}, jagged=True, link_branch="GhostTrack")


def test_group_link_only_on_jagged() -> None:
    with pytest.raises(ConfigError, match="only valid for jagged"):
        UprootGroupConfig(
            branches={"pt": "pt"},
            jagged=False,
            link_branch="GhostTrack",
            target_prefix="InDetTrackParticlesAuxDyn.",
        )


def test_empty_groups_raises() -> None:
    with pytest.raises(ConfigError, match="at least one group"):
        UprootReader(groups={})


def test_missing_branch_raises(ej_file) -> None:
    path, _ = ej_file
    reader = UprootReader(
        groups={"jets": {"branches": {"pt": "NOT_A_BRANCH"}, "jagged": True, "truncate": 4}},
        filename=path,
        tree="AnalysisMiniTree",
    )
    with pytest.raises(SchemaError):
        reader.prepare()


# --------------------------------------------------------------------------- #
# 5. ElementLink dereference (unroll=jets) — synthetic fake tree
# --------------------------------------------------------------------------- #

_GHOST_IDX = [[[0, 2], [1]], [[0]], [[2, 0, 1], [], [1]]]
_TRK_D0 = [[10.0, 11.0, 12.0], [20.0], [30.0, 31.0, 32.0]]
_TRK_Z0 = [[-1.0, -2.0, -3.0], [-4.0], [-5.0, -6.0, -7.0]]


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


def _linked_reader():
    return UprootReader(
        groups={
            "jets": {"branches": {"pt": "pt"}, "prefix": "AnalysisJetsAuxDyn.", "jagged": False},
            "tracks": {
                "branches": {"d0": "d0", "z0": "z0"},
                "jagged": True,
                "pad_max": 4,
                "link_branch": "GhostTrack",
                "target_prefix": "InDetTrackParticlesAuxDyn.",
            },
        },
        tree="CollectionTree",
        unroll="jets",
    )


def _expected(field_vals):
    return [
        [field_vals[ev_idx][i] for i in jet_links]
        for ev_idx, ev_jets in enumerate(_GHOST_IDX)
        for jet_links in ev_jets
    ]


def test_linked_deref_plain_index() -> None:
    import awkward as ak

    reader = _linked_reader()
    t = _FakeTree({
        "AnalysisJetsAuxDyn.GhostTrack": ak.Array(_GHOST_IDX),
        "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
        "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
    })
    block = reader._read_linked_block(t, reader.groups["tracks"], ["d0", "z0"], 0, 3)
    assert block["d0"].tolist() == _expected(_TRK_D0)
    assert block["z0"].tolist() == _expected(_TRK_Z0)


def test_linked_deref_struct_and_null_links() -> None:
    import awkward as ak

    idx = [[[0, 2, 0], [1]], [[0]], [[2, 0, 1], [], [1]]]
    key = [[[0, 490, 0], [490]], [[490]], [[490, 490, 490], [], [490]]]  # 0 == null
    links = ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(idx)}, depth_limit=None)
    reader = _linked_reader()
    t = _FakeTree({
        "AnalysisJetsAuxDyn.GhostTrack": links,
        "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
        "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
    })
    block = reader._read_linked_block(t, reader.groups["tracks"], ["d0", "z0"], 0, 3)
    assert block["d0"][0].tolist() == [12.0]  # e0j0: nulls dropped, keep persIndex 2
    assert block["d0"][3].tolist() == [32.0, 30.0, 31.0]  # e2j0: {2,0,1}


def test_linked_deref_multi_container_raises() -> None:
    import awkward as ak

    key = [[[1, 2], [1]], [[1]], [[1, 1, 1], [], [1]]]  # jet0/ev0 spans keys {1,2}
    links = ak.zip(
        {"m_persKey": ak.Array(key), "m_persIndex": ak.Array(_GHOST_IDX)}, depth_limit=None
    )
    reader = _linked_reader()
    t = _FakeTree({
        "AnalysisJetsAuxDyn.GhostTrack": links,
        "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
        "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
    })
    with pytest.raises(SchemaError, match="multiple target containers"):
        reader._read_linked_block(t, reader.groups["tracks"], ["d0", "z0"], 0, 3)


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
# 7. 1:1 ElementLink JOIN — extra fields merged onto an existing group
# --------------------------------------------------------------------------- #
#
# Distinct from section 5: `link_branch` BUILDS a constituent stream out of a
# target container (1:many), while a join augments elements the group already
# serves (1:1) and leaves the group's shape alone. The physical case is the HH4b
# PHYSLITE p6697, whose jets reach GN2v01 through `AnalysisJetsAuxDyn.btaggingLink`.

_JOIN_GROUP = {
    "branches": {"pt": "pt"},
    "prefix": "AnalysisJetsAuxDyn.",
    "jagged": True,
    "pad_max": 4,
    "join_branch": "btaggingLink",
    "join_prefix": "BTagging_AntiKt4EMPFlowAuxDyn.",
    "join_branches": {"GN2v01_pb": "GN2v01_pb", "GN2v01_pc": "GN2v01_pc"},
}
# 3 entries; jet -> b-tagging object is a permutation within each entry
_BTAG_IDX = [[2, 0, 1], [0], [1, 0]]
_BTAG_PB = [[0.1, 0.2, 0.3], [0.4], [0.5, 0.6]]
_BTAG_PC = [[10.0, 20.0, 30.0], [40.0], [50.0, 60.0]]


def _joined_reader(**over):
    group = {**_JOIN_GROUP, **over}
    return UprootReader(groups={"jets": group}, tree="CollectionTree", unroll=None)


def _join_tree(idx=None, key=None, pb=None, pc=None):
    import awkward as ak

    idx = _BTAG_IDX if idx is None else idx
    links = (
        ak.Array(idx)
        if key is None
        else ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(idx)}, depth_limit=None)
    )
    return _FakeTree({
        "AnalysisJetsAuxDyn.btaggingLink": links,
        "BTagging_AntiKt4EMPFlowAuxDyn.GN2v01_pb": ak.Array(_BTAG_PB if pb is None else pb),
        "BTagging_AntiKt4EMPFlowAuxDyn.GN2v01_pc": ak.Array(_BTAG_PC if pc is None else pc),
    })


def test_join_serves_direct_then_joined_fields_in_order() -> None:
    cfg = UprootReader._parse_group("jets", dict(_JOIN_GROUP))
    assert tuple(cfg.served_branches) == ("pt", "GN2v01_pb", "GN2v01_pc")
    assert cfg.is_joined and not cfg.is_linked


def test_join_gathers_by_pers_index() -> None:
    reader = _joined_reader()
    block = reader._read_joined_cols(
        _join_tree(), reader.groups["jets"], ["GN2v01_pb", "GN2v01_pc"], 0, 3
    )
    # entry 0 jets point at btagging objects 2,0,1 -> 0.3, 0.1, 0.2
    assert block["GN2v01_pb"].tolist() == [[0.3, 0.1, 0.2], [0.4], [0.6, 0.5]]
    assert block["GN2v01_pc"].tolist() == [[30.0, 10.0, 20.0], [40.0], [60.0, 50.0]]


def test_join_shape_matches_the_groups_own_branches() -> None:
    reader = _joined_reader()
    block = reader._read_joined_cols(_join_tree(), reader.groups["jets"], ["GN2v01_pb"], 0, 3)
    import awkward as ak

    assert ak.num(block["GN2v01_pb"], axis=1).tolist() == [len(e) for e in _BTAG_IDX]


def test_join_honours_a_struct_link_with_a_single_container_key() -> None:
    key = [[490, 490, 490], [490], [490, 490]]
    reader = _joined_reader()
    block = reader._read_joined_cols(
        _join_tree(key=key), reader.groups["jets"], ["GN2v01_pb"], 0, 3
    )
    assert block["GN2v01_pb"].tolist() == [[0.3, 0.1, 0.2], [0.4], [0.6, 0.5]]


def test_join_refuses_a_null_link() -> None:
    key = [[490, 0, 490], [490], [490, 490]]  # one thinned/absent b-tagging object
    reader = _joined_reader()
    with pytest.raises(SchemaError, match="null ElementLink"):
        reader._read_joined_cols(_join_tree(key=key), reader.groups["jets"], ["GN2v01_pb"], 0, 3)


def test_join_refuses_multiple_target_containers() -> None:
    key = [[490, 491, 490], [490], [490, 490]]
    reader = _joined_reader()
    with pytest.raises(SchemaError, match="multiple target containers"):
        reader._read_joined_cols(_join_tree(key=key), reader.groups["jets"], ["GN2v01_pb"], 0, 3)


def test_join_refuses_an_index_past_the_target() -> None:
    reader = _joined_reader()
    tree = _join_tree(idx=[[2, 0, 3], [0], [1, 0]])  # 3 is past entry 0's 3 objects
    with pytest.raises(SchemaError, match="out of range"):
        reader._read_joined_cols(tree, reader.groups["jets"], ["GN2v01_pb"], 0, 3)


def test_join_refuses_a_misaligned_target_container() -> None:
    reader = _joined_reader()
    tree = _join_tree(pb=[[0.1, 0.2, 0.3], [0.4]])  # 2 entries vs the link's 3
    with pytest.raises(SchemaError, match="not aligned"):
        reader._read_joined_cols(tree, reader.groups["jets"], ["GN2v01_pb"], 0, 3)


def test_join_config_needs_all_three_keys() -> None:
    with pytest.raises(ConfigError, match="needs 'join_branch'"):
        UprootGroupConfig(branches={"pt": "pt"}, join_branch="btaggingLink")


def test_join_config_refuses_a_field_declared_twice() -> None:
    with pytest.raises(ConfigError, match="declared in both"):
        UprootGroupConfig(
            branches={"pt": "pt", "GN2v01_pb": "GN2v01_pb"},
            join_branch="btaggingLink",
            join_prefix="BTagging_AntiKt4EMPFlowAuxDyn.",
            join_branches={"GN2v01_pb": "GN2v01_pb"},
        )


def test_join_config_refuses_a_link_and_a_join_on_one_group() -> None:
    with pytest.raises(ConfigError, match="already sets 'link_branch'"):
        UprootGroupConfig(
            branches={"d0": "d0"},
            link_branch="GhostTrack",
            target_prefix="InDetTrackParticlesAuxDyn.",
            join_branch="btaggingLink",
            join_prefix="BTagging_AntiKt4EMPFlowAuxDyn.",
            join_branches={"GN2v01_pb": "GN2v01_pb"},
        )


def test_join_on_a_jagged_group_under_unroll_is_refused() -> None:
    with pytest.raises(ConfigError, match="two levels below the entry axis"):
        UprootReader(
            groups={
                "rows": {"branches": {"pt": "pt"}, "jagged": False},
                "jets": dict(_JOIN_GROUP),
            },
            unroll="rows",
        )


# --------------------------------------------------------------------------- #
# 8. row cuts that reduce a constituent axis (sum/count)
# --------------------------------------------------------------------------- #


def _agg_reader(path, cut: str, truncate: int = 8):
    from salt.data import GlobalObjectCuts

    return UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": truncate},
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
    """`truncate` caps the served multiplicity, so the reduction has to see the cap."""
    path, arrays = ej_file
    njets = np.array(arrays["njets"])
    reader = _agg_reader(path, "sum(jets.valid) >= 2", truncate=2)
    assert len(reader) == int((np.minimum(njets, 2) >= 2).sum())  # 4, not 3


def test_count_is_the_multiplicity(ej_file) -> None:
    path, arrays = ej_file
    njets = np.array(arrays["njets"])
    reader = _agg_reader(path, "count(jets.pt) >= 4")
    assert len(reader) == int((njets >= 4).sum())


def test_a_reduction_sees_the_constituent_cuts(ej_file) -> None:
    from salt.data import GlobalObjectCuts

    path, arrays = ej_file
    thresh = 100_000.0
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": True, "truncate": 8}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
        constituent_cuts={"jets": {"on_fail": "drop", "cuts": [f"pt > {thresh}"]}},
        cuts=GlobalObjectCuts(global_cuts=("sum(jets.valid) >= 2",)),
    )
    n_pass = np.array([
        int((np.asarray(ev) > thresh).sum()) for ev in arrays["recojet_antikt4PFlow_pt_NOSYS"]
    ])
    assert len(reader) == int((n_pass >= 2).sum())
    served = reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]
    assert np.all(served["valid"].sum(axis=1) >= 2)


def test_a_predicate_reduction_counts_passing_constituents(ej_file) -> None:
    path, arrays = ej_file
    reader = _agg_reader(path, "sum(jets.pt > 100000.0) >= 2")
    n_pass = np.array([
        int((np.asarray(ev) > 100_000.0).sum()) for ev in arrays["recojet_antikt4PFlow_pt_NOSYS"]
    ])
    assert len(reader) == int((n_pass >= 2).sum())


@pytest.mark.parametrize(
    ("cut", "match"),
    [
        ("sum(tracks.valid) >= 4", "not a configured group"),
        ("sum(event.eventNumber) >= 4", "declared jagged=False"),
        ("sum(jets.nope) >= 4", "not configured branches"),
    ],
)
def test_reduction_config_errors(cut: str, match: str) -> None:
    from salt.data import GlobalObjectCuts

    with pytest.raises(ConfigError, match=match):
        UprootReader(
            groups={
                "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
                "event": {"branches": dict(_EVENT), "jagged": False},
            },
            tree="AnalysisMiniTree",
            unroll=None,
            cuts=GlobalObjectCuts(global_cuts=(cut,)),
        )


def test_join_reads_uproot_qualified_link_members() -> None:
    """A real POOL file names the members after the whole branch path, not bare.

    ``t['AnalysisJetsAuxDyn.btaggingLink'].array()`` comes back with fields
    ``AnalysisJetsAuxDyn.btaggingLink.m_persKey`` / ``...m_persIndex``. Matching
    only the bare name silently falls through to the "plain integer index"
    fixture path and hands the gather a record array.
    """
    import awkward as ak

    pre = "AnalysisJetsAuxDyn.btaggingLink"
    links = ak.zip(
        {
            f"{pre}.m_persKey": ak.Array([[490, 490, 490], [490], [490, 490]]),
            f"{pre}.m_persIndex": ak.Array(_BTAG_IDX),
        },
        depth_limit=1,  # record at the OUTER level, as uproot delivers it
    )
    reader = _joined_reader()
    tree = _FakeTree({
        pre: links,
        "BTagging_AntiKt4EMPFlowAuxDyn.GN2v01_pb": ak.Array(_BTAG_PB),
    })
    block = reader._read_joined_cols(tree, reader.groups["jets"], ["GN2v01_pb"], 0, 3)
    assert block["GN2v01_pb"].tolist() == [[0.3, 0.1, 0.2], [0.4], [0.6, 0.5]]


def test_link_member_lookup_is_unambiguous() -> None:
    import awkward as ak

    from salt.data.readers.uproot_reader import _link_member

    links = ak.zip(
        {"a.m_persIndex": ak.Array([[0]]), "b.m_persIndex": ak.Array([[1]])}, depth_limit=1
    )
    with pytest.raises(SchemaError, match="cannot tell which one"):
        _link_member(links, "m_persIndex")


# --------------------------------------------------------------------------- #
# 6. the hot read path opens each file ONCE per process
# --------------------------------------------------------------------------- #


def _cache_reader(path: Path) -> UprootReader:
    return UprootReader(
        groups={
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
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

    for lo in range(0, 8, 2):
        reader.read(slice(lo, lo + 2), Mode.FIT)

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
