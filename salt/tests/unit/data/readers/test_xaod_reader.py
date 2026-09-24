"""Behavioural tests for `salt.data.xAODReader` — ElementLink/PHYSLITE dereference."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from salt.data import (
    MultiSampleReader,
    SampleConfig,
    UprootGroupConfig,
    UprootReader,
    xAODGroupConfig,
    xAODReader,
)
from salt.graph.errors import ConfigError, SchemaError

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.tests._fixtures.easyjet_minitree import (  # noqa: E402
    build_fixture_arrays,
    write_minitree,
)
from salt.tests.unit.data.readers.test_uproot_reader import _FakeTree  # noqa: E402


@pytest.fixture
def ej_file(tmp_path: Path) -> tuple[Path, dict]:
    arrays = build_fixture_arrays(seed=99)
    return write_minitree(tmp_path / "ej.root", arrays), arrays


# --------------------------------------------------------------------------- #
# 1. group config validation (no file I/O)
# --------------------------------------------------------------------------- #


def test_link_without_unroll_raises() -> None:
    with pytest.raises(ConfigError, match="require unroll"):
        xAODReader(
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
        xAODGroupConfig(branches={"d0": "d0"}, jagged=True, link_branch="GhostTrack")


def test_group_link_only_on_jagged() -> None:
    with pytest.raises(ConfigError, match="only valid for jagged"):
        xAODGroupConfig(
            branches={"pt": "pt"},
            jagged=False,
            link_branch="GhostTrack",
            target_prefix="InDetTrackParticlesAuxDyn.",
        )


def test_parse_group_promotes_a_plain_base_config() -> None:
    """`xAODReader` accepts a plain base config and promotes it; the reverse redirects."""
    cfg = xAODReader._parse_group("jets", UprootGroupConfig(branches={"pt": "pt"}, jagged=False))
    assert isinstance(cfg, xAODGroupConfig)
    assert cfg.link_branch is None
    assert cfg.served_branches == {"pt": "pt"}

    with pytest.raises(ConfigError, match="xAODReader"):
        UprootReader._parse_group("jets", xAODGroupConfig(branches={"pt": "pt"}, jagged=False))


# --------------------------------------------------------------------------- #
# 2. ElementLink dereference (unroll=jets) — synthetic fake tree
# --------------------------------------------------------------------------- #

_GHOST_IDX = [[[0, 2], [1]], [[0]], [[2, 0, 1], [], [1]]]
_TRK_D0 = [[10.0, 11.0, 12.0], [20.0], [30.0, 31.0, 32.0]]
_TRK_Z0 = [[-1.0, -2.0, -3.0], [-4.0], [-5.0, -6.0, -7.0]]


def _linked_reader():
    return xAODReader(
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
# 3. lazy import contract
# --------------------------------------------------------------------------- #


def test_no_top_level_uproot_awkward_import() -> None:
    import salt.data.readers.xaod_reader as mod

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
# 4. 1:1 ElementLink JOIN — extra fields merged onto an existing group
# --------------------------------------------------------------------------- #
#
# Distinct from section 2: `link_branch` BUILDS a constituent stream (1:many);
# a join augments elements the group already serves (1:1), leaving its shape
# alone. Physical case: HH4b PHYSLITE p6697, jets -> GN2v01 via
# `AnalysisJetsAuxDyn.btaggingLink`.

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
    return xAODReader(groups={"jets": group}, tree="CollectionTree", unroll=None)


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
    cfg = xAODReader._parse_group("jets", dict(_JOIN_GROUP))
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
        xAODGroupConfig(branches={"pt": "pt"}, join_branch="btaggingLink")


def test_join_config_refuses_a_field_declared_twice() -> None:
    with pytest.raises(ConfigError, match="declared in both"):
        xAODGroupConfig(
            branches={"pt": "pt", "GN2v01_pb": "GN2v01_pb"},
            join_branch="btaggingLink",
            join_prefix="BTagging_AntiKt4EMPFlowAuxDyn.",
            join_branches={"GN2v01_pb": "GN2v01_pb"},
        )


def test_join_config_refuses_a_link_and_a_join_on_one_group() -> None:
    with pytest.raises(ConfigError, match="already sets 'link_branch'"):
        xAODGroupConfig(
            branches={"d0": "d0"},
            link_branch="GhostTrack",
            target_prefix="InDetTrackParticlesAuxDyn.",
            join_branch="btaggingLink",
            join_prefix="BTagging_AntiKt4EMPFlowAuxDyn.",
            join_branches={"GN2v01_pb": "GN2v01_pb"},
        )


def test_join_on_a_jagged_group_under_unroll_is_refused() -> None:
    with pytest.raises(ConfigError, match="two levels below the entry axis"):
        xAODReader(
            groups={
                "rows": {"branches": {"pt": "pt"}, "jagged": False},
                "jets": dict(_JOIN_GROUP),
            },
            unroll="rows",
        )


def test_join_reads_uproot_qualified_link_members() -> None:
    """A real POOL file names ElementLink members after the whole branch path, not
    bare; matching only the bare name falls through to the synthetic-fixture path.
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

    from salt.data.readers.xaod_reader import _link_member

    links = ak.zip(
        {"a.m_persIndex": ak.Array([[0]]), "b.m_persIndex": ak.Array([[1]])}, depth_limit=1
    )
    with pytest.raises(SchemaError, match="cannot tell which one"):
        _link_member(links, "m_persIndex")


# --------------------------------------------------------------------------- #
# 5. `_group_block` routes linked / joined / direct groups correctly
# --------------------------------------------------------------------------- #


def test_group_block_routes_linked_joined_and_direct() -> None:
    import awkward as ak

    reader = _linked_reader()
    t = _FakeTree({
        "AnalysisJetsAuxDyn.GhostTrack": ak.Array(_GHOST_IDX),
        "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
        "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
    })
    linked_cfg = reader.groups["tracks"]
    linked_block = reader._group_block(t, linked_cfg, ["d0", "z0"], 0, 3, {})
    expected_block = reader._read_linked_block(t, linked_cfg, ["d0", "z0"], 0, 3)
    assert linked_block["d0"].tolist() == expected_block["d0"].tolist() == _expected(_TRK_D0)
    assert linked_block["z0"].tolist() == expected_block["z0"].tolist()
    assert reader._direct_branches(linked_cfg, ["d0", "z0"]) == {}

    joined_reader = _joined_reader()
    jt = _join_tree()
    cfg = joined_reader.groups["jets"]
    shared = {"AnalysisJetsAuxDyn.pt": ak.Array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])}
    joined_block = joined_reader._group_block(jt, cfg, ["pt", "GN2v01_pb"], 0, 3, shared)
    assert joined_block["pt"].tolist() == shared["AnalysisJetsAuxDyn.pt"].tolist()
    assert joined_block["GN2v01_pb"].tolist() == [[0.3, 0.1, 0.2], [0.4], [0.6, 0.5]]
    assert joined_reader._direct_branches(cfg, ["pt", "GN2v01_pb"]) == {
        "pt": "AnalysisJetsAuxDyn.pt"
    }


# --------------------------------------------------------------------------- #
# 6. clone/restage/fingerprint identity
# --------------------------------------------------------------------------- #


def test_with_source_and_restage_keep_the_subclass(ej_file, tmp_path) -> None:
    """`with_source`/`restage`/`MultiSampleReader.with_source` all build `type(self)`."""
    path, _ = ej_file
    r = xAODReader(
        groups={
            "jets": {
                "branches": {"pt": "recojet_antikt4PFlow_pt_NOSYS"},
                "jagged": True,
                "pad_max": 4,
            }
        },
        tree="AnalysisMiniTree",
    )
    r2 = r.with_source(path, num=2, stage="val")
    assert type(r2) is xAODReader
    assert r2.stage == "val"
    assert isinstance(r2.groups["jets"], xAODGroupConfig)

    restaged = r.restage(tmp_path / "staged")
    assert type(restaged) is xAODReader

    multi = MultiSampleReader(
        samples=[SampleConfig(name="s", label=1, reader=r, sources={"val": str(path)})]
    )
    revived = multi.with_source("ignored", stage="val")
    assert type(revived.samples[0].reader) is xAODReader


def test_fingerprint_differs_by_reader_class(ej_file) -> None:
    path, _ = ej_file
    group = {
        "branches": {"pt": "recojet_antikt4PFlow_pt_NOSYS"},
        "jagged": True,
        "pad_max": 4,
    }
    r_x = xAODReader(groups={"jets": dict(group)}, filename=path, tree="AnalysisMiniTree")
    r_u = UprootReader(groups={"jets": dict(group)}, filename=path, tree="AnalysisMiniTree")

    fp_x = r_x.config_fingerprint()
    fp_u = r_u.config_fingerprint()
    assert fp_x["reader"] == "salt.data.readers.xaod_reader.xAODReader"
    assert fp_u["reader"] == "salt.data.readers.uproot_reader.UprootReader"

    base_x = {k: v for k, v in fp_x.items() if k != "reader"}
    base_u = {k: v for k, v in fp_u.items() if k != "reader"}
    assert base_x != base_u
