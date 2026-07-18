"""Unit tests for `PhysliteReader` / `XAODReader` ElementLink dereference (plan 54).

The real DAOD_PHYSLITE ElementLink record layout (``struct{m_persKey, m_persIndex}``
under a dotted aux branch) cannot be faithfully written by uproot and read back via
``t[branch]`` — so the *dereference logic* is exercised here on synthetic awkward
arrays (both the plain-int index layout and an in-memory ``m_persKey/m_persIndex``
record twin), driving the real `XAODReader._read_linked_block` via a fake tree; the
full real-file read (dotted branches + genuine ElementLink struct) is covered by
experiment 25.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from salt.core.data import PhysliteGroupConfig, PhysliteReader
from salt.core.data.xaod_reader import _pers_index, _pers_key  # noqa: PLC2701 - test internals
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.spec import Mode

ak = pytest.importorskip("awkward")
np = pytest.importorskip("numpy")


class _FakeBranch:
    """A tree branch whose ``.array`` slices a pre-built event-axis awkward array."""

    def __init__(self, arr):
        self._arr = arr

    def array(self, entry_start=None, entry_stop=None, library="ak"):  # noqa: ARG002
        return self._arr[entry_start:entry_stop]


class _FakeTree:
    """A minimal uproot-tree stand-in: maps exact branch names to `_FakeBranch`."""

    def __init__(self, branches):
        self._b = {name: _FakeBranch(arr) for name, arr in branches.items()}

    def __getitem__(self, name):
        return self._b[name]


def _reader() -> PhysliteReader:
    return PhysliteReader(
        groups={
            "jets": PhysliteGroupConfig(branches={"pt": "pt"}, jagged=False),
            "tracks": PhysliteGroupConfig(
                branches={"d0": "d0", "z0": "z0"},
                jagged=True,
                pad_max=4,
                link_branch="GhostTrack",
                target_collection="InDetTrackParticles",
            ),
        },
    )


# Synthetic event structure: 3 events, jets per event [2, 1, 3].
# Per jet: a list of persIndex values into that event's InDetTrackParticles.
_GHOST_IDX = [
    [[0, 2], [1]],     # event 0: jet0 -> tracks {0,2}; jet1 -> track {1}
    [[0]],             # event 1: jet0 -> track {0}
    [[2, 0, 1], [], [1]],  # event 2: jet0 -> {2,0,1}; jet1 -> {}; jet2 -> {1}
]
_TRK_D0 = [[10.0, 11.0, 12.0], [20.0], [30.0, 31.0, 32.0]]
_TRK_Z0 = [[-1.0, -2.0, -3.0], [-4.0], [-5.0, -6.0, -7.0]]


def _expected(field_vals):
    """The expected per-jet gathered track values (events flattened -> [jet][track])."""
    return [
        [field_vals[ev_idx][i] for i in jet_links]
        for ev_idx, ev_jets in enumerate(_GHOST_IDX)
        for jet_links in ev_jets
    ]


def _run_deref(links_arr):
    """Drive `XAODReader._read_linked_block` with a fake tree carrying `links_arr`."""
    reader = _reader()
    cfg = reader.groups["tracks"]
    t = _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": links_arr,
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )
    return reader._read_linked_block(t, cfg, ["d0", "z0"], 0, 3)  # noqa: SLF001


def test_deref_plain_int_index_gathers_target_by_persindex():
    """A plain-int GhostTrack index vector gathers the right per-jet track columns."""
    block = _run_deref(ak.Array(_GHOST_IDX))
    assert block["d0"].tolist() == _expected(_TRK_D0)
    assert block["z0"].tolist() == _expected(_TRK_Z0)


def test_deref_real_elementlink_struct_layout():
    """The real ``{m_persKey, m_persIndex}`` record twin dereferences identically."""
    # build a record array [event][jet][link]{m_persKey, m_persIndex} in memory
    key_vals = [[[123] * len(j) for j in ev] for ev in _GHOST_IDX]
    links = ak.zip(
        {"m_persKey": ak.Array(key_vals), "m_persIndex": ak.Array(_GHOST_IDX)},
        depth_limit=None,
    )
    assert "m_persIndex" in ak.fields(links)
    block = _run_deref(links)
    assert block["d0"].tolist() == _expected(_TRK_D0)
    assert block["z0"].tolist() == _expected(_TRK_Z0)


def test_pers_index_and_key_extraction():
    """`_pers_index`/`_pers_key` read the struct fields, or pass a plain int through."""
    plain = ak.Array(_GHOST_IDX)
    assert _pers_index(plain) is plain
    assert _pers_key(plain) is None
    links = ak.zip(
        {"m_persKey": ak.Array([[[1]]]), "m_persIndex": ak.Array([[[0]]])}, depth_limit=None
    )
    assert ak.fields(_pers_index(links)) == []  # the m_persIndex leaf, no record fields
    assert _pers_key(links) is not None


def test_null_elementlinks_dropped():
    """Null ElementLinks (m_persKey==0, e.g. thinned tracks) are dropped, not gathered."""
    # event 0: jet0 links -> persIndex {0(null), 2, 1(null)}; jet1 -> {1}
    idx = [[[0, 2, 0], [1]], [[0]], [[2, 0, 1], [], [1]]]
    key = [[[0, 490, 0], [490]], [[490]], [[490, 490, 490], [], [490]]]  # 0 == null
    links = ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(idx)}, depth_limit=None)
    block = _run_deref(links)  # flat jet order: e0j0, e0j1, e1j0, e2j0, e2j1, e2j2
    assert block["d0"][0].tolist() == [12.0]  # e0j0: nulls dropped, keep persIndex 2
    assert block["d0"][1].tolist() == [11.0]  # e0j1: persIndex 1
    assert block["d0"][2].tolist() == [20.0]  # e1j0: persIndex 0 -> d0[e1][0]
    assert block["d0"][3].tolist() == [32.0, 30.0, 31.0]  # e2j0: {2,0,1}, no nulls


def test_multi_target_container_raises():
    """Links spanning >1 target container (multiple m_persKey values) are rejected."""
    key_vals = [[[1, 2], [1]], [[1]], [[1, 1, 1], [], [1]]]  # jet0/ev0 spans keys {1,2}
    links = ak.zip(
        {"m_persKey": ak.Array(key_vals), "m_persIndex": ak.Array(_GHOST_IDX)},
        depth_limit=None,
    )
    with pytest.raises(SchemaError, match="multiple target containers"):
        _run_deref(links)


def test_group_config_link_validation():
    """link_branch/target_collection must be set together and only on jagged streams."""
    with pytest.raises(ConfigError, match="must be set together"):
        PhysliteGroupConfig(branches={"d0": "d0"}, jagged=True, link_branch="GhostTrack")
    with pytest.raises(ConfigError, match="only valid for jagged"):
        PhysliteGroupConfig(
            branches={"pt": "pt"},
            jagged=False,
            link_branch="GhostTrack",
            target_collection="InDetTrackParticles",
        )
    ok = PhysliteGroupConfig(
        branches={"d0": "d0"},
        jagged=True,
        pad_max=5,
        link_branch="GhostTrack",
        target_collection="InDetTrackParticles",
    )
    assert ok.is_linked


def test_physlite_defaults_and_declare_io():
    """PhysliteReader defaults to AnalysisJets; declare_io keeps concrete pad_max dims."""
    from salt.core.graph.spec import flatten_spec  # noqa: PLC0415

    reader = _reader()
    assert reader.jet_collection == "AnalysisJets"
    assert reader.aux_prefix == "AnalysisJetsAuxDyn."
    io = reader.declare_io(Mode.FIT)
    keys = set(flatten_spec(io.produces))
    # tracks stream jagged with pad_max=4 -> raw.tracks + masks.tracks; jets scalar
    assert "raw.tracks" in keys
    assert "masks.tracks" in keys
    assert "raw.jets" in keys


def test_no_top_level_uproot_awkward_import():
    """The physlite + xaod reader modules must lazy-import uproot/awkward."""
    import salt.core.data.physlite_reader as pmod  # noqa: PLC0415
    import salt.core.data.xaod_reader as xmod  # noqa: PLC0415

    for mod in (pmod, xmod):
        tree = ast.parse(Path(mod.__file__).read_text())
        top: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        assert "uproot" not in top
        assert "awkward" not in top
