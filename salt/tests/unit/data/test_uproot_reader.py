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
        filename=path, tree="AnalysisMiniTree", unroll=None,
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
        filename=path, tree="AnalysisMiniTree",
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
        filename=path, tree="AnalysisMiniTree", unroll="jets",
    )
    njets = np.array(arrays["njets"])
    assert len(reader) == int(njets.sum())  # one row per jet, events flattened away
    out = reader.read(slice(0, len(reader)), Mode.FIT)
    raw = out["raw.jets"]
    assert raw.shape == (int(njets.sum()),)
    assert "masks.jets" not in out  # jagged=False row-carrier -> no pad mask
    # values == the flattened per-event fixture arrays
    expected_pt = np.concatenate(
        [arrays["recojet_antikt4PFlow_pt_NOSYS"][ev] for ev in range(len(njets))]
    ).astype(raw["pt"].dtype)
    np.testing.assert_array_equal(raw["pt"], expected_pt)


def test_jet_rows_cut_on_pt(ej_file) -> None:
    from salt.data import Cut, CutSpec

    path, arrays = ej_file
    thresh = 100_000.0
    reader = UprootReader(
        groups={"jets": {"branches": dict(_JET), "jagged": False}},
        filename=path, tree="AnalysisMiniTree", unroll="jets",
        cuts=CutSpec(global_cuts=(Cut("pt", ">", thresh),)),
    )
    all_pt = np.concatenate(
        [arrays["recojet_antikt4PFlow_pt_NOSYS"][ev] for ev in range(arrays["n_events"])]
    )
    assert len(reader) == int((all_pt > thresh).sum())
    served = reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]
    assert np.all(served["pt"] > thresh)


def test_pre_flattened_entry_is_jet(ej_file) -> None:
    """A per-entry scalar stream with unroll=None serves each entry directly (pre-flattened)."""
    path, arrays = ej_file
    reader = UprootReader(
        groups={"event": {"branches": dict(_EVENT), "jagged": False}},
        filename=path, tree="AnalysisMiniTree", unroll=None,
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
        filename=path, tree="AnalysisMiniTree",
    )
    b = UprootReader(
        groups={
            "event": {"branches": dict(_EVENT), "jagged": False},
            "jets": {"branches": dict(_JET), "jagged": True, "truncate": 8},
        },
        filename=path, tree="AnalysisMiniTree",
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
        {"branches": {"d0": "d0"}, "jagged": True, "link_branch": "GhostTrack",
         "target_collection": "InDetTrackParticles"},
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
                    "branches": {"d0": "d0"}, "jagged": True, "link_branch": "GhostTrack",
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
            branches={"pt": "pt"}, jagged=False, link_branch="GhostTrack",
            target_prefix="InDetTrackParticlesAuxDyn.",
        )


def test_empty_groups_raises() -> None:
    with pytest.raises(ConfigError, match="at least one group"):
        UprootReader(groups={})


def test_missing_branch_raises(ej_file) -> None:
    path, _ = ej_file
    reader = UprootReader(
        groups={"jets": {"branches": {"pt": "NOT_A_BRANCH"}, "jagged": True, "truncate": 4}},
        filename=path, tree="AnalysisMiniTree",
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
                "branches": {"d0": "d0", "z0": "z0"}, "jagged": True, "pad_max": 4,
                "link_branch": "GhostTrack", "target_prefix": "InDetTrackParticlesAuxDyn.",
            },
        },
        tree="CollectionTree", unroll="jets",
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
    t = _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": ak.Array(_GHOST_IDX),
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )
    block = reader._read_linked_block(t, reader.groups["tracks"], ["d0", "z0"], 0, 3)
    assert block["d0"].tolist() == _expected(_TRK_D0)
    assert block["z0"].tolist() == _expected(_TRK_Z0)


def test_linked_deref_struct_and_null_links() -> None:
    import awkward as ak

    idx = [[[0, 2, 0], [1]], [[0]], [[2, 0, 1], [], [1]]]
    key = [[[0, 490, 0], [490]], [[490]], [[490, 490, 490], [], [490]]]  # 0 == null
    links = ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(idx)}, depth_limit=None)
    reader = _linked_reader()
    t = _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": links,
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )
    block = reader._read_linked_block(t, reader.groups["tracks"], ["d0", "z0"], 0, 3)
    assert block["d0"][0].tolist() == [12.0]  # e0j0: nulls dropped, keep persIndex 2
    assert block["d0"][3].tolist() == [32.0, 30.0, 31.0]  # e2j0: {2,0,1}


def test_linked_deref_multi_container_raises() -> None:
    import awkward as ak

    key = [[[1, 2], [1]], [[1]], [[1, 1, 1], [], [1]]]  # jet0/ev0 spans keys {1,2}
    links = ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(_GHOST_IDX)},
                   depth_limit=None)
    reader = _linked_reader()
    t = _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": links,
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )
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
