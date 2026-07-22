"""Equivalence gate (plan 02 T3): the unified `UprootReader` produces byte-identical
output to the legacy `EasyjetReader`/`XAODReader` presets it replaces.

Temporary scaffolding — runs old and new readers side by side on every existing
fixture and asserts bit-identical `read()` arrays, schema, ``__len__`` and pickle
round-trip. Deleted together with the old classes once the gate is green.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from salt.data import EasyjetGroupConfig, EasyjetReader, UprootReader
from salt.graph.spec import Mode

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.tests._fixtures.easyjet_minitree import (  # noqa: E402
    build_fixture_arrays,
    write_minitree,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _assert_read_equal(old: dict, new: dict) -> None:
    """Assert two reader read() outputs are byte-identical (keys, dtypes, values)."""
    assert set(old) == set(new), f"key mismatch: {set(old) ^ set(new)}"
    for key in old:
        a, b = old[key], new[key]
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        if a.dtype.names:  # structured array — compare field by field
            assert a.dtype.names == b.dtype.names, f"{key}: field order differs"
            for f in a.dtype.names:
                np.testing.assert_array_equal(a[f], b[f], err_msg=f"{key}.{f}")
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)


def _assert_equivalent(old_reader, new_reader, slices, mode=Mode.FIT) -> None:
    """Full equivalence: len, schema, per-slice reads, pickle round-trip re-read."""
    assert len(old_reader) == len(new_reader)
    old_reader.prepare()
    new_reader.prepare()
    for s in old_reader.groups:
        assert dict(old_reader.schema.groups[s].fields) == dict(new_reader.schema.groups[s].fields)
    for sl in slices:
        _assert_read_equal(old_reader.read(sl, mode), new_reader.read(sl, mode))
    # pickle round-trip: unpickled reader re-reads identically
    old_rt = pickle.loads(pickle.dumps(old_reader))
    new_rt = pickle.loads(pickle.dumps(new_reader))
    assert len(old_rt) == len(new_rt)
    _assert_read_equal(
        old_rt.read(slice(0, len(old_rt)), mode), new_rt.read(slice(0, len(new_rt)), mode)
    )


# --------------------------------------------------------------------------- #
# 1. easyjet (unroll=None) — synthetic AnalysisMiniTree fixture (always runs)
# --------------------------------------------------------------------------- #

_EJ_JET = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "phi": "recojet_antikt4PFlow_phi",
    "m": "recojet_antikt4PFlow_m_NOSYS",
    "GN2v01_pb": "recojet_antikt4PFlow_GN2v01_pb",
    "GN2v01_pc": "recojet_antikt4PFlow_GN2v01_pc",
    "GN2v01_pu": "recojet_antikt4PFlow_GN2v01_pu",
    "label": "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
}
_EJ_EVENT = {"eventNumber": "eventNumber", "mcChannelNumber": "mcChannelNumber"}


def _ej_old(truncate):
    return {
        "jets": EasyjetGroupConfig(branches=dict(_EJ_JET), jagged=True, truncate=truncate),
        "event": EasyjetGroupConfig(branches=dict(_EJ_EVENT), jagged=False),
    }


def _ej_new(truncate):
    return {
        "jets": {"branches": dict(_EJ_JET), "jagged": True, "truncate": truncate},
        "event": {"branches": dict(_EJ_EVENT), "jagged": False},
    }


@pytest.fixture
def ej_single(tmp_path: Path) -> Path:
    return write_minitree(tmp_path / "ej.root", build_fixture_arrays(seed=1234))


@pytest.fixture
def ej_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ej_dir"
    d.mkdir()
    write_minitree(d / "file_000001.root", build_fixture_arrays(seed=1))
    write_minitree(d / "file_000002.root", build_fixture_arrays(seed=2))
    return d


@pytest.mark.parametrize("truncate", [8, None])
def test_easyjet_equivalence_single(ej_single: Path, truncate) -> None:
    old = EasyjetReader(groups=_ej_old(truncate), filename=ej_single, tree="AnalysisMiniTree")
    new = UprootReader(
        groups=_ej_new(truncate), filename=ej_single, tree="AnalysisMiniTree", unroll=None
    )
    n = len(old)
    _assert_equivalent(old, new, [slice(0, n), slice(1, 3), slice(0, 2)])


def test_easyjet_equivalence_single_test_mode(ej_single: Path) -> None:
    old = EasyjetReader(groups=_ej_old(8), filename=ej_single, tree="AnalysisMiniTree")
    new = UprootReader(groups=_ej_new(8), filename=ej_single, tree="AnalysisMiniTree", unroll=None)
    _assert_read_equal(old.read(slice(1, 3), Mode.TEST), new.read(slice(1, 3), Mode.TEST))


def test_easyjet_equivalence_multifile_boundary(ej_dir: Path) -> None:
    old = EasyjetReader(groups=_ej_old(8), filename=ej_dir, tree="AnalysisMiniTree")
    new = UprootReader(groups=_ej_new(8), filename=ej_dir, tree="AnalysisMiniTree", unroll=None)
    n = len(old)
    na = build_fixture_arrays(seed=1)["n_events"]
    _assert_equivalent(old, new, [slice(0, n), slice(na - 1, na + 2), slice(3, 9)])


# --------------------------------------------------------------------------- #
# 2. xAOD direct (unroll=jets) — real DAOD_FTAG1LITE sample (skips if absent)
# --------------------------------------------------------------------------- #

_F1L_SAMPLE = Path(
    os.environ.get(
        "FTAG1LITE_SAMPLE",
        "/data/atlas_samples/test-athena-derivations/ftag1lite_stock_smoke_100evt/smoke_100evt/"
        "Athena_main_latest_20260424-082808/DAOD_FTAG1LITE.601589.e8549_s4159_r15530.pool.root",
    )
)
_AUX = "AntiKt4EMPFlowJetsAuxDyn."
_F1L_JET = {
    "pt": "pt", "eta": "eta_calibrated", "phi": "phi_calibrated", "mass": "mass_calibrated",
    "GN3_pb": "GN3EPCLV01_pb", "flavour_label": "HadronConeExclTruthLabelID",
}
_F1L_TRK = {
    "d0": "ft1l_trk_d0_bf16", "z0SinTheta": "ft1l_trk_z0SinTheta_bf16", "pt": "ft1l_trk_pt_bf16",
    "numberOfPixelHits": "ft1l_trk_numberOfPixelHits", "origin": "ft1l_trk_ftagTruthOriginLabel",
    "vertexing": "ft1l_trk_ftagTruthVertexIndex",
}

pytestmark_f1l = pytest.mark.skipif(
    not _F1L_SAMPLE.exists(), reason=f"FTAG1LITE sample not found at {_F1L_SAMPLE}"
)


def _f1l_new(pad_max, cuts=None):
    return UprootReader(
        groups={
            "jets": {"branches": dict(_F1L_JET), "prefix": _AUX, "jagged": False},
            "tracks": {
                "branches": dict(_F1L_TRK), "prefix": _AUX, "jagged": True, "pad_max": pad_max
            },
        },
        filename=_F1L_SAMPLE, tree="CollectionTree", unroll="jets", cuts=cuts,
    )


def _f1l_old(pad_max, cuts=None):
    from salt.data import FTAG1LiteGroupConfig, FTAG1LiteReader

    return FTAG1LiteReader(
        groups={
            "jets": FTAG1LiteGroupConfig(branches=dict(_F1L_JET), jagged=False),
            "tracks": FTAG1LiteGroupConfig(branches=dict(_F1L_TRK), jagged=True, pad_max=pad_max),
        },
        filename=_F1L_SAMPLE, tree="CollectionTree", cuts=cuts,
    )


@pytestmark_f1l
@pytest.mark.parametrize("pad_max", [40, 4])
def test_ftag1lite_equivalence(pad_max) -> None:
    old, new = _f1l_old(pad_max), _f1l_new(pad_max)
    n = len(old)
    slices = [slice(0, n), slice(0, 7), slice(max(0, n - 5), n)]
    _assert_equivalent(old, new, slices)


@pytestmark_f1l
def test_ftag1lite_equivalence_with_cut() -> None:
    from salt.data import Cut, CutSpec

    for cut in (Cut("pt", ">=", 50_000.0), Cut("flavour_label", "==", 5)):
        spec = CutSpec(global_cuts=(cut,))
        old, new = _f1l_old(40, spec), _f1l_new(40, spec)
        n = len(old)
        _assert_equivalent(old, new, [slice(0, n)])


@pytestmark_f1l
def test_ftag1lite_equivalence_cut_offbranch() -> None:
    """Cut variable not among configured branches — read at index-build on both."""
    from salt.data import Cut, CutSpec, FTAG1LiteGroupConfig, FTAG1LiteReader

    spec = CutSpec(global_cuts=(Cut("HadronConeExclTruthLabelID", "==", 5),))
    old = FTAG1LiteReader(
        groups={
            "jets": FTAG1LiteGroupConfig(branches={"pt": "pt"}, jagged=False),
            "tracks": FTAG1LiteGroupConfig(
                branches={"d0": "ft1l_trk_d0_bf16"}, jagged=True, pad_max=40
            ),
        },
        filename=_F1L_SAMPLE, cuts=spec,
    )
    new = UprootReader(
        groups={
            "jets": {"branches": {"pt": "pt"}, "prefix": _AUX, "jagged": False},
            "tracks": {
                "branches": {"d0": "ft1l_trk_d0_bf16"}, "prefix": _AUX, "jagged": True,
                "pad_max": 40,
            },
        },
        filename=_F1L_SAMPLE, tree="CollectionTree", unroll="jets", cuts=spec,
    )
    n = len(old)
    _assert_equivalent(old, new, [slice(0, n)])


# --------------------------------------------------------------------------- #
# 3. ElementLink dereference (unroll=jets) — old vs new _read_linked_block
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


def _new_linked_reader():
    return UprootReader(
        groups={
            "jets": {"branches": {"pt": "pt"}, "prefix": "AnalysisJetsAuxDyn.", "jagged": False},
            "tracks": {
                "branches": {"d0": "d0", "z0": "z0"}, "jagged": True, "pad_max": 4,
                "link_branch": "GhostTrack", "target_collection": "InDetTrackParticles",
            },
        },
        tree="CollectionTree", unroll="jets",
    )


def _old_linked_reader():
    from salt.data import PhysliteGroupConfig, PhysliteReader

    return PhysliteReader(
        groups={
            "jets": PhysliteGroupConfig(branches={"pt": "pt"}, jagged=False),
            "tracks": PhysliteGroupConfig(
                branches={"d0": "d0", "z0": "z0"}, jagged=True, pad_max=4,
                link_branch="GhostTrack", target_collection="InDetTrackParticles",
            ),
        },
    )


def _fake_tree():
    import awkward as ak

    return _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": ak.Array(_GHOST_IDX),
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )


def test_linked_deref_equivalence_plain_index() -> None:
    old, new = _old_linked_reader(), _new_linked_reader()
    t = _fake_tree()
    old_block = old._read_linked_block(t, old.groups["tracks"], ["d0", "z0"], 0, 3)
    new_block = new._read_linked_block(t, new.groups["tracks"], ["d0", "z0"], 0, 3)
    for f in ("d0", "z0"):
        assert old_block[f].tolist() == new_block[f].tolist()


def test_linked_deref_equivalence_struct_and_nulls() -> None:
    import awkward as ak

    idx = [[[0, 2, 0], [1]], [[0]], [[2, 0, 1], [], [1]]]
    key = [[[0, 490, 0], [490]], [[490]], [[490, 490, 490], [], [490]]]
    links = ak.zip({"m_persKey": ak.Array(key), "m_persIndex": ak.Array(idx)}, depth_limit=None)
    t = _FakeTree(
        {
            "AnalysisJetsAuxDyn.GhostTrack": links,
            "InDetTrackParticlesAuxDyn.d0": ak.Array(_TRK_D0),
            "InDetTrackParticlesAuxDyn.z0": ak.Array(_TRK_Z0),
        }
    )
    old, new = _old_linked_reader(), _new_linked_reader()
    old_block = old._read_linked_block(t, old.groups["tracks"], ["d0", "z0"], 0, 3)
    new_block = new._read_linked_block(t, new.groups["tracks"], ["d0", "z0"], 0, 3)
    for f in ("d0", "z0"):
        assert old_block[f].tolist() == new_block[f].tolist()
