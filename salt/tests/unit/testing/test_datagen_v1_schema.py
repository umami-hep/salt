"""Pin ``SALT_V1_SCHEMA`` to the salt-v1 training file the test suite consumes, per flag combo.

The expected group/field lists are the salt-v1 training-file layout the test suite is written
against, hard-coded, so a schema edit that drops or reorders a column fails here first.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from salt.testing.datagen import SALT_V1_SCHEMA, generate_data, load_schema, write_dataset
from salt.tests._fixtures.dummy_h5 import dummy_h5
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

N_JETS = 1000

JET_FLOATS = [
    "pt",
    "eta",
    "mass",
    "pt_btagJes",
    "eta_btagJes",
    # truth / regression-target / weight columns, between the kinematics and softMuon_*
    "ptFromTruthDressedWZJet",
    "HadronConeExclTruthLabelPt",
    "HadronConeExclTruthLabelLxy",
    "n_tracks",
    "n_truth_promptLepton",
    "sample_weight",
    "softMuon_pt",
    "softMuon_dR",
    "softMuon_eta",
    "softMuon_phi",
    "softMuon_qOverPratio",
    "softMuon_momentumBalanceSignificance",
    "softMuon_scatteringNeighbourSignificance",
    "softMuon_pTrel",
    "softMuon_ip3dD0",
    "softMuon_ip3dZ0",
    "softMuon_ip3dD0Significance",
    "softMuon_ip3dZ0Significance",
    "softMuon_ip3dD0Uncertainty",
    "softMuon_ip3dZ0Uncertainty",
    "R10TruthLabel_R22v1",
    "R10TruthLabel_R22v1_TruthJetMass",
    "R10TruthLabel_R22v1_TruthJetPt",
    "GN2Xv00_phbb",
    "GN2Xv00_phcc",
    "GN2Xv00_ptop",
    "GN2Xv00_pqcd",
    "GN2XWithMassv00_phbb",
    "GN2XWithMassv00_phcc",
    "GN2XWithMassv00_ptop",
    "GN2XWithMassv00_pqcd",
    "Xbb2020v3_Higgs",
    "Xbb2020v3_Top",
    "Xbb2020v3_QCD",
]
JET_LABELS = ["flavour_label", "HadronConeExclTruthLabelID", "HadronGhostInitialTruthLabelPdgId"]

TRACK_FLOATS = [
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "IP3D_signed_d0_significance",
    "IP3D_signed_z0_significance",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinThetaSignificance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
    "numberOfPixelHoles",
    "numberOfSCTHoles",
    "pt",
    "eta",
    "phi",
    "subjetIndex",
    "leptonID",
    "muon_quality",
    "muon_qOverPratio",
    "muon_momentumBalanceSignificance",
    "muon_scatteringNeighbourSignificance",
]
TRACK_LABELS = [
    "ftagTruthOriginLabel",
    "ftagTruthTypeLabel",
    "ftagTruthVertexIndex",
    "ftagTruthParentBarcode",
]

FLOW_FLOATS = ["pt", "energy", "deta", "dphi", "isCharged"]

ELECTRON_FLOATS = [
    "pt",
    "ptfrac",
    "ptrel",
    "dr",
    "abs_eta",
    "eta",
    "phi",
    "ftag_et",
    "qOverP",
    "d0RelativeToBeamspot",
    "d0RelativeToBeamspotSignificance",
    "ftag_z0AlongBeamspotSignificance",
    "ftag_ptVarCone30OverPt",
    "numberOfPixelHits",
    "numberOfSCTHitsInclDead",
    "ftag_deltaPOverP",
    "eProbabilityHT",
    "deltaEta1",
    "deltaPhiRescaled2",
    "ftag_energyOverP",
    "Rhad",
    "Rhad1",
    "Eratio",
    "weta2",
    "Rphi",
    "Reta",
    "wtots1",
    "f1",
    "f3",
]
ELECTRON_LABELS = ["ftagTruthOriginLabel", "ftagTruthVertexIndex"]

HADRON_FLOATS = ["pt", "Lxy", "deta", "dphi", "mass"]
HADRON_INTS = ["barcode", "flavour"]

FLOATS = {
    "jets": JET_FLOATS,
    "tracks": TRACK_FLOATS,
    "tracks_dr": TRACK_FLOATS,
    "tracks_ghost": TRACK_FLOATS,
    "flows": FLOW_FLOATS,
    "flow": FLOW_FLOATS,
    "electrons": ELECTRON_FLOATS,
    "truth_hadrons": HADRON_FLOATS,
}
INTS = {
    "jets": JET_LABELS,
    "tracks": TRACK_LABELS,
    "tracks_dr": TRACK_LABELS,
    "tracks_ghost": TRACK_LABELS,
    "flows": [],
    "flow": [],
    "electrons": ELECTRON_LABELS,
    "truth_hadrons": HADRON_INTS,
}
CONSTITUENTS = {"tracks", "tracks_dr", "tracks_ghost", "flows", "flow", "electrons", "truth_hadrons"}
GROUPS = set(FLOATS)

# every flag combination the migrated importers pass to dummy_h5
COMBOS: dict[str, dict[str, bool]] = {
    "default": {},
    "inc_taus": {"inc_taus": True},
    "is_gn3": {"is_gn3": True},
    "make_xbb": {"make_xbb": True},
    "inc_params": {"inc_params": True},
}
LABEL_VALUES = {
    "default": {0, 1, 2},
    "inc_taus": {0, 1, 2},
    "is_gn3": {0, 1, 2, 3, 4, 5},
    "make_xbb": {0, 1, 2, 3},
    "inc_params": {0, 1, 2},
}
CLASS_NAMES = {
    "default": ["bjets", "cjets", "ujets"],
    "inc_taus": ["bjets", "cjets", "ujets", "taujets"],
    # the names the shipped GN3 configs declare in class_names
    "is_gn3": ["ghostbjets", "ghostcjets", "ghostsjets", "ghostudjets", "ghostgjets", "ghosttaujets"],
    "make_xbb": ["hbb", "hcc", "top", "qcd"],
    "inc_params": ["bjets", "cjets", "ujets"],
}
FLAVOUR_WEIGHTS_LEN = {"default": 4, "inc_taus": 4, "is_gn3": 6, "make_xbb": 4, "inc_params": 4}


def _names(attr) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in attr]


@pytest.fixture(scope="module", params=list(COMBOS))
def generated(request) -> tuple[str, dict[str, np.ndarray]]:
    """``(combo, generate_data(SALT_V1_SCHEMA, flags))`` for every combo."""
    return request.param, generate_data(load_schema(SALT_V1_SCHEMA), COMBOS[request.param])


@pytest.fixture(scope="module", params=list(COMBOS))
def written(request, tmp_path_factory):
    """``(combo, h5, norm_dict, class_dict)`` written by ``dummy_h5`` for every combo."""
    base = tmp_path_factory.mktemp(f"v1_{request.param}")
    return request.param, *dummy_h5(base, **COMBOS[request.param])


def test_group_set(generated):
    """Exactly the salt-v1 groups; ``parameters`` only with inc_params."""
    combo, data = generated
    expected = GROUPS | ({"parameters"} if combo == "inc_params" else set())
    assert set(data) == expected


def test_field_names_and_dtypes(generated):
    """Every group carries exactly the salt-v1 columns, in order, f4/i4/bool."""
    _, data = generated
    for group in GROUPS:
        arr = data[group]
        valid = ["valid"] if group in CONSTITUENTS else []
        assert list(arr.dtype.names) == FLOATS[group] + INTS[group] + valid, group
        for name in FLOATS[group]:
            assert arr.dtype[name] == np.dtype("f4"), (group, name)
        for name in INTS[group]:
            assert arr.dtype[name] == np.dtype("i4"), (group, name)
        if valid:
            assert arr.dtype["valid"] == np.dtype(bool), group


def test_shapes(generated):
    """1000 jets; constituent widths tracks 40, flows 40, electrons 10, hadrons 5."""
    _, data = generated
    widths = {"tracks": 40, "tracks_dr": 40, "tracks_ghost": 40, "flows": 40, "flow": 40}
    widths |= {"electrons": 10, "truth_hadrons": 5}
    assert data["jets"].shape == (N_JETS,)
    for group, width in widths.items():
        assert data[group].shape == (N_JETS, width), group


def test_aliases_are_copies(generated):
    """tracks_dr/tracks_ghost duplicate tracks and flow duplicates flows, byte for byte."""
    _, data = generated
    for alias, src in (("tracks_dr", "tracks"), ("tracks_ghost", "tracks"), ("flow", "flows")):
        assert data[alias].tobytes() == data[src].tobytes(), alias


def test_lxy_nan_exactly_for_light_jets(generated):
    """``HadronConeExclTruthLabelLxy`` is NaN iff ``flavour_label == 0``."""
    _, data = generated
    jets = data["jets"]
    np.testing.assert_array_equal(
        np.isnan(jets["HadronConeExclTruthLabelLxy"]), jets["flavour_label"] == 0
    )
    other = [n for n in JET_FLOATS if n != "HadronConeExclTruthLabelLxy"]
    for name in other:
        assert np.isfinite(jets[name]).all(), name


def test_jet_label_values(generated):
    """flavour_label range per flag; the other jet labels draw from their declared sets."""
    combo, data = generated
    jets = data["jets"]
    assert set(np.unique(jets["flavour_label"]).tolist()) == LABEL_VALUES[combo]
    assert set(np.unique(jets["HadronConeExclTruthLabelID"]).tolist()) <= {0, 4, 5}
    pdg = set(np.unique(jets["HadronGhostInitialTruthLabelPdgId"]).tolist())
    assert pdg <= {0, 15, -511, 521, 10511, 441}
    assert ((jets["sample_weight"] >= 0) & (jets["sample_weight"] < 1)).all()


def test_valid_slots_lead_each_row(generated):
    """Valid constituents are packed to the front; tracks/hadrons have >=1 valid slot."""
    _, data = generated
    for group in CONSTITUENTS:
        valid = data[group]["valid"].astype(np.int8)
        assert (np.diff(valid, axis=1) <= 0).all(), group
    assert data["tracks"]["valid"][:, 0].all()
    assert data["truth_hadrons"]["valid"][:, 0].all()


def test_invalid_tracks_are_filled(generated):
    """Invalid tracks: floats NaN, labels -1, ftagTruthTypeLabel 0; valid floats finite."""
    _, data = generated
    tracks = data["tracks"]
    invalid = ~tracks["valid"]
    for name in TRACK_FLOATS:
        np.testing.assert_array_equal(np.isnan(tracks[name]), invalid, err_msg=name)
    for name in ("ftagTruthOriginLabel", "ftagTruthVertexIndex", "ftagTruthParentBarcode"):
        assert (tracks[name][invalid] == -1).all(), name
    assert (tracks["ftagTruthTypeLabel"][invalid] == 0).all()
    valid_type = set(np.unique(tracks["ftagTruthTypeLabel"][~invalid]).tolist())
    assert valid_type <= {-2, -3, 5, -5, 6, -6}


def test_invalid_flows_are_nan(generated):
    """Invalid flow slots are NaN in every feature; valid ones finite."""
    _, data = generated
    flows = data["flows"]
    for name in FLOW_FLOATS:
        np.testing.assert_array_equal(np.isnan(flows[name]), ~flows["valid"], err_msg=name)


def test_electrons_keep_values_in_invalid_slots(generated):
    """``mask_invalid: false``: no electron feature is ever NaN, valid or not."""
    _, data = generated
    for name in ELECTRON_FLOATS:
        assert np.isfinite(data["electrons"][name]).all(), name


def test_hadrons_fill_sort_and_barcodes(generated):
    """Invalid hadrons NaN/-1; valid flavours sorted descending; barcodes unique per jet."""
    _, data = generated
    hadrons = data["truth_hadrons"]
    valid = hadrons["valid"]
    for name in HADRON_FLOATS:
        np.testing.assert_array_equal(np.isnan(hadrons[name]), ~valid, err_msg=name)
    for name in HADRON_INTS:
        assert (hadrons[name][~valid] == -1).all(), name
    assert set(np.unique(hadrons["flavour"][valid]).tolist()) <= {-1, 4, 5}
    later_valid = valid[:, 1:]
    assert (np.diff(hadrons["flavour"], axis=1)[later_valid] <= 0).all()
    for row in range(N_JETS):
        ids = hadrons["barcode"][row][valid[row]]
        assert len(np.unique(ids)) == len(ids), row
        assert ((ids >= 0) & (ids < 10000)).all(), row


def test_track_parent_barcode_links_a_valid_hadron(generated):
    """Every valid track's ``ftagTruthParentBarcode`` is a valid hadron barcode of its jet."""
    _, data = generated
    tracks, hadrons = data["tracks"], data["truth_hadrons"]
    for row in range(N_JETS):
        parents = tracks["ftagTruthParentBarcode"][row][tracks["valid"][row]]
        pool = hadrons["barcode"][row][hadrons["valid"][row]]
        assert np.isin(parents, pool).all(), row


def test_parameters_mass_values(generated):
    """``parameters.mass`` draws from {5, 16, 25, 40, 55}, one per jet (inc_params only)."""
    combo, data = generated
    if combo != "inc_params":
        assert "parameters" not in data
        return
    params = data["parameters"]
    assert list(params.dtype.names) == ["mass"]
    assert params.dtype["mass"] == np.dtype("f4")
    assert params.shape == (N_JETS,)
    assert set(np.unique(params["mass"]).tolist()) <= {5.0, 16.0, 25.0, 40.0, 55.0}


def test_file_attrs_and_class_names(written):
    """unique_jets/config file attrs and the flavour_label class-name attr per flag."""
    combo, h5, _, _ = written
    with h5py.File(h5) as f:
        assert int(f.attrs["unique_jets"]) == N_JETS
        assert f.attrs["config"] == "{}"
        assert "_flags" not in f.attrs
        assert _names(f["jets"].attrs["flavour_label"]) == CLASS_NAMES[combo]
        assert set(f) == GROUPS | ({"parameters"} if combo == "inc_params" else set())


def test_computed_dicts(written):
    """The norm dict covers every float feature; class-dict lengths match the heads."""
    combo, _, nd, cd = written
    norm = yaml.safe_load(nd.read_text())
    assert set(norm) == GROUPS
    for group, names in FLOATS.items():
        assert list(norm[group]) == names, group
        for name, stats in norm[group].items():
            assert np.isfinite(stats["mean"]), (group, name)
            assert stats["std"] > 0, (group, name)
    classes = yaml.safe_load(cd.read_text())
    assert len(classes["jets"]["flavour_label"]) == FLAVOUR_WEIGHTS_LEN[combo]
    for group in ("tracks", "tracks_dr", "tracks_ghost"):
        assert len(classes[group]["ftagTruthOriginLabel"]) == 8, group
    if combo in {"default", "inc_taus", "inc_params"}:
        assert classes["jets"]["flavour_label"][3] == 0.0  # declared, never drawn


class TestWriteDataset:
    def test_returns_what_it_wrote(self, tmp_path):
        """The returned arrays are byte-identical to the written datasets."""
        data = write_dataset(tmp_path / "a.h5", SALT_V1_SCHEMA)
        with h5py.File(tmp_path / "a.h5") as f:
            assert set(f) == set(data)
            for name, arr in data.items():
                assert f[name][:].tobytes() == arr.tobytes(), name

    def test_dicts_written_only_when_asked(self, tmp_path):
        """No norm/class dict paths -> only the H5 is written."""
        write_dataset(tmp_path / "a.h5", SALT_V1_SCHEMA)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.h5"]

    def test_deterministic_and_seed_override(self, tmp_path):
        """Same schema seed -> identical data; ``seed=`` changes it."""
        first = write_dataset(tmp_path / "a.h5", SALT_V1_SCHEMA)
        again = write_dataset(tmp_path / "b.h5", SALT_V1_SCHEMA)
        other = write_dataset(tmp_path / "c.h5", SALT_V1_SCHEMA, seed=7)
        assert first["jets"].tobytes() == again["jets"].tobytes()
        assert first["jets"].tobytes() != other["jets"].tobytes()

    def test_accepts_a_parsed_schema(self, tmp_path):
        """A parsed ``Schema`` and its path produce the same file contents."""
        from_path = write_dataset(tmp_path / "a.h5", SALT_V1_SCHEMA)
        from_obj = write_dataset(tmp_path / "b.h5", load_schema(SALT_V1_SCHEMA))
        assert from_path["tracks"].tobytes() == from_obj["tracks"].tobytes()


class TestDummyH5:
    def test_paths(self, tmp_path):
        """Returns ``(dir/name, dir/dummy_norm_dict.yaml, dir/dummy_class_dict.yaml)``, all written."""
        h5, nd, cd = dummy_h5(tmp_path, name="x_ttbar.h5")
        assert (h5, nd, cd) == (
            tmp_path / "x_ttbar.h5",
            tmp_path / "dummy_norm_dict.yaml",
            tmp_path / "dummy_class_dict.yaml",
        )
        assert h5.is_file() and nd.is_file() and cd.is_file()

    def test_keeps_parity_dicts_in_the_same_dir(self, tmp_path):
        """``write_parity_norm_dict``'s norm_dict.yaml/class_dict.yaml survive a dummy_h5 call."""
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        before = nd.read_text(), cd.read_text()
        dummy_h5(tmp_path)
        assert (nd.read_text(), cd.read_text()) == before

    def test_unknown_flag_raises(self, tmp_path):
        """A misspelt flag is a TypeError, not a silently ignored kwarg."""
        with pytest.raises(TypeError, match="inc_tau"):
            dummy_h5(tmp_path, inc_tau=True)
        assert not list(tmp_path.iterdir())
