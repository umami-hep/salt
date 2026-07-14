"""Synthetic-input writers for the test suite: the dummy structured-H5
writer and the dummy norm/class-dict writer.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.utils.array_utils import join_structured_arrays

__all__ = [
    "ELECTRON_VARS",
    "FLOW_VARS",
    "HADRON_VARS",
    "JET_VARS",
    "TRACK_VARS",
    "write_dummy_file",
    "write_dummy_norm_dict",
]

DEFAULT_NTRACK = 40

# Example feature name lists used by the dummy writers below
JET_VARS = [
    "pt",
    "eta",
    "mass",
    "pt_btagJes",
    "eta_btagJes",
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

TRACK_VARS = [
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
    # Soft muon specific
    "muon_quality",
    "muon_qOverPratio",
    "muon_momentumBalanceSignificance",
    "muon_scatteringNeighbourSignificance",
]

FLOW_VARS = [
    "pt",
    "energy",
    "deta",
    "dphi",
    "isCharged",
]

ELECTRON_VARS = [
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

HADRON_VARS = ["pt", "Lxy", "deta", "dphi", "mass"]

# Module-level generator + torch seeding, faithful to the v1 helper: the FIRST
# write_dummy_file call in a fresh process is deterministic (default_rng(42)).
rng = np.random.default_rng(42)
torch.manual_seed(42)


def write_dummy_norm_dict(nd_path: Path, cd_path: Path, is_gn3: bool = False) -> None:
    """Write dummy normalization and class dictionaries to YAML files.

    Parameters
    ----------
    nd_path : Path
        Output path for the normalization dictionary YAML.
    cd_path : Path
        Output path for the class dictionary YAML.
    is_gn3 : bool, optional
        If ``True``, write GN3-style class labels, by default ``False``.
    """
    sd: dict[str, dict[str, dict[str, float]]] = {}
    sd["jets"] = {n: {"std": 1.0, "mean": 1.0} for n in JET_VARS}
    sd["tracks"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    sd["tracks_dr"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    sd["tracks_ghost"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    sd["electrons"] = {n: {"std": 1.0, "mean": 1.0} for n in ELECTRON_VARS}
    sd["flows"] = {n: {"std": 1.0, "mean": 1.0} for n in FLOW_VARS}
    # Duplicate "flows" to "flow" to support old naming
    sd["flow"] = {n: {"std": 1.0, "mean": 1.0} for n in FLOW_VARS}
    with open(nd_path, "w") as file:
        yaml.dump(sd, file, sort_keys=False)

    cd: dict[str, dict[str, list[float]]] = {}
    if is_gn3:
        cd["jets"] = {"flavour_label": [1.0, 2.0, 2.0, 2.0, 1.0, 1.0]}
    else:
        cd["jets"] = {"HadronConeExclTruthLabelID": [1.0, 2.0, 2.0, 2.0]}
        cd["jets"]["flavour_label"] = cd["jets"]["HadronConeExclTruthLabelID"]
    cd["tracks"] = {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]}
    cd["tracks_ghost"] = {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]}
    with open(cd_path, "w") as file:
        yaml.dump(cd, file, sort_keys=False)


def write_dummy_file(
    fname: str | Path,
    sd_fname: str | Path,
    make_xbb: bool = False,
    inc_taus: bool = False,
    inc_params: bool = False,
    is_gn3: bool = False,
) -> None:
    """Create a synthetic HDF5 file with structured groups for tests.

    Parameters
    ----------
    fname : str | Path
        Output HDF5 file path.
    sd_fname : str | Path
        Path to a normalization/class dictionary YAML (used to extract variable lists).
    make_xbb : bool, optional
        If ``True``, include Xbb-style flavour labels, by default ``False``.
    inc_taus : bool, optional
        If ``True``, include a tau class name in jet labels, by default ``False``.
    inc_params : bool, optional
        If ``True``, add a ``parameters`` dataset with simple values, by default ``False``.
    is_gn3 : bool, optional
        If ``True``, write GN3-style class labels, by default ``False``.
    """
    with open(sd_fname) as f:
        sd = yaml.safe_load(f)

    jet_vars = [
        "pt",
        "eta",
        "mass",
        "pt_btagJes",
        "eta_btagJes",
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

    params = ["mass"]

    track_vars = list(sd["tracks"])
    electron_vars = list(sd["electrons"])

    # settings
    n_jets = 1000
    jet_features = len(jet_vars)
    n_tracks_per_jet = 40
    track_features = len(track_vars)
    n_flows_per_jet = 40
    flow_features = len(FLOW_VARS)
    n_electrons_per_jet = 10
    electron_features = len(electron_vars)
    n_hadrons_per_jet = 5
    hadron_features = len(HADRON_VARS)

    # setup jets
    shapes_jets = {
        "inputs": [n_jets, jet_features + 3],
    }

    # setup tracks
    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, track_features + 4],
        "valid": [n_jets, n_tracks_per_jet],
    }

    # setup flow
    shapes_flow = {
        "inputs": [n_jets, n_flows_per_jet, flow_features],
        "valid": [n_jets, n_flows_per_jet],
    }

    # setup electrons
    shapes_electrons = {
        "inputs": [n_jets, n_electrons_per_jet, electron_features + 2],
        "valid": [n_jets, n_electrons_per_jet],
    }

    # setup hadrons
    shapes_hadrons = {
        "inputs": [n_jets, n_hadrons_per_jet, hadron_features + 2],
        "valid": [n_jets, n_hadrons_per_jet],
    }

    # setup parameters
    shapes_params = {
        "inputs": [n_jets, len(params)],
    }

    # setup jets
    jets_dtype = np.dtype(
        [(n, "f4") for n in jet_vars]
        + [
            ("flavour_label", "i4"),
            ("HadronConeExclTruthLabelID", "i4"),
            ("HadronGhostInitialTruthLabelPdgId", "i4"),
        ]
    )
    jets = rng.random(shapes_jets["inputs"])
    jets = u2s(jets, jets_dtype)
    if make_xbb:
        jets["flavour_label"] = rng.choice([0, 1, 2, 3], size=n_jets)
    elif is_gn3:
        jets["flavour_label"] = rng.choice([0, 1, 2, 3, 4, 5], size=n_jets)
    else:
        jets["flavour_label"] = rng.choice([0, 1, 2], size=n_jets)

    jets["HadronConeExclTruthLabelID"] = rng.choice([0, 4, 5], size=n_jets)
    jets["HadronConeExclTruthLabelLxy"][jets["flavour_label"] == 0] = np.nan
    jets["HadronGhostInitialTruthLabelPdgId"] = rng.choice(
        [0, 15, -511, 521, 10511, 441],
        size=n_jets,
    )

    jets["sample_weight"] = rng.uniform(0.0, 1.0, size=n_jets)

    # setup hadrons
    hadrons_dtype = np.dtype(
        [(n, "f4") for n in HADRON_VARS] + [("barcode", "i4"), ("flavour", "i4")]
    )
    hadrons = rng.random(shapes_hadrons["inputs"])
    valid = rng.choice([True, False], size=shapes_hadrons["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    # Set hadron features to NaN and labels to -1 for invalid entries
    n_float_vars = len(HADRON_VARS)
    mask = ~valid["valid"]
    hadrons[mask, :n_float_vars] = np.nan
    hadrons[mask, n_float_vars:] = -1
    hadrons = u2s(hadrons, hadrons_dtype)
    hadrons = join_structured_arrays([hadrons, valid])
    hadrons["barcode"] = rng.integers(0, 10000, size=(n_jets, n_hadrons_per_jet))
    hadrons["barcode"][~hadrons["valid"]] = -1
    hadrons["flavour"] = rng.choice([-1, 4, 5], size=(n_jets, n_hadrons_per_jet))
    hadrons["flavour"] = np.sort(hadrons["flavour"], axis=-1)[:, ::-1]
    hadrons["flavour"][~hadrons["valid"]] = -1

    # setup tracks
    tracks_dtype = np.dtype(
        [(n, "f4") for n in track_vars]
        + [
            ("ftagTruthOriginLabel", "i4"),
            ("ftagTruthTypeLabel", "i4"),
            ("ftagTruthVertexIndex", "i4"),
            ("ftagTruthParentBarcode", "i4"),
        ]
    )
    tracks = rng.random(shapes_tracks["inputs"])
    valid = rng.choice([True, False], size=shapes_tracks["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    # Set track features to NaN and labels to -1 for invalid entries
    n_float_vars = len(TRACK_VARS)
    mask = ~valid["valid"]
    tracks[mask, :n_float_vars] = np.nan
    tracks[mask, n_float_vars:] = -1
    tracks = u2s(tracks, tracks_dtype)
    tracks = join_structured_arrays([tracks, valid])
    hadron_track_select = rng.choice(np.arange(5), size=(n_jets, n_tracks_per_jet))
    track_barcodes = hadrons["barcode"][np.arange(n_jets)[:, None], hadron_track_select]
    tracks["ftagTruthParentBarcode"] = track_barcodes
    tracks["ftagTruthParentBarcode"][~tracks["valid"]] = -1
    tracks["ftagTruthTypeLabel"] = rng.choice(
        [-2, -3, 5, -5, 6, -6],
        size=(n_jets, n_tracks_per_jet),
    )
    tracks["ftagTruthTypeLabel"][~tracks["valid"]] = 0

    # setup flow
    flow_dtype = np.dtype([(n, "f4") for n in FLOW_VARS])
    flows = rng.random(shapes_flow["inputs"])
    valid = rng.choice([True, False], size=shapes_flow["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    flows[~valid["valid"]] = np.nan
    flows = u2s(flows, flow_dtype)
    flows = join_structured_arrays([flows, valid])

    # setup electrons
    electrons_dtype = np.dtype(
        [(n, "f4") for n in electron_vars]
        + [("ftagTruthOriginLabel", "i4"), ("ftagTruthVertexIndex", "i4")]
    )
    electrons = rng.random(shapes_electrons["inputs"])
    electrons = u2s(electrons, electrons_dtype)
    valid = rng.choice([True, False], size=shapes_electrons["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    electrons = join_structured_arrays([electrons, valid])

    # setup parameters
    params_dtype = np.dtype([(n, "f4") for n in params])
    params_arr = rng.random(shapes_params["inputs"])
    params_arr = u2s(params_arr, params_dtype)
    if inc_params:
        params_arr["mass"] = rng.choice([5, 16, 25, 40, 55], size=(n_jets))

    # write file
    with h5py.File(fname, "w") as f:
        f.attrs["unique_jets"] = len(jets)
        f.attrs["config"] = "{}"
        f.create_dataset("jets", data=jets)
        if make_xbb:
            f["jets"].attrs["flavour_label"] = ["hbb", "hcc", "top", "qcd"]
        elif is_gn3:
            f["jets"].attrs["flavour_label"] = [
                "ghostsplitbjets",
                "ghostsplitcjets",
                "ghostsplitsjets",
                "ghostsplitudjets",
                "ghostsplitgjets",
                "ghostsplittaujets",
            ]
        else:
            f["jets"].attrs["flavour_label"] = ["bjets", "cjets", "ujets"] + (
                ["taus"] if inc_taus else []
            )
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("tracks_dr", data=tracks)
        f.create_dataset("tracks_ghost", data=tracks)
        f.create_dataset("electrons", data=electrons)
        f.create_dataset("flows", data=flows)
        # Duplicate "flows" to "flow" to support old naming
        f.create_dataset("flow", data=flows)
        f.create_dataset("truth_hadrons", data=hadrons)
        if inc_params:
            f.create_dataset("parameters", data=params_arr)
