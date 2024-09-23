"""Mostly helper functions for generating random inputs."""

from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from numpy.random import default_rng

from salt.utils.array_utils import join_structured_arrays
from salt.utils.tensor_utils import attach_context

DEFAULT_NTRACK = 40

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
rng = np.random.default_rng(42)
torch.manual_seed(42)


def inputs_sep_no_pad(n_batch: int, n_track: int, n_jet_feat: int, n_track_feat: int):
    jets = torch.rand(n_batch, n_jet_feat)
    tracks = torch.rand(n_batch, n_track, n_track_feat)
    return jets, tracks


def inputs_sep_no_pad_multi_sequece(
    n_batch: int, n_seq_list: list[int], n_jet_feat: int, n_seq_feat_list: list[int]
):
    jets = torch.rand(n_batch, n_jet_feat)
    sequences = []
    for n_seq, n_seq_feat in zip(n_seq_list, n_seq_feat_list, strict=False):
        sequences.append(torch.rand(n_batch, n_seq, n_seq_feat))
    return jets, sequences


def inputs_sep_with_pad(
    n_batch: int, n_track: int, n_jet_feat: int, n_track_feat: int, p_valid=0.5
):
    jets, tracks = inputs_sep_no_pad(n_batch, n_track, n_jet_feat, n_track_feat)
    mask = get_random_mask(n_batch, n_track, p_valid)
    return jets, tracks, mask


def inputs_sep_with_pad_multi_sequece(
    n_batch: int, n_seq_list: list[int], n_jet_feat: int, n_seq_feat_list: list[int], p_valid=0.5
):
    jets, sequences = inputs_sep_no_pad_multi_sequece(
        n_batch, n_seq_list, n_jet_feat, n_seq_feat_list
    )
    masks = [get_random_mask(n_batch, n_seq, p_valid) for n_seq in n_seq_list]
    return jets, sequences, masks


def get_random_mask(n_batch: int, n_track: int, p_valid: float = 0.5):
    a = rng.choice(a=[True, False], size=(n_batch, n_track), p=[1 - p_valid, p_valid])
    a = np.sort(a, axis=-1)[:, ::-1]
    if n_track > 0 and p_valid > 0:  # ensure at least one valid track
        a[:, 0] = False
    return torch.tensor(a.copy())


def inputs_concat(n_batch: int, n_track: int, n_jet_feat: int, n_track_feat: int):
    jets, tracks = inputs_sep_no_pad(n_batch, DEFAULT_NTRACK, n_jet_feat, n_track_feat)
    inputs = attach_context(tracks, jets)
    mask = get_random_mask(n_batch, n_track)
    return inputs, mask


def write_dummy_norm_dict(nd_path: Path, cd_path: Path):
    sd: dict = {}
    sd["jets"] = {n: {"std": 1.0, "mean": 1.0} for n in JET_VARS}
    sd["tracks"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    sd["tracks_dr"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    sd["electrons"] = {n: {"std": 1.0, "mean": 1.0} for n in ELECTRON_VARS}
    sd["flow"] = {n: {"std": 1.0, "mean": 1.0} for n in TRACK_VARS}
    with open(nd_path, "w") as file:
        yaml.dump(sd, file, sort_keys=False)

    cd: dict = {}
    cd["jets"] = {"HadronConeExclTruthLabelID": [1.0, 2.0, 2.0, 2.0]}
    cd["jets"]["flavour_label"] = cd["jets"]["HadronConeExclTruthLabelID"]
    cd["tracks"] = {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]}
    with open(cd_path, "w") as file:
        yaml.dump(cd, file, sort_keys=False)


def get_dummy_inputs(n_jets=1000, n_jet_features=2, n_track_features=21, n_tracks_per_jet=40):
    shapes_jets = {
        "inputs": [n_jets, n_jet_features],
        "labels": [n_jets],
        "add_labels": [n_jets],
    }

    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, n_jet_features + n_track_features],
        "labels/ftagTruthOriginLabel": [n_jets, n_tracks_per_jet],
        "labels/ftagTruthParentBarcode": [n_jets, n_tracks_per_jet],
        "labels/ftagTruthVertexIndex": [n_jets, n_tracks_per_jet],
        "labels/truthOriginLabel": [n_jets, n_tracks_per_jet],
        "labels/truthVertexIndex": [n_jets, n_tracks_per_jet],
        "valid": [n_jets, n_tracks_per_jet],
    }

    rng = default_rng(seed=42)
    jets = {}
    for key, shape in shapes_jets.items():
        jets[key] = rng.random(shape)

    tracks = {}
    for key, shape in shapes_tracks.items():
        arr = rng.random(shape)
        if key == "valid":
            arr = arr.astype(bool)
            arr[:, 10:] = False
        if key == "labels":
            arr = rng.integers(0, 8, shape)
        tracks[key] = arr

    return jets, tracks


def write_dummy_file(fname, sd_fname, make_xbb=False, inc_taus=False, inc_params=False):
    """TODO: merge with atlas-ftag-tools test file generation."""
    with open(sd_fname) as f:
        sd = yaml.safe_load(f)

    jet_vars = [
        "pt",
        "eta",
        "mass",
        "pt_btagJes",
        "eta_btagJes",
        "HadronConeExclTruthLabelPt",
        "HadronConeExclTruthLabelLxy",
        "n_tracks",
        "n_truth_promptLepton",
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
    n_electrons_per_jet = 10
    electron_features = len(electron_vars)
    n_hadrons_per_jet = 5
    hadron_features = len(HADRON_VARS)
    # setup jets
    shapes_jets = {
        "inputs": [n_jets, jet_features + 2],
    }

    # setup tracks
    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, track_features + 3],
        "valid": [n_jets, n_tracks_per_jet],
    }

    shapes_electrons = {
        "inputs": [n_jets, n_electrons_per_jet, electron_features + 2],
        "valid": [n_jets, n_electrons_per_jet],
    }

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
        + [("flavour_label", "i4"), ("HadronConeExclTruthLabelID", "i4")]
    )
    jets = rng.random(shapes_jets["inputs"])
    jets = u2s(jets, jets_dtype)
    if make_xbb:
        jets["flavour_label"] = rng.choice([0, 1, 2, 3], size=n_jets)
    else:
        jets["flavour_label"] = rng.choice([0, 1, 2], size=n_jets)

    jets["HadronConeExclTruthLabelID"] = rng.choice([0, 4, 5], size=n_jets)
    jets["HadronConeExclTruthLabelLxy"][jets["flavour_label"] == 0] = np.nan

    # setup hadrons
    hadrons_dtype = np.dtype(
        [(n, "f4") for n in HADRON_VARS] + [("barcode", "i4"), ("flavour", "i4")]
    )
    hadrons = rng.random(shapes_hadrons["inputs"])
    valid = rng.choice([True, False], size=shapes_hadrons["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    hadrons[~valid["valid"]] = np.nan
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
            ("ftagTruthVertexIndex", "i4"),
            ("ftagTruthParentBarcode", "i4"),
        ]
    )
    tracks = rng.random(shapes_tracks["inputs"])
    valid = rng.choice([True, False], size=shapes_tracks["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    tracks[~valid["valid"]] = np.nan
    tracks = u2s(tracks, tracks_dtype)
    tracks = join_structured_arrays([tracks, valid])
    hadron_track_select = rng.choice(np.arange(5), size=(n_jets, n_tracks_per_jet))
    track_barcodes = hadrons["barcode"][np.arange(n_jets)[:, None], hadron_track_select]
    tracks["ftagTruthParentBarcode"] = track_barcodes
    tracks["ftagTruthParentBarcode"][~tracks["valid"]] = -1

    # setup electrons
    electrons_dtype = np.dtype(
        [(n, "f4") for n in electron_vars]
        + [
            ("ftagTruthOriginLabel", "i4"),
            ("ftagTruthVertexIndex", "i4"),
        ]
    )
    electrons = rng.random(shapes_electrons["inputs"])
    electrons = u2s(electrons, electrons_dtype)
    valid = rng.choice([True, False], size=shapes_electrons["valid"])
    valid = np.sort(valid, axis=-1)[:, ::-1].view(dtype=np.dtype([("valid", bool)]))
    electrons = join_structured_arrays([electrons, valid])

    # setup parameters
    params_dtype = np.dtype([(n, "f4") for n in params])
    params = rng.random(shapes_params["inputs"])
    params = u2s(params, params_dtype)
    if inc_params:
        params["mass"] = rng.choice([5, 16, 25, 40, 55], size=(n_jets))

    with h5py.File(fname, "w") as f:
        f.attrs["unique_jets"] = len(jets)
        f.attrs["config"] = "{}"
        f.create_dataset("jets", data=jets)
        if make_xbb:
            f["jets"].attrs["flavour_label"] = ["hbb", "hcc", "top", "qcd"]
        else:
            f["jets"].attrs["flavour_label"] = ["bjets", "cjets", "ujets"] + (
                ["taus"] if inc_taus else []
            )
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("tracks_dr", data=tracks)
        f.create_dataset("electrons", data=electrons)
        f.create_dataset("flow", data=tracks)
        f.create_dataset("truth_hadrons", data=hadrons)
        if inc_params:
            f.create_dataset("PARAMETERS", data=params)


def as_half(typestr) -> np.dtype:
    """Cast float type to half precision."""
    t = np.dtype(typestr)
    if t.kind != "f" or t.itemsize != 2:
        return t
    return np.dtype("f2")
