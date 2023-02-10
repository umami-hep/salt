"""Mostly helper functions for generating random inputs."""
import json

import h5py
import numpy as np
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from numpy.random import default_rng
from torch import Tensor

from salt.utils.arrays import join_structured_arrays

DEFAULT_NTRACK = 40

rng = np.random.default_rng(42)


def concat_jet_track(jets: Tensor, tracks: Tensor):
    n_track = tracks.shape[-2]
    jets_repeat = torch.repeat_interleave(jets[:, None, :], n_track, dim=1)
    inputs = torch.cat([jets_repeat, tracks], dim=2)
    return inputs


def inputs_sep_no_pad(n_batch: int, n_track: int, n_feat: int):
    jets = torch.rand(n_batch, 2)
    tracks = torch.rand(n_batch, n_track, n_feat - 2)
    return jets, tracks


def inputs_sep_with_pad(n_batch: int, n_track: int, n_feat: int, p_valid=0.5):
    jets, tracks = inputs_sep_no_pad(n_batch, n_track, n_feat)
    mask = get_random_mask(n_batch, n_track, p_valid)
    return jets, tracks, mask


def get_random_mask(n_batch: int, n_track: int, p_valid: float = 0.5):
    a = rng.choice(a=[True, False], size=(n_batch, n_track), p=[1 - p_valid, p_valid])
    if n_track > 0 and p_valid > 0:  # ensure at least one valid track
        a[:, 0] = False
    return torch.tensor(a)


def inputs_concat(n_batch: int, n_track: int, n_feat: int):
    jets, tracks = inputs_sep_no_pad(n_batch, DEFAULT_NTRACK, n_feat)
    inputs = concat_jet_track(jets, tracks)
    mask = get_random_mask(n_batch, n_track)
    return inputs, mask


def write_dummy_scale_dict(fname, n_jet_features: int, n_track_features: int):
    jet_vars = [f"dummy_jet_var_{i}" for i in range(n_jet_features)]
    track_vars = [f"dummy_track_var_{i}" for i in range(n_track_features)]
    sd: dict = {}
    sd["jets"] = {n: {"scale": 1, "shift": 1} for n in jet_vars}
    sd["tracks"] = {n: {"scale": 1, "shift": 1} for n in track_vars}
    with open(fname, "w") as f:
        f.write(json.dumps(sd))


def get_dummy_inputs(
    n_jets=1000,
    n_jet_features=2,
    n_track_features=21,
    n_tracks_per_jet=40,
    jets_name="jets",
    tracks_name="tracks",
):
    shapes_jets = {
        "inputs": [n_jets, n_jet_features],
        "labels": [n_jets],
    }

    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, n_jet_features + n_track_features],
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


def write_dummy_train_file(fname, sd_path, jets_name="jets", tracks_name="tracks"):
    with open(sd_path) as f:
        sd = json.load(f)
    kwargs = {"n_jet_features": len(sd[jets_name]), "n_track_features": len(sd[tracks_name])}
    jets, tracks = get_dummy_inputs(jets_name=jets_name, tracks_name=tracks_name, **kwargs)
    kwargs["n_track_features"] = 4
    flow = get_dummy_inputs(jets_name=jets_name, tracks_name="flow", **kwargs)[1]
    with h5py.File(fname, "w") as f:
        g_jets = f.create_group(jets_name)
        g_tracks = f.create_group(tracks_name)
        g_flow = f.create_group("flow")

        for key, arr in jets.items():
            g_jets.create_dataset(key, data=arr)
            g_jets[key].attrs['"labels"'] = ["bjets", "cjets", "ujets"]

        for key, arr in tracks.items():
            g_tracks.create_dataset(key, data=arr)
            if key == "inputs":
                var = list(sd[jets_name].keys()) + list(sd[tracks_name].keys())
                g_tracks[key].attrs[f"{tracks_name}_variables"] = var

        for key, arr in flow.items():
            g_flow.create_dataset(key, data=arr)
            if key == "inputs":
                var = list(sd[jets_name].keys()) + list(sd[tracks_name].keys())
                g_flow[key].attrs["flow_variables"] = var


def write_dummy_test_file(fname, sd_fname):
    with open(sd_fname) as f:
        sd = json.load(f)

    # TODO: read these from the predictionwriter config
    jet_vars = [
        "pt",
        "eta",
        "HadronConeExclTruthLabelID",
        "n_tracks",
        "n_truth_promptLepton",
        "dummy_jet_var_0",
        "dummy_jet_var_1",
    ]

    # settings
    n_jets = 1000
    jet_features = len(jet_vars)
    n_tracks_per_jet = 40
    track_features = 21

    # setup jets
    shapes_jets = {
        "inputs": [n_jets, jet_features],
    }

    # setup tracks
    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, track_features],
        "valid": [n_jets, n_tracks_per_jet],
    }

    # setup jets
    jets_dtype = np.dtype([(n, "f4") for n in jet_vars])
    jets = rng.random(shapes_jets["inputs"])
    jets = u2s(jets, jets_dtype)

    # setup tracks
    tracks_dtype = np.dtype([(n, "f4") for n in sd["tracks"]])
    tracks = rng.random(shapes_tracks["inputs"])
    tracks = u2s(tracks, tracks_dtype)
    valid = rng.random(shapes_tracks["valid"])
    valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
    tracks = join_structured_arrays([tracks, valid])

    with h5py.File(fname, "w") as f:
        f.create_dataset("jets", data=jets)
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("flow", data=tracks)
