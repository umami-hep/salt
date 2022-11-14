"""Mostly helper functions for generating random inputs."""

import torch

DEFAULT_NTRACK = 40


def concat_jet_track(jets, tracks):
    n_track = tracks.shape[-2]
    jets_repeat = torch.repeat_interleave(jets[:, None, :], n_track, dim=1)
    inputs = torch.cat([jets_repeat, tracks], dim=2)
    return inputs


def inputs_sep_no_pad(n_batch, n_track, n_feat):
    jets = torch.rand(n_batch, 2)
    tracks = torch.rand(n_batch, n_track, n_feat - 2)
    return jets, tracks


def inputs_sep_with_pad(n_batch, n_track, n_feat):
    jets, tracks = inputs_sep_no_pad(n_batch, n_track, n_feat)
    mask = get_mask(n_batch, n_track)
    return jets, tracks, mask


def get_mask(n_batch, n_track):
    mask = torch.zeros((n_batch, DEFAULT_NTRACK)).bool()
    mask[n_track:] = True


def inputs_concat(n_batch, n_track, n_feat):
    jets, tracks = inputs_sep_no_pad(n_batch, DEFAULT_NTRACK, n_feat)
    inputs = concat_jet_track(jets, tracks)
    mask = get_mask(n_batch, n_track)
    return inputs, mask


def generate_scale_dict(jet_features: int, track_features: int):
    jet_vars = ["pt", "eta"]
    track_vars = [f"test_{i}" for i in range(track_features)]
    sd: dict = {}
    sd["jets"] = {n: {"scale": 1, "shift": 1} for n in jet_vars}
    sd["tracks_loose"] = {n: {"scale": 1, "shift": 1} for n in track_vars}
    return sd
