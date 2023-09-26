from pathlib import Path

import torch
from torch import nn

from salt.data.scaler import NormDictScaler
from salt.models import Dense


class InitNet(nn.Module):
    def __init__(
        self,
        name: str,
        dense_config: dict,
        norm_dict: Path | None = None,
        variables: dict | None = None,
        input_names: dict | None = None,
        concat_jet_tracks: bool = False,
    ):
        """Initialiser network which can optionally handle input normalisation.

        Parameters
        ----------
        name : str
            Name of the input, must match the input types in the data config.
        dense_config : dict
            Keyword arguments for the dense networkfor performing the the initial embedding.
        norm_dict : Path | None, optional
            Path to yaml file containing normalisation parameters, by default None
        variables : dict | None, optional
            Input variables used in the forward pass, by default None
        input_names : dict | None, optional
            Names of the h5 group to access for each type of input, by default None
        concat_jet_tracks : bool, optional
            Concatenate jet inputs with track-type inputs, by default False
        """
        super().__init__()

        self.name = name
        self.net = Dense(**dense_config)
        self.concat_jet_tracks = concat_jet_tracks
        if bool(norm_dict) != bool(variables) != bool(input_names):
            raise ValueError("Must provide either all or none of norm_dict, variables, input_names")

        if norm_dict is not None and self.name != "edge":
            assert input_names is not None
            self.scaler = NormDictScaler(norm_dict, input_names, variables, concat_jet_tracks)
            means, stds = self.scaler.norm_params[self.name]
            self.register_buffer("means", torch.from_numpy(means))
            self.register_buffer("stds", torch.from_numpy(stds))
        else:
            self.register_buffer("means", None)
            self.register_buffer("stds", None)

    def forward(self, inputs: dict):
        x = inputs[self.name]
        if self.means is not None:
            x = (x - self.means) / self.stds

        return self.net(x)
