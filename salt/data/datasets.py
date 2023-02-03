import json
from typing import Mapping

import h5py
import torch
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch import Tensor
from torch.utils.data import Dataset

from salt.utils.inputs import concat_jet_track


class TrainJetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: Mapping,
        labels: Mapping = None,
        num_jets: int = -1,
        exclude: dict = None,
    ):
        """A simple map-style dataset for loading jets from an umami-
        preprocessed training file.

        Parameters
        ----------
        filename : str
            Input h5 filepath.
        inputs : Mapping
            Names of the h5 group to access for each type of input.
        labels : Mapping
            Mapping from task name to label name. Set automatically by the CLI.
        num_jets : int, optional
            Number of jets to use, by default -1 which will use all jets.
        exclude :
            Dict of variables in the datasets not to consider for training (TODO: think
            if the jet variables concatenated to tracks need to be excluded here as well)
        """
        super().__init__()

        # check labels have been configured
        assert labels is not None

        # open file and check
        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        self.check_file(inputs)

        # get datasets
        for g in inputs.values():
            if f"{g}/inputs" not in self.file.keys():
                raise ValueError(f"No group {g}/inputs in file {self.file.filename}")
        self.inputs = {n: self.file[f"{g}/inputs"] for n, g in inputs.items()}
        self.valids = {
            n: self.file[f"{g}/valid"] for n, g in inputs.items() if "valid" in self.file[g].keys()
        }
        self.labels = {n: self.file[l] for n, l in labels.items()}

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

        # set dict of variables to exclude
        self.exclude = exclude

        if self.exclude is None:
            self.exclude = {}

    def __len__(self):
        return int(self.num_jets)

    def __getitem__(self, jet_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        jet_idx
            Either an int corresponding to a single jet to load,
            or a numpy slice corresponding to a batch of jets.

        Returns
        -------
        tuple
            Inputs (dict of tensor), masks (dict of tensor) and labels (dict of tensor).
            Each tensor will contain a single element or a batch of elements.
        """

        inputs = {}
        masks = {}

        # read inputs
        for name in self.inputs:
            inputs[name] = torch.as_tensor(self.inputs[name][jet_idx], dtype=torch.float)

            # make mask for variables to exclude.
            if self.exclude.get(name, []):
                var_mask = [
                    var not in self.exclude[name]
                    for var in list(self.inputs[name].attrs.values())[0]
                ]
                inputs[name] = inputs[name][..., var_mask]

            if name in self.valids:
                masks[name] = ~torch.as_tensor(self.valids[name][jet_idx], dtype=torch.bool)

        # read labels
        labels = {n: torch.as_tensor(l[jet_idx], dtype=torch.long) for n, l in self.labels.items()}

        return inputs, masks, labels

    def get_num_jets(self, num_jets_requested: int):
        num_jets_available = len(self.inputs["jet"])

        # not enough jets
        if num_jets_requested > num_jets_available:
            raise ValueError(
                f"You asked for {num_jets_requested:,} jets, but only"
                f" {num_jets_available:,} jets are available in the file"
                f" {self.filename}."
            )

        # use all jets
        if num_jets_requested < 0:
            return num_jets_available

        # use requested jets
        else:
            return num_jets_requested

    def check_file(self, inputs: Mapping):
        error_str = (
            "Perhaps you used the wrong track group name or have used an old umami training file."
            " Please make sure you use an input file which is produced using a version of umami"
            " which includes !648 and !665, i.e. versions >=0.17."
        )

        if inputs["track"] not in self.file:
            raise KeyError(
                f"The input file '{self.filename}' does not a contain an object named"
                f" '{inputs['track']}'."
                + error_str
            )

        def check_group(g: h5py.Group):
            if not isinstance(g, h5py.Group):
                raise TypeError(
                    f"The input file '{self.filename}' contains a top level object"
                    f" '{g}' which is not a group."
                    + error_str
                )

            if "inputs" not in g.keys():
                raise KeyError(
                    f"The group '{g}' in file '{self.filename}' does not contain an"
                    " inputs dataset. This is unexpected."
                    + error_str
                )

            if inputs["track"] in g.name:
                track_vars = list(g["inputs"].attrs.values())[0]
                if not any(["jet" in v for v in track_vars]):
                    raise ValueError(
                        "Expected to find some variables called jet_* in the 'inputs'"
                        f" dataset for the group '{g}'. You should make sure to specify"
                        " `concat_jet_tracks: True` in your preprocessing config."
                    )

        [check_group(self.file[g]) for g in self.file]


class TestJetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: dict,
        scale_dict: str,
        num_jets: int = -1,
        exclude: dict = None,
    ):
        """A map-style dataset for loading jets from a structured array file
        produced by the tdd or prepared by umami.

        Parameters
        ----------
        filename : str
            Input h5 filepath
        inputs : dict
            Names of the h5 group to access for each type of input
        scale_dict : str
            Path to umami preprocessing scale dict file
        num_jets : int, optional
            Number of jets to use, by default -1
        """
        super().__init__()

        # open file
        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        self.inputs_names = inputs

        # make sure the input file looks okay
        self.check_file(inputs)

        # get scale dict and input variables
        # TODO: get track variables somewhere else?
        self.sd, self.vars = self.get_scale_dict(scale_dict, inputs)

        # get fields
        self.inputs = {n: self.file[d].fields(self.vars[n]) for n, d in inputs.items()}
        self.valids = {
            n: self.file[g].fields("valid")
            for n, g in inputs.items()
            if "valid" in self.file[g].dtype.names
        }

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

        # set dict of variables to exclude
        self.exclude = exclude

        if self.exclude is None:
            self.exclude = {}

    def __len__(self):
        return int(self.num_jets)

    def __getitem__(self, jet_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        jet_idx
            Either an int corresponding to a single jet to load,
            or a numpy slice corresponding to a batch of jets.

        Returns
        -------
        tuple
            Inputs (tensor) and labels (dict of tensor). Each tensor
            will contain a single element or a batch of elements.
        """

        inputs = {}
        masks = {}

        # read inputs
        for i_type, i_name in self.inputs_names.items():
            include = [v for v in self.vars[i_type] if v not in self.exclude.get(i_type, [])]
            inputs_ = self.inputs[i_type][jet_idx][include]
            inputs_ = self.scale_inputs(inputs_, self.sd[i_name], include)
            inputs[i_type] = torch.as_tensor(s2u(inputs_), dtype=torch.float)

            # TODO: added for subjets since no valid vlag
            inputs[i_type] = torch.nan_to_num(inputs[i_type])

            if i_type in self.valids:
                masks[i_type] = ~torch.as_tensor(self.valids[i_type][jet_idx], dtype=torch.bool)

        # concatenate
        for name in inputs:
            if name in self.valids or name == "track":  # TODO: added for subjets since no valid
                inputs[name] = concat_jet_track(inputs["jet"], inputs[name])

                # TODO: added for subjets since no valid vlag
                if "track" not in masks:
                    masks["track"] = torch.zeros(inputs["track"].shape[:-1]).bool()

                # fill nan
                inputs[name][masks[name]] = 0

        return inputs, masks, None

    def get_scale_dict(self, scale_dict_path: str, inputs: dict):
        with open(scale_dict_path) as f:
            scale_dict = json.load(f)

        variables = {
            "jet": scale_dict[inputs["jet"]].keys(),
            "track": scale_dict[inputs["track"]].keys(),
        }

        # hacky atm
        if "flow" in inputs and inputs["flow"] in scale_dict:
            variables["flow"] = scale_dict[inputs["flow"]].keys()

        return scale_dict, variables

    def scale_inputs(self, inputs: Tensor, sd: dict, include: list):
        """Normalise jet inputs."""
        inputs = inputs.astype([(n, "f4") for n in inputs.dtype.names])
        for name, tf in sd.items():
            if name in include:
                inputs[name] = (inputs[name] - tf["shift"]) / tf["scale"]
        return inputs

    def get_num_jets(self, num_jets_requested: int):
        num_jets_available = len(self.inputs["jet"])

        # not enough jets
        if num_jets_requested > num_jets_available:
            raise ValueError(
                f"You asked for {num_jets_requested:,} jets, but only"
                f" {num_jets_available:,} jets are available in the file"
                f" {self.filename}."
            )

        # use all jets
        if num_jets_requested < 0:
            return num_jets_available

        # use requested jets
        else:
            return num_jets_requested

    def check_file(self, inputs: Mapping):
        for inp in inputs.values():
            if inp not in self.file:
                raise KeyError(
                    f"The input file '{self.filename}' does not a contain an object named '{inp}'."
                )

            if not isinstance(self.file[inp], h5py.Dataset):
                raise KeyError(f"The object '{inp}' in file '{self.filename}' is not a dataset.")
