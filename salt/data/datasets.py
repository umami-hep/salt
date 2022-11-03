import json

import h5py
import torch
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch.utils.data import Dataset


class TrainJetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: dict,
        tasks: dict,
        num_jets: int = -1,
    ):
        """A simple map-style dataset for loading jets from an umami-
        preprocessed training file.

        Parameters
        ----------
        filename : str
            Input h5 filepath
        inputs : dict
            Names of the h5 group to access for each type of input
        tasks : dict
            Dict containing information about each aux task, used to
            load the labels for each task
        num_jets : int, optional
            Number of jets to use, by default -1
        """
        super().__init__()

        # open file
        self.filename = filename
        self.file = h5py.File(self.filename, "r")

        # make sure the input file looks okay
        self.check_file(inputs)

        # get datasets
        self.inputs = {n: self.file[f"{g}/inputs"] for n, g in inputs.items()}
        self.valid = {"track": self.file[f"{inputs['track']}/valid"]}
        self.labels = {n: self.file[f"{g}/labels"] for n, g in inputs.items()}

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

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

        # read inputs (assume we have already concatenated jet features in umami)
        inputs = torch.FloatTensor(self.inputs["track"][jet_idx])
        mask = ~torch.tensor(self.valid["track"][jet_idx])

        # read labels
        jet_class_label = torch.tensor(self.labels["jet"][jet_idx]).long()
        track_labels = torch.LongTensor(self.labels["track"][jet_idx])

        labels = {
            "jet_classification": jet_class_label,
            "track_classification": track_labels[..., 0],
        }

        return inputs, mask, labels

    def get_num_jets(self, num_jets_requested):
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

    def check_file(self, inputs):
        error_str = (
            "Perhaps you have specified an old-style umami training file. Please make"
            " sure you use an input file which is produced using a version of umami"
            " which includes !648, i.e. versions >=0.15."
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
                )

            if inputs["track"] in g.name:
                track_vars = list(g["inputs"].attrs.values())[0]
                if not any(["jet" in v for v in track_vars]):
                    raise ValueError(
                        "Expected to find some variables called jet_* in the 'inputs'"
                        f" dataset for the group '{g}'. You should make sure to specify"
                        " `concat_jet_tracks: True` in your umami preprocessing"
                        " config."
                    )

        [check_group(self.file[g]) for g in self.file]


class StructuredJetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: dict,
        scale_dict: str,
        num_jets: int = -1,
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

        # make sure the input file looks okay
        self.check_file(inputs)

        # get scale dict and input variables
        self.sd, self.vars = self.get_scale_dict(scale_dict, inputs)

        # get fields
        self.inputs = {n: self.file[d].fields(self.vars[n]) for n, d in inputs.items()}
        self.valid = self.file[inputs["track"]].fields("valid")

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

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

        # read inputs
        jets = torch.FloatTensor(s2u(self.inputs["jet"][jet_idx]))
        tracks = torch.FloatTensor(s2u(self.inputs["track"][jet_idx]))
        mask = ~torch.tensor(self.valid[jet_idx])

        # scale
        jets, tracks = self.scale_inputs(jets, tracks, mask)

        # concatenate
        jets_repeat = torch.repeat_interleave(jets[:, None, :], tracks.shape[1], dim=1)
        tracks = torch.cat([jets_repeat, tracks], dim=2)
        tracks[mask] = 0

        return tracks, mask, None

    def get_scale_dict(self, scale_dict_path: str, inputs: dict):
        # open scale dict
        with open(scale_dict_path) as f:
            scale_dict = json.load(f)

        # reformat
        sd: dict = {i: {} for i in inputs.keys()}
        for tf in scale_dict[inputs["jet"]]:
            sd["jet"][tf["name"]] = {
                "scale": tf["scale"],
                "shift": tf["shift"],
            }

        for name, tf in scale_dict[inputs["track"]].items():
            sd["track"][name] = tf

        variables = {
            "jet": sd["jet"].keys(),
            "track": sd["track"].keys(),
        }

        return sd, variables

    def scale_inputs(self, jets, tracks, mask):
        # normalise jet inputs
        for i, tf in enumerate(self.sd["jet"].values()):
            jets[:, i] = (jets[:, i] - tf["shift"]) / tf["scale"]

        # normalise track inputs
        for i, tf in enumerate(self.sd["track"].values()):
            tracks[..., i] = torch.where(
                ~mask,
                (tracks[..., i] - tf["shift"]) / tf["scale"],
                tracks[..., i],
            )

        return jets, tracks

    def get_num_jets(self, num_jets_requested):
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

    def check_file(self, inputs):
        for inp in inputs.values():
            if inp not in self.file:
                raise KeyError(
                    f"The input file '{self.filename}' does not a contain an object"
                    f" named '{inp}'."
                )

            if not isinstance(self.file[inp], h5py.Dataset):
                raise KeyError(
                    f"The object '{inp}' in file '{self.filename}' is not a dataset."
                )
