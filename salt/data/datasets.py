from collections.abc import Mapping

import h5py
import numpy as np
import torch
import yaml
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch.utils.data import Dataset

from salt.data.edge_features import get_dtype_edge, get_inputs_edge
from salt.utils.inputs import as_half, concat_jet_track


class JetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: dict,
        norm_dict: str,
        variables: dict,
        num_jets: int = -1,
        concat_jet_tracks: bool = True,
        labels: Mapping | None = None,
        nan_to_num: bool = False,
        num_inputs: dict | None = None,
    ):
        """A map-style dataset for loading jets from a structured array file.

        Parameters
        ----------
        filename : str
            Input h5 filepath containing structured arrays
        inputs : dict
            Names of the h5 group to access for each type of input
        norm_dict : str
            Path to file containing normalisation parameters
        variables : dict
            Variables and labels to use for the training
        num_jets : int, optional
            Number of jets to use, by default -1
        concat_jet_tracks : bool, optional
            Concatenate jet inputs with track-type inputs, by default True
        labels : Mapping
            Mapping from task name to label name. Set automatically by the CLI.
        nan_to_num : bool, optional
            Convert nans to zeros, by default False
        num_inputs : dict, optional
            Truncate the number of inputs to this number, by default None
        """
        super().__init__()

        # check labels have been configured
        self.labels = labels if labels is not None else {}

        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        self.input_names = inputs
        self.input_types = dict(map(reversed, self.input_names.items()))  # type:ignore
        self.concat_jet_tracks = concat_jet_tracks
        self.nan_to_num = nan_to_num
        self.num_inputs = num_inputs

        # check that num_inputs contains valid keys
        if self.num_inputs is not None and not set(self.num_inputs).issubset(self.input_names):
            raise ValueError(
                f"num_inputs keys {self.num_inputs.keys()} must be a subset of input_names keys"
                f" {self.input_names.keys()}"
            )

        with open(norm_dict) as f:
            self.norm_dict = yaml.safe_load(f)

        self.variables = variables
        if self.variables is None:
            self.variables = {
                self.input_types[k]: list(v.keys())
                for k, v in self.norm_dict.items()
                if k in self.input_types
            }

        # make sure the input file looks okay
        self.check_file(self.input_names)

        # setup fields
        self.dss = {}
        self.arrays = {}
        for input_type, input_name in self.input_names.items():
            self.dss[input_type] = self.file[input_name]
            variables = [lab for (g, lab) in self.labels.values() if g == input_type]  # type:ignore
            variables += self.variables[input_type]
            if input_type == "edge":
                dtype = get_dtype_edge(self.file[input_name], variables)
            else:
                dtype = get_dtype(self.file[input_name], variables)
            self.arrays[input_type] = np.array(0, dtype=dtype)

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

        # get norm params as arrays
        self.norm = {}
        for input_type, input_name in self.input_names.items():
            nd = self.norm_dict[input_name]
            var = self.variables[input_type]
            if input_type == "edge":
                means = np.array([0.0 for v in var], dtype=np.float32)
                stds = np.array([1.0 for v in var], dtype=np.float32)
            else:
                mean_key = "mean" if "mean" in nd[var[0]] else "shift"
                std_key = "std" if "std" in nd[var[0]] else "scale"
                means = np.array([nd[v][mean_key] for v in var], dtype=np.float32)
                stds = np.array([nd[v][std_key] for v in var], dtype=np.float32)
            self.norm[input_type] = {"mean": means, "std": stds}

        # for each regression task the target value must also be scaled and shifted
        for name, (_group, input_name) in self.labels.items():
            if "regression" in name:
                if input_name in self.norm_dict["jets"]:
                    nd = self.norm_dict["jets"]
                    mean_key = "mean" if "mean" in nd[input_name] else "shift"
                    std_key = "std" if "std" in nd[input_name] else "scale"
                    means = np.array(nd[input_name][mean_key], dtype=np.float32)
                    stds = np.array(nd[input_name][std_key], dtype=np.float32)
                    self.norm[input_name] = {"mean": means, "std": stds}
                else:
                    means = np.array(1, dtype=np.float32)
                    stds = np.array(1, dtype=np.float32)
                    self.norm[input_name] = {"mean": means, "std": stds}
                    print("No scaling for the regression target was found in the norm dict")

    def __len__(self):
        return int(self.num_jets)

    def __getitem__(self, jet_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        jet_idx
            A numpy slice corresponding to a batch of jets.

        Returns
        -------
        tuple
            Dict of tensor for each of the inputs, masks, and labels.
            Each tensor will contain a batch of samples.
        """
        inputs = {}
        labels = {}
        masks = {}

        # loop over input types
        for input_type in self.input_names:
            # load data (inputs + labels) for this input type
            batch = self.arrays[input_type]
            shape = (jet_idx.stop - jet_idx.start,) + self.dss[input_type].shape[1:]
            batch.resize(shape, refcheck=False)
            self.dss[input_type].read_direct(batch, jet_idx)

            # truncate inputs
            if self.num_inputs is not None and input_type in self.num_inputs:
                batch = batch[:, : int(self.num_inputs[input_type])]

            # process inputs for this input type
            if input_type == "edge":
                inputs[input_type] = torch.from_numpy(
                    get_inputs_edge(batch, self.variables[input_type])
                )
            else:
                scaled_inputs = self.scale_input(batch, input_type)
                inputs[input_type] = torch.from_numpy(scaled_inputs)

            # process labels for this input type
            for name, (group, label) in self.labels.items():
                if input_type == group:
                    dtype = torch.long if np.issubdtype(batch[label].dtype, np.integer) else None
                    labels[name] = torch.as_tensor(batch[label].copy(), dtype=dtype)

                # hack to handle the old umami train file format
                if input_type == "jet" and group == "/":
                    labels[name] = torch.as_tensor(self.file["labels"][jet_idx], dtype=torch.long)

                # scale target value of regression
                if (
                    input_type == "jet"
                    and "regression" in name
                    and "jet_regression_denominator" not in self.labels.keys()
                ):
                    shift, scale = self.norm[label]["mean"], self.norm[label]["std"]
                    labels[name] = (labels[name] - shift) / scale

            # get the padding mask
            if "valid" in batch.dtype.names and input_type != "edge":
                masks[input_type] = ~torch.from_numpy(batch["valid"])

        # concatenate jet and track inputs, and fill padded entries with zeros
        for name in inputs:
            if self.concat_jet_tracks and name not in ["jet", "global", "edge"]:
                inputs[name] = concat_jet_track(inputs["jet"], inputs[name])
                inputs[name][masks[name]] = 0

        return inputs, masks, labels

    def scale_input(self, batch: dict, input_type: str):
        """Normalise jet inputs."""
        inputs = s2u(batch[self.variables[input_type]], dtype=np.float32)
        if self.nan_to_num:
            inputs = np.nan_to_num(inputs)
        return (inputs - self.norm[input_type]["mean"]) / self.norm[input_type]["std"]

    def get_num_jets(self, num_jets_requested: int):
        num_jets_available = len(self.dss["jet"])

        # not enough jets
        if num_jets_requested > num_jets_available:
            raise ValueError(
                f"Requested {num_jets_requested:,} jets, but only {num_jets_available:,} are"
                f" available in the file {self.filename}."
            )

        # use all jets
        if num_jets_requested < 0:
            return num_jets_available

        # use requested jets
        return num_jets_requested

    def check_file(self, inputs: Mapping):
        keys = set(inputs.values())
        available = set(self.file.keys())
        if missing := keys - available:
            raise KeyError(
                f"The input file '{self.filename}' does not contain the following keys: {missing}."
                f" Available keys: {available}"
            )
        for inp in inputs.values():
            if not isinstance(self.file[inp], h5py.Dataset):
                raise KeyError(f"The object '{inp}' in file '{self.filename}' is not a dataset.")


def get_dtype(ds, variables=None) -> np.dtype:
    """Return a dtype based on an existing dataset and requested variables."""
    if variables is None:
        variables = ds.dtype.names
    if "valid" in ds.dtype.names and "valid" not in variables:
        variables.append("valid")

    if missing := set(variables) - set(ds.dtype.names):
        raise ValueError(
            f"Variables {missing} were not found in dataset {ds.name} in file {ds.file.filename}"
        )

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in variables])
