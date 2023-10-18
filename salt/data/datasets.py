from collections.abc import Mapping

import h5py
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch.utils.data import Dataset

from salt.data.edge_features import get_dtype_edge, get_inputs_edge
from salt.data.scaler import NormDictScaler
from salt.utils.inputs import as_half, concat_jet_track


class JetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        input_names: dict,
        norm_dict: str,
        variables: dict,
        stage: str,
        num_jets: int = -1,
        labels: Mapping | None = None,
        concat_jet_tracks: bool = True,
        num_inputs: dict | None = None,
        nan_to_num: bool = False,
        norm_in_model: bool = False,
        parameters: dict | None = None,
    ):
        """A map-style dataset for loading jets from a structured array file.

        Parameters
        ----------
        filename : str
            Input h5 filepath containing structured arrays
        input_names : dict
            Names of the h5 group to access for each type of input
        norm_dict : str
            Path to file containing normalisation parameters
        variables : dict
            Input variables used in the forward pass
        stage: str
            Either 'fit' or 'test'
        num_jets : int, optional
            Number of jets to use, by default -1
        labels : Mapping
            Mapping from task name to label name, set automatically by the CLI
        concat_jet_tracks : bool, optional
            Concatenate jet inputs with track-type inputs, by default True
        num_inputs : dict, optional
            Truncate the number of track-like inputs to this number, by default None
        nan_to_num : bool, optional
            Convert nans to zeros, by default False
        norm_in_model : bool, optional
            Normalise inputs in the model rather than in this Dataloader, by default False
            TODO: remove this and default to True when backwards compatability no longer needed
        parameters: dict
            Variables used to parameterise the network, by default None.
        """
        super().__init__()

        # check labels have been configured
        self.labels = labels if labels is not None else {}

        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        self.input_names = input_names
        self.concat_jet_tracks = concat_jet_tracks
        self.num_inputs = num_inputs
        self.nan_to_num = nan_to_num
        self.norm_in_model = norm_in_model
        self.parameters = parameters
        self.stage = stage

        # check that num_inputs contains valid keys
        if self.num_inputs is not None and not set(self.num_inputs).issubset(self.input_names):
            raise ValueError(
                f"num_inputs keys {self.num_inputs.keys()} must be a subset of input_names keys"
                f" {self.input_names.keys()}"
            )

        # make sure the input file looks okay
        self.check_file(self.input_names)

        # create scaler
        self.scaler = NormDictScaler(norm_dict, input_names, variables)
        self.input_variables = self.scaler.variables
        assert self.input_variables is not None

        # check parameters listed in variables appear in the same order in the parameters block
        if "parameters" in self.input_variables:
            assert self.parameters is not None
            assert self.input_variables["parameters"] is not None
            assert len(self.input_variables["parameters"]) == len(self.parameters)
            for idx, param_key in enumerate(self.parameters.keys()):
                assert self.input_variables["parameters"][idx] == param_key

        # setup datasets and accessor arrays
        self.dss = {}
        self.arrays = {}
        for input_type, input_name in self.input_names.items():
            self.dss[input_type] = self.file[input_name]
            variables = self.labels[input_type] if input_type in self.labels else []
            variables += self.input_variables[input_type]
            if input_type == "edge":
                dtype = get_dtype_edge(self.file[input_name], variables)
            else:
                dtype = get_dtype(self.file[input_name], variables)
            self.arrays[input_type] = np.array(0, dtype=dtype)

        # set number of jets
        self.num_jets = self.get_num_jets(num_jets)

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

            # truncate track-like inputs
            if self.num_inputs is not None and input_type in self.num_inputs:
                assert int(self.num_inputs[input_type]) <= batch.shape[1]
                batch = batch[:, : int(self.num_inputs[input_type])]

            # process inputs for this input type
            if input_type == "edge":
                inputs[input_type] = torch.from_numpy(
                    get_inputs_edge(batch, self.input_variables[input_type])
                )
            elif input_type == "parameters":
                flat_array = s2u(batch[self.input_variables[input_type]], dtype=np.float32)

                for ind, param in enumerate(self.parameters):
                    if self.stage == "fit":
                        # assign random values to jets with parameters not set to those in the
                        # train list, values are chosen at random from those in the train list
                        # according to probabilities if given, else with equal probability
                        try:
                            prob = self.parameters[param]["prob"]
                        except KeyError:
                            prob = None
                        mask = ~np.isin(flat_array[:, ind], self.parameters[param]["train"])
                        random = np.random.choice(
                            self.parameters[param]["train"], size=np.sum(mask), p=prob
                        )
                        flat_array[mask, ind] = random

                    if self.stage == "test":
                        # assign parameter values for all jets to those passed in the 'test' option
                        test_arr = np.full(np.shape(flat_array)[0], self.parameters[param]["test"])
                        flat_array[:, ind] = test_arr

                inputs[input_type] = torch.from_numpy(flat_array)

            else:
                flat_array = s2u(batch[self.input_variables[input_type]], dtype=np.float32)
                if self.nan_to_num:
                    flat_array = np.nan_to_num(flat_array)
                if not self.norm_in_model:
                    flat_array = self.scaler(flat_array, input_type)
                inputs[input_type] = torch.from_numpy(flat_array)

            # process labels for this input type
            if input_type in self.labels:
                labels[input_type] = {}
                for label in self.labels[input_type]:
                    dtype = torch.long if np.issubdtype(batch[label].dtype, np.integer) else None
                    labels[input_type][label] = torch.as_tensor(batch[label].copy(), dtype=dtype)

                # hack to handle the old umami train file format
                if input_type == "jet" and "/" in self.labels:
                    if "jet" not in labels:
                        labels["jet"] = {}
                    for label in self.labels["/"]:
                        labels[input_type][label] = torch.as_tensor(
                            self.file["labels"][jet_idx], dtype=torch.long
                        )

            # get the padding mask
            if "valid" in batch.dtype.names and input_type != "edge" and input_type != "parameters":
                masks[input_type] = ~torch.from_numpy(batch["valid"])

        # concatenate jet (and jet parameters) and track inputs, and fill padded entries with zeros
        for name in inputs:
            if self.concat_jet_tracks and name not in ["jet", "global", "edge", "parameters"]:
                inputs[name] = concat_jet_track(inputs["jet"], inputs[name])
                if "parameters" in self.input_names:
                    inputs[name] = concat_jet_track(inputs["parameters"], inputs[name])
                inputs[name][masks[name]] = 0

        return inputs, masks, labels

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

    variables_flat = []
    for item in variables:
        if isinstance(item, list):
            for subitem in item:
                variables_flat.append(subitem)
        else:
            variables_flat.append(item)

    if missing := set(variables_flat) - set(ds.dtype.names):
        raise ValueError(
            f"Variables {missing} were not found in dataset {ds.name} in file {ds.file.filename}"
        )

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in variables])
