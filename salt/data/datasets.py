from copy import deepcopy

import h5py
import numpy as np
import torch
from ftag import Cuts
from ftag.track_selector import TrackSelector
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch.utils.data import Dataset

from salt.data.edge_features import get_dtype_edge, get_inputs_edge
from salt.stypes import Vars
from salt.utils.array_utils import maybe_copy
from salt.utils.configs import MaskformerConfig
from salt.utils.inputs import as_half
from salt.utils.mask_utils import build_target_masks


class SaltDataset(Dataset):
    def __init__(
        self,
        filename: str,
        norm_dict: str,
        variables: Vars,
        stage: str,
        num: int = -1,
        labels: Vars = None,
        mf_config: MaskformerConfig | None = None,
        input_map: dict[str, str] | None = None,
        num_inputs: dict | None = None,
        nan_to_num: bool = False,
        global_object: str = "jets",
        PARAMETERS: dict | None = None,
        selections: dict[str, list[str]] | None = None,
    ):
        """An efficient map-style dataset for loading data from an H5 file containing structured
        arrays.

        Parameters
        ----------
        filename : str
            Input h5 filepath containing structured arrays
        norm_dict : str
            Path to file containing normalisation parameters
        variables : Vars
            Input variables used in the forward pass for each input type
        stage : str
            Stage of the training process
        num : int, optional
            Number of input samples to use. If `-1`, use all input samples
        labels : Vars
            List of required labels for each input type
        mf_config : MaskformerConfig, optional
            Config for Maskformer matching, by default None
        input_map : dict, optional
            Map names to the corresponding dataset names in the input h5 file.
            If not provided, the input names will be used as the dataset names.
        num_inputs : dict, optional
            Truncate the number of constituent inputs to this number, to speed up training
        nan_to_num : bool, optional
            Convert nans to zeros when loading inputs
        global_object : str
            Name of the global input object, as opposed to the constituent-level
            inputs
        PARAMETERS: dict
            Variables used to parameterise the network, by default None.
        selections : dict, optional
            Selections to apply to the input data, by default None.
        """
        super().__init__()
        # check labels have been configured
        self.labels = labels if labels is not None else {}

        # default input mapping: use input names as dataset names
        # allow only partial maps to be provided
        if input_map is None:
            input_map = {k: k for k in variables}

        if "GLOBAL" in input_map:
            input_map["GLOBAL"] = global_object

        if "PARAMETERS" in input_map:
            input_map["PARAMETERS"] = global_object

        self.input_map = input_map
        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        self.num_inputs = num_inputs
        self.nan_to_num = nan_to_num
        self.global_object = global_object
        self.selections = selections
        self.selectors = {}
        if self.selections:
            for key, value in self.selections.items():
                self.selectors[key] = TrackSelector(Cuts.from_list(value))

        # If MaskFormer matching is enabled, extract the relevent labels
        self.mf_config = deepcopy(mf_config)
        if self.mf_config:
            self.input_map["objects"] = self.mf_config.object.name

        self.variables = variables
        self.norm_dict = norm_dict
        self.PARAMETERS = PARAMETERS
        self.stage = stage
        self.rng = np.random.default_rng()

        # check that num_inputs contains valid keys
        if self.num_inputs is not None and not set(self.num_inputs).issubset(self.variables):
            raise ValueError(
                f"num_inputs keys {self.num_inputs.keys()} must be a subset of input variables"
                f" {self.variables.keys()}"
            )

        self.check_file()

        self.input_variables = variables
        assert self.input_variables is not None

        # check parameters listed in variables appear in the same order in the PARAMETERS block
        if "PARAMETERS" in self.input_variables:
            assert self.PARAMETERS is not None
            assert self.input_variables["PARAMETERS"] is not None
            assert len(self.input_variables["PARAMETERS"]) == len(self.PARAMETERS)
            for idx, param_key in enumerate(self.PARAMETERS.keys()):
                assert self.input_variables["PARAMETERS"][idx] == param_key

        # setup datasets and accessor arrays
        self.dss = {}
        self.arrays = {}
        for internal, external in self.input_map.items():
            self.dss[internal] = self.file[external]
            this_vars = self.labels[internal].copy() if internal in self.labels else []
            this_vars += self.input_variables.get(internal, [])
            if internal == "EDGE":
                dtype = get_dtype_edge(self.file[external], this_vars)
            else:
                dtype = get_dtype(self.file[external], this_vars)
            self.arrays[internal] = np.array(0, dtype=dtype)
        if self.global_object not in self.dss:
            self.dss[self.global_object] = self.file[self.global_object]

        # set number of objects
        self.num = self.get_num(num)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return int(self.num)

    def __getitem__(self, object_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        object_idx
            A numpy slice corresponding to a batch of objects.

        Returns
        -------
        tuple
            Dict of tensor for each of the inputs, pad_masks, and labels.
            Each tensor will contain a batch of samples.
        """
        inputs = {}
        labels = {}
        pad_masks = {}

        # loop over input types
        for input_name in self.input_map:
            # load data (inputs + labels) for this input type
            batch = self.arrays[input_name]
            shape = (object_idx.stop - object_idx.start,) + self.dss[input_name].shape[1:]
            batch.resize(shape, refcheck=False)
            self.dss[input_name].read_direct(batch, object_idx)

            # truncate constituent inputs
            if self.num_inputs is not None and input_name in self.num_inputs:
                assert int(self.num_inputs[input_name]) <= batch.shape[1]
                batch = batch[:, : int(self.num_inputs[input_name])]

            # apply selections for constituent inputs
            if self.selectors and (selector := self.selectors.get(input_name)):
                batch = selector(batch)

            # load edge inputs for this input type
            if input_name == "EDGE":
                inputs[input_name] = torch.from_numpy(
                    get_inputs_edge(batch, self.input_variables[input_name])
                )

            # load PARAMETERS for this input type
            elif input_name == "PARAMETERS":
                flat_array = s2u(batch[self.input_variables[input_name]], dtype=np.float32)

                for ind, param in enumerate(self.PARAMETERS):
                    if self.stage == "fit":
                        # assign random values to inputs with parameters not set to those in the
                        # train list, values are chosen at random from those in the train list
                        # according to probabilities if given, else with equal probability
                        try:
                            prob = self.PARAMETERS[param]["prob"]
                        except KeyError:
                            prob = None
                        mask = ~np.isin(flat_array[:, ind], self.PARAMETERS[param]["train"])
                        random = self.rng.choice(
                            self.PARAMETERS[param]["train"], size=np.sum(mask), p=prob
                        )
                        flat_array[mask, ind] = random

                    if self.stage == "test":
                        # assign parameter values for all objects passed in the 'test' option
                        test_arr = np.full(np.shape(flat_array)[0], self.PARAMETERS[param]["test"])
                        flat_array[:, ind] = test_arr

                inputs[input_name] = torch.from_numpy(flat_array)

            # load standard inputs for this input type
            elif self.input_variables.get(input_name):
                flat_array = s2u(batch[self.input_variables[input_name]], dtype=np.float32)
                if self.nan_to_num:
                    flat_array = np.nan_to_num(flat_array)
                inputs[input_name] = torch.from_numpy(maybe_copy(flat_array))

                # apply the input padding mask
                if "valid" in batch.dtype.names and input_name not in {
                    "EDGE",
                    "PARAMETERS",
                    self.global_object,
                    "GLOBAL",
                }:
                    pad_masks[input_name] = ~torch.from_numpy(batch["valid"])
                    inputs[input_name][pad_masks[input_name]] = 0

                # check inputs are finite
                if not torch.isfinite(inputs[input_name]).all():
                    raise ValueError(f"Non-finite inputs for '{input_name}' in {self.filename}.")

            # process labels for this input type
            if input_name in self.labels:
                labels[input_name] = {}
                for label in self.labels[input_name]:
                    dtype = torch.long if np.issubdtype(batch[label].dtype, np.integer) else None
                    batch_label = maybe_copy(batch[label])
                    labels[input_name][label] = torch.as_tensor(batch_label, dtype=dtype)
                    x = torch.as_tensor(batch_label, dtype=dtype)
                    if input_name == "objects" and label == self.mf_config.object.class_label:
                        for k, v in self.mf_config.object.class_map.items():
                            x[x == k] = v
                            labels[input_name]["object_class"] = x
                    labels[input_name][label] = x

                # hack to handle the old umami train file format
                if input_name == self.global_object and "/" in self.labels:
                    if self.global_object not in labels:
                        labels[self.global_object] = {}
                    for label in self.labels["/"]:
                        labels[input_name][label] = torch.as_tensor(
                            self.file["labels"][object_idx], dtype=torch.long
                        )

        if self.mf_config:
            labels["objects"]["masks"] = build_target_masks(
                labels["objects"][self.mf_config.object.id_label],
                labels[self.mf_config.constituent.name][self.mf_config.constituent.id_label],
            )

        return inputs, pad_masks, labels

    def get_num(self, num_requested: int):
        num_available = len(self.dss[self.global_object])

        # not enough objects
        if num_requested > num_available:
            raise ValueError(
                f"Requested {num_requested:,} from {self.global_object}, but only"
                f" {num_available:,} are available in the file {self.filename}."
            )

        # use all objects
        if num_requested < 0:
            return num_available

        # use requested number of objects
        return num_requested

    def check_file(self):
        keys = {self.input_map[k] for k in self.variables}
        available = set(self.file.keys())
        if missing := keys - available - {"EDGE", "GLOBAL", "PARAMETERS"}:
            raise KeyError(
                f"Input file '{self.filename}' does not contain keys {missing}."
                f" Available keys: {available}"
            )
        for k, v in self.variables.items():
            if k == "EDGE":
                continue
            if k == "PARAMETERS":
                continue
            if k == "GLOBAL":
                k = self.global_object  # noqa: PLW2901
            name = self.input_map[k]
            if not isinstance(self.file[name], h5py.Dataset):
                raise KeyError(f"The object '{name}' in file '{self.filename}' is not a dataset.")
            if missing := set(v) - set(self.file[name].dtype.names):
                raise KeyError(
                    f"Variables {missing} are missing from dataset '{name}' in input file"
                    f" '{self.filename}'."
                )


def get_dtype(ds, variables=None) -> np.dtype:
    """Return a dtype based on an existing dataset and requested variables."""
    if variables is None:
        variables = ds.dtype.names
    if "valid" in ds.dtype.names and "valid" not in variables:
        variables.append("valid")

    variables_flat = []
    for item in variables:
        if isinstance(item, list):
            variables_flat += item
        else:
            variables_flat.append(item)

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in variables])
