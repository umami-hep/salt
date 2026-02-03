import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from ftag import Cuts, Labeller
from ftag.track_selector import TrackSelector
from ftag.vds import create_virtual_file
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from torch.utils.data import Dataset

from salt.data.edge_features import get_dtype_edge, get_inputs_edge
from salt.stypes import Vars
from salt.utils.array_utils import maybe_copy
from salt.utils.configs import LabellerConfig, MaskformerConfig
from salt.utils.inputs import as_half
from salt.utils.mask_utils import build_target_masks


class SaltDataset(Dataset):
    """An efficient map-style dataset for loading data from an H5 file containing structured
    arrays.

    Parameters
    ----------
    filename : str | Path
        Input h5 filepath containing structured arrays
    norm_dict : str | Path
        Path to file containing normalisation parameters
    variables : Vars
        Input variables used in the forward pass for each input type
    stage : str
        Stage of the training process
    num : int, optional
        Number of input samples to use. If `-1`, use all input samples
    labeller_config : LabellerConfig | None, optional
        Configuration to apply relabelling on-the-fly for jet classification
    labels : Vars | None, optional
        List of required labels for each input type
    mf_config : MaskformerConfig | None, optional
        Config for Maskformer matching, by default None
    input_map : dict[str, str] | None, optional
        Map names to the corresponding dataset names in the input h5 file.
        If not provided, the input names will be used as the dataset names.
    vds_path : str | None, optional
        When using a wildcard as input h5 file, SaltDataset will create a VDS
        (virtual dataset) file. By default (None), this file will be created in a
        folder, which will be created exactly where the wildcard points to. For
        example, if the wildcard points to /path/to/somewhere/pp_train_*.h5, the
        following folder will be created: /path/to/somewhere/pp_train_vds/ and
        in there the file vds.h5. If the vds should be created somewhere else,
        set this option to the path where the h5 file will be created, e.g.
        /path/to/somewhere/my_personal_vds_folder/my_personal_vds.h5
    num_inputs : dict | None, optional
        Truncate the number of constituent inputs to this number, to speed up training
    non_finite_to_num : bool, optional
        Convert nans and infs to zeros when loading inputs
    global_object : str, optional
        Name of the global input object, as opposed to the constituent-level
        inputs
    parameters : dict | None, optional
        Variables used to parameterise the network, by default None.
    selections : dict[str, list[str]] | None, optional
        Selections to apply to the input data, by default None.
    ignore_finite_checks : bool, optional
        Ignoring check for non-finite inputs.
    recover_malformed : bool, optional
        Converts to invalid tracks from malformed inputs in truthOriginLabel.
    transforms : list[Callable] | None, optional
        Transformations to apply to the data, by default None.

    Raises
    ------
    FileNotFoundError
        If no file could be found that matches the given wildcard in filename.
    PermissionError
        If no permission are given to create the VDS file
    ValueError
        if use_labeller is set to true but the classes for relabelling are not supplied.
    """

    def __init__(
        self,
        filename: str | Path,
        norm_dict: str | Path,
        variables: Vars,
        stage: str,
        num: int = -1,
        labeller_config: LabellerConfig | None = None,
        labels: Vars | None = None,
        mf_config: MaskformerConfig | None = None,
        input_map: dict[str, str] | None = None,
        vds_path: str | None = None,
        num_inputs: dict | None = None,
        non_finite_to_num: bool = False,
        global_object: str = "jets",
        parameters: dict | None = None,
        selections: dict[str, list[str]] | None = None,
        ignore_finite_checks: bool = False,
        recover_malformed: bool = False,
        transforms: list[Callable] | None = None,
    ):
        super().__init__()
        # check labels have been configured
        self.labels = labels if labels is not None else {}

        # default input mapping: use input names as dataset names
        # allow only partial maps to be provided
        if input_map is None:
            input_map = {k: k for k in variables}

        if "global" in input_map:
            input_map["global"] = global_object

        if "parameters" in input_map:
            input_map["parameters"] = global_object

        self.input_map = input_map

        # Define the dataset that will be used
        self.filename = Path(filename)

        if "*" in self.filename.name:
            # Check that the wildcard matches at least one file
            matches = list(self.filename.parent.glob(self.filename.name))
            if not matches:
                raise FileNotFoundError(f"No files match wildcard: {self.filename}")

            if vds_path is None:
                # Replace the wildcard marker with "vds"
                vds_name = self.filename.name.replace("*", "vds")

                # Get the path of the new folder for the vds
                vds_dir = self.filename.parent / vds_name.strip(".h5")

                # Try to create the new folder/file.
                try:
                    vds_dir.mkdir(parents=True, exist_ok=True)

                    vds_out_path = Path(
                        create_virtual_file(
                            pattern=self.filename,
                            out_fname=vds_dir / "vds.h5",
                        )
                    )

                # If no permissions, raise error with custom message
                except PermissionError as err:
                    raise PermissionError(
                        f"No permissions to create a VDS folder/file in {vds_out_path}."
                        "Please use the custom vds_path option."
                    ) from err

            else:
                # Ensure the parent directory exists
                Path(vds_path).parent.mkdir(parents=True, exist_ok=True)

                # Create the virtual file
                vds_out_path = Path(
                    create_virtual_file(
                        pattern=self.filename,
                        out_fname=Path(vds_path),
                    )
                )

            # Set the file to the new VDS file
            self.filename = vds_out_path

        # Get the file correctly
        self.file = h5py.File(self.filename, "r")

        self.num_inputs = num_inputs
        self.non_finite_to_num = non_finite_to_num
        self.global_object = global_object
        self.selections = selections
        self.selectors = {}
        self.transforms = transforms
        if self.selections:
            for key, value in self.selections.items():
                self.selectors[key] = TrackSelector(Cuts.from_list(value))

        # If MaskFormer matching is enabled, extract the relevent labels
        self.mf_config = deepcopy(mf_config)
        if self.mf_config:
            self.input_map["objects"] = self.mf_config.object.name

        self.variables = variables
        self.norm_dict = norm_dict
        self.parameters = parameters
        self.stage = stage
        self.labeller_config = labeller_config
        if labeller_config and labeller_config.use_labeller:
            self.labeller = Labeller(labeller_config.class_names, labeller_config.require_labels)
        else:
            self.labeller = None
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

        # check parameters listed in variables appear in the same order in the parameters block
        if "parameters" in self.input_variables:
            assert self.parameters is not None
            assert self.input_variables["parameters"] is not None
            assert len(self.input_variables["parameters"]) == len(self.parameters)
            for idx, param_key in enumerate(self.parameters.keys()):
                assert self.input_variables["parameters"][idx] == param_key

        # set number of objects
        self.num = self.get_num(num)
        self.ignore_finite_checks = ignore_finite_checks
        self.recover_malformed = recover_malformed

        self._is_setup = False

    def _setup(self):
        """Setup the dataset."""
        # setup datasets and accessor arrays
        self.dss = {}
        self.arrays = {}
        file = h5py.File(self.filename, "r")

        for internal, external in self.input_map.items():
            self.dss[internal] = file[external]
            if (internal == external) and internal == self.global_object:
                this_vars = list(self.file[internal].dtype.fields.keys())
            else:
                this_vars = self.labels[internal].copy() if internal in self.labels else []
                this_vars += self.input_variables.get(internal, [])
            if internal == "EDGE":
                dtype = get_dtype_edge(file[external], this_vars)
            else:
                dtype = get_dtype(file[external], this_vars)
            self.arrays[internal] = np.array(0, dtype=dtype)
        if self.global_object not in self.dss:
            self.dss[self.global_object] = file[self.global_object]

        self._is_setup = True

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return int(self.num)

    def __getitem__(
        self, object_idx: slice
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
    ]:
        """Return one batch from the dataset.

        Parameters
        ----------
        object_idx : slice
            A Python slice selecting a contiguous batch of objects. Must have
            ``start`` and ``stop`` set (used to size reads).

        Returns
        -------
        tuple
            A tuple ``(inputs, pad_masks, labels)``:
            - **inputs** : dict[str, torch.Tensor]
            Mapping from input name to a tensor batch.
            - **pad_masks** : dict[str, torch.Tensor]
            Masks for padded entries (True = padded/invalid) per input that supports it.
            - **labels** : dict[str, dict[str, torch.Tensor]]
            Nested mapping ``labels[input_name][label_name] -> tensor``.

        Raises
        ------
        ValueError
            If non finite input values are found in the jets.
        """
        inputs: dict[str, torch.Tensor] = {}
        labels: dict[str, dict[str, torch.Tensor]] = {}
        pad_masks: dict[str, torch.Tensor] = {}

        # Some weird bug seem to occur due to hdf5 not being thread safe. The below ensures
        # that each worker has its own copy of the file to prevent
        if not self._is_setup:
            self._setup()
        # loop over input types
        for input_name in self.input_map:
            # load data (inputs + labels) for this input type
            batch = self.arrays[input_name]
            shape = (object_idx.stop - object_idx.start, *self.dss[input_name].shape[1:])
            batch.resize(shape, refcheck=False)
            self.dss[input_name].read_direct(batch, object_idx)

            # apply selections for constituent inputs
            if self.selectors and (selector := self.selectors.get(input_name)):
                batch = selector(batch)

            # truncate constituent inputs
            if self.num_inputs is not None and input_name in self.num_inputs:
                assert int(self.num_inputs[input_name]) <= batch.shape[1]
                batch = batch[:, : int(self.num_inputs[input_name])]

            # load edge inputs for this input type
            if input_name == "EDGE":
                inputs[input_name] = torch.from_numpy(
                    get_inputs_edge(batch, self.input_variables[input_name])
                )

            # load parameters for this input type
            elif input_name == "parameters":
                flat_array = s2u(batch[self.input_variables[input_name]], dtype=np.float32)

                # Ensure parameters is a dict here for mypy (and logic)
                assert self.parameters is not None, (
                    "self.parameters must be provided when using 'parameters' input"
                )
                params: dict[str, dict[str, Any]] = self.parameters

                for ind, param in enumerate(params):
                    if self.stage == "fit":
                        # assign random values to inputs with parameters not set to those in the
                        # train list, values are chosen at random from those in the train list
                        # according to probabilities if given, else with equal probability
                        try:
                            prob = params[param]["prob"]
                        except KeyError:
                            prob = None
                        mask = ~np.isin(flat_array[:, ind], params[param]["train"])
                        random = self.rng.choice(params[param]["train"], size=np.sum(mask), p=prob)
                        flat_array[mask, ind] = random

                    if self.stage == "test":
                        # assign parameter values for all objects passed in the 'test' option
                        test_arr = np.full(np.shape(flat_array)[0], params[param]["test"])
                        flat_array[:, ind] = test_arr

                inputs[input_name] = torch.from_numpy(flat_array)

            # load standard inputs for this input type
            elif self.input_variables.get(input_name):
                # Apply transforms while variable names are still accessible
                struct_array = batch[self.input_variables[input_name]]
                if self.transforms:
                    for transform in self.transforms:
                        struct_array = transform(struct_array, input_name)

                flat_array = s2u(struct_array, dtype=np.float32)

                if self.non_finite_to_num:
                    flat_array = np.nan_to_num(flat_array, posinf=0, neginf=0)
                inputs[input_name] = torch.from_numpy(maybe_copy(flat_array))

                # apply the input padding mask
                if "valid" in batch.dtype.names and input_name not in {
                    "EDGE",
                    "parameters",
                    self.global_object,
                    "global",
                }:
                    pad_masks[input_name] = ~torch.from_numpy(batch["valid"])
                    inputs[input_name][pad_masks[input_name]] = 0

                # check inputs are finite
                if not torch.isfinite(inputs[input_name]).all():
                    if self.ignore_finite_checks:
                        warnings.warn(
                            f"Non-finite inputs for '{input_name}' in {self.filename.name}."
                            "But ignore finite flag is on, make sure this is intentional.",
                            stacklevel=2,
                        )
                    else:
                        raise ValueError(
                            f"Non-finite inputs for '{input_name}' in {self.filename.name}."
                        )

            # process labels for this input type
            self.process_labels(labels, batch, input_name)

        if self.mf_config:
            labels["objects"]["masks"] = build_target_masks(
                labels["objects"][self.mf_config.object.id_label],
                labels[self.mf_config.constituent.name][self.mf_config.constituent.id_label],
            )
        return inputs, pad_masks, labels

    def get_num(self, num_requested: int):
        num_available = len(self.file[self.global_object])

        # not enough objects
        if num_requested > num_available:
            raise ValueError(
                f"Requested {num_requested:,} from {self.global_object}, but only"
                f" {num_available:,} are available in the file {self.filename.name}."
            )

        # use all objects
        if num_requested < 0:
            return num_available

        # use requested number of objects
        return num_requested

    def check_file(self):
        keys = {self.input_map[k] for k in self.variables}
        available = set(self.file.keys())
        if missing := keys - available - {"EDGE", "global", "parameters"}:
            raise KeyError(
                f"Input file '{self.filename.name}' does not contain keys {missing}."
                f" Available keys: {available}"
            )
        for k, v in self.variables.items():
            match k:
                case "EDGE" | "parameters":
                    continue
                case "global":
                    k = self.global_object  # noqa: PLW2901
                case _:
                    pass
            name = self.input_map[k]
            if not isinstance(self.file[name], h5py.Dataset):
                raise KeyError(
                    f"The object '{name}' in file '{self.filename.name}' is not a dataset."
                )
            if missing := set(v) - set(self.file[name].dtype.names):
                raise KeyError(
                    f"Variables {missing} are missing from dataset '{name}' in input file"
                    f" '{self.filename.name}'."
                )

    def process_labels(self, labels, batch, input_name):
        if (len(self.labels) != 0) and (input_name in self.labels):
            labels[input_name] = {}
            for label in self.labels[input_name]:
                if (
                    self.labeller_config
                    and self.labeller_config.use_labeller
                    and input_name == self.global_object
                    and label == "flavour_label"
                ):
                    labeller = self.labeller
                    all_train_vars = list(self.file["jets"].dtype.fields.keys())
                    for var in labeller.variables:
                        if var not in all_train_vars:
                            raise ValueError("Not enough fields to apply labelling cuts.")
                    labels[input_name][label] = self.labeller_config.class_names
                    labels_on_the_fly = self.labeller.get_labels(batch)
                    labels_np = labels_on_the_fly
                else:
                    batch_labels = maybe_copy(batch[label])
                    labels_np = batch_labels
                if label == "ftagTruthOriginLabel" and len(np.unique(labels_np)) > 9:
                    labels_np = malformed_truthorigin_check(self, labels_np)
                dtype = torch.long if np.issubdtype(labels_np.dtype, np.integer) else None
                labels[input_name][label] = torch.as_tensor(labels_np, dtype=dtype)
                x = torch.as_tensor(labels_np, dtype=dtype)
                if (
                    input_name == "objects"
                    and self.mf_config
                    and label == self.mf_config.object.class_label
                ):
                    for k, v in self.mf_config.object.class_map.items():
                        x[x == k] = v
                        labels[input_name]["object_class"] = x
                labels[input_name][label] = x
            return labels[input_name]
        return None


def get_dtype(ds: h5py.Dataset, variables: Iterable[str] | None = None) -> np.dtype:
    """Return a structured dtype based on an existing dataset and requested variables.

    Parameters
    ----------
    ds : h5py.Dataset
        Input dataset providing the source structured dtype (``ds.dtype``).
    variables : Iterable[str] | None, optional
        Variable names to include in the returned dtype. If ``None``, use
        ``ds.dtype.names``. If the dataset contains a ``"valid"`` field and
        it is not listed, it will be appended automatically.

    Returns
    -------
    numpy.dtype
        Structured dtype consisting of the requested fields. Each field's
        element dtype is converted via :func:`salt.utils.inputs.as_half`.
    """
    # Normalize to a concrete, mutable list
    if variables is None:
        variables_list: list[str] = list(ds.dtype.names or [])
    else:
        variables_list = list(variables)

    if "valid" in (ds.dtype.names or ()) and "valid" not in variables_list:
        variables_list.append("valid")

    variables_flat: list[str] = []
    for item in variables_list:
        if isinstance(item, list):
            variables_flat.extend(item)
        else:
            variables_flat.append(item)

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in variables_flat])


def malformed_truthorigin_check(
    ds: Any,
    input_batch_label: np.ndarray,
) -> np.ndarray:
    """Check and optionally correct malformed ``truthOriginLabels`` in a batch.

    Parameters
    ----------
    ds : Any
        An object with a boolean attribute ``recover_malformed``.
        If ``True``, malformed labels are converted to ``-1``.
        If ``False``, a ``ValueError`` is raised.
    input_batch_label : np.ndarray
        Array of truth origin labels for the current batch.

    Returns
    -------
    np.ndarray
        Either the original labels or a corrected array with all
        invalid labels replaced by ``-1`` if ``ds.recover_malformed`` is ``True``.

    Raises
    ------
    ValueError
        If malformed labels are detected and ``ds.recover_malformed`` is ``False``.

    Notes
    -----
    Valid truth origin labels are assumed to be in the inclusive range
    ``[-1, 7]``. Any value outside this range is considered malformed.

    Examples
    --------
    >>> class Dummy:
    ...     recover_malformed = True
    >>> ds = Dummy()
    >>> labels = np.array([0, 1, 99, -5])
    >>> malformed_truthorigin_check(ds, labels)
    array([ 0,  1, -1, -1])
    """
    if ds.recover_malformed:
        warnings.warn(
            "Malformed truthOriginLabels found with values and counts: "
            f"{np.unique(input_batch_label, return_counts=True)}"
            "Recover flag is on, converting to invalid and continuing."
            "You may still want to check these tracks.",
            stacklevel=2,
        )
        return np.where(
            (input_batch_label > 7) | (input_batch_label < -1),
            -1,
            input_batch_label,
        )
    raise ValueError(
        "Malformed truthOriginLabels found with values and counts: "
        f"{np.unique(input_batch_label, return_counts=True)}"
        "Recover flag is off, failing."
    )
