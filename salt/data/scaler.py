from pathlib import Path

import numpy as np
import yaml


class NormDictScaler:
    def __init__(
        self,
        norm_dict: str | Path,
        input_names: dict,
        variables: dict | None = None,
        concat_jet_tracks: bool = False,
    ):
        """Normalise variables using a dictionary of normalisation parameters.

        Parameters
        ----------
        norm_dict : str | Path
            Path to file containing normalisation parameters
        input_names : dict
            Names of the h5 group to access for each type of input
        variables : dict
            Variables and labels to use for the training
        concat_jet_tracks : bool, optional
            Concatenate jet inputs with track-type inputs, by default True
        """
        self.input_names = input_names
        self.input_types = dict(map(reversed, self.input_names.items()))  # type: ignore
        with open(norm_dict) as f:
            self.norm_dict = yaml.safe_load(f)

        if missing := set(self.input_types) - (avai := set(self.norm_dict)):
            raise ValueError(f"No norm params for {missing} in {norm_dict}. Choose from {avai}.")

        # if no variables are specified, use all variables in the norm_dict
        self.variables = variables
        if self.variables is None:
            self.variables = {
                self.input_types[k]: list(v.keys())
                for k, v in self.norm_dict.items()
                if k in self.input_types
            }
        assert self.variables is not None

        # get norm params as arrays
        self.norm_params = {}
        for input_type in self.variables:
            if input_type == "edge":
                continue
            nd = self.norm_dict[self.input_names[input_type]]
            var = self.variables[input_type]
            mean_key = "mean" if "mean" in nd[var[0]] else "shift"
            std_key = "std" if "std" in nd[var[0]] else "scale"

            means = [nd[v][mean_key] for v in var]
            stds = [nd[v][std_key] for v in var]
            if concat_jet_tracks and input_type not in ["jet", "global", "edge"]:
                jets_name = self.input_names["jet"]
                jet_nd = self.norm_dict[jets_name]
                jet_var = self.variables["jet"]
                means = [jet_nd[v][mean_key] for v in jet_var] + means
                stds = [jet_nd[v][std_key] for v in jet_var] + stds

            means = np.array(means, dtype=np.float32)
            stds = np.array(stds, dtype=np.float32)
            self.norm_params[input_type] = (means, stds)

    def __call__(self, array: np.ndarray, input_type: str):
        """Normalise all variables of a given type in parallel.

        Parameters
        ----------
        array : np.ndarray
            Array to normalise
        input_type : str
            Type of input array

        Returns
        -------
        np.ndarray
            Normalised array
        """
        means, stds = self.norm_params[input_type]
        return (array - means) / stds
