from pathlib import Path

import torch
import yaml
from torch import nn

from salt.typing import Tensors, Vars


class InputNorm(nn.Module):
    def __init__(
        self, norm_dict: Path, variables: Vars, global_object: str, input_map: dict[str, str]
    ) -> None:
        """Normalise inputs on the fly using a pre-computed normalisation dictionary.

        Parameters
        ----------
        norm_dict : Path
            Path to file containing normalisation parameters
        variables : dict
            Input variables for each type of input
        global_object : str
            Name of the global input object, as opposed to the constituent-level
            inputs
        input_map : dict
            Map names to the corresponding dataset names in the input h5 file.
            Set automatically by the framework.
        """
        super().__init__()
        self.variables = variables
        self.global_object = global_object
        self.NO_NORM = ["EDGE", "PARAMETERS"]
        with open(norm_dict) as f:
            self.norm_dict = yaml.safe_load(f)

        # get the keys that need to be normalised
        if input_map is None:
            input_map = {k: k for k in variables}
        keys = {input_map[k] for k in set(variables.keys())}
        if "EDGE" in keys:
            keys.remove("EDGE")
        if "GLOBAL" in keys:
            keys.remove("GLOBAL")
            keys.add(self.global_object)

        # check we have all required keys in the normalisation dictionary
        if missing := keys - set(self.norm_dict):
            raise ValueError(
                f"Missing input types {missing} in {norm_dict}. Choose from"
                f" {self.norm_dict.keys()}."
            )

        # check we have all required variables for each input type
        for k, vs in variables.items():
            if k in self.NO_NORM:
                continue
            name = input_map[k]
            if k == "GLOBAL":
                name = self.global_object
            if missing := set(vs) - set(self.norm_dict[name]):
                raise ValueError(
                    f"Missing variables {missing} for {name} in {norm_dict}. Choose from"
                    f" {self.norm_dict[name].keys()}."
                )

            # store normalisation parameters with the model
            means = torch.as_tensor([self.norm_dict[name][v]["mean"] for v in vs])
            stds = torch.as_tensor([self.norm_dict[name][v]["std"] for v in vs])
            self.register_buffer(f"{k}_means", means)
            self.register_buffer(f"{k}_stds", stds)

            # check normalisation parameters are ok
            if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
                raise ValueError(f"Non-finite normalisation parameters for {name} in {norm_dict}.")
            if any(stds == 0):
                raise ValueError(f"Zero standard deviation for {name} in {norm_dict}.")

    def forward(self, inputs: Tensors) -> Tensors:
        for k, x in inputs.items():
            if k in self.NO_NORM:
                continue
            inputs[k] = (x - getattr(self, f"{k}_means")) / getattr(self, f"{k}_stds")
        return inputs
