from torch import nn

from salt.models import Dense
from salt.utils.tensor_utils import attach_context
from salt.utils.typing import Tensors, Vars


class InitNet(nn.Module):
    def __init__(
        self,
        input_name: str,
        dense_config: dict,
        variables: Vars,
        global_object: str,
        attach_global: bool = True,
        muP: bool = False,
    ):
        """Initial input embedding network which can handle input concatenation.

        Parameters
        ----------
        input_name : str
            Name of the input, must match the input types in the data config
        dense_config : dict
            Keyword arguments for [`salt.models.Dense`][salt.models.Dense],
            the dense network producing the initial embedding. The `input_size`
            argument is inferred automatically by the framework
        variables : Vars
            Input variables used in the forward pass, set automatically by the framework
        global_object : str
            Name of the global object, set automatically by the framework
        attach_global : str, optional
            Concatenate global-level inputs with constituent-level inputs before embedding
        muP: bool, optional,
            Whether to use the muP parametrisation (impacts initialisation).
        """
        super().__init__()

        # set input size
        if "input_size" not in dense_config:
            dense_config["input_size"] = len(variables[input_name])
            if attach_global and input_name != "EDGE":
                dense_config["input_size"] += len(variables[global_object])
                dense_config["input_size"] += len(variables.get("PARAMETERS", []))

        self.input_name = input_name
        self.net = Dense(**dense_config)
        self.variables = variables
        self.attach_global = attach_global
        self.global_object = global_object
        self.muP = muP
        if muP:
            self.net.reset_parameters()

    def forward(self, inputs: Tensors):
        x = inputs[self.input_name]

        if self.attach_global:
            x = attach_context(x, inputs[self.global_object])

        if "PARAMETERS" in self.variables:
            x = attach_context(x, inputs["PARAMETERS"])

        return self.net(x)
