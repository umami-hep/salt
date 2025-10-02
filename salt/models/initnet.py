from torch import nn

from salt.models import Dense, FeaturewiseTransformation
from salt.models.posenc import PositionalEncoder
from salt.stypes import Tensors, Vars
from salt.utils.tensor_utils import attach_context


class InitNet(nn.Module):
    """Initial input embedding network.

    This class can handle global input concatenation and positional encoding.

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
    attach_global : bool, optional
        Concatenate global-level inputs with constituent-level inputs before embedding,
        by default True
    pos_enc : PositionalEncoder | None, optional
        Positional encoder module to use. See
        [`salt.models.PositionalEncoder`][salt.models.PositionalEncoder] for details.
        By default None
    mup: bool, optional
        Whether to use the muP parametrisation (impacts initialisation), by default False
    featurewise: FeaturewiseTransformation | None, optional
        Networks to apply featurewise transformations to inputs, set automatically by
        the framework. By default None
    """

    def __init__(
        self,
        input_name: str,
        dense_config: dict,
        variables: Vars,
        global_object: str,
        attach_global: bool = True,
        pos_enc: PositionalEncoder | None = None,
        mup: bool = False,
        featurewise: FeaturewiseTransformation | None = None,
    ):
        super().__init__()

        # set input size
        if "input_size" not in dense_config:
            dense_config["input_size"] = len(variables[input_name])
            if attach_global and input_name != "EDGE":
                dense_config["input_size"] += len(variables[global_object])
                if not featurewise:
                    dense_config["input_size"] += len(variables.get("parameters", []))

        self.input_name = input_name
        self.net = Dense(**dense_config)
        self.variables = variables
        self.attach_global = attach_global
        self.global_object = global_object
        self.pos_enc = pos_enc
        self.mup = mup
        if mup:
            self.net.reset_parameters()
        self.featurewise = featurewise

    def forward(self, inputs: Tensors):
        # get the inputs for this init net
        x = inputs[self.input_name]

        # add global features
        if self.attach_global:
            x = attach_context(x, inputs[self.global_object])

        # add parameters if not using featurewise transformations
        if "parameters" in self.variables and not self.featurewise:
            x = attach_context(x, inputs["parameters"])

        # apply featurewise transformations
        if self.featurewise is not None:
            x = self.featurewise(inputs, x)

        # inital projection
        x = self.net(x)

        # add positional encoding
        if self.pos_enc:
            this_vars = self.variables[self.input_name]
            input_indices = [this_vars.index(v) for v in self.pos_enc.variables]
            x += self.pos_enc(inputs[self.input_name][..., input_indices])

        return x
