import torch
from torch import Tensor, nn

from salt.models import Dense
from salt.stypes import Vars


class FeaturewiseTransformation(nn.Module):
    def __init__(
        self,
        layer: str,
        variables: Vars,
        dense_config_scale: dict | None = None,
        dense_config_bias: dict | None = None,
        apply_norm: bool = False,
    ):
        """Perform feature wise transformations on the features of a layer.
        https://distill.pub/2018/feature-wise-transformations/.

        Parameters
        ----------
        layer : str
            layer to scale/bias (either "input", "encoder", or "global")
        variables : Vars
            Input variables used in the forward pass, set automatically by the framework
        dense_config_scale : dict
            Keyword arguments for [salt.models.Dense][salt.models.Dense],
            the dense network performing the scaling.
        dense_config_bias : dict
            Keyword arguments for [salt.models.Dense][salt.models.Dense],
            the dense network performing the biasing.
        apply_norm : bool
            Apply layer normalisation to the transformed features. By default false.
        """
        super().__init__()

        self.layer = layer
        if layer not in {"input", "encoder", "global"}:
            raise ValueError(
                "Select either 'input', 'encoder' or 'global' layers for featurewise nets."
            )

        self.scale_net = None
        self.bias_net = None
        self.num_features = None
        self.norm = None

        if dense_config_scale:
            dense_config_scale["input_size"] = len(variables.get("PARAMETERS", []))
            self.scale_net = Dense(**dense_config_scale)
            self.num_features = self.scale_net.output_size
        if dense_config_bias:
            dense_config_bias["input_size"] = len(variables.get("PARAMETERS", []))
            self.bias_net = Dense(**dense_config_bias)
            self.num_features = self.bias_net.output_size

        if not self.bias_net and not self.scale_net:
            raise ValueError(
                "Need to specify at least one dense_config_scale or dense_config_bias."
            )

        if apply_norm:
            self.norm = nn.LayerNorm(self.num_features)

    def forward(self, inputs: dict, features: Tensor):
        if "PARAMETERS" not in inputs:
            raise ValueError("Featurewise transformations require 'PARAMETERS'.")
        x = inputs["PARAMETERS"]
        if self.scale_net:
            features = self.scale_net(x).unsqueeze(1) * features
        if self.bias_net:
            features = torch.add(features, self.bias_net(x).unsqueeze(1))
        if self.norm:
            features = self.norm(features)
        return features
