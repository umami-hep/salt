from torch import nn

from salt.stypes import Tensors, Vars
from salt.utils.edge_features import calculate_edge_features, check_edge_config


class EdgeConstructor(nn.Module):
    """Constructs edge features for a specified input modality.

    Parameters
    ----------
    input_name : str
        Name of the input modality to construct edge features for.
    edge_features : list[str]
        List of edge feature names to compute (e.g., ``["dR", "mass"]``).
    variables : Vars
        A dictionary mapping input names to their feature lists.
    indices_map : dict | None, optional
        Mapping from feature name to index within the input features tensor.
    """

    def __init__(
        self,
        input_name: str,
        edge_features: list[str],
        variables: Vars,
        indices_map: dict | None = None,
    ) -> None:
        super().__init__()
        self.input_name = input_name
        self.edge_features = edge_features
        if indices_map is None:
            check_edge_config(edge_features, variables[input_name])
            self.indices_map = {}
            for i, var_name in enumerate(variables[input_name]):
                self.indices_map[var_name] = i
        else:
            check_edge_config(edge_features, list(indices_map.keys()))
            self.indices_map = indices_map

    def forward(self, inputs: Tensors) -> Tensors:
        x = inputs[self.input_name]
        ebatch = calculate_edge_features(x, self.indices_map, self.edge_features)
        inputs[f"_edge_features_{self.input_name}"] = ebatch
        return inputs
