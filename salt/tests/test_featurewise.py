import pytest
import torch

from salt.models import FeaturewiseTransformation


@pytest.fixture
def dense_config():
    return {"output_size": 5, "hidden_layers": [5], "activation": "ReLU"}


@pytest.fixture
def variables():
    return {"PARAMETERS": ["x"]}


def test_featurewise_forward(dense_config, variables):
    layer = "input"

    featurewise = FeaturewiseTransformation(
        layer,
        variables,
        dense_config_scale=dense_config,
        dense_config_bias=dense_config,
    )

    inputs = {"PARAMETERS": torch.randn(1, 1)}
    features = torch.rand(1, 10, 5)
    transformed_features = featurewise(inputs, features)

    expected_output_shape = features.shape
    output_shape = transformed_features.shape
    # featurewise transformations should preserve size
    assert expected_output_shape == output_shape
