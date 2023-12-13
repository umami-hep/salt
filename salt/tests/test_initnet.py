import pytest
import torch

from salt.models import InitNet, PositionalEncoder


class MockInputs:
    def __getitem__(self, key):
        if key == "test_input":
            return torch.rand(10, 10, 5)
        if key == "global_object":
            return torch.rand(10, 2)
        raise KeyError


@pytest.fixture
def dense_config():
    return {"hidden_layers": [10, 20], "output_size": 32}


@pytest.fixture
def variables():
    return {"test_input": ["x", "y", "z", "phi", "theta"], "global_object": ["pt", "eta"]}


@pytest.mark.parametrize(
    "pos_enc, attach_global",
    [(None, False), (None, True), (PositionalEncoder(["x", "y"], 32), True)],
)
def test_init_net_forward(dense_config, variables, pos_enc, attach_global):
    input_name = "test_input"
    global_object = "global_object"

    net = InitNet(
        input_name=input_name,
        dense_config=dense_config,
        variables=variables,
        global_object=global_object,
        attach_global=attach_global,
        pos_enc=pos_enc,
    )

    inputs = MockInputs()
    output = net(inputs)

    # Assert output shape is correct
    expected_output_shape = dense_config["output_size"]
    assert output.shape[-1] == expected_output_shape
