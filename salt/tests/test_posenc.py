import pytest
import torch

from salt.models.posenc import PositionalEncoder


@pytest.fixture
def dim():
    return 32


@pytest.mark.parametrize("variables", [["x"], ["x", "y"], ["x", "y", "z"]])
def test_positional_encoder_initialization(variables):
    dim = 32
    encoder = PositionalEncoder(variables, dim)
    assert encoder.variables == variables
    assert encoder.dim == dim
    assert encoder.per_input_dim == dim // (2 * len(variables))
    assert encoder.last_dim == dim % (2 * len(variables))


@pytest.mark.parametrize("variables", [["x"], ["x", "y"], ["x", "y", "z"]])
def test_positional_encoder_call(variables, dim):
    encoder = PositionalEncoder(variables, dim)
    inputs = torch.rand(10, len(variables))
    encoded = encoder(inputs)

    # Verify output shape
    expected_shape = (10, dim)
    assert encoded.shape == expected_shape

    if dim % (2 * len(variables)) != 0:
        assert torch.all(encoded[..., -1] == 0)


@pytest.mark.parametrize("variables", [["x"], ["x", "y"], ["x", "y", "z"]])
def test_pos_enc_method(variables, dim):
    encoder = PositionalEncoder(variables, dim)
    xs = torch.rand(10)
    dim_per_var = dim // (2 * len(variables))

    # Test non-symmetric encoding
    encoded_non_sym = encoder.pos_enc(xs, dim_per_var)
    assert encoded_non_sym.shape == (10, 2 * dim_per_var)

    # Test symmetric encoding
    encoded_sym = encoder.pos_enc(xs, dim_per_var, symmetric=True)
    assert encoded_sym.shape == (10, 2 * dim_per_var)
