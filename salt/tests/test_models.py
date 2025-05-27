import copy

import numpy as np
import pytest
import torch
from torch import nn

from salt.models import (
    Dense,
    GATv2Attention,
    GlobalAttentionPooling,
    MultiheadAttention,
    ScaledDotProductAttention,
    TensorCrossAttentionPooling,
    TransformerCrossAttentionEncoder,
    TransformerEncoder,
)
from salt.utils.inputs import get_random_mask


def test_dense() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU")
    net(torch.rand(10))


def test_dense_context() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(10), torch.rand(4))


def test_dense_context_broadcast() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(1, 10, 10), torch.rand(1, 4))


@pytest.mark.parametrize("pooling", [GlobalAttentionPooling, TensorCrossAttentionPooling])
def test_pooling(pooling) -> None:
    if pooling != GlobalAttentionPooling:
        net = pooling(10, 1, {"num_heads": 1, "attention": ScaledDotProductAttention()})
    else:
        net = pooling(10)

    x = {"emb": torch.rand(1, 5, 10)}
    out = net(x)

    x = {"emb": torch.cat([x["emb"], torch.zeros((1, 1, x["emb"].shape[2]))], dim=1)}
    mask = get_random_mask(1, 6, p_valid=1)
    mask[:, -1] = True
    mask = {"mask": mask}
    out_with_mask = net(x, pad_mask=mask)
    assert torch.all(out == out_with_mask)


def test_transformer() -> None:
    net = TransformerEncoder(
        embed_dim=10,
        num_layers=2,
        mha_config={
            "num_heads": 2,
            "attention": ScaledDotProductAttention(),
        },
        dense_config={
            "activation": "ReLU",
            "hidden_layers": [10],
        },
    )
    # basic test
    net(torch.rand(10, 10, 10))

    # test zero track case
    assert torch.all(net(torch.rand(10, 0, 10)) == torch.empty((10, 0, 10)))

    # test fully padded case
    out = net(torch.rand(1, 10, 10), pad_mask=get_random_mask(1, 10, p_valid=0))
    assert not torch.isnan(out).any()

    # test that adding a padded track does not change the output
    tracks = torch.rand(1, 10, 10)
    mask = get_random_mask(1, 10, p_valid=1)
    out = net(tracks, pad_mask=mask)
    tracks = torch.cat([tracks, torch.zeros((1, 1, tracks.shape[2]))], dim=1)
    mask = torch.zeros(tracks.shape[:-1]).bool()
    mask[:, -1] = True
    out_with_pad = net(tracks, pad_mask=mask)[:, :-1]
    tensor_check = out.data.half() == out_with_pad.data.half()
    assert torch.all(tensor_check)


def test_transformer_cross_attention_encoder() -> None:
    net = TransformerCrossAttentionEncoder(
        input_names=["type1", "type2"],
        embed_dim=10,
        num_layers=2,
        mha_config={
            "num_heads": 2,
            "attention": ScaledDotProductAttention(),
        },
        sa_dense_config={
            "activation": "ReLU",
            "hidden_layers": [10],
        },
    )
    # Basic Functionality Test
    x = {
        "type1": torch.rand(10, 10, 10),
        "type2": torch.rand(10, 10, 10),
    }
    mask = {
        "type1": get_random_mask(10, 10, p_valid=1),
        "type2": get_random_mask(10, 10, p_valid=1),
    }
    net(x, mask)

    # Zero Input Case Test
    x["type1"] = torch.rand(10, 0, 10)
    mask["type1"] = torch.empty((10, 0), dtype=bool)
    assert torch.all(net(x, mask)["type1"] == torch.empty((10, 0, 10)))

    # Padded Case Test
    x["type1"] = torch.rand(1, 10, 10)
    x["type2"] = torch.rand(1, 10, 10)
    mask["type1"] = get_random_mask(1, 10, p_valid=0)
    mask["type2"] = get_random_mask(1, 10, p_valid=1)
    out = net(x, mask)
    assert not torch.isnan(out["type1"]).any()

    # Padding Invariance Test
    del x
    x = {
        "type1": torch.rand(1, 10, 10),
        "type2": torch.rand(1, 10, 10),
    }
    extended_x = copy.deepcopy(x)
    del extended_x["type1"]
    extended_x.update({
        "type1": torch.cat([copy.deepcopy(x["type1"]), torch.zeros((1, 1, 10))], dim=1)
    })
    mask["type1"] = get_random_mask(1, 10, p_valid=1)
    out = net(x, mask)
    mask["type1"] = torch.zeros(extended_x["type1"].shape[:-1]).bool()
    mask["type1"][:, -1] = True
    out_with_pad = net(extended_x, mask)["type1"][:, :-1]
    torch.testing.assert_close(out["type1"], out_with_pad)


def test_mha_allvalid_mask() -> None:
    n_batch = 2
    n_trk = 3
    n_head = 1
    n_dim = 5

    net = MultiheadAttention(
        embed_dim=n_dim,
        num_heads=n_head,
        attention=ScaledDotProductAttention(),
    )

    q = k = v = torch.rand((n_batch, n_trk, n_dim))
    valid_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)
    out_no_mask = net(q, k, v, q_mask=None, kv_mask=None)
    out_mask = net(q, k, v, q_mask=valid_mask, kv_mask=valid_mask)
    assert torch.all(out_no_mask == out_mask)


def test_gatv2():
    n_batch = 10
    n_trk = 10
    n_head = 4
    n_dim = 16
    head_dim = n_dim // n_head

    net = MultiheadAttention(
        attention=GATv2Attention(n_head, head_dim, activation="SiLU"),
        embed_dim=n_dim,
        num_heads=n_head,
    )

    q = k = v = torch.rand((n_batch, n_trk, n_dim))
    valid_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)
    out_no_mask = net(q, k, v, q_mask=None, kv_mask=None)
    out_mask = net(q, k, v, q_mask=valid_mask, kv_mask=valid_mask)
    assert torch.all(out_no_mask == out_mask)

    q_mask = kv_mask = get_random_mask(n_batch, n_trk)
    net(q, k, v, q_mask=q_mask, kv_mask=kv_mask)


def test_mha_qkv_different_dims():
    n_batch = 3
    n_trk = 5
    n_head = 2
    k_dim = 7
    q_dim = 8
    v_dim = 9

    net = MultiheadAttention(
        num_heads=n_head,
        embed_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        attention=ScaledDotProductAttention(),
    )

    q = torch.rand((n_batch, n_trk, q_dim))
    k = torch.rand((n_batch, n_trk + 1, k_dim))
    v = torch.rand((n_batch, n_trk + 1, v_dim))

    q_mask = get_random_mask(n_batch, n_trk)
    kv_mask = get_random_mask(n_batch, n_trk + 1)
    net(q, k, v, q_mask=q_mask, kv_mask=kv_mask)


@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 1.0])
def test_mha_vs_torch(frac_pad):
    torch.manual_seed(0)
    n_batch = 4
    n_trk = 10
    n_head = 1
    n_dim = 8
    frac_pad = 0.2

    t_net, s_net = get_pytorch_salt_mha(n_dim, n_head)

    mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
    mask[..., : int(frac_pad * n_trk)] = True  # set some padding

    x = torch.rand((n_batch, n_trk, n_dim))
    torch_out = t_net(x, x, x, key_padding_mask=mask)[0]
    out = s_net(x, x, x, kv_mask=mask)

    np.testing.assert_allclose(
        torch_out.detach().numpy(), out.detach().numpy(), rtol=1e-6, atol=1e-6
    )


def get_pytorch_salt_mha(n_dim, n_head):
    """Get a pytorch and salt MHA layer with equivalent weights."""
    t_net = nn.MultiheadAttention(
        embed_dim=n_dim, num_heads=n_head, batch_first=True, bias=True, add_zero_attn=True
    )

    s_net = MultiheadAttention(
        attention=ScaledDotProductAttention(), embed_dim=n_dim, num_heads=n_head
    )

    weights = torch.rand((3 * n_dim, n_dim))
    bias = torch.rand(3 * n_dim)
    t_net.in_proj_weight = nn.Parameter(weights)
    t_net.in_proj_bias = nn.Parameter(bias)

    wq, wk, wv = weights.chunk(3)
    bq, bk, bv = bias.chunk(3)
    s_net.linear_q.weight = nn.Parameter(wq)
    s_net.linear_k.weight = nn.Parameter(wk)
    s_net.linear_v.weight = nn.Parameter(wv)
    s_net.linear_q.bias = nn.Parameter(bq)
    s_net.linear_k.bias = nn.Parameter(bk)
    s_net.linear_v.bias = nn.Parameter(bv)

    s_net.linear_out.weight = t_net.out_proj.weight
    s_net.linear_out.bias = t_net.out_proj.bias

    return t_net, s_net
