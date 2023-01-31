import timeit

import numpy as np
import pytest
import torch
import torch.nn as nn

from salt.models import (
    Dense,
    GATv2Attention,
    MultiheadAttention,
    ScaledDotProductAttention,
    TransformerEncoder,
)


def test_dense() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU")
    net(torch.rand(10))


def test_dense_context() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(10), torch.rand(4))


def test_dense_context_broadcast() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(1, 10, 10), torch.rand(1, 4))


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
            "norm_layer": "LayerNorm",
        },
    )
    net(torch.rand(10, 10, 10))


def test_mha_mask() -> None:
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
    valid_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
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
    valid_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
    out_no_mask = net(q, k, v, q_mask=None, kv_mask=None)
    out_mask = net(q, k, v, q_mask=valid_mask, kv_mask=valid_mask)

    assert torch.all(out_no_mask == out_mask)

    q_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
    kv_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
    # TODO: function to generate a realistic padding mask
    q_mask[:, 2:] = kv_mask[:, 4:] = True  # set padding
    q_mask[1, 3:] = kv_mask[1, 5:] = True  # set padding
    net(q, k, v, q_mask=q_mask, kv_mask=kv_mask)


@pytest.mark.parametrize("attention", [ScaledDotProductAttention])
def test_mha_qkv_different_dims(attention):
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

    q_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)  # all valid
    kv_mask = torch.zeros((n_batch, n_trk + 1), dtype=torch.bool)  # all valid
    q_mask[:, 2:] = kv_mask[:, 3:] = True  # set padding
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


def test_mha_vs_torch_timing():
    n_batch = 100
    n_trk = 40
    n_head = 8
    n_dim = 128
    device = "cpu"

    x = torch.rand((n_batch, n_trk, n_dim), device=device)
    mask = torch.zeros((n_batch, n_trk), dtype=torch.bool, device=device)  # all valid
    mask[..., : int(0.5 * n_trk)] = True  # set some padding

    t_net, s_net = get_pytorch_salt_mha(n_dim, n_head)

    torch_out = t_net(x, x, x, key_padding_mask=mask)[0]
    out = s_net(x, x, x, kv_mask=mask)

    t_time = timeit.timeit(lambda: t_net(x, x, x, key_padding_mask=mask), number=100)
    s_time = timeit.timeit(lambda: s_net(x, x, x, kv_mask=mask), number=100)

    # ensure the salt mha implementation is not slower than the referenece pytorch
    # allow for some tolerance to avoid false positives
    assert t_time * 1.25 > s_time, (
        f"salt ({s_time:.2f}s) was significantly slower than pytorch {t_time:.2f}s! Check for"
        " regressions."
    )
    np.testing.assert_allclose(
        torch_out.cpu().detach().numpy(), out.cpu().detach().numpy(), rtol=1e-3, atol=1e-3
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
