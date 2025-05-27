import torch

from salt.models import (
    MultiheadAttention,
    ScaledDotProductAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from salt.utils.inputs import get_random_mask


def test_transformer_edges() -> None:
    net = TransformerEncoder(
        embed_dim=10,
        edge_embed_dim=10,
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
    net(torch.rand(10, 10, 10), torch.rand(10, 10, 10, 10))

    # test zero track case
    assert torch.all(
        net(torch.rand(10, 0, 10), torch.rand(10, 0, 0, 10)) == torch.empty((10, 0, 10))
    )

    # test fully padded case
    out = net(
        torch.rand(1, 10, 10), torch.rand(1, 10, 10, 10), pad_mask=get_random_mask(1, 10, p_valid=0)
    )
    assert not torch.isnan(out).any()

    # test that adding a padded track does not change the output
    tracks = torch.rand(1, 10, 10)
    edges = torch.rand(1, 10, 10, 10)
    mask = get_random_mask(1, 10, p_valid=1)
    out = net(tracks, edges, pad_mask=mask)
    tracks = torch.cat([tracks, torch.zeros((1, 1, tracks.shape[2]))], dim=1)
    edges = torch.cat([edges, torch.zeros((1, 1, 10, edges.shape[3]))], dim=1)
    edges = torch.cat([edges, torch.zeros((1, 11, 1, edges.shape[3]))], dim=2)
    mask = torch.zeros(tracks.shape[:-1]).bool()
    mask[:, -1] = True
    out_with_pad = net(tracks, edges, pad_mask=mask)[:, :-1]
    tensor_check = out.data.half() == out_with_pad.data.half()
    assert torch.all(tensor_check)


def test_mha_edges_allvalid_mask() -> None:
    n_batch = 2
    n_trk = 3
    n_head = 1
    n_dim = 5

    net = MultiheadAttention(
        embed_dim=n_dim,
        num_heads=n_head,
        attention=ScaledDotProductAttention(),
        edge_embed_dim=n_dim,
    )

    q = k = v = torch.rand((n_batch, n_trk, n_dim))
    edges = torch.rand((n_batch, n_trk, n_trk, n_dim))
    valid_mask = torch.zeros((n_batch, n_trk), dtype=torch.bool)
    out_no_mask, _ = net(q, k, v, edges, q_mask=None, kv_mask=None)
    out_mask, _ = net(q, k, v, edges, q_mask=valid_mask, kv_mask=valid_mask)
    assert torch.all(out_no_mask == out_mask)


def test_mha_edges_qkv_different_dims():
    n_batch = 3
    n_trk = 5
    n_head = 2
    k_dim = 7
    q_dim = 8
    v_dim = 9
    edge_dim = 10

    net = MultiheadAttention(
        num_heads=n_head,
        embed_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        edge_embed_dim=edge_dim,
        attention=ScaledDotProductAttention(),
    )

    q = torch.rand((n_batch, n_trk, q_dim))
    k = torch.rand((n_batch, n_trk + 1, k_dim))
    v = torch.rand((n_batch, n_trk + 1, v_dim))
    edges = torch.rand((n_batch, n_trk, n_trk + 1, edge_dim))

    q_mask = get_random_mask(n_batch, n_trk)
    kv_mask = get_random_mask(n_batch, n_trk + 1)
    net(q, k, v, edges, q_mask=q_mask, kv_mask=kv_mask)


def test_edge_updates():
    n_batch = 2
    n_trk = 3
    n_head = 1
    n_dim = 5

    x = torch.rand((n_batch, n_trk, n_dim))
    edges = torch.rand((n_batch, n_trk, n_trk, n_dim))

    # case where edges are updated
    net = TransformerEncoderLayer(
        embed_dim=n_dim,
        mha_config={
            "num_heads": n_head,
            "attention": ScaledDotProductAttention(),
        },
        edge_embed_dim=n_dim,
        update_edges=True,
    )

    _, edges_out = net(x, edges)
    assert ~torch.all(edges == edges_out)

    # case where edges are not updated
    net = TransformerEncoderLayer(
        embed_dim=n_dim,
        mha_config={
            "num_heads": n_head,
            "attention": ScaledDotProductAttention(),
        },
        edge_embed_dim=n_dim,
        update_edges=False,
    )

    _, edges_out = net(x, edges)
    assert torch.all(edges == edges_out)
