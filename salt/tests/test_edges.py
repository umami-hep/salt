import pytest
import torch

from salt.models.attention import EdgeAttention
from salt.models.transformer import EncoderLayer, NormResidual, Transformer
from salt.utils.inputs import get_random_mask


def get_inputs(batch_size, seq_len, dim, edge_dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, dim)
    mask = torch.rand(batch_size, seq_len) > frac_pad
    edges = torch.randn(batch_size, seq_len, seq_len, edge_dim)
    mask[:, 0] = False  # Make sure something can send
    return x, mask, edges


@pytest.mark.parametrize("seq_len", [1, 10])
@pytest.mark.parametrize("num_heads", [1, 2])
def test_edge_attention(
    seq_len,
    num_heads,
) -> None:
    batch_size = 2
    embed_dim = 32
    edge_dim = 33
    attn = EdgeAttention(
        embed_dim=embed_dim,
        edge_embed_dim=edge_dim,
        num_heads=num_heads,
        dropout=0.1,
        do_qk_norm=True,
        do_v_norm=True,
        update_edges=False,
    )

    x, mask, edge_x = get_inputs(batch_size, seq_len, embed_dim, edge_dim, 0.5)
    output, edge_output = attn(x, edge_x=edge_x, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert edge_output.shape == (batch_size, seq_len, seq_len, edge_dim)
    assert not torch.isnan(output).any()
    assert not torch.isnan(edge_output).any()
    assert torch.all(edge_x == edge_output)


@pytest.mark.parametrize("norm_type", ["pre", "post", "none"])
def test_norm_residual_edge_attention(norm_type) -> None:
    batch_size = 2
    seq_len = 5
    embed_dim = 32
    edge_dim = 33
    attn = EdgeAttention(
        embed_dim=embed_dim,
        edge_embed_dim=edge_dim,
        num_heads=2,
        dropout=0.1,
        do_qk_norm=True,
        do_v_norm=True,
        update_edges=False,
    )
    nr = NormResidual(attn, ls_init=0.1, embed_dim=embed_dim, norm_type=norm_type)

    x, mask, edge_x = get_inputs(batch_size, seq_len, embed_dim, edge_dim, 0.5)
    output, edge_output = nr(x, edge_x=edge_x, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert edge_output.shape == (batch_size, seq_len, seq_len, edge_dim)
    assert not torch.isnan(output).any()
    assert not torch.isnan(edge_output).any()
    assert torch.all(edge_x == edge_output)


def test_edge_attention_update_edges() -> None:
    batch_size, seq_len, edge_dim = 2, 5, 33
    embed_dim = 32
    attn = EdgeAttention(
        embed_dim=embed_dim,
        edge_embed_dim=edge_dim,
        num_heads=2,
        dropout=0.1,
        do_qk_norm=True,
        do_v_norm=True,
        update_edges=True,
    )
    x, mask, edge_x = get_inputs(batch_size, seq_len, embed_dim, edge_dim, 0.5)
    output, edge_output = attn(x, edge_x=edge_x, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert edge_output.shape == (batch_size, seq_len, seq_len, edge_dim)
    assert not torch.isnan(output).any()
    assert not torch.isnan(edge_output).any()
    assert ~torch.all(edge_x == edge_output)


def test_edge_attention_mup() -> None:
    batch_size, seq_len, embed_dim, edge_dim = 2, 5, 16, 8
    num_heads = 2
    attn = EdgeAttention(
        embed_dim=embed_dim,
        edge_embed_dim=edge_dim,
        num_heads=num_heads,
        update_edges=True,
        mup=True,
    )

    x, mask, edge_x = get_inputs(batch_size, seq_len, embed_dim, edge_dim, 0.5)
    output, edge_output = attn(x, edge_x=edge_x, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert edge_output.shape == (batch_size, seq_len, seq_len, edge_dim)
    assert not torch.isnan(output).any()
    assert not torch.isnan(edge_output).any()


@pytest.mark.parametrize("drop_registers", [True, False], ids=["DropRegisters", "KeepRegisters"])
def test_transformer_edges(drop_registers) -> None:
    bs, seq_len, embed_dim, edge_embed_dim = 2, 5, 10, 16
    num_registers = 3
    x = torch.rand(bs, seq_len, embed_dim)
    edge_x = torch.rand(bs, seq_len, seq_len, edge_embed_dim)

    x_zero_track = torch.rand(bs, 0, embed_dim)
    edge_x_zero_track = torch.rand(bs, 0, 0, edge_embed_dim)

    net = Transformer(
        embed_dim=embed_dim,
        edge_embed_dim=edge_embed_dim,
        num_layers=2,
        attn_kwargs={
            "num_heads": 2,
        },
        dense_kwargs={
            "activation": "ReLU",
        },
        update_edges=True,
        num_registers=num_registers,
        drop_registers=drop_registers,
    )

    # basic test
    out = net(x, edge_x=edge_x, pad_mask=get_random_mask(bs, seq_len, p_valid=0.5))[0]
    assert out.shape == (
        bs,
        seq_len if drop_registers else seq_len + num_registers,
        embed_dim,
    ), "Transformer with edges forward produced incorrect output shape"
    assert not torch.isnan(out).any(), "Transformer with edges forward produced NaNs"

    # test zero track case
    out_zero_track = net(
        x_zero_track, edge_x=edge_x_zero_track, pad_mask=get_random_mask(bs, 0, p_valid=0.5)
    )[0]
    assert out_zero_track.shape == (
        bs,
        0 if drop_registers else num_registers,
        embed_dim,
    ), "Transformer with edges zero-track forward produced incorrect output shape"
    assert not torch.isnan(out_zero_track).any(), (
        "Transformer with edges zero-track forward produced NaNs"
    )

    # test fully padded case
    out, _ = net(x, edge_x=edge_x, pad_mask=get_random_mask(bs, seq_len, p_valid=0.0))
    assert not torch.isnan(out).any(), "Transformer with edges forward produced NaNs"


def test_edge_updates():
    n_batch = 2
    n_trk = 3
    n_head = 1
    n_dim = 5

    x = torch.rand((n_batch, n_trk, n_dim))
    edges = torch.rand((n_batch, n_trk, n_trk, n_dim))

    # case where edges are updated
    net = EncoderLayer(
        embed_dim=n_dim,
        attn_kwargs={
            "num_heads": n_head,
        },
        edge_embed_dim=n_dim,
        update_edges=True,
    )

    _, edges_out = net(x, edge_x=edges)
    assert ~torch.all(edges == edges_out)

    # case where edges are not updated
    net = EncoderLayer(
        embed_dim=n_dim,
        attn_kwargs={
            "num_heads": n_head,
        },
        edge_embed_dim=n_dim,
        update_edges=False,
    )

    _, edges_out = net(x, edge_x=edges)
    assert torch.all(edges == edges_out)
