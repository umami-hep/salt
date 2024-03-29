import time

import pytest
import torch
from torch import nn

from salt.models.attention import MultiheadAttention
from salt.models.layernorm import RMSNorm
from salt.models.transformer import TransformerEncoderLayer
from salt.models.transformer_v2 import Attention, DecoderLayer, EncoderLayer, merge_masks

N_BATCH = 10
Q_SEQ = 20
KV_SEQ = 10
DIM = 16


def create_bool_tensor(shape, value):
    return torch.full(shape, value, dtype=torch.bool)


def test_merge_masks_none_inputs():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    mask = merge_masks(None, None, None, q_shape, k_shape)
    assert mask is None


def test_merge_masks_only_q_mask():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    q_mask = create_bool_tensor(q_shape[:-1], False)
    mask = merge_masks(q_mask, None, None, q_shape, k_shape)
    assert mask.shape == (N_BATCH, Q_SEQ, KV_SEQ)


def test_merge_masks_only_kv_mask():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    kv_mask = create_bool_tensor(k_shape[:-1], False)
    mask = merge_masks(None, kv_mask, None, q_shape, k_shape)
    assert mask.shape == (N_BATCH, Q_SEQ, KV_SEQ)


def test_merge_masks_q_and_kv_masks():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    q_mask = create_bool_tensor(q_shape[:-1], False)
    kv_mask = create_bool_tensor(k_shape[:-1], True)
    mask = merge_masks(q_mask, kv_mask, None, q_shape, k_shape)
    assert mask.shape == (N_BATCH, Q_SEQ, KV_SEQ)
    assert torch.all(mask)


def test_merge_masks_with_attn_mask():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    attn_mask = create_bool_tensor((3, 4, 5), False)
    mask = merge_masks(None, None, attn_mask, q_shape, k_shape)
    assert mask.shape == attn_mask.shape
    assert torch.equal(mask, attn_mask)


def test_merge_masks_different_shapes():
    q_shape = (2, 3, 10)
    k_shape = (2, 4, 10)
    q_mask = create_bool_tensor(q_shape[:-1], False)
    kv_mask = create_bool_tensor(k_shape[:-1], False)
    attn_mask = create_bool_tensor((2, 3, 4), False)
    mask = merge_masks(q_mask, kv_mask, attn_mask, q_shape, k_shape)
    assert mask.shape == attn_mask.shape


def test_padding_mask():
    torch_attn = nn.MultiheadAttention(8, 1, batch_first=True)

    # this is a correct full attention mask for padded inputs
    # we only need to pad the keys so that no query receives information from them
    # if we were to also pad the queries, the padded tokens would become nan as
    # they would not recieve any incomming signal
    attn_mask = torch.tensor([[[False, False, True], [False, False, True], [False, False, True]]])

    # confirm that the value of the padded tokens has no effect on the output
    x = torch.ones(1, 3, 8)
    x[:, 2] = 0
    out1 = torch_attn(x, x, x, attn_mask=attn_mask)[0]
    x[:, 2] = 10
    out2 = torch_attn(x, x, x, attn_mask=attn_mask)[0]
    torch.testing.assert_close(out1, out2)

    # this kind of mask is overkill and leads to nans
    # attn_mask = torch.tensor([[
    #   [False, False,  True],
    #   [False, False,  True],
    #   [ True,  True,  True]
    # ]])


def compare_attention_outputs(custom_attn, torch_attn, q, k, v, kv_mask=None):
    """Helper function to compare outputs of custom and torch attention modules."""
    custom_output = custom_attn(q, k, v, kv_mask=kv_mask)
    torch_output, _ = torch_attn(q, k, v, key_padding_mask=kv_mask)
    torch.testing.assert_close(custom_output, torch_output)
    assert not torch.isnan(custom_output).any()


def get_models(dim, num_heads, add_zero_attn):
    salt_attn = Attention(dim, num_heads=num_heads, add_zero_attn=add_zero_attn)
    torch_attn = nn.MultiheadAttention(
        dim, num_heads, batch_first=True, add_zero_attn=add_zero_attn
    )

    # Set the weights of the custom attention module to be the same as the torch module
    weights = torch.rand((3 * dim, dim))
    bias = torch.rand(3 * dim)
    torch_attn.in_proj_weight = nn.Parameter(weights)
    torch_attn.in_proj_bias = nn.Parameter(bias)

    wq, wk, wv = weights.chunk(3)
    bq, bk, bv = bias.chunk(3)
    salt_attn.wq.weight = nn.Parameter(wq)
    salt_attn.wk.weight = nn.Parameter(wk)
    salt_attn.wv.weight = nn.Parameter(wv)
    salt_attn.wq.bias = nn.Parameter(bq)
    salt_attn.wk.bias = nn.Parameter(bk)
    salt_attn.wv.bias = nn.Parameter(bv)
    salt_attn.wo.weight = torch_attn.out_proj.weight
    salt_attn.wo.bias = torch_attn.out_proj.bias
    return salt_attn, torch_attn


def get_test_inputs(batch_size, seq_len, dim, frac_pad=0.0):
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)
    kv_mask = torch.rand(batch_size, seq_len) < frac_pad
    q[kv_mask] = 0
    k[kv_mask] = 0
    v[kv_mask] = 0
    return q, k, v, kv_mask


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("seq_len", [0, 1, 2, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("add_zero_attn", [False, True])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 1.0])
def test_attention_output(batch_size, seq_len, dim, num_heads, add_zero_attn, frac_pad):
    salt_attn, torch_attn = get_models(dim, num_heads, add_zero_attn=add_zero_attn)
    q, k, v, kv_mask = get_test_inputs(batch_size, seq_len, dim, frac_pad=frac_pad)

    # if not adding a dummy token to attend to, ensure at least one element is not masked
    if not add_zero_attn and kv_mask.shape[-1] != 0:
        kv_mask[..., 0] = False

    compare_attention_outputs(salt_attn, torch_attn, q, k, v, kv_mask)


@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5])
def test_v1_v2_attention_output(dim, num_heads, frac_pad):
    v1_attn = MultiheadAttention(
        dim, num_heads, {"class_path": "salt.models.ScaledDotProductAttention"}
    )
    v2_attn = Attention(dim, num_heads=num_heads, add_zero_attn=False)
    v1_attn.linear_q = v2_attn.wq
    v1_attn.linear_k = v2_attn.wk
    v1_attn.linear_v = v2_attn.wv
    v1_attn.linear_out = v2_attn.wo
    q, k, v, kv_mask = get_test_inputs(10, 20, dim, frac_pad=frac_pad)
    v1_out = v1_attn(q, k, v, kv_mask=kv_mask)
    v2_out = v2_attn(q, k, v, kv_mask=kv_mask)
    torch.testing.assert_close(v1_out, v2_out)


@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0])  # note that this fails for frac_pad > 0
def test_v1_v2_encoder_output(dim, num_heads, frac_pad):
    v1_enc = TransformerEncoderLayer(
        dim,
        {
            "num_heads": num_heads,
            "attention": {"class_path": "salt.models.ScaledDotProductAttention"},
        },
        {"activation": "ReLU"},
    )
    v2_enc = EncoderLayer(
        dim,
        attn_kwargs={"num_heads": num_heads, "add_zero_attn": False},
        dense_kwargs={"gated": False},
    )

    v1_enc.mha.linear_q = v2_enc.attn.attention.wq
    v1_enc.mha.linear_k = v2_enc.attn.attention.wk
    v1_enc.mha.linear_v = v2_enc.attn.attention.wv
    v1_enc.mha.linear_out = v2_enc.attn.attention.wo

    v1_enc.dense.net[0] = v2_enc.dense.in_proj
    v1_enc.dense.net[2] = v2_enc.dense.out_proj
    v1_enc.norm1 = v2_enc.attn_norm
    v1_enc.norm2 = v2_enc.dense_norm

    q, _, _, kv_mask = get_test_inputs(10, 20, dim, frac_pad=frac_pad)

    v1_out = v1_enc(q, pad_mask=kv_mask)
    v2_out = v2_enc(q, pad_mask=kv_mask)

    torch.testing.assert_close(v1_out, v2_out)


def test_times_torch_vs_salt():  # pragma: no cover
    # skip if cuda is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size, seq_len, dim, num_heads = 1000, 40, 128, 8
    salt_attn, torch_attn = get_models(dim, num_heads, add_zero_attn=True)
    q, k, v, kv_mask = get_test_inputs(batch_size, seq_len, dim, frac_pad=0.5)

    # move tensors and models to cuda
    q, k, v, kv_mask = q.cuda(), k.cuda(), v.cuda(), kv_mask.cuda()
    salt_attn.cuda()
    torch_attn.cuda()

    # avoid torch fast path
    salt_attn.training = True
    torch_attn.training = True

    # warm up
    for _ in range(10):
        salt_attn(q, k, v, kv_mask=kv_mask)
        torch_attn(q, k, v, key_padding_mask=kv_mask)

    salt_times = []
    for _ in range(50):
        start = time.time()
        salt_attn(q, k, v, kv_mask=kv_mask)
        end = time.time()
        salt_times.append(end - start)

    torch_times = []
    for _ in range(50):
        start = time.time()
        torch_attn(q, k, v, key_padding_mask=kv_mask)
        end = time.time()
        torch_times.append(end - start)

    salt_mean = sum(salt_times) / len(salt_times)
    torch_mean = sum(torch_times) / len(torch_times)
    salt_median = sorted(salt_times)[len(salt_times) // 2]
    torch_median = sorted(torch_times)[len(torch_times) // 2]

    assert salt_mean < torch_mean, f"mean: {salt_mean} vs {torch_mean}"
    assert salt_median < torch_median, f"median: {salt_median} vs {torch_median}"


def test_RMSNorm():
    rmsnorm = RMSNorm(10)
    x = torch.randn(5, 10)
    rmsnorm(x)


def test_DecoderLayer():
    layer = DecoderLayer(embed_dim=32, attn_kwargs={"num_heads": 2})
    x = torch.randn(5, 10, 32)
    y = torch.randn(5, 10, 32)
    layer(x, y, pad_mask=None)
