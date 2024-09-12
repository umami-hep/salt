import importlib.util

import pytest
import torch
from torch import nn
from torch.utils.benchmark import Timer

from salt.models.attention import MultiheadAttention
from salt.models.layernorm import RMSNorm
from salt.models.transformer_v2 import (
    Attention,
    DecoderLayer,
    TransformerV2,
    change_attn_backends,
    merge_masks,
    redo_padding,
    undo_padding,
)

N_BATCH = 10
Q_SEQ = 20
KV_SEQ = 10
DIM = 16


def create_bool_tensor(shape, value):
    return torch.full(shape, value, dtype=torch.bool)


def test_merge_masks_none_inputs():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    mask = merge_masks(None, None, q_shape)
    assert mask is None


def test_merge_masks_only_attn_mask():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    attn_shape = (N_BATCH, Q_SEQ, KV_SEQ)
    attn_mask = create_bool_tensor(attn_shape, False)
    mask = merge_masks(None, attn_mask, q_shape)
    assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)


def test_merge_masks_only_kv_mask():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    kv_mask = create_bool_tensor(k_shape[:-1], False)
    mask = merge_masks(kv_mask, None, q_shape)
    assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)


def test_merge_masks_attn_and_kv_masks():
    q_shape = (N_BATCH, Q_SEQ, DIM)
    k_shape = (N_BATCH, KV_SEQ, DIM)
    attn_shape = (N_BATCH, Q_SEQ, KV_SEQ)
    kv_mask = create_bool_tensor(k_shape[:-1], False)
    attn_mask = create_bool_tensor(attn_shape, True)
    mask = merge_masks(kv_mask, attn_mask, q_shape)
    assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)
    assert torch.all(mask)


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


def get_models(dim, num_heads) -> tuple:
    salt_attn = Attention(dim, num_heads=num_heads)
    torch_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    salt_attn.in_proj_weight = torch_attn.in_proj_weight
    salt_attn.in_proj_bias = torch_attn.in_proj_bias
    salt_attn.out_proj.weight = torch_attn.out_proj.weight
    salt_attn.out_proj.bias = torch_attn.out_proj.bias
    return salt_attn, torch_attn


def get_cross_attn_inputs(batch_size, q_len, kv_len, dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    q = torch.randn(batch_size, q_len, dim)
    kv = torch.randn(batch_size, kv_len, dim)
    kv_mask = torch.rand(batch_size, kv_len) > frac_pad
    kv_mask[:, 0] = False  # Make sure something can send
    return q, kv, kv_mask


def get_self_attn_inputs(batch_size, seq_len, dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, dim)
    mask = torch.rand(batch_size, seq_len) > frac_pad
    mask[:, 0] = False  # Make sure something can send
    return x, mask


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("q_len", [1, 10])
@pytest.mark.parametrize("kv_len", [1, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 0.9])
def test_cross_attention(
    batch_size,
    q_len,
    kv_len,
    dim,
    frac_pad,
) -> None:
    salt_attn, torch_attn = get_models(dim, 2)
    q, kv, kv_mask = get_cross_attn_inputs(batch_size, q_len, kv_len, dim, frac_pad)
    custom_output = salt_attn(q, kv, kv_mask=kv_mask)
    torch_output, _ = torch_attn(q, kv, kv, key_padding_mask=kv_mask)
    torch.testing.assert_close(custom_output, torch_output)
    assert not torch.isnan(custom_output).any()


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("seq_len", [1, 2, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 0.9])
def test_self_attention(
    batch_size,
    seq_len,
    dim,
    num_heads,
    frac_pad,
) -> None:
    salt_attn, torch_attn = get_models(dim, num_heads)
    x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad)
    custom_output = salt_attn(x, mask=mask)
    torch_output, _ = torch_attn(x, x, x, key_padding_mask=mask)
    torch.testing.assert_close(custom_output, torch_output)
    assert not torch.isnan(custom_output).any()


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("seq_len", [1, 2, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5])
@pytest.mark.parametrize("attn_type", ["torch-flash", "torch-meff", "flash-varlen"])
def test_attention_backends(
    batch_size,
    seq_len,
    dim,
    num_heads,
    frac_pad,
    attn_type,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn not available")

    # FlashVarlenAttention requires half precision
    with torch.autocast("cuda", enabled=True):
        # Get the inputs and move to device
        x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad)
        x = x.cuda()
        mask = mask.cuda()

        # Change the masking to None for the torch backends as they dont support it
        if "torch" in attn_type:
            mask = None

        # Perform the standard attention (math)
        attn = Attention(dim, num_heads=num_heads).to("cuda")
        output = attn(x, mask=mask)

        # ensure zero padded
        if mask is not None:
            output *= ~mask.unsqueeze(-1)

        # Switch to the attention backend
        attn.set_backend(attn_type)
        if attn_type == "flash-varlen":
            x_p, culens, maxlen = undo_padding(x, mask)
            output_2 = attn(x_p, mask=mask, culens=culens, maxlen=maxlen)
            output_2 = redo_padding(output_2, mask)
        else:
            output_2 = attn(x, mask=mask)

        # Test all close with less strict due to half precision
        torch.testing.assert_close(output, output_2, atol=1e-3, rtol=1e-3)
        assert not torch.isnan(output_2).any()


def sync_v1v2_attn(v1_attn, v2_attn):
    wq, wk, wv = v2_attn.in_proj_weight.chunk(3)
    bq, bk, bv = v2_attn.in_proj_bias.chunk(3)
    v1_attn.linear_q.weight.data = wq
    v1_attn.linear_k.weight.data = wk
    v1_attn.linear_v.weight.data = wv
    v1_attn.linear_q.bias.data = bq
    v1_attn.linear_k.bias.data = bk
    v1_attn.linear_v.bias.data = bv


@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 0.9])
def test_v1_v2_attention_output(dim, num_heads, frac_pad):
    v1_attn = MultiheadAttention(
        dim, num_heads, {"class_path": "salt.models.ScaledDotProductAttention"}
    )
    v2_attn = Attention(dim, num_heads=num_heads)
    sync_v1v2_attn(v1_attn, v2_attn)
    v1_attn.linear_out = v2_attn.out_proj
    q, kv, kv_mask = get_cross_attn_inputs(10, 20, 20, dim, frac_pad=frac_pad)
    v1_out = v1_attn(q, kv, kv_mask=kv_mask)
    v2_out = v2_attn(q, kv, kv_mask=kv_mask)
    torch.testing.assert_close(v1_out, v2_out)


@pytest.mark.parametrize("num_registers", [1, 4])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("ls_init", [None, 0.1])
@pytest.mark.parametrize("drop_path", [0, 0.1])
def test_transformerv2_tensor_input(num_registers, num_layers, ls_init, drop_path):
    x, mask = get_self_attn_inputs(5, 10, 32, 0.5)
    trans = TransformerV2(
        num_layers=num_layers,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        num_registers=num_registers,
        ls_init=ls_init,
        drop_path=drop_path,
    )
    x, mask = trans(x, pad_mask=mask)
    assert x.shape == (5, 10 + num_registers, 32)
    assert not x.isnan().any()


@pytest.mark.parametrize("ls_init", [None, 0.1])
@pytest.mark.parametrize("drop_path", [0, 0.1])
def test_decoder_layer(ls_init, drop_path):
    q, kv, kv_mask = get_cross_attn_inputs(5, 10, 5, 32, 0.5)
    decoder = DecoderLayer(
        embed_dim=32,
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        ls_init=ls_init,
        drop_path=drop_path,
    )
    x = decoder(q, kv=kv, kv_mask=kv_mask)
    assert x.shape == q.shape
    assert not x.isnan().any()


@pytest.mark.parametrize("num_registers", [1, 4])
def test_transformerv2_dict_input(num_registers):
    x1, m1 = get_self_attn_inputs(5, 10, 32, 0.5)
    x2, m2 = get_self_attn_inputs(5, 3, 32, 0.5)
    x3, m3 = get_self_attn_inputs(5, 2, 32, 0.5)
    x = {"m1": x1, "m2": x2, "m3": x3}  # Multimodal inputs
    mask = {"m1": m1, "m2": m2, "m3": m3}
    trans = TransformerV2(
        num_layers=3,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        num_registers=num_registers,
    )
    x, mask = trans(x, pad_mask=mask)
    assert x.shape == (5, 10 + 3 + 2 + num_registers, 32)
    assert all(k in mask for k in ["m1", "m2", "m3", "REGISTERS"])


def test_times_torch_vs_salt() -> None:
    # skip if cuda is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Define the input parameters for the timings
    batch_size, seq_len, dim, num_heads = 1000, 64, 128, 8
    salt_attn, torch_attn = get_models(dim, num_heads)
    x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad=0.5)

    # move tensors and models to cuda
    x = x.cuda()
    mask = mask.cuda()
    salt_attn.cuda()
    torch_attn.cuda()

    # avoid torch fast path
    salt_attn.training = True
    torch_attn.training = True

    # Using timers also performs warm up
    salt_timer = Timer(
        stmt="salt_attn(x, kv_mask=mask)",
        globals={"salt_attn": salt_attn, "x": x, "mask": mask},
        label="salt",
        num_threads=1,
    )

    torch_timer = Timer(
        stmt="torch_attn(x, x, x, key_padding_mask=mask)",
        globals={"torch_attn": torch_attn, "x": x, "mask": mask},
        label="torch",
        num_threads=1,
    )

    salt_time = salt_timer.timeit(300).mean
    torch_time = torch_timer.timeit(300).mean
    assert salt_time < torch_time, f"mean: {salt_time} vs {torch_time}"


def test_times_varlen_vs_default() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn not available")

    # FlashVarlenAttention requires half precision
    with torch.autocast("cuda", enabled=True):
        # Define the input parameters for the timings
        num_layers = 4
        num_heads = 4
        batch_size = 256
        seq_len = 64
        dim = 128
        x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad=0.5)

        # Create the transformers
        standard_attn = TransformerV2(
            num_layers=num_layers,
            embed_dim=dim,
            attn_type="torch-math",
            dense_kwargs={"activation": "SiLU"},
            attn_kwargs={"num_heads": num_heads},
        )

        varlen_attn = TransformerV2(
            num_layers=num_layers,
            embed_dim=dim,
            attn_type="flash-varlen",
            dense_kwargs={"activation": "SiLU"},
            attn_kwargs={"num_heads": num_heads},
        )

        # move tensors and models to cuda
        x = x.cuda()
        mask = mask.cuda()
        standard_attn.cuda()
        varlen_attn.cuda()

        # Time the models
        s_timer = Timer(
            stmt="standard_attn(x, pad_mask=mask)",
            globals={"standard_attn": standard_attn, "x": x, "mask": mask},
            label="salt",
            num_threads=1,
        )
        v_timer = Timer(
            stmt="varlen_attn(x, pad_mask=mask)",
            globals={"varlen_attn": varlen_attn, "x": x, "mask": mask},
            label="salt",
            num_threads=1,
        )
        st = s_timer.timeit(20).mean
        vt = v_timer.timeit(20).mean
        assert vt < st, f"mean: {vt} vs {st}"


def test_RMSNorm():
    rmsnorm = RMSNorm(10)
    x = torch.randn(5, 10)
    rmsnorm(x)


def test_DecoderLayer():
    layer = DecoderLayer(embed_dim=32, attn_kwargs={"num_heads": 2})
    x = torch.randn(5, 10, 32)
    kv = torch.randn(5, 10, 32)
    layer(x, kv=kv)


def test_change_attn_backends():
    model = TransformerV2(
        num_layers=3,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
    )

    # change the backend
    change_attn_backends(model, "torch-meff")
    assert model.attn_type == "torch-meff"
    for layer in model.layers:
        assert layer.attn.fn.attn_type == "torch-meff"

    # no cuda, so it should not be able to set flash-varlen, and isntead fall back to torch-math
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            change_attn_backends(model, "flash-varlen")
        assert model.attn_type == "torch-math"
        for layer in model.layers:
            assert layer.attn.fn.attn_type == "torch-math"

    # check this works for a module that wraps a transformer
    wrapper = nn.Sequential(model)
    change_attn_backends(wrapper, "torch-flash")
    assert model.attn_type == "torch-flash"
    for layer in model.layers:
        assert layer.attn.fn.attn_type == "torch-flash"

    # check this works for a base attention layer
    attn = Attention(32, num_heads=2, attn_type="torch-math")
    change_attn_backends(attn, "torch-flash")
    assert attn.attn_type == "torch-flash"
