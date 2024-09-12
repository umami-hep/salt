"""Efficient Transformer implementation.

Updated transformer implementation based on
https://github.com/mistralai/mistral-src

Features:
- native SDP kernels (including flash)
- gated linear units https://arxiv.org/abs/2002.05202
- RMSNorm https://arxiv.org/abs/1910.07467
"""

import warnings
from functools import partial

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn

import salt.models.layernorm as layernorms
from salt.stypes import Tensors
from salt.utils.tensor_utils import redo_padding, undo_padding

ATTN_TYPES = ["torch-math", "torch-flash", "torch-meff", "flash-varlen"]


def merge_masks(
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Tensor,
) -> BoolTensor | None:
    """Create a full attention mask which incorporates the padding information.

    Using pytorch transformer convention for padding
        False: Real node
        True:  Zero padded

    Using pytorch transformer convention for attention mask
        False:  Not allowed in attention mechanism
        True:   Allowed in attention mechanism

    Designing attention mask such that padded tokens can't send information.
    But they can receive them.
    This prevents Nans in the attention scores caused by the softmax

    Parameters
    ----------
    kv_mask : BoolTensor | None
        Mask for the keys and values, of shape (batch, kv_len).
    attn_mask : BoolTensor | None
        Full attention mask, of shape (batch, q_len, kv_len).
    q_shape : Size
        Shape of the queries tensor, (batch, q_len, dim).
    """
    # Create the full mask which combines the attention and padding masks
    mask = None

    # if the kv_mask mask exists, ensure that padded tokens never send information
    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # convert the mask such that True is a valid token

    # include the attention mask
    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask & mask

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if mask is not None:
        mask = mask.unsqueeze(1)

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def change_attn_backends(module: nn.Module, backend: str) -> None:
    """Recursively change the attention backend of a module and all its children.

    Used primarily for switching back to torch-math for ONNX exports.
    """
    if isinstance(module, TransformerV2):
        module.set_backend(backend)
        return
    if isinstance(module, Attention):
        module.set_backend(backend)
        return
    for child in module.children():
        change_attn_backends(child, backend)


def projection_packed(
    q: Tensor,
    kv: Tensor | None,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple:
    """Efficient input projection for MHA when using a single linear layer.

    Essentially the same as torch.nn.functional._in_projection_packed
    But here we use chunk which is 40x faster than unflatten
    Not sure why they don't use chunk in the original implementation...

    Parameters
    ----------
    q : Tensor
        The queries tensor of shape (batch, q_len, dim).
    kv : Tensor | None
        The keys and values tensor of shape (batch, kv_len, dim).
    weight : Tensor
        The packed weight tensor of the input lienar projection with shape (3 * dim, dim).
    bias : Tensor | None
        The optional packed bias tensor of the input linear projection with shape (3 * dim).

    Returns
    -------
    q_proj, k_proj, v_proj : tuple
        The projected queries, keys, and values tensors.
    """
    # If the q tensor is the only input, then we assume we are doing self-attention.
    # This is made (slightly) faster by using a single linear layer, then chunking rather than
    # three seperate linear layers processed one at a time.
    if kv is None:
        return F.linear(q, weight, bias).chunk(3, dim=-1)

    # If the kv tensor is present, then we are doing cross-attention.
    # This means we must project the q and kv tensors seperately.
    # The kv linear layer can remain packed, allowing us to project together then chunk,
    # using the same trick as above. We must however first seperate weights (and biases if present)
    # of the linear layers for the q and kv parts. We use torch.split which returns a veiw of the
    # original tensor so this step doesnt required any extra memory or much time.
    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    # Now we can do the seperate projections
    q_proj = F.linear(q, w_q, b_q)
    k_proj, v_proj = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj


def torch_attn(
    q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float, backend: str
) -> Tensor:
    """Torch dot product attention with a switchable backend."""
    with torch.backends.cuda.sdp_kernel(
        enable_math=True,  # always enabled as a fallback
        enable_mem_efficient=(backend == "torch-meff"),
        enable_flash=(backend == "torch-flash"),
    ):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_type: str = "torch-meff",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Multihead attention module.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input.
        num_heads : int
            Number of attention heads.
        attn_type : str, optional
            Name of backend kernel to use.
        dropout : float, optional
            Dropout rate.
        bias : bool, optional
            Whether to include bias terms.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        assert attn_type in ATTN_TYPES, "Invalid attention type!"

        # Attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.attn_type = attn_type

        # Better parallelism for self-attention when using parameters directly
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str):
        # Check the attention backend
        self.attn_type = attn_type
        if self.attn_type == "flash-varlen":
            why_not_flash = ""
            try:
                from flash_attn import flash_attn_varlen_qkvpacked_func

                self._flash_attn = flash_attn_varlen_qkvpacked_func
            except ImportError:
                why_not_flash = (
                    "Requires the flash_attn package, CUDA 12+, and A100+, and must be installed "
                    "separately. See salt/setup/install_flash.sh for installation instructions."
                )
            if not torch.cuda.is_available():
                why_not_flash = "No GPU available."
            if not next(self.parameters()).is_cuda:
                why_not_flash = "A GPU is available but not being used."
            if why_not_flash:
                warnings.warn(
                    f"Cannot use flash-varlen backend. {why_not_flash} Reverting to torch-math.",
                    stacklevel=2,
                )
                self.attn_type = "torch-math"
        return self.attn_type

    def reset_parameters(self):
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def _flash_forward(self, x: Tensor, culens: Tensor, maxlen: int) -> Tensor:
        """FlashAttention backend."""
        # Perform the packed input projection
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out = self._flash_attn(qkv, culens, maxlen, dropout)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Mix with final linear layer
        return self.out_proj(a_out)

    def _torch_forward(
        self, x: Tensor, kv: Tensor, mask: BoolTensor, kv_mask: BoolTensor, attn_mask: BoolTensor
    ) -> Tensor:
        """Attention using pytorch."""
        # Otherwise perform standard attention
        B, S, D = x.shape

        # input projections -> B, S, D
        q, k, v = projection_packed(x, kv, self.in_proj_weight, self.in_proj_bias)

        # transform tensors to (B, Nh, S, Hd)
        shape = (B, -1, self.num_heads, self.head_dim)  # Dont use S for cross attn
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        # run attention
        s_mask = mask if kv is None else kv_mask  # Who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        dropout = self.dropout if self.training else 0.0
        a_out = torch_attn(q, k, v, mask, dropout, self.attn_type)

        # recombine heads
        a_out = a_out.transpose(1, 2).contiguous().view(B, S, D)

        # mix with final linear layer
        return self.out_proj(a_out)

    def forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
        culens: Tensor | None = None,
        maxlen: int | None = None,
    ) -> Tensor:
        """Attention forward pass, dispatches to the appropriate backend.

        Parameters
        ----------
        x : Tensor
            The pointcloud of shape (batch, x_len, dim).
        kv : Tensor
            Optional second pointcloud for cross-attn with shape (batch, kv_len, dim).
        mask : BoolTensor, optional
            Mask for the pointcloud x, by default None.
        kv_mask : BoolTensor, optional
            Mask the kv pointcloud, by default None.
        attn_mask : BoolTensor, optional
            Full attention mask, by default None.
        culens : Tensor, optional
            Cumulative lengths of the sequences in x, by default None.
            Only used for the flash-varlen backend.
        maxlen : int, optional
            Maximum length of a sequence in the x, by default None.
            Only used for the flash-varlen backend.

        Returns
        -------
        Tensor
            Output of shape (batch, x_len, dim).
        """
        if self.attn_type == "flash-varlen":
            assert kv is None, "flash-varlen only supports self attention!"
            assert attn_mask is None, "flash-varlen does not support attention masks!"
            assert culens is not None, "flash-varlen requires culens!"
            assert maxlen is not None, "flash-varlen requires maxlen!"
            return self._flash_forward(x, culens, maxlen)

        # Otherwise perform standard attention
        return self._torch_forward(x, kv, mask, kv_mask, attn_mask)


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        """Dense update with gated linear unit.

        See [2002.05202](https://arxiv.org/abs/2002.05202).

        Parameters
        ----------
        embed_dim : int
            Dimension of the input and output.
        hidden_dim : int | None, optional
            Dimension of the hidden layer. If None, defaults to embed_dim * 2.
        activation : str, optional
            Activation function.
        dropout : float, optional
            Dropout rate.
        bias : bool, optional
            Whether to include bias in the linear layers.
        gated : bool, optional
            Whether to gate the output of the hidden layer.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


class LayerScale(nn.Module):
    """Applies the LayerScale operation from the Cait vision transformer.

    Effective at improving stability and speed of deep transformers.
    Now the standard for vision transformers
    https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Drop paths for a stochastic depth neural network.

    Used for regularisation when applied to the main path of a residual block.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class PreNormResidual(nn.Module):
    """Wraps a module with pre-norm with a residual connection.

    Optionally also applies:
    - LayerScale
    - DropPath (Stochastic Depth)

    Neat way of doing the most common transformer pattern:
    - x = x + drop(scale * fn(norm(x)))
    """

    def __init__(
        self,
        fn: nn.Module,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        embed_dim: int = 0,
    ) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        norm : str, optional
            The normalization method, by default "LayerNorm".
        ls_init : float | None, optional
            The initial value for the layerscale, by default 1e-3.
            If None, then no layerscale is applied.
        drop_path : float, optional
            The drop path rate, by default 0.0.
        embed_dim : int
            The dimension of the input and output.
            If zero we will try get it from the fn's own embed_dim attribute.
        """
        super().__init__()
        dim = embed_dim or fn.embed_dim
        assert dim > 0, "Could not determine embed_dim from fn"
        self.fn = fn
        self.norm = getattr(layernorms, norm)(dim)
        self.ls = LayerScale(dim, ls_init) if ls_init is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.drop_path(self.ls(self.fn(self.norm(x), *args, **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ) -> None:
        """Encoder layer consisting of a self-attention and a feed-forward layer.

        Parameters
        ----------
        embed_dim : int
            Dimension of the embeddings at each layer.
        norm : str, optional
            Normalization style, by default "LayerNorm".
        drop_path : float, optional
            Drop path rate, by default 0.0.
        ls_init : float | None, optional
            Initial value for the layerscale, by default 1e-3.
        dense_kwargs : dict | None, optional
            Keyword arguments for [salt.models.transformer_v2.GLU][salt.models.transformer_v2.GLU].
        attn_kwargs : dict | None, optional
            Keyword arguments for
            [salt.models.transformer_v2.Attention][salt.models.transformer_v2.Attention].
        """
        super().__init__()

        # Safe defaults
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        # Attributes
        self.embed_dim = embed_dim

        # Submodules
        residual = partial(PreNormResidual, norm=norm, ls_init=ls_init, drop_path=drop_path)
        self.attn = residual(Attention(embed_dim, **attn_kwargs))
        self.dense = residual(GLU(embed_dim, **dense_kwargs))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        ls_init: float | None = 1e-3,
        drop_path: float = 0.0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ):
        super().__init__()

        # Safe defaults
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        # Attributes
        self.embed_dim = embed_dim

        # Submodules
        residual = partial(PreNormResidual, norm=norm, ls_init=ls_init, drop_path=drop_path)
        self.self_attn = residual(Attention(embed_dim=embed_dim, **attn_kwargs))
        self.cross_attn = residual(Attention(embed_dim=embed_dim, **attn_kwargs))
        self.dense = residual(GLU(embed_dim, **dense_kwargs))

    def forward(
        self,
        x: Tensor,
        *,  # Indicates that kv is required
        kv: Tensor,
        mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.self_attn(x, kv_mask=mask)
        x = self.cross_attn(x, kv=kv, kv_mask=kv_mask)
        return self.dense(x)


class TransformerV2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        out_dim: int | None = None,
        norm: str = "LayerNorm",
        attn_type: str = "torch-math",
        do_final_norm: bool = True,
        num_registers: int = 1,
        drop_registers: bool = False,
        **kwargs,
    ) -> None:
        """Transformer model consisting of a stack of Transformer encoder layers.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        embed_dim : int
            Dimension of the embeddings at each layer.
        out_dim : int | None, optional
            Optionally project the output to a different dimension.
        norm : str, optional
            Normalization style, by default "LayerNorm".
        attn_type : str, optional
            The backend for the attention mechanism, by default "torch-flash".
            Provided here because the varlen backend requires pre/post processing.
        do_final_norm : bool, optional
            Whether to apply a final normalization layer, by default True.
        num_registers : int, optional
            The number of registers to add to the END of the input sequence.
            Registers are randomly initialised tokens of the same dimension as
            any other inputs after initialiser networks. See 2309.16588.
        drop_registers : bool, optional
            If to drop the registers from the outputs
        kwargs : dict
            Keyword arguments for [salt.models.transformer_v2.EncoderLayer].
        """
        super().__init__()

        # Check the inputs
        if num_registers < 1:
            raise ValueError(
                "Some jets have no tracks, which causes NaNs in the attention scores. ",
                "To avoid this, set num_registers to at least 1",
            )

        # Attributes
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.out_dim = out_dim or embed_dim
        self.do_final_norm = do_final_norm
        self.do_out_proj = out_dim is not None
        self.attn_type = attn_type
        self.num_registers = num_registers
        self.drop_registers = drop_registers

        # Submodules
        kwargs["attn_kwargs"]["attn_type"] = self.attn_type
        self.layers = torch.nn.ModuleList([
            EncoderLayer(embed_dim=embed_dim, norm=norm, **kwargs) for _ in range(num_layers)
        ])

        # Check and set the attention type
        assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
        self.set_backend(self.attn_type)

        # Optional submodules
        if self.do_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, out_dim)
        if self.do_final_norm:
            self.out_norm = getattr(layernorms, norm)(self.out_dim)
        if self.num_registers:
            self.registers = nn.Parameter(
                torch.normal(torch.zeros((self.num_registers, self.embed_dim)), std=1e-4)
            )
            self.register_buffer("register_mask", torch.zeros(num_registers, dtype=torch.bool))
        self.featurewise = nn.ModuleList()

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self,
        x: Tensor,
        pad_mask: BoolTensor,
        inputs: Tensors | None = None,
        **kwargs,
    ) -> Tensor:
        # Add the registers to the sequence and the mask
        if self.num_registers:
            x, pad_mask = self._add_registers(x, pad_mask)

        # Combine the input sequences if they are dictionaries (don't overwrite pad_mask)
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # If using the varlen backend, pack the sequence and store the cumulative lengths
        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        # Run through the main transformer encoder layers
        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            x = layer(x, mask=mask, **kwargs)

        # Run through the optional layers
        if self.do_out_proj:
            x = self.out_proj(x)
        if self.do_final_norm:
            x = self.out_norm(x)

        # If using the varlen backend, unpack the sequence
        if self.attn_type == "flash-varlen":
            x = redo_padding(x, mask)

        # Optionally drop the registers from the output
        if self.drop_registers:
            x = x[:, : -self.num_registers]
            if isinstance(pad_mask, dict):
                del pad_mask["REGISTERS"]
            elif isinstance(pad_mask, Tensor):
                pad_mask = pad_mask[:, : -self.num_registers]

        return x, pad_mask

    def _add_registers(self, x: Tensor | dict, pad_mask: BoolTensor | dict | None) -> tuple:
        """Add the learnable registers to the end of the input sequence."""
        # Get the batch size and expand the registers to match
        B = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)

        # Add as a key or concatenate at the end
        reg = self.registers.expand(B, -1, -1)
        if isinstance(x, dict):
            x["REGISTERS"] = reg
        else:
            x = torch.cat([x, reg], dim=1)

        # Also include a mask for the registers
        if pad_mask is not None:
            reg_mask = self.register_mask.expand(B, -1)
            if isinstance(pad_mask, dict):
                pad_mask["REGISTERS"] = reg_mask
            else:
                pad_mask = torch.cat([pad_mask, reg_mask], dim=-1)

        return x, pad_mask
