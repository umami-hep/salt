"""Config-constructed GraphModules for the salt graph execution model
(encoder/pooling/loss pipeline), plus absorbed v1 Dense/Transformer/pooling
layers. Key vocabulary: ``inputs.<stream>`` -> ``normed.<stream>`` ->
``embed.<stream>`` -> ``seq.x``/``seq.mask`` -> ``encoded.seq`` ->
``encoded.<stream>`` / ``pooled.global``; task modules live in
`salt.core.nn.tasks`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, final

import torch
import yaml
from torch import BoolTensor, Size, Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import pad, softmax

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    KEY_SEP,
    GraphModule,
    Mode,
    TensorSpec,
    flatten_spec,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn.bind import ResolvedSchema

# Absorbed v1 Dense/Transformer/pooling layer family — copied here as
# self-contained classes so the v1 originals in salt/models/* stay untouched.

__all__ = [
    "Concat",
    "Dense",
    "EdgeEmbed",
    "EdgeFeatures",
    "FeaturewiseTransformation",
    "GlobalAttentionPooling",
    "LossGLS",
    "LossSum",
    "Normaliser",
    "PositionalEncoder",
    "Split",
    "StreamEmbed",
    "Transformer",
    "TransformerEncoder",
    "VectorConcat",
]

_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned before compile."""

_SEQ_LEN = sym_dim("S", "seq")
_ENC_LEN = sym_dim("L", "enc")

_EDGE_FEATURES = ("dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass")
"""Recognised edge-feature names. EdgeFeatures rejects anything outside this
set at config time."""


# ---------------------------------------------------------------------------
# inlined v1 math helpers (byte-faithful copies; v1 originals untouched)
# ---------------------------------------------------------------------------


def add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dimensions (after the batch dim) to reach a target rank.

    Raises
    ------
    ValueError
        If ``ndim`` is smaller than ``x.ndim``.
    """
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is smaller than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x


def attach_context_single(x: Tensor, context: Tensor) -> Tensor:
    """Broadcast ``context`` to match ``x`` and concatenate along the last dim.

    Returns ``cat([context, x], dim=-1)`` — context is PREPENDED.

    Raises
    ------
    RuntimeError
        If ``context`` is ``None``.
    ValueError
        If the provided context has more dimensions than the input.
    """
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    if dim_diff > 0:
        context = add_dims(context, x.dim())
        context = context.expand(*x.shape[:-1], -1)

    return torch.cat([context, x], dim=-1)


def attach_context(x: Tensor | dict[str, Tensor], context: Tensor) -> Tensor | dict[str, Tensor]:
    """Concatenate a context tensor to inputs (tensor or dict of tensors)."""
    if isinstance(x, dict):
        return {key: attach_context_single(val, context) for key, val in x.items()}
    return attach_context_single(x, context)


def check_edge_config(
    edge_features: list[str],
    available_vars: list[str],
) -> None:
    """Check the requested edge features are recognized and have the required input variables.

    Raises
    ------
    ValueError
        If an edge feature is not recognized or if required indices are missing.
    """
    req_vars: list[str] = []
    for variable in edge_features:
        if variable == "dR":
            req_vars.extend(["eta", "phi"])
        elif variable == "z":
            req_vars.extend(["pt"])
        elif variable == "kt":
            req_vars.extend(["eta", "phi", "pt"])
        elif variable == "isSelfLoop":
            continue
        elif variable == "subjetIndex":
            req_vars.extend(["subjetIndex"])
        elif variable == "mass":
            req_vars.extend(["pt", "eta", "phi", "energy"])
        else:
            raise ValueError(f"Edge feature {variable} not recognized")

    missing = set(req_vars) - set(available_vars)
    if missing:
        raise ValueError(
            f"Indices of {missing} required for edge features calculation were not specified."
        )


def calculate_edge_features(
    batch: Tensor,
    indices_map: dict[str, int],
    variables: list[str],
) -> Tensor:
    """Compute pairwise dR/kt/z/subjetIndex/isSelfLoop/mass edge features.

    Returns
    -------
    Tensor
        Edge features of shape ``[B, N, N, num_edge_features]``.
    """
    ebatch = torch.zeros(
        (batch.shape[0], batch.shape[1], batch.shape[1], len(variables)),
        dtype=batch.dtype,
        device=batch.device,
    )

    # intermediate quantities
    if "dR" in variables or "kt" in variables:
        dphi = batch[:, :, indices_map["phi"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["phi"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
        dphi -= (dphi > math.pi).type_as(dphi) * 2 * math.pi
        deta = batch[:, :, indices_map["eta"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["eta"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
    if "kt" in variables or "z" in variables:
        pt_min = torch.minimum(
            batch[:, :, indices_map["pt"]].unsqueeze(1).expand(-1, batch.shape[1], -1),
            batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1]),
        )
    if "mass" in variables:
        pt = batch[:, :, indices_map["pt"]]
        eta = batch[:, :, indices_map["eta"]]
        phi = batch[:, :, indices_map["phi"]]
        energy = batch[:, :, indices_map["energy"]]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * (torch.exp(eta) - torch.exp(-eta)) / 2

    # fill edge features
    for i, variable in enumerate(variables):
        if variable == "dR":
            ebatch[:, :, :, i] = torch.log(torch.sqrt(torch.square(deta) + torch.square(dphi)))
        elif variable == "kt":
            ebatch[:, :, :, i] = torch.log(
                pt_min * torch.sqrt(torch.square(deta) + torch.square(dphi))
            )
        elif variable == "z":
            pt_sum = batch[:, :, indices_map["pt"]].unsqueeze(1).expand(
                -1, batch.shape[1], -1
            ) + batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            ebatch[:, :, :, i] = torch.log(pt_min / pt_sum)
        elif variable == "isSelfLoop":
            ebatch[:, :, :, i] = (
                torch.eye(batch.shape[1], dtype=ebatch.dtype, device=batch.device)
                .unsqueeze(0)
                .expand(batch.shape[0], -1, -1)
            )
        elif variable == "subjetIndex":
            sji1 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(1).expand(-1, batch.shape[1], -1)
            )
            sji2 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            )
            ebatch[:, :, :, i] = torch.logical_and(torch.eq(sji1, sji2), sji1 >= 0)
        elif variable == "mass":
            e1 = energy.unsqueeze(1).expand(-1, batch.shape[1], -1)
            e2 = energy.unsqueeze(2).expand(-1, -1, batch.shape[1])
            px1 = px.unsqueeze(1).expand(-1, batch.shape[1], -1)
            px2 = px.unsqueeze(2).expand(-1, -1, batch.shape[1])
            py1 = py.unsqueeze(1).expand(-1, batch.shape[1], -1)
            py2 = py.unsqueeze(2).expand(-1, -1, batch.shape[1])
            pz1 = pz.unsqueeze(1).expand(-1, batch.shape[1], -1)
            pz2 = pz.unsqueeze(2).expand(-1, -1, batch.shape[1])
            e_sum = e1 + e2
            px_sum = px1 + px2
            py_sum = py1 + py2
            pz_sum = pz1 + pz2
            mass2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
            mass2 = torch.clamp_min(mass2, 1e-8)
            ebatch[:, :, :, i] = 0.5 * torch.log(mass2)

    return torch.nan_to_num(ebatch, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_tensor_dict(
    x: dict[str, Tensor],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Tensor:
    """Concatenate a dict of tensors along ``dim=1`` (``include``/``exclude`` are exclusive).

    Raises
    ------
    ValueError
        If both ``include`` and ``exclude`` are provided.
    """
    if include and exclude:
        raise ValueError("Cannot use 'include' and 'exclude' together")
    if include:
        return torch.cat([x[emb] for emb in include], dim=1)
    if exclude:
        return torch.cat([x[emb] for emb in x if emb not in exclude], dim=1)
    return torch.cat(list(x.values()), dim=1)


def masked_softmax(x: Tensor, mask: BoolTensor | None, dim: int = -1) -> Tensor:
    """Softmax that ignores padded elements: masked (``True``) entries are set to
    ``-inf`` before the softmax and zeroed after."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def undo_padding(seq: Tensor, mask: BoolTensor) -> tuple[Tensor, Tensor, int]:
    """Remove padded elements; return the packed sequence + flash-varlen metadata.

    ``mask == True`` means padded (the mask is flipped internally).

    Returns
    -------
    tuple[Tensor, Tensor, int]
        The packed (unpadded) sequence, the cumulative lengths (``int32``), and
        the maximum valid sequence length.
    """
    mask = ~mask  # convert mask: True -> valid token
    seqlens = mask.sum(dim=-1)
    maxlen = int(seqlens.max().item())
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[mask], culens, maxlen


def redo_padding(unpadded_seq: Tensor, mask: BoolTensor) -> Tensor:
    """Re-apply padding to an unpadded sequence (zeros at padded positions)."""
    mask = ~mask  # convert mask: True -> valid token
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out


# ===========================================================================
# Absorbed v1 Dense / Transformer / pooling layer family
# ---------------------------------------------------------------------------
# Copied verbatim (math + attribute layout + parameter registration order)
# from the v1 originals in salt/models/*, which stay untouched.
# ===========================================================================


class Dense(nn.Module):
    """A fully connected feed forward neural network, with optional context.

    Parameters
    ----------
    input_size : int
        Input size
    output_size : int | None, optional
        Output size. If not specified this will be the same as the input size, by default None
    hidden_layers : list[int] | None, optional
        Number of nodes per layer, if not specified, the network will have
        a single hidden layer with size `input_size * hidden_dim_scale`, by default None
    hidden_dim_scale : int, optional
        Scale factor for the hidden layer size, by default 2
    activation : str, optional
        Activation function for hidden layers. Must be a valid torch.nn activation function.
        By default "ReLU"
    final_activation : str | None, optional
        Activation function for the output layer. Must be a valid torch.nn activation function.
        By default None
    dropout : float, optional
        Apply dropout with the supplied probability, by default 0.0
    bias : bool, optional
        Whether to use bias in the linear layers, by default True
    context_size : int, optional
        Size of the context tensor, 0 means no context information is provided, by default 0
    mup : bool, optional
        Whether to use the muP parametrisation (impacts initialisation), by default None
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: str = "ReLU",
        final_activation: str | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        context_size: int = 0,
        mup: bool = False,
    ) -> None:
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.mup = mup

        self.node_list = [input_size + context_size, *hidden_layers, output_size]

        layers = []

        num_layers = len(self.node_list) - 1
        for i in range(num_layers):
            if dropout:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(self.node_list[i], self.node_list[i + 1], bias=bias))

            if i != num_layers - 1:
                layers.append(getattr(nn, activation)())
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

        if self.mup:
            self._reset_parameters()

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if self.context_size:
            x = attach_context(x, context)
        return self.net(x)

    def _reset_parameters(self):
        """Initialise the weights and biases for muP."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                std = 1.0 / layer.weight.shape[0] ** 0.5
                nn.init.normal_(layer.weight, std=std)
                nn.init.constant_(layer.bias, 0)


# ---------------------------------------------------------------------------
# layernorm.py absorption (hybrid/RMS norm)
# ---------------------------------------------------------------------------


class LayerNorm(nn.LayerNorm):
    """Faster LayerNorm by setting elementwise_affine=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)


class RMSNorm(torch.nn.Module):
    """RMSNorm from https://arxiv.org/abs/1910.07467 (LLaMA implementation)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class _Layernorms:
    """Namespace exposing `LayerNorm`/`RMSNorm` by name for ``getattr(_LAYERNORMS, norm)``."""

    LayerNorm = LayerNorm
    RMSNorm = RMSNorm


_LAYERNORMS = _Layernorms()


# ---------------------------------------------------------------------------
# featurewise.py + posenc.py absorption — FiLM + positional encoding
# ---------------------------------------------------------------------------

_FEATUREWISE_LAYERS: frozenset[str] = frozenset({"input", "encoder", "global"})
"""Valid FiLM ``layer`` placements: ``input`` applies scale/bias before a
`StreamEmbed`'s projection; ``encoder`` at the start of every encoder layer;
``global`` to the pooled/encoded representation before pooling."""

_POSENC_SYM_VARS: frozenset[str] = frozenset({"phi"})
"""Variables whose positional encoding is symmetric (sin/cos of the sin/cos)."""


class FeaturewiseTransformation(nn.Module):
    """Feature-wise (FiLM) scale/bias from per-event ``parameters``.

    https://distill.pub/2018/feature-wise-transformations/.

    Parameters
    ----------
    layer : str
        Which pipeline stage to scale/bias — one of ``{"input", "encoder", "global"}``.
    num_params : int
        Number of per-event conditioning parameters (the FiLM net input width).
    num_features : int
        Output width of the FiLM scale/bias nets (the embed/encoder width).
    dense_config_scale : dict | None, optional
        Extra `salt.core.nn.Dense` kwargs for the scaling net; must not set
        width keys. When None (and ``dense_config_bias`` is set), bias-only.
    dense_config_bias : dict | None, optional
        Extra `salt.core.nn.Dense` kwargs for the biasing net. When None (and
        ``dense_config_scale`` is set), scale-only.
    apply_norm : bool, optional
        Apply a `torch.nn.LayerNorm` to the transformed features, by default False.

    Raises
    ------
    ConfigError
        If `layer` is invalid, a dense config sets width keys, or neither
        scale nor bias net is configured.
    """

    def __init__(
        self,
        layer: str,
        num_params: int,
        num_features: int,
        dense_config_scale: dict | None = None,
        dense_config_bias: dict | None = None,
        apply_norm: bool = False,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        if layer not in _FEATUREWISE_LAYERS:
            raise ConfigError(
                f"FeaturewiseTransformation: layer must be one of {sorted(_FEATUREWISE_LAYERS)}, "
                f"got {layer!r} (v1 featurewise.py:44)"
            )
        if num_params < 1:
            raise ConfigError(
                f"FeaturewiseTransformation: num_params must be >= 1, got {num_params}"
            )
        if num_features < 1:
            raise ConfigError(
                f"FeaturewiseTransformation: num_features must be >= 1, got {num_features}"
            )
        scale_cfg = dict(dense_config_scale or {})
        bias_cfg = dict(dense_config_bias or {})
        for cfg in (scale_cfg, bias_cfg):
            _reject_width_keys(
                "FeaturewiseTransformation", cfg, ("input_size", "output_size", "context_size")
            )
        # a net is built iff its dense config is truthy — None or {} builds nothing
        build_scale = bool(dense_config_scale)
        build_bias = bool(dense_config_bias)
        if not build_scale and not build_bias:
            raise ConfigError(
                "FeaturewiseTransformation: specify at least one (non-empty) dense_config_scale "
                "or dense_config_bias (v1 featurewise.py:63-66)"
            )
        self.layer = layer
        self.num_params = int(num_params)
        self.num_features = int(num_features)
        self._build_scale = build_scale
        self._build_bias = build_bias
        self.scale_cfg = scale_cfg
        self.bias_cfg = bias_cfg
        self.apply_norm = bool(apply_norm)
        self.scale_net: nn.Module | None = None
        self.bias_net: nn.Module | None = None
        self.norm: nn.Module | None = None
        self._built = False

    def build(self) -> None:
        """Construct the scale/bias `Dense` nets + optional norm (idempotent)."""
        if self._built:
            return
        if self._build_scale:
            self.scale_net = Dense(
                input_size=self.num_params, output_size=self.num_features, **self.scale_cfg
            )
        if self._build_bias:
            self.bias_net = Dense(
                input_size=self.num_params, output_size=self.num_features, **self.bias_cfg
            )
        if self.apply_norm:
            self.norm = nn.LayerNorm(self.num_features)
        self._built = True

    def forward(self, params: Tensor, features: Tensor) -> Tensor:
        """Apply the FiLM scale/bias (and optional norm) to ``features``.

        Parameters
        ----------
        params : Tensor
            Per-event conditioning parameters ``[B, n_params]``.
        features : Tensor
            Features to transform ``[B, T, num_features]`` (or ``[B, num_features]``
            for the global layer).
        """
        assert self._built, "FeaturewiseTransformation.forward before build()"
        if self.scale_net is not None:
            features = self.scale_net(params).unsqueeze(1) * features
        if self.bias_net is not None:
            features = torch.add(features, self.bias_net(params).unsqueeze(1))
        if self.norm is not None:
            features = self.norm(features)
        return features


class PositionalEncoder(nn.Module):
    """Sin/cos positional encoding over coordinate variables.

    Evenly shares the embedding space between the encoded variables; any
    remaining dimensions are left as zeros. Parameter-free (``@torch.no_grad``).

    Parameters
    ----------
    variables : Sequence[str]
        Variable names to encode. Symmetric variables (``phi``) get the
        sin-of-sin / sin-of-cos symmetric encoding.
    dim : int
        Total positional-encoding width, split evenly across variables
        (``per_input_dim = dim // (2 * len(variables))``, remainder zero-padded).
    alpha : int, optional
        Frequency scaling factor, by default 100.

    Raises
    ------
    ConfigError
        If `variables` is empty or `dim` is too small to give each variable at
        least one frequency band.
    """

    def __init__(self, variables: Sequence[str], dim: int, alpha: int = 100) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.variables = tuple(variables)
        if not self.variables:
            raise ConfigError("PositionalEncoder: variables must be a non-empty sequence")
        self.dim = int(dim)
        self.alpha = int(alpha)
        self.per_input_dim = self.dim // (2 * len(self.variables))
        self.last_dim = self.dim % (2 * len(self.variables))
        if self.per_input_dim < 1:
            raise ConfigError(
                f"PositionalEncoder: dim={self.dim} too small for {len(self.variables)} variables "
                f"(per_input_dim = dim // (2*n_vars) = {self.per_input_dim} < 1)"
            )

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Encode each coordinate column (in ``variables`` order); concat along the last dim.

        Returns
        -------
        Tensor
            The positional encoding ``[..., dim]``.
        """
        encodings: list[Tensor] = []
        for i, var in enumerate(self.variables):
            symmetric = var in _POSENC_SYM_VARS
            encodings.append(self.pos_enc(inputs[..., i], self.per_input_dim, symmetric=symmetric))
        if self.last_dim > 0:
            encodings.append(torch.zeros_like(encodings[0][..., : self.last_dim]))
        return torch.cat(encodings, dim=-1)

    def pos_enc(self, xs: Tensor, dim: int, symmetric: bool = False) -> Tensor:
        """One variable's sin/cos encoding.

        Returns
        -------
        Tensor
            The ``[..., 2*dim]`` encoding for this variable.
        """
        xs = xs.unsqueeze(-1)
        kwargs = {"device": xs.device, "dtype": xs.dtype}
        omegas = self.alpha * torch.logspace(0, 2 / (dim) - 1, dim, 10_000, **kwargs)
        if symmetric:
            p1 = (xs.sin() * omegas).sin()
            p2 = (xs.cos() * omegas).sin()
        else:
            p1 = (xs * omegas).sin()
            p2 = (xs * omegas).cos()
        return torch.cat((p1, p2), dim=-1)


# ---------------------------------------------------------------------------
# attention.py absorption (MultiheadAttention / EdgeAttention / SDPA helpers)
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None


def check_flash_attn() -> str:
    """Check if Flash Attention is available and compatible.

    Returns
    -------
    str
        Empty string if Flash Attention is available, otherwise a reason why not.
    """
    if not torch.cuda.is_available():
        return "No GPU available."

    gpu_name = torch.cuda.get_device_name(0)
    # (major, minor), e.g. (8, 0)
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    sm_version = float(f"{major}.{minor}")

    if sm_version < 8.0:
        return (
            f"GPU '{gpu_name}' with SM {sm_version} is not compatible. "
            "Flash Attention 2 requires SM 8.0 or newer (Ampere+)."
        )
    if _flash_attn_func is None:
        return (
            "Flash attention is required but not found! Please ensure the package is installed "
            "correctly. If not, please install the flash attention package as described in "
            "https://ftag-salt.docs.cern.ch/setup/#install-the-salt-package"
        )
    return ""


ATTN_TYPES = ["torch-math", "torch-flash", "torch-meff", "flash-varlen"]


def merge_masks(
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
) -> BoolTensor | None:
    """Create a full attention mask which incorporates padding information.

    Padded tokens can't **send** information but can **receive** it (prevents
    softmax NaNs).

    Returns
    -------
    BoolTensor | None
        Combined mask of shape ``[B, 1, L_q, L_kv]`` (broadcastable over heads), or ``None``.
    """
    mask = None

    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # flip: True now means "valid token" (not padded)

    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask & mask

    if mask is not None:
        mask = mask.unsqueeze(1)  # broadcast over heads

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int) -> tuple[Tensor, Tensor]:
    """Repeat keys and values along a dimension.

    Returns
    -------
    tuple[Tensor, Tensor]
        Repeated ``(keys, values)`` tensors.
    """
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def projection_packed(
    q: Tensor,
    kv: Tensor | None,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Efficient input projection for MHA using a single packed linear layer.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Projected queries, keys, and values: ``(Q, K, V)``.
    """
    if kv is None:
        return functional.linear(q, weight, bias).chunk(3, dim=-1)

    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    q_proj = functional.linear(q, w_q, b_q)
    k_proj, v_proj = functional.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj


def torch_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: BoolTensor | None,
    dropout: float,
    softmax_scale: float,
    backend: str,
) -> Tensor:
    """Scaled dot-product attention with a switchable torch backend.

    Returns
    -------
    Tensor
        Attention output of shape ``[B, H, L_q, D_h]``.
    """
    backends = [SDPBackend.MATH]
    if backend == "torch-flash":
        backends += [SDPBackend.FLASH_ATTENTION]
    elif backend == "torch-meff":
        backends += [SDPBackend.EFFICIENT_ATTENTION]
    with sdpa_kernel(backends=backends):
        return functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout, scale=softmax_scale
        )


class Attention(nn.Module):
    """Multihead attention module with optional differential attention and norms.

    Parameters
    ----------
    embed_dim : int
        Input (and output) embedding dimension.
    num_heads : int, optional
        Number of attention heads. The default is ``1``.
    attn_type : str, optional
        Backend kernel to use. One of ``{"torch-math", "torch-flash", "torch-meff",
        "flash-varlen"}``. The default is ``"torch-meff"``.
    dropout : float, optional
        Dropout rate applied in attention. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms in projections. The default is ``True``.
    do_qk_norm : bool, optional
        Whether to apply RMSNorm to Q and K per head. The default is ``False``.
    do_v_norm : bool, optional
        Whether to apply RMSNorm to V per head. The default is ``False``.
    mup: bool, optional
        Whether to use the muP parametrisation. The default is ``False``.
        Impacts init and scale of dot product sqrt(head_dim) -> head_dim.
        Ref: https://arxiv.org/abs/2203.03466
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_type: str = "torch-meff",
        dropout: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
        do_v_norm: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        assert attn_type in ATTN_TYPES, "Invalid attention type!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.attn_type = attn_type
        self.do_qk_norm = do_qk_norm
        self.do_v_norm = do_v_norm
        self.mup = mup

        self.scale = 1 / self.head_dim if mup else 1 / math.sqrt(self.head_dim)

        if self.do_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        if self.do_v_norm:
            self.v_norm = RMSNorm(self.head_dim)

        # packed QKV projection: better parallelism for self-attention than 3 separate Linears
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str) -> str:
        """Set and validate the attention backend.

        Returns
        -------
        str
            Effective backend set (may fall back to ``"torch-math"``).
        """
        self.attn_type = attn_type
        if self.attn_type == "flash-varlen":
            why_not_flash = check_flash_attn()
            if why_not_flash:
                warnings.warn(
                    f"Cannot use flash-varlen backend. {why_not_flash} Reverting to torch-math.",
                    stacklevel=2,
                )
                self.attn_type = "torch-math"
            else:
                self._flash_attn = _flash_attn_func
        return self.attn_type

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        if self.mup:
            # muP init: https://arxiv.org/abs/2203.03466
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=1.0 / self.head_dim**0.5)  # K,V proj
            nn.init.constant_(self.in_proj_weight[: self.embed_dim, :], 0.0)  # Q projection
            nn.init.normal_(self.out_proj.weight, std=(1.0 / self.embed_dim) ** 0.5)  # Output proj
            if self.bias:
                nn.init.constant_(self.in_proj_bias, 0.0)
                nn.init.constant_(self.out_proj.bias, 0.0)
            return

        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj.reset_parameters()
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)

    def _flash_forward(self, x: Tensor, culens: Tensor, maxlen: int) -> Tensor:
        """FlashAttention backend.

        Returns
        -------
        Tensor
            Output of shape ``[N_total, D]``.
        """
        qkv = functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        if self.do_qk_norm or self.do_v_norm:
            dtype = qkv.dtype
            q, k, v = qkv.unbind(1)
            if self.do_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if self.do_v_norm:
                v = self.v_norm(v)
            qkv = torch.stack([q, k, v], dim=1).to(dtype)

        dropout = self.dropout if self.training else 0.0
        a_out = self._flash_attn(qkv, culens, maxlen, dropout, softmax_scale=self.scale)
        a_out = a_out.reshape(-1, self.embed_dim)

        return self.out_proj(a_out)

    def _torch_forward(
        self, x: Tensor, kv: Tensor, mask: BoolTensor, kv_mask: BoolTensor, attn_mask: BoolTensor
    ) -> Tensor:
        """Attention using PyTorch SDPA backends.

        Returns
        -------
        Tensor
            Output of shape ``[B, L_q, D]``.
        """
        b, s, d = x.shape

        q, k, v = projection_packed(x, kv, self.in_proj_weight, self.in_proj_bias)

        shape = (b, -1, self.num_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.do_v_norm:
            v = self.v_norm(v)

        s_mask = mask if kv is None else kv_mask  # Who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        dropout = self.dropout if self.training else 0.0
        a_out = torch_attn(
            q, k, v, mask, dropout=dropout, softmax_scale=self.scale, backend=self.attn_type
        )

        a_out = a_out.transpose(1, 2).contiguous().view(b, s, d)
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
        """Attention forward pass, dispatching to the appropriate backend.

        Returns
        -------
        Tensor
            Output of shape ``[B, L_q, D]``.
        """
        if self.attn_type == "flash-varlen":
            assert kv is None, "flash-varlen only supports self attention!"
            assert attn_mask is None, "flash-varlen does not support attention masks!"
            assert culens is not None, "flash-varlen requires culens!"
            assert maxlen is not None, "flash-varlen requires maxlen!"
            return self._flash_forward(x, culens, maxlen)

        return self._torch_forward(x, kv, mask, kv_mask, attn_mask)


class EdgeAttention(nn.Module):
    """Multihead attention module with optional norms, including edge features.

    Parameters
    ----------
    embed_dim : int
        Input (and output) embedding dimension.
    edge_embed_dim : int
        Model embedding dimension for edge features.
    num_heads : int, optional
        Number of attention heads. The default is ``1``.
    dropout : float, optional
        Dropout rate applied in attention. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms in projections. The default is ``True``.
    do_qk_norm : bool, optional
        Whether to apply RMSNorm to Q and K per head. The default is ``False``.
    do_v_norm : bool, optional
        Whether to apply RMSNorm to V per head. The default is ``False``.
    update_edges : bool, optional
        Indicate whether to update edge features, by default False
    mup: bool, optional
        Whether to use the muP parametrisation. The default is ``False``.
        Impacts init and scale of dot product sqrt(head_dim) -> head_dim.
        Ref: https://arxiv.org/abs/2203.03466
    """

    def __init__(
        self,
        embed_dim: int,
        edge_embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
        do_v_norm: bool = False,
        update_edges: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.edge_embed_dim = edge_embed_dim
        self.edge_head_dim = edge_embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.do_qk_norm = do_qk_norm
        self.do_v_norm = do_v_norm
        self.update_edges = update_edges
        self.mup = mup

        self.scale = 1 / self.head_dim if mup else 1 / math.sqrt(self.head_dim)

        if self.do_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        if self.do_v_norm:
            self.v_norm = RMSNorm(self.head_dim)

        # packed QKV projection: better parallelism for self-attention than 3 separate Linears
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.linear_e = nn.Linear(self.edge_embed_dim, self.num_heads, bias=bias)
        self.linear_g = nn.Linear(self.edge_embed_dim, self.num_heads, bias=bias)
        if self.update_edges:
            self.linear_e_out = nn.Linear(self.num_heads, self.edge_embed_dim, bias=bias)
        else:
            self.register_buffer("linear_e_out", None)

        self.reset_parameters()

    def set_backend(self, attn_type: str) -> str:
        warnings.warn(
            "EdgeAttention does not support different backends yet. Using raw attention.",
            stacklevel=2,
        )
        return attn_type

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        if self.mup:
            # muP init: https://arxiv.org/abs/2203.03466
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=1.0 / self.head_dim**0.5)  # K,V proj
            nn.init.constant_(self.in_proj_weight[: self.embed_dim, :], 0.0)  # Q projection
            linear_layers = [self.out_proj]
            nn.init.normal_(self.linear_e.weight, std=(1.0 / self.edge_embed_dim) ** 0.5)
            nn.init.normal_(self.linear_g.weight, std=(1.0 / self.edge_embed_dim) ** 0.5)
            linear_layers.extend([self.linear_e, self.linear_g])
            if self.update_edges:
                nn.init.normal_(self.linear_e_out.weight, std=(1.0 / self.num_heads) ** 0.5)
                linear_layers.append(self.linear_e_out)
            if self.bias:
                nn.init.constant_(self.in_proj_bias, 0.0)
                for layer in linear_layers:
                    nn.init.constant_(layer.bias, 0.0)
            return

        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)

        layers = [self.linear_e, self.linear_g, self.out_proj]
        if self.update_edges:
            layers.append(self.linear_e_out)
        for layer in layers:
            layer.reset_parameters()
            if self.bias:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        edge_x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Attention with edge features biasing the scores and gating the output.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output of shape ``[B, L_q, D]`` and updated edge features of shape
            ``[B, L_q, L_kv, E]``.
        """
        b, s, d = x.shape
        q, k, v = projection_packed(x, kv, self.in_proj_weight, self.in_proj_bias)

        shape = (b, -1, self.num_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.do_v_norm:
            v = self.v_norm(v)

        s_mask = mask if kv is None else kv_mask  # who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        e = self.linear_e(edge_x)  # (B, L_q, L_kv, num_heads)
        g = functional.sigmoid(self.linear_g(edge_x))  # (B, L_q, L_kv, num_heads)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, L_q, L_kv)
        attn_scores = attn_scores + e.permute(0, 3, 1, 2)  # add edge embeddings

        if self.dropout > 0.0 and self.training:
            attn_scores = functional.dropout(attn_scores, p=self.dropout)

        edge_out = edge_x
        if self.update_edges:
            edge_out = self.linear_e_out(
                attn_scores.permute(0, 2, 3, 1)  # (B, L_q, L_kv, num_heads)
            )
        # Compute attention weights
        masked_scores = (
            torch.masked_fill(attn_scores, ~mask, float("-inf"))
            if mask is not None
            else attn_scores
        )
        attn_weights = torch.softmax(masked_scores, dim=-1)  # (B, num_heads, L_q, L_kv)

        attn_weights = attn_weights * g.permute(0, 3, 1, 2)  # apply gating

        a_out = torch.matmul(attn_weights, v)  # (B, num_heads, L_q, head_dim)

        a_out = a_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(a_out), edge_out


# ---------------------------------------------------------------------------
# transformer.py absorption (GLU / LayerScale / DropPath / NormResidual /
# EncoderLayer / Transformer — the pieces TransformerEncoder uses)
# ---------------------------------------------------------------------------

try:
    from mup import MuReadout as _MuReadout

except ImportError:
    _MuReadout = None


class GLU(nn.Module):
    """Dense update with a (gated) linear unit. See https://arxiv.org/abs/2002.05202.

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    hidden_dim : int | None, optional
        Hidden dimension. If ``None``, defaults to ``2 * embed_dim``.
    activation : str, optional
        Name of the activation class in ``torch.nn`` (e.g., ``"SiLU"``).
    dropout : float, optional
        Dropout probability. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms. The default is ``True``.
    gated : bool, optional
        If ``True``, uses a gated branch (splits hidden in two). The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
        mup: bool = False,
    ):
        super().__init__()
        self.mup = mup

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        if self.mup:
            for proj in [self.in_proj, self.out_proj]:
                nn.init.normal_(proj.weight, mean=0.0, std=1.0 / (proj.weight.shape[0] ** 0.5))
                if bias:
                    nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the GLU block.

        Returns
        -------
        Tensor
            Output tensor of shape ``[B, L, D]``.
        """
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


class LayerScale(nn.Module):
    """Applies the LayerScale operation from CaiT (stabilizes deep transformers).

    Reference: https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale the input by a learnable vector ``gamma``.

        Returns
        -------
        Tensor
            Scaled inputs as Tensor
        """
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth / drop-path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Randomly drop residual paths during training.

        Returns
        -------
        Tensor
            Output tensor with stochastic depth applied when training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class NormResidual(nn.Module):
    """Residual wrapper with normalization, LayerScale, and DropPath.

    Represents PostNorm/PreNorm/NoNorm patterns and forwards edge features
    for `EdgeAttention`.

    Parameters
    ----------
    fn : GLU | Attention | EdgeAttention
        The wrapped non-resizing module.
    norm : str, optional
        Normalization class name. The default is ``"LayerNorm"``.
    ls_init : float | None, optional
        Initial value for LayerScale. If ``None``, LayerScale is disabled.
    drop_path : float, optional
        Drop-path rate for stochastic depth. The default is ``0.0``.
    embed_dim : int, optional
        Input/output dimension. If ``0``, attempts to read ``fn.embed_dim``.
    norm_type : str, optional
        One of ``{"pre", "post", "none"}``. The default is ``"pre"``.
    """

    def __init__(
        self,
        fn: GLU | Attention | EdgeAttention,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        embed_dim: int = 0,
        norm_type: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        dim = embed_dim or fn.embed_dim
        assert dim > 0, "Could not determine embed_dim from fn"
        self.fn = fn
        if self.norm_type != "none":
            self.norm = getattr(_LAYERNORMS, norm)(dim)
        self.ls = LayerScale(dim, ls_init) if ls_init is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

        self.edges = bool(isinstance(fn, EdgeAttention))

    def _forward_edges(self, x: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Apply residual wrapper around ``fn`` that returns edge features.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output tensor with residual + the edge features from ``fn``.
        """
        if self.norm_type == "pre":
            fn_out, edge_out = self.fn(self.norm(x), *args, **kwargs)
            res_out = x + self.drop_path(self.ls(fn_out))
            return res_out, edge_out
        if self.norm_type == "post":
            fn_out, edge_out = self.fn(x, *args, **kwargs)
            res_out = self.norm(x + self.drop_path(self.ls(fn_out)))
            return res_out, edge_out
        fn_out, edge_out = self.fn(x, *args, **kwargs)
        res_out = x + self.drop_path(self.ls(fn_out))
        return res_out, edge_out

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        """Apply residual wrapper around ``fn``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Output tensor with residual; a tuple ``(output, edge_out)`` for `EdgeAttention`.
        """
        if self.edges:
            return self._forward_edges(x, *args, **kwargs)
        if self.norm_type == "pre":
            return x + self.drop_path(self.ls(self.fn(self.norm(x), *args, **kwargs)))
        if self.norm_type == "post":
            return self.norm(x + self.drop_path(self.ls(self.fn(x, *args, **kwargs))))
        return x + self.drop_path(self.ls(self.fn(x, *args, **kwargs)))


@final
class EncoderLayer(nn.Module):
    """Transformer encoder layer: self-attention + feed-forward.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    norm : str, optional
        Normalization style. The default is ``"LayerNorm"``.
    ls_init : float | None, optional
        Initial LayerScale value. If ``None``, LayerScale is disabled.
    drop_path : float, optional
        Drop-path rate. The default is ``0.0``.
    depth : int, optional
        Layer depth index, used for differential attention weighting. The default is ``1``.
    dense_kwargs : dict | None, optional
        Keyword args for :class:`GLU`.
    attn_kwargs : dict | None, optional
        Keyword args for :class:`Attention`.
    norm_type : str, optional
        One of ``{"pre", "post", "hybrid"}``. The default is ``"pre"``.
    edge_embed_dim : int, optional
        Model embedding dimension for edge features. The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated after attention. The default is ``False``
    mup: bool, optional
        Whether to use μP parameterization. The default is ``False``.
    num_dense: int, optional
        Number of dense layers to stack in the feed-forward block. The default is ``1``.
    """

    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        depth: int = 1,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        norm_type: str = "pre",
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
        num_dense: int = 1,
    ) -> None:
        super().__init__()
        self.mup = mup
        self.num_dense = num_dense

        assert num_dense >= 1, "num_dense must be at least 1"
        self.update_edges = update_edges

        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        self.embed_dim = embed_dim
        self.norm_type = norm_type
        if norm_type == "hybrid":
            attn_kwargs["do_qk_norm"] = True
            attn_kwargs["do_v_norm"] = True
            residual_norm_type = "pre" if depth == 0 else "none"
            self.norm = (
                nn.Identity(embed_dim) if depth == 0 else getattr(_LAYERNORMS, norm)(embed_dim)
            )
        else:
            residual_norm_type = norm_type

        if self.mup:
            attn_kwargs["mup"] = True
            dense_kwargs["mup"] = True

        attn_class: type[Attention | EdgeAttention]
        if edge_embed_dim > 0:
            attn_class = EdgeAttention
            attn_kwargs["edge_embed_dim"] = edge_embed_dim
            attn_kwargs["update_edges"] = update_edges

            self.edge_prenorm = getattr(_LAYERNORMS, norm)(edge_embed_dim)
            if self.update_edges:
                self.edge_postnorm = getattr(_LAYERNORMS, norm)(edge_embed_dim)
        else:
            attn_class = Attention

        residual = partial(
            NormResidual,
            norm=norm,
            ls_init=ls_init,
            drop_path=drop_path,
            norm_type=residual_norm_type,
        )
        self.attn = residual(attn_class(embed_dim, **attn_kwargs))
        if num_dense == 1:
            self.dense = residual(GLU(embed_dim, **dense_kwargs))
        else:
            self.dense = nn.Sequential(*[
                residual(GLU(embed_dim, **dense_kwargs)) for _ in range(num_dense)
            ])

    def forward(
        self, x: Tensor, edge_x: Tensor | None = None, **kwargs: Any
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention and feed-forward.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            The updated token embeddings (and edges when ``edge_x`` is provided).
        """
        if edge_x is not None:
            x, edge_x = self.attn(x, edge_x=edge_x, **kwargs)
            if self.update_edges:
                edge_x = edge_x + self.edge_postnorm(edge_x)
        else:
            x = self.attn(x, **kwargs)

        x = self.dense(self.norm(x)) if self.norm_type == "hybrid" else self.dense(x)

        if edge_x is not None:
            return x, edge_x
        return x


@final
class Transformer(nn.Module):
    """Transformer encoder stack with optional registers and output projection.

    Parameters
    ----------
    num_layers : int
        Number of encoder layers.
    embed_dim : int
        Embedding dimension.
    out_dim : int | None, optional
        Optional output projection dimension. If ``None``, equals ``embed_dim``.
    norm : str, optional
        Normalization style. The default is ``"LayerNorm"``.
    attn_type : str, optional
        Attention backend. The default is ``"torch-math"``.
    do_final_norm : bool, optional
        Whether to apply a final normalization layer. The default is ``True``.
    num_registers : int, optional
        Number of learned register tokens. The default is ``1``.
    drop_registers : bool, optional
        If ``True``, registers are dropped from outputs. The default is ``False``.
    edge_embed_dim : int, optional
        Model embedding dimension for edge features. The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated after attention. The default is ``False``
    mup: bool, optional
        Whether to use μP parameterization. The default is ``False``.
    **kwargs : Any
        Extra keyword arguments forwarded to :class:`EncoderLayer`.

    Raises
    ------
    ValueError
        If ``num_registers < 1``.
    """

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
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if num_registers < 1:
            raise ValueError(
                "Some global objects (graphs) might have no constituents (nodes), "
                "which causes NaNs in the attention scores. "
                "To avoid this, set num_registers to at least 1",
            )

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.out_dim = out_dim or embed_dim
        self.do_final_norm = do_final_norm
        self.do_out_proj = out_dim is not None
        self.attn_type = attn_type
        self.num_registers = num_registers
        self.drop_registers = drop_registers
        self.edge_embed_dim = edge_embed_dim
        self.update_edges = update_edges
        self.mup = mup

        if self.update_edges:
            assert edge_embed_dim > 0, "Cannot update edges with edge_embed_dim=0"

        if self.mup:
            assert _MuReadout is not None, "mup is not installed!"
            assert self.do_out_proj, (
                "Need the out_dim layer for muP, \
                as this is the last layer of the muP-part of the model"
            )

        # attn_type is meaningless for EdgeAttention (edge_embed_dim > 0): it has no
        # pluggable backend, so only set it when the plain Attention path is used
        if edge_embed_dim == 0:
            kwargs["attn_kwargs"]["attn_type"] = self.attn_type

        self.layers = torch.nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                norm=norm,
                depth=depth,
                edge_embed_dim=edge_embed_dim,
                update_edges=update_edges,
                **kwargs,
            )
            for depth in range(num_layers)
        ])

        if self.edge_embed_dim == 0:
            assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
            self.set_backend(self.attn_type)

        if self.do_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
            if self.mup and _MuReadout is not None:
                self.out_proj = _MuReadout(embed_dim, self.out_dim)
                self.out_proj.bias.data.zero_()
                self.out_proj.weight.data.zero_()
        if self.do_final_norm:
            self.out_norm = getattr(_LAYERNORMS, norm)(self.out_dim)
        if self.num_registers:
            self.registers = nn.Parameter(
                torch.normal(torch.zeros((self.num_registers, self.embed_dim)), std=1e-4)
            )
            self.register_buffer("register_mask", torch.zeros(num_registers, dtype=torch.bool))
        self.featurewise = nn.ModuleList()

    def set_backend(self, attn_type: str) -> None:
        """Set the attention backend for all layers."""
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        pad_mask: BoolTensor | dict[str, BoolTensor],
        inputs: Any | None = None,
        edge_x: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, BoolTensor | dict[str, BoolTensor]]:
        """Run the encoder stack.

        Returns
        -------
        tuple[Tensor, BoolTensor | dict[str, BoolTensor]]
            Tuple of ``(encoded, pad_mask)`` where ``encoded`` has shape ``[B, L, D_out]``.
        """
        if self.num_registers:
            x, pad_mask = self._add_registers(x, pad_mask)

        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # zero-pad edges to the register-augmented sequence length (registers attend
        # to each other/tokens with no edge features, so they contribute zero bias)
        if edge_x is not None:
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (
                            edge_x.shape[0],
                            x.shape[1] - edge_x.shape[1],
                            edge_x.shape[2],
                            edge_x.shape[3],
                        ),
                        device=edge_x.device,
                    ),
                ],
                dim=1,
            )
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (
                            edge_x.shape[0],
                            edge_x.shape[1],
                            x.shape[1] - edge_x.shape[2],
                            edge_x.shape[3],
                        ),
                        device=edge_x.device,
                    ),
                ],
                dim=2,
            )

        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            if edge_x is not None:
                x, edge_x = layer(x, edge_x=edge_x, mask=mask, **kwargs)
            else:
                x = layer(x, mask=mask, **kwargs)

        if self.do_out_proj:
            x = self.out_proj(x)
        if self.do_final_norm:
            x = self.out_norm(x)

        if self.attn_type == "flash-varlen":
            x = redo_padding(x, mask)

        if self.drop_registers:
            x = x[:, : -self.num_registers]
            if isinstance(pad_mask, dict):
                del pad_mask["REGISTERS"]
            elif isinstance(pad_mask, Tensor):
                pad_mask = pad_mask[:, : -self.num_registers]

        return x, pad_mask

    def _add_registers(
        self, x: Tensor | dict[str, Tensor], pad_mask: BoolTensor | dict[str, BoolTensor] | None
    ) -> tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]:
        """Add the learnable registers to the end of the input sequence (and mask).

        Returns
        -------
        tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]
            Updated ``(x, pad_mask)`` including appended registers.
        """
        batch_size = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)

        reg = self.registers.expand(batch_size, -1, -1)
        if isinstance(x, dict):
            x["REGISTERS"] = reg
        else:
            x = torch.cat([x, reg], dim=1)

        if pad_mask is not None:
            reg_mask = self.register_mask.expand(batch_size, -1)
            if isinstance(pad_mask, dict):
                pad_mask["REGISTERS"] = reg_mask
            else:
                pad_mask = torch.cat([pad_mask, reg_mask], dim=-1)

        return x, pad_mask


# ---------------------------------------------------------------------------
# pooling.py absorption (GlobalAttentionPooling math — composed by the v2
# GlobalAttentionPooling GraphModule below)
# ---------------------------------------------------------------------------


class _GlobalAttentionPoolingV1(nn.Module):
    """Global attention pooling over concatenated node embeddings.

    Named with the ``V1`` suffix because the config-facing `GraphModule` below
    is also called `GlobalAttentionPooling`; this is the inner ``nn.Module`` it
    composes.

    Parameters
    ----------
    input_size : int
        Dimensionality of each node embedding feature vector.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(
        self,
        x: dict[str, Tensor] | dict,
        pad_mask: dict | None = None,
    ) -> Tensor:
        """Apply global attention pooling.

        Returns
        -------
        Tensor
            Pooled tensor of shape ``[B, D]``.
        """
        x_flat = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x_flat), pad_mask, dim=1)
        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_flat.shape[0], 1, x_flat.shape[2]), device=x_flat.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_flat = torch.cat([x_flat, x_pad], dim=1)

        return (x_flat * weights).sum(dim=1)


def _stream_len(stream: str) -> str:
    """Return the shared symbolic token-count dim for a sequence stream, e.g. ``"T:tracks"``."""
    return sym_dim("T", stream)


class Normaliser(nn.Module):
    """Config-constructed input normalisation, replacing v1 `InputNorm`.

    The DEFAULT input normaliser: loads a precomputed ``norm_dict.yaml``
    (fixed per-stream, per-variable ``{mean, std}``). For a self-normalising
    variant that learns statistics online (no norm dict), use
    ``class_path: salt.core.nn.MaskedInputNormaliser`` instead.

    ``materialise()`` is the only file-touching hook (loads the norm dict and
    fills the buffers) — skipped on checkpoint load, where values arrive via
    the state_dict. Produces NEW ``normed.<stream>`` keys; never mutates
    ``inputs.*``.
    """

    def __init__(
        self,
        norm_dict: str | Path,
        streams: Sequence[str],
        global_object: str | None = None,
    ) -> None:
        """Capture config only (no file I/O here).

        Parameters
        ----------
        norm_dict : str | Path
            Path to the normalisation dictionary YAML; read at `materialise`, never here.
        streams : Sequence[str]
            Streams to normalise.
        global_object : str | None, optional
            The stream that is a per-object vector (``[B, F]``) rather than
            a padded sequence (``[B, T, F]``), by default None.

        Raises
        ------
        ConfigError
            If `streams` is empty, contains duplicates, or `global_object`
            is not one of them.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Normaliser: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Normaliser: duplicate streams in {tuple(streams)}")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"Normaliser: global_object {global_object!r} is not in streams {tuple(streams)}"
            )
        self.norm_dict_path = Path(norm_dict)
        self.streams = tuple(streams)
        self.global_object = global_object
        self._fields: dict[str, tuple[str, ...]] = {}
        self._bound = False

    def _spec(self, stream: str) -> TensorSpec:
        """Build the shared spec for ``inputs.<stream>`` / ``normed.<stream>``.

        Returns
        -------
        TensorSpec
            ``("B", F)`` for the global object, ``("B", "T:<stream>", F)``
            for sequence streams.
        """
        width = sym_dim("F", f"{self.name}.{stream}")
        shape: tuple[int | str, ...] = (
            ("B", width) if stream == self.global_object else ("B", _stream_len(stream), width)
        )
        return TensorSpec(shape=shape, dtype="float32")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream."""
        del mode
        return IO(
            requires=unflatten_spec({f"inputs.{s}": self._spec(s) for s in self.streams}),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate normalisation buffers (means/stds per stream) from the resolved schema.

        The boolean ``materialised`` buffer guards against silently training
        on un-normalised values.

        Raises
        ------
        RuntimeError
            If called twice (rebinding would discard loaded values).
        """
        if self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: bind() called twice (design §2.3)")
        for stream in self.streams:
            key = f"inputs.{stream}"
            width = schema.width(key)
            self._fields[stream] = schema.fields_of(key)
            self.register_buffer(f"means_{stream}", torch.zeros(width))
            self.register_buffer(f"stds_{stream}", torch.ones(width))
        self.register_buffer("materialised", torch.tensor(False))
        self._bound = True

    def preflight(self) -> None:
        """Fail-fast, data-free norm-dict validation.

        Reads ONLY the norm-dict YAML: the file must exist, parse, and carry
        every configured stream; when already bound, per-variable mean/std
        entries are checked too. Called by `SaltModule.setup` on fresh fits
        (hard error) and by ``salt2 graph validate`` (warning).

        Raises
        ------
        ConfigError
            On a missing/unparsable norm dict, a missing stream, or (when
            bound) missing/non-finite/zero-std variable entries.
        """
        path = self.norm_dict_path
        prefix = f"Normaliser {self.name!r} preflight"
        fix = (
            f"  fix: point model.modules.{self.name}.init_args.norm_dict at the "
            "preprocessing norm_dict.yaml for this sample"
        )
        if not path.is_file():
            raise ConfigError(f"{prefix}: norm dict not found: {path}\n{fix}")
        try:
            with open(path) as fh:
                norm_dict = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(
                f"{prefix}: norm dict {path} is not valid YAML: {err}\n{fix}"
            ) from err
        if not isinstance(norm_dict, dict):
            raise ConfigError(f"{prefix}: norm dict {path} must be a mapping\n{fix}")
        for stream in self.streams:
            if stream not in norm_dict:
                raise ConfigError(
                    f"{prefix}: missing input type {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            if not self._fields:
                continue  # unbound (the data-free `salt2 graph validate` path)
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ConfigError(
                    f"{prefix}: missing variables {sorted(missing)} for {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to {path}, or remove "
                    f"them from the features variable list "
                    f"(config: data.modules.features.init_args.variables.{stream})"
                )
            for variable in variables:
                entry = norm_dict[stream][variable]
                try:
                    mean, std = float(entry["mean"]), float(entry["std"])
                except (KeyError, TypeError, ValueError):
                    raise ConfigError(
                        f"{prefix}: entry for {stream}.{variable} in {path} must be a "
                        f"{{mean, std}} mapping, got {entry!r}"
                    ) from None
                if not (torch.isfinite(torch.tensor(mean)) and torch.isfinite(torch.tensor(std))):
                    raise ConfigError(
                        f"{prefix}: non-finite normalisation parameters for "
                        f"{stream}.{variable} in {path}."
                    )
                if std == 0:
                    raise ConfigError(
                        f"{prefix}: zero standard deviation for {stream}.{variable} in {path}."
                    )

    def materialise(self) -> None:
        """Fill the buffers from the norm dict (the only file I/O this module does).

        Missing streams/variables, non-finite values, and zero stds are errors.

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the norm dict is missing this module's streams or variables,
            or contains non-finite means/stds or zero stds.
        """
        if not self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: materialise() before bind()")
        with open(self.norm_dict_path) as fh:
            norm_dict = yaml.safe_load(fh)
        for stream in self.streams:
            if stream not in norm_dict:
                raise ValueError(
                    f"Missing input type {stream!r} in {self.norm_dict_path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ValueError(
                    f"Missing variables {sorted(missing)} for {stream!r} in "
                    f"{self.norm_dict_path}. Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to "
                    f"{self.norm_dict_path}, or remove them from the features variable "
                    f"list (config: data.modules.features.init_args.variables.{stream})"
                )
            means = torch.as_tensor(
                [float(norm_dict[stream][v]["mean"]) for v in variables], dtype=torch.float32
            )
            stds = torch.as_tensor(
                [float(norm_dict[stream][v]["std"]) for v in variables], dtype=torch.float32
            )
            if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
                raise ValueError(
                    f"Non-finite normalisation parameters for {stream!r} in {self.norm_dict_path}."
                )
            if (stds == 0).any():
                raise ValueError(
                    f"Zero standard deviation for {stream!r} in {self.norm_dict_path}."
                )
            with torch.no_grad():
                getattr(self, f"means_{stream}").copy_(means)
                getattr(self, f"stds_{stream}").copy_(stds)
        self.materialised.fill_(True)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Produce ``normed.<stream> = (inputs.<stream> - means) / stds``.

        Raises
        ------
        RuntimeError
            If the buffers were never materialised (fresh fit without
            `materialise`) — identity values would silently train un-normalised.
        """
        del mode
        # skip under tracing: the tensor->bool read would emit a spurious
        # TracerWarning on every export. Tracing is still guarded —
        # OnnxAdapter rejects unmaterialised modules at construction, before
        # any trace (the eager path keeps this check).
        if not torch.jit.is_tracing() and not bool(self.materialised):
            raise RuntimeError(
                f"Normaliser {self.name!r}: forward before materialise() — on a fresh fit "
                "call materialise(); on checkpoint load the state_dict provides the values "
                "(design §2.3)"
            )
        return {
            f"normed.{s}": (b.get(f"inputs.{s}") - getattr(self, f"means_{s}"))
            / getattr(self, f"stds_{s}")
            for s in self.streams
        }


class MaskedInputNormaliser(nn.Module):
    """Self-normalising input layer with online masked running statistics.

    OPT-IN alternative to the default fixed-norm-dict `Normaliser`: learns its
    own mean/var online from the valid (non-padded) objects of every training
    batch, instead of reading a precomputed ``norm_dict.yaml``. No file I/O.

    Unlike ``nn.BatchNorm``, this layer ALWAYS applies the running buffers, even
    in train mode (``normed = (x - running_mean) / sqrt(running_var + eps)``);
    the running stats are updated from masked batch moments separately, only
    when training and off tracing. In eval/inference/ONNX this reduces to a
    frozen affine transform with no mask dependency. Produces NEW
    ``normed.<stream>`` keys; never mutates ``inputs.*``.

    The legacy ``norm_dict`` constructor arg is accepted for config
    compatibility but ignored (no file is ever read).
    """

    def __init__(
        self,
        streams: Sequence[str],
        global_object: str | None = None,
        norm_dict: str | Path | None = None,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
    ) -> None:
        """Capture config only (no file I/O here).

        Parameters
        ----------
        streams : Sequence[str]
            Streams to normalise.
        global_object : str | None, optional
            The stream that is a per-object vector (``[B, F]``) rather than
            a padded sequence (``[B, T, F]``), by default None. The global
            object has no pad mask — every row is valid.
        norm_dict : str | Path | None, optional
            DEPRECATED / IGNORED. Kept only for config compatibility, by default None.
        momentum : float | None, optional
            EMA momentum for the running-stat update (BatchNorm default ``0.1``):
            ``running = (1 - momentum) * running + momentum * batch``. Pass
            ``None`` for a cumulative moving average that converges to the true
            masked dataset mean/var, by default 0.1.
        eps : float, optional
            Added under the sqrt for numerical stability, by default 1e-5.

        Raises
        ------
        ConfigError
            If `streams` is empty, contains duplicates, `global_object` is
            not one of them, or `momentum`/`eps` are out of range.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("MaskedInputNormaliser: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"MaskedInputNormaliser: duplicate streams in {tuple(streams)}")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"MaskedInputNormaliser: global_object {global_object!r} is not in "
                f"streams {tuple(streams)}"
            )
        if momentum is not None and not 0.0 <= momentum <= 1.0:
            raise ConfigError(
                f"MaskedInputNormaliser: momentum must be None or in [0, 1], got {momentum}"
            )
        if eps <= 0.0:
            raise ConfigError(f"MaskedInputNormaliser: eps must be positive, got {eps}")
        # norm_dict is intentionally ignored (stats are learned online); kept in
        # the signature only so existing configs / CLI overrides still parse.
        del norm_dict
        self.streams = tuple(streams)
        self.global_object = global_object
        self.momentum = None if momentum is None else float(momentum)
        self.eps = float(eps)
        self._bound = False

    def _spec(self, stream: str) -> TensorSpec:
        """Build the shared spec for ``inputs.<stream>`` / ``normed.<stream>``.

        The last dim is the instance-scoped symbol ``F:<name>.<stream>`` on
        BOTH sides, so the concrete width declared by the dataset boundary
        propagates to ``normed.<stream>`` through unification (bind.py).

        Returns
        -------
        TensorSpec
            ``("B", F)`` for the global object, ``("B", "T:<stream>", F)``
            for sequence streams.
        """
        width = sym_dim("F", f"{self.name}.{stream}")
        shape: tuple[int | str, ...] = (
            ("B", width) if stream == self.global_object else ("B", _stream_len(stream), width)
        )
        return TensorSpec(shape=shape, dtype="float32")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` (+ training ``masks.<stream>``) -> ``normed.<stream>``.

        Sequence streams also require ``masks.<stream>`` (``True == padded``),
        but only as an OPTIONAL, training-mode port — the mask is read only
        when updating the running statistics, and a stream without a
        pad-mask producer treats every object as valid. Gating to
        ``Mode.TRAINING`` (rather than FIT alone) keeps TEST/ONNX free of any
        mask dependency, so the exported graph is a pure affine transform.
        """
        del mode
        requires: dict[str, TensorSpec] = {f"inputs.{s}": self._spec(s) for s in self.streams}
        for stream in self.streams:
            if stream == self.global_object:
                continue
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)),
                dtype="bool",
                kind="pad_mask",
                modes=Mode.TRAINING,
                optional=True,
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate the running-statistic buffers (identity init: mean 0 / var 1) from the schema.

        Raises
        ------
        RuntimeError
            If called twice (rebinding would discard learned values).
        """
        if self._bound:
            raise RuntimeError(
                f"MaskedInputNormaliser {self.name!r}: bind() called twice (design §2.3)"
            )
        for stream in self.streams:
            width = schema.width(f"inputs.{stream}")
            self.register_buffer(f"running_mean_{stream}", torch.zeros(width))
            self.register_buffer(f"running_var_{stream}", torch.ones(width))
            self.register_buffer(
                f"num_batches_tracked_{stream}", torch.zeros((), dtype=torch.long)
            )
            # total VALID-object count seen (cumulative-averaging path only)
            self.register_buffer(f"num_objects_seen_{stream}", torch.zeros((), dtype=torch.long))
        self._bound = True

    @staticmethod
    def _masked_moments(x: Tensor, valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-feature ``(sum, sumsq, count)`` over valid objects.

        `x` is ``[B, T, F]`` (sequence) or ``[B, F]`` (global); `valid` is the
        ``[B, T]`` keep-mask (``True == valid``), or None for the global
        object (all rows kept).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(sum_[F], sumsq_[F], count_scalar)``.
        """
        # valid is None for the global object (all rows valid); otherwise gather
        # the valid (non-padded) rows of the sequence stream.
        flat = x.reshape(-1, x.shape[-1]) if valid is None else x[valid]
        count = torch.tensor(flat.shape[0], dtype=x.dtype, device=x.device)
        return flat.sum(0), (flat * flat).sum(0), count

    @torch.no_grad()
    def _update_running_stats(self, stream: str, x: Tensor, valid: Tensor | None) -> None:
        """EMA-update the running stats for one stream from masked batch moments.

        All-reduces the per-feature sum/sumsq/count across DDP ranks before
        deriving the batch mean/var (never averaging per-rank mean/var
        directly, which is wrong when counts differ across ranks). Skips an
        all-padded (global count 0) stream.
        """
        s_sum, s_sumsq, count = self._masked_moments(x, valid)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.cat([s_sum, s_sumsq, count.reshape(1)])
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            f = s_sum.shape[0]
            s_sum, s_sumsq, count = packed[:f], packed[f : 2 * f], packed[2 * f]
        if count.item() == 0:
            return  # all-padded batch (global) — nothing valid to learn from
        batch_mean = s_sum / count
        # population (biased) variance over valid objects: E[x^2] - E[x]^2,
        # clamped to >= 0 against tiny float negatives.
        batch_var = (s_sumsq / count - batch_mean * batch_mean).clamp_min(0.0)
        getattr(self, f"num_batches_tracked_{stream}").add_(1)
        running_mean = getattr(self, f"running_mean_{stream}")
        running_var = getattr(self, f"running_var_{stream}")
        if self.momentum is None:
            # cumulative moving average: object-count-weighted parallel moment
            # combination (Chan et al.) — converges to the exact pooled mean/var
            seen = getattr(self, f"num_objects_seen_{stream}")
            n_old = seen.to(batch_mean.dtype)
            n_new = n_old + count
            delta = batch_mean - running_mean
            new_mean = running_mean + delta * (count / n_new)
            m_old = running_var * n_old
            m_new = batch_var * count
            new_var = (m_old + m_new + delta * delta * (n_old * count / n_new)) / n_new
            running_mean.copy_(new_mean)
            running_var.copy_(new_var)
            seen.add_(count.long())
        else:
            mom = self.momentum
            running_mean.mul_(1 - mom).add_(batch_mean, alpha=mom)
            running_var.mul_(1 - mom).add_(batch_var, alpha=mom)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply running-stat normalisation; update the stats in train mode.

        Always applies the frozen running buffers:
        ``normed.<stream> = (inputs.<stream> - running_mean) / sqrt(running_var + eps)``.
        The running stats update only when ``self.training``, the plan mode is
        training, and the graph is not being traced — so a TEST/ONNX plan
        never reads a mask even if the module is left in ``training=True``.

        Returns
        -------
        dict[str, Tensor]
            The newly produced ``normed.<stream>`` keys only.
        """
        updating = self.training and bool(mode & Mode.TRAINING) and not torch.jit.is_tracing()
        if updating:
            for stream in self.streams:
                x = b.get(f"inputs.{stream}")
                # global object has no pad mask; a sequence stream with a pad
                # mask gathers valid rows via ~pad_mask (mask True == padded);
                # a mask-less stream (optional port with no producer) treats
                # every object as valid (valid=None).
                mask_key = f"masks.{stream}"
                if stream == self.global_object or mask_key not in b:
                    valid = None
                else:
                    valid = ~b.get(mask_key)
                self._update_running_stats(stream, x, valid)
        out: dict[str, Tensor] = {}
        for stream in self.streams:
            mean = getattr(self, f"running_mean_{stream}")
            var = getattr(self, f"running_var_{stream}")
            out[f"normed.{stream}"] = (b.get(f"inputs.{stream}") - mean) / torch.sqrt(
                var + self.eps
            )
        return out


class StreamEmbed(nn.Module):
    """Config-constructed per-stream initial embedding, replacing v1 `InitNet`.

    Composes a fresh v1 `Dense` built at `bind`, with input width inferred
    from the resolved input and context widths. Context entries are
    PREPENDED in list order (``cat([context, x])``), so the feature layout is
    ``[ctx[-1], ..., ctx[0], stream]``.

    Stream rank is INFERRED from the bound input, not configured: a
    padded-sequence input ``[B, T, F]`` -> ``embed.<s> [B, T, D]``; a per-jet
    global input ``[B, F]`` -> ``embed.<s> [B, D]`` with no token axis. This is
    what makes DL1 a jets-only MLP (no encoder, no pooling).

    ``mup: true`` builds the internal `Dense` with ``mup=True`` at `bind`,
    applying the muP weight init; the forward math is otherwise unchanged.

    Optional, both OFF by default (unset => today's behaviour):

    - ``featurewise:`` — an input-layer `FeaturewiseTransformation` (FiLM):
      reads ``inputs.parameters`` and applies ``scale * x + bias`` to the
      embed INPUT before the `Dense` projection. When active, ``parameters``
      is NOT also concatenated as context.
    - ``pos_enc:`` — a `PositionalEncoder` ADDED to the embed OUTPUT (after
      projection), over the configured coordinate variables.
    """

    MUP_WIDTH_ARG = "out_dim"
    """The init_arg the muP shape-generation tooling (``salt2 mup-shapes``)
    sweeps for this module to produce the infshapes."""

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        context: Sequence[str] = (),
        input: str | None = None,  # noqa: A002 - YAML surface name
        mup: bool = False,
        featurewise: dict[str, Any] | None = None,
        pos_enc: dict[str, Any] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        stream : str
            The stream to embed.
        out_dim : int
            Output embedding width (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense`; must not contain width keys.
        context : Sequence[str], optional
            Dotted bundle keys attached as context, in prepend order (see
            class docstring), by default ``()``.
        input : str | None, optional
            Input key override, by default ``normed.<stream>``. The embed's
            rank is INFERRED from this bound input.
        mup : bool, optional
            Whether to use the muP parametrisation for the embed's internal
            `Dense`, by default False.
        featurewise : dict[str, Any] | None, optional
            Optional input-layer FiLM config (see class docstring): kwargs for
            `FeaturewiseTransformation` minus the width args. By default None.
        pos_enc : dict[str, Any] | None, optional
            Optional positional-encoding config (see class docstring): kwargs
            for `PositionalEncoder`. By default None.

        Raises
        ------
        ConfigError
            If `dense` configures widths (inferred at bind) or `out_dim` is not positive.
        """
        super().__init__()
        self.name = _UNNAMED
        if out_dim < 1:
            raise ConfigError(f"StreamEmbed: out_dim must be >= 1, got {out_dim}")
        _reject_width_keys("StreamEmbed", dense, ("input_size", "output_size", "context_size"))
        if "mup" in (dense or {}):
            raise ConfigError(
                "StreamEmbed: set mup on the module (init_args.mup), not inside dense — the flag "
                "is threaded into the composed v1 Dense at bind (design §3.4 muP architectural "
                "port)"
            )
        self.stream = stream
        self.out_dim = out_dim
        self.dense_cfg = dict(dense or {})
        self.context = tuple(context)
        self.input_key = input if input is not None else f"normed.{stream}"
        self.mup = bool(mup)
        self.net: nn.Module | None = None
        # -- optional input-layer FiLM ---------------------------------------
        self.featurewise_cfg = dict(featurewise) if featurewise is not None else None
        self.params_key = "inputs.parameters"
        self.featurewise: FeaturewiseTransformation | None = None
        if self.featurewise_cfg is not None:
            self.params_key = self.featurewise_cfg.pop("parameters", self.params_key)
            if self.featurewise_cfg.get("layer", "input") != "input":
                raise ConfigError(
                    "StreamEmbed featurewise: layer must be 'input' (the embed is the input-layer "
                    "FiLM site; use TransformerEncoder for encoder/global layers)"
                )
            self.featurewise_cfg["layer"] = "input"
        # -- optional positional encoding ------------------------------------
        self.pos_enc_cfg = dict(pos_enc) if pos_enc is not None else None
        self.pos_enc: PositionalEncoder | None = None
        self.pos_enc_indices: tuple[int, ...] = ()

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + context keys -> ``embed.<stream>``.

        Rank-AGNOSTIC: the input require and the ``embed.<s>`` produce both
        declare ``shape=None``, so the embed inherits its rank from the bound
        input. The ``out_dim`` width is contributed at bind via `derived_widths`.
        """
        del mode
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(shape=None, dtype="float32"),
        }
        for key in self.context:
            # rank/width unconstrained: context may be a [B, F] global vector
            # or broadcastable — widths resolve from the producer side
            requires[key] = TensorSpec(shape=None, dtype="float32")
        if self.featurewise_cfg is not None:
            # the per-event conditioning parameters: a rank-2 [B, n_params] global stream
            requires[self.params_key] = TensorSpec(shape=("B", sym_dim("P", self.name)), dtype="float32")
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(shape=None, dtype="float32"),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute the ``embed.<stream>`` last-dim width (``out_dim``, a config constant)."""
        del widths
        return {f"embed.{self.stream}": self.out_dim}

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` (``input_size = width(input) + sum(width(ctx))``).

        Also builds the optional input-layer FiLM (sized from ``parameters``
        width and the embed input width) and resolves the optional positional
        encoder's variable column indices from the input's declared fields.
        """
        input_size = schema.width(self.input_key) + sum(schema.width(key) for key in self.context)
        self.net = Dense(
            input_size=input_size, output_size=self.out_dim, mup=self.mup, **self.dense_cfg
        )
        if self.featurewise_cfg is not None:
            self.featurewise = FeaturewiseTransformation(
                num_params=schema.width(self.params_key),
                num_features=input_size,
                **self.featurewise_cfg,
            )
            self.featurewise.name = self.name
            self.featurewise.build()
        if self.pos_enc_cfg is not None:
            cfg = dict(self.pos_enc_cfg)
            cfg.setdefault("dim", self.out_dim)
            self.pos_enc = PositionalEncoder(**cfg)
            self.pos_enc.name = self.name
            # resolve the coordinate column indices by NAME from the input fields
            fields = schema.fields_of(self.input_key)
            try:
                self.pos_enc_indices = tuple(fields.index(v) for v in self.pos_enc.variables)
            except ValueError as err:
                raise ConfigError(
                    f"StreamEmbed {self.name!r} pos_enc: variable not found in {self.input_key!r} "
                    f"fields {list(fields)}: {err}"
                ) from None

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Attach context (prepended) and project; optional FiLM/pos-enc are no-ops when unset."""
        del mode
        x = b.get(self.input_key)
        raw = x  # the raw input columns (pos_enc reads these at the variable idx)
        for key in self.context:
            x = attach_context(x, b.get(key))  # cat([context, x])
        if self.featurewise is not None:
            # FiLM on the embed INPUT, before the projection
            x = self.featurewise(b.get(self.params_key), x)
        assert self.net is not None, "forward before bind()"
        out = self.net(x)
        if self.pos_enc is not None:
            # ADD the positional encoding to the embed OUTPUT
            out = out + self.pos_enc(raw[..., self.pos_enc_indices])
        return {f"embed.{self.stream}": out}


class Concat(nn.Module):
    """Concatenate embedded streams into one sequence.

    Produces ``seq.x`` / ``seq.mask`` / ``seq.layout``; the concat ORDER is
    the configured list and nothing else. ``seq.layout`` is a dict-valued
    meta leaf ``{stream: (start, stop)}`` over the pre-register sequence,
    consumed by `Split`.

    Registers live in `TransformerEncoder`, not here (the composed v1
    `Transformer` appends its own register tokens internally and requires
    ``num_registers >= 1``), so nonzero ``registers`` is rejected.
    """

    def __init__(self, streams: Sequence[str], registers: int = 0) -> None:
        """Capture the explicit stream order.

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates, or if `registers`
            is nonzero (registers live in `TransformerEncoder`, see class docstring).
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Concat: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Concat: duplicate streams in {tuple(streams)}")
        if registers != 0:
            raise ConfigError(
                "Concat: registers are internal to TransformerEncoder in M2 (the composed v1 "
                "Transformer appends them, transformer.py:679-681) — set "
                "encoder num_registers instead; Concat-owned registers land at M7 (design §5.1)"
            )
        self.streams = tuple(streams)
        self.registers = registers

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``embed.*``/``masks.*`` per stream -> seq keys.

        All streams share one instance-scoped embed-width symbol — equal
        widths are a genuine concat constraint. In ONNX mode an additional
        ``seq.offsets`` int64 tensor is produced (the trace-safe stream
        boundary table consumed by `Split`'s export branch).
        """
        del mode
        embed_dim = sym_dim("E", self.name)
        requires: dict[str, TensorSpec] = {}
        for stream in self.streams:
            requires[f"embed.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream), embed_dim), dtype="float32"
            )
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)), dtype="bool", kind="pad_mask"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, embed_dim), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                "seq.layout": TensorSpec(kind="meta"),
                "seq.offsets": TensorSpec(
                    shape=(len(self.streams) + 1,), dtype="int64", modes=Mode.ONNX
                ),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute the ``seq.x`` last-dim width from the per-stream embed widths.

        `StreamEmbed` produces ``embed.<stream>`` with ``shape=None`` (rank
        inferred from the bound input), so there is no concrete shape for the
        dim table to bind ``embed_dim`` from directly — this hook forwards the
        resolved embed width to ``seq.x`` (needed on the encoderless-pool path,
        which has no encoder require to bind it otherwise). The per-stream
        embeds share one width, so the FIRST resolved input width is used.

        Returns
        -------
        dict[str, int]
            ``{"seq.x": embed_width}`` once an ``embed.<stream>`` width is
            resolved, otherwise ``{}``.
        """
        for stream in self.streams:
            width = widths.get(f"embed.{stream}")
            if width is not None:
                return {"seq.x": width}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Any]:
        """Concatenate streams along the token dim and record the layout.

        In ONNX mode the per-stream token counts are ALSO published as the
        ``seq.offsets`` cumulative-boundary tensor, built with
        ``torch.onnx.operators.shape_as_tensor`` so the boundaries trace to
        Shape/Concat/CumSum nodes that stay symbolic under tracing.
        """
        xs = [b.get(f"embed.{stream}") for stream in self.streams]
        masks = [b.get(f"masks.{stream}") for stream in self.streams]
        layout: dict[str, tuple[int, int]] = {}
        start = 0
        for stream, x in zip(self.streams, xs, strict=True):
            layout[stream] = (start, start + x.shape[1])
            start += x.shape[1]
        out: dict[str, Any] = {
            "seq.x": torch.cat(xs, dim=1),
            "seq.mask": torch.cat(masks, dim=1),
            "seq.layout": layout,
        }
        if mode & Mode.ONNX:
            lengths = [torch.onnx.operators.shape_as_tensor(mask)[1:2] for mask in masks]
            zero = torch.zeros(1, dtype=torch.int64)
            out["seq.offsets"] = torch.cumsum(torch.cat([zero, *lengths]), dim=0)
        return out


class VectorConcat(nn.Module):
    """Ordered concatenation of ``[B, D_i]`` vectors into one ``[B, Dsum]`` key.

    Replaces v1's post-pooling ``'global'`` magic key: GN3's 2-feature
    ``global`` stream is fed past the encoder and concatenated onto the pooled
    representation (``pooled_dim = 256 + 2``, the width every GN3 task
    consumes). Concat ORDER is the config list — for converted GN3
    checkpoints, ``inputs: [pooled.global, normed.global]`` (pooled first)
    matches v1's task first-layer weight layout. Output width ``Dsum =
    sum(D_i)`` is resolved at bind via `derived_widths`. ONNX uses the
    ``export.inputs`` ``alias:`` mechanism (`OnnxAdapter`), not this module;
    VectorConcat itself is mode-agnostic.
    """

    def __init__(self, inputs: Sequence[str], out: str = "pooled.global") -> None:
        """Capture the explicit ordered input list and the output key.

        Parameters
        ----------
        inputs : Sequence[str]
            Dotted bundle keys to concatenate, in the EXACT order they appear
            in the output. Must be non-empty with no duplicates.
        out : str, optional
            The produced concatenated key, by default ``"pooled.global"`` (the
            v1 ``global_rep`` slot every GN3 task reads).

        Raises
        ------
        ConfigError
            If `inputs` is empty, contains duplicates, or names `out` itself
            (a self-feed).
        """
        super().__init__()
        self.name = _UNNAMED
        if not inputs:
            raise ConfigError("VectorConcat: inputs must be a non-empty sequence")
        if len(set(inputs)) != len(tuple(inputs)):
            raise ConfigError(f"VectorConcat: duplicate inputs in {tuple(inputs)}")
        if out in inputs:
            raise ConfigError(
                f"VectorConcat: out {out!r} appears in inputs {tuple(inputs)} — a module cannot "
                "consume its own output (design §2.1 write-once)"
            )
        self.inputs = tuple(inputs)
        self.out_key = out

    def declare_io(self, mode: Mode) -> IO:
        """Declare each ``[B, D_i]`` input -> the ``[B, Dsum]`` output.

        Each input gets its OWN instance-scoped width symbol (inputs are
        genuinely different widths, so they must NOT share a symbol); the
        output's ``Dsum`` is resolved at bind via `derived_widths`.
        """
        del mode
        requires: dict[str, TensorSpec] = {
            key: TensorSpec(shape=("B", sym_dim(f"D{i}", self.name)), dtype="float32")
            for i, key in enumerate(self.inputs)
        }
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", sym_dim("Dsum", self.name)), dtype="float32"),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute ``Dsum = sum(D_i)`` once every input width is resolved.

        Returns
        -------
        dict[str, int]
            ``{out: sum(widths[input])}`` when every input width is resolved,
            otherwise ``{}``.
        """
        if all(key in widths for key in self.inputs):
            return {self.out_key: sum(widths[key] for key in self.inputs)}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Concatenate the inputs along the feature dim, in the configured order."""
        del mode
        return {self.out_key: torch.cat([b.get(key) for key in self.inputs], dim=-1)}


class EdgeFeatures(nn.Module):
    """Config-constructed pairwise edge-feature builder.

    Requires the RAW (un-normalised) ``inputs.<stream>`` ``[B, T, F]`` plus
    ``masks.<stream>``, and produces ``edges.<stream>`` ``[B, T, T, E]`` where
    ``E = len(features)``. Produces a NEW key; never mutates ``inputs.*``.

    The per-element math (dR/kt/z/subjetIndex/isSelfLoop/mass) is byte-faithful
    to v1's inlined `calculate_edge_features`/`check_edge_config`. The
    ``indices_map`` (variable name -> column index) is resolved at `bind` from
    the resolved schema's declared fields — column lookups resolve by NAME,
    never by YAML list position.

    ONNX: the produced tensor carries the SAME ``T:<stream>`` symbol on both
    token axes, so both trace as dynamic; the math is all
    unsqueeze/expand/elementwise ops driven by the dynamic token count.
    """

    def __init__(
        self,
        stream: str,
        features: Sequence[str],
        out: str | None = None,
        input: str | None = None,  # noqa: A002 - YAML surface name
    ) -> None:
        """Capture config only (no schema/data access here).

        Parameters
        ----------
        stream : str
            The stream whose pairwise edges are built (``inputs.<stream>``).
        features : Sequence[str]
            Edge feature names, in produced column order — any of
            ``{"dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass"}``.
            Required input variables are checked at `bind` against the
            resolved fields.
        out : str | None, optional
            Produced edge key, by default ``edges.<stream>``.
        input : str | None, optional
            Raw input key override, by default ``inputs.<stream>``.

        Raises
        ------
        ConfigError
            If `features` is empty, has duplicates, or names an unrecognised
            edge feature.
        """
        super().__init__()
        self.name = _UNNAMED
        if not features:
            raise ConfigError("EdgeFeatures: features must be a non-empty sequence (design §6.7)")
        if len(set(features)) != len(tuple(features)):
            raise ConfigError(f"EdgeFeatures: duplicate features in {tuple(features)}")
        unknown = [f for f in features if f not in _EDGE_FEATURES]
        if unknown:
            raise ConfigError(
                f"EdgeFeatures: unrecognised edge feature(s) {unknown} — choose from "
                f"{sorted(_EDGE_FEATURES)} (v1 check_edge_config, edge_features.py:30-43)"
            )
        self.stream = stream
        self.features = tuple(features)
        self.out_key = out if out is not None else f"edges.{stream}"
        self.input_key = input if input is not None else f"inputs.{stream}"
        self.indices_map: dict[str, int] | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw ``inputs.<stream>`` + ``masks.<stream>`` -> ``edges.<stream>``.

        The produced edge tensor is ``[B, T, T, E]`` with the SAME
        ``T:<stream>`` symbol on both token axes (a square pairwise matrix).
        """
        del mode
        tlen = _stream_len(self.stream)
        edge_dim = len(self.features)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", tlen, sym_dim("F", self.name)), dtype="float32"
                ),
                f"masks.{self.stream}": TensorSpec(
                    shape=("B", tlen), dtype="bool", kind="pad_mask"
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", tlen, tlen, edge_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Resolve the variable-name -> column-index map and validate features.

        ``check_edge_config`` validates that every feature's required
        variables are present (e.g. ``dR`` needs ``eta``/``phi``).
        """
        fields = schema.fields_of(self.input_key)
        check_edge_config(list(self.features), list(fields))
        self.indices_map = {name: i for i, name in enumerate(fields)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute the pairwise edge features on the RAW input.

        Returns a FRESH tensor; never mutates ``inputs.*``.
        """
        del mode
        assert self.indices_map is not None, "forward before bind()"
        x = b.get(self.input_key)
        edges = calculate_edge_features(x, self.indices_map, list(self.features))
        return {self.out_key: edges}


class EdgeEmbed(nn.Module):
    """Config-constructed edge-feature embedding.

    An edge-typed `StreamEmbed`: maps ``edges.<stream>`` ``[B, T, T, E]`` ->
    ``edges.<stream>_emb`` ``[B, T, T, D_e]`` with an internal v1 `Dense` (an
    ``nn.Linear`` stack over the last dim, so it embeds each pairwise edge
    independently). No context, no muP, no per-stream rank inference — edges
    are always the rank-4 pairwise matrix.

    ONNX: both ``T:<stream>`` token axes flow through unchanged (dynamic).

    Carries no FiLM/positional-encoding, faithfully to v1: those live on
    `StreamEmbed` for the constituent streams and on `TransformerEncoder` for
    the encoder/global FiLM.
    """

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        input: str | None = None,  # noqa: A002 - YAML surface name
        out: str | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        stream : str
            The stream whose edge tensor is embedded.
        out_dim : int
            Output edge-embedding width ``D_e`` (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense`; must not contain width keys.
        input : str | None, optional
            Edge input key override, by default ``edges.<stream>``.
        out : str | None, optional
            Produced embedded edge key, by default ``edges.<stream>_emb``.

        Raises
        ------
        ConfigError
            If `dense` configures widths (inferred at bind) or `out_dim` is not positive.
        """
        super().__init__()
        self.name = _UNNAMED
        if out_dim < 1:
            raise ConfigError(f"EdgeEmbed: out_dim must be >= 1, got {out_dim}")
        _reject_width_keys("EdgeEmbed", dense, ("input_size", "output_size", "context_size"))
        self.stream = stream
        self.out_dim = out_dim
        self.dense_cfg = dict(dense or {})
        self.input_key = input if input is not None else f"edges.{stream}"
        self.out_key = out if out is not None else f"edges.{stream}_emb"
        self.net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``edges.<stream>`` -> ``edges.<stream>_emb`` (``[B,T,T,E] -> [B,T,T,D_e]``)."""
        del mode
        tlen = _stream_len(self.stream)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", tlen, tlen, sym_dim("E", self.name)), dtype="float32"
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", tlen, tlen, self.out_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` with ``input_size = width(edges.<stream>) = E``."""
        self.net = Dense(
            input_size=schema.width(self.input_key), output_size=self.out_dim, **self.dense_cfg
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Project the edge tensor through the internal `Dense` ([B,T,T,E] -> [B,T,T,D_e])."""
        del mode
        assert self.net is not None, "forward before bind()"
        return {self.out_key: self.net(b.get(self.input_key))}


class TransformerEncoder(nn.Module):
    """Config-constructed transformer encoder, composing a fresh v1 `Transformer`.

    Registers, packing, and the out projection stay INTERNAL; the register
    pad mask is published as the NEW key ``masks.registers`` — the caller's
    mask dict is never mutated.

    ``drop_registers`` (the MaskFormer encoder passthrough) strips the
    register rows from the output AFTER the layer stack runs — registers are
    still appended internally and visible to every attention layer (so a
    constituent-less jet has something to attend to), but the returned
    ``encoded.seq`` is sliced back to the stream tokens and no
    ``masks.registers`` key is produced.

    ``norm_type`` (``"pre"`` default, or ``"post"`` / ``"hybrid"``) is
    forwarded verbatim to every composed v1 ``EncoderLayer``.

    muP — the ``mup: true`` flag builds the composed v1 `Transformer` with
    ``mup=True``. This is NARROWER than the name suggests: v1's
    ``Transformer(mup=True)`` does NOT propagate ``mup`` down to its
    EncoderLayers, so the ONLY effect is the out-proj swap to
    ``mup.MuReadout`` (weight+bias zeroed). The attention softmax scale, the
    attention muP init, and the GLU/dense muP init are all NOT engaged by
    this flag — those live on ``Attention(mup=True)``/``Dense(mup=True)``
    directly (see `StreamEmbed`). This is a faithful port of v1's actual
    (narrower) encoder-mup behaviour, not a bug.

    At export (`set_export_mode`), a `MuReadout` out-proj is FOLDED into a
    plain `nn.Linear` with the output multiplier baked into the weights —
    numerically equal to the MuReadout forward — so the exported graph is a
    plain transformer.

    Optional FiLM: the ``featurewise:`` config is a LIST of
    `FeaturewiseTransformation` configs, each with ``layer: encoder`` (one per
    encoder layer, applied at the start of each layer) or ``layer: global``
    (applied to the encoder output before it's published). OFF by default.
    """

    _NORM_TYPES = ("pre", "post", "hybrid")
    """The encoder-layer norm placements the wrapper forwards. ``"none"`` is a
    residual-only v1 mode with no shipped v2 config — rejected loudly here."""

    MUP_WIDTH_ARG = "dim"
    """The init_arg the muP shape-generation tooling (``salt2 mup-shapes``)
    sweeps for this module to produce the infshapes."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        attention: dict[str, Any],
        out_dim: int | None = None,
        dense: dict[str, Any] | None = None,
        norm: str = "LayerNorm",
        num_registers: int = 1,
        norm_type: str = "pre",
        drop_registers: bool = False,
        mup: bool = False,
        edges: str | None = None,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        featurewise: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Build the composed v1 `Transformer` from config.

        Parameters
        ----------
        dim : int
            Embedding width of ``seq.x``.
        num_layers : int
            Number of encoder layers.
        attention : dict[str, Any]
            Attention config; MUST contain ``num_heads``. ``attn_type``
            (default ``"torch-math"``) selects the backend.
        out_dim : int | None, optional
            Output projection width, by default None (= `dim`, no projection).
        dense : dict[str, Any] | None, optional
            v1 ``dense_kwargs`` (``activation``, ``gated``, ...), by default None.
        norm : str, optional
            Normalisation layer name, by default ``"LayerNorm"``.
        num_registers : int, optional
            Learned register tokens appended INSIDE the encoder, by default 1 (v1 minimum).
        norm_type : str, optional
            Per-layer norm placement, one of ``{"pre", "post", "hybrid"}``, by
            default ``"pre"``. ``"hybrid"`` forces ``do_qk_norm``/``do_v_norm``
            and applies a pre-FFN norm; the placement logic lives in the
            composed v1 layer.
        drop_registers : bool, optional
            Strip the register rows from ``encoded.seq`` after the layer stack
            (see the class docstring), by default False.
        mup : bool, optional
            Whether to use the muP parametrisation (see the class docstring
            for the narrower-than-expected scope), by default False. Requires
            an out projection (``out_dim`` set).
        edges : str | None, optional
            The edge-embed bundle key the encoder consumes (e.g.
            ``"edges.tracks_emb"``). When set, every composed v1
            ``EncoderLayer`` swaps its `Attention` for an `EdgeAttention`. By
            default None (no edge path).
        edge_embed_dim : int, optional
            The edge-embed width ``D_e``. REQUIRED (positive) when `edges` is
            set (cross-checked against the resolved edge-embed width at
            bind); must be 0 when `edges` is None. By default 0.
        update_edges : bool, optional
            Whether the encoder updates the edge tensor each layer. Requires
            `edges` set. The updated edges stay INTERNAL to the encoder — only
            ``encoded.seq`` is published. By default False.

        Raises
        ------
        ConfigError
            If `attention` is missing ``num_heads``, `norm_type` is not one of
            ``{"pre", "post", "hybrid"}``, `mup` is set without an `out_dim`,
            `edges` is set without a positive `edge_embed_dim` (or vice versa),
            or `update_edges` is set without `edges`.
        """
        super().__init__()
        self.name = _UNNAMED
        if not isinstance(attention, Mapping) or "num_heads" not in attention:
            raise ConfigError(
                "TransformerEncoder: attention config must be a mapping containing 'num_heads' "
                "(v1 Transformer requires attn_kwargs, transformer.py:600-601)"
            )
        if norm_type not in self._NORM_TYPES:
            raise ConfigError(
                f"TransformerEncoder: norm_type must be one of {self._NORM_TYPES}, got "
                f"{norm_type!r}"
            )
        if mup and out_dim is None:
            raise ConfigError(
                "TransformerEncoder: mup requires an out_dim — the MuReadout out-proj is the last "
                "muP layer of the model and has no layer to live on without one "
                "(v1 transformer.py:594-597)"
            )
        # edges <-> edge_embed_dim must be set together: EncoderLayer picks
        # EdgeAttention iff edge_embed_dim > 0, and update_edges needs an edge
        # tensor to update. Reject inconsistent combinations at config time.
        if (edges is None) != (edge_embed_dim <= 0):
            raise ConfigError(
                "TransformerEncoder: 'edges' and 'edge_embed_dim' must be set together — "
                f"got edges={edges!r}, edge_embed_dim={edge_embed_dim}. Set both for an edge "
                "encoder (v1 GN2XE.yaml:79 edge_embed_dim with an edge_init_net), or neither "
                "(FD §6.7 1422-1424)"
            )
        if update_edges and edges is None:
            raise ConfigError(
                "TransformerEncoder: update_edges requires an 'edges' port — there is no edge "
                "tensor to update without one (v1 transformer.py:589-590, GN2XE.yaml:80)"
            )
        attn_kwargs = dict(attention)
        attn_type = attn_kwargs.pop("attn_type", "torch-math")
        self.dim = dim
        self.norm_type = norm_type
        self.drop_registers = bool(drop_registers)
        self.mup = bool(mup)
        self.edges_key = edges
        self.edge_embed_dim = int(edge_embed_dim)
        self.update_edges = bool(update_edges)
        self.encoder = Transformer(
            num_layers=num_layers,
            embed_dim=dim,
            out_dim=out_dim,
            norm=norm,
            attn_type=attn_type,
            do_final_norm=True,
            num_registers=num_registers,
            drop_registers=self.drop_registers,
            edge_embed_dim=self.edge_embed_dim,
            update_edges=self.update_edges,
            attn_kwargs=attn_kwargs,
            dense_kwargs=dict(dense) if dense is not None else None,
            norm_type=norm_type,
            mup=self.mup,
        )
        if self.mup:
            # MuReadout.forward/width_mult() assert infshape is set. Set base shapes
            # from the module onto itself (rescale_params=False) so a standalone mup
            # encoder is forward-runnable/traceable without a real shape file; import
            # locally to avoid a hard mup dependency for non-mup encoders.
            from mup import set_base_shapes  # noqa: PLC0415

            set_base_shapes(self.encoder, self.encoder, rescale_params=False)
        self.out_dim = self.encoder.out_dim
        self.num_registers = num_registers
        self.num_layers = int(num_layers)
        self.params_key = "inputs.parameters"
        self._encoder_film_cfg: dict[str, Any] | None = None
        self._global_film_cfg: dict[str, Any] | None = None
        self.featurewise_global: FeaturewiseTransformation | None = None
        for fw in featurewise or ():
            fw = dict(fw)
            layer = fw.get("layer")
            # one params key is shared by all FiLM entries; take it from any entry
            pk = fw.pop("parameters", None)
            if pk is not None:
                self.params_key = pk
            if layer == "encoder":
                if self._encoder_film_cfg is not None:
                    raise ConfigError(
                        "TransformerEncoder featurewise: at most one 'encoder'-layer FiLM entry "
                        "(v1 replicates ONE config across all encoder layers, saltmodel.py:268-269)"
                    )
                self._encoder_film_cfg = fw
            elif layer == "global":
                if self._global_film_cfg is not None:
                    raise ConfigError(
                        "TransformerEncoder featurewise: at most one 'global'-layer FiLM entry"
                    )
                self._global_film_cfg = fw
            elif layer == "input":
                raise ConfigError(
                    "TransformerEncoder featurewise: layer 'input' belongs on StreamEmbed "
                    "(featurewise:), not the encoder — only 'encoder'/'global' here"
                )
            else:
                raise ConfigError(
                    f"TransformerEncoder featurewise: each entry needs layer in "
                    f"{{'encoder', 'global'}}, got {layer!r}"
                )

    @property
    def edge_stream(self) -> str | None:
        """The stream the edge port belongs to, or None when no edge path.

        ``"edges.tracks_emb"`` -> ``"tracks"``. Used by the bind-time
        edge-stream-first validator to check the edge stream is
        `Concat.streams[0]`.
        """
        if self.edges_key is None:
            return None
        # "edges.<stream>_emb" -> "<stream>": strip the namespace + _emb suffix
        leaf = self.edges_key.split(KEY_SEP, 1)[1] if KEY_SEP in self.edges_key else self.edges_key
        return leaf.removesuffix("_emb")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` (+ ``edges.<stream>_emb``) -> ``encoded.seq``.

        ``masks.registers`` is produced ONLY when ``drop_registers`` is False —
        with registers dropped there are no register rows left to mask, and
        `GlobalAttentionPooling` treats ``masks.registers`` as OPTIONAL.

        When an `edges` port is configured, the encoder additionally REQUIRES
        the edge-embed tensor ``[B, T, T, D_e]`` whose BOTH token axes share
        the stream's ``T:<stream>`` symbol. Updated edges stay INTERNAL — this
        module still produces only ``encoded.seq`` (+ optional register mask).
        """
        del mode
        produces: dict[str, TensorSpec] = {
            "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, self.out_dim), dtype="float32"),
        }
        if not self.drop_registers:
            produces["masks.registers"] = TensorSpec(
                shape=("B", self.num_registers), dtype="bool", kind="pad_mask"
            )
        requires: dict[str, TensorSpec] = {
            "seq.x": TensorSpec(shape=("B", _SEQ_LEN, self.dim), dtype="float32"),
            "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
        }
        if self.edges_key is not None:
            stream = self.edge_stream
            assert stream is not None  # edges_key implies edge_stream (config invariant)
            tlen = _stream_len(stream)
            # both token axes share T:<stream> — the dynamic-T export prerequisite
            requires[self.edges_key] = TensorSpec(
                shape=("B", tlen, tlen, self.edge_embed_dim), dtype="float32"
            )
        if self._encoder_film_cfg is not None or self._global_film_cfg is not None:
            # per-event conditioning parameters feeding the encoder/global FiLM
            requires[self.params_key] = TensorSpec(
                shape=("B", sym_dim("P", self.name)), dtype="float32"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec(produces),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Cross-check the edge-embed width against the configured ``edge_embed_dim``.

        The composed v1 `Transformer` already built its `EdgeAttention`
        projections from ``edge_embed_dim`` at ``__init__``, so this only
        VALIDATES that the resolved edge-embed width matches — a mismatch is a
        named `ConfigError` here rather than a silent runtime shape error.
        No-op when no edge port is configured.

        Also builds the optional encoder/global FiLM: per-layer encoder FiLMs
        are sized to the encoder embed width and populated into the absorbed
        `Transformer`'s ``featurewise`` ModuleList; the global FiLM is sized to
        the encoder output width.

        Raises
        ------
        ConfigError
            When the resolved edge-embed width differs from ``edge_embed_dim``.
        """
        if self._encoder_film_cfg is not None or self._global_film_cfg is not None:
            num_params = schema.width(self.params_key)
            if self._encoder_film_cfg is not None:
                # one FiLM per encoder layer, replicating the same config across all layers
                for _ in range(self.num_layers):
                    film = FeaturewiseTransformation(
                        num_params=num_params, num_features=self.dim, **self._encoder_film_cfg
                    )
                    film.name = self.name
                    film.build()
                    self.encoder.featurewise.append(film)
            if self._global_film_cfg is not None:
                self.featurewise_global = FeaturewiseTransformation(
                    num_params=num_params, num_features=self.out_dim, **self._global_film_cfg
                )
                self.featurewise_global.name = self.name
                self.featurewise_global.build()
        if self.edges_key is None:
            return
        resolved = schema.width(self.edges_key)
        if resolved != self.edge_embed_dim:
            raise ConfigError(
                f"TransformerEncoder {self.name!r}: edge_embed_dim={self.edge_embed_dim} but the "
                f"resolved {self.edges_key!r} width is {resolved} — set edge_embed_dim to the "
                "EdgeEmbed out_dim (the encoder's EdgeAttention projections were sized from "
                "edge_embed_dim at construction, attention.py:535; FD §6.7 1422-1424)"
            )

    def set_export_mode(self) -> None:
        """Prepare the encoder for tracing: torch-math backend + MuReadout fold.

        `MuReadout`'s forward applies an output multiplier before the linear —
        a non-``nn.Linear`` op that traces to an unsupported/incorrect graph.
        `_fold_mu_readout` swaps it for a plain `nn.Linear` with the
        multiplier baked into the weights (numerically equal to the MuReadout
        forward). Idempotent — a second call no-ops.
        """
        # EdgeAttention has no pluggable backend — it is ALWAYS raw torch
        # attention (set_backend just warns) and already trace-safe; only
        # switch the backend for the non-edge encoder
        if self.edges_key is None:
            self.encoder.set_backend("torch-math")
        if self.mup:
            self._fold_mu_readout()

    def _fold_mu_readout(self) -> None:
        """Fold the composed v1 `MuReadout` out-proj into a plain `nn.Linear` for export.

        No-op unless the encoder has a `MuReadout` out projection (a non-mup
        or already-folded encoder is left untouched — idempotent).
        """
        from mup import MuReadout  # noqa: PLC0415

        proj = getattr(self.encoder, "out_proj", None)
        if not isinstance(proj, MuReadout):
            return  # non-mup / no out-proj / already folded — nothing to do
        # output_mult and width_mult scale only the linear term, not the bias
        # (MuReadout.forward: super().forward(output_mult * x / width_mult)).
        mult = float(proj.output_mult) / float(proj.width_mult())
        has_bias = proj.bias is not None
        folded = nn.Linear(proj.in_features, proj.out_features, bias=has_bias)
        with torch.no_grad():
            folded.weight.copy_(proj.weight * mult)
            if has_bias:
                folded.bias.copy_(proj.bias)
        folded.to(proj.weight.device, proj.weight.dtype)
        folded.eval()
        self.encoder.out_proj = folded

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Encode the sequence; publish the register mask as a NEW key.

        With ``drop_registers`` only ``encoded.seq`` is produced (no
        ``masks.registers``, gated out in `declare_io`). When an `edges` port
        is configured, the edge-embed tensor is passed as the ``edge_x``
        kwarg; the per-layer edge update stays internal — only ``encoded.seq``
        (+ optional register mask) is published.
        """
        del mode
        # FRESH dicts: _add_registers INSERTS a "REGISTERS" key into both —
        # never hand it bundle-owned dicts.
        xs: dict[str, Tensor] = {"seq": b.get("seq.x")}
        pad: dict[str, Tensor] = {"seq": b.get("seq.mask")}
        kwargs: dict[str, Tensor] = {}
        if self.edges_key is not None:
            kwargs["edge_x"] = b.get(self.edges_key)
        if len(self.encoder.featurewise) > 0:
            # encoder-layer FiLM: thread the per-event parameters into the absorbed
            # Transformer forward, which applies featurewise[i](params, x) per layer
            kwargs["inputs"] = b.get(self.params_key)
        encoded, out_pad = self.encoder(xs, pad_mask=pad, **kwargs)
        if self.featurewise_global is not None:
            # global-layer FiLM: scale/bias the encoder OUTPUT before it is pooled
            encoded = self.featurewise_global(b.get(self.params_key), encoded)
        if self.drop_registers:
            # registers stripped from encoded.seq; no register mask to publish
            return {"encoded.seq": encoded}
        return {"encoded.seq": encoded, "masks.registers": out_pad["REGISTERS"]}


class Split(nn.Module):
    """Per-stream slices of ``encoded.seq`` via the ``seq.layout`` meta leaf.

    Register rows sit AFTER every stream in the encoder output, so the
    pre-register layout offsets remain valid slices of ``encoded.seq``.

    ONNX export uses dynamic ``index_select`` slicing (via `Concat`'s
    ``seq.offsets`` tensor) rather than the eager branch's Python-int
    ``seq.layout`` offsets: with >=2 dynamic sequence axes, tracing bakes
    Python-int offsets as constants, which is silently WRONG once a second
    stream is present (verified empirically with >=2 dynamic axes producing
    mis-sliced output at every probed grid point).
    """

    def __init__(self, streams: Sequence[str]) -> None:
        """Capture the streams to slice out.

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Split: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Split: duplicate streams in {tuple(streams)}")
        self.streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``encoded.seq`` + ``seq.layout`` -> ``encoded.<stream>`` per stream.

        In ONNX mode the `Concat` ``seq.offsets`` boundary tensor is
        additionally required (the trace-safe slicing path, see class docstring).
        """
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, width), dtype="float32"),
                "seq.layout": TensorSpec(kind="meta"),
                "seq.offsets": TensorSpec(shape=None, dtype="int64", modes=Mode.ONNX),
            }),
            produces=unflatten_spec({
                f"encoded.{stream}": TensorSpec(
                    shape=("B", _stream_len(stream), width), dtype="float32"
                )
                for stream in self.streams
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Slice each configured stream out of the encoded sequence.

        ONNX mode uses dynamic ``index_select`` slicing driven by
        ``seq.offsets`` (see class docstring); the stream's position in the
        offsets table is its position in the ``seq.layout`` dict, so a
        `Split` over a stream SUBSET stays correct without the full concat list.
        """
        encoded = b.get("encoded.seq")
        layout = b.get("seq.layout")
        out: dict[str, Tensor] = {}
        if mode & Mode.ONNX:
            offsets = b.get("seq.offsets")
            order = list(layout)
            for stream in self.streams:
                i = order.index(stream)
                idx = torch.arange(offsets[i + 1] - offsets[i]) + offsets[i]
                out[f"encoded.{stream}"] = encoded.index_select(1, idx)
            return out
        for stream in self.streams:
            start, stop = layout[stream]
            out[f"encoded.{stream}"] = encoded[:, start:stop]
        return out


class GlobalAttentionPooling(nn.Module):
    """Config-constructed global attention pooling, with explicit input/out ports.

    Composes a fresh v1 `GlobalAttentionPooling` built at `bind`. Pools the
    register-augmented sequence with the post-register mask order
    streams-then-REGISTERS. Two wirings, one class:

    - **With an encoder**: ``input`` is ``encoded.seq`` and `TransformerEncoder`
      publishes ``masks.registers`` (register rows sit after every stream).
    - **Encoder-less** (DiPS/DeepSets family, every regression config): no
      ``encoder:`` block, so nothing produces ``masks.registers``. ``input``
      points at ``seq.x`` and ``masks.registers`` is declared OPTIONAL — absent
      from the plan when no producer exists, so the pad dict is just
      ``{"seq": seq.mask}``.
    """

    def __init__(self, input: str = "encoded.seq", out: str = "pooled.global") -> None:  # noqa: A002
        """Capture the explicit input/output ports."""
        super().__init__()
        self.name = _UNNAMED
        self.input_key = input
        self.out_key = out
        self.pool_net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + masks -> the pooled vector.

        The input's width symbol is shared with the produced key, so the
        pooled width resolves from the producing module's declaration.
        """
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", sym_dim("L", self.name), width), dtype="float32"
                ),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                # OPTIONAL: produced by `TransformerEncoder` on the WITH-encoder path,
                # ABSENT on the encoder-less path — the planner drops it when no
                # module produces it, so encoder-less configs still plan-compile.
                "masks.registers": TensorSpec(
                    shape=("B", sym_dim("R", "registers")),
                    dtype="bool",
                    kind="pad_mask",
                    optional=True,
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", width), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed v1 pooling with the inferred gate width."""
        self.pool_net = _GlobalAttentionPoolingV1(input_size=schema.width(self.input_key))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the sequence with the (optionally register-augmented) mask dict."""
        del mode
        assert self.pool_net is not None, "forward before bind()"
        x = {"seq": b.get(self.input_key)}
        # streams-then-REGISTERS dict order is numerics-critical: pooling cats
        # mask values in dict order. The encoder-less path has no register row,
        # so the pad dict is just {"seq": seq.mask}.
        pad = {"seq": b.get("seq.mask")}
        if "masks.registers" in b:
            pad["REGISTERS"] = b.get("masks.registers")
        return {self.out_key: self.pool_net(x, pad_mask=pad)}


class LossSum(nn.Module):
    """Weighted sum of per-task losses -> ``loss.total``.

    Per-task weights are applied INSIDE the tasks, so the default here is a
    plain sum; `weights` is an extra per-loss-key multiplier.

    The ``losses.**`` auto-collection is a framework wildcard: since the
    kernel rejects wildcard *requires*, narrowing happens framework-side —
    `collect_loss_keys` scans sibling modules' declared produces and `narrow`
    fixes the concrete key list before plan compilation. An explicit
    ``losses:`` config list skips collection entirely.
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; ``losses=None`` defers to framework narrowing.

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected via `collect_loss_keys`/`narrow`).
        weights : Mapping[str, float] | None, optional
            Per-loss multipliers keyed like `losses`, default 1.0 each.
        """
        super().__init__()
        self.name = _UNNAMED
        self._loss_keys: tuple[str, ...] | None = (
            tuple(_loss_key(k) for k in losses) if losses is not None else None
        )
        self.weights = {_loss_key(k): float(v) for k, v in (weights or {}).items()}
        if self._loss_keys is not None:
            self._check_weight_keys()

    def _check_weight_keys(self) -> None:
        """Reject weight entries that name no summed loss key.

        Raises
        ------
        ConfigError
            Naming the unknown weight keys and the known loss keys.
        """
        assert self._loss_keys is not None
        if unknown := sorted(set(self.weights) - set(self._loss_keys)):
            raise ConfigError(
                f"LossSum: weights for unknown loss keys {unknown} — summed keys are "
                f"{list(self._loss_keys)}"
            )

    @property
    def narrowed(self) -> bool:
        """Whether the loss-key list is fixed (explicit config or `narrow`)."""
        return self._loss_keys is not None

    @staticmethod
    def collect_loss_keys(
        modules: Mapping[str, GraphModule], mode: Mode = Mode.FIT
    ) -> tuple[str, ...]:
        """Scan sibling modules (LossSum instances skipped) for declared ``losses.*`` produces.

        Returns
        -------
        tuple[str, ...]
            All declared loss keys, in module-dict declaration order.
        """
        keys: list[str] = []
        for module in modules.values():
            if isinstance(module, LossSum):
                continue
            for key, spec in flatten_spec(module.declare_io(mode).produces).items():
                if key.startswith("losses.") and spec.active_in(mode):
                    keys.append(key)
        return tuple(keys)

    def narrow(self, loss_keys: Iterable[str]) -> None:
        """Fix the concrete loss-key list (framework-side ``losses.**`` narrowing).

        Raises
        ------
        ConfigError
            If explicit ``losses:`` config already fixed the keys, the list
            is empty, or a configured weight names no key.
        """
        if self._loss_keys is not None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys already fixed to {list(self._loss_keys)}"
            )
        keys = tuple(_loss_key(k) for k in loss_keys)
        if not keys:
            raise ConfigError(
                f"LossSum {self.name!r}: narrowed to an empty loss-key list — no module "
                "declares a losses.* produce (design §3.3)"
            )
        self._loss_keys = keys
        self._check_weight_keys()

    def declare_io(self, mode: Mode) -> IO:
        """Declare the narrowed loss keys -> ``loss.total`` (TRAINING only, empty in TEST/ONNX).

        Raises
        ------
        ConfigError
            If the loss keys were never fixed (no ``losses:`` config and no
            `narrow` call).
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})
        if self._loss_keys is None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys not fixed — pass losses: in config or let "
                "the framework narrow via collect_loss_keys()/narrow() before compile "
                "(losses.** is a framework wildcard, design §3.3)"
            )
        loss_spec = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        return IO(
            requires=unflatten_spec(dict.fromkeys(self._loss_keys, loss_spec)),
            produces=unflatten_spec({"loss.total": loss_spec}),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Sum the (optionally weighted) loss leaves -> ``{"loss.total": scalar}``."""
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        total = sum(self.weights.get(key, 1.0) * b.get(key) for key in self._loss_keys)
        return {"loss.total": total}


class LossGLS(LossSum):
    """Geometric-mean (GLS) combination of per-task losses -> ``loss.total``.

    Subclasses `LossSum` to share the ``losses.**`` framework wildcard,
    `declare_io`, and the `collect_loss_keys`/`narrow` integration. The ONLY
    behavioural change is the combination rule (`forward`): the n-task
    geometric mean ``(∏ losses)^(1/n)``.

    GLS does NOT utilise loss weights — the geometric mean of per-task losses
    is only meaningful when no task is pre-scaled (a weighted task loss
    ``w*L`` would contribute ``w^(1/n)`` to the product, an arbitrary rescale
    of the mean). Two weight surfaces are both guarded to 1.0: this module's
    per-loss ``weights`` (rejected in ``__init__``), and the task-side
    ``weight`` applied inside each task head (checked via
    `check_task_weights`, called by `SaltModule.__init__` before any plan
    compiles, since this module can't see sibling task state at construction).
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; reject any per-loss weight (GLS ignores weights).

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected, as for `LossSum`).
        weights : Mapping[str, float] | None, optional
            Accepted only for parity with the `LossSum` signature: any entry
            != 1.0 is rejected.

        Raises
        ------
        ConfigError
            If any configured weight is not 1.0 (GLS ignores weights — set
            them to 1, or use `LossSum` for a weighted sum).
        """
        super().__init__(losses=losses, weights=weights)
        # exact == 1.0 is the faithful semantic; weights are config literals,
        # never computed values
        if bad := {k: v for k, v in self.weights.items() if v != 1.0}:  # noqa: RUF069
            raise ConfigError(
                f"LossGLS: per-loss weights are not utilised by the geometric mean — got "
                f"{bad}; set all weights to 1.0, or use LossSum for a weighted sum "
                "(v1 modelwrapper.py:139-142)"
            )

    @staticmethod
    def check_task_weights(modules: Mapping[str, GraphModule]) -> None:
        """Assert every loss-producing task carries ``weight == 1.0``.

        Called by `SaltModule.__init__` when a `LossGLS` is present, BEFORE
        any `declare_io`/compile, so a weighted task under GLS fails loudly at
        assembly rather than silently rescaling the geometric mean. Modules
        without a numeric ``weight`` attribute (`Normaliser`, `Concat`,
        `LossSum`/`LossGLS`, ...) are ignored — only the loss producers carry it.

        Raises
        ------
        ConfigError
            Naming each task whose ``weight`` is not 1.0.
        """
        offenders = {
            name: float(module.weight)
            for name, module in modules.items()
            if not isinstance(module, LossSum)
            # duck-typed numeric check: LossSum carries `weights` (a dict), not
            # `weight`, and is excluded above regardless
            and isinstance(getattr(module, "weight", None), (int, float))
            and float(module.weight) != 1.0  # noqa: RUF069 - exact, the v1 semantic
        }
        if offenders:
            raise ConfigError(
                f"LossGLS: GLS does not utilise task weights — set all task weights to 1.0, "
                f"got {offenders} (v1 modelwrapper.py:139-142)"
            )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Combine the loss leaves by their geometric mean: ``(∏ losses)^(1/n)``."""
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        product = math.prod(b.get(key) for key in self._loss_keys)
        return {"loss.total": torch.pow(product, 1.0 / len(self._loss_keys))}


def _loss_key(key: str) -> str:
    """Normalise a configured loss reference to a dotted ``losses.`` key."""
    return key if key.startswith("losses.") else f"losses.{key}"


def _reject_width_keys(who: str, cfg: Mapping[str, Any] | None, banned: tuple[str, ...]) -> None:
    """Reject configured width keys — widths are inferred at bind.

    Raises
    ------
    ConfigError
        Naming the offending keys.
    """
    if cfg and (bad := sorted(set(cfg) & set(banned))):
        raise ConfigError(
            f"{who}: dense config must not set {bad} — widths are inferred at bind from the "
            "resolved schema (design §2.3 kills YAML width arithmetic)"
        )
