"""Standalone, config-constructed GraphModules for the GN2v2 surface (design §5.1, §9.2).

M2 porting policy (plan 05): these modules are constructed from plain config
kwargs — no live v1 instances, no ``input_size`` arithmetic in YAML — but may
COMPOSE fresh v1 layer classes internally where that is faithful (full code
absorption is M7). What is new here is the lifecycle (design §2.3):

``__init__`` captures config only; ``declare_io`` is static; ``bind(schema)``
builds width-dependent layers from the `ResolvedSchema`; ``materialise()`` is
the only place file I/O happens (Normaliser values). Forward passes read
declared bundle keys and return ONLY newly produced keys — bundle values are
never mutated (write-once, design §2.1; the executor's debug mode enforces
it).

Key vocabulary follows design §5.1: ``inputs.<stream>`` -> ``normed.<stream>``
-> ``embed.<stream>`` -> ``seq.x``/``seq.mask``/``seq.layout`` ->
``encoded.seq`` (+ ``masks.registers``) -> ``encoded.<stream>`` (`Split`) /
``pooled.global``; task modules live in `salt.core.nn.tasks`.

Documented M2 deviations from design §5.1 (both honest, both encoder-driven):

- **Registers stay inside `TransformerEncoder`** (its composed v1
  `Transformer` appends them, transformer.py:679-681, and *requires*
  ``num_registers >= 1``). `Concat` accepts the design's ``registers:`` arg
  but rejects nonzero values until M7 absorbs the encoder.
- **`Normaliser` takes an explicit ``streams:`` list** instead of
  demand-driven ``normed.*`` narrowing: the M1 kernel supports wildcard
  *produces* for framework modules, but the matching ``inputs.<s>``
  *requires* cannot be narrowed (planner.py rejects wildcard requires).
  TODO(M3+): demand-driven narrowing once the kernel grows consumer-side
  framework wildcards (same gap as `LossSum.collect_loss_keys`).

Symbolic-dim convention: token-count dims are stream-scoped and shared
(``"T:tracks"``); feature-width dims are *instance-scoped*
(``"D:<instance>"``) so two modules' unknown widths never falsely unify —
`resolve_bind_schema` still resolves them through the specs observed on the
same key (bind.py).
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

# v2-NATIVE absorption of the v1 layer/Dense/pooling family (M7 W2c-2). The
# v1 ``Dense`` (dense.py), the v1 ``Transformer`` stack + its transitive deps
# (transformer.py / attention.py / layernorm.py) and the v1
# ``GlobalAttentionPooling`` (pooling.py) are COPIED VERBATIM below as
# self-contained v2-native classes (``Dense``, ``Transformer`` &c.,
# ``_GlobalAttentionPoolingV1``) — the v1 originals stay UNTOUCHED in
# salt/models/* as the bitwise gate ORACLE (parity_gn2 / gates_m6 MU1/MU2/ED1/ED2
# weight-load a fresh v1 instance from these absorbed modules' state_dicts and
# compare forwards bitwise, so the absorbed structure + math are byte-faithful by
# construction). The small pure math helpers (``attach_context`` +
# ``calculate_edge_features`` / ``check_edge_config``) were inlined at M7 W2b
# BYTE-FAITHFULLY from v1 ``salt.utils.tensor_utils`` (tensor_utils.py:168-268)
# and v1 ``salt.utils.edge_features`` (edge_features.py:10-146); the four
# tensor-dict / varlen helpers the absorbed Transformer + pooling need
# (``masked_softmax`` / ``flatten_tensor_dict`` / ``undo_padding`` /
# ``redo_padding``) are inlined likewise from tensor_utils.py:30-165.

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
"""Placeholder instance name — the config dict key is assigned before compile (design §2.2)."""

_SEQ_LEN = sym_dim("S", "seq")
_ENC_LEN = sym_dim("L", "enc")

_EDGE_FEATURES = ("dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass")
"""The recognised edge-feature vocabulary (v1 `check_edge_config`,
edge_features.py:30-43; design §6.7). EdgeFeatures rejects anything outside this
set at config time; the per-feature required input variables (e.g. ``dR`` needs
``eta``/``phi``) are validated at `bind` against the resolved schema fields."""


# ---------------------------------------------------------------------------
# inlined v1 math helpers (M7 W2b) — byte-faithful copies, v1 originals untouched
# ---------------------------------------------------------------------------


def add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dimensions to reach a target rank.

    M7 W2b inline of v1 ``salt.utils.tensor_utils.add_dims`` (tensor_utils.py:168-197),
    byte-faithful. The new singleton dimensions are inserted after the batch
    dimension (i.e., at position 1 repeatedly) until ``x.ndim == ndim``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ndim : int
        Target number of dimensions.

    Returns
    -------
    Tensor
        Tensor reshaped with added singleton dimensions.

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
    """Concatenate a context tensor to a single tensor with broadcasting.

    M7 W2b inline of v1 ``salt.utils.tensor_utils.attach_context_single``
    (tensor_utils.py:200-240), byte-faithful. The ``context`` tensor is expanded
    (via :func:`add_dims` and broadcast) so its rank matches ``x``; it is then
    concatenated with ``x`` along the last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(B, ..., F)``.
    context : Tensor
        Context tensor of shape ``(B, F_ctx)`` or broadcastable to
        ``(B, ..., F_ctx)``.

    Returns
    -------
    Tensor
        Concatenation of ``context`` and ``x`` along the feature dimension,
        with shape ``(B, ..., F_ctx + F)``.

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
    """Concatenate a context tensor to inputs (tensor or dict of tensors).

    M7 W2b inline of v1 ``salt.utils.tensor_utils.attach_context``
    (tensor_utils.py:243-268), byte-faithful. A convenience wrapper over
    :func:`attach_context_single` that applies the operation to either a single
    tensor or every tensor in a dictionary.

    Parameters
    ----------
    x : Tensor | dict[str, Tensor]
        Input tensor or dictionary of tensors to which the context will be
        concatenated along the last dimension.
    context : Tensor
        Context tensor of shape ``(B, F_ctx)`` (or broadcastable to each
        input).

    Returns
    -------
    Tensor | dict[str, Tensor]
        If ``x`` is a tensor, returns a tensor with context concatenated.
        If ``x`` is a dict, returns a dict with each value concatenated
        with the context.
    """
    if isinstance(x, dict):
        return {key: attach_context_single(val, context) for key, val in x.items()}
    return attach_context_single(x, context)


def check_edge_config(
    edge_features: list[str],
    available_vars: list[str],
) -> None:
    """Check the provided edge feature configuration for validity.

    M7 W2b inline of v1 ``salt.utils.edge_features.check_edge_config``
    (edge_features.py:10-49), byte-faithful.

    Parameters
    ----------
    edge_features : list[str]
        List of edge features to compute.
    available_vars : list[str]
        List of available variables.

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
    """Calculate edge features for a given batch of graphs.

    M7 W2b inline of v1 ``salt.utils.edge_features.calculate_edge_features``
    (edge_features.py:52-146), byte-faithful — the pairwise dR/kt/z/subjetIndex/
    isSelfLoop/mass math, ``torch.zeros`` accumulator + final ``nan_to_num``.

    Parameters
    ----------
    batch : Tensor
        Input batch of node features of shape ``[B, N, D]``.
    indices_map : dict[str, int]
        Mapping variable names to indices in the node feature tensor.
    variables : list[str]
        List of edge features to compute.

    Returns
    -------
    Tensor
        Computed edge features tensor of shape ``[B, N, N, num_edge_features]``.
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
    """Flatten (concatenate) a dictionary of tensors into one tensor.

    M7 W2c-2 inline of v1 ``salt.utils.tensor_utils.flatten_tensor_dict``
    (tensor_utils.py:30-71), byte-faithful. All tensors are concatenated along
    ``dim=1``; ``include``/``exclude`` are mutually exclusive subset selectors.

    Returns
    -------
    Tensor
        Single tensor formed by concatenating the selected tensors along ``dim=1``.

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
    """Apply softmax while ignoring (masking) padded elements.

    M7 W2c-2 inline of v1 ``salt.utils.tensor_utils.masked_softmax``
    (tensor_utils.py:74-105), byte-faithful. Elements where ``mask`` is ``True``
    are set to ``-inf`` before the softmax and zeroed after.

    Returns
    -------
    Tensor
        Tensor after masked softmax.
    """
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def undo_padding(seq: Tensor, mask: BoolTensor) -> tuple[Tensor, Tensor, int]:
    """Remove padded elements and return packed sequence info.

    M7 W2c-2 inline of v1 ``salt.utils.tensor_utils.undo_padding``
    (tensor_utils.py:108-141), byte-faithful — the flash-varlen packer. Convention
    ``mask == True`` -> padded element; the mask is flipped internally.

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
    """Re-apply padding to an unpadded sequence.

    M7 W2c-2 inline of v1 ``salt.utils.tensor_utils.redo_padding``
    (tensor_utils.py:144-165), byte-faithful.

    Returns
    -------
    Tensor
        Padded tensor (zeros at padded positions, values at valid positions).
    """
    mask = ~mask  # convert mask: True -> valid token
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out


# ===========================================================================
# v2-native absorption of the v1 Dense / Transformer / pooling family (M7 W2c-2)
# ---------------------------------------------------------------------------
# The classes below are COPIED VERBATIM (math + attribute layout + parameter
# registration order) from the v1 originals so a fresh v1 instance can
# ``load_state_dict`` a v2-native module's composed sub-net (the gates_m6
# MU1/MU2/ED1/ED2 oracle pattern) and the forwards agree BITWISE. The v1
# originals (salt/models/dense.py, transformer.py, attention.py, layernorm.py,
# pooling.py) stay UNTOUCHED as the gate oracle.
# ===========================================================================


class Dense(nn.Module):
    """A fully connected feed forward neural network, with optional context.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.Dense`` (dense.py:6-103),
    COPIED VERBATIM — same layer list (``net``), same muP init
    (``_reset_parameters``), same ``attach_context`` forward. The v1 original is
    UNTOUCHED as the gate oracle (gates_m6 VS1/ED1 weight-load a fresh v1 Dense
    from this absorbed Dense's ``state_dict()`` and compare bitwise).

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

        # Save the networks input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.mup = mup

        # build nodelist
        self.node_list = [input_size + context_size, *hidden_layers, output_size]

        # input and hidden layers
        layers = []

        num_layers = len(self.node_list) - 1
        for i in range(num_layers):
            if dropout:
                layers.append(nn.Dropout(dropout))

            # linear projection
            layers.append(nn.Linear(self.node_list[i], self.node_list[i + 1], bias=bias))

            # activation for all but the final layer
            if i != num_layers - 1:
                layers.append(getattr(nn, activation)())

            # final layer: return logits by default, or activation if specified
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        # build the net
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
    """Faster LayerNorm by setting elementwise_affine=False.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.layernorm.LayerNorm``
    (layernorm.py:5-9), VERBATIM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)


class RMSNorm(torch.nn.Module):
    """RMSNorm from https://arxiv.org/abs/1910.07467.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.layernorm.RMSNorm``
    (layernorm.py:12-27), VERBATIM — follows the LLaMA implementation.
    """

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
    """Namespace mirroring v1 ``import salt.models.layernorm as layernorms``.

    The v1 EncoderLayer / Transformer resolve the norm class by NAME
    (``getattr(layernorms, norm)``, transformer.py:223,355,371). This namespace
    exposes the absorbed `LayerNorm` / `RMSNorm` under the same attribute names so
    the verbatim ``getattr(_LAYERNORMS, norm)`` lookups below are byte-faithful.
    """

    LayerNorm = LayerNorm
    RMSNorm = RMSNorm


_LAYERNORMS = _Layernorms()


# ---------------------------------------------------------------------------
# featurewise.py + posenc.py absorption (M7 W-FILM) — v2-native FiLM + posenc
# ---------------------------------------------------------------------------

_FEATUREWISE_LAYERS: frozenset[str] = frozenset({"input", "encoder", "global"})
"""The v1 FiLM ``layer`` placements (featurewise.py:44). ``input`` applies the
scale/bias BEFORE a `StreamEmbed`'s projection (initnet.py:85-86); ``encoder``
applies it at the START of every encoder layer (transformer.py:727-728);
``global`` applies it to the pooled/encoded representation before pooling
(saltmodel.py:165-166). All three are wired as OPTIONAL config blocks (off by
default)."""

_POSENC_SYM_VARS: frozenset[str] = frozenset({"phi"})
"""v1 `PositionalEncoder` ``SYM_VARS`` (posenc.py:4) — the variables whose
encoding is symmetric (a sin/cos of the sin/cos), faithful to v1."""


class FeaturewiseTransformation(nn.Module):
    """Feature-wise (FiLM) scale/bias from per-event ``parameters`` (M7 W-FILM).

    v2-native absorption of v1 ``salt.models.FeaturewiseTransformation``
    (featurewise.py:8-81), COPIED VERBATIM (the v1 original stays UNTOUCHED as the
    FILM1 gate oracle). The internal scale/bias nets are `salt.core.nn.Dense` (the
    v2-native absorbed Dense, byte-identical to v1's so a weight-transfer between
    them is exact) — NOT v1's ``salt.models.Dense``. The conditioning signal is
    the per-event ``parameters`` tensor ``[B, n_params]`` read from the bundle
    (``inputs.parameters``, the raw — NOT normalised — parameters, faithful to v1
    where ``parameters`` is in ``InputNorm.NO_NORM``, inputnorm.py:45).

    Forward (featurewise.py:71-81, verbatim math):
    ``features = scale_net(p).unsqueeze(1) * features`` then
    ``features = features + bias_net(p).unsqueeze(1)`` then optional `LayerNorm`.
    The ``unsqueeze(1)`` broadcasts the per-event ``[B, num_features]`` scale/bias
    over the token axis of a ``[B, T, num_features]`` feature tensor.

    https://distill.pub/2018/feature-wise-transformations/.

    Parameters
    ----------
    layer : str
        Which pipeline stage to scale/bias — one of ``{"input", "encoder",
        "global"}`` (featurewise.py:44; see ``_FEATUREWISE_LAYERS``).
    num_params : int
        Number of per-event conditioning parameters (the FiLM net input width;
        v1 inferred it from ``len(variables["parameters"])``, featurewise.py:55).
    num_features : int
        Output width of the FiLM scale/bias nets — the feature width the
        transformation is applied to (the embed/encoder width). Threaded in at
        bind from the resolved schema (v1 read it off the Dense ``output_size``,
        featurewise.py:57,61).
    dense_config_scale : dict | None, optional
        Extra `salt.core.nn.Dense` kwargs for the scaling net (``hidden_layers``,
        ``activation``, ...); must not set width keys. When None (and
        ``dense_config_bias`` is set) the FiLM applies a bias-only transform. By
        default None.
    dense_config_bias : dict | None, optional
        Extra `salt.core.nn.Dense` kwargs for the biasing net. When None (and
        ``dense_config_scale`` is set) the FiLM applies a scale-only transform. By
        default None.
    apply_norm : bool, optional
        Apply a `torch.nn.LayerNorm` to the transformed features, by default
        False.

    Raises
    ------
    ConfigError
        If `layer` is not one of ``{"input", "encoder", "global"}``, if a
        dense config sets width keys (inferred at bind), or if neither scale
        nor bias net is configured.
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
        # v1 builds a net iff the corresponding dense config is TRUTHY
        # (featurewise.py:54 ``if dense_config_scale:`` / :58) — so a None or an
        # empty {} config builds nothing. Mirror v1 exactly.
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
        """Construct the scale/bias `Dense` nets + optional norm (idempotent).

        Mirrors v1 ``FeaturewiseTransformation.__init__`` (featurewise.py:54-69)
        verbatim, but ``input_size``/``output_size`` come from the captured
        ``num_params``/``num_features`` (resolved at bind) instead of from
        ``variables``. Called once at bind; a second call no-ops so the parity
        gate can rebuild idempotently.
        """
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
        """Apply the FiLM scale/bias to ``features`` (featurewise.py:71-81 verbatim).

        Parameters
        ----------
        params : Tensor
            The per-event conditioning parameters ``[B, n_params]`` (the bundle's
            ``inputs.parameters`` — v1 ``inputs["parameters"]``).
        features : Tensor
            The features to transform ``[B, T, num_features]`` (or ``[B,
            num_features]`` for the global layer).

        Returns
        -------
        Tensor
            The scaled/biased (and optionally normed) features — a FRESH tensor
            (the multiply/add allocate new tensors), never aliasing ``features``.
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
    """Sin/cos positional encoding over coordinate variables (M7 W-FILM).

    v2-native absorption of v1 ``salt.models.posenc.PositionalEncoder``
    (posenc.py:7-85), COPIED VERBATIM MINUS the v1 ``print()`` debug lines
    (posenc.py:30,33,56) — the encoding math is the parity-bearing part and is
    byte-faithful. The v1 original stays UNTOUCHED as the FILM1 gate oracle.

    Evenly shares the embedding space between the encoded variables; any
    remaining dimensions are left as zeros. The ``@torch.no_grad`` forward and the
    sin/cos order are faithful to v1 (the encoding is a fixed, parameter-free
    function of the coordinates).

    Parameters
    ----------
    variables : Sequence[str]
        Variable names to encode (the coordinate columns). Symmetric variables
        (``phi``, ``_POSENC_SYM_VARS``) get the sin-of-sin / sin-of-cos symmetric
        encoding (posenc.py:79-81).
    dim : int
        Total positional-encoding width. Split evenly: ``per_input_dim = dim //
        (2 * len(variables))`` per variable, with ``dim % (2 * len(variables))``
        trailing zeros (posenc.py:31-32).
    alpha : int, optional
        Frequency scaling factor, by default 100 (posenc.py:25).

    Raises
    ------
    ConfigError
        If `variables` is empty or `dim` is too small to give each variable at
        least one frequency band (``per_input_dim < 1``).
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
        """Encode each coordinate column; concat along the last dim (posenc.py:36-57).

        Parameters
        ----------
        inputs : Tensor
            Coordinate tensor ``[..., len(variables)]`` (the selected columns, in
            ``variables`` order).

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
        """One variable's sin/cos encoding (posenc.py:59-85, verbatim).

        Parameters
        ----------
        xs : Tensor
            One coordinate column ``[...]``.
        dim : int
            Per-variable half-width (``per_input_dim``).
        symmetric : bool, optional
            Symmetric (phi-style) encoding, by default False.

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.check_flash_attn``
    (attention.py:29-59), VERBATIM.

    Returns
    -------
    str
        Empty string if Flash Attention is available, otherwise a reason why not.
    """
    # 1. Check CUDA Availability
    if not torch.cuda.is_available():
        return "No GPU available."

    # 2. Get CUDA & GPU Info
    gpu_name = torch.cuda.get_device_name(0)
    # Compute capability is a tuple (major, minor), e.g., (8, 0)
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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.merge_masks``
    (attention.py:65-112), VERBATIM. Padded tokens can't **send** information but
    can **receive** it (prevents softmax NaNs).

    Returns
    -------
    BoolTensor | None
        Combined mask of shape ``[B, 1, L_q, L_kv]`` (broadcastable over heads), or ``None``.
    """
    mask = None

    # If the kv_mask exists, ensure padded tokens never send information
    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # convert the mask so that True indicates a valid token

    # Combine with the explicit attention mask if present
    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask & mask

    # Unsqueeze for head broadcasting
    if mask is not None:
        mask = mask.unsqueeze(1)

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int) -> tuple[Tensor, Tensor]:
    """Repeat keys and values along a dimension.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.repeat_kv``
    (attention.py:115-136), VERBATIM.

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.projection_packed``
    (attention.py:139-176), VERBATIM — uses ``chunk`` (faster than ``unflatten``).

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.torch_attn``
    (attention.py:179-220), VERBATIM.

    Returns
    -------
    Tensor
        Attention output of shape ``[B, H, L_q, D_h]``.
    """
    backends = [SDPBackend.MATH]  # Default backend
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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.Attention``
    (attention.py:223-462), COPIED VERBATIM (same packed in-proj parameters,
    muP init, RMSNorm q/k/v norms, backend dispatch).

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

        # Attributes
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

        # Better parallelism for self-attention when using parameters directly
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
        # Check the attention backend
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

        # Standard init
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
        # Perform the packed input projection
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

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out = self._flash_attn(qkv, culens, maxlen, dropout, softmax_scale=self.scale)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Mix with final linear layer
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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.attention.EdgeAttention``
    (attention.py:465-658), COPIED VERBATIM — the edge-bias / edge-gate / optional
    edge-update math (consumes the W2b-inlined edge features). The v1 original is
    the bitwise oracle for gates_m6 ED1/ED2.

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

        # Attributes
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

        # Better parallelism for self-attention when using parameters directly
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Edge feature projections
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

        # Standard init
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)

        # Linear layers
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
        """Attention using PyTorch SDPA backends.

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

        s_mask = mask if kv is None else kv_mask  # Who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        e = self.linear_e(edge_x)  # (B, L_q, L_kv, num_heads)
        g = functional.sigmoid(self.linear_g(edge_x))  # (B, L_q, L_kv, num_heads)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, L_q, L_kv)
        attn_scores = attn_scores + e.permute(0, 3, 1, 2)  # add edge embeddings

        if self.dropout > 0.0 and self.training:
            attn_scores = functional.dropout(attn_scores, p=self.dropout)

        # Prepare edge output
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
    """Dense update with a (gated) linear unit.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.GLU``
    (transformer.py:53-125), VERBATIM. See https://arxiv.org/abs/2002.05202.

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.LayerScale``
    (transformer.py:128-151), VERBATIM. Reference: https://arxiv.org/abs/2103.17239
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
    """Stochastic depth / drop-path regularization.

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.DropPath``
    (transformer.py:154-180), VERBATIM.
    """

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.NormResidual``
    (transformer.py:183-283), VERBATIM. Represents PostNorm/PreNorm/NoNorm
    patterns and forwards edge features for `EdgeAttention`.

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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.EncoderLayer``
    (transformer.py:286-425), VERBATIM — the hybrid-norm placement logic, the
    Attention/EdgeAttention selection, and the residual submodule layout.

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

        # Safe defaults
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        # Attributes
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

        # Choose attention type
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

        # Submodules
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

    M7 W2c-2 v2-native absorption of v1 ``salt.models.transformer.Transformer``
    (transformer.py:511-789), COPIED VERBATIM — the register tokens, the muP
    ``MuReadout`` out-proj swap (the M6 coord-check canonical slope), the
    featurewise ``ModuleList`` hook, and the register/edge zero-pad forward. The
    v1 original is the bitwise oracle for gates_m6 MU1/ED1 (weight-loaded from
    this absorbed encoder's ``state_dict()``).

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

        # Check the inputs
        if num_registers < 1:
            raise ValueError(
                "Some global objects (graphs) might have no constituents (nodes), "
                "which causes NaNs in the attention scores. "
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

        # Set the attention type if no edge features are used
        if edge_embed_dim == 0:
            kwargs["attn_kwargs"]["attn_type"] = self.attn_type

        # Submodules
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

        # Only set the attention type if no edge features are used
        if self.edge_embed_dim == 0:
            # Check and set the attention type
            assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
            self.set_backend(self.attn_type)

        # Optional submodules
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
        # Add the registers to the sequence and the mask
        if self.num_registers:
            x, pad_mask = self._add_registers(x, pad_mask)

        # Combine the input sequences if they are dictionaries (don't overwrite pad_mask)
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # Pad edges by num_registers if using edge features
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

        # If using the varlen backend, pack the sequence and store the cumulative lengths
        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        # Run through the main transformer encoder layers
        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            if edge_x is not None:
                x, edge_x = layer(x, edge_x=edge_x, mask=mask, **kwargs)
            else:
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

    def _add_registers(
        self, x: Tensor | dict[str, Tensor], pad_mask: BoolTensor | dict[str, BoolTensor] | None
    ) -> tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]:
        """Add the learnable registers to the end of the input sequence (and mask).

        Returns
        -------
        tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]
            Updated ``(x, pad_mask)`` including appended registers.
        """
        # Get the batch size and expand the registers to match
        batch_size = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)

        # Add as a key or concatenate at the end
        reg = self.registers.expand(batch_size, -1, -1)
        if isinstance(x, dict):
            x["REGISTERS"] = reg
        else:
            x = torch.cat([x, reg], dim=1)

        # Also include a mask for the registers
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

    M7 W2c-2 v2-native absorption of v1
    ``salt.models.pooling.GlobalAttentionPooling`` (pooling.py:12-65), COPIED
    VERBATIM — the dict-order mask concatenation (numerics-critical, pooling.py:56)
    and the zero-token ONNX pad. Named with the ``V1`` suffix because the
    config-facing v2 GraphModule below is also called `GlobalAttentionPooling`;
    this is the inner ``nn.Module`` it composes (the same composition the M2 port
    used, now self-contained). The v1 original is the bitwise pooling oracle.

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
    """Return the shared symbolic token-count dim for a sequence stream.

    Returns
    -------
    str
        The symbolic dim, e.g. ``"T:tracks"``.
    """
    return sym_dim("T", stream)


class Normaliser(nn.Module):
    """Config-constructed input normalisation (design §6.3, replaces v1 `InputNorm`).

    This is the DEFAULT input normaliser: it loads a precomputed
    ``norm_dict.yaml`` (fixed per-stream, per-variable ``{mean, std}``) and
    preserves v1 forward parity. For an opt-in self-normalising variant that
    learns its statistics online (no norm dict), select
    ``class_path: salt.core.nn.MaskedInputNormaliser`` instead.

    Lifecycle (design §2.3): ``__init__`` records the ``norm_dict`` *path*
    and the stream list — no file I/O. ``bind(schema)`` allocates per-stream
    buffers (``means_<stream>`` / ``stds_<stream>``, the design §6.3 names)
    sized from the resolved ``inputs.<stream>`` widths, and captures the
    declared variable names. ``materialise()`` is the ONLY file-touching
    hook: it loads the norm dict and fills the buffers — skipped on
    checkpoint load, where values arrive via the state_dict (the
    ``materialised`` flag buffer travels with them).

    Unlike v1's ``InputNorm.forward`` (which rebinds its input dict's keys
    in place, inputnorm.py:103-106), this module produces NEW
    ``normed.<stream>`` keys and never mutates ``inputs.*`` (design §2.1).
    Because ``normed.*`` and ``inputs.*`` are distinct keys, consumers of
    raw inputs (edge features) need no ordering hack (design §6.3).
    """

    def __init__(
        self,
        norm_dict: str | Path,
        streams: Sequence[str],
        global_object: str | None = None,
    ) -> None:
        """Capture config only (design §2.3 — no file I/O here).

        Parameters
        ----------
        norm_dict : str | Path
            Path to the normalisation dictionary YAML; read at
            `materialise`, never here.
        streams : Sequence[str]
            Streams to normalise (explicit M2 surface — see the module
            docstring for the demand-driven TODO).
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
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        return IO(
            requires=unflatten_spec({f"inputs.{s}": self._spec(s) for s in self.streams}),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate normalisation buffers from the resolved schema (config-only).

        Buffers are named ``means_<stream>`` / ``stds_<stream>`` (design
        §6.3 checkpoint layout) and initialised to identity (0/1); the
        boolean ``materialised`` buffer guards against silently training on
        un-normalised values. Field names are captured here for
        `materialise`'s per-variable lookup; a `BindError` propagates from
        the schema if a stream's width or fields are not statically
        resolved.

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
        """Fail-fast, data-free norm-dict validation (M3 leftover, design §2.3).

        Without this check a wrong ``norm_dict`` path/content only surfaces
        at `materialise` — after dataset setup and plan compilation. The
        preflight reads ONLY the norm-dict YAML (config I/O in the design
        §2.6 sense — no H5/bind I/O): the file must exist, parse, and carry
        every configured stream; when the module is already bound (the
        `SaltModule.setup` call site) the per-variable mean/std entries are
        checked too, mirroring `materialise`'s validation. Called by
        `SaltModule.setup` on fresh fits (hard error) and by ``salt2 graph
        validate`` (warning — data-less machines stay supported).

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
        """Fill the buffers from the norm dict (the ONLY file I/O, design §2.3).

        Mirrors v1 `InputNorm`'s validation (inputnorm.py:56-88): missing
        streams/variables, non-finite values, and zero stds are errors.

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

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5); inputs are never
            mutated.

        Raises
        ------
        RuntimeError
            If the buffers were never materialised (fresh fit without
            `materialise`) — identity values would silently train
            un-normalised.
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

    OPT-IN alternative to the default fixed-norm-dict `Normaliser` (select
    via ``class_path: salt.core.nn.MaskedInputNormaliser``). The default
    `Normaliser` loads a precomputed ``norm_dict.yaml`` and preserves v1
    parity; THIS module instead learns its own statistics online.

    A ``BatchNorm``-style replacement for the v1 ``InputNorm``: instead of
    reading a pre-computed ``norm_dict.yaml`` (per-stream, per-variable
    ``{mean, std}``), this module **learns its own normalisation statistics
    online during training**, accumulating running mean/var over the
    *valid, non-padded* objects of every training batch. There is therefore
    no norm-dict dependency, no preprocessing pass, and no file I/O.

    Apply-with-running-stats (NOT batch stats)
    -----------------------------------------
    Unlike true ``nn.BatchNorm``, which normalises with *batch* statistics
    in train mode, this layer ALWAYS applies the **running** buffers:
    ``normed = (x - running_mean) / sqrt(running_var + eps)``. For input
    normalisation that keeps the rescaling stable within and across batches
    (batch-stat normalisation would make the input mapping batch-composition
    dependent and noisy). The running buffers are updated *separately* from
    the masked batch statistics during training only — closer to a "norm
    dict that adapts" than to classic BatchNorm. Init is identity (mean 0,
    var 1), so a fresh model is a near-passthrough that warms up.

    Masked online update (train only)
    ---------------------------------
    On a *training* forward (``self.training`` True, à la BatchNorm), for
    each stream the per-feature batch ``sum``, ``sumsq`` and ``count`` are
    computed over VALID objects only — sequence streams gather ``x[~mask]``
    (``masks.<stream>`` with ``True == padded``); the ``global_object`` has
    no mask (all rows valid). Under DDP these three quantities are
    all-reduced across ranks (never the per-rank mean/var, which cannot be
    averaged correctly when counts/means differ across ranks) before the
    global batch mean/var are derived. The EMA update is then
    ``running = (1 - momentum) * running + momentum * batch`` with
    ``num_batches_tracked += 1``. Empty (all-padded) streams skip the update.

    Eval / inference / ONNX (frozen)
    --------------------------------
    In eval mode (``self.training`` False) and under tracing the update
    branch and the mask read are both skipped: ``forward`` reduces to
    ``(x - running_mean) / sqrt(running_var + eps)``. The ``masks.<stream>``
    requirement is therefore declared **training-only** (``Mode.TRAINING``),
    so TEST/ONNX graphs do not demand a mask input and the export op is a
    pure affine transform with constant buffers.

    Lifecycle
    ---------
    ``__init__`` records config only (no I/O). ``bind(schema)`` allocates
    per-stream buffers ``running_mean_<stream>`` (zeros) /
    ``running_var_<stream>`` (ones) / ``num_batches_tracked_<stream>``
    (long 0) sized from the resolved ``inputs.<stream>`` widths. There is no
    ``materialise``/``preflight`` file hook — the buffers self-populate
    during training and ride in the checkpoint state_dict; on checkpoint
    load the stored buffers are restored verbatim and no warmup is needed.
    The legacy ``norm_dict`` constructor arg is kept for config
    compatibility but is **ignored** (no file is ever read).

    Unlike v1's ``InputNorm.forward`` (which rebinds its input dict's keys
    in place, inputnorm.py:103-106), this module produces NEW
    ``normed.<stream>`` keys and never mutates ``inputs.*`` (design §2.1).
    """

    def __init__(
        self,
        streams: Sequence[str],
        global_object: str | None = None,
        norm_dict: str | Path | None = None,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
    ) -> None:
        """Capture config only (design §2.3 — no file I/O here).

        Parameters
        ----------
        streams : Sequence[str]
            Streams to normalise (explicit M2 surface).
        global_object : str | None, optional
            The stream that is a per-object vector (``[B, F]``) rather than
            a padded sequence (``[B, T, F]``), by default None. The global
            object has no pad mask — every row is valid.
        norm_dict : str | Path | None, optional
            DEPRECATED / IGNORED. Kept only for config compatibility — the
            statistics are now learned online, never read from a file. No
            I/O is performed regardless of this value, by default None.
        momentum : float | None, optional
            EMA momentum for the running-stat update (BatchNorm default
            ``0.1``): ``running = (1 - momentum) * running + momentum *
            batch``. A fixed momentum tracks an exponentially-weighted
            trajectory and does NOT converge to the exact finite-dataset
            aggregate. Pass ``None`` for cumulative moving average
            (``momentum = 1 / num_batches_tracked``, like BatchNorm) which
            DOES converge to the true masked dataset mean/var, by default
            0.1.
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

        Every stream requires ``inputs.<stream>`` and produces
        ``normed.<stream>`` in all modes. Sequence streams ALSO require
        ``masks.<stream>`` (the pad mask, ``True == padded``) but ONLY in
        training modes (``Mode.TRAINING == FIT | VAL``) and as an OPTIONAL
        port: the mask is read only when updating the running statistics
        (under ``self.training``), and a stream without a pad-mask producer
        (e.g. a rank-2/rank-3 stream that is all-valid) simply has every
        object treated as valid. Gating the require to ``Mode.TRAINING``
        (rather than ``FIT`` alone) keeps the FIT and VAL plans structurally
        identical (``SaltModule._assert_fit_val_identical``) while leaving
        TEST/ONNX free of any mask dependency — so the exported graph is a
        pure affine transform. The ``global_object`` declares no mask at all.

        Returns
        -------
        IO
            The declared requires/produces.
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
        """Allocate the running-statistic buffers from the resolved schema.

        Per stream, registers ``running_mean_<stream>`` (zeros),
        ``running_var_<stream>`` (ones) and ``num_batches_tracked_<stream>``
        (long 0), sized from the resolved ``inputs.<stream>`` width. Init is
        identity normalisation (mean 0 / var 1) so a fresh model is a
        near-passthrough that warms up as the running stats accumulate. A
        `BindError` propagates from the schema if a stream's width is not
        statically resolved.

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

        `x` is ``[B, T, F]`` (sequence) or ``[B, F]`` (global); for a
        sequence stream `valid` is the boolean keep-mask ``[B, T]`` (the
        complement of the pad mask, ``True == valid``); for the global
        object `valid` is None (all rows kept). Reduction collapses every
        non-feature axis so the three returned tensors are ``[F]``; `count`
        is a scalar tensor (number of valid objects, summed over the batch).

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

        Computes per-feature ``(sum, sumsq, count)`` over valid objects,
        all-reduces those three quantities across DDP ranks (so the running
        buffers reflect the GLOBAL batch, not a single rank's shard — never
        averaging per-rank mean/var, which is wrong when counts/means
        differ), derives the global batch mean/var, and applies the EMA
        ``running = (1 - momentum) * running + momentum * batch``. An
        all-padded stream (global count 0) is skipped. ``num_batches_tracked``
        increments only when an update actually happens.
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
            # Cumulative (BatchNorm momentum=None): OBJECT-count-weighted pooling
            # via the parallel/chunked moment combination (Chan et al.), so the
            # buffers equal the EXACT pooled masked dataset mean/var regardless
            # of per-batch valid counts (converges to the true aggregate).
            seen = getattr(self, f"num_objects_seen_{stream}")
            n_old = seen.to(batch_mean.dtype)
            n_new = n_old + count
            delta = batch_mean - running_mean
            # combined mean
            new_mean = running_mean + delta * (count / n_new)
            # combined population variance (M2 accumulation form)
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

        Always applies the frozen running buffers,
        ``normed.<stream> = (inputs.<stream> - running_mean)
        / sqrt(running_var + eps)``, producing NEW keys and never mutating
        ``inputs.*`` (design §2.1/§2.5). The running statistics are updated
        from the masked batch moments (valid objects only) iff ALL of:
        ``self.training`` is True (BatchNorm-style gate — Lightning calls
        ``.train()`` for fit, ``.eval()`` for val/test), the plan ``mode`` is
        a training mode (``Mode.TRAINING``), and the graph is not being
        traced. Conjoining the mode keeps the update (and the mask read) in
        lockstep with the training-only ``masks.<stream>`` requirement, so a
        TEST/ONNX plan never reads a mask even on a module left in the
        default ``training=True`` state. In eval/inference/ONNX the op
        reduces to a pure affine transform with constant buffers.

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
    """Config-constructed per-stream initial embedding (design §9.2, replaces v1 `InitNet`).

    Composes a fresh v1 `Dense` built at `bind` — the dense input width is
    inferred from the resolved input and context widths, never configured
    (design §2.3 kills ``input_size`` YAML arithmetic, initnet.py:54-59).

    Context entries are attached with v1's ``attach_context`` semantics
    (tensor_utils.py:240: ``cat([context, x])`` — context PREPENDED), applied
    in list order; the final feature layout is therefore
    ``[ctx[-1], ..., ctx[0], stream]``. With the single GN2 entry
    ``context: [normed.jets]`` this reproduces v1's ``[global, stream]``
    column order exactly (checkpoint layout, design §5.1).

    Stream rank — INFERRED from the bound input (M7 W1.5 wave R; supersedes the
    M6-6 ``vector:`` flag): StreamEmbed no longer carries a rank flag of its own.
    Its rank is whatever its BOUND INPUT carries, and the embed simply preserves
    it: a padded-sequence input ``inputs.<s> [B, T, F]`` -> ``embed.<s> [B, T, D]``,
    a per-jet GLOBAL input ``inputs.<s> [B, F]`` -> ``embed.<s> [B, D]`` with NO
    token axis. The rank originates at ONE place — the reader's per-group
    ``global_object:`` flag (reader.py GroupConfig), which drives the
    reader/``Features`` boundary rank and the matching `Normaliser`
    ``global_object`` rank-2 handling (modules.py `Normaliser._spec`). The embed
    declares rank-AGNOSTIC specs (``shape=None`` on both its ``normed.<s>``
    require and its ``embed.<s>`` produce), so the producer (Normaliser/reader)
    sets the input rank and the consumer (Concat / a ``sequence:`` task head)
    sets the output rank — they are consistent BY CONSTRUCTION because both come
    from the single reader flag. The embed's ``out_dim`` width is contributed via
    the `derived_widths` bind hook (design §6.6), since a ``shape=None`` produce
    declares no last dim. This is what makes DL1 a jets-only MLP: a ``[B, F]``
    embed consumed straight by a (``sequence: false``) task head, with no encoder
    and no pooling — exactly v1's no-encoder/no-pool `InitNet`/`SaltModel` path
    (saltmodel.py:90-92 guard, :155-156 embed_xs, :170-172 global_rep = embed_xs).
    The internal v1 `Dense`/``attach_context`` are rank-agnostic (``nn.Linear``
    over ``[..., F]``, tensor_utils.py:236-240 expands a lower-rank context), so
    the forward math is identical — only the rank flowing through the spec changes.
    ONNX: a rank-2 embed is a plain `nn.Linear`-stack with no T axis, trivially
    traceable (B the sole dynamic axis), exactly as `Normaliser`'s ``[B, F]``
    global-object path traces today (no special-casing).

    muP — the ``mup:`` flag (M6 sub-wave B; plan 12; design §3.4 KEEP-architecture/
    BREAK-routing): the model-side embed is the first stage of the muP-parametrised
    forward (v1 ``InitNet(mup=True)`` -> ``Dense(mup=True)``, initnet.py:67-68,
    dense.py:48,88-89). When ``mup: true`` the composed v1 `Dense` is built with
    ``mup=True`` at `bind`, which runs the muP weight init in ``Dense.__init__``
    (``_reset_parameters``: every linear weight ``~N(0, 1/fan_out)``, bias zeroed,
    dense.py:96-102) instead of the default torch init. NOTHING else about the
    embed changes — the forward math (``net(attach_context(x))``) is byte-identical
    to the non-muP path; muP affects ONLY the initial parameter distribution
    (training dynamics), not the frozen forward. (Note: v1 ``InitNet(mup=True)``
    re-runs the reset via the misnamed ``self.net.reset_parameters()`` call,
    initnet.py:69 — but `Dense` exposes only ``_reset_parameters`` (with the
    underscore, dense.py:96) and ``nn.Module`` has no ``reset_parameters``, so this
    call RAISES ``AttributeError: 'Dense' object has no attribute 'reset_parameters'``
    at InitNet construction. v1's ``InitNet(mup=True)`` path is therefore BROKEN at
    construction — not merely a redundant no-op. v2 relies on ``Dense(mup=True)``'s
    own ``__init__`` reset (dense.py:88-89,96-102), which IS the intended muP init,
    so the v2 single-reset-in-``__init__`` is byte-faithful to v1's intended muP
    embed distribution, not a behaviour drop.) The ``apply_to`` ROUTING half
    (which named modules carry ``mup: true``, the shape-path/MuAdamW wiring) is a
    SEPARATE later stage — this is the architectural port only.

    Optional FiLM + positional encoding (M7 W-FILM, design §6.4 — input-layer
    half of v1 `InitNet`). Two OPTIONAL config blocks, both OFF by default (unset
    => byte-identical to today):

    - ``featurewise:`` — an INPUT-layer `FeaturewiseTransformation` (FiLM). When
      set, the embed reads the per-event ``inputs.parameters`` ``[B, n_params]``
      and applies ``scale * x + bias`` to the embed INPUT (before the `Dense`
      projection), exactly as v1 `InitNet.forward` (initnet.py:85-86: featurewise
      is applied to ``x`` BEFORE ``self.net(x)``). Faithful to v1, when featurewise
      is active the ``parameters`` are NOT also concatenated as context
      (initnet.py:81 ``not self.featurewise``) — the FiLM IS the conditioning path.
    - ``pos_enc:`` — a `PositionalEncoder` added to the embed OUTPUT. When set, the
      sin/cos encoding of the configured coordinate variables (selected by NAME
      from the input's fields) is ADDED to the projected embedding, exactly as v1
      `InitNet.forward` (initnet.py:92-95: ``x += pos_enc(inputs[input][...,
      idx])`` AFTER the projection). The pos_enc reads the RAW embed input columns
      (the same key the `Dense` consumes) at the variable indices.

    Both are NO-OP when unset.
    """

    MUP_WIDTH_ARG = "out_dim"
    """The init_arg the muP shape-generation tooling sweeps for this module
    (design §3.4; v1 ``parameter_name: output_size`` for ``init_nets``,
    GN2_muP.yaml:12-15). ``salt2 mup-shapes`` mutates ``init_args.out_dim`` to
    the base/delta widths to produce the infshapes."""

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        context: Sequence[str] = (),
        input: str | None = None,  # noqa: A002 - design §5.1 YAML surface name
        mup: bool = False,
        featurewise: dict[str, Any] | None = None,
        pos_enc: dict[str, Any] | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The stream to embed.
        out_dim : int
            Output embedding width (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense` (``hidden_layers``,
            ``activation``, ...); must not contain width keys, by default
            None.
        context : Sequence[str], optional
            Dotted bundle keys attached as context, in v1 prepend order (see
            class docstring), by default ``()``.
        input : str | None, optional
            Input key override, by default ``normed.<stream>``. The embed's
            rank is INFERRED from this bound input — no rank flag (M7 W1.5
            wave R; see the class docstring).
        mup : bool, optional
            Whether to use the muP parametrisation for the embed's internal
            `Dense` (M6 sub-wave B), by default False. When True the composed v1
            `Dense` is built with ``mup=True`` at `bind`, applying the muP weight
            init (``~N(0, 1/fan_out)`` weights, zeroed biases, dense.py:96-102);
            the forward is unchanged (muP affects init only). The v1 surface is
            ``init_net.dense_config.mup: True`` (GN2_muP.yaml:35).
        featurewise : dict[str, Any] | None, optional
            OPTIONAL input-layer FiLM config (M7 W-FILM; see class docstring). The
            kwargs for `FeaturewiseTransformation` MINUS the width args (``layer``
            is forced to ``"input"``; ``num_params``/``num_features`` are inferred
            at bind). May carry an explicit ``parameters`` key naming the bundle
            key of the per-event conditioning tensor (default ``inputs.parameters``).
            By default None (no FiLM — byte-identical to today).
        pos_enc : dict[str, Any] | None, optional
            OPTIONAL positional-encoding config (M7 W-FILM; see class docstring).
            The kwargs for `PositionalEncoder` (``variables``, ``dim``, ``alpha``)
            — ``dim`` defaults to ``out_dim`` (the embed width the encoding is added
            to). By default None (no positional encoding — byte-identical to today).

        Raises
        ------
        ConfigError
            If `dense` configures widths (``input_size`` etc. — inferred at
            bind, design §2.3) or `out_dim` is not positive.
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
        # -- optional input-layer FiLM (M7 W-FILM) ---------------------------
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
        # -- optional positional encoding (M7 W-FILM) ------------------------
        self.pos_enc_cfg = dict(pos_enc) if pos_enc is not None else None
        self.pos_enc: PositionalEncoder | None = None
        self.pos_enc_indices: tuple[int, ...] = ()

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + context keys -> ``embed.<stream>``.

        Rank-AGNOSTIC (M7 W1.5 wave R): the input require and the ``embed.<s>``
        produce both declare ``shape=None``, so the embed inherits its rank from
        the bound input — the producer (Normaliser/reader, keyed on the single
        reader ``global_object:`` flag) sets the input rank, and the consumer
        (Concat / a ``sequence:`` task head) sets the output rank; both come from
        that one flag so they agree by construction. The ``out_dim`` width of the
        ``embed.<s>`` produce is contributed at bind via `derived_widths` (a
        ``shape=None`` spec declares no last dim, design §6.6).

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(shape=None, dtype="float32"),
        }
        for key in self.context:
            # rank/width unconstrained here: context may be a [B, F] global
            # vector or broadcastable — widths resolve from the producer side
            requires[key] = TensorSpec(shape=None, dtype="float32")
        if self.featurewise_cfg is not None:
            # the per-event conditioning parameters: a rank-2 [B, n_params]
            # global stream (M7 W-FILM). Width resolves from the producer side.
            requires[self.params_key] = TensorSpec(shape=("B", sym_dim("P", self.name)), dtype="float32")
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(shape=None, dtype="float32"),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute the ``embed.<stream>`` last-dim width (design §6.6).

        The produce declares ``shape=None`` (rank inferred from the bound input,
        see `declare_io`), so the ``out_dim`` width is supplied here instead of
        via a spec last dim. `out_dim` is a config constant, so this is
        unconditional (no dependence on resolved input widths).

        Returns
        -------
        dict[str, int]
            ``{"embed.<stream>": out_dim}``.
        """
        del widths
        return {f"embed.{self.stream}": self.out_dim}

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` with the inferred input width (design §2.3).

        ``input_size = width(input) + sum(width(ctx))`` — exactly v1's
        inference (initnet.py:54-59) driven by the resolved schema instead
        of CLI variable injection. When ``self.mup`` the `Dense` is built with
        ``mup=True`` so its ``__init__`` applies the muP weight init
        (``_reset_parameters``, dense.py:88-89,96-102); the forward is unchanged.

        The optional input-layer FiLM (M7 W-FILM) is built here too: the FiLM
        nets are sized ``num_params = width(parameters)`` (input) /
        ``num_features = input_size`` (output, the embed INPUT width the FiLM is
        applied to BEFORE the projection, faithful to v1 initnet.py:85-86). The
        optional positional encoder resolves its variable column indices from the
        input's declared fields (initnet.py:93-94) and defaults ``dim`` to
        ``out_dim`` (the embed OUTPUT width it is added to).
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
            # (v1 initnet.py:93-94 ``[this_vars.index(v) for v in pos_enc.variables]``)
            fields = schema.fields_of(self.input_key)
            try:
                self.pos_enc_indices = tuple(fields.index(v) for v in self.pos_enc.variables)
            except ValueError as err:
                raise ConfigError(
                    f"StreamEmbed {self.name!r} pos_enc: variable not found in {self.input_key!r} "
                    f"fields {list(fields)}: {err}"
                ) from None

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Attach context (prepended, v1 order) and project.

        With the optional FiLM (M7 W-FILM) the per-event ``parameters`` scale/bias
        is applied to the embed INPUT before the projection (v1 initnet.py:85-86);
        with the optional positional encoder the sin/cos encoding of the configured
        coordinate columns is ADDED to the embed OUTPUT (v1 initnet.py:92-95). Both
        are NO-OP when unset (byte-identical to the no-FiLM path).

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        x = b.get(self.input_key)
        raw = x  # the raw input columns (pos_enc reads these at the variable idx)
        for key in self.context:
            x = attach_context(x, b.get(key))  # cat([context, x]) — tensor_utils.py:240
        if self.featurewise is not None:
            # FiLM on the embed INPUT, before the projection (v1 initnet.py:85-86)
            x = self.featurewise(b.get(self.params_key), x)
        assert self.net is not None, "forward before bind()"
        out = self.net(x)
        if self.pos_enc is not None:
            # ADD the positional encoding to the embed OUTPUT (v1 initnet.py:92-95)
            out = out + self.pos_enc(raw[..., self.pos_enc_indices])
        return {f"embed.{self.stream}": out}


class Concat(nn.Module):
    """Concatenate embedded streams into one sequence (design §5.1).

    Produces ``seq.x`` / ``seq.mask`` / ``seq.layout``; the concat ORDER is
    the configured list and nothing else (v1 relied on init_nets dict
    insertion order, saltmodel.py:128-129 / transformer.py:684-686).
    ``seq.layout`` is a dict-valued meta leaf ``{stream: (start, stop)}``
    over the pre-register sequence, consumed by `Split`.

    M2 deviation (documented in the module docstring): the design's
    ``registers:`` belong here, but the composed v1 `Transformer` appends
    its own register tokens internally and requires ``num_registers >= 1``
    — so nonzero ``registers`` is rejected until M7 absorbs the encoder.
    """

    def __init__(self, streams: Sequence[str], registers: int = 0) -> None:
        """Capture the explicit stream order (design §5.1).

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates, or if `registers`
            is nonzero (M2: registers live in `TransformerEncoder`, see
            class docstring).
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
        boundary table consumed by `Split`'s export branch, design §7).

        Returns
        -------
        IO
            The declared requires/produces.
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

        Concat shares one ``embed_dim`` symbol between its ``embed.<stream>``
        requires and its ``seq.x`` produce (equal embed widths are a genuine
        concat constraint). Since `StreamEmbed` now produces ``embed.<stream>``
        with ``shape=None`` (rank inferred from the bound input, M7 W1.5 wave R),
        its width arrives via `StreamEmbed.derived_widths` rather than a spec last
        dim — so there is no longer a concrete ``embed.<stream>`` shape for the
        dim table to bind ``embed_dim`` from. When an ENCODER follows, the
        encoder's concrete ``seq.x`` require width still binds it; the
        ENCODERLESS-pool path (regression DiPS body) has no such anchor, so this
        hook forwards the resolved embed width to ``seq.x`` directly (design §6.6,
        the same mechanism as `VectorConcat`'s ``Dsum``). The per-stream embeds
        share one width (the concat constraint), so the FIRST resolved input
        width is the ``seq.x`` width.

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
        ``torch.onnx.operators.shape_as_tensor`` on the per-stream pad masks
        so the boundaries trace to Shape/Concat/CumSum nodes that stay
        symbolic under ``dynamo=False`` tracing (design §7 multi-stream
        trace-safety; the mode branch itself is static Python). Eager
        FIT/VAL/TEST numerics are untouched.

        Returns
        -------
        dict[str, Any]
            The newly produced keys only (design §2.5). ``torch.cat``
            allocates fresh tensors even for one input, so the seq leaves
            never alias the per-stream leaves.
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
    """Ordered concatenation of ``[B, D_i]`` vectors into one ``[B, Dsum]`` key (design §6.6).

    The v2 spelling of v1's post-pooling ``'global'`` magic key
    (saltmodel.py:175-177): ``global_rep = cat([global_rep, global_feats],
    dim=-1)``. v1 fed GN3's 2-feature ``global`` stream PAST the encoder and
    concatenated it onto the pooled representation, producing the
    ``&pooled_dim 258`` (256 + 2) every GN3 task consumes. The synthesis killed
    ``'global'`` as a magic key without a replacement — a silent feature-drop
    hazard (both design critics rated HIGH: all widths still unify after the
    drop, so no ``ShapeError`` fires). This module is the explicit replacement.

    There is **no v1 class** for this — v1 inlined the cat in
    ``SaltModel.forward``. The nearest v1 primitives are the ``merge_dict``
    stream merge (saltmodel.py:135-147, sequence-level, code-only) and
    ``InitNet.attach_global`` (initnet.py:77-82, context prepend); neither is a
    standalone post-pooling vector concat. The correctness anchors are therefore
    design-conformance, not v1-byte-parity (plan 10 sub-wave B, L3):

    1. **Concat ORDER is the config list.** For converted GN3 checkpoints the
       converter emits ``inputs: [pooled.global, normed.global]`` (pooled first,
       global features last) so the task first-layer weight layouts line up with
       v1's ``cat([global_rep, global_feats])`` (design §6.6 1390-1391). The
       order is the configured list and nothing else.
    2. **Output width ``Dsum = sum(D_i)``.** Resolved at bind via
       `derived_widths` (there is no concrete edge for the dim table to bind
       ``Dsum`` to — only the concat knows the sum; design §6.6 "Dsum unified at
       bind", bind.py second pass).
    3. **ONNX is the ``alias:`` mechanism, not this module.** Athena feeds ONE
       jet tensor that v1 clones into ``global`` (to_onnx.py:377-378); v2
       reproduces that with ``export.inputs`` ``alias:`` (`OnnxAdapter`,
       onnx/adapter.py, design §6.6/§7) — a name-resolved column gather binding
       the aliased port from the source tensor. VectorConcat itself is
       mode-agnostic: ``torch.cat`` over ``[B, D]`` vectors has no dynamic
       feature axis, so the forward is identical in every mode (unlike
       sequence-level `Concat`/`Split`, which need ``seq.offsets`` trace-safety).

    4 configs depend on it: GN2emu, GN3V01, GN3_SoftE, GN3EPCLV01.
    """

    def __init__(self, inputs: Sequence[str], out: str = "pooled.global") -> None:
        """Capture the explicit ordered input list and the output key (design §6.6).

        Parameters
        ----------
        inputs : Sequence[str]
            Dotted bundle keys to concatenate, in the EXACT order they appear
            in the output (pooled first for converted GN3 checkpoints, design
            §6.6 1390-1391). Must be non-empty with no duplicates.
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
        """Declare each ``[B, D_i]`` input -> the ``[B, Dsum]`` output (design §6.6).

        Each input gets its OWN instance-scoped width symbol (the inputs are
        genuinely different widths — pooled 256 vs global 2 — so they must NOT
        share a symbol). The output's ``Dsum`` symbol is resolved at bind from
        the sum of the resolved input widths (`derived_widths`); the dim table
        alone cannot bind it (no concrete edge), which is exactly why the
        bind-time second pass exists (design §6.6 "Dsum unified at bind").

        Returns
        -------
        IO
            The declared requires/produces.
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
        """Contribute ``Dsum = sum(D_i)`` once every input width is resolved (design §6.6).

        Called by `resolve_bind_schema`'s second pass with the widths resolved
        so far. Returns the output width when ALL inputs are known, else an
        empty dict (a mode where some input is absent — the hook is
        order-insensitive across plans). This is the ONLY place the concat sum
        is known; the union-find dim table cannot infer it from edges alone.

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
        """Concatenate the inputs along the feature dim, in the configured order.

        Mode-agnostic: ``[B, D]`` vectors have no dynamic feature axis, so the
        same ``torch.cat`` traces correctly for ONNX (design §6.6). ``torch.cat``
        allocates a fresh tensor, so the output never aliases an input leaf
        (design §2.1 write-once).

        Returns
        -------
        dict[str, Tensor]
            The newly produced key only (design §2.5).
        """
        del mode
        return {self.out_key: torch.cat([b.get(key) for key in self.inputs], dim=-1)}


class EdgeFeatures(nn.Module):
    """Config-constructed pairwise edge-feature builder (design §6.7, M6 sub-wave C).

    The v2 replacement for v1's ``EdgeConstructor`` (edge_constructor.py:7) +
    ``calculate_edge_features`` (edge_features.py:52). A NetModule: it requires
    the **raw** ``inputs.<stream>`` ``[B, T, F]`` (design §6.3 — edges are built
    on UN-normalised values, which is exactly why ``inputs.*`` and ``normed.*``
    are distinct keys and no ordering hack is needed) plus ``masks.<stream>``,
    and produces ``edges.<stream>`` ``[B, T, T, E]`` where ``E = len(features)``.

    v1 wired this as in-place mutation: ``EdgeConstructor.forward`` wrote
    ``inputs[f"_edge_features_{name}"]`` back into the input dict
    (edge_constructor.py:41-44) and the SaltModel sorted init_nets so the edge
    stream landed first (saltmodel.py:69-80). v2 makes it an ordinary graph
    edge: a NEW ``edges.<stream>`` key, never mutating ``inputs.*`` (design
    §2.1 write-once); the edge-stream-first / EdgeAttention-backend constraints
    become bind-time validator rules on the ENCODER port (design §6.7
    1425-1431), authored when the encoder gains its ``edges:`` arg (a later M6
    sub-wave C stage — NOT here).

    Faithfulness: the per-element math is the v1 functions
    (`calculate_edge_features`, `check_edge_config`) — INLINED into this module
    BYTE-FAITHFULLY at M7 W2b (modules.py, copied from v1 edge_features.py:10-146,
    v1 original untouched) so the dR/kt/z/subjetIndex/isSelfLoop/mass values are
    byte-identical to v1. The ``indices_map`` (variable name -> column index)
    is resolved at `bind` from the resolved schema's declared ``inputs.<stream>``
    fields (the dataset-side `Features` declaration, design §2.2 — column
    lookups resolve by NAME, never by YAML list position), exactly v1's
    ``EdgeConstructor`` ``indices_map`` built from ``variables[input_name]``
    (edge_constructor.py:34-36). ``masks.<stream>`` is a declared dependency
    (design §6.7 1418) — the v1 math does not consume it (it relies on
    ``nan_to_num`` to zero the inf/nan from zero-padded rows, edge_features.py:146,
    and the upstream `Features` processor already zeroes padded rows,
    processors.py:145), so the v2 forward composes the v1 function verbatim and
    keeps the mask as the contract dependency that ties this module to its
    stream's pad mask.

    ONNX (load-bearing, design §6.7 ONNX note / plan 12 sub-wave C): the
    produced ``edges.<stream>`` carries the SAME ``T:<stream>`` symbol on BOTH
    token axes, so a downstream export marks both as dynamic; the v1 math is all
    ``unsqueeze``/``expand``/elementwise ops driven by ``batch.shape[1]`` (the
    dynamic token count), tracing to shape-derived nodes — no baked track count.
    The encoder-side register zero-pad is a SEPARATE later stage; this module's
    forward is itself trace-safe (no Python-int shape bakes).

    M7 W2b: the v1 ``calculate_edge_features`` / ``check_edge_config`` math is now
    INLINED into this module (byte-faithful, above), so the edge builder no longer
    imports the v1 ``salt.utils.edge_features`` tree — the composed-function
    dependency is dropped.
    """

    def __init__(
        self,
        stream: str,
        features: Sequence[str],
        out: str | None = None,
        input: str | None = None,  # noqa: A002 - design §5.1 YAML surface name
    ) -> None:
        """Capture config only (design §2.3 — no schema/data access here).

        Parameters
        ----------
        stream : str
            The stream whose pairwise edges are built (``inputs.<stream>``).
        features : Sequence[str]
            Edge feature names, in produced column order — any of
            ``{"dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass"}`` (the v1
            `check_edge_config` vocabulary, edge_features.py:30-43). Validated
            against the recognised set here; required input variables are
            checked at `bind` against the resolved fields.
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

        The input is the RAW (un-normalised) stream (design §6.3); the produced
        edge tensor is ``[B, T, T, E]`` with the SAME ``T:<stream>`` symbol on
        both token axes (a square per-stream pairwise matrix) and a concrete
        last dim ``E = len(features)``. ``masks.<stream>`` is a declared
        ``pad_mask`` dependency (design §6.7 1418).

        Returns
        -------
        IO
            The declared requires/produces.
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

        Mirrors v1 ``EdgeConstructor.__init__`` (edge_constructor.py:32-36):
        the ``indices_map`` is built from the declared ``inputs.<stream>``
        fields (the dataset-side `Features` variable list, design §2.2 — column
        lookups by NAME), then `check_edge_config` validates that every feature's
        required variables are present (e.g. ``dR`` needs ``eta``/``phi``).
        """
        fields = schema.fields_of(self.input_key)
        check_edge_config(list(self.features), list(fields))
        self.indices_map = {name: i for i, name in enumerate(fields)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute the pairwise edge features on the RAW input (design §6.7).

        Composes the v1 `calculate_edge_features` verbatim (byte-faithful), so
        the dR/kt/z/... values match v1 exactly. Returns a FRESH tensor
        (``torch.zeros`` + ``nan_to_num`` inside the v1 function), never
        mutating ``inputs.*`` (design §2.1 write-once).

        Returns
        -------
        dict[str, Tensor]
            The newly produced ``edges.<stream>`` key only (design §2.5).
        """
        del mode
        assert self.indices_map is not None, "forward before bind()"
        x = b.get(self.input_key)
        edges = calculate_edge_features(x, self.indices_map, list(self.features))
        return {self.out_key: edges}


class EdgeEmbed(nn.Module):
    """Config-constructed edge-feature embedding (design §6.7 1420-1421, M6 sub-wave C).

    An edge-typed `StreamEmbed`: it maps ``edges.<stream>`` ``[B, T, T, E]`` ->
    ``edges.<stream>_emb`` ``[B, T, T, D_e]`` with an internal v1 `Dense`. The
    v2 replacement for v1's ``edge_init_nets`` (saltmodel.py:69-80) — but
    WITHOUT v1's sort-first hack (the ``assert len(edge_init_nets) == 1`` and
    the init_nets re-sort so the edge stream is first): in v2 the edge stream's
    leading position in the concat is a bind-time VALIDATOR rule on the encoder
    port (design §6.7 1425-1431), not a silent runtime sort.

    Lifecycle (design §2.3): ``__init__`` captures config; ``bind`` builds the
    `Dense` with ``input_size = width(edges.<stream>) = E`` (the edge-feature
    count, resolved from the schema — design §2.3 kills ``input_size`` YAML
    arithmetic, exactly as `StreamEmbed`); ``forward`` projects. The internal v1
    `Dense` is an ``nn.Linear`` stack over the LAST dim (dense.py:91-94), so it
    applies cleanly to a rank-4 ``[B, T, T, E]`` tensor (it embeds each
    pairwise edge independently) — the v1 ``edge_init_nets[0]`` `InitNet` does
    the same (it is a `Dense` over the ``[B, L, L, E]`` edge matrix,
    saltmodel.py:132). No context, no muP, no per-stream rank inference — edges
    are always the rank-4 pairwise matrix.

    ONNX: a plain ``nn.Linear``-stack over the last dim has no dynamic feature
    axis; the two ``T:<stream>`` token axes flow through unchanged, so the
    embed traces with both token axes dynamic (the dynamic-T contract is
    enforced on the export side, plan 12 sub-wave C).

    FiLM / positional encoding (M7 W-FILM): the edge embed carries NEITHER — and
    this is FAITHFUL to v1, not a drop. v1's ``init_featurewise`` (saltmodel.py:
    262-279) attaches `FeaturewiseTransformation` ONLY to the constituent
    ``init_nets``, NEVER to the ``edge_init_nets``; and v1 builds the edge
    `InitNet` with ``featurewise=None`` / ``pos_enc=None`` (saltmodel.py:75-77 —
    the edge init nets get no featurewise/pos_enc kwargs). So an edge embed is a
    PLAIN projection of the pairwise edge-feature matrix with no per-event
    conditioning and no coordinate encoding (the FiLM/posenc input-layer port
    lives on `StreamEmbed` for the constituent streams; the encoder/global FiLM
    lives on `TransformerEncoder`). Nothing to wire here.
    """

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        input: str | None = None,  # noqa: A002 - design §5.1 YAML surface name
        out: str | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The stream whose edge tensor is embedded.
        out_dim : int
            Output edge-embedding width ``D_e`` (concrete, config-fixed; v1
            ``edge_init_nets[0].dense_config.output_size``, GN2XE.yaml:70).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense` (``hidden_layers``,
            ``activation``, ...); must not contain width keys, by default None.
        input : str | None, optional
            Edge input key override, by default ``edges.<stream>``.
        out : str | None, optional
            Produced embedded edge key, by default ``edges.<stream>_emb``.

        Raises
        ------
        ConfigError
            If `dense` configures widths (inferred at bind, design §2.3) or
            `out_dim` is not positive.
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
        """Declare ``edges.<stream>`` ``[B, T, T, E]`` -> ``edges.<stream>_emb`` ``[B, T, T, D_e]``.

        Both token axes share the stream's ``T:<stream>`` symbol; the input
        last dim is a require symbol (resolved at bind), the produced last dim
        is the concrete ``out_dim``.

        Returns
        -------
        IO
            The declared requires/produces.
        """
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
        """Build the internal `Dense` with the inferred edge-feature width (design §2.3).

        ``input_size = width(edges.<stream>) = E`` — the edge-feature count,
        resolved from the schema (v1 inferred it from the edge tensor's last
        dim, saltmodel.py:132). No context (edges carry no context entries).
        """
        self.net = Dense(
            input_size=schema.width(self.input_key), output_size=self.out_dim, **self.dense_cfg
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Project the rank-4 edge tensor through the internal `Dense`.

        The `Dense` embeds the last (edge-feature) dim, leaving both token axes
        intact: ``[B, T, T, E] -> [B, T, T, D_e]`` (v1 edge_init_nets,
        saltmodel.py:132).

        Returns
        -------
        dict[str, Tensor]
            The newly produced ``edges.<stream>_emb`` key only (design §2.5).
        """
        del mode
        assert self.net is not None, "forward before bind()"
        return {self.out_key: self.net(b.get(self.input_key))}


class TransformerEncoder(nn.Module):
    """Config-constructed transformer encoder (design §9.2, composes a fresh v1 `Transformer`).

    Everything width-relevant is config (``dim``, ``out_dim``), so the
    composed v1 instance is built in ``__init__`` (design §2.3 permits
    config-only layer construction there; `bind` is for schema-derived
    widths). Registers, packing, and the out projection stay INTERNAL
    (design §2.5 composite modules); the register pad mask is published as
    the NEW key ``masks.registers`` — the caller's mask dict is never
    mutated (v1's ``_add_registers`` inserts ``"REGISTERS"`` into it,
    transformer.py:777,785).

    ``drop_registers`` (the MaskFormer encoder passthrough, M5 sub-wave C;
    v1 transformer.py:560,584,745-750) makes the composed v1 `Transformer`
    strip the register rows from its output AFTER the layer stack runs: the
    registers are still appended internally and visible to every attention
    layer (they exist so a constituent-less jet has SOMETHING to attend to,
    transformer.py:569-574), but the returned ``encoded.seq`` is sliced back
    to the stream tokens only (``x[:, :-num_registers]``) and v1 drops the
    ``"REGISTERS"`` pad entry (``del pad_mask["REGISTERS"]``,
    transformer.py:745-750). So with ``drop_registers=True`` this module
    produces NO ``masks.registers`` key — there are no register rows left in
    ``encoded.seq`` for a downstream consumer to mask — exactly the v1
    MaskFormer encoder shape its `MaskDecoder` reads (``embed_xs`` with the
    registers already gone, maskformer.py:151-156,172). The shipped
    MaskFormer.yaml sets ``drop_registers: true`` (MaskFormer.yaml:31). The
    design's longer-term home for this is a `Concat`-owned ``registers:`` +
    ``drop_registers_after:`` (FD 1121-1125); in the M2 architecture
    registers live INSIDE this encoder (the composed v1 `Transformer`
    appends them and requires ``num_registers >= 1``), so the M5 passthrough
    is the faithful, byte-for-byte v1 mechanism — the Concat-owned form lands
    when M7 absorbs the encoder.

    ``norm_type`` (``"pre"`` default, or ``"post"`` / ``"hybrid"``) is
    forwarded verbatim to every composed v1 ``EncoderLayer`` (M5 sub-wave B;
    the GN3V01 flagship + GN3_Hybrid / GN3EPCLV01 are ``"hybrid"``). The
    placement logic — hybrid forcing ``do_qk_norm``/``do_v_norm``, the
    depth-0 residual-norm special case, and the pre-FFN norm — all lives in
    the composed v1 layer (transformer.py:350-356,421); the wrapper only
    threads the flag through the `Transformer` ``**kwargs`` passthrough.

    muP — the ``mup:`` flag (M6 sub-wave B; plan 12; design §3.4 KEEP-architecture/
    BREAK-routing). When ``mup: true`` the composed v1 `Transformer` is built with
    ``mup=True``. **IMPORTANT — what this flag actually does is NARROWER than the
    name suggests, and this is a faithful port of v1's behaviour.** v1
    ``Transformer.__init__`` takes ``mup`` as a *named* parameter
    (transformer.py:563), so it is consumed by `Transformer` and does NOT flow into
    its ``**kwargs``; when it builds its ``EncoderLayer`` list (transformer.py:604-614)
    it does NOT pass ``mup`` down. ``EncoderLayer`` therefore defaults to
    ``mup=False`` and never sets ``attn_kwargs["mup"]/dense_kwargs["mup"]``
    (transformer.py:362-364). The ONLY thing ``mup=True`` wires from the encoder flag
    is the **out-proj swap**: ``nn.Linear`` -> ``mup.MuReadout`` with weight AND bias
    zeroed (transformer.py:623-628), an output-scaling Linear subclass.

    Consequently, for THIS code path (the production GN2_muP encoder, built via
    ``class_path: salt.models.Transformer`` with ``mup: true``):

    - the attention softmax scale STAYS ``1/sqrt(head_dim)`` — the muP ``1/head_dim``
      scale (attention.py:275) is **NOT** active;
    - the attention muP init (Q-projection zeroed + K/V scaled, attention.py:319-326)
      is **NOT** run — the Q-rows of ``in_proj_weight`` are non-zero;
    - the GLU/dense-linear muP init (transformer.py:99-103, dense.py:96-102) is
      **NOT** engaged inside the encoder.

    Those attention/dense-level muP behaviours live in ``Attention(mup=True)`` /
    ``Dense(mup=True)``, which v1 only constructs when ``mup`` is threaded all the
    way down — something v1's ``Transformer(mup=True)`` does not do. So they were
    inactive in v1's production GN2_muP encoder too, and the v2 port is byte-faithful
    to v1's ACTUAL encoder-mup behaviour (MU1's bitwise ``torch.equal`` vs an
    independent v1 ``Transformer(mup=True)`` passes; the dense/embed muP init that IS
    active lives on the embed's ``Dense(mup=True)``, see `StreamEmbed`). This matches
    the MU1 gate's faithfulness_note verbatim. (If full attention-level muP were
    desired for correct attention HP-transfer, that would be a v1 *behaviour change*,
    out of scope for a parity-faithful port — raise it explicitly; do not assume the
    encoder's attention is muP-parametrised.)

    The flag is the ONLY init_arg taken here — the M7 ``featurewise:`` / ``edges:``
    ports stay out (their TODO line is below). The ``apply_to`` ROUTING half (which
    named modules carry ``mup: true``, ``mup-shapes``, the MuAdamW swap) is a
    SEPARATE later stage — this is the architectural port only.

    MuReadout needs ``infshape`` (set via ``mup.set_base_shapes``) before its
    ``forward`` / ``width_mult()`` work. The ROUTING stage applies a real shape
    file; here, the architectural default sets the base shapes from the module
    onto ITSELF with ``rescale_params=False`` — ``width_mult() == 1.0`` and the
    zeroed weights are untouched — so a ``mup: true`` encoder is forward-runnable
    and traceable standalone (a non-unit ``width_mult`` only enters once a wider
    model is set against a narrower base by the routing stage).

    ONNX export contract (load-bearing; plan 12 sub-wave B). ``MuReadout`` is a
    NON-``nn.Linear`` subclass whose forward applies an output multiplier
    (``output_mult * x / width_mult``) before the linear — tracing it as-is risks
    an unsupported/incorrect graph. At export (`set_export_mode`, the §7.2 protocol
    the `OnnxAdapter` invokes on every submodule) the multiplier is FOLDED into the
    weight/bias and the out-proj is swapped for a plain ``nn.Linear`` in the traced
    graph: ``W_folded = (output_mult / width_mult) * W`` and ``bias_folded = bias``
    (the multiplier scales only the ``x @ Wᵀ`` term, NOT the bias — see the
    MuReadout forward). The folded Linear's forward is numerically EQUAL to the
    MuReadout forward (bitwise when ``output_mult/width_mult == 1.0``, the
    architectural default; ``<=1e-6`` otherwise — the float32 multiply-order is the
    only difference). Inference math is then identity to a standard Linear (muP
    affects training, not the frozen forward), so the exported graph is a plain
    transformer.

    Optional FiLM (M7 W-FILM, design §6.4 — the encoder + global halves of v1
    featurewise). The ``featurewise:`` config is a LIST of
    `FeaturewiseTransformation` configs, each with a ``layer`` of ``"encoder"`` or
    ``"global"`` (an ``"input"`` entry belongs on `StreamEmbed`, rejected here):

    - ``layer: encoder`` — one `FeaturewiseTransformation` per encoder layer,
      applied to ``x`` at the START of every layer (v1 transformer.py:727-728).
      The per-layer FiLMs are populated into the absorbed `Transformer`'s
      ``featurewise`` ``ModuleList`` (the verbatim v1 forward already applies
      them, modules.py:2001-2003), driven by the per-event ``inputs.parameters``
      ``[B, n_params]`` tensor threaded through the encoder forward.
    - ``layer: global`` — a single `FeaturewiseTransformation` applied to the
      ENCODER OUTPUT (``encoded.seq``) before it is published, faithful to v1
      where the global FiLM scales ``preds["embed_xs"]`` (the encoder output)
      right before pooling (saltmodel.py:165-166).

    OFF by default (no ``featurewise:`` => byte-identical to today; the absorbed
    Transformer's ``featurewise`` ModuleList stays empty and the verbatim forward
    skips the FiLM application). The ``edges:`` port was wired in M6 sub-wave C
    (FD §6.7) — only the featurewise port lands here.
    """

    _NORM_TYPES = ("pre", "post", "hybrid")
    """The encoder-layer norm placements the wrapper forwards (v1 EncoderLayer,
    transformer.py:307-308,350-356). ``"none"`` is a residual-only v1 mode with
    no shipped v2 config — rejected loudly here rather than silently passed."""

    MUP_WIDTH_ARG = "dim"
    """The init_arg the muP shape-generation tooling sweeps for this module
    (design §3.4; v1 ``parameter_name: embed_dim`` for ``encoder``,
    GN2_muP.yaml:12-15). ``salt2 mup-shapes`` mutates ``init_args.dim`` to the
    base/delta widths to produce the infshapes."""

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
            (default ``"torch-math"``) selects the backend; the remaining
            keys are v1 ``attn_kwargs`` (REQUIRED by v1: transformer.py
            writes ``attn_type`` into them, transformer.py:600-601).
        out_dim : int | None, optional
            Output projection width, by default None (= `dim`, no
            projection).
        dense : dict[str, Any] | None, optional
            v1 ``dense_kwargs`` (``activation``, ``gated``, ...), by
            default None.
        norm : str, optional
            Normalisation layer name, by default ``"LayerNorm"``.
        num_registers : int, optional
            Learned register tokens appended INSIDE the encoder, by default
            1 (v1 minimum — transformer.py:558-559).
        norm_type : str, optional
            Per-layer norm placement, one of ``{"pre", "post", "hybrid"}``,
            by default ``"pre"``. Forwarded verbatim to every v1
            ``EncoderLayer`` via the `Transformer` ``**kwargs`` passthrough
            (transformer.py:611). ``"hybrid"`` (the GN3V01 flagship) makes the
            EncoderLayer force ``do_qk_norm``/``do_v_norm`` on its `Attention`,
            use a residual ``norm_type`` of ``"pre"`` at depth 0 / ``"none"``
            after, and apply a pre-FFN norm in ``forward`` (transformer.py:
            350-356,421). The wrapper only passes the flag through — all that
            placement logic lives in the composed v1 layer.
        drop_registers : bool, optional
            Strip the register rows from ``encoded.seq`` after the layer stack
            (the MaskFormer encoder passthrough; v1 transformer.py:745-750), by
            default False. Registers stay visible to every attention layer; only
            the OUTPUT sequence is sliced back to the stream tokens, and no
            ``masks.registers`` key is produced (see the class docstring).
        mup : bool, optional
            Whether to use the muP parametrisation (M6 sub-wave B), by default
            False. When True the composed v1 `Transformer` is built with
            ``mup=True``, which (faithfully to v1) wires ONLY the `MuReadout`
            out-proj swap — v1's ``Transformer(mup=True)`` does NOT pass ``mup``
            down to its EncoderLayers, so the 1/d attention scale and the
            attention/dense muP init are NOT active in this encoder path (see the
            class docstring for the full explanation). v1 surface is the encoder's
            ``mup: True`` (GN2_muP.yaml:51). Requires an out projection (``out_dim``
            set) — `MuReadout` is the last muP layer (v1 transformer.py:594-597).
            The export-time fold to a plain `nn.Linear` happens in `set_export_mode`.
        edges : str | None, optional
            The EDGE-EMBED bundle key the encoder consumes (M6 sub-wave C; FD
            §6.7 1422-1424). When set (e.g. ``"edges.tracks_emb"``) every
            composed v1 ``EncoderLayer`` swaps its `Attention` for an
            `EdgeAttention` (transformer.py:365-373): the edge tensor biases the
            attention scores and gates the softmax output (attention.py:630-654).
            The edge-stream-first / EdgeAttention-backend-forcing constraints are
            BIND-TIME validators on this port (FD 1425-1431; `validate_edge_port`
            in saltmodule.py) — the named-error replacements for v1's silent
            sort-first hack (saltmodel.py:69-80) and flash bypass
            (transformer.py:599-601). By default None (no edge path).
        edge_embed_dim : int, optional
            The edge-embed width ``D_e`` (v1 ``edge_embed_dim``, GN2XE.yaml:79).
            REQUIRED (positive) when `edges` is set — the composed v1
            `Transformer` builds its `EdgeAttention` projections from it at
            ``__init__`` (transformer.py:368), so it cannot be deferred to bind;
            cross-checked against the resolved ``edges.<stream>_emb`` width at
            bind (so a mismatch is a named error, not a silent shape bug). Must
            be 0 when `edges` is None. By default 0.
        update_edges : bool, optional
            Whether the encoder UPDATES the edge tensor each layer (v1
            ``update_edges``, GN2XE.yaml:80; EdgeAttention edge-out projection
            attention.py:641-644, EncoderLayer edge post-norm
            transformer.py:416-417). Requires `edges` set. The updated edges
            stay INTERNAL to the encoder (v2 produces only ``encoded.seq`` — the
            edge update is a per-layer refinement, not a published output, exactly
            as v1 keeps ``edge_x`` inside `Transformer.forward`,
            transformer.py:729-733). By default False.

        Raises
        ------
        ConfigError
            If `attention` is missing ``num_heads``, `norm_type` is not one of
            ``{"pre", "post", "hybrid"}``, `mup` is set without an `out_dim`
            (MuReadout has no layer to live on — v1 transformer.py:594-597),
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
        # edge-port consistency (FD §6.7 1422-1424): edges <-> edge_embed_dim are
        # paired — the v1 EncoderLayer picks EdgeAttention iff edge_embed_dim > 0
        # (transformer.py:366), and update_edges needs an edge tensor to update
        # (transformer.py:589-590). Reject the inconsistent combinations loudly
        # at config time rather than building a half-wired encoder.
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
            # MuReadout.forward/width_mult() assert ``infshape`` is set (via
            # mup.set_base_shapes). The ROUTING stage supplies a real shape file;
            # the architectural default sets the base shapes from the module onto
            # ITSELF with rescale_params=False — width_mult()==1.0, the zeroed
            # MuReadout weights untouched — so a standalone mup encoder is
            # forward-runnable and traceable (a non-unit width_mult enters only
            # when a wider model is set against a narrower base by the routing
            # stage). Import locally to keep the module import surface lean and to
            # avoid a hard mup dependency for non-mup encoders.
            from mup import set_base_shapes  # noqa: PLC0415

            set_base_shapes(self.encoder, self.encoder, rescale_params=False)
        self.out_dim = self.encoder.out_dim
        self.num_registers = num_registers
        # -- optional encoder/global FiLM (M7 W-FILM) ------------------------
        self.num_layers = int(num_layers)
        self.params_key = "inputs.parameters"
        self._encoder_film_cfg: dict[str, Any] | None = None
        self._global_film_cfg: dict[str, Any] | None = None
        self.featurewise_global: FeaturewiseTransformation | None = None
        for fw in featurewise or ():
            fw = dict(fw)
            layer = fw.get("layer")
            # one params key is shared by all FiLM entries (v1 reads the single
            # inputs["parameters"], featurewise.py:74); take it from any entry.
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

        ``"edges.tracks_emb"`` -> ``"tracks"`` (the second dotted component is
        the stream + the ``_emb`` suffix). Used by the bind-time
        edge-stream-first validator (FD §6.7 1425-1431) to check the edge
        stream is `Concat.streams[0]`.

        Returns
        -------
        str | None
            The edge stream name, or None.
        """
        if self.edges_key is None:
            return None
        # "edges.<stream>_emb" -> "<stream>" (the EdgeEmbed out-key convention,
        # modules.py EdgeEmbed.out_key); strip the namespace + the _emb suffix.
        leaf = self.edges_key.split(KEY_SEP, 1)[1] if KEY_SEP in self.edges_key else self.edges_key
        return leaf.removesuffix("_emb")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` (+ ``edges.<stream>_emb``) -> ``encoded.seq``.

        ``masks.registers`` is produced ONLY when ``drop_registers`` is False:
        with the registers dropped from ``encoded.seq`` there are no register
        rows left to mask (the v1 ``del pad_mask["REGISTERS"]`` shape,
        transformer.py:745-750), and a downstream `GlobalAttentionPooling`
        treats ``masks.registers`` as OPTIONAL — so a drop-registers config
        plan-compiles exactly like the encoder-less path (modules.py:1065-1070).

        When an `edges` port is configured (M6 sub-wave C; FD §6.7 1422-1424)
        the encoder additionally REQUIRES the edge-embed tensor
        ``edges.<stream>_emb`` ``[B, T, T, D_e]`` — a rank-4 pairwise tensor
        whose BOTH token axes share the stream's ``T:<stream>`` symbol (the
        square pairwise matrix `EdgeEmbed` produces, modules.py EdgeEmbed). The
        updated edges stay INTERNAL (v1 keeps ``edge_x`` inside the forward,
        transformer.py:729-733), so this module still produces only
        ``encoded.seq`` (+ the optional register mask) — no edge output key.

        Returns
        -------
        IO
            The declared requires/produces (widths concrete from config).
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
            # (FD §6.7 ONNX note); last dim is the concrete edge-embed width.
            requires[self.edges_key] = TensorSpec(
                shape=("B", tlen, tlen, self.edge_embed_dim), dtype="float32"
            )
        if self._encoder_film_cfg is not None or self._global_film_cfg is not None:
            # the per-event conditioning parameters: a rank-2 [B, n_params] global
            # stream feeding the encoder/global FiLM (M7 W-FILM)
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
        projections from ``edge_embed_dim`` at ``__init__`` (transformer.py:368),
        so this `bind` only VALIDATES that the resolved ``edges.<stream>_emb``
        width matches — a mismatch (the EdgeEmbed ``out_dim`` was changed without
        updating the encoder, or vice versa) is a named `ConfigError` here rather
        than a silent runtime shape error inside the v1 ``linear_e``
        (attention.py:535). No-op when no edge port is configured.

        Also builds the optional encoder/global FiLM (M7 W-FILM): the per-layer
        encoder FiLMs are sized ``num_features = dim`` (the encoder embed width
        the FiLM scales at the start of each layer, v1 transformer.py:727-728) and
        populated into the absorbed `Transformer`'s ``featurewise`` ModuleList; the
        global FiLM is sized ``num_features = out_dim`` (the encoder output width
        it scales before pooling, v1 saltmodel.py:165-166). ``num_params`` is the
        resolved ``parameters`` width on both.

        Raises
        ------
        ConfigError
            When the resolved edge-embed width differs from ``edge_embed_dim``.
        """
        if self._encoder_film_cfg is not None or self._global_film_cfg is not None:
            num_params = schema.width(self.params_key)
            if self._encoder_film_cfg is not None:
                # one FiLM per encoder layer — v1 replicates the SAME config across
                # all num_layers layers (saltmodel.py:268-269). Populate the
                # absorbed Transformer's featurewise ModuleList (its verbatim
                # forward applies featurewise[i](params, x) per layer).
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

        Two export-time transformations (design §2.5/§7.2; the `OnnxAdapter`
        invokes this on every submodule, adapter.py:285-295):

        1. Force the deterministic torch-math attention backend (test/ONNX
           semantics, the v1 ``change_attn_backends`` replacement).
        2. **muP MuReadout -> plain nn.Linear fold** (plan 12 sub-wave B export
           contract). When ``self.mup`` the v1 out projection is a
           ``mup.MuReadout`` whose forward applies an output multiplier
           (``output_mult * x / width_mult``) before the linear — a
           NON-``nn.Linear`` op that traces to an unsupported/incorrect graph.
           `_fold_mu_readout` swaps it for a plain ``nn.Linear`` with the
           multiplier baked into the weights: ``W_folded = (output_mult /
           width_mult) * W``, ``bias_folded = bias`` (the multiplier scales only
           the ``x @ Wᵀ`` term, NOT the bias). The folded forward is numerically
           equal to the MuReadout forward (bitwise when
           ``output_mult/width_mult == 1.0``, the architectural default; ``<=1e-6``
           otherwise). Idempotent — a second call no-ops once the swap has
           happened (the out-proj is then already a plain `nn.Linear`).
        """
        # EdgeAttention has no pluggable backend — it is ALWAYS raw torch
        # attention (set_backend just warns, attention.py:544-549) and is
        # already trace-safe; only switch the backend for the non-edge encoder
        # (the v1 EncoderLayer skips the backend assignment when edge_embed_dim>0,
        # transformer.py:599-601,616-620 — the now-NAMED constraint ED2 enforces).
        if self.edges_key is None:
            self.encoder.set_backend("torch-math")
        if self.mup:
            self._fold_mu_readout()

    def _fold_mu_readout(self) -> None:
        """Fold the composed v1 `MuReadout` out-proj into a plain `nn.Linear` for export.

        Deterministic and numerically equal to the `MuReadout` forward within
        parity tolerance (plan 12 sub-wave B). No-op unless the encoder has a
        `MuReadout` out projection (a non-mup encoder, or an already-folded one,
        is left untouched — idempotent).
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

        With ``drop_registers`` the composed v1 `Transformer` strips the
        register rows from its output and deletes the ``"REGISTERS"`` pad entry
        (transformer.py:745-750), so only ``encoded.seq`` is produced — no
        ``masks.registers`` (the produce is gated out in `declare_io`).

        When an `edges` port is configured (M6 sub-wave C; FD §6.7 1422-1424)
        the edge-embed tensor ``edges.<stream>_emb`` ``[B, T, T, D_e]`` is passed
        as the v1 ``edge_x`` kwarg. The register zero-pad to the
        register-augmented sequence length stays INSIDE the composed v1
        `Transformer` (transformer.py:689-719) — it builds the pad from the
        DYNAMIC ``x.shape[1]``, the SHAPE-DERIVED pad the ONNX trace needs (FD
        §6.7 ONNX note; ED1 ONNX-trace assertion). The per-layer edge update is
        internal (transformer.py:729-733), so this module still publishes only
        ``encoded.seq`` (+ the optional register mask) — no edge output key.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        # FRESH dicts: v1's _add_registers INSERTS a "REGISTERS" key into
        # both (transformer.py:777,785) — never hand it bundle-owned dicts.
        xs: dict[str, Tensor] = {"seq": b.get("seq.x")}
        pad: dict[str, Tensor] = {"seq": b.get("seq.mask")}
        kwargs: dict[str, Tensor] = {}
        if self.edges_key is not None:
            kwargs["edge_x"] = b.get(self.edges_key)
        if len(self.encoder.featurewise) > 0:
            # encoder-layer FiLM (M7 W-FILM): thread the per-event parameters into
            # the absorbed Transformer forward; its verbatim loop applies
            # featurewise[i](params, x) at the start of each layer (v1
            # transformer.py:727-728). The v2 FiLM signature is forward(params, x),
            # so `inputs` IS the [B, n_params] parameters tensor here.
            kwargs["inputs"] = b.get(self.params_key)
        encoded, out_pad = self.encoder(xs, pad_mask=pad, **kwargs)
        if self.featurewise_global is not None:
            # global-layer FiLM (M7 W-FILM): scale/bias the encoder OUTPUT before
            # it is pooled, exactly as v1 saltmodel.py:165-166
            encoded = self.featurewise_global(b.get(self.params_key), encoded)
        if self.drop_registers:
            # registers stripped from encoded.seq; v1 also removed "REGISTERS"
            # from the pad dict, so there is no register mask to publish
            return {"encoded.seq": encoded}
        return {"encoded.seq": encoded, "masks.registers": out_pad["REGISTERS"]}


class Split(nn.Module):
    """Per-stream slices of ``encoded.seq`` via the ``seq.layout`` meta leaf.

    The PRODUCTION task path (design §3.3): tasks consume per-stream
    ``encoded.<stream>`` tensors instead of reconstructing slices from
    pad-mask dict order (v1 ``input_name_mask``, task.py:58-78). Register
    rows sit AFTER every stream in the encoder output, so the pre-register
    layout offsets remain valid slices of ``encoded.seq``.

    Export-mode implementation (design §7 / risk 7, adjudicated empirically
    in the M4 recipe spike, 2026-06-12): the eager branch slices with
    Python-int ``seq.layout`` offsets, which ``dynamo=False`` tracing bakes
    as constants. A single-stream probe (L=0..60) showed the JIT tracer
    keeps ``size()``-derived ints symbolic through single-stream slicing —
    silently CORRECT for one dynamic sequence axis — but a two-stream probe
    mis-sliced at ALL 15 (L_trk, L_el) grid points: with >=2 dynamic axes
    the baked offsets are provably wrong. The ONNX branch therefore slices
    with ``index_select`` over an index range built from the `Concat`
    ``seq.offsets`` tensor (``shape_as_tensor``-derived, design §7
    mechanism (A)) — proven correct on the full two-axis grid including
    zero-length streams in the same spike; the design's fallback
    (per-stream encoder outputs) was NOT needed. The mode branch is static
    Python; eager FIT/VAL/TEST numerics are bit-identical to the M2 port,
    and ``index_select`` over a contiguous range equals the eager narrow
    slicing exactly.
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
        additionally required (the trace-safe slicing path, see the class
        docstring).

        Returns
        -------
        IO
            The declared requires/produces (one shared instance-scoped
            width symbol — slicing preserves the feature dim).
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
        ``seq.offsets`` (see the class docstring for the risk-7 evidence);
        the stream's position in the offsets table is its position in the
        ``seq.layout`` dict (static Python — `Concat` insertion order), so
        a `Split` over a stream SUBSET stays correct without knowing the
        full concat list.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
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
    """Config-constructed global attention pooling (design §5.1, explicit input/out ports).

    Composes a fresh v1 `GlobalAttentionPooling` built at `bind` (gate width
    inferred from the resolved input). Pools the register-augmented
    sequence with the post-register mask order streams-then-REGISTERS —
    exactly where v1's ``_add_registers`` leaves the dict
    (transformer.py:777,785; pooling cats mask values in dict order,
    pooling.py:56). The zero-token ONNX pad stays inside the composed v1
    class (pooling.py:59-63).

    Two wirings, one class (design §5.1; v1 saltmodel.py:90-93,155-156,
    169-170):

    - **With an encoder** (the GN2 path): ``input`` is ``encoded.seq`` and
      `TransformerEncoder` publishes ``masks.registers`` — the register rows
      sit after every stream, so the pad dict is streams-then-REGISTERS.
    - **Encoder-less** (M5; the DiPS/DeepSets family, every regression
      config + ``legacy/dips.yaml``): ``init_nets`` + ``pool_net`` with NO
      ``encoder:`` block, so nothing produces ``masks.registers`` (v1
      saltmodel.py:155-156 pools ``flatten_tensor_dict(xs)`` directly). The
      config points ``input`` at ``seq.x`` (the `Concat` output) and
      ``masks.registers`` is declared OPTIONAL — absent from the plan when no
      producer exists (planner `_collect_demand` skips optional requires,
      `_build_edges` binds no edge), so the config plan-compiles and the pad
      dict is just ``{"seq": seq.mask}``. v1's pooling cats mask values in
      dict order, so dropping the REGISTERS entry is the exact v1
      encoder-less semantics — NOT an approximation. The WITH-encoder path is
      untouched: when the encoder produces ``masks.registers`` the optional
      require still binds and the REGISTERS pad row is still consumed.
    """

    def __init__(self, input: str = "encoded.seq", out: str = "pooled.global") -> None:  # noqa: A002 - design §5.1 YAML surface name
        """Capture the explicit input/output ports (design §5.1)."""
        super().__init__()
        self.name = _UNNAMED
        self.input_key = input
        self.out_key = out
        self.pool_net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + masks -> the pooled vector.

        The input's width symbol is shared with the produced key, so the
        pooled width resolves from the producing module's declaration.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", sym_dim("L", self.name), width), dtype="float32"
                ),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                # OPTIONAL: produced by `TransformerEncoder` on the WITH-encoder
                # path, ABSENT on the encoder-less path (init_nets+pool_net, no
                # encoder — v1 saltmodel.py:90-93,155-156). Optional means the
                # planner drops it when no module produces it, so encoder-less
                # configs plan-compile (design §2.2; M5 encoder-less pooling).
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
        """Build the composed v1 pooling with the inferred gate width (design §2.3)."""
        self.pool_net = _GlobalAttentionPoolingV1(input_size=schema.width(self.input_key))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the sequence with the (optionally register-augmented) mask dict.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        assert self.pool_net is not None, "forward before bind()"
        x = {"seq": b.get(self.input_key)}
        # streams-then-REGISTERS dict order is numerics-critical: pooling
        # cats mask values in dict order (pooling.py:56). The encoder-less
        # path has no register row (no encoder produced one), so the pad dict
        # is just {"seq": seq.mask} — the exact v1 saltmodel.py:155-156,170
        # encoder-less semantics; absent optional ports are probed, not read
        # (executor optional-port contract).
        pad = {"seq": b.get("seq.mask")}
        if "masks.registers" in b:
            pad["REGISTERS"] = b.get("masks.registers")
        return {self.out_key: self.pool_net(x, pad_mask=pad)}


class LossSum(nn.Module):
    """Weighted sum of per-task losses -> ``loss.total`` (design §3.3).

    Owns loss combination outright, replacing the smeared v1 ownership
    (``ModelWrapper.total_loss`` wsum branch, modelwrapper.py:195-197 —
    per-task weights are applied INSIDE the tasks, as in v1, so the default
    here is a plain sum; `weights` is an extra per-loss-key multiplier).

    The ``losses.**`` auto-collection is a FRAMEWORK wildcard (design §3.3):
    the M1 kernel rejects wildcard *requires* (planner.py), so the
    narrowing happens framework-side — `collect_loss_keys` scans sibling
    modules' declared produces and `narrow` fixes the concrete key list
    before plan compilation (`SaltModule` calls both; tests may too). An
    explicit ``losses:`` config list skips collection entirely.
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
        """Whether the loss-key list is fixed (explicit config or `narrow`).

        Returns
        -------
        bool
            True once the concrete loss keys are known — the framework
            (`SaltModule`) narrows un-fixed instances before compile
            (design §3.3).
        """
        return self._loss_keys is not None

    @staticmethod
    def collect_loss_keys(
        modules: Mapping[str, GraphModule], mode: Mode = Mode.FIT
    ) -> tuple[str, ...]:
        """Scan sibling modules for declared ``losses.*`` produces (framework narrowing).

        Parameters
        ----------
        modules : Mapping[str, GraphModule]
            The full module dict (LossSum instances are skipped).
        mode : Mode, optional
            The mode whose declarations are scanned, by default `Mode.FIT`.

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
        """Declare the narrowed loss keys -> ``loss.total`` (TRAINING only).

        Returns
        -------
        IO
            Empty in TEST/ONNX (the module is mode-inactive there).

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
        """Sum the (optionally weighted) loss leaves.

        Returns
        -------
        dict[str, Tensor]
            ``{"loss.total": scalar}``.
        """
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        total = sum(self.weights.get(key, 1.0) * b.get(key) for key in self._loss_keys)
        return {"loss.total": total}


class LossGLS(LossSum):
    """Geometric-mean (GLS) combination of per-task losses -> ``loss.total`` (design §3.3).

    A NetModule SIBLING of `LossSum` (it subclasses it to share the
    ``losses.**`` framework wildcard, `declare_io`, and the
    `collect_loss_keys`/`narrow` integration — `SaltModule`'s narrow loop
    already keys off ``isinstance(_, LossSum)``, so a `LossGLS` is narrowed
    for free, saltmodule.py:157-158). The ONLY behavioural change is the
    combination rule (`forward`): the n-task GEOMETRIC MEAN
    ``(∏ losses)^(1/n)`` (GLS = geometric loss strategy), reproducing the v1
    ``loss_mode == "GLS"`` branch (modelwrapper.py:194-196) which `LossSum`'s
    weighted sum replaced for the default ``wsum`` mode.

    GLS does NOT utilise loss weights. v1 enforces this with a loud
    construction-time guard — ``assert all(task.weight == 1.0 for task in
    self.model.tasks)`` (modelwrapper.py:139-142) — because the geometric mean
    of per-task losses is only meaningful when no task is pre-scaled (a
    weighted task loss ``w*L`` contributes ``w^(1/n)`` to the product, an
    arbitrary rescale of the mean — silent divergence, plan 10 risk
    "LossGLS gates 14 configs"). v2 has TWO weight surfaces, both guarded to
    1.0:

    - This module's per-loss ``weights`` multiplier (the `LossSum` extra) —
      rejected in ``__init__`` so a GLS config can never carry one.
    - The task-side ``weight: float`` applied INSIDE each composed v1 head
      before it publishes ``losses.<task>`` (tasks.py:88, task.py:243) — the
      EXACT v1 ``task.weight`` surface. This is sibling state the loss module
      cannot see at construction, so the framework calls
      `check_task_weights(modules)` from `SaltModule.__init__`'s narrow loop
      (the v1 ctor guard's v2 home), failing loudly before any plan compiles.
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
            default None (auto-collected via `collect_loss_keys`/`narrow`,
            as for `LossSum`).
        weights : Mapping[str, float] | None, optional
            Accepted only for parity with the `LossSum` signature: GLS does
            NOT utilise weights, so any entry != 1.0 is rejected (v1
            modelwrapper.py:139-142).

        Raises
        ------
        ConfigError
            If any configured weight is not 1.0 (GLS ignores weights — set
            them to 1, or use `LossSum` for a weighted sum).
        """
        super().__init__(losses=losses, weights=weights)
        # exact == 1.0 is the faithful v1 semantic (modelwrapper.py:140 asserts
        # task.weight == 1.0); weights are config literals, never computed values
        if bad := {k: v for k, v in self.weights.items() if v != 1.0}:  # noqa: RUF069
            raise ConfigError(
                f"LossGLS: per-loss weights are not utilised by the geometric mean — got "
                f"{bad}; set all weights to 1.0, or use LossSum for a weighted sum "
                "(v1 modelwrapper.py:139-142)"
            )

    @staticmethod
    def check_task_weights(modules: Mapping[str, GraphModule]) -> None:
        """Assert every loss-producing task carries ``weight == 1.0`` (the v1 guard).

        The v2 home of v1's ``ModelWrapper.__init__`` GLS assertion
        (``all(task.weight == 1.0 for task in self.model.tasks)``,
        modelwrapper.py:139-142). Called by `SaltModule.__init__` when a
        `LossGLS` is present, BEFORE any `declare_io`/compile, so a weighted
        task under GLS fails loudly at assembly rather than silently
        rescaling the geometric mean (plan 10 risk). Inspects the public
        numeric ``weight`` every task module exposes (tasks.py:88 coerces it
        to ``float``; the guard accepts ``int`` too so a future un-coerced
        weight is still caught); modules without a numeric ``weight`` attribute
        (`Normaliser`, `Concat`, `LossSum`/`LossGLS`, ...) are ignored — only
        the loss producers carry it.

        Parameters
        ----------
        modules : Mapping[str, GraphModule]
            The full configured module dict.

        Raises
        ------
        ConfigError
            Naming each task whose ``weight`` is not 1.0.
        """
        offenders = {
            name: float(module.weight)
            for name, module in modules.items()
            if not isinstance(module, LossSum)
            # duck-typed numeric check (int OR float): the v2 task base coerces
            # ``self.weight = float(weight)`` (tasks.py:88) so a YAML ``weight: 2``
            # already arrives as 2.0 and is caught, but guarding ``(int, float)``
            # keeps a future task module that stored an un-coerced int weight from
            # silently slipping past the GLS guard. (LossSum carries ``weights`` —
            # a dict — not ``weight``, and is excluded above regardless.)
            and isinstance(getattr(module, "weight", None), (int, float))
            and float(module.weight) != 1.0  # noqa: RUF069 - exact, the v1 semantic
        }
        if offenders:
            raise ConfigError(
                f"LossGLS: GLS does not utilise task weights — set all task weights to 1.0, "
                f"got {offenders} (v1 modelwrapper.py:139-142)"
            )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Combine the loss leaves by their geometric mean.

        ``(∏_k losses[k])^(1/n)`` over the narrowed loss keys — the v1
        ``loss_mode == "GLS"`` reduction (``math.prod`` then ``pow(·, 1/n)``,
        modelwrapper.py:194-196). Weights are guaranteed 1.0 by ``__init__``
        and `check_task_weights`, so none appear here (a weighted product
        would be the divergence v1's guard forbids).

        Returns
        -------
        dict[str, Tensor]
            ``{"loss.total": (∏ losses)^(1/n)}``.
        """
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        product = math.prod(b.get(key) for key in self._loss_keys)
        return {"loss.total": torch.pow(product, 1.0 / len(self._loss_keys))}


def _loss_key(key: str) -> str:
    """Normalise a configured loss reference to a dotted ``losses.`` key.

    Returns
    -------
    str
        ``"losses.<name>"`` for bare task names; dotted keys unchanged.
    """
    return key if key.startswith("losses.") else f"losses.{key}"


def _reject_width_keys(who: str, cfg: Mapping[str, Any] | None, banned: tuple[str, ...]) -> None:
    """Reject configured width keys — widths are inferred at bind (design §2.3).

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
