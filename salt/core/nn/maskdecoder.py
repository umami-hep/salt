"""Config-constructed MaskFormer object decoder.

Refines a fixed bank of learnable object queries against an encoded constituent
sequence via a stack of cross/self-attention decoder layers, producing per-object
class logits/probs and per-constituent mask logits. Does not compute the loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import _UNNAMED, IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.attention import Attention
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.dense import Dense
from salt.core.nn.glu import GLU
from salt.core.nn.stream_embed import _stream_len

__all__ = ["MaskDecoder", "MaskDecoderLayer"]


def get_masks(
    x: Tensor,
    q: Tensor,
    mask_net: nn.Module,
    input_pad_mask: Tensor | None = None,
) -> Tensor:
    """Compute mask logits over input tokens conditioned on queries.

    Padded input positions are driven to the dtype minimum rather than left at the
    raw einsum value, so they vanish after a downstream softmax/sigmoid threshold.
    """
    mask_tokens = mask_net(q)
    pred_masks = torch.einsum("bqe,ble->bql", mask_tokens, x)

    if input_pad_mask is not None:
        pred_masks[input_pad_mask.unsqueeze(1).expand_as(pred_masks)] = torch.finfo(
            pred_masks.dtype
        ).min

    return pred_masks


class MaskDecoder(nn.Module):
    """MaskFormer object decoder over an encoded constituent sequence.

    Builds width-fixed submodules (query bank, decoder layers, class/mask heads) sized
    by ``embed_dim`` at construction. Produces only the object predictions
    (``objects.embed/class_logits/class_probs/masks``) — the loss lives in the separate
    `MaskFormerMatchedLoss`. ``num_classes`` is ``class_net.output_size - 1`` (the last
    class is the "null"/no-object category); ``output_size == 1`` is a binary special
    case that sigmoid-expands to a 2-column ``class_probs``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_objects: int,
        num_layers: int,
        class_net: Mapping[str, Any],
        md: Mapping[str, Any] | None = None,
        mask_net: Mapping[str, Any] | None = None,
        input: str = "encoded.seq",  # noqa: A002 - design §5.2 YAML surface name
        out_stream: str = "objects",
    ) -> None:
        """Capture config and build the width-fixed submodules.

        Parameters
        ----------
        embed_dim : int
            Query / node embedding width. Sizes the query bank, decoder layers,
            and the ``mask_net`` output.
        num_objects : int
            Number of learnable object queries ``M``.
        num_layers : int
            Number of decoder layers.
        class_net : Mapping[str, Any]
            ``Dense`` config for the class head. Must set ``output_size`` (the
            class count ``C``, last class = null) and must not set ``input_size``
            (inferred from ``embed_dim``).
        md : Mapping[str, Any] | None, optional
            Per-layer `MaskDecoderLayer` config — ``n_heads`` (required),
            ``mask_attention``, ``bidirectional_ca``.
        mask_net : Mapping[str, Any] | None, optional
            ``Dense`` config for the mask head (queries -> mask tokens). Defaults
            to a plain ``Dense(embed_dim, embed_dim)``; ``input_size`` is forbidden.
        input : str, optional
            The encoded-sequence key, by default ``encoded.seq``.
        out_stream : str, optional
            The produced object stream name, by default ``objects``.

        Raises
        ------
        ConfigError
            On a non-positive ``embed_dim`` / ``num_objects`` / ``num_layers``, a
            ``class_net`` / ``mask_net`` that sets ``input_size``, a ``class_net``
            missing ``output_size`` or with ``output_size < 1``, or an ``md``
            config missing ``n_heads``.
        """
        super().__init__()
        self.name = _UNNAMED
        if embed_dim < 1:
            raise ConfigError(f"MaskDecoder: embed_dim must be >= 1, got {embed_dim}")
        if num_objects < 1:
            raise ConfigError(f"MaskDecoder: num_objects must be >= 1, got {num_objects}")
        if num_layers < 1:
            raise ConfigError(f"MaskDecoder: num_layers must be >= 1, got {num_layers}")

        class_cfg = dict(class_net)
        if "input_size" in class_cfg:
            raise ConfigError(
                "MaskDecoder: class_net.input_size is inferred from embed_dim — remove it "
                "(design §2.3 kills YAML width arithmetic)"
            )
        n_classes = class_cfg.pop("output_size", None)
        if n_classes is None or n_classes < 1:
            raise ConfigError(
                "MaskDecoder: class_net.output_size is required and >= 1 (the object class "
                "count C, last class = null; v1 maskformer.py:48-49, MaskFormer.yaml:49)"
            )
        mask_cfg = dict(mask_net or {})
        if "input_size" in mask_cfg:
            raise ConfigError(
                "MaskDecoder: mask_net.input_size is inferred from embed_dim — remove it "
                "(design §2.3)"
            )
        md_cfg = dict(md or {})
        if "n_heads" not in md_cfg:
            raise ConfigError(
                "MaskDecoder: md config must contain 'n_heads' (the per-layer attention head "
                "count; v1 MaskDecoderLayer, maskformer.py:375-382, MaskFormer.yaml:43)"
            )

        self.embed_dim = embed_dim
        self.num_objects = num_objects
        self.num_classes = n_classes - 1  # last class is the null/no-object category
        self.input_key = input
        self.out_stream = out_stream

        self.inital_q = nn.Parameter(torch.empty((num_objects, embed_dim)))
        nn.init.normal_(self.inital_q)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.class_net = Dense(input_size=embed_dim, output_size=n_classes, **class_cfg)
        self.mask_net = Dense(
            input_size=embed_dim, output_size=mask_cfg.pop("output_size", embed_dim), **mask_cfg
        )

        # every layer shares the ONE mask_net instance, not a per-layer copy
        self.layers = nn.ModuleList([
            MaskDecoderLayer(embed_dim, mask_net=self.mask_net, **md_cfg)
            for _ in range(num_layers)
        ])

    def _input_stream(self) -> str:
        """The constituent stream name the masks span (``input`` after the dotted prefix)."""
        parts = self.input_key.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``input`` + ``seq.mask`` -> the four ``objects.*`` keys (all modes)."""
        del mode
        stream = self._input_stream()
        tok = _stream_len(stream)
        emb = sym_dim("E", self.name)
        n_classes = self.num_classes + 1
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(shape=("B", tok, emb), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", tok), dtype="bool", kind="pad_mask"),
            }),
            produces=unflatten_spec({
                f"{self.out_stream}.embed": TensorSpec(
                    shape=("B", self.num_objects, emb), dtype="float32"
                ),
                f"{self.out_stream}.class_logits": TensorSpec(
                    shape=("B", self.num_objects, n_classes), dtype="float32"
                ),
                f"{self.out_stream}.class_probs": TensorSpec(
                    shape=("B", self.num_objects, n_classes), dtype="float32"
                ),
                f"{self.out_stream}.masks": TensorSpec(
                    shape=("B", self.num_objects, tok), dtype="float32"
                ),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Validate the resolved input width equals ``embed_dim`` (all submodules are pre-sized)."""
        width = schema.width(self.input_key)
        if width != self.embed_dim:
            raise ConfigError(
                f"MaskDecoder {self.name!r}: input {self.input_key!r} resolves to width {width} "
                f"but embed_dim is {self.embed_dim} — they must match (the queries attend to the "
                f"encoded sequence; v1 maskformer.py:48,172)"
            )

    def set_export_mode(self) -> None:
        """Force the deterministic ``torch-math`` attention backend for export/test."""
        for module in self.modules():
            backend = getattr(module, "set_backend", None)
            if callable(backend):
                backend("torch-math")

    def _get_preds(self, q: Tensor, x: Tensor, pad_mask: Tensor | None) -> dict[str, Tensor]:
        """Class logits/probs + mask logits from queries; ``output_size==1`` sigmoid-expands."""
        class_logits = self.class_net(q)
        if class_logits.shape[-1] == 1:
            class_probs = class_logits.sigmoid()
            class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
        else:
            class_probs = class_logits.softmax(-1)
        pred_masks = get_masks(x, q, self.mask_net, pad_mask)
        return {"class_logits": class_logits, "class_probs": class_probs, "masks": pred_masks}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Refine the queries against the encoded sequence; produce the object predictions."""
        del mode
        x = b.get(self.input_key)
        pad_mask = b.get("seq.mask")

        q = self.norm1(self.inital_q.expand(x.shape[0], -1, -1))
        x = self.norm2(x)

        # append a dummy zero token (and mask) so ONNX never sees a zero-length
        # attention dimension when the constituent sequence is empty
        xpad = torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        x = torch.cat([x, xpad], dim=1)
        padpad = torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=pad_mask.dtype)
        pad_mask = torch.cat([pad_mask, padpad], dim=1)

        for layer in self.layers:
            q, x = layer(q, x, kv_mask=pad_mask)
        preds = self._get_preds(q, x, pad_mask)

        # un-pad: drop the dummy token appended above from the mask logits
        masks = preds["masks"][:, :, :-1]
        return {
            f"{self.out_stream}.embed": q,
            f"{self.out_stream}.class_logits": preds["class_logits"],
            f"{self.out_stream}.class_probs": preds["class_probs"],
            f"{self.out_stream}.masks": masks,
        }


class MaskDecoderLayer(nn.Module):
    """Single decoder layer used in `MaskDecoder`.

    Applies (1) cross-attention from queries to inputs, (2) self-attention
    among queries, (3) a gated feed-forward (GLU) update, and optionally
    (4) a bidirectional cross-attention update from inputs to queries.
    Mask-guided attention can be enabled to sparsify cross-attention.

    Parameters
    ----------
    mask_attention : bool
        If ``True``, build a boolean attention mask from predicted masks to
        restrict cross-attention to confident positions.
    bidirectional_ca : bool
        If ``True``, also update inputs via cross-attention from queries.
    mask_net : nn.Module
        Module mapping queries to mask tokens used when ``mask_attention=True``.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mask_attention: bool,
        bidirectional_ca: bool,
        mask_net: nn.Module,
    ) -> None:
        super().__init__()

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        self.q_ca = Attention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_sa = Attention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_dense = GLU(embed_dim)
        if bidirectional_ca:
            self.kv_ca = Attention(embed_dim=embed_dim, num_heads=n_heads)
            self.kv_dense = GLU(embed_dim)
        self.mask_net = mask_net
        # applied at the END of forward, after the last q/kv residual updates
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        kv_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one decoder layer step (cross-attn, self-attn, GLU, optional bidirectional CA)."""
        attn_mask = None
        if self.mask_attention:
            # True = attend, False = masked (transformers-2 SDPA convention)
            attn_mask = get_masks(kv, q, self.mask_net, kv_mask).sigmoid()
            attn_mask = (attn_mask > 0.9).detach()
            # a query with every position masked would get an all-False attention
            # row (NaN softmax) — force those rows fully open instead
            newmask = torch.all(attn_mask == 0, dim=-1, keepdim=True).expand(attn_mask.shape)

            attn_mask = attn_mask | newmask

        q = q + self.q_ca(q, kv=kv, kv_mask=kv_mask, attn_mask=attn_mask)
        q = q + self.q_sa(q)
        q = q + self.q_dense(q)

        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)
                newmask = torch.all(attn_mask == 1, dim=-1, keepdim=True).expand(attn_mask.shape)
                attn_mask = attn_mask | ~newmask.bool()

            kv = kv + self.kv_ca(kv, q, attn_mask=attn_mask)
            kv = kv + self.kv_dense(kv)
        q = self.norm1(q)
        kv = self.norm2(kv)
        return q, kv
