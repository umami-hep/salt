"""Config-constructed MaskFormer object decoder for the v2 core (design §5.2, FD 1129-1141).

M5 sub-wave C (plan 10). The v2 spelling of v1's ``salt.models.maskformer.MaskDecoder``
(maskformer.py:12,47,61,74,124-201): a fixed bank of learnable object queries is refined
against the encoded constituent sequence by a stack of cross/self-attention decoder
layers, and from the final queries the module predicts per-object class logits/probs and
per-constituent mask logits.

M2/M5 porting policy (plan 05): this COMPOSES the verbatim v1 building blocks — the
learnable query `nn.Parameter`, the two pre-norms, the v1 `MaskDecoderLayer` stack, and
two v1 `Dense` heads (``class_net`` / ``mask_net``) — and reproduces v1's INFERENCE
forward path (the dummy-token trick + the layer loop + `get_preds`, maskformer.py:166-203
with ``labels=None``) byte-for-byte. Full code absorption is M7.

What is DELIBERATELY NOT here (FD 1139, 1158-1171, 1220-1227):

- **No loss.** v1's `MaskDecoder.forward` calls ``self.mask_loss(...)`` when labels are
  present (maskformer.py:200-201); the class CE, mask dice/focal, and matched regression
  loss have EXACTLY ONE owner — the separate ``salt.core.nn.MaskFormerMatchedLoss`` module
  (a FIT|VAL-only sibling, the design's single-ownership rule). This decoder only PRODUCES
  ``objects.{embed,class_logits,class_probs,masks}`` in every mode; it never builds a
  `MaskFormerLoss`/`HungarianMatcher`, so it carries no matcher/scipy weight.
- **No ``aux_loss`` deep supervision** (maskformer.py:42-44,186-198, the
  ``intermediate_outputs`` collection). PARKED (FD §10; shipped MaskFormer.yaml:39 sets
  ``aux_loss: false``) — re-deferred, not implemented.

Key vocabulary (FD 1129-1138): ``input`` (default ``encoded.seq``, the encoder output
with its registers ALREADY dropped — see `TransformerEncoder.drop_registers`) ->
``objects.embed`` ``[B, M, D]`` (the refined queries — what the object-regression task
reads), ``objects.class_logits`` / ``objects.class_probs`` ``[B, M, C]``, and
``objects.masks`` ``[B, M, T]`` (per-object mask logits over the constituents).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.modules import _stream_len

# composed v1 layers (M2 porting policy, plan 05 — absorbed at M7)
from salt.models import Dense as V1Dense
from salt.models.maskformer import MaskDecoderLayer as V1MaskDecoderLayer
from salt.models.maskformer import get_masks as v1_get_masks

__all__ = ["MaskDecoder"]

_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned before compile (design §2.2)."""


class MaskDecoder(nn.Module):
    """MaskFormer object decoder over an encoded constituent sequence (design §5.2).

    Composes the v1 `MaskDecoderLayer` stack + two v1 `Dense` heads and reproduces v1's
    INFERENCE forward (the dummy-token trick + layer loop + `get_preds`,
    maskformer.py:166-203 with ``labels=None``) verbatim. Produces only the object
    predictions — the loss is owned by the separate `MaskFormerMatchedLoss` (FD 1139).

    Lifecycle (design §2.3): ``__init__`` captures config and builds the width-fixed
    submodules (the query bank, layers, and heads are all sized by the configured
    ``embed_dim``, so no schema-derived width exists — `bind` only validates the input
    width matches ``embed_dim``). ``declare_io`` is static; forward reads ``input`` and
    returns ONLY the four new ``objects.*`` keys (write-once, design §2.1).

    The ``num_classes`` the matched loss uses is ``class_net.output_size - 1`` (the last
    class is the "null"/no-object category, validated at construction): for the shipped
    MaskFormer.yaml ``class_net.output_size == 3`` (b, c, null) -> ``num_classes == 2``
    (maskformer.py:48-49, MaskFormer.yaml:49,58). The binary special case
    (``output_size == 1``) reproduces v1's sigmoid expansion to a 2-column ``class_probs``
    (maskformer.py:106-110).
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
        """Capture config and build the width-fixed submodules (design §2.3).

        Parameters
        ----------
        embed_dim : int
            Query / node embedding width (the encoder ``out_dim``; v1
            maskformer.py:48,61). The query bank, decoder layers, and the
            ``mask_net`` output are all sized by it.
        num_objects : int
            Number of learnable object queries ``M`` (v1 ``num_objects``,
            maskformer.py:54,61).
        num_layers : int
            Number of decoder layers (v1 maskformer.py:67-69).
        class_net : Mapping[str, Any]
            ``Dense`` config for the class head (mapping queries ``[B, M, D]`` ->
            class logits ``[B, M, C]``). MUST set ``output_size`` (the class count
            ``C``, last class = null) and MUST NOT set ``input_size`` (it is
            ``embed_dim``, design §2.3 kills YAML width arithmetic).
        md : Mapping[str, Any] | None, optional
            Per-layer `MaskDecoderLayer` config — ``n_heads`` (REQUIRED),
            ``mask_attention``, ``bidirectional_ca`` (v1 maskformer.py:40,68,
            MaskFormer.yaml:40-43), by default None (rejected: ``n_heads`` is
            required).
        mask_net : Mapping[str, Any] | None, optional
            ``Dense`` config for the mask head (queries -> mask tokens ``[B, M, D]``).
            ``output_size`` defaults to ``embed_dim`` (the einsum needs matching
            widths, maskformer.py:234); ``input_size`` is forbidden, by default None
            (a plain ``Dense(embed_dim, embed_dim)``, MaskFormer.yaml:51-55).
        input : str, optional
            The encoded-sequence key, by default ``encoded.seq`` (the encoder output
            with registers already dropped; FD 1132). v1 reads ``preds["embed_xs"]``
            (maskformer.py:172).
        out_stream : str, optional
            The produced object stream name, by default ``objects`` (so the keys are
            ``objects.embed`` etc. and the object-regression task reads
            ``objects.embed``; FD 1145).

        Raises
        ------
        ConfigError
            On a non-positive ``embed_dim`` / ``num_objects`` / ``num_layers``, a
            ``class_net`` / ``mask_net`` that sets ``input_size`` (inferred from
            ``embed_dim``), a ``class_net`` missing ``output_size`` or with
            ``output_size < 1``, or an ``md`` config missing ``n_heads``.
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

        # v1 query bank + pre-norms (maskformer.py:61-62,71-72)
        self.inital_q = nn.Parameter(torch.empty((num_objects, embed_dim)))
        nn.init.normal_(self.inital_q)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # v1 heads: class_net [D -> C], mask_net [D -> D] (maskformer.py:64-65)
        self.class_net = V1Dense(input_size=embed_dim, output_size=n_classes, **class_cfg)
        self.mask_net = V1Dense(
            input_size=embed_dim, output_size=mask_cfg.pop("output_size", embed_dim), **mask_cfg
        )

        # v1 layer stack — every layer shares the ONE mask_net (maskformer.py:67-69)
        self.layers = nn.ModuleList([
            V1MaskDecoderLayer(embed_dim, mask_net=self.mask_net, **md_cfg)
            for _ in range(num_layers)
        ])

    def _input_stream(self) -> str:
        """The constituent stream name the masks span (``input`` after the dotted prefix).

        Returns
        -------
        str
            ``"seq"`` for ``encoded.seq``, ``"tracks"`` for ``encoded.tracks``, ... — the
            second dotted component of `input`, used only for the symbolic mask token dim.
        """
        parts = self.input_key.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``input`` + ``seq.mask`` -> the four ``objects.*`` keys (all modes).

        The input is the encoded constituent sequence ``[B, T, D]`` (registers already
        dropped) and ``seq.mask`` ``[B, T]`` suppresses padded constituents in the mask
        logits (maskformer.py:113,236-239). All four products are active in every mode —
        FIT|VAL feed the matched loss / metrics, TEST|ONNX feed the object writer / export
        reduces (FD 1137-1138, 1211-1214).

        Returns
        -------
        IO
            The declared requires/produces.
        """
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
        """Validate the resolved input width equals ``embed_dim`` (design §2.3).

        All submodules are width-fixed at ``__init__`` (sized by ``embed_dim``), so there
        is nothing to construct here — but a mismatched encoder output width would only
        surface as a cryptic matmul error at the first forward, so it is caught loudly now.

        Raises
        ------
        ConfigError
            If the resolved ``input`` width is not ``embed_dim``.
        """
        width = schema.width(self.input_key)
        if width != self.embed_dim:
            raise ConfigError(
                f"MaskDecoder {self.name!r}: input {self.input_key!r} resolves to width {width} "
                f"but embed_dim is {self.embed_dim} — they must match (the queries attend to the "
                f"encoded sequence; v1 maskformer.py:48,172)"
            )

    def set_export_mode(self) -> None:
        """Force the deterministic torch-math attention backend (design §7.2, §2.5).

        The composed v1 `MaskDecoderLayer`s build their cross/self attention with the v1
        default backend (``torch-meff``, attention.py); the export/test path needs the
        deterministic ``torch-math`` SDPA kernel, exactly as `TransformerEncoder.
        set_export_mode` forces it on the encoder (modules.py:915). The `OnnxAdapter`
        invokes this recursively before tracing (adapter.py:144-147).
        """
        for module in self.modules():
            backend = getattr(module, "set_backend", None)
            if callable(backend):
                backend("torch-math")

    def _get_preds(self, q: Tensor, x: Tensor, pad_mask: Tensor | None) -> dict[str, Tensor]:
        """Class logits/probs + mask logits from queries (v1 `get_preds`, maskformer.py:104-115).

        Reproduces v1 verbatim: the class head maps queries -> logits, the binary special
        case (``output_size == 1``) sigmoid-expands to a 2-column ``class_probs`` while the
        multi-class case softmaxes, and the mask logits come from v1 `get_masks` (the
        ``einsum('bqe,ble->bql')`` of ``mask_net(q)`` against ``x``, with padded positions
        suppressed; maskformer.py:113,234-239).

        Returns
        -------
        dict[str, Tensor]
            ``{"class_logits", "class_probs", "masks"}`` (the ``objects.*`` suffixes).
        """
        class_logits = self.class_net(q)
        if class_logits.shape[-1] == 1:
            class_probs = class_logits.sigmoid()
            class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
        else:
            class_probs = class_logits.softmax(-1)
        pred_masks = v1_get_masks(x, q, self.mask_net, pad_mask)
        return {"class_logits": class_logits, "class_probs": class_probs, "masks": pred_masks}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Refine the queries against the encoded sequence; produce the object predictions.

        Reproduces v1 `MaskDecoder.forward`'s INFERENCE path (``labels=None``,
        maskformer.py:166-203) verbatim: expand + pre-norm the query bank, pre-norm the
        encoded sequence, append a single dummy (zero) constituent token to keep ONNX
        happy with a possibly-empty sequence (maskformer.py:177-184), run the decoder
        layers, take `get_preds` on the final queries, then UN-PAD — slice the dummy token
        out of both ``x`` and the mask logits (maskformer.py:194-196). The dummy-token
        trick lives entirely INSIDE this module in every mode (FD 1138), so the export
        graph traces a fixed shape regardless of the dynamic constituent count.

        Returns
        -------
        dict[str, Tensor]
            The four new ``objects.*`` keys only (design §2.5). ``objects.embed`` is the
            refined queries ``[B, M, D]`` (the object-regression task input); the
            intermediate un-padded ``x`` (v1's ``objects.x``, maskformer.py:195) is an
            internal value, not a declared product.
        """
        del mode
        x = b.get(self.input_key)
        pad_mask = b.get("seq.mask")

        # expand + pre-norm the learnable queries; pre-norm the sequence
        # (v1 maskformer.py:174-175)
        q = self.norm1(self.inital_q.expand(x.shape[0], -1, -1))
        x = self.norm2(x)

        # append a dummy zero token to the sequence (and the mask) so ONNX never
        # sees a zero-length attention dimension (v1 maskformer.py:177-184)
        xpad = torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        x = torch.cat([x, xpad], dim=1)
        padpad = torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=pad_mask.dtype)
        pad_mask = torch.cat([pad_mask, padpad], dim=1)

        # refine the queries through the decoder layer stack (v1 maskformer.py:187-191,
        # aux_loss path omitted — PARKED)
        for layer in self.layers:
            q, x = layer(q, x, kv_mask=pad_mask)
        preds = self._get_preds(q, x, pad_mask)

        # un-pad: drop the dummy token from x and the mask logits (v1 maskformer.py:194-196)
        masks = preds["masks"][:, :, :-1]
        return {
            f"{self.out_stream}.embed": q,
            f"{self.out_stream}.class_logits": preds["class_logits"],
            f"{self.out_stream}.class_probs": preds["class_probs"],
            f"{self.out_stream}.masks": masks,
        }
