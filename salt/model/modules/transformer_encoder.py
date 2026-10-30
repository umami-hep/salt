"""TransformerEncoder GraphModule (config-constructed encoder front-end)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import (
    IO,
    KEY_SEP,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.bind import ResolvedSchema
from salt.model.modules.stream_embed import _stream_len
from salt.model.nn.featurewise import FeaturewiseTransformation
from salt.model.nn.transformer import Transformer

_SEQ_LEN = sym_dim("S", "seq")


_ENC_LEN = sym_dim("L", "enc")


class TransformerEncoder(SaltModelModule):
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
    directly (see `StreamEmbed`). This narrower scope is intentional, not a bug.

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

    _FILM_LAYERS = ("encoder", "global")
    """The FiLM ``layer`` placements the encoder hosts (``input`` lives on `StreamEmbed`)."""

    MUP_WIDTH_ARG = "dim"
    """The init_arg the muP shape-generation tooling (``salt mup-shapes``)
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
                "encoder (v1 GN2XE.yaml:79 edge_embed_dim with an edge_init_net), or neither"
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
            update_edges=bool(update_edges),
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
            from mup import set_base_shapes

            set_base_shapes(self.encoder, self.encoder, rescale_params=False)
        self.out_dim = self.encoder.out_dim
        self.num_registers = num_registers
        self.num_layers = int(num_layers)
        # FiLM configs by layer; one params key is shared by all entries (taken from any)
        self.params_key = "inputs.parameters"
        films: dict[str, dict[str, Any]] = {}
        for entry in featurewise or ():
            fw = dict(entry)
            layer = fw.get("layer")
            if (key := fw.pop("parameters", None)) is not None:
                self.params_key = key
            if layer == "input":
                raise ConfigError(
                    "TransformerEncoder featurewise: layer 'input' belongs on StreamEmbed "
                    "(featurewise:), not the encoder — only 'encoder'/'global' here"
                )
            if layer not in self._FILM_LAYERS:
                raise ConfigError(
                    f"TransformerEncoder featurewise: each entry needs layer in "
                    f"{{'encoder', 'global'}}, got {layer!r}"
                )
            if layer in films:
                # v1 replicates ONE encoder config across all layers (saltmodel.py:268-269)
                raise ConfigError(
                    f"TransformerEncoder featurewise: at most one {layer!r}-layer FiLM entry"
                )
            films[layer] = fw
        self._encoder_film_cfg = films.get("encoder")
        self._global_film_cfg = films.get("global")
        self.featurewise_global: FeaturewiseTransformation | None = None

    @property
    def edge_stream(self) -> str | None:
        """The edge port's stream (``"edges.tracks_emb"`` -> ``"tracks"``), or None."""
        if self.edges_key is None:
            return None
        # "edges.<stream>_emb" -> "<stream>": strip the namespace + _emb suffix
        return self.edges_key.split(KEY_SEP, 1)[-1].removesuffix("_emb")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` (+ edge-embed when `edges` is set) -> ``encoded.seq``."""
        del mode
        produces = {"encoded.seq": TensorSpec(shape=("B", _ENC_LEN, self.out_dim), dtype="float32")}
        if not self.drop_registers:
            produces["masks.registers"] = TensorSpec(
                shape=("B", self.num_registers), dtype="bool", kind="pad_mask"
            )
        requires = {
            "seq.x": TensorSpec(shape=("B", _SEQ_LEN, self.dim), dtype="float32"),
            "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
        }
        if self.edges_key is not None:
            # both token axes share T:<stream> — the dynamic-T export prerequisite
            stream = self.edge_stream
            assert stream is not None  # edges_key implies edge_stream (config invariant)
            tlen = _stream_len(stream)
            requires[self.edges_key] = TensorSpec(
                shape=("B", tlen, tlen, self.edge_embed_dim), dtype="float32"
            )
        if self._encoder_film_cfg is not None or self._global_film_cfg is not None:
            # per-event conditioning parameters feeding the encoder/global FiLM
            requires[self.params_key] = TensorSpec(
                shape=("B", sym_dim("P", self.name)), dtype="float32"
            )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the optional encoder/global FiLM modules; validate the resolved edge width."""
        enc_cfg, global_cfg = self._encoder_film_cfg, self._global_film_cfg
        if enc_cfg is not None or global_cfg is not None:
            num_params = schema.width(self.params_key)
            if enc_cfg is not None:
                # one FiLM per encoder layer, replicating the same config across all layers
                for _ in range(self.num_layers):
                    film = FeaturewiseTransformation(
                        num_params=num_params, num_features=self.dim, **enc_cfg
                    )
                    film.name = self.name
                    self.encoder.featurewise.append(film)
            if global_cfg is not None:
                self.featurewise_global = FeaturewiseTransformation(
                    num_params=num_params, num_features=self.out_dim, **global_cfg
                )
                self.featurewise_global.name = self.name
        if self.edges_key is None:
            return
        resolved = schema.width(self.edges_key)
        if resolved != self.edge_embed_dim:
            raise ConfigError(
                f"TransformerEncoder {self.name!r}: edge_embed_dim={self.edge_embed_dim} but the "
                f"resolved {self.edges_key!r} width is {resolved} — set edge_embed_dim to the "
                "EdgeEmbed out_dim (the encoder's EdgeAttention projections were sized from "
                "edge_embed_dim at construction, attention.py:535)"
            )

    def set_export_mode(self) -> None:
        """Prepare for tracing: force the torch-math attention backend and fold MuReadout.

        Idempotent — a second call no-ops.
        """
        # EdgeAttention has no pluggable backend — it is ALWAYS raw torch
        # attention and already trace-safe; only switch the non-edge encoder
        if self.edges_key is None:
            self.encoder.set_backend("torch-math")
        if self.mup:
            self._fold_mu_readout()

    def _fold_mu_readout(self) -> None:
        """Fold the `MuReadout` out-proj into a plain `nn.Linear`; no-op if already folded."""
        from mup import MuReadout

        proj = getattr(self.encoder, "out_proj", None)
        if not isinstance(proj, MuReadout):
            return  # non-mup / no out-proj / already folded — nothing to do
        # output_mult and width_mult scale only the linear term, not the bias
        # (MuReadout.forward: super().forward(output_mult * x / width_mult)).
        mult = float(proj.output_mult) / float(proj.width_mult())
        folded = nn.Linear(proj.in_features, proj.out_features, bias=proj.bias is not None)
        with torch.no_grad():
            folded.weight.copy_(proj.weight * mult)
            if proj.bias is not None:
                folded.bias.copy_(proj.bias)
        self.encoder.out_proj = folded.to(proj.weight.device, proj.weight.dtype).eval()

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Encode the sequence; publishes the register mask too unless ``drop_registers``."""
        del mode
        # FRESH dicts: _add_registers INSERTS a "REGISTERS" key into both —
        # never hand it bundle-owned dicts.
        xs, pad = {"seq": b.get("seq.x")}, {"seq": b.get("seq.mask")}
        kwargs: dict[str, Tensor] = {}
        if self.edges_key is not None:
            kwargs["edge_x"] = b.get(self.edges_key)
        if len(self.encoder.featurewise) > 0:
            # encoder-layer FiLM: the composed Transformer applies featurewise[i](params, x)
            kwargs["inputs"] = b.get(self.params_key)
        encoded, out_pad = self.encoder(xs, pad_mask=pad, **kwargs)
        if self.featurewise_global is not None:
            # global-layer FiLM: scale/bias the encoder OUTPUT before it is pooled
            encoded = self.featurewise_global(b.get(self.params_key), encoded)
        if self.drop_registers:
            # registers stripped from encoded.seq; no register mask to publish
            return {"encoded.seq": encoded}
        return {"encoded.seq": encoded, "masks.registers": out_pad["REGISTERS"]}
