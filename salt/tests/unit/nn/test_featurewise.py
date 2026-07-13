"""Unit tests for FeaturewiseTransformation AND PositionalEncoder.

One test class covers both; named for the bigger half (mirrors featurewise.py + posenc.py).
"""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import (
    Bundle,
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.core.nn import (
    FeaturewiseTransformation,
    PositionalEncoder,
    ResolvedSchema,
    StreamEmbed,
    TransformerEncoder,
)


class TestFeaturewiseAndPosenc:
    """M7 W-FILM: v2-native FiLM + positional encoding wiring on StreamEmbed/encoder."""

    _DC = {"hidden_layers": [8], "activation": "ReLU"}

    def test_off_by_default_streamembed_noop(self):
        # no featurewise/pos_enc => no requires beyond input/context, no FiLM/PE
        embed = StreamEmbed(stream="tracks", out_dim=8)
        embed.name = "track_embed"
        io = embed.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"normed.tracks"}
        assert embed.featurewise is None
        assert embed.pos_enc is None

    def test_off_by_default_encoder_noop(self):
        enc = TransformerEncoder(dim=16, num_layers=2, attention={"num_heads": 2})
        enc.name = "encoder"
        io = enc.declare_io(Mode.FIT)
        # no inputs.parameters require when no FiLM configured
        assert "inputs.parameters" not in flatten_spec(io.requires)
        assert enc.featurewise_global is None
        # the absorbed Transformer's featurewise ModuleList stays empty
        assert len(enc.encoder.featurewise) == 0

    def test_streamembed_featurewise_declares_parameters(self):
        embed = StreamEmbed(
            stream="tracks",
            out_dim=8,
            featurewise={"dense_config_scale": dict(self._DC), "dense_config_bias": dict(self._DC)},
        )
        embed.name = "track_embed"
        req = flatten_spec(embed.declare_io(Mode.FIT).requires)
        assert "inputs.parameters" in req
        # rank-2 [B, n_params] global stream
        assert req["inputs.parameters"].shape == ("B", "P:track_embed")

    def test_streamembed_featurewise_built_at_bind(self):
        embed = StreamEmbed(
            stream="tracks",
            out_dim=8,
            featurewise={"dense_config_scale": dict(self._DC)},
        )
        embed.name = "track_embed"
        schema = ResolvedSchema(
            widths={"normed.tracks": 5, "inputs.parameters": 3},
            fields={"normed.tracks": tuple(f"v{i}" for i in range(5))},
        )
        embed.bind(schema)
        assert isinstance(embed.featurewise, FeaturewiseTransformation)
        # FiLM applied to the embed INPUT width (initnet.py:85-86)
        assert embed.featurewise.num_features == 5
        assert embed.featurewise.num_params == 3
        # forward consumes inputs.parameters and produces the embed
        x = torch.randn(4, 6, 5)
        p = torch.randn(4, 3)
        b = Bundle({"normed": {"tracks": x}, "inputs": {"parameters": p}})
        out = embed(b, Mode.FIT)["embed.tracks"]
        assert tuple(out.shape) == (4, 6, 8)

    def test_streamembed_pos_enc_resolves_indices_and_adds(self):
        embed = StreamEmbed(
            stream="tracks",
            out_dim=8,
            pos_enc={"variables": ["phi", "eta"]},
        )
        embed.name = "track_embed"
        schema = ResolvedSchema(
            widths={"normed.tracks": 4},
            fields={"normed.tracks": ("d0", "phi", "eta", "z0")},
        )
        embed.bind(schema)
        assert isinstance(embed.pos_enc, PositionalEncoder)
        # indices resolved by NAME (initnet.py:93-94): phi->1, eta->2
        assert embed.pos_enc_indices == (1, 2)
        assert embed.pos_enc.dim == 8  # defaults to out_dim
        x = torch.randn(4, 6, 4)
        b = Bundle({"normed": {"tracks": x}})
        out = embed(b, Mode.FIT)["embed.tracks"]
        # output = net(x) + pos_enc(x[..., (1,2)])
        expected = embed.net(x) + embed.pos_enc(x[..., (1, 2)])
        assert torch.equal(out, expected)

    def test_streamembed_pos_enc_unknown_variable_rejected(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, pos_enc={"variables": ["nope"]})
        embed.name = "track_embed"
        schema = ResolvedSchema(
            widths={"normed.tracks": 2}, fields={"normed.tracks": ("a", "b")}
        )
        with pytest.raises(ConfigError, match="not found"):
            embed.bind(schema)

    def test_streamembed_featurewise_non_input_layer_rejected(self):
        with pytest.raises(ConfigError, match="layer must be 'input'"):
            StreamEmbed(
                stream="tracks",
                out_dim=8,
                featurewise={"layer": "encoder", "dense_config_scale": dict(self._DC)},
            )

    def test_encoder_featurewise_builds_per_layer_and_global(self):
        enc = TransformerEncoder(
            dim=16,
            num_layers=3,
            out_dim=8,
            attention={"num_heads": 2},
            featurewise=[
                {"layer": "encoder", "dense_config_scale": dict(self._DC)},
                {"layer": "global", "dense_config_scale": dict(self._DC)},
            ],
        )
        enc.name = "encoder"
        req = flatten_spec(enc.declare_io(Mode.FIT).requires)
        assert "inputs.parameters" in req
        schema = ResolvedSchema(widths={"inputs.parameters": 4})
        enc.bind(schema)
        # one FiLM per encoder layer, sized to dim; one global FiLM sized to out_dim
        assert len(enc.encoder.featurewise) == 3
        for film in enc.encoder.featurewise:
            assert film.num_features == 16
            assert film.num_params == 4
        assert enc.featurewise_global is not None
        assert enc.featurewise_global.num_features == 8

    def test_encoder_featurewise_input_layer_rejected(self):
        with pytest.raises(ConfigError, match="layer 'input' belongs on StreamEmbed"):
            TransformerEncoder(
                dim=16,
                num_layers=1,
                attention={"num_heads": 2},
                featurewise=[{"layer": "input", "dense_config_scale": dict(self._DC)}],
            )

    def test_featurewise_requires_a_net(self):
        with pytest.raises(ConfigError, match="at least one"):
            FeaturewiseTransformation(layer="input", num_params=2, num_features=4)

    def test_featurewise_bad_layer_rejected(self):
        with pytest.raises(ConfigError, match="layer must be one of"):
            FeaturewiseTransformation(
                layer="nope", num_params=2, num_features=4, dense_config_scale={"hidden_layers": [4]}
            )

    def test_posenc_dim_too_small_rejected(self):
        with pytest.raises(ConfigError, match="too small"):
            PositionalEncoder(variables=["a", "b", "c"], dim=2)
