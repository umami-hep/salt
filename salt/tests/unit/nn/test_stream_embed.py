"""Unit tests for StreamEmbed (mirror of salt/model/modules/stream_embed.py)."""

from __future__ import annotations

import pytest
import torch

from salt.graph import (
    Bundle,
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.model.modules import (
    ResolvedSchema,
    StreamEmbed,
)
from salt.model.nn.dense import Dense
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
)
from salt.tests.unit.nn.conftest import B, T, fit_bundle


class TestStreamEmbed:
    def test_width_keys_rejected(self):
        with pytest.raises(ConfigError, match="input_size"):
            StreamEmbed(stream="tracks", out_dim=8, dense={"input_size": 21})

    def test_bind_infers_input_plus_context_width(self, gn2v2):
        modules, _, _ = gn2v2
        embed = modules["track_embed"]
        # 19 track vars + 2 jet context vars — initnet.py:54-59 inference
        assert embed.net.input_size == len(TRACK_VARIABLES) + len(JET_VARIABLES)
        assert embed.net.output_size == 16

    def test_declare_io(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, context=["normed.jets"])
        embed.name = "track_embed"
        io = embed.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"normed.tracks", "normed.jets"}
        assert set(flatten_spec(io.produces)) == {"embed.tracks"}
        # rank-agnostic produce (M7 W1.5 wave R): shape=None, width via derived_widths
        assert flatten_spec(io.produces)["embed.tracks"].shape is None
        assert embed.derived_widths({}) == {"embed.tracks": 8}

    def test_context_prepended_in_v1_order(self, gn2v2):
        """Single GN2 context entry reproduces v1's [global, stream] layout."""
        modules, _, _ = gn2v2
        b = fit_bundle()
        normed = modules["norm"](b, Mode.FIT)
        for key, val in normed.items():
            b.set(key, val)
        out = modules["track_embed"](b, Mode.FIT)
        x = torch.cat(
            [
                b.get("normed.jets").unsqueeze(1).expand(B, T, -1),
                b.get("normed.tracks"),
            ],
            dim=-1,
        )
        assert torch.equal(out["embed.tracks"], modules["track_embed"].net(x))


class TestStreamEmbedVector:
    """Rank INFERRED from the bound input (M7 W1.5 wave R; collapses the M6-6 flag)."""

    def test_declare_io_is_rank_agnostic(self):
        embed = StreamEmbed(stream="jets", out_dim=8)
        embed.name = "jet_embed"
        io = embed.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        prod = flatten_spec(io.produces)
        assert set(req) == {"normed.jets"}
        assert set(prod) == {"embed.jets"}
        # rank-agnostic: no flag, the rank flows from producer/consumer
        assert req["normed.jets"].shape is None
        assert prod["embed.jets"].shape is None

    def test_derived_widths_contribute_out_dim(self):
        """The ``embed.<s>`` width comes from `derived_widths` (shape=None produce)."""
        embed = StreamEmbed(stream="jets", out_dim=8)
        embed.name = "jet_embed"
        assert embed.derived_widths({}) == {"embed.jets": 8}

    def test_bind_infers_vector_width(self):
        """A vector stream's Dense input width is the resolved [B, F] width."""
        embed = StreamEmbed(stream="jets", out_dim=16, dense={"hidden_layers": [32]})
        embed.name = "jet_embed"
        embed.bind(ResolvedSchema(widths={"normed.jets": 2}))
        assert embed.net.input_size == 2
        assert embed.net.output_size == 16

    def test_forward_rank_two_bitwise_vs_plain_dense(self):
        """[B, F] embed forward == an INDEPENDENT no-context Dense (the module adds
        nothing beyond its net). DEL-1: the reference was the v1 Dense; the v2
        Dense is its verbatim port, so the assertion's teeth are unchanged."""
        torch.manual_seed(0)
        embed = StreamEmbed(stream="jets", out_dim=4, dense={"hidden_layers": [8, 8]})
        embed.name = "jet_embed"
        embed.bind(ResolvedSchema(widths={"normed.jets": 2}))
        # independent reference Dense with the SAME hyper-params + weights
        ref = Dense(input_size=2, output_size=4, hidden_layers=[8, 8])
        ref.load_state_dict(embed.net.state_dict())
        x = torch.randn(B, 2)
        b = Bundle()
        b.set("normed.jets", x)
        out = embed(b, Mode.FIT)
        assert out["embed.jets"].shape == (B, 4)
        assert torch.equal(out["embed.jets"], ref(x))


class TestStreamEmbedMup:
    """The ``mup:`` flag on `StreamEmbed` (M6 sub-wave B, plan 12 muP arch port)."""

    def test_default_is_not_mup(self):
        embed = StreamEmbed(stream="tracks", out_dim=8)
        assert embed.mup is False

    def test_mup_flag_threads_into_dense_at_bind(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, mup=True)
        embed.name = "track_embed"
        embed.bind(ResolvedSchema(widths={"normed.tracks": 5}))
        assert embed.mup is True
        assert embed.net.mup is True

    def test_mup_in_dense_rejected(self):
        # mup is a module-level init_arg, not a dense width/option key
        with pytest.raises(ConfigError, match="set mup on the module"):
            StreamEmbed(stream="tracks", out_dim=8, dense={"mup": True})

    # DEL-1: test_mup_init_matches_independent_v1_dense_mup retired with the v1
    # tree — its essence was v1-vs-v2 muP init parity (a v2-vs-v2 rewrite would
    # be circular). Parity closure: git checkout 29c67a1.

    def test_mup_forward_is_unchanged(self):
        """muP affects init only — the forward math is the standard Dense forward."""
        torch.manual_seed(1)
        embed = StreamEmbed(stream="tracks", out_dim=4, dense={"hidden_layers": [8]}, mup=True)
        embed.name = "track_embed"
        embed.bind(ResolvedSchema(widths={"normed.tracks": 5}))
        ref = Dense(input_size=5, output_size=4, hidden_layers=[8], mup=True)
        ref.load_state_dict(embed.net.state_dict())
        x = torch.randn(B, T, 5)
        b = Bundle()
        b.set("normed.tracks", x)
        out = embed(b, Mode.FIT)
        assert torch.equal(out["embed.tracks"], ref(x))
