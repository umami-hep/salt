"""Unit tests for bind-schema resolution (mirror of salt/core/nn/bind.py)."""

from __future__ import annotations

import pytest

from salt.core.nn import (
    BindError,
)
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
)

# bind schema resolution


class TestResolvedSchema:
    def test_widths_resolve_through_the_graph(self, gn2v2):
        """Symbolic widths bind from source/config declarations (design §2.3)."""
        _, _, schema = gn2v2
        assert schema.width("inputs.jets") == len(JET_VARIABLES)
        assert schema.width("inputs.tracks") == len(TRACK_VARIABLES)
        # through the Normaliser's shared symbol
        assert schema.width("normed.tracks") == len(TRACK_VARIABLES)
        # config-concrete produces
        assert schema.width("embed.tracks") == 16
        assert schema.width("encoded.seq") == 16
        # through Split / Pooling instance symbols
        assert schema.width("encoded.tracks") == 16
        assert schema.width("pooled.global") == 16

    def test_fields_come_from_the_source_declaration(self, gn2v2):
        _, _, schema = gn2v2
        assert schema.fields_of("inputs.tracks") == tuple(TRACK_VARIABLES)
        with pytest.raises(BindError, match="no declared fields"):
            schema.fields_of("encoded.seq")

    def test_unknown_width_raises_with_suggestions(self, gn2v2):
        _, _, schema = gn2v2
        with pytest.raises(BindError, match="encoded.tracks"):
            schema.width("encoded.trakcs")

    def test_meta_and_scalar_keys_have_no_width(self, gn2v2):
        _, _, schema = gn2v2
        for key in ("seq.layout", "loss.total"):
            with pytest.raises(BindError):
                schema.width(key)

    def test_conflicting_widths_raise(self):
        from salt.core.nn.bind import _DimBindings

        dims = _DimBindings()
        dims.bind("F:x", 3, "here")
        with pytest.raises(BindError, match="conflicting widths"):
            dims.bind("F:x", 4, "there")
