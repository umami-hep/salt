"""Unit tests for `salt.model.validation` — edge-port / class-names / origin-weighting checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from salt.graph.errors import ConfigError
from salt.model.modules import Concat
from salt.model.validation import (
    _concat_first_stream,
    _edge_encoders,
    check_class_names,
    resolve_origin_weighting,
    validate_edge_port,
)


def _concat(streams: list[str]) -> Concat:
    """A `Concat` with `.name` set — `declare_io` reads `self.name`."""
    concat = Concat(streams=streams)
    concat.name = "concat"
    return concat


def _edge_encoder(
    attn_type: str = "torch-math", edge_stream: str = "tracks"
) -> SimpleNamespace:
    return SimpleNamespace(
        edges_key="edges.tracks_emb",
        edge_stream=edge_stream,
        encoder=SimpleNamespace(attn_type=attn_type),
    )


class TestEdgeEncoders:
    def test_selects_only_modules_with_a_non_none_edges_key(self) -> None:
        modules = {
            "e": SimpleNamespace(edges_key="edges.tracks_emb"),
            "n": SimpleNamespace(edges_key=None),
            "p": SimpleNamespace(),
        }
        assert _edge_encoders(modules) == [("e", modules["e"])]


class TestConcatFirstStream:
    def test_returns_the_name_and_first_stream(self) -> None:
        concat = _concat(["tracks", "jets"])
        assert _concat_first_stream({"concat": concat}) == ("concat", "tracks")

    def test_empty_modules_returns_none(self) -> None:
        assert _concat_first_stream({}) is None

    def test_no_seq_x_producer_returns_none(self) -> None:
        assert _concat_first_stream({"e": SimpleNamespace()}) is None


class TestValidateEdgePort:
    def test_no_edge_encoders_is_a_no_op(self) -> None:
        assert validate_edge_port({}) == 0

    def test_a_well_formed_edge_encoder_counts_one(self) -> None:
        modules = {"enc": _edge_encoder(), "concat": _concat(["tracks"])}
        assert validate_edge_port(modules) == 1

    def test_missing_concat_raises(self) -> None:
        modules = {"enc": _edge_encoder()}
        with pytest.raises(ConfigError, match="no Concat is configured"):
            validate_edge_port(modules)

    def test_edge_stream_not_first_raises(self) -> None:
        modules = {"enc": _edge_encoder(), "concat": _concat(["jets", "tracks"])}
        with pytest.raises(ConfigError, match="NOT the first stream"):
            validate_edge_port(modules)

    def test_unsupported_attn_type_raises(self) -> None:
        modules = {
            "enc": _edge_encoder(attn_type="flash-varlen"),
            "concat": _concat(["tracks"]),
        }
        with pytest.raises(ConfigError, match="EdgeAttention supports ONLY"):
            validate_edge_port(modules)


class TestCheckClassNames:
    @staticmethod
    def _reader(attrs: dict) -> SimpleNamespace:
        return SimpleNamespace(schema_group=lambda stream: SimpleNamespace(attrs=attrs))  # noqa: ARG005

    @staticmethod
    def _module(class_names: list[str]) -> SimpleNamespace:
        return SimpleNamespace(class_names=class_names, stream="jets", label="flavour_label")

    def test_matching_order_counts_one(self) -> None:
        reader = self._reader({"flavour_label": ["bjets", "cjets", "ujets"]})
        module = self._module(["bjets", "cjets", "ujets"])
        assert check_class_names({"jets_classification": module}, reader) == 1

    def test_reordered_class_names_raise_naming_the_config_path(self) -> None:
        reader = self._reader({"flavour_label": ["bjets", "cjets", "ujets"]})
        module = self._module(["bjets", "ujets", "cjets"])
        with pytest.raises(ConfigError, match="DIFFERENT ORDER") as excinfo:
            check_class_names({"jets_classification": module}, reader)
        assert "model.modules.jets_classification.init_args.class_names" in str(excinfo.value)

    def test_different_class_set_raises(self) -> None:
        reader = self._reader({"flavour_label": ["bjets", "cjets", "ujets"]})
        module = self._module(["bjets", "cjets", "taujets"])
        with pytest.raises(ConfigError, match="class sets differ"):
            check_class_names({"jets_classification": module}, reader)

    def test_reader_without_schema_group_counts_zero(self) -> None:
        module = self._module(["bjets", "cjets", "ujets"])
        assert check_class_names({"jets_classification": module}, SimpleNamespace()) == 0

    def test_schema_group_returning_none_counts_zero(self) -> None:
        reader = SimpleNamespace(schema_group=lambda stream: None)  # noqa: ARG005
        module = self._module(["bjets", "cjets", "ujets"])
        assert check_class_names({"jets_classification": module}, reader) == 0

    def test_non_string_list_attr_counts_zero(self) -> None:
        reader = self._reader({"flavour_label": [1, 2]})
        module = self._module(["bjets", "cjets"])
        assert check_class_names({"jets_classification": module}, reader) == 0

    def test_module_lacking_class_names_counts_zero(self) -> None:
        reader = self._reader({"flavour_label": ["bjets", "cjets", "ujets"]})
        module = SimpleNamespace(stream="jets", label="flavour_label")
        assert check_class_names({"jets_classification": module}, reader) == 0


class TestResolveOriginWeighting:
    def test_counts_the_resolvers_that_return_true(self) -> None:
        modules = {
            "a": SimpleNamespace(resolve_origin_names=lambda reader: True),  # noqa: ARG005
            "b": SimpleNamespace(resolve_origin_names=lambda reader: False),  # noqa: ARG005
            "c": SimpleNamespace(),
        }
        assert resolve_origin_weighting(modules, reader=object()) == 1
