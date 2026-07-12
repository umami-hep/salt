"""Unit tests for the setup-graph type system (`salt.core.graph.setup_spec`)."""

from __future__ import annotations

import pytest

from salt.core.graph.setup_spec import (
    SOURCE_KINDS,
    SetupIO,
    SourceSpec,
    flatten_source_spec,
    unflatten_source_spec,
)
from salt.core.graph.spec import TensorSpec


def src(kind="path", **kwargs):
    return SourceSpec(kind=kind, **kwargs)


# §3 type system — SourceSpec / SetupIO


class TestSourceSpec:
    def test_kinds(self):
        assert SOURCE_KINDS == ("path", "scalar")
        assert SourceSpec(kind="path").kind == "path"
        assert SourceSpec(kind="scalar").kind == "scalar"

    def test_no_tensor_attributes(self):
        # a setup leaf is structurally incapable of being mistaken for a tensor
        s = SourceSpec()
        assert not hasattr(s, "shape")
        assert not hasattr(s, "dtype")
        assert not hasattr(s, "fields")

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="invalid source kind"):
            SourceSpec(kind="data")  # a tensor Kind, not a SourceKind

    def test_stage_gating(self):
        s = SourceSpec(stages=("train",))
        assert s.active_in("train")
        assert not s.active_in("val")

    def test_bare_string_stages_rejected(self):
        with pytest.raises(TypeError, match="not a bare string"):
            SourceSpec(stages="train")

    def test_empty_stages_rejected(self):
        with pytest.raises(ValueError, match="active in no stage"):
            SourceSpec(stages=())

    def test_invalid_stage_rejected(self):
        with pytest.raises(ValueError, match="invalid setup stage"):
            SourceSpec(stages=("fit",))  # 'fit' is a Mode, not a SetupStage


class TestSetupIO:
    def test_default_empty(self):
        assert SetupIO().is_empty()

    def test_flatten_roundtrip(self):
        flat = {"source.r.train.pattern": src(), "artifacts.r.num": src(kind="scalar")}
        nested = unflatten_source_spec(flat)
        assert flatten_source_spec(nested) == flat

    def test_rejects_tensorspec_leaf(self):
        with pytest.raises((TypeError, ValueError)):
            SetupIO(produces={"x": TensorSpec()})  # tensor leaf on the setup face
