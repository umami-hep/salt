"""Tests for `salt.core.writers.base` — the `Writer` base + `validate_specs`.

`WriterCallback.validate_specs` proves writer inputs exist AND kind/dtype-unify
on the writer->producer edge, before the first batch (M5 sub-wave D; design
§2.7/§8). The `Writer` ABC contract these exercise lives in
``salt/core/writers/base.py``.

(Split out of the former monolithic ``test_writers.py``; shared fixtures /
toy-writers / constants come from ``salt.tests._fixtures.writers_common``.)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from salt.core.graph.errors import ConfigError, KindError, ShapeError
from salt.core.graph.spec import TensorSpec
from salt.core.writers import (
    InputCopyWriter,
    PadMaskWriter,
    TaskWriter,
    Writer,
    WriterCallback,
)
from salt.tests._fixtures.writers_common import (  # noqa: F401  (data/modules are fixtures)
    NamedFeatureWriter,
    data,
    modules,
)


def _stub_reader() -> SimpleNamespace:
    """A minimal H5StructuredReader surface for `WriterCallback._declare_ctx`."""
    groups = {
        "jets": SimpleNamespace(global_object=True, dataset="jets"),
        "tracks": SimpleNamespace(global_object=False, dataset="tracks"),  # the sequence stream
    }
    return SimpleNamespace(streams=("jets", "tracks"), groups=groups)


def _test_producer_specs(modules) -> dict[str, TensorSpec]:
    """Model TEST-produced ports unioned with the real dataset-boundary leaves.

    Reproduces `SaltModule._validate_writer_specs`'s union (model ports take
    precedence) with the boundary specs the shipped writers consume: the
    ``pad_mask`` masks (``PadMaskWriter``/``JetCountWriter``) and ``meta.rows``
    (``InputCopyWriter``) carry the kinds/dtypes the dataset boundary serves.
    """
    boundary = {
        "masks.jets": TensorSpec(shape=("B", "T:jets"), dtype="bool", kind="pad_mask"),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
        "meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta"),
        "inputs.jets": TensorSpec(shape=("B", "F:jets"), dtype="float32", kind="data"),
        "inputs.tracks": TensorSpec(shape=("B", "T:tracks", "F:tracks"), dtype="float32"),
    }
    return boundary | WriterCallback.model_producer_specs(modules)


class TestValidateSpecs:
    """`validate_specs` proves writer inputs exist AND kind/dtype-unify (§2.7)."""

    def test_shipped_writers_pass(self, modules):
        # the base2.yaml writer trio against real model ports + boundary
        cb = WriterCallback(
            modules={
                "inputs_copy": InputCopyWriter(),
                "tasks": TaskWriter(),
                "pad_mask": PadMaskWriter(),
            }
        )
        # no raise: TaskWriter's preds.* requires are dtype/kind unconstrained,
        # PadMaskWriter's masks.* is bool/pad_mask (matches boundary), meta.rows
        # is int64/meta (matches boundary)
        cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_custom_writer_named_feature_passes(self, modules):
        # NamedFeatureWriter requires inputs.jets float32/data — matches boundary
        cb = WriterCallback(modules={"feat": NamedFeatureWriter(), "tasks": TaskWriter()})
        cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_wrong_kind_is_loud(self, modules):
        # a writer demanding the pad mask as plain data — the mask-polarity /
        # label-vs-feature guard the kind type exists for
        class WrongKind(Writer):
            def requires(self, ctx):
                del ctx
                return {"masks.tracks": TensorSpec(shape=None, dtype="bool", kind="data")}

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"bad": WrongKind()})
        with pytest.raises(KindError, match=r"kind='data'.*kind='pad_mask'|writer 'bad'"):
            cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_wrong_dtype_is_loud(self, modules):
        # the FD §2.7 ask: a deliberately wrong-dtype writer input caught at
        # compile (the boundary serves masks.tracks as bool; the writer says f4)
        class WrongDtype(Writer):
            def requires(self, ctx):
                del ctx
                return {"masks.tracks": TensorSpec(shape=None, dtype="float32", kind="pad_mask")}

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"bad": WrongDtype()})
        with pytest.raises(ShapeError, match="dtype mismatch on 'masks.tracks'"):
            cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_wrong_dtype_on_model_pred_is_loud(self, modules):
        # also catch a wrong dtype declared against a MODEL-produced port
        # (classification publishes float32; the writer says int64)
        class WrongPredDtype(Writer):
            def requires(self, ctx):
                del ctx
                return {
                    "preds.jets.jets_classification": TensorSpec(dtype="int64", kind="data"),
                }

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"bad": WrongPredDtype()})
        with pytest.raises(ShapeError, match="preds.jets.jets_classification"):
            cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_missing_required_input_is_loud(self, modules):
        class MissingInput(Writer):
            def requires(self, ctx):
                del ctx
                return {"preds.tracks.does_not_exist": TensorSpec(dtype=None, kind="data")}

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"bad": MissingInput()})
        with pytest.raises(ConfigError, match="input does not exist"):
            cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_optional_absent_require_is_fine(self, modules):
        # an OPTIONAL require with no producer must NOT raise (the spec.optional
        # short-circuit, mirroring the planner's optional-edge handling)
        class OptionalAbsent(Writer):
            def requires(self, ctx):
                del ctx
                return {
                    "preds.tracks.maybe": TensorSpec(dtype=None, kind="data", optional=True),
                    "masks.tracks": TensorSpec(shape=None, dtype="bool", kind="pad_mask"),
                }

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"ok": OptionalAbsent()})
        cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))

    def test_unconstrained_dtype_unifies_with_anything(self, modules):
        # a writer leaving dtype=None accepts whatever the producer publishes —
        # the TaskWriter's deliberately unconstrained preds.* contract stays
        # legal even against a dtype-declared boundary key
        class Unconstrained(Writer):
            def requires(self, ctx):
                del ctx
                return {"masks.tracks": TensorSpec(shape=None, dtype=None, kind="pad_mask")}

            def columns(self, ctx):
                del ctx
                return {"jets": np.dtype([("x", "i4")])}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

        cb = WriterCallback(modules={"ok": Unconstrained()})
        cb.validate_specs(modules, _stub_reader(), _test_producer_specs(modules))
