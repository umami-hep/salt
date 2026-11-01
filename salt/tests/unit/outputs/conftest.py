"""Shared helpers for the producer test mirrors."""

from __future__ import annotations

from salt.model.bind import ResolvedSchema
from salt.model.modules.tasks import ClassificationTaskModule

_FLOAT_TOL = 1e-6

_STREAM_J = "jets"
_STREAM_T = "tracks"


# helpers — build + bind a real task head


def _bind_classification(stream, label, class_names, sequence, *, loss=None, input_key=None):
    """Build + bind a `ClassificationTaskModule` against a hand-built schema."""
    module = ClassificationTaskModule(
        stream=stream,
        label=label,
        class_names=class_names,
        input=input_key,
        sequence=sequence,
        loss=loss,
    )
    module.name = f"{stream}_cls"
    schema = ResolvedSchema(widths={module.input_key: 8})
    module.bind(schema)
    return module
