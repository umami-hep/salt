"""Shared helpers for the producer test mirrors (split from test_producers.py, W45.2c)."""

from __future__ import annotations

from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.tasks import ClassificationTaskModule, RegressionTaskModule
from salt.core.outputs import TaskOutput


_FLOAT_TOL = 1e-6

_STREAM_J = "jets"
_STREAM_T = "tracks"


# helpers — build + bind a real task head; run its op on a synthetic bundle


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


def _bind_regression(
    stream, targets, *, sequence, denoms=None, norm=None, scaler=None, gaussian=False, fields=()
):
    """Build + bind a `RegressionTaskModule` against a hand-built schema."""
    module = RegressionTaskModule(
        stream=stream,
        targets=targets,
        sequence=sequence,
        gaussian=gaussian,
        target_denominators=denoms,
        norm_params=norm,
        scaler=scaler,
    )
    module.name = f"{stream}_reg"
    widths = {module.input_key: 8}
    schema = ResolvedSchema(widths=widths, fields={module.input_feature_key: fields})
    module.bind(schema)
    return module


def _producer(op, *, task="t", stream=_STREAM_J, name="out"):
    """Construct a named `TaskOutput` carrying `op`."""
    producer = TaskOutput(task=task, stream=stream, name=name, op=op)
    producer.name = "producer"
    return producer
