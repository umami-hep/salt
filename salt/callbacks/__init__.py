"""Lightning callbacks: bundle-consuming metrics and run-dir artifacts.

Includes checkpointing, a progress bar, confusion-matrix / MaskFormer metric
callbacks, graph/plan artifact writers, and the generated adapter that drives
a runtime output sink from the Lightning test loop.
"""

from __future__ import annotations

from salt.callbacks.checkpoint import Checkpoint
from salt.callbacks.confusion_matrix import ConfusionMatrix
from salt.callbacks.graph_artifacts import GraphArtifacts
from salt.callbacks.maskformer_confusion_matrix import MaskformerConfusionMatrix
from salt.callbacks.maskformer_metrics import MaskformerMetrics
from salt.callbacks.progress import ProgressBar
from salt.callbacks.schedule import StageScopedCallbacks, TrainingScheduleCallback
from salt.callbacks.sink_adapter import SinkAdapter, attach_runtime_sink

__all__ = [
    "Checkpoint",
    "ConfusionMatrix",
    "GraphArtifacts",
    "MaskformerConfusionMatrix",
    "MaskformerMetrics",
    "ProgressBar",
    "SinkAdapter",
    "StageScopedCallbacks",
    "TrainingScheduleCallback",
    "attach_runtime_sink",
]
