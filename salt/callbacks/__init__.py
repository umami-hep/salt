"""Lightning callbacks: bundle-consuming metrics and run-dir artifacts.

Includes checkpointing, a progress bar, confusion-matrix / MaskFormer metric
callbacks, a kinematic-reweight plotter, graph/plan artifact writers, and
the generated adapter that drives a runtime output sink from the Lightning test loop.
"""

from __future__ import annotations

from salt.callbacks.checkpoint import Checkpoint, StepCheckpoint
from salt.callbacks.confusion_matrix import ConfusionMatrix
from salt.callbacks.graph_artifacts import GraphArtifacts
from salt.callbacks.input_normalisation_plotter import InputNormalisationPlotter
from salt.callbacks.kinematic_reweight_plotter import KinematicReweightPlotter
from salt.callbacks.maskformer_confusion_matrix import MaskformerConfusionMatrix
from salt.callbacks.maskformer_metrics import MaskformerMetrics
from salt.callbacks.prior_closure_metrics import PriorClosureMetrics
from salt.callbacks.progress import ProgressBar
from salt.callbacks.schedule import StageScopedCallbacks, TrainingScheduleCallback
from salt.callbacks.sink_adapter import SinkAdapter, attach_runtime_sink

__all__ = [
    "Checkpoint",
    "ConfusionMatrix",
    "GraphArtifacts",
    "InputNormalisationPlotter",
    "KinematicReweightPlotter",
    "MaskformerConfusionMatrix",
    "MaskformerMetrics",
    "PriorClosureMetrics",
    "ProgressBar",
    "SinkAdapter",
    "StageScopedCallbacks",
    "StepCheckpoint",
    "TrainingScheduleCallback",
    "attach_runtime_sink",
]
