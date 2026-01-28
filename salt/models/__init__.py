"""Model definitions from the SALT framework."""

from salt.models.dense import Dense
from salt.models.featurewise import FeaturewiseTransformation
from salt.models.initnet import InitNet
from salt.models.inputnorm import InputNorm
from salt.models.maskformer_loss import MaskFormerLoss
from salt.models.pooling import (
    GlobalAttentionPooling,
    NodeQueryGAP,
    Pooling,
)
from salt.models.posenc import PositionalEncoder
from salt.models.r21xbb import R21Xbb
from salt.models.saltmodel import SaltModel
from salt.models.task import (
    ClassificationTask,
    GaussianRegressionTask,
    RegressionTask,
    RegressionTaskBase,
    TaskBase,
    VertexingTask,
)
from salt.models.transformer import Transformer

# Alias for backward compatibility
TransformerV2 = Transformer

__all__ = [
    "ClassificationTask",
    "Dense",
    "FeaturewiseTransformation",
    "GaussianRegressionTask",
    "GlobalAttentionPooling",
    "InitNet",
    "InputNorm",
    "MaskFormerLoss",
    "NodeQueryGAP",
    "Pooling",
    "PositionalEncoder",
    "R21Xbb",
    "RegressionTask",
    "RegressionTaskBase",
    "SaltModel",
    "TaskBase",
    "Transformer",
    "TransformerV2",
    "VertexingTask",
]
