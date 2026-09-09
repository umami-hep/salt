"""Model definitions from the SALT framework."""

from salt.models.custom_losses import BetaNLLLoss
from salt.models.dense import Dense
from salt.models.edge_constructor import EdgeConstructor
from salt.models.featurewise import FeaturewiseTransformation
from salt.models.initnet import InitNet
from salt.models.inputnorm import InputNorm
from salt.models.mdn_task import MixtureDensityTask, MixtureGaussianNLLLoss
from salt.models.pooling import (
    ClassAttentionPooling,
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
from salt.models.transformer_v2 import TransformerV2

__all__ = [
    "BetaNLLLoss",
    "ClassAttentionPooling",
    "ClassificationTask",
    "Dense",
    "EdgeConstructor",
    "FeaturewiseTransformation",
    "GaussianRegressionTask",
    "GlobalAttentionPooling",
    "InitNet",
    "InputNorm",
    "MixtureDensityTask",
    "MixtureGaussianNLLLoss",
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
