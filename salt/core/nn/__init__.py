"""salt.core.nn — model-side GraphModules for the v2 core.

Package-level exports are the standalone, config-constructed modules
(one module per file under `salt.core.nn`, plus `salt.core.nn.tasks`).
"""

from __future__ import annotations

from salt.core.nn.bind import (
    BindError,
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.concat import Concat
from salt.core.nn.edge_embed import EdgeEmbed, EdgeFeatures
from salt.core.nn.featurewise import FeaturewiseTransformation
from salt.core.nn.losses import LossGLS, LossSum
from salt.core.nn.maskdecoder import MaskDecoder
from salt.core.nn.maskformer_matched_loss import MaskFormerMatchedLoss
from salt.core.nn.norm import MaskedInputNormaliser, Normaliser
from salt.core.nn.pooling import GlobalAttentionPooling
from salt.core.nn.posenc import PositionalEncoder
from salt.core.nn.split import Split
from salt.core.nn.state_dict import map_v1_state_dict
from salt.core.nn.stream_embed import StreamEmbed
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.core.nn.transformer_encoder import TransformerEncoder
from salt.core.nn.vector_concat import VectorConcat

__all__ = [
    "BindError",
    "ClassificationTaskModule",
    "Concat",
    "EdgeEmbed",
    "EdgeFeatures",
    "FeaturewiseTransformation",
    "GlobalAttentionPooling",
    "LossGLS",
    "LossSum",
    "MaskDecoder",
    "MaskFormerMatchedLoss",
    "MaskedInputNormaliser",
    "Normaliser",
    "PositionalEncoder",
    "RegressionTaskModule",
    "ResolvedSchema",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
    "VectorConcat",
    "VertexingTaskModule",
    "bind_all",
    "map_v1_state_dict",
    "materialise_all",
    "resolve_bind_schema",
]
