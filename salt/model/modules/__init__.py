"""salt.model.modules — model-side GraphModules for the v2 core.

Package-level exports are the standalone, config-constructed modules
(one module per file under `salt.model.modules`, plus `salt.model.modules.tasks`).
"""

from __future__ import annotations

from salt.model.base import SaltModelModule
from salt.model.bind import (
    BindError,
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.model.modules.edge_embed import EdgeEmbed, EdgeFeatures
from salt.model.modules.losses import LossGLS, LossSum
from salt.model.modules.maskdecoder import MaskDecoder
from salt.model.modules.maskformer_matched_loss import MaskFormerMatchedLoss
from salt.model.modules.norm import MaskedInputNormaliser, Normaliser
from salt.model.modules.plumbing import Concat, Split, VectorConcat
from salt.model.modules.pooling import GlobalAttentionPooling
from salt.model.modules.stream_embed import StreamEmbed
from salt.model.modules.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.model.modules.transformer_encoder import TransformerEncoder
from salt.model.nn.featurewise import FeaturewiseTransformation
from salt.model.nn.posenc import PositionalEncoder

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
    "SaltModelModule",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
    "VectorConcat",
    "VertexingTaskModule",
    "bind_all",
    "materialise_all",
    "resolve_bind_schema",
]
