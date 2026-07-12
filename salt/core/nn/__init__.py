"""salt.core.nn — model-side GraphModules for the v2 core.

Package-level exports are the standalone, config-constructed modules
(`salt.core.nn.modules`, `salt.core.nn.tasks`). v1-wrapping parity
modules that wrap LIVE v1 instances live in `salt.core.nn.wrappers` /
`salt.core.nn.from_v1` — import those explicitly, they are not re-exported.
"""

from __future__ import annotations

from salt.core.nn.bind import (
    BindError,
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.from_v1 import from_v1, v1_sinks, v1_sources
from salt.core.nn.maskdecoder import MaskDecoder
from salt.core.nn.maskformer_loss import MaskFormerMatchedLoss
from salt.core.nn.modules import (
    Concat,
    EdgeEmbed,
    EdgeFeatures,
    FeaturewiseTransformation,
    GlobalAttentionPooling,
    LossGLS,
    LossSum,
    MaskedInputNormaliser,
    Normaliser,
    PositionalEncoder,
    Split,
    StreamEmbed,
    TransformerEncoder,
    VectorConcat,
)
from salt.core.nn.state_dict import map_v1_state_dict
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)

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
    "from_v1",
    "map_v1_state_dict",
    "materialise_all",
    "resolve_bind_schema",
    "v1_sinks",
    "v1_sources",
]
