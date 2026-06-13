"""salt.core.nn — model-side GraphModules for the v2 core.

Two families live here:

- **Standalone, config-constructed modules** (`salt.core.nn.modules`,
  `salt.core.nn.tasks` — M2, plan 05): the design §5.1 YAML surface
  (``class_path: salt.core.nn.Normaliser`` etc.), with the two-phase
  bind lifecycle (`salt.core.nn.bind`) and the v1->v2 state-dict mapping
  (`salt.core.nn.state_dict`). These are the package-level exports.
- **v1-wrapping parity modules** (`salt.core.nn.wrappers`,
  `salt.core.nn.from_v1` — plan 04): wrap LIVE v1 instances for the GN2
  forward-parity gate. Import their same-named classes explicitly from
  `salt.core.nn.wrappers`.
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
    GlobalAttentionPooling,
    LossGLS,
    LossSum,
    Normaliser,
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
    "GlobalAttentionPooling",
    "LossGLS",
    "LossSum",
    "MaskDecoder",
    "MaskFormerMatchedLoss",
    "Normaliser",
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
