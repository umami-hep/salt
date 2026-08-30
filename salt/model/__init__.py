"""salt.model — the Lightning integration and model base classes.

Public surface (import-friendly canonical paths):
  - ``salt.model.SaltModule``        — the LightningModule (also at ``.saltmodule``)
  - ``salt.model.SaltModelModule``   — model-module base (also at ``.base``)

Model-side packages:
  - ``salt.model.modules`` — config-constructed GraphModules (stream_embed,
    transformer_encoder, pooling, norm, plumbing, losses, tasks, maskdecoder, …)
  - ``salt.model.nn``      — raw torch building blocks (transformer, attention,
    dense, layernorm, posenc, featurewise, matcher, MaskFormer loss internals)
"""

from __future__ import annotations

from salt.model.base import SaltModelModule
from salt.model.saltmodule import CKPT_KEY, SaltModule

__all__ = [
    "CKPT_KEY",
    "SaltModelModule",
    "SaltModule",
]
