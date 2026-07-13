"""Config-constructed task modules (classification, regression, vertexing).

Each task composes a v1 task head (loss math kept verbatim) at `bind`, declares
its label/mask/context dependencies, and publishes ``preds.<stream>.<task>``
plus ``losses.<task>`` (FIT|VAL only).
"""

from __future__ import annotations

# private-helper re-exports: the pre-split module surface (`salt.core.nn.tasks.<x>`)
# stays importable — salt/core/main.py and tests import these by that path
from salt.core.nn.tasks.base import _checked_weight_source as _checked_weight_source
from salt.core.nn.tasks.base import _loss_cfg as _loss_cfg
from salt.core.nn.tasks.base import _loss_class as _loss_class
from salt.core.nn.tasks.base import _parse_expose as _parse_expose
from salt.core.nn.tasks.base import _TaskModuleBase as _TaskModuleBase
from salt.core.nn.tasks.classification import ClassificationTaskModule
from salt.core.nn.tasks.edge import VertexingTaskModule
from salt.core.nn.tasks.regression import RegressionTaskModule

__all__ = ["ClassificationTaskModule", "RegressionTaskModule", "VertexingTaskModule"]
