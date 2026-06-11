"""salt.core.nn — v1-wrapping GraphModules for the GN2 forward-parity gate.

See `salt.core.nn.wrappers` for the wrapper classes (and the documented
deviations from design §3.3/§5.1) and `salt.core.nn.from_v1` for the
weight-sharing graph builder + plan-boundary helpers.
"""

from __future__ import annotations

from salt.core.nn.from_v1 import from_v1, v1_sinks, v1_sources
from salt.core.nn.wrappers import (
    Concat,
    ConstituentTask,
    GlobalObjectTask,
    Normaliser,
    Pooling,
    Split,
    StreamEmbed,
    TransformerEncoder,
)

__all__ = [
    "Concat",
    "ConstituentTask",
    "GlobalObjectTask",
    "Normaliser",
    "Pooling",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
    "from_v1",
    "v1_sinks",
    "v1_sources",
]
