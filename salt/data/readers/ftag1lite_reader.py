"""`FTAG1LiteReader` — an `XAODReader` preset for DAOD_FTAG1LITE POOL files.

FTAG1LITE ships per-jet track lists as direct double-jagged aux decorations
(``ft1l_trk_*``), so no ElementLink dereference is needed — this is just the xAOD
reader with the ``AntiKt4EMPFlowJets`` jet collection as the default preset.
"""

from __future__ import annotations

from dataclasses import dataclass

from salt.data.readers.uproot_reader import UprootGroupConfig
from salt.data.readers.xaod_reader import XAODReader

__all__ = ["FTAG1LiteGroupConfig", "FTAG1LiteReader"]


@dataclass(frozen=True)
class FTAG1LiteGroupConfig(UprootGroupConfig):
    """Per-stream reader configuration for `FTAG1LiteReader`.

    `branches` maps the v2 field name (``pt``, ``flavour_label``, ``d0``) to the
    bare aux-store branch name; the reader prepends ``<jet_collection>AuxDyn.``.
    `jagged=False` is a jet-level scalar stream (``[event][jet]`` -> ``(B,)``);
    `jagged=True` is a constituent stream (``[event][jet][track]`` -> ``(B,
    pad_max)`` padded). `pad_max` caps the per-jet constituent multiplicity; None
    auto-resolves the file-wide max.
    """


class FTAG1LiteReader(XAODReader):
    """`XAODReader` preset for DAOD_FTAG1LITE POOL files (``AntiKt4EMPFlowJets``).

    See `XAODReader` for the full parameter set; the only difference is the default
    jet collection. Constituent streams are direct ``ft1l_trk_*`` double-jagged aux
    decorations (no ElementLink dereference).
    """

    _DEP_NAME = "FTAG1LiteReader"
    _DEP_EXTRA = "root"
    JET_COLLECTION = "AntiKt4EMPFlowJets"
