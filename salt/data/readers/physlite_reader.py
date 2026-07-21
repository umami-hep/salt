"""`PhysliteReader` — an `XAODReader` preset for DAOD_PHYSLITE POOL files.

PHYSLITE ships pre-calibrated ``AnalysisJets`` and associates jets to tracks via
ElementLink aux vectors (``AnalysisJetsAuxDyn.GhostTrack`` -> ``InDetTrackParticles``)
rather than per-jet decorations, so constituent streams use the ElementLink
dereference path (``link_branch`` + ``target_collection`` on the group config).

Schema confirmed on a real file (exp 25, mc23 ttbar): jet collection
``AnalysisJets``, link branch ``AnalysisJetsAuxDyn.GhostTrack``
(``ElementLink<DataVector<xAOD::IParticle>>``: per-jet vectors of
``{m_persKey, m_persIndex}``), target container ``InDetTrackParticles``.
"""

from __future__ import annotations

from dataclasses import dataclass

from salt.data.readers.uproot_reader import UprootGroupConfig
from salt.data.readers.xaod_reader import XAODReader

__all__ = ["PhysliteGroupConfig", "PhysliteReader"]


@dataclass(frozen=True)
class PhysliteGroupConfig(UprootGroupConfig):
    """Per-stream reader configuration for `PhysliteReader`.

    Jet-level streams (``jagged=False``) map ``branches`` onto
    ``AnalysisJetsAuxDyn.<branch>``. Constituent streams (``jagged=True``) set
    ``link_branch`` (e.g. ``GhostTrack``) + ``target_collection`` (e.g.
    ``InDetTrackParticles``); ``branches`` then map onto
    ``<target_collection>AuxDyn.<branch>`` and are gathered per jet by the link
    vector's ``m_persIndex``. `pad_max` caps the per-jet track multiplicity.
    """


class PhysliteReader(XAODReader):
    """`XAODReader` preset for DAOD_PHYSLITE POOL files (``AnalysisJets``).

    See `XAODReader` for the full parameter set. The jet collection defaults to
    ``AnalysisJets`` (pre-calibrated), and constituent streams reach tracks through
    the ElementLink dereference path — configure ``link_branch: GhostTrack`` +
    ``target_collection: InDetTrackParticles`` on each track stream.
    """

    _DEP_NAME = "PhysliteReader"
    _DEP_EXTRA = "physlite"
    JET_COLLECTION = "AnalysisJets"
