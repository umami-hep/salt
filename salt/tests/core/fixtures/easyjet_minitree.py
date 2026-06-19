"""Deterministic synthetic easyjet ``AnalysisMiniTree`` ROOT fixture builder.

Writes a tiny ROOT file (a handful of events, jagged jets) mirroring the real
easyjet output-tree layout the `EasyjetReader` reads: ``recojet_antikt4PFlow_*``
jagged jet branches (float kinematics + GN2 scores + an integer
``HadronConeExclTruthLabelID`` flavour label) and scalar event-level branches
(``eventNumber``, ``mcChannelNumber``). Everything is generated from a fixed
seed so the committed regression test reproduces it byte-for-byte WITHOUT
depending on ``/data/atlas_samples`` — the plan's "≥1 reproducible fixture test"
requirement.

The numbers are deterministic NumPy arrays (the fixture IS the ground truth the
round-trip test compares against), so the builder also returns the raw per-event
awkward/numpy values for the test to assert against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Per-event jet multiplicities — deliberately uneven (incl. a 0-jet event) so the
# pad mask / valid-length logic is exercised on the boundary cases.
_NJETS = [3, 1, 5, 0, 2, 4]
TREE_NAME = "AnalysisMiniTree"


def build_fixture_arrays(seed: int = 1234) -> dict[str, Any]:
    """Build the deterministic per-event fixture values (ground truth).

    Returns
    -------
    dict[str, Any]
        ``njets`` (list[int]), ``n_events`` (int), and the per-event jagged jet
        fields (lists of np arrays) + scalar event fields (np arrays). Branch
        names match the real easyjet file. Used both to WRITE the ROOT file and
        as the round-trip ORACLE.
    """
    rng = np.random.default_rng(seed)
    njets = list(_NJETS)
    n = len(njets)
    # jagged jet fields (float kinematics + scores, int flavour label)
    pt = [rng.uniform(20_000, 300_000, size=k).astype(np.float32) for k in njets]
    eta = [rng.uniform(-2.5, 2.5, size=k).astype(np.float32) for k in njets]
    phi = [rng.uniform(-np.pi, np.pi, size=k).astype(np.float32) for k in njets]
    mass = [rng.uniform(5_000, 40_000, size=k).astype(np.float32) for k in njets]
    pb = [rng.uniform(0, 1, size=k).astype(np.float32) for k in njets]
    pc = [rng.uniform(0, 1, size=k).astype(np.float32) for k in njets]
    pu = [rng.uniform(0, 1, size=k).astype(np.float32) for k in njets]
    # flavour label: values from the easyjet universe {5=b,4=c,15=tau,0=light}
    label_universe = np.array([5, 4, 15, 0], dtype=np.int32)
    label = [
        label_universe[rng.integers(0, 4, size=k)].astype(np.int32) for k in njets
    ]
    # scalar event fields
    event_number = np.arange(1000, 1000 + n, dtype=np.uint64)
    mc_channel = np.full(n, 603404, dtype=np.uint32)
    return {
        "n_events": n,
        "njets": njets,
        "recojet_antikt4PFlow_pt_NOSYS": pt,
        "recojet_antikt4PFlow_eta": eta,
        "recojet_antikt4PFlow_phi": phi,
        "recojet_antikt4PFlow_m_NOSYS": mass,
        "recojet_antikt4PFlow_GN2v01_pb": pb,
        "recojet_antikt4PFlow_GN2v01_pc": pc,
        "recojet_antikt4PFlow_GN2v01_pu": pu,
        "recojet_antikt4PFlow_HadronConeExclTruthLabelID": label,
        "eventNumber": event_number,
        "mcChannelNumber": mc_channel,
    }


def write_minitree(path: Path, arrays: dict[str, Any]) -> Path:
    """Write the fixture arrays to a ROOT ``AnalysisMiniTree`` at ``path``.

    Returns
    -------
    Path
        The written file path.
    """
    import awkward as ak
    import uproot

    n = arrays["n_events"]
    njets = arrays["njets"]
    branches: dict[str, Any] = {}
    for name, val in arrays.items():
        if name in ("n_events", "njets"):
            continue
        if isinstance(val, list):  # jagged: list of per-event arrays
            branches[name] = ak.Array(val)
        else:  # scalar event-level
            branches[name] = val
    del n, njets
    path = Path(path)
    with uproot.recreate(path) as f:
        f[TREE_NAME] = branches
    return path
