"""Deterministic synthetic easyjet ``AnalysisMiniTree`` ROOT fixture builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Per-event jet multiplicities — deliberately uneven (incl. a 0-jet event) so the
# pad mask / valid-length logic is exercised on the boundary cases.
_NJETS = [3, 1, 5, 0, 2, 4]
TREE_NAME = "AnalysisMiniTree"


def build_fixture_arrays(seed: int = 1234) -> dict[str, Any]:
    """Build the deterministic per-event fixture values (ground truth)."""
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


JET_BRANCHES = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "phi": "recojet_antikt4PFlow_phi",
    "m": "recojet_antikt4PFlow_m_NOSYS",
    "GN2v01_pb": "recojet_antikt4PFlow_GN2v01_pb",
    "GN2v01_pc": "recojet_antikt4PFlow_GN2v01_pc",
    "GN2v01_pu": "recojet_antikt4PFlow_GN2v01_pu",
}
"""Field name -> on-disk branch, matching ``salt/configs/readers/easyjet_events.yaml``.

``_NOSYS`` marks the systematics-affected variables (``pt``, ``m``) and is
deliberately absent from ``eta``/``phi`` — it is not a uniform suffix.
"""


def write_minitree(path: Path, arrays: dict[str, Any]) -> Path:
    """Write the fixture arrays to a ROOT ``AnalysisMiniTree`` at ``path``."""
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


def write_sample_pair(directory: Path) -> tuple[Path, Path]:
    """Two deterministic minitrees — a signal and a background sample.

    The pair a `MultiSampleReader` fragment needs: same schema, different files,
    different seeds. Returns ``(signal, background)``.
    """
    directory = Path(directory)
    signal = write_minitree(directory / "signal.root", build_fixture_arrays(seed=1))
    background = write_minitree(directory / "background.root", build_fixture_arrays(seed=2))
    return signal, background


def write_jets_norm_dict(path: Path, fields: list[str], seed: int = 1) -> Path:
    """A ``{jets: {field: {mean, std}}}`` norm dict over the fixture's own values.

    `Normaliser` rejects a non-finite or zero ``std``; a constant column (or an
    all-padding one) falls back to 1.0.
    """
    import yaml

    arrays = build_fixture_arrays(seed=seed)
    norm: dict[str, dict[str, dict[str, float]]] = {"jets": {}}
    for field in fields:
        chunks = [np.asarray(v) for v in arrays[JET_BRANCHES[field]] if len(v)]
        values = np.concatenate(chunks)
        std = float(np.std(values))
        norm["jets"][field] = {
            "mean": float(np.mean(values)),
            "std": std if np.isfinite(std) and std != 0.0 else 1.0,
        }
    path = Path(path)
    path.write_text(yaml.safe_dump(norm, sort_keys=False))
    return path


def write_sourced_fragment(raw: dict, out: Path, files: dict[str, Path]) -> Path:
    """A reader fragment with each sample's per-stage ``sources:`` filled in.

    Derived from the shipped fragment rather than hand-written, so the groups
    under test are the SHIPPED groups and only the file list is supplied. Needed
    because ``samples`` is a jsonargparse list, which the CLI cannot address
    element-wise. ``files`` maps sample name -> file.
    """
    import yaml

    samples = raw["data"]["modules"]["reader"]["init_args"]["samples"]
    for sample in samples:
        path = files[sample["name"]]
        sample["sources"] = {stage: str(path) for stage in ("train", "val", "test")}
        # `sources` binds only at stage-clone time (MultiSampleReader.with_source),
        # so a run-free path — `salt graph validate`, which calls prepare() on the
        # PROTOTYPE — still sees a sub-reader with no file. Set filename too.
        sample["reader"].setdefault("init_args", {})["filename"] = str(path)
    out = Path(out)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out
