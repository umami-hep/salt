"""Strict FTAG1LITE container-name categories: MC subcampaign (from the reco r-tag), parton
shower (from the physics short) and simulation type (from the first a/s tag), plus the
`ContainerOneHot` processor over the reader's per-row category code columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from salt.data.base import Processor
from salt.graph.errors import ConfigError
from salt.graph.planner import PlanStep
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec

SUBCAMPAIGNS: tuple[str, ...] = ("mc20a", "mc20d", "mc20e", "mc23a", "mc23c", "mc23d", "mc23e")
"""MC subcampaign vocabulary, in one-hot column order."""

SHOWERS: tuple[str, ...] = ("Pythia8", "Herwig7", "Sherpa", "other")
"""Parton-shower vocabulary, in one-hot column order; ``other`` is the non-fatal fallback."""

SIM_TYPES: tuple[str, ...] = ("FullSim", "AF3")
"""Simulation-type vocabulary, in one-hot column order."""

CATEGORY_FIELDS: tuple[str, ...] = ("subcampaign", "shower", "sim_type")
"""The int64 code columns `FTAG1LITEReader` injects on its ``unroll`` stream."""

RTAG_SUBCAMPAIGN: dict[str, str] = {
    "r13167": "mc20a",
    "r13144": "mc20d",
    "r13145": "mc20e",
    "r14622": "mc23a",
    "r15540": "mc23a",
    "r15224": "mc23d",
    "r15530": "mc23d",
    "r16083": "mc23e",
}
"""Reco r-tag -> subcampaign (AMI preInclude Campaigns MC20a/d/e and MC23a/d/e).
An r-tag missing here is a `ConfigError`, never a fallback bucket."""

MERGE_RTAGS: frozenset[str] = frozenset({"r13146", "r15514", "r15516", "r15970"})
"""AMI merge-step r-tags: they follow the reco tag and never decide the subcampaign."""

SHOWER_TOKENS: dict[str, str] = {
    "PhPy8EG": "Pythia8",
    "Py8EG": "Pythia8",
    "Py8": "Pythia8",
    "Pythia8EvtGen": "Pythia8",
    "PowhegPythia8EvtGen": "Pythia8",
    "H7EG": "Herwig7",
    "PhH7EG": "Herwig7",
    "Sh": "Sherpa",
    "Sherpa": "Sherpa",
}
"""First ``_``-token of the physics short -> shower (exact match; anything else is other)."""

_DSID = re.compile(r"\.(\d{6})\.")
_RTAG = re.compile(r"r\d+")
_SIMTAG = re.compile(r"[as]\d+")
_VOCABS = dict(zip(CATEGORY_FIELDS, (SUBCAMPAIGNS, SHOWERS, SIM_TYPES), strict=True))


@dataclass(frozen=True)
class ContainerCategories:
    """One container's parsed identity and categories."""

    name: str
    dsid: str
    physics_short: str
    rtag: str
    subcampaign: str
    shower: str
    sim_type: str | None


def parse_container_categories(name: str, *, require_sim_type: bool = False) -> ContainerCategories:
    """Categorise one container from its directory name (``<scope>__<did>``,
    ``<scope>_<did>`` or a bare DID; only the basename is read).

    The subcampaign comes from the FIRST ``r<N>`` token of the final dot-segment (the AMI
    tag string): two-r-tag DIDs carry the reco tag first and the merge tag second. The
    simulation type comes from the FIRST ``a<N>`` (AF3) or ``s<N>`` (FullSim) token, since
    an ``s`` tag after an ``a`` tag is a HITS merge; without either it is None, or a
    `ConfigError` when `require_sim_type` is set.

    Raises
    ------
    ConfigError
        Naming the container and the offending token, when the name has no DSID, physics
        short or AMI tag segment; when the AMI tags carry no r-tag; when the first r-tag is
        a merge tag or not in `RTAG_SUBCAMPAIGN`; or (with `require_sim_type`) when no
        simulation tag is present.
    """
    base = Path(name).name
    m = _DSID.search(base)
    if m is None:
        raise ConfigError(f"FTAG1LITE container {base!r}: no '.<6-digit DSID>.' dot-segment")
    segments = base[m.end() :].split(".")
    physics_short = segments[0]
    if not physics_short:
        raise ConfigError(
            f"FTAG1LITE container {base!r}: no physics-short segment after DSID {m.group(1)!r}"
        )
    if len(segments) < 2 or not segments[-1]:
        raise ConfigError(
            f"FTAG1LITE container {base!r}: no AMI tag segment after physics short "
            f"{physics_short!r}"
        )
    tags = segments[-1]
    tokens = tags.split("_")
    rtag = next((t for t in tokens if _RTAG.fullmatch(t)), None)
    if rtag is None:
        raise ConfigError(f"FTAG1LITE container {base!r}: AMI tag string {tags!r} has no r-tag")
    if rtag in MERGE_RTAGS:
        raise ConfigError(
            f"FTAG1LITE container {base!r}: first r-tag {rtag!r} is a merge tag — the reco "
            "r-tag must come first in the AMI tag string"
        )
    subcampaign = RTAG_SUBCAMPAIGN.get(rtag)
    if subcampaign is None:
        raise ConfigError(
            f"FTAG1LITE container {base!r}: r-tag {rtag!r} has no known subcampaign "
            f"(known: {sorted(RTAG_SUBCAMPAIGN)})"
        )
    sim_tag = next((t for t in tokens if _SIMTAG.fullmatch(t)), None)
    sim_type = None if sim_tag is None else ("AF3" if sim_tag[0] == "a" else "FullSim")
    if sim_type is None and require_sim_type:
        raise ConfigError(
            f"FTAG1LITE container {base!r}: AMI tag string {tags!r} has no a<N>/s<N> simulation tag"
        )
    shower = SHOWER_TOKENS.get(physics_short.split("_")[0], "other")
    return ContainerCategories(
        name=base,
        dsid=m.group(1),
        physics_short=physics_short,
        rtag=rtag,
        subcampaign=subcampaign,
        shower=shower,
        sim_type=sim_type,
    )


def category_codes(c: ContainerCategories) -> dict[str, int]:
    """Vocabulary indices per `CATEGORY_FIELDS` column; ``sim_type`` is -1 when None."""
    sim = -1 if c.sim_type is None else SIM_TYPES.index(c.sim_type)
    codes = (SUBCAMPAIGNS.index(c.subcampaign), SHOWERS.index(c.shower), sim)
    return dict(zip(CATEGORY_FIELDS, codes, strict=True))


def onehot_fields(include_sim_type: bool) -> tuple[str, ...]:
    """One-hot column names: subcampaigns, showers, then (optionally) simulation types."""
    fields = [f"subcampaign_{s}" for s in SUBCAMPAIGNS] + [f"shower_{s}" for s in SHOWERS]
    if include_sim_type:
        fields += [f"sim_type_{s}" for s in SIM_TYPES]
    return tuple(fields)


class ContainerOneHot(Processor):
    """Category codes -> one-hot: ``raw.<stream>`` -> ``inputs.<out>`` float32 ``[B, F]``.

    Kinematic-prior conditioning only, never a tagger input; active in every mode. Each
    reader-injected int64 ``subcampaign`` / ``shower`` (/ ``sim_type``) code sets one 0/1
    column per block (not to be normalised). Columns (`fields`, carried on the produced spec):
    ``subcampaign_<x>``, ``shower_<x>``, then ``sim_type_<x>`` with `include_sim_type`.
    A code outside its vocabulary (``sim_type == -1`` included) is a `ValueError`.
    """

    def __init__(
        self, stream: str = "jets", out: str = "container_onehot", include_sim_type: bool = False
    ) -> None:
        super().__init__()
        self.stream = str(stream)
        self.out = str(out)
        self.include_sim_type = bool(include_sim_type)
        self._codes = CATEGORY_FIELDS if self.include_sim_type else CATEGORY_FIELDS[:2]

    @property
    def fields(self) -> tuple[str, ...]:
        """The one-hot column names, in column order."""
        return onehot_fields(self.include_sim_type)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>`` -> ``inputs.<out>`` (``[B, len(fields)]`` float32)."""
        del mode
        fields = self.fields
        requires = {f"raw.{self.stream}": TensorSpec(kind="data")}
        produces = {
            f"inputs.{self.out}": TensorSpec(
                shape=("B", len(fields)), dtype="float32", kind="data", fields=fields
            )
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand the category code columns from ``raw.<stream>``."""
        del step
        return {self.stream: dict.fromkeys(self._codes, self.name)}

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Build a fresh one-hot array; raises `ValueError` on an out-of-vocabulary code."""
        del rows, mode
        raw = batch.get(f"raw.{self.stream}")
        out = np.zeros((len(raw), len(self.fields)), dtype=np.float32)
        index, offset = np.arange(len(raw)), 0
        for col in self._codes:
            size = len(_VOCABS[col])
            codes = np.asarray(raw[col], dtype=np.int64)
            bad = (codes < 0) | (codes >= size)
            if bad.any():
                raise ValueError(
                    f"ContainerOneHot module {self.name!r}: {self.stream}.{col} code(s) "
                    f"{np.unique(codes[bad]).tolist()} outside [0, {size})"
                )
            out[index, offset + codes] = 1.0
            offset += size
        return {f"inputs.{self.out}": out}
