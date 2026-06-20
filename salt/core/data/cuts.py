"""Reusable index-build-time selections for v2 `Reader`s (plan 19, Track C).

A `CutSpec` is a small, frozen, reader-AGNOSTIC description of which jet-level
(global-object) rows are *eligible* to be served. It is evaluated ONCE at the
reader's index-build stage (`prepare`) — NOT as a per-batch processor and NOT as
a `Dataset` wrapper.

Why index-build, not a post-read wrapper
----------------------------------------
Row-dropping cuts in a per-batch wrapper break the ``(B, …)`` batch contract
(every reader read is a CONTIGUOUS slice, ``dataset.py:353`` rejects fancy
indices) and downstream length-coherence. Event/jet selections decide *which rows
are eligible*, so they belong at the reader's index-build stage: each batch then
stays a contiguous slice of the *filtered* row index, ``__len__`` is the filtered
count, and `Features` / `Labels` / the sampler all see a clean dense stream. This
mirrors v1 semantics, where selections run inside the read on the structured array
BEFORE any bundle key exists (``base.py`` Reader docstring).

Per-split selection
-------------------
``CutSpec`` carries a ``global_cuts`` tuple applied to every split, plus an
optional ``per_split`` map (``{"train": (...), "val": (...), "test": (...)}``).
The reader's ``with_source(stage=...)`` clone passes the bound stage through, and
``for_split(stage)`` returns ``global + this split's`` cuts. A train/val split by
``eventNumber`` parity, or a per-split ``pt`` floor, is therefore a config change
with no reader-code change.

The comparison operators reuse ``processors._OPERATORS`` (the v1
``datasets.py:29-36`` comparator table) so a cut's ``op`` is exactly the v1
conditional-replacement operator set: ``==  !=  >=  <=  >  <``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from salt.core.data.processors import _OPERATORS
from salt.core.graph.errors import ConfigError

__all__ = ["Cut", "CutSpec"]


@dataclass(frozen=True)
class Cut:
    """One scalar comparison on a jet-level (global-object) field (plan 19).

    Parameters
    ----------
    field : str
        The jet-level field name to cut on. A *bare* field name (``"pt"``,
        ``"flavour_label"``) addresses the jets stream's structured field; a
        DOTTED name (``"jets.pt"``) is accepted too and the leading ``jets.``
        (or any ``<stream>.``) prefix is stripped — the cut is evaluated against
        the jet-level scalar record the reader builds in `prepare`.
    op : str
        One of ``==  !=  >=  <=  >  <`` (reuses ``processors._OPERATORS``).
    value : float | int
        The right-hand side of the comparison.
    comment : str, optional
        Free-text provenance note (ignored by the evaluator).

    Raises
    ------
    ConfigError
        On an empty ``field`` or an ``op`` outside the supported operator set.
    """

    field: str
    op: str
    value: float | int
    comment: str = ""

    def __post_init__(self) -> None:
        if not self.field:
            raise ConfigError("Cut: 'field' must be a non-empty string (plan 19)")
        if self.op not in _OPERATORS:
            raise ConfigError(
                f"Cut: unknown op {self.op!r} — expected one of {sorted(_OPERATORS)} "
                "(reuses v1 datasets.py:29-36 operators) (plan 19)"
            )

    @property
    def bare_field(self) -> str:
        """The field name with any leading ``<stream>.`` prefix stripped.

        Returns
        -------
        str
            The bare structured-field name to look up on the jet record.
        """
        return self.field.rsplit(".", 1)[-1]

    def mask(self, jet_scalars: np.ndarray) -> np.ndarray:
        """Evaluate this cut over a jet-level structured array.

        Parameters
        ----------
        jet_scalars : np.ndarray
            A structured ``(N_jets,)`` array carrying the cut field.

        Returns
        -------
        np.ndarray
            A ``(N_jets,)`` bool mask: True where the jet PASSES the cut.

        Raises
        ------
        KeyError
            If `bare_field` is not a field of ``jet_scalars``.
        """
        names = jet_scalars.dtype.names or ()
        fname = self.bare_field
        if fname not in names:
            raise KeyError(
                f"Cut field {self.field!r} (-> {fname!r}) is not a jet-level scalar field; "
                f"available: {sorted(names)} (plan 19). Add it to the jets-stream branches so "
                "it is read at index-build."
            )
        return _OPERATORS[self.op](jet_scalars[fname], self.value)


@dataclass(frozen=True)
class CutSpec:
    """Global + per-split jet eligibility, applied at index-build (plan 19).

    Parameters
    ----------
    global_cuts : tuple[Cut, ...], optional
        Cuts applied to EVERY split (AND-combined).
    per_split : Mapping[str, tuple[Cut, ...]], optional
        Per-split extra cuts, keyed by stage (``"train"``/``"val"``/``"test"``).
        ``for_split(stage)`` returns ``global_cuts + per_split[stage]``.

    Raises
    ------
    ConfigError
        On an unknown stage key in ``per_split`` or a non-`Cut` entry.
    """

    global_cuts: tuple[Cut, ...] = ()
    per_split: Mapping[str, tuple[Cut, ...]] = field(default_factory=dict)

    _STAGE_KEYS = ("train", "val", "test")

    def __post_init__(self) -> None:
        for c in self.global_cuts:
            if not isinstance(c, Cut):
                raise ConfigError(f"CutSpec.global_cuts must contain Cut instances, got {c!r}")
        unknown = set(self.per_split) - set(self._STAGE_KEYS)
        if unknown:
            raise ConfigError(
                f"CutSpec.per_split: unknown stage keys {sorted(unknown)} — expected a subset "
                f"of {list(self._STAGE_KEYS)} (plan 19)"
            )
        for stage, cuts in self.per_split.items():
            for c in cuts:
                if not isinstance(c, Cut):
                    raise ConfigError(
                        f"CutSpec.per_split[{stage!r}] must contain Cut instances, got {c!r}"
                    )

    def for_split(self, split: str | None) -> tuple[Cut, ...]:
        """The effective cuts for a stage: global + that split's extras.

        Parameters
        ----------
        split : str | None
            The stage (``"train"``/``"val"``/``"test"``); ``None`` applies only
            the global cuts.

        Returns
        -------
        tuple[Cut, ...]
            The AND-combined cut tuple for this split.
        """
        extra = self.per_split.get(split, ()) if split is not None else ()
        return (*self.global_cuts, *extra)

    def eligible(self, jet_scalars: np.ndarray, split: str | None) -> np.ndarray:
        """Bool mask over jets: True where ALL effective cuts pass (plan 19).

        Parameters
        ----------
        jet_scalars : np.ndarray
            A structured ``(N_jets,)`` array carrying every cut field.
        split : str | None
            The stage selecting per-split cuts.

        Returns
        -------
        np.ndarray
            A ``(N_jets,)`` bool mask of eligible (passing) jets. With no cuts,
            every jet is eligible.

        Raises
        ------
        KeyError
            On a cut whose field is absent from ``jet_scalars`` (propagated from
            `Cut.mask`).
        """
        n = len(jet_scalars)
        keep = np.ones(n, dtype=bool)
        for c in self.for_split(split):
            keep &= c.mask(jet_scalars)
        return keep

    def fields(self, split: str | None = None) -> tuple[str, ...]:
        """The bare jet-field names referenced by the effective cuts (plan 19).

        Used by the reader to guarantee the cut variables are read at
        index-build even when not otherwise demanded.

        Parameters
        ----------
        split : str | None, optional
            Restrict to one split's effective cuts; ``None`` (default) returns
            the union across global + every per-split.

        Returns
        -------
        tuple[str, ...]
            The deduplicated bare field names, in first-seen order.
        """
        if split is not None:
            cuts = self.for_split(split)
        else:
            cuts = (*self.global_cuts, *(c for cs in self.per_split.values() for c in cs))
        seen: dict[str, None] = {}
        for c in cuts:
            seen.setdefault(c.bare_field, None)
        return tuple(seen)
