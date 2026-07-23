"""`CutSpec` — reusable index-build-time row selections for v2 `Reader`s,
evaluated once in `prepare` (global cuts plus optional per-split cuts).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from salt.data.processors.multi_target import _OPERATORS
from salt.graph.errors import ConfigError

__all__ = ["Cut", "CutSpec"]


@dataclass(frozen=True)
class Cut:
    """One scalar comparison on a row-level field.

    Parameters
    ----------
    field : str
        The row-level field name to cut on — a field on the reader's sample
        axis (jets for jet readers, events for event readers). A bare field
        name (``"pt"``, ``"flavour_label"``) addresses the sample-axis stream's
        structured field; a dotted name (``"jets.pt"``) is accepted too and the
        leading ``<stream>.`` prefix is stripped — the cut is evaluated against
        the row-level scalar record the reader builds in `prepare`.
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
            raise ConfigError("Cut: 'field' must be a non-empty string")
        if self.op not in _OPERATORS:
            raise ConfigError(
                f"Cut: unknown op {self.op!r} — expected one of {sorted(_OPERATORS)}"
            )

    @property
    def bare_field(self) -> str:
        """The field name with any leading ``<stream>.`` prefix stripped."""
        return self.field.rsplit(".", 1)[-1]

    def mask(self, rows: np.ndarray) -> np.ndarray:
        """Evaluate this cut over a row-level structured array.

        Parameters
        ----------
        rows : np.ndarray
            A structured ``(N_rows,)`` array carrying the cut field.

        Returns
        -------
        np.ndarray
            A ``(N_rows,)`` bool mask: True where the row PASSES the cut.

        Raises
        ------
        KeyError
            If `bare_field` is not a field of ``rows``.
        """
        names = rows.dtype.names or ()
        fname = self.bare_field
        if fname not in names:
            raise KeyError(
                f"Cut field {self.field!r} (-> {fname!r}) is not a row-level scalar field; "
                f"available: {sorted(names)}. Add it to the reader's sample-axis stream so "
                "it is read at index-build."
            )
        return _OPERATORS[self.op](rows[fname], self.value)


@dataclass(frozen=True)
class CutSpec:
    """Global + per-split row eligibility, applied at index-build.

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
                f"of {list(self._STAGE_KEYS)}"
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

    def eligible(self, rows: np.ndarray, split: str | None) -> np.ndarray:
        """Bool mask over rows: True where all effective cuts pass.

        Parameters
        ----------
        rows : np.ndarray
            A structured ``(N_rows,)`` array carrying every cut field.
        split : str | None
            The stage selecting per-split cuts.

        Returns
        -------
        np.ndarray
            A ``(N_rows,)`` bool mask of eligible (passing) rows. With no cuts,
            every row is eligible.

        Raises
        ------
        KeyError
            On a cut whose field is absent from ``rows`` (propagated from
            `Cut.mask`).
        """
        n = len(rows)
        keep = np.ones(n, dtype=bool)
        for c in self.for_split(split):
            keep &= c.mask(rows)
        return keep

    def fields(self, split: str | None = None) -> tuple[str, ...]:
        """The bare field names referenced by the effective cuts.

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
