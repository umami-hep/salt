"""The v2 cut vocabulary: one `Cut` predicate (simple comparison or whitelisted
expression) shared by `GlobalObjectCuts` (sample-axis rows, index-build) and
`ConstituentCuts` (per-stream constituents, ``on_fail: mask | drop``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from salt.data.processors.multi_target import _OPERATORS
from salt.data.readers.expressions import parse_expression
from salt.graph.errors import ConfigError

__all__ = ["ConstituentCuts", "Cut", "CutSpec", "GlobalObjectCuts"]

_ON_FAIL = ("mask", "drop")
VALID_FIELD = "valid"
"""The per-constituent validity field every jagged stream carries."""


@dataclass(frozen=True)
class Cut:
    """One keep predicate — a scalar comparison or a whitelisted expression.

    Two equivalent forms:

    - simple: ``field`` / ``op`` / ``value`` — ``Cut("pt", ">=", 20000)``;
    - expression: ``expr`` — ``Cut(expr="(numberOfPixelSharedHits +
      numberOfSCTSharedHits / 2) < 1.1")``, parsed by an ``ast``-whitelist
      evaluator (`salt.data.readers.expressions`) and evaluated vectorised.

    An expression that is a bare ``<field> <op> <number>`` normalises to the
    simple form (``field``/``op``/``value`` are filled in, ``expr`` is kept as
    provenance), so ``"d0 < 3.5"`` and ``Cut("d0", "<", 3.5)`` are the same cut.
    `Cut.parse` accepts a string, a mapping, or a `Cut` — the surface every cut
    container normalises through, so YAML may write plain strings.

    Parameters
    ----------
    field : str, optional
        The field to cut on. A bare name (``"pt"``) addresses the stream's
        structured field; a dotted name (``"jets.pt"``) is accepted and the
        leading ``<stream>.`` prefix is stripped.
    op : str, optional
        One of ``==  !=  >=  <=  >  <``.
    value : float | int | None, optional
        The right-hand side of the comparison.
    expr : str, optional
        A cut expression, mutually exclusive with ``field``/``op``/``value``.
    comment : str, optional
        Free-text provenance note (ignored by the evaluator).

    Raises
    ------
    ConfigError
        When both forms (or neither) are given, on an empty ``field``, an ``op``
        outside the supported set, a missing ``value``, or an expression the
        whitelist rejects.
    """

    field: str = ""
    op: str = ""
    value: float | int | None = None
    expr: str = ""
    comment: str = ""

    def __post_init__(self) -> None:
        simple_given = bool(self.field) or bool(self.op) or self.value is not None
        if self.expr and simple_given:
            raise ConfigError(f"Cut: give either expr={self.expr!r} or field/op/value, not both")
        if not self.expr:
            if not simple_given:
                raise ConfigError("Cut: give either a field/op/value triple or an expr")
            if not self.field:
                raise ConfigError("Cut: 'field' must be a non-empty string")
            if self.op not in _OPERATORS:
                raise ConfigError(
                    f"Cut: unknown op {self.op!r} — expected one of {sorted(_OPERATORS)}"
                )
            if self.value is None:
                raise ConfigError(f"Cut: 'value' must be set for field {self.field!r}")
            return
        simple = parse_expression(self.expr).simple
        if simple is not None:  # normalise a bare comparison onto the simple form
            fld, op, value = simple
            object.__setattr__(self, "field", fld)
            object.__setattr__(self, "op", op)
            object.__setattr__(self, "value", value)

    @classmethod
    def parse(cls, spec: Cut | str | Mapping[str, Any]) -> Cut:
        """Normalise a config entry (a `Cut`, an expression string, or a mapping) to a `Cut`.

        Returns
        -------
        Cut
            The normalised cut.

        Raises
        ------
        ConfigError
            When `spec` is none of those three forms, or is an invalid cut.
        """
        if isinstance(spec, Cut):
            return spec
        if isinstance(spec, str):
            return cls(expr=spec)
        if isinstance(spec, Mapping):
            return cls(**dict(spec))
        raise ConfigError(
            f"cut entry {spec!r} is not a Cut, an expression string, or a mapping of field/op/value"
        )

    @property
    def bare_field(self) -> str:
        """The simple form's field name with any leading ``<stream>.`` prefix stripped."""
        return self.field.rsplit(".", 1)[-1]

    @property
    def fields(self) -> tuple[str, ...]:
        """The bare field names this cut reads — the read-planner contract."""
        if self.field:
            return (self.bare_field,)
        return parse_expression(self.expr).fields

    def mask(self, source: Mapping[str, Any] | np.ndarray) -> Any:
        """Evaluate this cut to a keep mask (True where the entry PASSES).

        Parameters
        ----------
        source : Mapping[str, Any] | np.ndarray
            A structured array (any shape) or a mapping of column arrays carrying
            every field this cut references.

        Returns
        -------
        Any
            A bool mask shaped like the source columns. A NaN value FAILS every
            ordering / equality cut (numpy semantics; only ``!=`` is True vs NaN).

        Raises
        ------
        KeyError
            When a referenced field is absent from `source`.
        """
        if not self.field:
            return parse_expression(self.expr).evaluate(source)
        names = source.dtype.names or () if isinstance(source, np.ndarray) else tuple(source)
        fname = self.bare_field
        if fname not in names:
            raise KeyError(
                f"Cut field {self.field!r} (-> {fname!r}) is not available; "
                f"present: {sorted(names)}. Add it to the stream so it is read."
            )
        return _OPERATORS[self.op](source[fname], self.value)


def _parse_cuts(specs: Sequence[Any], where: str) -> tuple[Cut, ...]:
    """Normalise a config cut sequence to `Cut` instances, naming `where` on failure.

    Returns
    -------
    tuple[Cut, ...]
        The normalised cuts.

    Raises
    ------
    ConfigError
        On an entry that is not a valid cut.
    """
    out = []
    for spec in specs:
        try:
            out.append(Cut.parse(spec))
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return tuple(out)


@dataclass(frozen=True)
class GlobalObjectCuts:
    """Global + per-split row eligibility on the sample axis, applied at index-build.

    The reader-uniform ``cuts:`` surface: cuts here drop whole ROWS (jets for a
    jet reader, events for an event reader) and so change ``__len__``. Per-
    constituent filtering is `ConstituentCuts`, never this.

    Parameters
    ----------
    global_cuts : Sequence[Cut | str | Mapping], optional
        Cuts applied to EVERY split (AND-combined). Entries normalise through
        `Cut.parse`.
    per_split : Mapping[str, Sequence[Cut | str | Mapping]], optional
        Per-split extra cuts, keyed by stage (``"train"``/``"val"``/``"test"``).
        ``for_split(stage)`` returns ``global_cuts + per_split[stage]``.

    Raises
    ------
    ConfigError
        On an unknown stage key in ``per_split`` or an entry that is not a cut.
    """

    global_cuts: tuple[Cut, ...] = ()
    per_split: Mapping[str, tuple[Cut, ...]] = field(default_factory=dict)

    _STAGE_KEYS = ("train", "val", "test")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_cuts",
            _parse_cuts(tuple(self.global_cuts), "GlobalObjectCuts.global_cuts"),
        )
        unknown = set(self.per_split) - set(self._STAGE_KEYS)
        if unknown:
            raise ConfigError(
                f"GlobalObjectCuts.per_split: unknown stage keys {sorted(unknown)} — expected "
                f"a subset of {list(self._STAGE_KEYS)}"
            )
        object.__setattr__(
            self,
            "per_split",
            {
                stage: _parse_cuts(tuple(cuts), f"GlobalObjectCuts.per_split[{stage!r}]")
                for stage, cuts in self.per_split.items()
            },
        )

    def for_split(self, split: str | None) -> tuple[Cut, ...]:
        """The effective cuts for a stage: global + that split's extras.

        Parameters
        ----------
        split : str | None
            The stage (``"train"``/``"val"``/``"test"``); ``None`` applies only the
            global cuts.

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

        A `KeyError` propagates from `Cut.mask` for a cut whose field is absent
        from ``rows``.
        """
        keep = np.ones(len(rows), dtype=bool)
        for c in self.for_split(split):
            keep &= c.mask(rows)
        return keep

    def fields(self, split: str | None = None) -> tuple[str, ...]:
        """The bare field names referenced by the effective cuts.

        Used by the reader to guarantee the cut variables are read at index-build
        even when not otherwise demanded.

        Parameters
        ----------
        split : str | None, optional
            Restrict to one split's effective cuts; ``None`` (default) returns the
            union across global + every per-split.

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
            for f in c.fields:
                seen.setdefault(f, None)
        return tuple(seen)


CutSpec = GlobalObjectCuts
"""Deprecated alias kept so pre-rename ``cuts:`` config/class paths keep resolving."""


@dataclass(frozen=True)
class ConstituentCuts:
    """Per-stream constituent cuts with explicit failure semantics.

    Constituent cuts never change the reader's length: they act WITHIN a row.

    - ``on_fail: mask`` — a failing constituent keeps its slot and is blanked in
      place: float fields become NaN, signed-int ``-1``, unsigned-int ``0``, bool
      ``False`` (so ``valid`` becomes False). Positions and multiplicity are
      preserved. Constituents already invalid (padding) are left untouched.
    - ``on_fail: drop`` — a failing constituent is REMOVED and the row re-padded,
      so a failing constituent never wastes a served slot.

    Parameters
    ----------
    cuts : Sequence[Cut | str | Mapping], optional
        The AND-combined keep predicates; entries normalise through `Cut.parse`,
        so YAML may write expression strings (``["d0 < 3.5"]``).
    on_fail : str
        ``"mask"`` or ``"drop"`` — required, no default: the two modes hand the
        model different inputs.

    Raises
    ------
    ConfigError
        On an unset/unknown ``on_fail`` or an entry that is not a cut.
    """

    cuts: tuple[Cut, ...] = ()
    on_fail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cuts", _parse_cuts(tuple(self.cuts), "ConstituentCuts.cuts"))
        if self.on_fail not in _ON_FAIL:
            raise ConfigError(
                f"ConstituentCuts: on_fail must be set explicitly to one of {list(_ON_FAIL)}, "
                f"got {self.on_fail!r} — 'mask' blanks a failing constituent in place, 'drop' "
                "removes it and re-pads"
            )

    @property
    def fields(self) -> tuple[str, ...]:
        """The bare field names referenced, deduplicated in first-seen order."""
        seen: dict[str, None] = {}
        for c in self.cuts:
            for f in c.fields:
                seen.setdefault(f, None)
        return tuple(seen)

    def keep(self, source: Mapping[str, Any] | np.ndarray) -> Any:
        """The AND of every cut's keep mask over a constituent-shaped source.

        Parameters
        ----------
        source : Mapping[str, Any] | np.ndarray
            A structured constituent array or a mapping of column arrays.

        Returns
        -------
        Any
            The combined keep mask (True where the constituent PASSES), or None
            when no cut is configured.
        """
        keep = None
        for c in self.cuts:
            this = c.mask(source)
            keep = this if keep is None else (keep & this)
        return keep

    def apply(self, batch: np.ndarray) -> np.ndarray:
        """Apply these cuts to a structured ``(B, T)`` constituent batch.

        Parameters
        ----------
        batch : np.ndarray
            The structured ``(B, T)`` constituent batch, carrying ``valid``.

        Returns
        -------
        np.ndarray
            ``mask`` blanks failing constituents IN PLACE and returns `batch`;
            ``drop`` returns a compacted, re-padded copy.

        Raises
        ------
        KeyError
            When the stream carries no ``valid`` field.
        """
        if not self.cuts:
            return batch
        if VALID_FIELD not in (batch.dtype.names or ()):
            raise KeyError(
                f"constituent cuts need the {VALID_FIELD!r} field on the stream; "
                f"present: {sorted(batch.dtype.names or ())}"
            )
        return self._mask(batch) if self.on_fail == "mask" else self._drop(batch)

    def _mask(self, batch: np.ndarray) -> np.ndarray:
        """Blank failing (previously valid) constituents in place, per dtype kind.

        Returns
        -------
        np.ndarray
            The same (mutated) `batch`.

        Raises
        ------
        TypeError
            On a field dtype outside float / signed int / unsigned int / bool.
        """
        valid = batch[VALID_FIELD]
        removed = np.zeros_like(valid, dtype=bool)
        for c in self.cuts:
            removed[valid & ~c.mask(batch)] = True
        for name in batch.dtype.names or ():
            col = batch[name]
            kind = col.dtype.kind
            if kind == "f":
                col[removed] = np.nan
            elif kind == "i":
                col[removed] = -1
            elif kind == "u":
                col[removed] = 0
            elif kind == "b":
                col[removed] = False
            else:
                raise TypeError(
                    f"constituent cut masking: unsupported dtype {col.dtype} for field {name!r}"
                )
        return batch

    def _drop(self, batch: np.ndarray) -> np.ndarray:
        """Remove failing constituents, compact each row (order preserved) and re-pad.

        Returns
        -------
        np.ndarray
            A compacted, re-padded copy of `batch`.
        """
        from salt.data.readers.stream import pad_fill

        valid = batch[VALID_FIELD]
        keep = self.keep(batch) & valid
        t_dim = batch.shape[1]
        order = np.argsort(~keep, axis=1, kind="stable")  # kept first, order preserved
        out = np.take_along_axis(batch, order, axis=1)
        new_valid = np.arange(t_dim)[None, :] < keep.sum(axis=1)[:, None]
        pad = ~new_valid
        for name in out.dtype.names or ():
            if name == VALID_FIELD:
                continue
            col = out[name]
            col[pad] = pad_fill(col.dtype)
        out[VALID_FIELD] = new_valid
        return out
