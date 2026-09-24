"""`ConstituentSelection` — the one constituent-cut engine for every reader.

A pure post-read processor: rewrites ``raw.<stream>``/``masks.<stream>`` via the
planner's rewrite contract instead of each reader fusing its own cut/sort/pad.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from salt.data.base import Processor
from salt.data.readers.cuts import VALID_FIELD, Cut, _parse_cuts
from salt.data.readers.stream import pad_fill
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec

_KNOWN_KEYS = frozenset({"cuts", "pad_max", "sort"})


@dataclass(frozen=True)
class _ConstituentCuts:
    """AND-combined per-constituent keep predicates; `apply` drops the failures and re-pads.

    Acts WITHIN a row (never changes the reader's length). A cut carrying a
    constituent-axis reduction (``sum``) is refused — that is a row cut.
    """

    cuts: tuple[Cut, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cuts", _parse_cuts(tuple(self.cuts), "ConstituentSelection.cuts"))
        reduced = [agg.source for c in self.cuts for agg in c.aggregations]
        if reduced:
            raise ConfigError(
                f"ConstituentSelection.cuts: {reduced} reduce the constituent axis, but a "
                "constituent cut is evaluated ON that axis — it decides one constituent at a "
                "time and cannot see the row. Put a reduction in the reader's row-level "
                "cuts: instead"
            )

    @property
    def fields(self) -> tuple[str, ...]:
        """Bare field names referenced, deduplicated in first-seen order."""
        seen: dict[str, None] = {}
        for c in self.cuts:
            for f in c.fields:
                seen.setdefault(f, None)
        return tuple(seen)

    def keep(self, source: Mapping[str, Any] | np.ndarray) -> Any:
        """AND of every cut's keep mask over a constituent-shaped source; None when no cut."""
        keep = None
        for c in self.cuts:
            this = c.mask(source)
            keep = this if keep is None else (keep & this)
        return keep

    def apply(self, batch: np.ndarray) -> np.ndarray:
        """Drop failing constituents, compact each row (order preserved), re-pad.

        No cuts -> `batch` unchanged.
        """
        if not self.cuts:
            return batch
        if VALID_FIELD not in (batch.dtype.names or ()):
            raise KeyError(
                f"constituent cuts need the {VALID_FIELD!r} field on the stream; "
                f"present: {sorted(batch.dtype.names or ())}"
            )
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


@dataclass(frozen=True)
class _StreamSelection:
    """One stream's resolved cut -> sort -> truncate configuration."""

    cuts: _ConstituentCuts
    pad_max: int | None
    sort: dict[str, str] | None


def _parse_sort(spec: Mapping[str, Any] | None, stream: str) -> dict[str, str] | None:
    """Normalise a ``{var, mode}`` sort spec; None passes through.

    Raises ConfigError on an empty/missing ``var`` or an unknown ``mode``.
    """
    if spec is None:
        return None
    var = spec.get("var")
    mode = spec.get("mode", "descending")
    if not var:
        raise ConfigError(
            f"ConstituentSelection[{stream!r}].sort: 'var' must be a non-empty field name"
        )
    if mode not in {"ascending", "descending"}:
        raise ConfigError(
            f"ConstituentSelection[{stream!r}].sort: unknown mode {mode!r} — expected one of "
            "('ascending', 'descending')"
        )
    return {"var": str(var), "mode": str(mode)}


def _sorted_by(arr: np.ndarray, sort: dict[str, str]) -> np.ndarray:
    """Per-row stable sort of `arr` on ``sort['var']``, invalid slots pushed last.

    Raises KeyError when ``sort['var']`` is absent from `arr`.
    """
    var = sort["var"]
    if var not in (arr.dtype.names or ()):
        raise KeyError(f"ConstituentSelection sort: field {var!r} not present in the stream")
    key = arr[var]
    ordering_key = key if sort["mode"] == "ascending" else -key.astype(np.float64)
    order_key = np.where(arr[VALID_FIELD], ordering_key, np.inf)
    order = np.argsort(order_key, axis=1, kind="stable")
    return np.take_along_axis(arr, order, axis=1)


class ConstituentSelection(Processor):
    """Per-stream constituent cut -> sort -> truncate, applied after the read.

    The single cut engine shared by every reader: rewrites
    ``raw.<stream>``/``masks.<stream>`` in place of each reader's own fused
    cut/sort/pad path. Row/event cuts (`GlobalObjectCuts`) are unaffected —
    they stay on the reader's ``cuts:`` and act once at index-build, never here.

    Order per stream is **cut -> sort -> truncate**: a failing constituent is
    dropped and the row compacted and re-padded, so a failing constituent never
    wastes a served slot; ``sort`` runs after the cut.

    Byte-parity with a reader's former fused path holds exactly when the
    reader's served width is >= every row's PRE-cut multiplicity (truncating
    before vs. after the cut agree only then).

    Parameters
    ----------
    streams : Mapping[str, Mapping[str, Any] | None]
        Per-stream selection, keyed by stream name. Each entry:

        - ``cuts`` (sequence of `Cut` | str | mapping, optional) — AND-combined
          keep predicates; default ``()``; a failing constituent is dropped and
          the row re-padded;
        - ``pad_max`` (int, optional) — truncate width after cut/sort; ``None``
          (default) keeps the reader's served width;
        - ``sort`` (mapping, optional) — ``{var, mode}``, ``mode``
          ``"ascending"``/``"descending"`` (default ``"descending"``).

        A ``None`` entry is skipped (a jsonargparse null deletes an inherited
        stream's selection). A stream with no ``cuts``/``sort``/``pad_max`` is a
        legal no-op.

    Raises
    ------
    ConfigError
        On empty ``streams``, an unknown key in a stream entry, an invalid
        ``sort``, or ``pad_max`` < 1.
    ShapeError
        At plan time when ``pad_max`` exceeds a reader group's concrete
        ``pad_max`` for the stream.
    """

    def __init__(self, streams: Mapping[str, Mapping[str, Any] | None]) -> None:
        super().__init__()
        if not streams:
            raise ConfigError("ConstituentSelection needs at least one entry in 'streams'")
        self.streams: dict[str, _StreamSelection] = {}
        for stream, cfg in streams.items():
            if cfg is None:
                continue
            self.streams[stream] = self._checked_stream(stream, dict(cfg))

    @staticmethod
    def _checked_stream(stream: str, cfg: dict[str, Any]) -> _StreamSelection:
        """Validate + normalise one stream's config entry to a `_StreamSelection`."""
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ConfigError(
                f"ConstituentSelection[{stream!r}]: unknown keys {sorted(unknown)} — expected "
                f"a subset of {sorted(_KNOWN_KEYS)}"
            )
        cc = _ConstituentCuts(cuts=tuple(cfg.get("cuts", ())))
        pad_max = cfg.get("pad_max")
        if pad_max is not None:
            if not isinstance(pad_max, int) or isinstance(pad_max, bool):
                raise ConfigError(
                    f"ConstituentSelection[{stream!r}].pad_max must be an int, got {pad_max!r}"
                )
            if pad_max < 1:
                raise ConfigError(
                    f"ConstituentSelection[{stream!r}].pad_max must be >= 1, got {pad_max}"
                )
        sort = _parse_sort(cfg.get("sort"), stream)
        return _StreamSelection(cuts=cc, pad_max=pad_max, sort=sort)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<s>``/``masks.<s>`` requires (narrowed to the cut/sort
        fields) and a same-shape rewrite of both, for every configured stream.
        """
        del mode  # cuts are data semantics in every mode
        requires: dict[str, TensorSpec] = {}
        rewrites: dict[str, TensorSpec] = {}
        for stream, sel in self.streams.items():
            needed: dict[str, None] = {}
            for f in sel.cuts.fields:
                needed.setdefault(f, None)
            if sel.sort is not None:
                needed.setdefault(sel.sort["var"], None)
            requires[f"raw.{stream}"] = TensorSpec(kind="data", fields=tuple(needed) or None)
            requires[f"masks.{stream}"] = TensorSpec(dtype="bool", kind="pad_mask")
            shape = ("B", sel.pad_max) if sel.pad_max is not None else None
            rewrites[f"raw.{stream}"] = TensorSpec(shape=shape, kind="data")
            rewrites[f"masks.{stream}"] = TensorSpec(shape=shape, dtype="bool", kind="pad_mask")
        return IO(requires=unflatten_spec(requires), rewrites=unflatten_spec(rewrites))

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Cut -> sort -> truncate each configured stream (a bare stream copies through)."""
        del rows, mode
        out: dict[str, np.ndarray] = {}
        for stream, sel in self.streams.items():
            raw = batch.get(f"raw.{stream}")
            if VALID_FIELD not in (raw.dtype.names or ()):
                raise KeyError(
                    f"ConstituentSelection[{stream!r}] needs the {VALID_FIELD!r} field on the "
                    f"stream; present: {sorted(raw.dtype.names or ())}"
                )
            arr = np.array(raw, copy=True)  # never alias the reader's reusable buffer
            arr = sel.cuts.apply(arr)
            if sel.sort is not None:
                arr = _sorted_by(arr, sel.sort)
            if sel.pad_max is not None:
                arr = np.ascontiguousarray(arr[:, : sel.pad_max])
            out[f"raw.{stream}"] = arr
            out[f"masks.{stream}"] = ~arr[VALID_FIELD]
        return out
