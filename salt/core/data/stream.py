"""Shared `Reader`-base stream-assembly helpers.

Folds the per-reader jagged pad / file-offset / dtype-sentinel code that the
easyjet, ftag1lite, and multisample readers each need into one shared surface:

- `StreamConfig` — a per-stream description of the cut -> sort -> truncate ->
  pad pipeline (``pad_max``, ``sort``, ``cuts``, ``jagged``). Lives on the
  `Reader` base so every reader speaks the same vocabulary.
- `_cut_sort_truncate_pad` — the shared assembly: cut (per-constituent
  keep-mask, drop-then-pad: a cut constituent is removed before padding,
  never wasting a ``pad_max`` slot) -> sort (argsort by ``sort.var`` + permute
  all fields) -> truncate to the leading ``pad_max`` -> pad + ``valid`` with
  dtype-aware sentinels (float ``0.0``, signed-int label ``-1``, unsigned
  ``0``, bool ``False``).
- `OffsetIndex` — cumulative row offsets across a deterministic file list plus
  the covering-range mapping a contiguous global slice needs.

**Hard invariant.** With no cuts and no sort configured (the default),
`_cut_sort_truncate_pad` produces a result that is byte-for-byte identical to
the readers' contiguous truncate+pad+valid path: ``valid`` is computed first
from the per-row counts, each field is truncated to the leading ``pad_max``,
padded per dtype, cast to the schema dtype, and assembled into a structured
``(B, T)`` array with a trailing ``valid`` bool field. The cut/sort machinery
is skipped entirely on the default path — it engages only when a
`StreamConfig` carries a non-empty ``cuts`` tuple or a ``sort`` spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from salt.core.graph.errors import ConfigError

if TYPE_CHECKING:
    from salt.core.data.cuts import Cut
    from salt.core.schema import GroupSchema

__all__ = ["INT_PAD_SENTINEL", "OffsetIndex", "StreamConfig", "pad_fill"]

INT_PAD_SENTINEL = -1
"""Pad fill for SIGNED-int (label) fields: never a real class, folded to
``ignore_index=-1`` downstream. Floats pad to 0.0; unsigned/counts pad to 0; bool
pads to False; the ``valid`` field is set explicitly, never via these fills."""

_SORT_MODES = ("ascending", "descending")


@dataclass(frozen=True)
class StreamConfig:
    """Per-stream cut -> sort -> truncate -> pad pipeline description.

    The shared vocabulary the `Reader` base uses to drive
    `_cut_sort_truncate_pad`. The per-reader group configs (`GroupConfig`,
    `EasyjetGroupConfig`, `FTAG1LiteGroupConfig`) map onto it.

    Parameters
    ----------
    pad_max : int
        The served constituent multiplicity ``T`` (the leading N kept after sort +
        truncate). Resolved by the reader at index-build (config ``truncate`` /
        ``pad_max`` or the file-wide max). Must be ``>= 1``.
    sort : Mapping[str, str] | None, optional
        Constituent sort spec ``{"var": <field>, "mode": "ascending"|"descending"}``.
        ``None`` (default) keeps the file order — the parity-preserving path. The
        permutation is applied to every field of the stream (and any aligned labels)
        so a sort never desynchronises features from labels.
    cuts : tuple[Cut, ...], optional
        Per-constituent keep cuts (drop-then-pad). ``()`` (default) keeps every
        constituent — the parity-preserving path. A constituent failing any cut is
        removed before padding (never wastes a ``pad_max`` slot). Reuses
        `salt.core.data.cuts.Cut`.
    jagged : bool, optional
        Whether this is a variable-length sequence stream (padded to ``pad_max``
        with a ``valid`` field + pad mask). ``False`` is a scalar / global-object
        stream — `_cut_sort_truncate_pad` is not used for those. Default ``True``.

    Raises
    ------
    ConfigError
        On ``pad_max < 1``, a malformed ``sort`` spec, a non-`Cut` entry in
        ``cuts``, or cuts/sort configured on a non-jagged stream.
    """

    pad_max: int
    sort: dict[str, str] | None = None
    cuts: tuple[Cut, ...] = ()
    jagged: bool = True

    def __post_init__(self) -> None:
        from salt.core.data.cuts import Cut  # noqa: PLC0415 - lazy: avoid base<-stream<-cuts cycle

        if self.pad_max < 1:
            raise ConfigError(f"StreamConfig: pad_max must be >= 1, got {self.pad_max}")
        if self.sort is not None:
            var = self.sort.get("var")
            mode = self.sort.get("mode", "descending")
            if not var:
                raise ConfigError("StreamConfig.sort: 'var' must be a non-empty field name")
            if mode not in _SORT_MODES:
                raise ConfigError(
                    f"StreamConfig.sort: unknown mode {mode!r} — expected one of {_SORT_MODES}"
                )
            # normalise (frozen dataclass — set via object.__setattr__)
            object.__setattr__(self, "sort", {"var": str(var), "mode": str(mode)})
        for c in self.cuts:
            if not isinstance(c, Cut):
                raise ConfigError(f"StreamConfig.cuts must contain Cut instances, got {c!r}")
        if not self.jagged and (self.cuts or self.sort is not None):
            raise ConfigError(
                "StreamConfig: cuts/sort are only valid for jagged (sequence) streams"
            )

    @property
    def engages_pipeline(self) -> bool:
        """Whether cut/sort are configured (the additive path engages).

        The parity-protecting predicate: ``False`` means `_cut_sort_truncate_pad`
        runs the byte-identical contiguous truncate+pad+valid path; ``True`` means
        the drop-then-pad / sort machinery engages.
        """
        return bool(self.cuts) or self.sort is not None


def pad_fill(dt: np.dtype | None, arr: Any = None) -> Any:
    """The pad fill value for a field, by dtype kind.

    float -> 0.0 (zeroed again after masking downstream); SIGNED int (labels) ->
    ``-1`` sentinel (never a real class; folded to ``ignore_index=-1``); unsigned
    int / counts -> 0; bool -> False.

    Parameters
    ----------
    dt : np.dtype | None
        The field dtype. When ``None`` the kind is taken from ``arr``'s innermost
        content dtype (the awkward fallback used before a schema dtype is known).
    arr : Any, optional
        An awkward array whose ``layout.content`` dtype kind is used when ``dt`` is
        ``None``.
    """
    kind = dt.kind if dt is not None else np.asarray(arr.layout.content).dtype.kind
    if kind == "f":
        return 0.0
    if kind == "i":
        return INT_PAD_SENTINEL
    if kind == "b":
        return False
    return 0  # unsigned ints / counts / other: 0


def _cut_sort_truncate_pad(
    cols: dict[str, Any],
    fields: list[str],
    stream_cfg: StreamConfig,
    b: int,
    gschema: GroupSchema | None = None,
    labels: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut -> sort -> truncate -> pad a jagged stream into a structured ``(B, T)`` array.

    The shared assembly factored out of the per-reader ``_assemble_jagged``
    methods. The pipeline, in order:

    1. **cut (drop-then-pad).** When ``stream_cfg.cuts`` is non-empty, a
       per-constituent keep mask is built (AND of every `Cut` evaluated against
       the jagged columns) and applied to all fields + labels — a failing
       constituent is removed (not masked in place), so it never wastes a
       ``pad_max`` slot. With no cuts this step is a no-op.
    2. **sort.** When ``stream_cfg.sort`` is set, an argsort of the sort field
       per row gives a permutation applied to every field + label (so features
       and labels stay aligned). With no sort the file order is kept.
    3. **truncate.** Keep the leading ``pad_max`` constituents (``arr[:, :T]``).
    4. **pad + valid.** ``valid`` is computed first from the post-cut/sort
       per-row counts (clipped to ``T``); each field is ``ak.pad_none`` +
       ``ak.fill_none`` per dtype (`pad_fill`), densified, cast to the schema
       dtype, and assembled into a structured ``(B, T)`` array with a trailing
       ``valid`` bool field.

    With ``stream_cfg.engages_pipeline`` False (no cuts/sort — the default)
    steps 1-2 are skipped and steps 3-4 reproduce the readers' previous
    contiguous path byte-for-byte.

    Parameters
    ----------
    cols : dict[str, Any]
        ``{field: jagged awkward array}`` of length ``b`` (depth-1 ``[row][const]``).
    fields : list[str]
        The served field names, in config order (the structured-array field order).
    stream_cfg : StreamConfig
        The cut/sort/pad spec for this stream (``pad_max`` resolved).
    b : int
        The number of rows (batch size) — the output's leading dim.
    gschema : GroupSchema | None, optional
        The stream's schema group (field -> dtype). When given each field is cast to
        its schema dtype (the readers' existing ``block.astype(dt)``); when ``None``
        the awkward-inferred dtype is kept.
    labels : dict[str, Any] | None, optional
        Additional jagged columns aligned to ``cols`` (e.g. constituent labels) that
        must be permuted/cut IN LOCKSTEP but are NOT emitted as structured fields.
        Primarily for future cut/sort callers that carry labels separately; the
        retrofitted readers pass label fields inside ``cols`` and leave this ``None``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(structured (B, T) array, valid (B, T) bool)``. Propagates `KeyError`
        from `_apply_cut_and_sort` when a configured cut / sort field is absent.
    """
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    t_dim = stream_cfg.pad_max
    work = dict(cols)
    aligned = dict(labels) if labels is not None else {}

    if stream_cfg.engages_pipeline:
        work, aligned = _apply_cut_and_sort(work, aligned, fields, stream_cfg)

    # valid length FIRST: post-cut/sort per-row count, clipped to T (parity path)
    first = work[fields[0]]
    counts = np.asarray(ak.num(first, axis=1)) if b > 0 else np.zeros(0, dtype=np.int64)
    valid = np.arange(t_dim)[None, :] < np.minimum(counts, t_dim)[:, None]  # (B, T) bool

    dtype_fields: list[tuple[str, np.dtype]] = []
    blocks: dict[str, np.ndarray] = {}
    for f in fields:
        arr = work[f][:, :t_dim]  # truncate to served multiplicity (leading)
        padded = ak.pad_none(arr, t_dim, axis=1, clip=True)
        dt = np.dtype(gschema.fields[f]) if gschema is not None else None
        fill = pad_fill(dt, arr)
        dense = ak.to_numpy(ak.fill_none(padded, fill, axis=1))
        block = np.asarray(dense)
        if dt is not None:
            block = block.astype(dt, copy=False)
        blocks[f] = block
        dtype_fields.append((f, block.dtype))
    dtype_fields.append(("valid", np.dtype("bool")))
    raw = np.empty((b, t_dim), dtype=np.dtype(dtype_fields))
    for f in fields:
        raw[f] = blocks[f]
    raw["valid"] = valid
    return raw, valid


def _apply_cut_and_sort(
    work: dict[str, Any],
    aligned: dict[str, Any],
    fields: list[str],
    stream_cfg: StreamConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply per-constituent drop-then-pad cuts then a per-row sort to jagged columns.

    A maximal-lockstep permutation: the keep mask (cuts) and the argsort (sort) are
    BOTH applied to every entry of ``work`` and ``aligned`` so features and labels
    never desynchronise.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The cut+sorted ``(work, aligned)`` column dicts.

    Raises
    ------
    KeyError
        If a cut field or the sort field is missing from the columns.
    """
    import awkward as ak  # noqa: PLC0415 - optional reader extra (lazy)

    all_cols = {**work, **aligned}

    # --- 1. cut: drop-then-pad (a failing constituent is REMOVED) ---
    if stream_cfg.cuts:
        from salt.core.data.processors import _OPERATORS  # noqa: PLC0415

        keep = None
        for c in stream_cfg.cuts:
            fname = c.bare_field
            if fname not in all_cols:
                raise KeyError(
                    f"StreamConfig cut field {c.field!r} (-> {fname!r}) is not a constituent "
                    f"field of this stream; available: {sorted(all_cols)}"
                )
            this = _OPERATORS[c.op](all_cols[fname], c.value)
            keep = this if keep is None else (keep & this)
        if keep is not None:
            work = {f: work[f][keep] for f in work}
            aligned = {f: aligned[f][keep] for f in aligned}

    # --- 2. sort: argsort by sort.var, permute ALL fields + labels in lockstep ---
    if stream_cfg.sort is not None:
        var = stream_cfg.sort["var"]
        if var not in {**work, **aligned}:
            raise KeyError(
                f"StreamConfig sort field {var!r} is not a constituent field of this stream; "
                f"available: {sorted({**work, **aligned})}"
            )
        key = work[var] if var in work else aligned[var]
        order = ak.argsort(key, axis=1, ascending=stream_cfg.sort["mode"] == "ascending")
        work = {f: work[f][order] for f in work}
        aligned = {f: aligned[f][order] for f in aligned}

    del fields  # field order is preserved by the caller's structured assembly
    return work, aligned


@dataclass
class OffsetIndex:
    """Cumulative row offsets across a file list + covering-range mapping.

    Factored from the readers' file-table offset bookkeeping: the easyjet/H5
    simple cumulative ``(start, n)`` table and the ftag1lite event->jet
    covering search live here as one helper.

    The simple flavour (``offsets`` only) maps a contiguous global row slice to
    the ``(file_index, local_start, local_stop)`` runs that cover it — the
    per-file ``entry_start``/``entry_stop`` reads the easyjet reader stitches.
    The covering flavour (a per-entry cumulative array) maps a contiguous
    slice over a filtered / derived index back to a covering range over an
    underlying coarser index (the ftag1lite kept-jet -> covering-event
    search).

    Parameters
    ----------
    counts : list[int]
        The per-file served row counts, in deterministic (sorted) file order.

    Attributes
    ----------
    offsets : list[int]
        Cumulative offsets ``[0, counts[0], counts[0]+counts[1], ...]`` (length
        ``len(counts) + 1``); ``offsets[i]`` is the global index of file ``i``'s
        first row, ``offsets[-1]`` is the total.
    """

    counts: list[int]
    offsets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        acc = [0]
        for n in self.counts:
            acc.append(acc[-1] + int(n))
        self.offsets = acc

    @property
    def total(self) -> int:
        """The total served row count = sum of per-file counts."""
        return self.offsets[-1]

    def file_starts(self) -> list[int]:
        """The global start offset of each file (``offsets[:-1]``)."""
        return self.offsets[:-1]

    def runs(self, rows: slice) -> list[tuple[int, int, int]]:
        """Decompose a global row slice into per-file ``(file_index, lo, hi)`` runs.

        Each run is the local ``[lo, hi)`` row range within file ``file_index``
        that the global slice ``[rows.start, rows.stop)`` covers — exactly the
        ``entry_start``/``entry_stop`` per-file reads a multi-file reader
        stitches. Files with no overlap are skipped; the runs are returned in
        file order.

        Returns
        -------
        list[tuple[int, int, int]]
            ``(file_index, local_lo, local_hi)`` for each overlapping file.
        """
        start, stop = rows.start, rows.stop
        out: list[tuple[int, int, int]] = []
        for i, n in enumerate(self.counts):
            f_start = self.offsets[i]
            lo = max(start, f_start)
            hi = min(stop, f_start + n)
            if lo >= hi:
                continue
            out.append((i, lo - f_start, hi - f_start))
        return out

    @staticmethod
    def covering_range(cum: np.ndarray, lo: int, hi: int) -> tuple[int, int, int]:
        """Map a derived-index range ``[lo, hi)`` to a covering coarse range.

        Given a cumulative-count array ``cum`` (length ``n_coarse + 1``, where
        ``cum[k]`` is the number of derived rows in the first ``k`` coarse units
        — e.g. the per-event cumulative kept-jet counts), return the smallest
        coarse range ``[c0, c1)`` whose derived rows include ``[lo, hi)``, plus
        the offset of ``lo`` within ``c0``'s first derived row. This is the
        ftag1lite kept-jet -> covering-event search, generalised.

        Parameters
        ----------
        cum : np.ndarray
            Cumulative derived-row counts, ``cum[0] == 0``, length ``n_coarse + 1``.
        lo, hi : int
            The derived-index range (e.g. local kept-jet ``[jlo, jhi)``).

        Returns
        -------
        tuple[int, int, int]
            ``(c0, c1, offset_in_block)``: coarse start/stop and the offset of ``lo``
            within the ``[c0, c1)`` derived block.
        """
        c0 = int(np.searchsorted(cum, lo, side="right") - 1)
        c1 = int(np.searchsorted(cum, hi, side="left"))
        offset_in_block = lo - int(cum[c0])
        return c0, c1, offset_in_block
