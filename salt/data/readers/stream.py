"""Shared `Reader`-base stream-assembly helpers: `StreamConfig`, `_truncate_pad`
(truncate -> pad + valid), `OffsetIndex` (cumulative file offsets for contiguous
global slices), and `stage_file` (the multi-process-safe file-staging primitive).
"""

from __future__ import annotations

import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from filelock import FileLock, Timeout

from salt.graph.errors import ConfigError
from salt.utils import file_utils as fu

if TYPE_CHECKING:
    from salt.schema import GroupSchema

__all__ = [
    "INT_PAD_SENTINEL",
    "LOCK_TIMEOUT_S",
    "OffsetIndex",
    "StreamConfig",
    "pad_fill",
    "stage_file",
]

INT_PAD_SENTINEL = -1
"""Pad fill for SIGNED-int (label) fields: never a real class, folded to
``ignore_index=-1`` downstream. Floats pad to 0.0; unsigned/counts pad to 0; bool
pads to False; the ``valid`` field is set explicitly, never via these fills."""

LOCK_TIMEOUT_S = 1800
"""Seconds a `FileLock` waits before giving up (virtual-dataset build, file staging)."""


def stage_file(src: Path, dst: Path) -> Path:
    """Copy `src` to `dst` once, multi-process-safe (FileLock + ``.done`` marker,
    so a DDP rank / dataloader-worker stampede copies the file exactly once).

    Parameters
    ----------
    src : Path
        Source file to stage.
    dst : Path
        Destination path (its parent is created if missing).

    Returns
    -------
    Path
        ``dst`` (the staged copy).

    Raises
    ------
    RuntimeError
        If acquiring the lock times out.
    """
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    done_path = dst.with_suffix(dst.suffix + ".done")
    # Fast path: already staged (marker present and the copy is in place).
    if done_path.exists() and dst.is_file():
        return dst

    lock_path = dst.with_suffix(dst.suffix + ".lock")
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=LOCK_TIMEOUT_S)
    except Timeout as exc:
        raise RuntimeError(f"Timeout waiting for staging lock: {lock_path}") from exc

    try:
        # Re-check under the lock — a contender may have just finished.
        if done_path.exists() and dst.is_file():
            return dst
        fu.copy_file(src, dst)  # no-op if dst already present (file_utils.copy_file)
        marker_tmp = done_path.with_name(done_path.name + f".tmp.{os.getpid()}")
        marker_tmp.write_text(f"ok pid={os.getpid()} time={time.time()}\n")
        marker_tmp.replace(done_path)
        return dst
    finally:
        with suppress(Exception):
            lock.release()


@dataclass(frozen=True)
class StreamConfig:
    """Per-stream truncate -> pad spec.

    The shared vocabulary the `Reader` base uses to drive `_truncate_pad`. The
    reader group configs (`GroupConfig`, `UprootGroupConfig`) map onto it.

    Parameters
    ----------
    pad_max : int
        The served constituent multiplicity ``T`` (the leading N kept after
        truncate). Resolved by the reader at index-build (config ``pad_max`` or
        the file-wide max). Must be ``>= 1``.
    jagged : bool, optional
        Whether this is a variable-length sequence stream (padded to ``pad_max``
        with a ``valid`` field + pad mask). ``False`` is a scalar / global-object
        stream — `_truncate_pad` is not used for those. Default ``True``.

    Raises
    ------
    ConfigError
        On ``pad_max < 1``.
    """

    pad_max: int
    jagged: bool = True

    def __post_init__(self) -> None:
        if self.pad_max < 1:
            raise ConfigError(f"StreamConfig: pad_max must be >= 1, got {self.pad_max}")


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


def _truncate_pad(
    cols: dict[str, Any],
    fields: list[str],
    stream_cfg: StreamConfig,
    b: int,
    gschema: GroupSchema | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate -> pad a jagged stream into a structured ``(B, T)`` array.

    1. truncate: keep the leading ``pad_max`` constituents.
    2. pad: ``valid`` computed from the truncated per-row count; each field
       ``ak.pad_none`` + fill per dtype (`pad_fill`), cast to the schema dtype,
       with a trailing ``valid`` bool field.

    Returns ``(structured (B, T) array, valid (B, T) bool)``.
    """
    import awkward as ak

    t_dim = stream_cfg.pad_max

    # valid length FIRST: per-row count, clipped to T (parity path)
    first = cols[fields[0]]
    counts = np.asarray(ak.num(first, axis=1)) if b > 0 else np.zeros(0, dtype=np.int64)
    valid = np.arange(t_dim)[None, :] < np.minimum(counts, t_dim)[:, None]  # (B, T) bool

    dtype_fields: list[tuple[str, np.dtype]] = []
    blocks: dict[str, np.ndarray] = {}
    for f in fields:
        # `pad_none(..., clip=True)` already TRUNCATES to `t_dim` as well as
        # padding to it, so slicing to `[:, :t_dim]` first only builds a whole
        # intermediate jagged array for `pad_none` to redo the same cut on.
        arr = cols[f]
        padded = ak.pad_none(arr, t_dim, axis=1, clip=True)
        dt = np.dtype(gschema.fields[f]) if gschema is not None else None
        fill = pad_fill(dt, arr)
        # Fill the pad slots in NUMPY, not awkward. `ak.fill_none` walks the
        # array and builds a second awkward array for `ak.to_numpy` to then
        # materialise, so the pair costs two traversals; `to_numpy` on an
        # option-type array already hands back a masked array carrying exactly
        # the same information, and filling that is one vectorised numpy write.
        # A row set that happens to need no padding comes back unmasked, hence
        # the isinstance check rather than an unconditional `.filled`.
        dense = ak.to_numpy(padded)
        block = dense.filled(fill) if isinstance(dense, np.ma.MaskedArray) else np.asarray(dense)
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


@dataclass
class OffsetIndex:
    """Cumulative row offsets across a file list + covering-range mapping.

    The simple flavour (``offsets`` only) maps a contiguous global row slice to
    the ``(file_index, local_start, local_stop)`` runs that cover it. The
    covering flavour (a per-entry cumulative array) maps a contiguous slice
    over a filtered / derived index back to a covering range over an
    underlying coarser index (kept-row -> covering-entry).

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
        the offset of ``lo`` within ``c0``'s first derived row.

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
