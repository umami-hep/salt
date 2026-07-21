from collections.abc import Sequence
from typing import Any

import numpy as np


def join_structured_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Join a list of structured NumPy arrays into a single structured array.

    All arrays must be structured (named fields) and share the same first
    dimension; the merged dtype concatenates the input field descriptors.
    """
    assert len(arrays) > 0, "arrays must be a non-empty list of structured arrays"
    # Combine dtype descriptors (a list of (name, format[, shape]) tuples)
    dtype: list = sum((a.dtype.descr for a in arrays), [])
    # Allocate an empty structured array with the combined dtype
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    # Copy each field by name
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def listify(maybe_list: list[Any] | Any | None) -> Sequence[Any]:
    """Convert a scalar or list to a list, preserving ``None``.

    Returns ``None`` unchanged; a list unchanged; otherwise wraps the value as
    a single-element list.
    """
    if maybe_list is None:
        return None  # type: ignore[return-value]
    if isinstance(maybe_list, list):
        return maybe_list
    return [maybe_list]


def maybe_pad(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Pad ``src`` to match ``tgt``'s shape along the sequence (2nd) dimension.

    Convenience helper for 2D arrays where the second dimension is a
    variable-length sequence. Returns ``src`` unchanged if the shapes already
    match; otherwise a zero-padded copy along axis 1.
    """
    if src.shape == tgt.shape:
        return src
    seq_len = tgt.shape[1] if tgt.ndim == 2 else None
    if seq_len and seq_len != src.shape[1]:
        n_pad = seq_len - src.shape[1]
        src = np.pad(src, ((0, 0), (0, n_pad)), mode="constant")
    return src


def maybe_copy(src: np.ndarray) -> np.ndarray:
    """Return ``src`` unchanged if C-contiguous, otherwise a contiguous copy."""
    if src.flags.c_contiguous:
        return src
    return src.copy()
