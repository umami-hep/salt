from collections.abc import Sequence
from typing import Any

import numpy as np


def join_structured_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Join a list of structured NumPy arrays into a single structured array.

    Notes
    -----
    This follows the approach discussed in:
    https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list[np.ndarray]
        List of **structured** arrays (same shape) to join. Each array must
        have a structured dtype (with named fields). All arrays must share the
        same first dimension (i.e., number of rows).

    Returns
    -------
    np.ndarray
        A merged structured array whose dtype is the concatenation of the
        input field descriptors and whose shape matches the inputs.
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
    """Convert a scalar or list to a list (preserving ``None``).

    Parameters
    ----------
    maybe_list : list[Any] | Any | None
        A scalar, list, or ``None``.

    Returns
    -------
    Sequence[Any] | None
        ``None`` if the input is ``None``; otherwise a list. If the input was
        already a list, it is returned unchanged; otherwise it is wrapped as a
        single-element list.
    """
    if maybe_list is None:
        return None  # type: ignore[return-value]
    if isinstance(maybe_list, list):
        return maybe_list
    return [maybe_list]


def maybe_pad(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Pad ``src`` to match the shape of ``tgt`` along the sequence dimension.

    This is a convenience helper for 2D arrays where the second dimension is a
    variable-length sequence. If both arrays already have the same shape, the
    input ``src`` is returned unchanged.

    Parameters
    ----------
    src : np.ndarray
        Source array to be padded. Expected shape is ``(N, L_src)`` or any
        shape equal to ``tgt.shape``.
    tgt : np.ndarray
        Target array that defines the desired shape. If 2D, its second
        dimension (``L_tgt``) is used to compute the padding for ``src``.

    Returns
    -------
    np.ndarray
        The original ``src`` if no padding is required; otherwise a copy of
        ``src`` padded with zeros on the sequence dimension to match ``tgt``.
    """
    if src.shape == tgt.shape:
        return src
    seq_len = tgt.shape[1] if tgt.ndim == 2 else None
    if seq_len and seq_len != src.shape[1]:
        n_pad = seq_len - src.shape[1]
        src = np.pad(src, ((0, 0), (0, n_pad)), mode="constant")
    return src


def maybe_copy(src: np.ndarray) -> np.ndarray:
    """Return a contiguous array, copying only if needed.

    Parameters
    ----------
    src : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        ``src`` itself if it is already C-contiguous; otherwise a contiguous
        copy of ``src``.
    """
    if src.flags.c_contiguous:
        return src
    return src.copy()
