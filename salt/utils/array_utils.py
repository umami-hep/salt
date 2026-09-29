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
