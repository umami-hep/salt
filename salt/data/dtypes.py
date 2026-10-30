"""Dtype helpers for H5 structured-array reads.

``get_dtype`` builds a structured numpy dtype for a requested variable
subset, casting float16-like fields to half via ``as_half``.
``salt.data.readers.reader`` imports ``get_dtype`` from here.
"""

from collections.abc import Iterable
from typing import Any

import h5py
import numpy as np


def as_half(typestr: Any) -> np.dtype:
    """Cast a float16-like dtype specifier to half precision (``f2``); other dtypes pass
    through unchanged.
    """
    t = np.dtype(typestr)
    if t.kind != "f" or t.itemsize != 2:
        return t
    return np.dtype("f2")


def get_dtype(ds: h5py.Dataset, variables: Iterable[str] | None = None) -> np.dtype:
    """Structured dtype for the requested `variables` (default: all of ``ds.dtype.names``),
    each field cast via `as_half`; a present ``"valid"`` field is auto-included.
    """
    names = list(ds.dtype.names or [])
    wanted: list[Any] = names if variables is None else list(variables)
    if "valid" in names and "valid" not in wanted:
        wanted.append("valid")
    # a nested list item contributes its members
    flat = [v for item in wanted for v in (item if isinstance(item, list) else [item])]
    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in flat])
