"""Core-local copy of the v1 dtype helpers (M7 W2a relocation).

``get_dtype`` (v1 ``salt.data.datasets.get_dtype``) builds a structured numpy
dtype for a requested variable subset, casting float16-like fields to half via
``as_half`` (v1 ``salt.utils.inputs.as_half``, pulled transitively). Both are
copied BYTE-FAITHFULLY (identical logic); the v1 originals stay in place as the
RS1 gate oracle. ``salt.core.data.reader`` imports ``get_dtype`` from here.
"""

from collections.abc import Iterable
from typing import Any

import h5py
import numpy as np


def as_half(typestr: Any) -> np.dtype:
    """Return a NumPy dtype, “casting” float16-like specifiers to half precision.

    Parameters
    ----------
    typestr : Any
        Any object understood by :class:`numpy.dtype` (e.g., ``"f2"``, ``np.float16``,
        or an existing dtype).

    Returns
    -------
    numpy.dtype
        If ``typestr`` corresponds to a floating type of itemsize ``2`` bytes, returns
        ``np.dtype("f2")``; otherwise returns the dtype constructed from ``typestr``.
    """
    t = np.dtype(typestr)
    if t.kind != "f" or t.itemsize != 2:
        return t
    return np.dtype("f2")


def get_dtype(ds: h5py.Dataset, variables: Iterable[str] | None = None) -> np.dtype:
    """Return a structured dtype based on an existing dataset and requested variables.

    Parameters
    ----------
    ds : h5py.Dataset
        Input dataset providing the source structured dtype (``ds.dtype``).
    variables : Iterable[str] | None, optional
        Variable names to include in the returned dtype. If ``None``, use
        ``ds.dtype.names``. If the dataset contains a ``"valid"`` field and
        it is not listed, it will be appended automatically.

    Returns
    -------
    numpy.dtype
        Structured dtype consisting of the requested fields. Each field's
        element dtype is converted via :func:`as_half`.
    """
    # Normalize to a concrete, mutable list
    if variables is None:
        variables_list: list[str] = list(ds.dtype.names or [])
    else:
        variables_list = list(variables)

    if "valid" in (ds.dtype.names or ()) and "valid" not in variables_list:
        variables_list.append("valid")

    variables_flat: list[str] = []
    for item in variables_list:
        if isinstance(item, list):
            variables_flat.extend(item)
        else:
            variables_flat.append(item)

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in variables_flat])
