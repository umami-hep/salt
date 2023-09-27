import numpy as np


def join_structured_arrays(arrays: list):
    """Join a list of structured numpy arrays.

    See https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list
        List of structured numpy arrays to join

    Returns
    -------
    np.array
        A merged structured array
    """
    dtype: list = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray


def listify(maybe_list):
    """Convert a scalar or list to a list.
    If None, returns None.

    Parameters
    ----------
    maybe_list
        A scalar, list or None

    Returns
    -------
    list
        A list or None
    """
    if maybe_list is None:
        return None
    if isinstance(maybe_list, list):
        return maybe_list
    return [maybe_list]


def maybe_pad(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Pad src to the shape of tgt if necessary."""
    if src.shape == tgt.shape:
        return src
    seq_len = tgt.shape[1] if tgt.ndim == 2 else None
    if seq_len and seq_len != src.shape[1]:
        n_pad = seq_len - src.shape[1]
        src = np.pad(src, ((0, 0), (0, n_pad)), mode="constant")
    return src
