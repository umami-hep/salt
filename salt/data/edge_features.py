from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from salt.utils.inputs import as_half

if TYPE_CHECKING:  # pragma: no cover
    import h5py


def get_dtype_edge(
    ds: h5py.Dataset,
    variables: Sequence[str] | None,
) -> np.dtype:
    """Derive an edge feature NumPy dtype from a dataset and a list of variables.

    This utility inspects a dataset's structured dtype and selects only
    the fields needed to compute the requested edge features. It also
    checks that all required base variables are present.

    Parameters
    ----------
    ds : h5py.Dataset
        An HDF5 dataset with a structured dtype (``ds.dtype.names``)
        containing the base variables needed for edge features.
    variables : Sequence[str] | None
        Names of edge variables to include. Recognized edge features are:

        * ``"dR"`` → requires ``eta``, ``phi``.
        * ``"z"`` → requires ``pt``.
        * ``"kt"`` → requires ``eta``, ``phi``, ``pt``.
        * ``"isSelfLoop"`` → skipped (no extra vars).
        * ``"subjetIndex"`` → requires ``subjetIndex``.

        If ``None``, a ``ValueError`` is raised.

    Returns
    -------
    numpy.dtype
        A structured dtype containing only the fields required to compute
        the specified edge features. Each field's dtype is converted by
        the ``as_half`` function (must be in scope).

    Raises
    ------
    ValueError
        If ``variables`` is ``None``.
        If any required base variable is missing from ``ds.dtype.names``.
        If an unknown edge feature name is supplied.

    Examples
    --------
    >>> import h5py, numpy as np
    >>> ds = h5py.File("edges.h5")["edges"]
    >>> dtype = get_dtype_edge(ds, ["dR", "z"])
    >>> dtype.names
    ('eta', 'phi', 'pt')
    """
    if variables is None:
        raise ValueError("Edge variables need to be specified if edge features are to be included")

    req_vars: list[str] = []
    for variable in variables:
        if variable == "dR":
            req_vars.extend(["eta", "phi"])
        elif variable == "z":
            req_vars.extend(["pt"])
        elif variable == "kt":
            req_vars.extend(["eta", "phi", "pt"])
        elif variable == "isSelfLoop":
            continue
        elif variable == "subjetIndex":
            req_vars.extend(["subjetIndex"])
        else:
            raise ValueError(f"Edge feature {variable} not recognized")

    if "valid" in ds.dtype.names and "valid" not in variables:
        req_vars.append("valid")

    missing = set(req_vars) - set(ds.dtype.names)
    if missing:
        raise ValueError(
            f"Variables {missing} required for edge feature calculation were not found "
            f"in dataset {ds.name} in file {ds.file.filename}"
        )

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in req_vars])


def get_inputs_edge(
    batch: np.ndarray,
    variables: Sequence[str],
) -> np.ndarray:
    """Compute edge features for all pairs of nodes in a batch.

    This constructs a 4-D array of shape ``(N, M, M, K)`` where:

    * ``N`` = number of batches (first dimension of ``batch``)
    * ``M`` = number of nodes per batch (second dimension of ``batch``)
    * ``K`` = number of requested edge features (``len(variables)``)

    Each slice ``ebatch[n, :, :, k]`` contains the computed edge feature
    ``variables[k]`` for all node-to-node pairs in batch ``n``.

    Parameters
    ----------
    batch : np.ndarray
        Structured or record array containing at least the fields used by
        the requested edge variables:

        * ``eta``, ``phi`` for ``"dR"`` and ``"kt"``
        * ``pt`` for ``"z"`` and ``"kt"``
        * ``subjetIndex`` for ``"subjetIndex"``

    variables : Sequence[str]
        Names of edge variables to compute. Recognised values:

        * ``"dR"``: log of ΔR distance
        * ``"kt"``: log of kT distance
        * ``"z"``: log of pT fraction
        * ``"isSelfLoop"``: indicator for self-edges
        * ``"subjetIndex"``: indicator for same-subjet edges

    Returns
    -------
    np.ndarray
        Array of shape ``(N, M, M, len(variables))`` with computed edge features.
        All NaNs and infs are replaced by 0.

    Notes
    -----
    This function uses ``np.log`` of intermediate values; if you supply
    zero or negative inputs, NaNs may be produced but are set to 0.0 in
    the returned array.

    Examples
    --------
    >>> ebatch = get_inputs_edge(batch, ["dR", "z"])
    >>> ebatch.shape
    (n_batch, n_nodes, n_nodes, 2)
    """
    ebatch = np.zeros(
        (batch.shape[0], batch.shape[1], batch.shape[1], len(variables)),
        dtype=np.float32,
    )

    with np.errstate(divide="ignore"):
        # intermediate quantities
        if "dR" in variables or "kt" in variables:
            dphi = np.expand_dims(batch["phi"].astype(ebatch.dtype), 1).repeat(
                batch.shape[1], 1
            ) - np.expand_dims(batch["phi"].astype(ebatch.dtype), 2).repeat(batch.shape[1], 2)
            dphi -= np.ones_like(dphi) * (dphi > math.pi) * 2 * math.pi
            deta = np.expand_dims(batch["eta"].astype(ebatch.dtype), 1).repeat(
                batch.shape[1], 1
            ) - np.expand_dims(batch["eta"].astype(ebatch.dtype), 2).repeat(batch.shape[1], 2)
        if "kt" in variables or "z" in variables:
            pt_min = np.minimum(
                np.expand_dims(batch["pt"].astype(ebatch.dtype), 1).repeat(batch.shape[1], 1),
                np.expand_dims(batch["pt"].astype(ebatch.dtype), 2).repeat(batch.shape[1], 2),
            )
        # fill edge features
        for i, variable in enumerate(variables):
            if variable == "dR":
                ebatch[:, :, :, i] = np.log(np.sqrt(np.square(deta) + np.square(dphi)))
            elif variable == "kt":
                ebatch[:, :, :, i] = np.log(pt_min * np.sqrt(np.square(deta) + np.square(dphi)))
            elif variable == "z":
                pt_sum = np.expand_dims(batch["pt"].astype(ebatch.dtype), 1).repeat(
                    batch.shape[1], 1
                ) + np.expand_dims(batch["pt"].astype(ebatch.dtype), 2).repeat(batch.shape[1], 2)
                ebatch[:, :, :, i] = np.log(pt_min / pt_sum)
            elif variable == "isSelfLoop":
                ebatch[:, :, :, i] = np.identity(batch.shape[1], dtype=ebatch.dtype)
            elif variable == "subjetIndex":
                sji1 = np.expand_dims(batch["subjetIndex"].astype(ebatch.dtype), 1).repeat(
                    batch.shape[1], 1
                )
                sji2 = np.expand_dims(batch["subjetIndex"].astype(ebatch.dtype), 2).repeat(
                    batch.shape[1], 2
                )
                ebatch[:, :, :, i] = np.logical_and(np.equal(sji1, sji2), sji1 >= 0)

    return np.nan_to_num(ebatch, nan=0.0, posinf=0.0, neginf=0.0)
