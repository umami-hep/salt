import math

import numpy as np

from salt.utils.inputs import as_half


def get_dtype_edge(ds, variables) -> np.dtype:
    """Return a dtype based on derived edge variables."""
    if variables is None:
        raise ValueError("Edge variables need to be specified if edge features are to be included")

    req_vars = []
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

    if missing := set(req_vars) - set(ds.dtype.names):
        raise ValueError(
            f"Variables {missing} required for edge feature calculation were not found in dataset"
            f" {ds.name} in file {ds.file.filename}"
        )

    return np.dtype([(n, as_half(x)) for n, x in ds.dtype.descr if n in req_vars])


def get_inputs_edge(batch, variables):
    """Calculate edge features from batch info."""
    ebatch = np.zeros(
        (batch.shape[0], batch.shape[1], batch.shape[1], len(variables)), dtype=np.float32
    )

    with np.errstate(divide="ignore"):
        # calculate useful intermediate quantities
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
        # calculate edge feature information and fill batch
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
                ebatch[:, :, :, i] = np.identity(batch.shape[1])
            elif variable == "subjetIndex":
                sji1 = np.expand_dims(batch["subjetIndex"].astype(ebatch.dtype), 1).repeat(
                    batch.shape[1], 1
                )
                sji2 = np.expand_dims(batch["subjetIndex"].astype(ebatch.dtype), 2).repeat(
                    batch.shape[1], 2
                )
                ebatch[:, :, :, i] = np.logical_and(np.equal(sji1, sji2), sji1 >= 0)

    return np.nan_to_num(ebatch, nan=0.0, posinf=0.0, neginf=0.0)
