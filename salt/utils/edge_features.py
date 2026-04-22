from __future__ import annotations

import math

import torch

from salt.stypes import Tensor


def check_edge_config(
    edge_features: list[str],
    available_vars: list[str],
):
    """Check the provided edge feature configuration for validity.

    Parameters
    ----------
    edge_features : list[str]
        List of edge features to compute.
    available_vars : list[str]
        List of available variables.

    Raises
    ------
    ValueError
        If an edge feature is not recognized or if required indices are missing.
    """
    req_vars: list[str] = []
    for variable in edge_features:
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
        elif variable == "mass":
            req_vars.extend(["pt", "eta", "phi", "energy"])
        else:
            raise ValueError(f"Edge feature {variable} not recognized")

    missing = set(req_vars) - set(available_vars)
    if missing:
        raise ValueError(
            f"Indices of {missing} required for edge features calculation were not specified."
        )


def calculate_edge_features(
    batch: Tensor,
    indices_map: dict[str, int],
    variables: list[str],
) -> Tensor:
    """Calculate edge features for a given batch of graphs.

    Parameters
    ----------
    batch : Tensor
        Input batch of node features of shape ``[B, N, D]``.
    indices_map : dict[str, int]
        Mapping variable names to indices in the node feature tensor.
    variables : list[str]
        List of edge features to compute.

    Returns
    -------
    Tensor
        Computed edge features tensor of shape ``[B, N, N, num_edge_features]``.
    """
    ebatch = torch.zeros(
        (batch.shape[0], batch.shape[1], batch.shape[1], len(variables)),
        dtype=batch.dtype,
        device=batch.device,
    )

    # intermediate quantities
    if "dR" in variables or "kt" in variables:
        dphi = batch[:, :, indices_map["phi"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["phi"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
        dphi -= (dphi > math.pi).type_as(dphi) * 2 * math.pi
        deta = batch[:, :, indices_map["eta"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["eta"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
    if "kt" in variables or "z" in variables:
        pt_min = torch.minimum(
            batch[:, :, indices_map["pt"]].unsqueeze(1).expand(-1, batch.shape[1], -1),
            batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1]),
        )
    if "mass" in variables:
        pt = batch[:, :, indices_map["pt"]]
        eta = batch[:, :, indices_map["eta"]]
        phi = batch[:, :, indices_map["phi"]]
        energy = batch[:, :, indices_map["energy"]]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * (torch.exp(eta) - torch.exp(-eta)) / 2

    # fill edge features
    for i, variable in enumerate(variables):
        if variable == "dR":
            ebatch[:, :, :, i] = torch.log(torch.sqrt(torch.square(deta) + torch.square(dphi)))
        elif variable == "kt":
            ebatch[:, :, :, i] = torch.log(
                pt_min * torch.sqrt(torch.square(deta) + torch.square(dphi))
            )
        elif variable == "z":
            pt_sum = batch[:, :, indices_map["pt"]].unsqueeze(1).expand(
                -1, batch.shape[1], -1
            ) + batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            ebatch[:, :, :, i] = torch.log(pt_min / pt_sum)
        elif variable == "isSelfLoop":
            ebatch[:, :, :, i] = (
                torch.eye(batch.shape[1], dtype=ebatch.dtype, device=batch.device)
                .unsqueeze(0)
                .expand(batch.shape[0], -1, -1)
            )
        elif variable == "subjetIndex":
            sji1 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(1).expand(-1, batch.shape[1], -1)
            )
            sji2 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            )
            ebatch[:, :, :, i] = torch.logical_and(torch.eq(sji1, sji2), sji1 >= 0)
        elif variable == "mass":
            e1 = energy.unsqueeze(1).expand(-1, batch.shape[1], -1)
            e2 = energy.unsqueeze(2).expand(-1, -1, batch.shape[1])
            px1 = px.unsqueeze(1).expand(-1, batch.shape[1], -1)
            px2 = px.unsqueeze(2).expand(-1, -1, batch.shape[1])
            py1 = py.unsqueeze(1).expand(-1, batch.shape[1], -1)
            py2 = py.unsqueeze(2).expand(-1, -1, batch.shape[1])
            pz1 = pz.unsqueeze(1).expand(-1, batch.shape[1], -1)
            pz2 = pz.unsqueeze(2).expand(-1, -1, batch.shape[1])
            e_sum = e1 + e2
            px_sum = px1 + px2
            py_sum = py1 + py2
            pz_sum = pz1 + pz2
            mass2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
            mass2 = torch.clamp_min(mass2, 1e-8)
            ebatch[:, :, :, i] = 0.5 * torch.log(mass2)

    return torch.nan_to_num(ebatch, nan=0.0, posinf=0.0, neginf=0.0)
