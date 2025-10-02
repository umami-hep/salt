import torch
from torch import Tensor


def symmetrize_edge_scores(scores: Tensor, node_numbers: Tensor) -> Tensor:
    """Make directed edge scores symmetric within each graph and squash to (0, 1).

    For a batch of graphs concatenated along the edge dimension, this function
    averages each directed edge score with its opposite-direction counterpart and
    returns the elementwise sigmoid of the averaged scores.

    Parameters
    ----------
    scores : Tensor
        Concatenated edge-score tensor of shape ``(E, 1)``, where ``E`` is the sum
        over graphs of ``n_i * (n_i - 1)`` (directed edges without self-loops).
    node_numbers : Tensor
        Tensor of shape ``(B,)`` (integer dtype) with the number of nodes in each
        graph of the batch. Graphs with ``n <= 1`` are ignored.

    Returns
    -------
    Tensor
        Symmetrized, sigmoid-squashed edge scores of shape ``(E, 1)``.
    """
    node_numbers = node_numbers[node_numbers > 1].long()  # remove graphs without edges
    edge_numbers = (node_numbers * (node_numbers - 1)).long()

    # cumulative edge counts (prefix sums) and per-graph edge offsets
    cum_edges = torch.cumsum(edge_numbers, 0).long()
    edge_offsets = torch.cat([torch.tensor([0], device=cum_edges.device).long(), cum_edges[:-1]])

    # compute indices of the opposite-direction edges within each graph's block
    sym_ind = torch.cat([
        torch.arange(n - 1, n * (n - 1) ** 2 + 1, n - 1, device=node_numbers.device)
        for n in node_numbers
    ])
    sym_ind += torch.cat([
        torch.arange(0, n - 1, device=sym_ind.device).repeat(n).sort()[0] * n for n in node_numbers
    ])
    sym_ind = sym_ind % torch.cat([n.repeat(n) for n in edge_numbers])
    sym_ind += torch.cat([
        edge_offsets[i].repeat(edge_numbers[i]) for i in range(len(edge_offsets))
    ])

    edge_scores = (scores + scores[sym_ind]) / 2.0
    return torch.sigmoid(edge_scores.float())


def update_node_indices(
    scores: Tensor,
    node_indices: Tensor,
    update_indices: Tensor,
    node_numbers: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run one union-find-style update step over a batch of graphs.

    For each graph in the batch, the function updates the ``node_indices`` array by
    propagating the minimum index along edges whose scores are at least ``0.5``.
    Graphs that do not change during this step are marked as finished in
    ``update_indices`` (set to ``False``).

    Parameters
    ----------
    scores : Tensor
        Concatenated edge-score tensor of shape ``(E, 1)`` (after any
        symmetrization), where ``E = Σ_i n_i (n_i - 1)``.
    node_indices : Tensor
        Concatenated per-node indices of shape ``(N,)`` (integer dtype) to be
        updated in-place (where ``N = Σ_i n_i``).
    update_indices : Tensor
        Boolean tensor of shape ``(B,)`` indicating which graphs should still be
        updated (``True`` means “continue updating”).
    node_numbers : Tensor
        Tensor of shape ``(B,)`` (integer dtype) with the number of nodes in each
        graph.

    Returns
    -------
    tuple[Tensor, Tensor]
        Updated ``(node_indices, update_indices)``.
    """
    edge_offset = node_offset = 0
    for i, _nnodes in enumerate(node_numbers):
        nnodes = int(_nnodes.item())
        nedges = nnodes * (nnodes - 1)

        if nnodes <= 1:
            update_indices[i] = False

        if not update_indices[i]:
            edge_offset += nedges
            node_offset += nnodes
            continue

        node_ids = torch.arange(0, nnodes)
        edge_ids = torch.transpose(
            torch.stack((node_ids.repeat(nnodes).sort()[0], node_ids.repeat(nnodes))), 0, 1
        )
        edge_ids = edge_ids[edge_ids[:, 0] != edge_ids[:, 1]]

        max_val = torch.max(node_indices[node_offset : node_offset + nnodes]) + 1

        # source and destination component ids for each directed edge
        src_ids = node_indices[node_offset : node_offset + nnodes][edge_ids[:, 0]]
        dest_ids = node_indices[node_offset : node_offset + nnodes][edge_ids[:, 1]]

        # only consider edges with score >= 0.5
        dest_ids[scores[edge_offset : edge_offset + nedges, 0] < 0.5] = max_val
        min_ids = torch.minimum(src_ids, dest_ids)

        # assign each node the minimum id observed across its incident edges
        node_indices_prev = torch.clone(node_indices[node_offset : node_offset + nnodes])
        node_indices[node_offset : node_offset + nnodes] = torch.min(
            torch.reshape(min_ids, (nnodes, -1)), dim=-1
        )[0]

        # if nothing changed for this graph, stop updating it
        update_indices[i] = torch.any(
            ~torch.eq(node_indices_prev, node_indices[node_offset : node_offset + nnodes])
        )

        edge_offset += nedges
        node_offset += nnodes

    return node_indices, update_indices


def get_node_assignment(output: Tensor, mask: Tensor) -> Tensor:
    """Compute connected-component/cluster assignments from edge scores.

    This wraps :func:`symmetrize_edge_scores` and repeatedly calls
    :func:`update_node_indices` until convergence to produce per-node
    component indices.

    Parameters
    ----------
    output : Tensor
        Concatenated directed edge scores of shape ``(E, 1)`` for a batch
        of graphs.
    mask : Tensor
        Padding mask of shape ``(B, S)`` (``True`` indicates padded).
        Used to determine the number of nodes per graph.

    Returns
    -------
    Tensor
        Concatenated per-node component indices of shape ``(N, 1)``.
    """
    # pad mask with one extra “track” to avoid ONNX issues
    mask = torch.cat(
        [mask, torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)], dim=1
    )
    node_numbers = (~mask).sum(dim=-1).long()

    # symmetrize edge scores
    scores = symmetrize_edge_scores(output, node_numbers) if output.shape[0] > 0 else output

    # union-find-like propagation until no more changes
    node_indices = torch.cat([torch.arange(nnodes) for nnodes in node_numbers])
    update_indices = torch.ones_like(node_numbers, dtype=torch.bool)
    while torch.any(update_indices):
        node_indices, update_indices = update_node_indices(
            scores, node_indices, update_indices, node_numbers
        )

    return node_indices.unsqueeze(-1)


@torch.jit.script
def get_node_assignment_jit(output: Tensor, mask: Tensor) -> Tensor:
    """TorchScript wrapper for :func:`get_node_assignment`.

    Parameters
    ----------
    output : Tensor
        Concatenated directed edge scores of shape ``(E, 1)``.
    mask : Tensor
        Padding mask of shape ``(B, S)`` (``True`` indicates padded).

    Returns
    -------
    Tensor
        Concatenated per-node component indices of shape ``(N, 1)``.
    """
    return get_node_assignment(output, mask)
