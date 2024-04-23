import torch
from torch import Tensor


def symmetrize_edge_scores(scores: Tensor, node_numbers: Tensor):
    """Function to make edge scores symmetric.

    Output has same format as input: (edges in batch, 1). Node number
    array gives number of nodes per graph in batch as a tensor.
    """
    node_numbers = node_numbers[node_numbers > 1].long()  # remove jets without edges
    edge_numbers = (node_numbers * (node_numbers - 1)).long()

    # calculate cumulative edge numbers and offsets for symmetric index calculation
    cum_edges = torch.cumsum(edge_numbers, 0).long()
    edge_offsets = torch.cat([torch.tensor([0], device=cum_edges.device).long(), cum_edges[:-1]])

    torch.tensor_split(scores, cum_edges)

    # calculate opposite edge indices (assumes edges sorted by src then dest or vice versa)
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
    scores: Tensor, node_indices: Tensor, update_indices: Tensor, node_numbers: Tensor
):
    """Run a single step of the union find algorithm.

    Takes a score matrix with shape (edges in batch, 1) and a tensor with
    the number of nodes in each graph of the batch as well as a tensor specifying
    for which graphs the algorithm has already terminated. Returns updated vertex
    index and algorithm termination tensors.
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

        # create arrays with node indices for source and destination nodes of each edge
        src_ids = node_indices[node_offset : node_offset + nnodes][edge_ids[:, 0]]
        dest_ids = node_indices[node_offset : node_offset + nnodes][edge_ids[:, 1]]

        # pick out minimum between source and destination node ids, filter edges with scores < 0.5
        dest_ids[scores[edge_offset : edge_offset + nedges, 0] < 0.5] = max_val
        min_ids = torch.minimum(src_ids, dest_ids)

        # get the lowest node id for each node across all edges for each node
        node_indices_prev = torch.clone(node_indices[node_offset : node_offset + nnodes])
        node_indices[node_offset : node_offset + nnodes] = torch.min(
            torch.reshape(min_ids, (nnodes, -1)), dim=-1
        )[0]

        update_indices[i] = torch.any(
            ~torch.eq(node_indices_prev, node_indices[node_offset : node_offset + nnodes])
        )

        edge_offset += nedges
        node_offset += nnodes

    return node_indices, update_indices


def get_node_assignment(output: Tensor, mask: Tensor):
    """Run edge score symmetrization and union find.

    Wrapper function which returns reconstructed vertex indices in shape
    (nodes in batch, 1). Assumes mask of shape (batch, max_tracks).
    """
    # pad mask with additional track to avoid onnx error
    mask = torch.cat(
        [mask, torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)], dim=1
    )
    node_numbers = (~mask).sum(dim=-1).long()

    # symmetrize edge scores
    scores = symmetrize_edge_scores(output, node_numbers) if output.shape[0] > 0 else output

    # update node assignments until no more changes occur
    node_indices = torch.cat([torch.arange(nnodes) for nnodes in node_numbers])
    update_indices = torch.ones_like(node_numbers, dtype=torch.bool)
    while torch.any(update_indices):
        node_indices, update_indices = update_node_indices(
            scores, node_indices, update_indices, node_numbers
        )

    return node_indices.unsqueeze(-1)


@torch.jit.script
def get_node_assignment_jit(output: Tensor, mask: Tensor):
    return get_node_assignment(output, mask)
