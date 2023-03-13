import torch
from torch import Tensor


def symmetrize_edge_scores(scores: Tensor, node_numbers: list):
    """Make edge scores symmetric.

    Output has same format as input: (edges in batch, 1). Node number
    array gives number of nodes per graph in batch as a list.
    """
    edge_scores = torch.zeros_like(scores)

    edge_offset = 0
    for nnodes in node_numbers:
        # generate map of edge ID's (assumes fully-connected graphs, edges sorted by src then dest)
        node_ids = torch.arange(0, nnodes)
        edge_ids = torch.transpose(
            torch.stack((torch.repeat_interleave(node_ids, nnodes), node_ids.repeat(nnodes))), 0, 1
        )
        edge_ids = edge_ids[edge_ids[:, 0] != edge_ids[:, 1]]

        # find indices of edges where source node > destination node and vice versa
        e_ij = torch.where(edge_ids[:, 0] > edge_ids[:, 1])[0]
        e_ji = torch.where(edge_ids[:, 1] > edge_ids[:, 0])[0]

        # sort e_ji indices by destination node (to match source node sorting of e_ij)
        sort_ji1 = torch.argsort(edge_ids[:, 1][e_ji], stable=True)
        e_ij += edge_offset
        e_ji = e_ji[sort_ji1] + edge_offset

        # symmetrize output scores for edge pairs that have source and destination node in common
        edge_scores[e_ji, 0] = (scores[e_ij, 0] + scores[e_ji, 0]) / 2
        edge_scores[e_ij, 0] = (scores[e_ij, 0] + scores[e_ji, 0]) / 2

        edge_offset += nnodes * (nnodes - 1)

    return torch.sigmoid(edge_scores)


def update_node_indices(scores: Tensor, node_indices: list, update_indices: list):
    """Run a single step of the union find algorithm.

    Takes a score matrix with shape (edges in batch, 1) and a list with
    the number of nodes in each graph of the batch as well as a list of
    booleans specifying which graphs still need updating. Returns an updated
    list of vertex indices for each node.
    """
    node_numbers = [nodes.size(dim=0) for nodes in node_indices]

    edge_offset = node_offset = 0
    for i, nnodes in enumerate(node_numbers):
        if not update_indices[i]:
            edge_offset += nnodes * (nnodes - 1)
            node_offset += nnodes
            continue

        node_ids = torch.arange(0, nnodes)
        edge_ids = torch.transpose(
            torch.stack((torch.repeat_interleave(node_ids, nnodes), node_ids.repeat(nnodes))), 0, 1
        )
        edge_ids = edge_ids[edge_ids[:, 0] != edge_ids[:, 1]]

        max_val = torch.max(node_indices[i]) + 1

        # create arrays with node indices for source and destination nodes of each edge
        src_ids = node_indices[i][edge_ids[:, 0]]
        dest_ids = node_indices[i][edge_ids[:, 1]]

        # pick out minimum between source and destination node ids, filter edges with scores < 0.5
        dest_ids[scores[edge_offset : edge_offset + nnodes * (nnodes - 1), 0] < 0.5] = max_val
        min_ids = torch.minimum(src_ids, dest_ids)

        # get the lowest node id for each node across all edges for each node
        if nnodes > 1:
            node_indices[i] = torch.min(torch.reshape(min_ids, (nnodes, -1)), dim=-1)[0]

        edge_offset += nnodes * (nnodes - 1)
        node_offset += nnodes

    return node_indices


def get_node_assignment(output: Tensor, node_numbers: list):
    """Run edge score symmetrization and union find.

    Wrapper function which returns reconstructed vertex indices in shape
    (nodes in batch, 1). Node number array gives number of nodes per graph
    in batch as a list (can be derived from mask).
    """
    # symmetrize edge scores
    scores = symmetrize_edge_scores(output, node_numbers)

    # update node assignments until no more changes occur
    node_indices = [torch.arange(nnodes) for nnodes in node_numbers if nnodes > 0]
    update_indices = [True for i in range(len(node_indices))]

    while any(update_indices):
        node_indices_prev = node_indices
        node_indices = update_node_indices(scores, node_indices_prev.copy(), update_indices)
        update_indices = [
            torch.any(nodes != node_indices_prev[i]).item() for i, nodes in enumerate(node_indices)
        ]

    return torch.cat(node_indices).unsqueeze(-1)
