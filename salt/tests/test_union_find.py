import torch

from salt.utils.union_find import get_node_assignment


def test_no_vertex() -> None:
    """Test union find on a 2 track jet with no vertices."""
    labels = torch.tensor([0, 1])
    pairwise_probs = torch.tensor([[-1.0, -1.0]])
    mask = torch.tensor([[False, False]])
    assert torch.equal(
        get_node_assignment(pairwise_probs.flatten().unsqueeze(dim=-1), mask),
        labels.unsqueeze(dim=-1),
    )


def test_single_vertex_1() -> None:
    """Test union find on a 3 track jet with one two track vertex."""
    labels = torch.tensor([0, 1, 0])
    pairwise_probs = torch.tensor([[-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    mask = torch.tensor([[False, False, False]])
    assert torch.equal(
        get_node_assignment(pairwise_probs.flatten().unsqueeze(dim=-1), mask),
        labels.unsqueeze(dim=-1),
    )


def test_single_vertex_2() -> None:
    """Test union find on a 3 track jet with one three track vertex."""
    labels = torch.tensor([0, 0, 0])
    pairwise_probs = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mask = torch.tensor([[False, False, False]])
    assert torch.equal(
        get_node_assignment(pairwise_probs.flatten().unsqueeze(dim=-1), mask),
        labels.unsqueeze(dim=-1),
    )


def test_single_vertex_3() -> None:
    """Test union find on a 4 track jet with one three track vertex."""
    labels = torch.tensor([0, 1, 1, 1])
    pairwise_probs = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]
    )
    mask = torch.tensor([[False, False, False, False]])
    assert torch.equal(
        get_node_assignment(pairwise_probs.flatten().unsqueeze(dim=-1), mask),
        labels.unsqueeze(dim=-1),
    )


def test_mult_vertices() -> None:
    """Test union find on a 3 and 4 track jet.

    The former with one three track vertex and the latter with two two track vertices.
    """
    labels_jet1 = torch.tensor([0, 0, 2])
    labels_jet2 = torch.tensor([0, 1, 1, 0])
    labels = torch.cat([labels_jet1.flatten(), labels_jet2.flatten()], dim=0)
    pairwise_probs_jet1 = torch.tensor([[1.0, -1.0], [1.0, -1.0], [-1.0, -1.0]])
    pairwise_probs_jet2 = torch.tensor(
        [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    pairwise_probs = torch.cat(
        [pairwise_probs_jet1.flatten(), pairwise_probs_jet2.flatten()], dim=0
    )
    mask = torch.tensor([[False, False, False, True], [False, False, False, False]])
    assert torch.equal(
        get_node_assignment(pairwise_probs.flatten().unsqueeze(dim=-1), mask),
        labels.flatten().unsqueeze(dim=-1),
    )
