import copy

import numpy as np
import pytest
import torch
from torch import nn

from salt.models import (
    Dense,
    GlobalAttentionPooling,
)
from salt.utils.inputs import get_random_mask


def test_dense() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU")
    net(torch.rand(10))


def test_dense_context() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(10), torch.rand(4))


def test_dense_context_broadcast() -> None:
    net = Dense(10, 10, [10, 10], activation="ReLU", context_size=4)
    net(torch.rand(1, 10, 10), torch.rand(1, 4))


# @pytest.mark.parametrize("pooling", [GlobalAttentionPooling, TensorCrossAttentionPooling])
def test_pooling() -> None:
    net = GlobalAttentionPooling(10)

    x = {"emb": torch.rand(1, 5, 10)}
    out = net(x)

    x = {"emb": torch.cat([x["emb"], torch.zeros((1, 1, x["emb"].shape[2]))], dim=1)}
    mask = get_random_mask(1, 6, p_valid=1)
    mask[:, -1] = True
    mask = {"mask": mask}
    out_with_mask = net(x, pad_mask=mask)
    assert torch.all(out == out_with_mask)
