import torch

from salt.models.dense import Dense


def test_dense():
    net = Dense(10, 10, [10, 10], activation="relu")
    net(torch.rand(10))
