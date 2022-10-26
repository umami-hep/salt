import torch

from salt.models import Dense, Transformer


def test_dense():
    net = Dense(10, 10, [10, 10], activation="ReLU")
    net(torch.rand(10))


def test_transformer():
    net = Transformer(10, 2, 2, activation="ReLU")
    net(torch.rand(10, 10, 10))
