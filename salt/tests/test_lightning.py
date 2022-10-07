import torch

from salt.lightning import MyModel
from salt.models.dense import Dense


def test_lightning():
    net = Dense(10, 10, [10])
    model = MyModel(net)
    model(torch.rand(10))
