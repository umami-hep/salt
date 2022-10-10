import torch

from salt.lightning import LightningTagger
from salt.models.dense import Dense


def test_lightning():
    net = Dense(10, 10, [10])
    model = LightningTagger(net)
    model(torch.rand(10))
