import torch

from salt.lightning import LightningTagger
from salt.models.tagger import JetTagger


def test_lightning():
    net = JetTagger(21, 3, [64, 64], activation=torch.nn.ReLU)
    model = LightningTagger(net, loss_weights={0: 1})
    model(torch.rand(10, 10, 21))
