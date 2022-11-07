import torch
import torch.nn as nn


class Pooling(nn.Module):
    ...


class GlobalAttentionPooling(Pooling):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(self, x, mask):
        weights = torch.softmax(self.gate_nn(x), dim=1)
        weights[mask] == 0
        return (x * weights).sum(dim=1)
