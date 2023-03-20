import torch
from torch import Tensor, nn

from salt.models.attention import masked_softmax


class Pooling(nn.Module):
    ...


class GlobalAttentionPooling(Pooling):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            mask = mask.unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x), mask, dim=1)

        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weights = torch.cat([weights, torch.zeros((weights.shape[0], 1, weights.shape[2]))], dim=1)
        x = torch.cat([x, torch.zeros((x.shape[0], 1, x.shape[2]))], dim=1)

        return (x * weights).sum(dim=1)
