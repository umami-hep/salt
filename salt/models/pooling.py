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
        return (x * weights).sum(dim=1)
