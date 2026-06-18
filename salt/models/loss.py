import torch
from torch import nn


class WeightedSumCompositeLoss(nn.Module):
    """Weighted sum of several loss modules."""

    def __init__(
        self,
        loss_list: list[nn.Module],
        loss_weights: list[float] | None = None,
    ):
        if loss_weights is None:
            loss_weights = [1.0] * len(loss_list)
        else:
            assert len(loss_weights) == len(loss_list), (
                "Length of loss_weights must match length of loss_list"
            )
        self.loss_weights = torch.tensor(loss_weights)
        self.loss_list = loss_list

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        final_loss = 0.0

        for weight, loss in zip(self.loss_weights, self.loss_list, strict=False):
            final_loss += weight * loss(preds, targets)

        return final_loss
