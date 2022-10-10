import torch.nn as nn


class JetClassificationLoss(nn.Module):
    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0):
        """Jet cross entropy classification loss.

        Parameters
        ----------
        weight : float, optional
            Apply weighting to the computed loss, by default 1.0
        label_smoothing : float, optional
            Apply label smoothing, by default 0.0
        """
        super().__init__()
        self.weight = weight
        self.jet_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, true):
        return self.jet_loss(preds, true) * self.weight
