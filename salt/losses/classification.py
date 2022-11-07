import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        """Cross entropy classification loss.

        Parameters
        ----------
        label_smoothing : float, optional
            Apply label smoothing, by default 0.0
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, true):
        if preds.dim() == 3:
            # for tracks, flatten across jets and compute loss on unmasked inputs
            preds = preds.flatten(end_dim=1)
            true = true.flatten(end_dim=1)
            mask = true != -1  # TODO: pass valid flag (or use MaskedTensor)
            preds = preds[mask]
            true = true[mask]

        return self.loss(preds, true)
