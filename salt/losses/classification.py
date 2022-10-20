import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, label_smoothing: float = 0.0):
        """Jet cross entropy classification loss.

        Parameters
        ----------
        weight : str
            Name of the loss, used to access relevant preds and labels.
        weight : float, optional
            Apply weighting to the computed loss, by default 1.0
        label_smoothing : float, optional
            Apply label smoothing, by default 0.0
        """
        super().__init__()
        self.name = name
        self.weight = weight
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, true):
        preds = preds[self.name]
        true = true[self.name]

        if preds.dim() == 3:
            # for tracks, flatten across jets and compute loss on unmasked inputs
            preds = preds.flatten(end_dim=1)
            true = true.flatten(end_dim=1)
            mask = true != -1  # TODO: pass in mask from the valid flag
            preds = preds[mask]
            true = true[mask]

        return self.loss(preds, true) * self.weight
