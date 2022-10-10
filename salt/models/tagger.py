import torch.nn as nn

from salt.models.dense import Dense


class JetTagger(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list,
        activation: nn.Module,
    ):
        """Jet tagger model.

        Parameters
        ----------
        input_size : int
            Number of input features per track
        output_size : int
            Number of output classes
        hidden_layers : list
            Number of nodes per hidden layer
        activation : nn.Module
            Activation function
        """
        super().__init__()

        self.track_net = Dense(input_size, hidden_layers[-1], hidden_layers, activation)
        self.jet_net = Dense(hidden_layers[-1], output_size, hidden_layers, activation)

    def forward(self, x):
        embd_tracks = self.track_net(x)
        preds = self.jet_net(embd_tracks.sum(axis=1))
        return preds
