import torch
import torch.nn as nn


class JetTagger(nn.Module):
    def __init__(
        self,
        init_net: nn.Module,
        gnn: nn.Module,
        pool_net: nn.Module,
        jet_net: nn.Module,
        track_net: nn.Module,
        tasks: dict,
    ):
        """Jet tagger model.

        Parameters
        ----------
        init_net : nn.Module
            Initialisation network
        gnn : nn.Module
            Graph neural network
        pool_net : nn.Module
            Pooling network
        jet_net : nn.Module
            Jet classification network
        track_net : nn.Module
            Track classification network
        tasks : dict
            Dict of tasks to perform
        """
        super().__init__()

        self.init_net = init_net
        self.gnn = gnn
        self.jet_net = jet_net
        self.pool_net = pool_net
        self.track_net = track_net
        self.tasks = tasks

    def get_track_mask(self, tracks: torch.Tensor):
        """The input track h5 dataset is filled for a fixed number of tracks
        per jet.

        The inputs of padded tracks are all either 0 or -1.
        """
        mask = ((tracks == 0) | (tracks == -1)).all(axis=-1)
        mask[:, 0] = False  # hack to make the MHA work
        return mask

    def forward(self, x):
        mask = self.get_track_mask(x)
        embd_x = self.init_net(x)
        embd_x = self.gnn(embd_x, mask=mask)
        pooled = self.pool_net(embd_x, mask=mask)

        preds = {}
        if "jet_classification" in self.tasks:
            preds["jet_classification"] = self.jet_net(pooled)
        if "track_classification" in self.tasks:
            preds["track_classification"] = self.track_net(embd_x)

        return preds
