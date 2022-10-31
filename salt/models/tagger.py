import torch.nn as nn


class JetTagger(nn.Module):
    def __init__(
        self,
        tasks: dict,
        init_net: nn.Module = None,
        gnn: nn.Module = None,
        pool_net: nn.Module = None,
        jet_net: nn.Module = None,
        track_net: nn.Module = None,
    ):
        """Jet tagger model.

        Parameters
        ----------
        tasks : dict
            Dict of tasks to perform
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
        """
        super().__init__()

        self.init_net = init_net
        self.gnn = gnn
        self.jet_net = jet_net
        self.pool_net = pool_net
        self.track_net = track_net
        self.tasks = tasks

        if "jet_classification" in self.tasks and not jet_net:
            raise ValueError("Can't run jet classification without a jet net.")
        if "track_classification" in self.tasks and not track_net:
            raise ValueError("Can't run track classification without a track net.")

    def forward(self, x, mask):
        mask[..., 0] = False  # hack to make the MHA work
        embd_x = self.init_net(x)
        if self.gnn:
            embd_x = self.gnn(embd_x, mask=mask)
        pooled = self.pool_net(embd_x, mask=mask)

        preds = {}
        if self.jet_net and "jet_classification" in self.tasks:
            preds["jet_classification"] = self.jet_net(pooled)
        if self.track_net and "track_classification" in self.tasks:
            preds["track_classification"] = self.track_net(embd_x)

        return preds
