from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from salt.models import Pooling
from salt.utils.tensor_utils import attach_context


class JetTagger(nn.Module):
    def __init__(
        self,
        init_nets: ModuleList,
        pool_net: Pooling,
        tasks: ModuleList,
        gnn: nn.Module = None,
    ):
        """Jet constituent tagger.

        # TODO: add option to pool separately for each task

        Parameters
        ----------
        init_nets : ModuleList
            Initialisation networks
        gnn : nn.Module
            Graph neural network
        pool_net : nn.Module
            Pooling network
        tasks : ModuleList
            Task networks
        """
        super().__init__()

        self.init_nets = init_nets
        self.pool_net = pool_net
        self.tasks = tasks
        self.gnn = gnn

        # ensure unique names
        assert len({init_net.name for init_net in self.init_nets}) == len(self.init_nets)
        assert len({task.name for task in self.tasks}) == len(self.tasks)

        # check init nets have the same output embedding size (unless an edge init net is present)
        sizes = {list(init_net.parameters())[-1].shape[0] for init_net in self.init_nets}
        names = {init_net.name for init_net in self.init_nets}
        assert (
            len(sizes) == 1
            or ("edge" in names and len(sizes) == 2)
            or ("electron" in names and len(sizes) == 2)
        )

    def forward(self, inputs: dict, mask: dict, labels: dict | None = None):
        # initial embeddings
        embed_x = {}
        edge_x = None
        for init_net in self.init_nets:
            if init_net.name != "edge":
                embed_x[init_net.name] = init_net(inputs)
            else:
                edge_x = init_net(inputs)

        if self.gnn:
            embed_x = self.gnn(embed_x, mask=mask, edge_x=edge_x)
            global_feats = inputs.get("global", None)
            if global_feats is not None:
                embed_x = attach_context(embed_x, global_feats)

        # pooling
        pooled = self.pool_net(embed_x, mask=mask)

        # run tasks
        preds, loss = self.tasks_forward(pooled, embed_x, mask, labels)

        return preds, loss

    def tasks_forward(
        self,
        pooled: Tensor,
        embed_x: Tensor,
        mask: Mapping,
        labels: dict | None = None,
    ):
        preds = {}
        loss = {}

        if isinstance(embed_x, dict):
            embed_x = torch.cat(list(embed_x.values()), dim=1)

        for task in self.tasks:
            if task.input_type == "jet":
                task_input = pooled
                task_mask = None
                context = None
            else:
                task_input = embed_x
                task_mask = mask
                context = pooled
            task_preds, task_loss = task(task_input, labels, task_mask, context=context)
            preds[task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
