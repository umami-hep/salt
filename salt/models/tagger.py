from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from salt.models import Pooling


class JetTagger(nn.Module):
    def __init__(
        self,
        init_nets: ModuleList,
        pool_net: Pooling,
        tasks: ModuleList,
        gnn: nn.Module = None,
    ):
        """Jet constituent tagger.

        # TODO: multiple inputs not compatible with aux tasks
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

        # check init nets have the same output embedding size
        sizes = {list(init_net.parameters())[-1].shape[0] for init_net in self.init_nets}
        assert len(sizes) == 1

    def forward(self, inputs: dict, mask: dict, labels: dict = None):
        # initial embeddings
        embed_x = {}
        for init_net in self.init_nets:
            embed_x[init_net.name] = init_net(inputs)

        # concatenate different input groups
        # TODO: use a flag as to which input type this is
        embed_x = torch.cat(list(embed_x.values()), dim=1)
        combined_mask = None
        if mask is not None:
            combined_mask = torch.cat(list(mask.values()), dim=1)

        # graph network
        if self.gnn:
            embed_x = self.gnn(embed_x, mask=combined_mask)

        # pooling
        pooled = self.pool_net(embed_x, mask=combined_mask)

        # run tasks
        preds, loss = self.tasks_forward(pooled, embed_x, mask, labels)

        return preds, loss

    def tasks_forward(
        self,
        pooled: Tensor,
        embed_x: Tensor,
        mask: Mapping,
        labels: dict = None,
    ):
        preds = {}
        loss = {}
        # TODO: move this login into the task class, including per element loss weighting
        for task in self.tasks:
            if "jet" in task.name:  # TODO: make robust with a flag to say what the task is on
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
