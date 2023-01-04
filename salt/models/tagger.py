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
        init_net : Dense
            Initialisation network
        gnn : nn.Module
            Graph neural network
        pool_net : nn.Module
            Pooling network
        jet_net : Task
            Jet classification task
        track_net : Task
            Track classification task
        """
        super().__init__()

        self.init_nets = init_nets
        self.pool_net = pool_net
        self.tasks = tasks
        self.gnn = gnn

        # check init nets have the same output embedding size
        sizes = {list(init_net.parameters())[-1].shape[0] for init_net in self.init_nets}
        assert len(sizes) == 1

    def forward(self, inputs: dict, mask: dict, labels: dict):
        # initial embeddings
        embed_x = {}
        for i, init_net in enumerate(self.init_nets):
            embed_x[init_net.name] = init_net(inputs)

        # concatenate different things
        # TODO: check if it helps to add a flag as to which input type this is
        all_x = torch.cat(list(embed_x.values()), dim=1)
        if mask is not None:
            mask = torch.cat(list(mask.values()), dim=1)

        # graph network
        if self.gnn:
            all_x = self.gnn(all_x, mask=mask)

        # pooling
        pooled = self.pool_net(all_x, mask=mask)

        # concat global rep
        pooled_repeat = torch.repeat_interleave(pooled[:, None, :], all_x.shape[1], dim=1)
        all_x = torch.cat([pooled_repeat, all_x], dim=-1)

        # run tasks
        preds, loss = self.tasks_forward(pooled, all_x, mask, labels)

        return preds, loss

    def tasks_forward(self, pooled: Tensor, embed_x: Tensor, mask: Tensor, labels: dict):
        preds = {}
        loss = {}
        for task in self.tasks:
            if "jet" in task.name:  # TODO: make robust with a flag to say what the task is on
                task_input = pooled
                task_mask = None
            else:
                task_input = embed_x
                task_mask = mask
            task_labels = labels[task.name] if labels is not None else None
            task_preds, task_loss = task(task_input, task_labels, task_mask)
            preds[task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
