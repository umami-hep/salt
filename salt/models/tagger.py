import torch
from torch import Tensor, nn

from salt.models import InitNet, Pooling
from salt.utils.typing import BoolTensors, NestedTensors, Tensors


class JetTagger(nn.Module):
    def __init__(
        self,
        init_nets: list[dict],
        pool_net: Pooling,
        tasks: nn.ModuleList,
        encoder: nn.Module = None,
    ):
        """A generic constituent-based tagger model.

        This model can do more than just object tagging, as it can can be
        configured with any number of input types and tasks.

        Parameters
        ----------
        init_nets : nn.ModuleList
            Keyword arguments for initialisation networks.
            See [`salt.models.InitNet`][salt.models.InitNet]
        encoder : nn.Module
            Input encoder model
        pool_net : nn.Module
            Pooling network which computes a global representation of the object
            by aggregating over the constituents
        tasks : nn.ModuleList
            Task networks, see [`salt.models.TaskBase`][salt.models.TaskBase]
        """
        super().__init__()

        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.pool_net = pool_net
        self.tasks = tasks
        self.encoder = encoder

    def forward(self, inputs: Tensors, mask: BoolTensors, labels: NestedTensors | None = None):
        # initial input embeddings
        embed_x = {}
        edge_x = None
        for init_net in self.init_nets:
            if init_net.input_name != "EDGE":
                embed_x[init_net.input_name] = init_net(inputs)
            else:
                edge_x = init_net(inputs)

        # input encoding
        if self.encoder:
            embed_x = self.encoder(embed_x, mask=mask, edge_x=edge_x)

        # pooling
        pooled = self.pool_net(embed_x, mask=mask)

        # add global features to pooled representation
        if (global_feats := inputs.get("GLOBAL")) is not None:
            pooled = torch.cat([pooled, global_feats], dim=-1)

        # run tasks
        preds, loss = self.tasks_forward(pooled, embed_x, mask, labels)

        return preds, loss

    def tasks_forward(
        self,
        pooled: Tensor,
        embed_x: Tensor,
        mask: BoolTensors,
        labels: NestedTensors | None = None,
    ):
        preds: NestedTensors = {}
        loss = {}

        if isinstance(embed_x, dict):
            embed_x = torch.cat(list(embed_x.values()), dim=1)

        for task in self.tasks:
            # TODO: put into task
            if task.input_name == task.global_object:
                task_input = pooled
                task_mask = None
                context = None
            else:
                task_input = embed_x
                task_mask = mask
                context = pooled
            task_preds, task_loss = task(task_input, labels, task_mask, context=context)
            if task.input_name not in preds:
                preds[task.input_name] = {}
            preds[task.input_name][task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
