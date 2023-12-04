import torch
from torch import Tensor, nn

from salt.models import InitNet, Pooling
from salt.typing import BoolTensors, NestedTensors, Tensors


class SaltModel(nn.Module):
    def __init__(
        self,
        init_nets: list[dict],
        tasks: nn.ModuleList,
        encoder: nn.Module = None,
        pool_net: Pooling = None,
    ):
        """A generic multi-modal, multi-task neural network.

        This model can be used to implement a wide range of models, including
        [DL1](https://ftag.docs.cern.ch/algorithms/taggers/dips/),
        [DIPS](https://ftag.docs.cern.ch/algorithms/taggers/dl1/),
        [GN2](https://ftag.docs.cern.ch/algorithms/taggers/GN2/)
        and more.

        Parameters
        ----------
        init_nets : nn.ModuleList
            Keyword arguments for one or more initialisation networks.
            See [`salt.models.InitNet`][salt.models.InitNet].
            Each initialisation network produces an initial input embedding for
            a single input type.
        tasks : nn.ModuleList
            Task heads, see [`salt.models.TaskBase`][salt.models.TaskBase].
            These can be used to implement object tagging, vertexing, regression,
            classification, etc.
        encoder : nn.Module
            Main input encoder, which takes the output of the initialisation
            networks and produces a single embedding for each constituent.
            If not provided this model is essentially a DeepSets model.
        pool_net : nn.Module
            Pooling network which computes a global representation of the object
            by aggregating over the constituents. If not provided, assume that
            the only inputs are global features (i.e. no constituents).
        """
        super().__init__()

        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.tasks = tasks
        self.encoder = encoder
        self.pool_net = pool_net

        # checks for the global object only setup
        if self.pool_net is None:
            assert self.encoder is None, "pool_net must be set if encoder is set"
            assert len(self.init_nets) == 1, "pool_net must be set if more than one init_net is set"
            assert self.init_nets[0].input_name == self.init_nets[0].global_object

    def forward(
        self, inputs: Tensors, masks: BoolTensors | None = None, labels: NestedTensors | None = None
    ) -> tuple[NestedTensors, Tensors]:
        """Forward pass through the `SaltModel`.

        Don't call this method directy, instead use `__call__`.

        Parameters
        ----------
        inputs : Tensors
            Dict of input tensors for each modality. Each tensor is of shape
            `(batch_size, num_inputs, input_size)`.
        masks : BoolTensors
            Dict of input padding mask tensors for each modality. Each tensor is of
            shape `(batch_size, num_inputs)`.
        labels : Tensors, optional
            Nested dict of label tensors. The outer dict is keyed by input modality,
            the inner dict is keyed by label variable name. Each tensor is of shape
            `(batch_size, num_inputs)`. If not specified, assume we are running model
            inference (i.e. no loss computation).

        Returns
        -------
        preds : NestedTensors
            Dict of model predictions for each task, separated by input modality.
            Tensors have varying shapes depending on the task.
        loss : Tensors
            Dict of losses for each task, aggregated over the batch.
        """
        # initial input embeddings
        initial_embeddings = {}
        edge_x = None
        for init_net in self.init_nets:
            if init_net.input_name != "EDGE":
                initial_embeddings[init_net.input_name] = init_net(inputs)
            else:
                edge_x = init_net(inputs)

        # input encoding
        combined_embeddings = initial_embeddings
        if self.encoder:
            combined_embeddings = self.encoder(initial_embeddings, mask=masks, edge_x=edge_x)

        # pooling
        if self.pool_net:
            global_rep = self.pool_net(combined_embeddings, mask=masks)
        else:
            global_rep = initial_embeddings[self.global_object]

        # add global features to global_rep representation
        if (global_feats := inputs.get("GLOBAL")) is not None:
            global_rep = torch.cat([global_rep, global_feats], dim=-1)

        # run tasks
        preds, loss = self.run_tasks(global_rep, combined_embeddings, masks, labels)

        return preds, loss

    def run_tasks(
        self,
        global_rep: Tensor,
        embed_x: Tensor,
        mask: BoolTensors,
        labels: NestedTensors | None = None,
    ):
        preds: NestedTensors = {}
        loss = {}

        if isinstance(embed_x, dict):
            embed_x = torch.cat(list(embed_x.values()), dim=1)

        for task in self.tasks:
            if task.input_name == task.global_object:
                task_preds, task_loss = task(global_rep, labels, None, context=None)
            else:
                task_preds, task_loss = task(embed_x, labels, mask, context=global_rep)
            if task.input_name not in preds:
                preds[task.input_name] = {}
            preds[task.input_name][task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
