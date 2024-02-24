import torch
from torch import Tensor, cat, nn

from salt.models import InitNet, Pooling
from salt.stypes import BoolTensors, NestedTensors, Tensors
from salt.utils.tensor_utils import flatten_tensor_dict, maybe_flatten_tensors


class SaltModel(nn.Module):
    def __init__(
        self,
        init_nets: list[dict],
        tasks: nn.ModuleList,
        encoder: nn.Module = None,
        mask_decoder: nn.Module = None,
        pool_net: Pooling = None,
        num_register_tokens: int = 0,
        merge_dict: dict[str, list[str]] | None = None,
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
        mask_decoder : nn.Module
            Mask decoder, which takes the output of the encoder and produces a
            series of learned embeddings to represent object masks
        pool_net : nn.Module
            Pooling network which computes a global representation of the object
            by aggregating over the constituents. If not provided, assume that
            the only inputs are global features (i.e. no constituents).
        num_register_tokens : int
            Number of randomly initialised register tokens of the same length as
            any other input sequences after initialiser networks (e.g. tracks).
            See https://arxiv.org/abs/2309.16588.
        merge_dict : dict[str, list[str]] | None
            A dictionary that lets the salt concatenate all the input
            representations of the inputs in list[str] and act on them
            in following layers (e.g. transformer or tasks) as if they
            are coming from one input type
        """
        super().__init__()

        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.tasks = tasks
        self.encoder = encoder
        self.mask_decoder = mask_decoder

        self.pool_net = pool_net
        self.merge_dict = merge_dict
        self.num_register_tokens = num_register_tokens

        # init register tokens
        if self.num_register_tokens and self.encoder:
            self.registers = torch.nn.Parameter(
                torch.normal(
                    torch.zeros((self.num_register_tokens, self.encoder.embed_dim)), std=1e-4
                )
            )
            self.register_mask = torch.zeros(self.num_register_tokens, dtype=torch.bool)
            self.register_buffer("register_mask_buffer", self.register_mask)
        else:
            self.registers = None
            self.register_mask = None

        # checks for the global object only setup
        if self.pool_net is None:
            assert self.encoder is None, "pool_net must be set if encoder is set"
            assert len(self.init_nets) == 1, "pool_net must be set if more than one init_net is set"
            assert self.init_nets[0].input_name == self.init_nets[0].global_object

    def forward(
        self,
        inputs: Tensors,
        pad_masks: BoolTensors | None = None,
        labels: NestedTensors | None = None,
    ) -> tuple[NestedTensors, Tensors]:
        """Forward pass through the `SaltModel`.

        Don't call this method directy, instead use `__call__`.

        Parameters
        ----------
        inputs : Tensors
            Dict of input tensors for each modality. Each tensor is of shape
            `(batch_size, num_inputs, input_size)`.
        pad_masks : BoolTensors
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
        # initial input projections
        xs = {}
        for init_net in self.init_nets:
            xs[init_net.input_name] = init_net(inputs)

        if self.num_register_tokens:
            batch_size = xs[next(iter(xs))].shape[0]
            xs["REGISTERS"] = self.registers.expand(batch_size, -1, -1)
            if pad_masks:
                pad_masks["REGISTERS"] = self.register_mask_buffer.expand(batch_size, -1)

        # handle edge features if present
        edge_x = xs.pop("EDGE", None)
        kwargs = {} if edge_x is None else {"edge_x": edge_x}

        if isinstance(self.merge_dict, dict):
            for merge_name, merge_types in self.merge_dict.items():
                xs[merge_name] = cat([xs.pop(mt) for mt in merge_types], dim=1)
            if pad_masks is not None:
                for merge_name, merge_types in self.merge_dict.items():
                    pad_masks[merge_name] = cat([pad_masks.pop(mt) for mt in merge_types], dim=1)
            for merge_name, merge_types in self.merge_dict.items():
                if labels is not None:
                    labels[merge_name] = {}
                    for var in labels[merge_types[0]]:
                        labels[merge_name].update({
                            var: cat([labels[mt][var] for mt in merge_types], dim=1)
                        })

        # Generate embedding from encoder, or by concatenating the init net outputs
        if self.encoder:
            preds = {"embed_xs": self.encoder(xs, pad_mask=pad_masks, **kwargs)}
        else:
            preds = {"embed_xs": flatten_tensor_dict(xs)}

        if self.encoder:
            preds["embed_xs"] = self.encoder(xs, pad_mask=pad_masks, **kwargs)
        preds, labels, loss = (
            self.mask_decoder(preds, self.tasks, pad_masks, labels)
            if self.mask_decoder
            else (preds, labels, {})
        )

        # pooling
        if self.pool_net:
            global_rep = self.pool_net(preds, pad_mask=pad_masks)
        else:
            global_rep = preds["embed_xs"]

        # add global features to global representation
        if (global_feats := inputs.get("GLOBAL")) is not None:
            global_rep = torch.cat([global_rep, global_feats], dim=-1)
        preds["global_rep"] = global_rep

        # run tasks
        task_preds, task_loss = self.run_tasks(preds, pad_masks, labels)
        preds.update(task_preds)
        loss.update(task_loss)

        return preds, loss

    def run_tasks(
        self,
        preds: dict[str, Tensor],
        masks: BoolTensors | None,
        labels: NestedTensors | None = None,
    ):
        loss = {}

        preds["embed_xs"] = maybe_flatten_tensors(preds["embed_xs"])

        for task in self.tasks:
            if task.input_name == task.global_object:
                task_preds, task_loss = task(preds["global_rep"], labels, None, context=None)
            elif task.input_name == "objects":
                task_preds, task_loss = task(preds["objects"]["embed"], labels, masks, context=None)
            else:
                task_preds, task_loss = task(
                    preds["embed_xs"], labels, masks, context=preds["global_rep"]
                )
            if task.input_name not in preds:
                preds[task.input_name] = {}
            preds[task.input_name][task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
