import torch
from torch import Tensor, cat, nn

from salt.models import FeaturewiseTransformation, InitNet, Pooling
from salt.stypes import BoolTensors, NestedTensors, Tensors
from salt.utils.tensor_utils import flatten_tensor_dict, maybe_flatten_tensors


class SaltModel(nn.Module):
    """Generic multi-modal, multi-task neural network.

    This model can implement a wide range of architectures such as
    [DL1](https://ftag.docs.cern.ch/algorithms/taggers/dips/),
    [DIPS](https://ftag.docs.cern.ch/algorithms/taggers/dl1/),
    [GN2](https://ftag.docs.cern.ch/algorithms/taggers/GN2/) and more.

    Parameters
    ----------
    init_nets : list[dict]
        Keyword arguments for one or more initialisation networks. Each
        initialisation network produces an initial input embedding for a
        single input type. See :class:`salt.models.InitNet`.
    tasks : nn.ModuleList
        Task heads to apply to the encoded/poolled features. See
        :class:`salt.models.TaskBase`.
    encoder : nn.Module | None, optional
        Main input encoder which takes the outputs of the init nets and
        produces per-constituent embeddings. If not provided, the model is
        effectively a DeepSets model. The default is ``None``.
    mask_decoder : nn.Module | None, optional
        Mask decoder which takes the encoder output and produces learned
        embeddings to represent object masks. The default is ``None``.
    pool_net : Pooling | None, optional
        Pooling network computing a global representation by aggregating
        over the constituents. If not provided, assume only global features
        are present. The default is ``None``.
    merge_dict : dict[str, list[str]] | None, optional
        Dictionary specifying which input types should be concatenated into
        a single stream (e.g., transformer input). The default is ``None``.
    featurewise_nets : list[dict] | None, optional
        Keyword arguments for featurewise transformation networks that
        perform per-feature scaling and biasing. The default is ``None``.
    """

    def __init__(
        self,
        init_nets: list[dict],
        tasks: nn.ModuleList,
        encoder: nn.Module | None = None,
        mask_decoder: nn.Module | None = None,
        pool_net: Pooling | None = None,
        merge_dict: dict[str, list[str]] | None = None,
        featurewise_nets: list[dict] | None = None,
    ):
        super().__init__()

        # init featurewise networks
        if featurewise_nets:
            self.init_featurewise(featurewise_nets, init_nets, encoder)

        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.tasks = tasks
        self.encoder = encoder
        self.mask_decoder = mask_decoder

        self.pool_net = pool_net
        self.merge_dict = merge_dict

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
        """Forward pass through the model.

        Parameters
        ----------
        inputs : Tensors
            Dict of input tensors for each modality of shape
            ``(batch_size, num_inputs, input_size)``.
        pad_masks : BoolTensors | None, optional
            Dict of input padding mask tensors for each modality of shape
            ``(batch_size, num_inputs)``. The default is ``None``.
        labels : NestedTensors | None, optional
            Nested dict of label tensors. Outer dict keyed by input modality,
            inner dict keyed by label variable. Each tensor has shape
            ``(batch_size, num_inputs)``. If ``None``, run inference without
            loss computation.

        Returns
        -------
        preds : NestedTensors
            Dict of model predictions for each task separated by input modality.
        loss : Tensors
            Dict of losses for each task aggregated over the batch.
        """
        # initial input projections
        xs = {}

        for init_net in self.init_nets:
            xs[init_net.input_name] = init_net(inputs)

        # handle edge features if present
        edge_x = xs.pop("EDGE", None)
        kwargs = {} if edge_x is None else {"edge_x": edge_x}

        # merge multiple streams if requested
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

        # encode
        if self.encoder:
            embed_xs = self.encoder(xs, pad_mask=pad_masks, inputs=inputs, **kwargs)
            if isinstance(embed_xs, tuple):
                embed_xs, pad_masks = embed_xs
            preds = {"embed_xs": embed_xs}
        else:
            preds = {"embed_xs": flatten_tensor_dict(xs)}

        preds, labels, loss = (
            self.mask_decoder(preds, self.tasks, pad_masks, labels)
            if self.mask_decoder
            else (preds, labels, {})
        )

        # apply featurewise transformation to global track embeddings if configured
        if hasattr(self, "featurewise_global") and self.featurewise_global:
            preds["embed_xs"] = self.featurewise_global(inputs, preds["embed_xs"])

        # pooling
        if self.pool_net:
            global_rep = self.pool_net(preds, pad_mask=pad_masks)
        else:
            global_rep = preds["embed_xs"]

        # add global features to global representation
        if (global_feats := inputs.get("global")) is not None:
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
    ) -> tuple[dict, dict]:
        """Run all configured tasks given encoded/poolled features.

        Parameters
        ----------
        preds : dict[str, Tensor]
            Predictions dict with at least ``"embed_xs"`` and ``"global_rep"``.
            Updated with task predictions keyed by input name and task name.
        masks : BoolTensors | None
            Padding masks passed to tasks requiring per-constituent masks.
        labels : NestedTensors | None, optional
            Nested dict of label tensors. The default is ``None``.

        Returns
        -------
        dict
            Updated predictions dict including per-task outputs.
        dict
            Loss dict mapping task name to loss tensor.
        """
        loss: dict = {}

        preds["embed_xs"] = maybe_flatten_tensors(preds["embed_xs"])

        for task in self.tasks:
            if task.input_name == task.global_object:
                task_preds, task_loss = task(preds["global_rep"], labels, None, context=None)
            elif self.mask_decoder and task.input_name == "objects":
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

    def init_featurewise(
        self,
        featurewise_nets: list[dict],
        init_nets: list[dict],
        encoder: nn.Module,
    ) -> None:
        """Initialize featurewise transformation modules.

        Depending on the ``layer`` key in each featurewise net config, attach
        :class:`FeaturewiseTransformation` modules to input init nets, encoder
        layers, or to the global representation.

        Parameters
        ----------
        featurewise_nets : list[dict]
            Config dicts for each featurewise transformation specifying at least
            ``"layer"``. Accepted values are ``"input"``, ``"encoder"``,
            or ``"global"``.
        init_nets : list[dict]
            Same list passed to ``__init__`` containing init net configurations.
            Modified in place when ``layer=="input"``.
        encoder : nn.Module
            Encoder module; must have ``num_layers`` and ``featurewise`` list
            attributes when ``layer=="encoder"``.

        Raises
        ------
        ValueError
            If featurewise transform is requested for the encoder, but no encoder is defined
            If the given type of layer for the featurewise net is not allowed
        """
        for featurewise_net in featurewise_nets:
            if featurewise_net.get("layer") == "input":
                for init_net in init_nets:
                    init_net["featurewise"] = FeaturewiseTransformation(**featurewise_net)
            elif featurewise_net.get("layer") == "encoder":
                if encoder:
                    for _layer in range(encoder.num_layers):
                        encoder.featurewise.append(FeaturewiseTransformation(**featurewise_net))
                else:
                    raise ValueError(
                        "Requested featurewise transforms for encoder, no encoder configured"
                    )
            elif featurewise_net.get("layer") == "global":
                self.featurewise_global = FeaturewiseTransformation(**featurewise_net)
            else:
                raise ValueError(
                    "Select either 'input', 'encoder' or 'global' layers for featurewise nets."
                )
