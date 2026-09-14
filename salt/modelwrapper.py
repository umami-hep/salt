import math
import warnings
from collections.abc import Mapping
from typing import Any

import lightning
import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from salt.models import EdgeConstructor, InputNorm
from salt.models.transformer import change_attn_backends
from salt.optim import HybridMuonAdamW
from salt.utils.muP_utils.configuration_muP import instantiate_mup

try:
    from lion_pytorch import Lion

    _lion_available = True

except ImportError:
    _lion_available = False

try:
    from mup import MuAdamW

    _mup_available = True

except ImportError:
    _mup_available = False


def check_unique(modules: nn.ModuleList, attr_name: str) -> None:
    """Check that a specific attribute is unique across all modules.

    Parameters
    ----------
    modules : nn.ModuleList
        List of PyTorch modules.
    attr_name : str
        Name of the attribute to check.
    """
    assert len({getattr(m, attr_name) for m in modules}) == len(modules), (
        f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"
    )


class ModelWrapper(lightning.LightningModule):
    """A generic wrapper class for Salt-compatible models.

    This class wraps [`SaltModel`][salt.models.SaltModel], but can also be used to
    wrap arbitrary PyTorch models for training with Lightning. It handles:

    - A generic forward pass including input normalization
    - Training, validation, and test steps with logging
    - Sanity checks on the model configuration
    - Optimizer and scheduler setup

    Parameters
    ----------
    model : nn.Module
        Model to be wrapped.
    lrs_config : Mapping[str, float]
        Learning rate schedule configuration.
    global_object : str
        Name of the global input object, as opposed to constituent-level inputs.
    norm_config : dict | None, optional
        Keyword arguments for [`salt.models.InputNorm`][salt.models.InputNorm].
    name : str, optional
        Name of the model, used for logging and inference outputs. Default is ``"salt"``.
    mup_config : dict | None, optional
        Configuration for mup scaling. Default is ``None``.
    loss_mode : str, optional
        Loss reduction mode. Default is ``"wsum"``. Other options: ``"GLS"``, ``"DWA"``.
    dwa_temperature : float, optional
        Softmax temperature ``T`` for ``loss_mode="DWA"``. Larger values flatten the
        weights towards uniform. Default is ``2.0``, as in the original paper.
    optimizer : str, optional
        Optimizer to use. Default is ``"AdamW"``. Other options: ``"lion"``, ``"HybridMuonAdamW"``.
    optimizer_kwargs : dict | None, optional
        Extra keyword arguments forwarded to the optimizer constructor, e.g.
        ``{"fused": true}`` to enable the fused CUDA AdamW kernel. Default is ``None``.
    edge_constructors : list[dict] | None, optional
        Edge constructors configuration. By default None
    """

    def __init__(
        self,
        model: nn.Module,
        lrs_config: Mapping[str, float],
        global_object: str,
        norm_config: dict | None = None,
        name: str = "salt",
        mup_config: dict | None = None,
        loss_mode: str = "wsum",
        dwa_temperature: float = 2.0,
        optimizer: str = "AdamW",
        optimizer_kwargs: dict | None = None,
        edge_constructors: list[dict] | None = None,
    ):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.save_hyperparameters(logger=False)

        self.model = model
        self.lrs_config = lrs_config
        self.global_object = global_object
        self.name = name
        self.mup = mup_config or {}
        self.last_val_batch_outs = None

        # MuP initialization if configured
        if self.mup:
            load_path = self.mup.get("shape_path")
            instantiate_mup(model, load_path)

        # propagate metadata to tasks
        self.model.global_object = self.global_object
        for task in self.model.tasks:
            task.global_object = self.global_object
            task.model_name = self.name

        # sanity checks
        check_unique(self.model.init_nets, "input_name")
        check_unique(self.model.tasks, "name")

        assert len({t.net.output_size for t in self.model.init_nets}) == 1

        # initialize edge constructors
        self.edge_constructor: EdgeConstructor | None
        if edge_constructors:
            assert len(edge_constructors) <= 1, (
                "At most one edge constructor is supported at this moment"
            )
            self.edge_constructor = EdgeConstructor(**edge_constructors[0])
        else:
            self.edge_constructor = None

        # input normalizer
        assert norm_config is not None
        self.norm = InputNorm(**norm_config)

        allowed_loss_modes = ["wsum", "GLS", "DWA"]
        assert loss_mode in allowed_loss_modes, f"Loss mode must be one of {allowed_loss_modes}"
        self.loss_mode = loss_mode
        if loss_mode in {"GLS", "DWA"}:
            assert all(task.weight == 1.0 for task in self.model.tasks), (
                f"{loss_mode} does not utilise task weights - set all weights to 1"
            )

        assert dwa_temperature > 0.0, "dwa_temperature must be positive"
        self.dwa_temperature = dwa_temperature
        # per-task mean losses of the last two epochs, and the weights derived from them
        self._dwa_prev: dict[str, float] = {}
        self._dwa_prev2: dict[str, float] = {}
        self._dwa_weights: dict[str, float] = {}
        self._dwa_reset_accumulator()

        # Set the optimizer
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

    def _get_optimizer_class(self) -> type[Optimizer]:
        """
        Resolve and validate the optimizer class.

        Returns
        -------
        Type[Optimizer]
            The optimizer class to instantiate.

        Raises
        ------
        ImportError
            If a requested optimizer backend is not available.
        ValueError
            If an unsupported optimizer name is provided.
        """
        opt_name = self.optimizer

        if opt_name == "lion":
            if not _lion_available:
                raise ImportError(
                    "Lion optimizer requested but not available. "
                    "Check installation of lion-pytorch."
                )
            return Lion

        if opt_name == "AdamW":
            return MuAdamW if (self.mup and _mup_available) else AdamW

        if opt_name == "HybridMuonAdamW":
            return HybridMuonAdamW

        raise ValueError(f"Optimizer '{opt_name}' is not supported.")

    def total_loss(self, loss: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the final loss given per-task losses.

        Parameters
        ----------
        loss : dict[str, torch.Tensor]
            Dictionary of per-task losses.

        Returns
        -------
        Tensor
            Final reduced loss.
        """
        if self.loss_mode == "DWA":
            # the first two epochs have no loss ratio yet, so weights stay uniform
            if not self._dwa_weights:
                return torch.stack(list(loss.values())).sum()
            return torch.stack([self._dwa_weights[k] * v for k, v in loss.items()]).sum()
        if self.loss_mode == "GLS":
            loss_prod = math.prod(subloss for subloss in loss.values())
            return torch.pow(loss_prod, 1.0 / len(loss))
        return sum(subloss for subloss in loss.values())

    def forward(
        self,
        inputs: torch.Tensor | dict,
        pad_masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Forward pass through the wrapped model with input normalization.

        Parameters
        ----------
        inputs : torch.Tensor | dict
            Model inputs.
        pad_masks : torch.Tensor | None, optional
            Padding masks for variable-length inputs.
        labels : torch.Tensor | None, optional
            Training targets. If not provided, inference mode is assumed.

        Returns
        -------
        Any
            Whatever is returned by the wrapped model's forward pass.
        """
        if self.edge_constructor:
            inputs = self.edge_constructor(inputs)
        x = self.norm(inputs)
        return self.model(x, pad_masks, labels)

    def _dwa_reset_accumulator(self) -> None:
        """Clear the running per-task loss sums for the current epoch."""
        self._dwa_epoch_sums: dict[str, torch.Tensor] = {}
        self._dwa_epoch_count: int = 0

    def _dwa_accumulate(self, loss: dict[str, torch.Tensor]) -> None:
        """Add this step's per-task losses to the running epoch sums.

        Parameters
        ----------
        loss : dict[str, torch.Tensor]
            Per-task losses, plus the reduced ``"loss"`` entry which is skipped.
        """
        for name, value in loss.items():
            if name == "loss":
                continue
            detached = value.detach()
            current = self._dwa_epoch_sums.get(name)
            self._dwa_epoch_sums[name] = detached.clone() if current is None else current + detached
        self._dwa_epoch_count += 1

    def _dwa_update_weights(self) -> None:
        """Roll the loss history forward and recompute the DWA task weights.

        Weights follow Liu et al., "End-to-End Multi-Task Learning with Attention":
        ``w_i = N * softmax(r_i / T)`` with ``r_i = L_i(t-1) / L_i(t-2)``, so a task
        whose loss is falling slowly, or rising, receives more weight. They sum to
        ``N``, matching the scale of a plain sum. As in the reference, the ratios are
        not bounded, so a task whose loss jumps by orders of magnitude between epochs
        can saturate the softmax and take most of the weight.
        """
        if not self._dwa_epoch_count:
            return

        names = sorted(self._dwa_epoch_sums)
        means = torch.stack([self._dwa_epoch_sums[k] for k in names]) / self._dwa_epoch_count
        # average across ranks so every process derives identical weights
        if self.trainer.world_size > 1:
            gathered = self.all_gather(means)
            assert isinstance(gathered, torch.Tensor)
            means = gathered.mean(dim=0)
        self._dwa_prev2 = self._dwa_prev
        self._dwa_prev = {k: float(v) for k, v in zip(names, means, strict=True)}
        self._dwa_reset_accumulator()

        if not self._dwa_prev2:
            return

        assert set(self._dwa_prev) == set(self._dwa_prev2), "DWA task names changed between epochs"
        assert all(v > 0.0 for v in self._dwa_prev2.values()), (
            "DWA needs positive task losses to form the loss ratio"
        )
        ratios = torch.tensor(
            [self._dwa_prev[k] / self._dwa_prev2[k] for k in names], dtype=torch.float32
        )
        weights = len(names) * torch.softmax(ratios / self.dwa_temperature, dim=0)
        self._dwa_weights = {k: float(weights[i]) for i, k in enumerate(names)}

    def on_train_epoch_end(self) -> None:
        """Recompute the DWA weights from the epoch that just finished."""
        if self.loss_mode == "DWA":
            self._dwa_update_weights()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the DWA loss history so a resumed run keeps its weights.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            Checkpoint dict that Lightning is about to write.
        """
        if self.loss_mode == "DWA":
            checkpoint["dwa_state"] = {
                "prev": self._dwa_prev,
                "prev2": self._dwa_prev2,
                "weights": self._dwa_weights,
            }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Normalise ``torch.compile`` key prefixes in the saved state_dict.

        Training with ``--compile`` wraps ``self.model`` in
        ``torch._dynamo.OptimizedModule``, which prepends ``_orig_mod.`` to every
        parameter key when the checkpoint is saved. Loading such a checkpoint
        into a non-compiled module fails with a strict state_dict mismatch.
        Mirrors the offline :mod:`salt.utils.repair_ckpt` logic, applied
        in-memory before Lightning calls ``load_state_dict``.
        """
        if "dwa_state" in checkpoint:
            dwa_state = checkpoint["dwa_state"]
            self._dwa_prev = dwa_state["prev"]
            self._dwa_prev2 = dwa_state["prev2"]
            self._dwa_weights = dwa_state["weights"]

        state_dict = checkpoint.get("state_dict")
        if not state_dict or not any("_orig_mod." in k for k in state_dict):
            return
        checkpoint["state_dict"] = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def shared_step(self, batch: tuple, evaluation: bool = False):
        """Unpack a batch, run forward, and compute loss.

        Parameters
        ----------
        batch : tuple
            A batch of ``(inputs, pad_masks, labels)``.
        evaluation : bool, optional
            If True, skip loss computation.

        Returns
        -------
        preds
            Model predictions.
        labels
            Ground-truth labels.
        pad_masks
            Padding masks.
        loss : dict | None
            Dictionary of per-task and total loss values, or ``None`` in evaluation mode.
        """
        inputs, pad_masks, labels = batch
        preds, loss = self(inputs, pad_masks, labels)

        if evaluation:
            return preds, labels, pad_masks, None

        loss["loss"] = self.total_loss(loss)
        return preds, labels, pad_masks, loss

    def log_losses(self, loss: dict[str, torch.Tensor], stage: str) -> None:
        """Log per-task and total losses.

        Parameters
        ----------
        loss : dict[str, torch.Tensor]
            Dictionary of losses.
        stage : str
            Training stage, e.g. ``"train"`` or ``"val"``.
        """
        kwargs: dict[str, Any] = {"sync_dist": len(self.trainer.device_ids) > 1}
        self.log(f"{stage}/loss", loss["loss"], **kwargs)
        for t, loss_value in loss.items():
            n = f"{stage}/{t}_loss" if "loss" not in t else f"{stage}/{t}"
            self.log(n, loss_value, **kwargs)
        # effective per-task gradient weights, empty unless running under DWA
        for t_name, weight in self._dwa_weights.items():
            self.log(f"{stage}/{t_name}_dwa_weight", weight, **kwargs)

    def training_step(self, batch: tuple) -> dict[str, Any]:
        """Lightning training step.

        Parameters
        ----------
        batch : tuple
            Batch that is to be trained

        Returns
        -------
        dict[str, Any]
            Dict with the losses and the outputs

        Raises
        ------
        RuntimeError
            If the loss is NaN
        """
        preds, labels, pad_masks, loss = self.shared_step(batch)
        if loss["loss"].isnan():
            raise RuntimeError(
                "Loss is NaN - check dataset for NaNs or infs. "
                "See 'docs/training.md - NaNs' for more info."
            )
        if self.loss_mode == "DWA":
            self._dwa_accumulate(loss)
        self.log_losses(loss, stage="train")
        outputs = {"preds": preds, "labels": labels, "pad_masks": pad_masks}
        return {**loss, "outputs": outputs}

    def validation_step(self, batch: tuple):
        """Lightning validation step.

        Parameters
        ----------
        batch : tuple
            Batch that is to be validated

        Returns
        -------
        dict[str, Any]
            Dict with the losses and the outputs
        """
        preds, labels, pad_masks, loss = self.shared_step(batch)
        self.log_losses(loss, stage="val")
        outputs = {"preds": preds, "labels": labels, "pad_masks": pad_masks}
        return {**loss, "outputs": outputs}

    def test_step(self, batch: tuple):
        """Lightning test step.

        Parameters
        ----------
        batch : tuple
            Batch that is to be tested

        Returns
        -------
        tuple
            Evaluation results
        """
        if (
            type(self.model.encoder).__name__ == "Transformer"
            and self.trainer.precision == "32-true"
        ):
            change_attn_backends(self, backend="torch-math")
        inputs, pad_masks, labels = batch
        batch = (inputs, pad_masks, labels)
        return self.shared_step(batch, evaluation=True)[0]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        """Configure optimizer and learning-rate scheduler for Lightning.

        This method resolves the optimizer class based on the configuration,
        instantiates it with shared hyperparameters, and attaches a
        OneCycleLR scheduler.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            A tuple containing:
                - List with a single instantiated optimizer.
                - List with a scheduler configuration dictionary compatible
                with Lightning (interval='step').
        """
        optimizer_class = self._get_optimizer_class()

        optimizer_kwargs = {
            "lr": self.lrs_config["initial"],
            "weight_decay": self.lrs_config.get("weight_decay", 1e-5),
            **self.optimizer_kwargs,
        }

        # HybridMuonAdamW wants names for good parameter selection
        if optimizer_class is HybridMuonAdamW:
            optimizer = optimizer_class(self.named_parameters(), **optimizer_kwargs)

        else:
            optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # IMPORTANT: OneCycleLR takes the optimizer as the FIRST positional argument.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
            last_epoch=int(self.lrs_config.get("last_epoch", -1)),
            cycle_momentum=optimizer_class is not HybridMuonAdamW,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @property
    def input_dims(self) -> dict[str, int]:
        """Return dimensionality of each input object after normalization.

        Returns
        -------
        dict[str, int]
            Mapping from object name to feature dimension.
        """
        return {k: len(v) for k, v in self.norm.variables.items()}
