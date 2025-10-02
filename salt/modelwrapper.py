import math
import warnings
from collections.abc import Mapping
from typing import Any

import lightning
import torch
from torch import nn
from torch.optim import AdamW

from salt.models import InputNorm
from salt.models.transformer_v2 import change_attn_backends
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
    assert len({getattr(m, attr_name) for m in modules}) == len(
        modules
    ), f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"


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
        Loss reduction mode. Default is ``"wsum"``. Other option: ``"GLS"``.
    optimizer : str, optional
        Optimizer to use. Default is ``"AdamW"``. Other option: ``"lion"``.
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
        optimizer: str = "AdamW",
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

        assert len({t.net.output_size for t in self.model.init_nets if t.input_name != "EDGE"}) == 1

        # input normalizer
        assert norm_config is not None
        self.norm = InputNorm(**norm_config)

        allowed_loss_modes = ["wsum", "GLS"]
        assert loss_mode in allowed_loss_modes, f"Loss mode must be one of {allowed_loss_modes}"
        self.loss_mode = loss_mode
        if loss_mode == "GLS":
            assert all(
                task.weight == 1.0 for task in self.model.tasks
            ), "GLS does not utilise task weights - set all weights to 1"

        allowed_optimizers = ["lion", "AdamW"]
        assert optimizer in allowed_optimizers, (
            f"Optimizer {optimizer} not implemented, " f"please choose from {allowed_optimizers}"
        )
        self.optimizer = optimizer

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
        x = self.norm(inputs)
        return self.model(x, pad_masks, labels)

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
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        self.log(f"{stage}/loss", loss["loss"], **kwargs)
        for t, loss_value in loss.items():
            n = f"{stage}/{t}_loss" if "loss" not in t else f"{stage}/{t}"
            self.log(n, loss_value, **kwargs)

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
            type(self.model.encoder).__name__ == "TransformerV2"
            and self.trainer.precision == "32-true"
        ):
            change_attn_backends(self, backend="torch-math")
        inputs, pad_masks, _ = batch
        batch = (inputs, pad_masks, None)
        return self.shared_step(batch, evaluation=True)[0]

    def configure_optimizers(self):
        """Configure optimizer and learning-rate scheduler for Lightning.

        Returns
        -------
        tuple
            Tuple of the optimizer and the scheduler

        Raises
        ------
        ImportError
            When Lion should be used but isn't found
        """
        if self.optimizer == "lion":
            if _lion_available:
                opt = Lion(
                    self.parameters(),
                    lr=self.lrs_config["initial"],
                    weight_decay=self.lrs_config.get("weight_decay", 1e-5),
                )
            else:
                raise ImportError("Lion is not available! Please check the installation")
        else:  # AdamW or MuAdamW
            optimizer = MuAdamW if self.mup and _mup_available else AdamW
            opt = optimizer(
                self.parameters(),
                lr=self.lrs_config["initial"],
                weight_decay=self.lrs_config.get("weight_decay", 1e-5),
            )

        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
            last_epoch=int(self.lrs_config.get("last_epoch", -1)),
        )
        return [opt], [{"scheduler": sch, "interval": "step"}]

    @property
    def input_dims(self) -> dict[str, int]:
        """Return dimensionality of each input object after normalization.

        Returns
        -------
        dict[str, int]
            Mapping from object name to feature dimension.
        """
        return {k: len(v) for k, v in self.norm.variables.items()}
