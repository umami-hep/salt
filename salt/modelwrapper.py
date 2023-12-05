import warnings
from collections.abc import Mapping

import lightning as L
import torch
from torch import nn

from salt.models import InputNorm


def check_unique(modules: nn.ModuleList, attr_name: str) -> None:
    assert len({getattr(m, attr_name) for m in modules}) == len(
        modules
    ), f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"


class ModelWrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lrs_config: Mapping[str, float],
        global_object: str,
        norm_config: dict | None = None,
        name: str = "salt",
        muP_config: dict | None = None,
    ):
        """A wrapper class for any model implemented in Salt.

        This wrapper class allows is as generic as possible. It wraps
        [`SaltModel`][salt.models.SaltModel], but could also be used to
        wrap any other model if you want to do train something that doesn't
        fit into the [`SaltModel`][salt.models.SaltModel] architecture.

        This class is responsible for containing things that are common to all
        salt models. These are:

        - A generic forward pass, including input normalisation
        - Training, validation and test steps, which include logging
        - Some sanity checks on the model configuration

        Parameters
        ----------
        model : nn.Module
            Model to be wrapped
        lrs_config: Mapping
            LRS config which has to be set manually for now
            https://github.com/omni-us/jsonargparse/issues/170#issuecomment-1288167674
        global_object : str
            Name of the global input object, as opposed to the constituent-level
            inputs. This argument is set automatically by the framework.
        norm_config : dict, optional
            Keyword arguments for [`salt.models.InputNorm`][salt.models.InputNorm].
        name: str, optional
            Name of the model, used for logging and inference output names
        muP_config: dict, optional
            The muP configuration.
        """
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.save_hyperparameters(logger=False)

        self.model = model
        self.lrs_config = lrs_config
        self.global_object = global_object
        self.name = name
        self.muP = muP_config if muP_config else {}
        # Here the model should pick it up
        if self.muP:
            from salt.utils.muP_utils.configuration_muP import instantiate_mup

            load_path = None
            if "shape_path" in self.muP:
                load_path = self.muP["shape_path"]
            instantiate_mup(model, load_path)

        # all tasks should inherit the global object type
        self.model.global_object = self.global_object
        for task in self.model.tasks:
            task.global_object = self.global_object

        # ensure unique names for init_nets and tasks
        check_unique(self.model.init_nets, "input_name")
        check_unique(self.model.tasks, "name")

        # check that the model has the same output size for all init nets
        assert len({t.net.output_size for t in self.model.init_nets if t.input_name != "EDGE"}) == 1

        # create input normaliser
        assert norm_config is not None
        self.norm = InputNorm(**norm_config)

    def forward(self, inputs, pad_masks=None, labels=None):
        """Generic forward pass through any salt-compatible model.

        This function performs input normalisation and then calls the `self.model`'s
        forward pass. Don't call this method directy, instead use `__call__`.

        Parameters
        ----------
        inputs
            Any generic input to the model.
        pad_masks
            Input padding masks.
        labels
            Training targets. If not specified, assume we are running model inference
            (i.e. no loss computation).

        Returns
        -------
        Whatever is returned by `self.model`'s forward pass.
        """
        x = self.norm(inputs)
        return self.model(x, pad_masks, labels)

    def shared_step(self, batch, evaluation=False):
        """Function used to unpack the batch, run the forward pass, and compute
        losses, used by training, validation and test steps.

        Parameters
        ----------
        batch : tuple
            A single batch of inputs, pad_masks and labels
        evaluation : bool
            If true, don't compute the losses and return early

        Returns
        -------
        preds
            Model predictions
        labels
            True labels
        loss
            Reduced loss over the input batch
        """
        # unpack the batch
        inputs, pad_masks, labels = batch

        # forward pass through model
        preds, loss = self(inputs, pad_masks, labels)

        if evaluation:
            return preds, labels, pad_masks, None

        # compute total loss
        loss["loss"] = sum(subloss for subloss in loss.values())

        return preds, labels, pad_masks, loss

    def log_losses(self, loss, stage):
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        self.log(f"{stage}_loss", loss["loss"], **kwargs)
        for t, loss_value in loss.items():
            n = f"{stage}_{t}_loss" if "loss" not in t else f"{stage}_{t}"
            self.log(n, loss_value, **kwargs)

    def training_step(self, batch, batch_idx):
        # foward pass
        preds, labels, _, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="train")

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        # foward pass
        _, _, _, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="val")

        # return loss (and maybe more stuff)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, pad_masks, labels = batch
        batch = (inputs, pad_masks, None)
        return self.shared_step(batch, evaluation=True)[0]

    def configure_optimizers(self):
        if self.muP:
            from mup import MuAdamW as AdamW
        else:
            from torch.optim import AdamW
        opt = AdamW(
            self.parameters(),
            lr=self.lrs_config["initial"],
            weight_decay=self.lrs_config.get("weight_decay", 1e-5),
        )

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
            last_epoch=int(self.lrs_config.get("last_epoch", -1)),
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]

    @property
    def input_dims(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.norm.variables.items()}
