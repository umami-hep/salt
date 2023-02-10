import warnings
from typing import Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LightningTagger(pl.LightningModule):
    def __init__(self, model: nn.Module, lrs_config: Mapping):
        """Lightning jet tagger model.

        Parameters
        ----------
        model : nn.Module
            Network and loss function defintions
        lrs_config: Mapping
            LRS config which has to be set manually for now
            https://github.com/omni-us/jsonargparse/issues/170#issuecomment-1288167674
        """

        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.save_hyperparameters(logger=False)

        self.model = model
        self.lrs_config = lrs_config

        self.in_dims = [list(net.parameters())[0].shape[1] for net in self.model.init_nets]

    def forward(self, x, mask, labels=None):
        """Forward pass through the model.

        Don't call this method directy.
        """
        return self.model(x, mask, labels)

    def shared_step(self, batch, evaluation=False):
        """Function used to unpack the batch, run the forward pass, and compute
        losses, used by training, validation and test steps.

        Parameters
        ----------
        batch : tuple
            A single batch of inputs, masks and labels
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
        inputs, mask, labels = batch

        # forward pass through model
        preds, loss = self(inputs, mask, labels)

        if evaluation:
            return preds, labels, mask, None

        # compute total loss
        loss["loss"] = sum(subloss for subloss in loss.values())

        return preds, labels, mask, loss

    def log_losses(self, loss, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True)
        for t, l in loss.items():
            n = f"{stage}_{t}_loss" if "loss" not in t else f"{stage}_{t}"
            self.log(n, l, sync_dist=True)

    def training_step(self, batch, batch_idx):
        # foward pass
        preds, labels, _, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="train")

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        # foward pass
        preds, labels, _, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="val")

        # return loss (and maybe more stuff)
        return_dict = loss

        return return_dict

    def test_step(self, batch, batch_idx):
        preds, _, mask, _ = self.shared_step(batch, evaluation=True)
        return preds, mask

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lrs_config["initial"], weight_decay=1e-5
        )

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["max"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]
