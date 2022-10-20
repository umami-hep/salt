import pytorch_lightning as pl
import torch
import torch.nn as nn

from salt.losses.classification import ClassificationLoss


class LightningTagger(pl.LightningModule):
    def __init__(self, net: nn.Module, tasks: dict):
        """Lightning jet tagger model.

        Parameters
        ----------
        net : nn.Module
            Network to use
        tasks : dict
            Dict of tasks and weights
        """

        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.model = net
        self.tasks = tasks

        self.losses = {}
        for task_name, opts in self.tasks.items():
            self.losses[task_name] = ClassificationLoss(
                task_name, weight=opts["weight"]
            )

    def forward(self, x, mask):
        """Forward pass through the model.

        Don't call this method directy.
        """
        return self.model(x, mask)

    def shared_step(self, batch, evaluation=False):
        """Function used to unpack the batch, run the forward pass, and compute
        losses, used by training, validation and test steps.

        Parameters
        ----------
        batch : _type_
            batch of inputs and labels

        Returns
        -------
        preds
            Model predictions
        labels
            True labels
        loss
            Reduced loss over the input batch
        """

        # separate graphs and true labels
        inputs, mask, labels = batch

        # get the model prediction
        preds = self(inputs, mask)

        if evaluation:
            return labels, preds, None

        # compute loss
        loss = {"loss": 0}
        for task in self.tasks:
            task_loss = self.losses[task](preds, labels)
            loss["loss"] += task_loss
            loss[f"{task}_loss"] = task_loss.detach()

        return preds, labels, loss

    def log_losses(self, loss, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True)
        for task in self.tasks:
            self.log(
                f"{stage}_{task}_loss",
                loss[f"{task}_loss"],
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        """Here you compute and return the training loss, compute additional
        metrics, and perform logging."""

        # foward pass
        preds, labels, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="train")

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Operates on a single batch of data from the validation set.

        In this step you'd might generate examples or calculate anything
        of interest like accuracy.
        """

        # foward pass
        preds, labels, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="val")

        # return loss (and maybe more stuff)
        return_dict = loss

        return return_dict

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

        # cosine warm restarts
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2)

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=1e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=1000,
            final_div_factor=1000,
            pct_start=0.1,
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]
