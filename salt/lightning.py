import socket
import subprocess

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
            Dict of tasks and loss weights for each task
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

        # separate graphs and true labels
        inputs, mask, labels = batch

        # get the model prediction
        preds = self(inputs, mask)

        if evaluation:
            return preds, labels, mask, None

        # compute loss
        loss = {"loss": 0}
        for task in self.tasks:
            task_loss = self.losses[task](preds, labels)
            loss["loss"] += task_loss
            loss[f"{task}_loss"] = task_loss.detach()

        return preds, labels, mask, loss

    def log_losses(self, loss, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True)
        for task in self.tasks:
            self.log(
                f"{stage}_{task}_loss",
                loss[f"{task}_loss"],
                sync_dist=True,
            )

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

    def on_train_start(self):
        if not self.trainer.is_global_zero:
            return

        if not self.logger:
            return

        exp = self.logger.experiment
        trainer = self.trainer
        train_loader = trainer.datamodule.train_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        train_dset = train_loader.dataset
        val_dset = val_loader.dataset

        # inputs
        exp.log_parameter("num_jets_train", len(train_dset))
        exp.log_parameter("num_jets_val", len(val_dset))
        exp.log_parameter("batch_size", train_loader.batch_size)
        # TODO: log input variables from datasets

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        exp.log_parameter("trainable_params", num_params)

        # resources
        exp.log_parameter("num_gpus", trainer.num_devices)
        exp.log_parameter("gpu_ids", trainer.device_ids)
        exp.log_parameter("num_workers", train_loader.num_workers)

        # version info
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        exp.log_parameter("git_hash", git_hash.decode("ascii").strip())
        exp.log_parameter("timestamp", trainer.timestamp)
        exp.log_parameter("torch_version", torch.__version__)
        exp.log_parameter("lightning_version", pl.__version__)
        exp.log_parameter("cuda_version", torch.version.cuda)
        exp.log_parameter("hostname", socket.gethostname())
