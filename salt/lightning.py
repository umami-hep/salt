import pytorch_lightning as pl
import torch.nn as nn

from salt.losses.jet import JetClassificationLoss


class LightningTagger(pl.LightningModule):
    def __init__(self, net: nn.Module, loss_weights: dict):
        """Lightning jet tagger model.

        Parameters
        ----------
        net : nn.Module
            Network to use
        loss_weights : dict
            Weights for the different losses
        """

        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.model = net
        self.loss_weights = loss_weights
        self.jet_loss = JetClassificationLoss()

    def forward(self, x):
        """Forward pass through the model.

        Don't call this method directy.
        """
        return self.model(x)

    def shared_step(self, batch, evaluation=False):
        """Function used to unpack the batch, run the forward pass, and compute
        losses, used by training, validation and test steps.

        Parameters
        ----------
        batch : _type_
            batch of inputs and labels

        Returns
        -------
        y_true
            True labels
        y_preds
            Model predictions
        loss
            Reduced loss over the input batch
        """

        # separate graphs and true labels
        x, y_true = batch

        # get the model prediction
        y_pred = self(x)

        if evaluation:
            return y_true, y_pred, None

        # compute classification_loss
        loss = {"loss": 0}
        jet_loss = self.jet_loss(y_pred, y_true)
        loss["loss"] += jet_loss
        # loss["jet_loss"] = jet_loss.detach()

        return y_true, y_pred, loss

    def log_losses(self, loss, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True)
        # for loss_type in self.config["logging_losses"]:
        #    self.log(
        #        f"{stage}_{loss_type}", loss[loss_type], sync_dist=True, batch_size=1
        #    )

    def training_step(self, batch, batch_idx):
        """Here you compute and return the training loss, compute additional
        metrics, and perform logging."""

        # foward pass
        y_true, y_pred, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="train")

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Operates on a single batch of data from the validation set.

        In this step you'd might generate examples or calculate anything
        of interest like accuracy.
        """

        # foward pass
        y_true, y_pred, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, stage="val")

        # return loss (and maybe more stuff)
        return_dict = loss

        return return_dict

    """
    configure in CLI instead
    def configure_optimizers(self):

        # optimise the whole model
        opt = torch.optim.AdamW(
            self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=1e-5
        )

        # cosine warm restarts
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2)

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=2e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=4,
            final_div_factor=10,
            pct_start=0.05,
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]
    """
