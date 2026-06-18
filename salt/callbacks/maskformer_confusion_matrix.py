import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import confusion_matrix
from torch import Tensor


class MaskformerConfusionMatrix(Callback):
    """Callback to log normalized confusion matrix for MaskFormer class labels to Comet.

    Parameters
    ----------
    only_val : bool
        Whether to log confusion matrix only on validation epoch end. Default is True.
    log_every_n_epochs : int
        Log confusion matrix every N epochs. Default is 1.
    normalize : bool
        Whether to normalize the confusion matrix. Default is True.
    """

    def __init__(self, only_val: bool = True, log_every_n_epochs: int = 1, normalize: bool = True):
        self.only_val = only_val
        self.log_every_n_epochs = log_every_n_epochs
        self.normalize = normalize
        self.val_preds: list[Tensor] = []
        self.val_targets: list[Tensor] = []
        self.train_preds: list[Tensor] = []
        self.train_targets: list[Tensor] = []

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        model = module.model
        if not model.mask_decoder:
            raise ValueError(
                "Model requires mask_decoder to use MaskformerConfusionMatrix callback"
            )

        # Get class names from datamodule
        self.classes = trainer.datamodule.train_dataloader().dataset.mf_config.object.class_names
        self.null_index = len(self.classes) - 1

    def on_validation_batch_end(
        self,
        trainer,
        module,  # noqa: ARG002
        outputs,
        batch,  # noqa: ARG002
        batch_idx,  # noqa: ARG002
    ):
        if trainer.fast_dev_run:
            return

        # Extract predictions and labels
        preds = outputs["outputs"]["preds"]["objects"]
        labels = outputs["outputs"]["labels"]["objects"]

        obj_class_pred = preds["class_logits"].argmax(-1).detach().cpu()
        obj_class_tgt = labels["object_class"].detach().cpu()

        # Store for epoch-end aggregation
        self.val_preds.append(obj_class_pred.view(-1))
        self.val_targets.append(obj_class_tgt.view(-1))

    def on_train_batch_end(
        self,
        trainer,
        module,  # noqa: ARG002
        outputs,
        batch,  # noqa: ARG002
        batch_idx,  # noqa: ARG002
    ):
        if self.only_val or trainer.fast_dev_run:
            return

        # Extract predictions and labels
        preds = outputs["outputs"]["preds"]["objects"]
        labels = outputs["outputs"]["labels"]["objects"]

        obj_class_pred = preds["class_logits"].argmax(-1).detach().cpu()
        obj_class_tgt = labels["object_class"].detach().cpu()

        # Store for epoch-end aggregation
        self.train_preds.append(obj_class_pred.view(-1))
        self.train_targets.append(obj_class_tgt.view(-1))

    def _log_confusion_matrix(self, module: LightningModule, stage: str):
        """Create and log confusion matrix to Comet."""
        if stage == "val":
            all_preds = torch.cat(self.val_preds).numpy()
            all_targets = torch.cat(self.val_targets).numpy()
        else:
            all_preds = torch.cat(self.train_preds).numpy()
            all_targets = torch.cat(self.train_targets).numpy()

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(len(self.classes)))

        # Normalize if requested
        if self.normalize:
            cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if self.normalize else "d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax,
            vmin=0,
            vmax=1 if self.normalize else None,
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        title = f"{'Normalized ' if self.normalize else ''}Confusion Matrix - {stage.capitalize()}"
        ax.set_title(title)
        plt.tight_layout()

        # Log to Comet if available
        if hasattr(module.logger, "experiment"):
            module.logger.experiment.log_figure(
                figure_name=f"{stage}/confusion_matrix",
                figure=fig,
                step=module.current_epoch,
            )

        plt.close(fig)

    def on_validation_epoch_end(self, trainer, module):
        if trainer.fast_dev_run or len(self.val_preds) == 0:
            return

        # Only log every N epochs
        if module.current_epoch % self.log_every_n_epochs != 0:
            self.val_preds.clear()
            self.val_targets.clear()
            return

        self._log_confusion_matrix(module, "val")

        # Clear storage
        self.val_preds.clear()
        self.val_targets.clear()

    def on_train_epoch_end(self, trainer, module):
        if self.only_val or trainer.fast_dev_run or len(self.train_preds) == 0:
            return

        # Only log every N epochs
        if module.current_epoch % self.log_every_n_epochs != 0:
            self.train_preds.clear()
            self.train_targets.clear()
            return

        self._log_confusion_matrix(module, "train")

        # Clear storage
        self.train_preds.clear()
        self.train_targets.clear()
