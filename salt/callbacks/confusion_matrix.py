from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT


class ConfusionMatrixCallback(Callback):
    """Callback to log the confusion matrix during training, at the end of each epoch.

    Parameters
    ----------
    task_name : str
        Name of the task for which the confusion matrix will be logged
    class_names_override : list[str] | dict[str, str] | None
        Class names to use for the confusion matrix.
        It can be a list of class names or a mapping between the existing and new ones.
        If None, the class names from the task will be used.
    """

    def __init__(
        self, task_name: str, class_names_override: list[str] | dict[str, str] | None = None
    ) -> None:
        self.task_name = task_name
        self.class_names_override = class_names_override

    def setup(self, trainer: Trainer, _pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return

        self.truth_labels: list[int] = []
        self.pred_labels: list[int] = []

        for task in trainer.model.model.tasks:
            if task.name == self.task_name:
                self.task_input_name = task.input_name
                self.task_label_name = task.label
                if isinstance(self.class_names_override, dict):
                    self.task_class_names = [
                        self.class_names_override.get(name, name) for name in task.class_names
                    ]
                else:
                    self.task_class_names = self.class_names_override or task.class_names
                break

    def on_validation_batch_end(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        pred_labels_batch = torch.argmax(
            outputs["outputs"]["preds"][self.task_input_name][self.task_name], dim=-1
        )
        truth_labels_batch = outputs["outputs"]["labels"][self.task_input_name][
            self.task_label_name
        ]

        self.truth_labels.extend(truth_labels_batch)
        self.pred_labels.extend(pred_labels_batch)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
    ) -> None:
        if isinstance(trainer.logger, CometLogger):
            # Log the confusion matrix to Comet
            trainer.logger.experiment.log_confusion_matrix(
                y_true=self.truth_labels,
                y_predicted=self.pred_labels,
                labels=self.task_class_names,
                epoch=trainer.current_epoch,
            )
        self.truth_labels = []
        self.pred_labels = []
