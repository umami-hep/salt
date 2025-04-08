import torch
import torch.nn.functional as nnf
import torchmetrics as tm
from lightning import Callback, LightningModule, Trainer
from torch.nn import Module, ModuleDict

from salt.utils.mask_utils import mask_from_logits, reco_metrics


class MaskformerMetrics(Callback):
    def __init__(self, only_val=True, mask_criteria: dict[str, tuple] | None = None):
        """Callback to log metrics for Maskformer model.

        The following metrics are logged.
            - class_exact_match
            - not null class efficiency and purity
            - per class efficiency and purity
            - mean absolute error for regression tasks
            - mask match efficiency and fake rate, for different criteria

        Parameters
        ----------
        only_val: bool
            Whether to log metrics only on validation step. Default is true. It is not recommended
            to log metrics on training step, as it will slow down the training process.
        mask_criteria: dict[str, tuple] | None
            Dictionary with the criteria to evaluate the mask match. The keys are the name of the
            criteria, and the values are tuples (R, P) where R is the mimimum recall and P is the
            minimum purity for the criteria to be met. Default is None, which will define the
            'perfect' critera as (1, 1) and the 'loose' criteria as (0.5, 0.5).

        """
        self.mask_criteria = mask_criteria
        if not self.mask_criteria:
            self.mask_criteria = {"perfect": (1, 1), "loose": (0.5, 0.5)}
        self.only_val = only_val

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        model = module.model
        if not model.mask_decoder:
            raise ValueError("Model requires mask_decoder to use MaskformerMetrics callback")
        kwargs = {"sync_dist": len(trainer.device_ids) > 1}

        def log(metrics, stage):
            for t, loss_value in metrics.items():
                module.log(f"{stage}/{t}", loss_value, **kwargs)

        self.log = log
        # make Not hard-coded
        self.classes = trainer.datamodule.train_dataloader().dataset.mf_config.object.class_names
        self.null_index = len(self.classes) - 1
        nclasses = len(self.classes)
        mask_size = 40

        module.maskformer_metrics = Module()
        self.metrics = module.maskformer_metrics
        # class label metrics
        self.metrics.class_exact_match = tm.classification.MulticlassExactMatch(
            num_classes=nclasses
        )
        self.metrics.class_accuracy_micro = tm.classification.MulticlassAccuracy(
            num_classes=nclasses, average="micro"
        )
        self.metrics.class_accuracy_macro = tm.classification.MulticlassAccuracy(
            num_classes=nclasses, average="macro"
        )

        # per-class metrics
        self.metrics.per_class_eff = ModuleDict({
            k: tm.classification.BinaryRecall() for k in self.classes if k != "null"
        })
        self.metrics.per_class_pur = ModuleDict({
            k: tm.classification.BinaryPrecision() for k in self.classes if k != "null"
        })
        # not-null metrics
        self.metrics.notnull_eff = tm.classification.BinaryRecall()
        self.metrics.notnull_pur = tm.classification.BinaryPrecision()

        # mask metrics
        self.metrics.mask_exact_match = tm.classification.MultilabelExactMatch(num_labels=mask_size)

    def _get_metrics(self, module, outputs):
        metrics = {}

        preds, labels, pad_mask = (outputs["preds"], outputs["labels"], outputs["pad_masks"])
        qpreds, qlabels = preds["objects"], labels["objects"]

        # class metrics
        obj_class_pred = qpreds["class_logits"].argmax(-1).detach()
        obj_class_tgt = qlabels["object_class"].detach()

        self.metrics.class_exact_match(obj_class_pred, obj_class_tgt)
        metrics["class_exact_match"] = self.metrics.class_exact_match
        self.metrics.class_accuracy_micro(obj_class_pred.view(-1), obj_class_tgt.view(-1))
        metrics["class_accuracy_micro"] = self.metrics.class_accuracy_micro
        self.metrics.class_accuracy_macro(obj_class_pred.view(-1), obj_class_tgt.view(-1))
        metrics["class_accuracy_macro"] = self.metrics.class_accuracy_macro

        # per-class metrics
        for i, k in enumerate(self.classes):
            if k == "null":
                obj_present_tgt = obj_class_tgt != self.null_index
                obj_present_pred = obj_class_pred != self.null_index
                self.metrics.notnull_eff(obj_present_pred, obj_present_tgt)
                self.metrics.notnull_pur(obj_present_pred, obj_present_tgt)
                metrics["notnull_eff"] = self.metrics.notnull_eff
                metrics["notnull_pur"] = self.metrics.notnull_pur
            else:
                is_class_tgt = obj_class_tgt.view(-1) == i
                is_class_pred = obj_class_pred.view(-1) == i
                self.metrics.per_class_eff[k](is_class_pred, is_class_tgt)
                self.metrics.per_class_pur[k](is_class_pred, is_class_tgt)
                metrics[f"{k}_eff"] = self.metrics.per_class_eff[k]
                metrics[f"{k}_pur"] = self.metrics.per_class_pur[k]

        pred_masks = qpreds["masks"].detach()
        tgt_masks = qlabels["masks"].detach()
        pred_masks = mask_from_logits(pred_masks, "sigmoid", pad_mask["tracks"], obj_class_pred)

        for key, (recall, purity) in self.mask_criteria.items():
            eff, fr = reco_metrics(
                pred_masks, tgt_masks, min_recall=recall, min_purity=purity, reduce=True
            )
            metrics[f"query_{key}_match_eff"] = eff
            metrics[f"query_{key}_match_fake"] = fr

        if "regression" in qpreds:
            task = [
                t
                for t in module.model.tasks
                if t.input_name == "objects" and t.name == "regression"
            ]
            if len(task) != 1:
                raise ValueError(
                    "Can only run regression maskformer metrics for a single query regression task"
                )
            task = task[0]

            valid_idx = ~torch.isnan(qlabels["regression"]).all(-1)
            for i, t in enumerate(task.targets):
                unscaled_preds = task.scaler.inverse(t, qpreds["regression"][valid_idx][:, i])
                targets = qlabels[t][valid_idx]
                metrics[f"query_{t}_mae"] = nnf.l1_loss(unscaled_preds, targets)

        return metrics

    def _get_log_metrics(self, module, outputs, stage):
        metrics = self._get_metrics(module, outputs["outputs"])
        self.log(metrics, stage)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):  # noqa: ARG002
        if not self.only_val and not trainer.fast_dev_run:
            self._get_log_metrics(module, outputs, "train")

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):  # noqa: ARG002
        if not trainer.fast_dev_run:
            self._get_log_metrics(module, outputs, "val")
