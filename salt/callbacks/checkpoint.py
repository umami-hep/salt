import os

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self) -> None:
        """A callback to checkpoint files during training."""
        filename = "{epoch:03d}-{val_loss:.4f}"
        super().__init__(save_top_k=-1, filename=filename)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # dynamically set the output dirpath form the trainer timestamp
        if stage == "fit":
            self.dirpath = os.path.join(trainer.out_dir, "ckpts")
            self.timestamp = trainer.timestamp
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
