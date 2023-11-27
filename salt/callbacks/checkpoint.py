from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from s3path import S3Path


class Checkpoint(ModelCheckpoint):
    def __init__(self, monitor_loss: str, fname_string: str = "val_loss") -> None:
        """Create checkpoint files during training."""
        filename = "epoch={epoch:03d}-" + fname_string + "={" + monitor_loss + ":.5f}"
        super().__init__(save_top_k=-1, filename=filename, auto_insert_metric_name=False)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            if trainer.fast_dev_run:
                return

            if "s3:/" in trainer.log_dir[:4]:
                log_dir = S3Path(trainer.log_dir.replace("s3://", "").replace("s3:/", ""))
                self.dirpath = "s3://" + str(log_dir / "ckpts")
            elif "s3:/" in trainer.log_dir:
                raise ValueError(
                    f"trainer.log_dir should start with 's3:/', instead of {trainer.log_dir}"
                )
            else:
                log_dir = Path(trainer.log_dir)
                self.dirpath = str(log_dir / "ckpts")

            # this could be used to add the timestamp to the filename
            # self.timestamp = log_dir.name

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
