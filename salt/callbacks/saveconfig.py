import shutil
import socket
import subprocess
from pathlib import Path

import h5py
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.cli import LightningArgumentParser, Namespace


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config
        preserves this structure.
    Raises:
        RuntimeError: If the config file already exists in the directory
        to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        super().__init__()
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # save only on rank zero to avoid race conditions.
        if stage != "fit" or self.already_saved:
            return

        self.trainer = trainer
        self.plm = pl_module

        # get path info
        log_dir = Path(trainer.log_dir)
        assert log_dir is not None
        trainer.timestamp = log_dir.name
        config_path = Path(log_dir / self.config_filename)

        # broadcast whether to fail to all ranks
        file_exists = config_path.exists()
        file_exists = trainer.strategy.broadcast(file_exists)
        if file_exists and not self.overwrite:
            raise RuntimeError(
                f"{self.__class__.__name__} expected {config_path} to NOT exist."
                " Aborting to avoid overwriting results of a previous run."
            )

        if trainer.is_global_zero:
            # save configs
            self.save_config(config_path)

            # save metadata
            self.save_metadata(config_path)

            # broadcast so that all ranks are in sync on future calls to .setup()
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, config_path):
        # the `log_dir` needs to be created as we rely on the logger
        # to do it usually but it hasn't logged anything at this point
        config_path.parent.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"Created output dir {config_path.parent}")
        print("-" * 100, "\n")

        # copy the scale dict
        sd_path = Path(config_path.parent / Path(self.config.data.scale_dict).name)
        shutil.copyfile(self.config.data.scale_dict, sd_path)
        self.config.data.scale_dict = str(sd_path.resolve())

        # write config
        self.parser.save(
            self.config,
            str(config_path),
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )

        # log files as assets
        if self.plm.logger is not None:
            self.plm.logger.experiment.log_asset(config_path)
            self.plm.logger.experiment.log_asset(sd_path)

    def save_metadata(self, config_path):
        # TODO: log input variables from datasets

        trainer = self.trainer
        logger = self.plm.logger

        train_loader = trainer.datamodule.train_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        train_dset = train_loader.dataset
        val_dset = val_loader.dataset

        meta = {}

        meta["train_file"] = str(train_dset.filename)
        meta["val_file"] = str(val_dset.filename)
        meta["num_jets_train"] = len(train_dset)
        meta["num_jets_val"] = len(val_dset)
        batch_size = train_loader.batch_size
        batch_size = train_loader.sampler.batch_size if not batch_size else batch_size
        meta["batch_size"] = batch_size
        params = sum(p.numel() for p in self.plm.parameters() if p.requires_grad)
        meta["trainable_params"] = params

        meta["num_gpus"] = trainer.num_devices
        meta["gpu_ids"] = trainer.device_ids
        meta["num_workers"] = train_loader.num_workers

        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        meta["git_hash"] = git_hash.decode("ascii").strip()
        if logger:
            meta["out_dir"] = logger.save_dir
        if hasattr(self.trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp
        meta["torch_version"] = str(torch.__version__)
        meta["lightning_version"] = str(pl.__version__)
        meta["cuda_version"] = torch.version.cuda
        meta["hostname"] = socket.gethostname()

        # save the jet classes, which is stored as an attr in the training file
        with h5py.File(meta["train_file"]) as file:
            jet_name = self.config["data"]["inputs"]["jet"]
            jet_classes = file[f"{jet_name}/labels"].attrs["label_classes"]
            meta["jet_classes"] = dict(zip(range(len(jet_classes)), jet_classes))

        if logger:
            logger.log_hyperparams(meta)

        meta_path = Path(config_path.parent / "metadata.yaml")
        with open(meta_path, "w") as file:
            yaml.dump(meta, file, sort_keys=False)
