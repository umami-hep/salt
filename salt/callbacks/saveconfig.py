import contextlib
import json
import shutil
import socket
from contextlib import suppress
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, Namespace

from salt.utils.git_check import get_git_hash


def get_attr(file, attribute, key=None):
    obj = file if key is None else file[key]
    value = dict(obj.attrs).get(attribute)
    if np.issubdtype(type(value), np.integer):
        value = int(value)
    if isinstance(value, str):
        with suppress(json.decoder.JSONDecodeError, TypeError):
            value = json.loads(value)
    return value


class SaveConfigCallback(Callback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        """Saves a LightningCLI config to the log_dir when training starts.

        Parameters
        ----------
        parser : LightningArgumentParser
            The parser object used to parse the configuration.
        config : Namespace
            The parsed configuration that will be saved.
        config_filename : str, optional
            Filename for the config file.
        overwrite : bool, optional
            Whether to overwrite an existing config file.
        multifile : bool, optional
            When input is multiple config files, saved config

        Raises
        ------
        RuntimeError:
            If the config file already exists in the directory
            to avoid overwriting a previous run

        """
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

        # copy the norm dict
        nd_path = Path(config_path.parent / Path(self.config.data.norm_dict).name)
        shutil.copyfile(self.config.data.norm_dict, nd_path)
        self.config.data.norm_dict = str(nd_path.resolve())

        # copy the class dict
        if self.config.data.class_dict:
            cd_path = Path(config_path.parent / Path(self.config.data.class_dict).name)
            shutil.copyfile(self.config.data.class_dict, cd_path)
            self.config.data.class_dict = str(cd_path.resolve())

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
            self.plm.logger.experiment.log_asset(nd_path)
            self.plm.logger.experiment.log_asset(cd_path)

    def save_metadata(self, config_path):
        trainer = self.trainer
        logger = self.plm.logger

        train_loader = trainer.datamodule.train_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        train_dset = train_loader.dataset
        val_dset = val_loader.dataset
        jet_name = self.config["data"]["input_names"]["jet"]

        meta = {}

        meta["train_file"] = str(train_dset.filename)
        meta["val_file"] = str(val_dset.filename)
        meta["num_jets_train"] = len(train_dset)
        meta["num_jets_val"] = len(val_dset)
        meta["total_jets_train"] = len(train_dset.file[jet_name])
        meta["total_jets_val"] = len(val_dset.file[jet_name])
        batch_size = train_loader.batch_size
        batch_size = batch_size if batch_size else train_loader.sampler.batch_size
        meta["batch_size"] = batch_size
        params = sum(p.numel() for p in self.plm.parameters() if p.requires_grad)
        meta["trainable_params"] = params
        meta["num_gpus"] = trainer.num_devices
        meta["gpu_ids"] = trainer.device_ids
        meta["num_workers"] = train_loader.num_workers

        with contextlib.suppress(KeyError):
            meta["num_unique_jets_train"] = get_attr(train_dset.file, "unique_jets")
            meta["num_unique_jets_val"] = get_attr(val_dset.file, "unique_jets")
            meta["dsids"] = get_attr(train_dset.file, "dsids")

        meta["git_hash"] = get_git_hash()
        if logger:
            meta["out_dir"] = logger.save_dir
            meta["log_url"] = logger.experiment.url
        if hasattr(self.trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp
        meta["torch_version"] = str(torch.__version__)
        meta["lightning_version"] = str(L.__version__)
        meta["cuda_version"] = torch.version.cuda
        meta["hostname"] = socket.gethostname()

        if logger:
            logger.log_hyperparams(meta)

        # save the jet classes, which is stored as an attr in the training file
        with h5py.File(meta["train_file"]) as file:
            try:
                jet_classes = file[f"{jet_name}"].attrs["flavour_label"]
            except KeyError:
                jet_classes = "not available"
            meta["jet_classes"] = dict(zip(range(len(jet_classes)), jet_classes, strict=True))

        with contextlib.suppress(KeyError):
            meta["jet_counts_train"] = get_attr(train_dset.file, "jet_counts")
            meta["jet_counts_val"] = get_attr(val_dset.file, "jet_counts")
            meta["pp_config_train"] = get_attr(train_dset.file, "config")
            meta["pp_config_val"] = get_attr(val_dset.file, "config")

        meta_path = Path(config_path.parent / "metadata.yaml")
        with open(meta_path, "w") as file:
            yaml.dump(meta, file, sort_keys=False)
