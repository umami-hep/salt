import contextlib
import json
import os
import shutil
import socket
from contextlib import suppress
from pathlib import Path
from typing import Any

import h5py
import lightning
import numpy as np
import torch
import yaml
from ftag.git_check import get_git_hash
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, Namespace
from lightning.pytorch.loggers.comet import CometLogger

try:  # pragma: no cover
    from s3fs import S3FileSystem
    from s3path import S3Path

    _HAS_S3 = True
except ImportError:  # pragma: no cover
    _HAS_S3 = False


def get_attr(file: h5py.File, attribute: str, key: str | None = None) -> Any:
    """Retrieve an HDF5 attribute from a file or one of its datasets.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file handle.
    attribute : str
        Name of the attribute to fetch.
    key : str | None, optional
        Dataset key within the file. If provided, the attribute is read
        from ``file[key]`` instead of the file root.

    Returns
    -------
    Any
        The attribute value, converted where possible:

        * NumPy integer scalars are cast to built-in ``int``.
        * If the value is a string containing JSON, it is parsed
          into a Python object (list/dict/etc.) with graceful failure.

        ``None`` if the attribute is not present.

    Examples
    --------
    >>> with h5py.File("example.h5", "r") as f:
    ...     n_jets = get_attr(f, "n_jets")
    ...     config = get_attr(f, "config_json", key="my_dataset")
    """
    obj = file if key is None else file[key]
    value: Any = dict(obj.attrs).get(attribute)

    # Only cast if we actually have a NumPy/integer scalar
    if isinstance(value, (np.integer | int)):
        return int(value)

    # Try to parse JSON-encoded strings
    if isinstance(value, str):
        with suppress(json.decoder.JSONDecodeError, TypeError):
            return json.loads(value)

    return value


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Parameters
    ----------
    parser : LightningArgumentParser
        The parser object used to parse the configuration.
    config : Namespace
        The parsed configuration that will be saved.
    config_filename : str, optional
        Filename for the config file, by default "config.yaml"
    overwrite : bool, optional
        Whether to overwrite an existing config file, by default False
    multifile : bool, optional
        When input is multiple config files, saved config, by default False
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
        """Setup the config callback.

        Parameters
        ----------
        trainer : Trainer
            Trainer instance that is used
        pl_module : LightningModule
            Pytorch lightning module that is trained
        stage : str
            Stage in which the config is stored

        Raises
        ------
        ImportError
            S3FileSystem and S3Path not found
        ValueError
            If the trainer.log_dir is not starting with "s3:/"
        RuntimeError
            If the config file already exists in the directory
            to avoid overwriting a previous run
        """
        # save only on rank zero to avoid race conditions.
        if stage != "fit" or self.already_saved:
            return

        self.trainer = trainer
        self.plm = pl_module

        # get path info
        if "s3:/" in trainer.log_dir[:4]:
            if not _HAS_S3:
                raise ImportError("S3FileSystem and S3Path not found!")

            self.use_S3 = True
            self.make_path = S3Path
            log_dir = self.make_path("/" + trainer.log_dir.replace("s3://", "").replace("s3:/", ""))
            self.S3_session = S3FileSystem(anon=False)
        elif "s3:/" in trainer.log_dir:
            raise ValueError(
                f"trainer.log_dir should start with 's3:/', instead of {trainer.log_dir}"
            )
        else:
            self.use_S3 = False
            self.make_path = Path
            log_dir = self.make_path(trainer.log_dir)

        assert log_dir is not None
        trainer.timestamp = log_dir.name
        # broadcast whether to fail to all ranks
        config_path = self.make_path(log_dir / self.config_filename)
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

    def save_config(self, config_path: Path) -> None:
        """Save the config file.

        Parameters
        ----------
        config_path : Path
            Path under which the config file will be stored
        """
        # the `log_dir` needs to be created as we rely on the logger
        # to do it usually but it hasn't logged anything at this point
        config_path.parent.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"Created output dir {config_path.parent} {'on S3' if self.use_S3 else ''}")
        print("-" * 100, "\n")

        # copy the norm dict
        nd_path = self.make_path(
            config_path.parent / self.make_path(self.config.data.norm_dict).name
        )
        self.config.data.norm_dict = self.write_file(self.config.data.norm_dict, nd_path)
        # copy the class dict
        if self.config.data.class_dict:
            cd_path = self.make_path(
                config_path.parent / self.make_path(self.config.data.class_dict).name
            )
            self.config.data.class_dict = self.write_file(self.config.data.class_dict, cd_path)

        # write config
        self.write_yaml_file(self.config, config_path)

        # log files as assets
        # currently cannot save log files as assests on S3
        if isinstance(self.plm.logger, CometLogger) and not self.use_S3:
            self.plm.logger.experiment.log_asset(config_path)
            self.plm.logger.experiment.log_asset(nd_path)
            if self.config.data.class_dict:
                self.plm.logger.experiment.log_asset(cd_path)

    def save_metadata(self, config_path: Path):
        """Save the metadata.

        Parameters
        ----------
        config_path : Path
            Path under which the metadata will be stored
        """
        trainer = self.trainer
        logger = self.plm.logger

        train_loader = trainer.datamodule.train_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        train_dset = train_loader.dataset
        val_dset = val_loader.dataset

        global_object = self.plm.global_object
        if input_map := self.config["data"]["input_map"]:
            global_object = input_map[global_object]

        meta: dict[str, Any] = {}

        meta["train_file"] = str(train_dset.filename)
        meta["val_file"] = str(val_dset.filename)
        meta["num_train"] = len(train_dset)
        meta["num_val"] = len(val_dset)
        batch_size = train_loader.batch_size
        batch_size = batch_size or train_loader.sampler.batch_size
        meta["batch_size"] = batch_size
        params = sum(p.numel() for p in self.plm.parameters() if p.requires_grad)
        meta["trainable_params"] = params
        meta["num_gpus"] = trainer.num_devices
        meta["gpu_ids"] = trainer.device_ids
        meta["num_workers"] = train_loader.num_workers

        # Read metadata from the train file
        with h5py.File(meta["train_file"]) as train_file:
            meta["available_train"] = len(train_file[global_object])

            # TODO(@jabarr): update UPP to call attribute objects instead of jets
            # https://github.com/umami-hep/umami-preprocessing/issues/56
            with contextlib.suppress(KeyError):
                meta["num_unique_jets_train"] = get_attr(train_file, "unique_jets")
                meta["dsids"] = get_attr(train_file, "dsids")
                meta["jet_counts_train"] = get_attr(train_file, "jet_counts")
                meta["pp_config_train"] = get_attr(train_file, "config")

            # save the object classes, which is stored as an attr in the training file
            try:
                object_classes = train_file[global_object].attrs["flavour_label"]
            except KeyError:
                object_classes = "not available"
            meta["object_classes"] = dict(
                zip(range(len(object_classes)), object_classes, strict=True)
            )

        # Read metadata from the validation file
        with h5py.File(meta["val_file"]) as val_file:
            meta["available_val"] = len(val_file[global_object])

            # TODO(@jabarr): update UPP to call attribute objects instead of jets
            # https://github.com/umami-hep/umami-preprocessing/issues/56
            with contextlib.suppress(KeyError):
                meta["num_unique_jets_val"] = get_attr(val_file, "unique_jets")
                meta["jet_counts_val"] = get_attr(val_file, "jet_counts")
                meta["pp_config_val"] = get_attr(val_file, "config")

        meta["salt_hash"] = get_git_hash(Path(__file__).parent)
        if logger:
            meta["out_dir"] = os.environ["COMET_OFFLINE_DIRECTORY"]
            if not self.use_S3:
                # Currently not available on S3
                meta["log_url"] = logger.experiment.url
        if hasattr(self.trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp
        meta["torch_version"] = str(torch.__version__)
        meta["lightning_version"] = str(lightning.__version__)
        meta["cuda_version"] = torch.version.cuda
        meta["hostname"] = socket.gethostname()

        if logger and not self.use_S3:
            # Currently not available on S3
            logger.log_hyperparams(meta)

        meta_path = self.make_path(config_path.parent / "metadata.yaml")
        self.write_dump_yaml_file(meta, meta_path)

    def write_file(self, file: str, store_path: Path) -> str:
        """Writes the file on S3 at the store_path.

        Parameters
        ----------
        file : str
            Local path of the file
        store_path : Path
            Path to which the file is written

        Returns
        -------
        str
            The output path
        """
        if self.use_S3:
            self.S3_session.put(file, store_path)
            outpath = file
        else:
            shutil.copyfile(file, store_path)
            outpath = str(store_path.resolve())
        return outpath

    def write_yaml_file(self, yaml_file: str, store_path: Path) -> None:
        """Writes the file on S3 at the store_path."""
        if self.use_S3:
            local_path = Path("local_config.yaml")
            self.parser.save(
                yaml_file,
                str(local_path),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            self.S3_session.put(str(local_path), store_path)
            local_path.unlink()
        else:
            self.parser.save(
                yaml_file,
                str(store_path),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

    def write_dump_yaml_file(self, meta, store_path):
        """Writes the file on S3 at the store_path."""
        if self.use_S3:
            local_path = Path("metadata.yaml")
            with open(str(local_path), "w") as file:
                yaml.dump(meta, file, sort_keys=False)
            self.S3_session.put(str(local_path), store_path)
            local_path.unlink()
        else:
            with open(store_path, "w") as file:
                yaml.dump(meta, file, sort_keys=False)
