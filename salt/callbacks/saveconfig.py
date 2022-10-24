import shutil
from datetime import datetime
from pathlib import Path

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
        self.config_filename = Path(self.config.config[0].rel_path).name
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return
        if stage != "fit":
            return

        trainer.timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")

        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None

        trainer.out_dir = Path(log_dir / Path(trainer.timestamp))
        config_path = Path(trainer.out_dir / self.config_filename)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = config_path.exists()
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist."
                    " Aborting to avoid overwriting results of a previous run. You can"
                    " delete the previous config file, set"
                    " `LightningCLI(save_config_callback=None)` to disable config"
                    " saving, or set `LightningCLI(save_config_overwrite=True)` to"
                    " overwrite the config file."
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger
            # to do it usually but it hasn't logged anything at this point
            config_path.parent.mkdir(parents=True, exist_ok=True)
            print("-" * 100)
            print(f"Created output dir {trainer.out_dir}")
            print("-" * 100, "\n")
            self.parser.save(
                self.config,
                str(config_path),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

            # copy the scale dict
            sd_path = Path(config_path.parent / Path(self.config.data.scale_dict).name)
            shutil.copyfile(self.config.data.scale_dict, sd_path)

            # log files as assets
            pl_module.logger.experiment.log_asset(config_path)
            pl_module.logger.experiment.log_asset(sd_path)

            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
