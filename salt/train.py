"""Top level training script, powered by the lightning CLI."""

import comet_ml  # noqa F401
from pytorch_lightning.cli import LightningCLI

from salt.data.datamodules import JetDataModule
from salt.lightning import LightningTagger
from salt.utils.logging import get_comet_logger


def main():
    logger = get_comet_logger("test")
    logger.experiment.log_parameter("sample", "test")
    cli = LightningCLI(LightningTagger, JetDataModule)
    cli.instantiate_trainer(logger=logger)


if __name__ == "__main__":
    main()
