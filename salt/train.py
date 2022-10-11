"""Top level training script, powered by the lightning CLI."""

import comet_ml  # noqa F401
from pytorch_lightning.cli import LightningCLI

from salt.callbacks.saveconfig import SaveConfigCallback
from salt.data.datamodules import JetDataModule
from salt.lightning import LightningTagger


def main():
    LightningCLI(
        LightningTagger,
        JetDataModule,
        env_parse=True,
        save_config_callback=SaveConfigCallback,
    )


if __name__ == "__main__":
    main()
