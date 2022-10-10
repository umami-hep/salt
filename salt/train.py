"""Top level training script, powered by the lightning CLI."""

from pytorch_lightning.cli import LightningCLI

from salt.data.datamodules import JetDataModule
from salt.lightning import LightningTagger


def main():
    LightningCLI(LightningTagger, JetDataModule)


if __name__ == "__main__":
    main()
