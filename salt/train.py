"""Top level training script, powered by the lightning CLI."""

# main.py
from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from pytorch_lightning.demos.boring_classes import BoringDataModule

from salt.lightning import MyModel


def main():
    LightningCLI(MyModel, BoringDataModule)


if __name__ == "__main__":
    main()
