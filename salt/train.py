"""Top level training script, powered by the lightning CLI."""

# main.py
import torch
from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

cli = LightningCLI(DemoModel, BoringDataModule)
# note: don't call fit!!
