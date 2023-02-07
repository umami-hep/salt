"""Top level training script, powered by the lightning CLI."""

import pathlib

import comet_ml  # noqa F401
from pytorch_lightning.cli import ArgsType

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import JetDataModule
from salt.lightning import LightningTagger
from salt.utils.cli import SaltCLI

config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    SaltCLI(
        model_class=LightningTagger,
        datamodule_class=JetDataModule,
        save_config_callback=SaveConfigCallback,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]},
            "test": {"default_config_files": [f"{config_dir}/base.yaml"]},
        },
    )


if __name__ == "__main__":
    main()
