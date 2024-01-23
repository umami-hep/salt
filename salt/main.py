"""Top level training script, powered by the lightning CLI."""

import pathlib

import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import SaltDataModule
from salt.modelwrapper import ModelWrapper
from salt.utils.cli import SaltCLI


def main(args: ArgsType = None) -> None:
    config_dir = pathlib.Path(__file__).parent / "configs"

    SaltCLI(
        model_class=ModelWrapper,
        datamodule_class=SaltDataModule,
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
