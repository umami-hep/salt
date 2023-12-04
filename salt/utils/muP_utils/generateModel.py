"""Top level training script, powered by the lightning CLI."""

import pathlib
import sys

import comet_ml  # noqa F401
from lightning.pytorch.cli import ArgsType
from torch import save

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import JetDataModule
from salt.modelwrapper import ModelWrapper
from salt.utils.cli import SaltCLI
from salt.utils.muP_utils.configuration_muP import get_model_path


def generateModel(args: ArgsType = None) -> None:
    config_dir = pathlib.Path(__file__).parent.parent.parent / "configs"

    if args is None:
        model_type = sys.argv.pop(1)
    else:
        model_type = args[0]
        args = args[1:]
        sys.argv = [sys.argv[0]]

    cli = SaltCLI(
        model_class=ModelWrapper,
        datamodule_class=JetDataModule,
        save_config_callback=SaveConfigCallback,
        args=args,
        run=False,
        parser_kwargs={
            "default_env": True,
            "default_config_files": [f"{config_dir}/base.yaml"],
        },
    )
    if "temp" in model_type:
        # Store the model with the ModelWrapper
        print(f"Saving model_type={model_type} at {get_model_path(model_type)}")
        save(cli.model, get_model_path(model_type))
    else:
        # Store the model inside the ModelWrapper
        print(f"Saving model_type={model_type} at {get_model_path(model_type)}")
        save(cli.model.model, get_model_path(model_type))


if __name__ == "__main__":
    generateModel()
