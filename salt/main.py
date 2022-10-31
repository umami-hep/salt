"""Top level training script, powered by the lightning CLI."""

import comet_ml  # noqa F401
from pytorch_lightning.cli import ArgsType

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import JetDataModule
from salt.lightning import LightningTagger
from salt.utils.cli import SaltCLI


def main(args: ArgsType = None) -> None:
    SaltCLI(
        model_class=LightningTagger,
        datamodule_class=JetDataModule,
        save_config_callback=SaveConfigCallback,
        env_parse=True,
        args=args,
        parser_kwargs={
            "fit": {"default_config_files": ["configs/base.yaml"]},
            "test": {"default_config_files": ["configs/base.yaml"]},
        },
    )


if __name__ == "__main__":
    main()
