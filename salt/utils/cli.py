import glob
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from pytorch_lightning.cli import LightningCLI


def get_best_epoch(config_path: Path) -> Path:
    """Find the best perfoming epoch.

    Parameters
    ----------
    config_path : Path
        Path to saved training config file.

    Returns
    -------
    Path
        Path to best checkpoint for the training run.
    """
    ckpt_dir = Path(config_path.parent / "ckpts")
    print("No --ckpt_path specified, looking for best checkpoint in", ckpt_dir)
    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
    exp = r"(?<=val_loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print("Using checkpoint", ckpt)
    return ckpt


class SaltCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            "--name", default="salt", help="Name for this training run."
        )

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        if self.subcommand == "fit":
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
            log = "trainer.logger"
            name = sc["name"]
            if sc[log]:
                sc[f"{log}.init_args.experiment_name"] = name
                key = f"{log}.init_args.save_dir"
            else:
                key = "trainer.default_root_dir"
            log_dir = Path(sc[key])
            sc[key] = str(Path(log_dir / f"{name}_{timestamp}").resolve())

        if self.subcommand == "test":
            print("\n" + "-" * 100)

            # modify callbacks when testing
            self.save_config_callback = None
            sc["trainer.logger"] = False
            for c in sc["trainer.callbacks"]:
                if hasattr(c, "init_args") and hasattr(c.init_args, "refresh_rate"):
                    c.init_args.refresh_rate = 1

            # use the best epoch for testing
            if sc["ckpt_path"] is None:
                config = sc["config"]
                assert len(config) == 1
                best_epoch_path = get_best_epoch(Path(config[0].rel_path))
                sc["ckpt_path"] = best_epoch_path

            # ensure only one devices is used for testing
            n_devices = sc["trainer.devices"]
            if isinstance(n_devices, str) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            print("-" * 100 + "\n")
