import glob
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jsonargparse.typing import register_type
from pytorch_lightning.cli import LightningCLI


# handle list -> tensor config
def serializer(x):
    return x.tolist()


def deserializer(x):
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


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
    exp = r"(?<=loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print("Using checkpoint", ckpt)
    return ckpt


class SaltCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--name", default="salt", help="Name for this training run.")
        # TODO: link arguments
        # parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        # parser.link_arguments("name", "trainer.logger.init_args.experiment_name")

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        if self.subcommand == "fit":
            # get timestamped output dir for this run
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
            log = "trainer.logger"
            name = sc["name"]

            # handle cases depending on whether the logger is present
            log_dir = sc["trainer.default_root_dir"]
            if sc[log]:
                sc[f"{log}.init_args.experiment_name"] = name  # TODO: link arguments
                log_dir = sc[f"{log}.init_args.save_dir"]
            log_dir = Path(log_dir)

            # handle case where we re-use an existing config: use parent of timestampped dir
            try:
                datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")
                log_dir = log_dir.parent
            except ValueError:
                ...

            # set the timestampped dir
            log_dir = str(Path(log_dir / f"{name}_{timestamp}").resolve())
            sc["trainer.default_root_dir"] = log_dir
            if sc[log]:
                sc[f"{log}.init_args.save_dir"] = log_dir

            # add the labels from the model config to the data config
            labels = {}
            model_dict = vars(sc.model.model.init_args)
            for submodel in model_dict["tasks"]["init_args"]["modules"]:
                assert "Task" in submodel["class_path"]
                task = submodel["init_args"]
                labels[task["name"]] = task["label"]
            sc["data"]["labels"] = labels

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
            if (isinstance(n_devices, str) or isinstance(n_devices, int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            print("-" * 100 + "\n")
