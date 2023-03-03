import glob
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jsonargparse.typing import register_type
from pytorch_lightning.cli import LightningCLI


# add support for converting yaml lists to tensors
def serializer(x: torch.Tensor) -> list:
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
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
        parser.link_arguments("name", "trainer.logger.init_args.experiment_name")
        parser.link_arguments("name", "model.name")
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        if self.subcommand == "fit":
            # reduce precision to imrprove performance
            # don't do this during evaluation as you will get variation wrt Athena
            torch.set_float32_matmul_precision("medium")

            # get timestamped output dir for this run
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
            log = "trainer.logger"
            name = sc["name"]
            log_dir = Path(sc["trainer.default_root_dir"])

            # handle case where we re-use an existing config: use parent of timestampped dir
            try:
                datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")
                log_dir = log_dir.parent
            except ValueError:
                pass

            # set the timestampped dir
            log_dir_timestamp = str(Path(log_dir / f"{name}_{timestamp}").resolve())
            sc["trainer.default_root_dir"] = log_dir_timestamp
            if sc[log]:
                sc[f"{log}.init_args.save_dir"] = log_dir_timestamp

            # add the labels from the model config to the data config
            labels = {}
            model_dict = vars(sc.model.model.init_args)

            # modify the input size to harmonize with the number of excluded variables
            exclude = sc["data.exclude"]

            if exclude is None:
                exclude = {}

            for i, submodel in enumerate(model_dict["init_nets"]["init_args"]["modules"]):
                if exclude.get(submodel["init_args"]["name"]):
                    submodel["init_args"]["net"]["init_args"]["input_size"] -= len(
                        exclude.get(submodel["init_args"]["name"])
                    )

            for submodel in model_dict["tasks"]["init_args"]["modules"]:
                assert "Task" in submodel["class_path"]
                task = submodel["init_args"]
                labels[task["name"]] = task["label"]
                if task["label_denominator"]:
                    labels[task["name"] + "_denominator"] = task["label_denominator"]
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

            # ensure only one device is used for testing
            n_devices = sc["trainer.devices"]
            if (isinstance(n_devices, str) or isinstance(n_devices, int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            # disable move_files_temp
            sc["data.move_files_temp"] = None

            print("-" * 100 + "\n")
