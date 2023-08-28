import glob
import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from jsonargparse.typing import register_type
from lightning.pytorch.cli import LightningCLI

from salt.utils.git_check import check_for_uncommitted_changes, create_and_push_tag


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
        parser.add_argument("--force", action="store_true", help="Run with uncomitted changes.")
        parser.add_argument("--tag", action="store_true", help="Push a tag for the current code.")
        parser.add_argument(
            "--compile", action="store_true", help="Compile the model to speed up training."
        )

    def fit(self, model, **kwargs):
        if self.config[self.subcommand]["compile"]:
            model = torch.compile(model, mode="reduce-overhead")
        self.trainer.fit(model, **kwargs)

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        # add normalisation to init nets
        if sc.data.norm_in_model:
            for init_net in sc.model.model.init_args.init_nets.init_args.modules:
                init_net.init_args.norm_dict = sc.data.norm_dict
                init_net.init_args.variables = sc.data.variables
                init_net.init_args.input_names = sc.data.input_names
                init_net.init_args.concat_jet_tracks = sc.data.concat_jet_tracks

        # add the labels from the model config to the data config
        labels = {}
        model_dict = vars(sc.model.model.init_args)
        for submodel in model_dict["tasks"]["init_args"]["modules"]:
            assert "Task" in submodel["class_path"]
            task = submodel["init_args"]
            if self.subcommand == "fit":
                labels[task["name"]] = (task["input_type"], task["label"])
            if denominator := task.get("label_denominator"):
                labels[task["name"] + "_denominator"] = (task["input_type"], denominator)
        sc["data"]["labels"] = labels

        if self.subcommand == "fit":
            self.add_jet_class_names()

            # reduce precision to improve performance
            # don't do this during evaluation as you will get increased variation wrt Athena
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
            dirname = f"{name}_{timestamp}"
            log_dir_timestamp = str(Path(log_dir / dirname).resolve())
            sc["trainer.default_root_dir"] = log_dir_timestamp
            if sc[log]:
                sc[f"{log}.init_args.save_dir"] = log_dir_timestamp

            # run git checks
            if not sc["force"]:
                check_for_uncommitted_changes()
                if sc["tag"]:
                    create_and_push_tag(dirname)

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
            if (isinstance(n_devices, str | int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            # disable move_files_temp
            sc["data.move_files_temp"] = None

            print("-" * 100 + "\n")

    def add_jet_class_names(self) -> None:
        # add flavour label class names to jet classification task, if it exists
        sc = self.config[self.subcommand]
        for task in sc.model.model.init_args.tasks.init_args.modules:
            args = task.init_args
            if (
                args.name == "jet_classification"
                and args.label == "flavour_label"
                and args.class_names is None
            ):
                with h5py.File(sc.data.train_file) as f:
                    name = sc.data.input_names[args.input_type]
                    args.class_names = f[name].attrs[args.label]
