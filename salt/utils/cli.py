import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from ftag.git_check import check_for_uncommitted_changes, create_and_push_tag
from jsonargparse.typing import register_type
from lightning.pytorch.cli import LightningCLI

from salt.utils.array_utils import listify


# add support for converting yaml lists to tensors
def serializer(x: torch.Tensor) -> list:
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


def get_best_epoch(config_path: Path) -> str:
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
    ckpts = list(Path(ckpt_dir).glob("*.ckpt"))
    exp = r"(?<=loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print("Using checkpoint", ckpt)
    return str(ckpt)


class SaltCLI(LightningCLI):
    def apply_links(self, parser) -> None:
        parser.link_arguments("name", "trainer.logger.init_args.experiment_name")
        parser.link_arguments("name", "model.name")
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("data.global_object", "model.global_object")

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("-n", "--name", default="salt", help="Name for this training run.")
        parser.add_argument(
            "-f", "--force", action="store_true", help="Run with uncomitted changes."
        )
        parser.add_argument(
            "-t", "--tag", action="store_true", help="Push a tag for the current code."
        )
        parser.add_argument(
            "--compile", action="store_true", help="Compile the model to speed up training."
        )
        self.apply_links(parser)

    def fit(self, model, **kwargs):
        if self.config[self.subcommand]["compile"]:
            # unfortunately compiling in place doesn't work
            # https://github.com/pytorch/pytorch/pull/97565
            # https://github.com/pytorch/pytorch/issues/101107
            model.model = torch.compile(model.model)
        self.trainer.fit(model, **kwargs)

    def before_instantiate_classes(self) -> None:
        """A lot of automatic configuration is done here."""
        sc = self.config[self.subcommand] if self.subcommand else self.config

        if "config_S3" in sc.data:
            from salt.utils.file_utils import require_S3_CLI, setup_S3_CLI

            if require_S3_CLI(sc.data["config_S3"]):
                sc.data = setup_S3_CLI(sc.data)

        # add the labels from the model config to the data config
        labels: dict = {}
        model_dict = vars(sc.model.model.init_args)
        for submodel in model_dict["tasks"]["init_args"]["modules"]:
            assert "Task" in submodel["class_path"]
            task = submodel["init_args"]
            if task["input_name"] not in labels:
                labels[task["input_name"]] = []
            # Check if there is a merge dict and try if exists
            if hasattr(self.config, "fit"):
                fit_config = self.config.fit.model.model.init_args
                if hasattr(fit_config, "merge_dict"):
                    merge_dict = self.config.fit.model.model.init_args.merge_dict
                    if merge_dict is not None:
                        task_input_name = task["input_name"]
                        if task_input_name in merge_dict:
                            for inp in merge_dict[task["input_name"]]:
                                if inp not in labels:
                                    labels[inp] = []
                                if self.subcommand == "fit":
                                    if label := task.get("label"):
                                        labels[inp].append(label)
                                    if weight := task.get("sample_weight"):
                                        labels[inp].append(weight)
                                    if targets := task.get("targets"):
                                        for target in listify(targets):
                                            labels[inp].append(target)
            if self.subcommand == "fit":
                if label := task.get("label"):
                    labels[task["input_name"]].append(label)
                if weight := task.get("sample_weight"):
                    labels[task["input_name"]].append(weight)
                if targets := task.get("targets"):
                    for target in listify(targets):
                        labels[task["input_name"]].append(target)
            if denominators := task.get("target_denominators"):
                for denominator in listify(denominators):
                    labels[task["input_name"]].append(denominator)

        if model_dict.get("mask_decoder"):
            if not (mf_config := sc["data"].get("mf_config")):
                raise ValueError("Mask decoder requires 'mf_config' in data config.")
            if mf_config.constituent.name not in labels:
                raise ValueError(
                    f"The constituent name {mf_config.constituent.name} is not in the data labels. "
                    "Ensure that the constituent name is in the input_map of the data config."
                )
            # Needed in case no tasks other than mask prediction/classification
            if "objects" not in labels:
                labels["objects"] = []
            labels["objects"] += [
                mf_config.object.id_label,
                mf_config.object.class_label,
            ]
            labels[mf_config.constituent.name] += [mf_config.constituent.id_label]
        sc["data"]["labels"] = labels

        # add norm
        sc["model"]["norm_config"] = {}
        sc["model"]["norm_config"]["norm_dict"] = sc.data.norm_dict
        sc["model"]["norm_config"]["variables"] = sc.data.variables
        sc["model"]["norm_config"]["global_object"] = sc.data.global_object
        sc["model"]["norm_config"]["input_map"] = sc.data.input_map

        if self.subcommand == "fit" or self.subcommand is None:
            # add variables to init nets
            for init_net in sc.model.model.init_args.init_nets:
                init_net["variables"] = sc.data.variables
                init_net["global_object"] = sc.data.global_object
            # add variables to featurewise nets
            if sc.model.model.init_args.featurewise_nets:
                for featurewise_net in sc.model.model.init_args.featurewise_nets:
                    featurewise_net["variables"] = sc.data.variables

            # extract object class names from h5 attrs (requires FTAG preprocessing)
            self.add_object_class_names()

            # if class weights are not specified, read them from class_dict
            for task in sc.model.model.init_args.tasks.init_args.modules:
                if not task["init_args"].get("use_class_dict"):
                    continue
                if (cd_fname := sc.data.class_dict) is None:
                    raise ValueError("use_class_dict=True requires class_dict to be specified")
                if task["init_args"]["loss"]["init_args"]["weight"] is not None:
                    raise ValueError(
                        "Class weights already specified, disable use_class_dict or remove weights."
                    )
                with open(cd_fname) as f:
                    class_dict = yaml.safe_load(f)
                input_name = task["init_args"]["input_name"]
                if task["init_args"]["label"] in class_dict[input_name]:
                    class_weights = class_dict[input_name][task["init_args"]["label"]]
                    task["init_args"]["loss"]["init_args"]["weight"] = class_weights
                else:
                    raise ValueError(
                        f"Label {task['init_args']['label']} not found in class_dict. "
                        "Use use_class_dict=False and specify class weights manually."
                    )

            # reduce precision to improve performance
            # don't do this during evaluation as you will get increased variation wrt Athena
            torch.set_float32_matmul_precision("medium")

            # get timestamped output dir for this run
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
            log = "trainer.logger"
            name = sc["name"]
            log_dir = sc["trainer.default_root_dir"]
            log_dir = Path(log_dir) if "s3://" not in log_dir else Path(log_dir[5:])
            # handle case where we re-use an existing config: use parent of timestampped dir
            try:
                datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")
                log_dir = log_dir.parent
            except ValueError:
                pass

            # set the timestampped dir
            dirname = f"{name}_{timestamp}"
            if "s3:/" not in sc["trainer.default_root_dir"]:
                log_dir_timestamp = str(Path(log_dir / dirname).resolve())
            else:
                log_dir_timestamp = str(Path(log_dir / dirname))
                log_dir_timestamp = "s3://" + log_dir_timestamp
            sc["trainer.default_root_dir"] = log_dir_timestamp
            if sc[log]:
                sc[f"{log}.init_args.save_dir"] = log_dir_timestamp

            # run git checks
            if not sc["force"] and not sc.trainer.fast_dev_run:
                path = Path(__file__).parent
                check_for_uncommitted_changes(path)
                if sc["tag"]:
                    create_and_push_tag(
                        path,
                        "atlas-flavor-tagging-tools/algorithms/salt",
                        dirname,
                        "automated salt tag",
                    )

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
                sc["ckpt_path"] = get_best_epoch(Path(config[0].rel_path))

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

    def add_object_class_names(self) -> None:
        # add flavour_label class names to global object classification task, if it exists
        sc = self.config[self.subcommand] if self.subcommand else self.config
        for task in sc.model.model.init_args.tasks.init_args.modules:
            t_args = task.init_args
            if not (
                t_args.name == f"{sc.data.global_object}_classification"
                and t_args.label == "flavour_label"
                and t_args.class_names is None
            ):
                return
            name = sc.data.input_map[t_args.input_name] if sc.data.input_map else t_args.input_name
            with h5py.File(sc.data.train_file) as f:
                if t_args.label in f[name].attrs:
                    t_args.class_names = f[name].attrs[t_args.label]
                else:
                    raise ValueError(
                        f"'{t_args.label}' not found in the h5 attrs of group '{name}' in file "
                        f"{sc.data.train_file}. Specify class_names manually."
                    )
