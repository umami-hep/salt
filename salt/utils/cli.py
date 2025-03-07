import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from ftag.git_check import check_for_uncommitted_changes, create_and_push_tag
from jsonargparse import Namespace as JsonNamespace
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
        parser.add_argument(
            "-oc", "--overwrite_config", action="store_true", help="Overwrite config file."
        )
        parser.add_argument(
            "-ls",
            "--log_suffix",
            default=None,
            type=str,
            help="Appended to model name to create the log directory.",
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
        config = self.config[self.subcommand] if self.subcommand else self.config
        sc_tasks = config.model.model.init_args.tasks.init_args.modules

        labels: dict = {}
        for task in sc_tasks:
            assert "Task" in task["class_path"]
            self.collect_labels_from_task(
                task["init_args"], config.model.model.init_args.get("merge_dict"), labels
            )

        if config.model.model.init_args.get("mask_decoder"):
            if not (maskformer_config := config.data.get("mf_config")):
                raise ValueError("Mask decoder requires 'mf_config' in data config.")
            if maskformer_config.constituent.name not in labels:
                raise ValueError(
                    f"The constituent name {maskformer_config.constituent.name} is not in the"
                    " data labels. Ensure that the constituent name is in the input_map of the"
                    " data config."
                )

            # Needed in case no tasks other than mask prediction/classification
            if "objects" not in labels:
                labels["objects"] = []
            labels["objects"] += [
                maskformer_config.object.id_label,
                maskformer_config.object.class_label,
            ]
            labels[maskformer_config.constituent.name] += [maskformer_config.constituent.id_label]

        config.data.labels = labels

        # add norm
        config.model.norm_config = {
            "norm_dict": config.data.norm_dict,
            "variables": config.data.variables,
            "global_object": config.data.global_object,
            "input_map": config.data.input_map,
        }

        if self.subcommand == "fit" or self.subcommand is None:
            # add variables to inititialization networks
            for init_net in config.model.model.init_args.init_nets:
                init_net["variables"] = config.data.variables
                init_net["global_object"] = config.data.global_object

            # add variables to feature-wise networks
            if config.model.model.init_args.featurewise_nets:
                for featurewise_net in config.model.model.init_args.featurewise_nets:
                    featurewise_net["variables"] = config.data.variables

            for task in sc_tasks:
                # extract object class names from h5 attrs (requires FTAG preprocessing)
                if class_names := self.get_object_class_names(task, config.data):
                    task.init_args.class_names = class_names

                # if class weights are not specified read them from class_dict
                if task.init_args.get("use_class_dict") and (
                    class_weights := self.get_class_weights_from_class_dict(task, config.data)
                ):
                    task.init_args.loss.init_args.weight = torch.Tensor(class_weights)

            # reduce precision to improve performance
            # don't do this during evaluation as you will get increased variation wrt Athena
            torch.set_float32_matmul_precision("medium")

            output_dir_path = self.get_output_dir_path(
                config.trainer.default_root_dir, config.name, config.log_suffix
            )
            config.trainer.default_root_dir = output_dir_path
            if config.trainer.logger:
                config.trainer.logger.init_args.save_dir = output_dir_path

            # run git checks
            if not config.force and not config.trainer.fast_dev_run:
                path = Path(__file__).parent
                check_for_uncommitted_changes(path)
                if config.tag:
                    create_and_push_tag(
                        path,
                        "atlas-flavor-tagging-tools/algorithms/salt",
                        Path(output_dir_path).stem,
                        "automated salt tag",
                    )

            if config.overwrite_config:
                self.save_config_kwargs["overwrite"] = True

        if self.subcommand == "test":
            print("\n" + "-" * 100)

            # no logger, callback refresh rate 1 for testing
            self.save_config_callback = None
            config.trainer.logger = False
            for callback in config.trainer.callbacks:
                if hasattr(callback, "init_args") and hasattr(callback.init_args, "refresh_rate"):
                    callback.init_args.refresh_rate = 1

            # use best epoch for testing
            if not config.ckpt_path:
                assert len(config.config) == 1
                config.ckpt_path = get_best_epoch(Path(config.config[0].rel_path))

            if isinstance(config.trainer.devices, str | int) and int(config.trainer.devices) > 1:
                print("Setting --trainer.devices=1")
                config.trainer.devices = "1"
            elif isinstance(config.trainer.devices, list) and len(config.trainer.devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            config.data.move_files_temp = None

            print("-" * 100 + "\n")

    @staticmethod
    def collect_labels_from_task(
        task: JsonNamespace,
        merge_dict: JsonNamespace | None,
        labels: dict,
    ) -> None:
        def collect_labels(task: JsonNamespace):
            labels = []
            if label := task.get("label"):
                labels.append(label)
            if weight := task.get("sample_weight"):
                labels.append(weight)
            if targets := task.get("targets"):
                labels.extend(listify(targets))

            if denominators := task.get("target_denominators"):
                labels.extend(listify(denominators))

            return labels

        if task["input_name"] in labels:
            labels[task["input_name"]].extend(collect_labels(task))
        else:
            labels[task["input_name"]] = collect_labels(task)

        if not merge_dict or task["input_name"] not in merge_dict:
            return

        for label_name in merge_dict[task["input_name"]]:
            if label_name in labels:
                labels[label_name].extend(collect_labels(task))
            else:
                labels[label_name] = collect_labels(task)

    @staticmethod
    def get_object_class_names(task: JsonNamespace, data: JsonNamespace) -> list | None:
        if not (
            task.init_args.name == f"{data.global_object}_classification"
            and task.init_args.label == "flavour_label"
            and task.init_args.class_names is None
        ):
            return None

        name = (
            data.input_map[task.init_args.input_name]
            if data.input_map
            else task.init_args.input_name
        )
        with h5py.File(data.train_file) as f:
            if task.init_args.label not in f[name].attrs:
                raise ValueError(
                    f"'{task.init_args.label}' not found in the h5 attrs of group '{name}' in file "
                    f"{data.train_file}. Specify class_names manually."
                )

            return list(f[name].attrs[task.init_args.label])

    @staticmethod
    def get_class_weights_from_class_dict(
        task: JsonNamespace,
        data: JsonNamespace,
    ) -> list:
        if (class_dict_path := data.class_dict) is None:
            raise ValueError("use_class_dict=True requires class_dict to be specified")

        if task.init_args.loss.init_args.weight is not None:
            raise ValueError(
                "Class weights already specified, disable use_class_dict or remove weights."
            )

        with open(class_dict_path) as f:
            class_dict = yaml.safe_load(f)

        input_name = task.init_args.input_name
        if data.input_map is not None:
            input_name = data.input_map[input_name]

        if task.init_args.label not in class_dict[input_name]:
            raise ValueError(
                f"Label {task.init_args.label} not found in class_dict. "
                "Use use_class_dict=False and specify class weights manually."
            )

        return class_dict[input_name][task.init_args.label]

    @staticmethod
    def get_output_dir_path(default_output_dir_path: str, name: str, log_suffix: str | None) -> str:
        log_dir = Path(default_output_dir_path)
        try:
            datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")
            log_dir = log_dir.parent
        except ValueError:
            pass

        if log_suffix is not None:
            dirname = f"{name}_{log_suffix}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
            dirname = f"{name}_{timestamp}"

        return str(Path(log_dir / dirname).resolve())
