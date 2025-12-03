import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
    """Serialize a PyTorch tensor to a plain Python list (for YAML/JSON).

    Parameters
    ----------
    x : torch.Tensor
        Tensor to serialize.

    Returns
    -------
    list
        The tensor converted to a nested Python list.
    """
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
    """Deserialize a plain Python list into a PyTorch tensor.

    Parameters
    ----------
    x : list
        Nested list representation of a tensor.

    Returns
    -------
    torch.Tensor
        Tensor reconstructed from the list.
    """
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


def get_best_epoch(config_path: Path) -> str:
    """Find the checkpoint with the lowest validation loss for a training run.

    This scans the sibling ``ckpts/`` directory of ``config_path`` for ``*.ckpt`` files and
    picks the one with the smallest ``loss=...`` value embedded in the filename.

    Parameters
    ----------
    config_path : Path
        Path to the saved training config file (``config.yaml``).

    Returns
    -------
    str
        Absolute or relative string path to the best checkpoint file.
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
    """Lightning CLI wrapper that wires SALT configuration conveniences.

    This class adds:
    - Argument links between high-level fields and nested config fields.
    - Automatic label collection for tasks (including MaskFormer object/constituent support).
    - Optional model compilation for speed.
    - Git cleanliness checks and optional tagging on training starts.
    - Output directory normalization and timestamping.
    - Best-epoch checkpoint selection for testing.
    """

    def apply_links(self, parser: Any) -> None:
        """Link top-level CLI arguments to nested config entries.

        Parameters
        ----------
        parser : Any
            The Lightning/JsonArg parser instance whose arguments should be linked.
        """
        parser.link_arguments("name", "trainer.logger.init_args.experiment_name")
        parser.link_arguments("name", "model.name")
        parser.link_arguments("data.global_object", "model.global_object")

    def add_arguments_to_parser(self, parser: Any) -> None:
        """Add SALT-specific CLI arguments to the parser.

        Parameters
        ----------
        parser : Any
            The Lightning/JsonArg parser instance to extend with additional arguments.
        """
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

    def fit(self, model: Any, **kwargs: Any) -> None:
        """Run training with optional compilation.

        If ``--compile`` is set, the wrapped model module (``model.model``) is compiled
        before training.

        Parameters
        ----------
        model : Any
            The LightningModule wrapper instance created by the CLI.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`pytorch_lightning.Trainer.fit`.
        """
        if self.config[self.subcommand]["compile"]:
            # unfortunately compiling in place doesn't work
            # https://github.com/pytorch/pytorch/pull/97565
            # https://github.com/pytorch/pytorch/issues/101107
            model.model = torch.compile(model.model)
        self.trainer.fit(model, **kwargs)

    def before_instantiate_classes(self) -> None:
        """Perform automatic configuration/patches prior to class instantiation.

        This hook:
        - Collects all labels required by configured tasks (including MaskFormer).
        - Populates normalization config onto the model section.
        - Injects variable/global object info into init nets and featurewise nets.
        - Optionally loads class names from HDF5 attributes and class weights from a class dict.
        - Adjusts precision and output directories.
        - Runs git checks and optional tagging for training runs.
        - Normalizes test-time configuration and selects the best checkpoint automatically.

        Raises
        ------
        ValueError
            If the mf_config is not in the data config
            If the constituent name is not in the data labels
            If no trainer device is set
        """
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
                if (
                    task.init_args.get("use_class_dict")
                    and (config.ckpt_path is None)
                    and (class_weights := self.get_class_weights_from_class_dict(task, config.data))
                ):
                    task.init_args.loss.init_args.weight = torch.Tensor(class_weights)

            # reduce precision to improve performance
            # don't do this during evaluation as you will get increased variation wrt Athena
            torch.set_float32_matmul_precision("medium")

            # Ensure that the output dir path has a suffix or a timestemp
            output_dir_path = self.get_output_dir_path(
                config.trainer.default_root_dir, config.name, config.log_suffix
            )

            # Set the default root dir to the output path
            config.trainer.default_root_dir = output_dir_path

            # Check the status of the logger in the config
            if config.trainer.logger:
                # Check if the comet api key is available
                comet_api_key = os.getenv("COMET_API_KEY")

                # If online is true but no API key is given, set offline to False
                if not comet_api_key:
                    config.trainer.logger.init_args["online"] = False

                # Setup the offline output directory
                os.environ["COMET_OFFLINE_DIRECTORY"] = output_dir_path
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)

            # run git checks
            if not config.force and not config.trainer.fast_dev_run:
                path = Path(__file__).parent
                check_for_uncommitted_changes(path)
                if config.tag:
                    create_and_push_tag(
                        path,
                        "aft/algorithms/salt",
                        Path(output_dir_path).stem,
                        "automated salt tag",
                    )

            # Set config overwrite
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
        """Accumulate label field names required by a task into ``labels``.

        Parameters
        ----------
        task : JsonNamespace
            The task configuration namespace (``init_args`` of the task).
        merge_dict : JsonNamespace | None
            Optional mapping of merged input names. If provided, labels are also
            replicated to these merged inputs.
        labels : dict
            Mutable dictionary mapping input stream name → list of label field names.
        """

        def collect_labels(task: JsonNamespace) -> list:
            """Collect label-like fields from a task config (helper).

            Parameters
            ----------
            task : JsonNamespace
                Task as json namespace instance

            Returns
            -------
            list
                List with the labels
            """
            labels: list = []
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
        """Read class names for object classification from HDF5 attributes if applicable.

        Only applies when the task targets ``flavour_label`` on the global object and
        the task has ``class_names=None``.

        Parameters
        ----------
        task : JsonNamespace
            Task configuration namespace (contains ``init_args``).
        data : JsonNamespace
            Data configuration namespace (contains ``train_file``, ``input_map`` etc.).

        Returns
        -------
        list or None
            List of class names if discovered; otherwise ``None``.

        Raises
        ------
        ValueError
            If the init_args label is not in the h5 attrs.
        """
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
        """Load class weights for a classification task from a class-dict YAML.

        Parameters
        ----------
        task : JsonNamespace
            Task configuration namespace (contains ``init_args`` and a loss block).
        data : JsonNamespace
            Data configuration namespace; must contain ``class_dict`` and optionally
            an ``input_map``.

        Returns
        -------
        list
            List of class weights aligned with the task's class order.

        Raises
        ------
        ValueError
            If the class dict path is missing, weights already provided, or label is absent.
        """
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
        """Build the run output directory path with either a suffix or a timestamp.

        If ``default_output_dir_path`` already ends with a timestamp-like suffix, strip it.

        Parameters
        ----------
        default_output_dir_path : str
            Base path (from trainer config) where logs/checkpoints should be written.
        name : str
            The run name (typically the model name).
        log_suffix : str | None
            Optional suffix to append. If ``None``, a timestamp of the form
            ``YYYYMMDD-THHMMSS`` is appended.

        Returns
        -------
        str
            Absolute path to the resulting output directory.
        """
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
