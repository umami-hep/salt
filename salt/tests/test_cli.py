import pathlib
import unittest
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
from lightning.pytorch.cli import ArgsType

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import SaltDataModule
from salt.modelwrapper import ModelWrapper
from salt.utils.cli import SaltCLI
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict


class CliTestCase(unittest.TestCase):
    train_h5_path = "./dummy_train_inputs.h5"
    nd_path = "./dummy_norm_dict.yaml"
    cd_path = "./dummy_class_dict.yaml"
    tmp_files: ClassVar[list[str]] = [train_h5_path, nd_path, cd_path]

    def setUp(self):
        """Writing dummy data for setup."""
        write_dummy_norm_dict(self.nd_path, self.cd_path)
        write_dummy_file(self.train_h5_path, self.nd_path, make_xbb=True)

    def test_add_object_class_names(self, args: ArgsType = None) -> None:
        """Instanting cli and calling related config should throw no errors."""
        args = [f"--data.train_file={self.train_h5_path}", "--trainer.default_root_dir=./"]

        config_dir = pathlib.Path(__file__).parent / "configs"
        print(config_dir)
        cli = SaltCLI(
            model_class=ModelWrapper,
            datamodule_class=SaltDataModule,
            save_config_callback=SaveConfigCallback,
            run=False,
            args=args,
            parser_kwargs={
                "default_env": True,
                "default_config_files": [f"{config_dir}/test_config.yaml"],
            },
        )
        cli.add_object_class_names()

    def test_add_object_class_names_labellerFalse(self, args: ArgsType = None) -> None:
        """Labels should return the same value as class names when labeller False."""
        args = ["--data.train_file=./dummy_train_inputs.h5", "--trainer.default_root_dir=./"]

        config_dir = pathlib.Path(__file__).parent / "configs"
        print(config_dir)
        cli = SaltCLI(
            model_class=ModelWrapper,
            datamodule_class=SaltDataModule,
            save_config_callback=SaveConfigCallback,
            run=False,
            args=args,
            parser_kwargs={
                "default_env": True,
                "default_config_files": [f"{config_dir}/test_config.yaml"],
            },
        )
        f = h5py.File((self.train_h5_path), "r")
        labels = f["jets"].attrs["flavour_label"]
        tasks = cli.config.model.model.init_args.tasks.init_args
        class_names = tasks.modules[0].init_args.class_names
        assert np.all(labels == class_names)

    def test_add_object_class_names_labellerTrue(self, args: ArgsType = None) -> None:
        """Class names should override labels when labeller True."""
        config_class_names = ["hbb", "hcc", "top", "qcdnonbb", "qcdbb"]
        args = [
            "--data.train_file=./dummy_train_inputs.h5",
            "--trainer.default_root_dir=./",
            "--model.model.init_args.tasks.init_args.modules.init_args.use_labeller=True",
            f"--model.model.init_args.tasks.init_args.modules.init_args.class_names={config_class_names}",
        ]
        config_dir = pathlib.Path(__file__).parent / "configs"
        cli = SaltCLI(
            model_class=ModelWrapper,
            datamodule_class=SaltDataModule,
            save_config_callback=SaveConfigCallback,
            run=False,
            args=args,
            parser_kwargs={
                "default_env": True,
                "default_config_files": [f"{config_dir}/test_config.yaml"],
            },
        )

        f = h5py.File(("./dummy_train_inputs.h5"), "r")
        labels = f["jets"].attrs["flavour_label"]
        tasks = cli.config.model.model.init_args.tasks.init_args
        class_names = tasks.modules[0].init_args.class_names
        assert labels.shape[0] != len(class_names)
        assert np.all(labels == ["hbb", "hcc", "top", "qcd"])
        assert np.all(class_names == ["hbb", "hcc", "top", "qcdnonbb", "qcdbb"])
        assert config_class_names == class_names

    def tearDown(self):
        """Deleting dummy data for tearDown."""
        for f in self.tmp_files:
            Path(f).unlink()
