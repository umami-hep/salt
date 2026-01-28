import pathlib

import pytest

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import SaltDataModule
from salt.modelwrapper import ModelWrapper
from salt.utils.cli import SaltCLI
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict


@pytest.fixture
def test_files(tmp_path):
    """Create dummy data files in a temporary directory."""
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    nd_path = tmp_path / "dummy_norm_dict.yaml"
    cd_path = tmp_path / "dummy_class_dict.yaml"

    write_dummy_norm_dict(nd_path, cd_path)
    write_dummy_file(train_h5_path, nd_path, make_xbb=True)

    return {
        "train_h5_path": train_h5_path,
        "nd_path": nd_path,
        "cd_path": cd_path,
        "tmpdir": tmp_path,
    }


def test_initialization(test_files) -> None:
    """Instanting cli and calling related config should throw no errors."""
    args = [
        f"--data.train_file={test_files['train_h5_path']}",
        f"--trainer.default_root_dir={test_files['tmpdir']}",
        f"--data.norm_dict={test_files['nd_path']}",
        f"--data.class_dict={test_files['cd_path']}",
    ]

    config_dir = pathlib.Path(__file__).parent / "configs"
    _ = SaltCLI(
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
