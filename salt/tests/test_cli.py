import pathlib

import pytest
import yaml
from jsonargparse import Namespace

from salt.callbacks import SaveConfigCallback
from salt.data.datamodules import SaltDataModule
from salt.modelwrapper import ModelWrapper
from salt.utils.cli import SaltCLI, get_best_epoch
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


def test_get_best_epoch(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch=000-loss=0.50.ckpt").touch()
    (ckpt_dir / "epoch=001-loss=0.25.ckpt").touch()
    best = get_best_epoch(tmp_path / "config.yaml")
    assert best.endswith("epoch=001-loss=0.25.ckpt")


def test_get_best_epoch_negative_loss(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch=000-loss=0.50.ckpt").touch()
    (ckpt_dir / "epoch=001-loss=-0.25.ckpt").touch()
    best = get_best_epoch(tmp_path / "config.yaml")
    assert best.endswith("epoch=001-loss=-0.25.ckpt")


def class_weights_from_dict(tmp_path, weights, label_map=None):
    cd_path = tmp_path / "class_dict.yaml"
    cd_path.write_text(yaml.dump({"jets": {"label": weights}}))
    loss = Namespace(init_args=Namespace(weight=None))
    task = Namespace(
        init_args=Namespace(input_name="jets", label="label", label_map=label_map, loss=loss)
    )
    data = Namespace(class_dict=str(cd_path), input_map=None)
    return SaltCLI.get_class_weights_from_class_dict(task, data)


def test_class_dict_list(tmp_path):
    assert class_weights_from_dict(tmp_path, [1.0, 4.0, 2.0]) == [1.0, 4.0, 2.0]


def test_class_dict_mapping(tmp_path):
    weights = class_weights_from_dict(tmp_path, {0: 1.0, 1: 4.0, 2: 2.0})
    assert weights == pytest.approx([1.0, 4.0, 2.0])


def test_class_dict_mapping_label_map(tmp_path):
    # charge-style task: several PDG IDs per class, 5554 is mapped but not in the class dict
    weights = class_weights_from_dict(
        tmp_path,
        {-5122: 8.0, -511: 2.0, -15: 4.0, 0: 1.0, 15: 4.0, 511: 2.0, 5122: 8.0},
        label_map={511: 1, -511: 0, 5122: 0, -5122: 1, 5554: 0, 0: 2, 15: 2, -15: 2},
    )
    assert weights == pytest.approx([1.6, 1.6, 2 / 3])


def test_class_dict_mapping_missing_class(tmp_path):
    with pytest.raises(ValueError, match="No class_dict entries"):
        class_weights_from_dict(tmp_path, {0: 1.0, 2: 2.0})


def test_class_dict_mapping_label_map_missing_class(tmp_path):
    with pytest.raises(ValueError, match="No class_dict entries"):
        class_weights_from_dict(tmp_path, {0: 1.0, 4: 4.0}, label_map={0: 0, 4: 1, 5: 2})
