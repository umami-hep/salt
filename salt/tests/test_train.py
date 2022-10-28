import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.main import main

test_dir = Path(__file__).resolve().parent
config_path = Path(test_dir.parent / "configs")
tmp_dir = Path("/tmp/salt_tests/")
tmp_dir.mkdir(parents=True, exist_ok=True)
w = "ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning:"
h5_fname = Path(tmp_dir / "test.h5")
sd_fname = Path(tmp_dir / "scale-dict.json")


def train_template(args=None) -> None:
    if args is None:
        args = []

    # setup args
    args += [f"--config={config_path}/simple.yaml"]
    args += [f"--data.scale_dict={sd_fname}"]
    args += [f"--data.train_file={h5_fname}"]
    args += [f"--data.val_file={h5_fname}"]
    args += ["--data.num_jets_train=2500"]
    args += ["--data.num_jets_val=2500"]
    args += ["--data.batch_size=100"]
    args += ["--data.num_workers=0"]
    args += ["--trainer.logger.offline=True"]
    args += ["--trainer.max_epochs=2"]
    args += ["--trainer.accelerator=cpu"]
    args += [f"--trainer.logger.save_dir={tmp_dir}"]
    args += [f"--trainer.default_root_dir={tmp_dir}"]

    # run
    main(args)


@pytest.mark.filterwarnings(w)
def test_train_batched() -> None:
    args = ["fit", "--data.batched_read=True"]
    train_template(args)


@pytest.mark.filterwarnings(w)
def test_train_unbatched() -> None:
    args = ["fit", "--data.batched_read=False"]
    train_template(args)


@pytest.mark.filterwarnings(w)
def test_train_movefilestemp() -> None:
    args = ["fit", "--data.move_files_temp=/dev/shm/test_files"]
    train_template(args)


def setup_module():
    """setup any state specific to the execution of the given module."""
    # get test file
    generate_test_input(h5_fname)


def generate_test_input(fpath: Path) -> None:
    """Generate test input file.

    Parameters
    ----------
    fpath : Path
        Path to test h5 file.
    """

    # settings
    n_jets = 2500
    jet_features = 2
    n_tracks_per_jet = 40
    track_features = 23

    # setup jets
    key_words_jets = ["inputs", "labels"]
    shapes_jets = [
        [n_jets, jet_features],
        [n_jets],
    ]

    # setup tracks
    key_words_tracks = ["inputs", "labels", "valid"]
    shapes_tracks = [
        [n_jets, n_tracks_per_jet, track_features],
        [n_jets, n_tracks_per_jet, jet_features],
        [n_jets, n_tracks_per_jet],
    ]

    # create h5 file
    rng = np.random.default_rng(seed=65)
    with h5py.File(fpath, "w") as f:
        # write jets
        g_jets = f.create_group("jets")
        for key, shape in zip(key_words_jets, shapes_jets):
            arr = rng.random(shape)
            g_jets.create_dataset(key, data=arr)

        # write tracks
        g_tracks = f.create_group("tracks_loose")
        for key, shape in zip(key_words_tracks, shapes_tracks):
            arr = rng.random(shape)
            if key == "valid":
                arr = arr.astype(bool)
                arr[:, 10:] = False
            g_tracks.create_dataset(key, data=arr)
            if key == "inputs":
                g_tracks[key].attrs['"tracks_variables"'] = ["jet_eta", "pt"]

        # create scale dict file
        with open(sd_fname, "w") as f:
            f.write(json.dumps({"test": "test"}))
