import json
from pathlib import Path

import h5py
import pytest
from numpy.random import default_rng

from salt.main import main
from salt.utils.inputs import generate_scale_dict

w = "ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning:"


@pytest.mark.filterwarnings(w)
class TestTrain:
    test_dir = Path(__file__).resolve().parent
    config_path = Path(test_dir.parent / "configs")
    tmp_dir = Path("/tmp/salt_tests/")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    h5_fname = Path(tmp_dir / "test_train.h5")
    sd_fname = Path(tmp_dir / "scale-dict.json")

    @classmethod
    def setup_class(cls):
        """Setup tests (runs once)."""
        cls.generate_train_input(cls.h5_fname)

    def train_template(self, args=None, model="gnn") -> None:
        if args is None:
            args = []

        # setup args
        args += [f"--config={self.config_path}/base.yaml"]
        args += [f"--config={self.config_path}/{model}.yaml"]
        args += [f"--data.scale_dict={self.sd_fname}"]
        args += [f"--data.train_file={self.h5_fname}"]
        args += [f"--data.val_file={self.h5_fname}"]
        args += ["--data.num_jets_train=1000"]
        args += ["--data.num_jets_val=1000"]
        args += ["--data.batch_size=100"]
        args += ["--data.num_workers=0"]
        args += ["--trainer.logger.offline=True"]
        args += ["--trainer.max_epochs=2"]
        args += ["--trainer.accelerator=cpu"]
        args += [f"--trainer.logger.save_dir={self.tmp_dir}"]
        args += [f"--trainer.default_root_dir={self.tmp_dir}"]

        # run
        main(args)

    @pytest.mark.depends(name="train")
    def test_train_batched(self) -> None:
        args = ["fit", "--data.batched_read=True"]
        self.train_template(args)

    def test_train_unbatched(self) -> None:
        args = ["fit", "--data.batched_read=False"]
        self.train_template(args)

    def test_train_dev(self) -> None:
        args = ["fit", "--trainer.fast_dev_run=10"]
        self.train_template(args)

    def test_train_movefilestemp(self) -> None:
        tmp_path = Path(self.tmp_dir / "dev" / "shm")
        args = ["fit", f"--data.move_files_temp={tmp_path}"]
        self.train_template(args)
        assert not Path(tmp_path).exists()

    def test_train_distributed(self) -> None:
        args = ["fit", "--trainer.devices=2", "--data.num_workers=2"]
        self.train_template(args)

    def test_train_dips(self) -> None:
        args = ["fit"]
        self.train_template(args, model="dips")

    def test_train_regression(self) -> None:
        args = ["fit"]
        self.train_template(args, model="regression")

    @classmethod
    def generate_train_input(cls, fpath: Path) -> None:
        """Generate dummy training input file at the specified path.

        Parameters
        ----------
        fpath : Path
            Path to test h5 file.
        """

        # settings
        n_jets = 1000
        jet_features = 2
        n_tracks_per_jet = 40
        track_features = 21
        tracks = "tracks_loose"

        # setup jets
        shapes_jets = {
            "inputs": [n_jets, jet_features],
            "labels": [n_jets],
        }

        # setup tracks
        shapes_tracks = {
            "inputs": [n_jets, n_tracks_per_jet, jet_features + track_features],
            "labels": [n_jets, n_tracks_per_jet, 2],
            "valid": [n_jets, n_tracks_per_jet],
        }

        # create h5 file
        rng = default_rng(seed=42)
        with h5py.File(fpath, "w") as f:
            # write jets
            g_jets = f.create_group("jets")
            for key, shape in shapes_jets.items():
                arr = rng.random(shape)
                g_jets.create_dataset(key, data=arr)
                if key == "labels":
                    g_jets[key].attrs['"labels"'] = ["bjets", "cjets", "ujets"]

            # write tracks
            g_tracks = f.create_group(tracks)
            for key, shape in shapes_tracks.items():
                arr = rng.random(shape)
                if key == "valid":
                    arr = arr.astype(bool)
                    arr[:, 10:] = False
                if key == "labels":
                    arr = rng.integers(0, 8, shape)
                g_tracks.create_dataset(key, data=arr)
                if key == "inputs":
                    g_tracks[key].attrs[f"{tracks}_variables"] = ["jet_eta", "pt"]
                if key == "labels":
                    g_tracks[key].attrs[f"{tracks}_truth_variables"] = [
                        "truthOriginLabel",
                        "truthVertexIndex",
                    ]

            # create scale dict file
            with open(cls.sd_fname, "w") as f:
                f.write(json.dumps(generate_scale_dict(jet_features, track_features)))
