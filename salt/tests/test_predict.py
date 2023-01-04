import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.main import main
from salt.utils.arrays import join_structured_arrays

w = "ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning:"


@pytest.mark.filterwarnings(w)
@pytest.mark.depends(on=["train"])
class TestPredict:
    test_dir = Path(__file__).resolve().parent
    tmp_dir = Path("/tmp/salt_tests/")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    h5_fname = tmp_dir / "test_predict_ttbar_blah.h5"
    config_path = None
    sd_path = None

    @classmethod
    def setup_class(cls):
        """Setup tests (runs once)."""
        cls.generate_test_input(cls.h5_fname)

    def predict_template(self, args=None) -> None:
        if args is None:
            args = []

        # setup args
        args += [f"--config={self.config_path}"]
        args += [f"--data.test_file={self.h5_fname}"]
        args += ["--data.num_jets_test=1000"]

        # run
        main(args)

    def test_predict(self) -> None:
        args = ["test"]
        self.predict_template(args)

    @classmethod
    def generate_test_input(cls, fpath: Path) -> None:
        """Generate dummy test input file at the specified path.

        Parameters
        ----------
        fpath : Path
            Path to test h5 file.
        """

        # look in the test dir for a training run
        dirs = [x for x in cls.tmp_dir.iterdir() if x.is_dir()]
        test_dir = [x for x in dirs if Path(x / "config.yaml").exists()][0]
        config_path = Path(test_dir / "config.yaml")
        sd_path = [x for x in test_dir.iterdir() if x.suffix == ".json"][0]

        cls.test_dir = test_dir
        cls.config_path = config_path
        cls.sd_path = sd_path

        with open(cls.sd_path) as f:
            sd = json.load(f)

        # TODO: avoid redefinition (by making these vars configurable?)
        jet_vars = [
            "pt",
            "eta",
            "HadronConeExclTruthLabelID",
            "n_tracks_loose",
            "n_truth_promptLepton",
        ]

        # settings
        n_jets = 1000
        jet_features = len(jet_vars)
        n_tracks_per_jet = 40
        track_features = 21

        # setup jets
        shapes_jets = {
            "inputs": [n_jets, jet_features],
        }

        # setup tracks
        shapes_tracks = {
            "inputs": [n_jets, n_tracks_per_jet, track_features],
            "valid": [n_jets, n_tracks_per_jet],
        }

        rng = np.random.default_rng(seed=65)

        # setup jets
        jets_dtype = np.dtype([(n, "f4") for n in jet_vars])
        jets = rng.random(shapes_jets["inputs"])
        jets = u2s(jets, jets_dtype)

        # setup tracks
        tracks_dtype = np.dtype([(n, "f4") for n in sd["tracks"]])
        tracks = rng.random(shapes_tracks["inputs"])
        tracks = u2s(tracks, tracks_dtype)
        valid = rng.random(shapes_tracks["valid"])
        valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
        tracks = join_structured_arrays([tracks, valid])

        with h5py.File(fpath, "w") as f:
            f.create_dataset("jets", data=jets)
            f.create_dataset("tracks", data=tracks)
