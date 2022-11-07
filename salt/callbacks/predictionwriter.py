from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from pytorch_lightning import Callback, LightningModule, Trainer

from salt.utils.arrays import join_structured_arrays


class PredictionWriter(Callback):
    def __init__(self, write_tracks: bool = False) -> None:
        """A callback to write test outputs to h5 file."""
        super().__init__()

        # list of variables to copy from test file
        self.jet_variables = [
            "pt",
            "eta",
            "HadronConeExclTruthLabelID",
            "n_tracks_loose",
            "n_truth_promptLepton",
        ]

        self.write_tracks = write_tracks
        self.track_cols = [
            "Pileup",
            "Fake",
            "Primary",
            "FromB",
            "FromBC",
            "FromC",
            "FromTau",
            "OtherSecondary",
        ]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        # inputs names
        self.jet = trainer.datamodule.inputs["jet"]
        self.track = trainer.datamodule.inputs["track"]

        # place to store intermediate outputs
        self.tasks = [t.name for t in pl_module.model.get_tasks()]
        self.outputs: dict = {t: [] for t in self.tasks}
        self.mask: list = []

        # get test dataset
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.file = self.ds.file
        self.num_jets = len(self.ds)

        # get jet class prediction column names
        train_file = trainer.datamodule.train_dataloader().dataset.file
        jet_classes = list(train_file[f"{self.jet}/labels"].attrs.values())[0]
        self.jet_cols = [f"salt_p{c[0]}" for c in jet_classes]

        # get output path
        out_dir = Path(trainer._ckpt_path).parent
        out_basename = str(Path(trainer._ckpt_path).stem)
        sample = str(Path(self.ds.filename).name).split("_")[2]
        fname = f"{out_basename}__test_{sample}.h5"
        self.out_path = Path(out_dir / fname)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dl_idx):
        # append test outputs
        [self.outputs[t].append(outputs[0][t]) for t in self.tasks]
        self.mask.append(outputs[1])

    def on_test_end(self, trainer, pl_module):
        # concat test batches
        outputs = {t: torch.cat(self.outputs[t]) for t in self.tasks}

        # softmax jet classification outputs
        jet_class_preds = torch.softmax(outputs["jet_classification"], dim=-1)

        # create output jet dataframe
        dtype = np.dtype([(n, "f2") for n in self.jet_cols])
        jets = u2s(jet_class_preds.cpu().numpy(), dtype)
        jets2 = self.file[self.jet].fields(self.jet_variables)[: self.num_jets]
        jets = join_structured_arrays((jets, jets2))

        if self.write_tracks and "track_classification" in pl_module.tasks:
            # masked softmax
            t = outputs["track_classification"].cpu()
            mask = torch.cat(self.mask).unsqueeze(dim=-1).cpu()
            t = F.softmax(t.masked_fill(mask, float("-inf")), dim=-1).numpy()

            # convert to structured array
            t = t.view(dtype=np.dtype([(name, "f4") for name in self.track_cols]))
            t = t.reshape(t.shape[0], t.shape[1])

            # add other other vars to the array
            track_vars = ["truthOriginLabel", "truthVertexIndex"]
            t2 = self.file[self.track].fields(track_vars)[: self.num_jets]
            t = join_structured_arrays((t, t2))

        # write to h5 file
        print("\n" + "-" * 100)
        if self.out_path.exists():
            print("Warning! Overwriting existing file.")

        with h5py.File(self.out_path, "w") as f:
            self.create_dataset(f, jets, self.jet)

            if self.write_tracks and "track_classification" in pl_module.tasks:
                self.create_dataset(f, t, self.track)

        print("Created output file", self.out_path)
        print("-" * 100, "\n")

    def create_dataset(self, f, a, name, half_precision=True):
        # convert down to float16
        if half_precision:

            def half(t):
                t = np.dtype(t)
                if t.kind == "f" and t.itemsize == 2:
                    return "f2"
                return t

            a = np.array(a, dtype=[(n, half(t)) for n, t in a.dtype.descr])

        # write
        f.create_dataset(name, data=a, compression="lzf")
