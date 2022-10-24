from pathlib import Path

import h5py
import pandas as pd
import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class PredictionWriter(Callback):
    def __init__(self) -> None:
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

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # place to store intermediate outputs
        self.tasks = pl_module.tasks
        self.outputs: dict = {t: [] for t in self.tasks}

        # get test dataset
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.file = self.ds.file
        self.num_jets = len(self.ds)

        # get flav prediction column names
        train_file = trainer.datamodule.train_dataloader().dataset.file
        class_names = list(train_file["jets/labels"].attrs.values())[0]
        self.cols = [f"salt_p{c[0]}" for c in class_names]

        # get output path
        out_dir = Path(trainer._ckpt_path).parent
        out_basename = str(Path(trainer._ckpt_path).stem)
        sample = str(Path(self.ds.filename).name).split("_")[2]
        fname = f"{out_basename}__test_{sample}.h5"
        self.out_path = Path(out_dir / fname)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dl_idx):
        # append test outputs
        [self.outputs[t].append(outputs[t]) for t in self.tasks]

    def on_test_end(self, trainer, pl_module):
        # concat test batches
        self.outputs = {t: torch.cat(self.outputs[t]) for t in self.tasks}

        # softmax jet classification outputs
        flav_preds = torch.softmax(self.outputs["jet_classification"], dim=1)

        # create output dataframe
        jet_df = pd.DataFrame(flav_preds.cpu().numpy(), columns=self.cols)
        for v in self.jet_variables:
            jet_df[v] = self.file["jets"].fields(v)[: self.num_jets]

        # write to h5 file
        # TODO: warn on overwrite
        print("\n" + "-" * 100)
        print("Created output file", self.out_path)
        print("-" * 100, "\n")
        with h5py.File(self.out_path, "w") as f:
            self.create_dataset(f, jet_df, "jets")

    def create_dataset(self, f, df, name):
        f.create_dataset(name, data=df.to_records(index=False), compression="lzf")
