from pathlib import Path

import h5py
import numpy as np
import torch
from ftag import Flavours
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.models.task import ClassificationTask, RegressionTaskBase
from salt.utils.array_utils import join_structured_arrays
from salt.utils.union_find import get_node_assignment


class PredictionWriter(Callback):
    def __init__(
        self,
        jet_variables: list | None = None,
        track_variables: list | None = None,
        write_tracks: bool = False,
        half_precision: bool = False,
        jet_classes: list | None = None,
    ) -> None:
        """Write test outputs to h5 file.

        Parameters
        ----------
        jet_variables : list
            List of jet variables to copy from test file
        track_variables : list
            List of track variables to copy from test file
        write_tracks : bool
            If true, write track outputs to file
        half_precision : bool
            If true, write outputs at half precision
        jet_classes : list
            List of flavour names with the index corresponding to the label values
        """
        super().__init__()

        self.jet_variables = jet_variables
        self.track_variables = track_variables
        self.write_tracks = write_tracks
        self.half_precision = half_precision
        self.jet_classes = jet_classes
        self.track_origin_cols = [
            "Pileup",
            "Fake",
            "Primary",
            "FromB",
            "FromBC",
            "FromC",
            "FromTau",
            "OtherSecondary",
        ]
        self.track_type_cols = [
            "NoTruth",
            "Other",
            "Pion",
            "Kaon",
            "Electron",
            "Muon",
        ]

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        # some basic info
        self.trainer = trainer
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_suff = trainer.datamodule.test_suff
        self.file = self.ds.file
        self.num_jets = len(self.ds)
        self.norm_dict = self.ds.scaler.norm_dict

        # inputs names
        self.jet = self.ds.input_names["jet"]
        self.track = self.ds.input_names.get("track")

        # place to store intermediate outputs
        self.tasks = module.model.tasks
        self.outputs: dict = {task: [] for task in self.tasks}
        self.mask: list = []

        # get jet class names for output file
        for task in self.tasks:
            if task.name != "jet_classification":
                continue
            if self.jet_classes is None:
                if task.class_names is not None:
                    self.jet_classes = task.class_names
                else:
                    raise ValueError(
                        "Couldn't infer jet classes from model. "
                        "Please provide a list of jet classes."
                    )
            jet_px = [f"{Flavours[c].px}" if c in Flavours else f"p{c}" for c in self.jet_classes]
            self.jet_class_cols = [f"{module.name}_{px}" for px in jet_px]

        # decide whether to write tracks
        self.task_names = [task.name for task in self.tasks]
        self.write_tracks = self.write_tracks and any(
            t in self.task_names for t in ["track_origin", "track_vertexing", "jet_regression"]
        )

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        stem = str(Path(self.ds.filename).stem)
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        suffix = f"_{self.test_suff}" if self.test_suff is not None else ""
        return Path(out_dir / f"{out_basename}__test_{sample}{suffix}.h5")

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):
        preds = outputs
        inputs, masks, labels = batch
        this_batch = False

        for task in self.tasks:
            task_preds = preds[task.name]
            task_mask = masks.get(task.input_type)
            if isinstance(task, ClassificationTask):
                task_preds = task.run_inference(task_preds, task_mask)
            elif issubclass(type(task), RegressionTaskBase):
                task_preds = task.run_inference(task_preds, labels)
            self.outputs[task].append(task_preds.cpu())

            if self.write_tracks and task_mask is not None and not this_batch:
                self.mask.append(task_mask.cpu())
                this_batch = True

    def on_test_end(self, trainer, module):
        # concat test batches
        outputs = {task: torch.cat(self.outputs[task]) for task in self.tasks}

        # handle jets
        precision_str = "f2" if self.half_precision else "f4"
        jet_outs = []
        for task in self.tasks:
            if task.input_type != "jet":
                continue

            if task.name == "jet_classification":
                dtype = np.dtype([(n, precision_str) for n in self.jet_class_cols])
                jets = outputs[task].float().cpu().numpy()
                jet_outs.append(u2s(jets, dtype))

            if issubclass(type(task), RegressionTaskBase):
                dtype = np.dtype([(f"{task.label}_{task.name}", precision_str)])
                jets = outputs[task].float().cpu().unsqueeze(-1).numpy()
                jet_outs.append(u2s(jets, dtype))

        jets = join_structured_arrays(jet_outs)
        if self.jet_variables is None:
            self.jet_variables = self.file[self.jet].dtype.names
        jets2 = self.file[self.jet].fields(self.jet_variables)[: self.num_jets]
        jets = join_structured_arrays((jets, jets2))

        if self.write_tracks:
            outputs = {task.name: outputs[task] for task in self.tasks}
            if self.track_variables is None:
                self.track_variables = self.file[self.track].dtype.names
            t = self.file[self.track].fields(self.track_variables)[: self.num_jets]

            # add output for track classification
            if "track_origin" in self.task_names:
                t2 = outputs["track_origin"].float().cpu().numpy()
                t2 = t2.view(dtype=np.dtype([(name, "f4") for name in self.track_origin_cols]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

            if "track_type" in self.task_names:
                t2 = outputs["track_type"].float().cpu().numpy()
                t2 = t2.view(dtype=np.dtype([(name, "f4") for name in self.track_type_cols]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

            # add output for vertexing
            if "track_vertexing" in self.task_names:
                mask = torch.cat(self.mask)
                t2 = outputs["track_vertexing"]
                # could switch this to running on individual batches if memory becomes an issue
                t2 = get_node_assignment(t2, mask)
                t2 = mask_fill_flattened(t2, mask).float().cpu().numpy()

                # convert to structured array
                t2 = t2.view(dtype=np.dtype([("VertexIndex", "f4")]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

        # write to h5 file
        print("\n" + "-" * 100)
        if self.output_path.exists():
            print("Warning! Overwriting existing file.")

        with h5py.File(self.output_path, "w") as f:
            self.create_dataset(f, jets, self.jet, self.half_precision)
            if self.write_tracks:
                self.create_dataset(f, t, self.track, self.half_precision)

        print("Created output file", self.output_path)
        print("-" * 100, "\n")

    def create_dataset(self, f, a, name, half_precision):
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


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def mask_fill_flattened(flat_array, mask):
    filled = torch.full((mask.shape[0], mask.shape[1], flat_array.shape[1]), float("-inf"))
    mask = mask.to(torch.bool)
    start_index = end_index = 0

    for i in range(mask.shape[0]):
        if mask[i].shape[0] > 0:
            end_index += (~mask[i]).to(torch.long).sum()
            filled[i, : end_index - start_index] = flat_array[start_index:end_index]
            start_index = end_index

    return filled
