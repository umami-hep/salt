from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from ftag import Flavours as Flavs
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.models.task import ClassificationTask, RegressionTaskBase, VertexingTask
from salt.utils.array_utils import join_structured_arrays, maybe_pad


class PredictionWriter(Callback):
    def __init__(
        self,
        write_tracks: bool = False,
        half_precision: bool = False,
        jet_classes: list | None = None,
        extra_vars: dict[str, list[str]] | None = None,
    ) -> None:
        """Write test outputs to h5 file.

        Parameters
        ----------
        write_tracks : bool
            If true, write track outputs to file
        half_precision : bool
            If true, write outputs at half precision
        jet_classes : list
            List of flavour names with the index corresponding to the label values
        extra_vars : dict
            Extra variables to write to file for each input type. If not specified for a given input
            type, all variables in the test file will be written.
        """
        super().__init__()
        if extra_vars is None:
            extra_vars = defaultdict(list)
        self.extra_vars = extra_vars
        self.write_tracks = write_tracks
        self.half_precision = half_precision
        self.precision = "f2" if self.half_precision else "f4"
        self.jet_classes = jet_classes

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
        self.input_names = self.ds.input_names

        # place to store intermediate outputs
        self.tasks = module.model.tasks
        self.outputs: dict = {input_type: {} for input_type in {t.input_type for t in self.tasks}}
        self.masks: dict = {}

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
        add_mask = False
        for task in self.tasks:
            if not self.write_tracks and task.input_type == "track":
                continue

            this_preds = preds[task.input_type][task.name]
            this_mask = masks.get(task.input_type)

            if isinstance(task, ClassificationTask):
                # special case for jet classification output names
                if task.name == "jet_classification":
                    flavs = [f"{Flavs[c].px}" if c in Flavs else f"p{c}" for c in self.jet_classes]
                    task.class_names = [f"{module.name}_{px}" for px in flavs]
                this_preds = task.run_inference(this_preds, this_mask, self.precision)
            elif isinstance(task, VertexingTask):
                this_preds = task.run_inference(this_preds, this_mask)
            elif issubclass(type(task), RegressionTaskBase):
                this_preds = task.run_inference(this_preds, labels, self.precision)
            if task.name not in self.outputs[task.input_type]:
                self.outputs[task.input_type][task.name] = []
            self.outputs[task.input_type][task.name].append(this_preds)
            if this_mask is not None and add_mask is False:
                if task.input_type not in self.masks:
                    self.masks[task.input_type] = []
                self.masks[task.input_type].append(this_mask)
                add_mask = True

    def on_test_end(self, trainer, module):
        print("\n" + "-" * 100)
        if self.output_path.exists():
            print("Warning! Overwriting existing file.")
        f = h5py.File(self.output_path, "w")

        for input_type, outputs in self.outputs.items():
            input_name = self.input_names[input_type]

            # get input variables
            input_variables = self.extra_vars[input_type]
            if not input_variables:
                input_variables = self.file[input_name].dtype.names
            inputs = self.file[input_name].fields(input_variables)[: self.num_jets]

            # get output variables
            this_outputs = [inputs]
            for preds in outputs.values():
                x = np.concatenate(preds)  # concat test batches
                maybe_pad(x, inputs)
                this_outputs.append(maybe_pad(x, inputs))

            # add mask if present
            if input_type in self.masks:
                mask = np.concatenate(self.masks[input_type])  # concat test batches
                mask = u2s(np.expand_dims(mask, -1), dtype=np.dtype([("mask", "?")]))
                this_outputs.append(maybe_pad(mask, inputs))

            # join structured arrays
            this_outputs = join_structured_arrays(this_outputs)

            # write the dataset for this input type
            self.create_dataset(f, this_outputs, input_name, self.half_precision)

        f.close()
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
