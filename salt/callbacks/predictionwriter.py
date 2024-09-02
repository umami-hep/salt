from collections import defaultdict
from pathlib import Path

import numpy as np
from ftag import Flavours as Flavs
from ftag.hdf5 import H5Writer
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.models.task import (
    ClassificationTask,
    GaussianRegressionTask,
    RegressionTask,
    VertexingTask,
)
from salt.stypes import Vars
from salt.utils.array_utils import join_structured_arrays, maybe_pad
from salt.utils.mask_utils import indices_from_mask


class PredictionWriter(Callback):
    def __init__(
        self,
        write_tracks: bool = False,
        write_objects: bool = False,
        half_precision: bool = False,
        object_classes: list | None = None,
        extra_vars: Vars | None = None,
    ) -> None:
        """Write test outputs to h5 file.

        This callback will write the outputs of the model to an h5 evaluation file. The outputs
        are produced by calling the `run_inference` method of each task. The output file
        is written to the same directory as the checkpoint file, and has the same name
        as the checkpoint file, but with the suffix `__test_<sample><suffix>.h5`. The file will
        contain one dataset for each input type, with the same name as the input type in the test
        file.

        Parameters
        ----------
        write_tracks : bool
            If False, skip any tasks with `input_name="tracks"`.
        write_objects : bool
            If False, skip any tasks with `input_name="objects"` and outputs of the
            MaskDecoder. Default is False
        half_precision : bool
            If true, write outputs at half precision
        object_classes : list
            List of flavour names with the index corresponding to the label values. This is used
            to construct the global object classification probability output names.
        extra_vars : Vars
            Extra variables to write to file for each input type. If not specified for a given input
            type, all variables in the test file will be written.
        """
        super().__init__()
        if extra_vars is None:
            extra_vars = defaultdict(list)
        self.extra_vars = extra_vars
        self.write_tracks = write_tracks
        self.write_objects = write_objects
        self.half_precision = half_precision
        self.precision = "f2" if self.half_precision else "f4"
        self.object_classes = object_classes

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        self.writer = None
        # some basic info
        self.trainer = trainer
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.batch_size = trainer.datamodule.batch_size
        self.test_suff = trainer.datamodule.test_suff
        self.file = self.ds.file
        self.num = len(self.ds)
        self.norm_dict = self.ds.norm_dict

        # inputs names
        self.input_map = self.ds.input_map

        # check extra vars exist
        for input_type, vars_to_check in self.extra_vars.items():
            if input_type in self.input_map:
                dataset_name = self.input_map[input_type]
                available_vars = set(self.file[dataset_name].dtype.names)
                if missing_vars := set(vars_to_check) - available_vars:
                    raise ValueError(
                        "The following variables are missing for input type"
                        f"'{input_type}': {missing_vars}"
                    )
            else:
                raise ValueError(f"Input type '{input_type}' is not recognized in input_map.")

        # place to store intermediate outputs
        self.tasks = module.model.tasks
        self.outputs: dict = {input_name: {} for input_name in {t.input_name for t in self.tasks}}
        self.pad_masks: dict = {}

        # get object class names for output file
        for task in self.tasks:
            if task.name != f"{module.global_object}_classification":
                continue
            if self.object_classes is None:
                if task.class_names is not None:
                    self.object_classes = task.class_names
                else:
                    raise ValueError(
                        "Couldn't infer object classes from model. "
                        "Please provide a list of object classes."
                    )

        if self.write_objects:
            if not module.model.mask_decoder:
                raise ValueError("write_objects=True but no mask decoder found in model.")
            if not self.write_tracks:
                print("-" * 50)
                print("WARNING: If outputting mask objects, you probably also want tracks")

            # Add objects to outputs if there are no main tasks for this
            if "objects" not in self.outputs:
                self.outputs["objects"] = {}

            self.object_params = {
                "class_label": self.ds.mf_config.object.class_label,
                "label_map": [f"p{name}" for name in self.ds.mf_config.object.class_names],
            }

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        stem = str(Path(self.ds.filename).stem)
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        suffix = f"_{self.test_suff}" if self.test_suff is not None else ""
        return Path(out_dir / f"{out_basename}__test_{sample}{suffix}.h5")

    def _write_batch_outputs(self, batch_outputs, pad_masks, batch_idx):
        to_write = {}
        blow = batch_idx * self.batch_size
        bhigh = (batch_idx + 1) * self.batch_size
        for input_name, outputs in batch_outputs.items():
            this_outputs = []

            if input_name in self.input_map:
                name = self.input_map[input_name]

                # get input variables
                input_variables = self.extra_vars[name]
                if not input_variables:
                    input_variables = self.file[name].dtype.names
                if name in self.file:
                    inputs = self.file[name].fields(input_variables)[blow:bhigh]
                    this_outputs.append(inputs)
            else:
                name = input_name
                inputs = None

            for preds in outputs.values():
                if inputs is not None:
                    this_outputs.append(maybe_pad(preds, inputs))
                else:
                    this_outputs.append(preds)

            # add mask if present
            if name in pad_masks:
                pad_mask = pad_masks[name].cpu()
                pad_mask = u2s(np.expand_dims(pad_mask, -1), dtype=np.dtype([("mask", "?")]))
                this_outputs.append(maybe_pad(pad_mask, inputs))

            to_write[name] = join_structured_arrays(this_outputs)

        # If the writer hasn't been created yet, create it now that we have the dtypes and shapes
        if self.writer is None:
            dtypes = {k: v.dtype for k, v in to_write.items()}
            shapes = {k: (self.num,) + v.shape[1:] for k, v in to_write.items()}
            self.writer = H5Writer(
                dst=self.output_path,
                dtypes=dtypes,
                shapes=shapes,
                shuffle=False,
                precision="half" if self.half_precision else "full",
            )
        self.writer.write(to_write)

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):  # noqa: ARG002
        preds = outputs
        _, pad_masks, labels = batch
        add_mask = False
        to_write: dict = {input_name: {} for input_name in {t.input_name for t in self.tasks}}
        out_pads = {}
        for task in self.tasks:
            if (
                (not self.write_tracks and task.input_name == "tracks")
                or (not self.write_objects and task.input_name == "objects")
                or (issubclass(type(task), GaussianRegressionTask))
            ):
                continue

            this_preds = preds[task.input_name][task.name]
            this_pad_masks = pad_masks.get(task.input_name)

            if isinstance(task, ClassificationTask):
                # special case for object classification output names
                if task.name == f"{module.global_object}_classification":
                    flavs = [
                        f"{Flavs[c].px}" if c in Flavs else f"p{c}" for c in self.object_classes
                    ]
                    task.class_names = [f"{module.name}_{px}" for px in flavs]
                this_preds = task.run_inference(this_preds, this_pad_masks, self.precision)
            elif isinstance(task, VertexingTask):
                this_preds = task.run_inference(this_preds, this_pad_masks)
            elif issubclass(type(task), RegressionTask):
                this_preds = task.run_inference(this_preds, labels, self.precision)
            elif issubclass(type(task), GaussianRegressionTask):
                if f"{module.name}_" not in task.name:
                    task.name = f"{module.name}_" + task.name
                means, stddevs = task.run_inference(this_preds, labels, self.precision)
            if task.name not in to_write[task.input_name]:
                to_write[task.input_name][task.name] = []
            if issubclass(type(task), GaussianRegressionTask):
                # if task.name + "_stddev" not in self.outputs[task.input_name]:
                #     to_write[task.input_name][task.name + "_stddev"] = []

                to_write[task.input_name][task.name] = means
                to_write[task.input_name][task.name + "_stddev"] = stddevs
            else:
                to_write[task.input_name][task.name] = this_preds

            if this_pad_masks is not None and add_mask is False:
                out_pads[task.input_name] = this_pad_masks
                add_mask = True

        if self.write_objects:
            # Generate the object outputs of the form (B, N) where N is the number of objects
            objects = outputs["objects"]

            probs_dtype = np.dtype([
                (f"{module.name}_{n}", self.precision) for n in self.object_params["label_map"]
            ])
            to_write["objects"]["object_class_probs"] = u2s(
                objects["class_probs"].cpu().float().numpy(), dtype=probs_dtype
            )
            to_write["objects"]["object_class_targets"] = u2s(
                labels["objects"][self.object_params["class_label"]].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("class_label", "i8")]),
            )

            # Write the mask indices to the tracks
            mask_indices = indices_from_mask(objects["masks"].cpu().sigmoid() > 0.5)
            dtype = np.dtype([(f"{module.name}_MaskIndex", "i8")])
            mask_indices = mask_indices.int().cpu().numpy()
            mask_indices = np.where(~this_pad_masks, mask_indices, -1)
            to_write["tracks"]["mask_index"] = u2s(np.expand_dims(mask_indices, -1), dtype)

            # Write the truth mask and mask logits to their own dset
            to_write["object_masks"] = {}
            to_write["object_masks"]["tgt_masks"] = u2s(
                labels["objects"]["masks"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_mask", "i8")]),
            )
            to_write["object_masks"]["mask_logits"] = u2s(
                objects["masks"].cpu().float().unsqueeze(-1).numpy(),
                dtype=np.dtype([("mask_logits", self.precision)]),
            )

        self._write_batch_outputs(to_write, out_pads, batch_idx)

    def on_test_end(self, trainer, module):  # noqa: ARG002
        if self.writer is not None:
            self.writer.close()
