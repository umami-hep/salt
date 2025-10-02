"""Runs the Integrated Gradients : https://gitlab.cern.ch/atlas-flavourtagging-algorithms/salt-attribution
algorithm on the evaluation file and writes the results to a file.
"""

from pathlib import Path

import h5py
import torch
from lightning import Callback, LightningModule, Trainer

try:
    from captum.attr import IntegratedGradients
    from salt_attribution.utils import (
        SaltModelCaptumWrapper,
        calculate_flow_sizes,
        calculate_track_sizes,
        find_all_predictions_and_inputs,
        numpy_dict_2_h5,
        sampled_baseline_feature_attribution,
        tensor_dict_2_device,
        tensor_dict_2_ndarray,
        topk_entropy_baselines,
    )

    HAS_IG = True

except ImportError as e:
    FAILED_IG = e
    HAS_IG = False


class IntegratedGradientWriter(Callback):
    """Callback to run Integrated Gradients on the test set and save the results to a file.

    Parameters
    ----------
    input_keys : dict
        Dictionary of input keys to be used for the model. This should take the form:
        input_keys:
            inputs: ["jets", "tracks", ... [any other inputs]]
        pad_masks: ["tracks", ... [any other pad masks]]
    output_keys : list
        A list of keys representing the nested output of the model we wish to use. E.g., if
        the model returns {'jets' : {'jets_classification' : [predictions ]}} then
        'output_keys' should be : ['jets', 'jets_classification']
    add_softmax : bool, optional
        Whether to add softmax to the model outputs, by default True.
    n_baselines : int, optional
        Number of baselines to use for each jet, by default 5.
    min_allowed_track_sizes : int, optional
        Only calculate attributions for jets with at least this many tracks, by default 5.
    max_allowed_track_sizes : int, optional
        Only calculate attributions for jets with at most this many tracks, by default 15.
    min_allowed_flow_sizes : int | None, optional
        Only calculate attributions for jets with at least this many tracks, by default None,
        meaning no minimum flow size is applied.
    max_allowed_flow_sizes : int | None, optional
        Only calculate attributions for jets with at most this many tracks, by default None,
        meaning no maximum flow size is applied
    tracks_name : str, optional
        Name of the tracks in the output file, by default "tracks".
    flows_name : str, optional
        Name of the flows in the output file, by default "flows".
    n_jets : int, optional
        Number of jets to use for the attribution calculation, by default -1, which means
        half of jets in the test set are used.
    n_steps : int, optional
        Number of steps to use for the estimation of the integrated gradients integral.
        Default is 50.
    internal_batch_size : int, optional
        Batch size that Captum uses when calculating integrated gradients, by default -1, which
        means the same batch size as the dataloader is used.
    normalize_deltas : bool, optional
        Whether to normalize the convergence deltas, by default True.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    Raises
    ------
    ImportError
        When captum and salt-attribution are not available
    ValueError
        If neither min/max flow count was set or both
    """

    def __init__(
        self,
        input_keys: dict,
        output_keys: list,
        add_softmax: bool = True,
        n_baselines: int = 5,
        min_allowed_track_sizes: int = 5,
        max_allowed_track_sizes: int = 15,
        min_allowed_flow_sizes: int | None = None,
        max_allowed_flow_sizes: int | None = None,
        tracks_name: str = "tracks",
        flows_name: str = "flows",
        n_jets: int = -1,
        n_steps: int = 50,
        internal_batch_size: int = -1,
        normalize_deltas: bool = True,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = True
        if not HAS_IG:
            raise ImportError(
                FAILED_IG.msg
                + "\n"
                + "IntegratedGradientWriter requires captum and salt-attribution. "
                "Please install them with `pip install -r requirements-ig.txt`."
            )

        self.add_softmax = add_softmax
        self.n_baselines = n_baselines
        self.min_allowed_track_sizes = min_allowed_track_sizes
        self.max_allowed_track_sizes = max_allowed_track_sizes
        self.allowed_track_sizes = torch.arange(
            min_allowed_track_sizes, max_allowed_track_sizes + 1
        )

        if min_allowed_flow_sizes is not None and max_allowed_flow_sizes is not None:
            self.allowed_flow_sizes = torch.arange(
                min_allowed_flow_sizes, max_allowed_flow_sizes + 1
            )
            self.do_flows = True
        elif min_allowed_flow_sizes is None and max_allowed_flow_sizes is None:
            self.allowed_flow_sizes = None
            self.do_flows = False
        else:
            raise ValueError("Either both min/max flow count must be set, or both must be None.")

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.tracks_name = tracks_name
        self.flows_name = flows_name
        self.n_steps = n_steps
        self.n_jets = n_jets
        self.internal_batch_size = internal_batch_size
        self.normalize_deltas = normalize_deltas

        self.overwrite = overwrite
        self.input_keys = input_keys

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:  # noqa: ARG002
        if stage != "test":
            return
        self.trainer = trainer
        self.dm = trainer.datamodule
        self.ds = self.dm.test_dataloader().dataset
        self.test_suff = self.dm.test_suff

        if self.n_jets == -1:
            # We use less than the full dataloader,
            # as not all jets will fulfil the track requirements
            self.n_jets = int(len(self.ds) * 0.5)
            print(f"n_jets not set, using half of all jets ({self.n_jets}) in the dataset")
        if self.internal_batch_size == -1:
            self.internal_batch_size = self.dm.batch_size
            print(
                "internal_batch_size not set, using the evaluation batch size "
                f"({self.internal_batch_size})"
            )

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        stem = str(Path(self.ds.filename).stem)
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        suffix = f"_{self.test_suff}" if self.test_suff is not None else ""
        return Path(out_dir / f"{out_basename}__attributions_{sample}{suffix}.h5")

    def on_test_start(self, trainer, pl_module):  # noqa: ARG002
        print("Will run Integrated Gradients on test set")
        print("This may take a while!")
        print(f"\nSaving attributions to {self.output_path}")
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output file {self.output_path} already exists. "
                "Please set `overwrite=True` to overwrite the file."
            )
        model = trainer.model
        model.zero_grad()

        captum_wrapped_model = SaltModelCaptumWrapper(
            model,
            inputs_keys=self.input_keys,
            output_keys=self.output_keys,
            add_softmax=self.add_softmax,
        )

        attributor = IntegratedGradients(captum_wrapped_model)

        dataloader = self.dm.test_dataloader()
        inputs, pad_masks, labels = next(iter(dataloader))

        device = model.device
        inputs = tensor_dict_2_device(inputs, device)
        pad_masks = tensor_dict_2_device(pad_masks, device)
        labels = tensor_dict_2_device(labels, device)

        with h5py.File(self.output_path, "w") as f:
            # Find the baselines
            print("Finding predictions")
            all_preds_and_inputs = find_all_predictions_and_inputs(
                model=model,
                dataloader=dataloader,
                allowed_track_sizes=self.allowed_track_sizes,
                allowed_flow_sizes=self.allowed_flow_sizes,
                output_keys=self.output_keys,
                num_of_jets=self.n_jets,
                add_softmax=self.add_softmax,
                device=device,
                verbose=self.verbose,
                track_name=self.tracks_name,
                flow_name=self.flows_name,
            )

            all_track_sizes = calculate_track_sizes(
                all_preds_and_inputs.pad_masks, track_name=self.tracks_name
            )
            all_flow_sizes = (
                calculate_flow_sizes(all_preds_and_inputs.pad_masks, flow_name=self.flows_name)
                if self.do_flows
                else None
            )
            numpy_dict_2_h5(f, tensor_dict_2_ndarray(all_preds_and_inputs._asdict()))

            print("\nFinding baselines")
            jet_baselines, track_baselines, flow_baselines, baseline_entropies = (
                topk_entropy_baselines(
                    preds=all_preds_and_inputs.preds,
                    inputs=all_preds_and_inputs.inputs,
                    track_sizes=all_track_sizes,
                    allowed_track_sizes=self.allowed_track_sizes,
                    flow_sizes=all_flow_sizes,
                    allowed_flow_sizes=self.allowed_flow_sizes,
                    n_baselines=self.n_baselines,
                    track_name=self.tracks_name,
                    flow_name=self.flows_name,
                )
            )
            print("Found baselines")

            baseline_grp = f.create_group("baselines")

            baseline_grp.create_dataset("jets", data=jet_baselines.numpy())
            baseline_grp.create_dataset(self.tracks_name, data=track_baselines.numpy())
            baseline_grp.create_dataset("baseline_entropies", data=baseline_entropies.numpy())

            print("\nFinding attributions")

            # We need inference mode to be false here, as we need to keep track of gradients
            with torch.inference_mode(mode=False):
                feature_attribs = sampled_baseline_feature_attribution(
                    model=model,
                    attributor=attributor,
                    dataloader=dataloader,
                    allowed_track_sizes=self.allowed_track_sizes,
                    allowed_flow_sizes=self.allowed_flow_sizes,
                    output_keys=self.output_keys,
                    jet_baselines=jet_baselines,
                    track_baselines=track_baselines,
                    flow_baselines=flow_baselines,
                    add_softmax=self.add_softmax,
                    n_steps=self.n_steps,
                    internal_batch_size=self.internal_batch_size,
                    device=device,
                    num_of_jets=self.n_jets,
                    normalize_deltas=self.normalize_deltas,
                    verbose=self.verbose,
                    track_name=self.tracks_name,
                    flow_name=self.flows_name,
                )

            attributions_grp = f.create_group("attributions")
            numpy_dict_2_h5(attributions_grp, tensor_dict_2_ndarray(feature_attribs._asdict()))
