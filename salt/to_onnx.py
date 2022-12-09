import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from ruamel.yaml import YAML
from torch import Tensor
from torch.nn.functional import softmax
from tqdm import tqdm

from salt.lightning import LightningTagger
from salt.utils.inputs import concat_jet_track, inputs_sep_no_pad, inputs_sep_with_pad

# https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx
TRACK_SELECTIONS = [
    "all",
    "ip3d",
    "dipsLoose202102",
    "r22default",
    "r22loose",
    "dipsTightUpgrade",
    "dipsLooseUpgrade",
]


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A script to convert a salt model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Saved training config.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Checkpoint path.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--track_selection",
        type=str,
        help="Track selection, matches `trk_select_regexes` in 'DataPrepUtilities.cxx'",
        choices=TRACK_SELECTIONS,
        required=True,
    )
    parser.add_argument(
        "--sd_path",
        type=Path,
        help="Scale dict path. Taken from the config if not provided",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Model name, used to create the *_px outputs.",
        default="salt",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing exported ONNX model.",
        action="store_true",
    )

    return parser.parse_args(args)


def get_probs(outputs: Tensor):
    outputs = softmax(outputs, dim=-1)
    return torch.split(outputs.squeeze(), 1, -1)


class ONNXModel(LightningTagger):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model=model, lrs_config={})
        assert len(self.in_dims) == 1, "Multi input ONNX models are not yet supported."
        jets, tracks = inputs_sep_no_pad(1, 40, self.in_dims[0])
        self.example_input_array = jets, tracks.squeeze(0)

    def forward(self, jets: Tensor, tracks: Tensor, labels=None):
        # in athena the jets have a batch dimension but the tracks do not
        tracks = tracks.unsqueeze(0)

        # concatenate jet and track inputs
        inputs = {"track": concat_jet_track(jets, tracks)}

        # don't pass padded inputs, so all tracks are taken to be real
        mask = {"track": torch.zeros(tracks.shape[:-1]).bool()}

        # get probabilities
        outputs = self.model(inputs, mask, None)[0]["jet_classification"]
        return get_probs(outputs)


def add_metadata(onnx_path, model_name, sd_path, track_selection, output_names):
    print("\n" + "-" * 100)
    print("Adding Metadata...")
    print(f"Using scale dict {sd_path}")

    with open(sd_path) as f:
        scale_dict = json.load(f)

    # load and check the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    metadata = {"layers": [], "nodes": []}

    metadata["input_sequences"] = [
        {
            "name": f"tracks_{track_selection}_sd0sort",
            "variables": [
                {"name": k, "offset": v["shift"], "scale": v["scale"]}
                for k, v in scale_dict["tracks_loose"].items()
            ],
        }
    ]
    metadata["inputs"] = [
        {
            "name": "jet_var",
            "variables": [
                {"name": k, "offset": v["shift"], "scale": v["scale"]}
                for k, v in scale_dict["jets"].items()
            ],
        }
    ]
    metadata["outputs"] = {model_name: {"labels": output_names, "node_index": 0}}

    metadata = {"gnn_config": json.dumps(metadata)}

    for k, v in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = v

    onnx_model.doc_string = f"{model_name}"

    onnx.save(onnx_model, onnx_path)
    print("-" * 100)


def compare_output(pt_model, onnx_session, n_track=40):
    n_batch = 1
    n_feat = pt_model.in_dims[0]

    jets, tracks, mask = inputs_sep_with_pad(n_batch, n_track, n_feat)
    inputs = {"track": concat_jet_track(jets, tracks)}

    pred_pt = pt_model(inputs, mask, None)[0]["jet_classification"]
    pred_pt = [p.detach().numpy() for p in get_probs(pred_pt)]

    ort_inputs = {
        "jet_features": jets.numpy(),
        "track_features": tracks.squeeze(0).numpy(),
    }
    pred_onnx = onnx_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        pred_pt, pred_onnx, rtol=1e-06, atol=1e-06, err_msg="Torch vs ONNX check failed"
    )

    assert not np.isnan(np.array(pred_onnx)).any()  # non nans
    assert not (np.array(pred_onnx) == 0).any()  # no trivial zeros


def compare_outputs(pt_model, onnx_path):
    print("\n" + "-" * 100)
    print("Validating ONNX model...")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for n_track in tqdm(range(1, 40), leave=False):
        for _ in range(10):
            compare_output(pt_model, session, n_track)

    print(
        "Sucess! Pytorch and ONNX models are consistent.\nIt should be safe to ignore"
        " any above warnings, but you should still verify consistency in Athena."
    )
    print("-" * 100)


def main(args=None):
    # parse args
    args = parse_args(args)

    # get the config file
    config_path = args.config
    with open(config_path) as fp:
        config = YAML().load(fp)
        sd_path = config["data"]["scale_dict"] if not args.sd_path else args.sd_path

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pt_model = LightningTagger.load_from_checkpoint(args.ckpt_path)
        pt_model.eval()

        onnx_model = ONNXModel.load_from_checkpoint(args.ckpt_path)
        onnx_model.eval()

    print("\n" + "-" * 100)
    print("Converting model to ONNX...")
    print("-" * 100)

    onnx_path = args.ckpt_path.parent.parent / "model.onnx"
    if onnx_path.exists() and not args.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")

    input_names = ["jet_features", "track_features"]
    output_names = ["pb", "pc", "pu"]
    onnx_model.to_onnx(
        onnx_path,
        opset_version=16,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=False,
        dynamic_axes={
            "track_features": {0: "n_tracks"},
        },
    )

    add_metadata(onnx_path, args.name, sd_path, args.track_selection, output_names)
    compare_outputs(pt_model, onnx_path)
    print("\n" + "-" * 100)
    print(f"Done! Saved ONNX model at {onnx_path}")
    print("-" * 100)
    print()


if __name__ == "__main__":
    main()
