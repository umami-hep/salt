import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch import Tensor, nn
from torch.nn.functional import softmax
from tqdm import tqdm

from salt.callbacks.predictionwriter import mask_fill_flattened
from salt.lightning_module import LightningTagger
from salt.utils.git_check import check_for_uncommitted_changes, get_git_hash
from salt.utils.inputs import concat_jet_track, inputs_sep_no_pad, inputs_sep_with_pad
from salt.utils.union_find import get_node_assignment

torch.manual_seed(42)
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
        "--ckpt_path",
        type=Path,
        help="Checkpoint path.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Saved training config. If not provided, look in the parent directory of `ckpt_path`.",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--track_selection",
        type=str,
        help="Track selection, must match `trk_select_regexes` in 'DataPrepUtilities.cxx'",
        choices=TRACK_SELECTIONS,
        default="r22default",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Model name, used in the *_px outputs. Taken from the config if not provided",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing exported ONNX model.",
        action="store_true",
    )
    parser.add_argument(
        "--include_aux",
        help="Include auxiliary task outputs (if available)",
        action="store_true",
    )
    parser.add_argument("--force", help="Run with uncomitted changes.", action="store_true")

    return parser.parse_args(args)


def get_probs(outputs: Tensor):
    outputs = softmax(outputs, dim=-1)
    return torch.split(outputs.squeeze(), 1, -1)


class ONNXModel(LightningTagger):
    def __init__(self, model: nn.Module, include_aux=False) -> None:
        super().__init__(model=model, lrs_config={})
        self.include_aux = include_aux

        assert len(self.in_dims) == 1, "Multi input ONNX models are not yet supported."

        jets, tracks = inputs_sep_no_pad(1, 40, 2, self.in_dims[0] - 2)
        self.example_input_array = jets, tracks.squeeze(0)

    def forward(self, jets: Tensor, tracks: Tensor, labels=None):
        # in athena the jets have a batch dimension but the tracks do not
        tracks = concat_jet_track(jets, tracks.unsqueeze(0))

        # return class probabilities
        outputs = self.model({"track": tracks}, None)[0]
        onnx_outputs = get_probs(outputs["jet_classification"])

        if self.include_aux and "track_vertexing" in outputs:
            mask = torch.zeros(tracks.shape[:-1])
            edge_scores = outputs["track_vertexing"]
            vertex_indices = get_node_assignment(edge_scores, mask)
            vertex_list = mask_fill_flattened(vertex_indices, mask)  # get list of vertex indices
            onnx_outputs += (vertex_list,)

        return onnx_outputs


def compare_output(pt_model, onnx_session, include_aux, n_track=40):
    n_batch = 1

    jets, tracks, mask = inputs_sep_with_pad(
        n_batch, n_track, 2, pt_model.in_dims[0] - 2, p_valid=1
    )

    inputs_pt = {"track": concat_jet_track(jets, tracks)}
    outputs_pt = pt_model(inputs_pt, {"track": mask})[0]
    pred_pt_jc = [p.detach().numpy() for p in get_probs(outputs_pt["jet_classification"])]

    inputs_onnx = {
        "jet_features": jets.numpy(),
        "track_features": tracks.squeeze(0).numpy(),
    }
    outputs_onnx = onnx_session.run(None, inputs_onnx)

    # test jet classification
    if include_aux and "track_vertexing" in outputs_pt:
        pred_onnx_jc = outputs_onnx[:-1]
    else:
        pred_onnx_jc = outputs_onnx

    np.testing.assert_allclose(
        pred_pt_jc,
        pred_onnx_jc,
        rtol=1e-04,
        atol=1e-04,
        err_msg="Torch vs ONNX check failed for jet classification",
    )

    assert not np.isnan(np.array(pred_onnx_jc)).any()  # non nans
    assert not (np.array(pred_onnx_jc) == 0).any()  # no trivial zeros

    # test vertexing
    if include_aux and "track_vertexing" in outputs_pt:
        pred_pt_scores = outputs_pt["track_vertexing"].detach()
        pred_pt_indices = get_node_assignment(pred_pt_scores, mask)
        pred_pt_vtx = mask_fill_flattened(pred_pt_indices, mask)

        pred_onnx_vtx = outputs_onnx[-1]
        np.testing.assert_allclose(
            pred_pt_vtx,
            pred_onnx_vtx,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for vertexing",
        )


def compare_outputs(pt_model, onnx_path, include_aux):
    print("\n" + "-" * 100)
    print("Validating ONNX model...")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for n_track in tqdm(range(0, 40), leave=False):
        for _ in range(10):
            compare_output(pt_model, session, include_aux, n_track)

    print(
        "Success! Pytorch and ONNX models are consistent, but you should verify this in"
        " Athena.\nFor more info see: https://ftag-salt.docs.cern.ch/export/#athena-validation"
    )
    print("-" * 100)


def main(args=None):
    # parse args
    args = parse_args(args)

    if not args.force:
        check_for_uncommitted_changes()

    if not (config_path := args.config):
        config_path = args.ckpt_path.parents[1] / "config.yaml"
        assert config_path.is_file(), f"Could not find config file at {config_path}"

    config = yaml.safe_load(config_path.read_text())
    model_name = args.name if args.name else config["name"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pt_model = LightningTagger.load_from_checkpoint(args.ckpt_path)
        pt_model.eval()

        onnx_model = ONNXModel.load_from_checkpoint(args.ckpt_path, include_aux=args.include_aux)
        onnx_model.eval()

    print("\n" + "-" * 100)
    print("Converting model to ONNX...")
    print("-" * 100)

    # configure paths
    base_path = args.ckpt_path.parent.parent
    onnx_path = base_path / "network.onnx"
    if onnx_path.exists() and not args.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")

    # configure inputs and outputs
    jet_classes = yaml.safe_load((base_path / "metadata.yaml").read_text())["jet_classes"]
    input_names = ["jet_features", "track_features"]
    output_names = [f"p{flav.rstrip('jets')}" for flav in jet_classes.values()]
    output_types = ["float" for _ in jet_classes.values()]
    dynamic_axes = {"track_features": {0: "n_tracks"}}

    if args.include_aux and "track_vertexing" in config["data"]["labels"]:
        output_names.append("vertex_indices")
        output_types.append("vec_char")
        dynamic_axes["vertex_indices"] = {0: "n_tracks"}

    # export
    onnx_model.to_onnx(
        onnx_path,
        opset_version=16,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # add metadata
    add_metadata(
        config_path,
        args.ckpt_path,
        onnx_path,
        model_name,
        args.track_selection,
        output_names,
        output_types,
    )

    # validate
    compare_outputs(pt_model, onnx_path, args.include_aux)
    print("\n" + "-" * 100)
    print(f"Done! Saved ONNX model at {onnx_path}")
    print("-" * 100)
    print()


def add_metadata(
    config_path,
    ckpt_path,
    onnx_path,
    model_name,
    track_selection,
    output_names,
    output_types,
):
    print("\n" + "-" * 100)
    print("Adding Metadata...")

    # load and check the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # add metadata
    metadata = {"ckpt_path": str(ckpt_path.resolve()), "layers": [], "nodes": []}
    config = yaml.safe_load(config_path.read_text())
    metadata["config.yaml"] = config
    jet_vars = config["data"]["variables"]["jet"]
    trk_vars = config["data"]["variables"]["track"]
    metadata["metadata.yaml"] = yaml.safe_load((config_path.parent / "metadata.yaml").read_text())

    # add inputs
    metadata["inputs"] = [
        {
            "name": "jet_var",
            "variables": [
                {"name": k.removesuffix("_btagJes"), "offset": 0.0, "scale": 1.0} for k in jet_vars
            ],
        }
    ]
    metadata["input_sequences"] = [
        {
            "name": f"tracks_{track_selection}_sd0sort",
            "variables": [{"name": k, "offset": 0.0, "scale": 1.0} for k in trk_vars],
        }
    ]
    metadata["outputs"] = {
        model_name: {
            "labels": output_names,
            "types": output_types,
            "node_index": 0,
        }
    }

    # Save the git hash of the repo used for exporting onnx model
    metadata["export_git_hash"] = get_git_hash()

    # write metadata as json string
    metadata = {"gnn_config": json.dumps(metadata)}
    for k, v in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = v

    onnx_model.doc_string = f"{model_name}"
    onnx.save(onnx_model, onnx_path)
    print("-" * 100)


if __name__ == "__main__":
    main()
