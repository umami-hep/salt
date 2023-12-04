import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from ftag.git_check import check_for_uncommitted_changes, get_git_hash
from torch import Tensor
from torch.nn.functional import softmax
from tqdm import tqdm

from salt.models.task import mask_fill_flattened
from salt.modelwrapper import ModelWrapper
from salt.utils.inputs import inputs_sep_no_pad, inputs_sep_with_pad
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
        "-a",
        "--include_aux",
        help="Include auxiliary task outputs (if available)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Run with uncomitted changes.",
        action="store_true",
    )

    return parser.parse_args(args)


def get_probs(outputs: Tensor):
    outputs = softmax(outputs, dim=-1)
    return tuple(output.squeeze() for output in torch.split(outputs, 1, -1))


class ONNXModel(ModelWrapper):
    def __init__(self, include_aux=False, **kwargs) -> None:
        super().__init__(**kwargs)
        assert len(self.model.init_nets) == 1, "Multi input ONNX models are not yet supported."
        self.include_aux = include_aux
        self.const = "tracks"
        jets, tracks = inputs_sep_no_pad(
            1, 40, self.input_dims[self.global_object], self.input_dims[self.const]
        )
        self.example_input_array = jets, tracks.squeeze(0)

    def forward(self, jets: Tensor, tracks: Tensor, labels=None):  # type: ignore
        # in athena the jets have a batch dim but the tracks don't, so add it here
        tracks = tracks.unsqueeze(0)

        # forward pass
        outputs = super().forward({self.global_object: jets, self.const: tracks}, None)[0]

        # get class probabilities
        onnx_outputs = get_probs(outputs[self.global_object]["jet_classification"])

        # add aux outputs
        if self.include_aux:
            track_outs = outputs[self.const]
            if "track_origin" in track_outs:
                outputs_track = torch.argmax(track_outs["track_origin"], dim=-1)
                outputs_track = outputs_track.squeeze(0).char()
                onnx_outputs += (outputs_track,)

            if "track_vertexing" in track_outs:
                mask = torch.zeros(tracks.shape[:-1], dtype=torch.bool)
                edge_scores = track_outs["track_vertexing"]
                vertex_indices = get_node_assignment(edge_scores, mask)
                vertex_list = mask_fill_flattened(vertex_indices, mask)
                onnx_outputs += (vertex_list.reshape(-1).char(),)

        return onnx_outputs


def compare_output(pt_model, onnx_session, include_aux, n_track=40):
    n_batch = 1

    jets, tracks, mask = inputs_sep_with_pad(
        n_batch, n_track, pt_model.input_dims["jets"], pt_model.input_dims["tracks"], p_valid=1
    )

    inputs_pt = {"jets": jets, "tracks": tracks}
    outputs_pt = pt_model(inputs_pt, {"tracks": mask})[0]
    pred_pt_jc = [p.detach().numpy() for p in get_probs(outputs_pt["jets"]["jet_classification"])]

    inputs_onnx = {
        "jet_features": jets.numpy(),
        "track_features": tracks.squeeze(0).numpy(),
    }
    outputs_onnx = onnx_session.run(None, inputs_onnx)

    # test jet classification
    pred_onnx_jc = outputs_onnx[: len(pred_pt_jc)]

    np.testing.assert_allclose(
        pred_pt_jc,
        pred_onnx_jc,
        rtol=1e-04,
        atol=1e-04,
        err_msg="Torch vs ONNX check failed for jet classification",
    )

    assert not np.isnan(np.array(pred_onnx_jc)).any()  # non nans
    assert not (np.array(pred_onnx_jc) == 0).any()  # no trivial zeros

    # test track origin
    if include_aux:
        if n_track == 0:
            return

        pred_pt_origin = torch.argmax(outputs_pt["tracks"]["track_origin"], dim=-1).detach().numpy()
        pred_onnx_origin = outputs_onnx[len(pred_pt_jc) : len(pred_pt_jc) + len(pred_pt_origin)][0]

        np.testing.assert_allclose(
            pred_pt_origin.squeeze(),
            pred_onnx_origin,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track origin",
        )

    # test vertexing
    if include_aux:
        pred_pt_scores = outputs_pt["tracks"]["track_vertexing"].detach()
        pred_pt_indices = get_node_assignment(pred_pt_scores, mask)
        pred_pt_vtx = mask_fill_flattened(pred_pt_indices, mask)

        pred_onnx_vtx = outputs_onnx[-1]
        np.testing.assert_allclose(
            pred_pt_vtx.squeeze(),
            pred_onnx_vtx,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for vertexing",
        )


def compare_outputs(pt_model, onnx_path, include_aux):
    print("\n" + "-" * 100)
    print("Validating ONNX model...")

    sess_options = ort.SessionOptions()
    # suppress warnings due to unoptimized subgraphs - https://github.com/microsoft/onnxruntime/issues/14694
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    for n_track in tqdm(range(0, 40), leave=False):
        print(n_track)
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
        check_for_uncommitted_changes(Path(__file__).parent)

    if not (config_path := args.config):
        config_path = args.ckpt_path.parents[1] / "config.yaml"
        assert config_path.is_file(), f"Could not find config file at {config_path}"

    config = yaml.safe_load(config_path.read_text())
    model_name = args.name if args.name else config["name"]
    model_name = model_name.replace("-", "_")  # dashes not allowed for AuxVars

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pt_model = ModelWrapper.load_from_checkpoint(
            args.ckpt_path, map_location=torch.device("cpu")
        )
        pt_model.eval()
        pt_model.float()

        onnx_model = ONNXModel.load_from_checkpoint(
            args.ckpt_path, include_aux=args.include_aux, map_location=torch.device("cpu")
        )
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
    dynamic_axes = {"track_features": {0: "n_tracks"}}

    if args.include_aux:
        if "track_origin" in [t.name for t in pt_model.model.tasks]:
            out_name = f"{model_name}_TrackOrigin"
            output_names.append(out_name)
            dynamic_axes[out_name] = {0: "n_tracks"}

        if "track_vertexing" in [t.name for t in pt_model.model.tasks]:
            out_name = f"{model_name}_VertexIndex"
            output_names.append(out_name)
            dynamic_axes[out_name] = {0: "n_tracks"}

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
    jet_vars = config["data"]["variables"]["jets"]
    trk_vars = config["data"]["variables"]["tracks"]
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

    # add model version instead of specifying outputs
    metadata["onnx_model_version"] = "v1"

    # Save the git hash of the repo used for exporting onnx model
    metadata["salt_export_hash"] = get_git_hash(Path(__file__).parent)

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
