import argparse
import json
import warnings
from pathlib import Path

import onnx
import torch
import yaml
from ftag.git_check import check_for_uncommitted_changes, get_git_hash

from salt.models.maskformer import get_maskformer_outputs
from salt.models.task import mask_fill_flattened
from salt.models.transformer_v2 import change_attn_backends
from salt.modelwrapper import ModelWrapper
from salt.onnx.check import compare_outputs
from salt.utils.configs import MaskformerConfig
from salt.utils.union_find import get_node_assignment_jit

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
        "-mf",
        "--object_name",
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Run with uncomitted changes.",
        action="store_true",
    )

    return parser.parse_args(args)


class ONNXModel(ModelWrapper):
    def __init__(
        self,
        onnx_feature_map: list[dict],
        name: str | None = None,
        include_aux: bool = False,
        object_name: str | None = None,
        mf_config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name or self.name
        assert "_" not in self.name, "Model name cannot contain underscores."
        assert "-" not in self.name, "Model name cannot contain dashes."
        self.include_aux = include_aux

        if sum([bool(object_name), bool(mf_config)]) not in {0, 2}:
            raise ValueError("If one of object name or mf config is defined, so must the other.")
        self.object = object_name
        self.mf_config = MaskformerConfig(**mf_config) if mf_config else None
        if self.object and self.mf_config:
            self.object_params = {
                "class_label": self.mf_config.object.class_label,
                "label_map": [f"p{name}" for name in self.mf_config.object.class_names],
            }

        self.feature_map = onnx_feature_map
        self.aux_sequence_object = "tracks"

        example_input_list = []
        self.salt_names = []
        self.input_names = []
        self.aux_sequence_index = 1

        for i, feature in enumerate(self.feature_map):
            if feature["name_salt"] == self.global_object:
                example_input_list.append(torch.rand(1, self.input_dims[self.global_object]))
            else:
                example_input_list.append(
                    torch.rand(1, 40, self.input_dims[feature["name_salt"]]).squeeze(0)
                )
            if feature["name_salt"] == self.aux_sequence_object:
                self.aux_sequence_index = i
            self.salt_names.append(feature["name_salt"])
            self.input_names.append(feature["name_athena_out"])
        self.example_input_array = tuple(example_input_list)

    @property
    def global_task(self):
        global_tasks = [t for t in self.model.tasks if t.input_name == self.global_object]
        assert len(global_tasks) <= 1, "Multi global task ONNX models are not yet supported."
        return global_tasks[0] if global_tasks else None

    @property
    def model_name(self) -> str:
        # aux variables are not allowed to have dashes in Athena
        return self.name.replace("-", "_")

    @property
    def output_names(self) -> list[str]:
        """The output names are a list of strings, one for each output of the model."""
        # get the global task output names
        outputs = self.global_task.output_names if self.global_task else []

        # aux task output names
        if self.include_aux:
            if "track_origin" in [t.name for t in self.model.tasks]:
                out_name = f"{self.model_name}_TrackOrigin"
                outputs.append(out_name)

            if "track_vertexing" in [t.name for t in self.model.tasks]:
                out_name = f"{self.model_name}_VertexIndex"
                outputs.append(out_name)
        if self.object:
            regression_task = [
                t for t in self.model.tasks if t.input_name == "objects" and t.name == "regression"
            ]
            assert len(regression_task) == 1, "Object outputs require a regression task"
            # First we append the leading jet regression variables
            outputs += [
                f"{self.model_name}_leading_{self.object}_{v}" for v in regression_task[0].targets
            ]
            outputs += [f"{self.model_name}_{self.object}Index"]

        return outputs

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Let ONNX know which inputs/outputs have dynamic shape (i.e. can vary in length)."""
        # dynamic inputs
        dynamic_axes = {}
        for feature in self.feature_map:
            if not feature["is_global"]:
                dynamic_axes.update({feature["name_athena_out"]: {0: feature["athena_num_name"]}})

        # dynamic outputs
        if self.include_aux:
            if "track_origin" in [t.name for t in self.model.tasks]:
                out_name = f"{self.model_name}_TrackOrigin"
                dynamic_axes[out_name] = {0: "n_tracks"}
            if "track_vertexing" in [t.name for t in self.model.tasks]:
                out_name = f"{self.model_name}_VertexIndex"
                dynamic_axes[out_name] = {0: "n_tracks"}
        if self.object:
            out_name = f"{self.model_name}_{self.object}"
            dynamic_axes[out_name] = {0: "n_tracks"}
        return dynamic_axes

    def forward(self, *args):  # type: ignore[override]
        """Forward pass through the model.

        The arguments must be passed in the same order they appear in the feature map.
        """
        # in athena the jets have a batch dim but the tracks don't, so add it here
        assert len(args) == len(self.salt_names), "Number of inputs does not match feature map."
        assert (
            len(args[0].shape) == 2
        ), "Jets should have a batch dimension, and variable dimension but not a sequence dimension"
        input_dict = {self.global_object: args[0]}
        input_dict.update({
            self.salt_names[i]: args[i].unsqueeze(0) for i in range(1, len(self.salt_names))
        })

        # forward pass
        outputs = super().forward(input_dict, None)[0]

        # get the global task output
        onnx_outputs = ()
        onnx_outputs += self.global_task.get_onnx(
            outputs[self.global_object][self.global_task.name]
        )

        # add aux outputs
        if self.include_aux:
            tracks = args[self.aux_sequence_index].unsqueeze(0)
            track_outs = outputs[self.aux_sequence_object]
            if "track_origin" in track_outs:
                outputs_track = torch.argmax(track_outs["track_origin"], dim=-1)
                outputs_track = outputs_track.squeeze(0).char()
                onnx_outputs += (outputs_track,)

            if "track_vertexing" in track_outs:
                pad_mask = torch.zeros(tracks.shape[:-1], dtype=torch.bool)
                edge_scores = track_outs["track_vertexing"]
                vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
                vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
                onnx_outputs += (vertex_list.reshape(-1).char(),)

        if self.object:
            assert "objects" in outputs, "No MF objects in outputs"
            regression_tasks = [
                t for t in self.model.tasks if t.input_name == "objects" and t.name == "regression"
            ]
            assert len(regression_tasks) == 1, "Object outputs require a regression task"
            regression_task = regression_tasks[0]

            # Get the (hopefully) correctly (un)scaled regression predictions
            for i, t in enumerate(regression_task.targets):
                unscaled_preds = regression_task.scaler.inverse(
                    t, outputs["objects"]["regression"][:, :, i]
                )
                outputs["objects"]["regression"][:, :, i] = unscaled_preds

            # Extract the mf outputs.
            # TODO: write all regression values, this will require work on the athena end as well
            # https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/issues/53
            leading_reg, indices, class_probs, regression = get_maskformer_outputs(  # noqa: F841
                outputs["objects"], apply_reorder=True
            )

            for r in leading_reg[0]:
                onnx_outputs += (r,)
            onnx_outputs += (indices.reshape(-1).char(),)

        return onnx_outputs


def get_default_onnx_feature_map(track_selection, inputs):
    feature_map = [
        {
            "name_athena_in": "jet_var",
            "name_athena_out": "jet_features",
            "name_salt": "jets",
            "is_global": True,
        },
    ]

    if "tracks" in inputs:
        feature_map.append({
            "name_athena_in": f"tracks_{track_selection}_sd0sort",
            "name_athena_out": "track_features",
            "athena_num_name": "n_tracks",
            "name_salt": "tracks",
            "is_global": False,
        })

    if "flow" in inputs:
        feature_map.append({
            "name_athena_in": f"flows_{track_selection}_sd0sort",
            "name_athena_out": "flow_features",
            "athena_num_name": "n_flow",
            "name_salt": "flow",
            "is_global": False,
        })

    if "hits" in inputs:
        feature_map.append({
            "name_athena_in": "hits_var",
            "name_athena_out": "hit_features",
            "athena_num_name": "n_hits",
            "name_salt": "hits",
            "is_global": False,
        })

    return feature_map


def main(args=None):
    # parse args
    args = parse_args(args)

    if not args.force:
        check_for_uncommitted_changes(Path(__file__).parent)

    if not (config_path := args.config):
        config_path = args.ckpt_path.parents[1] / "config.yaml"
        assert config_path.is_file(), f"Could not find config file at {config_path}"

    config = yaml.safe_load(config_path.read_text())
    # Default config that only uses jets and tracks sorted in a default way

    onnx_feature_map = get_default_onnx_feature_map(
        args.track_selection, list(config["data"]["variables"].keys())
    )

    # instantiate pytorch and wrapper models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pt_model = ModelWrapper.load_from_checkpoint(
            args.ckpt_path,
            map_location=torch.device("cpu"),
            norm_config=config["model"]["norm_config"],
        )
        pt_model.eval()
        pt_model.float()

        if args.object_name:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            mf_config = config["data"].get("mf_config")
            if not mf_config:
                raise ValueError("No mf_config in config")
        else:
            mf_config = {}

        onnx_model = ONNXModel.load_from_checkpoint(
            args.ckpt_path,
            onnx_feature_map=onnx_feature_map,
            name=args.name,
            include_aux=args.include_aux,
            object_name=args.object_name,
            mf_config=mf_config,
            map_location=torch.device("cpu"),
            norm_config=config["model"]["norm_config"],
        )

        onnx_model.eval()
        change_attn_backends(
            onnx_model.model, "torch-math"
        )  # Only applies to transformer_v2 layers

    print("\n" + "-" * 100)
    print("Converting model to ONNX...")
    print("-" * 100)

    # configure paths
    base_path = args.ckpt_path.parent.parent
    onnx_path = base_path / "network.onnx"
    if onnx_path.exists() and not args.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")

    # export
    onnx_model.to_onnx(
        onnx_path,
        opset_version=16,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes=onnx_model.dynamic_axes,
    )

    # add metadata
    add_metadata(
        config_path,
        config,
        args.ckpt_path,
        onnx_path,
        onnx_model.model_name,
        onnx_model.output_names,
        onnx_feature_map,
    )
    seq_names_onnx = []
    seq_names_salt = []
    for feature in onnx_feature_map:
        if feature["is_global"]:
            continue
        seq_names_salt.append(feature["name_salt"])
        seq_names_onnx.append(feature["name_athena_out"])

    # validate pytorch and exported onnx models
    compare_outputs(
        pt_model,
        onnx_path,
        args.include_aux,
        seq_names_salt=seq_names_salt,
        seq_names_onnx=seq_names_onnx,
    )
    print("\n" + "-" * 100)
    print(f"Done! Saved ONNX model at {onnx_path}")
    print("-" * 100)
    print()


def add_metadata(
    config_path,
    config,
    ckpt_path,
    onnx_path,
    model_name,
    output_names,
    onnx_feature_map,
):
    print("\n" + "-" * 100)
    print("Adding Metadata...")

    # load and check the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # add metadata
    metadata = {"ckpt_path": str(ckpt_path.resolve()), "layers": [], "nodes": []}
    metadata["config.yaml"] = config
    metadata["metadata.yaml"] = yaml.safe_load((config_path.parent / "metadata.yaml").read_text())
    metadata["salt_export_hash"] = get_git_hash(Path(__file__).parent)

    # careful - this stuff is used in athena
    metadata["onnx_model_version"] = "v1"
    metadata["output_names"] = output_names
    metadata["model_name"] = model_name
    metadata["inputs"] = []
    metadata["input_sequences"] = []
    for feature in onnx_feature_map:
        if feature["is_global"]:  # global features similar to jet features
            metadata["inputs"] += [
                {
                    "name": feature["name_athena_in"],
                    "variables": [
                        {"name": k.removesuffix("_btagJes"), "offset": 0.0, "scale": 1.0}
                        for k in config["data"]["variables"][feature["name_salt"]]
                    ],
                }
            ]
        else:  # feature sequences simmilar to tracks features
            metadata["input_sequences"] += [
                {
                    "name": feature["name_athena_in"],
                    "variables": [
                        {"name": k, "offset": 0.0, "scale": 1.0}
                        for k in config["data"]["variables"][feature["name_salt"]]
                    ],
                },
            ]

    # write metadata as json string
    metadata = {"gnn_config": json.dumps(metadata)}

    for k, v in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = v

    onnx_model.doc_string = model_name
    onnx.save(onnx_model, onnx_path)
    print("-" * 100)


if __name__ == "__main__":
    main()
