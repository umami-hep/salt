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
from salt.utils.get_structured_input_dict import get_structured_input_dict
from salt.utils.union_find import get_node_assignment_jit

torch.manual_seed(42)
# https://gitlab.cern.ch/atlas/athena/-/blob/main/PhysicsAnalysis/JetTagging/FlavorTagInference/Root/TracksLoader.cxx#L74
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
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Space separated list of tasks to include in the ONNX model. "
            "By default all global tasks are included.",
        ),
    )
    parser.add_argument(
        "-mf",
        "--object_name",
    )
    parser.add_argument(
        "--combine_outputs",
        type=str,
        nargs="+",
        default=[],
        help="Space seperated list of items described in 'parse_output_combination'",
    )
    parser.add_argument(
        "-r",
        "--rename",
        type=str,
        nargs="+",
        default=[],
        help="Space seperated list of them form 'oldname1:newname1 oldname2:newname2'",
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
        variable_map: dict,
        name: str | None = None,
        tasks_to_output: list[str] | None = None,
        object_name: str | None = None,
        mf_config: dict | None = None,
        combine_outputs: list[tuple] | None = None,
        rename_outputs: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name or self.name
        assert "_" not in self.name, "Model name cannot contain underscores."
        assert "-" not in self.name, "Model name cannot contain dashes."
        for task in self.model.tasks:
            task.model_name = self.name
        self.tasks_to_output = tasks_to_output or []
        if not self.tasks_to_output:
            self.tasks_to_output = [
                t.name for t in self.model.tasks if t.input_name == self.global_object
            ]
            print("No tasks specified, dumping all global tasks: ", self.tasks_to_output)
        else:
            # Need aux tasks to be after global tasks, and in the following order for checks to work
            for t in ["track_origin", "track_vertexing", "track_type"]:
                if t in self.tasks_to_output:
                    self.tasks_to_output.remove(t)
                    self.tasks_to_output.append(t)

        self.combine_outputs = combine_outputs or []
        self.rename_outputs = rename_outputs or {}
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
        self.variable_map = variable_map

        example_input_list = []
        num_tracks = 40
        self.salt_names = []
        self.input_names = []
        self.aux_sequence_index = 1

        for i, feature in enumerate(self.feature_map):
            if feature["name_salt"] == self.global_object:
                example_input_list.append(torch.rand(1, self.input_dims[self.global_object]))
            else:
                example_input_list.append(
                    torch.rand(1, num_tracks, self.input_dims[feature["name_salt"]]).squeeze(0)
                )
            if feature["name_salt"] == self.aux_sequence_object:
                self.aux_sequence_index = i
            self.salt_names.append(feature["name_salt"])
            self.input_names.append(feature["name_athena_out"])
        self.example_input_array = tuple(example_input_list)

    @property
    def global_tasks(self):
        return [t for t in self.model.tasks if t.input_name == self.global_object]

    @property
    def output_names(self) -> list[str]:
        """The output names are a list of strings, one for each output of the model."""
        outputs = []
        for t in self.global_tasks:
            if t.name in self.tasks_to_output:
                outputs += t.output_names

        # Allow renaming and combining of global tasks
        if self.rename_outputs:
            for oldname, newname in self.rename_outputs.items():
                full_oldname = f"{self.name}_{oldname}"
                full_newname = f"{self.name}_{newname}"
                assert full_oldname in outputs, f"Output {oldname} not found in outputs"
                idx = outputs.index(full_oldname)
                outputs[idx] = full_newname

        if self.combine_outputs:
            for output_name, parsed_inputs in self.combine_outputs:
                for _, input_name in parsed_inputs:
                    assert (
                        f"{self.name}_{input_name}" in outputs
                    ), f"Output {input_name} not found in outputs"
                outputs.append(f"{self.name}_{output_name}")

        if "track_origin" in self.tasks_to_output:
            out_name = f"{self.name}_TrackOrigin"
            outputs.append(out_name)

        if "track_vertexing" in self.tasks_to_output:
            out_name = f"{self.name}_VertexIndex"
            outputs.append(out_name)

        if "track_type" in self.tasks_to_output:
            out_name = f"{self.name}_TrackType"
            outputs.append(out_name)
        if self.object:
            regression_task = [
                t for t in self.model.tasks if t.input_name == "objects" and t.name == "regression"
            ]
            assert len(regression_task) == 1, "Object outputs require a regression task"
            # First we append the leading jet regression variables
            outputs += [
                f"{self.name}_leading_{self.object}_{v}" for v in regression_task[0].targets
            ]
            outputs += [f"{self.name}_{self.object}Index"]

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
        if "track_origin" in self.tasks_to_output:
            out_name = f"{self.name}_TrackOrigin"
            dynamic_axes[out_name] = {0: "n_tracks"}
        if "track_vertexing" in self.tasks_to_output:
            out_name = f"{self.name}_VertexIndex"
            dynamic_axes[out_name] = {0: "n_tracks"}
        if "track_type" in self.tasks_to_output:
            out_name = f"{self.name}_TrackType"
            dynamic_axes[out_name] = {0: "n_tracks"}
        if self.object:
            out_name = f"{self.name}_{self.object}"
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

        structured_input_dict = get_structured_input_dict(
            input_dict, self.variable_map, self.global_object
        )

        masks = {
            k: torch.zeros((1, args[i].shape[0]), dtype=torch.bool)
            for i, k in enumerate(self.salt_names)
            if k != self.global_object
        }

        outputs = super().forward(input_dict, masks)[0]

        # get the global tasks outputs
        onnx_outputs = sum(
            (
                t.get_onnx(outputs[self.global_object][t.name], labels=structured_input_dict)
                for t in self.global_tasks
            ),
            (),
        )

        if self.combine_outputs:
            onnx_out_names = self.output_names
            for _, parsed_inputs in self.combine_outputs:
                output = sum([
                    scale * onnx_outputs[onnx_out_names.index(f"{self.name}_{input_name}")]
                    for scale, input_name in parsed_inputs
                ])
                onnx_outputs += (output,)

        # add aux outputs
        if "track_origin" in self.tasks_to_output:
            track_origins = outputs[self.aux_sequence_object]["track_origin"]
            track_origins = torch.concatenate(
                [track_origins, torch.zeros((1, 1, track_origins.shape[-1]))], dim=1
            )

            outputs_track = torch.argmax(track_origins, dim=-1)[:, :-1]
            outputs_track = outputs_track.squeeze(0).char()

            onnx_outputs += (outputs_track,)

        if "track_vertexing" in self.tasks_to_output:
            tracks = args[self.aux_sequence_index].unsqueeze(0)
            pad_mask = torch.zeros(tracks.shape[:-1], dtype=torch.bool)
            edge_scores = outputs[self.aux_sequence_object]["track_vertexing"]

            vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
            vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
            onnx_outputs += (vertex_list.reshape(-1).char(),)

        if "track_type" in self.tasks_to_output:
            track_type = outputs[self.aux_sequence_object]["track_type"]
            track_type = torch.concatenate(
                [track_type, torch.zeros((1, 1, track_type.shape[-1]))], dim=1
            )

            outputs_track = torch.argmax(track_type, dim=-1)[:, :-1]
            outputs_track = outputs_track.squeeze(0).char()
            onnx_outputs += (outputs_track,)

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


def get_default_onnx_feature_map(track_selection, inputs, global_name):
    feature_map = []

    for input_name in inputs:
        if input_name == global_name:
            feature_map.append({
                "name_athena_in": f"{global_name.removesuffix('s')}_var",
                "name_athena_out": f"{global_name.removesuffix('s')}_features",
                "name_salt": global_name,
                "is_global": True,
            })
        elif "tracks" in input_name or "flows" in input_name:
            base_name = input_name.split("_")[0]
            feature_map.append({
                "name_athena_in": f"{base_name}_{track_selection}_sd0sort",
                "name_athena_out": f"{base_name.removesuffix('s')}_features",
                "athena_num_name": f"n_{base_name}",
                "name_salt": base_name,
                "is_global": False,
            })
        else:
            feature_map.append({
                "name_athena_in": f"{input_name}_var",
                "name_athena_out": f"{input_name.removesuffix('s')}_features",
                "athena_num_name": f"n_{input_name}",
                "name_salt": input_name,
                "is_global": False,
            })

    return feature_map


def parse_output_combination(arg_str):
    """Parses a string which represents combinations of outputs to be included in the final
    onnx model.

    Parameters
    ----------
    arg_str : str
        String of the form:
            - light,0.5*q,0.1*g,0.7*s
                Represents p_light=0.5*p_q + 0.1*p_g + 0.7*p_s
            - light,q,g,s
                Represents p_light=p_q + p_g + p_s
        A commar separated list of where the first is the output name, and each other item
        represents a scaled sum

    Returns
    -------
    output_name : str
        The name of the output
    parsed_inputs : list[tuple[float, str]]
        A list of tuples where the first element is the scale and the second is the input name
    """
    all_terms = arg_str.split(",")
    output_name = all_terms[0]
    terms = all_terms[1:]
    parsed_inputs = []
    for term in terms:
        if "*" in term:
            scale, input_name = term.split("*")
            parsed_inputs.append((float(scale), input_name))
        else:
            parsed_inputs.append((1, term))
    return output_name, parsed_inputs


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
        args.track_selection,
        list(config["data"]["variables"].keys()),
        config["data"]["global_object"],
    )

    combine_outputs = []
    if args.combine_outputs:
        for output in args.combine_outputs:
            output_name, parsed_inputs = parse_output_combination(output)
            combine_outputs.append((output_name, parsed_inputs))

    rename_outputs = dict(x.split(":") for x in args.rename)
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
        change_attn_backends(pt_model.model, "torch-math")

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
            variable_map=config["data"]["variables"],
            name=args.name,
            tasks_to_output=args.tasks,
            object_name=args.object_name,
            mf_config=mf_config,
            map_location=torch.device("cpu"),
            norm_config=config["model"]["norm_config"],
            combine_outputs=combine_outputs,
            rename_outputs=rename_outputs,
        )

        onnx_model.eval()
        onnx_model.float()
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
        onnx_model.name,
        onnx_model.output_names,
        onnx_feature_map,
        combine_outputs,
        rename_outputs,
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
        global_object=config["data"]["global_object"],
        seq_names_salt=seq_names_salt,
        seq_names_onnx=seq_names_onnx,
        variable_map=config["data"]["variables"],
        tasks_to_output=onnx_model.tasks_to_output,
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
    combine_outputs,
    rename_outputs,
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
    metadata["combine_outputs"] = combine_outputs
    metadata["rename_outputs"] = rename_outputs
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
