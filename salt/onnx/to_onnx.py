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

# Ensure deterministic behavior for synthetic tests
torch.manual_seed(42)

# Track selection tags that mirror Athena-side choices
# (see link below for the loader code used on the Athena side)
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


def parse_args(args: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for exporting a SALT model to ONNX.

    Parameters
    ----------
    args : list[str] | None
        Argument list to parse (as from ``sys.argv[1:]``). If ``None``,
        arguments are read from ``sys.argv`` by :func:`argparse.ArgumentParser.parse_args`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    # Configure argparse with defaults and helpful descriptions
    parser = argparse.ArgumentParser(
        description="A script to convert a salt model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint path (required)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Checkpoint path.",
        required=True,
    )
    # Training config (optional; inferred if not provided)
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Saved training config. If not provided, look in the parent directory of `ckpt_path`.",
        required=False,
    )
    # Track selection type (must match Athena-side loader choices)
    parser.add_argument(
        "-t",
        "--track_selection",
        type=str,
        help="Track selection, must match `trk_select_regexes` in 'DataPrepUtilities.cxx'",
        choices=TRACK_SELECTIONS,
        default="r22default",
    )
    # Optional name override for model outputs (prefix)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Model name, used in the *_px outputs. Taken from the config if not provided",
        required=False,
    )
    # Overwrite existing ONNX file
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing exported ONNX model.",
        action="store_true",
    )
    # Subset of tasks to include in the export (by default: all global tasks)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Space separated list of tasks to include in the ONNX model. "
            "By default all global tasks are included."
        ),
    )
    # Optional: include MaskFormer object outputs (requires mf_config)
    parser.add_argument(
        "-mf",
        "--object_name",
    )
    # Combine multiple outputs linearly into a new one (e.g., 'light,0.5*q,0.5*g')
    parser.add_argument(
        "--combine_outputs",
        type=str,
        nargs="+",
        default=[],
        help="Space seperated list of items described in 'parse_output_combination'",
    )
    # Rename outputs on export (e.g., 'old:new')
    parser.add_argument(
        "-r",
        "--rename",
        type=str,
        nargs="+",
        default=[],
        help="Space seperated list of them form 'oldname1:newname1 oldname2:newname2'",
    )
    # Allow running with uncommitted changes (suppress safety check)
    parser.add_argument(
        "-f",
        "--force",
        help="Run with uncomitted changes.",
        action="store_true",
    )

    # Parse provided args list (or sys.argv if None)
    return parser.parse_args(args)


class ONNXModel(ModelWrapper):
    """ONNX export wrapper for a trained SALT model.

    This wrapper:
      - sets up example inputs for tracing/export,
      - enforces a consistent output naming scheme (including renames/combines),
      - exposes dynamic axes information,
      - and formats outputs for ONNX consumers (e.g., Athena).
    """

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
        # Initialize base wrapper (loads underlying model and config)
        super().__init__(**kwargs)

        # Enforce output naming constraints and propagate name to tasks
        self.name = name or self.name
        assert "_" not in self.name, "Model name cannot contain underscores."
        assert "-" not in self.name, "Model name cannot contain dashes."
        for task in self.model.tasks:
            task.model_name = self.name

        # Decide which tasks to export (defaults to all global tasks)
        self.tasks_to_output = tasks_to_output or []
        if not self.tasks_to_output:
            self.tasks_to_output = [
                t.name for t in self.model.tasks if t.input_name == self.global_object
            ]
            print("No tasks specified, dumping all global tasks: ", self.tasks_to_output)
        else:
            # Ensure auxiliary sequence tasks (track_*) appear after global tasks
            # and in the order expected by downstream checks
            for t in ["track_origin", "track_vertexing", "track_type"]:
                if t in self.tasks_to_output:
                    self.tasks_to_output.remove(t)
                    self.tasks_to_output.append(t)

        # Configure optional output rewrites (renames and linear combinations)
        self.combine_outputs = combine_outputs or []
        self.rename_outputs = rename_outputs or {}

        # If we export MaskFormer object outputs, both name and config must be set
        if sum([bool(object_name), bool(mf_config)]) not in {0, 2}:
            raise ValueError("If one of object name or mf config is defined, so must the other.")
        self.object = object_name
        self.mf_config = MaskformerConfig(**mf_config) if mf_config else None
        if self.object and self.mf_config:
            self.object_params = {
                "class_label": self.mf_config.object.class_label,
                "label_map": [f"p{name}" for name in self.mf_config.object.class_names],
            }

        # Store feature mapping and variable map used to form inputs
        self.feature_map = onnx_feature_map
        self.aux_sequence_object = "tracks"
        self.variable_map = variable_map

        # Build example inputs aligned with the feature map for ONNX tracing
        example_input_list = []
        num_tracks = 40
        self.salt_names = []
        self.input_names = []
        self.aux_sequence_index = 1

        # Iterate in feature-map order to create per-input placeholders
        for i, feature in enumerate(self.feature_map):
            if feature["name_salt"] == self.global_object:
                example_input_list.append(torch.rand(1, self.input_dims[self.global_object]))
            elif feature["name_salt"] == "EDGE":
                example_input_list.append(
                    torch.rand(
                        1, num_tracks, num_tracks, self.input_dims[feature["name_salt"]]
                    ).squeeze(0)
                )
            else:
                example_input_list.append(
                    torch.rand(1, num_tracks, self.input_dims[feature["name_salt"]]).squeeze(0)
                )
            if feature["name_salt"] == self.aux_sequence_object:
                self.aux_sequence_index = i
            self.salt_names.append(feature["name_salt"])
            self.input_names.append(feature["name_athena_out"])

        # Tuple of inputs in export order (ONNX expects a tuple)
        self.example_input_array = tuple(example_input_list)

    @property
    def global_tasks(self) -> list:
        """Tasks operating on the global object stream.

        Returns
        -------
        list
            List of task modules associated with :attr:`global_object`.
        """
        return [t for t in self.model.tasks if t.input_name == self.global_object]

    @property
    def output_names(self) -> list[str]:
        """Final list of ONNX output names (after renames/combines).

        Includes:
          - global task outputs,
          - optional renamed/combined outputs,
          - optional auxiliary per-track outputs (``track_*``),
          - optional MaskFormer object outputs.

        Returns
        -------
        list[str]
            Ordered output names for the exported ONNX model.
        """
        # Start with global task outputs (ordered by tasks_to_output)
        outputs = []
        for t in self.global_tasks:
            if t.name in self.tasks_to_output:
                outputs += t.output_names

        # Apply output renaming (checked for existence)
        if self.rename_outputs:
            for oldname, newname in self.rename_outputs.items():
                full_oldname = f"{self.name}_{oldname}"
                full_newname = f"{self.name}_{newname}"
                assert full_oldname in outputs, f"Output {oldname} not found in outputs"
                idx = outputs.index(full_oldname)
                outputs[idx] = full_newname

        # Add combined outputs (just names here; values are produced in forward)
        if self.combine_outputs:
            for output_name, parsed_inputs in self.combine_outputs:
                for _, input_name in parsed_inputs:
                    assert f"{self.name}_{input_name}" in outputs, (
                        f"Output {input_name} not found in outputs"
                    )
                outputs.append(f"{self.name}_{output_name}")

        # Append auxiliary per-track outputs if requested
        if "track_origin" in self.tasks_to_output:
            out_name = f"{self.name}_TrackOrigin"
            outputs.append(out_name)

        if "track_vertexing" in self.tasks_to_output:
            out_name = f"{self.name}_VertexIndex"
            outputs.append(out_name)

        if "track_type" in self.tasks_to_output:
            out_name = f"{self.name}_TrackType"
            outputs.append(out_name)

        # If exporting MaskFormer object outputs, append the object-specific fields
        if self.object:
            regression_task = [
                t for t in self.model.tasks if t.input_name == "objects" and t.name == "regression"
            ]
            assert len(regression_task) == 1, "Object outputs require a regression task"
            # Leading-object regression variables
            outputs += [
                f"{self.name}_leading_{self.object}_{v}" for v in regression_task[0].targets
            ]
            # Mask/object index output
            outputs += [f"{self.name}_{self.object}Index"]

        return outputs

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Dynamic axes mapping for ONNX export.

        Returns
        -------
        dict[str, dict[int, str]]
            Mapping from IO tensor names to a dict of axis-index → dynamic-axis-name.
            Sequence inputs/outputs (e.g., tracks) have the first axis (0) marked dynamic.
        """
        # Mark variable-length input sequences as dynamic along axis 0
        dynamic_axes: dict[str, dict[int, str]] = {}
        for feature in self.feature_map:
            if "EDGE" in feature["name_athena_out"]:
                dynamic_axes.update({
                    feature["name_athena_out"]: {
                        0: feature["athena_num_name"],
                        1: feature["athena_num_name"],
                    }
                })
            elif not feature["is_global"]:
                dynamic_axes.update({feature["name_athena_out"]: {0: feature["athena_num_name"]}})

        # Mark variable-length per-track outputs as dynamic along axis 0
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

    def forward(self, *args: torch.Tensor):  # type: ignore[override]
        """Produce ONNX-friendly outputs given Athena-ordered feature tensors.

        Notes
        -----
        The arguments must be provided in the exact order specified by the
        ``onnx_feature_map``. The first argument (global object) has shape
        ``[batch, features]``. Sequence inputs (e.g., tracks) have shape
        ``[length, features]`` (Athena-style, no batch dimension).

        Parameters
        ----------
        *args : torch.Tensor
            Feature tensors ordered to match :attr:`input_names`. The first
            tensor corresponds to the global object; the remaining are sequences.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Flat tuple of ONNX outputs in the order given by :attr:`output_names`.
        """
        # Basic input checks for shape/order consistency
        assert len(args) == len(self.salt_names), "Number of inputs does not match feature map."
        assert len(args[0].shape) == 2, (
            "Jets should have a batch dimension, "
            "and variable dimension but not a sequence dimension"
        )

        # Build SALT-style input dict: add a batch dimension for sequences
        input_dict = {self.global_object: args[0]}
        input_dict.update({
            self.salt_names[i]: args[i].unsqueeze(0) for i in range(1, len(self.salt_names))
        })

        # Construct structured inputs (variable mapping, etc.) for tasks
        structured_input_dict = get_structured_input_dict(
            input_dict, self.variable_map, self.global_object
        )

        # Build padding masks (all valid by default for export-time inputs)
        masks = {
            k: torch.zeros((1, args[i].shape[0]), dtype=torch.bool)
            for i, k in enumerate(self.salt_names)
            if k != self.global_object and "EDGE" not in k
        }

        # Run the wrapped model forward (returns predictions dict and losses)
        outputs = super().forward(input_dict, masks)[0]

        # Gather global task outputs and convert to ONNX-friendly tensors
        onnx_outputs = sum(
            (
                t.get_onnx(outputs[self.global_object][t.name], labels=structured_input_dict)
                for t in self.global_tasks
            ),
            (),
        )

        # Optionally append new outputs formed as linear combinations of existing ones
        if self.combine_outputs:
            onnx_out_names = self.output_names
            for _, parsed_inputs in self.combine_outputs:
                output = sum(
                    scale * onnx_outputs[onnx_out_names.index(f"{self.name}_{input_name}")]
                    for scale, input_name in parsed_inputs
                )
                onnx_outputs += (output,)

        # Auxiliary per-track outputs: track_origin (argmax of per-track logits)
        if "track_origin" in self.tasks_to_output:
            track_origins = outputs[self.aux_sequence_object]["track_origin"]
            # Append a padded value to match ONNX expectations; remove later
            track_origins = torch.concatenate(
                [track_origins, torch.zeros((1, 1, track_origins.shape[-1]))], dim=1
            )
            outputs_track = torch.argmax(track_origins, dim=-1)[:, :-1]
            outputs_track = outputs_track.squeeze(0).char()
            onnx_outputs += (outputs_track,)

        # Auxiliary per-track outputs: track_vertexing (graph assignment)
        if "track_vertexing" in self.tasks_to_output:
            tracks = args[self.aux_sequence_index].unsqueeze(0)
            pad_mask = torch.zeros(tracks.shape[:-1], dtype=torch.bool)
            edge_scores = outputs[self.aux_sequence_object]["track_vertexing"]
            vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
            vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
            onnx_outputs += (vertex_list.reshape(-1).char(),)

        # Auxiliary per-track outputs: track_type (argmax of per-track logits)
        if "track_type" in self.tasks_to_output:
            track_type = outputs[self.aux_sequence_object]["track_type"]
            # Append a padded value to match ONNX expectations; remove later
            track_type = torch.concatenate(
                [track_type, torch.zeros((1, 1, track_type.shape[-1]))], dim=1
            )
            outputs_track = torch.argmax(track_type, dim=-1)[:, :-1]
            outputs_track = outputs_track.squeeze(0).char()
            onnx_outputs += (outputs_track,)

        # Optional MaskFormer object outputs: leading regression values and mask indices
        if self.object:
            assert "objects" in outputs, "No MF objects in outputs"
            regression_tasks = [
                t for t in self.model.tasks if t.input_name == "objects" and t.name == "regression"
            ]
            assert len(regression_tasks) == 1, "Object outputs require a regression task"
            regression_task = regression_tasks[0]

            # Invert scaling for object regression targets (per target)
            for i, t in enumerate(regression_task.targets):
                unscaled_preds = regression_task.scaler.inverse(
                    t, outputs["objects"]["regression"][:, :, i]
                )
                outputs["objects"]["regression"][:, :, i] = unscaled_preds

            # Convert masks/logits to usable outputs (leading reg, indices, class probs, full reg)
            leading_reg, indices, _, _ = get_maskformer_outputs(
                outputs["objects"], apply_reorder=True
            )

            # Emit leading-object regression scalars and the object indices
            for r in leading_reg[0]:
                onnx_outputs += (r,)
            onnx_outputs += (indices.reshape(-1).char(),)

        # Return a flat tuple of ONNX outputs in the expected order
        return onnx_outputs


def get_default_onnx_feature_map(
    track_selection: str, inputs: list[str], global_name: str, input_map: dict
) -> list[dict]:
    """Build a default Athena↔SALT feature mapping (typical jets+tracks setup).

    The mapping describes:
      - input tensor names expected by Athena (in/out),
      - SALT-side names,
      - whether inputs are global vs. sequences,
      - and the dynamic-length symbol for sequence inputs.

    Parameters
    ----------
    track_selection : str
        Track selection key that determines which track sequence is used (Athena-side).
    inputs : list[str]
        Training-time input names used in SALT (e.g., ``["jets", "tracks"]``).
    global_name : str
        Name of the global input stream (e.g., ``"jets"``).
    input_map: dict
        Dictionary containing underlying feature mapping (relevant for edge features).

    Returns
    -------
    list[dict]
        Feature map describing input/output names and sequence properties.
    """
    # Accumulate a list of feature descriptors (one per input)
    feature_map: list[dict] = []

    # Iterate through training inputs to construct names consistently
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
        # For backwards compatibility, due to mismatching
        # naming convention flow/flows between athena and TDD
        elif input_name == "flow":
            base_name = input_name.split("_")[0]
            feature_map.append({
                "name_athena_in": f"flows_{track_selection}_sd0sort",
                "name_athena_out": f"{base_name.removesuffix('s')}_features",
                "athena_num_name": f"n_{base_name}",
                "name_salt": base_name,
                "is_global": False,
            })
        elif "EDGE" in input_name:
            base_name = input_map[input_name].split("_")[0]
            feature_map.append({
                "name_athena_in": f"{input_name}_{base_name}",
                "name_athena_out": f"{input_name}_{base_name.removesuffix('s')}_features",
                "athena_num_name": f"n_{base_name}",
                "name_salt": input_name,
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


def parse_output_combination(arg_str: str) -> tuple[str, list[tuple[float, str]]]:
    """Parse a linear-combination spec for creating a new exported output.

    Examples
    --------
    - ``light,0.5*q,0.1*g,0.7*s`` → ``p_light = 0.5*p_q + 0.1*p_g + 0.7*p_s``
    - ``light,q,g,s`` → ``p_light = p_q + p_g + p_s``

    Parameters
    ----------
    arg_str : str
        Comma-separated specification: first item is the new output name,
        subsequent items are either feature names or scaled terms (``scale*name``).

    Returns
    -------
    output_name : str
        Name of the new output to be appended.
    parsed_inputs : list[tuple[float, str]]
        List of ``(scale, input_name)`` tuples to be summed.
    """
    # Split on commas and separate output name from term list
    all_terms = arg_str.split(",")
    output_name = all_terms[0]
    terms = all_terms[1:]

    # Convert each term into (scale, name); default scale is 1
    parsed_inputs: list[tuple[float, str]] = []
    for term in terms:
        if "*" in term:
            scale, input_name = term.split("*")
            parsed_inputs.append((float(scale), input_name))
        else:
            parsed_inputs.append((1, term))
    return output_name, parsed_inputs


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint: load model, export to ONNX, attach metadata, and validate.

    Parameters
    ----------
    args : list[str] | None, optional
        Argument vector for the CLI. If ``None``, parse from ``sys.argv``.

    Raises
    ------
    ValueError
        If mf_config is not in config
    FileExistsError
        If the onnx file already exists
    """
    # Parse the command-line arguments
    parsed_args = parse_args(args)

    # Enforce clean working tree unless --force is specified
    if not parsed_args.force:
        check_for_uncommitted_changes(Path(__file__).parent)

    # Resolve config path (either provided or inferred near the checkpoint)
    if not (config_path := parsed_args.config):
        config_path = parsed_args.ckpt_path.parents[1] / "config.yaml"
        assert config_path.is_file(), f"Could not find config file at {config_path}"

    # Load training configuration
    config = yaml.safe_load(config_path.read_text())

    # Build a default feature map (jets + tracks) from config
    onnx_feature_map = get_default_onnx_feature_map(
        parsed_args.track_selection,
        list(config["data"]["variables"].keys()),
        config["data"]["global_object"],
        config["data"]["input_map"],
    )

    # Parse output combination specs into a list of (name, [(scale, term), ...])
    combine_outputs: list[tuple[str, list[tuple[float, str]]]] = []
    if parsed_args.combine_outputs:
        for output in parsed_args.combine_outputs:
            output_name, parsed_inputs = parse_output_combination(output)
            combine_outputs.append((output_name, parsed_inputs))

    # Parse renames specified as "old:new" pairs
    rename_outputs = dict(x.split(":") for x in parsed_args.rename)

    # Load the PyTorch model and switch attention backend for stable export
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pt_model = ModelWrapper.load_from_checkpoint(
            parsed_args.ckpt_path,
            map_location=torch.device("cpu"),
            norm_config=config["model"]["norm_config"],
        )
        pt_model.eval()
        pt_model.float()
        change_attn_backends(pt_model.model, "torch-math")

        # If object outputs are requested, load MF config from the training config
        if parsed_args.object_name:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            mf_config = config["data"].get("mf_config")
            if not mf_config:
                raise ValueError("No mf_config in config")
        else:
            mf_config = {}

        # Build the ONNX wrapper around the trained model
        onnx_model = ONNXModel.load_from_checkpoint(
            parsed_args.ckpt_path,
            onnx_feature_map=onnx_feature_map,
            variable_map=config["data"]["variables"],
            name=parsed_args.name,
            tasks_to_output=parsed_args.tasks,
            object_name=parsed_args.object_name,
            mf_config=mf_config,
            map_location=torch.device("cpu"),
            norm_config=config["model"]["norm_config"],
            combine_outputs=combine_outputs,
            rename_outputs=rename_outputs,
        )

        # Set to eval/float and ensure attention uses torch-math kernels
        onnx_model.eval()
        onnx_model.float()
        change_attn_backends(
            onnx_model.model, "torch-math"
        )  # Only applies to transformer_v2 layers

    # Announce export start
    print("\n" + "-" * 100)
    print("Converting model to ONNX...")
    print("-" * 100)

    # Construct output ONNX path next to the checkpoint (unless overridden)
    base_path = parsed_args.ckpt_path.parent.parent
    onnx_path = base_path / "network.onnx"
    if onnx_path.exists() and not parsed_args.overwrite:
        raise FileExistsError(f"Found existing file '{onnx_path}'.")

    # Export the model to ONNX using our wrapper's shapes and dynamic axes
    onnx_model.to_onnx(
        onnx_path,
        opset_version=16,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes=onnx_model.dynamic_axes,
    )

    # Attach metadata (config, hashes, IO schema) to the ONNX file
    add_metadata(
        config_path=config_path,
        config=config,
        ckpt_path=parsed_args.ckpt_path,
        onnx_path=onnx_path,
        model_name=onnx_model.name,
        output_names=onnx_model.output_names,
        onnx_feature_map=onnx_feature_map,
        combine_outputs=combine_outputs,
        rename_outputs=rename_outputs,
    )

    # Prepare sequence names for ONNX runtime validation
    seq_names_onnx: list[str] = []
    seq_names_salt: list[str] = []
    edge_name_onnx = None
    edge_name_salt = None
    for feature in onnx_feature_map:
        if feature["is_global"]:
            continue
        if "EDGE" in feature["name_salt"]:
            edge_name_salt = feature["name_salt"]
            edge_name_onnx = feature["name_athena_out"]
            continue
        seq_names_salt.append(feature["name_salt"])
        seq_names_onnx.append(feature["name_athena_out"])

    # Compare PyTorch vs ONNX numerics across many synthetic cases
    compare_outputs(
        pt_model,
        onnx_path,
        global_object=config["data"]["global_object"],
        seq_names_salt=seq_names_salt,
        seq_names_onnx=seq_names_onnx,
        variable_map=config["data"]["variables"],
        tasks_to_output=onnx_model.tasks_to_output,
        edge_name_salt=edge_name_salt,
        edge_name_onnx=edge_name_onnx,
    )

    # Final success message with path of saved model
    print("\n" + "-" * 100)
    print(f"Done! Saved ONNX model at {onnx_path}")
    print("-" * 100)
    print()


def add_metadata(
    config_path: Path,
    config: dict,
    ckpt_path: Path,
    onnx_path: Path,
    model_name: str,
    output_names: list[str],
    onnx_feature_map: list[dict],
    combine_outputs: list[tuple[str, list[tuple[float, str]]]],
    rename_outputs: dict[str, str],
) -> None:
    """Attach training/config metadata and IO schema to the exported ONNX model.

    Parameters
    ----------
    config_path : Path
        Path to the training configuration YAML file.
    config : dict
        Parsed training configuration mapping.
    ckpt_path : Path
        Path to the original checkpoint file (.ckpt).
    onnx_path : Path
        Destination path of the exported ONNX file.
    model_name : str
        Name to store in the ONNX model doc string and metadata.
    output_names : list[str]
        Ordered list of output tensor names for the ONNX graph.
    onnx_feature_map : list[dict]
        Feature mapping used at export time (Athena↔SALT).
    combine_outputs : list[tuple[str, list[tuple[float, str]]]]
        Output combinations specified as (new_name, [(scale, input_name), ...]).
    rename_outputs : dict[str, str]
        Mapping from old output name to new output name.
    """
    # Announce metadata step
    print("\n" + "-" * 100)
    print("Adding Metadata...")

    # Load and validate the ONNX graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Build a metadata dict including paths, hashes, and I/O schema
    metadata: dict = {"ckpt_path": str(ckpt_path.resolve()), "layers": [], "nodes": []}
    metadata["config.yaml"] = config
    metadata["metadata.yaml"] = yaml.safe_load((config_path.parent / "metadata.yaml").read_text())
    metadata["salt_export_hash"] = get_git_hash(Path(__file__).parent)

    # Versioning and top-level schema keys used by Athena
    metadata["onnx_model_version"] = "v1"
    metadata["output_names"] = output_names
    metadata["model_name"] = model_name
    metadata["inputs"] = []
    metadata["input_sequences"] = []
    metadata["combine_outputs"] = combine_outputs
    metadata["rename_outputs"] = rename_outputs

    # Describe inputs and sequences (variable lists are copied from training config)
    for feature in onnx_feature_map:
        if feature["is_global"]:
            # Global (e.g., jet) inputs — offsets/scales are informational placeholders
            metadata["inputs"] += [
                {
                    "name": feature["name_athena_in"],
                    "variables": [
                        {"name": k.removesuffix("_btagJes"), "offset": 0.0, "scale": 1.0}
                        for k in config["data"]["variables"][feature["name_salt"]]
                    ],
                }
            ]
        else:
            # Sequence inputs (e.g., tracks)
            metadata["input_sequences"] += [
                {
                    "name": feature["name_athena_in"],
                    "variables": [
                        {"name": k, "offset": 0.0, "scale": 1.0}
                        for k in config["data"]["variables"][feature["name_salt"]]
                    ],
                },
            ]

    # Encode metadata as JSON string in the ONNX model's metadata properties
    metadata = {"gnn_config": json.dumps(metadata)}
    for k, v in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = v

    # Store a short doc string at the model level and save file
    onnx_model.doc_string = model_name
    onnx.save(onnx_model, onnx_path)
    print("-" * 100)


if __name__ == "__main__":
    # Standard CLI entrypoint
    main()
