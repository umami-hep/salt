import json
from types import SimpleNamespace

import onnx
import pytest
import torch
import yaml
from onnx import TensorProto, helper
from torch import nn

from salt.models import ClassificationTask, GlobalAttentionPooling, SaltModel
from salt.modelwrapper import ModelWrapper
from salt.onnx import to_onnx as to_onnx_module
from salt.onnx.check import compare_outputs
from salt.onnx.to_onnx import (
    ONNXModel,
    add_metadata,
    get_default_onnx_feature_map,
    parse_args,
    parse_output_combination,
    update_config,
)


LRS_CONFIG = {
    "initial": 1e-7,
    "max": 5e-4,
    "end": 1e-5,
    "pct_start": 0.01,
    "weight_decay": 1e-5,
}


VARIABLES = {
    "event": ["met", "met_phi"],
    "jet": ["pt", "eta"],
    "el": ["pt"],
}
INPUT_MAP = {
    "event": "event",
    "jet": "jet",
    "el": "el",
}


@pytest.fixture
def norm_dict_path(tmp_path):
    norm_dict = {
        input_name: {var: {"mean": 0.0, "std": 1.0} for var in variables}
        for input_name, variables in VARIABLES.items()
    }
    path = tmp_path / "norm_dict.yaml"
    path.write_text(yaml.dump(norm_dict))
    return path


@pytest.fixture
def feature_map():
    return get_default_onnx_feature_map("r22default", list(VARIABLES), "event")


def build_task(
    name: str, input_name: str = "event", hidden_layers: list[int] | None = None
) -> ClassificationTask:
    return ClassificationTask(
        name=name,
        input_name=input_name,
        label="label",
        class_names=["sig", "bkg"],
        loss=nn.CrossEntropyLoss(),
        dense_config={
            "input_size": 4,
            "output_size": 2,
            "hidden_layers": hidden_layers or [4],
        },
    )


def build_onnx_model(
    norm_dict_path,
    feature_map,
    tasks: nn.ModuleList | None = None,
    attach_global: bool = False,
    **kwargs,
) -> ONNXModel:
    model = SaltModel(
        init_nets=[
            {
                "input_name": "jet",
                "attach_global": attach_global,
                "variables": VARIABLES,
                "global_object": "event",
                "dense_config": {
                    "output_size": 4,
                    "hidden_layers": [4],
                },
            },
            {
                "input_name": "el",
                "attach_global": attach_global,
                "variables": VARIABLES,
                "global_object": "event",
                "dense_config": {
                    "output_size": 4,
                    "hidden_layers": [4],
                },
            },
        ],
        pool_net=GlobalAttentionPooling(input_size=4),
        tasks=tasks or nn.ModuleList([build_task("events_classification")]),
    )

    return ONNXModel(
        model=model,
        lrs_config=LRS_CONFIG,
        global_object="event",
        norm_config={
            "norm_dict": norm_dict_path,
            "variables": VARIABLES,
            "global_object": "event",
            "input_map": INPUT_MAP,
        },
        onnx_feature_map=feature_map,
        variable_map=VARIABLES,
        **kwargs,
    )


def test_parse_args_batched_standalone(tmp_path):
    ckpt_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    args = parse_args([
        "--ckpt_path",
        str(ckpt_path),
        "--config",
        str(config_path),
        "--track_selection",
        "all",
        "--name",
        "eventmodel",
        "--overwrite",
        "--tasks",
        "events_classification",
        "--object_name",
        "hadrons",
        "--combine_outputs",
        "sum,psig,pbkg",
        "--rename",
        "psig:signal",
        "--force",
        "--batched-standalone",
    ])

    assert args.ckpt_path == ckpt_path
    assert args.config == config_path
    assert args.track_selection == "all"
    assert args.name == "eventmodel"
    assert args.overwrite
    assert args.tasks == ["events_classification"]
    assert args.object_name == "hadrons"
    assert args.combine_outputs == ["sum,psig,pbkg"]
    assert args.rename == ["psig:signal"]
    assert args.force
    assert args.batched_standalone


def test_parse_args_rejects_unknown_track_selection(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--ckpt_path", str(tmp_path / "model.ckpt"), "--track_selection", "bad"])


def test_get_default_onnx_feature_map_for_event_inputs(feature_map):
    assert feature_map == [
        {
            "name_athena_in": "event_var",
            "name_athena_out": "event_features",
            "name_salt": "event",
            "is_global": True,
        },
        {
            "name_athena_in": "jet_var",
            "name_athena_out": "jet_features",
            "athena_num_name": "n_jet",
            "name_salt": "jet",
            "is_global": False,
        },
        {
            "name_athena_in": "el_var",
            "name_athena_out": "el_features",
            "athena_num_name": "n_el",
            "name_salt": "el",
            "is_global": False,
        },
    ]


def test_get_default_onnx_feature_map_covers_special_input_names():
    feature_map = get_default_onnx_feature_map(
        "ip3d",
        ["jets", "global", "tracks_loose", "flows", "electrons", "flow", "_edge"],
        "jets",
    )

    assert feature_map == [
        {
            "name_athena_in": "jet_var",
            "name_athena_out": "jet_features",
            "name_salt": "jets",
            "is_global": True,
        },
        {
            "name_athena_in": "tracks_ip3d_sd0sort",
            "name_athena_out": "track_features",
            "athena_num_name": "n_tracks",
            "name_salt": "tracks_loose",
            "is_global": False,
        },
        {
            "name_athena_in": "flows_ip3d_sd0sort",
            "name_athena_out": "flow_features",
            "athena_num_name": "n_flows",
            "name_salt": "flows",
            "is_global": False,
        },
        {
            "name_athena_in": "electrons_r22default",
            "name_athena_out": "electron_features",
            "athena_num_name": "n_electrons",
            "name_salt": "electrons",
            "is_global": False,
        },
        {
            "name_athena_in": "flows_ip3d_sd0sort",
            "name_athena_out": "flow_features",
            "athena_num_name": "n_flow",
            "name_salt": "flow",
            "is_global": False,
        },
    ]


def test_default_export_schema_keeps_athena_sequence_inputs(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map)

    assert onnx_model.input_names == ["event_features", "jet_features", "el_features"]
    assert onnx_model.example_input_array[0].shape == (1, 2)
    assert onnx_model.example_input_array[1].shape == (40, 2)
    assert onnx_model.example_input_array[2].shape == (40, 1)
    assert onnx_model.dynamic_axes == {
        "jet_features": {0: "n_jet"},
        "el_features": {0: "n_el"},
    }


def test_model_name_rejects_athena_unsafe_characters(norm_dict_path, feature_map):
    with pytest.raises(AssertionError, match="underscores"):
        build_onnx_model(norm_dict_path, feature_map, name="bad_name")

    with pytest.raises(AssertionError, match="dashes"):
        build_onnx_model(norm_dict_path, feature_map, name="bad-name")


def test_output_names_reject_unknown_rename_or_combine(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map, rename_outputs={"missing": "new"})
    with pytest.raises(AssertionError, match="Output missing not found"):
        _ = onnx_model.output_names

    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        combine_outputs=[("sum", [(1.0, "missing")])],
    )
    with pytest.raises(AssertionError, match="Output missing not found"):
        _ = onnx_model.output_names


def test_auxiliary_task_names_are_ordered_last(norm_dict_path, feature_map):
    tasks = nn.ModuleList([
        build_task("events_classification"),
        build_task("track_type", input_name="jet"),
        build_task("track_origin", input_name="jet"),
        build_task("track_vertexing", input_name="jet"),
    ])

    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        tasks=tasks,
        tasks_to_output=[
            "track_type",
            "events_classification",
            "track_origin",
            "track_vertexing",
        ],
    )

    assert onnx_model.tasks_to_output == [
        "events_classification",
        "track_origin",
        "track_vertexing",
        "track_type",
    ]
    assert onnx_model.output_names[-3:] == [
        "salt_TrackOrigin",
        "salt_VertexIndex",
        "salt_TrackType",
    ]
    assert onnx_model.dynamic_axes["salt_TrackOrigin"] == {0: "n_tracks"}
    assert onnx_model.dynamic_axes["salt_VertexIndex"] == {0: "n_tracks"}
    assert onnx_model.dynamic_axes["salt_TrackType"] == {0: "n_tracks"}


def test_include_latent_numbers_outputs_sequentially(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map, include_latent=True)

    # L1 is the pooled latent, L2+ the hidden layers of the classification head
    assert onnx_model.latent_output_names == ["salt_LatentL1", "salt_LatentL2"]
    assert onnx_model.output_names == ["salt_psig", "salt_pbkg", "salt_LatentL1", "salt_LatentL2"]


def test_include_latent_keeps_auxiliary_outputs_last(norm_dict_path, feature_map):
    tasks = nn.ModuleList([
        build_task("events_classification"),
        build_task("track_origin", input_name="jet"),
    ])

    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        tasks=tasks,
        tasks_to_output=["events_classification", "track_origin"],
        include_latent=True,
    )

    # latents sit between the global outputs and the per-track tail, so name slicing
    # that reads track outputs from the end is unaffected
    assert onnx_model.output_names == [
        "salt_psig",
        "salt_pbkg",
        "salt_LatentL1",
        "salt_LatentL2",
        "salt_TrackOrigin",
    ]


def test_include_latent_numbers_repeated_hidden_widths_uniquely(norm_dict_path, feature_map):
    tasks = nn.ModuleList([build_task("events_classification", hidden_layers=[4, 4, 4])])

    onnx_model = build_onnx_model(
        norm_dict_path, feature_map, tasks=tasks, include_latent=True
    )

    # every hidden layer here has the same width, so names must not be keyed on it
    assert onnx_model.latent_output_names == [
        "salt_LatentL1",
        "salt_LatentL2",
        "salt_LatentL3",
        "salt_LatentL4",
    ]

    onnx_model(torch.rand(1, 2), torch.rand(3, 2), torch.rand(3, 1))
    captures = onnx_model._latent_captures
    assert len({id(captures[name]) for name in onnx_model.latent_output_names}) == 4


def test_include_latent_forward_shape_matches_interface(norm_dict_path, feature_map):
    athena = build_onnx_model(norm_dict_path, feature_map, include_latent=True)
    outputs = athena(torch.rand(1, 2), torch.rand(3, 2), torch.rand(3, 1))
    # Athena runs one global object per call, so latents are rank-1 [D]
    assert [tuple(out.shape) for out in outputs[-2:]] == [(4,), (4,)]

    batched = build_onnx_model(
        norm_dict_path, feature_map, include_latent=True, batched_standalone=True
    )
    outputs = batched(
        torch.rand(2, 2),
        torch.rand(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.rand(2, 3, 1),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    # the batched interface keeps the batch dimension its dynamic_axes declares
    assert [tuple(out.shape) for out in outputs[-2:]] == [(2, 4), (2, 4)]
    for name in batched.latent_output_names:
        assert batched.dynamic_axes[name] == {0: "batch"}


def test_latent_task_selects_the_named_head(norm_dict_path, feature_map):
    tasks = nn.ModuleList([
        build_task("events_classification"),
        build_task("events_charge", hidden_layers=[4, 4]),
    ])

    # global_object is "event", so "<global_object>_classification" does not match either
    # head and the two candidates are ambiguous
    with pytest.raises(ValueError, match="events_classification, events_charge"):
        build_onnx_model(norm_dict_path, feature_map, tasks=tasks, include_latent=True)

    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        tasks=tasks,
        include_latent=True,
        latent_task="events_charge",
    )
    assert onnx_model.latent_output_names == ["salt_LatentL1", "salt_LatentL2", "salt_LatentL3"]

    with pytest.raises(ValueError, match="not a global classification task"):
        build_onnx_model(
            norm_dict_path, feature_map, tasks=tasks, include_latent=True, latent_task="nope"
        )


def test_batched_standalone_schema_adds_batch_axes_and_masks(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map, batched_standalone=True)

    assert onnx_model.input_names == [
        "event_features",
        "jet_features",
        "jet_features_mask",
        "el_features",
        "el_features_mask",
    ]
    assert onnx_model.example_input_array[0].shape == (2, 2)
    assert onnx_model.example_input_array[1].shape == (2, 40, 2)
    assert onnx_model.example_input_array[2].shape == (2, 40)
    assert onnx_model.example_input_array[3].shape == (2, 40, 1)
    assert onnx_model.example_input_array[4].shape == (2, 40)
    assert onnx_model.dynamic_axes == {
        "event_features": {0: "batch"},
        "jet_features": {0: "batch", 1: "n_jet"},
        "jet_features_mask": {0: "batch", 1: "n_jet"},
        "el_features": {0: "batch", 1: "n_el"},
        "el_features_mask": {0: "batch", 1: "n_el"},
        "salt_psig": {0: "batch"},
        "salt_pbkg": {0: "batch"},
    }


def test_batched_standalone_forward_preserves_batch_dimension(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map, batched_standalone=True)
    outputs = onnx_model(
        torch.rand(3, 2),
        torch.rand(3, 5, 2),
        torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, True, True, True],
                [False, True, True, True, True],
            ]
        ),
        torch.rand(3, 2, 1),
        torch.tensor([[False, False], [False, True], [True, True]]),
    )

    assert len(outputs) == 2
    assert outputs[0].shape == (3,)
    assert outputs[1].shape == (3,)


def test_tasks_to_output_filters_global_tasks(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        tasks=nn.ModuleList([build_task("task_a"), build_task("task_b")]),
        tasks_to_output=["task_b"],
        batched_standalone=True,
    )

    outputs = onnx_model(
        torch.rand(2, 2),
        torch.rand(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.rand(2, 3, 1),
        torch.zeros(2, 3, dtype=torch.bool),
    )

    assert onnx_model.output_names == ["salt_psig", "salt_pbkg"]
    assert len(outputs) == 2


def test_output_names_support_rename_and_combine(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        rename_outputs={"psig": "signal"},
        combine_outputs=[("sum", [(1.0, "signal"), (1.0, "pbkg")])],
    )

    assert onnx_model.output_names == ["salt_signal", "salt_pbkg", "salt_sum"]


def test_forward_supports_combined_outputs(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(
        norm_dict_path,
        feature_map,
        combine_outputs=[("sum", [(1.0, "psig"), (1.0, "pbkg")])],
        batched_standalone=True,
    )

    outputs = onnx_model(
        torch.rand(2, 2),
        torch.rand(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.rand(2, 3, 1),
        torch.zeros(2, 3, dtype=torch.bool),
    )

    assert len(outputs) == 3
    assert torch.allclose(outputs[2], outputs[0] + outputs[1])


def test_forward_checks_input_count_and_shapes(norm_dict_path, feature_map):
    onnx_model = build_onnx_model(norm_dict_path, feature_map, batched_standalone=True)

    with pytest.raises(AssertionError, match="Number of inputs"):
        onnx_model(torch.rand(2, 2))

    with pytest.raises(AssertionError, match="event should have"):
        onnx_model(
            torch.rand(2, 1, 2),
            torch.rand(2, 3, 2),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.rand(2, 3, 1),
            torch.zeros(2, 3, dtype=torch.bool),
        )

    with pytest.raises(AssertionError, match="jet should have shape"):
        onnx_model(
            torch.rand(2, 2),
            torch.rand(3, 2),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.rand(2, 3, 1),
            torch.zeros(2, 3, dtype=torch.bool),
        )


def test_batched_standalone_rejects_auxiliary_track_outputs(norm_dict_path, feature_map):
    tasks = nn.ModuleList([
        build_task("events_classification"),
        build_task("track_origin", input_name="jet"),
    ])

    with pytest.raises(ValueError, match="global task outputs only"):
        build_onnx_model(
            norm_dict_path,
            feature_map,
            tasks=tasks,
            tasks_to_output=["events_classification", "track_origin"],
            batched_standalone=True,
        )


def test_batched_standalone_rejects_object_outputs(norm_dict_path, feature_map):
    with pytest.raises(ValueError, match="object outputs"):
        build_onnx_model(
            norm_dict_path,
            feature_map,
            object_name="hadrons",
            mf_config={
                "object": {
                    "class_label": "label",
                    "class_names": ["b"],
                }
            },
            batched_standalone=True,
        )


def test_batched_standalone_export_matches_pytorch_over_padded_sweep(
    norm_dict_path, feature_map, tmp_path
):
    # attach_global=True so the global features are consumed, else the exporter prunes the input
    onnx_model = build_onnx_model(
        norm_dict_path, feature_map, attach_global=True, batched_standalone=True
    )
    onnx_model.eval()

    # PyTorch reference shares the SaltModel weights, so drift is purely an export artefact
    pt_model = ModelWrapper(
        model=onnx_model.model,
        lrs_config=LRS_CONFIG,
        global_object="event",
        norm_config={
            "norm_dict": norm_dict_path,
            "variables": VARIABLES,
            "global_object": "event",
            "input_map": INPUT_MAP,
        },
    )
    pt_model.eval()

    onnx_path = tmp_path / "network.onnx"
    onnx_model.to_onnx(
        onnx_path,
        opset_version=20,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes=onnx_model.dynamic_axes,
        dynamo=False,
    )

    # batched=True sweeps padded sequence lengths through a real ORT session, asserting parity
    compare_outputs(
        pt_model,
        onnx_path,
        global_object="event",
        seq_names_salt=["jet", "el"],
        seq_names_onnx=["jet_features", "el_features"],
        variable_map=VARIABLES,
        tasks_to_output=onnx_model.tasks_to_output,
        batched=True,
    )


def test_object_name_and_maskformer_config_must_be_paired(norm_dict_path, feature_map):
    with pytest.raises(ValueError, match="so must the other"):
        build_onnx_model(norm_dict_path, feature_map, object_name="hadrons")

    with pytest.raises(ValueError, match="so must the other"):
        build_onnx_model(
            norm_dict_path,
            feature_map,
            mf_config={
                "object": {
                    "class_label": "label",
                    "class_names": ["b"],
                }
            },
        )


def test_parse_output_combination():
    assert parse_output_combination("light,0.5*q,g,2*s") == (
        "light",
        [(0.5, "q"), (1, "g"), (2.0, "s")],
    )


def test_update_config_converts_weight_and_removes_deprecated_loss_args():
    config = {
        "model": {
            "init_args": {
                "tasks": {
                    "init_args": {
                        "modules": [
                            {
                                "init_args": {
                                    "loss": {
                                        "init_args": {
                                            "weight": [1.0, 2.0],
                                            "size_average": True,
                                            "reduce": False,
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    }

    update_config(config)
    loss_args = config["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"][
        "loss"
    ]["init_args"]

    assert torch.equal(loss_args["weight"], torch.tensor([1.0, 2.0]))
    assert "size_average" not in loss_args
    assert "reduce" not in loss_args


def test_add_metadata_writes_expected_gnn_config(tmp_path, monkeypatch, feature_map):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n")
    (tmp_path / "metadata.yaml").write_text("tag: value\n")
    ckpt_path = tmp_path / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir()
    ckpt_path.write_text("")
    onnx_path = tmp_path / "network.onnx"

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["salt_psig"],
                value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0]),
            )
        ],
        name="empty",
        inputs=[helper.make_tensor_value_info("event_features", TensorProto.FLOAT, [1, 2])],
        outputs=[helper.make_tensor_value_info("salt_psig", TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph), onnx_path)
    monkeypatch.setattr(to_onnx_module, "get_git_hash", lambda _: "abc123")

    add_metadata(
        config_path=config_path,
        config={
            "data": {
                "variables": VARIABLES,
            }
        },
        ckpt_path=ckpt_path,
        onnx_path=onnx_path,
        model_name="salt",
        output_names=["salt_psig"],
        onnx_feature_map=feature_map,
        combine_outputs=[("sum", [(1.0, "psig")])],
        rename_outputs={"psig": "signal"},
    )

    loaded = onnx.load(onnx_path)
    metadata = {prop.key: prop.value for prop in loaded.metadata_props}
    gnn_config = json.loads(metadata["gnn_config"])

    assert loaded.doc_string == "salt"
    assert gnn_config["ckpt_path"] == str(ckpt_path.resolve())
    assert gnn_config["metadata.yaml"] == {"tag": "value"}
    assert gnn_config["salt_export_hash"] == "abc123"
    assert gnn_config["output_names"] == ["salt_psig"]
    assert gnn_config["model_name"] == "salt"
    assert gnn_config["inputs"] == [
        {
            "name": "event_var",
            "variables": [
                {"name": "met", "offset": 0.0, "scale": 1.0},
                {"name": "met_phi", "offset": 0.0, "scale": 1.0},
            ],
        }
    ]
    assert gnn_config["input_sequences"] == [
        {
            "name": "jet_var",
            "variables": [
                {"name": "pt", "offset": 0.0, "scale": 1.0},
                {"name": "eta", "offset": 0.0, "scale": 1.0},
            ],
        },
        {
            "name": "el_var",
            "variables": [{"name": "pt", "offset": 0.0, "scale": 1.0}],
        },
    ]
    assert gnn_config["combine_outputs"] == [["sum", [[1.0, "psig"]]]]
    assert gnn_config["rename_outputs"] == {"psig": "signal"}


def write_main_config(path, *, mf_config=None):
    config = {
        "name": "test",
        "model": {
            "model": {
                "class_path": "salt.models.SaltModel",
                "init_args": {
                    "tasks": {
                        "init_args": {
                            "modules": [
                                {
                                    "init_args": {
                                        "loss": {
                                            "init_args": {
                                                "weight": None,
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
            },
            "lrs_config": {
                "initial": 1e-7,
                "max": 5e-4,
                "end": 1e-5,
                "pct_start": 0.01,
                "weight_decay": 1e-5,
            },
        },
        "data": {
            "global_object": "event",
            "input_map": INPUT_MAP,
            "variables": VARIABLES,
        },
    }
    if mf_config is not None:
        config["data"]["mf_config"] = mf_config
    path.write_text(yaml.dump(config, sort_keys=False))


class DummyLoadedModel:
    def __init__(self):
        self.model = nn.Module()

    def eval(self):
        return self

    def float(self):
        return self


class DummyONNXModel(DummyLoadedModel):
    name = "salt"
    input_names = ["event_features", "jet_features", "el_features"]
    output_names = ["salt_psig"]
    dynamic_axes = {"jet_features": {0: "n_jet"}, "el_features": {0: "n_el"}}
    tasks_to_output = ["events_classification"]

    def __init__(self):
        super().__init__()
        self.to_onnx_calls = []

    def to_onnx(self, *args, **kwargs):
        self.to_onnx_calls.append((args, kwargs))


def patch_main_heavy_dependencies(monkeypatch, dummy_onnx_model):
    calls = SimpleNamespace(metadata=None, compare=None)

    monkeypatch.setattr(to_onnx_module, "check_for_uncommitted_changes", lambda _: None)
    monkeypatch.setattr(to_onnx_module, "change_attn_backends", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        to_onnx_module.ModelWrapper,
        "load_from_checkpoint",
        lambda *args, **kwargs: DummyLoadedModel(),
    )
    monkeypatch.setattr(
        to_onnx_module.ONNXModel,
        "load_from_checkpoint",
        lambda *args, **kwargs: dummy_onnx_model,
    )

    def fake_add_metadata(*args, **kwargs):
        calls.metadata = {"args": args, "kwargs": kwargs}

    def fake_compare_outputs(*args, **kwargs):
        calls.compare = {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(to_onnx_module, "add_metadata", fake_add_metadata)
    monkeypatch.setattr(to_onnx_module, "compare_outputs", fake_compare_outputs)
    return calls


def test_main_exports_and_validates_default_path(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_main_config(config_path)
    ckpt_path = tmp_path / "training" / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("")
    dummy_onnx_model = DummyONNXModel()
    calls = patch_main_heavy_dependencies(monkeypatch, dummy_onnx_model)

    to_onnx_module.main([
        "--ckpt_path",
        str(ckpt_path),
        "--config",
        str(config_path),
        "--combine_outputs",
        "sum,psig",
        "--rename",
        "psig:signal",
        "--force",
    ])

    onnx_path = ckpt_path.parent.parent / "network.onnx"
    assert len(dummy_onnx_model.to_onnx_calls) == 1
    args, kwargs = dummy_onnx_model.to_onnx_calls[0]
    assert args == (onnx_path,)
    assert kwargs["input_names"] == dummy_onnx_model.input_names
    assert kwargs["output_names"] == dummy_onnx_model.output_names
    assert calls.metadata["kwargs"]["combine_outputs"] == [("sum", [(1, "psig")])]
    assert calls.metadata["kwargs"]["rename_outputs"] == {"psig": "signal"}
    assert calls.compare["kwargs"]["seq_names_salt"] == ["jet", "el"]
    assert calls.compare["kwargs"]["seq_names_onnx"] == ["jet_features", "el_features"]
    assert calls.compare["kwargs"]["tasks_to_output"] == ["events_classification"]
    assert calls.compare["kwargs"]["batched"] is False


def test_main_uses_batched_validation_when_requested(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_main_config(config_path)
    ckpt_path = tmp_path / "training" / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("")
    dummy_onnx_model = DummyONNXModel()
    calls = patch_main_heavy_dependencies(monkeypatch, dummy_onnx_model)

    with pytest.warns(UserWarning, match="will not work in Athena"):
        to_onnx_module.main([
            "--ckpt_path",
            str(ckpt_path),
            "--config",
            str(config_path),
            "--batched-standalone",
            "--force",
        ])

    assert calls.compare["kwargs"]["batched"] is True
    assert calls.compare["kwargs"]["seq_names_salt"] == ["jet", "el"]
    assert calls.compare["kwargs"]["seq_names_onnx"] == ["jet_features", "el_features"]


def test_main_infers_config_path_and_checks_worktree(tmp_path, monkeypatch):
    train_dir = tmp_path / "training"
    ckpt_path = train_dir / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("")
    config_path = train_dir / "config.yaml"
    write_main_config(config_path)
    dummy_onnx_model = DummyONNXModel()
    patch_main_heavy_dependencies(monkeypatch, dummy_onnx_model)
    checked_paths = []
    monkeypatch.setattr(
        to_onnx_module,
        "check_for_uncommitted_changes",
        lambda path: checked_paths.append(path),
    )

    to_onnx_module.main(["--ckpt_path", str(ckpt_path)])

    assert checked_paths == [to_onnx_module.Path(to_onnx_module.__file__).parent]


def test_main_refuses_to_overwrite_existing_onnx(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_main_config(config_path)
    ckpt_path = tmp_path / "training" / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("")
    (ckpt_path.parent.parent / "network.onnx").write_text("")
    patch_main_heavy_dependencies(monkeypatch, DummyONNXModel())

    with pytest.raises(FileExistsError, match="network.onnx"):
        to_onnx_module.main([
            "--ckpt_path",
            str(ckpt_path),
            "--config",
            str(config_path),
            "--force",
        ])


def test_main_raises_when_object_export_has_no_maskformer_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_main_config(config_path)
    ckpt_path = tmp_path / "training" / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("")
    patch_main_heavy_dependencies(monkeypatch, DummyONNXModel())

    with pytest.raises(ValueError, match="No mf_config"):
        to_onnx_module.main([
            "--ckpt_path",
            str(ckpt_path),
            "--config",
            str(config_path),
            "--object_name",
            "hadrons",
            "--force",
        ])
