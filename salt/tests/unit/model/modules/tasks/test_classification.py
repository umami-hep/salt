"""Unit tests for ClassificationTaskModule."""

from __future__ import annotations

import pytest
import torch

from salt.graph import (
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.model.modules import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.model.modules.tasks import (
    ClassificationTaskModule,
)
from salt.tests._fixtures.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
)

# task modules


class TestClassificationTaskModule:
    def test_class_names_required(self):
        with pytest.raises(ConfigError, match="class_names"):
            ClassificationTaskModule(stream="jets", label="flavour_label", class_names=[])

    def test_sequence_inference(self):
        seq_task = ClassificationTaskModule(stream="tracks", label="l", class_names=["a", "b"])
        glob_task = ClassificationTaskModule(
            stream="jets", label="l", class_names=["a", "b"], input="pooled.global"
        )
        assert seq_task.sequence and not glob_task.sequence

    def test_declare_io_mode_gating(self):
        task = ClassificationTaskModule(
            stream="tracks", label="lbl", class_names=["a", "b"], context="pooled.global"
        )
        task.name = "track_origin"
        io = task.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        assert set(req) == {
            "encoded.tracks",
            "pooled.global",
            "masks.tracks",
            "labels.tracks.lbl",
        }
        assert req["labels.tracks.lbl"].kind == "label"
        assert req["labels.tracks.lbl"].modes == Mode.TRAINING
        assert req["masks.tracks"].kind == "pad_mask"
        produces = flatten_spec(io.produces)
        assert produces["preds.tracks.track_origin"].modes == Mode.ALL
        assert produces["losses.track_origin"].modes == Mode.TRAINING
        assert produces["losses.track_origin"].kind == "loss"

    def test_bind_builds_head_with_inferred_widths(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"]
        assert head.net.input_size == 16
        assert head.net.context_size == 16
        assert head.net.output_size == len(ORIGIN_CLASSES)
        assert head.loss.ignore_index == -1  # v1 contract (task.py:123-124)

    def test_weight_source_literal_conflict(self):
        with pytest.raises(ConfigError, match="already specified"):
            ClassificationTaskModule(
                stream="tracks",
                label="l",
                class_names=["a", "b"],
                loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"weight": [1, 2]}},
                weight_source={"from_class_dict": "cd.yaml"},
            )

    def test_weight_source_materialise(self, norm_paths):
        """Class weights fill the CE buffer at materialise."""
        modules = build_gn2v2_modules(norm_paths[0], class_dict=norm_paths[1])
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        head = modules["track_origin"]
        assert torch.equal(head.loss.weight, torch.ones(8))  # bind: allocated, identity
        materialise_all(modules)
        expected = torch.tensor([4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3])
        assert torch.allclose(head.loss.weight, expected)
        # the buffer is in the state dict (resume inherits it, cli.py:253-267)
        assert "loss.weight" in modules["track_origin"].state_dict()

    def test_weight_source_length_mismatch_raises(self, norm_paths):
        """The fixture class dict has 4 flavour weights but 3 class names."""
        task = ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            weight_source={"from_class_dict": str(norm_paths[1])},
        )
        task.name = "jets_classification"
        task.bind(ResolvedSchema(widths={"pooled.global": 16}))
        with pytest.raises(ValueError, match="3 class_names"):
            task.materialise()
