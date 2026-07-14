"""Plan 50 Phase C gates — per-task target-label emission in TEST mode.

Label emission is a FIRST-CLASS GRAPH CONTRACT: ``get_output(mode=TEST)``
emits each task's target-label column(s), ``get_output_manifest(mode=TEST)``
declares them statically, ``output_time_requires(mode=TEST)`` demands exactly
the label keys they read — and in ONNX/export mode NONE of the three ever
names a label.
"""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import Mode
from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import flatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.core.outputs.run_task_output import RunTaskOutput

_RUN = "GN2"


# -- builders ----------------------------------------------------------------


def _cls(stream="jets", *, sequence=False, label_map=None, write_targets=True):
    module = ClassificationTaskModule(
        stream=stream,
        label="flavour_label" if not sequence else "ftagTruthOriginLabel",
        class_names=["bjets", "cjets", "ujets"],
        input=None if sequence else "pooled.global",
        sequence=sequence,
        label_map=label_map,
        write_targets=write_targets,
    )
    module.name = "jets_classification" if not sequence else "track_origin"
    module.bind(ResolvedSchema(widths={module.input_key: 8}))
    return module


def _reg(targets=("mHH",), *, sequence=False, denoms=None, gaussian=False, write_targets=True):
    module = RegressionTaskModule(
        stream="tracks" if sequence else "jets",
        targets=list(targets),
        input=None if sequence else "pooled.global",
        sequence=sequence,
        target_denominators=list(denoms) if denoms else None,
        norm_params={"mean": [1.0] * len(targets), "std": [2.0] * len(targets)}
        if denoms is None
        else None,
        gaussian=gaussian,
        write_targets=write_targets,
    )
    module.name = "regression"
    module.bind(
        ResolvedSchema(
            widths={module.input_key: 8},
            fields={module.input_feature_key: tuple(denoms or ())},
        )
    )
    return module


def _vtx(*, write_targets=True):
    module = VertexingTaskModule(
        stream="tracks",
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        dense={"hidden_layers": [4], "activation": "ReLU"},
        write_targets=write_targets,
    )
    module.name = "track_vertexing"
    module.bind(ResolvedSchema(widths={module.input_key: 8}))
    return module


# -- classification -----------------------------------------------------------


def test_cls_global_test_manifest_declares_target_column():
    """Global head TEST manifest: preds then one unprefixed i4 ``target_{task}`` column."""
    module = _cls()
    manifest = module.get_output_manifest(Mode.TEST, _RUN)
    assert [f.h5_name for f in manifest] == ["pb", "pc", "pu", "target_jets_classification"]
    target = manifest[-1]
    assert target.dtype == "i4"
    assert target.axis == "global"
    assert target.prefix is False  # labels are model-independent: never run-name prefixed
    assert target.onnx_name is None
    assert target.value is None


def test_cls_global_get_output_emits_remapped_label():
    """Global head: the target value is the label as consumed (post label_map remap)."""
    module = _cls(label_map={5: 0, 4: 1, 0: 2})
    b = Bundle({
        "preds": {"jets": {module.name: torch.randn(4, 3)}},
        "labels": {"jets": {"flavour_label": torch.tensor([5, 4, 0, 5])}},
    })
    *_, target = module.get_output(b, Mode.TEST, _RUN)
    assert target.h5_name == "target_jets_classification"
    assert target.value.tolist() == [0, 1, 2, 0]


def test_cls_seq_target_pads_and_invalid_read_minus_one():
    """Seq head: padded and label==-2 positions read -1 (the loss's consumed labels)."""
    module = _cls(stream="tracks", sequence=True)
    mask = torch.tensor([[False, False, True]])
    labels = torch.tensor([[1, -2, 7]])
    b = Bundle({
        "preds": {"tracks": {module.name: torch.randn(1, 3, 3)}},
        "masks": {"tracks": mask},
        "labels": {"tracks": {"ftagTruthOriginLabel": labels}},
    })
    *_, target = module.get_output(b, Mode.TEST, _RUN)
    assert target.h5_name == "target_track_origin"
    assert target.axis == "per_token"
    assert target.value.tolist() == [[1, -1, -1]]


def test_cls_output_time_requires_demands_exactly_the_label_key():
    """TEST demands the task's label key (+ pad mask for a seq head); ONNX never does."""
    module = _cls()
    assert module.output_time_requires(Mode.TEST) == ["labels.jets.flavour_label"]
    assert module.output_time_requires(Mode.ONNX) == []
    seq = _cls(stream="tracks", sequence=True)
    assert seq.output_time_requires(Mode.TEST) == [
        "masks.tracks",
        "labels.tracks.ftagTruthOriginLabel",
    ]
    assert seq.output_time_requires(Mode.ONNX) == ["masks.tracks"]


# -- regression ---------------------------------------------------------------


def test_reg_test_manifest_declares_one_physical_target_per_target():
    """Regression TEST manifest: preds then one unprefixed f4 ``target_{task}_{target}``."""
    module = _reg(targets=("mHH", "dR"))
    manifest = module.get_output_manifest(Mode.TEST, _RUN)
    assert [f.h5_name for f in manifest] == [
        "mHH",
        "dR",
        "target_regression_mHH",
        "target_regression_dR",
    ]
    for f in manifest[-2:]:
        assert f.dtype == "f4" and f.prefix is False and f.onnx_name is None


def test_reg_get_output_emits_unscaled_physical_targets():
    """The emitted target is the RAW label value (physical space), NOT the scaled target."""
    module = _reg(targets=("mHH",))  # norm_params mean=1 std=2: scaled space differs
    raw = torch.tensor([3.0, 5.0, 7.0])
    b = Bundle({
        "preds": {"jets": {module.name: torch.randn(3, 1)}},
        "labels": {"jets": {"mHH": raw.clone()}},
    })
    *_, target = module.get_output(b, Mode.TEST, _RUN)
    torch.testing.assert_close(target.value, raw, rtol=0, atol=0)


def test_reg_seq_target_pads_are_nan():
    """Per-token regression targets NaN-fill padded positions (matching the pred columns)."""
    module = _reg(targets=("dEta",), sequence=True)
    mask = torch.tensor([[False, True]])
    b = Bundle({
        "preds": {"tracks": {module.name: torch.randn(1, 2, 1)}},
        "masks": {"tracks": mask},
        "labels": {"tracks": {"dEta": torch.tensor([[0.5, 9.9]])}},
    })
    *_, target = module.get_output(b, Mode.TEST, _RUN)
    assert target.value[0, 0] == 0.5
    assert torch.isnan(target.value[0, 1])


def test_reg_gaussian_targets_are_r_not_2r():
    """A gaussian head emits ONE target column per target (R), not 2R (_stddev has no truth)."""
    module = _reg(targets=("pt",), gaussian=True)
    manifest = module.get_output_manifest(Mode.TEST, _RUN)
    assert [f.h5_name for f in manifest] == ["pt", "pt_stddev", "target_regression_pt"]


def test_reg_ratio_head_demands_targets_and_denoms_in_test_only():
    """Ratio head TEST demand = denominator labels + target labels; ONNX = the input Feature."""
    module = _reg(targets=("m_over_mHH",), denoms=("mHH",))
    assert module.output_time_requires(Mode.TEST) == [
        "labels.jets.mHH",
        "labels.jets.m_over_mHH",
    ]
    assert module.output_time_requires(Mode.ONNX) == ["inputs.jets"]


# -- vertexing ----------------------------------------------------------------


def test_vtx_test_manifest_declares_target_column():
    """Vertexing TEST manifest: VertexIndex then an unprefixed i4 ``target_{task}`` column."""
    module = _vtx()
    manifest = module.get_output_manifest(Mode.TEST, _RUN)
    assert [f.h5_name for f in manifest] == ["VertexIndex", "target_track_vertexing"]
    target = manifest[-1]
    assert target.dtype == "i4" and target.axis == "per_token" and target.prefix is False


def test_vtx_get_output_emits_label_with_pads_minus_one():
    """The vertexing target is the raw per-token vertex-index label; pads read -1."""
    module = _vtx()
    n_valid = 2  # the edge head scores ordered pairs of VALID tracks only
    b = Bundle({
        "preds": {"tracks": {module.name: torch.rand(n_valid * (n_valid - 1), 1)}},
        "masks": {"tracks": torch.tensor([[False, False, True]])},
        "labels": {"tracks": {"ftagTruthVertexIndex": torch.tensor([[0, 0, 5]])}},
    })
    *_, target = module.get_output(b, Mode.TEST, _RUN)
    assert target.value.tolist() == [[0, 0, -1]]


def test_vtx_output_time_requires_label_in_test_only():
    """Vertexing TEST demand = pad mask + vertex label; ONNX = pad mask only."""
    module = _vtx()
    assert module.output_time_requires(Mode.TEST) == [
        "masks.tracks",
        "labels.tracks.ftagTruthVertexIndex",
    ]
    assert module.output_time_requires(Mode.ONNX) == ["masks.tracks"]


# -- export mode is label-free for EVERY task family ---------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: _cls(),
        lambda: _cls(stream="tracks", sequence=True),
        lambda: _reg(targets=("mHH", "dR")),
        lambda: _reg(targets=("dEta",), sequence=True),
        lambda: _reg(targets=("pt",), gaussian=True),
        lambda: _reg(targets=("m_over_mHH",), denoms=("mHH",)),
        lambda: _vtx(),
    ],
    ids=[
        "cls-global",
        "cls-seq",
        "reg-global",
        "reg-seq",
        "reg-gaussian",
        "reg-ratio",
        "vtx",
    ],
)
def test_export_mode_manifest_and_requires_are_label_free(build):
    """ONNX/export mode: no manifest field is a target column, no ``labels.*`` demanded."""
    module = build()
    manifest = module.get_output_manifest(Mode.ONNX, _RUN)
    assert manifest, "every family exports at least one ONNX field"
    for f in manifest:
        for name in (f.h5_name, f.onnx_name):
            assert name is None or not name.startswith("target_"), (
                f"ONNX manifest leaked a label column {name!r}"
            )
    assert all(not d.startswith("labels.") for d in module.output_time_requires(Mode.ONNX)), (
        "ONNX output_time_requires demanded a label — export/inference must be label-free"
    )


# -- disable flag --------------------------------------------------------------


def test_write_targets_false_removes_columns_and_demand():
    """``write_targets: false``: no target column in the TEST manifest, no label demand."""
    for module in (
        _cls(write_targets=False),
        _reg(targets=("mHH",), write_targets=False),
        _vtx(write_targets=False),
    ):
        manifest = module.get_output_manifest(Mode.TEST, _RUN)
        assert all(
            f.h5_name is None or not f.h5_name.startswith("target_") for f in manifest
        ), f"{type(module).__name__} emitted a target column with write_targets=False"
        assert all(
            not d.startswith("labels.") for d in module.output_time_requires(Mode.TEST)
        ), f"{type(module).__name__} demanded a label with write_targets=False"


# -- RunTaskOutput integration: the section carries the contract ---------------


class _FixtureTasks:
    """A tiny two-task model dict: one global cls head + one seq cls head."""

    def build(self):
        jets = _cls()
        origin = _cls(stream="tracks", sequence=True)
        return {jets.name: jets, origin.name: origin}


def test_run_task_output_declares_target_leaves_and_label_demand_in_test():
    """RunTaskOutput.declare_io(TEST) produces the target leaves and requires the labels."""
    modules = _FixtureTasks().build()
    rt = RunTaskOutput(tasks=list(modules))
    rt.name = "run_tasks"
    rt.bind_model_modules(modules)
    io = rt.declare_io(Mode.TEST)
    req = flatten_spec(io.requires)
    prod = set(flatten_spec(io.produces))
    assert "labels.jets.flavour_label" in req
    assert "labels.tracks.ftagTruthOriginLabel" in req
    assert req["labels.jets.flavour_label"].kind == "label"
    # label dep dtype is UNCONSTRAINED (int64 class labels vs float32 regression targets)
    assert req["labels.jets.flavour_label"].dtype is None
    assert "outputs.jets.jets_classification.target_jets_classification" in prod
    assert "outputs.tracks.track_origin.target_track_origin" in prod


def test_run_task_output_onnx_declares_no_label_leaf_or_demand():
    """RunTaskOutput.declare_io(ONNX) names no target leaf and demands no label."""
    modules = _FixtureTasks().build()
    rt = RunTaskOutput(tasks=list(modules))
    rt.name = "run_tasks"
    rt.bind_model_modules(modules)
    io = rt.declare_io(Mode.ONNX)
    req = flatten_spec(io.requires)
    prod = flatten_spec(io.produces)
    assert all(not k.startswith("labels.") for k in req)
    assert all("target_" not in k for k in prod)
    assert all(
        "target_" not in key for key, _ in rt.manifest_fields(Mode.ONNX)
    ), "the ONNX section manifest must be label-free"


def test_run_task_output_forward_writes_target_leaf_values():
    """RunTaskOutput.forward(TEST) writes the consumed-label tensor into the target leaf."""
    modules = {"jets_classification": _cls(label_map={5: 0, 4: 1, 0: 2})}
    rt = RunTaskOutput(tasks=["jets_classification"])
    rt.name = "run_tasks"
    rt.bind_model_modules(modules)
    b = Bundle({
        "preds": {"jets": {"jets_classification": torch.randn(2, 3)}},
        "labels": {"jets": {"flavour_label": torch.tensor([5, 0])}},
    })
    produced = rt.forward(b, Mode.TEST)
    leaf = "outputs.jets.jets_classification.target_jets_classification"
    assert produced[leaf].tolist() == [0, 2]
