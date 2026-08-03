"""The producer manifest protocol: a module minting ``outputs.*`` leaves names them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.outputs import (
    ClassProbs,
    Combination,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    SeqClassIndex,
    SeqClassProbs,
)
from salt.outputs.sink import collect_manifest_fields
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, write_parity_norm_dict

pytestmark = pytest.mark.cpu_always

_MF_TARGETS = ("pt", "Lxy", "deta", "dphi", "mass")


@pytest.fixture
def gn2v2_modules(tmp_path):
    """The GN2v2 model modules (a global `jets_classification` + per-token `track_origin`)."""
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    return build_gn2v2_modules(tmp_path / "norm_dict.yaml")


def _bound(node, modules, name="producer"):
    """Name a producer and bind the model modules it resolves names from."""
    node.name = name
    node.bind_model_modules(modules)
    return node


# -- TaskOutput and its pre-wired subclasses ---------------------------------


def test_class_probs_names_come_from_the_task(gn2v2_modules):
    """`ClassProbs` mints one global float32 field per class, named by the source task."""
    node = _bound(ClassProbs(task="jets_classification", stream="jets"), gn2v2_modules)
    fields = node.manifest_fields(Mode.ONNX)
    assert [key for key, _ in fields] == [node.output_key] * 3
    assert [f.resolved_onnx_name for _, f in fields] == ["pb", "pc", "pu"]
    assert {f.onnx_dtype for _, f in fields} == {"float32"}
    assert {f.axis for _, f in fields} == {"global"}


def test_seq_class_index_names_come_from_the_task(gn2v2_modules):
    """`SeqClassIndex` mints ONE int8 per-token field, pascal-cased by the source task."""
    node = _bound(SeqClassIndex(task="track_origin", stream="tracks"), gn2v2_modules)
    ((key, field),) = node.manifest_fields(Mode.ONNX)
    assert key == "outputs.tracks.track_origin"
    assert field.resolved_onnx_name == "TrackOrigin"
    assert field.onnx_dtype == "int8"
    assert field.axis == "per_token"


def test_seq_class_probs_declares_no_onnx_field(gn2v2_modules):
    """Per-token probability columns are an eval-H5 representation with no ONNX twin."""
    node = _bound(SeqClassProbs(task="track_origin", stream="tracks"), gn2v2_modules)
    assert node.manifest_fields(Mode.ONNX) == []


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.TEST])
def test_task_output_is_onnx_only(gn2v2_modules, mode):
    """The eval-H5 side rides the section's RunTaskOutput, so these declare ONNX only."""
    node = _bound(ClassProbs(task="jets_classification", stream="jets"), gn2v2_modules)
    assert node.manifest_fields(mode) == []


def test_task_output_unbound_modules_is_a_named_error():
    node = ClassProbs(task="jets_classification", stream="jets")
    with pytest.raises(ConfigError, match="no model modules bound"):
        node.manifest_fields(Mode.ONNX)


def test_task_output_missing_task_names_the_candidates(gn2v2_modules):
    node = _bound(ClassProbs(task="typo_task", stream="jets"), gn2v2_modules)
    with pytest.raises(ConfigError, match="'typo_task' is not a model module"):
        node.manifest_fields(Mode.ONNX)


# -- Combination -------------------------------------------------------------


def test_combination_names_itself():
    """A combination has no source task — it names its own single global scalar."""
    node = Combination(
        source="outputs.jets.jets_classification", name="pbc", terms={0: 1.0, 1: 1.0}
    )
    ((key, field),) = node.manifest_fields(Mode.ONNX)
    assert key == "outputs.jets.pbc"
    assert field.resolved_onnx_name == "pbc"
    assert field.axis == "global"
    assert field.onnx_dtype == "float32"
    assert node.manifest_fields(Mode.TEST) == []


# -- MaskFormerObjects -------------------------------------------------------


def _mf_modules(targets=_MF_TARGETS, n_reg=None):
    """A `MaskFormerObjects` bound to a regression-task stub exposing `targets`."""
    node = MaskFormerObjects(
        n_reg=len(targets) if n_reg is None else n_reg, index_name="HadronIndex"
    )
    node.name = "maskformer_objects"
    node.bind_model_modules({"regression": SimpleNamespace(targets=targets)})
    return node


def test_maskformer_leading_names_derive_from_the_regression_targets():
    """One ``leading_objects_<target>`` global per target — read off the task, not typed."""
    node = _mf_modules()
    leading = [
        f.resolved_onnx_name
        for key, f in node.manifest_fields(Mode.ONNX)
        if key == node.leading_key
    ]
    assert leading == [f"leading_objects_{t}" for t in _MF_TARGETS]


def test_maskformer_index_is_int8_per_token():
    """The producer declares its own int8 per-token dtype for the constituent index."""
    node = _mf_modules()
    (field,) = [f for key, f in node.manifest_fields(Mode.ONNX) if key == node.index_key]
    assert node.index_key == "outputs.tracks.HadronIndex"
    assert field.resolved_onnx_name == "HadronIndex"
    assert field.onnx_dtype == "int8"
    assert field.axis == "per_token"


def test_maskformer_vertex_leaves_are_declared_but_not_final():
    """The two ``vertices_*`` leaves are visible in the manifest and dropped by the collector."""
    node = _mf_modules()
    declared = dict(node.manifest_fields(Mode.ONNX))
    assert declared[node.vertices_class_probs_key].final is False
    assert declared[node.vertices_regression_key].final is False
    collected = {key for key, _ in collect_manifest_fields([node], Mode.ONNX)}
    assert collected == {node.leading_key, node.index_key}


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.TEST])
def test_maskformer_manifest_is_onnx_only(mode):
    """The TEST index column rides the H5 sink's object_groups, not this manifest."""
    assert _mf_modules().manifest_fields(mode) == []


def test_maskformer_target_count_must_match_n_reg():
    """The leading leaf is sliced to n_reg, so a target/n_reg mismatch is refused."""
    node = _mf_modules(targets=("pt", "Lxy"), n_reg=5)
    with pytest.raises(ConfigError, match="declares 2 targets.*but n_reg is 5"):
        node.manifest_fields(Mode.ONNX)


def test_maskformer_requires_a_targets_bearing_task():
    node = MaskFormerObjects(n_reg=2)
    node.name = "maskformer_objects"
    node.bind_model_modules({"regression": SimpleNamespace()})
    with pytest.raises(ConfigError, match="exposes no `targets`"):
        node.manifest_fields(Mode.ONNX)


def test_maskformer_unbound_modules_is_a_named_error():
    node = MaskFormerObjects(n_reg=2)
    node.name = "maskformer_objects"
    with pytest.raises(ConfigError, match="no model modules bound"):
        node.manifest_fields(Mode.ONNX)


# -- MFLeadVertexDecorator ---------------------------------------------------


def test_lead_vertex_decorator_names_its_jet_level_scalars():
    """Each configured jet-level output is one global float ONNX field."""
    node = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
    )
    fields = node.manifest_fields(Mode.ONNX)
    assert [key for key, _ in fields] == list(node.output_keys)
    assert [f.resolved_onnx_name for _, f in fields] == ["lead_vertex_pt", "lead_vertex_mass"]
    assert node.manifest_fields(Mode.TEST) == []
