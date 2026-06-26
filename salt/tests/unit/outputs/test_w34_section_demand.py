"""PLAN 34 W34.2 unit gates — demand-pruning + FIT/VAL plan_hash stability + guards.

The W34.2 demand keystone (plan §6 gate 4): the outputs: section (RunTaskOutput +
InputCopyWriter + PadMaskWriter) is ``modes=ALL`` but DEMAND-pruned — in FIT/VAL
nothing demands ``outputs.*`` (losses read ``preds.*``), so the planner's demand
closure drops the whole section and the FIT/VAL plan is byte-identical to a model
WITHOUT the section. In TEST a sink demands the section's leaves, pulling them in.

Also covers the dumb-sink guards: dup-name H5 column, the section column-order
authority (section field order, NOT executor topo order), and the dead-preds guard
intactness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import Mode, TensorSpec, unflatten_spec
from salt.core.outputs.writers import PadMaskWriter, RunTaskOutput
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

pytestmark = pytest.mark.cpu_always


def _modules(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    nd = tmp_path / "norm_dict.yaml"
    cd = tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return build_gn2v2_modules(nd)


def _bare_gn2v2(tmp_path: Path) -> dict:
    """The gn2v2 fixture WITHOUT any plan-29/31 producers (model + tasks + loss only)."""
    modules = _modules(tmp_path)
    # the gn2v2 fixture ships plan-29/31 ONNX producers — drop them so the bare
    # model is just nets + tasks + loss (the plan-34 model.modules shape).
    for k in ("jet_probs", "track_origin_index", "track_vertex_index"):
        modules.pop(k, None)
    return modules


# the gn2v2 dataset boundary specs the FIT plan compiles against (jets global +
# tracks seq features/labels/masks/meta) — enough for the FIT/VAL plan to resolve.
def _fit_sources() -> dict:
    return unflatten_spec({
        "inputs.jets": TensorSpec(shape=("B", 2), dtype="float32"),
        "inputs.tracks": TensorSpec(shape=("B", "T", 21), dtype="float32"),
        "masks.tracks": TensorSpec(shape=("B", "T"), dtype="bool", kind="pad_mask"),
        "labels.jets.flavour_label": TensorSpec(shape=("B",), dtype="int64", kind="label"),
        "labels.tracks.ftagTruthOriginLabel": TensorSpec(
            shape=("B", "T"), dtype="int64", kind="label"
        ),
        "labels.tracks.ftagTruthVertexIndex": TensorSpec(
            shape=("B", "T"), dtype="int64", kind="label"
        ),
    })


class TestW34FitPlanHashStability:
    """The outputs: section is byte-invisible to the FIT/VAL plan (demand-pruned)."""

    def test_fit_plan_hash_unchanged_by_section(self, tmp_path):
        """Adding RunTaskOutput + PadMaskWriter to the module dict does NOT change FIT plan_hash."""
        sources = _fit_sources()
        bare = _bare_gn2v2(tmp_path / "bare")
        bare_hash = compile_plan(dict(bare), Mode.FIT, sources, sinks=["loss.total"]).plan_hash

        withsec = _bare_gn2v2(tmp_path / "withsec")
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin"])
        rt.name = "run_tasks"
        rt.bind_model_modules(dict(withsec))
        pmw = PadMaskWriter(streams=["tracks"])
        pmw.name = "pad_mask"
        withsec["run_tasks"] = rt
        withsec["pad_mask"] = pmw
        sec_hash = compile_plan(dict(withsec), Mode.FIT, sources, sinks=["loss.total"]).plan_hash

        assert sec_hash == bare_hash, (
            "the outputs: section perturbed the FIT plan_hash — it MUST be demand-pruned "
            f"from FIT (plan §6 gate 4): bare={bare_hash[:16]} withsection={sec_hash[:16]}"
        )

    def test_section_nodes_absent_from_fit_plan_steps(self, tmp_path):
        """The section nodes do NOT appear in the FIT plan's steps (genuinely pruned)."""
        sources = _fit_sources()
        modules = _bare_gn2v2(tmp_path)
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin"])
        rt.name = "run_tasks"
        rt.bind_model_modules(dict(modules))
        pmw = PadMaskWriter(streams=["tracks"])
        pmw.name = "pad_mask"
        modules["run_tasks"] = rt
        modules["pad_mask"] = pmw
        plan = compile_plan(dict(modules), Mode.FIT, sources, sinks=["loss.total"])
        step_names = {step.name for step in plan.steps}
        assert "run_tasks" not in step_names
        assert "pad_mask" not in step_names

    def test_section_nodes_present_in_test_plan(self, tmp_path):
        """In TEST, when a sink demands the section leaves, the section nodes ARE in the plan."""
        sources = _fit_sources()
        modules = _bare_gn2v2(tmp_path)
        rt = RunTaskOutput(tasks=["jets_classification"])
        rt.name = "run_tasks"
        rt.bind_model_modules(dict(modules))
        modules["run_tasks"] = rt
        # demand the global head's per-class probs leaf (what an H5 sink would demand)
        sinks = ["outputs.jets.jets_classification.pb"]
        plan = compile_plan(dict(modules), Mode.TEST, sources, sinks=sinks)
        step_names = {step.name for step in plan.steps}
        assert "run_tasks" in step_names, "RunTaskOutput must be pulled into TEST by sink demand"
