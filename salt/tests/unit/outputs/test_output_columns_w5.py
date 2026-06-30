"""W5.0 unit gate — producer ``output_columns`` reproduces the legacy task schema.

Plan 31 W5.0: each conversion producer self-describes its output columns as a
field manifest (`OutputField`), resolved FROM THE TASK IT WRAPS by reusing the
legacy ``task.output_names()`` / ``class_suffixes`` / ``output_suffixes`` /
`VERTEX_INDEX` logic. This gate asserts, per task family, that the manifest's
H5 names (and ONNX names) MATCH the legacy `TaskWriter`-derived schema exactly,
so the W5.1 auto-collecting sinks reproduce the legacy column schema byte-for-byte:

- **classification (global)** — `ClassProbs`: H5 + ONNX = the per-class suffixes.
- **seq-class probs (per-token H5)** — `SeqClassProbs`: H5 = per-class suffixes,
  no ONNX (the ONNX is the argmax index).
- **seq-class index (per-token ONNX)** — `SeqClassIndex`: no H5, ONNX = the
  pascal-case task name (``track_origin -> TrackOrigin``), int8 per-token.
- **regression** — identity `TaskOutput` over a `RegressionTaskModule`: H5 +
  ONNX = ``output_suffixes`` (``custom_output_names`` else targets; doubled for
  gaussian), covering custom-name and multi-target cases.
- **vertexing (ONNX)** — `VertexUnionFind`: no H5, ONNX = `VERTEX_INDEX`, int8.
- **combination** — `Combination`: H5 + ONNX = its own name, global float.
"""

from __future__ import annotations

import pytest

from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.core.outputs import (
    ClassProbs,
    Combination,
    Regression,
    SeqClassIndex,
    SeqClassProbs,
    TaskOutput,
    VertexUnionFind,
)
from salt.core.outputs.names import VERTEX_INDEX

RUN = "MyRun"


def _named(modules: dict) -> dict:
    """Assign each module its dict key as ``.name`` (the SaltModule binding contract)."""
    for name, module in modules.items():
        module.name = name
    return modules


def _legacy_h5_suffixes(task, run_name: str) -> list[str]:
    """The legacy ``task.output_names`` H5 suffixes (run-name prefix stripped)."""
    out = []
    for column, _dtype in task.output_names(run_name):
        out.append(column[len(run_name) + 1 :] if column.startswith(f"{run_name}_") else column)
    return out


# ---------------------------------------------------------------------------
# classification — global + per-token (probs H5 + index ONNX)
# ---------------------------------------------------------------------------


def _class_modules() -> dict:
    return _named({
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=["Pileup", "Fake", "Primary", "FromB"],
            context="pooled.global",
            dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
    })


def test_classprobs_global_matches_legacy():
    mods = _class_modules()
    jet_probs = ClassProbs(task="jets_classification", stream="jets")
    jet_probs.name = "jet_probs"
    fields = jet_probs.output_columns(RUN, mods)
    # H5 names == legacy task.output_names suffixes (pb/pc/pu)
    assert [f.h5_name for f in fields] == _legacy_h5_suffixes(mods["jets_classification"], RUN)
    # ONNX defaults to the same suffixes; all global float, final
    assert [f.resolved_onnx_name for f in fields] == ["pb", "pc", "pu"]
    assert all(f.axis == "global" and f.dtype == "f4" and f.final for f in fields)


def test_seqclassprobs_per_token_h5_only_matches_legacy():
    mods = _class_modules()
    probs = SeqClassProbs(task="track_origin", stream="tracks")
    probs.name = "track_origin_probs"
    fields = probs.output_columns(RUN, mods)
    assert [f.h5_name for f in fields] == _legacy_h5_suffixes(mods["track_origin"], RUN)
    # the H5 prob columns have NO ONNX representation (the ONNX is the argmax index)
    assert all(f.onnx_name is None for f in fields)
    assert all(f.axis == "per_token" and f.dtype == "f4" and f.final for f in fields)


def test_seqclassindex_per_token_onnx_only_matches_legacy():
    mods = _class_modules()
    index = SeqClassIndex(task="track_origin", stream="tracks")
    index.name = "track_origin_index"
    fields = index.output_columns(RUN, mods)
    assert len(fields) == 1
    f = fields[0]
    # the ONNX argmax index has NO H5 column; the Athena name is pascal_case(task)
    assert f.h5_name is None
    assert f.resolved_onnx_name == "TrackOrigin"  # pascal_case('track_origin')
    assert f.dtype == "int8" and f.axis == "per_token" and f.final
    # matches the legacy onnx_outputs name exactly
    legacy = mods["track_origin"].onnx_outputs()
    assert legacy[0].name == f.resolved_onnx_name


# ---------------------------------------------------------------------------
# regression — identity TaskOutput; custom names + multi-target + gaussian
# ---------------------------------------------------------------------------


def _reg_modules() -> dict:
    return _named({
        "reg_normed": RegressionTaskModule(
            stream="jets",
            input="pooled.global",
            targets="HadronConeExclTruthLabelPt",
            norm_params={"mean": 1.0, "std": 1.0},
            loss="MSELoss",
            dense={"hidden_layers": [4], "activation": "SiLU"},
        ),
        "reg_multinorm": RegressionTaskModule(
            stream="jets",
            input="pooled.global",
            targets=["R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt"],
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            loss="MSELoss",
            dense={"hidden_layers": [4], "activation": "SiLU"},
        ),
        "reg_ratio": RegressionTaskModule(
            stream="jets",
            input="pooled.global",
            targets="HadronConeExclTruthLabelPt",
            target_denominators="pt_btagJes",
            custom_output_names="pt",
            loss="MSELoss",
            dense={"hidden_layers": [4], "activation": "SiLU"},
        ),
    })


@pytest.mark.parametrize("task_name", ["reg_normed", "reg_multinorm", "reg_ratio"])
def test_regression_identity_passthrough_matches_legacy(task_name):
    mods = _reg_modules()
    prod = TaskOutput(task=task_name, stream="jets")  # identity (W5 stopgap)
    prod.name = f"{task_name}_out"
    fields = prod.output_columns(RUN, mods)
    # H5 names == legacy output_names suffixes (targets / custom names)
    assert [f.h5_name for f in fields] == _legacy_h5_suffixes(mods[task_name], RUN)
    # ONNX defaults to the same suffix (== onnx_outputs names)
    legacy_onnx = mods[task_name].onnx_outputs()[0]
    legacy_onnx_names = legacy_onnx.names if legacy_onnx.names is not None else [legacy_onnx.name]
    assert [f.resolved_onnx_name for f in fields] == list(legacy_onnx_names)
    assert all(f.axis == "global" and f.dtype == "f4" and f.final for f in fields)


def test_regression_descale_producer_matches_legacy():
    mods = _reg_modules()
    prod = Regression(task="reg_ratio", stream="jets", targets=["HadronConeExclTruthLabelPt"],
                      target_denominators=["pt_btagJes"])
    prod.name = "reg_ratio_descale"
    fields = prod.output_columns(RUN, mods)
    assert [f.h5_name for f in fields] == _legacy_h5_suffixes(mods["reg_ratio"], RUN)


def test_regression_gaussian_doubles_suffixes():
    mods = _named({
        "reg_gauss": RegressionTaskModule(
            stream="jets",
            input="pooled.global",
            targets=["pt", "mass"],
            gaussian=True,
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            loss="GaussianNLLLoss",
            dense={"hidden_layers": [4], "activation": "SiLU"},
        ),
    })
    prod = TaskOutput(task="reg_gauss", stream="jets")
    prod.name = "reg_gauss_out"
    fields = prod.output_columns(RUN, mods)
    # gaussian: R means then R _stddev suffixes (legacy output_suffixes doubling)
    assert [f.h5_name for f in fields] == _legacy_h5_suffixes(mods["reg_gauss"], RUN)
    assert [f.h5_name for f in fields] == ["pt", "mass", "pt_stddev", "mass_stddev"]


# ---------------------------------------------------------------------------
# vertexing — ONNX union-find index (no H5 column, deferred family)
# ---------------------------------------------------------------------------


def test_vertex_union_find_onnx_only_matches_legacy():
    mods = _named({
        "track_vertexing": VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            context="pooled.global",
            dense={"hidden_layers": [4], "activation": "ReLU"},
        ),
    })
    node = VertexUnionFind(task="track_vertexing", stream="tracks")
    node.name = "track_vertex_index"
    fields = node.output_columns(RUN, mods)
    assert len(fields) == 1
    f = fields[0]
    assert f.h5_name is None  # vertexing H5 is a deferred family (no producer column)
    assert f.resolved_onnx_name == VERTEX_INDEX
    assert f.dtype == "int8" and f.axis == "per_token" and f.final
    # matches the legacy vertexing onnx_outputs suffix exactly
    assert mods["track_vertexing"].onnx_outputs()[0].name == VERTEX_INDEX


# ---------------------------------------------------------------------------
# combination — self-named global float (H5 + ONNX)
# ---------------------------------------------------------------------------


def test_combination_self_named():
    node = Combination(source="outputs.jets.jets_classification", name="pbc", terms={0: 1.0, 1: 1.0})
    node.name = "pbc"
    fields = node.output_columns(RUN, {})
    assert len(fields) == 1
    f = fields[0]
    assert f.h5_name == "pbc" and f.resolved_onnx_name == "pbc"
    assert f.dtype == "f4" and f.axis == "global" and f.final
