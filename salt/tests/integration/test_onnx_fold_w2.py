"""End-to-end gates for the plan-29 W2 folded ONNX path (design §6, §8 W2 row)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from salt.core.graph import Bundle, Executor, Mode
from salt.core.nn import bind_all, resolve_bind_schema
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    check_onnx,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.core.outputs import (
    ClassProbs,
    Combination,
    OnnxExportLeaf,
    OnnxExportSink,
    SeqClassIndex,
    VertexUnionFind,
)
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    compile_gn2v2,
    write_parity_norm_dict,
)

ORACLE_DIR = Path("/tmp/w2_oracle")

VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}

# plan 29 W2 B3: these are pure-CPU ONNX-trace gates (oracle byte-identity,
# folded==legacy argmax, combination==pb+pc, check_onnx incl L=0). They are NOT
# GPU/heavy integration tests despite living under tests/integration/ — the
# `cpu_always` marker (conftest.py) opts them OUT of the GPU skip so they run on
# EVERY CI invocation, with or without --run-integration. Skipping them silently
# would let an ONNX-contract regression (a reorder/rename/redtype, a folded-vs-
# legacy drift, or a double-softmax) ship unnoticed.
pytestmark = pytest.mark.cpu_always


def _export_cfg() -> ExportConfig:
    return ExportConfig(
        model_name="GN2v2",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )


# GO: the W0 oracle export contract is byte-identical post-W2 (legacy path)


def _folded_gn2_export(tmp_path):
    """A deterministically-weighted GN2 export through the FOLDED path (W4): ClassProbs +"""
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    torch.manual_seed(42)  # deterministic non-trivial weights (retired v1 transfer stand-in)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    jp = ClassProbs(task="jets_classification", stream="jets"); jp.name = "jet_probs"
    ti = SeqClassIndex(task="track_origin", stream="tracks"); ti.name = "track_origin_index"
    vi = VertexUnionFind(task="track_vertexing", stream="tracks"); vi.name = "track_vertex_index"
    sink = OnnxExportSink(outputs=[
        OnnxExportLeaf(key="outputs.jets.jets_classification", names=["pb", "pc", "pu"]),
        OnnxExportLeaf(key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8", per_token=True),
        OnnxExportLeaf(key="outputs.tracks.track_vertexing", name="VertexIndex", dtype="int8", per_token=True),
    ]); sink.name = "onnx_export"
    modules.update({"jet_probs": jp, "track_origin_index": ti, "track_vertex_index": vi, "onnx_export": sink})
    resolved = resolve_export_config(_export_cfg(), "GN2_v2")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    result = export_graph(
        modules, _export_cfg(), VARIABLES, tmp_path / "folded.onnx", outputs=[], run_name="GN2_v2"
    )
    return result, modules


@pytest.mark.skipif(
    not (ORACLE_DIR / "gn2v2.json").is_file(),
    reason="W0 oracle goldens not present (run /tmp/w2_oracle/dump_onnx_meta.py)",
)
def test_folded_gn2v2_export_contract_matches_oracle(tmp_path):
    """The MIGRATED folded gn2v2 export contract matches the W0 golden EXACTLY (W4)."""
    golden = json.loads((ORACLE_DIR / "gn2v2.json").read_text())
    result, _ = _folded_gn2_export(tmp_path)
    adapter = result.adapter
    assert adapter.output_names == golden["output_names"]  # ORDERED list-equality
    assert adapter.output_dtypes == golden["output_dtypes"]
    assert json.loads(json.dumps(adapter.dynamic_axes)) == golden["dynamic_axes"]
    example = adapter.example_inputs(sequence_length=5)
    with torch.no_grad():
        out_tuple = adapter(*example)
    assert len(out_tuple) == golden["output_tuple_len"]
    assert len(adapter.output_names) == golden["output_tuple_len"]
    session = make_session(result.onnx_path)
    assert [o.name for o in session.get_outputs()] == golden["output_names"]


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    """A deterministically-weighted GN2 export through the FOLDED path (SeqClassIndex +
    Combination + sink)."""
    tmp = tmp_path_factory.mktemp("onnx_fold")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    torch.manual_seed(42)  # deterministic non-trivial weights (retired v1 transfer stand-in)
    modules = build_gn2v2_modules(tmp / "norm_dict.yaml")
    jet_probs = ClassProbs(task="jets_classification", stream="jets")
    jet_probs.name = "jet_probs"
    track_index = SeqClassIndex(task="track_origin", stream="tracks")
    track_index.name = "track_origin_index"
    pbc = Combination(source="outputs.jets.jets_classification", name="pbc", terms={0: 1.0, 1: 1.0})
    pbc.name = "pbc"
    export_sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(key="outputs.jets.jets_classification", names=["pb", "pc", "pu"]),
            OnnxExportLeaf(key="outputs.jets.pbc", name="pbc"),
            OnnxExportLeaf(
                key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8", per_token=True
            ),
        ]
    )
    export_sink.name = "onnx_export"
    modules.update({
        "jet_probs": jet_probs,
        "track_origin_index": track_index,
        "pbc": pbc,
        "onnx_export": export_sink,
    })
    resolved = resolve_export_config(_export_cfg(), "GN2_v2")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    result = export_graph(
        modules, _export_cfg(), VARIABLES, tmp / "folded.onnx", outputs=[], run_name="GN2_v2"
    )
    return SimpleNamespace(result=result, modules=modules)


# folded export correctness


def test_folded_export_contract(folded):
    """The folded sink names the conversion leaves: split + combine + per-token int8 axis."""
    adapter = folded.result.adapter
    assert adapter.output_names == [
        "GN2v2_pb",
        "GN2v2_pc",
        "GN2v2_pu",
        "GN2v2_pbc",
        "GN2v2_TrackOrigin",
    ]
    assert adapter.output_dtypes == ["float32", "float32", "float32", "float32", "int8"]
    assert adapter.dynamic_axes["GN2v2_TrackOrigin"] == {0: "n_tracks"}


def test_folded_check_onnx_agrees_including_zero_tokens(folded):
    """torch-vs-ort 1e-6 incl. L=0 — the folded conversions trace correctly (§6.4 gate)."""
    grid = [{"tracks": length} for length in (0, 1, 2, 7, 21)]
    result = check_onnx(
        folded.result.adapter,
        folded.result.onnx_path,
        trials=2,
        float_rtol=1e-6,
        float_atol=1e-6,
        lengths_grid=grid,
    )
    assert result.passed, result.failures
    assert result.n_cases == 2 * len(grid)


def test_folded_combination_equals_pb_plus_pc(folded):
    """The ``pbc`` combination leaf == ``pb + pc`` bitwise (folds the combine loop, Q2)."""
    session = make_session(folded.result.onnx_path)
    gen = torch.Generator().manual_seed(11)
    jets = torch.rand(1, len(JET_VARIABLES), generator=gen).numpy()
    tracks = torch.rand(7, len(TRACK_VARIABLES), generator=gen).numpy()
    out = dict(
        zip(
            folded.result.adapter.output_names,
            session.run(None, {"jet_features": jets, "track_features": tracks}),
            strict=True,
        )
    )
    pbc = np.ravel(out["GN2v2_pbc"])
    pb_plus_pc = np.ravel(out["GN2v2_pb"]) + np.ravel(out["GN2v2_pc"])
    np.testing.assert_array_equal(pbc, pb_plus_pc)



# folded correctness vs the v1 weight oracle + the W0 oracle contract (W4: the
# legacy reduce path is RETIRED — folded is the SOLE path. The argmax/union-find
# math equivalence vs v1's chains is proven in test_onnx_fold_w3.py).


def test_folded_pb_pc_pu_equal_single_softmax_of_raw_logits(tmp_path):
    """POST-P4 no-double-softmax: the folded pb/pc/pu == ONE softmax of the RAW
    TEST-plan logits, computed through the independent Executor path (the eager
    v2 model on the SAME weights — replaces the retired v1 forward oracle)."""
    result, modules = _folded_gn2_export(tmp_path)
    session = make_session(result.onnx_path)
    test_plan = compile_gn2v2(modules, Mode.TEST)
    gen = torch.Generator().manual_seed(31)
    for _ in range(4):
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(7, len(TRACK_VARIABLES), generator=gen)
        out = dict(zip(result.adapter.output_names,
                       session.run(None, {"jet_features": jets.numpy(), "track_features": tracks.numpy()}),
                       strict=True))
        folded = np.array([np.ravel(out[f"GN2v2_{s}"])[0] for s in ("pb", "pc", "pu")])
        b = Bundle()
        b.set("inputs.jets", jets.clone())
        b.set("inputs.tracks", tracks.unsqueeze(0).clone())
        b.set("masks.tracks", torch.zeros((1, 7), dtype=torch.bool))
        with torch.no_grad():
            res = Executor(test_plan).run(b)
        # ONE softmax of the raw TEST logits — a double softmax in the folded
        # path would break the 1e-6 agreement
        ref = torch.softmax(res.get("preds.jets.jets_classification"), dim=-1).numpy().ravel()
        np.testing.assert_allclose(folded, ref, atol=1e-6)


@pytest.mark.skipif(
    not (ORACLE_DIR / "gn2v2.json").is_file(),
    reason="W0 oracle goldens not present",
)
def test_folded_int8_track_origin_check_onnx(tmp_path):
    """The folded int8 TrackOrigin leaf is torch-vs-ort exact incl L=0 (folds _bind_argmax)."""
    result, _ = _folded_gn2_export(tmp_path)
    grid = [{"tracks": length} for length in (0, 1, 2, 7, 21)]
    cr = check_onnx(result.adapter, result.onnx_path, trials=2, float_rtol=1e-6, float_atol=1e-6, lengths_grid=grid)
    assert cr.passed, cr.failures
