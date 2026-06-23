"""End-to-end gates for the plan-29 W2 folded ONNX path (design §6, §8 W2 row).

W2 folds the global-float (split_scalars naming), the int8 argmax, and the
combination conversions into the executor-traced forward, with a declare-only
`OnnxExportSink` naming the conversion leaves instead of the off-graph reduce
manifest. The HARD CONSTRAINT is that the LEGACY reduce path stays
BITWISE-identical (the W0 oracle) and the folded path is correct + agrees with
the legacy reduces (R8 hybrid). This file asserts:

- **GO oracle identity**: the W0 golden (``/tmp/w2_oracle/*.json``) export
  contract (ordered output_names, dtypes, dynamic_axes, tuple length) is
  byte-identical post-W2 for the legacy fixtures (gn2v2 + two_stream) — a
  reordered/renamed/redtyped output is a FAIL;
- **folded export correctness**: a softmax+argmax+combine folded config traces,
  ``check_onnx`` agrees torch-vs-ort 1e-6 incl. L=0, the int8 leaf is exact;
- **folded == legacy reduces**: the folded int8 argmax leaf is bitwise-equal to
  the legacy ``argmax`` reduce on the SAME weights (folding ``_bind_argmax``);
- **the combination leaf** equals ``pb + pc`` bitwise (folding the combine loop).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
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
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
)
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

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


# ---------------------------------------------------------------------------
# GO: the W0 oracle export contract is byte-identical post-W2 (legacy path)
# ---------------------------------------------------------------------------


def _folded_gn2_export(tmp_path):
    """A weight-matched GN2 export through the FOLDED path (W4): ClassProbs +
    SeqClassIndex + VertexUnionFind named by an OnnxExportSink — pb/pc/pu,
    TrackOrigin int8, VertexIndex int8 (the v1/oracle tuple)."""
    v1 = build_test_gn2(tmp_path)
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
    nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(v1.state_dict(), modules), strict=False
    )
    return export_graph(modules, _export_cfg(), VARIABLES, tmp_path / "folded.onnx", outputs=[], run_name="GN2_v2")


@pytest.mark.skipif(
    not (ORACLE_DIR / "gn2v2.json").is_file(),
    reason="W0 oracle goldens not present (run /tmp/w2_oracle/dump_onnx_meta.py)",
)
def test_folded_gn2v2_export_contract_matches_oracle(tmp_path):
    """The MIGRATED folded gn2v2 export contract matches the W0 golden EXACTLY (W4).

    POST-W4 the gn2v2 export is the FOLDED conversion-node path (ClassProbs +
    SeqClassIndex + VertexUnionFind named by an OnnxExportSink) — the off-graph
    reduce manifest is retired. The ORDERED output_names, per-output dtypes,
    dynamic_axes map, and output-tuple length must still be byte-identical to
    ``/tmp/w2_oracle/gn2v2.json`` (raw .onnx bytes differ — node names move when
    the conversions fold, §6.4; the criterion is the contract-field identity).
    """
    golden = json.loads((ORACLE_DIR / "gn2v2.json").read_text())
    result = _folded_gn2_export(tmp_path)
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
    """A weight-matched GN2 export through the FOLDED path (SeqClassIndex + Combination + sink).

    Same v1 weights as the W0 oracle, but the ONNX outputs come from folded
    conversion nodes named by an `OnnxExportSink` instead of the reduce manifest:
    jets split_scalars (pb/pc/pu), a ``pbc`` combination, and the int8 argmax
    ``TrackOrigin`` leaf (folding ``_bind_argmax``).

    Returns
    -------
    SimpleNamespace
        ``.result`` (ExportResult), ``.v1`` (the weight oracle).
    """
    tmp = tmp_path_factory.mktemp("onnx_fold")
    v1 = build_test_gn2(tmp)
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
    nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(v1.state_dict(), modules), strict=False
    )
    result = export_graph(
        modules, _export_cfg(), VARIABLES, tmp / "folded.onnx", outputs=[], run_name="GN2_v2"
    )
    return SimpleNamespace(result=result, v1=v1, modules=modules)


# ---------------------------------------------------------------------------
# folded export correctness
# ---------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# folded correctness vs the v1 weight oracle + the W0 oracle contract (W4: the
# legacy reduce path is RETIRED — folded is the SOLE path. The argmax/union-find
# math equivalence vs v1's chains is proven in test_onnx_fold_w3.py).
# ---------------------------------------------------------------------------


def test_folded_pb_pc_pu_equal_single_v1_softmax(tmp_path):
    """POST-P4 no-double-softmax: the folded pb/pc/pu == ONE softmax of the v1 raw logits.

    The classification task now publishes RAW logits in ONNX (P4), so the folded
    ClassProbs node applies the SINGLE softmax inside the traced graph. The
    exported pb/pc/pu must match a single softmax of the weight-matched v1 head's
    raw logits within 1e-6 — proving the cutover did NOT double-softmax.
    """
    result = _folded_gn2_export(tmp_path)
    v1 = build_test_gn2(tmp_path)
    v1.eval()
    session = make_session(result.onnx_path)
    gen = torch.Generator().manual_seed(31)
    for _ in range(4):
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(7, len(TRACK_VARIABLES), generator=gen)
        out = dict(zip(result.adapter.output_names,
                       session.run(None, {"jet_features": jets.numpy(), "track_features": tracks.numpy()}),
                       strict=True))
        folded = np.array([np.ravel(out[f"GN2v2_{s}"])[0] for s in ("pb", "pc", "pu")])
        with torch.no_grad():
            preds, _ = v1({"jets": jets.clone(), "tracks": tracks.unsqueeze(0).clone()},
                          {"tracks": torch.zeros((1, 7), dtype=torch.bool)}, None)
        ref = torch.softmax(preds["jets"]["jets_classification"], dim=-1).numpy().ravel()
        np.testing.assert_allclose(folded, ref, atol=1e-6)


@pytest.mark.skipif(
    not (ORACLE_DIR / "gn2v2.json").is_file(),
    reason="W0 oracle goldens not present",
)
def test_folded_int8_track_origin_check_onnx(tmp_path):
    """The folded int8 TrackOrigin leaf is torch-vs-ort exact incl L=0 (folds _bind_argmax)."""
    result = _folded_gn2_export(tmp_path)
    grid = [{"tracks": length} for length in (0, 1, 2, 7, 21)]
    cr = check_onnx(result.adapter, result.onnx_path, trials=2, float_rtol=1e-6, float_atol=1e-6, lengths_grid=grid)
    assert cr.passed, cr.failures
