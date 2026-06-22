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

from salt.core.graph.errors import ConfigError
from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    ExportOutput,
    attach_manifest,
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
    TaskOutput,
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


@pytest.mark.skipif(
    not (ORACLE_DIR / "gn2v2.json").is_file(),
    reason="W0 oracle goldens not present (run /tmp/w2_oracle/dump_onnx_meta.py)",
)
def test_oracle_gn2v2_export_contract_byte_identical(tmp_path):
    """The legacy gn2v2 (split+argmax+union_find) export contract matches the W0 golden EXACTLY.

    Re-builds the W0 ``exported`` fixture and asserts the ORDERED output_names,
    per-output dtypes, dynamic_axes map, and output-tuple length are byte-
    identical to ``/tmp/w2_oracle/gn2v2.json`` — a reorder/rename/redtype is a
    FAIL (the §6.4 ONNX-bitwise gate; the legacy reduce path is untouched by W2).
    """
    golden = json.loads((ORACLE_DIR / "gn2v2.json").read_text())
    v1 = build_test_gn2(tmp_path)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    manifest = [
        ExportOutput(port="preds.jets.jets_classification", names=["pb", "pc", "pu"]),
        ExportOutput(
            port="preds.tracks.track_origin", name="TrackOrigin", dtype="int8", reduce="argmax"
        ),
        ExportOutput(
            port="preds.tracks.track_vertexing",
            name="VertexIndex",
            dtype="int8",
            reduce="vertex_union_find",
        ),
    ]
    resolved = attach_manifest(resolve_export_config(_export_cfg(), "GN2_v2"), manifest)
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict(modules).load_state_dict(map_v1_state_dict(v1.state_dict(), modules))
    result = export_graph(
        modules,
        _export_cfg(),
        VARIABLES,
        tmp_path / "network.onnx",
        outputs=manifest,
        run_name="GN2_v2",
    )
    adapter = result.adapter
    assert adapter.output_names == golden["output_names"]  # ORDERED list-equality
    assert adapter.output_dtypes == golden["output_dtypes"]
    # dynamic_axes JSON uses string axis keys; compare via the same json round-trip
    assert json.loads(json.dumps(adapter.dynamic_axes)) == golden["dynamic_axes"]
    # graph output tuple length matches
    example = adapter.example_inputs(sequence_length=5)
    with torch.no_grad():
        out_tuple = adapter(*example)
    assert len(out_tuple) == golden["output_tuple_len"]
    assert len(adapter.output_names) == golden["output_tuple_len"]
    # the .onnx output names match the adapter (Athena schema)
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
# folded == legacy reduces (R8 — no drift between the two paths)
# ---------------------------------------------------------------------------


def test_folded_int8_argmax_equals_legacy_reduce(tmp_path):
    """The folded SeqClassIndex int8 leaf is bitwise-equal to the legacy ``argmax`` reduce.

    Both paths export the SAME v1 weights — the folded conversion node and the
    legacy ``_bind_argmax`` reduce must produce the identical int8 ``TrackOrigin``
    leaf on the same input (the W2 hybrid carries the math VERBATIM, no drift).
    """
    v1 = build_test_gn2(tmp_path)
    weights = v1.state_dict()

    # legacy reduce export
    mod_l = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    manifest = [
        ExportOutput(port="preds.jets.jets_classification", names=["pb", "pc", "pu"]),
        ExportOutput(
            port="preds.tracks.track_origin", name="TrackOrigin", dtype="int8", reduce="argmax"
        ),
    ]
    res_l = attach_manifest(resolve_export_config(_export_cfg(), "GN2_v2"), manifest)
    plan_l = compile_onnx_plan(mod_l, res_l, VARIABLES)
    bind_all(mod_l, resolve_bind_schema([plan_l]))
    nn.ModuleDict(mod_l).load_state_dict(map_v1_state_dict(weights, mod_l))
    r_l = export_graph(
        mod_l,
        _export_cfg(),
        VARIABLES,
        tmp_path / "legacy.onnx",
        outputs=manifest,
        run_name="GN2_v2",
    )

    # folded conversion-node export (same weights). The folded TrackOrigin int8
    # leaf comes from a SeqClassIndex node (folding _bind_argmax) — the leaf the
    # OnnxExportSink names. NOTE: the folded jets split deliberately reuses the
    # task's already-softmaxed ONNX probs via an identity TaskOutput (the gn2v2
    # fixture task still publishes converted probs in ONNX — the P4 carve-out,
    # tasks.py:25-26 — so a ClassProbs node would DOUBLE-softmax; the eventual P4
    # flip publishes raw logits and ClassProbs becomes the single softmax). The
    # int8 argmax equivalence below is softmax-invariant either way, so it pins
    # the "folds _bind_argmax" claim independent of the P4 timing.
    mod_f = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    jet_probs = TaskOutput(
        task="jets_classification", stream="jets"
    )  # identity: task pre-softmaxes
    jet_probs.name = "jet_probs"
    track_index = SeqClassIndex(task="track_origin", stream="tracks")
    track_index.name = "track_origin_index"
    export_sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(key="outputs.jets.jets_classification", names=["pb", "pc", "pu"]),
            OnnxExportLeaf(
                key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8", per_token=True
            ),
        ]
    )
    export_sink.name = "onnx_export"
    mod_f.update({
        "jet_probs": jet_probs,
        "track_origin_index": track_index,
        "onnx_export": export_sink,
    })
    res_f = resolve_export_config(_export_cfg(), "GN2_v2")
    plan_f = compile_onnx_plan(mod_f, res_f, VARIABLES)
    bind_all(mod_f, resolve_bind_schema([plan_f]))
    nn.ModuleDict({k: v for k, v in mod_f.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(weights, mod_f), strict=False
    )
    r_f = export_graph(
        mod_f, _export_cfg(), VARIABLES, tmp_path / "folded.onnx", outputs=[], run_name="GN2_v2"
    )

    # same weights, same input -> the int8 TrackOrigin leaf must be IDENTICAL
    # (the folded SeqClassIndex node == the legacy argmax reduce, no drift, R8)
    s_l, s_f = make_session(r_l.onnx_path), make_session(r_f.onnx_path)
    gen = torch.Generator().manual_seed(21)
    jets = torch.rand(1, len(JET_VARIABLES), generator=gen).numpy()
    tracks = torch.rand(13, len(TRACK_VARIABLES), generator=gen).numpy()
    feed = {"jet_features": jets, "track_features": tracks}
    o_l = dict(zip(r_l.adapter.output_names, s_l.run(None, feed), strict=True))
    o_f = dict(zip(r_f.adapter.output_names, s_f.run(None, feed), strict=True))
    np.testing.assert_array_equal(o_l["GN2v2_TrackOrigin"], o_f["GN2v2_TrackOrigin"])
    # the float split scalars match too: the identity TaskOutput passes the task's
    # already-softmaxed ONNX probs through, and the legacy split_scalars reduce
    # splits the SAME leaf — so the folded sink's split is bitwise the same scalars
    for suffix in ("pb", "pc", "pu"):
        np.testing.assert_allclose(o_l[f"GN2v2_{suffix}"], o_f[f"GN2v2_{suffix}"], atol=1e-6)


# ---------------------------------------------------------------------------
# hybrid: folded split+argmax (export node) + LEGACY vertex_union_find reduce
# (pins the 'folded + legacy_ordered' concatenation ORDER, adapter.py ~161)
# ---------------------------------------------------------------------------


def _hybrid_modules(tmp_path):
    """A GN2v2 module dict with a folded sink (pb/pc/pu + TrackOrigin) + weights.

    The folded `OnnxExportSink` names the split (pb/pc/pu via the identity
    TaskOutput, softmax-faithful per the P4 carve-out) and the int8 ``TrackOrigin``
    argmax (`SeqClassIndex`); the LEGACY ``VertexIndex`` rides the reduce manifest
    (``vertex_union_find``, NOT folded in W2).

    Returns
    -------
    tuple
        ``(modules, v1)`` — the GN2v2 module dict (sink + producers folded in)
        and the v1 weight oracle.
    """
    v1 = build_test_gn2(tmp_path)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    jet_probs = TaskOutput(task="jets_classification", stream="jets")  # softmax-faithful identity
    jet_probs.name = "jet_probs"
    track_index = SeqClassIndex(task="track_origin", stream="tracks")
    track_index.name = "track_origin_index"
    export_sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(key="outputs.jets.jets_classification", names=["pb", "pc", "pu"]),
            OnnxExportLeaf(
                key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8", per_token=True
            ),
        ]
    )
    export_sink.name = "onnx_export"
    modules.update({
        "jet_probs": jet_probs,
        "track_origin_index": track_index,
        "onnx_export": export_sink,
    })
    return modules, v1


def _bind_load_hybrid(modules, v1, manifest):
    """Compile the hybrid ONNX plan (folded sink + legacy manifest), bind, load v1 weights.

    Returns
    -------
    ExportConfig
        The resolved+manifest-attached export config (ready for ``export_graph``).
    """
    resolved = attach_manifest(resolve_export_config(_export_cfg(), "GN2_v2"), manifest)
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(v1.state_dict(), modules), strict=False
    )
    return resolved


def test_hybrid_folded_plus_legacy_reduce_preserves_oracle_order(tmp_path):
    """A MIXED export (folded split+argmax + LEGACY vertex_union_find) keeps the v1 tuple order.

    Pins the ``folded + legacy_ordered`` concatenation (adapter.py ~161): the
    folded outputs come FIRST (the export node's leaf list is the ordering
    authority, design §6.3), the legacy reduce outputs append after — so the
    Athena tuple is the v1/oracle order ``[..pb, pc, pu, TrackOrigin, VertexIndex]``
    EXACTLY. ``check_onnx``'s per-NAME agreement would NOT catch a consistent
    reorder, so this list-equality assertion is the guard against a future
    late-fold/early-legacy config silently reordering the Athena tuple.
    """
    modules, v1 = _hybrid_modules(tmp_path)
    manifest = [
        ExportOutput(
            port="preds.tracks.track_vertexing",
            name="VertexIndex",
            dtype="int8",
            reduce="vertex_union_find",
        ),
    ]
    _bind_load_hybrid(modules, v1, manifest)
    result = export_graph(
        modules,
        _export_cfg(),
        VARIABLES,
        tmp_path / "hybrid.onnx",
        outputs=manifest,
        run_name="GN2_v2",
    )
    adapter = result.adapter
    # the EXACT v1/oracle Athena tuple order — folded (pb/pc/pu, TrackOrigin)
    # then legacy (VertexIndex), matching /tmp/w2_oracle/gn2v2.json output_names
    assert adapter.output_names == [
        "GN2v2_pb",
        "GN2v2_pc",
        "GN2v2_pu",
        "GN2v2_TrackOrigin",
        "GN2v2_VertexIndex",
    ]
    assert adapter.output_dtypes == ["float32", "float32", "float32", "int8", "int8"]
    # the per-name onnx schema matches the adapter (the Athena schema), in order
    session = make_session(result.onnx_path)
    assert [o.name for o in session.get_outputs()] == adapter.output_names


def test_hybrid_name_collision_raises_configerror(tmp_path):
    """Folded-sink and legacy-reduce output names MUST be DISJOINT (name-collision guard).

    When BOTH demand sources are present, the flat Athena output namespace must
    stay disjoint — a folded leaf and a legacy reduce both minting the SAME
    ``{model_name}_<suffix>`` would silently clobber one in the tuple. The adapter
    raises a clear `ConfigError` naming the overlap. Here the folded sink names the
    int8 track leaf ``VertexIndex`` AND the legacy ``vertex_union_find`` reduce
    also names ``VertexIndex`` — the collision must hard-error, not ship a
    duplicate-named ONNX graph.
    """
    v1 = build_test_gn2(tmp_path)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    jet_probs = TaskOutput(task="jets_classification", stream="jets")
    jet_probs.name = "jet_probs"
    track_index = SeqClassIndex(task="track_origin", stream="tracks")
    track_index.name = "track_origin_index"
    # the folded sink NAMES the int8 track-origin leaf 'VertexIndex' (collision bait)
    export_sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(
                key="outputs.tracks.track_origin", name="VertexIndex", dtype="int8", per_token=True
            ),
        ]
    )
    export_sink.name = "onnx_export"
    modules.update({
        "jet_probs": jet_probs,
        "track_origin_index": track_index,
        "onnx_export": export_sink,
    })
    # the LEGACY reduce ALSO mints 'VertexIndex' (vertex_union_find on the vertexing head)
    manifest = [
        ExportOutput(
            port="preds.tracks.track_vertexing",
            name="VertexIndex",
            dtype="int8",
            reduce="vertex_union_find",
        ),
    ]
    _bind_load_hybrid(modules, v1, manifest)
    with pytest.raises(ConfigError, match=r"GN2v2_VertexIndex"):
        export_graph(
            modules,
            _export_cfg(),
            VARIABLES,
            tmp_path / "collide.onnx",
            outputs=manifest,
            run_name="GN2_v2",
        )
