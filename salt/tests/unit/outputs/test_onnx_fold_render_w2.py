"""Render gate for the plan-29 W2 folded ONNX path (design §7 render payoff, §8 W2 row).

The W2 render payoff (the ONNX counterpart of W1's H5 card): a config wiring the
folded conversions (argmax/split/combine) + the declare-only `OnnxExportSink`
renders an ``onnx_export (OnnxExportSink)`` card in ``--mode onnx`` with the
folded conversion nodes appearing on-graph — REPLACING the off-graph reduce
manifest (which drew NO export card). This pins:

- the folded ONNX config renders the ``onnx_export`` card + the conversion nodes
  (jet_probs / track_origin_index / pbc) with named-consumer edges into the sink,
  and NO ``<sinks>`` sentinel;
- the export sink is ONNX-only — pruned from TEST (no card there);
- the FIT plan_hash is byte-unchanged vs the same config with the folded nodes +
  export sink removed (resume-safe — the W2 nodes exist only for ONNX).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from salt.core.cli import load_config
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import Mode
from salt.core.render import dot_source

_CONFIGS = Path(__file__).parents[3] / "core" / "configs"
_DUMMY = str(_CONFIGS / "gn2v2-dummy.yaml")
_FOLD = str(_CONFIGS / "gn2v2-dummy-onnx-fold.yaml")
_OVERRIDES = [
    "model.modules.norm.init_args.norm_dict=unused.yaml",
    "trainer.logger=false",
]

_FOLDED_NODES = ("jet_probs", "track_origin_index", "pbc")


@pytest.fixture(scope="module")
def fold_cfg():
    """The folded-ONNX config (gn2v2-dummy + the W2 fold override).

    Returns
    -------
    GraphConfig
        The loaded config (modules include the folded conversions + onnx_export).
    """
    return load_config([_DUMMY, _FOLD], _OVERRIDES)


def _compile(cfg, mode):
    """Compile one mode from a loaded `GraphConfig` (the static CLI plan).

    Returns
    -------
    Plan
        The compiled plan.
    """
    return compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )


def test_onnx_render_has_named_export_sink_card(fold_cfg):
    """The ONNX DOT renders the OnnxExportSink as its OWN named card (design §7 W2 payoff)."""
    onnx = _compile(fold_cfg, Mode.ONNX)
    dot = dot_source(onnx, fold_cfg.modules)

    assert "<sinks>" not in dot  # the off-graph manifest sentinel is gone
    assert "onnx_export" in dot
    assert "OnnxExportSink" in dot
    # the folded conversion nodes render on-graph (not the off-graph manifest)
    for node in _FOLDED_NODES:
        assert node in onnx.module_names
        assert node in dot
    # named-consumer edges flow into the NAMED export node
    edges = set(re.findall(r'"(\w+)" -> "onnx_export"', dot))
    assert set(_FOLDED_NODES) <= edges


def test_export_sink_pruned_from_test_render(fold_cfg):
    """The ONNX-only export sink is pruned from TEST (no card there)."""
    test = _compile(fold_cfg, Mode.TEST)
    assert "onnx_export" not in test.module_names
    dot = dot_source(test, fold_cfg.modules)
    assert "onnx_export" not in dot


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL])
def test_fit_val_plan_hash_unchanged_by_folded_onnx_nodes(fold_cfg, mode):
    """FIT/VAL ``plan_hash`` is byte-identical with vs without the W2 folded nodes.

    The folded conversion producers + the OnnxExportSink exist ONLY for the ONNX
    export path — all are demand-pruned/inactive in FIT/VAL. So the folded config
    must hash IDENTICALLY in FIT/VAL to the same config with those nodes removed,
    or checkpoint resume would break (``saltmodule._verify_ckpt_hash`` on FIT).
    """
    full = _compile(fold_cfg, mode).plan_hash
    fold_only = {"jet_probs", "track_origin_index", "pbc", "onnx_export"}
    base_modules = {k: v for k, v in fold_cfg.modules.items() if k not in fold_only}
    base = compile_plan(
        base_modules,
        mode,
        fold_cfg.sources,
        schema=fold_cfg.schema,
        sinks=fold_cfg.sinks,
        sink_origins=fold_cfg.sink_origins.get(mode),
    ).plan_hash
    assert full == base
