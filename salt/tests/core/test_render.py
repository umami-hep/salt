"""Tests for `salt.core.render`: plan table, DOT source, matplotlib DAG renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.graph.planner import compile_plan
from salt.core.graph.probe import SYNTHETIC, probe_shapes
from salt.core.graph.spec import Mode, TensorSpec, unflatten_spec
from salt.core.render import dot_source, plan_table, render_graph
from salt.tests.core.toys import ToyEmbed, ToyHead, ToySource, ToyWildcardLabels

# the in-repo test-scale GN2v2 config (16-dim, no machine paths) — synthetic
# probing needs no data file, only a placeholder norm_dict the probe replaces
_DUMMY_CFG = str(Path(__file__).parent.parent.parent / "core" / "configs" / "gn2v2-dummy.yaml")
_NORM_PLACEHOLDER = ["model.modules.norm.init_args.norm_dict=unused.yaml"]


@pytest.fixture(scope="module")
def plan():
    modules = {
        "source": ToySource(),
        "embed": ToyEmbed(),
        "labels": ToyWildcardLabels(),
        "head": ToyHead(),
    }
    for name, module in modules.items():
        module.name = name
    return compile_plan(
        modules,
        Mode.FIT,
        sources=unflatten_spec({"raw.x": TensorSpec(shape=("B", 8), dtype="float32")}),
        schema=("labels.x",),
        sinks=["losses.total"],
    )


class TestPlanTable:
    def test_header_steps_and_sources(self, plan):
        text = plan_table(plan)
        assert f"plan [mode=FIT] {len(plan.steps)} steps" in text
        assert f"plan_hash={plan.plan_hash}" in text
        for step in plan.steps:
            assert step.name in text
        assert "sources: raw.x" in text

    def test_narrowed_wildcards_section(self, plan):
        text = plan_table(plan)
        assert "narrowed wildcards [mode=FIT]:" in text
        assert "labels: labels.x" in text


class TestDotSource:
    def test_port_cards_and_edges(self, plan):
        dot = dot_source(plan)
        assert "digraph salt_core_fit {" in dot
        # one HTML-like signature card per module
        assert '"embed" [label=<' in dot
        assert "<TABLE" in dot
        # in/out sections list the consumed and produced keys with shapes
        assert "<I>in</I>" in dot
        assert "<I>out</I>" in dot
        assert "embed.x" in dot
        assert "(B, 16)" in dot
        assert "losses.total" in dot
        # deduped node -> node edges (no per-key edge labels)
        assert '"<sources>" -> "source";' in dot
        assert '"embed" -> "head";' in dot
        assert '"head" -> "<sinks>";' in dot
        # kind-coloured rows: labels orange, losses red, preds blue
        assert "#b5651d" in dot  # label kind
        assert "#c0392b" in dot  # loss kind
        assert "#1f6fb2" in dot  # preds kind

    def test_probed_shapes_override_declared(self, plan):
        dot = dot_source(plan, probed_shapes={"embed.x": (16, 16)})
        assert "(16, 16)" in dot  # probed concrete shape wins over (B, 16)

    def test_pruned_rendered_dashed(self, plan):
        dot = dot_source(plan, modules={}, pruned=["aux"])
        assert "aux (pruned)" in dot
        assert "style=dashed" in dot
        assert "color=grey" in dot


class TestRenderGraph:
    @pytest.mark.parametrize("suffix", ["svg", "png"])
    def test_writes_image(self, plan, tmp_path, suffix):
        out = render_graph(plan, tmp_path / f"graph.{suffix}")
        assert out.stat().st_size > 0

    def test_title_and_pruned_footnote(self, plan, tmp_path):
        out = render_graph(plan, tmp_path / "graph.svg", title="my title", pruned=["aux"])
        text = out.read_text()
        assert "my title" in text
        assert "demand-pruned in FIT: aux" in text

    def test_creates_parent_dirs(self, plan, tmp_path):
        out = render_graph(plan, tmp_path / "a" / "b" / "graph.svg")
        assert out.exists()


class TestProbeShapes:
    """The one-batch SHAPE PROBE, synthetic path (no data file, design §4.3)."""

    @pytest.fixture(scope="class")
    def probed(self):
        # SYNTHETIC: no data file — the probe synthesises a batch from the
        # reader schema of the in-repo test-scale GN2v2 config.
        return probe_shapes(_DUMMY_CFG, _NORM_PLACEHOLDER, SYNTHETIC, Mode.FIT)

    def test_inputs_have_concrete_shapes(self, probed):
        # B == 16; tracks carry the 19 configured features over 40 synthetic tokens
        assert probed["inputs.jets"] == (16, 2)
        assert probed["inputs.tracks"] == (16, 40, 19)

    def test_embed_has_concrete_shape(self, probed):
        # the model-side embedding crossed the torch boundary with a real shape
        assert probed["embed.tracks"][:2] == (16, 40)
        assert len(probed["embed.tracks"]) == 3  # [B, T, embed_dim]

    def test_labels_populated_with_concrete_shapes(self, probed):
        # the narrowed task labels loaded off the synthetic batch
        assert probed["labels.jets.flavour_label"] == (16,)
        assert probed["labels.tracks.ftagTruthOriginLabel"] == (16, 40)

    def test_preds_have_concrete_shapes(self, probed):
        # forward ran end-to-end through the heads
        assert probed["preds.jets.jets_classification"][0] == 16
        assert probed["preds.tracks.track_origin"][:2] == (16, 40)

    def test_none_is_synthetic(self):
        # data_file=None is the same SYNTHETIC path as the sentinel
        shapes = probe_shapes(_DUMMY_CFG, _NORM_PLACEHOLDER, None, Mode.FIT)
        assert shapes["inputs.tracks"] == (16, 40, 19)

    def test_probed_shapes_feed_dot_source(self, plan, probed):
        # the probe output drops straight into the renderer as concrete shapes,
        # overriding the symbolic declared shape on every matching port row
        dot = dot_source(plan, probed_shapes={"embed.x": tuple(probed["inputs.tracks"][:2])})
        assert "(16, 40)" in dot
