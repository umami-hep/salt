"""Tests for `salt.core.render`: plan table, DOT source, matplotlib DAG renderer."""

from __future__ import annotations

import pytest

from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import Mode, TensorSpec, unflatten_spec
from salt.core.render import dot_source, plan_table, render_graph
from salt.tests.core.toys import ToyEmbed, ToyHead, ToySource, ToyWildcardLabels


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
    def test_nodes_edges_and_styling(self, plan):
        dot = dot_source(plan)
        assert "digraph salt_core_fit {" in dot
        assert '"embed"' in dot
        assert '"<sources>" -> "source"' in dot
        assert '"embed" -> "head" [label="embed.x (B, 16)"' in dot
        assert "color=orange" in dot  # label kind

    def test_pruned_rendered_dashed(self, plan):
        dot = dot_source(plan, modules={}, pruned=["aux"])
        assert "(pruned)" in dot
        assert "style=dashed" in dot


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
