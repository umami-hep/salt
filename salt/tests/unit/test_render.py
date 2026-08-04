"""Tests for `salt.graph.render`: plan table, DOT source, and the CLI dot render."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from salt.cli import load_config
from salt.cli import main as cli_main
from salt.graph.planner import compile_plan
from salt.graph.spec import PRIMARY_MODES, Mode, TensorSpec, unflatten_spec
from salt.model.bind import resolve_bind_schema
from salt.graph.render import _shape_str, dot_source, plan_table  # noqa: PLC2701
from salt.tests._fixtures.toys import ToyEmbed, ToyHead, ToySource, ToyWildcardLabels

# the in-repo test-scale GN2v2 config (16-dim, no machine paths) — the static
# width resolution needs no data file, only a placeholder norm_dict
_DUMMY_CFG = str(Path(__file__).parent.parent.parent / "configs" / "gn2v2-opendata.yaml")
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

    def test_concrete_feature_dim_unchanged_by_width(self, plan):
        # the toy plan's embed.x already declares a CONCRETE last dim (B, 16);
        # a resolved width for it is a no-op (the width agrees with the int) and,
        # crucially, never rewrites the already-concrete dim into something else
        dot = dot_source(plan, widths={"embed.x": 99})
        assert "(B, 16)" in dot
        assert "99" not in dot

    def test_pruned_rendered_dashed(self, plan):
        dot = dot_source(plan, modules={}, pruned=["aux"])
        assert "aux (pruned)" in dot
        assert "style=dashed" in dot
        assert "color=grey" in dot


_HAS_DOT = shutil.which("dot") is not None


class TestPlotCli:
    """`salt graph plot` renders the DOT to PNG+PDF via the dot binary."""

    def test_dot_sidecar_written_even_without_image(self, tmp_path):
        # a .dot target short-circuits before any dot invocation: the DOT
        # sidecar is the requested output and no image is produced
        out = tmp_path / "graph.dot"
        rc = cli_main(
            ["graph", "plot", "-c", _DUMMY_CFG, "--mode", "fit",
             "-o", str(out), "--set", _NORM_PLACEHOLDER[0]]
        )
        assert rc == 0
        assert out.exists()
        assert "digraph" in out.read_text()

    @pytest.mark.skipif(not _HAS_DOT, reason="graphviz `dot` binary not on PATH")
    def test_png_pdf_and_dot_emitted(self, tmp_path):
        # the full image path: PNG at the requested -o, a sibling PDF, and the
        # .dot sidecar — all via the in-container dot binary
        out = tmp_path / "graph.png"
        rc = cli_main(
            ["graph", "plot", "-c", _DUMMY_CFG, "--mode", "fit",
             "-o", str(out), "--set", _NORM_PLACEHOLDER[0]]
        )
        assert rc == 0
        png = out
        pdf = out.with_suffix(".pdf")
        dot = out.with_suffix(".dot")
        assert png.stat().st_size > 0
        assert pdf.stat().st_size > 0
        assert dot.exists()
        # PNG magic bytes / PDF header confirm dot actually rasterised
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert pdf.read_bytes()[:5] == b"%PDF-"


class TestShapeStrWidthSubstitution:
    """`_shape_str` substitutes the SYMBOLIC FEATURE dim, keeps data dims."""

    def test_symbolic_feature_dim_becomes_concrete(self):
        # E:enc is a feature family — the resolved width replaces it as the LAST
        # dim; the leading B and the sequence dim T:tracks stay symbolic
        spec = TensorSpec(shape=("B", "T:tracks", "E:enc"))
        assert _shape_str("encoded.tracks", spec, {"encoded.tracks": 16}) == "(B, T:tracks, 16)"

    def test_last_dim_data_family_stays_symbolic(self):
        # when the LAST dim is itself a data family (T/L/S), a width must NOT
        # turn it concrete — that would freeze a genuinely data-dependent dim
        spec = TensorSpec(shape=("B", "T:tracks"))
        assert _shape_str("labels.tracks.origin", spec, {"labels.tracks.origin": 100}) == (
            "(B, T:tracks)"
        )

    def test_batch_only_stays_symbolic(self):
        # a width on a batch-only key must NOT turn B into a concrete int (the
        # renderer joins a single dim with no trailing comma, like _fmt_shape)
        spec = TensorSpec(shape=("B",))
        assert _shape_str("labels.jets.flavour", spec, {"labels.jets.flavour": 3}) == "(B)"

    def test_no_width_keeps_declared_symbolic(self):
        spec = TensorSpec(shape=("B", "F:norm"))
        assert _shape_str("normed.jets", spec, None) == "(B, F:norm)"
        assert _shape_str("normed.jets", spec, {}) == "(B, F:norm)"

    def test_concrete_last_dim_left_alone(self):
        # an already-concrete feature dim is not a symbolic feature dim, so it is
        # untouched (the resolved width agrees with the declared int anyway)
        spec = TensorSpec(shape=("B", 9))
        assert _shape_str("preds.jets.cls", spec, {"preds.jets.cls": 9}) == "(B, 9)"

    def test_scalar_and_none_have_no_shape(self):
        assert not _shape_str("losses.total", TensorSpec(shape=()), {"losses.total": 1})
        assert not _shape_str("embed.x", TensorSpec(shape=None), {"embed.x": 16})


def _static_widths(cfg):
    """Resolve widths data-free across every compilable primary mode (no batch run)."""
    plans = []
    for plan_mode in PRIMARY_MODES:
        if plan_mode in cfg.mode_errors:
            continue
        try:
            plans.append(
                compile_plan(
                    cfg.modules, plan_mode, cfg.sources, schema=cfg.schema,
                    sinks=cfg.sinks, sink_origins=cfg.sink_origins.get(plan_mode),
                )
            )
        except Exception:  # noqa: BLE001, S112 - non-requested modes may not compile
            continue
    return dict(resolve_bind_schema(plans).widths)


class TestStaticWidthRender:
    """The DOT carries CONCRETE feature dims from the STATIC resolution — NO data."""

    @pytest.fixture(scope="class")
    def dot(self):
        # the real test-scale GN2v2 config, resolved with NO data file and NO
        # batch run — only the static bind schema (the widths that build the
        # nn.Linear layers) feeds the renderer.
        cfg = load_config([_DUMMY_CFG], _NORM_PLACEHOLDER)
        plan = compile_plan(
            cfg.modules, Mode.FIT, cfg.sources, schema=cfg.schema,
            sinks=cfg.sinks, sink_origins=cfg.sink_origins.get(Mode.FIT),
        )
        widths = _static_widths(cfg)
        return dot_source(plan, cfg.modules, widths=widths)

    def test_feature_dims_are_concrete(self, dot):
        # the normaliser / encoder / pool / concat feature widths render as the
        # resolved ints, NOT the symbolic F:/E:/D: dims
        assert "(B, T:tracks, 19)" in dot   # normed.tracks: 19 input features
        assert "(B, T:tracks, 16)" in dot   # encoded.tracks: 16 embed width
        assert "(B, 16)" in dot             # pooled.global: 16
        # no symbolic FEATURE dim leaked into the rendered shapes
        for leaked in ("F:norm", "E:concat", "D:pool", "D:split"):
            assert leaked not in dot

    def test_batch_and_sequence_dims_stay_symbolic(self, dot):
        # B and the sequence/token dims are genuinely data-dependent — symbolic
        assert "(B, T:tracks," in dot
        assert "L:enc" in dot
        assert "S:seq" in dot

    def test_render_is_data_free(self):
        # the whole `salt graph plot` path produces a DOT sidecar with NO data
        # file and NO --probe flag, and never emits the probe's labeller line
        cfg = load_config([_DUMMY_CFG], _NORM_PLACEHOLDER)
        widths = _static_widths(cfg)
        assert widths["encoded.tracks"] == 16
        assert widths["normed.tracks"] == 19


class TestPlotCliStaticWidths:
    """`salt graph plot` (no --probe) writes a DOT with concrete feature dims."""

    def test_dot_sidecar_has_concrete_feature_dims(self, tmp_path):
        out = tmp_path / "static.dot"
        rc = cli_main(
            ["graph", "plot", "-c", _DUMMY_CFG, "--mode", "fit",
             "-o", str(out), "--set", _NORM_PLACEHOLDER[0]]
        )
        assert rc == 0
        text = out.read_text()
        assert "(B, T:tracks, 19)" in text   # concrete feature dim, B/T symbolic
        assert "F:norm" not in text          # the symbolic feature dim is gone
