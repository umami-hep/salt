"""Compile-regression gate: the graph-break inventory of a compiled salt model.

`--compile` compiles every graph module in place (`SaltModule._apply_compile`),
so the unit that matters is one module's forward. This module replays a compiled
plan module-by-module under `torch._dynamo.explain` with the eager backend and
compares the graph breaks it finds against a checked-in ALLOWLIST. A break at a
site nobody signed off on fails the gate.

CPU-only and inductor-free (eager backend, tiny fixture models) — runs
unconditionally in the unit-test job.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from salt.graph import Bundle, Mode
from salt.graph.executor import canonical_produced
from salt.graph.planner import Plan
from salt.model.modules import bind_all, materialise_all, resolve_bind_schema
from salt.model.modules.tasks import RegressionTaskModule
from salt.tests._fixtures.gn2v2_fixture import (
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_batch,
    make_gn2_labels,
    write_parity_norm_dict,
)
from salt.tests._fixtures.v2_builders import (
    build_regression_modules,
    compile_regression,
    make_regression_labels,
)

B, T = 6, 10

# ---------------------------------------------------------------------------
# ALLOWLISTS — the signed-off graph breaks, per fixture config.
#
# A site is "<path relative to the repo root>:<function>", NOT file:line: line
# numbers churn on every unrelated edit above them, which would make this gate
# a nuisance rather than a ratchet. The value is how many breaks that site is
# allowed to take. More than that, or any site at all that is not listed here,
# fails. Fewer means the entry is stale and should be pruned — that warns.
#
# Every entry needs a reason. "It was already there" is not one.
# ---------------------------------------------------------------------------

# GN2v2 (norm -> stream embed -> concat -> encoder -> split -> pool -> three
# task heads -> loss). torch-math attention: the flash-varlen unpad/repad seam
# does not appear (there is no flash-attn on a CPU box, and the GPU seam has its
# own test in salt/tests/unit/utils/test_tensor_utils.py).
#
# Empty. Every module either captures whole or, in the vertexing head's case,
# captures nothing at all — see NO_GRAPH below, which is the other half of this
# gate.
GN2V2_ALLOWLIST: dict[str, int] = {}

# The DiPS-shaped regression plan (no encoder, no vertexing).
REGRESSION_ALLOWLIST: dict[str, int] = {}

# Modules dynamo captures NO graph for. A break-free module is not automatically
# a compiled module: if a forward's whole body is a `torch.compiler.disable`d
# call, dynamo emits no graph and reports no break either. That is a legitimate
# outcome exactly once here, and silence is not the way to record it.
GN2V2_NO_GRAPH: dict[str, str] = {
    "track_vertexing": (
        "VertexingTaskModule.forward reads its bundle leaves and hands everything to the "
        "torch.compiler.disable'd head_forward, so there is no tensor work left to capture. "
        "The head is uncapturable by construction: it compresses a [B, N, N] adjacency to one "
        "row per valid edge, which makes both the allocation size and the indices "
        "data-dependent, and the boolean compressions lower to aten.nonzero, which inductor "
        "refuses on CUDA."
    ),
}
REGRESSION_NO_GRAPH: dict[str, str] = {}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _site(frame) -> str:
    """``<repo-relative path>:<function>`` for a dynamo user-stack frame.

    Cut at the last ``/salt/`` so the key is the same whether salt is an
    editable checkout, a worktree under a directory also called ``salt``, or an
    installed package.

    Returns
    -------
    str
        The site key used by the allowlists.
    """
    text = Path(frame.filename).resolve().as_posix()
    index = text.rfind("/salt/")
    rel = text[index + 1 :] if index != -1 else text
    return f"{rel}:{frame.name}"


def _breaks_of(module_name: str, explanation) -> list[tuple[str, str]]:
    """Extract ``(site, description)`` for every graph break dynamo reported.

    Returns
    -------
    list[tuple[str, str]]
        One entry per break, in report order.
    """
    reasons = getattr(explanation, "break_reasons", None)
    assert reasons is not None, (
        "torch._dynamo.explain no longer exposes `break_reasons` — this gate reads the "
        "dynamo explain API directly and needs updating for this torch version"
    )
    found = []
    for reason in reasons:
        stack = list(getattr(reason, "user_stack", None) or [])
        frame = stack[-1] if stack else None
        site = _site(frame) if frame is not None else "<unknown>:<unknown>"
        where = f"{Path(frame.filename).name}:{frame.lineno}" if frame is not None else "?"
        text = " ".join(str(getattr(reason, "reason", "")).split())[:200]
        found.append((site, f"[{module_name}] {where} — {text}"))
    return found


def _forward_steps(plan: Plan) -> list:
    """Plan steps that are actually called as a tensor forward (sinks excluded).

    Returns
    -------
    list
        The forward `PlanStep`s in plan order.
    """
    steps = []
    for step in plan.steps:
        is_sink = getattr(step.module, "is_sink", None)
        if callable(is_sink) and is_sink():
            continue
        steps.append(step)
    return steps


def _advance(step, bundle: Bundle, mode: Mode) -> None:
    """Run one plan step eagerly and merge its produces into `bundle`."""
    produced = step.module(bundle, mode)
    expected = set(step.produces)
    bundle.merge(
        canonical_produced(produced, expected, step.name), who=step.name, expected=expected
    )


def break_inventory(plan: Plan, bundle: Bundle) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Graph breaks and captured-graph counts per module of `plan`, in plan order.

    Mirrors production: `--compile` compiles each graph module in place, so each
    module's forward is traced as its own dynamo entry point. Every module is
    traced under `explain` and then re-run eagerly, so the next module sees a
    fully populated bundle.

    Returns
    -------
    tuple[list[tuple[str, str]], dict[str, int]]
        The ``(site, description)`` breaks, and ``module name -> graph count``.
    """
    inventory: list[tuple[str, str]] = []
    graphs: dict[str, int] = {}
    for step in _forward_steps(plan):
        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        explanation = torch._dynamo.explain(step.module.forward)(bundle, plan.mode)  # noqa: SLF001
        inventory.extend(_breaks_of(step.name, explanation))
        graphs[step.name] = explanation.graph_count
        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        _advance(step, bundle, plan.mode)
    return inventory, graphs


def assert_allowlisted(inventory: list[tuple[str, str]], allowlist: dict[str, int]) -> None:
    """Fail on any break the allowlist does not cover; warn on stale entries."""
    counts: dict[str, int] = {}
    details: dict[str, list[str]] = {}
    for site, description in inventory:
        counts[site] = counts.get(site, 0) + 1
        details.setdefault(site, []).append(description)

    offenders = [
        f"  {site} — {counts[site]} break(s), allowed {allowlist.get(site, 0)}\n"
        + "\n".join(f"      {d}" for d in details[site])
        for site in sorted(counts)
        if counts[site] > allowlist.get(site, 0)
    ]
    assert not offenders, (
        "un-allowlisted graph break(s) under torch.compile:\n"
        + "\n".join(offenders)
        + "\n\nEither fix the break, or add the site to the allowlist in "
        "salt/tests/unit/model/test_compile_regression.py WITH a reason."
    )

    if stale := sorted(site for site, budget in allowlist.items() if counts.get(site, 0) < budget):
        warnings.warn(
            f"compile-regression allowlist entries no longer fire (prune them): {stale}",
            stacklevel=2,
        )


def assert_graphs_captured(graphs: dict[str, int], no_graph: dict[str, str]) -> None:
    """Every module must capture at least one graph, except the listed seams."""
    silent = sorted(name for name, count in graphs.items() if count == 0)
    unexpected = [name for name in silent if name not in no_graph]
    assert not unexpected, (
        f"module(s) {unexpected} captured NO graph at all — dynamo had nothing to compile. "
        "A break-free module is not the same as a compiled module: a forward whose whole body "
        "is a torch.compiler.disable'd call reports zero breaks AND zero graphs. Either give "
        "it something to capture, or record it in the NO_GRAPH map WITH a reason.\n"
        f"captured graphs per module: {graphs}"
    )
    if stale := sorted(set(no_graph) - set(silent)):
        warnings.warn(f"NO_GRAPH entries now capture a graph (prune them): {stale}", stacklevel=2)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def norm_dict_path(tmp_path):
    """Write the parity norm dict.

    Returns
    -------
    Path
        The norm-dict path.
    """
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


@pytest.fixture
def gn2v2_fit(norm_dict_path):
    """Build the GN2v2 fixture.

    Returns
    -------
    tuple
        The bound, materialised module dict and its FIT plan.
    """
    modules = build_gn2v2_modules(norm_dict_path)
    plan = compile_gn2v2(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema(plan))
    materialise_all(modules)
    return modules, plan


def gn2v2_bundle() -> Bundle:
    """A FIT bundle for the GN2v2 fixture (inputs + masks + labels).

    Returns
    -------
    Bundle
        The populated source bundle.
    """
    inputs, masks = make_gn2_batch(B, T)
    bundle = Bundle()
    for stream, x in inputs.items():
        bundle.set(f"inputs.{stream}", x)
    for stream, mask in masks.items():
        bundle.set(f"masks.{stream}", mask)
    for stream, fields in make_gn2_labels(B, T).items():
        for name, value in fields.items():
            bundle.set(f"labels.{stream}.{name}", value)
    return bundle


REG_TARGETS = ("R10TruthLabel_R22v1_TruthJetPt",)


@pytest.fixture
def regression_fit(norm_dict_path):
    """Build the regression fixture.

    Returns
    -------
    tuple
        The bound, materialised module dict and its FIT plan.
    """
    task = RegressionTaskModule(
        stream="jets",
        targets=list(REG_TARGETS),
        input="pooled.global",
        loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
    )
    task.name = "regression"
    modules = build_regression_modules(norm_dict_path, task)
    plan = compile_regression(modules, Mode.FIT, REG_TARGETS)
    bind_all(modules, resolve_bind_schema(plan))
    materialise_all(modules)
    return modules, plan


def regression_bundle() -> Bundle:
    """A FIT bundle for the regression fixture.

    Returns
    -------
    Bundle
        The populated source bundle.
    """
    inputs, masks = make_gn2_batch(B, T)
    bundle = Bundle()
    for stream, x in inputs.items():
        bundle.set(f"inputs.{stream}", x)
    bundle.set("masks.tracks", masks["tracks"])
    for key, value in make_regression_labels(B, REG_TARGETS).items():
        bundle.set(key, value)
    return bundle


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------


class TestGraphBreakAllowlist:
    """Only signed-off graph breaks may survive."""

    def test_gn2v2_fit_breaks_are_allowlisted(self, gn2v2_fit):
        _, plan = gn2v2_fit
        inventory, _ = break_inventory(plan, gn2v2_bundle())
        assert_allowlisted(inventory, GN2V2_ALLOWLIST)

    def test_regression_fit_breaks_are_allowlisted(self, regression_fit):
        _, plan = regression_fit
        inventory, _ = break_inventory(plan, regression_bundle())
        assert_allowlisted(inventory, REGRESSION_ALLOWLIST)

    def test_gn2v2_total_break_count_is_the_allowlist_budget(self, gn2v2_fit):
        """The count, not just the sites — a second break at an allowed site is new too."""
        _, plan = gn2v2_fit
        inventory, _ = break_inventory(plan, gn2v2_bundle())
        assert len(inventory) == sum(GN2V2_ALLOWLIST.values())

    def test_gn2v2_every_module_captures_a_graph(self, gn2v2_fit):
        """Zero breaks is only good news if there is also a graph."""
        _, plan = gn2v2_fit
        _, graphs = break_inventory(plan, gn2v2_bundle())
        assert_graphs_captured(graphs, GN2V2_NO_GRAPH)

    def test_regression_every_module_captures_a_graph(self, regression_fit):
        _, plan = regression_fit
        _, graphs = break_inventory(plan, regression_bundle())
        assert_graphs_captured(graphs, REGRESSION_NO_GRAPH)

    def test_the_gate_catches_a_deliberate_new_break(self, gn2v2_fit, monkeypatch):
        """Poison one module with an explicit break; the gate must reject it."""
        modules, plan = gn2v2_fit
        pool_type = type(modules["pool"])
        original = pool_type.forward

        def poisoned(self, b, mode):
            out = original(self, b, mode)
            torch._dynamo.graph_break()  # noqa: SLF001 - the dynamo test surface
            return {key: value * 1.0 for key, value in out.items()}

        monkeypatch.setattr(pool_type, "forward", poisoned)
        inventory, _ = break_inventory(plan, gn2v2_bundle())
        with pytest.raises(AssertionError, match="un-allowlisted graph break"):
            assert_allowlisted(inventory, GN2V2_ALLOWLIST)


class TestEncoderFullGraph:
    """The encoder is the compiled hot path: it must capture whole, no allowlist."""

    def test_encoder_compiles_with_fullgraph(self, gn2v2_fit):
        _, plan = gn2v2_fit
        bundle = gn2v2_bundle()
        encoder_step = None
        for step in _forward_steps(plan):
            if step.name == "encoder":
                encoder_step = step
                break
            _advance(step, bundle, plan.mode)
        assert encoder_step is not None, "the GN2v2 fixture plan has no encoder step"

        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        compiled = torch.compile(encoder_step.module.forward, backend="eager", fullgraph=True)
        out = compiled(bundle, plan.mode)
        expected = encoder_step.module(bundle, plan.mode)
        assert set(out) == set(expected)
        for key, value in expected.items():
            assert torch.equal(out[key], value)
