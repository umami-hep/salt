"""Plan rendering: §4.4 plan tables, Graphviz DOT, and the matplotlib DAG plot.

One shared home for the three renderings of a compiled `Plan`, consumed by
both the static graph tooling (``salt2 graph plan/plot``, `salt.core.cli`)
and the run-dir artifact callback (`salt.core.callbacks.GraphArtifacts`,
design §4.4):

- `plan_table` — the ordered §4.4 step table (binding constraints + narrowed
  wildcard results), byte-identical to the historical ``salt2 graph plan``
  stdout so ``plan_<mode>.txt`` artifacts and the CLI agree.
- `dot_source` — Graphviz DOT text (design §4.3 styling). Emitted alongside
  every image render; the ``dot`` binary is NOT assumed to exist.
- `render_graph` — the PRIMARY image renderer: a layered topological DAG
  drawn with matplotlib (always available in the salt container, unlike
  graphviz). Layout ported from the architecture-investigation
  ``plot_graph.py``: x = longest-path layer, namespace-coloured module
  boxes, staggered edge-key labels, dataset-boundary/sink ellipses.

matplotlib is imported lazily inside `render_graph` (and never via pyplot —
`matplotlib.figure.Figure` keeps the rendering backend-free), so importing
this module stays cheap for the table/DOT paths.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from salt.core.graph.planner import SINKS, SOURCES, Plan
from salt.core.graph.spec import KEY_SEP, TensorSpec, flatten_spec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from salt.core.graph.planner import PlanStep
    from salt.core.graph.spec import GraphModule

__all__ = ["dot_source", "plan_table", "render_graph"]

_WILDCARD_PARTS = frozenset({"*", "**"})


def _has_wildcard(key: str) -> bool:
    """Check whether a dotted key contains a wildcard component (design §2.2).

    Returns
    -------
    bool
        True if any component is ``"*"`` or ``"**"``.
    """
    return any(part in _WILDCARD_PARTS for part in key.split(KEY_SEP))


# ---------------------------------------------------------------------------
# §4.4 plan table (the salt2 graph plan stdout == plan_<mode>.txt artifact)
# ---------------------------------------------------------------------------


def plan_table(plan: Plan) -> str:
    """Format the ordered §4.4 plan table for one compiled plan.

    Shows each step's binding constraint (the latest predecessor forced by a
    key edge) and the narrowed wildcard results — including the full label
    list a dataset plan will load (design §3.1 debugging story, §4.4).

    Returns
    -------
    str
        The multi-line table (no trailing newline).
    """
    mode = plan.mode
    lines = [f"plan [mode={mode.name}] {len(plan.steps)} steps  plan_hash={plan.plan_hash}"]
    index = {step.name: i for i, step in enumerate(plan.steps)}
    for i, step in enumerate(plan.steps, start=1):
        binding = [
            (index[edge.producer], edge.key)
            for edge in plan.edges
            if edge.consumer == step.name and edge.producer in index
        ]
        if binding:
            latest, _ = max(binding)
            after = plan.steps[latest].name
        elif any(e.consumer == step.name and e.producer == SOURCES for e in plan.edges):
            after = SOURCES
        else:
            after = "(no inputs)"
        needs = ", ".join(sorted(step.requires)) or "nothing"
        # a wildcard producer narrowed to nothing (e.g. Labels in TEST/ONNX)
        # runs as a no-op and reads no fields — say so, or the step looks
        # like live label loading in a plan that provably has none (§4.2)
        noop = "" if step.produces else "  [narrowed to 0 keys — no-op]"
        lines.append(f"  {i:2d}. {step.name:<20} after {after:<20} (needs {needs}){noop}")
    narrowed_lines = []
    for step in plan.steps:
        declared = step.module.declare_io(mode).produces
        concrete = {key for key in flatten_spec(declared) if not _has_wildcard(key)}
        if extra := sorted(set(step.produces) - concrete):
            narrowed_lines.append(f"  {step.name}: {', '.join(extra)}")
    if narrowed_lines:
        lines.append(f"narrowed wildcards [mode={mode.name}]:")
        lines.extend(narrowed_lines)
    lines.append(f"sources: {', '.join(sorted(plan.sources)) or '(none)'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT (design §4.3) — emitted alongside every image render
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Escape a string for a double-quoted DOT identifier.

    Returns
    -------
    str
        The escaped text (without surrounding quotes).
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _quote(text: str) -> str:
    """Quote a string as a DOT identifier.

    Returns
    -------
    str
        The double-quoted, escaped identifier.
    """
    return f'"{_esc(text)}"'


def _label(*lines: str) -> str:
    r"""Build a quoted multi-line DOT label.

    Returns
    -------
    str
        The quoted label with ``\n`` separators.
    """
    return '"' + "\\n".join(_esc(line) for line in lines) + '"'


def _edge_style(key: str, spec: TensorSpec) -> list[str]:
    """Kind-based DOT edge styling (design §4.3).

    Labels orange, losses red, pad-masks dotted grey, ``preds.*`` blue.

    Returns
    -------
    list[str]
        Extra DOT edge attributes.
    """
    if spec.kind == "label":
        return ["color=orange", "fontcolor=orange"]
    if spec.kind == "loss":
        return ["color=red", "fontcolor=red"]
    if spec.kind == "pad_mask":
        return ["color=grey", "fontcolor=grey", "style=dotted"]
    if spec.kind == "meta":
        return ["color=grey", "fontcolor=grey"]
    if key.partition(KEY_SEP)[0] == "preds":
        return ["color=blue", "fontcolor=blue"]
    return []


def dot_source(
    plan: Plan,
    modules: Mapping[str, GraphModule] | None = None,
    pruned: Iterable[str] = (),
) -> str:
    r"""Render a compiled plan as Graphviz DOT text (design §4.3).

    Nodes are module instances (``name\nClassName``); edges are bundle keys
    with kind-based styling; demand-pruned modules (looked up in `modules`)
    render grey-dashed.

    Returns
    -------
    str
        The DOT source.
    """
    modules = dict(modules or {})
    lines = [
        f"digraph salt_core_{plan.mode.name.lower()} {{",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Helvetica"];',
    ]
    if any(edge.producer == SOURCES for edge in plan.edges):
        lines.append(f"  {_quote(SOURCES)} [shape=ellipse, style=dashed];")
    if any(edge.consumer == SINKS for edge in plan.edges):
        lines.append(f"  {_quote(SINKS)} [shape=ellipse, style=dashed];")
    lines.extend(
        f"  {_quote(step.name)} [label={_label(step.name, type(step.module).__name__)}];"
        for step in plan.steps
    )
    for name in pruned:
        cls = type(modules[name]).__name__ if name in modules else "?"
        label = _label(name, cls, "(pruned)")
        lines.append(f"  {_quote(name)} [label={label}, style=dashed, color=grey, fontcolor=grey];")
    for edge in plan.edges:
        if edge.producer == SOURCES:
            spec = plan.sources[edge.key]
        else:
            spec = plan.step(edge.producer).produces[edge.key]
        attrs = [f"label={_quote(edge.key)}", *_edge_style(edge.key, spec)]
        lines.append(f"  {_quote(edge.producer)} -> {_quote(edge.consumer)} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# matplotlib layered DAG (the PRIMARY image renderer — graphviz-free)
# ---------------------------------------------------------------------------

# module box fill by (first matching) produced namespace — the §4.3 styling
# translated to the layered plot (investigation plot_graph.py palette)
_NS_COLOURS = {
    "raw": "#b5d99c",
    "masks": "#b5d99c",
    "meta": "#d9d9d9",
    "inputs": "#7fbf7b",
    "labels": "#4daf4a",
    "normed": "#9ecae1",
    "embed": "#6baed6",
    "seq": "#6baed6",
    "encoded": "#3182bd",
    "pooled": "#3182bd",
    "preds": "#fdae6b",
    "losses": "#fb6a4a",
    "loss": "#de2d26",
}
_FALLBACK_COLOUR = "#cccccc"


def _module_colour(step: PlanStep) -> str:
    """Pick the box colour from the step's produced namespaces.

    Returns
    -------
    str
        A hex colour.
    """
    for namespace in sorted({key.split(KEY_SEP)[0] for key in step.produces}):
        if namespace in _NS_COLOURS:
            return _NS_COLOURS[namespace]
    return _FALLBACK_COLOUR


def _layered_layout(plan: Plan) -> dict[str, tuple[float, float]]:
    """Layered topological layout: x = longest-path layer, y = spread in layer.

    Returns
    -------
    dict[str, tuple[float, float]]
        Module name -> (x, y) position.
    """
    names = [step.name for step in plan.steps]
    preds: dict[str, set[str]] = defaultdict(set)
    for edge in plan.edges:
        if edge.producer in names and edge.consumer in names:
            preds[edge.consumer].add(edge.producer)
    layer: dict[str, int] = {}
    for name in names:  # steps are already topo-ordered
        layer[name] = 1 + max((layer[p] for p in preds[name] if p in layer), default=0)
    by_layer: dict[int, list[str]] = defaultdict(list)
    for name in names:
        by_layer[layer[name]].append(name)
    pos: dict[str, tuple[float, float]] = {}
    for x, members in by_layer.items():
        for i, name in enumerate(sorted(members)):
            pos[name] = (float(x) * 1.8, -(i - (len(members) - 1) / 2.0) * 2.0)
    return pos


def render_graph(
    plan: Plan,
    out_path: str | Path,
    title: str | None = None,
    pruned: Iterable[str] = (),
    dpi: int = 200,
) -> Path:
    """Render a compiled plan as a layered DAG image via matplotlib (design §4.3/§4.4).

    The primary renderer for ``salt2 graph plot`` and the run-dir
    ``graph_<mode>`` artifacts: matplotlib ships in the salt container,
    graphviz does not. Output format follows the file suffix (``.svg`` /
    ``.png`` / ``.pdf``). Demand-pruned module names are listed in a grey
    footnote rather than drawn (they have no plan edges to lay out).

    Parameters
    ----------
    plan : Plan
        The compiled plan to draw.
    out_path : str | Path
        Output image path (suffix selects the format).
    title : str | None, optional
        Figure title; defaults to mode + step count + plan-hash prefix.
    pruned : Iterable[str], optional
        Names of demand-pruned modules (footnote only), by default ().
    dpi : int, optional
        Raster DPI (PNG), by default 200.

    Returns
    -------
    Path
        The written image path.

    Raises
    ------
    ImportError
        When matplotlib is not installed (callers may fall back to DOT).
    """
    # lazy + pyplot-free: Figure needs no backend/global state (module docstring)
    try:
        from matplotlib.figure import Figure  # noqa: PLC0415 - lazy heavy import
        from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(f"render_graph needs matplotlib: {err}") from err

    out_path = Path(out_path)
    pos = _layered_layout(plan)
    steps = {step.name: step for step in plan.steps}
    names = set(pos)

    # boundary pseudo-nodes: framework sources feeding in, sink demands out
    boundary_in: dict[str, set[str]] = defaultdict(set)
    boundary_out: dict[str, set[str]] = defaultdict(set)
    internal: list[tuple[str, str, str]] = []
    for edge in plan.edges:
        p_in, c_in = edge.producer in names, edge.consumer in names
        if p_in and c_in:
            internal.append((edge.producer, edge.consumer, edge.key))
        elif c_in:
            boundary_in[edge.key].add(edge.consumer)
        elif p_in:
            boundary_out[edge.key].add(edge.producer)

    xs = [x for x, _ in pos.values()] or [0.0]
    ys = [y for _, y in pos.values()] or [0.0]
    min_x, max_x = min(xs), max(xs)
    yrange = max(ys) - min(ys)
    fig_w = min(30.0, max(10.0, 1.15 * (max_x - min_x) + 7.0))
    fig_h = max(6.5, 0.95 * yrange + 3.0)
    fig = Figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot()

    # place boundary ellipses left (sources) / right (sinks)
    bpos: dict[tuple[str, str], tuple[float, float]] = {}
    for i, key in enumerate(sorted(boundary_in)):
        bpos["in", key] = (min_x - 2.0, -(i - (len(boundary_in) - 1) / 2.0) * 1.3)
    for i, key in enumerate(sorted(boundary_out)):
        bpos["out", key] = (max_x + 2.0, -(i - (len(boundary_out) - 1) / 2.0) * 1.3)

    # group multi-key edges between one module pair; stagger labels along the
    # edge (cycled fraction t, alternating perpendicular offset) so labels
    # from one source never stack
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for src, dst, key in internal:
        grouped[src, dst].append(key)
    per_src_idx: dict[str, int] = defaultdict(int)
    for (src, dst), keys in sorted(grouped.items()):
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.46, y0),
                (x1 - 0.46, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.1,
                color="#555555",
                connectionstyle="arc3,rad=0.08",
                zorder=1,
            )
        )
        i = per_src_idx[src]
        per_src_idx[src] += 1
        t = (0.35, 0.58, 0.78)[i % 3]
        sign = 1 if i % 2 == 0 else -1
        ax.text(
            x0 + t * (x1 - x0),
            y0 + t * (y1 - y0) + sign * (0.22 + 0.05 * len(keys)),
            "\n".join(sorted(keys)),
            fontsize=7.6,
            ha="center",
            va="bottom" if sign > 0 else "top",
            color="#333333",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "#bbbbbb",
                "lw": 0.5,
                "alpha": 0.9,
            },
        )

    for (tag, key), (bx, by) in bpos.items():
        for module in sorted(boundary_in[key] if tag == "in" else boundary_out[key]):
            mx, my = pos[module]
            start, end = (
                ((bx + 0.75, by), (mx - 0.46, my))
                if tag == "in"
                else (
                    (mx + 0.46, my),
                    (bx - 0.75, by),
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=11,
                    lw=0.9,
                    color="#999999",
                    connectionstyle="arc3,rad=0.06",
                    zorder=1,
                )
            )
        ax.add_patch(Ellipse((bx, by), 1.5, 0.55, fc="#f2f2f2", ec="#888888", lw=1.0, zorder=2))
        ax.text(bx, by, key, fontsize=7.0, ha="center", va="center", style="italic", zorder=3)

    for name, (x, y) in pos.items():
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.46, y - 0.28),
                0.92,
                0.56,
                boxstyle="round,pad=0.06,rounding_size=0.10",
                fc=_module_colour(steps[name]),
                ec="#222222",
                lw=1.2,
                zorder=2,
            )
        )
        ax.text(x, y, name, fontsize=9.0, ha="center", va="center", weight="bold", zorder=3)

    all_x = [x for x, _ in (*pos.values(), *bpos.values())]
    all_y = [y for _, y in (*pos.values(), *bpos.values())]
    ax.set_xlim(min(all_x) - 1.4, max(all_x) + 1.4)
    ax.set_ylim(min(all_y) - 1.2, max(all_y) + 1.2)
    if title is None:
        title = (
            f"salt2 graph [mode={plan.mode.name}] — {len(plan.steps)} steps, "
            f"hash {plan.plan_hash[:10]}"
        )
    ax.set_title(title, fontsize=12)
    if pruned_list := sorted(pruned):
        ax.text(
            0.01,
            0.01,
            f"demand-pruned in {plan.mode.name}: {', '.join(pruned_list)}",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#888888",
            style="italic",
        )
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return out_path
