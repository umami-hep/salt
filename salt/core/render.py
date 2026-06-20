"""Plan rendering: §4.4 plan tables, Graphviz DOT, and the matplotlib DAG plot.

One shared home for the three renderings of a compiled `Plan`, consumed by
both the static graph tooling (``salt2 graph plan/plot``, `salt.core.cli`)
and the run-dir artifact callback (`salt.core.callbacks.GraphArtifacts`,
design §4.4):

- `plan_table` — the ordered §4.4 step table (binding constraints + narrowed
  wildcard results), byte-identical to the historical ``salt2 graph plan``
  stdout so ``plan_<mode>.txt`` artifacts and the CLI agree.
- `dot_source` — Graphviz DOT text (design §4.3 styling): the port-card
  layout, one HTML-like signature card per module (header + consumed/produced
  rows) and one deduped node->node arrow per producer/consumer pair. Emitted
  alongside every image render; the ``dot`` binary is NOT assumed to exist.
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


def _fmt_shape(spec: TensorSpec | None) -> str:
    """Format a spec's shape as ``"(d0, d1, ...)"`` for edge labels.

    Symbolic dims (``"B"``, ``"T:tracks"``) and concrete ints are joined as-is.
    A scalar/None-shape spec (and a missing spec) yields ``""`` so the edge
    keeps just its key rather than an empty ``"()"``.

    Returns
    -------
    str
        The parenthesised shape, or ``""`` when there is none.
    """
    # `not spec.shape` covers both None and the empty tuple () — a scalar
    # (e.g. a `losses.*` leaf) has no dims worth showing, so it keeps just its
    # key rather than a noisy "()".
    if spec is None or not spec.shape:
        return ""
    return "(" + ", ".join(str(dim) for dim in spec.shape) + ")"


def _edge_spec(plan: Plan, producer: str, key: str) -> TensorSpec | None:
    """Resolve the `TensorSpec` an edge carries (mirrors `dot_source`'s lookup).

    For a source edge (``producer == SOURCES``) the spec comes from
    `plan.sources`; otherwise from the producing step's `produces`. Uses
    ``.get`` so a missing key degrades to ``None`` rather than raising.

    Returns
    -------
    TensorSpec | None
        The resolved spec, or ``None`` if unavailable.
    """
    if producer == SOURCES:
        return plan.sources.get(key)
    return plan.step(producer).produces.get(key)


# Kind/key -> row font colour for the signature-card rows (design §4.3): the
# orange/red/blue accents the matplotlib renderer reserves for edges become row
# font colours here, since the port-card layout has no per-key wires to colour.
_KIND_COLOURS = {"label": "#b5651d", "loss": "#c0392b", "preds": "#1f6fb2"}
_ROW_DEFAULT_COLOUR = "#333333"
_SHAPE_COLOUR = "#888888"
_PRUNED_FILL = "#dddddd"


def _row_colour(key: str, spec: TensorSpec | None) -> str:
    """Kind-based row font colour for a signature-card row (design §4.3).

    Labels orange, losses red, ``preds.*`` blue; everything else the neutral
    default. Derived from the spec's `kind` (falling back to the key's leading
    namespace for ``preds.*``), mirroring the edge palette the matplotlib
    renderer uses.

    Returns
    -------
    str
        A hex colour.
    """
    if spec is not None and spec.kind in _KIND_COLOURS:
        return _KIND_COLOURS[spec.kind]
    if key.partition(KEY_SEP)[0] == "preds":
        return _KIND_COLOURS["preds"]
    return _ROW_DEFAULT_COLOUR


def _html_esc(text: str) -> str:
    """Escape a string for inclusion in a Graphviz HTML-like label.

    Only ``&``/``<``/``>`` are special inside an HTML-like label; quotes stay
    literal (the label is delimited by ``<...>``, not ``"..."``).

    Returns
    -------
    str
        The escaped text.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _shape_str(key: str, spec: TensorSpec | None, probed_shapes: Mapping[str, tuple] | None) -> str:
    """Shape string for a card row, preferring a probed concrete shape.

    When `probed_shapes` carries a (live-traced) shape for `key`, it is
    formatted like ``"(16, 40, 19)"``; otherwise the declared/symbolic shape
    from `spec` (`_fmt_shape`) is used. A scalar probed shape (``()``) stays
    bare, matching `_fmt_shape`'s scalar handling.

    Returns
    -------
    str
        The parenthesised shape, or ``""`` when there is none.
    """
    if probed_shapes is not None and key in probed_shapes:
        dims = tuple(probed_shapes[key])
        if not dims:
            return ""
        return "(" + ", ".join(str(dim) for dim in dims) + ")"
    return _fmt_shape(spec)


def _card_row(key: str, shape: str, colour: str, *, bold: bool) -> str:
    """One ``<TR>`` row of a signature card: kind-coloured key plus grey shape.

    Returns
    -------
    str
        The HTML-like table row.
    """
    name = f"<B>{_html_esc(key)}</B>" if bold else _html_esc(key)
    tail = f'  <FONT COLOR="{_SHAPE_COLOUR}">{_html_esc(shape)}</FONT>' if shape else ""
    return f'    <TR><TD ALIGN="LEFT"><FONT COLOR="{colour}">{name}{tail}</FONT></TD></TR>'


def _card_section(tag: str) -> str:
    """A faint italic ``in``/``out`` section divider row.

    Returns
    -------
    str
        The HTML-like table row.
    """
    return (
        '    <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#aaaaaa">'
        f"<I>{_html_esc(tag)}</I></FONT></TD></TR>"
    )


def _card_node(
    name: str,
    title: str,
    cls: str,
    fill: str,
    ins: list[tuple[str, str, str]],
    outs: list[tuple[str, str, str]],
) -> str:
    """Assemble one HTML-like signature-card node line.

    `ins`/`outs` are ``(key, shape, colour)`` triples for the consumed and
    produced rows; the header carries `title` + optional `cls` over a `fill`
    background.

    Returns
    -------
    str
        The full ``"name" [label=<<TABLE...>>];`` DOT node line.
    """
    sub = (
        f'<BR/><FONT POINT-SIZE="8" COLOR="#555555">{_html_esc(cls)}</FONT>'
        if cls
        else ""
    )
    rows = [
        f'    <TR><TD BGCOLOR="{fill}" ALIGN="CENTER"><B>{_html_esc(title)}</B>{sub}</TD></TR>'
    ]
    if ins:
        rows.append(_card_section("in"))
        rows.extend(_card_row(k, s, c, bold=False) for k, s, c in ins)
    if outs:
        rows.append(_card_section("out"))
        rows.extend(_card_row(k, s, c, bold=True) for k, s, c in outs)
    table = (
        '<\n    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">\n'
        + "\n".join(rows)
        + "\n    </TABLE>>"
    )
    return f"  {_quote(name)} [label={table}];"


def dot_source(
    plan: Plan,
    modules: Mapping[str, GraphModule] | None = None,
    pruned: Iterable[str] = (),
    probed_shapes: Mapping[str, tuple] | None = None,
) -> str:
    r"""Render a compiled plan as Graphviz DOT text — port-card layout (design §4.3).

    Each module renders as an HTML-like signature card: a colour-filled header
    (``name`` + ``ClassName``), an ``in`` section listing the keys the module
    CONSUMES, and an ``out`` section listing the keys it PRODUCES. Each row is a
    ``key`` plus its tensor shape, the key kind-coloured (labels orange, losses
    red, ``preds.*`` blue). The ``<sources>`` pseudo-node carries only an ``out``
    section (the framework boundary leaves); ``<sinks>`` only an ``in`` section.
    Edges collapse to ONE deduped ``producer -> consumer`` arrow per pair — the
    per-key detail lives in the cards, not on floating edge labels. Demand-pruned
    modules (looked up in `modules`) render as a grey dashed card.

    Parameters
    ----------
    plan : Plan
        The compiled plan to draw.
    modules : Mapping[str, GraphModule] | None, optional
        Module instances, used only to label `pruned` cards with their class.
    pruned : Iterable[str], optional
        Names of demand-pruned modules to draw grey/dashed, by default ().
    probed_shapes : Mapping[str, tuple] | None, optional
        Live-traced concrete shapes by dotted key; when a key is present its
        tuple is shown (e.g. ``"(16, 40, 19)"``) in preference to the
        declared/symbolic shape, by default None.

    Returns
    -------
    str
        The DOT source.
    """
    modules = dict(modules or {})
    # Graph styling (design §4.3): the validated port-card layout. plaintext
    # nodes carry HTML-like TABLE labels (the cards), so the shape/style/fill
    # live in the table, not the node attrs. splines=ortho draws clean right
    # angles between cards; the per-key detail is inside the cards (no floating
    # edge labels to detach), so ortho is safe here. Generous rank separation
    # keeps the LR rows of cards legible.
    lines = [
        f"digraph salt_core_{plan.mode.name.lower()} {{",
        "  rankdir=LR;",
        "  splines=ortho;",
        "  nodesep=0.6;",
        "  ranksep=1.5;",
        "  pad=0.4;",
        '  bgcolor="white";',
        '  node [shape=plaintext, fontname="Helvetica"];',
        '  edge [color="#777777", arrowsize=0.8, penwidth=1.3];',
    ]

    # consumed keys per module: each require edge into the module, with the
    # spec carried by the producing edge (mirrors the matplotlib renderer's
    # _edge_spec lookup). dict-insertion dedupes while preserving edge order.
    consumed: dict[str, dict[str, TensorSpec | None]] = {}
    for edge in plan.edges:
        consumed.setdefault(edge.consumer, {}).setdefault(
            edge.key, _edge_spec(plan, edge.producer, edge.key)
        )

    def _rows(items: Iterable[tuple[str, TensorSpec | None]]) -> list[tuple[str, str, str]]:
        return [
            (key, _shape_str(key, spec, probed_shapes), _row_colour(key, spec))
            for key, spec in items
        ]

    if any(edge.producer == SOURCES for edge in plan.edges):
        lines.append(
            _card_node(
                SOURCES, SOURCES, "", "#f5f5f5", [], _rows(sorted(plan.sources.items()))
            )
        )
    for step in plan.steps:
        ins = _rows(consumed.get(step.name, {}).items())
        outs = _rows(step.produces.items())
        cls = type(step.module).__name__
        lines.append(_card_node(step.name, step.name, cls, _module_colour(step), ins, outs))
    if any(edge.consumer == SINKS for edge in plan.edges):
        lines.append(
            _card_node(SINKS, SINKS, "", "#f5f5f5", _rows(consumed.get(SINKS, {}).items()), [])
        )
    for name in pruned:
        cls = type(modules[name]).__name__ if name in modules else "?"
        ins = _rows(consumed.get(name, {}).items())
        line = _card_node(name, f"{name} (pruned)", cls, _PRUNED_FILL, ins, [])
        # grey dashed border distinguishes a demand-pruned card from a live one
        lines.append(line.replace("];", ", style=dashed, color=grey];"))

    seen: set[tuple[str, str]] = set()
    for edge in plan.edges:
        pair = (edge.producer, edge.consumer)
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(f"  {_quote(edge.producer)} -> {_quote(edge.consumer)};")
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
    probed_shapes: Mapping[str, tuple] | None = None,
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
    probed_shapes : Mapping[str, tuple] | None, optional
        Live-traced concrete shapes by dotted key (the ``--probe`` result);
        when a key is present its tuple is shown in preference to the
        declared/symbolic shape, by default None.

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
        # each key paired with ITS OWN spec shape (never reuse one for the group);
        # a probed concrete shape (the --probe result) wins over the symbolic one
        label_lines = []
        for key in sorted(keys):
            shape = _shape_str(key, _edge_spec(plan, src, key), probed_shapes)
            label_lines.append(f"{key} {shape}" if shape else key)
        ax.text(
            x0 + t * (x1 - x0),
            y0 + t * (y1 - y0) + sign * (0.22 + 0.05 * len(keys)),
            "\n".join(label_lines),
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
        if tag == "in":
            shape = _shape_str(key, plan.sources.get(key), probed_shapes)
        else:
            producer = next(iter(boundary_out[key]))
            shape = _shape_str(key, _edge_spec(plan, producer, key), probed_shapes)
        text = f"{key}\n{shape}" if shape else key
        ax.text(bx, by, text, fontsize=7.0, ha="center", va="center", style="italic", zorder=3)

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
