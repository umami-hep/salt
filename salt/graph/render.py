"""Plan rendering for a compiled `Plan`: `plan_table` (ordered step table), `dot_source`/
`graph_dot` (Graphviz DOT text) and `render_image` (DOT -> PNG + PDF via the ``dot`` binary).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from salt.graph.errors import GraphError
from salt.graph.planner import SINKS, SOURCES, Plan
from salt.graph.spec import (
    KEY_SEP,
    TensorSpec,
    _has_wildcard,
    flatten_spec,
    is_symbolic_dim,
    split_symbolic_dim,
)
from salt.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from salt.graph.planner import PlanStep
    from salt.graph.spec import GraphModule, Mode

__all__ = ["dot_source", "graph_dot", "plan_table", "pruned_modules", "render_image"]

_LOG = get_logger(__name__)

# Symbolic dim families that are genuinely data-dependent (batch, per-stream
# token length, encoder layer length, merged seq length) and so stay symbolic
# in rendered shapes. Every other symbolic family is a config-fixed feature
# width resolved by `salt.model.bind.resolve_bind_schema`, so the renderer
# substitutes the concrete int instead.
_DATA_DIM_FAMILIES = frozenset({"B", "T", "L", "S"})

# ---------------------------------------------------------------------------
# plan table (the salt graph plan stdout == plan_<mode>.txt artifact)
# ---------------------------------------------------------------------------


def plan_table(plan: Plan) -> str:
    """Format the ordered plan table for one compiled plan.

    Shows each step's binding constraint (the latest predecessor forced by a
    key edge) and the narrowed wildcard results — including the full label
    list a dataset plan will load.
    """
    mode = plan.mode
    lines = [f"plan [mode={mode.name}] {len(plan.steps)} steps  plan_hash={plan.plan_hash}"]
    index = {step.name: i for i, step in enumerate(plan.steps)}
    for i, step in enumerate(plan.steps, start=1):
        producers = [edge.producer for edge in plan.edges if edge.consumer == step.name]
        if binding := [index[p] for p in producers if p in index]:
            after = plan.steps[max(binding)].name
        else:
            after = SOURCES if SOURCES in producers else "(no inputs)"
        needs = ", ".join(sorted(step.requires)) or "nothing"
        # a wildcard producer narrowed to nothing (e.g. Labels in TEST/ONNX)
        # runs as a no-op and reads no fields — say so, or the step looks
        # like live label loading in a plan that provably has none
        noop = "" if step.produces else "  [narrowed to 0 keys — no-op]"
        rewrites = f"  [rewrites {', '.join(sorted(step.rewrites))}]" if step.rewrites else ""
        lines.append(f"  {i:2d}. {step.name:<20} after {after:<20} (needs {needs}){noop}{rewrites}")
    narrowed_lines = []
    for step in plan.steps:
        io = step.module.declare_io(mode)
        concrete = {key for key in flatten_spec(io.produces) if not _has_wildcard(key)}
        # a step's own rewrite keys are declared (not narrowed wildcards) — a
        # rewriter must not be mis-reported as a narrowed wildcard producer
        concrete |= set(flatten_spec(getattr(io, "rewrites", {})))
        if extra := sorted(set(step.produces) - concrete):
            narrowed_lines.append(f"  {step.name}: {', '.join(extra)}")
    if narrowed_lines:
        lines += [f"narrowed wildcards [mode={mode.name}]:", *narrowed_lines]
    lines.append(f"sources: {', '.join(sorted(plan.sources)) or '(none)'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT — emitted alongside every image render
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Escape a string for a double-quoted DOT identifier (without surrounding
    quotes).
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _quote(text: str) -> str:
    """Quote a string as a DOT identifier."""
    return f'"{_esc(text)}"'


def _edge_spec(plan: Plan, producer: str, key: str) -> TensorSpec | None:
    """Resolve the `TensorSpec` an edge carries (mirrors `dot_source`'s lookup)."""
    return plan.sources.get(key) if producer == SOURCES else plan.step(producer).produces.get(key)


# Row font colour for the signature-card rows: the port-card layout has no
# per-key wires to colour, so the orange/red/blue kind accents are applied to
# the consumed/produced row text instead.
_KIND_COLOURS = {
    "label": "#b5651d",
    "loss": "#c0392b",
    "preds": "#1f6fb2",
    "default": "#333333",  # neutral row text
    "shape": "#888888",  # grey shape annotation
}
_TABLE_OPEN = '<\n    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'


def _row_colour(key: str, spec: TensorSpec | None) -> str:
    """Kind-based row font colour: labels orange, losses red, ``preds.*`` blue,
    else the neutral default.
    """
    if spec is not None and spec.kind in _KIND_COLOURS:
        return _KIND_COLOURS[spec.kind]
    if key.partition(KEY_SEP)[0] == "preds":
        return _KIND_COLOURS["preds"]
    return _KIND_COLOURS["default"]


def _html_esc(text: str) -> str:
    """Escape a string for inclusion in a Graphviz HTML-like label."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_feature_dim(dim: int | str) -> bool:
    """Whether a shape entry is a symbolic feature dim (resolvable to a width)
    — true for any symbolic family except the data-dependent ``B``/``T``/``L``/``S``.
    """
    return is_symbolic_dim(dim) and split_symbolic_dim(str(dim))[0] not in _DATA_DIM_FAMILIES


def _shape_str(key: str, spec: TensorSpec | None, widths: Mapping[str, int] | None) -> str:
    """Shape string for a card row: substitutes the concrete int for the shape's
    last dim when `widths` resolves `key` and that dim is a symbolic feature
    dim, e.g. ``encoded.tracks`` renders ``(B, T:tracks, 16)``.
    """
    if spec is None or not spec.shape:
        return ""
    dims: list[int | str] = list(spec.shape)
    if widths is not None and key in widths and _is_feature_dim(dims[-1]):
        dims[-1] = widths[key]
    return "(" + ", ".join(str(dim) for dim in dims) + ")"


def _card_row(key: str, shape: str, colour: str, *, bold: bool) -> str:
    """One ``<TR>`` row of a signature card: kind-coloured key plus grey shape."""
    name = f"<B>{_html_esc(key)}</B>" if bold else _html_esc(key)
    tail = f'  <FONT COLOR="{_KIND_COLOURS["shape"]}">{_html_esc(shape)}</FONT>' if shape else ""
    return f'    <TR><TD ALIGN="LEFT"><FONT COLOR="{colour}">{name}{tail}</FONT></TD></TR>'


def _card_section(tag: str) -> str:
    """A faint italic ``in``/``out`` section divider row."""
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
    *,
    badge: str | None = None,
) -> str:
    """Assemble one HTML-like signature-card node line: `ins`/`outs` are
    ``(key, shape, colour)`` row triples; the header carries `title` +
    optional `cls` over a `fill` background, plus an optional `badge` line
    (e.g. "frozen") under the class name.
    """
    sub = f'<BR/><FONT POINT-SIZE="8" COLOR="#555555">{_html_esc(cls)}</FONT>' if cls else ""
    badge_html = (
        f'<BR/><FONT POINT-SIZE="8" COLOR="#b30000"><B>{_html_esc(badge)}</B></FONT>'
        if badge
        else ""
    )
    rows = [
        '    <TR><TD BGCOLOR="'
        f'{fill}" ALIGN="CENTER"><B>{_html_esc(title)}</B>{sub}{badge_html}</TD></TR>'
    ]
    if ins:
        rows.append(_card_section("in"))
        rows.extend(_card_row(k, s, c, bold=False) for k, s, c in ins)
    if outs:
        rows.append(_card_section("out"))
        rows.extend(_card_row(k, s, c, bold=True) for k, s, c in outs)
    table = "\n".join([_TABLE_OPEN, *rows, "    </TABLE>>"])
    return f"  {_quote(name)} [label={table}];"


def dot_source(
    plan: Plan,
    modules: Mapping[str, GraphModule] | None = None,
    pruned: Iterable[str] = (),
    widths: Mapping[str, int] | None = None,
    frozen: frozenset[str] | None = None,
    title: str | None = None,
) -> str:
    r"""Render a compiled plan as Graphviz DOT text — port-card layout.

    Each module renders as an HTML-like signature card: a colour-filled header
    (``name`` + ``ClassName``), an ``in`` section listing the keys the module
    consumes, and an ``out`` section listing the keys it produces. Each row is a
    ``key`` plus its tensor shape, the key kind-coloured (labels orange, losses
    red, ``preds.*`` blue). The ``<sources>`` pseudo-node carries only an ``out``
    section; ``<sinks>`` only an ``in`` section. Edges collapse to one deduped
    ``producer -> consumer`` arrow per pair. Demand-pruned modules (looked up in
    `modules`) render as a grey dashed card.

    Parameters
    ----------
    modules : Mapping[str, GraphModule] | None, optional
        Module instances, used only to label `pruned` cards with their class.
    pruned : Iterable[str], optional
        Names of demand-pruned modules to draw grey/dashed, by default ().
    widths : Mapping[str, int] | None, optional
        Statically resolved per-key feature widths (dotted key -> concrete
        last-dim int), from `salt.model.bind.resolve_bind_schema`. When a
        key's declared shape ends in a symbolic feature dim, that dim is shown
        as the concrete width; data-dependent dims stay symbolic. By default None.
    frozen : frozenset[str] | None, optional
        Module names to style as frozen (grey fill + a "frozen" badge) — the
        rest render with their default namespace fill. Used by ``salt
        merge-config`` to show a `training_schedule` stage's freeze mask. By
        default None (no module is styled frozen; DOT output byte-identical to
        the unannotated render).
    title : str | None, optional
        A caption placed at the top of the graph (e.g. a stage header). By
        default None (no caption; DOT output byte-identical).
    """
    modules = dict(modules or {})
    # plaintext nodes carry HTML-like TABLE labels (the cards), so shape/style/
    # fill live in the table, not the node attrs. splines=ortho is safe since
    # there are no floating edge labels to detach.
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
    if title is not None:
        lines.extend(['  labelloc="t";', '  fontname="Helvetica";', f"  label={_quote(title)};"])

    # consumed keys per module: each require edge into the module, with the
    # spec carried by the producing edge (via the _edge_spec lookup).
    # dict-insertion dedupes while preserving edge order.
    consumed: dict[str, dict[str, TensorSpec | None]] = {}
    for edge in plan.edges:
        consumed.setdefault(edge.consumer, {}).setdefault(
            edge.key, _edge_spec(plan, edge.producer, edge.key)
        )

    def _rows(items: Iterable[tuple[str, TensorSpec | None]]) -> list[tuple[str, str, str]]:
        return [(key, _shape_str(key, spec, widths), _row_colour(key, spec)) for key, spec in items]

    if any(edge.producer == SOURCES for edge in plan.edges):
        lines.append(
            _card_node(SOURCES, SOURCES, "", "#f5f5f5", [], _rows(sorted(plan.sources.items())))
        )
    for step in plan.steps:
        ins = _rows(consumed.get(step.name, {}).items())
        outs = _rows(step.produces.items())
        cls = type(step.module).__name__
        badge = "frozen" if frozen is not None and step.name in frozen else None
        fill = _NS_COLOURS["frozen"] if badge else _module_colour(step)
        lines.append(_card_node(step.name, step.name, cls, fill, ins, outs, badge=badge))
    if any(edge.consumer == SINKS for edge in plan.edges):
        lines.append(
            _card_node(SINKS, SINKS, "", "#f5f5f5", _rows(consumed.get(SINKS, {}).items()), [])
        )
    for name in pruned:
        cls = type(modules[name]).__name__ if name in modules else "?"
        ins = _rows(consumed.get(name, {}).items())
        line = _card_node(name, f"{name} (pruned)", cls, _NS_COLOURS["pruned"], ins, [])
        # grey dashed border distinguishes a demand-pruned card from a live one
        lines.append(line.replace("];", ", style=dashed, color=grey];"))

    pairs = dict.fromkeys((edge.producer, edge.consumer) for edge in plan.edges)  # first-seen order
    lines.extend(f"  {_quote(producer)} -> {_quote(consumer)};" for producer, consumer in pairs)
    lines.append("}")
    return "\n".join(lines) + "\n"


def graph_dot(
    plan: Plan,
    modules: Mapping[str, GraphModule] | None,
    *,
    pruned: Iterable[str] = (),
    widths: Mapping[str, int] | None = None,
    title: str | None = None,
) -> str:
    """The DOT text ``salt graph plot`` and `salt.callbacks.GraphArtifacts` render.

    Parameters
    ----------
    plan : Plan
        The compiled plan to draw.
    modules : Mapping[str, GraphModule] | None
        Module instances (labels the `pruned` cards with their class).
    pruned : Iterable[str], optional
        Demand-pruned module names (see `pruned_modules`), by default ().
    widths : Mapping[str, int] | None, optional
        Resolved per-key feature widths (`salt.model.bind.resolve_bind_schema`), by default None.
    title : str | None, optional
        Graph caption, by default None.

    Returns
    -------
    str
        The `dot_source` text for these arguments.
    """
    return dot_source(plan, modules, pruned, widths=widths, title=title)


def pruned_modules(modules: Mapping[str, GraphModule], plan: Plan) -> list[str]:
    """Sorted names of the demand-pruned modules of `plan` — the `deadcode` ``key == "*"`` set.

    A configured module is pruned when it has a port active in ``plan.mode`` yet is
    not a plan step. A module with no active port in that mode (e.g. a TEST-only
    writer in FIT) is mode-gated, not pruned, and is left out — exactly as the
    planner never makes it a node.

    Parameters
    ----------
    modules : Mapping[str, GraphModule]
        Every module the plan was compiled from.
    plan : Plan
        The compiled plan.

    Returns
    -------
    list[str]
        Pruned module names, sorted.
    """
    live = set(plan.module_names)
    return sorted(
        name
        for name, module in modules.items()
        if name not in live and _active_in(module, plan.mode)
    )


def _active_in(module: GraphModule, mode: Mode) -> bool:
    """Whether any requires/produces/rewrites leaf of `module` is active in `mode` (the
    planner's node test, `salt.graph.planner._collect_nodes`).
    """
    io = module.declare_io(mode)
    trees = (io.requires, io.produces, getattr(io, "rewrites", {}))
    return any(spec.active_in(mode) for tree in trees for spec in flatten_spec(tree).values())


# ---------------------------------------------------------------------------
# DOT -> PNG + PDF via the Graphviz `dot` binary
# ---------------------------------------------------------------------------


def render_image(dot_path: Path, stem: Path, *, strict: bool) -> tuple[Path, Path] | None:
    """Rasterise `dot_path` to ``<stem>.png`` (150 dpi) and ``<stem>.pdf`` with Graphviz ``dot``.

    Silent on success; the caller reports the written paths.

    Parameters
    ----------
    dot_path : Path
        The DOT source file (already written).
    stem : Path
        Output path without suffix; ``.png``/``.pdf`` are appended.
    strict : bool
        True (``salt graph plot``): a missing ``dot`` or a failed render raises.
        False (the training callback): either logs a warning with the manual
        command and returns None — never raises.

    Returns
    -------
    tuple[Path, Path] | None
        ``(png, pdf)``, or None when a non-strict render did not complete.

    Raises
    ------
    GraphError
        With `strict`, when ``dot`` is not on PATH or a ``dot`` invocation fails
        (its stderr surfaced in the message).
    """
    png, pdf = stem.with_name(f"{stem.name}.png"), stem.with_name(f"{stem.name}.pdf")
    hint = f"render manually with: dot -Tpng {dot_path} -o {png}"
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        if strict:
            raise GraphError(
                "the Graphviz `dot` binary was not found on PATH — cannot render the "
                f"graph image. The DOT source was written to {dot_path}; render it "
                f"later, e.g. `dot -Tpng {dot_path} -o {png}`. On a native "
                "install (uv/conda), install Graphviz separately as a system "
                "package; the prebuilt salt containers already include it, so "
                "hitting this there is unexpected. See "
                "https://ftag-salt.docs.cern.ch/setup/#install-graphviz"
            )
        _LOG.warning(f"graphviz `dot` not on PATH — wrote {dot_path} only; {hint}")
        return None
    renders = (
        ([dot_bin, "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(png)], png),
        ([dot_bin, "-Tpdf", str(dot_path), "-o", str(pdf)], pdf),
    )
    for cmd, target in renders:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            failure = (
                f"`dot` failed to render {target} (exit {result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'}"
            )
            if strict:
                raise GraphError(failure)
            _LOG.warning(f"{failure}; {hint}")
            return None
    return png, pdf


# ---------------------------------------------------------------------------
# module box palette (shared by the DOT card fills)
# ---------------------------------------------------------------------------

# module box fill by (first matching) produced namespace
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
    # card fills that are not namespace-derived
    "pruned": "#dddddd",
    "frozen": "#bdbdbd",  # mid-grey, distinct from the lighter pruned fill
    "fallback": "#cccccc",
}


def _module_colour(step: PlanStep) -> str:
    """Pick the box colour from the step's produced namespaces."""
    for namespace in sorted({key.split(KEY_SEP)[0] for key in step.produces}):
        if namespace in _NS_COLOURS:
            return _NS_COLOURS[namespace]
    return _NS_COLOURS["fallback"]
