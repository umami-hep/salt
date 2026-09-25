"""Config-level model-graph checks shared by `SaltModule` (at `__init__`/
`setup`) and `salt graph validate`: encoder edge-port constraints,
`class_names` ↔ schema-attr agreement, and name-based `origin_weighting`
resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from salt.graph.errors import ConfigError
from salt.graph.spec import GraphModule, Mode, flatten_spec

__all__ = ["check_class_names", "resolve_origin_weighting", "validate_edge_port"]

# the only attention backend an EdgeAttention encoder may declare — other
# backends would silently bypass the edge computation.
_EDGE_OK_BACKENDS = frozenset({"torch-math"})


def _edge_encoders(modules: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The encoder modules that declare an edge port (duck-typed on a non-None
    ``edges_key`` attribute — a `TransformerEncoder` with ``edges:`` configured).
    """
    return [
        (name, module)
        for name, module in modules.items()
        if getattr(module, "edges_key", None) is not None
    ]


def _concat_first_stream(modules: Mapping[str, Any]) -> tuple[str, str] | None:
    """The `Concat`'s first stream (the edge-stream-first reference), identified
    by its produced ``seq.x`` key (not by attribute, since `Normaliser`/`Split`
    also carry ``streams``). The edge tensor's stream must equal this so the
    ``[B, T, T, D_e]`` edge matrix aligns with the encoder's leading sequence
    rows. None when no `Concat` is configured.
    """
    for name, module in modules.items():
        streams = getattr(module, "streams", None)
        if not (isinstance(streams, (list, tuple)) and streams):
            continue
        declare = getattr(module, "declare_io", None)
        if not callable(declare):
            continue
        # declare_io is config-only/static — cheap and side-effect-free to call here.
        produces = flatten_spec(declare(Mode.FIT).produces)
        if "seq.x" in produces:
            return name, streams[0]
    return None


def validate_edge_port(modules: Mapping[str, Any]) -> int:
    """Validate the encoder edge-port bind-time constraints.

    Runs at `SaltModule.__init__` and in ``salt graph validate``; no-op when
    no encoder declares an edge port. Rule (a) edge-stream-first: the edge
    tensor's stream must be ``Concat.streams[0]``, because the encoder
    zero-pads the ``[B, T, T, D_e]`` edge matrix assuming the edge stream's
    rows lead the concatenated sequence. Rule (b): an encoder with an edge
    port may not declare a backend outside ``{"torch-math"}`` — EdgeAttention
    only supports raw torch attention. Either violation is a `ConfigError`;
    returns the number of edge-bearing encoders validated.
    """
    encoders = _edge_encoders(modules)
    if not encoders:
        return 0
    concat = _concat_first_stream(modules)
    for name, module in encoders:
        edge_stream = module.edge_stream  # "edges.tracks_emb" -> "tracks"
        # -- rule (a): edge stream must be Concat.streams[0] --------------------
        if concat is None:
            raise ConfigError(
                f"encoder {name!r} declares an edge port (edges={module.edges_key!r}) but no "
                "Concat is configured — the edge tensor's stream must be the FIRST concat stream "
                "so the [B, T, T, D_e] edge matrix aligns with the leading sequence rows "
                "(v1 sort-first hack, saltmodel.py:66-73)"
            )
        concat_name, first_stream = concat
        if edge_stream != first_stream:
            raise ConfigError(
                f"encoder {name!r} edge port stream {edge_stream!r} (from edges="
                f"{module.edges_key!r}) is NOT the first stream of Concat {concat_name!r} "
                f"(streams[0]={first_stream!r}) — the edge tensor must align with the LEADING "
                "rows of the concatenated sequence (the encoder zero-pads it to the "
                "register-augmented length assuming the edge stream is first, "
                f"transformer.py:689-719). fix: put {edge_stream!r} first in {concat_name!r}'s "
                "streams (v1 did this silently via the init-net sort, saltmodel.py:66-73)"
            )
        # -- rule (b): no non-edge attention backend alongside an edge port -----
        attn_type = getattr(getattr(module, "encoder", None), "attn_type", "torch-math")
        if attn_type not in _EDGE_OK_BACKENDS:
            raise ConfigError(
                f"encoder {name!r} declares attention backend {attn_type!r} alongside an edge "
                f"port (edges={module.edges_key!r}), but EdgeAttention supports ONLY raw torch "
                f"attention ({sorted(_EDGE_OK_BACKENDS)}). v1 silently IGNORED the backend when "
                "edge features were on (transformer.py:599-601) — v2 makes that a named error so "
                "a flash-varlen edge config fails loudly instead of running unexpectedly-slow raw "
                "attention. fix: set the encoder's attention.attn_type to 'torch-math' (or drop "
                "the edge port)."
            )
    return len(encoders)


def check_class_names(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Cross-check configured ``class_names`` against schema label attrs.

    For every module declaring ``class_names`` + ``stream`` + ``label``
    (duck-typed), the configured list must match the schema artifact's label
    attr in set AND order — a reordered ``class_names`` is a silent physics
    mislabeling no shape check can catch. Runs at `SaltModule.setup` and in
    ``salt graph validate``; no-op without a schema artifact. Returns the
    number of lists compared; mismatch raises `ConfigError`.
    """
    schema_group = getattr(reader, "schema_group", None)
    if not callable(schema_group):
        return 0
    checked = 0
    for name, module in modules.items():
        class_names = getattr(module, "class_names", None)
        stream = getattr(module, "stream", None)
        label = getattr(module, "label", None)
        if class_names is None or stream is None or label is None:
            continue
        gschema = schema_group(stream)
        if gschema is None:
            continue
        attr = gschema.attrs.get(label)
        if not (
            isinstance(attr, (list, tuple)) and attr and all(isinstance(item, str) for item in attr)
        ):
            continue  # no class-name attr for this label — nothing to check
        checked += 1
        if list(class_names) != list(attr):
            configured, stored = list(class_names), list(attr)
            diagnosis = (
                "same classes, DIFFERENT ORDER — the model head indices would be "
                "silently mislabelled"
                if sorted(configured) == sorted(stored)
                else "the class sets differ"
            )
            raise ConfigError(
                f"class_names of module {name!r} (config: model.modules.{name}."
                f"init_args.class_names) do not match the {label!r} attr of the "
                f"{stream!r} group in the schema artifact ({diagnosis}).\n"
                f"  configured: {configured}\n"
                f"  schema:     {stored}\n"
                f"  fix: set class_names to the schema order (or re-dump the schema if "
                "the file genuinely changed)"
            )
    return checked


def resolve_origin_weighting(modules: Mapping[str, GraphModule], reader: Any) -> int:
    """Resolve name-based ``origin_weighting`` to ids before bind.

    Maps names to integer origin ids against the schema artifact for every
    module exposing ``resolve_origin_names`` (duck-typed). Runs at
    `SaltModule.setup` before the two-phase bind, so resolved ids are in
    place when `VertexingTaskModule.bind` builds the composed head. Integer-id
    weighting and schema-less readers are no-ops; a name-based config that
    resolves nothing fails loudly at bind. Returns the count resolved.
    """
    resolved = 0
    for module in modules.values():
        resolve = getattr(module, "resolve_origin_names", None)
        if callable(resolve) and resolve(reader):
            resolved += 1
    return resolved
