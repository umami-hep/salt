"""Schema dataclasses + loader for the schema-driven test-data generator
(field types: distribution, label, id, link).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Field types
FIELD_TYPES = ("distribution", "label", "id", "link")

# Distributions understood by the engine
DISTS = ("normal", "uniform", "lognormal", "exponential", "constant")

# Per-distribution default parameters
DIST_DEFAULTS: dict[str, dict[str, float]] = {
    "normal": {"mean": 0.0, "std": 1.0},
    "uniform": {"low": 0.0, "high": 1.0},
    "lognormal": {"mean": 0.0, "sigma": 1.0},
    "exponential": {"scale": 1.0},
    "constant": {"value": 0.0},
}


class SchemaError(ValueError):
    """Raised when a schema fails structural validation."""


# --------------------------------------------------------------------------- #
# Field specs
# --------------------------------------------------------------------------- #
@dataclass
class FieldSpec:
    """Base for all field specs."""

    name: str
    type: str
    dtype: str = "f4"
    invalid_fill: Any | None = None  # overrides the group/top-level sentinel


@dataclass
class DistributionField(FieldSpec):
    dist: str = "uniform"
    params: dict[str, float] = field(default_factory=dict)
    nan_where: str | None = None  # pinned grammar: "<field> == <int>"


@dataclass
class LabelField(FieldSpec):
    classes: list[int] = field(default_factory=list)
    sample_classes: list[int] | None = None
    class_names: list[str] | None = None
    class_names_append_if: dict[str, list[str]] | None = None
    weights: list[float] | None = None
    sorted: bool = False
    classes_by_flag: dict[str, list[int]] | None = None
    class_names_by_flag: dict[str, list[str]] | None = None

    def resolved_classes(self, flags: dict[str, bool]) -> list[int]:
        """Resolve the canonical class set given active flags.

        Precedence (mirrors the current code's ``if make_xbb / elif is_gn3 /
        else`` ladder): ``make_xbb`` > ``is_gn3`` > default.
        """
        if self.classes_by_flag:
            for flag in ("make_xbb", "is_gn3"):
                if flags.get(flag) and flag in self.classes_by_flag:
                    return list(self.classes_by_flag[flag])
        return list(self.classes)

    def resolved_sample_classes(self, flags: dict[str, bool]) -> list[int]:
        """The values actually drawn into the data, given active flags."""
        if self.classes_by_flag:
            for flag in ("make_xbb", "is_gn3"):
                if flags.get(flag) and flag in self.classes_by_flag:
                    # flag variants draw the full declared set
                    return list(self.classes_by_flag[flag])
        if self.sample_classes is not None:
            return list(self.sample_classes)
        return list(self.classes)

    def resolved_class_names(self, flags: dict[str, bool]) -> list[str] | None:
        """Resolve the class-name attr list given active flags."""
        if self.class_names_by_flag:
            for flag in ("make_xbb", "is_gn3"):
                if flags.get(flag) and flag in self.class_names_by_flag:
                    return list(self.class_names_by_flag[flag])
        if self.class_names is None:
            return None
        names = list(self.class_names)
        if self.class_names_append_if:
            for flag, extra in self.class_names_append_if.items():
                if flags.get(flag):
                    names = names + list(extra)
        return names


@dataclass
class IdField(FieldSpec):
    range: tuple[int, int] = (0, 10000)
    scope: str = "jet"
    unique: bool = True


@dataclass
class LinkField(FieldSpec):
    references: str = ""  # "<group>.<idfield>"
    scope: str = "jet"
    unmatched_fraction: float = 0.0
    unmatched_value: int | None = None
    required_match: bool = False
    select_over: str = "valid_ids"

    @property
    def ref_group(self) -> str:
        return self.references.split(".", 1)[0]

    @property
    def ref_field(self) -> str:
        return self.references.split(".", 1)[1]


# --------------------------------------------------------------------------- #
# Group + schema
# --------------------------------------------------------------------------- #
@dataclass
class GroupSpec:
    name: str
    kind: str  # "global" | "constituent"
    fields: list[FieldSpec] = field(default_factory=list)
    max_items: int | None = None
    valid_fraction: float = 0.5
    min_valid: int = 0
    mask_invalid: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)
    alias_of: str | None = None
    emit_if: str | None = None

    @property
    def is_constituent(self) -> bool:
        return self.kind == "constituent"


@dataclass
class Schema:
    n_samples: int
    groups: list[GroupSpec]
    seed: int = 42
    fill_float: float = float("nan")
    fill_int: int = -1
    file_attrs: dict[str, Any] = field(default_factory=dict)

    def group(self, name: str) -> GroupSpec:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"No group named {name!r} in schema")

    @property
    def config_attr(self) -> str:
        return str(self.file_attrs.get("config", "{}"))

    def unique_jets_from(self) -> str:
        explicit = self.file_attrs.get("unique_jets_from")
        if explicit:
            return explicit
        for g in self.groups:
            if g.kind == "global" and g.emit_if is None:
                return g.name
        # fall back to first global
        for g in self.groups:
            if g.kind == "global":
                return g.name
        return self.groups[0].name


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_field(raw: dict[str, Any]) -> FieldSpec:
    if "name" not in raw or "type" not in raw:
        raise SchemaError(f"Field spec missing 'name' or 'type': {raw}")
    ftype = raw["type"]
    if ftype not in FIELD_TYPES:
        raise SchemaError(f"Unknown field type {ftype!r} (field {raw.get('name')!r})")
    common = {
        "name": raw["name"],
        "type": ftype,
        "dtype": raw.get("dtype", "i4" if ftype in ("label", "id", "link") else "f4"),
        "invalid_fill": raw.get("invalid_fill"),
    }
    if ftype == "distribution":
        dist = raw.get("dist", "uniform")
        if dist not in DISTS:
            raise SchemaError(f"Unknown distribution {dist!r} (field {raw['name']!r})")
        return DistributionField(
            **common,
            dist=dist,
            params=dict(raw.get("params", {})),
            nan_where=raw.get("nan_where"),
        )
    if ftype == "label":
        if "classes" not in raw:
            raise SchemaError(f"label field {raw['name']!r} missing 'classes'")
        return LabelField(
            **common,
            classes=list(raw["classes"]),
            sample_classes=(list(raw["sample_classes"]) if "sample_classes" in raw else None),
            class_names=(list(raw["class_names"]) if "class_names" in raw else None),
            class_names_append_if=raw.get("class_names_append_if"),
            weights=(list(raw["weights"]) if "weights" in raw else None),
            sorted=bool(raw.get("sorted")),
            classes_by_flag=raw.get("classes_by_flag"),
            class_names_by_flag=raw.get("class_names_by_flag"),
        )
    if ftype == "id":
        if "range" not in raw:
            raise SchemaError(f"id field {raw['name']!r} missing 'range'")
        lo, hi = raw["range"]
        return IdField(
            **common,
            range=(int(lo), int(hi)),
            scope=raw.get("scope", "jet"),
            unique=bool(raw.get("unique", True)),
        )
    # link
    if "references" not in raw:
        raise SchemaError(f"link field {raw['name']!r} missing 'references'")
    return LinkField(
        **common,
        references=raw["references"],
        scope=raw.get("scope", "jet"),
        unmatched_fraction=float(raw.get("unmatched_fraction", 0.0)),
        unmatched_value=raw.get("unmatched_value"),
        required_match=bool(raw.get("required_match")),
        select_over=raw.get("select_over", "valid_ids"),
    )


def _parse_group(raw: dict[str, Any]) -> GroupSpec:
    if "name" not in raw or "kind" not in raw:
        raise SchemaError(f"Group missing 'name' or 'kind': {raw}")
    if raw["kind"] not in ("global", "constituent"):
        raise SchemaError(f"Group {raw['name']!r}: unknown kind {raw['kind']!r}")
    fields = [_parse_field(f) for f in raw.get("fields", [])]
    return GroupSpec(
        name=raw["name"],
        kind=raw["kind"],
        fields=fields,
        max_items=raw.get("max_items"),
        valid_fraction=float(raw.get("valid_fraction", 0.5)),
        min_valid=int(raw.get("min_valid", 0)),
        mask_invalid=bool(raw.get("mask_invalid", True)),
        attrs=dict(raw.get("attrs", {})),
        alias_of=raw.get("alias_of"),
        emit_if=raw.get("emit_if"),
    )


def parse_schema(doc: dict[str, Any]) -> Schema:
    """Parse a raw dict (from YAML/JSON) into a validated ``Schema``."""
    if "n_samples" not in doc:
        raise SchemaError("Schema missing required 'n_samples'")
    if "groups" not in doc:
        raise SchemaError("Schema missing required 'groups'")
    fill = doc.get("fill", {})
    schema = Schema(
        n_samples=int(doc["n_samples"]),
        groups=[_parse_group(g) for g in doc["groups"]],
        seed=int(doc.get("seed", 42)),
        fill_float=fill.get("float", float("nan")),
        fill_int=int(fill.get("int", -1)),
        file_attrs=dict(doc.get("file_attrs", {})),
    )
    _validate(schema)
    return schema


def _validate(schema: Schema) -> None:
    names = [g.name for g in schema.groups]
    if len(names) != len(set(names)):
        raise SchemaError("Duplicate group names in schema")

    # build id-field index for link validation
    id_fields: dict[str, set[str]] = {}
    for g in schema.groups:
        if g.alias_of is not None:
            continue
        id_fields[g.name] = {f.name for f in g.fields if isinstance(f, IdField)}

    for g in schema.groups:
        if g.alias_of is not None:
            if g.alias_of not in names:
                raise SchemaError(f"Group {g.name!r} alias_of unknown group {g.alias_of!r}")
            continue
        if g.is_constituent and g.max_items is None:
            raise SchemaError(f"constituent group {g.name!r} missing 'max_items'")
        field_names = {f.name for f in g.fields}
        for f in g.fields:
            if isinstance(f, LabelField):
                _validate_label(g, f)
            if isinstance(f, DistributionField) and f.nan_where is not None:
                _validate_nan_where(g, f, field_names)
            if isinstance(f, LinkField):
                _validate_link(schema, g, f, id_fields)

    _check_no_cycles(schema)
    _propagate_required_match_min_valid(schema)


def _propagate_required_match_min_valid(schema: Schema) -> None:
    """Guarantee a genuine referent exists in phase 1 for every required_match link.

    A ``required_match: true`` link to a CONSTITUENT group ``G`` raises the
    effective ``G.min_valid`` to ``max(G.min_valid, 1)`` — phase 1 must
    generate at least one genuinely-valid referent (real id, real payload)
    so the phase-2 resolver always finds a non-empty pool (promoting an
    already-invalid-filled slot at phase 2 would carry a sentinel id).
    Global references are rejected separately by `_validate_link`.
    """
    for g in schema.groups:
        if g.alias_of is not None:
            continue
        for f in g.fields:
            if isinstance(f, LinkField) and f.required_match:
                ref = schema.group(f.ref_group)
                if ref.kind == "constituent" and ref.min_valid < 1:
                    ref.min_valid = 1


def _validate_label(g: GroupSpec, f: LabelField) -> None:
    sample_classes = f.sample_classes if f.sample_classes is not None else f.classes
    if not set(sample_classes).issubset(set(f.classes)):
        raise SchemaError(
            f"{g.name}.{f.name}: sample_classes {sample_classes} not subset of classes {f.classes}"
        )
    if f.class_names is not None and len(f.class_names) != len(sample_classes):
        raise SchemaError(
            f"{g.name}.{f.name}: class_names length {len(f.class_names)} "
            f"!= len(sample_classes) {len(sample_classes)}"
        )
    if f.weights is not None and len(f.weights) != len(sample_classes):
        raise SchemaError(
            f"{g.name}.{f.name}: weights length {len(f.weights)} "
            f"!= len(sample_classes) {len(sample_classes)}"
        )


def _validate_nan_where(g: GroupSpec, f: DistributionField, field_names: set[str]) -> None:
    parsed = parse_nan_where(f.nan_where)
    if parsed is None:
        raise SchemaError(
            f"{g.name}.{f.name}: nan_where {f.nan_where!r} does not match "
            f"the pinned grammar '<field> == <int>'"
        )
    ref_field, _ = parsed
    if ref_field not in field_names:
        raise SchemaError(f"{g.name}.{f.name}: nan_where references unknown field {ref_field!r}")


def _validate_link(
    schema: Schema, g: GroupSpec, f: LinkField, id_fields: dict[str, set[str]]
) -> None:
    if "." not in f.references:
        raise SchemaError(f"{g.name}.{f.name}: references must be '<group>.<idfield>'")
    ref_group, ref_field = f.ref_group, f.ref_field
    if ref_group not in id_fields:
        raise SchemaError(f"{g.name}.{f.name}: references unknown group {ref_group!r}")
    if ref_field not in id_fields[ref_group]:
        raise SchemaError(f"{g.name}.{f.name}: references {f.references!r} is not an 'id' field")
    if f.required_match:
        ref = schema.group(ref_group)
        if ref.kind == "global":
            raise SchemaError(
                f"{g.name}.{f.name}: required_match=True against a global reference is unsupported"
            )


def _check_no_cycles(schema: Schema) -> None:
    edges: dict[str, set[str]] = {g.name: set() for g in schema.groups if g.alias_of is None}
    for g in schema.groups:
        if g.alias_of is not None:
            continue
        for f in g.fields:
            if isinstance(f, LinkField):
                # referenced -> referencing
                edges[f.ref_group].add(g.name)
    # Kahn's algorithm
    indeg = dict.fromkeys(edges, 0)
    for src, dsts in edges.items():
        for d in dsts:
            indeg[d] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for d in edges[n]:
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)
    if visited != len(edges):
        raise SchemaError("Reference graph contains a cycle")


def topo_sort_link_groups(schema: Schema) -> list[str]:
    """Return non-alias group names in dependency (topological) order over
    ``references`` edges (referenced group before referencing group).
    """
    edges: dict[str, set[str]] = {g.name: set() for g in schema.groups if g.alias_of is None}
    for g in schema.groups:
        if g.alias_of is not None:
            continue
        for f in g.fields:
            if isinstance(f, LinkField):
                edges[f.ref_group].add(g.name)
    indeg = dict.fromkeys(edges, 0)
    for src, dsts in edges.items():
        for d in dsts:
            indeg[d] += 1
    # stable order: preserve schema order among zero-indegree nodes
    order_index = {g.name: i for i, g in enumerate(schema.groups)}
    queue = sorted([n for n, d in indeg.items() if d == 0], key=lambda n: order_index[n])
    out: list[str] = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        new_ready = []
        for d in edges[n]:
            indeg[d] -= 1
            if indeg[d] == 0:
                new_ready.append(d)
        queue.extend(sorted(new_ready, key=lambda n: order_index[n]))
        queue.sort(key=lambda n: order_index[n])
    return out


def parse_nan_where(expr: str | None) -> tuple[str, int] | None:
    """Parse the pinned ``"<field> == <int>"`` grammar.

    Returns ``(field_name, int_literal)`` or ``None`` if the expression does not
    match. Never uses ``eval``.
    """
    if expr is None:
        return None
    if "==" not in expr:
        return None
    left, _, right = expr.partition("==")
    left, right = left.strip(), right.strip()
    if not left:
        return None
    try:
        literal = int(right)
    except (TypeError, ValueError):
        return None
    return left, literal


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_schema(path: str | Path | dict[str, Any] | Schema) -> Schema:
    """Load a schema from a YAML/JSON path, a raw dict, or pass through a Schema."""
    if isinstance(path, Schema):
        return path
    if isinstance(path, dict):
        return parse_schema(path)
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        doc = json.loads(text)
    else:
        doc = yaml.safe_load(text)
    return parse_schema(doc)


def as_schema(schema: dict[str, Any] | Schema) -> Schema:
    """Coerce a dict or Schema into a Schema (no file IO)."""
    if isinstance(schema, Schema):
        return schema
    return parse_schema(schema)
