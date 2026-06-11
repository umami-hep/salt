"""The dataset schema artifact for the salt v2 kernel.

Design §2.6: ``salt2 schema dump train.h5 -o schema.yaml`` writes a small YAML
describing each H5 group — field names + dtypes, ``valid`` presence, group
attrs (incl. class-name lists such as ``flavour_label``), and global file
attrs. The artifact is generated once and versioned next to ``norm_dict.yaml``;
reading it at config-parse time is config I/O, not data I/O.

`dump_schema` is the only function in this module (and in ``salt.core``, M1)
that touches a data file. Everything else operates on the in-memory `Schema`,
which feeds the planner's wildcard narrowing (design §2.2 rule (d)) via
`Schema.keys` / `Schema.validate_keys`.

The artifact is versioned from day one (``schema_version:``) and `load_schema`
is tolerant of additive change — unknown top-level and per-group keys are
ignored (design §11 risk 12).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from salt.core.graph.errors import SchemaError
from salt.core.graph.spec import KEY_SEP, join_key, split_key

__all__ = [
    "SCHEMA_VERSION",
    "GroupSchema",
    "KeyValidation",
    "Schema",
    "dump_schema",
    "load_schema",
    "save_schema",
]

SCHEMA_VERSION = 1
"""Current schema artifact version (design §11 risk 12)."""

_SUGGESTION_CUTOFF = 0.5


@dataclass(frozen=True)
class GroupSchema:
    """Schema of one H5 group: field names + dtypes and group attrs (design §2.6).

    `fields` maps field name to a numpy dtype name (``"float32"``); `attrs`
    holds the group attrs as plain Python values, including class-name lists
    such as the ``flavour_label`` attr used by the §4.1 class-names check.
    """

    fields: dict[str, str]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid(self) -> bool:
        """Whether the group carries a ``valid`` field (padded-sequence marker).

        Returns
        -------
        bool
            True if ``"valid"`` is among the fields.
        """
        return "valid" in self.fields


@dataclass(frozen=True)
class KeyValidation:
    """Report from `Schema.validate_keys` (design §2.6).

    `present` keys exist in the schema; `missing` keys name an existing group
    but an absent field; `unknown` keys name a group not in the schema (or are
    not ``group.field``-shaped at all). `suggestions` maps each bad key to its
    nearest schema keys, for §4.1-quality error messages.
    """

    present: tuple[str, ...]
    missing: tuple[str, ...]
    unknown: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every validated key is present in the schema.

        Returns
        -------
        bool
            True if there are no missing or unknown keys.
        """
        return not (self.missing or self.unknown)


@dataclass(frozen=True)
class Schema:
    """The dataset schema artifact (design §2.6).

    `groups` maps H5 group name to `GroupSchema`; `attrs` holds global file
    attrs; `schema_version` versions the artifact format (design §11 risk 12).
    """

    groups: dict[str, GroupSchema]
    attrs: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def keys(self) -> tuple[str, ...]:
        """Flatten the schema to dotted ``group.field`` keys.

        This is the flat universe handed to `compile_plan(schema=...)` so
        wildcard narrowing is statically validated (design §2.2 rule (d)).

        Returns
        -------
        tuple[str, ...]
            All ``group.field`` keys, group by group in declaration order.

        Raises
        ------
        SchemaError
            If a group or field name cannot form a dotted bundle key (e.g.
            contains ``"."`` — legal in H5/numpy, unaddressable as a key).
        """
        out: list[str] = []
        for group, gschema in self.groups.items():
            for fld in gschema.fields:
                if not group or not fld or KEY_SEP in group or KEY_SEP in fld:
                    raise SchemaError(
                        f"schema group {group!r} field {fld!r} cannot form a dotted bundle "
                        f"key — names containing {KEY_SEP!r} (or empty names) are not "
                        "addressable (design §2.1); regenerate the artifact with "
                        "salt2 schema dump"
                    )
                out.append(f"{group}{KEY_SEP}{fld}")
        return tuple(out)

    def validate_keys(self, keys: Iterable[str]) -> KeyValidation:
        """Validate demanded dotted keys against the schema (design §2.6).

        Used by the planner's wildcard narrowing and the validate CLI: each
        key is interpreted as ``group.field`` (the first component is the
        group). Malformed string keys (``"jets..pt"``, ``""``) are classified
        as unknown rather than raising; non-string keys are a caller bug and
        raise TypeError. Outputs are sorted for deterministic reports.

        Parameters
        ----------
        keys : Iterable[str]
            Dotted keys to check, e.g. ``["jets.pt", "tracks.d0"]``.

        Returns
        -------
        KeyValidation
            The missing/unknown report, with nearest-key suggestions.
        """
        universe = sorted(self.keys())
        present: list[str] = []
        missing: list[str] = []
        unknown: list[str] = []
        suggestions: dict[str, tuple[str, ...]] = {}
        for key in keys:
            try:
                parts = split_key(key)
            except ValueError:  # malformed key: report unknown, do not raise
                parts = ()
            group = parts[0] if parts else None
            fld = join_key(parts[1:]) if len(parts) > 1 else None
            if group is None or fld is None or group not in self.groups:
                unknown.append(key)
            elif fld in self.groups[group].fields:
                present.append(key)
            else:
                missing.append(key)
            if key not in present:
                near = get_close_matches(key, universe, n=3, cutoff=_SUGGESTION_CUTOFF)
                if near:
                    suggestions[key] = tuple(near)
        return KeyValidation(
            present=tuple(sorted(present)),
            missing=tuple(sorted(missing)),
            unknown=tuple(sorted(unknown)),
            suggestions=suggestions,
        )


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def load_schema(path: str | Path) -> Schema:
    """Load a schema artifact from YAML (design §2.6).

    Tolerant of additive change (design §11 risk 12): unknown top-level keys
    and unknown per-group keys are ignored, and a newer `schema_version` is
    accepted as long as the keys this reader needs are present.

    Returns
    -------
    Schema
        The parsed schema.

    Raises
    ------
    SchemaError
        If the file is not a mapping, the version field is missing/invalid,
        or the groups structure is malformed.
    """
    path = Path(path)
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except OSError as err:
        raise SchemaError(f"cannot read schema file {path}: {err}") from err
    except yaml.YAMLError as err:
        raise SchemaError(f"schema file {path} is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise SchemaError(
            f"schema file {path} must contain a mapping, got {type(raw).__name__} (design §2.6)"
        )
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SchemaError(
            f"schema file {path} has missing or invalid 'schema_version' "
            f"({version!r}) — artifacts are versioned from day one (design §11 risk 12)"
        )
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, dict):
        raise SchemaError(f"schema file {path} must declare a 'groups' mapping (design §2.6)")
    groups: dict[str, GroupSchema] = {}
    for name, node in groups_raw.items():
        gname = str(name)
        if KEY_SEP in gname:
            raise SchemaError(
                f"schema file {path}: group {gname!r} contains {KEY_SEP!r} — group names must "
                "be single dotted-key components (design §2.1); regenerate the artifact with "
                "salt2 schema dump"
            )
        if not isinstance(node, dict):
            raise SchemaError(f"schema file {path}: group {gname!r} must be a mapping")
        fields = node.get("fields")
        if not isinstance(fields, dict):
            raise SchemaError(
                f"schema file {path}: group {gname!r} must declare a 'fields' mapping"
            )
        for fld in fields:
            if KEY_SEP in str(fld):
                raise SchemaError(
                    f"schema file {path}: group {gname!r} field {fld!r} contains {KEY_SEP!r} — "
                    "field names must be single dotted-key components (design §2.1); "
                    "regenerate the artifact with salt2 schema dump"
                )
        attrs = node.get("attrs", {})
        if not isinstance(attrs, dict):
            raise SchemaError(f"schema file {path}: group {gname!r} 'attrs' must be a mapping")
        groups[gname] = GroupSchema(
            fields={str(k): str(v) for k, v in fields.items()},
            attrs=dict(attrs),
        )
    attrs = raw.get("attrs", {})
    if not isinstance(attrs, dict):
        raise SchemaError(f"schema file {path}: top-level 'attrs' must be a mapping")
    return Schema(groups=groups, attrs=dict(attrs), schema_version=version)


def save_schema(schema: Schema, path: str | Path) -> None:
    """Write a schema artifact to YAML (inverse of `load_schema`).

    Field and group order is preserved so the artifact diffs cleanly under
    version control (design §2.6: committed next to ``norm_dict.yaml``).
    """
    payload: dict[str, Any] = {
        "schema_version": schema.schema_version,
        "attrs": dict(schema.attrs),
        "groups": {
            name: {"fields": dict(gschema.fields), "attrs": dict(gschema.attrs)}
            for name, gschema in schema.groups.items()
        },
    }
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


# ---------------------------------------------------------------------------
# H5 -> Schema (the only file-touching function in salt.core, design §2.6)
# ---------------------------------------------------------------------------


def dump_schema(h5_path: str | Path) -> Schema:
    """Read a training file's structure into a `Schema` (design §2.6).

    Scrapes what v1 reads ad hoc (``cli.py:425-443``, ``datasets.py:579-607``):
    every top-level structured dataset becomes a group with its field names,
    dtypes, and attrs; global file attrs are kept too. Non-structured
    datasets and nested groups are ignored (the v1 layout is flat structured
    datasets at the file root — see ``salt.utils.inputs.write_dummy_file``).

    Dataset or field names containing ``"."`` are legal in HDF5/numpy but
    cannot be addressed as dotted bundle keys (design §2.1) — they are
    skipped with a warning on stderr so the artifact stays consumable by
    `Schema.keys` and the planner.

    Returns
    -------
    Schema
        The scraped schema, groups sorted by name.

    Raises
    ------
    SchemaError
        If the file cannot be opened as HDF5.
    """
    h5_path = Path(h5_path)
    try:
        h5file = h5py.File(h5_path, "r")
    except OSError as err:
        raise SchemaError(f"cannot open {h5_path} as HDF5: {err}") from err
    with h5file:
        groups: dict[str, GroupSchema] = {}
        for name in sorted(h5file):
            node = h5file[name]
            if not isinstance(node, h5py.Dataset) or node.dtype.names is None:
                continue
            if KEY_SEP in name:
                print(
                    f"WARNING: skipping dataset {name!r} in {h5_path}: names containing "
                    f"{KEY_SEP!r} cannot be addressed as bundle keys (design §2.1)",
                    file=sys.stderr,
                )
                continue
            fields: dict[str, str] = {}
            for fname in node.dtype.names:
                if KEY_SEP in fname:
                    print(
                        f"WARNING: skipping field {name}/{fname!r} in {h5_path}: names "
                        f"containing {KEY_SEP!r} cannot be addressed as bundle keys "
                        "(design §2.1)",
                        file=sys.stderr,
                    )
                    continue
                fields[fname] = _dtype_name(node.dtype[fname])
            attrs = {key: _plain(value) for key, value in node.attrs.items()}
            groups[name] = GroupSchema(fields=fields, attrs=attrs)
        global_attrs = {key: _plain(value) for key, value in h5file.attrs.items()}
    return Schema(groups=groups, attrs=global_attrs)


def _dtype_name(dtype: np.dtype) -> str:
    """Human-readable, reconstructible name for a structured-array field dtype.

    Returns
    -------
    str
        ``np.dtype(name)``-constructible name (``"float32"``); subarray
        fields fall back to ``str(dtype)``.
    """
    if dtype.subdtype is not None:
        return str(dtype)
    return dtype.name


def _plain(value: Any) -> Any:
    """Convert an h5py attr value to plain Python for YAML serialisation.

    Returns
    -------
    Any
        Lists/str/int/float/bool — no numpy types survive.
    """
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value
