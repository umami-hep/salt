"""`ObjectGroup` / `ObjectGroupField` — the H5 sink's declarative structured
object-group schema, fed from bundle leaves (generic, no MaskFormer knowledge).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from salt.graph.errors import ConfigError

_VALID_KINDS = ("data", "label", "pad_mask")
"""The demand kinds a field's source leaf may declare (must match the producer)."""

_FLOAT_DTYPES = ("f2", "f4", "f8", "float16", "float32", "float64")
"""Float H5 descriptors demoted to ``f2`` under half precision."""


@dataclass(frozen=True)
class ObjectGroupField:
    """One column-family in a structured output group, sourced from a bundle leaf.

    The producer emits a bare ``<leaf>`` tensor; the sink packs its last
    dimension into the named `suffixes` columns (``{run_name}_{suffix}`` when
    `prefix`, else the bare suffix). A single-suffix field wraps a
    channel-less leaf into one column; a multi-suffix field expands the leaf's
    trailing channel axis in `suffixes` order.

    Parameters
    ----------
    leaf : str
        The bundle key to source (e.g. ``objects.class_probs``,
        ``labels.objects.object_class``, ``outputs.tracks.object_index``). The
        sink demands it (keeping its producer alive) and casts it at write time.
    suffixes : Sequence[str]
        The per-channel column suffixes, in last-dim order.
    dtype : str, optional
        The H5 column dtype (numpy descriptor), by default ``"f4"``. A float
        dtype is demoted to ``f2`` when the sink runs half-precision.
    prefix : bool, optional
        Whether to prefix each suffix with ``{run_name}_``, by default True.
    kind : str, optional
        The source leaf's demand kind — one of ``data`` / ``label`` /
        ``pad_mask`` (must match the producer's declared kind), by default
        ``"data"``.

    Raises
    ------
    ConfigError
        For an empty leaf, an empty suffix list, or an unknown kind.
    """

    leaf: str
    suffixes: Sequence[str]
    dtype: str = "f4"
    prefix: bool = True
    kind: str = "data"

    def __post_init__(self) -> None:
        if not self.leaf:
            raise ConfigError("ObjectGroupField needs a non-empty source `leaf` bundle key")
        if not list(self.suffixes):
            raise ConfigError(
                f"ObjectGroupField {self.leaf!r} needs a non-empty `suffixes` list — name the "
                "per-channel columns the leaf's last dimension expands into"
            )
        if self.kind not in _VALID_KINDS:
            raise ConfigError(
                f"ObjectGroupField {self.leaf!r}: kind {self.kind!r} must be one of "
                f"{list(_VALID_KINDS)} (the source leaf's producer kind)"
            )

    def column_names(self, run_name: str) -> list[str]:
        """The H5 column names for this field (run-name prefixed unless bare)."""
        return [f"{run_name}_{s}" if self.prefix else str(s) for s in self.suffixes]

    def resolved_dtype(self, half_precision: bool) -> str:
        """The H5 dtype, float demoted to ``f2`` under half precision."""
        if half_precision and self.dtype in _FLOAT_DTYPES:
            return "f2"
        return self.dtype

    def np_dtype(self, run_name: str, half_precision: bool) -> np.dtype:
        """The structured numpy dtype this field contributes to its group."""
        dt = self.resolved_dtype(half_precision)
        return np.dtype([(col, dt) for col in self.column_names(run_name)])

    @classmethod
    def coerce(cls, obj: ObjectGroupField | Mapping[str, Any]) -> ObjectGroupField:
        """Build from a dataclass or a plain config mapping."""
        if isinstance(obj, ObjectGroupField):
            return obj
        return cls(**dict(obj))


@dataclass(frozen=True)
class ObjectGroup:
    """A declarative structured output group fed from bundle leaves.

    Fully generic (no MaskFormer knowledge). `name` is the H5 group.

    - With `shape` set, the group is a NON-reader output group whose per-row
      trailing shape is `shape` — each entry an ``int``, or a reader stream
      name resolved to that stream's file token length at open time (e.g.
      ``[5, "tracks"]`` -> ``(5, T_tracks)``).
    - With `shape` None, the group targets an EXISTING reader stream (`name`
      must be one) and its fields are per-token columns re-expanded to the
      file token length, appended after the reader/task columns.

    Parameters
    ----------
    name : str
        The H5 group name (a new non-reader group, or an existing reader
        stream when `shape` is None).
    fields : Sequence[ObjectGroupField]
        The column-families, in H5 column order.
    shape : Sequence[int | str] | None, optional
        The non-reader group's trailing per-row shape (ints, or reader stream
        names resolved to token lengths). None (the default) targets a reader
        stream.

    Raises
    ------
    ConfigError
        For an empty name or an empty field list.
    """

    name: str
    fields: Sequence[ObjectGroupField] = field(default_factory=tuple)
    shape: Sequence[int | str] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("ObjectGroup needs a non-empty `name` (the H5 group)")
        if not list(self.fields):
            raise ConfigError(
                f"ObjectGroup {self.name!r} needs a non-empty `fields` list — name the "
                "column-families the group serialises"
            )

    @classmethod
    def coerce(cls, obj: ObjectGroup | Mapping[str, Any]) -> ObjectGroup:
        """Build from a dataclass or a plain config mapping (coercing nested fields)."""
        if isinstance(obj, ObjectGroup):
            return obj
        data = dict(obj)
        data["fields"] = tuple(ObjectGroupField.coerce(f) for f in data.get("fields", ()))
        return cls(**data)
