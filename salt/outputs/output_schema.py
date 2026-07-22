"""Output-schema value objects — the sink-side serialisation contracts the
graph trace cannot recover (per-column suffixes, dtypes, run-name prefix,
structured object groups) plus the shared output-suffix constants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

from salt.graph.errors import ConfigError
from salt.graph.spec import KEY_SEP

__all__ = [
    "VERTEX_INDEX",
    "ObjectGroup",
    "ObjectGroupField",
    "OutputColumn",
    "OutputField",
    "pascal_case",
]

_OUTPUTS_NAMESPACE = "outputs"
"""The bundle namespace the output sinks demand from (``outputs.*`` leaves)."""

VERTEX_INDEX = "VertexIndex"
"""The vertexing output suffix — shared by TEST and ONNX.

TEST: the eval-H5 column (bare while ``prefix_vertex_column=false``,
v1 byte parity; ``{run_name}_VertexIndex`` once that flag flips). ONNX:
``{model_name}_VertexIndex``. One constant, one per-mode prefix rule, one
compat flag.
"""


def pascal_case(name: str) -> str:
    """Snake-case to Pascal-case — the default ONNX aux-output suffix rule.

    E.g. ``track_origin -> TrackOrigin``, ``track_type -> TrackType``.
    Overridable per task via the `TaskWriter` ``onnx_names:`` mapping. Each
    ``_``-separated part is upper-cased at its first letter only, so e.g.
    ``track_pVtx`` stays ``TrackPVtx``-free (that would require an explicit
    ``onnx_names`` entry).
    """
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


@dataclass(frozen=True)
class OutputField:
    """One output column a producer contributes to a sink's field manifest.

    ``h5_name``/``onnx_name`` can diverge (e.g. a per-token classification head
    writes per-class H5 probability columns but an ONNX argmax index) — either
    may be ``None`` if the field has no representation in that sink.

    Parameters
    ----------
    h5_name : str | None
        H5 column suffix (run-name-prefixed by the sink unless ``prefix=False``).
        ``None`` if the field has no H5 representation.
    onnx_name : str | None
        Flat ONNX output suffix; defaults to `h5_name`. ``None`` if the field
        has no ONNX representation.
    dtype : str
        Serialisation dtype (numpy descriptor for H5, mapped to the ONNX
        equivalent via `onnx_dtype`).
    axis : str
        ``"global"`` (jet-level scalar) or ``"per_token"`` (sequence column).
    final : bool
        ``False`` for an intermediate leaf consumed only by a downstream node
        (sinks must not auto-collect it); ``True`` (default) for a
        written/exported leaf.
    prefix : bool
        Whether the H5 column is ``{run_name}_{h5_name}`` or the bare
        `h5_name` (e.g. an unprefixed ``VertexIndex`` column).
    value : Tensor | None
        The graph-visible converted tensor. ``None`` for the static manifest
        path (name/dtype/axis minted before any forward); filled by the
        task-side ``get_output`` path with the converted torch tensor.
    """

    h5_name: str | None
    onnx_name: str | None = None
    dtype: str = "f4"
    axis: str = "global"
    final: bool = True
    prefix: bool = True
    value: Tensor | None = None

    def __post_init__(self) -> None:
        if self.axis not in {"global", "per_token"}:
            raise ConfigError(
                f"OutputField axis must be 'global' or 'per_token', got {self.axis!r}"
            )
        if self.h5_name is None and self.onnx_name is None:
            raise ConfigError(
                "OutputField needs at least one of h5_name / onnx_name (a field with neither "
                "has no representation in any sink)"
            )

    @property
    def resolved_onnx_name(self) -> str | None:
        """The ONNX suffix, defaulting to `h5_name` when not explicitly set."""
        return self.onnx_name if self.onnx_name is not None else self.h5_name

    @property
    def onnx_dtype(self) -> str:
        """The ONNX dtype namespace mapping of `dtype` (``f4 -> float32``, ``i1/i8 -> int8``)."""
        return "int8" if self.dtype in {"i1", "i8", "int8"} else "float32"


@dataclass(frozen=True)
class OutputColumn:
    """One ``outputs.*`` leaf's declarative H5 column schema.

    The producer emits a bare ``outputs.<stream>.<name>`` tensor; the sink
    owns the serialisation schema the trace cannot recover — the per-column
    suffixes, the H5 dtype, and the run-name prefix — without the sink
    reaching into the task.

    Parameters
    ----------
    key : str
        The ``outputs.<stream>.<name>`` leaf this column declares. Concrete
        and under the ``outputs`` namespace.
    suffixes : Sequence[str]
        Per-channel logical suffixes the leaf's last dim expands into, in
        last-dim order (e.g. ``["pb", "pc", "pu"]``). The H5 column is
        ``{run_name}_{suffix}`` when `prefix` (default), else the bare
        `suffix`.
    dtype : str, optional
        The H5 column dtype (numpy descriptor), by default ``"f4"``.
    prefix : bool, optional
        Whether to prefix each suffix with ``{run_name}_``, by default True
        (False for a bare-column family like ``VertexIndex``).

    Raises
    ------
    ConfigError
        For a non-``outputs`` key, a wildcard key, or an empty suffix list.
    """

    key: str
    suffixes: Sequence[str]
    dtype: str = "f4"
    prefix: bool = True

    def __post_init__(self) -> None:
        parts = self.key.split(KEY_SEP)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"OutputColumn key {self.key!r} contains a wildcard — sink demand keys "
                "are concrete (design §2.2)"
            )
        if parts[0] != _OUTPUTS_NAMESPACE:
            raise ConfigError(
                f"OutputColumn key {self.key!r} is not under the {_OUTPUTS_NAMESPACE!r} "
                "namespace — sinks consume producer outputs.* leaves, not raw predictions "
                "(design §2)"
            )
        if not list(self.suffixes):
            raise ConfigError(
                f"OutputColumn {self.key!r} needs a non-empty suffix list — name the "
                "per-channel columns the leaf expands into (design §1 declarative table)"
            )

    @property
    def stream(self) -> str:
        """The leaf's stream (``outputs.<stream>.<name>`` middle segment)."""
        return self.key.split(KEY_SEP)[1]

    def column_names(self, run_name: str) -> list[str]:
        """The H5 column names for this leaf (run-name prefixed unless bare)."""
        return [f"{run_name}_{s}" if self.prefix else str(s) for s in self.suffixes]

    def np_dtype(self, run_name: str) -> np.dtype:
        """The structured numpy dtype this leaf contributes to its group."""
        return np.dtype([(col, self.dtype) for col in self.column_names(run_name)])


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
