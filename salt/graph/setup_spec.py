"""Declared SETUP-time interfaces for the setup-graph kernel: `SourceSpec`
PATH / SCALAR-CONFIG leaves, kept disjoint from the tensor-only `spec.py`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, get_args

from salt.graph.spec import KEY_SEP, check_key_component, split_key

__all__ = [
    "SOURCE_KINDS",
    "SetupIO",
    "SetupStage",
    "SourceKind",
    "SourceNestedSpec",
    "SourceSpec",
    "flatten_source_spec",
    "iter_source_leaves",
    "unflatten_source_spec",
]

SetupStage: TypeAlias = Literal["train", "val", "test"]
"""Lightning-style setup stage.

NOT a `Mode`: ``datamodule.setup("fit")`` builds both the train and val
datasets in one call, so setup granularity is coarser than the per-batch
FIT/VAL `Mode` split. `Mode.ONNX` is never a setup stage (ONNX never runs the
setup pass).
"""

SETUP_STAGES: tuple[SetupStage, ...] = get_args(SetupStage)
"""All valid `SetupStage` values, in canonical order."""

SourceKind: TypeAlias = Literal["path", "scalar"]
"""SETUP-time port kinds, parallel to (and disjoint from) the tensor `Kind`.

PATH = a filesystem path string (a raw sample file, a VDS path, a ``/dev/shm``
staged path). SCALAR = a resolved CONFIG artifact: a dict of str/float, a
float, or an opaque resolved object (e.g. a ``num`` row-cap dict). A consumer
port may only bind a producer leaf of the same kind.
"""

SOURCE_KINDS: tuple[SourceKind, ...] = get_args(SourceKind)
"""All valid `SourceKind` values, for runtime validation."""


@dataclass(frozen=True)
class SourceSpec:
    """Static description of one SETUP-bundle leaf (SETUP-time only).

    A setup leaf is a filesystem PATH (``kind="path"``) or a resolved
    CONFIG/SCALAR artifact (``kind="scalar"``: dict of str/float, a float, or
    an opaque resolved object such as a ``num`` row-cap dict). It has no
    ``shape``, no ``dtype``, no symbolic dims, no ``fields`` — setup leaves are
    not tensors and never enter the per-batch bundle. That omission keeps a
    setup leaf structurally incapable of being mistaken for a `TensorSpec`, so
    `_unify_edge` (which reads ``.shape``/``.dtype``) must never run on a setup
    edge.

    `stages` gates the leaf per setup stage exactly as `TensorSpec.modes` gates
    per `Mode`; `optional` marks a consumed-if-present require.
    """

    kind: SourceKind = "path"
    stages: tuple[SetupStage, ...] = SETUP_STAGES
    optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"invalid source kind {self.kind!r}: must be one of {SOURCE_KINDS}")
        if isinstance(self.stages, str):
            # strings are iterable: stages="train" would silently become five
            # single-char stages instead of an error
            raise TypeError(
                f"stages must be a sequence of stage names, not a bare string: {self.stages!r} "
                f"(did you mean stages=({self.stages!r},)?)"
            )
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("stages must not be empty: a port active in no stage is dead")
        for stage in self.stages:
            if stage not in SETUP_STAGES:
                raise ValueError(f"invalid setup stage {stage!r}: must be one of {SETUP_STAGES}")
        if not isinstance(self.optional, bool):
            raise TypeError(f"optional must be bool, got {type(self.optional).__name__}")

    def active_in(self, stage: SetupStage) -> bool:
        """Whether this port is active in the given setup stage."""
        return stage in self.stages


SourceNestedSpec: TypeAlias = dict[str, "SourceNestedSpec | SourceSpec"]
"""Nested dict mirroring the setup-bundle layout, with `SourceSpec` leaves."""


def iter_source_leaves(
    nested: SourceNestedSpec, prefix: str = ""
) -> Iterator[tuple[str, SourceSpec]]:
    """Iterate the leaves of a nested setup spec, depth-first in dict order.

    The setup-time twin of `iter_spec_leaves`; the shared dotted-key handling
    (`check_key_component`) is reused, only the leaf type differs.

    Yields
    ------
    tuple[str, SourceSpec]
        ``(dotted_key, spec)`` pairs.

    Raises
    ------
    TypeError
        On leaves that are neither dicts nor `SourceSpec`.
    """
    for name, node in nested.items():
        check_key_component(name)
        key = f"{prefix}{KEY_SEP}{name}" if prefix else name
        if isinstance(node, SourceSpec):
            yield key, node
        elif isinstance(node, dict):
            yield from iter_source_leaves(node, prefix=key)
        else:
            raise TypeError(
                f"setup spec leaf at {key!r} must be SourceSpec or nested dict, "
                f"got {type(node).__name__}"
            )


def flatten_source_spec(nested: SourceNestedSpec) -> dict[str, SourceSpec]:
    """Flatten a nested setup spec to dotted keys (the setup-time twin of `flatten_spec`)."""
    return dict(iter_source_leaves(nested))


def unflatten_source_spec(flat: Mapping[str, SourceSpec]) -> SourceNestedSpec:
    """Rebuild a nested setup spec from dotted keys (inverse of `flatten_source_spec`).

    Raises
    ------
    ValueError
        If one key is a leaf-prefix of another (``"a.b"`` and ``"a.b.c"``).
    TypeError
        If a value is not a `SourceSpec`.
    """
    nested: SourceNestedSpec = {}
    for key, spec in flat.items():
        if not isinstance(spec, SourceSpec):
            raise TypeError(f"value for {key!r} must be SourceSpec, got {type(spec).__name__}")
        parts = split_key(key)
        node = nested
        for depth, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if isinstance(child, SourceSpec):
                clash = KEY_SEP.join(parts[: depth + 1])
                raise ValueError(f"key {key!r} clashes with leaf {clash!r}")  # noqa: TRY004
            node = child  # type: ignore[assignment]
        leaf = parts[-1]
        if leaf in node:
            raise ValueError(f"key {key!r} clashes with existing subtree or leaf")
        node[leaf] = spec
    return nested


@dataclass(frozen=True)
class SetupIO:
    """A `SaltDatasetModule`'s SETUP-time interface.

    The setup-time analogue of `IO`: required and produced nested specs with
    `SourceSpec` leaves. A separate type from `IO` (whose ``__post_init__``
    would reject `SourceSpec` leaves), so a PATH/SCALAR leaf can never be
    declared on the per-batch face and vice versa.
    """

    requires: SourceNestedSpec = field(default_factory=dict)
    produces: SourceNestedSpec = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate both trees eagerly (component names + leaf types).
        flatten_source_spec(self.requires)
        flatten_source_spec(self.produces)

    def is_empty(self) -> bool:
        """Whether this declares no setup ports at all (the default no-op face)."""
        return not self.requires and not self.produces
