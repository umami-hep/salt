"""Declared tensor interfaces for the salt v2 graph kernel.

Design §2.2: graph participants declare what they consume and produce as
nested dicts mirroring the bundle layout, with `TensorSpec` leaves. A `Mode`
flag gates ports, and `kind` typing makes mask-polarity and label/feature
mix-ups static type errors. Declarations are static — building them must not
touch data files, the network, or tensors.

Symbolic dims (``"B"``, ``"T:tracks"``, ``"F:tracks"``) are preserved as plain
strings here; unification across the graph happens in the planner (design §3.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Literal, Protocol, TypeAlias, get_args, runtime_checkable

__all__ = [
    "IO",
    "KEY_SEP",
    "KINDS",
    "PRIMARY_MODES",
    "GraphModule",
    "Kind",
    "Mode",
    "NestedSpec",
    "TensorSpec",
    "check_key_component",
    "flatten_spec",
    "is_symbolic_dim",
    "iter_spec_leaves",
    "join_key",
    "split_key",
    "split_symbolic_dim",
    "sym_dim",
    "unflatten_spec",
]

KEY_SEP = "."
"""Separator for dotted bundle keys (design §2.1)."""


class Mode(Flag):
    """Execution modes gating graph ports (design §2.2)."""

    FIT = auto()
    VAL = auto()
    TEST = auto()
    ONNX = auto()
    TRAINING = FIT | VAL
    ALL = FIT | VAL | TEST | ONNX


PRIMARY_MODES: tuple[Mode, ...] = (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX)
"""The four atomic modes, in canonical order (composites excluded)."""

Kind: TypeAlias = Literal["data", "pad_mask", "label", "loss", "meta"]
"""Port kinds: a consumer port can only bind a producer leaf of the same kind (design §2.2)."""

KINDS: tuple[Kind, ...] = get_args(Kind)
"""All valid `Kind` values, for runtime validation."""


# ---------------------------------------------------------------------------
# dotted-key helpers (shared by Bundle and the planner)
# ---------------------------------------------------------------------------


def check_key_component(name: str) -> str:
    """Validate a single dotted-key component (a nested-dict key).

    Components must be non-empty strings without the ``"."`` separator.

    Returns
    -------
    str
        The component, unchanged.

    Raises
    ------
    TypeError
        If `name` is not a string.
    ValueError
        If `name` is empty or contains the key separator.
    """
    if not isinstance(name, str):
        raise TypeError(f"key component must be str, got {type(name).__name__}: {name!r}")
    if not name:
        raise ValueError("key component must be a non-empty string")
    if KEY_SEP in name:
        raise ValueError(f"key component {name!r} must not contain {KEY_SEP!r}")
    return name


def split_key(key: str) -> tuple[str, ...]:
    """Split a dotted key into its components, validating each one.

    ``split_key("preds.jets.cls")`` returns ``("preds", "jets", "cls")``.

    Returns
    -------
    tuple[str, ...]
        The key components.

    Raises
    ------
    TypeError
        If `key` is not a string.
    ValueError
        For empty keys or empty components (``"a..b"``).
    """
    if not isinstance(key, str):
        raise TypeError(f"bundle key must be str, got {type(key).__name__}: {key!r}")
    if not key:
        raise ValueError("bundle key must be a non-empty string")
    parts = tuple(key.split(KEY_SEP))
    if any(not part for part in parts):
        raise ValueError(f"invalid dotted key {key!r}: empty component")
    return parts


def join_key(parts: tuple[str, ...] | list[str]) -> str:
    """Join validated components into a dotted key (inverse of `split_key`).

    Returns
    -------
    str
        The dotted key.

    Raises
    ------
    ValueError
        If `parts` is empty or any component is invalid.
    """
    if not parts:
        raise ValueError("cannot join an empty sequence of key components")
    for part in parts:
        check_key_component(part)
    return KEY_SEP.join(parts)


# ---------------------------------------------------------------------------
# symbolic-dim vocabulary (design §2.2 — strings only, unification in planner)
# ---------------------------------------------------------------------------

_SYM_SEP = ":"


def is_symbolic_dim(dim: int | str) -> bool:
    """Check whether a shape entry is symbolic (``"B"``, ``"T:tracks"``) rather than concrete.

    Returns
    -------
    bool
        True if `dim` is a symbolic dim string.
    """
    return isinstance(dim, str)


def sym_dim(family: str, qualifier: str | None = None) -> str:
    """Build a symbolic dim string, e.g. ``sym_dim("T", "tracks") == "T:tracks"``.

    `family` is the dim family (``"B"`` batch, ``"T"`` tokens, ``"F"`` features, ...);
    `qualifier` optionally names the stream the dim belongs to.

    Returns
    -------
    str
        The symbolic dim string.

    Raises
    ------
    ValueError
        If `family` or `qualifier` is empty or contains ``":"``.
    """
    if not family or _SYM_SEP in family:
        raise ValueError(f"invalid symbolic dim family {family!r}")
    if qualifier is None:
        return family
    if not qualifier or _SYM_SEP in qualifier:
        raise ValueError(f"invalid symbolic dim qualifier {qualifier!r}")
    return f"{family}{_SYM_SEP}{qualifier}"


def split_symbolic_dim(dim: str) -> tuple[str, str | None]:
    """Split a symbolic dim into ``(family, qualifier)``; ``"B"`` gives ``("B", None)``.

    Returns
    -------
    tuple[str, str | None]
        The dim family and the optional stream qualifier.

    Raises
    ------
    ValueError
        If the string is not a valid symbolic dim.
    """
    if not isinstance(dim, str) or not dim:
        raise ValueError(f"invalid symbolic dim {dim!r}: must be a non-empty string")
    family, sep, qualifier = dim.partition(_SYM_SEP)
    if not sep:
        return family, None
    if not family or not qualifier or _SYM_SEP in qualifier:
        raise ValueError(
            f"invalid symbolic dim {dim!r}: expected 'FAMILY' or 'FAMILY{_SYM_SEP}qualifier'"
        )
    return family, qualifier


# ---------------------------------------------------------------------------
# tensor specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Static description of one bundle leaf (design §2.2).

    `shape` mixes concrete ints and symbolic dim strings (``("B", "T:tracks", 23)``);
    None means unconstrained. `fields` carries last-dim column names so column
    lookups resolve by name, never by YAML list position arithmetic.
    """

    shape: tuple[int | str, ...] | None = None
    dtype: str | None = None
    kind: Kind = "data"
    modes: Mode = Mode.ALL
    optional: bool = False
    fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.shape is not None:
            if isinstance(self.shape, str):
                # strings are iterable: shape="BT" would silently become two
                # symbolic dims ("B", "T") instead of an error
                raise TypeError(
                    f"shape must be a sequence of dims, not a bare string: {self.shape!r} "
                    f"(did you mean shape=({self.shape!r},)?)"
                )
            object.__setattr__(self, "shape", tuple(self.shape))
            for dim in self.shape:  # type: ignore[union-attr]
                if isinstance(dim, str):
                    split_symbolic_dim(dim)
                elif isinstance(dim, int):
                    if dim < 0:
                        raise ValueError(f"concrete dim must be non-negative, got {dim}")
                else:
                    raise TypeError(
                        f"shape entries must be int or symbolic str, got {type(dim).__name__}"
                    )
        if self.kind not in KINDS:
            raise ValueError(f"invalid kind {self.kind!r}: must be one of {KINDS}")
        if not isinstance(self.modes, Mode):
            raise TypeError(f"modes must be a Mode flag, got {type(self.modes).__name__}")
        if not self.modes:
            raise ValueError("modes must not be empty: a port active in no mode is dead")
        if not isinstance(self.optional, bool):
            raise TypeError(f"optional must be bool, got {type(self.optional).__name__}")
        if self.fields is not None:
            if isinstance(self.fields, str):
                raise TypeError(
                    f"fields must be a sequence of names, not a bare string: {self.fields!r} "
                    f"(did you mean fields=({self.fields!r},)?)"
                )
            object.__setattr__(self, "fields", tuple(self.fields))
            for name in self.fields:  # type: ignore[union-attr]
                if not isinstance(name, str) or not name:
                    raise ValueError(f"fields must be non-empty strings, got {name!r}")
            last_dim = self.shape[-1] if self.shape else None
            if isinstance(last_dim, int) and len(self.fields) != last_dim:
                raise ValueError(
                    f"fields length {len(self.fields)} does not match concrete last dim {last_dim}"
                )

    def active_in(self, mode: Mode) -> bool:
        """Check whether this port is active in (overlaps) the given mode.

        Returns
        -------
        bool
            True if `self.modes` overlaps `mode`.
        """
        return bool(self.modes & mode)


NestedSpec: TypeAlias = dict[str, "NestedSpec | TensorSpec"]
"""Nested dict mirroring the bundle layout, with `TensorSpec` leaves (design §2.2)."""


def iter_spec_leaves(nested: NestedSpec, prefix: str = "") -> Iterator[tuple[str, TensorSpec]]:
    """Iterate the leaves of a nested spec, depth-first in dict order.

    Component names are validated via `check_key_component` (which raises
    ValueError/TypeError on invalid names).

    Yields
    ------
    tuple[str, TensorSpec]
        ``(dotted_key, spec)`` pairs.

    Raises
    ------
    TypeError
        On leaves that are neither dicts nor `TensorSpec`.
    """
    for name, node in nested.items():
        check_key_component(name)
        key = f"{prefix}{KEY_SEP}{name}" if prefix else name
        if isinstance(node, TensorSpec):
            yield key, node
        elif isinstance(node, dict):
            yield from iter_spec_leaves(node, prefix=key)
        else:
            raise TypeError(
                f"spec leaf at {key!r} must be TensorSpec or nested dict, got {type(node).__name__}"
            )


def flatten_spec(nested: NestedSpec) -> dict[str, TensorSpec]:
    """Flatten a nested spec to dotted keys (design §3.1, port resolution).

    Returns
    -------
    dict[str, TensorSpec]
        ``{dotted_key: spec}`` in depth-first dict order.
    """
    return dict(iter_spec_leaves(nested))


def unflatten_spec(flat: Mapping[str, TensorSpec]) -> NestedSpec:
    """Rebuild a nested spec from dotted keys (inverse of `flatten_spec`).

    Returns
    -------
    NestedSpec
        The nested spec tree.

    Raises
    ------
    ValueError
        If one key is a leaf-prefix of another (``"a.b"`` and ``"a.b.c"``).
    TypeError
        If a value is not a `TensorSpec`.
    """
    nested: NestedSpec = {}
    for key, spec in flat.items():
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"value for {key!r} must be TensorSpec, got {type(spec).__name__}")
        parts = split_key(key)
        node = nested
        for depth, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if isinstance(child, TensorSpec):
                clash = KEY_SEP.join(parts[: depth + 1])
                # structural clash between two declared keys, not a type mismatch
                raise ValueError(f"key {key!r} clashes with leaf {clash!r}")  # noqa: TRY004
            node = child  # type: ignore[assignment]
        leaf = parts[-1]
        if leaf in node:
            raise ValueError(f"key {key!r} clashes with existing subtree or leaf")
        node[leaf] = spec
    return nested


@dataclass(frozen=True)
class IO:
    """A module's declared interface: required and produced nested specs (design §2.2)."""

    requires: NestedSpec = field(default_factory=dict)
    produces: NestedSpec = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate both trees eagerly (component names + leaf types).
        flatten_spec(self.requires)
        flatten_spec(self.produces)


@runtime_checkable
class GraphModule(Protocol):
    """Protocol every graph participant implements (design §2.2).

    `declare_io` is a function of the module's own config only — it must not
    touch data files, the network, or tensors. `name` is the instance name
    (the config dict key).
    """

    name: str

    def declare_io(self, mode: Mode) -> IO:
        """Return the declared requires/produces for the given mode."""
        ...
