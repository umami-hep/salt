"""Runtime tensor bundle: a nested dict of tensors with dotted-path addressing
and write-once semantics.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any

from salt.core.graph.errors import DeclarationError, KeyCollisionError
from salt.core.graph.spec import KEY_SEP, check_key_component, split_key

__all__ = ["Bundle"]

_MISSING = object()


class Bundle:
    """Nested dict of tensors with dotted-path access and write-once semantics.

    All mutation goes through `set` / `merge`, which enforce write-once: a key
    (or a subtree/leaf prefix of it) may never be written twice. Leaf values
    may themselves be plain dicts (e.g. the ``seq.layout`` meta leaf) — they
    are stored opaquely and never confused with bundle structure.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Create a bundle, optionally adopting an initial nested dict.

        Dict values in `data` are treated as structure (dict-valued *leaves*
        must be added via `set`/`merge`); empty dicts contribute no keys.
        """
        self._data: dict[str, Any] = {}
        self._leaves: dict[str, Any] = {}  # dotted key -> leaf value, insertion-ordered
        self._nodes: set[str] = set()  # dotted keys that are structure (subtree roots)
        if data:
            self._adopt(data, prefix="")

    # -- read access --------------------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        """The underlying plain nested dict (live reference — do not mutate).

        Returns
        -------
        dict[str, Any]
            The nested payload dict.
        """
        return self._data

    def get(self, key: str) -> Any:
        """Return the leaf value at a dotted key, e.g. ``b.get("preds.jets.cls")``.

        Returns
        -------
        Any
            The leaf value (tensor, ndarray, or other plain payload).

        Raises
        ------
        KeyError
            If the key is missing, names a subtree rather than a leaf, or
            descends through an existing leaf.
        """
        split_key(key)
        value = self._leaves.get(key, _MISSING)
        if value is not _MISSING:
            return value
        if key in self._nodes:
            raise KeyError(f"bundle key {key!r} is a subtree, not a leaf; use subtree()")
        blocker = self._leaf_prefix_of(key)
        if blocker is not None:
            raise KeyError(f"bundle key {key!r} not found: {blocker!r} is a leaf, cannot descend")
        raise KeyError(f"bundle key {key!r} not found")

    def subtree(self, prefix: str) -> dict[str, Any]:
        """Return a copy of the nested dict under `prefix` (leaf values shared).

        The returned structure is freshly built, so mutating it cannot bypass
        the bundle's write-once semantics.

        Returns
        -------
        dict[str, Any]
            A fresh nested dict; leaf values are shared with the bundle.

        Raises
        ------
        KeyError
            If `prefix` is unknown or names a leaf rather than a subtree.
        """
        split_key(prefix)
        if prefix in self._leaves:
            raise KeyError(f"bundle key {prefix!r} is a leaf, not a subtree; use get()")
        if prefix not in self._nodes:
            raise KeyError(f"bundle subtree {prefix!r} not found")
        head = prefix + KEY_SEP
        out: dict[str, Any] = {}
        for key, value in self._leaves.items():
            if not key.startswith(head):
                continue
            parts = key[len(head) :].split(KEY_SEP)
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return out

    def keys(self) -> list[str]:
        """Return all flattened dotted leaf keys.

        Returns
        -------
        list[str]
            Dotted leaf keys, in insertion order.
        """
        return list(self._leaves)

    def __contains__(self, key: str) -> bool:
        # Mirror get(): malformed probe keys fail loudly instead of reading as
        # a permanent absence (a typo'd optional-port probe would otherwise be
        # silently False forever).
        split_key(key)
        return key in self._leaves

    def __len__(self) -> int:
        return len(self._leaves)

    def __repr__(self) -> str:
        return f"Bundle(keys={list(self._leaves)})"

    # -- write access (write-once) ------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a leaf at a dotted key; raises KeyCollisionError if the key exists.

        Write-once: collisions with existing leaves, existing subtrees, or
        leaf prefixes of `key` all raise KeyCollisionError.
        """
        parts = split_key(key)
        self._check_writable(key, who=None)
        self._insert(parts, key, value)

    def merge(self, produced: dict[str, Any], who: str, expected: AbstractSet[str]) -> None:
        """Merge a module's produced nested dict into the bundle (executor-only).

        Write-once enforced, and the produced key set is checked against the
        producing module's declaration on EVERY merge — the executor passes
        the declared key set as `expected` (Bundle stays decoupled from
        planner types). The check-then-insert order makes the merge atomic:
        on error, no produced key has been written. A produced key colliding
        with the bundle raises KeyCollisionError.

        Raises
        ------
        DeclarationError
            If the produced key set does not equal `expected`.
        """
        expected_set = set(expected)
        flat = _flatten_produced(produced, expected_set, prefix="")
        got = set(flat)
        if got != expected_set:
            missing = sorted(expected_set - got)
            extra = sorted(got - expected_set)
            raise DeclarationError(
                f"module {who!r} returned keys that do not match its declared produces: "
                f"missing={missing} unexpected={extra}"
            )
        # Pre-check every key before inserting any, so a collision leaves the
        # bundle untouched. Keys within one merge cannot collide with each
        # other (they come from a single nested dict).
        for key in flat:
            self._check_writable(key, who=who)
        for key, value in flat.items():
            self._insert(tuple(key.split(KEY_SEP)), key, value)

    # -- internals -----------------------------------------------------------

    def _adopt(self, data: dict[str, Any], prefix: str) -> None:
        """Recursively adopt an initial nested dict (dicts = structure)."""
        for name, value in data.items():
            check_key_component(name)
            key = f"{prefix}{KEY_SEP}{name}" if prefix else name
            if isinstance(value, dict):
                self._adopt(value, prefix=key)
            else:
                self._insert(tuple(key.split(KEY_SEP)), key, value)

    def _leaf_prefix_of(self, key: str) -> str | None:
        """Return the existing leaf that is a proper dotted prefix of `key`, if any.

        Returns
        -------
        str | None
            The blocking leaf key, or None.
        """
        parts = key.split(KEY_SEP)
        for depth in range(1, len(parts)):
            prefix = KEY_SEP.join(parts[:depth])
            if prefix in self._leaves:
                return prefix
        return None

    def _check_writable(self, key: str, who: str | None) -> None:
        """Check that writing `key` would not violate write-once.

        Raises
        ------
        KeyCollisionError
            If `key`, a subtree at `key`, or a leaf prefix of `key` exists.
        """
        via = f" (while merging from module {who!r})" if who is not None else ""
        if key in self._leaves:
            raise KeyCollisionError(f"bundle key {key!r} already exists (write-once){via}")
        if key in self._nodes:
            raise KeyCollisionError(
                f"cannot set {key!r}: it already exists as a subtree (write-once){via}"
            )
        blocker = self._leaf_prefix_of(key)
        if blocker is not None:
            raise KeyCollisionError(
                f"cannot set {key!r}: {blocker!r} is already a leaf (write-once){via}"
            )

    def _insert(self, parts: tuple[str, ...], key: str, value: Any) -> None:
        """Insert a pre-validated leaf into both the nested payload and the index."""
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        self._leaves[key] = value
        for depth in range(1, len(parts)):
            self._nodes.add(KEY_SEP.join(parts[:depth]))


def _flatten_produced(produced: dict[str, Any], expected: set[str], prefix: str) -> dict[str, Any]:
    """Flatten a produced nested dict to dotted keys, guided by the declaration.

    A dict value whose dotted path is itself in `expected` is a declared
    dict-valued leaf (e.g. ``seq.layout``) and is kept whole; undeclared dicts
    are treated as structure and descended for precise mismatch reporting.
    Empty undeclared dicts are recorded as leaves so they surface as
    unexpected keys rather than vanishing silently.

    Returns
    -------
    dict[str, Any]
        ``{dotted_key: value}`` in nested-dict iteration order.
    """
    flat: dict[str, Any] = {}
    for name, value in produced.items():
        check_key_component(name)
        key = f"{prefix}{KEY_SEP}{name}" if prefix else name
        if key in expected or not isinstance(value, dict) or not value:
            flat[key] = value
        else:
            flat.update(_flatten_produced(value, expected, prefix=key))
    return flat
