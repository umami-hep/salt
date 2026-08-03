"""The sink registry: where a run's output sinks are looked up."""

from __future__ import annotations

from typing import Any

from salt.outputs.sinks.sink import Node

__all__ = ["iter_sinks", "register_sink", "sink_registry"]

_REGISTRY_ATTR = "salt_sinks"
"""The trainer attribute the registry lives on."""


def sink_registry(owner: Any) -> list[Any]:
    """The sink list attached to `owner` (a trainer), created on first use.

    Sinks stopped being Lightning callbacks, so ``trainer.callbacks`` is no
    longer where they are found. Every wiring path — the ``callbacks:`` alias,
    the ``outputs:`` section, a command's implicit injection — appends here,
    and every discovery site reads `iter_sinks`.

    Parameters
    ----------
    owner : Any
        The trainer (or any object) the registry hangs off. None yields an
        empty, unattached list.

    Returns
    -------
    list[Any]
        The live registry list; mutating it registers sinks.
    """
    if owner is None:
        return []
    registry = getattr(owner, _REGISTRY_ATTR, None)
    if registry is None:
        registry = []
        try:
            setattr(owner, _REGISTRY_ATTR, registry)
        except AttributeError:  # a slotted / frozen stand-in: nothing to attach to
            return []
    return registry


def register_sink(owner: Any, sink: Any) -> None:
    """Register `sink` on `owner`'s registry, unless the same object is already there.

    Parameters
    ----------
    owner : Any
        The trainer carrying the registry.
    sink : Any
        The sink node to register.
    """
    registry = sink_registry(owner)
    if not any(existing is sink for existing in registry):
        registry.append(sink)


def looks_like_sink(obj: Any) -> bool:
    """Whether `obj` is a sink for discovery purposes.

    A `Node` instance, or an object duck-typed as one — the pre-split code
    selected on ``writer_demand`` / ``bind_output_section``, and test doubles
    and third-party sinks still arrive that way.

    Parameters
    ----------
    obj : Any
        A candidate registry or callbacks-list entry.

    Returns
    -------
    bool
        Whether `obj` should be treated as a sink.
    """
    if isinstance(obj, Node):
        return True
    return callable(getattr(obj, "writer_demand", None)) or callable(
        getattr(obj, "bind_output_section", None)
    )


def iter_sinks(owner: Any) -> list[Any]:
    """Every sink attached to `owner`, registry first.

    The single discovery surface. It also scans ``owner.callbacks`` for
    sink-shaped entries: that is the ``callbacks:`` alias path — a sink handed
    straight to a programmatically built trainer, which is how a config
    declared one before sinks moved into the ``outputs:`` section.

    Parameters
    ----------
    owner : Any
        The trainer to read. None yields an empty list.

    Returns
    -------
    list[Any]
        The attached sinks, in wiring order, deduplicated by identity.
    """
    if owner is None:
        return []
    found: list[Any] = list(sink_registry(owner))
    for entry in getattr(owner, "callbacks", None) or []:
        if looks_like_sink(entry) and not any(seen is entry for seen in found):
            found.append(entry)
    return found
