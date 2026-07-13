"""Exception hierarchy for the salt v2 graph kernel.

Planner-stage errors (connectivity, config, cycles, ...) are appended to this
module by the planner stage — see the anchor comment at the bottom.
"""

from __future__ import annotations

__all__ = [
    "AllModesDeadError",
    "ConfigError",
    "ConnectivityError",
    "CycleError",
    "DeclarationError",
    "GraphError",
    "KeyCollisionError",
    "KindError",
    "MutationError",
    "SchemaError",
    "ShapeError",
    "UndeclaredAccessError",
]

_SUGGESTION_CUTOFF = 0.5
"""`difflib.get_close_matches` cutoff for did-you-mean suggestions in error messages."""


class GraphError(Exception):
    """Base class for all errors raised by the salt.core graph kernel."""


class KeyCollisionError(GraphError):
    """Write-once violation: a bundle key was written more than once.

    Raised by `Bundle.set` / `Bundle.merge` when a dotted key (or a subtree /
    leaf prefix of it) already exists in the bundle.
    """


class DeclarationError(GraphError):
    """A module's returned key set does not match its declared produces.

    Raised by `Bundle.merge` on every merge — the executor passes the
    producing module's declared key set and the comparison is always on.
    """


# planner-stage errors appended below


class ConfigError(GraphError):
    """A module configuration is structurally invalid.

    Raised by the planner for config bugs that are not connectivity problems:
    instance-name mismatches, wildcard patterns on non-framework modules or in
    `requires`, non-concrete source/sink keys, or compiling a composite mode.
    """


class ConnectivityError(GraphError):
    """A required key has no producer, or a key has more than one.

    Messages name the consumer, the missing key, nearest-key suggestions, and
    modules that could produce the key in another mode. Also raised when a
    wildcard-narrowed key fails schema validation.
    """


class CycleError(GraphError):
    """The dependency graph contains a cycle.

    Cycles are always a config error — there is no fixpoint execution. The
    message names the cycle as a key-level chain. Also raised when a wildcard
    producer's narrowed outputs transitively feed its own inputs.
    """


class KindError(GraphError):
    """A consumer port's kind does not match its producer leaf's kind.

    Kind typing makes mask-polarity and label/feature mix-ups static type
    errors, not runtime surprises.
    """


class ShapeError(GraphError):
    """Shape or dtype unification failed across a graph edge.

    Symbolic dims are unified globally at validate time; the message names
    both endpoints and the conflicting sizes.
    """


class AllModesDeadError(ConfigError):
    """A configured module is dead in every mode.

    Per-mode pruning is a feature; a module whose outputs reach no sink in
    *any* mode is a config bug and fails validation loudly — silently dropping
    a configured selection or augmentation step would corrupt physics.
    """


class SchemaError(GraphError):
    """A demanded field is absent from the dataset schema.

    Reserved for the schema artifact tooling and the reader's bind-time check
    (``salt2 schema``); the planner reports schema-invalid wildcard narrowing
    as `ConnectivityError`.
    """


# executor-stage errors


class UndeclaredAccessError(GraphError):
    """An undeclared bundle access by a module under debug execution.

    Raised by the executor's read-tracking bundle view (``run(debug=True)``)
    when a module reads or probes a key absent from its declared requires for
    the running mode — the message names the module, the key, and the
    declaration to amend. Also raised when a module mutates its bundle view
    or grabs the raw payload: produced keys must be returned, not written.
    """


class MutationError(GraphError):
    """A module mutated a bundle tensor in place under debug execution.

    Bundle keys are write-once, but key-level checks cannot see ``tensor.add_``
    style in-place edits that corrupt an upstream key for later consumers.
    Under ``run(debug=True)`` the executor snapshots ``torch.Tensor._version``
    for every bundle leaf around each step and raises this error — naming the
    module and the key — when a version counter bumps. Read-time mutation is
    sanctioned only inside the reader, never in graph modules.
    """
