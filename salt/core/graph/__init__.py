"""salt.core.graph — the pure graph kernel (design §2, §3).

Public API: bundle, declared interfaces, planner, executor, and the
exception hierarchy.
"""

from __future__ import annotations

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import (
    AllModesDeadError,
    ConfigError,
    ConnectivityError,
    CycleError,
    DeclarationError,
    GraphError,
    KeyCollisionError,
    KindError,
    MutationError,
    SchemaError,
    ShapeError,
    UndeclaredAccessError,
)
from salt.core.graph.executor import Executor
from salt.core.graph.planner import (
    SINKS,
    SOURCES,
    DeadOutput,
    Edge,
    Plan,
    PlanStep,
    Sinks,
    compile_plan,
    deadcode,
)
from salt.core.graph.spec import (
    IO,
    KEY_SEP,
    KINDS,
    PRIMARY_MODES,
    GraphModule,
    Kind,
    Mode,
    NestedSpec,
    TensorSpec,
    check_key_component,
    flatten_spec,
    is_symbolic_dim,
    iter_spec_leaves,
    join_key,
    split_key,
    split_symbolic_dim,
    sym_dim,
    unflatten_spec,
)

__all__ = [
    "IO",
    "KEY_SEP",
    "KINDS",
    "PRIMARY_MODES",
    "SINKS",
    "SOURCES",
    "AllModesDeadError",
    "Bundle",
    "ConfigError",
    "ConnectivityError",
    "CycleError",
    "DeadOutput",
    "DeclarationError",
    "Edge",
    "Executor",
    "GraphError",
    "GraphModule",
    "KeyCollisionError",
    "Kind",
    "KindError",
    "Mode",
    "MutationError",
    "NestedSpec",
    "Plan",
    "PlanStep",
    "SchemaError",
    "ShapeError",
    "Sinks",
    "TensorSpec",
    "UndeclaredAccessError",
    "check_key_component",
    "compile_plan",
    "deadcode",
    "flatten_spec",
    "is_symbolic_dim",
    "iter_spec_leaves",
    "join_key",
    "split_key",
    "split_symbolic_dim",
    "sym_dim",
    "unflatten_spec",
]
