"""Execution of a compiled SETUP plan: walk steps in topological order, call
``module.setup(ctx, stage)``, merge into a shared write-once bundle.
"""

from __future__ import annotations

from salt.core.graph.bundle import Bundle
from salt.core.graph.executor import canonical_produced
from salt.core.graph.planner import Plan
from salt.core.graph.setup_spec import SetupStage

__all__ = ["run_setup_plan"]


def run_setup_plan(plan: Plan, stage: SetupStage, ctx: Bundle | None = None) -> Bundle:
    """Execute a compiled setup `plan` for `stage`, threading one write-once ctx.

    Walks ``plan.steps`` in topological order, calling
    ``module.setup(ctx, stage)`` and merging its produced keys into `ctx`
    under write-once + declaration checks (the setup analogue of
    `Executor.run`). A module may either merge into `ctx` itself and return
    it (detected by identity, nothing to add), or return only its newly
    produced dict, merged here against the step's declared ``produces`` as
    the expected key set. `ctx` defaults to a fresh `Bundle`; passing an
    existing one is how ``setup("fit")`` accumulates both ``"train"`` and
    ``"val"`` into one ctx.

    Raises `KeyCollisionError` on a write-once violation, `DeclarationError`
    on a produces mismatch.
    """
    if ctx is None:
        ctx = Bundle()
    for step in plan.steps:
        produced = step.module.setup(ctx, stage)  # type: ignore[attr-defined]
        # case (a): module merged into ctx itself and returned it — nothing to do.
        if produced is ctx:
            continue
        # case (b): module returned only its newly produced dict — the loop owns
        # the write-once merge against the step's declared key set.
        expected = set(step.produces)
        ctx.merge(
            canonical_produced(produced, expected, step.name),
            who=step.name,
            expected=expected,
        )
    return ctx
