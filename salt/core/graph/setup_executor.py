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
    ``module.setup(ctx, stage)`` on each module's live instance and merging its
    produced keys into `ctx` under write-once + declaration checks (the setup
    analogue of `Executor.run`).

    A module may either (a) merge into `ctx` itself and ``return ctx``, in
    which case this loop adds nothing, or (b) ``return`` only its newly
    produced dict, which this loop merges. Case (a) is detected by identity
    (``produced is ctx``); case (b) is merged with the step's declared
    ``produces`` as the ``expected`` key set — so a module that produces an
    extra/missing key fails loudly with its name, exactly like the per-batch
    executor.

    Parameters
    ----------
    ctx : Bundle | None, optional
        The shared write-once setup bundle to populate. A fresh `Bundle` is
        created when None; passing an existing one is how ``setup("fit")``
        accumulates both ``"train"`` and ``"val"`` into one ctx.

    Raises
    ------
    KeyCollisionError
        If a setup module writes a key already present (write-once).
    DeclarationError
        If a module's returned dict does not match its declared setup-produces.
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
