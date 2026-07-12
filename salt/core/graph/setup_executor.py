"""Execution of a compiled SETUP plan.

The setup graph's runtime is a small bespoke loop — deliberately not
`Executor.run`. `Executor.run` is hardwired to ``module(view, mode)`` and
validates ``callable(module)`` at construction, but a `DatasetModule.setup`
is a *named method*, not ``__call__``. So the setup pass is a third small loop
(parallel to `GraphDataset.__getitem__`, not the executor) that:

1. walks the compiled setup-plan steps in topological order;
2. calls ``produced = module.setup(ctx, stage)`` for each;
3. merges the produced keys into one shared write-once `Bundle` (the
   ``SetupBundle``) via ``Bundle.merge(produced, who, expected=set(step.produces))``.

The genuinely reused kernel pieces are exactly: the planner's topological
order (`compile_setup_plan`) and `Bundle`'s write-once ``merge``-with-
``expected``. Modules that produce nothing (the default no-op `setup`) simply
return the ctx unchanged and merge nothing.

For ``setup("fit")`` the datamodule runs this loop for both ``"train"`` and
``"val"`` into the same ctx (disjoint stage-qualified keys).
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
