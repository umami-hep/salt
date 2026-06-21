"""Output sinks — terminal consumers of the ``outputs.*`` dict (design §2 layer 2).

A sink is a Lightning callback that consumes the producers' ``outputs.*``
leaves and does something terminal with them. Sinks are independent and
composable: each takes, as config, the list of output names it wants.

The load-bearing mechanism is **demand**: a sink declares the ``outputs.*``
keys it needs via `writer_demand`, the duck-typed surface `SaltModule`
already consumes for the M4.5 `WriterCallback` (saltmodule.py
``_attached_writer`` / ``_boundary_demand``; design §8). Those keys become the
TEST plan's sinks, so the planner keeps the producers (and transitively the
``preds.*`` they read) alive in TEST while FIT/VAL prune them (design §2, §4
risk 4 — the demand-gating keystone). The sink need not serialise anything to
anchor demand; this P0 sink is a no-op collector that records the keys it saw
per batch, proving the demand path without pulling in the H5 serialisation
concerns deferred to the P1 `H5OutputWriter`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightning import Callback

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import KEY_SEP

__all__ = ["CollectOutputs"]

_OUTPUTS_NAMESPACE = "outputs"
"""The bundle namespace this sink demands from (design §2.1)."""


class CollectOutputs(Callback):
    """Minimal TEST-only sink: anchor demand on named ``outputs.*`` leaves.

    The smallest sink that proves the producer -> ``outputs.*`` -> sink demand
    path (design §4b P0). It declares the configured ``outputs.*`` keys as its
    demand (`writer_demand`), which `SaltModule` folds into the TEST plan's
    sinks; at test time it collects the demanded leaves per batch into
    `collected` (a no-op stand-in for the P1 `H5OutputWriter`'s serialisation).

    Parameters
    ----------
    outputs : Sequence[str]
        The ``outputs.<stream>.<name>`` keys this sink consumes — each must be
        a concrete (wildcard-free) key under the ``outputs`` namespace.

    Raises
    ------
    ConfigError
        For an empty list, a wildcard key, or a key outside the ``outputs``
        namespace (the sink consumes producer leaves, not raw predictions).
    """

    def __init__(self, outputs: Sequence[str]) -> None:
        super().__init__()
        keys = list(outputs or [])
        if not keys:
            raise ConfigError(
                "CollectOutputs needs a non-empty outputs list — name the "
                "outputs.<stream>.<name> leaves to consume (design §2 layer 2)"
            )
        for key in keys:
            parts = key.split(KEY_SEP)
            if any(part in {"*", "**"} for part in parts):
                raise ConfigError(
                    f"CollectOutputs output key {key!r} contains a wildcard — sink "
                    "demand keys are concrete (design §2.2)"
                )
            if parts[0] != _OUTPUTS_NAMESPACE:
                raise ConfigError(
                    f"CollectOutputs output key {key!r} is not under the "
                    f"{_OUTPUTS_NAMESPACE!r} namespace — sinks consume producer "
                    "outputs.* leaves, not raw predictions (design §2)"
                )
        self._outputs = tuple(keys)
        #: Per-batch list of the demanded leaves seen at test time (no-op
        #: collector; replaced by real serialisation in the P1 H5OutputWriter).
        self.collected: list[dict[str, Any]] = []

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` keys (read-only view).

        Returns
        -------
        tuple[str, ...]
            The configured keys, in declaration order.
        """
        return self._outputs

    def writer_demand(self, model_modules: Any, reader: Any) -> dict[str, str]:
        """The TEST demand this sink anchors (duck-typed `SaltModule` surface, design §8).

        Returns the configured ``outputs.*`` keys mapped to a §4.1-grade
        demander description, exactly the shape `SaltModule._boundary_demand`
        consumes from the M4.5 `WriterCallback`. The arguments mirror that
        contract and are unused here (this sink demands fixed model-produced
        keys, not reader-derived ones).

        Returns
        -------
        dict[str, str]
            ``{outputs key: "sink 'CollectOutputs' demanding <key>"}`` in
            declaration order.
        """
        del model_modules, reader
        return {key: f"sink 'CollectOutputs' demanding {key}" for key in self._outputs}

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Bundle,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect the demanded ``outputs.*`` leaves for one batch (no-op sink).

        The executed bundle carries the producers' ``outputs.*`` leaves
        (the demand kept them alive); this records them for the P0 collector
        contract. Replaced by real H5 serialisation in P1.
        """
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.collected.append({key: outputs.get(key) for key in self._outputs})
