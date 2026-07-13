"""`CollectOutputs` — the minimal TEST-only demand-anchoring sink."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightning import Callback

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import KEY_SEP

_OUTPUTS_NAMESPACE = "outputs"
"""The bundle namespace this sink demands from."""


class CollectOutputs(Callback):
    """Minimal TEST-only sink: anchor demand on named ``outputs.*`` leaves.

    Declares the configured ``outputs.*`` keys as its demand (`writer_demand`),
    which `SaltModule` folds into the TEST plan's sinks; at test time it
    collects the demanded leaves per batch into `collected` (a no-op stand-in
    for a real serialising sink).

    Parameters
    ----------
    outputs : Sequence[str]
        The ``outputs.<stream>.<name>`` keys this sink consumes — each must be
        a concrete (wildcard-free) key under the ``outputs`` namespace.

    Raises
    ------
    ConfigError
        For an empty list, a wildcard key, or a key outside the ``outputs``
        namespace.
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
        #: collector).
        self.collected: list[dict[str, Any]] = []

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` keys, in declaration order."""
        return self._outputs

    def writer_demand(self, model_modules: Any, reader: Any) -> dict[str, str]:
        """The TEST demand this sink anchors (duck-typed `SaltModule` surface)."""
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
        """Collect the demanded ``outputs.*`` leaves for one batch (no-op sink)."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.collected.append({key: outputs.get(key) for key in self._outputs})
