"""LossSum and LossGLS loss-combination GraphModules."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    GraphModule,
    Mode,
    TensorSpec,
    flatten_spec,
    unflatten_spec,
)
from salt.core.nn.base import SaltModelModule


class LossSum(SaltModelModule):
    """Weighted sum of per-task losses -> ``loss.total``.

    Per-task weights are applied INSIDE the tasks, so the default here is a
    plain sum; `weights` is an extra per-loss-key multiplier.

    The ``losses.**`` auto-collection is a framework wildcard: since the
    kernel rejects wildcard *requires*, narrowing happens framework-side —
    `collect_loss_keys` scans sibling modules' declared produces and `narrow`
    fixes the concrete key list before plan compilation. An explicit
    ``losses:`` config list skips collection entirely.
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; ``losses=None`` defers to framework narrowing.

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected via `collect_loss_keys`/`narrow`).
        weights : Mapping[str, float] | None, optional
            Per-loss multipliers keyed like `losses`, default 1.0 each.
        """
        super().__init__()
        self._loss_keys: tuple[str, ...] | None = (
            tuple(_loss_key(k) for k in losses) if losses is not None else None
        )
        self.weights = {_loss_key(k): float(v) for k, v in (weights or {}).items()}
        if self._loss_keys is not None:
            self._check_weight_keys()

    def _check_weight_keys(self) -> None:
        """Reject weight entries that name no summed loss key."""
        assert self._loss_keys is not None
        if unknown := sorted(set(self.weights) - set(self._loss_keys)):
            raise ConfigError(
                f"LossSum: weights for unknown loss keys {unknown} — summed keys are "
                f"{list(self._loss_keys)}"
            )

    @property
    def narrowed(self) -> bool:
        """Whether the loss-key list is fixed (explicit config or `narrow`)."""
        return self._loss_keys is not None

    @staticmethod
    def collect_loss_keys(
        modules: Mapping[str, GraphModule], mode: Mode = Mode.FIT
    ) -> tuple[str, ...]:
        """Scan sibling modules (LossSum instances skipped) for declared ``losses.*`` produces.

        Returns all declared loss keys, in module-dict declaration order.
        """
        keys: list[str] = []
        for module in modules.values():
            if isinstance(module, LossSum):
                continue
            for key, spec in flatten_spec(module.declare_io(mode).produces).items():
                if key.startswith("losses.") and spec.active_in(mode):
                    keys.append(key)
        return tuple(keys)

    def narrow(self, loss_keys: Iterable[str]) -> None:
        """Fix the concrete loss-key list (framework-side ``losses.**`` narrowing).

        Raises
        ------
        ConfigError
            If explicit ``losses:`` config already fixed the keys, the list
            is empty, or a configured weight names no key.
        """
        if self._loss_keys is not None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys already fixed to {list(self._loss_keys)}"
            )
        keys = tuple(_loss_key(k) for k in loss_keys)
        if not keys:
            raise ConfigError(
                f"LossSum {self.name!r}: narrowed to an empty loss-key list — no module "
                "declares a losses.* produce (design §3.3)"
            )
        self._loss_keys = keys
        self._check_weight_keys()

    def declare_io(self, mode: Mode) -> IO:
        """Declare the narrowed loss keys -> ``loss.total`` (TRAINING only); raises if unfixed."""
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})
        if self._loss_keys is None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys not fixed — pass losses: in config or let "
                "the framework narrow via collect_loss_keys()/narrow() before compile "
                "(losses.** is a framework wildcard, design §3.3)"
            )
        loss_spec = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        return IO(
            requires=unflatten_spec(dict.fromkeys(self._loss_keys, loss_spec)),
            produces=unflatten_spec({"loss.total": loss_spec}),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Sum the (optionally weighted) loss leaves -> ``{"loss.total": scalar}``."""
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        total = sum(self.weights.get(key, 1.0) * b.get(key) for key in self._loss_keys)
        return {"loss.total": total}


class LossGLS(LossSum):
    """Geometric-mean (GLS) combination of per-task losses -> ``loss.total``.

    Subclasses `LossSum` to share the ``losses.**`` framework wildcard,
    `declare_io`, and the `collect_loss_keys`/`narrow` integration. The ONLY
    behavioural change is the combination rule (`forward`): the n-task
    geometric mean ``(∏ losses)^(1/n)``.

    GLS does NOT utilise loss weights — the geometric mean of per-task losses
    is only meaningful when no task is pre-scaled (a weighted task loss
    ``w*L`` would contribute ``w^(1/n)`` to the product, an arbitrary rescale
    of the mean). Two weight surfaces are both guarded to 1.0: this module's
    per-loss ``weights`` (rejected in ``__init__``), and the task-side
    ``weight`` applied inside each task head (checked via
    `check_task_weights`, called by `SaltModule.__init__` before any plan
    compiles, since this module can't see sibling task state at construction).
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; reject any per-loss weight (GLS ignores weights).

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected, as for `LossSum`).
        weights : Mapping[str, float] | None, optional
            Accepted only for parity with the `LossSum` signature: any entry
            != 1.0 is rejected.

        Raises
        ------
        ConfigError
            If any configured weight is not 1.0 (GLS ignores weights — set
            them to 1, or use `LossSum` for a weighted sum).
        """
        super().__init__(losses=losses, weights=weights)
        # exact == 1.0 is the faithful semantic; weights are config literals,
        # never computed values
        if bad := {k: v for k, v in self.weights.items() if v != 1.0}:  # noqa: RUF069
            raise ConfigError(
                f"LossGLS: per-loss weights are not utilised by the geometric mean — got "
                f"{bad}; set all weights to 1.0, or use LossSum for a weighted sum "
                "(v1 modelwrapper.py:139-142)"
            )

    @staticmethod
    def check_task_weights(modules: Mapping[str, GraphModule]) -> None:
        """Assert every loss-producing task carries ``weight == 1.0``.

        Called by `SaltModule.__init__` when a `LossGLS` is present, BEFORE
        any `declare_io`/compile, so a weighted task under GLS fails loudly at
        assembly rather than silently rescaling the geometric mean. Modules
        without a numeric ``weight`` attribute are ignored. Raises
        `ConfigError` naming each offending task.
        """
        offenders = {
            name: float(module.weight)
            for name, module in modules.items()
            if not isinstance(module, LossSum)
            # duck-typed numeric check: LossSum carries `weights` (a dict), not
            # `weight`, and is excluded above regardless
            and isinstance(getattr(module, "weight", None), (int, float))
            and float(module.weight) != 1.0  # noqa: RUF069 - exact, the v1 semantic
        }
        if offenders:
            raise ConfigError(
                f"LossGLS: GLS does not utilise task weights — set all task weights to 1.0, "
                f"got {offenders} (v1 modelwrapper.py:139-142)"
            )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Combine the loss leaves by their geometric mean: ``(∏ losses)^(1/n)``."""
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        product = math.prod(b.get(key) for key in self._loss_keys)
        return {"loss.total": torch.pow(product, 1.0 / len(self._loss_keys))}


def _loss_key(key: str) -> str:
    """Normalise a configured loss reference to a dotted ``losses.`` key."""
    return key if key.startswith("losses.") else f"losses.{key}"
