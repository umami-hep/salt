"""`ClassWeight` — running per-class jet fraction -> class-balancing per-jet loss weight."""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.model.modules.kinematic_weight import _all_reduce
from salt.utils.logging import get_logger

__all__ = ["ClassWeight"]

log = get_logger(__name__)

_EMPTY = "ClassWeight %r: class index %d had zero accumulated jets — its weight is capped at w_max"
_FROZEN = "ClassWeight %r: q frozen at global step %d (epoch %d) from %d jets: %s"


class ClassWeight(SaltModelModule):
    """Class-balancing per-jet loss weight from the running per-class jet fractions ``q̂``.

    Produces ``labels.<stream>.<out>`` = ``w_cls_f = 1/(K q̂_f)`` clipped to ``[0, w_max]``,
    whose jet-mean is 1 identically (``Σ_f q̂_f/(K q̂_f) = 1``). ``q̂`` is the running class
    fraction of the stage-1 `Mode.FIT` batches (rank-reduced every batch, floored at `p_floor`
    while running, frozen unfloored at `freeze_accumulators`, the floor re-applied when the
    weight is formed), bit-identical to a `KinematicWeight` accumulating the same batches.
    A class-balanced prior learns ``P(f|x)/q_f``, so pair it with
    ``KinematicWeight(balanced_prior=True)``. TRAINING-only and label-only (no preds dependency,
    so it can sit before the prior head in the plan), no parameters; consumed as one entry of a
    task's ``sample_weight`` list, e.g. ``[nn_kinematic_weight, flavour_class_weight]``.

    Parameters
    ----------
    n_classes : int
        Number of classes ``K`` (non-bool int >= 2). Required.
    stream, label, out : str, optional
        The stream, its ``labels.<stream>.<label>`` class index and the produced field
        ``labels.<stream>.<out>``; default ``jets``/``flavour_label``/``flavour_class_weight``.
    p_floor, w_max : float, optional
        Class-fraction floor in ``(0, 1)``, default ``1e-4``; weight ceiling ``> 0``, default
        ``50.0`` (a class with zero accumulated jets is capped at `w_max`).

    Raises
    ------
    ConfigError
        On an invalid argument.
    RuntimeError
        From `freeze_accumulators` when nothing was accumulated.
    """

    counts: Tensor
    q: Tensor
    frozen: Tensor

    def __init__(
        self,
        *,
        n_classes: int,
        stream: str = "jets",
        label: str = "flavour_label",
        out: str = "flavour_class_weight",
        p_floor: float = 1e-4,
        w_max: float = 50.0,
    ) -> None:
        super().__init__()
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            msg = f"n_classes must be a non-bool int >= 2, got {n_classes!r}"
            raise ConfigError(f"ClassWeight: {msg}")
        if not 0 < p_floor < 1:
            raise ConfigError(f"ClassWeight: p_floor must satisfy 0 < p_floor < 1, got {p_floor}")
        if not w_max > 0:
            raise ConfigError(f"ClassWeight: w_max must be > 0, got {w_max}")
        self.n_classes, self.stream, self.label, self.out = n_classes, stream, label, out
        self.p_floor, self.w_max = float(p_floor), float(w_max)
        self.register_buffer("counts", torch.zeros(n_classes, dtype=torch.float64))
        self.register_buffer("q", torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float32))
        self.register_buffer("frozen", torch.tensor(False))

    @property
    def label_key(self) -> str:
        """The class-index label key ``labels.<stream>.<label>``."""
        return f"labels.{self.stream}.{self.label}"

    @property
    def weight_key(self) -> str:
        """The produced weight key ``labels.<stream>.<out>``."""
        return f"labels.{self.stream}.{self.out}"

    @property
    def accumulators_frozen(self) -> bool:
        """Whether `freeze_accumulators` has run."""
        return bool(self.frozen)

    def declare_io(self, mode: Mode) -> IO:
        """Empty outside `Mode.TRAINING`: a loss weight has no meaning without a loss."""
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})
        label = partial(TensorSpec, shape=("B",), kind="label", modes=Mode.TRAINING)
        requires = {self.label_key: label(dtype="int64")}
        produces = {self.weight_key: label(dtype="float32")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Weight each row by its class; a `Mode.FIT` call first accumulates the batch."""
        k = self.n_classes
        y = b.get(self.label_key).long()
        valid = (y >= 0) & (y < k)
        if not bool(self.frozen) and mode is Mode.FIT:
            with torch.no_grad():
                fresh = torch.bincount(y[valid], minlength=k).to(torch.float64)
                _all_reduce(fresh)
                self.counts += fresh
                if (total := self.counts.sum()) > 0:
                    self.q.copy_((self.counts / total).clamp_min(self.p_floor).to(torch.float32))
        w_cls = (1.0 / (k * self.q.clamp_min(self.p_floor))).clamp(0.0, self.w_max)
        out = torch.where(valid, w_cls[y.clamp(0, k - 1)], torch.ones_like(w_cls[:1]))
        return {self.weight_key: out.detach().to(torch.float32)}

    def freeze_accumulators(self, global_step: int, epoch: int) -> None:
        """Freeze the unfloored running ``q̂`` at a stage boundary; idempotent."""
        if bool(self.frozen):
            return
        if (total := self.counts.sum()) == 0:
            raise RuntimeError(
                f"ClassWeight {self.name!r}: freeze_accumulators called with no accumulated "
                "stage-1 train batches (counts all zero) — the class weight is undefined"
            )
        with torch.no_grad():
            self.q.copy_((self.counts / total).to(torch.float32))
            for class_idx in (c for c, n in enumerate(self.counts.tolist()) if n == 0):
                log.warning(_EMPTY, self.name, class_idx)
            self.frozen.fill_(True)
        prior = [round(v, 6) for v in self.q.tolist()]
        log.info(_FROZEN, self.name, global_step, epoch, int(total), prior)
