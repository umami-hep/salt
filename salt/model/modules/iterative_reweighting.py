"""`IterativeReweighting` — a module group expanding into N chained NN-reweighting pairs."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from salt.model.base import ModuleGroup, SaltModelModule
from salt.model.modules.kinematic_weight import KinematicWeight, _check
from salt.model.modules.tasks import ClassificationTaskModule

__all__ = ["IterativeReweighting"]

_OWNED: tuple[str, ...] = ("eta", "chain", "chain_exponents")


class IterativeReweighting(ModuleGroup):
    """``n_iterations`` (prior head, `KinematicWeight`) pairs of iterated NN reweighting.

    Iteration ``k``'s residual is ``r_k = min(w_max, (t/p_k)^eta)``; head ``k`` trains on the
    detached cumulative weight ``C_{k-1} = min(w_max, C_{k-2} r_{k-1})`` (``C_0 = 1``) times
    the prior's own ``sample_weight`` (e.g. a class weight); the last `KinematicWeight`
    outputs the geometric mean of the last `avg_last` cumulative weights, clamped at
    ``w_max``. The fixed point is ``P_w(f|x) = q_f``, the RAW ``q_f`` numerator re-anchoring
    the class fractions. The taper is applied inside the last head's clamp, so the geometric
    mean is exact unless that head's own residual exceeds ``w_max``. With ``n_iterations >
    1`` only targets ``inclusive``/``mean`` are allowed (the harmonic ``Z`` accumulates under
    the unweighted measure).

    Names: the last pair keeps `prior_name` / `weight_name` / ``weight_field`` (N=1 is
    exactly today's pair); pair ``k < N`` is ``<prior_name>_k`` / ``<weight_name>_k`` writing
    the cumulative ``<weight_field>_iter_k`` (FIT/VAL-only prior, no published variables).
    Joint vs sequential is the ``training_schedule``: every head trainable in one sample
    stage, or one sample stage per head with the others frozen; the main stage's ``frozen``
    list must name every head. Members cannot be overridden individually from the CLI —
    override the group's init_args.

    Parameters
    ----------
    n_iterations : int
        Number of (prior, weight) pairs, a non-bool int ``>= 1``.
    prior : dict
        `ClassificationTaskModule` init_args shared by every prior head; with
        ``n_iterations > 1`` its ``loss.init_args.reduction`` must be ``none``.
    weight : dict
        `KinematicWeight` init_args shared by every weight: ``mode: nn``, any ``task_name``
        equal to `prior_name`, no ``eta``/``chain``/``chain_exponents`` (the group's).
    eta : float, optional
        Per-iteration shrinkage exponent in ``(0, 1]``, by default ``1.0``.
    avg_last : int, optional
        Geometric-mean window ``1 <= avg_last <= n_iterations``, by default ``1``.
    prior_name, weight_name : str, optional
        The last pair's names, by default ``kinematic_prior`` / ``kinematic_weight``.

    Raises
    ------
    ConfigError
        On an invalid argument or a combination ruled out above.
    """

    def __init__(
        self,
        *,
        n_iterations: int,
        prior: dict,  # BARE `dict`: jsonargparse keeps the member specs raw
        weight: dict,
        eta: float = 1.0,
        avg_last: int = 1,
        prior_name: str = "kinematic_prior",
        weight_name: str = "kinematic_weight",
    ) -> None:
        super().__init__()
        check = partial(_check, "IterativeReweighting")
        n = n_iterations
        check((type(n) is not int or n < 1, f"n_iterations must be a non-bool int >= 1: {n!r}"))
        check(
            (
                type(avg_last) is not int or not 1 <= avg_last <= n,
                f"avg_last must be a non-bool int in [1, n_iterations={n}], got {avg_last!r}",
            ),
            (not isinstance(eta, int | float) or not 0 < eta <= 1, f"need 0 < eta <= 1: {eta!r}"),
            (not isinstance(prior, Mapping), f"prior must be a mapping, got {prior!r}"),
            (not isinstance(weight, Mapping), f"weight must be a mapping, got {weight!r}"),
        )
        loss, target = prior.get("loss"), weight.get("target", "inclusive")
        loss_args = (loss.get("init_args") or {}) if isinstance(loss, Mapping) else {}
        owned = [k for k in _OWNED if k in weight]
        check(
            (
                weight.get("task_name", prior_name) != prior_name,
                f"weight task_name {weight.get('task_name')!r} must equal {prior_name=!r}",
            ),
            (weight.get("mode") != "nn", f"weight mode must be 'nn', got {weight.get('mode')!r}"),
            (bool(owned), f"weight must not set {owned} — the group owns them"),
            (
                n > 1 and loss_args.get("reduction") != "none",
                "n_iterations > 1 needs prior.loss.init_args.reduction: none (a list "
                "sample_weight requires it)",
            ),
            (
                n > 1 and target not in {"inclusive", "mean"},
                f"target {target!r} with n_iterations > 1: their Z normalisation accumulates "
                "under the unweighted measure; iterate with target inclusive or mean",
            ),
        )
        self.n_iterations, self.avg_last, self.eta = n, avg_last, float(eta)
        self.prior, self.weight = dict(prior), dict(weight)
        self.prior_name, self.weight_name = prior_name, weight_name

    def expand(self) -> dict[str, SaltModelModule]:
        """The ordered ``{p_1: prior, w_1: weight, ..., p_N: prior, w_N: weight}`` members."""
        k_max, m = self.n_iterations, self.avg_last
        wf = self.weight.get("weight_field") or "nn_kinematic_weight"
        sw = self.prior.get("sample_weight")
        base_sw = [] if sw is None else [sw] if isinstance(sw, str) else list(sw)

        def field(j: int) -> str:
            return wf if j == k_max else f"{wf}_iter_{j}"

        members: dict[str, SaltModelModule] = {}
        for k in range(1, k_max + 1):
            last = k == k_max
            p_k = self.prior_name if last else f"{self.prior_name}_{k}"
            w_k = self.weight_name if last else f"{self.weight_name}_{k}"
            prior: dict[str, Any] = dict(self.prior)
            if k >= 2:
                prior["sample_weight"] = [*base_sw, field(k - 1)]
            if not last and "expose" not in prior:
                prior["expose"] = ["fit", "val"]
            weight = {**self.weight, "task_name": p_k, "weight_field": field(k), "eta": self.eta}
            if not last:
                weight["variables"] = None  # published variables only on the last pair
                if k >= 2:
                    weight |= {"chain": [field(k - 1)], "chain_exponents": [1.0]}
            elif k_max > 1:
                chain = [field(j) for j in range(k_max - max(m, 2) + 1, k_max)]
                exps = [*([1.0 / m] * (len(chain) - 1)), min(1.0, 2.0 / m)]
                weight |= {"chain": chain, "chain_exponents": exps, "eta": self.eta / m}
            members[p_k] = ClassificationTaskModule(**prior)
            members[w_k] = KinematicWeight(**weight)
        return members
