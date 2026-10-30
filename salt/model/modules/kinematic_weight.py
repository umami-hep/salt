"""`KinematicWeight` — stage-1 kinematic-prior posterior -> per-jet loss weight."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.model.base import SaltModelModule
from salt.model.bind import BindError, ResolvedSchema
from salt.utils.logging import get_logger

__all__ = ["KinematicWeight", "VariableSpec", "parse_variables"]

log = get_logger(__name__)

_MODES: tuple[str, ...] = ("nn", "binned")
_TARGETS: tuple[str, ...] = ("inclusive", "mean", "harmonic", "harmonic_equal")
_HARMONIC: tuple[str, ...] = ("harmonic", "harmonic_equal")

_SPARSE = "KinematicWeight %r: %d of %d cells hold fewer than %d jets (%.3f of accumulated jets)"
_FROZEN = "KinematicWeight %r: q_f frozen at global step %d (epoch %d) from %d jets: %s%s"


def _check(*rules: tuple[bool, str]) -> None:
    """Raise a `ConfigError` for the first violated ``(bad, message)`` rule, in order."""
    for bad, msg in rules:
        if bad:
            raise ConfigError(f"KinematicWeight: {msg}")


def _all_reduce(t: Tensor) -> None:
    """Sum `t` across ranks in place when a multi-rank process group is initialised."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


@dataclass(frozen=True)
class VariableSpec:
    """One reweight variable: continuous (binned on `edges`) or categorical (`categories`).

    Continuous bins the raw label times `scale` (then ``|x|`` if `abs`) on `edges`, an infinite
    first/last edge being an overflow bin; categorical indexes `categories` by an int64 code.
    """

    name: str
    edges: tuple[float, ...] | None = None
    scale: float = 1.0
    abs: bool = False
    categories: tuple[str, ...] | None = None

    @property
    def categorical(self) -> bool:
        return self.categories is not None

    @property
    def n_bins(self) -> int:
        """Number of bins (continuous) or categories (categorical)."""
        return len(self.categories) if self.categories is not None else len(self.edges or ()) - 1

    @property
    def dtype(self) -> str:
        return "int64" if self.categorical else "float32"

    @property
    def finite_edges(self) -> tuple[float, ...]:
        """The edges with infinite (overflow) entries dropped; ``()`` for a categorical axis."""
        return tuple(e for e in self.edges or () if math.isfinite(e))

    def transform(self, x: Tensor) -> Tensor:
        """Map a raw column onto this axis: scaled (optionally ``|x|``) floats, or codes."""
        if self.categorical:
            return x.long()
        x = x.float() * self.scale
        return x.abs() if self.abs else x


def _parse_one(name: str, entry: Any, owner: str) -> VariableSpec:
    where = f"{owner}: variable {name!r}"
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {entry!r}")
    if ("edges" in entry) == ("categories" in entry):
        raise ConfigError(f"{where} needs exactly one of edges (continuous) or categories")
    allowed = {"edges", "scale", "abs"} if "edges" in entry else {"categories"}
    if unknown := sorted(set(entry) - allowed):
        raise ConfigError(f"{where}: unknown key(s) {unknown}, allowed {sorted(allowed)}")
    if "categories" in entry:
        cats = tuple(str(c) for c in entry["categories"])
        if len(cats) < 2 or len(set(cats)) != len(cats):
            raise ConfigError(f"{where}: categories needs >= 2 unique names, got {list(cats)}")
        return VariableSpec(name, categories=cats)
    edges = tuple(float(e) for e in entry["edges"])
    if len(edges) < 2 or not all(b > a for a, b in pairwise(edges)):
        raise ConfigError(f"{where}: edges must be strictly increasing, >= 2, got {list(edges)}")
    # Strict increase already pins -inf to the first and +inf to the last position.
    if any(math.isinf(e) for e in edges[1:-1]):
        raise ConfigError(f"{where}: only the first/last edge may be infinite, got {list(edges)}")
    scale = float(entry.get("scale", 1.0))
    if not (math.isfinite(scale) and scale > 0):
        raise ConfigError(f"{where}: scale must be finite and > 0, got {scale}")
    return VariableSpec(name, edges=edges, scale=scale, abs=bool(entry.get("abs", False)))


def parse_variables(
    spec: Mapping[str, Mapping[str, Any]] | None, owner: str
) -> dict[str, VariableSpec]:
    """Validate a ``variables:`` mapping into `VariableSpec`s (`ConfigError` names `owner`)."""
    if not isinstance(spec, Mapping) or not spec:
        raise ConfigError(f"{owner}: variables must be a non-empty mapping, got {spec!r}")
    return {str(name): _parse_one(str(name), entry, owner) for name, entry in spec.items()}


def cell_index(
    values: Mapping[str, Tensor], specs: Mapping[str, VariableSpec], edges: Mapping[str, Tensor]
) -> tuple[Tensor, int]:
    """Row-major flat cell index ``(B,)`` int64 over already-transformed `values`, and n_cells.

    `edges`: continuous name -> float32 edges. Values beyond the outer edges clamp into the edge
    bins (a live loss weight must not crash on one stray jet); a categorical code outside
    ``[0, len(categories))`` raises `ValueError`.
    """
    idx = torch.zeros((), dtype=torch.int64)
    for spec in specs.values():
        x = values[spec.name]
        if spec.categorical:
            bad = (x < 0) | (x >= spec.n_bins)
            if bool(bad.any()):
                raise ValueError(
                    f"variable {spec.name!r}: code(s) {torch.unique(x[bad]).tolist()} outside "
                    f"[0, {spec.n_bins}) for categories {list(spec.categories or ())}"
                )
        else:
            x = (torch.bucketize(x, edges[spec.name], right=True) - 1).clamp(0, spec.n_bins - 1)
        idx = idx * spec.n_bins + x
    return idx.to(torch.int64), math.prod(s.n_bins for s in specs.values())


class KinematicWeight(SaltModelModule):
    """Turn a class posterior ``P(g|x)`` into a per-jet loss weight ``w_f(x) = t(x)/P(f|x)``.

    Produces ``labels.<stream>.<weight_field>`` in-graph, consumable as a task's
    ``sample_weight``. ``mode="nn"`` reads ``P`` off the live head ``preds.<stream>.<task_name>``;
    ``mode="binned"`` from an N-dim table of per-cell class counts over `variables`. ``q_f`` is
    configured (frozen from construction) or, with ``q_f: null`` plus `n_classes`, the per-class
    bincount; counts accumulate over the stage-1 FIT batches (rank-reduced every batch; ``q_f``
    meanwhile the running marginal floored at `p_floor`) and freeze at `freeze_accumulators`.

    With ``pi = q_f``, ``K`` classes and ``P`` clipped to ``[p_floor, 1]``, `target` gives

    - ``inclusive``: ``w = pi_f / P(f|x)``
    - ``mean``: ``w = pi_f / (K P(f|x)) * sum_g P(g|x) / pi_g``
    - ``harmonic`` / ``harmonic_equal``: ``w = pi_f / (Z P(f|x) S(x))``, with
      ``S = sum_g alpha_g pi_g / P(g|x)``, ``alpha = pi`` resp. ``1``, ``Z = E_x[1/S]``

    so the per-class mean weight is 1 for every target; ``w`` is clipped to ``[0, w_max]``.
    ``Z`` accumulates alongside ``q_f`` and is checkpointed and frozen with it (binned: exact
    from the table; nn: estimated while the prior still trains — a known bias, acceptable as
    ``Z`` only affects `w_max` clipping). A class with zero accumulated jets freezes to
    ``q_f = 0`` (weight 0) for ``inclusive``; any other target raises. With `balanced_prior`
    the recovered posterior is floored at `p_floor` and ``w`` clamped at `w_max` exactly as
    without it; a class weight consumed alongside (``ClassWeight``, clamped there) multiplies
    in the task head and the product is never re-clamped.

    Iterated reweighting: head k's `KinematicWeight` emits ``r_k = min(w_max, (t/p_k)^eta)``;
    through `chain` it multiplies in the previous cumulative weight, giving
    ``C_k = min(w_max, C_{k-1} r_k)``, and the LAST one carries `chain_exponents` tapered so the
    output is the geometric mean of the last `avg_last` cumulative weights (see
    `IterativeReweighting`). ``eta == 1`` and no chain is bitwise the single-iteration weight.

    Parameters
    ----------
    task_name : str, optional
        ``mode="nn"``: the prior head's ``model.modules`` key, by default ``"kinematic_prior"``.
    mode : {"nn", "binned"}
        Posterior source: the live prediction head or the accumulated table. Required.
    stream, label : str, optional
        Stream and its ``labels.<stream>.<label>`` class index, default ``jets``/``flavour_label``.
    weight_field : str | None, optional
        The produced field; ``None`` (default) derives ``f"{mode}_kinematic_weight"``.
    q_f, n_classes : Sequence[float] | None, int | None, optional
        Exactly one: a configured marginal (>= 2 entries > 0 summing to 1), or the class
        count (non-bool int >= 2) to accumulate it; ``binned`` and the harmonic targets need
        `n_classes`.
    variables : Mapping[str, Mapping[str, Any]] | None, optional
        Spec (see `parse_variables`); ``nn`` optional (publish only), ``binned`` required
        (table axes + publish). Each variable is published as
        ``labels.<stream>.<weight_field>_<name>`` (continuous after scale/abs, float32;
        categorical int64 codes) for `salt.callbacks.KinematicReweightPlotter`; an unconsumed
        published key is a planner ``DeadOutput`` warning (error under ``--strict`` validate).
    target : {"inclusive", "mean", "harmonic", "harmonic_equal"}, optional
        Weight target, by default ``"inclusive"``.
    p_floor, w_max : float, optional
        Posterior floor in ``(0, 1)``, default ``1e-4``; weight ceiling ``> 0`` (``inf``
        disables clipping), default ``50.0``.
    min_cell_count : int, optional
        ``binned``: freeze warns about cells holding fewer jets, by default ``100``.
    onehot : str, optional
        Categorical codes are decoded by argmax over their exact one-hot block of
        ``inputs.<onehot>`` (fields ``<variable>_<category>``, validated at `bind`).
    balanced_prior : bool, optional
        The prior head was trained class-balanced (its `sample_weight` is a `ClassWeight`), so
        it learned ``P_bal(f|x)`` proportional to ``P(f|x)/q_f``; recover
        ``P(f|x) = q_f P_bal(f|x) / sum_g q_g P_bal(g|x)`` before the `p_floor` clamp, using
        the running ``q_f`` during the priors stage and the frozen one after. ``nn`` only.
        Default ``False``.
    eta : float, optional
        Per-iteration shrinkage exponent in ``(0, 1]`` on the own factor, before the clamp.
    chain : Sequence[str] | None, optional
        Earlier weight fields ``<field>`` multiplied in after the clamp (then re-clamped).
    chain_exponents : Sequence[float] | None, optional
        One exponent ``> 0`` per `chain` field, by default all 1.

    Raises
    ------
    ConfigError
        On an invalid argument or a combination ruled out above.
    RuntimeError
        From `freeze_accumulators`: nothing accumulated, or a zero non-inclusive prior.
    """

    # Buffers (registered in __init__) declared for mypy;
    # otherwise nn.Module.__getattr__ types them `Tensor | Module`.
    q_f: Tensor
    q_f_counts: Tensor
    q_f_frozen: Tensor
    z_sum: Tensor
    z: Tensor
    cell_counts: Tensor

    def __init__(
        self,
        task_name: str = "kinematic_prior",
        *,
        mode: Literal["nn", "binned"],
        stream: str = "jets",
        label: str = "flavour_label",
        weight_field: str | None = None,
        q_f: Sequence[float] | None = None,
        n_classes: int | None = None,
        variables: Mapping[str, Mapping[str, Any]] | None = None,
        target: Literal["inclusive", "mean", "harmonic", "harmonic_equal"] = "inclusive",
        p_floor: float = 1e-4,
        w_max: float = 50.0,
        min_cell_count: int = 100,
        onehot: str = "container_onehot",
        balanced_prior: bool = False,
        eta: float = 1.0,
        chain: Sequence[str] | None = None,
        chain_exponents: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        acc = "the accumulate form (q_f: null + n_classes)"
        xor = (
            "give exactly one of q_f (a configured target marginal) or n_classes (accumulate "
            "q_f from stage-1 train batches), got "
        )
        _check(
            (mode not in _MODES, f"mode must be one of {_MODES}, got {mode!r}"),
            (target not in _TARGETS, f"target must be one of {_TARGETS}, got {target!r}"),
            (not 0 < p_floor < 1, f"p_floor must satisfy 0 < p_floor < 1, got {p_floor}"),
            (not w_max > 0, f"w_max must be > 0, got {w_max}"),
            (
                type(min_cell_count) is not int or min_cell_count < 1,
                f"min_cell_count must be a non-bool int >= 1, got {min_cell_count!r}",
            ),
            (q_f is not None and n_classes is not None, xor + "both"),
            (q_f is None and n_classes is None, xor + "neither"),
            (mode == "binned" and variables is None, "mode='binned' requires variables"),
            (mode == "binned" and q_f is not None, f"mode='binned' requires {acc}"),
            (
                q_f is not None and target in _HARMONIC,
                f"targets harmonic/harmonic_equal need {acc} so Z can be frozen alongside q_f",
            ),
            (
                balanced_prior and mode == "binned",
                "balanced_prior needs mode='nn' (a trained prior)",
            ),
            (not 0 < eta <= 1, f"eta must satisfy 0 < eta <= 1, got {eta}"),
            (chain is not None and len(chain) == 0, "chain must name at least one field"),
            (
                chain_exponents is not None
                and (chain is None or len(chain_exponents) != len(chain)),
                "chain_exponents must pair one exponent with every chain field",
            ),
            (any(e <= 0 for e in chain_exponents or ()), "chain_exponents must be > 0"),
        )
        if q_f is not None:
            q = [float(x) for x in q_f]
            _check(
                (len(q) < 2, f"q_f must have at least 2 entries, got {len(q)}"),
                (any(x <= 0 for x in q), f"q_f entries must all be > 0, got {q}"),
                (abs(sum(q) - 1.0) > 1e-3, f"q_f must sum to 1 (within 1e-3), got {sum(q)}"),
            )
        elif isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ConfigError(
                f"KinematicWeight: n_classes must be a non-bool int >= 2, got {n_classes!r}"
            )
        else:
            q = [1.0 / n_classes] * n_classes

        self.task_name = task_name
        self.mode: Literal["nn", "binned"] = mode
        self.stream = stream
        self.label = label
        self.weight_field = weight_field if weight_field is not None else f"{mode}_kinematic_weight"
        self.p_floor = float(p_floor)
        self.w_max = float(w_max)
        self.target: str = target
        self.min_cell_count = min_cell_count
        self.balanced_prior = bool(balanced_prior)
        self.eta = float(eta)
        self.chain: tuple[str, ...] = tuple(chain or ())
        exps = chain_exponents if chain_exponents is not None else (1.0,) * len(self.chain)
        self.chain_exponents: tuple[float, ...] = tuple(float(e) for e in exps)
        self.n_classes = k = len(q)
        self.register_buffer("q_f", torch.tensor(q, dtype=torch.float32))
        self.register_buffer("q_f_counts", torch.zeros(k, dtype=torch.float64))
        self.register_buffer("q_f_frozen", torch.tensor(q_f is not None))
        self.register_buffer("z_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("z", torch.ones((), dtype=torch.float64))

        self.onehot, owner = onehot, "KinematicWeight"
        self.variables = None if variables is None else parse_variables(variables, owner)
        self.categorical = tuple(n for n, s in (self.variables or {}).items() if s.categorical)
        self._onehot_cols: dict[str, slice] | None = None  # set at bind
        self.n_cells = 0
        if mode == "binned" and self.variables is not None:  # binned requires variables
            self.n_cells = math.prod(s.n_bins for s in self.variables.values())
            # Persistent (default): checkpointed, so a resumed stage 1 keeps its table.
            self.register_buffer("cell_counts", torch.zeros(self.n_cells, k, dtype=torch.float64))
            for name, spec in self.variables.items():
                if not spec.categorical:  # config-derived: non-persistent, follows .to(device)
                    edges = torch.tensor(spec.edges, dtype=torch.float32)
                    self.register_buffer(f"edges_{name}", edges, persistent=False)

    @property
    def label_key(self) -> str:
        """The class-index label key ``labels.<stream>.<label>``."""
        return f"labels.{self.stream}.{self.label}"

    @property
    def weight_key(self) -> str:
        """The produced weight key ``labels.<stream>.<weight_field>``."""
        return f"labels.{self.stream}.{self.weight_field}"

    @property
    def published(self) -> dict[str, str]:
        """Variable name -> its published key ``labels.<stream>.<weight_field>_<name>``."""
        return {name: f"{self.weight_key}_{name}" for name in self.variables or {}}

    @property
    def chain_keys(self) -> tuple[str, ...]:
        """The chained weight keys ``labels.<stream>.<field>``, in `chain` order."""
        return tuple(f"labels.{self.stream}.{f}" for f in self.chain)

    def declare_io(self, mode: Mode) -> IO:
        """Empty outside `Mode.TRAINING`: a loss weight has no meaning without a loss."""
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})

        def label(dtype: str) -> TensorSpec:
            return TensorSpec(shape=("B",), dtype=dtype, kind="label", modes=Mode.TRAINING)

        requires = {self.label_key: label("int64")}
        if self.mode == "nn":
            requires[f"preds.{self.stream}.{self.task_name}"] = TensorSpec(
                shape=("B", self.n_classes), dtype="float32"
            )
        for key in self.chain_keys:
            requires[key] = label("float32")
        specs = self.variables or {}
        for spec in specs.values():
            if not spec.categorical:
                requires[f"labels.{self.stream}.{spec.name}"] = label("float32")
        if self.categorical:
            requires[f"inputs.{self.onehot}"] = TensorSpec(
                shape=("B", sym_dim("F", self.name)), dtype="float32"
            )
        produces = {self.weight_key: label("float32")}
        produces.update({key: label(specs[n].dtype) for n, key in self.published.items()})
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Locate each categorical variable's exact one-hot block in ``inputs.<onehot>``."""
        self._onehot_cols = cols = dict[str, slice]()
        if not self.categorical:
            return
        key, specs = f"inputs.{self.onehot}", self.variables or {}
        try:
            fields = schema.fields_of(key)
        except BindError as err:
            raise ConfigError(
                f"KinematicWeight {self.name!r}: {key!r} carries no field names, so categorical "
                f"variables {list(self.categorical)} cannot be decoded"
            ) from err
        for name in self.categorical:
            # Exact block: a spec omitting a category would silently mis-decode by argmax.
            want = [f"{name}_{c}" for c in specs[name].categories or ()]
            got = [f for f in fields if f.startswith(f"{name}_")]
            i = fields.index(want[0]) if want[0] in fields else 0
            if list(fields[i : i + len(want)]) != want or len(got) != len(want):
                odd = [f for f in want if f not in got] + [f for f in got if f not in want]
                raise ConfigError(
                    f"KinematicWeight {self.name!r}: categorical variable {name!r} needs exactly "
                    f"the contiguous spec-order block {want} in {key!r}; found {got}, "
                    f"missing/offending {odd}; available: {list(fields)}"
                )
            cols[name] = slice(i, i + len(want))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Any]:
        """Compute the `target` weight ``clip(t(x) / P(y|x), 0, w_max)``, detached.

        ``label == -1`` rows get a harmless weight (the loss zeroes them). An accumulate-mode
        `Mode.FIT` call before `freeze_accumulators` first adds the batch to the counts (and
        ``Z``), so its own weight already uses the updated state; with `balanced_prior`, the nn
        posterior is first multiplied by ``q_f`` and renormalised.
        """
        k = self.n_classes
        y = b.get(self.label_key).long()
        y_idx = y.clamp(0, k - 1)
        valid = (y >= 0) & (y < k)
        specs = self.variables or {}
        values = self._values(b, specs)
        cell: Tensor | None = None
        if self.mode == "binned":
            edges = {n: self.get_buffer(f"edges_{n}") for n, s in specs.items() if s.edges}
            cell, _ = cell_index(values, specs, edges)

        accumulate = not bool(self.q_f_frozen) and mode is Mode.FIT
        if accumulate:
            with torch.no_grad():
                self._accumulate(y[valid], None if cell is None else cell[valid])

        if self.mode == "nn":
            p = torch.softmax(b.get(f"preds.{self.stream}.{self.task_name}"), dim=-1)
            if self.balanced_prior:  # undo the class balancing the prior was trained with
                p = p * self.q_f
                p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        else:
            rows = self.cell_counts[cell]
            p = (rows / rows.sum(-1, keepdim=True).clamp_min(1.0)).float()  # empty cell -> 0
        p = p.clamp(self.p_floor, 1.0)
        p_f = p.gather(1, y_idx[:, None]).squeeze(1)
        pi, pi_f = self.q_f, self.q_f[y_idx]
        if self.target == "inclusive":
            w = pi_f / p_f
        elif self.target == "mean":
            w = pi_f / (k * p_f) * (p / pi).sum(-1)
        else:
            s = self._harmonic_sum(p)
            if accumulate:
                with torch.no_grad():
                    # Needs P from the just-updated counts: a 2nd reduce (inclusive/mean: one).
                    z_inc = (1.0 / s[valid].double()).sum().reshape(1)
                    _all_reduce(z_inc)
                    self.z_sum += z_inc[0]
                    total = self.q_f_counts.sum()
                    if total > 0:
                        self.z.fill_(self.z_sum / total)  # running Z (first batch: its mean)
            w = pi_f / (self.z * p_f * s)
        if self.eta != 1.0:
            w = w.pow(self.eta)
        w = w.clamp(0.0, self.w_max)
        for key, e in zip(self.chain_keys, self.chain_exponents, strict=True):
            c = b.get(key).to(w.dtype)
            w = w * (c if e == 1.0 else c.pow(e))
        if self.chain:
            w = w.clamp(0.0, self.w_max)
        out = {self.weight_key: w.detach().to(torch.float32)}
        for name, key in self.published.items():
            dtype = torch.int64 if specs[name].categorical else torch.float32
            out[key] = values[name].detach().to(dtype)
        return out

    def _values(self, b: Bundle, specs: Mapping[str, VariableSpec]) -> dict[str, Tensor]:
        """Per-variable axis values: continuous labels after scale/abs, argmax-decoded codes."""
        if self.categorical and self._onehot_cols is None:
            raise RuntimeError(f"KinematicWeight {self.name!r}: bind() before forward")
        cols, onehot = self._onehot_cols or {}, f"inputs.{self.onehot}"
        return {
            name: b.get(onehot)[:, cols[name]].argmax(-1).to(torch.int64)
            if spec.categorical
            else spec.transform(b.get(f"labels.{self.stream}.{name}"))
            for name, spec in specs.items()
        }

    def _accumulate(self, y: Tensor, cell: Tensor | None) -> None:
        """Add one FIT batch's valid labels (binned: and their cells) to the counts."""
        k = self.n_classes
        fresh = [torch.bincount(y, minlength=k).to(torch.float64)]
        if cell is not None:
            table = torch.zeros(self.n_cells * k, dtype=torch.float64, device=y.device)
            fresh.append(table.scatter_add_(0, cell * k + y, torch.ones_like(y, dtype=table.dtype)))
        flat = torch.cat(fresh)
        # DDP's default broadcast_buffers=True overwrites every rank's buffers from rank 0 at
        # each forward, so reducing only at the freeze would drop ranks 1..W-1; reducing every
        # batch's fresh increments keeps the buffers rank-identical and the counts resume-safe.
        _all_reduce(flat)
        self.q_f_counts += flat[:k]
        if cell is not None:
            self.cell_counts += flat[k:].view(self.n_cells, k)
        total = self.q_f_counts.sum()
        if total > 0:
            self.q_f.copy_((self.q_f_counts / total).clamp_min(self.p_floor).to(torch.float32))

    def _harmonic_sum(self, p: Tensor) -> Tensor:
        """``S = sum_g alpha_g pi_g / P(g|x)`` over the last axis, ``alpha = pi`` or 1."""
        alpha_pi = self.q_f**2 if self.target == "harmonic" else self.q_f
        return (alpha_pi / p).sum(-1)

    def freeze_accumulators(self, global_step: int, epoch: int) -> None:
        """Freeze the accumulated ``q_f`` (and ``Z``, the binned table) at a stage boundary.

        `TrainingController.advance_to_stage` calls this on every module that defines it,
        at EVERY stage boundary (including a resume at the boundary, which is mid-epoch
        for a num_samples/frac_samples stage). Idempotent: a configured ``q_f`` is frozen
        from construction and a second call is a no-op.

        Raises
        ------
        RuntimeError
            With nothing accumulated, or a zero-count class under a non-inclusive `target`.
        """
        if bool(self.q_f_frozen):
            return
        total = self.q_f_counts.sum()
        if total == 0:
            raise RuntimeError(
                f"KinematicWeight {self.name!r}: freeze_accumulators called with no "
                "accumulated stage-1 train batches (q_f_counts all zero) — q_f is undefined"
            )
        empty = [c for c, count in enumerate(self.q_f_counts.tolist()) if count == 0]
        if empty and self.target != "inclusive":
            raise RuntimeError(
                f"KinematicWeight {self.name!r}: target {self.target!r} needs every class "
                f"prior > 0 but class index {empty[0]} had zero accumulated jets"
            )
        with torch.no_grad():
            self.q_f.copy_((self.q_f_counts / total).to(torch.float32))
            for class_idx in empty:
                log.warning(
                    "KinematicWeight %r: class index %d had zero accumulated jets — its "
                    "frozen q_f weight is 0",
                    self.name,
                    class_idx,
                )
            if self.mode == "binned":
                totals = self.cell_counts.sum(-1)
                sparse = totals < self.min_cell_count
                n = int(sparse.sum())
                frac = float(totals[sparse].sum() / total)
                level = logging.WARNING if n else logging.INFO
                log.log(level, _SPARSE, self.name, n, self.n_cells, self.min_cell_count, frac)
            if self.target in _HARMONIC:
                if self.mode == "nn":
                    self.z.fill_(self.z_sum / total)
                else:  # exact: Z = sum_c (N_c / N) / S_c over the frozen table
                    keep = totals > 0
                    p = (self.cell_counts[keep] / totals[keep][:, None]).clamp(self.p_floor, 1.0)
                    self.z.fill_((totals[keep] / total / self._harmonic_sum(p)).sum())
            self.q_f_frozen.fill_(True)
        prior = [round(v, 6) for v in self.q_f.tolist()]
        z_note = f", Z={float(self.z):.6g}" if self.target in _HARMONIC else ""
        log.info(_FROZEN, self.name, global_step, epoch, int(total), prior, z_note)

    @property
    def accumulators_frozen(self) -> bool:
        """Whether `freeze_accumulators` has run (always ``True`` for a configured ``q_f``)."""
        return bool(self.q_f_frozen)
