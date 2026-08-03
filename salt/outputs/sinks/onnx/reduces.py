"""Export output reduces: the live registry of bundle-port -> ONNX-output
conversions (`register_reduce`) plus shared math helpers used by the folded
conversion nodes in `salt.outputs.producers`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import TensorSpec
from salt.outputs.sinks.onnx.config import ExportOutput
from salt.utils.mask_utils import indices_from_mask
from salt.utils.union_find import mask_fill_flattened

__all__ = [
    "BoundReduce",
    "ReduceCtx",
    "ReduceSpec",
    "bind_reduce",
    "mask_fill_flattened",
    "per_token_reduces",
    "reduce_dtype",
    "register_reduce",
    "registered_reduces",
    "unregister_reduce",
]


# ---------------------------------------------------------------------------
# shared MaskFormer export math
# ---------------------------------------------------------------------------
# `get_maskformer_outputs` (null suppression + pT reorder + index math) is a pure
# function the folded MaskFormerObjects conversion node composes. The per-node ->
# batch unflatten `mask_fill_flattened` lives in `salt.utils.union_find` (its
# single home, alongside the union-find kernel) and is re-exported here so the
# ONNX-math seam keeps naming it.


def get_maskformer_outputs(
    objects: Mapping[str, Tensor],
    max_null: float = 0.5,
    apply_reorder: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert raw MaskFormer-style outputs to convenient per-object tensors.

    Thresholds the "null" class probability and suppresses masks/regression for
    objects with ``p_null > max_null``; converts per-position mask logits into
    sparse mask indices; optionally reorders objects so the "leading" object
    (highest ``regression[0]``, e.g. pT) is first.

    Parameters
    ----------
    objects : Mapping[str, Tensor]
        Keys: ``"masks"`` (mask logits ``[B, M, L]``), ``"class_probs"``
        (``[B, M, C]``, last class is null), ``"regression"`` (``[B, M, R]``).
    max_null : float, optional
        Maximum allowed null probability for an object to be kept, by default ``0.5``.
    apply_reorder : bool, optional
        Reorder objects in descending order of ``regression[..., 0]``, by default ``True``.

    Returns
    -------
    leading_regression : torch.Tensor
        ``[B, R]`` for the leading object (after optional reordering).
    obj_indices : torch.Tensor | None
        Sparse mask indices per object, ``[B, M]``, values in ``[0, L)`` (``NaN``
        when undefined); ``None`` when there are no tracks (``L == 0``).
    class_probs : torch.Tensor
        Possibly-reordered class probabilities, ``[B, M, C]``.
    regression : torch.Tensor
        Possibly-reordered regression tensor, ``[B, M, R]``, ``NaN`` for null objects.
    """
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]
    n_tracks = masks.shape[-1]
    n_obj = masks.shape[1]
    n_reg = regression.shape[-1]

    if n_tracks == 0:
        return (
            torch.full((1, n_obj), torch.nan),
            None,
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )
    null_preds = class_probs[:, :, -1] > max_null
    if not null_preds.any():
        return (
            torch.full((1, n_obj), torch.nan),
            torch.arange(n_tracks).unsqueeze(0).expand(1, n_tracks),
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )

    masks = masks.sigmoid() > 0.5
    expanded_null = null_preds.unsqueeze(-1).expand(-1, -1, masks.size(-1))
    masks[expanded_null] = torch.zeros_like(masks)[expanded_null]
    regression[null_preds] = torch.nan

    if apply_reorder:
        # leading object = highest regression[0] (e.g. pT); argsort doesn't handle
        # NaN reliably in Athena, so null entries go to -inf for the sort then
        # back to NaN afterward
        regression[null_preds] = -torch.inf
        order = torch.argsort(regression[:, :, 0], descending=True)
        regression[null_preds] = torch.nan
        order_expanded = order.unsqueeze(-1).expand(-1, -1, masks.size(-1))

        masks = torch.gather(masks, 1, order_expanded)
        class_probs = torch.gather(
            class_probs, 1, order.unsqueeze(-1).expand(-1, -1, class_probs.size(-1))
        )
        regression = torch.gather(
            regression, 1, order.unsqueeze(-1).expand(-1, -1, regression.size(-1))
        )
    leading_regression = regression[:, 0]

    obj_indices = indices_from_mask(masks)

    return leading_regression, obj_indices, class_probs, regression


@dataclass(frozen=True)
class ReduceCtx:
    """Static context for binding reduces (config + plan derived, no tensors).

    `model_name` prefixes every output name; `seq_dyn_axis` maps sequence
    streams to their configured dynamic-axis names (from ``export.inputs``);
    `produced_specs` are the ONNX plan's produced specs, used for static
    class-count validation.
    """

    model_name: str
    seq_dyn_axis: Mapping[str, str]
    produced_specs: Mapping[str, TensorSpec]


@dataclass(frozen=True)
class BoundReduce:
    """One bound output group: traced-graph function + generated ONNX metadata.

    `fn` consumes the executed bundle and returns the group's output
    tensors, in `output_names` order; `dynamic_axes` covers only this
    group's outputs.
    """

    port: str
    output_names: tuple[str, ...]
    dtypes: tuple[str, ...]
    dynamic_axes: Mapping[str, dict[int, str]]
    fn: Callable[[Bundle], tuple[Tensor, ...]]


Binder = Callable[[ExportOutput, "ReduceCtx"], BoundReduce]
"""A reduce binder: resolved `ExportOutput` + static `ReduceCtx` -> `BoundReduce`."""


@dataclass(frozen=True)
class ReduceSpec:
    """One live registry entry: a reduce's binder + its declared output dtype.

    `register_reduce` builds and registers these; `bind_reduce` looks up the
    binder by name.

    Parameters
    ----------
    name : str
        The registry key named in `ExportOutput.reduce`.
    binder : Binder
        Builds the traced-graph `BoundReduce` for one resolved entry.
    dtype : str
        The reduce's declared ONNX output dtype (``float32``/``int8``) — the
        single owner of the per-reduce dtype rule.
    per_token : bool
        Whether the reduce emits per-token (dynamic sequence axis) outputs.
    expects_names : bool
        Whether the reduce consumes the plural ``names`` field (per-class
        scalars) vs the singular ``name`` field.
    """

    name: str
    binder: Binder
    dtype: str
    per_token: bool
    expects_names: bool


_REGISTRY: dict[str, ReduceSpec] = {}
"""The live reduce registry."""


def register_reduce(
    name: str,
    binder: Binder,
    *,
    dtype: str,
    per_token: bool = False,
    expects_names: bool = False,
) -> None:
    """Register a reduce under `name` with its declared dtype.

    The single entry point that adds a reduce to the live registry — the field
    knowledge (binder, dtype, per-token placement) lives with the
    registration, not in a frozen config tuple.

    Parameters
    ----------
    name : str
        The registry key named in `ExportOutput.reduce`. Must be unique.
    binder : Binder
        The binder building the traced-graph `BoundReduce` (see `bind_reduce`).
    dtype : str
        The reduce's declared ONNX output dtype (``float32``/``int8``).
    per_token : bool, optional
        Whether the reduce emits dynamic-axis (per-token) outputs, by default
        False (a global output).
    expects_names : bool, optional
        Whether the reduce consumes the plural ``names`` field (per-class
        scalars) rather than the singular ``name``, by default False.

    Raises
    ------
    ConfigError
        On a non-string/empty name, an unknown declared dtype, or a duplicate
        registration (re-registering a name is a programming error — there is
        no silent override).
    """
    if not isinstance(name, str) or not name:
        raise ConfigError(f"register_reduce: name must be a non-empty string, got {name!r}")
    if dtype not in {"float32", "int8"}:
        raise ConfigError(
            f"register_reduce {name!r}: dtype must be 'float32' or 'int8', got {dtype!r} "
            "(the ONNX output dtypes salt export supports — adapter.output_dtypes/check.py:149)"
        )
    if name in _REGISTRY:
        raise ConfigError(
            f"register_reduce: reduce {name!r} is already registered "
            f"(registry: {sorted(_REGISTRY)}) — names are unique, no silent override"
        )
    _REGISTRY[name] = ReduceSpec(
        name=name,
        binder=binder,
        dtype=dtype,
        per_token=per_token,
        expects_names=expects_names,
    )


def unregister_reduce(name: str) -> None:
    """Remove a reduce from the live registry (the symmetric cleanup surface).

    The registry is process-global module state. A consumer that registers a
    TRANSIENT reduce (a gate probe, a test fixture, a notebook experiment) must
    remove it again so it leaves no residue — a leaked entry would make the
    registered set order-dependent for everything sharing the process. No-ops if
    the name is not registered (idempotent teardown).

    Parameters
    ----------
    name : str
        The reduce name to remove.
    """
    _REGISTRY.pop(name, None)


def registered_reduces() -> tuple[str, ...]:
    """The registered reduce names, sorted."""
    return tuple(sorted(_REGISTRY))


def reduce_spec(name: str) -> ReduceSpec:
    """Look up one registered reduce spec by name.

    Returns
    -------
    ReduceSpec
        The registered spec.

    Raises
    ------
    ConfigError
        When `name` is not a registered reduce.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"unknown reduce {name!r} — registry: {registered_reduces()} "
            "(register it via salt.outputs.sinks.onnx.reduces.register_reduce)"
        ) from None


def reduce_dtype(name: str) -> str:
    """The DECLARED ONNX output dtype of a registered reduce.

    `reduce_spec` raises `ConfigError` for an unregistered name.

    Returns
    -------
    str
        ``float32`` or ``int8``.
    """
    return reduce_spec(name).dtype


def per_token_reduces() -> tuple[str, ...]:
    """The registered reduces that emit per-token (dynamic-axis) outputs, sorted."""
    return tuple(sorted(name for name, spec in _REGISTRY.items() if spec.per_token))


def bind_reduce(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """Bind one resolved `ExportOutput` to its registered reduce implementation.

    `reduce_spec` raises `ConfigError` on an unknown reduce key; the registered
    binder raises it on a statically-detectable mismatch (class count, missing
    sequence stream).

    Returns
    -------
    BoundReduce
        The bound reduce.
    """
    spec = reduce_spec(str(out_cfg.reduce))
    return spec.binder(out_cfg, ctx)
