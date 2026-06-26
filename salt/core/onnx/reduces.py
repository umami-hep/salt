"""Export output reduces: the LIVE registry of bundle-port -> ONNX-output conversions (design §7.3).

This is the M5 ``register_reduce`` deliverable (M4.5 amendment 555-567): the
export reduce set is a LIVE registry — `register_reduce` binds a reduce *name*
to a binder + its DECLARED output dtype + a per-token flag, and
`salt.core.onnx.config` validates ``export.outputs`` against it (and defaults
each entry's dtype from the registered declaration) instead of the frozen
``KNOWN_REDUCES`` tuple it carried through M4.5. A custom (e.g. export-only)
writer can now ship its own export math: register a reduce once, then name it
in `ExportOutput.reduce` — the field knowledge (output names, dtype, dynamic
axes) has one owner, the registered binder. This unblocks sub-wave C's two new
MaskFormer reduces (``leading_object``/``object_index``), which register here.

Each entry binds one resolved `ExportOutput` to a `BoundReduce` that knows its
output names, dtypes and dynamic axes — so the exporter's
``output_names``/``dynamic_axes`` are GENERATED from the export config instead
of v1's hand-ordered list surgery (``to_onnx.py:182-187,243-338``). The reduce
functions run INSIDE the traced graph, at exactly v1's placement. The three
SHIPPED reduces (registered at import via `register_reduce`, byte-unchanged
from M4):

- ``split_scalars`` — per-class probability scalars (float32, global)
  (``task.py:285-301``: ``torch.split(probs, 1, -1)`` + squeeze). The task
  already published converted probabilities in ONNX mode (design §3.3), so
  this reduce is a pure splitter.
- ``argmax`` — int8 per-token argmax with the zero-row append/strip trick
  kept verbatim (``to_onnx.py:415-423``; the appended row keeps the trace
  valid for zero-token jets). v1 argmaxes RAW logits where the v2 task
  publishes run_inference-converted probabilities in ONNX mode — argmax is
  invariant under the (masked) softmax, so the int8 output is identical.
- ``vertex_union_find`` — int8 per-token in-graph union-find on the RAW edge
  scores the vertexing task publishes in ONNX mode (design §3.3 per-family
  exception): ``get_node_assignment_jit`` (still ``@torch.jit.script``,
  fake-pad-track workaround inside, ``union_find.py:151-153``) +
  ``mask_fill_flattened`` + ``.reshape(-1).char()`` — v1's exact chain
  (``to_onnx.py:426-432``).

The two MaskFormer reduces register here too (sub-wave C, plan 10), composing
v1's `get_maskformer_outputs` (null suppression + pT reorder + index math,
maskformer.py:244-349) byte-faithfully:

- ``leading_object`` — R float32 GLOBAL scalars, the leading object's leading-
  object regression values (v1 ``to_onnx.py:461-468``). It reorders + selects the
  leading object; it never inverts scaling.
  ⚠ plan 34 W34.3 CONSEQUENCE (forward-flip): v1 relied on the object-regression
  task publishing DE-SCALED ``preds.objects.regression`` in TEST|ONNX, so this
  reduce could reorder physical values directly. Since W34.3 flipped
  ``RegressionTaskModule.forward`` to RAW loss-space in TEST|ONNX, that leaf now
  carries RAW (scaled) values — so this reduce (and its folded ``MaskFormerObjects``
  producer, ``producers.py``) would reorder/select RAW values. Both are
  EXPORT-RETIRED today (the off-graph reduce manifest was retired at plan-29 W4 and
  no MaskFormer config wires the ``MaskFormerObjects`` producer — the MaskFormer
  eval/export migration is W6-DEFERRED), so nothing live is wrong. But the W6
  MaskFormer cutover MUST de-scale ``preds.objects.regression`` (via the object
  regression task's ``run_inference`` / a ``Regression`` producer) BEFORE the
  ``get_maskformer_outputs`` reorder, exactly as the global/seq regression heads now
  de-scale on the ``outputs:`` section. Tracked as a W6 hard rule, not a W34.3 fix.
- ``object_index`` — int8 PER-TOKEN constituent->object index (v1
  ``indices.reshape(-1).char()``, ``to_onnx.py:469``); its single suffix is the
  writer-declared `OBJECT_INDEX.onnx` (``HadronIndex``).

Both are declared on ONE port (``preds.<object_stream>.regression`` /
``<object_stream>.masks``) yet additionally read the OTHER object keys from the
executed bundle — the decoder is kept alive by the ``object_index`` ``masks``
sink, so ALL its products (class_probs/masks/embed) are present.

Torch-free seam: `salt.core.onnx.config`'s MODULE BODY introduces no
top-level torch or registry import — it does NOT import this module at top
level (this module imports ``config.ExportOutput``, so the dependency is
one-way). It queries the live registry through `registered_reduces` /
`reduce_dtype` / `per_token_reduces` via a DEFERRED import inside its
export-only resolution path (``config._resolve_output`` /
``combine_insertion_index`` / the ``KNOWN_REDUCES`` lazy attribute) — so this
torch-importing registry is reached only when an `export:` block is actually
resolved, never at fit-time parse. (A bare ``import salt.core.onnx.config``
still pulls torch into ``sys.modules`` transitively via the
``salt.core.onnx`` package ``__init__``, which imports the torch-using
adapter; the seam is config.py's own torch-free module body and the
deferred-import discipline, not the whole import path.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec
from salt.core.onnx.config import ExportOutput
from salt.core.utils.mask_utils import indices_from_mask

__all__ = [
    "BoundReduce",
    "ReduceCtx",
    "ReduceSpec",
    "bind_reduce",
    "per_token_reduces",
    "reduce_dtype",
    "register_reduce",
    "registered_reduces",
    "unregister_reduce",
]


# ---------------------------------------------------------------------------
# inlined v1 MaskFormer export math (M7 W2b)
# ---------------------------------------------------------------------------
# ``get_maskformer_outputs`` (null suppression + pT reorder + index math) and
# ``mask_fill_flattened`` (the per-node -> batch unflatten) are tiny pure
# functions the two MaskFormer object reduces + the vertex union-find reduce
# compose. Inlined here BYTE-FAITHFULLY from v1 ``salt.models.maskformer``
# (maskformer.py:244-349) and v1 ``salt.models.task`` (task.py:1010-1035) so the
# export graph traces identically while the core package no longer imports the v1
# ``models`` tree. ``indices_from_mask`` is the relocated core copy
# (``salt.core.utils.mask_utils``, also byte-faithful).


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def mask_fill_flattened(flat_array: Tensor, mask: Tensor) -> Tensor:
    """Unflatten a per-node array back to a batch-shaped tensor using a mask.

    M7 W2b inline of v1 ``salt.models.task.mask_fill_flattened`` (task.py:1009-1035),
    byte-faithful — the ``@torch.jit.script`` decorator is PRESERVED (the union-find
    export reduce inlines this scripted subgraph into the ONNX trace, to_onnx.py:431;
    the scripted form's loop semantics are load-bearing for export parity).

    Parameters
    ----------
    flat_array : Tensor
        Tensor of shape ``[N, F]`` with concatenated (valid) per-node values.
    mask : Tensor
        Boolean mask of shape ``[B, L]`` where valid (non-padded) positions are ``False``.

    Returns
    -------
    Tensor
        Filled tensor of shape ``[B, L, F]`` where padded positions are set to ``-inf``.
    """
    filled = torch.full((mask.shape[0], mask.shape[1], flat_array.shape[1]), float("-inf"))
    mask = mask.to(torch.bool)
    start_index = end_index = 0

    for i in range(mask.shape[0]):
        if mask[i].shape[0] > 0:
            end_index += (~mask[i]).to(torch.long).sum()
            filled[i, : end_index - start_index] = flat_array[start_index:end_index]
            start_index = end_index

    return filled


def get_maskformer_outputs(
    objects: Mapping[str, Tensor],
    max_null: float = 0.5,
    apply_reorder: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert raw MaskFormer-style outputs to convenient per-object tensors.

    M7 W2b inline of v1 ``salt.models.maskformer.get_maskformer_outputs``
    (maskformer.py:244-349), byte-faithful. This helper:
      1. Thresholds the "null" class probability and suppresses masks/regression
         for objects with ``p_null > max_null``.
      2. Converts per-position mask logits into sparse mask indices via
         :func:`salt.core.utils.mask_utils.indices_from_mask`.
      3. Optionally reorders objects so that the "leading" object is first
         (highest regression[0], e.g. pT in vertexing).

    Parameters
    ----------
    objects : Mapping[str, Tensor]
        Dictionary with keys at least:
        - ``"masks"``: mask logits of shape ``[B, M, L]``.
        - ``"class_probs"``: class probabilities of shape ``[B, M, C]`` (last class is null).
        - ``"regression"``: regression targets/predictions of shape ``[B, M, R]``.
    max_null : float, optional
        Maximum allowed null probability ``p_null`` for an object to be kept,
        by default ``0.5``.
    apply_reorder : bool, optional
        If ``True``, reorder objects in descending order of ``regression[..., 0]``,
        by default ``True``.

    Returns
    -------
    leading_regression : torch.Tensor
        Tensor of shape ``[B, R]`` for the leading object (after optional reordering).
    obj_indices : torch.Tensor | None
        Sparse indices of masks per object with shape ``[B, M]``; values are
        positions in ``[0, L)`` (or ``NaN`` when undefined). May be ``None``
        if there are no tracks (``L == 0``).
    class_probs : torch.Tensor
        Possibly-reordered class probabilities of shape ``[B, M, C]``.
    regression : torch.Tensor
        Possibly-reordered regression tensor of shape ``[B, M, R]`` with ``NaN``
        for objects deemed null.

    Notes
    -----
    - If there are no input tracks/tokens (``L == 0``), dummy tensors filled with
      ``NaN`` are returned for indices and regression.
    - Masks are thresholded at ``0.5`` after a sigmoid to produce boolean masks
      prior to conversion to indices.
    """
    # Convert the (N,M) -> (M,) mask indices
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]
    n_tracks = masks.shape[-1]
    n_obj = masks.shape[1]
    n_reg = regression.shape[-1]

    # If we have a jet with no tracks,
    if n_tracks == 0:
        return (
            torch.full((1, n_obj), torch.nan),
            None,
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )
    # For testing purposes - this will likely blow up our fake rate
    null_preds = class_probs[:, :, -1] > max_null
    if not null_preds.any():
        # If we have no predicted objects, we return dummy values
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
        # Define the leading object as the one with the highest regression[0] value
        # in vertexing case, this is the pT. We first set these values to non-nans, as
        # argsort will otherwise not work correctly when in athena, and then set them back
        regression[null_preds] = -torch.inf
        order = torch.argsort(regression[:, :, 0], descending=True)
        regression[null_preds] = torch.nan
        order_expanded = order.unsqueeze(-1).expand(-1, -1, masks.size(-1))

        # Use gather to reorder tensors along a specific dimension
        masks = torch.gather(masks, 1, order_expanded)
        class_probs = torch.gather(
            class_probs, 1, order.unsqueeze(-1).expand(-1, -1, class_probs.size(-1))
        )
        regression = torch.gather(
            regression, 1, order.unsqueeze(-1).expand(-1, -1, regression.size(-1))
        )
        # Define the leading object as that with the highest [0] (pt for vertexing)
    leading_regression = regression[:, 0]

    # Convert our masks (N,M), now in pT order, to be (M,) indices
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
    """One LIVE registry entry: a reduce's binder + its DECLARED output dtype.

    `register_reduce` builds and registers these; `bind_reduce` looks up the
    binder by name, and `salt.core.onnx.config._resolve_output` reads `dtype`
    (the reduce's declared ONNX output dtype) and `per_token` (whether the
    reduce emits dynamic-axis outputs) to default + validate the manifest entry
    without hard-coding the per-reduce dtype rules it carried through M4.5.

    Parameters
    ----------
    name : str
        The registry key named in `ExportOutput.reduce`.
    binder : Binder
        Builds the traced-graph `BoundReduce` for one resolved entry.
    dtype : str
        The reduce's declared ONNX output dtype (``float32``/``int8``) — the
        single owner of the per-reduce dtype rule (e.g. v1 ``.char()`` int8 for
        the aux reduces, ``to_onnx.py:422,432``). `_resolve_output` defaults an
        entry's ``dtype`` from this and rejects any other declared dtype.
    per_token : bool
        Whether the reduce emits per-token (dynamic sequence axis) outputs —
        the "sequence-aux" entries of v1's output order. `config`'s
        `combine_insertion_index` keys off this set (amendment merge
        condition 5: combines insert before the first per-token entry).
    expects_names : bool
        Whether the reduce consumes the plural ``names`` field (per-class
        scalars, ``split_scalars``) vs the singular ``name`` field (the single-
        output reduces). Drives `_resolve_output`'s name/names exclusivity rule.
    """

    name: str
    binder: Binder
    dtype: str
    per_token: bool
    expects_names: bool


_REGISTRY: dict[str, ReduceSpec] = {}
"""The live reduce registry — the M5 replacement for the frozen ``KNOWN_REDUCES``."""


def register_reduce(
    name: str,
    binder: Binder,
    *,
    dtype: str,
    per_token: bool = False,
    expects_names: bool = False,
) -> None:
    """Register a reduce under `name` with its declared dtype (the M5 public surface).

    The single entry point that adds a reduce to the LIVE registry validated by
    `salt.core.onnx.config`. Shipped reduces register at import; custom /
    export-only writers (and sub-wave C's MaskFormer reduces) register their own
    export math the same way — the field knowledge (binder, dtype, per-token
    placement) lives with the registration, not in a frozen config tuple.

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
        False (a global output). Drives the combine-insertion order in
        `config.combine_insertion_index`.
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
            "(the ONNX output dtypes salt2 export supports — adapter.output_dtypes/check.py:149)"
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
    """The registered reduce names, sorted (the live replacement for ``KNOWN_REDUCES``).

    Returns
    -------
    tuple[str, ...]
        Sorted registry keys.
    """
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
            "(register it via salt.core.onnx.reduces.register_reduce; design §7.3)"
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
    """The registered reduces that emit per-token (dynamic-axis) outputs, sorted.

    The live replacement for the frozen ``PER_TOKEN_REDUCES`` tuple; consumed by
    `config.combine_insertion_index` (combines insert before the first per-token
    entry, the v1 order, amendment merge condition 5).

    Returns
    -------
    tuple[str, ...]
        Sorted per-token reduce names.
    """
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


# ---------------------------------------------------------------------------
# plan-29 W4 atomic cutover: the SHIPPED reduces are RETIRED.
# ---------------------------------------------------------------------------
# Through W3 this module also shipped (and registered at import) the five
# bundle-port -> ONNX-output reduce binders:
#   split_scalars / argmax / vertex_union_find / leading_object / object_index
# W4 folds every one of those conversions into a real plan node in
# `salt.core.outputs.producers` (ClassProbs/SeqClassIndex/VertexUnionFind/
# MaskFormerObjects/Combination), so the off-graph reduce manifest +
# post-executor `reduce.fn` loop are gone. No live config registers a shipped
# reduce any more; the `OnnxExportSink` names the conversion `outputs.*` leaves.
#
# What REMAINS here is the public `register_reduce` SURFACE (the live registry
# `_REGISTRY`, `register_reduce`/`unregister_reduce`/`registered_reduces`/
# `reduce_spec`/`reduce_dtype`/`per_token_reduces`/`bind_reduce` + the
# `ReduceSpec`/`BoundReduce`/`ReduceCtx` dataclasses) so a downstream custom
# export-only writer can still register its own reduce (R7 — retiring the public
# API is a separate, owned migration). The two shared math helpers
# `mask_fill_flattened` + `get_maskformer_outputs` also stay: the folded
# `VertexUnionFind` / `MaskFormerObjects` conversion nodes import them.
