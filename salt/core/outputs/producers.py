"""Output producers — in-graph `GraphModule`s populating the ``outputs.*`` dict.

Design §2 (layer 1): a producer reads ``preds.<stream>.<task>`` (the raw
training-space prediction a task always publishes) and writes one or more
``outputs.<stream>.<name>`` leaves — the eval/export-ready tensors a sink
serialises. Producers are ordinary model-pipeline modules: ``declare_io(mode)
-> IO`` plus a ``forward(bundle, mode)``, so the existing planner /
executor / bind machinery carries them with no kernel change.

Mode handling (design §2 "no more per-task forward branching"): producers run
ONLY when demanded. Their ports are active in every mode (``modes=ALL``); what
gates them is DEMAND, not a hard mode flag — the keystone the architecture
rests on (design §4 risk 4, gated at P0):

- **FIT/VAL**: nothing demands ``outputs.*`` (losses/metrics read ``preds.*``
  directly), so the planner's demand closure (planner.py ``_demand_closure``)
  drops the producer — it is absent from the FIT/VAL ``plan.steps`` and the
  training path is bitwise-unperturbed.
- **TEST**: a sink demands the ``outputs.*`` leaf, which pulls the producer
  (and transitively its ``preds.*`` source) into the plan.
- **ONNX**: the export ports demand it and the producer is in the traced
  subgraph.

Gating by demand (not by ``modes=TEST|ONNX``) is what makes the absence in
FIT/VAL provably a PRUNE: the producer is genuinely collected as a node and
then dropped because its output reaches no sink, exactly the mechanism the
design relies on. (A hard ``modes`` flag would also hide it, but then a wiring
slip that left it demanded in FIT/VAL could go unnoticed.)

This module ships the generic ``TaskOutput`` producer for the trivial
majority (design §4b: "one generic ``TaskOutput`` producer for the
copy/softmax/argmax/de-scale majority, with dedicated classes only for the
genuinely non-trivial heads"). ``TaskOutput`` is parameterised by a
``ConversionOp`` (the eval math); the default op is an identity copy (a real
torch passthrough via ``torch.clone``, so the produced leaf never aliases the
source ``preds.*`` leaf, design §2.1). P1 ships the classification + regression
conversion ops — each reproduces the corresponding v1/M4.5 task
``run_inference`` math VERBATIM (`salt.core.nn.tasks`):

- `ClassProbsOp` — global softmax / sigmoid (`ClassificationTask.run_inference`,
  ``tasks.py:312-328``, non-sequence branch).
- `SeqClassIndexOp` — masked-softmax then per-token ``argmax`` to an int index
  leaf (the sequence-classification ONNX shape, e.g. ``TrackOrigin``;
  `_masked_softmax` ``tasks.py:118-137`` then argmax over the class dim).
- `RegressionDescaleOp` — de-scale by scaler / norm_params / ratio-denominator
  (`RegressionTask.run_inference`, ``tasks.py:522-548``); the ratio-denominator
  case consumes an input Feature leaf and gathers the denominator by name.

The thin subclasses `ClassProbs`, `SeqClassIndex` and `Regression` construct
`TaskOutput` with the matching op so a config names the producer family
directly (design §2 config sketch). The demand-gating / width-resolution
contract is identical for every op.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, split_key, sym_dim, unflatten_spec

# plan-29 W3 (the hard reduces): the two scripted union-find helpers + the
# byte-faithful MaskFormer export math are ALREADY inlined VERBATIM in
# salt.core.onnx.reduces (the legacy reduce path). The W3 conversion nodes
# (`VertexUnionFind`/`MaskFormerObject`) REUSE those exact copies — a single
# source of truth, so the folded node and the legacy reduce can NEVER drift.
from salt.core.onnx.reduces import get_maskformer_outputs, mask_fill_flattened
from salt.core.utils.array_utils import listify
from salt.core.utils.scalers import RegressionTargetScaler
from salt.core.utils.union_find import get_node_assignment_jit
from salt.core.outputs.names import VERTEX_INDEX, pascal_case

__all__ = [
    "ClassProbs",
    "ClassProbsOp",
    "Combination",
    "ConversionOp",
    "IdentityOp",
    "MFLeadVertexDecorator",
    "MaskFormerObject",
    "MaskFormerObjects",
    "OutputField",
    "Regression",
    "RegressionDescaleOp",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "TaskOutput",
    "VertexUnionFind",
]

_UNNAMED = "unnamed"
"""Placeholder instance name — config assembly assigns the dict key (design §2.2)."""


@dataclass(frozen=True)
class OutputField:
    """One self-described output column a producer mints (plan 31 W5.0 field manifest).

    The plan-31 producer field-manifest entry: a producer resolves its output
    columns FROM THE TASK IT WRAPS (reusing the legacy ``task.output_names()`` /
    ``class_suffixes`` / ``output_suffixes`` / `VERTEX_INDEX` logic), so the
    auto-collecting sinks reproduce the legacy `TaskWriter` schema EXACTLY without
    hand-listing per config. Each field carries BOTH serialisation names because
    they can DIVERGE (plan 31 §3.1): a classification per-token head's H5 columns
    are the per-class prob suffixes (``pb``/``pc`` ...) while its ONNX output is the
    argmax index under the Athena name (``TrackOrigin``); regression's H5 *and*
    ONNX share the target/custom suffix. ``onnx_name`` DEFAULTS to ``h5_name``.

    A field carrying ``h5_name=None`` has no H5 representation (e.g. the ONNX-only
    `SeqClassIndex` argmax leaf, whose H5 counterpart is `SeqClassProbs`'s prob
    columns); ``onnx_name=None`` has no ONNX representation (e.g. `SeqClassProbs`'s
    H5 prob columns, whose ONNX counterpart is `SeqClassIndex`). The H5 sink
    collects fields with an ``h5_name``; the ONNX sink collects fields with an
    ``onnx_name`` (plan 31 §3.4 — one metadata source, two representations).

    Parameters
    ----------
    h5_name : str | None
        The H5 column suffix (run-name-prefixed by the sink unless ``prefix=False``)
        — the legacy ``task.output_names()`` suffix. ``None`` when the field has no
        H5 representation (an ONNX-only leaf).
    onnx_name : str | None
        The flat ONNX (Athena) output suffix (``{model_name}_{onnx_name}``).
        Defaults to ``h5_name`` when not given; ``None`` when the field has no ONNX
        representation (an H5-only leaf).
    dtype : str
        The H5/ONNX serialisation dtype — the numpy descriptor for H5 (``"f4"`` /
        ``"i8"`` / ``"i1"``) AND the equivalent ONNX dtype (``"float32"`` /
        ``"int8"``). The sink maps between the two namespaces as needed.
    axis : str
        ``"global"`` (a jet-level scalar ``[N]``) or ``"per_token"`` (a sequence
        column ``[N, T]`` / a dynamic ONNX axis).
    final : bool
        ``True`` for a written/exported leaf (the default for a producer's primary
        output); ``False`` for an exposed INTERMEDIATE leaf consumed only by a
        downstream node (e.g. `MaskFormerObjects`' per-vertex leaves the
        `MFLeadVertexDecorator` reads), which the sinks must NOT auto-collect.
    prefix : bool
        Whether the H5 column is ``{run_name}_{h5_name}`` (the v1 default) or the
        bare ``h5_name`` (the v1 ``VertexIndex`` byte-parity column). ONNX always
        prefixes with ``{model_name}_``.
    value : Tensor | None
        The graph-visible converted result this field describes (plan 34 W34.1).
        ``None`` for the static plan-31 manifest path (``output_columns`` mints
        name/dtype/axis metadata BEFORE any forward, so it has no tensor); the
        plan-34 task-side ``get_output(b, mode, run_name)`` path FILLS it with the
        converted torch tensor (softmax / masked-softmax / argmax — traceable
        ops, so ONNX sees them in-graph). The sink applies the
        ``{run_name}``/``{model_name}`` prefix, the f4->f2 downcast and the H5
        packing — the value stays a raw torch tensor at full precision (no prefix,
        no pack baked in). A field describes a SINGLE serialisation leaf, so a
        multi-class head returns one field per class, each carrying that class's
        column of `value`'s last axis (see ``ClassificationTaskModule.get_output``).
    """

    h5_name: str | None
    onnx_name: str | None = None
    dtype: str = "f4"
    axis: str = "global"
    final: bool = True
    prefix: bool = True
    value: Tensor | None = None

    def __post_init__(self) -> None:
        if self.axis not in {"global", "per_token"}:
            raise ConfigError(
                f"OutputField axis must be 'global' or 'per_token', got {self.axis!r}"
            )
        if self.h5_name is None and self.onnx_name is None:
            raise ConfigError(
                "OutputField needs at least one of h5_name / onnx_name (a field with neither "
                "has no representation in any sink)"
            )

    @property
    def resolved_onnx_name(self) -> str | None:
        """The ONNX suffix, defaulting to `h5_name` when not explicitly set.

        Returns
        -------
        str | None
            `onnx_name` if given, else `h5_name` (None only for an H5-only field
            that explicitly passed ``onnx_name`` not set AND h5_name None — which
            ``__post_init__`` forbids, so this is None only when onnx is suppressed).
        """
        return self.onnx_name if self.onnx_name is not None else self.h5_name

    @property
    def onnx_dtype(self) -> str:
        """The ONNX dtype namespace mapping of `dtype` (``f4 -> float32``, ``i1/i8 -> int8``).

        Returns
        -------
        str
            ``"float32"`` or ``"int8"``.
        """
        return "int8" if self.dtype in {"i1", "i8", "int8"} else "float32"


def _resolve_task(model_modules: Mapping[str, Any], task_name: str, who: str) -> Any:
    """Look up the wrapped task module by name in the model module dict (W5.0).

    A producer resolves its output column NAMES from the task it wraps so the
    auto-collecting sinks reproduce the legacy ``task.output_names()`` schema
    EXACTLY (plan 31 §3.1). The sink — which holds the model module dict — threads
    it into ``output_columns(run_name, model_modules)``.

    Returns
    -------
    Any
        The resolved task module.

    Raises
    ------
    ConfigError
        When the named task is absent from the model module dict.
    """
    task = model_modules.get(task_name)
    if task is None:
        raise ConfigError(
            f"{who}: wrapped task {task_name!r} is not in the model modules — a conversion "
            "producer's output_columns() resolve their names from the task they wrap "
            "(plan 31 §3.1)"
        )
    return task


def _add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dims after the batch dim to reach ``ndim`` (v1 tensor_utils.py:168-198).

    Copied byte-faithfully from ``salt.core.nn.tasks._add_dims`` (itself inlined
    from v1 ``salt.utils.tensor_utils.add_dims``) — the broadcast helper the
    masked softmax relies on. Reproduced here (not imported) so the conversion
    op carries its math standalone and the parity test can compare against the
    task's copy directly.

    Returns
    -------
    Tensor
        ``x`` reshaped with the added singleton dimensions.

    Raises
    ------
    ValueError
        If ``ndim`` is smaller than ``x.ndim``.
    """
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is smaller than input ndim ({x.dim()})")
    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])
    return x


def _masked_softmax(x: Tensor, mask: Tensor | None, dim: int = -1) -> Tensor:
    """Softmax ignoring padded elements (v1 tensor_utils.py:74-107, byte-faithful).

    Copied VERBATIM from ``salt.core.nn.tasks._masked_softmax`` (the padded-aware
    softmax the absorbed ``ClassificationTask.run_inference`` uses for sequence
    heads, v1 task.py:263): elements where ``mask`` is ``True`` are set to
    ``-inf`` before the softmax and zeroed after.

    Returns
    -------
    Tensor
        Tensor after the masked softmax.
    """
    if mask is not None:
        mask = _add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)
    x = torch.softmax(x, dim=dim)
    if mask is not None:
        x = x.masked_fill(mask, 0)
    return x


# ===========================================================================
# Conversion ops (design §4b: the eval math behind the generic TaskOutput).
# Each op is a small strategy object (NOT an nn.Module) holding ITS config and
# reproducing one task's run_inference math VERBATIM. Three hooks let the
# generic TaskOutput stay op-agnostic:
#   - extra_requires(stream): extra input ports (pad mask / input Feature)
#   - derived_width(in_width): the output last-dim width given the input width
#     (identity/softmax/de-scale preserve it; argmax collapses it to 1)
#   - convert(b, mode, ...): the forward conversion on the raw preds tensor
# ===========================================================================


class ConversionOp:
    """The eval-math strategy behind `TaskOutput` (design §4b).

    Subclasses reproduce one task family's ``run_inference`` math VERBATIM.
    The base is the IDENTITY op kept for P0 byte-compat (a `torch.clone`
    passthrough). Three hooks parameterise the generic `TaskOutput`:
    `extra_requires` (extra input ports the op reads), `derived_width` (the
    output last-dim width as a function of the input width — design §6.6 / §4
    risk 6) and `convert` (the forward conversion).
    """

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Extra input ports this op reads beyond the source ``preds.*`` leaf.

        The default op needs nothing else. Override to demand a pad mask
        (masked softmax / argmax) or a raw input Feature (ONNX ratio
        denominators).

        Returns
        -------
        dict[str, TensorSpec]
            ``{key: spec}`` of additional requires (empty by default).
        """
        del stream
        return {}

    def derived_width(self, in_width: int) -> int:
        """The output leaf's last-dim width given the input ``preds.*`` width.

        Identity / softmax / de-scale preserve the width; ``argmax`` collapses
        the class dim to a single index column (overridden in `SeqClassIndexOp`).

        Returns
        -------
        int
            The output last-dim width.
        """
        return in_width

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """The field manifest this op's output leaf expands into (plan 31 W5.0).

        Resolved FROM THE WRAPPED TASK so the auto-collecting sinks reproduce the
        legacy ``task.output_names()`` schema EXACTLY. The base (identity copy)
        mirrors the wrapped task's own ``output_names``: one field per column, the
        suffix the bare task column name (run-name prefix STRIPPED — the sink
        re-prefixes), dtype the task descriptor, axis per the task's sequence flag.
        Subclasses override where the representation diverges (argmax index,
        regression descale). This default serves an IDENTITY ``TaskOutput`` over a
        task that owns its own rendering (e.g. the regression descale passthrough,
        the W5 stopgap).

        Parameters
        ----------
        task : Any
            The wrapped task module (resolved by the producer).
        run_name : str
            The run name — used only to STRIP the prefix the task baked in, so the
            returned suffix is bare (the sink re-prefixes per its own run name).

        Returns
        -------
        list[OutputField]
            One field per task output column.

        Raises
        ------
        ConfigError
            When the wrapped task ships no ``output_names`` rendering.
        """
        names = task.output_names(run_name)
        sequence = bool(getattr(task, "sequence", False))
        axis = "per_token" if sequence else "global"
        fields: list[OutputField] = []
        for column, dtype in names:
            suffix = column[len(run_name) + 1 :] if column.startswith(f"{run_name}_") else column
            prefix = column.startswith(f"{run_name}_")
            fields.append(
                OutputField(h5_name=suffix, dtype=str(dtype), axis=axis, final=True, prefix=prefix)
            )
        return fields

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Identity copy of the prediction leaf (P0 passthrough).

        A fresh tensor (``clone``) so the output never aliases the source
        ``preds.*`` leaf (write-once, design §2.1).

        Returns
        -------
        Tensor
            The converted (here: cloned) prediction tensor.
        """
        del mode, stream
        return b.get(pred_key).clone()


class IdentityOp(ConversionOp):
    """Explicit alias for the identity conversion op (the `ConversionOp` base).

    The base `ConversionOp` already IS the identity passthrough; this named
    subclass lets a config say ``op: {class_path: ...IdentityOp}`` explicitly
    when the default would otherwise be implicit.
    """


class ClassProbsOp(ConversionOp):
    """Global classification probabilities (v1 ``ClassificationTask.run_inference``).

    Reproduces the non-sequence branch of ``tasks.py:312-328`` VERBATIM:
    ``sigmoid`` for a ``BCEWithLogitsLoss`` head, else ``softmax(dim=-1)`` —
    the float32 per-class probabilities the eval H5 ``{run_name}_{px}`` columns
    carry (``tasks.py:1257-1271``). Width-preserving (C classes in, C probs
    out). The pad-mask / sequence branch is `SeqClassIndexOp`'s job; this op is
    for pooled (global) heads only and asserts the 2-D prediction shape v1
    asserts (``tasks.py:323``).

    Parameters
    ----------
    bce : bool, optional
        Whether the source head uses a ``BCEWithLogitsLoss`` (per-class
        ``sigmoid``) instead of the default ``CrossEntropyLoss`` (``softmax``),
        by default False. Mirrors the v1 ``isinstance(self.loss,
        BCEWithLogitsLoss)`` branch (``tasks.py:320``).
    """

    def __init__(self, bce: bool = False) -> None:
        self.bce = bool(bce)

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float global field per class — the ``class_suffixes`` (v1 task.py:140-151).

        The global (pooled) classification probabilities: H5 columns AND ONNX
        ``split_scalars`` scalars are the SAME per-class suffixes (``pb``/``pc`` ...),
        so ``onnx_name`` defaults to ``h5_name``. The legacy `TaskWriter` derives
        these from ``task.output_names(run_name)`` (H5) and
        ``task.onnx_outputs()`` (ONNX, the `class_suffixes`); both are reproduced
        here from the same `class_suffixes`.

        Returns
        -------
        list[OutputField]
            One ``f4`` global field per class, in class order.
        """
        del run_name
        return [
            OutputField(h5_name=px, dtype="f4", axis="global", final=True)
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Sigmoid (BCE) or softmax over the class dim (v1 ``tasks.py:312-328``).

        Returns
        -------
        Tensor
            ``[B, C]`` per-class probabilities.
        """
        del mode, stream
        preds = b.get(pred_key)
        if self.bce:
            return torch.sigmoid(preds)
        assert preds.ndim == 2, (
            "ClassProbsOp is a global (pooled) head; use SeqClassIndex for per-token heads"
        )
        return torch.softmax(preds, dim=-1)


class SeqClassIndexOp(ConversionOp):
    """Per-token class index (masked-softmax then ``argmax``, e.g. ``TrackOrigin``).

    Reproduces the sequence-classification index math: the masked softmax v1
    applies for per-token heads (``_masked_softmax``, ``tasks.py:118-137,322-327``)
    followed by ``argmax`` over the class dim — the integer per-token class index
    the ONNX ``argmax`` reduce emits (``reduces.py:_bind_argmax``). The class dim
    COLLAPSES, so the output last dim is 1 (`derived_width` returns 1). The pad
    mask is read from ``masks.<stream>`` (so padded tokens are ``-inf`` before the
    softmax — argmax is invariant under the softmax, ``reduces.py:28-30``).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask (a variable-length
        constituent sequence), by default True. False for a fixed-count query
        bank (the MaskFormer ``objects`` stream — no ``masks.objects``),
        mirroring v1's objects-stream exemption (``tasks.py:69,547``).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax.

        Returns
        -------
        dict[str, TensorSpec]
            ``{masks.<stream>: spec}`` when `has_pad_mask`, else empty.
        """
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def derived_width(self, in_width: int) -> int:
        """The argmax collapses the class dim to a single index column.

        Returns
        -------
        int
            Always 1 (one per-token index).
        """
        del in_width
        return 1

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One int8 per-token ONNX field, named ``pascal_case(task)`` (e.g. ``TrackOrigin``).

        The ONNX-ONLY argmax index leaf (folds ``reduces._bind_argmax``): the
        legacy sequence-classification ``onnx_outputs`` emits a single int8 ``argmax``
        entry whose suffix is the Pascal-case of the task instance name
        (``track_origin -> TrackOrigin``, v1 ``to_onnx.py:283-292``). It has NO H5
        column (``h5_name=None``): the TEST H5 carries the full prob vector via the
        sibling `SeqClassProbs` producer — the two-representation split (plan 31
        §3.1 / design §6.2 R2).

        Returns
        -------
        list[OutputField]
            One int8 per-token field with ``h5_name=None`` and the Athena
            ``onnx_name``.
        """
        del run_name
        return [
            OutputField(
                h5_name=None,
                onnx_name=pascal_case(task.name),
                dtype="int8",
                axis="per_token",
                final=True,
            )
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax then per-token ``argmax`` over the class dim (mode-branched).

        Two representations of the SAME argmax (design §6.2, R2 — "one
        conversion path per OUTPUT REPRESENTATION"):

        - **TEST**: ``[B, L]`` integer per-token class indices (``int64``) — the
          eval columns. The masked softmax zeroes padded tokens; argmax is
          invariant under it (``reduces.py:28-30``).
        - **ONNX**: the int8 ``[L]`` leaf with the zero-row append/strip trick
          carried VERBATIM from ``reduces._bind_argmax`` (``to_onnx.py:415-423``):
          a zero row is appended along the token axis so the traced argmax stays
          valid for zero-token jets, then stripped, ``.squeeze(0).char()`` to the
          Athena int8 output. v1 argmaxed RAW logits where the converted probs
          here are softmaxed first — argmax is invariant under the (masked)
          softmax, so the int8 output is identical (folds ``reduces._bind_argmax``).

        The branch is keyed on `mode`, NOT on a hard ``modes`` declaration: the
        zero-row trick is ONNX-shape-specific (``[1, L, C]`` traced batch) and
        must NOT run on a TEST batch (it would corrupt the eval int64 column);
        the masked-softmax+argmax body is shared. Same pattern as
        `RegressionDescaleOp._descale_source`.

        Returns
        -------
        Tensor
            TEST: ``[B, L]`` int64 per-token class indices. ONNX: ``[L]`` int8.
        """
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # zero-row append/strip VERBATIM (to_onnx.py:418-421 / reduces._bind_argmax):
            # the appended row keeps the traced argmax valid for zero-token jets,
            # then stripped; .squeeze(0).char() to the int8 [L] Athena output
            probs = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
            out = torch.argmax(probs, dim=-1)[:, :-1]
            return out.squeeze(0).char()
        return torch.argmax(probs, dim=-1)


class SeqClassProbsOp(ConversionOp):
    """Per-token class probabilities (masked-softmax, the sequence eval-H5 columns).

    Reproduces the sequence branch of ``ClassificationTask.run_inference``
    (``tasks.py:322-327``) VERBATIM: the masked softmax v1 applies for per-token
    heads (``_masked_softmax``, ``tasks.py:118-137``), giving the ``[B, L, C]``
    per-class probabilities the eval H5 ``{run_name}_{px}`` columns carry for a
    sequence classification head (e.g. ``track_origin``'s 8 origin probs,
    ``tasks.py:1257-1271``). Padded positions read ``0.0`` (the masked softmax
    zeroes them) — the v1 eval byte-parity quirk. Width-preserving (C classes in,
    C probs out). This is the eval-H5 counterpart of `SeqClassIndexOp` (which
    collapses to the ONNX argmax INDEX): the TEST H5 keeps the full prob columns,
    the ONNX export keeps the argmax — the two-representation split is exactly why
    they are distinct ops (design §1 dtype/split facts).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True (False
        for a fixed-count query bank — the MaskFormer ``objects`` stream).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax.

        Returns
        -------
        dict[str, TensorSpec]
            ``{masks.<stream>: spec}`` when `has_pad_mask`, else empty.
        """
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float per-token H5 field per class — the ``class_suffixes`` (v1 task.py:140-151).

        The TEST-H5-ONLY per-token prob columns: the legacy sequence-classification
        ``output_names`` emits one ``{run_name}_{px}`` ``f4`` column per class. There
        is NO ONNX representation here (``onnx_name=None``) — the ONNX export uses the
        sibling `SeqClassIndex` argmax index (the two-representation split, plan 31
        §3.1).

        Returns
        -------
        list[OutputField]
            One ``f4`` per-token field per class (H5-only), in class order.
        """
        del run_name
        return [
            OutputField(
                h5_name=px, onnx_name=None, dtype="f4", axis="per_token", final=True
            )
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax over the class dim (v1 ``tasks.py:322-327``).

        Returns
        -------
        Tensor
            ``[B, L, C]`` per-token per-class probabilities (padded tokens 0.0).
        """
        del mode
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        return _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)


class RegressionDescaleOp(ConversionOp):
    """De-scale regression predictions (v1 ``RegressionTask.run_inference``).

    Reproduces ``tasks.py:522-548`` VERBATIM — exactly one of three mutually
    exclusive scaling methods inverts the training-space prediction:

    - **ratio denominator** (``target_denominators``): ``pred[..., i] *=
      denom_i`` (``tasks.py:533-535``). The denominator source is mode-split
      (FD §3.3): TEST reads it from ``labels.<stream>.<denom>``; ONNX gathers it
      by NAME from the raw input Feature tensor ``inputs.<stream>``
      (``tasks.py:2281-2294``). The op declares both source ports and picks per
      mode in `convert`.
    - **norm_params** (mean/std): ``pred[..., i] = pred[..., i] * std_i + mean_i``
      (``tasks.py:536-539``).
    - **scaler** (functional `RegressionTargetScaler`): ``pred[..., i] =
      scaler.inverse(target_i, pred[..., i])`` (``tasks.py:540-542``).

    Width-preserving (R targets in, R de-scaled values out). The pad-mask NaN
    fill v1 applies after de-scaling (``tasks.py:545-546``) is NOT reproduced
    here: that NaN is a SINK-side serialisation concern (the H5 writer owns pad
    re-expansion, design §2 layer 2), and the eval ``get_h5`` reads the
    already-de-scaled tensor — the de-scaling MATH is what this op owns. (For an
    unmasked global head v1 applies no NaN fill anyway.)

    Parameters
    ----------
    stream : str
        The regressed stream (denominator labels live under
        ``labels.<stream>.<denom>``; the input Feature is ``inputs.<stream>``).
    targets : str | Sequence[str]
        The regression target name(s), in column order (R targets; a bare string
        for a single target). Used to index the functional scaler and to size the
        descale loop, matching v1's ``range(len(self.targets))``.
    target_denominators : str | Sequence[str] | None, optional
        Per-target ratio-denominator variable name(s) (``target/denom`` at train,
        ``pred * denom`` at de-scale), by default None. Mutually exclusive with
        `norm_params` / `scaler`.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": <scalar|list>, "std": <scalar|list>}`` per-target mean/std,
        by default None. Mutually exclusive.
    scaler : Mapping[str, Mapping[str, Any]] | None, optional
        Per-target functional scaling config (built into a
        `RegressionTargetScaler`), by default None. Mutually exclusive.

    Raises
    ------
    ConfigError
        On more than one scaling method, or a denominator/target count mismatch.
    """

    def __init__(
        self,
        stream: str,
        targets: str | Sequence[str],
        target_denominators: str | Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        scaler: Mapping[str, Mapping[str, Any]] | None = None,
        gaussian: bool = False,
        sequence: bool = False,
    ) -> None:
        self.stream = stream
        # accept a scalar string OR a sequence (mirrors RegressionTaskModule.targets,
        # tasks.py — the config YAML surface uses a bare string for a single target)
        self.targets = tuple(listify(targets))
        if not self.targets:
            raise ConfigError("RegressionDescaleOp: targets is required and non-empty")
        # plan 34 W34.3: a per-token (sequence) regression head NaN-fills padded
        # positions after de-scaling (v1 run_inference, tasks.py:557 / :649-650) — the
        # SAME quirk the task's get_output reproduces, so the producer path matches the
        # get_output path AND the legacy WriterCallback. The pad mask is read from
        # masks.<stream>; a global head NaN-fills nothing.
        self.sequence = bool(sequence)
        self.target_denominators = (
            tuple(listify(target_denominators)) if target_denominators is not None else None
        )
        self.norm_params = self._checked_norm_params(norm_params)
        self.scaler = RegressionTargetScaler(dict(scaler)) if scaler is not None else None
        # plan 34 W34.3: gaussian heads publish [..., 2R] (means ‖ raw variances)
        # and de-scale to [..., 2R] (means ‖ stddev = sqrt(softplus(var)) * std),
        # mirroring v1 GaussianRegressionTask.run_inference (tasks.py:616-652). The
        # gaussian descale supports norm_params (mean/std) and ratio denominators —
        # NOT a functional scaler (v1 has none for gaussian).
        self.gaussian = bool(gaussian)
        if self.gaussian and self.scaler is not None:
            raise ConfigError(
                "RegressionDescaleOp: gaussian de-scaling has no functional-scaler branch "
                "(v1 GaussianRegressionTask.run_inference, tasks.py:616-652) — use norm_params "
                "or target_denominators"
            )
        n_methods = sum(
            x is not None for x in (self.target_denominators, self.norm_params, self.scaler)
        )
        if n_methods > 1:
            raise ConfigError(
                "RegressionDescaleOp: only a single scaling method is allowed — set at most one "
                f"of target_denominators/norm_params/scaler (v1 tasks.py:355), got {n_methods}"
            )
        if self.target_denominators is not None and len(self.target_denominators) != len(
            self.targets
        ):
            raise ConfigError(
                f"RegressionDescaleOp: target_denominators {list(self.target_denominators)} "
                f"({len(self.target_denominators)}) must match targets {list(self.targets)} "
                f"({len(self.targets)}) (v1 tasks.py:361-366)"
            )
        # resolved at bind: the declared input-Feature column order, so the ONNX
        # de-scaling can gather denominators by NAME (FD §3.3, tasks.py:2191-2202).
        self._input_fields: tuple[str, ...] = ()

    @staticmethod
    def _checked_norm_params(
        norm_params: Mapping[str, Any] | None,
    ) -> dict[str, list[float]] | None:
        """Normalise + validate the ``norm_params`` mapping (v1 tasks.py:1929-1955).

        Returns
        -------
        dict[str, list[float]] | None
            ``{"mean": [...], "std": [...]}`` with both listified, or None.

        Raises
        ------
        ConfigError
            If the mapping is present but lacks ``mean``/``std``.
        """
        if norm_params is None:
            return None
        if set(norm_params) < {"mean", "std"}:
            raise ConfigError(
                f"RegressionDescaleOp: norm_params must carry 'mean' and 'std', got "
                f"{sorted(norm_params)} (v1 tasks.py:1947)"
            )
        return {
            "mean": [float(x) for x in listify(norm_params["mean"])],
            "std": [float(x) for x in listify(norm_params["std"])],
        }

    @property
    def input_feature_key(self) -> str:
        """The raw-input key carrying the ONNX denominator columns.

        Returns
        -------
        str
            ``inputs.<stream>``.
        """
        return f"inputs.{self.stream}"

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float field per regression output — the ``output_suffixes`` (v1 task.py:511-517).

        The de-scaled regression values: H5 columns AND ONNX ``split_scalars``
        scalars are the SAME suffixes (``custom_output_names`` else the targets;
        doubled for a gaussian head — R means then R ``_stddev``), so ``onnx_name``
        defaults to ``h5_name``. Reproduces the legacy ``task.output_names`` (H5) and
        ``task.onnx_outputs`` (ONNX) from the one ``output_suffixes`` source.

        Returns
        -------
        list[OutputField]
            One ``f4`` field per regression output, in column order (global for a
            pooled head, per-token for a sequence head).
        """
        del run_name
        axis = "per_token" if bool(getattr(task, "sequence", False)) else "global"
        return [
            OutputField(h5_name=suffix, dtype="f4", axis=axis, final=True)
            for suffix in task.output_suffixes
        ]

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the ratio-denominator sources (+ pad mask for a seq head).

        FIT|VAL|TEST read each denominator from ``labels.<stream>.<denom>``;
        ONNX gathers them by name from the raw ``inputs.<stream>`` Feature
        tensor. norm_params / scaler need no external source. A per-token
        (``sequence``) head additionally demands ``masks.<stream>`` for the
        post-de-scale NaN fill (plan 34 W34.3).

        Returns
        -------
        dict[str, TensorSpec]
            The denominator-source ports + (seq) the pad mask.
        """
        out: dict[str, TensorSpec] = {}
        if self.sequence:
            out[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        if self.target_denominators is None:
            return out
        for denom in self.target_denominators:
            out[f"labels.{stream}.{denom}"] = TensorSpec(
                shape=None,
                dtype="float32",
                kind="label",
                modes=Mode.FIT | Mode.VAL | Mode.TEST,
            )
        out[self.input_feature_key] = TensorSpec(
            shape=("B", sym_dim("F", f"{stream}.descale")),
            dtype="float32",
            modes=Mode.ONNX,
        )
        return out

    def bind(self, fields: tuple[str, ...]) -> None:
        """Capture the input Feature column order for the ONNX by-name gather.

        Mirrors ``RegressionTaskModule.bind`` (``tasks.py:2191-2202``): every
        ratio denominator must be a declared column of ``inputs.<stream>`` so
        the export graph can gather it by name; an absent denominator is a
        config error (the ONNX mode has no other source).

        Raises
        ------
        ConfigError
            If a ratio denominator is not a declared input Feature column.
        """
        self._input_fields = tuple(fields)
        if self.target_denominators is None:
            return
        present = set(self._input_fields)
        if missing := [d for d in self.target_denominators if d not in present]:
            raise ConfigError(
                f"RegressionDescaleOp: ratio denominators {missing} are not declared columns of "
                f"{self.input_feature_key!r} ({sorted(present)}) — the ONNX export graph de-scales "
                "from the input Feature tensor (tasks.py:2191-2202), so a denominator must be an "
                "input variable"
            )

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Invert the configured scaling (v1 ``RegressionTask.run_inference``).

        Returns
        -------
        Tensor
            ``[B, R]`` / ``[B, L, R]`` de-scaled physical values (float32).
        """
        # clone before the in-place de-scale: ``.float()`` on an already-float32
        # leaf returns the SAME tensor, so a bare ``b.get(...).float()`` would
        # mutate the bundle's ``preds.*`` leaf in place (the v1 task owns its
        # fresh ``preds`` and can mutate it; a producer must not, write-once §2.1)
        preds = b.get(pred_key).float().clone()
        if self.gaussian:
            return self._convert_gaussian(preds, b, mode, stream)
        if self.target_denominators is not None:
            denoms = self._descale_source(b, mode, stream)
            for i, denom in enumerate(self.target_denominators):
                preds[..., i] *= denoms[denom]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[..., i] *= self.norm_params["std"][i]
                preds[..., i] += self.norm_params["mean"][i]
        elif self.scaler is not None:
            for i in range(len(self.targets)):
                preds[..., i] = self.scaler.inverse(self.targets[i], preds[..., i])
        return self._nan_fill(preds, b, stream)

    def _nan_fill(self, preds: Tensor, b: Bundle, stream: str) -> Tensor:
        """NaN-fill padded positions for a seq head (v1 run_inference, tasks.py:557).

        Reproduces the post-de-scale ``torch.masked_fill(preds, mask.unsqueeze(-1),
        nan)`` the task's ``run_inference`` applies (and that the task's get_output
        reproduces), so the producer path matches the get_output path AND the legacy
        WriterCallback at padded positions. A global head NaN-fills nothing.

        Returns
        -------
        Tensor
            ``preds`` with padded positions set to NaN (seq head), else unchanged.
        """
        if not self.sequence:
            return preds
        mask = b.get(f"masks.{stream}")
        return torch.masked_fill(preds, mask.unsqueeze(-1), torch.nan)

    def _convert_gaussian(self, preds: Tensor, b: Bundle, mode: Mode, stream: str) -> Tensor:
        """De-scale a gaussian head's ``[..., 2R]`` means ‖ raw-variances (plan 34 W34.3).

        Mirrors v1 ``GaussianRegressionTask.run_inference`` (tasks.py:616-652)
        VERBATIM: the means in columns ``[0:R]`` de-scale like a plain regression
        head (ratio-denom OR mean/std), and the variances in ``[R:2R]`` become
        ``stddev = sqrt(softplus(var)) * std`` (norm_params) or are scaled by the
        denominator (ratio). The published ``[..., 2R]`` array is means ‖ stddevs
        (the FD 1567-1568 one-array contract the writer splits on ``_stddev``).

        Note: the v1 gaussian indexing is ``preds[:, i]`` / ``preds[:, i+1]`` for
        ``i in range(R)`` — index-aligned for the R=1 shipped gaussian configs;
        the loop is reproduced here on the last axis (``preds[..., j]``) so it
        broadcasts over a per-token sequence head too.

        Returns
        -------
        Tensor
            The de-scaled ``[..., 2R]`` means ‖ stddevs.
        """
        n = len(self.targets)
        if self.target_denominators is not None:
            denoms = self._descale_source(b, mode, stream)
            for i, denom in enumerate(self.target_denominators):
                # v1: preds[:, i] (mean) and preds[:, i + 1] (var) both * denom
                preds[..., i] *= denoms[denom]
                preds[..., i + 1] *= denoms[denom]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[..., i] *= self.norm_params["std"][i]
                preds[..., i] += self.norm_params["mean"][i]
                preds[..., i + 1] = (
                    torch.sqrt(nn.functional.softplus(preds[..., i + 1]))
                    * self.norm_params["std"][i]
                )
        del n
        # gaussian run_inference NaN-fills means + stds at padded positions
        # (tasks.py:648-650); _nan_fill masks the whole [..., 2R] array equivalently
        return self._nan_fill(preds, b, stream)

    def _descale_source(self, b: Bundle, mode: Mode, stream: str) -> dict[str, Tensor]:
        """Gather the per-denominator de-scaling source (FD §3.3 mode split).

        TEST reads ``labels.<stream>.<denom>``; ONNX gathers the denominator by
        NAME from the raw input Feature tensor (the only export-time source,
        ``tasks.py:2281-2294``).

        Returns
        -------
        dict[str, Tensor]
            ``{denom: tensor}`` for every ratio denominator.
        """
        assert self.target_denominators is not None
        if mode & Mode.ONNX:
            columns = b.get(self.input_feature_key)
            field_index = {name: i for i, name in enumerate(self._input_fields)}
            return {
                denom: columns[..., field_index[denom]] for denom in self.target_denominators
            }
        return {
            denom: b.get(f"labels.{stream}.{denom}") for denom in self.target_denominators
        }


class TaskOutput(nn.Module):
    """Generic producer: ``preds.<stream>.<task>`` -> ``outputs.<stream>.<name>``.

    The copy/softmax/argmax/de-scale majority needs no dedicated class — one
    parameterised producer reads a task's published prediction leaf, applies a
    `ConversionOp` (the eval math) and writes an ``outputs.*`` leaf (design
    §4b). The default op is an identity copy (a real torch passthrough, P0
    byte-compat); P1 ships the classification + regression ops, and the thin
    `ClassProbs` / `SeqClassIndex` / `Regression` subclasses pre-select one.

    The produced leaf's last-dim width is re-emitted via the `derived_widths`
    hook (design §6.6, §4 risk 6): a producer that minted a *fresh* symbolic
    last dim would get no resolved width in a TEST-only bind (no FIT plan to
    unify against), so the H5 sink could not size columns and the exporter would
    have no concrete last dim. `derived_widths` maps the bound input width
    through the op's `derived_width` onto the output instead (identity / softmax
    / de-scale preserve it; ``argmax`` collapses it to 1), so the width resolves
    from the TEST plan alone.

    Both ports declare ``shape=None`` (rank-agnostic): a task's prediction is
    ``[B, C]`` for a global head and ``[B, T, C]`` for a sequence head, and the
    generic producer copies/converts either without knowing the rank statically.
    The last-dim width still resolves — the source task's own produce spec
    carries the concrete last dim, which `resolve_bind_schema` records for the
    ``preds.*`` key and `derived_widths` then maps onto the output.

    Parameters
    ----------
    task : str
        The source task's instance name — the producer reads
        ``preds.<stream>.<task>``.
    stream : str
        The stream the source task publishes under (``preds.<stream>.<task>``).
        Also the stream the output is written under
        (``outputs.<stream>.<name>``).
    name : str, optional
        The output leaf name (``outputs.<stream>.<name>``); defaults to `task`
        so a "just copy the task output" producer needs no extra config.
    op : ConversionOp | None, optional
        The eval-math conversion op, by default the identity copy (P0
        passthrough). P1 ops reproduce a task family's ``run_inference``
        VERBATIM (`ClassProbsOp` / `SeqClassIndexOp` / `RegressionDescaleOp`).
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        op: ConversionOp | None = None,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.task = task
        self.stream = stream
        self.output_name = name if name is not None else task
        self.op = op if op is not None else ConversionOp()
        self.pred_key = f"preds.{stream}.{task}"
        self.output_key = f"outputs.{stream}.{self.output_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``preds.<stream>.<task>`` (+ op extras) -> ``outputs.<stream>.<name>``.

        Both ports are active in every mode (``modes=ALL``, the default): the
        producer is gated by DEMAND, not by a hard mode flag (see the module
        docstring), so FIT/VAL drop it via the planner's demand closure rather
        than via mode-inactivity — the absence is then provably a prune (design
        §4 risk 4). The op contributes extra requires (pad mask for masked
        softmax / argmax; the ratio-denominator source ports for de-scaling).
        Both shapes are ``None`` (rank-agnostic — a global vs sequence head
        differ in rank); the output last-dim width is bound via `derived_widths`
        (design §6.6) from the source task's resolved ``preds.*`` width through
        the op's `derived_width`, so it resolves in a TEST-only bind (design §4
        risk 6). ``kind="data"`` (the default) — a prediction tensor, never a
        loss/label leaf — so the producer<-task edge kind-unifies.

        Returns
        -------
        IO
            The declared requires/produces for this producer.
        """
        del mode
        pred_spec = TensorSpec(shape=None, dtype="float32")
        out_spec = TensorSpec(shape=None, dtype="float32")
        requires: dict[str, TensorSpec] = {self.pred_key: pred_spec}
        requires.update(self.op.extra_requires(self.stream))
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({self.output_key: out_spec}),
        )

    def bind(self, schema: Any) -> None:
        """Delegate to the op's ``bind`` (capture input-Feature field order, if any).

        Only `RegressionDescaleOp` needs a bind (to capture the
        ``inputs.<stream>`` column order for the ONNX by-name denominator gather,
        ``tasks.py:2191-2202``); other ops have no ``bind`` and this no-ops. The
        op's fields are resolved from the schema only when the input Feature key
        declares any — in a TEST-only bind without an ``inputs.<stream>`` field
        declaration there is nothing to resolve (TEST sources denominators from
        labels by name, design §3.3), so the bind stays tolerant.
        """
        op_bind = getattr(self.op, "bind", None)
        if not callable(op_bind):
            return
        try:
            fields = schema.fields_of(self.op.input_feature_key)
        except Exception:  # noqa: BLE001 — no field declaration in this (e.g. TEST-only) bind
            fields = ()
        op_bind(fields)

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Map the bound input prediction width through the op onto the output (design §6.6).

        The bind fixpoint (`salt.core.nn.bind._apply_derived_widths`) calls this
        once the input width is resolved; returning ``{}`` until then keeps the
        hook order-insensitive across plans. The op's `derived_width` decides
        how the input width maps (preserve for softmax/de-scale; collapse to 1
        for ``argmax``).

        Returns
        -------
        dict[str, int]
            ``{outputs.<stream>.<name>: width}`` once the input width is known,
            else an empty dict.
        """
        pred_width = widths.get(self.pred_key)
        if pred_width is None:
            return {}
        return {self.output_key: self.op.derived_width(pred_width)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply the op to the prediction leaf, writing the ``outputs.*`` leaf.

        Returns
        -------
        dict[str, Tensor]
            The newly produced ``outputs.<stream>.<name>`` leaf only
            (design §2.5). The op returns a fresh tensor (the identity op
            ``clone``s; the conversion ops build new tensors), so the output
            never aliases the source ``preds.*`` leaf (write-once, design §2.1).
        """
        converted = self.op.convert(b, mode, pred_key=self.pred_key, stream=self.stream)
        return {self.output_key: converted}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """The field manifest this producer's ``outputs.*`` leaf expands into (W5.0).

        Resolves the wrapped task (`task`) from `model_modules` and delegates to the
        op's `output_columns`, so the auto-collecting sinks reproduce the legacy
        ``task.output_names()`` / ``task.onnx_outputs()`` schema EXACTLY (plan 31
        §3.1). Each op family decides the H5/ONNX names + dtype + axis + the H5/ONNX
        split (a per-token classification head is two producers — `SeqClassProbs`
        for the H5 prob columns, `SeqClassIndex` for the ONNX argmax index).

        Parameters
        ----------
        run_name : str
            The run name (the task's H5 column prefix; the op strips it to bare
            suffixes the sink re-prefixes).
        model_modules : Mapping[str, Any]
            The model-side module dict, threaded by the sink so the producer can
            resolve the task it wraps.

        Returns
        -------
        list[OutputField]
            The field manifest for this producer's output leaf.
        """
        task = _resolve_task(model_modules, self.task, f"producer {self.name!r}")
        return self.op.output_columns(task, run_name)


class ClassProbs(TaskOutput):
    """Global-classification probability producer (design §2, `ClassProbsOp`).

    A thin `TaskOutput` pre-wired with `ClassProbsOp`: reads a pooled
    classification head's ``preds.<stream>.<task>`` logits and writes the
    per-class probabilities (``softmax`` / ``sigmoid``) to
    ``outputs.<stream>.<name>``. Width-preserving (C classes -> C probs).

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The stream the head publishes under (also the output stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    bce : bool, optional
        Whether the head uses ``BCEWithLogitsLoss`` (per-class ``sigmoid``)
        instead of ``CrossEntropyLoss`` (``softmax``), by default False.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        bce: bool = False,
    ) -> None:
        super().__init__(task=task, stream=stream, name=name, op=ClassProbsOp(bce=bce))


class SeqClassIndex(TaskOutput):
    """Per-token class-index producer (design §2, `SeqClassIndexOp`).

    A thin `TaskOutput` pre-wired with `SeqClassIndexOp`: reads a sequence
    classification head's ``preds.<stream>.<task>`` logits, applies the masked
    softmax and ``argmax`` over the class dim, and writes the integer per-token
    class index (e.g. ``TrackOrigin``) to ``outputs.<stream>.<name>``. The class
    dim collapses, so the output last-dim width is 1.

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The constituent stream the head publishes under (also the output
        stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True
        (False for a fixed-count query bank, the MaskFormer ``objects``
        stream).
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        has_pad_mask: bool = True,
    ) -> None:
        super().__init__(
            task=task, stream=stream, name=name, op=SeqClassIndexOp(has_pad_mask=has_pad_mask)
        )


class SeqClassProbs(TaskOutput):
    """Per-token class-probability producer (design §2, `SeqClassProbsOp`).

    A thin `TaskOutput` pre-wired with `SeqClassProbsOp`: reads a sequence
    classification head's ``preds.<stream>.<task>`` logits, applies the masked
    softmax over the class dim, and writes the ``[B, L, C]`` per-token per-class
    probabilities (e.g. ``track_origin``'s 8 origin probs) to
    ``outputs.<stream>.<name>`` — the eval-H5 columns (`SeqClassIndex` is the
    ONNX argmax counterpart). Width-preserving (C classes -> C probs).

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The constituent stream the head publishes under (also the output stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        has_pad_mask: bool = True,
    ) -> None:
        super().__init__(
            task=task, stream=stream, name=name, op=SeqClassProbsOp(has_pad_mask=has_pad_mask)
        )


class Regression(TaskOutput):
    """Regression de-scaling producer (design §2, `RegressionDescaleOp`).

    A thin `TaskOutput` pre-wired with `RegressionDescaleOp`: reads a regression
    head's ``preds.<stream>.<task>`` training-space values and writes the
    de-scaled physical values (scaler / norm_params / ratio-denominator) to
    ``outputs.<stream>.<name>``. Width-preserving (R targets -> R values). For
    the ratio-denominator case the op consumes the denominator source
    (``labels.<stream>.<denom>`` in TEST; ``inputs.<stream>`` by name in ONNX).

    Parameters
    ----------
    task : str
        The source regression task's instance name.
    stream : str
        The regressed stream (also the output stream and the denominator-source
        stream).
    targets : str | Sequence[str]
        The regression target name(s), in column order (a bare string for one).
    name : str | None, optional
        The output leaf name, by default `task`.
    target_denominators : str | Sequence[str] | None, optional
        Per-target ratio-denominator variable name(s), by default None. Mutually
        exclusive with `norm_params` / `scaler`.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": ..., "std": ...}`` per-target normalisation, by default None.
    scaler : Mapping[str, Mapping[str, Any]] | None, optional
        Per-target functional scaling config, by default None.
    gaussian : bool, optional
        Whether the source head is a gaussian (``mu``/``sigma``) head publishing
        ``[..., 2R]`` (means ‖ raw variances), de-scaled to means ‖ ``stddev =
        sqrt(softplus(var)) * std`` (plan 34 W34.3, v1 ``GaussianRegressionTask.
        run_inference``), by default False. The descaled width stays ``2R``; the
        sink splits on ``_stddev`` via the gaussian-doubled ``output_suffixes``.
    sequence : bool, optional
        Whether the source head is per-token (a sequence stream), by default False.
        A seq head NaN-fills padded positions after de-scaling (v1 run_inference,
        tasks.py:557 / :648-650) — the SAME quirk the task's get_output reproduces,
        so the producer path matches get_output AND the legacy WriterCallback.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        targets: str | Sequence[str],
        name: str | None = None,
        target_denominators: str | Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        scaler: Mapping[str, Mapping[str, Any]] | None = None,
        gaussian: bool = False,
        sequence: bool = False,
    ) -> None:
        super().__init__(
            task=task,
            stream=stream,
            name=name,
            op=RegressionDescaleOp(
                stream=stream,
                targets=targets,
                target_denominators=target_denominators,
                norm_params=norm_params,
                scaler=scaler,
                gaussian=gaussian,
                sequence=sequence,
            ),
        )


class Combination(nn.Module):
    """Linear-combination producer: a NEW ``outputs.*`` leaf from a source bundle leaf (Q2).

    Folds the v1/M4.5 export combine loop (``adapter.py:279-280``,
    ``to_onnx.py:404-412``) into a normal conversion plan node (design §6.2 /
    Decisions-Locked Q2). It reads a SOURCE prob/pred bundle leaf — an
    ``outputs.<stream>.<src>`` leaf a producer already minted (e.g.
    ``outputs.jets.jets_classification`` ``[B, ..., C]`` softmaxed probs) — and
    produces a NEW ``outputs.<stream>.<name>`` scalar leaf as a weighted sum over
    its last-dim channels:

        ``out = sum(scale * source[..., index])`` over ``terms``

    which is bitwise-equal to v1's ``pb + pc`` computed on the renamed scalars
    (the same float adds, in the same `terms` order — the source leaf is the
    same softmaxed prob vector the ``split_scalars`` reduce splits into
    ``GN2v2_pb`` ...). Because it reads a BUNDLE leaf (not renamed Athena output
    names), the v1 name-space dependency disappears; both sinks consume/name the
    new leaf like any other ``outputs.*`` leaf.

    It is a normal `GraphModule` conversion node: ``declare_io`` requires the
    source ``outputs.*`` leaf (``kind=data``, the SAME kind the producer that
    minted it emits) and produces the new ``outputs.<stream>.<name>`` leaf;
    ``forward`` runs the sum inside the executor's step loop — per-batch in TEST
    AND once in the ONNX trace, like every other conversion. The output last dim
    collapses to a scalar (``derived_widths`` re-emits width 1), so the leaf is a
    GLOBAL float scalar with no per-token axis (v1's combines are global,
    ``to_onnx.py:404-412``). The Athena tuple ORDER residual (R3) is owned by the
    export node's output list, NOT by this node's topo position.

    Parameters
    ----------
    source : str
        The source bundle leaf, an ``outputs.<stream>.<src>`` key (a producer
        leaf, e.g. ``outputs.jets.jets_classification``). The new leaf is written
        under the SAME stream as the source.
    name : str
        The new output leaf's last component (``outputs.<stream>.<name>``), e.g.
        ``pbc``.
    terms : Mapping[int, float]
        Source last-dim channel index -> scale, in combination order (e.g.
        ``{0: 1.0, 1: 1.0}`` for ``probs[..., 0] + probs[..., 1]``). At least one
        term; every index must be a non-negative int.

    Raises
    ------
    ConfigError
        For a non-``outputs`` source, a wildcard source, an empty ``terms``, or a
        negative/non-int channel index.
    """

    def __init__(
        self,
        source: str,
        name: str,
        terms: Mapping[int, float],
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        parts = split_key(source)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"Combination source {source!r} contains a wildcard — conversion sources are "
                "concrete (design §2.2)"
            )
        if len(parts) < 2 or parts[0] != "outputs":
            raise ConfigError(
                f"Combination source {source!r} must be an 'outputs.<stream>.<name>' producer "
                "leaf — a combination reads a bundle prob/pred leaf, not a raw prediction or a "
                "renamed Athena output (design §6.2 / Q2)"
            )
        if not terms:
            raise ConfigError(
                f"Combination {name!r}: 'terms' must map at least one source channel index to a "
                "scale (e.g. {0: 1.0, 1: 1.0} for probs[..., 0] + probs[..., 1])"
            )
        self.source = source
        self.output_name = name
        self.stream = parts[1]
        # preserve declaration order (jsonargparse builds an ordered dict); the
        # sum order is load-bearing for the v1 float bitwise-equality (R3)
        self.terms: tuple[tuple[int, float], ...] = tuple(
            (self._checked_index(index, name), float(scale)) for index, scale in terms.items()
        )
        self.output_key = f"outputs.{self.stream}.{name}"

    @staticmethod
    def _checked_index(index: Any, name: str) -> int:
        """Validate a source channel index is a non-negative int.

        Returns
        -------
        int
            The validated index.

        Raises
        ------
        ConfigError
            For a non-int or negative index.
        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(
                f"Combination {name!r}: source channel index {index!r} must be a non-negative "
                "int (the source leaf's last-dim position to weight)"
            )
        return index

    def declare_io(self, mode: Mode) -> IO:
        """Declare the source ``outputs.*`` leaf -> the new ``outputs.<stream>.<name>`` leaf.

        Both ports are active in every mode (``modes=ALL``): like the other
        conversion producers, the combination is gated by DEMAND, not a hard mode
        flag (FIT/VAL prune it via the demand closure). The source require carries
        ``kind="data"`` (the kind the producer that minted the source leaf emits),
        so the edge kind-unifies; both shapes are ``None`` (rank-agnostic — the
        source is a prob vector and the output collapses its last dim).

        Returns
        -------
        IO
            The declared requires/produces for this combination node.
        """
        del mode
        requires = {self.source: TensorSpec(shape=None, dtype="float32", kind="data")}
        produces = {self.output_key: TensorSpec(shape=None, dtype="float32", kind="data")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The combination collapses the source last dim to a single scalar column.

        The combined value is a weighted sum over selected channels, so the
        output last-dim width is always 1 — re-emitted via the bind fixpoint hook
        (design §6.6) exactly like `SeqClassIndexOp`'s collapse, so the H5 sink can
        size the column from a TEST-only bind.

        Returns
        -------
        dict[str, int]
            ``{outputs.<stream>.<name>: 1}``.
        """
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute ``sum(scale * source[..., index])`` over `terms` (v1 ``to_onnx.py:404-412``).

        The sum runs in the SAME order as the configured `terms` (load-bearing
        for the v1 float bitwise-equality, R3). A fresh tensor results from the
        index-and-add, so the output never aliases the source leaf (write-once
        §2.1).

        Returns
        -------
        dict[str, Tensor]
            The new ``outputs.<stream>.<name>`` global scalar leaf only.
        """
        del mode
        source = b.get(self.source)
        out = sum(scale * source[..., index] for index, scale in self.terms)
        return {self.output_key: out}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One float global field named after the combination (``pbc``) — H5 + ONNX (W5.0).

        A combination is its own self-named output (no wrapped task): a single
        ``f4`` GLOBAL scalar under `output_name`, present in BOTH the H5 columns and
        the ONNX tuple (``onnx_name`` defaults to ``h5_name``).

        Returns
        -------
        list[OutputField]
            A single ``f4`` global field named `output_name`.
        """
        del run_name, model_modules
        return [OutputField(h5_name=self.output_name, dtype="f4", axis="global", final=True)]


class VertexUnionFind(nn.Module):
    """In-graph union-find conversion node (plan-29 W3, folds ``reduces._bind_vertex_union_find``).

    Folds the v1/M4.5 ``vertex_union_find`` export reduce (``reduces.py:592-624``,
    ``to_onnx.py:426-432``) into a normal conversion plan node (design §6.2 mapping
    table, the SHARPEST W3 fold — R1). It reads the RAW ``preds.<stream>.<task>``
    ``[E, 1]`` edge scores the vertexing task publishes in ONNX mode (design §3.3
    per-family exception) plus the stream's ``masks.<stream>`` pad mask, runs the
    ``@torch.jit.script`` union-find INSIDE ``forward(b, Mode.ONNX)``:

        ``get_node_assignment_jit`` (``salt.core.utils.union_find``, ``@torch.jit.script``)
        -> ``mask_fill_flattened`` (``reduces.py``, ``@torch.jit.script``)
        -> ``.reshape(-1).char()``

    producing the int8 ``[L]`` per-token leaf v1's exact chain emits. The two
    scripted helpers + the ``.reshape(-1).char()`` are reproduced VERBATIM by
    reusing ``reduces.py``'s already-inlined byte-faithful copies (one source of
    truth, no drift). The CRITICAL trace-placement risk (R1): the
    ``@torch.jit.script`` subgraph must inline IDENTICALLY when called one frame
    deeper through ``Executor.run``'s step loop vs the legacy post-executor
    ``reduce.fn`` — ``torch.onnx.export(dynamo=False)`` traces the executed op
    sequence, not the Python call structure, so the same scripted subgraph inlines
    at its call site regardless of caller frame. (W3 byte-diffs the folded
    VertexIndex ONNX vs the legacy reduce on the same weights to PROVE it.)

    It is a normal `GraphModule` conversion node: ``declare_io`` requires the raw
    ``preds.<stream>.<task>`` edge-score leaf (``kind=data``, the SAME port the
    reduce reads) + the stream's ``masks.<stream>`` (``kind=pad_mask``), and
    produces the new int8 ``outputs.<stream>.<name>`` leaf the `OnnxExportSink`
    names. Because it declares the SAME ``preds.*`` port the reduce reads today, the
    demand-closure pulls the IDENTICAL vertexing task node into the ONNX plan (R8 —
    no rerouting through ``outputs.*``). The ``reshape(-1)`` collapse has no
    recoverable last dim in a bind, so ``derived_widths`` re-emits width 1 (R6) —
    a per-token int8 column.

    This node is ONNX-only by construction: the union-find chain is shaped for the
    traced ``[1, L, ...]`` export batch (the all-valid pad mask + the fake-pad-track
    workaround inside ``get_node_assignment``); it is never wired into a TEST H5
    config (the eval vertex columns come from a separate per-token path). The
    ``forward`` runs the scripted chain unconditionally — the trace is the only
    consumer.

    Parameters
    ----------
    task : str
        The source vertexing task's instance name — the node reads the RAW
        ``preds.<stream>.<task>`` ``[E, 1]`` edge scores.
    stream : str
        The constituent stream the head publishes under (``preds.<stream>.<task>``
        and ``masks.<stream>``). Also the stream the output is written under
        (``outputs.<stream>.<name>``).
    name : str, optional
        The output leaf name (``outputs.<stream>.<name>``); defaults to `task`.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.task = task
        self.stream = stream
        self.output_name = name if name is not None else task
        self.pred_key = f"preds.{stream}.{task}"
        self.mask_key = f"masks.{stream}"
        self.output_key = f"outputs.{stream}.{self.output_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the RAW ``preds.*`` edge scores + ``masks.*`` -> the int8 ``outputs.*`` leaf.

        Both ports are active in every mode (``modes=ALL``): the node is gated by
        DEMAND, not a hard mode flag (FIT/VAL/TEST prune it via the demand-closure
        — only the ONNX export sink demands its leaf). The ``preds.*`` require is
        the SAME raw port the legacy ``vertex_union_find`` reduce reads today
        (``kind=data``), so the demand-closure keeps the identical vertexing task
        node alive in the ONNX plan (R8); the ``masks.<stream>`` require
        (``kind=pad_mask``) is the all-valid pad mask the union-find consumes. Both
        shapes are ``None`` (rank-agnostic — the edge scores are ``[E, 1]``, the
        output collapses to ``[L]``); the output width is re-emitted via
        `derived_widths` (R6).

        Returns
        -------
        IO
            The declared requires/produces for this conversion node.
        """
        del mode
        # plan 31 W5.1/W5.2: ONNX-ONLY ports. VertexUnionFind is ONNX-only BY
        # CONSTRUCTION (the union-find chain is shaped for the traced [1, L, ...]
        # export batch; the TEST vertex column is the vertexing task's own get_h5,
        # a DEFERRED H5 family with no conversion producer). Gating the ports to
        # Mode.ONNX (not the demand-gating the SHARED softmax producers use) makes the
        # node INACTIVE in FIT/VAL/TEST — so a config that opts the vertexing head out
        # of TEST does not trip the planner's pre-prune connectivity check on this
        # node's preds.* require (which would otherwise be unsatisfiable in TEST). It
        # also prevents accidental TEST wiring of the export-shaped int8 leaf.
        requires = {
            self.pred_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
            self.mask_key: TensorSpec(
                shape=("B", sym_dim("T", self.stream)), dtype="bool", kind="pad_mask",
                modes=Mode.ONNX,
            ),
        }
        produces = {
            self.output_key: TensorSpec(shape=None, dtype="int8", kind="data", modes=Mode.ONNX)
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The union-find ``reshape(-1)`` collapses to a single per-token index column (R6).

        The output is one int8 index per token, so the last-dim width is always 1 —
        re-emitted via the bind fixpoint hook (design §6.6) like `SeqClassIndexOp`'s
        collapse, so a sink can size the column from a bind alone (R6: the
        ``reshape(-1)`` has no recoverable last dim).

        Returns
        -------
        dict[str, int]
            ``{outputs.<stream>.<name>: 1}``.
        """
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the ``@torch.jit.script`` union-find chain VERBATIM (v1 ``to_onnx.py:426-432``).

        ``get_node_assignment_jit`` (the scripted union-find on the raw edge scores
        + the all-valid pad mask, with the fake-pad-track workaround inside) ->
        ``mask_fill_flattened`` (the scripted per-node -> batch unflatten) ->
        ``.reshape(-1).char()`` — the IDENTICAL chain `reduces._bind_vertex_union_find`
        runs, now inside the executor's step loop instead of the post-executor
        reduce loop (R1). A fresh tensor results, so the output never aliases a
        bundle leaf (write-once §2.1).

        Returns
        -------
        dict[str, Tensor]
            The new int8 ``outputs.<stream>.<name>`` per-token leaf only.
        """
        del mode
        edge_scores = b.get(self.pred_key)  # RAW [E, 1] scores (design §3.3)
        pad_mask = b.get(self.mask_key)  # the all-valid pad mask (to_onnx.py:428)
        vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
        vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
        return {self.output_key: vertex_list.reshape(-1).char()}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One int8 per-token ONNX field on the shared `VERTEX_INDEX` suffix (W5.0).

        The union-find vertex assignment is an ONNX-ONLY leaf (folds
        ``reduces._bind_vertex_union_find``): the legacy vertexing ``onnx_outputs``
        emits one int8 ``vertex_union_find`` entry under the shared `VERTEX_INDEX`
        suffix (v1 ``to_onnx.py:286-288``). It has NO auto-collected H5 column
        (``h5_name=None``): the TEST H5 vertex column is the vertexing TASK's own
        ``get_h5`` (a DEFERRED family with no conversion producer, plan 31 §6 W5
        scope) — vertexing-bearing configs opt the head out of TEST eval.

        Returns
        -------
        list[OutputField]
            One int8 per-token field (ONNX-only) on `VERTEX_INDEX`.
        """
        del run_name, model_modules
        return [
            OutputField(
                h5_name=None,
                onnx_name=VERTEX_INDEX,
                dtype="int8",
                axis="per_token",
                final=True,
            )
        ]


class MaskFormerObjects(nn.Module):
    """MaskFormer object RECONSTRUCTION node (plan-29 W3, USER DESIGN 2026-06-22 — "the writer").

    The reconstruction half of the TWO-NODE MaskFormer split (the user design
    SUPERSEDES the doc's single ``MaskFormerObject`` §6.2 description): it runs
    `get_maskformer_outputs` ONCE inside ``forward(b, Mode.ONNX)`` — the
    null-suppression + pT reorder + index math — and EXPOSES its products so a
    downstream `MFLeadVertexDecorator` can READ the reordered per-vertex outputs
    without re-doing any heavy lifting (the user: the decorator is "basically just
    reading the vertex outputs that the writer outputs"). One node produces:

    - **object_index** (``outputs.<constituent_stream>.<index_name>``, int8
      PER-TOKEN): the per-constituent owning-object index,
      ``indices.reshape(-1).char()`` (folds ``reduces._bind_object_index`` BITWISE
      vs the legacy reduce, the §6.2 object_index row).
    - **leading_object** (``outputs.<stream>.<leading_name>``, float32 GLOBAL): the
      leading object's R de-scaled regression scalars (folds
      ``reduces._bind_leading_object``, the §6.2 leading_object row) — kept so the
      single-node fold gate stays valid and so back-compat ``leading_object``
      configs have a folded path; the legacy ``leading_object`` reduce stays
      UNTOUCHED (additive).
    - **vertices_class_probs** (``outputs.<stream>.<vertices_class_probs_name>``,
      float32 ``[B, M, C]``) and **vertices_regression**
      (``outputs.<stream>.<vertices_regression_name>``, float32 ``[B, M, R]``): the
      reordered (null-suppressed + pT-ordered) per-vertex class probabilities +
      regression tensors `get_maskformer_outputs` returns (its 3rd / 4th returns),
      exposed as intermediate ``outputs.*`` leaves for the `MFLeadVertexDecorator`
      (a node->node edge — the decorator's demand keeps THIS node alive).

    `get_maskformer_outputs` is reproduced VERBATIM by reusing ``reduces.py``'s
    already-inlined byte-faithful copy (one source of truth — the v1
    ``salt.models.maskformer.get_maskformer_outputs``, byte-faithful per the
    reduces.py docstring).

    Cross-node reads + clone discipline (R4): the node DECLARES all three reads it
    needs — ``objects.class_probs`` / ``objects.masks`` / ``preds.<stream>.<reg_task>``
    — as ``declare_io`` requires, so the demand-closure keeps the MaskDecoder + the
    object-regression task alive in the ONNX plan (and the debug-mode
    ``_ReadTrackedBundle`` never raises ``UndeclaredAccessError``). The three tensors
    are CLONED before `get_maskformer_outputs` (which mutates ``masks``/``regression``
    in place for null-suppression + pT reorder), so the write-once bundle is never
    mutated (else ``MutationError`` under debug, design §2.1).

    Derived widths (R6): ``object_index``'s ``reshape(-1)`` collapse has no
    recoverable last dim, so ``derived_widths`` re-emits width 1 for the index leaf;
    the leading-regression leaf keeps the source regression width (R targets); the
    per-vertex leaves are ``[B, M, C]`` / ``[B, M, R]`` whose last dim is C / R.

    ONNX-only by construction (like `VertexUnionFind`): the null-suppression +
    pT-reorder chain is shaped for the traced export batch; it is never wired into a
    TEST H5 config (the eval object columns come from the `MaskFormerObjectWriter`'s
    own TEST path).

    Parameters
    ----------
    regression_task : str
        The object-regression task's instance name — the node reads the DE-SCALED
        ``preds.<stream>.<regression_task>`` ``[B, M, R]`` predictions (v1's single
        object-regression key). The legacy reduces hardcode ``"regression"``; this
        node threads the name so a non-default task is supported.

        PARITY NOTE (R4): the folded ``object_index`` / ``leading_object`` leaves are
        bitwise-equal to the LEGACY ``_bind_object_index`` / ``_bind_leading_object``
        reduces ONLY when ``regression_task == "regression"`` — the legacy default the
        ``MaskFormerObjectWriter`` ASSERTS for ONNX export (``writers/maskformer.py:443``,
        because the legacy ``object_index`` reduce is declared on the masks port and so
        always reads the FIXED ``preds.<stream>.regression`` key). Threading a
        non-default ``regression_task`` reorders by a DIFFERENT regression tensor than
        the legacy reduce, so it is an UNTESTED superset of legacy behaviour with no
        parity oracle. It is internally consistent (the single ``get_maskformer_outputs``
        call reads the threaded key for both folds, so there is no stale-key trace
        error), but it has no folded==legacy guarantee. Keep ``regression_task`` at its
        ``"regression"`` default for any config that must match the legacy ONNX path.
    stream : str, optional
        The object stream the decoder publishes under (``objects.class_probs`` /
        ``objects.masks`` / ``preds.<stream>.<regression_task>``), by default
        ``"objects"``. Also the object output stream (leading + vertices leaves).
    leading_name : str, optional
        The leading-object regression leaf name (``outputs.<stream>.<leading_name>``),
        by default ``"leading_object"``.
    index_name : str, optional
        The per-constituent object-index leaf name (``outputs.<stream>.<index_name>``),
        by default ``"object_index"``.
    n_reg : int
        The object-regression target count R (the leading-regression output width).
        The leading leaf is sliced to ``leading_reg[:, :n_reg]`` so it reproduces the
        legacy ``leading_object`` reduce's ``leading_reg[0, i] for i in range(R)``
        (``reduces.py:728``) EXACTLY — including v1's no-objects/empty-track dummy
        path, where ``get_maskformer_outputs`` returns a ``[1, n_obj]`` leading
        tensor (``reduces.py:209``) and v1 takes only the first R columns. Must
        match the configured leading-object ``names`` count.
    constituent_stream : str, optional
        The constituent stream the per-token index leaf is written under and whose
        dynamic axis the index carries, by default ``"tracks"`` (v1's fixed
        ``aux_sequence_object``). The index leaf is
        ``outputs.<constituent_stream>.<index_name>``.
    vertices_class_probs_name : str, optional
        The exposed reordered per-vertex class-probs leaf name
        (``outputs.<stream>.<name>``), by default ``"vertices_class_probs"``.
    vertices_regression_name : str, optional
        The exposed reordered per-vertex regression leaf name
        (``outputs.<stream>.<name>``), by default ``"vertices_regression"``.
    """

    def __init__(
        self,
        n_reg: int,
        regression_task: str = "regression",
        stream: str = "objects",
        leading_name: str = "leading_object",
        index_name: str = "object_index",
        constituent_stream: str = "tracks",
        vertices_class_probs_name: str = "vertices_class_probs",
        vertices_regression_name: str = "vertices_regression",
    ) -> None:
        super().__init__()
        if not isinstance(n_reg, int) or isinstance(n_reg, bool) or n_reg < 1:
            raise ConfigError(
                f"MaskFormerObjects: n_reg must be a positive int (the leading-object regression "
                f"target count R, matching the export leading names), got {n_reg!r}"
            )
        self.name = _UNNAMED
        self.stream = stream
        self.constituent_stream = constituent_stream
        self.regression_task = regression_task
        self.leading_name = leading_name
        self.index_name = index_name
        self.n_reg = n_reg
        self.class_probs_key = f"{stream}.class_probs"
        self.masks_key = f"{stream}.masks"
        self.reg_key = f"preds.{stream}.{regression_task}"
        # the GLOBAL leading-regression leaf is written under the OBJECT stream; the
        # PER-TOKEN index leaf under the CONSTITUENT stream (its dynamic axis source)
        self.leading_key = f"outputs.{stream}.{leading_name}"
        self.index_key = f"outputs.{constituent_stream}.{index_name}"
        # the exposed reordered per-vertex outputs (object stream) the decorator reads
        self.vertices_class_probs_key = f"outputs.{stream}.{vertices_class_probs_name}"
        self.vertices_regression_key = f"outputs.{stream}.{vertices_regression_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ALL three maskformer reads -> the leading-regression + object-index leaves (R4).

        Both products are active in every mode (``modes=ALL``): the node is gated
        by DEMAND, not a hard mode flag. It DECLARES all three cross-node reads —
        ``objects.class_probs`` / ``objects.masks`` (the decoder products) and
        ``preds.<stream>.<reg_task>`` (the de-scaled object-regression
        predictions), all ``kind=data`` — so the demand-closure keeps the
        MaskDecoder + the regression task alive in the ONNX plan and the
        debug-mode read-tracker never raises (R4). It produces BOTH the float32
        GLOBAL leading-regression leaf (object stream) and the int8 PER-TOKEN
        object-index leaf (constituent stream). All shapes are ``None``
        (rank-agnostic); the index width is re-emitted via `derived_widths` (R6),
        the leading width follows the regression port (or `n_reg`).

        Returns
        -------
        IO
            The declared requires/produces for this conversion node.
        """
        del mode
        # ONNX-ONLY ports (W6b — same gate as `VertexUnionFind`). The
        # null-suppression + pT-reorder `get_maskformer_outputs` chain is shaped for
        # the traced export batch (this node is "ONNX-only by construction", the class
        # docstring), so its ports are gated to `Mode.ONNX` rather than the
        # demand-gating the SHARED softmax producers use. This makes the node INACTIVE
        # in FIT/VAL/TEST — so a config that wires it for ONNX export alongside an
        # object-regression head opted OUT of TEST eval (``expose: [fit, val, onnx]``,
        # the MaskFormer.yaml W6b cutover) does NOT trip the planner's pre-prune
        # connectivity check on this node's ``preds.<stream>.<reg_task>`` require
        # (unsatisfiable in TEST, where the regression pred is not exposed). The ONNX
        # plan_hash is unchanged (the ports are identical in ONNX).
        requires = {
            self.class_probs_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.masks_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
            self.reg_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
        }
        produces = {
            self.leading_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.index_key: TensorSpec(shape=None, dtype="int8", kind="data", modes=Mode.ONNX),
            # the exposed reordered per-vertex outputs the MFLeadVertexDecorator reads
            # (a node->node edge — the decorator's demand keeps this node alive)
            self.vertices_class_probs_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.vertices_regression_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Width-resolve the leaves: index collapses to 1 (R6), leading follows the reg port.

        The per-constituent index leaf's ``reshape(-1)`` has no recoverable last dim,
        so it is always width 1 (R6, like `SeqClassIndexOp`'s collapse). The
        leading-regression leaf is the first `n_reg` regression targets (R), so its
        width is the configured `n_reg` — both resolve from a bind alone (R6: the
        ``reshape(-1)`` / dummy-path shapes have no recoverable last dim). The exposed
        per-vertex leaves keep ``[B, M, C]`` / ``[B, M, R]`` last dims — the vertex
        regression width is `n_reg`; the class-probs width has no recoverable last dim
        in a bind (the decoder's class count is not threaded), so it is left
        unresolved (the decorator consumes it by trace, not by a sized H5 column).

        Returns
        -------
        dict[str, int]
            ``{index leaf: 1, leading leaf: n_reg, vertices_regression leaf: n_reg}``.
        """
        del widths
        return {
            self.index_key: 1,
            self.leading_key: self.n_reg,
            self.vertices_regression_key: self.n_reg,
        }

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run ``get_maskformer_outputs`` ONCE -> object_index + leading + the per-vertex leaves.

        ONE `get_maskformer_outputs` call (null-suppression + pT reorder) over the
        CLONED object trio (R4 — never mutate the write-once bundle) yields the
        leading regression ``[B, R]``, the per-constituent index ``[B, L]`` AND the
        reordered per-vertex ``class_probs [B, M, C]`` / ``regression [B, M, R]`` (its
        3rd / 4th returns). The leading scalars (``leading_reg[0]``, v1
        ``to_onnx.py:467-468``) slice into the global ``[B, R]`` leaf, the indices
        ``reshape(-1).char()`` (v1 ``to_onnx.py:469``) into the int8 ``[L]`` leaf —
        the SAME tensors the two legacy reduces emit — and the reordered per-vertex
        tensors pass through as the intermediate leaves the `MFLeadVertexDecorator`
        reads (the user design: the decorator just reads what "the writer" exposes).

        Returns
        -------
        dict[str, Tensor]
            The float32 GLOBAL leading-regression leaf, the int8 PER-TOKEN
            object-index leaf, and the two reordered per-vertex leaves.
        """
        del mode
        objects = {
            "class_probs": b.get(self.class_probs_key).clone(),
            "masks": b.get(self.masks_key).clone(),
            "regression": b.get(self.reg_key).clone(),
        }
        leading_reg, indices, vertices_class_probs, vertices_regression = get_maskformer_outputs(
            objects, apply_reorder=True
        )
        # `get_maskformer_outputs` returns indices=None at L == 0 (n_tracks == 0,
        # reduces.py:210). This NEVER happens on the export path — torch.onnx.export
        # always traces at a fixed L > 0 and onnxruntime then runs L == 0 through the
        # traced graph (the gate confirms it passes), so `indices` is a real tensor at
        # trace time and this guard folds to a constant-False branch that emits NO ops
        # (the same trace-safe pattern as the `n_obj == 0` guard in the decorator —
        # it cannot perturb the byte-identical L > 0 trace the GO2 fold gate pins). The
        # guard is defensive insurance only, removing the latent AttributeError a future
        # eager TEST-mode wiring of this node would hit (R1/R4 low-severity foot-gun);
        # legacy `_bind_object_index` (reduces.py:786) has the same unguarded crash.
        if indices is None:
            empty_index = torch.zeros(0, dtype=torch.int8)
        else:
            empty_index = indices.reshape(-1).char()  # v1 to_onnx.py:469
        # leading_reg is [B, R] in the normal path (maskformer.py:344) but [1, n_obj]
        # in v1's no-objects/empty-track dummy path (reduces.py:209,212,219,222); the
        # legacy reduce takes only the first R columns (leading_reg[0, i] for i in
        # range(R), reduces.py:728), so slice to [:, :n_reg] to reproduce both paths
        # EXACTLY. Emit the GLOBAL [B, R] leaf so the OnnxExportSink's split_scalars
        # names it into R per-target scalars (split(value, 1, -1).squeeze(), the SAME
        # naming split a ClassProbs softmax leaf rides — no per-target stack node).
        return {
            self.leading_key: leading_reg[:, : self.n_reg],
            self.index_key: empty_index,  # indices.reshape(-1).char() (v1 to_onnx.py:469)
            # the reordered (null-suppressed + pT-ordered) per-vertex outputs — passed
            # through verbatim as get_maskformer_outputs returns them (the decorator
            # does the lead-vertex selection; all heavy lifting is HERE, R4-cloned)
            self.vertices_class_probs_key: vertices_class_probs,
            self.vertices_regression_key: vertices_regression,
        }

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """The MaskFormer object leaves' field manifest — index + leading + intermediates (W5.0).

        MaskFormer eval/export migration is W6-DEFERRED (plan 31 §5): this manifest
        is provided so the auto-collecting sinks can be unit-tested for the
        ``final`` distinction (plan 31 R-C — the per-vertex leaves the
        `MFLeadVertexDecorator` reads must NOT be auto-collected). The
        ``object_index`` is the int8 per-token ONNX leaf (folds
        ``_bind_object_index``); the leading-regression is the global float ONNX
        leaf (folds ``_bind_leading_object``). The per-vertex ``vertices_class_probs``
        / ``vertices_regression`` are EXPOSED INTERMEDIATES (``final=False``).
        Resolving the legacy MaskFormer Athena names (``HadronIndex`` / per-object
        leading suffixes) stays a W6 concern; the W6 sink uses the explicit
        override, so the names here are the bare leaf names (placeholders that the
        dup-guard / final-flag unit tests exercise without claiming legacy parity).

        Returns
        -------
        list[OutputField]
            object_index (int8 per-token, final), leading_object (f4 global, final),
            and the two per-vertex intermediates (final=False).
        """
        del run_name, model_modules
        return [
            OutputField(
                h5_name=None, onnx_name=self.index_name, dtype="int8",
                axis="per_token", final=True,
            ),
            OutputField(h5_name=self.leading_name, dtype="f4", axis="global", final=True),
            OutputField(
                h5_name="vertices_class_probs", onnx_name=None, dtype="f4",
                axis="per_token", final=False,
            ),
            OutputField(
                h5_name="vertices_regression", onnx_name=None, dtype="f4",
                axis="per_token", final=False,
            ),
        ]


# DEPRECATED one-window alias (USER DESIGN 2026-06-22): the single-node
# ``MaskFormerObject`` is renamed ``MaskFormerObjects`` (the reconstruction node /
# "the writer") under the two-node split. The promoted node is a strict superset
# (same folds, plus the exposed per-vertex leaves), so an existing
# ``MaskFormerObject`` config / fold test keeps working — the alias resolves to the
# promoted node. Remove after the migration window.
MaskFormerObject = MaskFormerObjects
"""Deprecated alias for `MaskFormerObjects` (the two-node MaskFormer split rename)."""


class MFLeadVertexDecorator(nn.Module):
    """MaskFormer lead-vertex jet-level DECORATOR (plan-29 W3 NEW capability, USER DESIGN).

    The decoration half of the TWO-NODE MaskFormer split — a THIN selector that
    READS `MaskFormerObjects`'s exposed reordered per-vertex leaves
    (``vertices_class_probs [B, M, C]`` + ``vertices_regression [B, M, R]``) and
    emits jet-level GLOBAL scalar leaves (``jet.lead_vertex_pt`` /
    ``jet.lead_vertex_mass`` / ...). It is a brand-new jet-level capability (NOT a
    parity fold of any legacy reduce, so it has NO legacy oracle — it is UNIT-TESTED
    on hand-built inputs), separating object RECONSTRUCTION (Node 1a, the writer)
    from jet-level DECORATION (this node). The user: it is "basically just reading
    the vertex outputs that the writer outputs".

    The LEAD VERTEX is selected per-jet as the **highest-pT vertex** (by
    ``vertices_regression[..., pt_index]``) among the vertices that are ALL of

    - not null: ``vertices_class_probs[..., null_index] < pnull_threshold`` (the
      same null-probability cut `get_maskformer_outputs` applies, default 0.5), AND
    - not the primary vertex: the vertex's argmax predicted class
      (``argmax(vertices_class_probs[..., :])``) is NOT `pv_class_index`, AND
    - a real vertex class: the argmax predicted class is NOT `null_index` (user
      sign-off 2026-06-22 — with >=3 classes a vertex can have ``argmax==null`` yet
      ``pnull < threshold``; this third cut requires the most-likely class to be an
      actual vertex, not null).

    For each configured output ``{name: reg_index}`` it pulls the selected vertex's
    ``vertices_regression[..., reg_index]`` and writes the jet-level scalar
    ``outputs.<jet_stream>.<name>``.

    NOTE: this lead-vertex selection DIFFERS from the legacy ``leading_object``
    reduce (pT-only, NO PV exclusion / null cut on the *decorator* side — though
    null suppression already happened in Node 1a). It is therefore a NEW output, NOT
    a relocation; the legacy ``leading_object`` reduce + Node 1a's leading_object
    leaf both stay UNTOUCHED (additive).

    Trace-safe no-qualifying-vertex fill (R4/R6 discipline): when NO vertex
    qualifies (all-null jet, all-PV jet, or ``M == 0`` / ``L == 0`` empty inputs),
    the jet-level scalars are filled with NaN deterministically — a masked-argmax
    over a ``[B, M]`` validity mask with an all-``-inf`` pT column gathers a
    well-defined index whose outputs are then NaN-overwritten where the jet has no
    qualifying vertex, so the trace stays valid for every batch shape (no
    data-dependent control flow).

    Dummy-path NaN (surprising but consistent with v1 dummy semantics): when Node
    1a's `get_maskformer_outputs` hits its ``not null_preds.any()`` dummy path (NO
    object exceeds the null threshold, i.e. every object "looks real"), it returns
    an ALL-NaN ``vertices_regression`` while ``vertices_class_probs`` flows through
    REAL (low-pnull, non-NaN). The decorator then sees ``qualify=True`` for those
    vertices (low pnull, argmax != PV) but their pT is NaN, so ``masked_pt`` is NaN,
    ``any_qualify`` is True (the NaN-fill guard does NOT fire), and it gathers NaN
    regression -> the jet-level scalars are NaN. Likewise the ``n_tracks == 0`` /
    ``L == 0`` dummy path returns all-NaN regression. So lead-vertex scalars are NaN
    whenever Node 1a's dummy path is active — not only on the all-null / all-PV /
    ``M == 0`` paths above. This matches v1's "no predicted objects -> dummy NaN"
    semantics (v1's ``leading_object`` is also NaN there); surfaced here because the
    qualify mask passing vertices whose regression is undefined is counterintuitive.

    Parameters
    ----------
    source : str
        The `MaskFormerObjects` exposed per-vertex CLASS-PROBS leaf
        (``outputs.<object_stream>.<vertices_class_probs_name>``, ``[B, M, C]``). The
        regression source defaults to the same object stream's
        ``vertices_regression`` leaf unless `regression_source` overrides it.
    outputs : Mapping[str, int]
        ``{output_name: reg_index}`` — each jet-level scalar leaf
        ``outputs.<jet_stream>.<output_name>`` pulls the lead vertex's
        ``vertices_regression[..., reg_index]``.
    pt_index : int
        The ``vertices_regression`` channel that is the vertex pT (the selection
        key — highest pT wins). Must index a configured / valid regression channel.
    pv_class_index : int
        The vertex class index that marks the PRIMARY vertex (excluded from the
        lead-vertex selection).
    pnull_threshold : float, optional
        The null-probability cut: a vertex with
        ``class_probs[..., null_index] >= pnull_threshold`` is excluded, by default
        0.5 (the `get_maskformer_outputs` default).
    null_index : int | None, optional
        The class index of the NULL class in ``vertices_class_probs``, by default
        None = the LAST class (the v1 ``class_probs[:, :, -1]`` null convention,
        ``reduces.py:215``).
    jet_stream : str, optional
        The jet-level output stream (``outputs.<jet_stream>.<name>``), by default
        ``"jet"``.
    regression_source : str | None, optional
        Override for the per-vertex REGRESSION leaf, by default None = the
        ``source`` object stream's ``vertices_regression`` leaf.

    Raises
    ------
    ConfigError
        For a non-``outputs`` / wildcard source, an empty `outputs` map, a
        negative/non-int reg index or pt_index, or a missing pt selection channel.
    """

    def __init__(
        self,
        source: str,
        outputs: Mapping[str, int],
        pt_index: int,
        pv_class_index: int,
        pnull_threshold: float = 0.5,
        null_index: int | None = None,
        jet_stream: str = "jet",
        regression_source: str | None = None,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        parts = split_key(source)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"MFLeadVertexDecorator source {source!r} contains a wildcard — conversion "
                "sources are concrete (design §2.2)"
            )
        if len(parts) < 2 or parts[0] != "outputs":
            raise ConfigError(
                f"MFLeadVertexDecorator source {source!r} must be a "
                "'outputs.<object_stream>.<vertices_class_probs>' leaf the MaskFormerObjects "
                "node exposes (the per-vertex class probs) — it reads a bundle leaf, not a raw "
                "prediction (USER DESIGN 2026-06-22)"
            )
        if not outputs:
            raise ConfigError(
                "MFLeadVertexDecorator: 'outputs' must map at least one jet-level scalar name to "
                "the lead vertex's regression channel index (e.g. {lead_vertex_pt: 0})"
            )
        self.source = source
        self.object_stream = parts[1]
        # the regression leaf defaults to the same object stream's vertices_regression
        self.regression_source = (
            regression_source
            if regression_source is not None
            else f"outputs.{self.object_stream}.vertices_regression"
        )
        reg_parts = split_key(self.regression_source)
        if len(reg_parts) < 2 or reg_parts[0] != "outputs":
            raise ConfigError(
                f"MFLeadVertexDecorator regression_source {self.regression_source!r} must be a "
                "'outputs.<object_stream>.<vertices_regression>' leaf (the per-vertex regression)"
            )
        self.jet_stream = jet_stream
        self.pt_index = self._checked_index(pt_index, "pt_index")
        self.pv_class_index = self._checked_index(pv_class_index, "pv_class_index")
        self.pnull_threshold = float(pnull_threshold)
        self.null_index = null_index if null_index is None else self._checked_index(
            null_index, "null_index"
        )
        # preserve declaration order (jsonargparse builds an ordered dict)
        self.outputs_map: tuple[tuple[str, int], ...] = tuple(
            (name, self._checked_index(idx, f"outputs[{name!r}]")) for name, idx in outputs.items()
        )
        self.output_keys: tuple[str, ...] = tuple(
            f"outputs.{jet_stream}.{name}" for name, _ in self.outputs_map
        )

    @staticmethod
    def _checked_index(index: Any, what: str) -> int:
        """Validate an index is a non-negative int.

        Returns
        -------
        int
            The validated index.

        Raises
        ------
        ConfigError
            For a non-int or negative index.
        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(
                f"MFLeadVertexDecorator: {what} index {index!r} must be a non-negative int"
            )
        return index

    def declare_io(self, mode: Mode) -> IO:
        """Declare the two per-vertex source leaves -> the jet-level scalar leaves.

        Both ports are active in every mode (``modes=ALL``): the decorator is gated
        by DEMAND, not a hard mode flag (FIT/VAL prune it via the demand-closure).
        The per-vertex class-probs + regression requires (``kind=data``) are the
        leaves `MaskFormerObjects` mints — a node->node edge whose demand keeps the
        reconstruction node alive — and the node produces one GLOBAL jet-level scalar
        leaf per configured output. All shapes ``None`` (rank-agnostic); each output
        is a scalar so `derived_widths` re-emits width 1.

        Returns
        -------
        IO
            The declared requires/produces for this decorator node.
        """
        del mode
        requires = {
            self.source: TensorSpec(shape=None, dtype="float32", kind="data"),
            self.regression_source: TensorSpec(shape=None, dtype="float32", kind="data"),
        }
        produces = {
            key: TensorSpec(shape=None, dtype="float32", kind="data") for key in self.output_keys
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Every jet-level output is a single scalar column (width 1).

        Returns
        -------
        dict[str, int]
            ``{output leaf: 1}`` per configured output.
        """
        del widths
        return dict.fromkeys(self.output_keys, 1)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Select the lead vertex (highest-pT, non-null, non-PV) and emit jet-level scalars.

        Trace-safe, no data-dependent control flow:

        1. Read ``class_probs [B, M, C]`` + ``regression [B, M, R]`` (the exposed
           reordered per-vertex leaves). For ``M == 0`` (no objects) every output is
           an all-NaN ``[B]`` scalar (handled explicitly — masked-argmax over an
           empty object axis is undefined).
        2. Build the per-vertex QUALIFY mask: ``pnull < threshold`` (null-prob cut)
           AND ``argmax(class_probs) != pv_class_index`` (PV exclusion). Null
           vertices are excluded by the ``pnull < threshold`` cut — the SAME
           null-probability cut Node 1a's `get_maskformer_outputs` applies. Note
           Node 1a NaN-suppresses ``regression`` (and zeros ``masks``) for null
           vertices but does NOT NaN ``class_probs`` (it only reorders it), so a
           class-probs row is never NaN on the real two-node chain; the cut is the
           sole exclusion mechanism. (As defensive belt-and-braces, a NaN-class row
           would still be excluded — ``NaN < threshold`` is False — but that input
           does not occur upstream.)
        3. Masked-argmax the pT column (``regression[..., pt_index]`` with
           non-qualifying vertices set to ``-inf``) -> the lead-vertex index per jet.
        4. Gather each configured ``regression[..., reg_index]`` at the lead index;
           where NO vertex qualifies (the masked pT row is all ``-inf``), overwrite
           the output with NaN (the deterministic empty/all-null/all-PV fill).

        Returns
        -------
        dict[str, Tensor]
            ``{outputs.<jet_stream>.<name>: [B] float32 scalar}`` per configured output.
        """
        del mode
        class_probs = b.get(self.source)  # [B, M, C]
        regression = b.get(self.regression_source)  # [B, M, R]
        n_obj = class_probs.shape[1]
        batch = class_probs.shape[0]
        if n_obj == 0:
            # no object queries -> no vertex can qualify; every output is all-NaN.
            # This python-bool branch on M bakes a constant into the trace (a
            # TracerWarning), but M (the object-query axis) is FIXED by the decoder
            # and is NEVER a declared dynamic export input — no export config makes M
            # dynamic — so the frozen branch is harmless. Do NOT "fix" the warning by
            # making this data-dependent; that would break the trace if M were ever 0.
            nan = torch.full((batch,), torch.nan, dtype=torch.float32)
            return dict.fromkeys(self.output_keys, nan)
        null_idx = self.null_index if self.null_index is not None else class_probs.shape[-1] - 1
        pnull = class_probs[..., null_idx]  # [B, M]
        pred_class = torch.argmax(class_probs, dim=-1)  # [B, M]
        # qualify = not-null AND not-PV AND argmax-is-a-real-vertex-class.
        # THREE conditions (user sign-off 2026-06-22):
        #   (a) pnull < threshold      — the null-prob cut (the SAME cut Node 1a's
        #       get_maskformer_outputs applies);
        #   (b) argmax != pv_class_index — exclude the primary vertex;
        #   (c) argmax != null_idx     — the vertex's MOST-LIKELY class must be a real
        #       vertex class, NOT null. With >=3 classes a vertex can have argmax==null
        #       yet pnull<threshold (thin-spread probs, e.g. [.1,.15,.15,.2,.4]); (a)
        #       alone would let it qualify, so (c) is required for "actually a vertex".
        # Node 1a NaNs only `regression`/`masks`, never `class_probs` (it just reorders
        # it), so a class-probs row is never NaN on the real chain; NaN < threshold is
        # False, so a (non-occurring) NaN-class vertex is excluded — defensive only.
        qualify = (
            (pnull < self.pnull_threshold)
            & (pred_class != self.pv_class_index)
            & (pred_class != null_idx)
        )  # [B, M]
        pt = regression[..., self.pt_index]  # [B, M]
        # mask non-qualifying vertices to -inf so the argmax never picks them
        masked_pt = torch.where(qualify, pt, torch.full_like(pt, -torch.inf))  # [B, M]
        lead = torch.argmax(masked_pt, dim=-1)  # [B] index of the lead vertex per jet
        any_qualify = qualify.any(dim=-1)  # [B] does this jet have ANY lead vertex?
        out: dict[str, Tensor] = {}
        lead_exp = lead.unsqueeze(-1)  # [B, 1] for gather along the object axis
        for (key, (_name, reg_index)) in zip(self.output_keys, self.outputs_map, strict=True):
            col = regression[..., reg_index]  # [B, M]
            value = torch.gather(col, 1, lead_exp).squeeze(1)  # [B] lead-vertex value
            # deterministic NaN fill where no vertex qualifies (empty/all-null/all-PV)
            value = torch.where(any_qualify, value, torch.full_like(value, torch.nan))
            out[key] = value.float()
        return out

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One float global jet-level field per configured lead-vertex scalar (W5.0).

        The decorator's jet-level scalars (``lead_vertex_pt`` / ``lead_vertex_mass``
        / ...) are each a single ``f4`` GLOBAL output, present in both H5 and ONNX
        (``onnx_name`` defaults to ``h5_name``). A NEW capability with no legacy
        oracle (plan 31 W6 / design W3); provided for the auto-collect sinks +
        unit tests.

        Returns
        -------
        list[OutputField]
            One ``f4`` global field per configured output name, in declaration order.
        """
        del run_name, model_modules
        return [
            OutputField(h5_name=name, dtype="f4", axis="global", final=True)
            for name, _ in self.outputs_map
        ]
