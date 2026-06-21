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
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.utils.array_utils import listify
from salt.core.utils.scalers import RegressionTargetScaler

__all__ = [
    "ClassProbs",
    "ClassProbsOp",
    "ConversionOp",
    "IdentityOp",
    "Regression",
    "RegressionDescaleOp",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "TaskOutput",
]

_UNNAMED = "unnamed"
"""Placeholder instance name — config assembly assigns the dict key (design §2.2)."""


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

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax then per-token ``argmax`` over the class dim.

        Returns
        -------
        Tensor
            ``[B, L]`` integer per-token class indices (``int64``).
        """
        del mode
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
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
    targets : Sequence[str]
        The regression target names, in column order (R targets). Used to index
        the functional scaler and to size the descale loop, matching v1's
        ``range(len(self.targets))``.
    target_denominators : Sequence[str] | None, optional
        Per-target ratio-denominator variable names (``target/denom`` at train,
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
        targets: Sequence[str],
        target_denominators: Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        scaler: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.stream = stream
        self.targets = tuple(targets)
        if not self.targets:
            raise ConfigError("RegressionDescaleOp: targets is required and non-empty")
        self.target_denominators = (
            tuple(listify(target_denominators)) if target_denominators is not None else None
        )
        self.norm_params = self._checked_norm_params(norm_params)
        self.scaler = RegressionTargetScaler(dict(scaler)) if scaler is not None else None
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

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the ratio-denominator sources (FD §3.3 mode-split de-scaling).

        FIT|VAL|TEST read each denominator from ``labels.<stream>.<denom>``;
        ONNX gathers them by name from the raw ``inputs.<stream>`` Feature
        tensor. norm_params / scaler need no external source.

        Returns
        -------
        dict[str, TensorSpec]
            The denominator-source ports (empty when not a ratio head).
        """
        if self.target_denominators is None:
            return {}
        out: dict[str, TensorSpec] = {}
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
        return preds

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
    targets : Sequence[str]
        The regression target names, in column order.
    name : str | None, optional
        The output leaf name, by default `task`.
    target_denominators : Sequence[str] | None, optional
        Per-target ratio-denominator variable names, by default None. Mutually
        exclusive with `norm_params` / `scaler`.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": ..., "std": ...}`` per-target normalisation, by default None.
    scaler : Mapping[str, Mapping[str, Any]] | None, optional
        Per-target functional scaling config, by default None.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        targets: Sequence[str],
        name: str | None = None,
        target_denominators: Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        scaler: Mapping[str, Mapping[str, Any]] | None = None,
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
            ),
        )
