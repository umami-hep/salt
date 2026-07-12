"""Output producers — `GraphModule`s that populate ``outputs.*`` from ``preds.*``.

A producer reads a task's ``preds.<stream>.<task>`` prediction and writes one or
more ``outputs.<stream>.<name>`` leaves for TEST/ONNX; producers run only when
demanded, so FIT/VAL prune them via the planner's demand closure — no per-task
forward branching needed.
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

# The two scripted union-find helpers + the MaskFormer export math are inlined
# verbatim in salt.core.onnx.reduces (the legacy reduce path); the conversion
# nodes below reuse those exact copies so the folded node and the legacy
# reduce can never drift.
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
"""Placeholder instance name — the config assembly step assigns the dict key."""


@dataclass(frozen=True)
class OutputField:
    """One output column a producer contributes to a sink's field manifest.

    ``h5_name``/``onnx_name`` can diverge (e.g. a per-token classification head
    writes per-class H5 probability columns but an ONNX argmax index) — either
    may be ``None`` if the field has no representation in that sink.

    Parameters
    ----------
    h5_name : str | None
        H5 column suffix (run-name-prefixed by the sink unless ``prefix=False``).
        ``None`` if the field has no H5 representation.
    onnx_name : str | None
        Flat ONNX output suffix; defaults to `h5_name`. ``None`` if the field
        has no ONNX representation.
    dtype : str
        Serialisation dtype (numpy descriptor for H5, mapped to the ONNX
        equivalent via `onnx_dtype`).
    axis : str
        ``"global"`` (jet-level scalar) or ``"per_token"`` (sequence column).
    final : bool
        ``False`` for an intermediate leaf consumed only by a downstream node
        (sinks must not auto-collect it); ``True`` (default) for a
        written/exported leaf.
    prefix : bool
        Whether the H5 column is ``{run_name}_{h5_name}`` or the bare
        `h5_name` (e.g. the v1 ``VertexIndex`` byte-parity column).
    value : Tensor | None
        The graph-visible converted tensor. ``None`` for the static manifest
        path (name/dtype/axis minted before any forward); filled by the
        task-side ``get_output`` path with the converted torch tensor.
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
        """The ONNX suffix, defaulting to `h5_name` when not explicitly set."""
        return self.onnx_name if self.onnx_name is not None else self.h5_name

    @property
    def onnx_dtype(self) -> str:
        """The ONNX dtype namespace mapping of `dtype` (``f4 -> float32``, ``i1/i8 -> int8``)."""
        return "int8" if self.dtype in {"i1", "i8", "int8"} else "float32"


def _resolve_task(model_modules: Mapping[str, Any], task_name: str, who: str) -> Any:
    """Look up the wrapped task module by name in the model module dict.

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
    """Add singleton dims after the batch dim to reach ``ndim``.

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
    """Softmax that ignores padded elements (mask=True set to -inf before, 0 after)."""
    if mask is not None:
        mask = _add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)
    x = torch.softmax(x, dim=dim)
    if mask is not None:
        x = x.masked_fill(mask, 0)
    return x


# Conversion ops: small strategy objects (not nn.Module) holding config and
# reproducing one task family's eval math. Three hooks parameterise the
# generic TaskOutput: extra_requires (extra input ports), derived_width
# (output width given input width), convert (the forward conversion).


class ConversionOp:
    """Eval-math strategy behind `TaskOutput`; base is an identity passthrough."""

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Extra input ports this op reads beyond the source ``preds.*`` leaf (none by default)."""
        del stream
        return {}

    def derived_width(self, in_width: int) -> int:
        """The output leaf's last-dim width given the input width (unchanged by default)."""
        return in_width

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """Field manifest for this op's output leaf, resolved from the wrapped task.

        The base (identity) mirrors the wrapped task's own ``output_names``:
        one field per column, bare suffix (run-name prefix stripped — the sink
        re-prefixes), dtype from the task descriptor, axis from the task's
        sequence flag.

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
        """Identity copy of the prediction leaf (cloned so the output never aliases it)."""
        del mode, stream
        return b.get(pred_key).clone()


class IdentityOp(ConversionOp):
    """Explicit alias for the identity conversion op (the `ConversionOp` base)."""


class ClassProbsOp(ConversionOp):
    """Global classification probabilities: sigmoid (BCE) or softmax over classes.

    Parameters
    ----------
    bce : bool, optional
        Use per-class sigmoid (a ``BCEWithLogitsLoss`` head) instead of
        softmax (``CrossEntropyLoss``), by default False.
    """

    def __init__(self, bce: bool = False) -> None:
        self.bce = bool(bce)

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float global field per class (``class_suffixes``); same suffixes for H5 and ONNX."""
        del run_name
        return [
            OutputField(h5_name=px, dtype="f4", axis="global", final=True)
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Sigmoid (BCE) or softmax over the class dim."""
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

    The class dim collapses, so the output last dim is 1. The pad mask is
    read from ``masks.<stream>`` — padded tokens are ``-inf`` before the
    softmax (argmax is invariant under it).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True.
        False for a fixed-count query bank (e.g. the MaskFormer ``objects``
        stream, which has no ``masks.objects``).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax (when `has_pad_mask`)."""
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def derived_width(self, in_width: int) -> int:
        """The argmax collapses the class dim to a single index column (always 1)."""
        del in_width
        return 1

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One int8 per-token ONNX-only field named ``pascal_case(task)`` (e.g. ``TrackOrigin``).

        No H5 column (``h5_name=None``): the TEST H5 carries the full prob
        vector via the sibling `SeqClassProbs` producer instead.
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

        TEST returns ``[B, L]`` int64 indices. ONNX returns the int8 ``[L]``
        leaf using the zero-row append/strip trick: a zero row is appended
        along the token axis so the traced argmax stays valid for zero-token
        jets, then stripped and cast to int8 for the Athena output.
        """
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # zero-row append/strip keeps the traced argmax valid for zero-token
            # jets; strip it back off before casting to the int8 [L] output
            probs = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
            out = torch.argmax(probs, dim=-1)[:, :-1]
            return out.squeeze(0).char()
        return torch.argmax(probs, dim=-1)


class SeqClassProbsOp(ConversionOp):
    """Per-token class probabilities (masked-softmax; the sequence eval-H5 columns).

    The eval-H5 counterpart of `SeqClassIndexOp` (which collapses to the ONNX
    argmax index): the TEST H5 keeps the full prob columns, the ONNX export
    keeps the argmax. Padded positions read ``0.0`` (the masked softmax zeroes
    them).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True
        (False for a fixed-count query bank).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax (when `has_pad_mask`)."""
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float per-token H5-only field per class (``class_suffixes``); no ONNX leaf."""
        del run_name
        return [
            OutputField(
                h5_name=px, onnx_name=None, dtype="f4", axis="per_token", final=True
            )
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax over the class dim; ``[B, L, C]`` per-token probabilities."""
        del mode
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        return _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)


class RegressionDescaleOp(ConversionOp):
    """De-scale regression predictions back to physical values.

    Exactly one of three mutually exclusive scaling methods applies:

    - **ratio denominator**: ``pred[..., i] *= denom_i``. The denominator
      source is mode-split: TEST reads it from ``labels.<stream>.<denom>``;
      ONNX gathers it by name from the raw input Feature tensor
      ``inputs.<stream>``.
    - **norm_params** (mean/std): ``pred[..., i] = pred[..., i] * std_i + mean_i``.
    - **scaler** (functional `RegressionTargetScaler`): ``pred[..., i] =
      scaler.inverse(target_i, pred[..., i])``.

    Width-preserving (R targets in, R de-scaled values out).

    Parameters
    ----------
    stream : str
        The regressed stream (denominator labels live under
        ``labels.<stream>.<denom>``; the input Feature is ``inputs.<stream>``).
    targets : str | Sequence[str]
        The regression target name(s), in column order (a bare string for a
        single target).
    target_denominators : str | Sequence[str] | None, optional
        Per-target ratio-denominator variable name(s), by default None.
        Mutually exclusive with `norm_params` / `scaler`.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": <scalar|list>, "std": <scalar|list>}`` per-target
        mean/std, by default None. Mutually exclusive.
    scaler : Mapping[str, Mapping[str, Any]] | None, optional
        Per-target functional scaling config, by default None. Mutually
        exclusive.
    gaussian : bool, optional
        Whether the source head publishes ``[..., 2R]`` (means ‖ raw
        variances), de-scaled to means ‖ ``stddev``, by default False. No
        functional-scaler branch is supported for gaussian heads.
    sequence : bool, optional
        Whether the source head is per-token; a per-token head NaN-fills
        padded positions after de-scaling, by default False.

    Raises
    ------
    ConfigError
        On more than one scaling method, or a denominator/target count
        mismatch.
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
        # accept a scalar string OR a sequence (the config YAML surface uses a
        # bare string for a single target)
        self.targets = tuple(listify(targets))
        if not self.targets:
            raise ConfigError("RegressionDescaleOp: targets is required and non-empty")
        self.sequence = bool(sequence)
        self.target_denominators = (
            tuple(listify(target_denominators)) if target_denominators is not None else None
        )
        self.norm_params = self._checked_norm_params(norm_params)
        self.scaler = RegressionTargetScaler(dict(scaler)) if scaler is not None else None
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
        # de-scaling can gather denominators by name
        self._input_fields: tuple[str, ...] = ()

    @staticmethod
    def _checked_norm_params(
        norm_params: Mapping[str, Any] | None,
    ) -> dict[str, list[float]] | None:
        """Normalise + validate the ``norm_params`` mapping.

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
        """The raw-input key carrying the ONNX denominator columns (``inputs.<stream>``)."""
        return f"inputs.{self.stream}"

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float field per regression output (``output_suffixes``); same suffixes for H5/ONNX."""
        del run_name
        axis = "per_token" if bool(getattr(task, "sequence", False)) else "global"
        return [
            OutputField(h5_name=suffix, dtype="f4", axis=axis, final=True)
            for suffix in task.output_suffixes
        ]

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the ratio-denominator sources (+ pad mask for a sequence head).

        FIT|VAL|TEST read each denominator from ``labels.<stream>.<denom>``;
        ONNX gathers them by name from the raw ``inputs.<stream>`` Feature
        tensor. norm_params / scaler need no external source.
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

        Every ratio denominator must be a declared column of
        ``inputs.<stream>`` so the export graph can gather it by name.

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
        """Invert the configured scaling to the physical value."""
        # clone before the in-place de-scale: `.float()` on an already-float32
        # leaf returns the SAME tensor, so a bare `.float()` would mutate the
        # bundle's preds.* leaf in place — a producer must not (write-once)
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
        """NaN-fill padded positions for a sequence head (no-op for a global head)."""
        if not self.sequence:
            return preds
        mask = b.get(f"masks.{stream}")
        return torch.masked_fill(preds, mask.unsqueeze(-1), torch.nan)

    def _convert_gaussian(self, preds: Tensor, b: Bundle, mode: Mode, stream: str) -> Tensor:
        """De-scale a gaussian head's ``[..., 2R]`` means ‖ raw-variances.

        Means in columns ``[0:R]`` de-scale like a plain regression head
        (ratio-denom OR mean/std); variances in ``[R:2R]`` become ``stddev =
        sqrt(softplus(var)) * std`` (norm_params) or are scaled by the
        denominator (ratio). Published as ``[..., 2R]`` means ‖ stddevs (the
        sink splits on ``_stddev``).
        """
        n = len(self.targets)
        if self.target_denominators is not None:
            denoms = self._descale_source(b, mode, stream)
            for i, denom in enumerate(self.target_denominators):
                # mean (i) and var (i+1) both scale by the same denominator
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
        # NaN-fills means + stds at padded positions equivalently over [..., 2R]
        return self._nan_fill(preds, b, stream)

    def _descale_source(self, b: Bundle, mode: Mode, stream: str) -> dict[str, Tensor]:
        """Gather the per-denominator de-scaling source (mode-split: labels in TEST, inputs in ONNX)."""
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
    parameterised producer applies a `ConversionOp` (the eval math) to a
    task's published prediction and writes the result. The default op is an
    identity copy; the thin `ClassProbs` / `SeqClassIndex` / `Regression`
    subclasses pre-select a P1 op.

    Both ports declare ``shape=None`` (rank-agnostic — a global head is
    ``[B, C]``, a sequence head ``[B, T, C]``); the output last-dim width is
    resolved via `derived_widths`, which maps the bound input width through
    the op's `derived_width` (so it resolves from a TEST-only bind alone).

    Parameters
    ----------
    task : str
        The source task's instance name — the producer reads
        ``preds.<stream>.<task>``.
    stream : str
        The stream the source task publishes under; also the stream the
        output is written under.
    name : str, optional
        The output leaf name; defaults to `task`.
    op : ConversionOp | None, optional
        The eval-math conversion op, by default the identity copy.
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

        Both ports are active in every mode (``modes=ALL``): the producer is
        gated by demand, not a hard mode flag, so FIT/VAL drop it via the
        planner's demand closure. ``kind="data"`` (the default) so the
        producer<-task edge kind-unifies.
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

        Only `RegressionDescaleOp` needs a bind; other ops have no ``bind``
        and this no-ops.
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
        """Map the bound input prediction width through the op onto the output.

        Returns ``{}`` until the input width is resolved (keeps the hook
        order-insensitive across plans).
        """
        pred_width = widths.get(self.pred_key)
        if pred_width is None:
            return {}
        return {self.output_key: self.op.derived_width(pred_width)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply the op to the prediction leaf, writing the ``outputs.*`` leaf."""
        converted = self.op.convert(b, mode, pred_key=self.pred_key, stream=self.stream)
        return {self.output_key: converted}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """The field manifest this producer's ``outputs.*`` leaf expands into.

        Resolves the wrapped task (`task`) from `model_modules` and delegates
        to the op's `output_columns`.
        """
        task = _resolve_task(model_modules, self.task, f"producer {self.name!r}")
        return self.op.output_columns(task, run_name)


class ClassProbs(TaskOutput):
    """Global-classification probability producer (a `TaskOutput` pre-wired with `ClassProbsOp`).

    Reads a pooled classification head's logits and writes the per-class
    probabilities (``softmax`` / ``sigmoid``). Width-preserving.

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The stream the head publishes under (also the output stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    bce : bool, optional
        Whether the head uses ``BCEWithLogitsLoss`` (sigmoid) instead of
        ``CrossEntropyLoss`` (softmax), by default False.
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
    """Per-token class-index producer (a `TaskOutput` pre-wired with `SeqClassIndexOp`).

    Reads a sequence classification head's logits, applies masked softmax +
    ``argmax``, and writes the integer per-token class index (e.g.
    ``TrackOrigin``). The class dim collapses, so the output width is 1.

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
        (False for a fixed-count query bank).
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
    """Per-token class-probability producer (a `TaskOutput` pre-wired with `SeqClassProbsOp`).

    Reads a sequence classification head's logits, applies masked softmax,
    and writes the ``[B, L, C]`` per-token per-class probabilities — the
    eval-H5 columns (`SeqClassIndex` is the ONNX argmax counterpart).

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
    """Regression de-scaling producer (a `TaskOutput` pre-wired with `RegressionDescaleOp`).

    Reads a regression head's training-space values and writes the de-scaled
    physical values. Width-preserving. For the ratio-denominator case the op
    consumes the denominator source (``labels.<stream>.<denom>`` in TEST,
    ``inputs.<stream>`` by name in ONNX).

    Parameters
    ----------
    task : str
        The source regression task's instance name.
    stream : str
        The regressed stream (also the output stream and the
        denominator-source stream).
    targets : str | Sequence[str]
        The regression target name(s), in column order.
    name : str | None, optional
        The output leaf name, by default `task`.
    target_denominators : str | Sequence[str] | None, optional
        Per-target ratio-denominator variable name(s), by default None.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": ..., "std": ...}`` per-target normalisation, by default None.
    scaler : Mapping[str, Mapping[str, Any]] | None, optional
        Per-target functional scaling config, by default None.
    gaussian : bool, optional
        Whether the source head is a gaussian (``mu``/``sigma``) head
        publishing ``[..., 2R]``, by default False.
    sequence : bool, optional
        Whether the source head is per-token, by default False.
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
    """Linear-combination producer: a new ``outputs.*`` leaf from a source bundle leaf.

    Reads a source ``outputs.<stream>.<src>`` leaf a producer already minted
    (e.g. softmaxed class probs) and produces a new
    ``outputs.<stream>.<name>`` scalar leaf as a weighted sum over its
    last-dim channels: ``out = sum(scale * source[..., index])`` over
    `terms`. Because it reads a bundle leaf (not a renamed export name), both
    sinks consume/name the new leaf like any other ``outputs.*`` leaf.

    The output last dim collapses to a scalar (a GLOBAL float, no per-token
    axis).

    Parameters
    ----------
    source : str
        The source bundle leaf, an ``outputs.<stream>.<src>`` key. The new
        leaf is written under the SAME stream as the source.
    name : str
        The new output leaf's last component (``outputs.<stream>.<name>``).
    terms : Mapping[int, float]
        Source last-dim channel index -> scale, in combination order (e.g.
        ``{0: 1.0, 1: 1.0}`` for ``probs[..., 0] + probs[..., 1]``). At least
        one term; every index must be a non-negative int.

    Raises
    ------
    ConfigError
        For a non-``outputs`` source, a wildcard source, an empty `terms`, or
        a negative/non-int channel index.
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
        # sum order is load-bearing for the v1 float bitwise-equality
        self.terms: tuple[tuple[int, float], ...] = tuple(
            (self._checked_index(index, name), float(scale)) for index, scale in terms.items()
        )
        self.output_key = f"outputs.{self.stream}.{name}"

    @staticmethod
    def _checked_index(index: Any, name: str) -> int:
        """Validate a source channel index is a non-negative int.

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

        Both ports are active in every mode (``modes=ALL``): gated by demand,
        not a hard mode flag, like the other conversion producers.
        """
        del mode
        requires = {self.source: TensorSpec(shape=None, dtype="float32", kind="data")}
        produces = {self.output_key: TensorSpec(shape=None, dtype="float32", kind="data")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The combination collapses the source last dim to a single scalar column (width 1)."""
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute ``sum(scale * source[..., index])`` over `terms`, in declared order."""
        del mode
        source = b.get(self.source)
        out = sum(scale * source[..., index] for index, scale in self.terms)
        return {self.output_key: out}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One float global field named after the combination; present in both H5 and ONNX."""
        del run_name, model_modules
        return [OutputField(h5_name=self.output_name, dtype="f4", axis="global", final=True)]


class VertexUnionFind(nn.Module):
    """In-graph union-find conversion node (folds the legacy union-find export reduce).

    Reads the RAW ``preds.<stream>.<task>`` ``[E, 1]`` edge scores the
    vertexing task publishes in ONNX mode plus the stream's
    ``masks.<stream>`` pad mask, and runs the ``@torch.jit.script``
    union-find inside ``forward(b, Mode.ONNX)``:

        ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
        ``.reshape(-1).char()``

    producing the same int8 ``[L]`` per-token leaf the legacy chain emits.
    The two scripted helpers are reused verbatim from ``reduces.py`` (one
    source of truth, no drift).

    Because it declares the SAME ``preds.*`` port the legacy reduce reads,
    the demand-closure pulls the identical vertexing task node into the ONNX
    plan. The ``reshape(-1)`` collapse has no recoverable last dim in a bind,
    so `derived_widths` re-emits width 1.

    This node is ONNX-only by construction: the union-find chain is shaped
    for the traced export batch; it is never wired into a TEST H5 config
    (the eval vertex columns come from a separate per-token path).

    Parameters
    ----------
    task : str
        The source vertexing task's instance name — the node reads the RAW
        ``preds.<stream>.<task>`` ``[E, 1]`` edge scores.
    stream : str
        The constituent stream the head publishes under (also the stream the
        output is written under).
    name : str, optional
        The output leaf name; defaults to `task`.
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

        The output width is re-emitted via `derived_widths` (the
        ``reshape(-1)`` collapse has no recoverable last dim).
        """
        del mode
        # ONNX-only ports: this node is ONNX-only by construction (class
        # docstring), so gating to Mode.ONNX (rather than demand-gating like
        # the shared softmax producers) keeps it inactive in FIT/VAL/TEST —
        # a config that opts the vertexing head out of TEST does not trip
        # the planner's pre-prune connectivity check on this node's require.
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
        """The union-find ``reshape(-1)`` collapses to a single per-token index column (width 1)."""
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the ``@torch.jit.script`` union-find chain (verbatim, see class docstring)."""
        del mode
        edge_scores = b.get(self.pred_key)  # RAW [E, 1] scores
        pad_mask = b.get(self.mask_key)  # the all-valid pad mask
        vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
        vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
        return {self.output_key: vertex_list.reshape(-1).char()}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One int8 per-token ONNX-only field on the shared `VERTEX_INDEX` suffix.

        No auto-collected H5 column: the TEST H5 vertex column comes from the
        vertexing task's own ``get_h5`` instead.
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
    """MaskFormer object reconstruction node (the "writer" half of the two-node MaskFormer split).

    Runs `get_maskformer_outputs` once inside ``forward(b, Mode.ONNX)`` (the
    null-suppression + pT reorder + index math) and exposes its products so a
    downstream `MFLeadVertexDecorator` can read the reordered per-vertex
    outputs without redoing any of the heavy lifting. Produces:

    - **object_index** (``outputs.<constituent_stream>.<index_name>``, int8
      per-token): the per-constituent owning-object index.
    - **leading_object** (``outputs.<stream>.<leading_name>``, float32
      global): the leading object's R de-scaled regression scalars.
    - **vertices_class_probs** / **vertices_regression**
      (``outputs.<stream>.<...>``, float32 ``[B, M, C]`` / ``[B, M, R]``):
      the reordered (null-suppressed + pT-ordered) per-vertex tensors,
      exposed as intermediate leaves for `MFLeadVertexDecorator`.

    `get_maskformer_outputs` is reused verbatim from ``reduces.py`` (one
    source of truth). The node declares all three cross-node reads it needs
    (``objects.class_probs`` / ``objects.masks`` /
    ``preds.<stream>.<reg_task>``) so the demand-closure keeps the upstream
    decoder + regression task alive in the ONNX plan; all three tensors are
    cloned before `get_maskformer_outputs` (which mutates ``masks``/
    ``regression`` in place for null-suppression + pT reorder), so the
    write-once bundle is never mutated.

    ONNX-only by construction (like `VertexUnionFind`): never wired into a
    TEST H5 config.

    Parameters
    ----------
    regression_task : str
        The object-regression task's instance name — the node reads the
        de-scaled ``preds.<stream>.<regression_task>`` ``[B, M, R]``
        predictions.

        The folded ``object_index`` / ``leading_object`` leaves are
        bitwise-equal to the legacy reduces ONLY when
        ``regression_task == "regression"`` (the legacy default the export
        path asserts). A non-default `regression_task` reorders by a
        different regression tensor than the legacy reduce and has no parity
        oracle; keep the default for any config that must match the legacy
        ONNX path.
    stream : str, optional
        The object stream the decoder publishes under, by default
        ``"objects"``. Also the object output stream.
    leading_name : str, optional
        The leading-object regression leaf name, by default
        ``"leading_object"``.
    index_name : str, optional
        The per-constituent object-index leaf name, by default
        ``"object_index"``.
    n_reg : int
        The object-regression target count R (the leading-regression output
        width). The leading leaf is sliced to ``leading_reg[:, :n_reg]`` to
        reproduce the legacy reduce exactly, including the no-objects/
        empty-track dummy path. Must match the configured leading-object
        ``names`` count.
    constituent_stream : str, optional
        The constituent stream the per-token index leaf is written under and
        whose dynamic axis the index carries, by default ``"tracks"``.
    vertices_class_probs_name : str, optional
        The exposed reordered per-vertex class-probs leaf name, by default
        ``"vertices_class_probs"``.
    vertices_regression_name : str, optional
        The exposed reordered per-vertex regression leaf name, by default
        ``"vertices_regression"``.
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
        """Declare all three maskformer reads -> the leading-regression + object-index leaves.

        The index width has no recoverable last dim in a bind (width 1); the
        leading width follows the regression port (or `n_reg`) — both
        re-emitted via `derived_widths`.
        """
        del mode
        # ONNX-only ports (same gate as `VertexUnionFind`): the null-suppression
        # + pT-reorder chain is shaped for the traced export batch, so this node
        # is inactive in FIT/VAL/TEST — a config that wires it for ONNX export
        # alongside an object-regression head opted out of TEST eval does not
        # trip the planner's pre-prune connectivity check on this require.
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
        """Width-resolve the leaves: index collapses to 1, leading + vertex regression follow n_reg."""
        del widths
        return {
            self.index_key: 1,
            self.leading_key: self.n_reg,
            self.vertices_regression_key: self.n_reg,
        }

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run ``get_maskformer_outputs`` once -> object_index + leading + the per-vertex leaves."""
        del mode
        objects = {
            "class_probs": b.get(self.class_probs_key).clone(),
            "masks": b.get(self.masks_key).clone(),
            "regression": b.get(self.reg_key).clone(),
        }
        leading_reg, indices, vertices_class_probs, vertices_regression = get_maskformer_outputs(
            objects, apply_reorder=True
        )
        # get_maskformer_outputs returns indices=None at L == 0 (n_tracks == 0).
        # This never happens on the export path (torch.onnx.export always traces
        # at a fixed L > 0), so this guard folds to a constant-False branch that
        # emits no ops; it only removes a latent AttributeError for a future
        # eager TEST-mode wiring of this node.
        if indices is None:
            empty_index = torch.zeros(0, dtype=torch.int8)
        else:
            empty_index = indices.reshape(-1).char()
        # leading_reg is [B, R] normally but [1, n_obj] in the no-objects/
        # empty-track dummy path; slice to [:, :n_reg] to reproduce both paths
        # exactly (matches the legacy reduce's leading_reg[0, i] for i in range(R)).
        return {
            self.leading_key: leading_reg[:, : self.n_reg],
            self.index_key: empty_index,
            # the reordered (null-suppressed + pT-ordered) per-vertex outputs — passed
            # through verbatim as get_maskformer_outputs returns them (the decorator
            # does the lead-vertex selection; all heavy lifting is here, cloned above)
            self.vertices_class_probs_key: vertices_class_probs,
            self.vertices_regression_key: vertices_regression,
        }

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """The MaskFormer object leaves' field manifest — index + leading + intermediates.

        The per-vertex ``vertices_class_probs`` / ``vertices_regression``
        are exposed intermediates (``final=False``) that sinks must not
        auto-collect.
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


# One-window alias: `MaskFormerObject` was renamed `MaskFormerObjects`. The
# promoted node is a strict superset, so an existing `MaskFormerObject`
# config keeps working via this alias. Remove after the migration window.
MaskFormerObject = MaskFormerObjects
"""Deprecated alias for `MaskFormerObjects`."""


class MFLeadVertexDecorator(nn.Module):
    """MaskFormer lead-vertex jet-level decorator (the decoration half of the two-node split).

    A thin selector that reads `MaskFormerObjects`'s exposed reordered
    per-vertex leaves (``vertices_class_probs [B, M, C]`` +
    ``vertices_regression [B, M, R]``) and emits jet-level global scalar
    leaves (e.g. ``jet.lead_vertex_pt``). Not a parity fold of any legacy
    reduce — a new capability with no legacy oracle.

    The lead vertex is selected per-jet as the highest-pT vertex (by
    ``vertices_regression[..., pt_index]``) among the vertices that are ALL of:

    - not null: ``vertices_class_probs[..., null_index] < pnull_threshold``;
    - not the primary vertex: ``argmax(vertices_class_probs) != pv_class_index``;
    - a real vertex class: ``argmax(vertices_class_probs) != null_index``
      (needed because with >=3 classes a vertex can have ``argmax==null`` yet
      ``pnull < threshold``).

    For each configured output ``{name: reg_index}`` it pulls the selected
    vertex's ``vertices_regression[..., reg_index]``.

    NOTE: this selection differs from the legacy ``leading_object`` reduce
    (pT-only, no PV/null exclusion on the decorator side) — it is a NEW
    output, not a relocation; both stay untouched (additive).

    When no vertex qualifies (all-null jet, all-PV jet, or empty inputs), the
    jet-level scalars are filled with NaN deterministically via a
    masked-argmax over an all-``-inf`` pT column, so the trace stays valid
    for every batch shape (no data-dependent control flow).

    Note also: when `MaskFormerObjects`'s ``get_maskformer_outputs`` hits its
    "no object exceeds the null threshold" dummy path, it returns an all-NaN
    ``vertices_regression`` while ``vertices_class_probs`` flows through
    real — so the decorator's qualify mask can pass vertices whose
    regression is undefined, and the jet-level scalars end up NaN. This
    matches v1's "no predicted objects -> dummy NaN" semantics.

    Parameters
    ----------
    source : str
        The `MaskFormerObjects` exposed per-vertex class-probs leaf. The
        regression source defaults to the same object stream's
        ``vertices_regression`` leaf unless `regression_source` overrides it.
    outputs : Mapping[str, int]
        ``{output_name: reg_index}`` — each jet-level scalar leaf pulls the
        lead vertex's ``vertices_regression[..., reg_index]``.
    pt_index : int
        The ``vertices_regression`` channel that is the vertex pT (the
        selection key).
    pv_class_index : int
        The vertex class index that marks the primary vertex (excluded from
        selection).
    pnull_threshold : float, optional
        The null-probability cut, by default 0.5.
    null_index : int | None, optional
        The class index of the null class, by default None = the last class.
    jet_stream : str, optional
        The jet-level output stream, by default ``"jet"``.
    regression_source : str | None, optional
        Override for the per-vertex regression leaf, by default None.

    Raises
    ------
    ConfigError
        For a non-``outputs`` / wildcard source, an empty `outputs` map, a
        negative/non-int reg index or pt_index, or a missing pt selection
        channel.
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

        Both ports are active in every mode (``modes=ALL``, gated by demand).
        The requires are the leaves `MaskFormerObjects` mints — a node->node
        edge whose demand keeps that node alive.
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
        """Every jet-level output is a single scalar column (width 1)."""
        del widths
        return dict.fromkeys(self.output_keys, 1)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Select the lead vertex (highest-pT, non-null, non-PV) and emit jet-level scalars.

        Trace-safe, no data-dependent control flow: build the per-vertex
        qualify mask (not-null AND not-PV AND real-vertex-class), masked-
        argmax the pT column to find the lead index per jet, then gather
        each configured regression channel at that index (NaN where no
        vertex qualifies).
        """
        del mode
        class_probs = b.get(self.source)  # [B, M, C]
        regression = b.get(self.regression_source)  # [B, M, R]
        n_obj = class_probs.shape[1]
        batch = class_probs.shape[0]
        if n_obj == 0:
            # no object queries -> no vertex can qualify; every output is all-NaN.
            # M (the object-query axis) is fixed by the decoder and never a
            # declared dynamic export input, so this frozen branch is harmless.
            nan = torch.full((batch,), torch.nan, dtype=torch.float32)
            return dict.fromkeys(self.output_keys, nan)
        null_idx = self.null_index if self.null_index is not None else class_probs.shape[-1] - 1
        pnull = class_probs[..., null_idx]  # [B, M]
        pred_class = torch.argmax(class_probs, dim=-1)  # [B, M]
        # qualify = not-null AND not-PV AND argmax-is-a-real-vertex-class. The
        # third condition is needed because with >=3 classes a vertex can have
        # argmax==null yet pnull<threshold (thin-spread probs).
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
        """One float global jet-level field per configured lead-vertex scalar (no legacy oracle)."""
        del run_name, model_modules
        return [
            OutputField(h5_name=name, dtype="f4", axis="global", final=True)
            for name, _ in self.outputs_map
        ]
