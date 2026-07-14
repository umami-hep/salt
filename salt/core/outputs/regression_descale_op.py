"""`RegressionDescaleOp` — de-scale regression predictions back to physical values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode, TensorSpec, sym_dim
from salt.core.outputs.conversion_ops import ConversionOp
from salt.core.utils.array_utils import listify
from salt.core.utils.scalers import RegressionTargetScaler


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
        """Normalise + validate the ``norm_params`` mapping; raises `ConfigError`
        if present but lacking ``mean``/``std``.
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

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the ratio-denominator sources (+ pad mask for a sequence head):
        FIT/VAL/TEST from ``labels.<stream>.<denom>``, ONNX from the raw
        ``inputs.<stream>`` Feature tensor (by name); norm_params/scaler need none.
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
        """Capture the input Feature column order for the ONNX by-name gather;
        raises `ConfigError` if a ratio denominator is not a declared
        ``inputs.<stream>`` column.
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
