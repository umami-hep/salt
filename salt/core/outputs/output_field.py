"""`OutputField` — one output column a producer contributes to a sink's field manifest.

Also home to the shared producer-side private helpers (`_resolve_task`,
`_add_dims`, `_masked_softmax`) per the §3.4 split map.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from salt.core.graph.errors import ConfigError


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
