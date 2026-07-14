"""Shared task-module base class and config helpers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode, TensorSpec
from salt.core.nn.base import SaltModelModule
from salt.core.nn.dense import Dense, _reject_width_keys
from salt.core.onnx.config import ExportOutput
from salt.core.outputs.output_field import OutputField

_WIDTH_KEYS = ("input_size", "output_size", "context_size")


# Streams that are a fixed-count query bank (e.g. MaskFormer's `objects`, M
# learnable queries with no padding) rather than a variable-length pad-masked
# sequence. A sequence-mode task on such a stream must NOT require/consume a
# pad mask, or graph validation fails with a ConnectivityError (no module
# produces `masks.objects`).
_NO_PAD_MASK_STREAMS = frozenset({"objects"})


class _TaskModuleBase(SaltModelModule):
    """Shared config capture + loss construction for the task modules.

    The head itself (``net`` `Dense` layer + ``loss`` module) is built by the
    subclass ``bind()`` once the input/context widths are known.
    """

    def __init__(
        self,
        stream: str,
        label: str,
        input: str | None,  # noqa: A002 - matches the YAML config key name
        context: str | None,
        dense: dict[str, Any] | None,
        loss: str | dict[str, Any] | None,
        weight: float,
        default_loss: dict[str, Any],
        expose: Sequence[str] | None = None,
        write_targets: bool = True,
    ) -> None:
        super().__init__()
        _reject_width_keys(type(self).__name__, dense, _WIDTH_KEYS)
        self.stream = stream
        self.write_targets = bool(write_targets)
        # `input_name` is the head-math name for the task's stream tag (the
        # single source of both the label-dict key and the pad-mask selection)
        self.input_name = stream
        self.label = label
        self.input_key = input if input is not None else f"encoded.{stream}"
        self.context = context
        self.dense_cfg = dict(dense or {})
        self.loss_cfg = _loss_cfg(loss, default_loss)
        self.weight = float(weight)
        self.expose_modes = _parse_expose(expose, type(self).__name__)
        # the head layers — built by the subclass bind() (widths known there)
        self.net: Dense | None = None
        self.loss: nn.Module | None = None

    def input_name_mask(self, pad_masks: Mapping) -> Tensor:
        """Boolean mask selecting tokens from ``self.input_name``.

        Returns
        -------
        Tensor
            Boolean mask of shape ``[L]``, True for positions in ``self.input_name``.
        """
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (1 if (t == self.input_name) else 0)
                for t, m in pad_masks.items()
            ],
        ).bool()

    def _pred_spec(self, spec: TensorSpec) -> TensorSpec:
        """Re-stamp `spec` to the configured ``expose`` modes (kind/shape/dtype kept)."""
        if self.expose_modes == Mode.ALL:
            return spec
        return replace(spec, modes=spec.modes & self.expose_modes)

    @property
    def pred_key(self) -> str:
        """``preds.<stream>.<instance-name>``."""
        return f"preds.{self.stream}.{self.name}"

    @property
    def loss_key(self) -> str:
        """``losses.<instance-name>``, auto-collected by `LossSum`."""
        return f"losses.{self.name}"

    @property
    def label_key(self) -> str:
        """``labels.<stream>.<label>``."""
        return f"labels.{self.stream}.{self.label}"

    @property
    def has_pad_mask(self) -> bool:
        """True for a variable-length sequence stream; False for a non-sequence head
        or a fixed-count query bank (e.g. MaskFormer ``objects``).
        """
        return self.sequence and self.stream not in _NO_PAD_MASK_STREAMS

    def _emit_targets(self, mode: Mode) -> bool:
        """Whether `mode` gets target-label fields: TEST only, ``write_targets`` on
        (never ONNX — export/inference stays label-free).
        """  # noqa: DOC201 - private one-line predicate
        return self.write_targets and bool(mode & Mode.TEST)

    # -- output rendering: TEST columns + values, ONNX manifest -----------------
    #
    # Per-family rendering lives on the task; `TaskWriter` only orchestrates
    # (groups, prefixes-by-stream, pads, writes) and never branches on task
    # family. A family that ships no rendering inherits the base methods below,
    # which raise a ConfigError naming the missing rendering.

    onnx_renameable: bool = False
    """Whether `TaskWriter` ``onnx_names`` may override this task's ONNX suffix.

    True for classification (needed when class names collide across two
    classification tasks). False for vertexing (fixed `VERTEX_INDEX` suffix)
    and regression (suffixes are its own ``custom_output_names``).
    """

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """The TEST column schema for this task — ``(column_name, np_dtype_str)``.

        Parameters
        ----------
        run_name : str
            The run ``name:`` — the TEST column prefix.

        Raises
        ------
        ConfigError
            For a task family that ships no TEST rendering.
        """
        del run_name
        raise ConfigError(self._no_render_msg("TEST columns"))

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Render this task's formatted TEST values as a structured array.

        Reads the task's published ``preds.*`` leaf from `b` and renders it to
        a structured array whose dtype is ``np.dtype(self.output_names(run_name))``.

        Parameters
        ----------
        b : Bundle
            The executed TEST bundle (carries the converted ``preds.*`` leaf).
        run_name : str
            The run ``name:`` — the TEST column prefix (matches `output_names`).

        Raises
        ------
        ConfigError
            For a task family that ships no TEST rendering.
        """
        del b, run_name
        raise ConfigError(self._no_render_msg("TEST values"))

    def onnx_outputs(self) -> list[ExportOutput]:
        """Render this task's ONNX-manifest entries.

        One `ExportOutput` per family entry, referencing a shipped reduce by
        key: global classification -> ``split_scalars`` per-class suffixes;
        sequence classification -> one ``argmax`` int8 entry; vertexing ->
        ``vertex_union_find`` int8 on the shared `VERTEX_INDEX` constant;
        regression -> ``split_scalars`` per-target suffixes. `TaskWriter`
        decides WHICH tasks export; the task only renders its own entry.

        Raises
        ------
        ConfigError
            For a task family that ships no ONNX rendering.
        """
        raise ConfigError(self._no_render_msg("ONNX output"))

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Render this task's converted, graph-visible output fields.

        Reads the task's RAW ``preds.*`` leaf (loss-space) plus any output-time
        deps (`output_time_requires`) from `b`, applies the eval conversion
        (softmax / masked-softmax / argmax / descale / union-find) in
        TRACEABLE torch ops so ONNX sees them in-graph, and returns one
        `OutputField` per serialisation leaf. Each field carries the converted
        torch ``value`` plus the bare suffix + dtype + axis metadata — it does
        NOT pack a structured numpy array, prefix the run/model name, or
        downcast precision (the sink does all three).

        Target-label emission (plan 50 Phase C): in TEST mode, and unless the
        task's ``write_targets`` flag is off, the prediction fields are
        followed by the task's TARGET-LABEL field(s) — the labels the model
        targeted, as columns named ``target_{task}`` (classification /
        vertexing) or ``target_{task}_{target}`` (regression, one per target).
        Label columns are model-independent, so they are NEVER run-name
        prefixed (``prefix=False``). Per family: classification emits the
        class label as consumed (post ``label_map`` remap; padded / invalid
        ``-2`` positions read ``-1``, matching the loss); regression emits the
        UNSCALED physical target(s) (padded positions NaN, matching the
        de-scaled prediction columns); vertexing emits its per-token
        vertex-index label (padded positions ``-1``). In ONNX/export mode NO
        label field is ever emitted, declared, or demanded — the export graph
        must stay label-free.

        Parameters
        ----------
        b : Bundle
            The executed bundle (carries the RAW ``preds.*`` leaf + any
            output-time dep, e.g. ``masks.<stream>``).
        mode : Mode
            The execution mode — selects the H5 (probs) vs ONNX
            (split-scalars / argmax index) representation.
        run_name : str
            Accepted for symmetry with `get_h5`/`output_names` but NOT baked
            into the field names (the sink prefixes).

        Raises
        ------
        ConfigError
            For a task family that ships no output rendering.
        """
        del b, mode, run_name
        raise ConfigError(self._no_render_msg("output"))

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free serialisation-leaf metadata for `mode`.

        The bundle-free twin of `get_output`: the SAME `OutputField` list
        (names / dtypes / axis / final / prefix, same order) but with every
        ``value`` left ``None``, so sinks can resolve column names/dtypes/order
        before any batch runs. Includes the TEST-mode target-label fields
        (see `get_output`); the ONNX manifest is always label-free.

        Parameters
        ----------
        mode : Mode
            The execution mode — selects the H5 vs ONNX representation,
            exactly as `get_output`.
        run_name : str
            Accepted for symmetry; NOT baked into the names (the sink prefixes).

        Raises
        ------
        ConfigError
            For a task family that ships no output rendering.
        """
        del mode, run_name
        raise ConfigError(self._no_render_msg("output manifest"))

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred bundle keys `get_output` needs at output time.

        E.g. a per-token (sequence) head needs the stream pad mask
        (``masks.<stream>``) for the masked softmax; a global (pooled) head
        needs nothing extra. In TEST mode a ``write_targets`` task ALSO
        demands exactly the ``labels.*`` keys its target-label fields read;
        in ONNX mode no label key is ever demanded (label-free export /
        inference is a graph contract). The base returns ``[]``; per-family
        overrides declare their own.

        Parameters
        ----------
        mode : Mode
            The execution mode (for families whose deps are mode-split).

        Returns
        -------
        list[str]
            Dotted bundle keys (empty by default).
        """
        del mode
        return []

    def _no_render_msg(self, what: str) -> str:
        """The unsupported-family error message, naming the instance, type, and
        missing rendering.
        """
        return (
            f"task {self.name!r} ({type(self).__name__}) ships no {what} rendering — "
            "supported families are ClassificationTaskModule, VertexingTaskModule and "
            "RegressionTaskModule; give a custom task module output_names/get_h5/onnx_outputs "
            "methods, or write a custom Writer for its outputs (design §8)"
        )


def _parse_expose(expose: Sequence[str] | None, cls: str) -> Mode:
    """Parse a task ``expose:`` mode-name list into a `Mode` flag.

    ``None`` means all modes (``Mode.ALL``); a case-insensitive list of
    ``fit``/``val``/``test``/``onnx`` gates the prediction port to exactly
    those modes (e.g. ``expose: [fit, val]`` prunes a train-only aux task's
    prediction from the TEST/ONNX plans). Raises `ConfigError` on a non-list,
    empty list, or unknown mode name.
    """
    if expose is None:
        return Mode.ALL
    if isinstance(expose, str) or not isinstance(expose, Sequence):
        raise ConfigError(
            f"{cls}: expose must be a list of mode names (e.g. [fit, val]), got {expose!r} "
            "(design §4.2)"
        )
    if not expose:
        raise ConfigError(
            f"{cls}: expose may not be an empty list — a task exposed in no mode is dead; "
            "omit expose for all modes, or remove the task (design §4.2)"
        )
    valid = {m.name.lower(): m for m in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX)}
    modes = Mode.FIT & Mode.TEST  # empty seed (no mode); accumulate the named ones
    for raw in expose:
        if not isinstance(raw, str) or raw.lower() not in valid:
            raise ConfigError(
                f"{cls}: unknown expose mode {raw!r} — valid modes are "
                f"{sorted(valid)} (design §4.2)"
            )
        modes |= valid[raw.lower()]
    return modes


def _checked_weight_source(weight_source: Mapping[str, str] | None) -> dict[str, str] | None:
    """Validate a ``weight_source`` mapping is exactly ``{"from_class_dict": <path>}``, or None."""
    if weight_source is None:
        return None
    if set(weight_source) != {"from_class_dict"} or not isinstance(
        weight_source["from_class_dict"], (str, Path)
    ):
        raise ConfigError(
            f"weight_source must be {{'from_class_dict': <path>}}, got {dict(weight_source)!r} "
            "(design §3.3)"
        )
    return {"from_class_dict": str(weight_source["from_class_dict"])}


def _loss_cfg(loss: str | dict[str, Any] | None, default: dict[str, Any]) -> dict[str, Any]:
    """Normalise a loss config to ``{class_path, init_args}`` form; raises
    `ConfigError` if malformed.
    """
    if loss is None:
        cfg: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in default.items()}
        return cfg
    if isinstance(loss, str):
        return {"class_path": loss if "." in loss else f"torch.nn.{loss}"}
    if isinstance(loss, Mapping) and "class_path" in loss:
        return {
            "class_path": str(loss["class_path"]),
            "init_args": dict(loss.get("init_args", {})),
        }
    raise ConfigError(
        f"loss config must be a torch.nn class name or {{class_path, init_args}}, got {loss!r}"
    )


def _loss_class(cfg: Mapping[str, Any]) -> type[nn.Module]:
    """Resolve a loss ``class_path`` to its `nn.Module` subclass (config-only, no instantiation)."""
    path = cfg["class_path"]
    module_path, _, cls_name = path.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError) as err:
        raise ConfigError(f"cannot resolve loss class_path {path!r}: {err}") from err
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ConfigError(f"loss class_path {path!r} is not an nn.Module subclass")
    return cls
