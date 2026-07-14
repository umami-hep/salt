"""The MaskFormer object writer — eval columns AND ONNX object outputs.

TEST: per-object class probabilities, the MaskIndex column, truth/logit masks.
ONNX: the ``leading_object`` and ``object_index`` reduces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import GraphModule, Mode, TensorSpec
from salt.core.onnx.config import ExportOutput
from salt.core.outputs.writer_base import WriteCtx, Writer, WriterDeclareCtx
from salt.core.outputs.names import OBJECT_INDEX
from salt.core.utils.mask_utils import indices_from_mask

__all__ = ["MaskFormerObjectWriter"]


class MaskFormerObjectWriter(Writer):
    """Persist MaskFormer object predictions + truth (TEST), and declare the
    two object ONNX outputs (ONNX) — one writer, both modes.

    TEST demand: the decoder's ``<object_stream>.*`` keys and the
    `MaskFormerTargets` truth labels. ONNX manifest ports: the
    object-regression ``preds.<object_stream>.regression`` and the decoder
    ``<object_stream>.masks``.

    Parameters
    ----------
    object_classes : Sequence[str]
        All object class names, including the trailing ``null`` (e.g.
        ``["b", "c", "null"]``). Columns are ``{run_name}_p{name}`` per class
        (plain ``p`` prefix, no `Flavours` resolution). Length must equal the
        decoder's published ``<object_stream>.class_probs`` column count
        (``num_classes + 1``).
    object_stream : str, optional
        The decoder's object bundle stream, by default ``objects``; the
        ``objects``/``object_masks`` H5 groups derive from it.
    constituent_stream : str, optional
        The constituent reader stream the masks span (``MaskIndex`` lands
        here), by default ``tracks``.
    regression_task : str, optional
        The object-regression task instance name, by default ``regression``.
        For ONNX export it must be ``"regression"`` (see `onnx_outputs`);
        eval-only (``onnx: false``) writers may use any name.
    leading_prefix : str, optional
        The leading-object ONNX suffix template
        (``{leading_prefix}_{object_stream}_{t}``), by default ``leading``.
    onnx : bool, optional
        Participate in the ONNX manifest, by default True.

    Raises
    ------
    ConfigError
        On an empty ``object_classes`` list.
    """

    # v1 truth/raw column names — byte parity
    _CLASS_TARGET_COLUMN = "class_label"
    _TRUTH_MASK_COLUMN = "truth_mask"
    _MASK_LOGITS_COLUMN = "mask_logits"

    def __init__(
        self,
        object_classes: Sequence[str],
        object_stream: str = "objects",
        constituent_stream: str = "tracks",
        regression_task: str = "regression",
        leading_prefix: str = "leading",
        onnx: bool = True,
    ) -> None:
        self.object_classes = tuple(object_classes)
        if not self.object_classes:
            raise ConfigError(
                "MaskFormerObjectWriter: object_classes must be non-empty (the per-object "
                "class-probability columns, v1 predictionwriter.py:270)"
            )
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        self.regression_task = str(regression_task)
        self.leading_prefix = str(leading_prefix)
        self.onnx = bool(onnx)

    # -- shared per-family suffix helpers (ONE owner, both modes) ----------------

    def _class_suffixes(self) -> list[str]:
        """The per-object-class probability suffixes: ``p{name}`` per class, no `Flavours` resolution."""
        return [f"p{name}" for name in self.object_classes]

    def _class_key(self) -> str:
        """The decoder class-probability bundle key (``<object_stream>.class_probs``)."""
        return f"{self.object_stream}.class_probs"

    def _masks_key(self) -> str:
        """The decoder mask-logits bundle key (``<object_stream>.masks``)."""
        return f"{self.object_stream}.masks"

    def _pad_mask_key(self) -> str:
        """The constituent pad-mask bundle key (``masks.<constituent_stream>``)."""
        return f"masks.{self.constituent_stream}"

    def _class_label_key(self) -> str:
        """The truth object-class label key (``labels.<object_stream>.object_class``)."""
        return f"labels.{self.object_stream}.object_class"

    def _truth_mask_key(self) -> str:
        """The truth object-mask key (``labels.<object_stream>.masks``)."""
        return f"labels.{self.object_stream}.masks"

    def _reg_pred_key(self) -> str:
        """The object-regression prediction key (``preds.<object_stream>.regression``)."""
        return f"preds.{self.object_stream}.{self.regression_task}"

    # -- TEST role (executes) -----------------------

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Decoder ``<object_stream>.{class_probs,masks}`` + `MaskFormerTargets`
        truth labels (TEST-only, keeps targets alive) + the constituent pad
        mask (TEST-only; `write` uses it to set padded ``MaskIndex`` to -1).
        """
        del ctx
        return {
            self._class_key(): TensorSpec(shape=None, dtype=None),
            self._masks_key(): TensorSpec(shape=None, dtype=None),
            self._pad_mask_key(): TensorSpec(
                shape=None, dtype="bool", kind="pad_mask", modes=Mode.TEST
            ),
            self._class_label_key(): TensorSpec(
                shape=None, dtype="int64", kind="label", modes=Mode.TEST
            ),
            self._truth_mask_key(): TensorSpec(
                shape=None, dtype="bool", kind="label", modes=Mode.TEST
            ),
        }

    def extra_groups(self, ctx: WriteCtx) -> dict[str, tuple[int, ...]]:
        """``objects`` ``[total, M]`` and ``object_masks`` ``[total, M, T]``, M from
        the decoder's object axis, T from the constituent stream's file
        sequence length (raises `ConfigError` if it's not a sequence stream).
        ``MaskIndex`` rides the reader stream itself — no extra group needed.
        """
        num_objects = self._num_objects(ctx.model_modules)
        if self.constituent_stream not in ctx.seq_lengths:
            raise ConfigError(
                f"MaskFormerObjectWriter {self.name!r}: constituent stream "
                f"{self.constituent_stream!r} is not a sequence stream (no file token length) — "
                f"the object masks span its constituents (sequence streams: "
                f"{sorted(ctx.seq_lengths)})"
            )
        tok = ctx.seq_lengths[self.constituent_stream]
        return {self.object_stream: (num_objects,), "object_masks": (num_objects, tok)}

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the per-group object columns: ``objects`` (class probs + truth class),
        the constituent stream (``MaskIndex``), and ``object_masks`` (truth mask + mask logits).
        """
        fmt = "f2" if ctx.precision == "half" else "f4"
        prob_descr = [(f"{ctx.run_name}_{s}", fmt) for s in self._class_suffixes()]
        prob_descr.append((self._CLASS_TARGET_COLUMN, "i8"))
        return {
            self.object_stream: np.dtype(prob_descr),
            self.constituent_stream: np.dtype([(f"{ctx.run_name}_{OBJECT_INDEX.test}", "i8")]),
            "object_masks": np.dtype([
                (self._TRUTH_MASK_COLUMN, "i8"),
                (self._MASK_LOGITS_COLUMN, fmt),
            ]),
        }

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """The statically-derivable eval columns — the same names `columns` declares."""
        del ctx
        prob_cols = [f"{run_name}_{s}" for s in self._class_suffixes()]
        return {
            self.object_stream: [*prob_cols, self._CLASS_TARGET_COLUMN],
            self.constituent_stream: [f"{run_name}_{OBJECT_INDEX.test}"],
            "object_masks": [self._TRUTH_MASK_COLUMN, self._MASK_LOGITS_COLUMN],
        }

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Format one batch into the ``{objects, constituent_stream, object_masks}`` structured arrays."""
        del rows
        ctx = self.ctx
        fmt = "f2" if ctx.precision == "half" else "f4"
        class_probs = bundle.get(self._class_key())  # [B, M, C]
        masks = bundle.get(self._masks_key())  # [B, M, T]
        class_label = bundle.get(self._class_label_key())  # [B, M]
        truth_mask = bundle.get(self._truth_mask_key())  # [B, M, T]

        # objects group: per-class probs + truth class
        prob_dtype = np.dtype([(f"{ctx.run_name}_{s}", fmt) for s in self._class_suffixes()])
        prob_arr = u2s(class_probs.cpu().float().numpy(), prob_dtype)
        target_arr = u2s(
            class_label.cpu().unsqueeze(-1).numpy(),
            np.dtype([(self._CLASS_TARGET_COLUMN, "i8")]),
        )
        objects_out = self._join(prob_arr, target_arr)

        # MaskIndex on the constituent stream: per-constituent owning object via
        # indices_from_mask(sigmoid > 0.5) (noindex -2), padded constituents -> -1
        mask_indices = indices_from_mask(masks.cpu().sigmoid() > 0.5)  # [B, T] int64, -2 noindex
        mask_indices = mask_indices.int().cpu().numpy()
        pad = bundle.get(self._pad_mask_key()).cpu().numpy()  # True = padded
        mask_indices = np.where(~pad, mask_indices, -1)  # padded constituents -> -1
        index_arr = u2s(
            np.expand_dims(mask_indices, -1),
            np.dtype([(f"{ctx.run_name}_{OBJECT_INDEX.test}", "i8")]),
        )

        # object_masks group: truth mask + raw mask logits
        tgt_arr = u2s(
            truth_mask.cpu().unsqueeze(-1).numpy(),
            np.dtype([(self._TRUTH_MASK_COLUMN, "i8")]),
        )
        logits_arr = u2s(
            masks.cpu().float().unsqueeze(-1).numpy(),
            np.dtype([(self._MASK_LOGITS_COLUMN, fmt)]),
        )
        object_masks_out = self._join(tgt_arr, logits_arr)

        return {
            self.object_stream: objects_out,
            self.constituent_stream: index_arr,
            "object_masks": object_masks_out,
        }

    @staticmethod
    def _join(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Merge two same-shape structured arrays into one (field union, left first)."""
        descr = list(left.dtype.descr) + list(right.dtype.descr)
        out = np.empty(left.shape, dtype=np.dtype(descr))
        for name in left.dtype.names:
            out[name] = left[name]
        for name in right.dtype.names:
            out[name] = right[name]
        return out

    def _num_objects(self, model_modules: Mapping[str, GraphModule]) -> int:
        """The decoder's object-query count ``M`` (read from the module producing
        ``<object_stream>.masks``); raises `ConfigError` if none exposes
        ``num_objects`` for the stream.
        """
        for module in model_modules.values():
            num = getattr(module, "num_objects", None)
            out_stream = getattr(module, "out_stream", None)
            if isinstance(num, int) and out_stream == self.object_stream:
                return num
        raise ConfigError(
            f"MaskFormerObjectWriter {self.name!r}: no configured module produces object stream "
            f"{self.object_stream!r} with a num_objects attribute (a MaskDecoder out_stream) — "
            "the object H5 axis size cannot be resolved (design §5.2)"
        )

    # -- ONNX role (declares) --------------------------------

    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The two object ONNX outputs: leading-object regression scalars first,
        then the per-constituent object index. ``[]`` when ``onnx: false``.

        Raises `ConfigError` when ``regression_task`` isn't ``"regression"`` —
        the ``object_index`` reduce always reads the fixed
        ``preds.<object_stream>.regression`` key regardless of task name.
        """
        if not self.onnx:
            return []
        if self.regression_task != "regression":
            raise ConfigError(
                f"MaskFormerObjectWriter {self.name!r}: ONNX export requires "
                f"regression_task == 'regression', got {self.regression_task!r}. The "
                "object_index reduce reorders by the leading object's regression but is "
                "declared on the masks port, so it reads the fixed "
                "'preds.<object_stream>.regression' key (v1's single object-regression key, "
                "to_onnx.py:445-469) — a non-default task name would fetch a stale/absent key "
                "at trace time. Rename the object-regression task to 'regression', or set "
                "onnx: false to skip ONNX participation (eval-only)."
            )
        targets = self._regression_suffixes(ctx.model_modules)
        leading_names = [f"{self.leading_prefix}_{self.object_stream}_{t}" for t in targets]
        return [
            ExportOutput(
                port=self._reg_pred_key(),
                names=leading_names,
                reduce="leading_object",
            ),
            ExportOutput(
                port=self._masks_key(),
                name=OBJECT_INDEX.onnx,
                reduce="object_index",
                dtype="int8",
            ),
        ]

    def _regression_suffixes(self, model_modules: Mapping[str, GraphModule]) -> tuple[str, ...]:
        """The object-regression task's per-target suffixes (its
        ``output_suffixes``); raises `ConfigError` if the task isn't
        configured or carries none.
        """
        module = model_modules.get(self.regression_task)
        suffixes = getattr(module, "output_suffixes", None)
        if module is None or suffixes is None:
            candidates = sorted(
                name
                for name, mod in model_modules.items()
                if getattr(mod, "output_suffixes", None) is not None
            )
            raise ConfigError(
                f"MaskFormerObjectWriter {self.name!r}: regression_task "
                f"{self.regression_task!r} does not name a configured object-regression task "
                f"(needs output_suffixes — the leading-object ONNX scalar names, "
                f"to_onnx.py:300-303). Candidates: {candidates or '<none>'}"
            )
        return tuple(suffixes)
