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
    """Persist MaskFormer object predictions + truth, and declare the ONNX object outputs.

    Reproduces v1's object-writing block byte-for-byte in TEST and declares
    v1's two object ONNX outputs in ONNX — a single output manifest for both
    modes. The decoder's ``<object_stream>.*`` keys and the
    ``MaskFormerTargets`` truth labels are its TEST demand; the
    object-regression ``preds.<object_stream>.regression`` and the decoder
    ``<object_stream>.masks`` are its ONNX manifest ports.

    Parameters
    ----------
    object_classes : Sequence[str]
        All object class names, including the trailing ``null`` (v1
        ``object.class_names``; e.g. ``["b", "c", "null"]``). The
        class-probability columns are ``{run_name}_p{name}`` per class — a
        plain ``p`` prefix with no `Flavours` resolution (objects are not jet
        flavours). The list length must equal the decoder's published
        ``<object_stream>.class_probs`` column count (``num_classes + 1``).
    object_stream : str, optional
        The decoder's object bundle stream (the `MaskDecoder` ``out_stream`` and
        the `MaskFormerTargets` object stream), by default ``objects``. The
        ``objects``/``object_masks`` H5 groups derive from it.
    constituent_stream : str, optional
        The constituent reader stream the masks span, by default ``tracks``.
        The ``MaskIndex`` column lands on this stream.
    regression_task : str, optional
        The object-regression task instance name (its
        ``preds.<object_stream>.<regression_task>`` port is the
        ``leading_object`` ONNX entry), by default ``regression``. For ONNX
        export (``onnx: true``) it must be ``"regression"``: the
        ``object_index`` reduce is declared on the masks port and reads the
        fixed ``preds.<object_stream>.regression`` key, so a non-default name
        raises a `ConfigError` at `onnx_outputs` (eval-only ``onnx: false``
        writers may use any name).
    leading_prefix : str, optional
        The leading-object ONNX suffix template; each regression target ``t``
        becomes ``{leading_prefix}_{object_stream}_{t}``, by default
        ``leading``.
    onnx : bool, optional
        Participate in the ONNX manifest, by default True. ``False`` makes the
        writer eval-only.

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
        """Declare the consumed decoder predictions + truth labels (TEST demand).

        The decoder's ``<object_stream>.{class_probs,masks}`` and the
        `MaskFormerTargets` truth labels ``labels.<object_stream>.{object_class,
        masks}`` — the truth requires keep `MaskFormerTargets` alive in the TEST
        plan. Truth labels are TEST-only (the matched loss owns the FIT|VAL
        truth demand). The constituent pad mask ``masks.<constituent_stream>``
        is also declared (TEST-only, ``kind="pad_mask"``): `write` consumes it
        to set padded constituents' ``MaskIndex`` to ``-1``, so the plan must
        provide it rather than relying on incidental availability.
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
        """The two non-reader output groups: ``objects`` and ``object_masks``.

        ``objects`` is ``[total, M]`` (the object axis) and ``object_masks`` is
        ``[total, M, T]`` (object x constituent), where ``M`` is read from the
        decoder's published object axis and ``T`` from the constituent stream's
        file sequence length. The constituent ``MaskIndex`` column rides on the
        reader stream itself, so it needs no extra group.

        Raises
        ------
        ConfigError
            When the constituent stream has no file sequence length (it must be a
            sequence stream — the masks span its tokens).
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
        """The decoder's object-query count ``M`` (the ``objects`` group axis).

        Read from the decoder module producing ``<object_stream>.masks`` (its
        ``num_objects`` attribute) so the H5 object axis matches the model.

        Raises
        ------
        ConfigError
            When no configured module exposes ``num_objects`` for the object
            stream.
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
        """Declare the two object ONNX outputs, in v1 emission order (leading-object scalars
        then the per-constituent index).

        ``preds.<object_stream>.regression`` -> the ``leading_object`` reduce (R
        global float32 leading-object regression scalars, suffixed
        ``{leading_prefix}_{object_stream}_{t}`` per target) and
        ``<object_stream>.masks`` -> the ``object_index`` reduce (one int8
        per-constituent index named `OBJECT_INDEX.onnx` = ``HadronIndex``).
        Truth columns never appear: `requires` is TEST-only, so the labels
        demand never reaches the ONNX plan. The leading suffixes derive from
        the object-regression task's ``output_suffixes`` — one owner, both
        modes.

        Returns ``[]`` with ``onnx: false``. Propagates the
        `_regression_suffixes` `ConfigError` when the object-regression task is
        not a configured module with ``output_suffixes``.

        Raises
        ------
        ConfigError
            When ``regression_task`` is not ``"regression"``: the
            ``object_index`` reduce reorders constituents by the leading
            object's regression and is declared on ``<object_stream>.masks``,
            so it cannot carry the regression task name in its own port and
            reads the fixed ``preds.<object_stream>.regression`` key. A
            non-default ``regression_task`` would make the writer declare
            ``preds.<object_stream>.<other>`` while the reduce fetches
            ``preds.<object_stream>.regression`` at trace time — this loud
            config-resolution error replaces that latent trace-time KeyError.
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
        """The object-regression task's per-target suffixes (the leading entry names):
        ``output_suffixes`` of the configured object-regression task (its
        ``custom_output_names`` else its targets — one owner, both modes).

        Raises
        ------
        ConfigError
            When the object-regression task is not configured / carries no
            ``output_suffixes``.
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
