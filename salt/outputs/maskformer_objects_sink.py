"""`MaskFormerObjectsSink` — the sink-hosted MaskFormer object writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import _UNNAMED, GraphModule, Mode, TensorSpec
from salt.outputs.names import OBJECT_INDEX
from salt.outputs.run_task_output import OutputSectionWriter
from salt.utils.mask_utils import indices_from_mask


@dataclass(frozen=True)
class WriterDeclareCtx:
    """Config-only context for `MaskFormerObjectWriter.requires` (static, no file I/O).

    Parameters
    ----------
    model_modules : Mapping[str, GraphModule]
        The model-side module dict (instance name -> module).
    streams : tuple[str, ...]
        The reader's stream names, in config order.
    sequence_streams : tuple[str, ...]
        The subset of `streams` carrying a pad mask (``global_object: false`` groups).
    """

    model_modules: Mapping[str, GraphModule]
    streams: tuple[str, ...]
    sequence_streams: tuple[str, ...]


class _MFWriteCtxShim:
    """Minimal write context carrying the two fields `MaskFormerObjectWriter.write`
    reads — ``run_name`` + ``precision``.

    The host `H5OutputSink` owns those; the object axis ``M`` rides the node
    (``bind_model_modules``), the constituent token length rides the sink's
    `_ExtraGroupCtx`.
    """

    __slots__ = ("precision", "run_name")

    def __init__(self, run_name: str, precision: str) -> None:
        self.run_name = run_name
        self.precision = precision


class MaskFormerObjectWriter:
    """The MaskFormer object formatting core — TEST demand, column schema and
    per-batch structured arrays for the object eval-H5 groups.

    Hosted by `MaskFormerObjectsSink`, which delegates ``requires`` /
    ``columns`` / ``write`` / ``_num_objects`` so the emitted column set,
    dtypes, the ``[B, M]`` compactness and the ``-2``/``-1`` ``MaskIndex``
    sentinels are produced by one owner (v1 byte parity,
    predictionwriter.py:267-308).

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
        The object-regression task instance name, by default ``regression``
        (carried for config symmetry; inert on the eval-only sink path).

    Raises
    ------
    ConfigError
        On an empty ``object_classes`` list.
    """

    name: str = _UNNAMED
    """Instance name — assigned by the hosting sink."""

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

    # -- shared per-family suffix helpers ----------------------------------------

    def _class_suffixes(self) -> list[str]:
        """The per-object-class probability suffixes: ``p{name}``, no `Flavours` resolution."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return [f"p{name}" for name in self.object_classes]

    def _class_key(self) -> str:
        """The decoder class-probability bundle key (``<object_stream>.class_probs``)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return f"{self.object_stream}.class_probs"

    def _masks_key(self) -> str:
        """The decoder mask-logits bundle key (``<object_stream>.masks``)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return f"{self.object_stream}.masks"

    def _pad_mask_key(self) -> str:
        """The constituent pad-mask bundle key (``masks.<constituent_stream>``)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return f"masks.{self.constituent_stream}"

    def _class_label_key(self) -> str:
        """The truth object-class label key (``labels.<object_stream>.object_class``)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return f"labels.{self.object_stream}.object_class"

    def _truth_mask_key(self) -> str:
        """The truth object-mask key (``labels.<object_stream>.masks``)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
        return f"labels.{self.object_stream}.masks"

    # -- TEST role -----------------------

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Decoder ``<object_stream>.{class_probs,masks}`` + `MaskFormerTargets`
        truth labels (TEST-only, keeps targets alive) + the constituent pad
        mask (TEST-only; `write` uses it to set padded ``MaskIndex`` to -1).
        """  # noqa: DOC201 - writer-core method (docstring policy)
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

    def columns(self, ctx: Any) -> dict[str, np.dtype]:
        """Declare the per-group object columns: ``objects`` (class probs + truth class),
        the constituent stream (``MaskIndex``), and ``object_masks`` (truth mask + mask logits).
        Reads only ``ctx.run_name``/``ctx.precision``.
        """  # noqa: DOC201 - writer-core method (docstring policy)
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

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Format one batch into the ``{objects, constituent, object_masks}`` structured arrays."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
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
        """Merge two same-shape structured arrays into one (field union, left first)."""  # noqa: DOC201 - writer-core one-liner (docstring policy)
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
        """  # noqa: DOC201, DOC501 - writer-core method (docstring policy)
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


class MaskFormerObjectsSink(OutputSectionWriter):
    """Sink-hosted MaskFormer object writer.

    Hosts the `MaskFormerObjectWriter` TEST role on the `H5OutputSink`
    ``extra_groups`` seam, so MaskFormer's eval-H5 object columns are
    produced on the sink path.

    Like `InputCopyWriter` this is a manifest-only section node
    (``is_manifest_only`` -> not graph-folded; it mints no ``outputs.*`` leaf).
    Instead the host `H5OutputSink`:

    1. folds this node's `sink_requires` (the MaskDecoder products
       ``objects.{class_probs,masks}``, the `MaskFormerTargets` labels
       ``labels.objects.{object_class,masks}``, and the constituent pad mask
       ``masks.<constituent_stream>``) into its own TEST demand — anchoring the
       decoder + targets in the TEST plan and threading those leaves into the
       consume bundle; and
    2. calls this node's `write` per batch and merges the returned structured
       arrays into the eval H5 (the ``objects`` / ``object_masks`` non-reader
       extra groups + the ``MaskIndex`` column on the constituent reader stream).

    The object-prediction group is emitted as the v2-native ``objects`` group
    (named by ``object_stream``), a non-reader extra group. It is NOT merged
    into the ``truth_hadrons`` stream group — that merge, and the per-object
    regression eval columns, stay deferred.

    Byte parity: an internal `MaskFormerObjectWriter` (eval-only — ONNX is
    wired separately via the `MaskFormerObjects` conversion node +
    `OnnxExportSink`) owns ``columns`` + ``write``, so the emitted column set,
    dtypes, the ``[B, M]`` compactness and the ``-2``/``-1`` ``MaskIndex``
    sentinels are produced by one owner — the sink path can never drift.

    Parameters
    ----------
    object_classes : Sequence[str]
        All object class names including the trailing ``null`` (the per-object
        probability columns ``{run_name}_p{name}``).
    object_stream : str, optional
        The decoder object stream (the ``objects`` / ``object_masks`` H5 groups
        derive from it), by default ``objects``.
    constituent_stream : str, optional
        The constituent reader stream the masks span; the ``MaskIndex`` column
        lands here, by default ``tracks``.
    regression_task : str, optional
        The object-regression task instance name (carried for config symmetry;
        inert on the eval-only sink path), by default ``regression``.
    modes : Sequence[str] | None, optional
        The modes this writer runs in (``["test", "export"]`` subset; ``None``
        = both). The object eval columns are H5 (``test``) only — ONNX object
        outputs are wired separately (`MaskFormerObjects` conversion node +
        `OnnxExportSink`).
    """

    name = "maskformer_objects"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        object_classes: Sequence[str],
        object_stream: str = "objects",
        constituent_stream: str = "tracks",
        regression_task: str = "regression",
        modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(modes=modes)
        self.name = type(self).name
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        # the writer core owns the byte-identical columns()/write() ops; ONNX
        # participation is wired separately (the MaskFormerObjects conversion node
        # + OnnxExportSink), so this eval-only fold never declares an export manifest.
        self._writer = MaskFormerObjectWriter(
            object_classes=object_classes,
            object_stream=object_stream,
            constituent_stream=constituent_stream,
            regression_task=regression_task,
        )
        self._writer.name = self.name
        # resolved at bind (the object axis M comes from the bound MaskDecoder).
        self._model_modules: Mapping[str, Any] | None = None
        self._num_objects: int | None = None

    def is_manifest_only(self) -> bool:
        """Not graph-folded: the sink owns this node's demand + per-batch write.

        Like `InputCopyWriter`, the data never flows through the graph; the
        host `H5OutputSink` folds `sink_requires` into its demand and calls
        `write` itself.
        """
        return True

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model modules and resolve the object axis ``M``.

        Called by `SaltModule.compose_output_section` before any plan compiles.
        The object-query count ``M`` is read from the bound `MaskDecoder` (the
        ``num_objects`` attribute of the module whose ``out_stream`` is this
        node's ``object_stream``) — the `_ExtraGroupCtx` the sink later threads
        carries no model modules, so ``M`` must be resolved here.
        """
        self._model_modules = dict(model_modules)
        self._num_objects = self._writer._num_objects(self._model_modules)  # noqa: SLF001

    # -- schema (consumed by H5OutputSink at open_schema) -----------------------

    def extra_groups(self, ctx: Any) -> dict[str, tuple[int, ...]]:
        """The two non-reader output groups ``objects`` ``(M,)`` / ``object_masks`` ``(M, T)``.

        ``M`` is read from the bound decoder (`bind_model_modules`) — the
        sink's `_ExtraGroupCtx` carries only file geometry.

        Raises
        ------
        ConfigError
            When `bind_model_modules` has not run, or the constituent stream has no
            file token length (it must be a sequence stream).
        """
        if self._num_objects is None:
            raise ConfigError(
                f"MaskFormerObjectsSink {self.name!r}: bind_model_modules must run before "
                "extra_groups — the object axis M comes from the bound MaskDecoder (W6b)"
            )
        if self.constituent_stream not in ctx.seq_lengths:
            raise ConfigError(
                f"MaskFormerObjectsSink {self.name!r}: constituent stream "
                f"{self.constituent_stream!r} is not a sequence stream (no file token length) — "
                f"the object masks span its constituents (sequence streams: "
                f"{sorted(ctx.seq_lengths)})"
            )
        tok = ctx.seq_lengths[self.constituent_stream]
        return {self.object_stream: (self._num_objects,), "object_masks": (self._num_objects, tok)}

    def columns(self, ctx: Any) -> dict[str, np.dtype]:
        """Per-group object columns, delegated verbatim to `MaskFormerObjectWriter`
        (reads only ``ctx.run_name``/``ctx.precision``) for a byte-identical schema.
        """
        return self._writer.columns(ctx)

    # -- demand (folded by H5OutputSink into its TEST declare_io requires) -------

    def sink_requires(self) -> dict[str, TensorSpec]:
        """The decoder/truth/pad-mask leaves the host sink must demand on this node's behalf.

        A manifest-only node never anchors its own demand (it is not graph-
        folded); the host `H5OutputSink` folds these into its TEST
        ``declare_io`` requires so the MaskDecoder + `MaskFormerTargets` stay
        alive in the plan and the consume bundle carries the leaves `write`
        reads. Delegated to `MaskFormerObjectWriter.requires` (its ``requires``
        ignores the ctx).
        """
        ctx = WriterDeclareCtx(
            model_modules=self._model_modules or {}, streams=(), sequence_streams=()
        )
        return self._writer.requires(ctx)

    # -- per-batch data (called by H5OutputSink.consume) ------------------------

    def write(
        self, bundle: Bundle, rows: slice, run_name: str, precision: str
    ) -> dict[str, np.ndarray]:
        """One batch of structured arrays, delegated to `MaskFormerObjectWriter.write`
        via a `_MFWriteCtxShim` carrying ``run_name``/``precision``; the sink then
        re-expands ``MaskIndex`` and merges the ``objects``/``object_masks`` groups.
        """
        self._writer.ctx = _MFWriteCtxShim(run_name, precision)
        return self._writer.write(bundle, rows)
