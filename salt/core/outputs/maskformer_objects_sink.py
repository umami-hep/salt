"""`MaskFormerObjectsSink` — the sink-hosted MaskFormer object writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec
from salt.core.outputs.maskformer import MaskFormerObjectWriter
from salt.core.outputs.run_task_output import OutputSectionWriter
from salt.core.outputs.writer_base import WriterDeclareCtx


class _MFWriteCtxShim:
    """Minimal stand-in for the legacy `WriteCtx`, carrying only the two fields
    `MaskFormerObjectWriter.write` reads — ``run_name`` + ``precision``.

    The host `H5OutputSink` owns the run name and the f4/f2 float policy, so the
    relocated writer needs nothing else; the object axis ``M`` rides the node
    (``bind_model_modules``), the constituent token length rides the sink-
    supplied `_ExtraGroupCtx`.
    """

    __slots__ = ("precision", "run_name")

    def __init__(self, run_name: str, precision: str) -> None:
        self.run_name = run_name
        self.precision = precision


class MaskFormerObjectsSink(OutputSectionWriter):
    """Sink-hosted MaskFormer object writer.

    Relocates the legacy `salt.core.writers.MaskFormerObjectWriter` TEST role
    onto the `H5OutputSink` ``extra_groups`` seam so MaskFormer's eval-H5 object
    columns are produced on the sink path, not the legacy `WriterCallback`.

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

    Byte parity: an internal `MaskFormerObjectWriter` (``onnx=False`` — ONNX is
    wired separately via the `MaskFormerObjects` conversion node +
    `OnnxExportSink`) owns ``columns`` + ``write``, so the emitted column set,
    dtypes, the ``[B, M]`` compactness and the ``-2``/``-1`` ``MaskIndex``
    sentinels are produced by the same code as the legacy writer — the sink
    path can never drift from it.

    Parameters
    ----------
    object_classes : Sequence[str]
        All object class names including the trailing ``null`` (the per-object
        probability columns ``{run_name}_p{name}``; v1 ``object.class_names``).
    object_stream : str, optional
        The decoder object stream (the ``objects`` / ``object_masks`` H5 groups
        derive from it), by default ``objects``.
    constituent_stream : str, optional
        The constituent reader stream the masks span; the ``MaskIndex`` column
        lands here, by default ``tracks``.
    regression_task : str, optional
        The object-regression task instance name (carried for symmetry with the
        legacy writer; inert on the eval-only sink path — ``onnx=False``), by
        default ``regression``.
    """

    name = "maskformer_objects"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        object_classes: Sequence[str],
        object_stream: str = "objects",
        constituent_stream: str = "tracks",
        regression_task: str = "regression",
    ) -> None:
        super().__init__()
        self.name = type(self).name
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        # the legacy writer owns the byte-identical columns()/write() ops. onnx=False:
        # ONNX participation is wired separately (the MaskFormerObjects conversion node
        # + OnnxExportSink), so this eval-only fold never declares an export manifest.
        self._writer = MaskFormerObjectWriter(
            object_classes=object_classes,
            object_stream=object_stream,
            constituent_stream=constituent_stream,
            regression_task=regression_task,
            onnx=False,
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

        Mirrors `MaskFormerObjectWriter.extra_groups`, but reads ``M`` from the
        bound decoder (`bind_model_modules`) rather than ``ctx.model_modules``
        (the sink's `_ExtraGroupCtx` carries only file geometry).

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
        """The per-group object columns — delegated verbatim to the legacy writer.

        `MaskFormerObjectWriter.columns` reads only ``ctx.run_name`` +
        ``ctx.precision`` (both carried by the sink's `_ExtraGroupCtx`), so the
        byte-identical column schema (``{run_name}_p{class}`` f4 + ``class_label``
        i8 on ``objects``; the ``MaskIndex`` i8 on the constituent stream;
        ``truth_mask`` i8 + ``mask_logits`` f4 on ``object_masks``) is produced
        by the same code the legacy writer ships.
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
        """One batch of structured arrays — delegated verbatim to the legacy writer.

        The host sink supplies ``run_name`` + ``precision`` (it owns the column
        prefix + float policy); a `_MFWriteCtxShim` carries them to
        `MaskFormerObjectWriter.write`, which packs the exact legacy columns.
        The sink then re-expands the per-token ``MaskIndex`` to the file token
        length and merges the ``objects`` / ``object_masks`` extra groups.
        """
        self._writer.ctx = _MFWriteCtxShim(run_name, precision)
        return self._writer.write(bundle, rows)
