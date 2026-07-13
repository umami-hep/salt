"""Writer-side base class and contexts.

A `Writer` executes in TEST (H5 columns from bundle keys) and is read
declaratively in ONNX (`onnx_outputs` export-manifest entries, never traced).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import _UNNAMED, GraphModule, TensorSpec
from salt.core.onnx.config import ExportOutput

__all__ = ["ExportOnlyWriter", "WriteCtx", "Writer", "WriterDeclareCtx", "task_modules"]


def task_modules(model_modules: Mapping[str, GraphModule]) -> dict[str, GraphModule]:
    """Discover task modules by duck-typing (the ``tasks.py`` surface).

    A "task module" is any model module exposing string ``pred_key`` and
    ``stream`` attributes (`salt.core.nn.tasks._TaskModuleBase`; duck-typed so
    user task modules participate too). Order is the module-dict order — the
    config declaration order, which is also the v1 ``model.tasks`` order for
    converted configs.

    The conversion producers (`salt.core.outputs.TaskOutput` and its
    `ClassProbs`/`SeqClassIndex`/`Regression` subclasses) also expose
    ``pred_key``/``stream`` (they read ``preds.<stream>.<task>``), but they are
    NOT task heads — they produce an ``outputs.*`` leaf (``output_key``) and
    ship no TEST-column rendering. Exclude any module carrying an
    ``output_key`` so the TaskWriter never tries to render eval columns for a
    conversion node.
    """
    return {
        name: module
        for name, module in model_modules.items()
        if isinstance(getattr(module, "pred_key", None), str)
        and isinstance(getattr(module, "stream", None), str)
        and not isinstance(getattr(module, "output_key", None), str)  # exclude conversion producers
    }


@dataclass(frozen=True)
class WriterDeclareCtx:
    """Config-only context for `Writer.requires` (static, no file I/O).

    Built by `WriterCallback.writer_demand` from the model module dict and the
    configured reader — available before any plan is compiled, so the
    declared requires can anchor TEST demand.

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


@dataclass(frozen=True)
class WriteCtx:
    """Test-run context handed to `Writer.setup`.

    Parameters
    ----------
    output_path : Path
        The resolved output H5 file path.
    total : int
        Number of rows that will be written (the H5Writer fixed-mode size).
    run_name : str
        The run name (`SaltModule.name`) — the v1 ``model_name`` column prefix.
    source_path : Path
        The resolved test source file (VDS for wildcards) — input copies
        re-read from here by ``meta.rows``.
    streams : tuple[str, ...]
        Reader stream names, in config order.
    sequence_streams : tuple[str, ...]
        Streams carrying a pad mask.
    group_datasets : Mapping[str, str]
        Stream -> H5 dataset name (the v1 ``input_map``; output groups are
        named after the *file* datasets).
    seq_lengths : Mapping[str, int]
        Sequence stream -> the file's constituent dimension. Per-token writer
        fragments are zero-padded to this length (v1 ``maybe_pad`` truncation
        re-expansion).
    model_modules : Mapping[str, GraphModule]
        The model-side module dict.
    batch_size : int
        Configured test batch size.
    precision : str
        H5Writer float policy: ``"full"`` (f4) or ``"half"`` (f2).
    feature_fields : Mapping[str, tuple[str, ...]]
        Bundle key -> declared last-dim column names, from the bind-time
        `ResolvedSchema` (e.g. ``"inputs.tracks"`` -> the configured `Features`
        variable order). Lets custom writers resolve feature columns by name
        (``ctx.feature_fields["inputs.tracks"].index("d0")``) instead of
        hard-coding index arithmetic off the config. Empty when the
        LightningModule carries no resolved schema.
    """

    output_path: Path
    total: int
    run_name: str
    source_path: Path
    streams: tuple[str, ...]
    sequence_streams: tuple[str, ...]
    group_datasets: Mapping[str, str]
    seq_lengths: Mapping[str, int]
    model_modules: Mapping[str, GraphModule]
    batch_size: int
    precision: str = "full"
    feature_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class Writer(ABC):
    """The single output manifest: executes in TEST, is read in ONNX.

    Writers are plain config-constructed objects (no parameters, not
    ``nn.Module``); jsonargparse instantiates them from the top-level
    ``writers.modules`` dict and `WriterCallback` drives the TEST lifecycle:
    ``requires`` (static, demand wiring) -> ``setup`` -> ``columns`` (file
    created before the first batch) -> ``write`` per batch -> ``finalize``. In
    ONNX mode nothing here runs — the exporter only *reads* `onnx_outputs`
    (`WriterCallback` assembles the declarations into the export manifest).

    Custom writers are the user story for "I want a new column": subclass,
    declare requires, return a structured array per stream, and add a few
    YAML lines under ``writers.modules``. To additionally export, override
    `onnx_outputs` with ports + suffixes + a registered reduce (the live
    ``salt.core.onnx.reduces`` registry, viewed via
    ``salt.core.onnx.config.KNOWN_REDUCES``; custom export math registers
    through ``salt.core.onnx.reduces.register_reduce``) — ``write()`` is numpy
    and never traces.
    """

    name: str = _UNNAMED
    """Instance name — assigned from the ``writers.modules`` config key."""

    export_only: bool = False
    """Explicit export-only flag.

    False (default) for every writer with a TEST role. A writer whose TEST
    role is empty but whose ONNX manifest is not must set this flag — the
    blessed spelling is subclassing `ExportOnlyWriter` — or writer-role
    validation raises a `ConfigError` (the stub shape is never implicit).
    """

    @abstractmethod
    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the consumed bundle keys (static, config-only).

        Returns ``{dotted bundle key: spec}`` — these keys are TEST-graph sink
        demand: model-produced keys become plan sinks, dataset-namespace keys
        join the boundary demand.
        """

    def setup(self, ctx: WriteCtx) -> None:
        """Per-test-run setup; the default stores `ctx` as ``self.ctx``."""
        self.ctx = ctx

    @abstractmethod
    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the output columns up front (file created before any batch).

        Returns ``{stream: structured dtype}`` — field order is part of the
        output byte-schema; `WriterCallback` merges fragments across writers
        in config order with a collision check.
        """

    @abstractmethod
    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Format one batch.

        Parameters
        ----------
        bundle : Bundle
            The executed TEST bundle returned by ``test_step``.
        rows : slice
            Absolute source-file rows of this batch (from ``meta.rows``,
            already verified against the sink's running counter).

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` with first dimension
            ``rows.stop - rows.start`` and dtype exactly as declared by
            `columns`; per-token arrays padded to the file sequence length.
        """

    def finalize(self) -> None:  # noqa: B027 - intentional optional hook (default no-op)
        """Release any per-run resources (default: no-op)."""

    # -- ONNX role: declarative only, never executed -----------------------

    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The writer's ONNX-manifest entries.

        Default: the writer does not exist in the ONNX graph — the eval-only
        direction (`InputCopyWriter`, `PadMaskWriter`, truth columns). Static
        and config-only (`WriterDeclareCtx`) so export-on-a-laptop is
        preserved by construction. Entries carry logical suffixes
        (``name``/``names``); the exporter prefixes ``{export.model_name}_`` —
        writers never see the Athena name.
        """
        del ctx
        return []

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """Statically-derivable TEST column names per stream.

        The eval half of the ``salt2 graph resolve [--annotate]`` manifest
        annotation: writers that can name their eval columns from config alone
        override this; file-dependent writers (`InputCopyWriter` — columns
        ride from the source file) keep the empty default and are annotated as
        file-dependent. This is documentation surface only — the binding TEST
        schema remains `columns` (checked against this in the manifest-
        coherence gate).
        """
        del ctx, run_name
        return {}

    def extra_groups(self, ctx: WriteCtx) -> dict[str, tuple[int, ...]]:
        """Declare output groups that are not reader input streams.

        The shipped task/copy/mask writers write only into reader-stream H5
        groups (``jets``, ``tracks``, ...), and `WriterCallback._merge_columns`
        sizes those groups from the source file's per-stream geometry. A few
        writers need new groups with their own first-class shapes — the
        MaskFormer object writer's ``objects`` ``[total, M]`` and
        ``object_masks`` ``[total, M, T]`` groups, which have an object axis
        ``M`` that no reader stream carries. A writer declaring such columns in
        `columns` must size them here: ``{group: trailing shape}`` where
        ``trailing shape`` is the per-row shape without the leading ``total``
        row dim (``(M,)`` for ``objects``, ``(M, T)`` for ``object_masks``).
        The callback names the H5 dataset after the group and merges the extra
        groups alongside the reader-stream groups (same collision/order
        rules). Default: no extra groups.
        """
        del ctx
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ExportOnlyWriter(Writer):
    """The export-only writer pattern.

    A writer that exists only in the ONNX manifest: no TEST demand, no eval
    columns, nothing written — `onnx_outputs` is its single role. Use it to
    declare an Athena output with no eval analogue (the inverse of the default
    eval-only direction). Subclass and override `onnx_outputs` only — e.g.
    echoing the normalised global features as extra Athena outputs:

    .. code-block:: python

        class JetEcho(ExportOnlyWriter):
            def onnx_outputs(self, ctx):
                # split_scalars: one float32 output per suffix; the suffix
                # count must match the port's feature count
                return [ExportOutput(port="normed.jets", names=["jetEcho0", "jetEcho1"])]

    The ``export_only`` flag is what blesses the shape: a hand-rolled writer
    with empty ``columns()`` and a non-empty manifest without the flag is
    rejected by writer-role validation (`WriterCallback`) — the cargo-cult
    stub is never silently legal. The export contract: the output math is the
    entry's ``reduce``, which must name a registered reduce — the shipped keys
    ``split_scalars`` (the ``names:`` default), ``argmax`` or
    ``vertex_union_find`` (viewed via ``salt.core.onnx.config.KNOWN_REDUCES``),
    or a custom reduce registered through
    ``salt.core.onnx.reduces.register_reduce`` (a binder + declared dtype +
    per-token flag). There is no ``write()`` to put custom math in — export
    math lives in the registered reduce. One writer owns one export port —
    re-exporting a port another writer already declares (e.g. a ``preds.*``
    key the `TaskWriter` exports) needs that writer narrowed first
    (``onnx_tasks``/``onnx_streams``).
    """

    export_only: bool = True

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """No TEST demand — export-only writers never join the TEST plan."""
        del ctx
        return {}

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """No eval columns."""
        del ctx
        return {}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Nothing to write (and never called: no columns were declared)."""
        del bundle, rows
        return {}

    @abstractmethod
    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The export-manifest entries — the writer's single role.

        Must be non-empty (a both-empty writer is invalid).
        """
