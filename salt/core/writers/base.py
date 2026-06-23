"""Writer-side base class and contexts (design §2.7, §8; M4.5 unified manifest).

A `Writer` is the SINGLE output manifest of a salt2 model, with one role
per output mode (the M4.5 amendment, ``amendment-unified-writers.md``):

- **TEST — the writer executes** (unchanged from M3): it declares the
  bundle keys it consumes (`Writer.requires` — kind-typed `TensorSpec`s,
  exactly like a module's ``declare_io``) and turns each test batch into
  structured-array columns for the shared H5 sink owned by
  `salt.core.writers.callback.WriterCallback`. Because the requires are
  declared statically, demand-gating works end to end: writer-demanded keys
  keep their producers alive in the TEST plan, dataset-served keys (labels,
  masks, ``meta.rows``) flow into the boundary demand, and a produced
  ``preds.*`` key no writer consumes is a hard `ConfigError` (design §4.2,
  §8 — wired in `salt.core.saltmodule.SaltModule._model_sinks`).
- **ONNX — the writer is read, never run**: `Writer.onnx_outputs` declares
  the writer's export-manifest entries as M4 `ExportOutput` objects
  (amendment merge condition 2 — no parallel type), and the exporter
  assembles ``export.outputs`` from them. ONNX demand derives from the
  manifest (``{o.port for o in onnx_outputs(ctx)}``) — one demand mechanism
  in both output modes. **Export = reduces only: ``write()`` never traces
  and never runs inside Athena** (amendment cost 1) — the manifest is
  declarative precisely because arbitrary ``write()`` numpy cannot enter
  the traced graph. Export math comes from the LIVE reduce registry
  (``salt.core.onnx.reduces``; ``salt.core.onnx.config.KNOWN_REDUCES`` is a
  live VIEW of its registered names — ``split_scalars``, ``argmax``,
  ``vertex_union_find`` ship at import, plus the MaskFormer
  ``leading_object``/``object_index``). The public
  ``salt.core.onnx.reduces.register_reduce`` surface has LANDED (M5 sub-wave
  C/D, amendment addendum 555-567), so custom and export-only writers may
  register their OWN export math (a binder + declared dtype + per-token flag)
  and name it in ``ExportOutput.reduce``.

Both directions of mode-narrowing are first-class (amendment merge
condition 3): a writer that never overrides `Writer.onnx_outputs` is
**eval-only** (the default — `InputCopyWriter`, `PadMaskWriter`, truth
columns; ``TaskWriter(onnx=false)`` narrows per instance), and
`ExportOnlyWriter` is THE blessed **export-only** pattern (non-empty
manifest, no TEST role, explicit ``export_only`` flag). A writer with
neither role is a `ConfigError` (design principle 10 extended to writers).
Naming: writers declare logical *suffixes* (`salt.core.writers.names`);
TEST prefixes with the run name, ONNX with ``export.model_name``
(amendment §5).

Design-deviation note: §2.7 sketches ``columns(schema: ResolvedSchema)``.
The implemented signature is ``columns(ctx: WriteCtx)`` — writers need
*file* metadata (source dtypes for input copies, the file's sequence
lengths for v1's ``maybe_pad`` re-expansion) which the model-side
`ResolvedSchema` does not carry; `WriteCtx` bundles both. Functionally
equivalent: the output schema is still declared up front, so the file and
all datasets exist before the first batch (the empty-test-set fix, §8).

M5 sub-wave D — writer-spec validation LANDED: the `TensorSpec` VALUES of a
writer's declared requires now drive both KEY demand AND a static kind/dtype
check. Key serveability was always validated end to end (a model-produced key
anchors the TEST plan; an unserveable dataset-namespace key raises at
`SaltModule._boundary_demand`); `WriterCallback.validate_specs` (called from
TWO entry points: `SaltModule.setup` on the TEST path at run setup, AND
`salt2 graph validate` for the TEST mode data-free — `cli._cmd_validate`, the
canonical CI static-validator command — both after the TEST plan compiles and
before bind/the first batch) additionally unifies each writer-declared
require's kind/dtype against the leaf that actually produces it — a model
TEST-port or a dataset boundary source — reusing the planner's
`_unify_edge`/`KindError` rules (kind must match; dtypes must match when both
are declared, a ``None`` on either side unifies with anything). §2.7's
"kind-typed requires, exactly like ``declare_io``" now holds for real on the
writer→producer edge. SHAPE unification is intentionally not replayed: writer
requires carry symbolic-dim shapes whose batch/token dims only bind against
the live boundary inside the compiled plan (the plan's own `_unify_edge`
unifies model-side shapes end to end), so the writer-specific gap was kind and
dtype, which is what the validator closes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import GraphModule, TensorSpec
from salt.core.onnx.config import ExportOutput

__all__ = ["ExportOnlyWriter", "WriteCtx", "Writer", "WriterDeclareCtx", "task_modules"]

_UNNAMED = "unnamed"


def task_modules(model_modules: Mapping[str, GraphModule]) -> dict[str, GraphModule]:
    """Discover task modules by duck-typing (the ``tasks.py`` surface).

    A "task module" is any model module exposing string ``pred_key`` and
    ``stream`` attributes (`salt.core.nn.tasks._TaskModuleBase`; duck-typed
    so user task modules participate too — the `check_class_names`
    precedent, ``saltmodule.py``). Order is the module-dict order — the
    config declaration order, which is also the v1 ``model.tasks`` order
    for converted configs (design §9.2).

    The plan-29 conversion PRODUCERS (`salt.core.outputs.TaskOutput` and its
    `ClassProbs`/`SeqClassIndex`/`Regression` subclasses) also expose
    ``pred_key``/``stream`` (they READ ``preds.<stream>.<task>``), but they are
    NOT task heads — they produce an ``outputs.*`` leaf (``output_key``) and ship
    no TEST-column rendering. Exclude any module carrying an ``output_key`` so the
    TaskWriter never tries to render eval columns for a conversion node (W4).

    Parameters
    ----------
    model_modules : Mapping[str, GraphModule]
        The model-side module dict (``SaltModule`` net contents).

    Returns
    -------
    dict[str, GraphModule]
        ``{instance name: module}`` for every task-shaped module, in
        declaration order.
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

    Built by `WriterCallback.writer_demand` from the model module dict and
    the configured reader — available before any plan is compiled, so the
    declared requires can anchor TEST demand (design §8).

    Parameters
    ----------
    model_modules : Mapping[str, GraphModule]
        The model-side module dict (instance name -> module).
    streams : tuple[str, ...]
        The reader's stream names, in config order.
    sequence_streams : tuple[str, ...]
        The subset of `streams` carrying a pad mask (``global_object: false``
        groups, design §6.1).
    """

    model_modules: Mapping[str, GraphModule]
    streams: tuple[str, ...]
    sequence_streams: tuple[str, ...]


@dataclass(frozen=True)
class WriteCtx:
    """Test-run context handed to `Writer.setup` (design §2.7).

    Parameters
    ----------
    output_path : Path
        The resolved output H5 file path.
    total : int
        Number of rows that will be written (the H5Writer fixed-mode size).
    run_name : str
        The run name (`SaltModule.name`) — the v1 ``model_name`` column
        prefix (``task.py:140-151`` naming contract).
    source_path : Path
        The resolved test source file (VDS for wildcards) — input copies
        re-read from here by ``meta.rows``.
    streams : tuple[str, ...]
        Reader stream names, in config order.
    sequence_streams : tuple[str, ...]
        Streams carrying a pad mask.
    group_datasets : Mapping[str, str]
        Stream -> H5 dataset name (the v1 ``input_map``; output groups are
        named after the *file* datasets, ``predictionwriter.py:182-183``).
    seq_lengths : Mapping[str, int]
        Sequence stream -> the FILE's constituent dimension. Per-token
        writer fragments are zero-padded to this length (v1 ``maybe_pad``
        truncation re-expansion, ``array_utils.py:62-90``).
    model_modules : Mapping[str, GraphModule]
        The model-side module dict.
    batch_size : int
        Configured test batch size.
    precision : str
        H5Writer float policy: ``"full"`` (f4) or ``"half"`` (f2).
    feature_fields : Mapping[str, tuple[str, ...]]
        Bundle key -> declared last-dim column names, from the bind-time
        `ResolvedSchema` (e.g. ``"inputs.tracks"`` -> the configured
        `Features` variable order). Lets custom writers resolve feature
        columns BY NAME (``ctx.feature_fields["inputs.tracks"].index("d0")``)
        instead of hard-coding index arithmetic off the config — the exact
        failure class ``TensorSpec.fields`` exists to kill (M3-review
        ergonomics fix). Empty when the LightningModule carries no resolved
        schema.
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
    """The single output manifest: executes in TEST, is read in ONNX (design §2.7, M4.5).

    Writers are plain config-constructed objects (no parameters, not
    ``nn.Module``); jsonargparse instantiates them from the top-level
    ``writers.modules`` dict (design §5.1) and `WriterCallback` drives the
    TEST lifecycle: ``requires`` (static, demand wiring) -> ``setup`` ->
    ``columns`` (file created before the first batch) -> ``write`` per
    batch -> ``finalize``. In ONNX mode NOTHING here runs — the exporter
    only *reads* `onnx_outputs` (and `WriterCallback` assembles the
    declarations into the export manifest, M4.5 amendment §2/§4).

    Custom writers are the user story for "I want a new column": subclass,
    declare requires, return a structured array per stream, and add four
    YAML lines under ``writers.modules`` (design §8). To additionally
    export, override `onnx_outputs` with ports + suffixes + a registered
    reduce (the live ``salt.core.onnx.reduces`` registry, viewed via
    ``salt.core.onnx.config.KNOWN_REDUCES``; custom export math registers
    through the landed ``salt.core.onnx.reduces.register_reduce`` surface) —
    ``write()`` is numpy and never traces (amendment cost 1).
    """

    name: str = _UNNAMED
    """Instance name — assigned from the ``writers.modules`` config key."""

    export_only: bool = False
    """The explicit export-only flag (amendment merge condition 3).

    False (default) for every writer with a TEST role. A writer whose TEST
    role is empty but whose ONNX manifest is not MUST set this flag — the
    blessed spelling is subclassing `ExportOnlyWriter` — or writer-role
    validation raises a `ConfigError` (the stub shape is never implicit).
    """

    @abstractmethod
    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the consumed bundle keys (static, config-only).

        Parameters
        ----------
        ctx : WriterDeclareCtx
            Model module dict + reader stream info.

        Returns
        -------
        dict[str, TensorSpec]
            ``{dotted bundle key: spec}`` — these keys are TEST-graph sink
            demand (design §8): model-produced keys become plan sinks,
            dataset-namespace keys join the boundary demand.
        """

    def setup(self, ctx: WriteCtx) -> None:
        """Per-test-run setup; the default stores `ctx` as ``self.ctx``."""
        self.ctx = ctx

    @abstractmethod
    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare the output columns UP FRONT (file created before any batch).

        Parameters
        ----------
        ctx : WriteCtx
            The test-run context (same object as `setup` received).

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}`` — field order is part of the
            output byte-schema; `WriterCallback` merges fragments across
            writers in config order with a collision check (design §8).
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

    # -- ONNX role: declarative only, never executed (M4.5 amendment §2.1) ------

    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The writer's ONNX-manifest entries (M4's `ExportOutput`, condition 2).

        Default: the writer does not exist in the ONNX graph — the
        eval-only direction (`InputCopyWriter`, `PadMaskWriter`, truth
        columns; amendment §3). Static and config-only (`WriterDeclareCtx`)
        so export-on-a-laptop (design §2.3, §7 step 1) is preserved by
        construction. Entries carry logical SUFFIXES (``name``/``names``);
        the exporter prefixes ``{export.model_name}_`` (amendment §5 rule
        3) — writers never see the Athena name.

        Parameters
        ----------
        ctx : WriterDeclareCtx
            Model module dict + reader stream info (the same context
            `requires` receives).

        Returns
        -------
        list[ExportOutput]
            The manifest entries, in this writer's output order.
        """
        del ctx
        return []

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """Statically-derivable TEST column names per stream (design §4.4 annotation).

        The eval half of the ``salt2 graph resolve [--annotate]`` manifest
        annotation (amendment merge condition 8): writers that can name
        their eval columns from config alone override this; file-dependent
        writers (`InputCopyWriter` — columns ride from the source file)
        keep the empty default and are annotated as file-dependent. This is
        documentation surface ONLY — the binding TEST schema remains
        `columns` (checked against this in the U1 manifest-coherence gate).

        Parameters
        ----------
        ctx : WriterDeclareCtx
            Model module dict + reader stream info.
        run_name : str
            The run ``name:`` — the TEST column prefix (amendment §5
            rule 2).

        Returns
        -------
        dict[str, list[str]]
            ``{stream: [column names]}`` (empty by default).
        """
        del ctx, run_name
        return {}

    def extra_groups(self, ctx: WriteCtx) -> dict[str, tuple[int, ...]]:
        """Declare OUTPUT groups that are not reader input streams (design §8).

        The shipped task/copy/mask writers write ONLY into reader-stream H5
        groups (``jets``, ``tracks``, ...), and `WriterCallback._merge_columns`
        sizes those groups from the source file's per-stream geometry. A few
        writers need NEW groups with their own first-class shapes — the
        MaskFormer object writer's ``objects`` ``[total, M]`` and
        ``object_masks`` ``[total, M, T]`` groups (v1
        ``predictionwriter.py:267-308``), which have an object axis ``M`` that no
        reader stream carries. A writer declaring such columns in `columns`
        MUST size them here: ``{group: trailing shape}`` where ``trailing
        shape`` is the per-row shape WITHOUT the leading ``total`` row dim
        (``(M,)`` for ``objects``, ``(M, T)`` for ``object_masks``). The
        callback names the H5 dataset after the group and merges the extra
        groups alongside the reader-stream groups (same collision/order rules).
        Default: no extra groups (the shipped task/copy/mask writers).

        Parameters
        ----------
        ctx : WriteCtx
            The test-run context (same object as `setup` / `columns` receive) —
            carries ``seq_lengths`` for constituent-aligned trailing dims.

        Returns
        -------
        dict[str, tuple[int, ...]]
            ``{group name: trailing per-row shape}`` (empty by default).
        """
        del ctx
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ExportOnlyWriter(Writer):
    """THE export-only writer pattern (M4.5 amendment merge condition 3 — blessed).

    A writer that exists ONLY in the ONNX manifest: no TEST demand, no
    eval columns, nothing written — `onnx_outputs` is its single role.
    Use it to declare an Athena output with no eval analogue (the inverse
    of the default eval-only direction). Subclass and override
    `onnx_outputs` only — e.g. echoing the normalised global features as
    extra Athena outputs (run verbatim in ``test_manifest.py`` and, at
    export scale, in the U1 gate):

    .. code-block:: python

        class JetEcho(ExportOnlyWriter):
            def onnx_outputs(self, ctx):
                # split_scalars: one float32 output per suffix; the suffix
                # count must match the port's feature count
                return [ExportOutput(port="normed.jets", names=["jetEcho0", "jetEcho1"])]

    The ``export_only`` flag is what blesses the shape: a hand-rolled
    writer with empty ``columns()`` and a non-empty manifest WITHOUT the
    flag is rejected by writer-role validation (`WriterCallback`) — the
    cargo-cult stub is never silently legal. Remember the export contract:
    the output math is the entry's ``reduce``, which must name a registered
    reduce — the shipped keys ``split_scalars`` (the ``names:`` default),
    ``argmax`` or ``vertex_union_find`` (viewed via
    ``salt.core.onnx.config.KNOWN_REDUCES``), or a CUSTOM reduce you register
    through the landed ``salt.core.onnx.reduces.register_reduce`` surface (a
    binder + declared dtype + per-token flag; M5 sub-wave C/D, amendment
    addendum 555-567). There is no ``write()`` to put custom math in — export
    math lives in the registered reduce. One more rule of the manifest: one
    writer owns one export port — re-exporting a port another writer already
    declares (e.g. a ``preds.*`` key the `TaskWriter` exports) needs that
    writer narrowed first (``onnx_tasks``/``onnx_streams``).
    """

    export_only: bool = True

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """No TEST demand — export-only writers never join the TEST plan.

        Returns
        -------
        dict[str, TensorSpec]
            Always empty.
        """
        del ctx
        return {}

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """No eval columns.

        Returns
        -------
        dict[str, np.dtype]
            Always empty.
        """
        del ctx
        return {}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Nothing to write (and never called: no columns were declared).

        Returns
        -------
        dict[str, np.ndarray]
            Always empty.
        """
        del bundle, rows
        return {}

    @abstractmethod
    def onnx_outputs(self, ctx: WriterDeclareCtx) -> list[ExportOutput]:
        """The export-manifest entries — the writer's single role.

        Returns
        -------
        list[ExportOutput]
            Must be non-empty (a both-empty writer violates principle 10).
        """
