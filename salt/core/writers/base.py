"""Writer-side base class and contexts (design §2.7, §8).

A `Writer` is a TEST-mode graph *sink*: it declares the bundle keys it
consumes (`Writer.requires` — kind-typed `TensorSpec`s, exactly like a
module's ``declare_io``) and turns each test batch into structured-array
columns for the shared H5 sink owned by
`salt.core.writers.callback.WriterCallback`. Because the requires are
declared statically, demand-gating works end to end: writer-demanded keys
keep their producers alive in the TEST plan, dataset-served keys (labels,
masks, ``meta.rows``) flow into the boundary demand, and a produced
``preds.*`` key no writer consumes is a hard `ConfigError` (design §4.2, §8
— wired in `salt.core.saltmodule.SaltModule._model_sinks`).

Design-deviation note: §2.7 sketches ``columns(schema: ResolvedSchema)``.
The implemented signature is ``columns(ctx: WriteCtx)`` — writers need
*file* metadata (source dtypes for input copies, the file's sequence
lengths for v1's ``maybe_pad`` re-expansion) which the model-side
`ResolvedSchema` does not carry; `WriteCtx` bundles both. Functionally
equivalent: the output schema is still declared up front, so the file and
all datasets exist before the first batch (the empty-test-set fix, §8).

Design-deviation note (M3 review): the `TensorSpec` VALUES of a writer's
declared requires drive demand by KEY only — key serveability is validated
end to end (a model-produced key anchors the TEST plan; an unserveable
dataset-namespace key raises at `SaltModule._boundary_demand`) but
kind/dtype/shape unification against the producing port is not yet run for
writer sinks, so §2.7's "kind-typed requires, exactly like ``declare_io``"
holds nominally. Threading the specs through plan compilation (reusing the
planner's unification) is an M5 follow-up, recorded in the study CLAUDE.md
TODOs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import GraphModule, TensorSpec

__all__ = ["WriteCtx", "Writer", "WriterDeclareCtx", "task_modules"]

_UNNAMED = "unnamed"


def task_modules(model_modules: Mapping[str, GraphModule]) -> dict[str, GraphModule]:
    """Discover task modules by duck-typing (the ``tasks.py`` surface).

    A "task module" is any model module exposing string ``pred_key`` and
    ``stream`` attributes (`salt.core.nn.tasks._TaskModuleBase`; duck-typed
    so user task modules participate too — the `check_class_names`
    precedent, ``saltmodule.py``). Order is the module-dict order — the
    config declaration order, which is also the v1 ``model.tasks`` order
    for converted configs (design §9.2).

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
        The subset of `streams` carrying a pad mask (``vector: false``
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
    """Test-time sink module: bundle keys in, structured-array columns out (design §2.7).

    Writers are plain config-constructed objects (no parameters, not
    ``nn.Module``); jsonargparse instantiates them from the top-level
    ``writers.modules`` dict (design §5.1) and `WriterCallback` drives the
    lifecycle: ``requires`` (static, demand wiring) -> ``setup`` ->
    ``columns`` (file created before the first batch) -> ``write`` per
    batch -> ``finalize``.

    Custom writers are the user story for "I want a new column": subclass,
    declare requires, return a structured array per stream, and add four
    YAML lines under ``writers.modules`` (design §8).
    """

    name: str = _UNNAMED
    """Instance name — assigned from the ``writers.modules`` config key."""

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
