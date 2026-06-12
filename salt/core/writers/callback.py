"""`WriterCallback` — the single H5 sink driving the writer modules (design §8).

Built by `Salt2CLI` from the top-level ``writers:`` block (the documented
`SaveConfigCallback`-style wiring, design §8) so ``trainer.callbacks`` stays
free of writer plumbing and overrides address ``writers.modules.<name>``
stably (design §5.3).

Responsibilities:

- **demand**: `writer_demand` merges the writers' declared requires —
  `SaltModule` uses it as the TEST plan's sinks (replacing the M2
  anchor-on-all-preds) and as extra dataset-boundary demand, and raises the
  TEST dead-preds hard error (design §4.2, §8);
- **sink**: ONE ftag `H5Writer` in FIXED mode (``num_jets`` known up front —
  an empty test set still produces a valid empty file, fixing
  ``predictionwriter.py:215-226``), ``shuffle=False``, lz4, v1 dataset
  naming/precision semantics;
- **rows**: alignment is *data* (``meta.rows``) asserted against the sink's
  running counter — sharded or uneven-batch loaders fail loudly instead of
  silently corrupting alignment (design §8); multi-device test writing is
  out of scope (asserted single-device, as v1).

File creation happens at ``on_test_start`` (after checkpoint restore, before
the first batch): ``trainer.ckpt_path`` and the datamodule's test dataset
are both guaranteed by then, and the design's create-before-first-batch
contract still holds.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag.hdf5 import H5Writer
from lightning import Callback, LightningModule, Trainer

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import GraphModule
from salt.core.writers.base import WriteCtx, Writer, WriterDeclareCtx
from salt.utils.array_utils import join_structured_arrays

__all__ = ["WriterCallback"]

DEFAULT_OUTPUT = "{ckpt_dir}/{ckpt_stem}__test_{sample}.h5"
"""The v1-compatible output template (design §5.1, ``predictionwriter.py:164-171``)."""


class WriterCallback(Callback):
    """Assemble writer modules into the Lightning test loop (design §8).

    Parameters
    ----------
    modules : dict[str, Writer | None]
        Writer modules by instance name (the ``writers.modules`` config
        keys; ``None`` entries are dropped — design §5.3 null-deletion).
        Dict order is the per-group COLUMN order — ``base2.yaml`` ships
        ``inputs_copy -> tasks -> pad_mask`` to reproduce the v1 layout.
    output : str, optional
        Output path template; ``{ckpt_dir}``/``{ckpt_stem}`` come from
        ``trainer.ckpt_path`` and ``{sample}`` from the v1 test-file stem
        heuristic plus the datamodule's ``test_suff``
        (``predictionwriter.py:164-171``), by default `DEFAULT_OUTPUT`.
    half_precision : bool, optional
        Write float columns at f2 instead of f4 (the v1 flag), by default
        False.

    Raises
    ------
    ConfigError
        For an empty writer dict or a non-`Writer` entry.
    """

    def __init__(
        self,
        modules: dict[str, Writer | None],
        output: str = DEFAULT_OUTPUT,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        modules = {name: writer for name, writer in (modules or {}).items() if writer is not None}
        if not modules:
            raise ConfigError(
                "WriterCallback needs a non-empty writer dict — configure writers.modules "
                "(design §8; base2.yaml ships inputs_copy/tasks/pad_mask defaults)"
            )
        for name, writer in modules.items():
            if not isinstance(writer, Writer):
                raise ConfigError(
                    f"writers.modules.{name} ({type(writer).__name__}) is not a "
                    "salt.core.writers.Writer (design §2.7)"
                )
            writer.name = name
        self._writers: dict[str, Writer] = modules
        self.output = output
        self.half_precision = half_precision
        # per-test-run state (reset at on_test_start)
        self._h5: H5Writer | None = None
        self._rows_written = 0
        self._expected = 0
        self._group_of: dict[str, str] = {}
        self.output_path: Path | None = None

    @property
    def writers(self) -> dict[str, Writer]:
        """The assembled writer modules (read-only view).

        Returns
        -------
        dict[str, Writer]
            ``{instance name: writer}`` in config (column) order.
        """
        return dict(self._writers)

    # -- static demand (consumed by SaltModule, design §8) ----------------------

    def writer_demand(self, model_modules: dict[str, GraphModule], reader: Any) -> dict[str, str]:
        """Merge the writers' declared requires into one demand map.

        Called by `SaltModule` (duck-typed, before any plan compiles): the
        keys anchor the TEST plan sinks and extend the dataset-boundary
        demand; the values are §4.1-grade demander descriptions for error
        attribution.

        Parameters
        ----------
        model_modules : dict[str, GraphModule]
            The model-side module dict.
        reader : Any
            The configured reader prototype (``streams`` + ``groups`` with
            resolved ``vector`` flags — the `H5StructuredReader` surface).

        Returns
        -------
        dict[str, str]
            ``{dotted key: "writer 'name' (config: writers.modules.name)"}``
            in writer/declaration order.
        """
        out: dict[str, str] = {}
        for name, keys in self.per_writer_demand(model_modules, reader).items():
            for key in keys:
                out.setdefault(key, f"writer {name!r} (config: writers.modules.{name})")
        return out

    def per_writer_demand(
        self, model_modules: dict[str, GraphModule], reader: Any
    ) -> dict[str, list[str]]:
        """Each writer's declared requires, unmerged (static, config-only).

        The per-writer view behind `writer_demand`, also consumed by
        `salt.core.callbacks.GraphArtifacts` for the ``plan_test.txt``
        writer-sinks table — the answer to "which writer consumes
        ``preds.X``" that the merged demand map cannot give (M3-review fix;
        design §4.3, §8).

        Returns
        -------
        dict[str, list[str]]
            ``{writer instance name: [dotted keys]}`` in writer (config)
            order, each list in the writer's declaration order.
        """
        ctx = self._declare_ctx(model_modules, reader)
        return {name: list(writer.requires(ctx)) for name, writer in self._writers.items()}

    @staticmethod
    def _declare_ctx(model_modules: dict[str, GraphModule], reader: Any) -> WriterDeclareCtx:
        """Build the static declare context from the reader config.

        Returns
        -------
        WriterDeclareCtx
            Streams + sequence streams + model modules.

        Raises
        ------
        ConfigError
            When the reader does not expose the required surface.
        """
        streams = tuple(getattr(reader, "streams", ()) or ())
        groups = getattr(reader, "groups", None)
        if not streams or groups is None:
            raise ConfigError(
                f"writers need an H5StructuredReader-style reader exposing streams/groups — "
                f"got {type(reader).__name__} (design §8)"
            )
        sequence = tuple(s for s in streams if not getattr(groups[s], "vector", False))
        return WriterDeclareCtx(
            model_modules=model_modules, streams=streams, sequence_streams=sequence
        )

    # -- lightning hooks ---------------------------------------------------------

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Single-device assertion (multi-device test writing is out of scope, as v1).

        Raises
        ------
        ConfigError
            When testing on more than one device.
        """
        del pl_module
        if stage == "test" and trainer.world_size != 1:
            raise ConfigError(
                f"prediction writing requires a single device, got world_size="
                f"{trainer.world_size} — multi-device test writing is out of scope "
                "(design §8, v1 contract)"
            )

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Create the output file with the full schema BEFORE the first batch.

        A `ConfigError` propagates from `_write_ctx`/`_merge_columns` for a
        missing ``ckpt_path``, a foreign datamodule, an unknown output
        template key, or colliding writer columns.
        """
        ctx = self._write_ctx(trainer, pl_module)
        for writer in self._writers.values():
            writer.setup(ctx)
        dtypes, shapes = self._merge_columns(ctx)
        self._h5 = H5Writer(
            dst=ctx.output_path,
            dtypes=dtypes,
            shapes=shapes,
            shuffle=False,
            jets_name=next(iter(dtypes)),  # batch-sizing group (v1 jets_name semantics)
            precision="half" if self.half_precision else "full",
            num_jets=ctx.total,  # FIXED mode: valid empty file for an empty test set (§8)
        )
        self._rows_written = 0
        self._expected = ctx.total
        self.output_path = ctx.output_path

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Bundle,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect every writer's fragments for one batch and stream them.

        Raises
        ------
        ConfigError
            On a row-alignment break (``meta.rows`` vs the running counter)
            or a fragment with the wrong length.
        """
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        assert self._h5 is not None, "on_test_batch_end before on_test_start"
        rows_t = outputs.get("meta.rows")
        start, stop = int(rows_t[0]), int(rows_t[1])
        if start != self._rows_written:
            raise ConfigError(
                f"writer row alignment broke: batch rows [{start}, {stop}) but "
                f"{self._rows_written} rows written so far — sharded or uneven-batch test "
                "loaders are not supported (design §8 row-alignment contract)"
            )
        rows = slice(start, stop)
        fragments: dict[str, list[np.ndarray]] = {}
        for name, writer in self._writers.items():
            for stream, arr in writer.write(outputs, rows).items():
                if len(arr) != stop - start:
                    raise ConfigError(
                        f"writer {name!r} returned {len(arr)} rows for stream {stream!r}, "
                        f"expected {stop - start} (rows [{start}, {stop}))"
                    )
                fragments.setdefault(stream, []).append(arr)
        data = {
            self._group_of[stream]: (
                arrays[0] if len(arrays) == 1 else join_structured_arrays(arrays)
            )
            for stream, arrays in fragments.items()
        }
        self._h5.write(data)
        self._rows_written = stop

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Finalize the writers and close the sink."""
        del trainer, pl_module
        for writer in self._writers.values():
            writer.finalize()
        if self._h5 is None:
            return
        if self._rows_written == self._expected:
            self._h5.close()
        else:
            # a truncated loop (e.g. an interrupt) — close the raw handle;
            # H5Writer.close() would raise on the fixed-mode row mismatch
            print(
                f"WARNING: wrote {self._rows_written:,} of {self._expected:,} expected rows "
                f"to {self.output_path} — trailing rows are zero-filled"
            )
            self._h5.file.close()
        self._h5 = None
        print("-" * 100)
        print(f"Wrote eval file {self.output_path}")
        print("-" * 100)

    # -- setup plumbing -----------------------------------------------------------

    def _write_ctx(self, trainer: Trainer, pl_module: LightningModule) -> WriteCtx:
        """Assemble the per-run `WriteCtx` from the trainer state.

        Returns
        -------
        WriteCtx
            The run context handed to every writer's ``setup``.

        Raises
        ------
        ConfigError
            When ``ckpt_path`` is unset or the datamodule/reader lacks the
            required surface.
        """
        dm = getattr(trainer, "datamodule", None)
        dset = getattr(dm, "test_dset", None)
        if dset is None:
            raise ConfigError(
                "WriterCallback needs a GraphDataModule with a built test dataset — "
                f"got {type(dm).__name__} (design §8)"
            )
        reader = dset.reader
        declare = self._declare_ctx(self._model_modules(pl_module), reader)
        groups = reader.groups
        group_datasets = {stream: groups[stream].dataset for stream in declare.streams}
        source_path = Path(reader.source_path)
        seq_lengths: dict[str, int] = {}
        # reader-matching open flags (see InputCopyWriter.setup: HDF5 rejects
        # mixed SWMR flags on one file within a process)
        with h5py.File(source_path, "r", swmr=True, libver="latest") as f:
            for stream in declare.sequence_streams:
                seq_lengths[stream] = int(f[group_datasets[stream]].shape[1])
        total = self._expected_rows(trainer, len(dset), dm.batch_size)
        # bind-time field names (ResolvedSchema.fields) let custom writers
        # resolve feature columns by name (WriteCtx.feature_fields docstring)
        schema_fields = getattr(getattr(pl_module, "schema", None), "fields", None)
        return WriteCtx(
            output_path=self._output_path(trainer, dm, reader),
            total=total,
            run_name=getattr(pl_module, "name", "salt"),
            source_path=source_path,
            streams=declare.streams,
            sequence_streams=declare.sequence_streams,
            group_datasets=group_datasets,
            seq_lengths=seq_lengths,
            model_modules=declare.model_modules,
            batch_size=dm.batch_size,
            precision="half" if self.half_precision else "full",
            feature_fields={key: tuple(val) for key, val in (schema_fields or {}).items()},
        )

    @staticmethod
    def _model_modules(pl_module: LightningModule) -> dict[str, GraphModule]:
        """The model-side graph-module dict (duck-typed `SaltModule` surface).

        Returns
        -------
        dict[str, GraphModule]
            ``{instance name: module}``.

        Raises
        ------
        ConfigError
            When the LightningModule carries no graph-module dict.
        """
        modules = getattr(pl_module, "_graph_modules", None)
        if not modules:
            raise ConfigError(
                "WriterCallback needs a SaltModule-style LightningModule (graph-module "
                f"dict), got {type(pl_module).__name__} (design §8)"
            )
        return dict(modules)

    @staticmethod
    def _expected_rows(trainer: Trainer, total: int, batch_size: int) -> int:
        """Rows the (possibly ``limit_test_batches``-capped) loop will write.

        Returns
        -------
        int
            ``min(total, num_batches * batch_size)`` — with the sequential
            no-drop sampler only the LAST batch is partial, so a batch cap
            yields full batches.
        """
        num_batches = getattr(trainer, "num_test_batches", None)
        if not num_batches:
            return total
        limit = num_batches[0]
        if isinstance(limit, (int, float)) and math.isfinite(limit):
            return min(total, int(limit) * batch_size)
        return total

    def _output_path(self, trainer: Trainer, dm: Any, reader: Any) -> Path:
        """Render the output template (v1 ``predictionwriter.py:164-171``).

        Template keys: ``ckpt_dir``, ``ckpt_stem`` (from
        ``trainer.ckpt_path``) and ``sample`` — the v1 stem heuristic
        (``stem.split('_')[3]`` iff the test-file stem has exactly four
        underscore parts, else the whole stem) with the datamodule's
        ``test_suff`` appended as ``_{suffix}``.

        Returns
        -------
        Path
            The resolved output path.

        Raises
        ------
        ConfigError
            When ``ckpt_path`` is unset or the template names an unknown
            key.
        """
        ckpt_path = trainer.ckpt_path
        if ckpt_path is None:
            raise ConfigError(
                "prediction writing needs trainer.ckpt_path — run salt2 test with "
                "--ckpt_path <ckpt> (the output file is named after the checkpoint, "
                "v1 contract predictionwriter.py:164-171)"
            )
        stem = Path(getattr(reader, "filename", None) or reader.source_path).stem
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        test_suff = getattr(dm, "test_suff", None)
        if test_suff:
            sample = f"{sample}_{test_suff}"
        keys = {
            "ckpt_dir": str(Path(ckpt_path).parent),
            "ckpt_stem": Path(ckpt_path).stem,
            "sample": sample,
        }
        try:
            return Path(self.output.format(**keys))
        except KeyError as err:
            raise ConfigError(
                f"unknown writers.output template key {err} — available: "
                f"{sorted(keys)} (design §5.1)"
            ) from None

    def _merge_columns(
        self, ctx: WriteCtx
    ) -> tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]:
        """Merge per-writer column declarations into per-group dtypes/shapes.

        Column order = writer config order then each writer's own order —
        with the ``base2.yaml`` writer order this reproduces the v1 group
        layout (input copies, task columns, pad mask). Collisions are a
        `ConfigError` naming both writers (fixing the silent overwrite of
        ``array_utils.py:30-37``).

        Returns
        -------
        tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]
            ``(dtypes, shapes)`` keyed by H5 dataset name.

        Raises
        ------
        ConfigError
            On a column-name collision or an unknown stream.
        """
        descrs: dict[str, list] = {}
        owners: dict[tuple[str, str], str] = {}
        for name, writer in self._writers.items():
            for stream, dtype in writer.columns(ctx).items():
                if stream not in ctx.streams:
                    raise ConfigError(
                        f"writer {name!r} declared columns for unknown stream {stream!r} — "
                        f"reader streams are {list(ctx.streams)}"
                    )
                for descr in dtype.descr:
                    field = descr[0]
                    if (other := owners.get((stream, field))) is not None:
                        raise ConfigError(
                            f"column {field!r} in group {stream!r} is declared by writers "
                            f"{other!r} AND {name!r} — rename one output "
                            "(collision check, design §8)"
                        )
                    owners[stream, field] = name
                    descrs.setdefault(stream, []).append(descr)
        if not descrs:
            raise ConfigError(
                "the configured writers declare no output columns at all — check "
                "writers.modules (design §8)"
            )
        self._group_of = {stream: ctx.group_datasets[stream] for stream in descrs}
        dtypes = {self._group_of[stream]: np.dtype(descr) for stream, descr in descrs.items()}
        shapes = {
            self._group_of[stream]: (
                (ctx.total, ctx.seq_lengths[stream]) if stream in ctx.seq_lengths else (ctx.total,)
            )
            for stream in descrs
        }
        return dtypes, shapes
