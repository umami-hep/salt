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
- **ONNX manifest** (M4.5 unified manifest): `onnx_manifest` assembles the
  writers' declared `ExportOutput` entries into THE export-output manifest
  (writer config order; flat-namespace collision check naming both
  writers) — its ports are the ONNX plan sinks (one demand mechanism in
  both output modes, amendment §4) and the exporter's output list. Writers
  are READ here, never run;
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag.hdf5 import H5Writer
from lightning import Callback, LightningModule, Trainer

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError, KindError, ShapeError
from salt.core.graph.spec import GraphModule, Mode, TensorSpec, flatten_spec
from salt.core.onnx.config import ExportOutput
from salt.core.writers.base import WriteCtx, Writer, WriterDeclareCtx
from salt.core.utils.array_utils import join_structured_arrays

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
                "(design §8). plan 34 W34.4c: base2.yaml no longer ships a writers default; "
                "configs supply their own writers.modules OR a top-level outputs: section "
                "with dumb sinks instead"
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
            resolved ``global_object`` flags — the `H5StructuredReader` surface).

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
        self._validate_writer_roles(ctx)
        return {name: list(writer.requires(ctx)) for name, writer in self._writers.items()}

    def validate_specs(
        self,
        model_modules: dict[str, GraphModule],
        reader: Any,
        producer_specs: Mapping[str, TensorSpec],
    ) -> None:
        """Prove every writer's TEST inputs exist AND kind/dtype-unify, before the first batch.

        The static validator design §2.7 / §8 promises: "the static validator
        proves the writer's inputs exist before a single batch is read". Until
        M5 the `Writer.requires` `TensorSpec` VALUES drove demand by KEY only
        (the `salt.core.writers.base` deviation note) — key serveability was
        enforced (a model-produced key anchors the TEST plan; an unserveable
        dataset-namespace key raises at `SaltModule._boundary_demand`), but the
        kind/dtype each writer DECLARES for a consumed key was never unified
        against the port that actually produces it. A writer demanding
        ``preds.jets.classification`` as ``kind="label"`` (or ``dtype="int64"``
        where the task publishes ``float32``) was silently accepted: the demand
        map only looked at the dotted key, so a genuinely wrong consumer
        contract surfaced — if at all — as a confusing first-batch shape/dtype
        error inside ``write()``, not a config error.

        This closes the gap with the planner's own edge unification
        (`salt.core.graph.planner._bind_edges` / `_unify_edge`): for each
        writer-declared require with a non-optional, TEST-active spec, look up
        the producing leaf (a model-produced TEST port OR a dataset-boundary
        source — `producer_specs` is their union) and check

        - **existence** — a non-optional require with no producer is the §2.7
          "input does not exist" failure (normally pre-empted by the demand
          machinery; covered here so the validator's contract is honest
          standalone);
        - **kind** — consumer kind must equal producer kind (`KindError`,
          design §2.2 — the mask-polarity / label-vs-feature guard);
        - **dtype** — when BOTH sides declare a dtype, they must match
          (`ShapeError`, the planner's `_unify_edge` rule). A ``None`` on
          either side is "unconstrained" and unifies with anything (the
          `TaskWriter`'s deliberately unconstrained ``preds.*`` requires, which
          accept whatever dtype a task family publishes, stay legal).

        Shape unification is deliberately NOT replayed here: writer requires
        carry symbolic-dim shapes (``("B", ...)``) whose batch/token dims only
        bind against the live boundary inside the compiled plan, and the plan's
        own `_unify_edge` already unifies model-side shapes end to end. The
        writer-specific gap was kind/dtype on the writer→producer edge, which
        the plan never builds (writers are sinks demanded by KEY).

        Called from TWO sites: `SaltModule.setup` on the TEST path at run
        setup, AND `salt2 graph validate` for the TEST mode (`cli._cmd_validate`,
        the canonical CI static-validator command) data-free — both right after
        the TEST plan compiled (so `producer_specs` — model-produced ports plus
        the dataset boundary — is available) and before bind/the first batch.
        The data-free caller assembles `producer_specs` from the compiled TEST
        plan's steps + sources (the static equivalent of the runtime union of
        `model_producer_specs` and `GraphDataset.boundary_specs`).

        Parameters
        ----------
        model_modules : dict[str, GraphModule]
            The model-side module dict.
        reader : Any
            The configured reader prototype (`_declare_ctx` surface).
        producer_specs : Mapping[str, TensorSpec]
            Flat ``{dotted key: spec}`` of every TEST producer: the model
            modules' TEST-active produced ports unioned with the dataset
            boundary's served leaves (``masks``/``labels``/``meta``/``inputs``).
            Model-produced keys take precedence over a same-named boundary key
            (they are the executed leaf).

        Raises
        ------
        KindError
            A writer declares a consumed key with a kind differing from its
            producer (design §2.2).
        ShapeError
            A writer declares a consumed key with a dtype differing from its
            producer's declared dtype (both non-``None``; the `_unify_edge`
            rule).
        ConfigError
            A non-optional writer require has no TEST producer at all
            (existence half of the §2.7 contract).
        """
        ctx = self._declare_ctx(model_modules, reader)
        for name, writer in self._writers.items():
            where = f"writer {name!r} (config: writers.modules.{name})"
            for key, cspec in writer.requires(ctx).items():
                if not cspec.active_in(Mode.TEST):
                    continue  # not a TEST input (e.g. an ONNX-only require)
                pspec = producer_specs.get(key)
                if pspec is None:
                    if cspec.optional:
                        continue
                    raise ConfigError(
                        f"[mode=TEST] {where} declares require {key!r} but no model module "
                        "or dataset source produces it — the writer's input does not exist "
                        "(static-validator contract, design §2.7/§8).\n  fix: correct the "
                        f"writer's requires, or add a module/source producing {key!r}"
                    )
                if pspec.kind != cspec.kind:
                    raise KindError(
                        f"[mode=TEST] {where} declares require {key!r} with kind="
                        f"{cspec.kind!r}, but its producer provides kind={pspec.kind!r} "
                        "(design §2.2; writer requires are kind-typed exactly like a "
                        "module's declare_io)"
                    )
                if (
                    pspec.dtype is not None
                    and cspec.dtype is not None
                    and pspec.dtype != cspec.dtype
                ):
                    raise ShapeError(
                        f"[mode=TEST] dtype mismatch on {key!r}: {where} declares require "
                        f"dtype={cspec.dtype!r}, but its producer declares {pspec.dtype!r} "
                        "(the planner's _unify_edge rule extended to writer sinks, design "
                        "§2.7)"
                    )

    @staticmethod
    def model_producer_specs(
        model_modules: Mapping[str, GraphModule],
    ) -> dict[str, TensorSpec]:
        """The model modules' TEST-active produced leaves, in declaration order.

        Helper for `SaltModule.setup` to assemble the `validate_specs`
        ``producer_specs`` argument (model side); the dataset half comes from
        the TEST `GraphDataset.boundary_specs()`. First producer wins on a
        duplicate key (declaration order), mirroring
        `SaltModule._model_sinks`'s ``setdefault``.

        Returns
        -------
        dict[str, TensorSpec]
            ``{dotted key: spec}`` for every TEST-active produced port.
        """
        out: dict[str, TensorSpec] = {}
        for module in model_modules.values():
            for key, spec in flatten_spec(module.declare_io(Mode.TEST).produces).items():
                if spec.active_in(Mode.TEST):
                    out.setdefault(key, spec)
        return out

    # -- the ONNX-manifest assembly (M4.5 amendment §4) --------------------------

    def per_writer_onnx_manifest(
        self, model_modules: dict[str, GraphModule], reader: Any
    ) -> dict[str, list[ExportOutput]]:
        """Each writer's declared ONNX-manifest entries, unmerged (static).

        The per-writer view behind `onnx_manifest` — the sibling of
        `per_writer_demand` for the ONNX role (amendment §4): consumed for
        sink-origin attribution (`salt.core.cli`) and the manifest
        annotation (``salt2 graph resolve``).

        Returns
        -------
        dict[str, list[ExportOutput]]
            ``{writer instance name: entries}`` in writer (config) order,
            each list in the writer's declaration order.
        """
        ctx = self._declare_ctx(model_modules, reader)
        self._validate_writer_roles(ctx)
        return {name: list(writer.onnx_outputs(ctx)) for name, writer in self._writers.items()}

    def onnx_manifest(
        self, model_modules: dict[str, GraphModule], reader: Any
    ) -> list[ExportOutput]:
        """THE assembled export-output manifest (M4.5 unified manifest).

        Manifest order = writer config order, each writer's entries in its
        own declaration order (`TaskWriter` emits global-stream entries
        before sequence-stream entries — the v1 output order, amendment §5
        rule 5). The assembled manifest is a FLAT ONNX namespace: a suffix
        declared by two entries is a hard error naming both owning writers
        and ports, with ``onnx_names:`` as the fix (amendment §5 rule 4 —
        previously unrepresentable when ``export.outputs`` was
        hand-deduplicated by the author). Port duplicates (two writers
        exporting one port) are equally rejected.

        Returns
        -------
        list[ExportOutput]
            The flat manifest; its ports are the ONNX plan sinks and its
            entries the exporter's output list.

        Raises
        ------
        ConfigError
            On a suffix or port collision across writers.
        """
        per_writer = self.per_writer_onnx_manifest(model_modules, reader)
        suffix_owner: dict[str, tuple[str, str]] = {}
        port_owner: dict[str, str] = {}
        manifest: list[ExportOutput] = []
        for writer_name, entries in per_writer.items():
            for entry in entries:
                if (other := port_owner.get(entry.port)) is not None:
                    raise ConfigError(
                        f"ONNX manifest port {entry.port!r} is declared by writers "
                        f"{other!r} AND {writer_name!r} (config: writers.modules.*) — one "
                        "writer owns one export port (M4.5 amendment §4)"
                    )
                port_owner[entry.port] = writer_name
                for raw_suffix in entry.names if entry.names is not None else [entry.name]:
                    suffix = str(raw_suffix)
                    if (owner := suffix_owner.get(suffix)) is not None:
                        other_writer, other_port = owner
                        raise ConfigError(
                            f"ONNX output suffix {suffix!r} is declared TWICE in the flat "
                            f"export namespace: by writer {other_writer!r} (port "
                            f"{other_port!r}) and writer {writer_name!r} (port "
                            f"{entry.port!r}) — distinct H5 groups may share column names, "
                            "but ONNX outputs cannot; rename one side via "
                            f"writers.modules.{writer_name}.init_args.onnx_names (or "
                            "export.rename) — amendment §5 rule 4"
                        )
                    suffix_owner[suffix] = (writer_name, entry.port)
                manifest.append(entry)
        return manifest

    def column_manifests(
        self, model_modules: dict[str, GraphModule], reader: Any, run_name: str
    ) -> dict[str, dict[str, list[str]]]:
        """Each writer's statically-derivable eval columns (§4.4 annotation).

        Returns
        -------
        dict[str, dict[str, list[str]]]
            ``{writer instance name: {stream: [column names]}}`` in writer
            order — empty inner dicts mean "file-dependent or no static
            columns" (`Writer.column_manifest` default).
        """
        ctx = self._declare_ctx(model_modules, reader)
        return {
            name: dict(writer.column_manifest(ctx, run_name))
            for name, writer in self._writers.items()
        }

    def _validate_writer_roles(self, ctx: WriterDeclareCtx) -> None:
        """Reject writers with no role / the unblessed export-only stub shape.

        Design principle 10 extended to writers (amendment §4 item 3): a
        configured writer must do SOMETHING — an empty TEST demand
        (`requires`) and an empty ONNX manifest (`onnx_outputs`) together
        are a `ConfigError`. The export-only shape (no TEST demand,
        non-empty manifest) is legal ONLY behind the explicit
        ``export_only`` flag — `ExportOnlyWriter` is the blessed spelling
        (merge condition 3); and an ``export_only`` writer declaring TEST
        demand contradicts itself.

        Note (M5 decision point — SETTLED in sub-wave A; AM dec 4): this
        validation derives every writer's ONNX manifest on the TEST path too
        (`per_writer_demand` calls it during ``salt2 test`` demand
        assembly), so a task family with a TEST representation but NO export
        representation would raise `TaskWriter.onnx_outputs`'s
        unsupported-family error during eval unless narrowed away
        (``onnx: false``/``onnx_streams``/``onnx_tasks`` — the error names
        all three). AM dec 4 left two options for the M5 ``TaskWriter``
        regression extension: keep the loud coupling, or derive only
        export-representable families. **Decision: KEEP the loud coupling —
        it is the correct guard for a genuinely unrepresentable family — and
        make scalar regression representable in BOTH modes so it never
        reaches the raise.** Global scalar regression is already
        export-representable in v1 (`RegressionTask.output_names` +
        ``get_onnx``; AM pre-implementation check), so it joins
        classification + vertexing as a supported `TaskWriter` family rather
        than tripping the error. A regression config therefore does NOT
        crash ``salt2 test`` demand assembly (non-empty TEST demand AND
        non-empty manifest); an author who declines to export a
        TEST-representable regression task uses explicit ONNX narrowing
        (``onnx_outputs`` returns ``[]`` before the family dispatch, so the
        eval-only shape stays legal). Full rationale + the A2 consequence:
        ``salt/core/README.md`` (M4.5 amendment addendum). The loud raises
        (``salt/core/nn/tasks.py`` ``_TaskModuleBase.output_names`` /
        ``get_h5`` / ``onnx_outputs`` — the per-family rendering now lives on
        the task, not the writer) stay.

        Raises
        ------
        ConfigError
            Naming the writer and the violated rule.
        """
        for name, writer in self._writers.items():
            demand = writer.requires(ctx)
            manifest = writer.onnx_outputs(ctx)
            where = f"writer {name!r} (config: writers.modules.{name})"
            if writer.export_only:
                if demand:
                    raise ConfigError(
                        f"{where} sets export_only=True but declares TEST requires "
                        f"{sorted(demand)} — export-only writers have no TEST role "
                        "(ExportOnlyWriter contract, M4.5 amendment merge condition 3)"
                    )
                if not manifest:
                    raise ConfigError(
                        f"{where} sets export_only=True but declares no onnx_outputs — "
                        "an export-only writer's single role is its manifest "
                        "(M4.5 amendment merge condition 3)"
                    )
                continue
            if not demand and manifest:
                raise ConfigError(
                    f"{where} declares ONNX outputs but no TEST requires — this is the "
                    "export-only stub shape, which must be EXPLICIT: subclass "
                    "salt.core.writers.ExportOnlyWriter (or set export_only=True) "
                    "(M4.5 amendment merge condition 3)"
                )
            if not demand and not manifest:
                raise ConfigError(
                    f"{where} declares neither TEST requires nor ONNX outputs — a "
                    "configured writer must do something in at least one mode; remove it "
                    f"(writers.modules.{name}: null) or fix its declarations "
                    "(design principle 10, M4.5 amendment §4)"
                )

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
        sequence = tuple(s for s in streams if not getattr(groups[s], "global_object", False))
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
        ``array_utils.py:30-37``). This is also the RUNTIME half of the
        writer-role validation (`_validate_writer_roles` is the static
        half): a non-``export_only`` writer producing no columns raises —
        the merge-condition-3 stub error when it declares ONNX outputs,
        the principle-10 does-nothing error otherwise.

        Returns
        -------
        tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]
            ``(dtypes, shapes)`` keyed by H5 dataset name.

        Raises
        ------
        ConfigError
            On a column-name collision, an unknown stream, or a columnless
            non-export-only writer (stub / does-nothing shapes).
        """
        descrs: dict[str, list] = {}
        owners: dict[tuple[str, str], str] = {}
        declare = WriterDeclareCtx(
            model_modules=dict(ctx.model_modules),
            streams=ctx.streams,
            sequence_streams=ctx.sequence_streams,
        )
        # writer-declared OUTPUT groups that are not reader streams (e.g. the
        # MaskFormer object writer's `objects`/`object_masks` groups, design §8):
        # group -> trailing per-row shape, with the declaring writer for attribution.
        extra_shapes, extra_owner = self._collect_extra_groups(ctx)
        for name, writer in self._writers.items():
            columns = writer.columns(ctx)
            if not columns:
                if writer.export_only:
                    continue  # the blessed export-only pattern: no TEST columns by design
                # the runtime backstop behind the static role check —
                # columns() can be file-dependent, so "does nothing in
                # TEST" is only fully decidable here (design principle 10,
                # M4.5 amendment §4). The unblessed export-only stub shape
                # (no columns, non-empty manifest, no flag) gets the
                # merge-condition-3 error here too — never a silent
                # zero-contribution fall-through
                if writer.onnx_outputs(declare):
                    raise ConfigError(
                        f"writer {name!r} (config: writers.modules.{name}) declares ONNX "
                        "outputs but produced no TEST columns — this is the export-only "
                        "stub shape, which must be EXPLICIT: subclass "
                        "salt.core.writers.ExportOnlyWriter (or set export_only=True) "
                        "(M4.5 amendment merge condition 3)"
                    )
                raise ConfigError(
                    f"writer {name!r} (config: writers.modules.{name}) declares no "
                    "output columns AND no ONNX outputs — a configured writer must do "
                    "something in at least one mode (design principle 10, M4.5 "
                    "amendment §4)"
                )
            for stream, dtype in columns.items():
                if stream not in ctx.streams and stream not in extra_shapes:
                    raise ConfigError(
                        f"writer {name!r} declared columns for unknown group {stream!r} — "
                        f"reader streams are {list(ctx.streams)} and writer-declared extra "
                        f"groups are {sorted(extra_shapes)} (declare a non-reader output group "
                        "in Writer.extra_groups, design §8)"
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
        # reader streams map to their FILE dataset name (v1 group naming); writer-
        # declared extra groups (object outputs) use the group name as the dataset.
        self._group_of = {stream: ctx.group_datasets.get(stream, stream) for stream in descrs}
        del extra_owner  # attribution is consumed inside _collect_extra_groups' checks
        dtypes = {self._group_of[stream]: np.dtype(descr) for stream, descr in descrs.items()}
        shapes = {
            self._group_of[stream]: self._group_shape(stream, ctx, extra_shapes)
            for stream in descrs
        }
        return dtypes, shapes

    def _collect_extra_groups(
        self, ctx: WriteCtx
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
        """Merge the writers' `extra_groups` declarations (non-reader output groups).

        Each writer declaring a non-reader output group (the MaskFormer object
        writer's ``objects``/``object_masks``, design §8) sizes it via
        `Writer.extra_groups`; this merges those declarations with a uniqueness
        check (two writers cannot both own the same new group) and rejects an
        extra group that shadows a reader stream (collision-proof H5 dataset
        naming).

        Returns
        -------
        tuple[dict[str, tuple[int, ...]], dict[str, str]]
            ``(group -> trailing per-row shape, group -> declaring writer)``.

        Raises
        ------
        ConfigError
            On a duplicate extra group or one shadowing a reader stream.
        """
        shapes: dict[str, tuple[int, ...]] = {}
        owner: dict[str, str] = {}
        for name, writer in self._writers.items():
            for group, trailing in writer.extra_groups(ctx).items():
                if group in ctx.streams:
                    raise ConfigError(
                        f"writer {name!r} (config: writers.modules.{name}) declares extra group "
                        f"{group!r}, which shadows reader stream {group!r} — extra groups are "
                        "NON-reader output groups only (design §8)"
                    )
                if (other := owner.get(group)) is not None:
                    raise ConfigError(
                        f"extra output group {group!r} is declared by writers {other!r} AND "
                        f"{name!r} — one writer owns one extra group (design §8)"
                    )
                owner[group] = name
                shapes[group] = tuple(int(d) for d in trailing)
        return shapes, owner

    @staticmethod
    def _group_shape(
        stream: str, ctx: WriteCtx, extra_shapes: Mapping[str, tuple[int, ...]]
    ) -> tuple[int, ...]:
        """The fixed-mode H5 shape of one output group (leading ``total`` row dim).

        Reader sequence streams carry ``(total, file_seq_len)``, reader global
        streams ``(total,)`` (v1 group geometry); a writer-declared extra group
        carries ``(total, *trailing)`` from its `Writer.extra_groups` shape.

        Returns
        -------
        tuple[int, ...]
            The full H5 dataset shape.
        """
        if stream in extra_shapes:
            return (ctx.total, *extra_shapes[stream])
        if stream in ctx.seq_lengths:
            return (ctx.total, ctx.seq_lengths[stream])
        return (ctx.total,)
