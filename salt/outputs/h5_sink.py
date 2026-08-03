"""The H5 sink — `H5OutputSink`, the eval-H5 serialiser.

Keeps the deprecated one-window `H5OutputWriter` legacy alias export.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag.hdf5 import H5Writer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import (
    IO,
    KEY_SEP,
    Mode,
    TensorSpec,
    flatten_spec,
    sym_dim,
    unflatten_spec,
)
from salt.outputs.output_schema import ObjectGroup, ObjectGroupField, OutputColumn
from salt.outputs.sink import OutputSink, RuntimeSink, SinkContext
from salt.utils.array_utils import join_structured_arrays

_SinkCallback = OutputSink
"""Deprecated private alias for `OutputSink` (the sink base was made public)."""

DEFAULT_OUTPUT = "{ckpt_dir}/{ckpt_stem}__test_{sample}.h5"
"""The v1-compatible output template."""


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad a per-token array along axis 1 to the file sequence length.

    Positions beyond the model's (possibly truncated) sequence read as zeros
    (so a truncated-away ``mask`` position reads ``mask=False``).
    """
    if arr.ndim < 2 or arr.shape[1] >= length:
        return arr
    out = np.zeros((arr.shape[0], length, *arr.shape[2:]), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


class H5OutputSink(RuntimeSink):
    """The H5 sink: a terminal graph node serialising ``outputs.*`` to the eval H5.

    A `GraphModule` terminal node whose ``declare_io`` requires the demanded
    ``outputs.*`` leaves (+ ``meta.rows`` + each pad-mask stream) in TEST and
    produces nothing — so the planner keeps it in TEST (rendering its own
    card) while FIT/VAL/ONNX prune it.

    It accumulates the named ``outputs.*`` producer leaves per batch and
    writes them through one ftag `H5Writer` (FIXED mode, ``num_jets`` known
    up front — a valid empty file for an empty test set).

    Responsibilities the sink owns (the producers must NOT know about these):

    - **structured-array packing**: each producer leaf is a plain ``[B, C]``
      / ``[B, L, C]`` tensor; the sink ``u2s``-packs it into the declared
      ``{run_name}_{suffix}`` columns (`OutputColumn`).
    - **per-token pad re-expansion**: per-token leaves and the pad-mask
      column are zero-padded to the FILE sequence length (``_pad_to``).
    - **input-variable copies**: optional source-file columns re-read by
      absolute rows (``meta.rows``) through one cached handle, with SOURCE
      dtypes/order.
    - **pad-mask columns**: an optional boolean ``mask`` column per sequence
      stream (True = padded); truncated-away positions read ``mask=False``.
    - **output-group naming**: reader streams map to their FILE dataset name.

    `writer_demand` returns the consumed ``outputs.*`` leaves PLUS the source
    ``preds.*`` leaf of each producer feeding them (so the TEST dead-preds
    gate sees those predictions as consumed, since preds.* are consumed by
    producers, not by the sink), the ``meta.rows`` row anchor, and each
    pad-mask stream's mask.

    Note: keeping input copies / masks / pad re-expansion here means this
    sink carries extra DataModule/reader/source-file coupling (it reads the
    file's sequence length from ``reader.source_path``). The win is
    composable additional sinks, not decoupling of the H5 sink itself.

    Parameters
    ----------
    outputs : None
        RETIRED as a config surface (plan 50 Phase B). The H5 sink is now
        implicit — the ``salt test`` command wires it and derives its column
        schema from the bound top-level ``outputs:`` section (``RunTaskOutput``
        + ``InputCopyWriter`` + ``PadMaskWriter``). Only ``None``/``[]`` is
        accepted; any truthy value raises `ConfigError`. Do NOT wire this sink
        in ``callbacks:`` with an explicit ``OutputColumn`` table — declare the
        section instead (see ``gn2v2-opendata.yaml``).
    copy_inputs : Mapping[str, Sequence[str]] | None, optional
        Per-stream source-file variables to copy into the eval H5, in column
        order, by default None.
    write_pad_mask : bool | Sequence[str], optional
        Boolean ``mask`` column per sequence stream: ``True`` writes one for
        every demanded output's sequence stream; a list names the streams;
        by default False.
    output : str, optional
        Output path template (``{ckpt_dir}``/``{ckpt_stem}``/``{sample}``),
        by default `DEFAULT_OUTPUT`.
    half_precision : bool, optional
        Write float columns at f2 instead of f4, by default False.
    object_groups : Sequence[ObjectGroup] | None, optional
        Declarative structured output groups fed from bundle leaves — each an
        `ObjectGroup` naming a NON-reader group (with a trailing `shape`, e.g.
        the MaskFormer ``objects`` ``(M,)`` / ``object_masks`` ``(M, T)``
        groups) or extra per-token columns on an existing reader stream (e.g.
        the ``tracks`` ``HadronIndex`` column). Each field sources one bundle
        leaf, which the sink demands (keeping its producer alive) and packs.
        Generic: the sink has no per-consumer knowledge. Empty/None (the
        default) makes the mechanism a strict no-op (byte-identical schema).
    modes : Sequence[str] | None, optional
        Which planner modes to run in. `allowed_modes` is ``[test]``, so
        ``[test]`` is the only accepted list and omitting it (the default)
        means the same thing.

    Raises
    ------
    ConfigError
        For any truthy ``outputs`` value (the retired explicit-table surface),
        or (at run setup) a column-name collision / unknown stream / missing
        source variable.
    """

    name = "h5_output"
    """The graph-node instance name (overridable by the config dict key)."""

    def __init__(
        self,
        outputs: Sequence[OutputColumn | Mapping[str, Any]] | None = None,
        copy_inputs: Mapping[str, Sequence[str]] | None = None,
        write_pad_mask: bool | Sequence[str] = False,
        output: str = DEFAULT_OUTPUT,
        half_precision: bool = False,
        object_groups: Sequence[ObjectGroup | Mapping[str, Any]] | None = None,
        modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(modes=modes)
        # plan 50 Phase B: the explicit OutputColumn table is RETIRED as a config
        # surface — the H5 sink is now implicit (the command wires it) and derives
        # its column schema from the bound outputs: section (RunTaskOutput +
        # InputCopyWriter + PadMaskWriter). An explicit `outputs:` table is a hard
        # error pointing at the section mechanism. `outputs=None`/`[]` is the only
        # accepted value (the injected/dumb-section path).
        if outputs:
            raise ConfigError(
                "H5OutputSink no longer accepts an explicit `outputs:` OutputColumn table "
                "(plan 50 Phase B) — the H5 sink is implicit (the `salt test` command wires "
                "it) and derives its columns from the top-level `outputs:` section "
                "(RunTaskOutput + InputCopyWriter + PadMaskWriter). Declare the section, per "
                "`gn2v2-opendata.yaml`; use each RunTaskOutput's `modes:` list to control "
                "test-vs-export participation. Do NOT wire H5OutputSink in `callbacks:` at all."
            )
        # no explicit columns — resolved lazily from the bound outputs: section.
        self._explicit_columns: tuple[OutputColumn, ...] = ()
        self._columns: tuple[OutputColumn, ...] = ()
        self._columns_resolved = False
        self.copy_inputs = {s: list(v) for s, v in (copy_inputs or {}).items()}
        self.write_pad_mask = write_pad_mask
        self.output = output
        self.half_precision = half_precision
        # declarative structured output groups fed from bundle leaves. Empty by
        # default — the mechanism is then a strict no-op and the H5 schema is
        # byte-identical to a plain sink. Self-contained (no section-node lookup):
        # each field sources a bundle leaf the sink demands + packs.
        self._object_groups: tuple[ObjectGroup, ...] = tuple(
            ObjectGroup.coerce(g) for g in (object_groups or ())
        )
        # DUMB-SECTION mode: when an `outputs:` section is bound (RunTaskOutput +
        # InputCopyWriter + PadMaskWriter), the sink dumps ALL active outputs.*
        # leaves and derives its column schema + copy spec + mask streams from the
        # SECTION manifest in section declaration order. The explicit constructor
        # args are the OVERRIDE used when no section is bound. One of the two MUST
        # resolve at run setup.
        self._output_section: Mapping[str, Any] | None = None
        # which section selection the columns resolve from: Mode.TEST (the
        # `salt test` eval schema, the default) or Mode.ONNX (`salt inference`
        # writes STRICTLY the export output set — plan 50 Phase D, via
        # `use_export_selection`).
        self._section_mode: Mode = Mode.TEST
        self._section_columns: tuple[tuple[str, OutputColumn], ...] | None = None
        # dumb-section copy resolution: None until a section binds; True means
        # "copy every stream with a configured task" (v1 default, resolved at
        # open_schema against the reader).
        self._copy_all_tasked_streams: bool = False
        self._copy_variables: dict[str, list[str]] = {}
        # per-test-run state (reset at open_schema)
        self._h5: H5Writer | None = None
        self._rows_written = 0
        self._expected = 0
        self._run_name = "salt"
        self._group_of: dict[str, str] = {}
        self._seq_lengths: dict[str, int] = {}
        # non-reader object-group -> trailing per-row shape, resolved at
        # open_schema from the configured object_groups ({} when none).
        self._object_shapes: dict[str, tuple[int, ...]] = {}
        self._mask_streams: tuple[str, ...] = ()
        self._copy_handle: h5py.File | None = None
        self._copy_reads: dict[str, tuple[h5py.Dataset, list[str]]] = {}
        self.output_path: Path | None = None

    @property
    def columns(self) -> tuple[OutputColumn, ...]:
        """The configured output columns, in declaration (H5 column) order."""
        return self._columns

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` leaf keys, in declaration order."""
        return tuple(col.key for col in self._columns)

    # -- column resolution ------------------------------------------------------

    def is_test_sink(self) -> bool:
        """Always True (unlike the base probe): always demands ``meta.rows``, so this
        is resolution-free even before a dumb-section binds its columns.
        """
        return True

    # -- dumb-section binding -------------------------------------

    def use_export_selection(self) -> None:
        """Switch the section-derived selection to the EXPORT (``Mode.ONNX``) set.

        Plan 50 Phase D — ``salt inference`` writes STRICTLY the export
        output set to H5: columns resolve from ``manifest_fields(Mode.ONNX)``
        (one single-suffix column per ONNX leaf, named by the field's resolved
        ONNX name), and the section's copy/mask writers contribute only when
        their ``modes:`` include ``export``. Call BEFORE `bind_output_section`.
        """
        self._section_mode = Mode.ONNX
        self._columns_resolved = False

    def bind_output_section(self, section: Mapping[str, Any]) -> None:
        """Capture the ``outputs:`` section so the dumb sink dumps its leaves.

        The section is the ORDERED dict of section writers (`RunTaskOutput`,
        `InputCopyWriter`, `PadMaskWriter`). When bound, the sink switches to
        the DUMB path: it dumps ALL active ``outputs.*`` leaves and derives
        its column schema + input-copy spec + pad-mask streams from the
        SECTION manifest in SECTION DECLARATION ORDER (the H5 column-order
        authority) — the constructor knobs are ignored. The section is bound
        by the planner/SaltModule after the model. A copy/mask writer whose
        ``modes:`` list excludes this sink's selection mode contributes
        nothing (an export-only copy writer never adds ``salt test`` columns,
        and a test-only one never adds ``salt inference`` columns).
        """
        self._output_section = section
        # the section drives copy_inputs + write_pad_mask too (override the ctor
        # knobs in dumb mode): collect from the InputCopyWriter / PadMaskWriter
        # that RUN in this sink's selection mode.
        copy_inputs: dict[str, list[str]] = {}
        mask_streams: list[str] = []
        for writer in section.values():
            runs_in_mode = getattr(writer, "runs_in_mode", None)
            if callable(runs_in_mode) and not writer.runs_in_mode(self._section_mode):
                continue
            if callable(getattr(writer, "copy_spec", None)):
                spec = writer.copy_spec()
                streams = spec.get("streams")
                variables = spec.get("variables") or {}
                # streams None -> the v1 default (every stream with a configured
                # task), resolved at open_schema against the reader; encode that as
                # the sentinel {} (empty fields = all source fields) per stream we
                # learn at open time. We stash the spec and resolve in open_schema.
                if streams is None:
                    self._copy_all_tasked_streams = True
                    self._copy_variables = variables
                else:
                    self._copy_all_tasked_streams = False
                    for s in streams:
                        copy_inputs[s] = list(variables.get(s, []))
                    self._copy_variables = variables
            if callable(getattr(writer, "mask_streams", None)):
                mask_streams.extend(writer.mask_streams())
        if not getattr(self, "_copy_all_tasked_streams", False):
            self.copy_inputs = copy_inputs
        self.write_pad_mask = tuple(dict.fromkeys(mask_streams)) if mask_streams else False

    def _is_dumb_section(self) -> bool:
        """Whether an ``outputs:`` section is bound (the dumb-section path is active)."""
        return bool(self._output_section)

    def _run_task_outputs(self) -> list[Any]:
        """The bound section's `RunTaskOutput` writers, in section declaration order."""
        if not self._output_section:
            return []
        return [
            w
            for w in self._output_section.values()
            if callable(getattr(w, "is_run_task_output", None)) and w.is_run_task_output()
        ]

    def _resolve_columns(self, run_name: str) -> tuple[OutputColumn, ...]:
        """Resolve the H5 column table — explicit, or from the bound ``outputs:`` section.

        In explicit-``outputs`` mode returns the configured columns unchanged.
        With a bound dumb ``outputs:`` section the schema comes from the
        section's ``RunTaskOutput.manifest_fields`` in SECTION DECLARATION
        ORDER. With neither, a sink that has no way to know its columns is a
        config error.

        Raises
        ------
        ConfigError
            When neither explicit columns nor an ``outputs:`` section is
            configured.
        """
        if self._columns_resolved:
            return self._columns
        if self._is_dumb_section():
            return self._resolve_section_columns(run_name)
        raise ConfigError(
            "H5OutputSink has no columns to write — give it an explicit `outputs:` "
            "OutputColumn table, or compose a top-level `outputs:` section "
            "(RunTaskOutput + InputCopyWriter + PadMaskWriter) that binds to it "
            "(plan 34 W34.2)"
        )

    def _column_suffix(self, field: Any) -> str | None:
        """The field's column suffix under this sink's selection mode, or None.

        TEST keeps fields with an ``h5_name`` (the eval schema); the export
        selection (``Mode.ONNX``) keeps fields with a resolved ONNX name — the
        EXACT selection rule the `OnnxExportSink` tuple uses, so the inference
        H5 columns are 1:1 with the ONNX tuple by construction.
        """  # noqa: DOC201 - private helper, no Returns block per docstring policy
        if self._section_mode is Mode.TEST:
            return field.h5_name
        return field.resolved_onnx_name

    def _resolve_section_columns(self, run_name: str) -> tuple[OutputColumn, ...]:
        """Resolve H5 columns from the bound ``outputs:`` section manifest.

        Walks the section's `RunTaskOutput` writers'
        ``manifest_fields(<selection mode>)`` (value-free `OutputField`
        metadata, ``Mode.TEST`` unless `use_export_selection` switched to
        ``Mode.ONNX``) in SECTION DECLARATION ORDER, keeps the FINAL fields the
        selection names (`_column_suffix`), and assembles ONE `OutputColumn`
        per ``outputs.*`` leaf (suffixes in field order; export-mode leaves are
        single-suffix by construction). The SECTION field order is the H5
        column order authority (not executor topo order). Caches the resolved
        table (run-name-stable). The section's InputCopyWriter/PadMaskWriter
        contribute their columns through the copy / mask paths
        (`_merge_columns`), not here.

        Raises
        ------
        ConfigError
            When the section mints no final task column for the selection, or
            two leaves mint the same flat H5 column.
        """
        by_key: dict[str, list[tuple[str, Any]]] = {}
        key_order: list[str] = []
        for run_task in self._run_task_outputs():
            for output_key, field in run_task.manifest_fields(self._section_mode):
                suffix = self._column_suffix(field)
                if suffix is None:
                    continue
                if output_key not in by_key:
                    by_key[output_key] = []
                    key_order.append(output_key)
                by_key[output_key].append((suffix, field))
        if not key_order:
            raise ConfigError(
                "H5OutputSink (dumb-section) found no RunTaskOutput task with a final "
                f"{'export-selection' if self._section_mode is Mode.ONNX else 'H5'} column — "
                "wire a RunTaskOutput([tasks]) in the outputs: section (plan 34 W34.2; for "
                "salt inference the RunTaskOutput's modes: list must include 'export')"
            )
        seen_cols: dict[tuple[str, str], str] = {}
        columns: list[OutputColumn] = []
        for output_key in key_order:
            pairs = by_key[output_key]
            stream = output_key.split(KEY_SEP)[1]
            prefix = pairs[0][1].prefix
            dtype = pairs[0][1].dtype
            suffixes = [suffix for suffix, _ in pairs]
            col = OutputColumn(key=output_key, suffixes=suffixes, dtype=dtype, prefix=prefix)
            for column_name in col.column_names(run_name):
                if (other := seen_cols.get((stream, column_name))) is not None:
                    raise ConfigError(
                        f"H5OutputSink (dumb-section): flat column {column_name!r} in stream "
                        f"{stream!r} is minted by BOTH {other} AND {output_key!r} (plan 34 W34.2)"
                    )
                seen_cols[stream, column_name] = output_key
            columns.append(col)
        resolved = tuple(columns)
        self._columns = resolved
        self._columns_resolved = True
        return resolved

    def _ensure_columns(self) -> tuple[OutputColumn, ...]:
        """The resolved columns for the CURRENT run name (auto-collect aware).

        Uses the cached run name (`_run_name`, set at `open_schema`) for the
        prefix; before a run (static demand) the suffixes are
        run-name-independent, so a provisional resolve with the placeholder
        run name yields the same keys.
        """
        return self._resolve_columns(self._run_name)

    # -- graph node surface -------------------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """TEST requires the demanded ``outputs.*`` leaves, ``meta.rows``, and each
        pad-mask stream's ``masks.<stream>``; produces nothing (terminal node,
        planner keeps it). FIT/VAL/ONNX: empty requires/produces (pruned).
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        # dtype is None on the require: OutputColumn.dtype is the H5 numpy
        # descriptor, not the producer's torch dtype — constraining it would
        # conflict with the producer's declared dtype; the sink consumes
        # whatever leaf the producer emits and casts at write time.
        req: dict[str, TensorSpec] = {
            col.key: TensorSpec(shape=None, dtype=None, kind="data")
            for col in self._ensure_columns()
        }
        req["meta.rows"] = TensorSpec(shape=None, dtype="int64", kind="meta")
        for stream in self._pad_mask_streams():
            if self._is_dumb_section():
                # in dumb-section mode the PadMaskWriter section node produces
                # outputs.<stream>.mask; the sink DEMANDS that leaf (keeping
                # PadMaskWriter alive in the plan) and reads the bool mask from
                # it — the section feeds the sink through the graph.
                req[f"outputs.{stream}.mask"] = TensorSpec(
                    shape=None, dtype=None, kind="data"
                )
            else:
                req[f"masks.{stream}"] = TensorSpec(
                    shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
                )
        # each object-group field sources a bundle leaf the sink must demand
        # (so its producer — a decoder head, a truth-label processor, a
        # reconstruction node like MaskFormerObjects — stays alive in the TEST
        # plan and its leaf threads into the consume bundle). Empty for the
        # shipped non-object-group sinks (byte-identical no-op).
        for key, spec in self._object_group_requires().items():
            req.setdefault(key, spec)
        return IO(requires=unflatten_spec(req), produces={})

    def _object_group_requires(self) -> dict[str, TensorSpec]:
        """The source-leaf demand of every object-group field, for the sink to anchor.

        The sink consumes each leaf and casts at write time, so the require
        dtype is unconstrained; the field's ``kind`` must match its producer's
        declared kind (data / label / pad_mask) or the planner's kind-unify
        raises.
        """
        out: dict[str, TensorSpec] = {}
        for group in self._object_groups:
            for field in group.fields:
                out.setdefault(
                    field.leaf, TensorSpec(shape=None, dtype=None, kind=field.kind)
                )
        return out

    # -- static demand (consumed by SaltModule) ----------------------

    def writer_demand(self, model_modules: Mapping[str, Any], reader: Any) -> dict[str, str]:
        """The sink's TEST ``declare_io`` requires, each mapped to a demander
        description; discovered by `SaltModule._attached_writer` via
        ``callable(getattr(cb, "writer_demand", None))``.
        """
        del model_modules, reader
        who = "sink 'H5OutputSink' demanding"
        return {key: f"{who} {key}" for key in flatten_spec(self.declare_io(Mode.TEST).requires)}

    def _pad_mask_streams(self) -> tuple[str, ...]:
        """The sequence streams a pad-mask column is requested for.

        ``True`` -> every demanded output's stream, de-duplicated in
        first-seen order.
        """
        if self.write_pad_mask is False:
            return ()
        if self.write_pad_mask is True:
            seen: dict[str, None] = {}
            for col in self._ensure_columns():
                seen.setdefault(col.stream, None)
            return tuple(seen)
        return tuple(dict.fromkeys(self.write_pad_mask))

    # -- node lifecycle (relocated VERBATIM from on_test_*) --------

    def open_schema(self, ctx: SinkContext) -> None:
        """Create the eval H5 with the full schema BEFORE the first batch.

        Resolves the output path + total rows from the context's datamodule,
        opens the source handle for input copies, merges the output /
        input-copy / pad-mask columns into per-group dtypes/shapes, and
        creates the FIXED-mode `H5Writer`.

        Raises
        ------
        ConfigError
            For a missing ``ckpt_path``, a foreign datamodule, an unknown
            output template key, a non-sequence pad-mask stream, a column
            collision, or a missing input-copy source variable. Also when a
            reader advertising no h5py-openable structured source
            (``reader.h5_source is None`` — a uproot/ROOT reader, a
            `MultiSampleReader`, or a global-only custom reader) is paired with
            pad-mask columns or input-copying, which genuinely need that source
            file — such a reader with neither demand is fine.
        """
        dm = ctx.datamodule
        dset = getattr(dm, "test_dset", None)
        if dset is None:
            raise ConfigError(
                "H5OutputSink needs a GraphDataModule with a built test dataset — "
                f"got {type(dm).__name__} (design §5.1)"
            )
        reader = dset.reader
        self._run_name = ctx.run_name
        streams = tuple(getattr(reader, "streams", ()) or ())
        groups = getattr(reader, "groups", None)
        self._mask_streams = self._pad_mask_streams()
        # The structured-H5 path opens an h5py source to probe per-stream
        # sequence lengths (pad-mask columns) and to copy input fields. Key on
        # the reader's advertised CAPABILITY (`h5_source`), not its type: a
        # reader with no h5py-openable source (`h5_source is None`) — a
        # uproot/ROOT reader whose `.groups` are a non-H5 config shape, a
        # MultiSampleReader wrapping any reader, or a global-only custom reader
        # — takes the no-source path, writing task outputs only. That path is
        # fine when NEITHER pad-mask columns NOR input-copying is demanded;
        # both genuinely need the source file (design §5.1). Capability-keying
        # (no isinstance) makes MultiSampleReader delegation work for free.
        h5_source = getattr(reader, "h5_source", None)
        copy_requested = bool(self.copy_inputs) or self._copy_all_tasked_streams
        if h5_source is None:
            if self._mask_streams or copy_requested:
                want = " and ".join(
                    label
                    for label, needed in (
                        ("pad-mask columns", bool(self._mask_streams)),
                        ("input-copying", copy_requested),
                    )
                    if needed
                )
                raise ConfigError(
                    f"H5OutputSink: {want} need an H5StructuredReader-style reader "
                    f"exposing an h5py-openable source (reader.h5_source) — "
                    f"{type(reader).__name__} advertises none (design §5.1)"
                )
            sequence_streams: tuple[str, ...] = ()
            group_datasets: dict[str, str] = {}
            source_path: Path | None = None
            self._seq_lengths = {}
        else:
            sequence_streams = tuple(
                s for s in streams if not getattr(groups[s], "global_object", False)
            )
            group_datasets = {stream: groups[stream].dataset for stream in streams}
            source_path = Path(reader.source_path)
            # reader-matching open flags (HDF5 rejects mixed SWMR flags on one file
            # within a process)
            with h5py.File(source_path, "r", swmr=True, libver="latest") as f:
                self._seq_lengths = {
                    stream: int(f[group_datasets[stream]].shape[1]) for stream in sequence_streams
                }
        for stream in self._mask_streams:
            if stream not in sequence_streams:
                raise ConfigError(
                    f"H5OutputSink: pad-mask stream {stream!r} is not a sequence stream — "
                    f"pad masks exist for {list(sequence_streams)} only (design §6.1)"
                )
        total = self._expected_rows(ctx, len(dset), dm.batch_size)
        # resolve the NON-reader object groups' trailing shapes (object_groups).
        # Empty object_groups -> {} so self._object_shapes stays empty and the
        # column merge below is byte-identical to a plain sink (the no-op path).
        self._object_shapes = self._resolve_object_shapes(streams)
        if source_path is not None:  # the no-source branch reaches here only with copying off
            self._open_copies(source_path, group_datasets, streams)
        dtypes, shapes = self._merge_columns(
            streams, sequence_streams, group_datasets, total
        )
        self.output_path = self._output_path(ctx, dm, reader)
        self._h5 = H5Writer(
            dst=self.output_path,
            dtypes=dtypes,
            shapes=shapes,
            shuffle=False,
            jets_name=next(iter(dtypes)),  # batch-sizing group (v1 jets_name semantics)
            precision="half" if self.half_precision else "full",
            num_jets=total,  # FIXED mode: valid empty file for an empty test set
        )
        self._rows_written = 0
        self._expected = total

    def consume(self, bundle: Bundle) -> None:
        """Serialise one batch of ``outputs.*`` (+ input copies + masks).

        Validates row alignment against the running counter (raises
        `ConfigError` on a break or wrong-row-count fragment), packs each
        demanded leaf, re-reads input copies, builds pad masks, merges
        same-group fragments, and streams one `H5Writer.write`.
        """
        assert self._h5 is not None, "consume before open_schema"
        rows_t = bundle.get("meta.rows")
        start, stop = int(rows_t[0]), int(rows_t[1])
        if start != self._rows_written:
            raise ConfigError(
                f"H5OutputSink row alignment broke: batch rows [{start}, {stop}) but "
                f"{self._rows_written} rows written so far — sharded or uneven-batch test "
                "loaders are not supported (design §5 row-alignment contract)"
            )
        rows = slice(start, stop)
        n = stop - start
        fragments: dict[str, list[np.ndarray]] = {}
        # input copies first (the v1 inputs_copy -> tasks -> pad_mask column order)
        for stream, arr in self._copy_fragments(rows).items():
            fragments.setdefault(stream, []).append(arr)
        # output columns next
        for arr_stream, arr in self._output_fragments(bundle).items():
            fragments.setdefault(arr_stream, []).append(arr)
        # pad masks last
        for stream, arr in self._mask_fragments(bundle).items():
            fragments.setdefault(stream, []).append(arr)
        # object groups pack their declared fields from the demanded bundle
        # leaves. NON-reader groups (e.g. `objects` / `object_masks`) are new
        # groups; a reader-stream group (e.g. the `tracks` HadronIndex column) is
        # re-expanded to the file token length like any per-token column.
        # Appended AFTER the copy/output/mask fragments so the join order
        # matches `_merge_columns` (object-group columns come last).
        for stream, arr in self._object_group_fragments(bundle).items():
            fragments.setdefault(stream, []).append(arr)
        for stream, arrs in fragments.items():
            for arr in arrs:
                if len(arr) != n:
                    raise ConfigError(
                        f"H5OutputSink: fragment for group {stream!r} has {len(arr)} rows, "
                        f"expected {n} (rows [{start}, {stop}))"
                    )
        data = {
            self._group_of[stream]: (arrs[0] if len(arrs) == 1 else join_structured_arrays(arrs))
            for stream, arrs in fragments.items()
        }
        self._h5.write(data)
        self._rows_written = stop

    def flush(self) -> None:
        """Close the source handle and the sink, warning on a truncated loop."""
        self._close_copies()
        if self._h5 is None:
            return
        if self._rows_written == self._expected:
            self._h5.close()
        else:
            print(
                f"WARNING: wrote {self._rows_written:,} of {self._expected:,} expected rows "
                f"to {self.output_path} — trailing rows are zero-filled"
            )
            self._h5.file.close()
        self._h5 = None
        print("-" * 100)
        print(f"Wrote eval file {self.output_path}")
        print("-" * 100)

    def close_if_open(self) -> None:
        """Idempotently close any open handle on an interrupted test.

        If ``consume`` raised mid-test, Lightning's ``on_test_end`` may not
        run, leaking the FIXED-mode handle and leaving a half-written file
        open. This closes the raw `H5Writer.file` handle (and the cached
        source handle) WITHOUT the full-count assertion, matching `flush`'s
        truncated branch, and is a no-op once already closed.
        """
        self._close_copies()
        if self._h5 is None:
            return
        self._h5.file.close()
        self._h5 = None

    # -- per-batch fragment builders --------------------------------------------

    def _output_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Pack each demanded ``outputs.*`` leaf into its declared columns.

        Per-token (sequence-stream) leaves are re-expanded to the file
        sequence length; a 2-D ``[B, C]`` global leaf is packed directly.
        """
        out: dict[str, np.ndarray] = {}
        frags: dict[str, list[np.ndarray]] = {}
        for col in self._ensure_columns():
            tensor = bundle.get(col.key)
            values = tensor.detach().cpu().numpy()
            # the producer leaf may be [B] / [B, L] (collapsed index columns) or
            # [B, C] / [B, L, C] — give u2s an explicit trailing channel axis so a
            # single-suffix column packs the same as the v1 eval path
            if values.ndim == 1 or (values.ndim == 2 and col.stream in self._seq_lengths):
                values = values[..., np.newaxis]
            arr = u2s(np.ascontiguousarray(values), col.np_dtype(self._run_name))
            if arr.ndim == 2:
                arr = _pad_to(arr, self._seq_lengths[col.stream])
            frags.setdefault(col.stream, []).append(arr)
        for stream, arrs in frags.items():
            out[stream] = arrs[0] if len(arrs) == 1 else join_structured_arrays(arrs)
        return out

    def _mask_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Build the boolean ``mask`` column per requested sequence stream.

        True = padded, padded to the file sequence length with ``mask=False``.
        """
        out: dict[str, np.ndarray] = {}
        for stream in self._mask_streams:
            # in dumb-section mode read the PadMaskWriter's outputs.<stream>.mask
            # leaf (fed through the graph); otherwise the bundle's masks.<stream>
            # directly (the producer path).
            mask_key = (
                f"outputs.{stream}.mask" if self._is_dumb_section() else f"masks.{stream}"
            )
            mask = bundle.get(mask_key).detach().cpu().numpy()
            arr = u2s(np.expand_dims(mask, -1), dtype=np.dtype([("mask", "?")]))
            out[stream] = _pad_to(arr, self._seq_lengths[stream])
        return out

    def _object_group_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Pack each object group's declared fields from the demanded bundle leaves.

        Each field sources one bundle leaf and packs its last dimension into
        the field's named columns; a group's fields are joined in declaration
        order. A reader-stream group's fragment is re-expanded to the file
        token length; a NON-reader group's per-row shape is guarded against the
        schema-declared shape (raises `ConfigError` — e.g. ``object_masks``
        must span the full file constituent width, incompatible with a reader
        ``truncate``). Empty ``object_groups`` -> ``{}`` (the no-op path).
        """
        out: dict[str, np.ndarray] = {}
        for group in self._object_groups:
            frags = [self._pack_object_field(bundle, f) for f in group.fields]
            arr = frags[0] if len(frags) == 1 else join_structured_arrays(frags)
            if group.name in self._seq_lengths and arr.ndim >= 2:
                arr = _pad_to(arr, self._seq_lengths[group.name])
            if group.name in self._object_shapes:
                declared = self._object_shapes[group.name]
                actual = arr.shape[1:]
                if actual != declared:
                    raise ConfigError(
                        f"H5OutputSink: object group {group.name!r} fragment per-row shape "
                        f"{actual} does not match the schema-declared per-row shape {declared} "
                        "— a reader `truncate` narrower than the file width is unsupported for a "
                        "non-reader object group (its trailing axes must span the full file width)"
                    )
            out[group.name] = arr
        return out

    def _pack_object_field(self, bundle: Bundle, field: ObjectGroupField) -> np.ndarray:
        """Pack one field's source leaf into its declared columns (u2s), casting at write time.

        A single-suffix field wraps a channel-less leaf into one column; a
        multi-suffix field expands the leaf's trailing channel axis in
        ``suffixes`` order.
        """
        values = bundle.get(field.leaf).detach().cpu().numpy()
        if len(field.suffixes) == 1:
            values = values[..., np.newaxis]
        return u2s(
            np.ascontiguousarray(values), field.np_dtype(self._run_name, self.half_precision)
        )

    def _copy_fragments(self, rows: slice) -> dict[str, np.ndarray]:
        """Re-read this batch's input-copy columns by absolute rows, at full file
        sequence length, SOURCE dtypes/order.
        """
        return {
            stream: ds.fields(fields)[rows.start : rows.stop]
            for stream, (ds, fields) in self._copy_reads.items()
        }

    # -- setup plumbing ----------------------------------------------------------

    def _open_copies(
        self, source_path: Path, group_datasets: Mapping[str, str], streams: tuple[str, ...]
    ) -> None:
        """Open the cached source handle and resolve per-stream copy field lists;
        raises `ConfigError` for an unknown copy stream or a missing copy variable.
        """
        self._close_copies()
        # dumb-section: InputCopyWriter(streams=None) means "every reader stream"
        # (the v1 default); resolve it now that the reader streams are known.
        if self._copy_all_tasked_streams:
            self.copy_inputs = {s: list(self._copy_variables.get(s, [])) for s in streams}
        if not self.copy_inputs:
            return
        if unknown := sorted(set(self.copy_inputs) - set(streams)):
            raise ConfigError(
                f"H5OutputSink: copy_inputs streams {unknown} are not reader streams — "
                f"streams are {list(streams)}"
            )
        self._copy_handle = h5py.File(source_path, "r", swmr=True, libver="latest")
        for stream in streams:  # reader config order
            if stream not in self.copy_inputs:
                continue
            ds = self._copy_handle[group_datasets[stream]]
            file_fields = list(ds.dtype.names or ())
            fields = self.copy_inputs[stream] or file_fields
            if missing := sorted(set(fields) - set(file_fields)):
                raise ConfigError(
                    f"H5OutputSink: copy_inputs variables {missing} missing for stream "
                    f"{stream!r} in {source_path.name!r} (v1 extra_vars contract)"
                )
            self._copy_reads[stream] = (ds, list(fields))

    def _close_copies(self) -> None:
        """Close the cached source handle (re-entrant across test runs)."""
        if self._copy_handle is not None:
            self._copy_handle.close()
            self._copy_handle = None
        self._copy_reads = {}

    def _resolve_object_shapes(self, streams: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        """Resolve each NON-reader object group's trailing per-row shape.

        A `shape` entry is an ``int``, or a reader stream name resolved to that
        stream's file token length. A non-reader group name may NOT shadow a
        reader stream, and two groups may NOT share a name. A reader-stream
        group (``shape is None``) must name an actual reader stream. Returns
        ``{group -> trailing shape}`` for the non-reader groups only; empty
        ``object_groups`` -> ``{}`` (the byte-identical no-op path).

        Raises
        ------
        ConfigError
            For a duplicate group name, a non-reader group shadowing a reader
            stream, a reader-stream group naming an unknown stream, or a shape
            token naming a stream with no file token length.
        """
        shapes: dict[str, tuple[int, ...]] = {}
        seen: set[str] = set()
        for group in self._object_groups:
            if group.name in seen:
                raise ConfigError(
                    f"H5OutputSink: object group {group.name!r} is declared twice — one group "
                    "owns one H5 group name"
                )
            seen.add(group.name)
            if group.shape is None:
                if group.name not in streams:
                    raise ConfigError(
                        f"H5OutputSink: object group {group.name!r} has no `shape` (a reader-"
                        f"stream group) but {group.name!r} is not a reader stream — reader "
                        f"streams are {list(streams)}; give it a `shape` for a non-reader group"
                    )
                continue
            if group.name in streams:
                raise ConfigError(
                    f"H5OutputSink: non-reader object group {group.name!r} shadows reader stream "
                    f"{group.name!r} — drop its `shape` to add columns to the reader stream, or "
                    "rename the group"
                )
            trailing: list[int] = []
            for dim in group.shape:
                if isinstance(dim, str):
                    if dim not in self._seq_lengths:
                        raise ConfigError(
                            f"H5OutputSink: object group {group.name!r} shape token {dim!r} is "
                            f"not a sequence stream — sequence streams are "
                            f"{sorted(self._seq_lengths)}"
                        )
                    trailing.append(self._seq_lengths[dim])
                else:
                    trailing.append(int(dim))
            shapes[group.name] = tuple(trailing)
        return shapes

    def _merge_columns(
        self,
        streams: tuple[str, ...],
        sequence_streams: tuple[str, ...],
        group_datasets: Mapping[str, str],
        total: int,
    ) -> tuple[dict[str, np.dtype], dict[str, tuple[int, ...]]]:
        """Merge copy / output / pad-mask / object-group columns into per-group dtypes/shapes.

        Column order: input copies, then output columns (declaration order),
        the pad mask, then the object-group columns. Output groups are named
        after the FILE dataset. A column collision or an output leaf for an
        unknown stream raises `ConfigError` naming both contributors.
        """
        descrs: dict[str, list] = {}
        owners: dict[tuple[str, str], str] = {}

        def _add(stream: str, dtype: np.dtype, who: str) -> None:
            if stream not in streams and stream not in self._object_shapes:
                raise ConfigError(
                    f"H5OutputSink: {who} targets unknown group {stream!r} — reader streams "
                    f"are {list(streams)} and object groups are {sorted(self._object_shapes)} "
                    "(declare a non-reader group's `shape` in object_groups)"
                )
            for descr in dtype.descr:
                field_name = descr[0]
                if (other := owners.get((stream, field_name))) is not None:
                    raise ConfigError(
                        f"column {field_name!r} in group {stream!r} is declared by {other} AND "
                        f"{who} — rename one output (collision check, design §8)"
                    )
                owners[stream, field_name] = who
                descrs.setdefault(stream, []).append(descr)

        for stream, (ds, fields) in self._copy_reads.items():
            _add(
                stream,
                np.dtype([(fname, ds.dtype[fname]) for fname in fields]),
                f"copy_inputs[{stream!r}]",
            )
        for col in self._ensure_columns():
            _add(col.stream, col.np_dtype(self._run_name), f"output {col.key!r}")
        for stream in self._mask_streams:
            _add(stream, np.dtype([("mask", "?")]), f"pad mask[{stream!r}]")
        # object-group columns — appended AFTER the copy/task/mask columns
        # (mirroring the retired extra-group merge order). Each field declares
        # its column dtypes from its own spec; the per-column uniqueness /
        # attribution check is the SAME `_add` owners map a reader column rides.
        # Dead code when object_groups is empty (byte-identical no-op).
        for group in self._object_groups:
            for field in group.fields:
                _add(
                    group.name,
                    field.np_dtype(self._run_name, self.half_precision),
                    f"object_groups[{group.name!r}].{field.leaf!r}",
                )
        if not descrs:
            raise ConfigError("H5OutputSink declares no output columns at all (design §8)")
        self._group_of = {stream: group_datasets.get(stream, stream) for stream in descrs}
        dtypes = {self._group_of[stream]: np.dtype(descr) for stream, descr in descrs.items()}
        shapes = {
            self._group_of[stream]: self._group_shape(stream, sequence_streams, total)
            for stream in descrs
        }
        return dtypes, shapes

    def _group_shape(
        self, stream: str, sequence_streams: tuple[str, ...], total: int
    ) -> tuple[int, ...]:
        """The fixed-mode H5 shape of one output group (leading ``total`` row dim).

        A reader sequence stream carries ``(total, file_seq_len)``, a reader
        global stream ``(total,)``, and a non-reader object group
        ``(total, *trailing)`` from its `object_groups` shape.
        """
        if stream in self._object_shapes:
            return (total, *self._object_shapes[stream])
        if stream in sequence_streams:
            return (total, self._seq_lengths[stream])
        return (total,)

    @staticmethod
    def _expected_rows(ctx: SinkContext, total: int, batch_size: int) -> int:
        """Rows the (possibly ``limit_test_batches``-capped) loop will write.

        ``min(total, num_batches * batch_size)`` — with the sequential
        no-drop sampler only the last batch is partial.
        """
        num_batches = ctx.num_test_batches
        if not num_batches:
            return total
        limit = num_batches[0]
        if isinstance(limit, (int, float)) and math.isfinite(limit):
            return min(total, int(limit) * batch_size)
        return total

    def _output_path(self, ctx: SinkContext, dm: Any, reader: Any) -> Path:
        """Render the output template; raises `ConfigError` when ``ckpt_path``
        is unset or the template names an unknown key.
        """
        ckpt_path = ctx.ckpt_path
        if ckpt_path is None:
            raise ConfigError(
                "H5OutputSink needs a checkpoint path — run salt test with --ckpt_path "
                "<ckpt> (the output file is named after the checkpoint, v1 contract)"
            )
        # name the output after the reader's file. A single-file reader exposes
        # `filename`/`source_path`; a MultiSampleReader (N sources, neither
        # attribute) falls back to its first staged source, else the run name.
        src = getattr(reader, "filename", None) or getattr(reader, "source_path", None)
        if src is None:
            srcs = reader.sources() if hasattr(reader, "sources") else []
            src = srcs[0] if srcs else self._run_name
        stem = Path(src).stem
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
                f"unknown H5OutputSink output template key {err} — available: {sorted(keys)}"
            ) from None


# DEPRECATED one-window alias (design Q4): the node-shaped sink was renamed
# H5OutputWriter -> H5OutputSink. Downstream configs that wire
# `salt.outputs.H5OutputWriter` (incl. gn2v2-dummy-cutover.yaml) keep
# working — the alias resolves to the promoted node. Remove after the migration
# window.
H5OutputWriter = H5OutputSink
"""Deprecated alias for `H5OutputSink`."""
