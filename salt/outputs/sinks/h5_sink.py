"""The H5 sink — `H5OutputSink`, the eval-H5 serialiser."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from operator import attrgetter
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
    Mode,
    TensorSpec,
    unflatten_spec,
)
from salt.outputs.output_schema import ObjectGroup, ObjectGroupField, OutputColumn, group_columns
from salt.outputs.sinks.sink import RuntimeSink, SinkContext, collect_manifest_fields
from salt.utils.array_utils import join_structured_arrays
from salt.utils.logging import console, get_logger

_LOG = get_logger(__name__)

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


def _join(arrs: list[np.ndarray]) -> np.ndarray:
    """One group's structured fragments as one array (a lone fragment passes through)."""
    return arrs[0] if len(arrs) == 1 else join_structured_arrays(arrs)


class H5OutputSink(RuntimeSink):
    """The H5 sink: a terminal graph node serialising ``outputs.*`` to the eval H5.

    A `GraphModule` terminal node whose ``declare_io`` requires the demanded
    ``outputs.*`` leaves (+ ``meta.rows`` + each pad-mask stream) in TEST and
    produces nothing — so the planner keeps it in TEST (rendering its own
    card) while FIT/VAL/ONNX prune it.

    It accumulates the named ``outputs.*`` producer leaves per batch and
    writes them through one ftag `H5Writer` (FIXED mode, ``num_jets`` known
    up front — a valid empty file for an empty test set).

    The sink (not the producers) owns: structured-array packing of plain
    ``[B, C]``/``[B, L, C]`` leaves into ``{run_name}_{suffix}`` columns;
    per-token pad re-expansion to the FILE sequence length; input-variable
    copies re-read from the source file by absolute rows (``meta.rows``) with
    SOURCE dtypes/order; optional boolean pad-mask columns (True = padded;
    truncated-away positions read False); and reader-stream -> FILE dataset
    naming. `writer_demand` returns the consumed ``outputs.*`` leaves PLUS the
    source ``preds.*`` leaf of each producer feeding them (so the TEST
    dead-preds gate sees those predictions as consumed), ``meta.rows``, and
    each pad-mask stream's mask.

    Parameters
    ----------
    copy_inputs : Mapping[str, Sequence[str]] | None, optional
        Deprecated, ignored; kept so pre-round-5 saved configs still load.
        Must be left at the default (None); use `InputCopyWriter` in the
        ``outputs:`` section.
    write_pad_mask : bool | Sequence[str], optional
        Deprecated, ignored; kept so pre-round-5 saved configs still load.
        Must be left at the default (False); use `PadMaskWriter` in the
        ``outputs:`` section.
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
    consumes : Sequence[str] | None, optional
        fnmatch patterns over the ``outputs.*`` leaf key narrowing the
        collected column schema, by default None (every declared final leaf).

    Raises
    ------
    ConfigError
        (At construction) `copy_inputs` / `write_pad_mask` given a non-default
        value; (at run setup) a column-name collision / unknown stream /
        missing source variable, or no bound producer minting a column for
        the selection.
    """

    name = "h5_output"
    """The graph-node instance name (overridable by the config dict key)."""

    def __init__(
        self,
        copy_inputs: Mapping[str, Sequence[str]] | None = None,
        write_pad_mask: bool | Sequence[str] = False,
        output: str = DEFAULT_OUTPUT,
        half_precision: bool = False,
        object_groups: Sequence[ObjectGroup | Mapping[str, Any]] | None = None,
        modes: Sequence[str] | None = None,
        consumes: Sequence[str] | None = None,
    ) -> None:
        if copy_inputs is not None:
            raise ConfigError(
                "H5OutputSink: copy_inputs is no longer supported (deprecated, ignored; kept "
                "only so pre-round-5 saved configs still load) — request input copies with an "
                "InputCopyWriter entry in the outputs: section"
            )
        if write_pad_mask is not False:
            raise ConfigError(
                "H5OutputSink: write_pad_mask is no longer supported (deprecated, ignored; kept "
                "only so pre-round-5 saved configs still load) — request pad-mask columns with a "
                "PadMaskWriter entry in the outputs: section"
            )
        super().__init__(modes=modes, consumes=consumes)
        self._columns: tuple[OutputColumn, ...] = ()
        self._columns_resolved = False
        # section-driven copy/mask state: set by bind_output_section, empty until then
        self.copy_inputs: dict[str, list[str]] = {}
        self.write_pad_mask: tuple[str, ...] | bool = False
        self.output = output
        self.half_precision = half_precision
        # empty by default — then a strict no-op, H5 schema byte-identical
        self._object_groups = tuple(ObjectGroup.coerce(g) for g in object_groups or ())
        # Mode.TEST = eval schema (default); Mode.ONNX = strictly the export
        # output set (`use_export_selection`, salt inference).
        self._section_mode: Mode = Mode.TEST
        # section copy resolution: False until a section binds; True means
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
        is resolution-free even before the section binds its columns.
        """
        return True

    # -- section binding -------------------------------------

    def use_export_selection(self) -> None:
        """Switch the section-derived selection to the EXPORT (``Mode.ONNX``) set.

        ``salt inference`` writes STRICTLY the export
        output set to H5: columns resolve from ``manifest_fields(Mode.ONNX)``
        (one single-suffix column per ONNX leaf, named by the field's resolved
        ONNX name), and the section's copy/mask writers contribute only when
        their ``modes:`` include ``export``. Call BEFORE `bind_output_section`.
        """
        self._section_mode = Mode.ONNX
        self._columns_resolved = False

    def _invalidate_manifest(self) -> None:
        """Drop the DERIVED column table after a manifest source rebinds."""
        self._columns_resolved = False

    def bind_output_section(self, section: Mapping[str, Any]) -> None:
        """Capture the ``outputs:`` section the sink derives its columns from.

        The section is the ORDERED dict of section writers (`RunTaskOutput`,
        `InputCopyWriter`, `PadMaskWriter`): the sink derives its column
        schema + input-copy spec + pad-mask streams from the SECTION manifest
        in SECTION DECLARATION ORDER (the H5 column-order authority); the
        section's copy/mask writers set the sink's copy/mask state. A
        copy/mask writer whose ``modes:`` exclude this sink's selection mode
        contributes nothing.
        """
        # local: input_copy_writer -> run_task_output -> salt.model.base would cycle
        from salt.outputs.input_copy_writer import InputCopyWriter
        from salt.outputs.pad_mask_writer import PadMaskWriter
        from salt.outputs.run_task_output import OutputSectionWriter

        super().bind_output_section(section)
        # copy_inputs / write_pad_mask also come from the section's writers (this sink's mode)
        copy_inputs: dict[str, list[str]] = {}
        mask_streams: list[str] = []
        mode = self._section_mode
        for writer in section.values():
            if not isinstance(writer, OutputSectionWriter) or not writer.runs_in_mode(mode):
                continue
            if isinstance(writer, InputCopyWriter):
                # streams None -> every stream with a configured task, resolved
                # at open_schema against the reader (variables stashed until then)
                self._copy_all_tasked_streams = writer.streams is None
                self._copy_variables = {k: list(v) for k, v in writer.variables.items()}
                for s in writer.streams or ():
                    copy_inputs[s] = list(writer.variables.get(s, []))
            if isinstance(writer, PadMaskWriter):
                mask_streams.extend(writer.streams)
        if not self._copy_all_tasked_streams:
            self.copy_inputs = copy_inputs
        self.write_pad_mask = tuple(dict.fromkeys(mask_streams)) if mask_streams else False

    def _resolve_columns(self, run_name: str) -> tuple[OutputColumn, ...]:
        """Resolve the H5 column table from the bound manifest sources (cached).

        Collects every FINAL field the bound producers declare for the
        selection mode, narrows by ``consumes:``, and assembles ONE
        `OutputColumn` per ``outputs.*`` leaf from the fields the selection
        names: TEST keeps fields with an ``h5_name`` (the eval schema); the
        export selection (``Mode.ONNX``) keeps fields with a resolved ONNX name
        — the EXACT rule the `OnnxExportSink` tuple uses, so the inference H5
        columns are 1:1 with the ONNX tuple by construction. MANIFEST field
        order (not executor topo order) is the H5 column-order authority.
        Copy/mask writers contribute through `_merge_columns`, not here.
        `ConfigError` when no producer mints a final column for the selection,
        or two leaves mint the same flat H5 column.
        """
        if self._columns_resolved:
            return self._columns
        columns = group_columns(
            self._filter_consumed(
                collect_manifest_fields(self._manifest_sources(), self._section_mode)
            ),
            attrgetter("h5_name" if self._section_mode is Mode.TEST else "resolved_onnx_name"),
        )
        if not columns:
            raise ConfigError(
                "H5OutputSink found no RunTaskOutput task with a final "
                f"{'export-selection' if self._section_mode is Mode.ONNX else 'H5'} column — "
                "wire a RunTaskOutput([tasks]) in the outputs: section (for "
                "salt inference the RunTaskOutput's modes: list must include 'export')"
            )
        seen_cols: dict[tuple[str, str], str] = {}
        for col in columns:
            for column_name in col.column_names(run_name):
                if (other := seen_cols.get((col.stream, column_name))) is not None:
                    raise ConfigError(
                        f"H5OutputSink: flat column {column_name!r} in stream "
                        f"{col.stream!r} is minted by BOTH {other} AND {col.key!r}"
                    )
                seen_cols[col.stream, column_name] = col.key
        self._columns, self._columns_resolved = tuple(columns), True
        return self._columns

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
        pad-mask stream's ``outputs.<stream>.mask``; produces nothing (terminal
        node, planner keeps it). FIT/VAL/ONNX: empty requires/produces (pruned).
        """
        del mode  # non-TEST modes never get here (`allowed_modes` = TEST; `Node` gates)
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
            # the PadMaskWriter section node produces outputs.<stream>.mask; the
            # sink demands that leaf (keeping PadMaskWriter alive in the plan) and
            # reads the bool mask from it — the section feeds the sink through
            # the graph.
            req[f"outputs.{stream}.mask"] = TensorSpec(shape=None, dtype=None, kind="data")
        # demand each object-group field's source leaf (first declaration wins)
        # so its producer stays alive in the TEST plan; empty for non-object-group
        # sinks. Dtype unconstrained (cast at write time); the field's ``kind``
        # must match its producer's declared kind or the planner's kind-unify raises.
        for field in (f for group in self._object_groups for f in group.fields):
            req.setdefault(field.leaf, TensorSpec(shape=None, dtype=None, kind=field.kind))
        return IO(requires=unflatten_spec(req), produces={})

    def _pad_mask_streams(self) -> tuple[str, ...]:
        """The PadMaskWriter-declared streams a pad-mask column is requested for."""
        if not self.write_pad_mask:
            return ()
        return tuple(dict.fromkeys(self.write_pad_mask))

    # -- node lifecycle (relocated VERBATIM from on_test_*) --------

    def open_schema(self, ctx: SinkContext) -> None:
        """Create the eval H5 with the full schema BEFORE the first batch.

        Resolves the output path + total rows from the context's datamodule,
        opens the source handle for input copies, merges the output /
        input-copy / pad-mask columns into per-group dtypes/shapes, and
        creates the FIXED-mode `H5Writer`.

        `ConfigError` on: missing ``ckpt_path``, foreign datamodule, unknown
        output template key, non-sequence pad-mask stream, column collision,
        missing input-copy source variable, or a reader with no h5py-openable
        source (``reader.h5_source is None``) paired with pad-mask columns or
        input-copying (which genuinely need the source file).
        """
        dm = ctx.datamodule
        if (dset := getattr(dm, "test_dset", None)) is None:
            raise ConfigError(
                "H5OutputSink needs a SaltDataModule with a built test dataset — "
                f"got {type(dm).__name__}"
            )
        reader = dset.reader
        self._run_name = ctx.run_name
        streams = tuple(getattr(reader, "streams", ()) or ())
        groups = getattr(reader, "groups", None)
        self._mask_streams = self._pad_mask_streams()
        # key on the reader's advertised CAPABILITY (`h5_source`), not its type:
        # a reader with no h5py-openable source takes the no-source path (task
        # outputs only), fine unless pad-mask columns or input-copying are
        # demanded — both genuinely need the source file.
        h5_source = getattr(reader, "h5_source", None)
        copy_requested = bool(self.copy_inputs) or self._copy_all_tasked_streams
        if h5_source is None:
            if self._mask_streams or copy_requested:
                needs = {"pad-mask columns": self._mask_streams, "input-copying": copy_requested}
                want = " and ".join(label for label, needed in needs.items() if needed)
                raise ConfigError(
                    f"H5OutputSink: {want} need an H5StructuredReader-style reader "
                    f"exposing an h5py-openable source (reader.h5_source) — "
                    f"{type(reader).__name__} advertises none"
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
                    f"pad masks exist for {list(sequence_streams)} only"
                )
        total = self._expected_rows(ctx, len(dset), dm.batch_size)
        # resolve the NON-reader object groups' trailing shapes (object_groups).
        # Empty object_groups -> {} so self._object_shapes stays empty and the
        # column merge below is byte-identical to a plain sink (the no-op path).
        self._object_shapes = self._resolve_object_shapes(streams)
        if source_path is not None:  # the no-source branch reaches here only with copying off
            self._open_copies(source_path, group_datasets, streams)
        dtypes, shapes = self._merge_columns(streams, sequence_streams, group_datasets, total)
        self.output_path = self._output_path(ctx, dm, reader)
        self._h5 = H5Writer(
            dst=self.output_path,
            dtypes=dtypes,
            shapes=shapes,
            shuffle=False,
            jets_name=next(iter(dtypes)),  # batch-sizing group (v1 jets_name semantics)
            precision="half" if self.half_precision else "full",
            num_jets=total,  # FIXED mode: valid empty file for an empty test set
            # h5py-native lzf (NOT ftag's default lz4 via hdf5plugin): plain
            # h5py reads the file back anywhere with zero setup
            compression="lzf",
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
                "loaders are not supported (row-alignment contract)"
            )
        n = stop - start
        fragments: dict[str, list[np.ndarray]] = {}
        # input copies, output columns, pad masks, then object groups: the v1
        # inputs_copy -> tasks -> pad_mask column order, as `_merge_columns` lays it out
        for part in (
            self._copy_fragments(slice(start, stop)),
            self._output_fragments(bundle),
            self._mask_fragments(bundle),
            self._object_group_fragments(bundle),
        ):
            for stream, arr in part.items():
                fragments.setdefault(stream, []).append(arr)
        for stream, arrs in fragments.items():
            for arr in arrs:
                if len(arr) != n:
                    raise ConfigError(
                        f"H5OutputSink: fragment for group {stream!r} has {len(arr)} rows, "
                        f"expected {n} (rows [{start}, {stop}))"
                    )
        self._h5.write({self._group_of[stream]: _join(arrs) for stream, arrs in fragments.items()})
        self._rows_written = stop

    def flush(self) -> None:
        """Close the source handle and the sink, warning on a truncated loop."""
        self._close_copies()
        if self._h5 is None:
            return
        if self._rows_written == self._expected:
            self._h5.close()
        else:
            _LOG.warning(
                f"wrote {self._rows_written:,} of {self._expected:,} expected rows "
                f"to {self.output_path} — trailing rows are zero-filled"
            )
            self._h5.file.close()
        self._h5 = None
        console("-" * 100)
        console(f"Wrote eval file {self.output_path}")
        console("-" * 100)

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
        frags: dict[str, list[np.ndarray]] = {}
        for col in self._ensure_columns():
            values = bundle.get(col.key).detach().cpu().numpy()
            # the producer leaf may be [B] / [B, L] (collapsed index columns) or
            # [B, C] / [B, L, C] — give u2s an explicit trailing channel axis so a
            # single-suffix column packs the same as the v1 eval path
            if values.ndim == 1 or (values.ndim == 2 and col.stream in self._seq_lengths):
                values = values[..., np.newaxis]
            arr = u2s(np.ascontiguousarray(values), col.np_dtype(self._run_name))
            if arr.ndim == 2:
                arr = _pad_to(arr, self._seq_lengths[col.stream])
            frags.setdefault(col.stream, []).append(arr)
        return {stream: _join(arrs) for stream, arrs in frags.items()}

    def _mask_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Build the boolean ``mask`` column per requested sequence stream.

        True = padded, padded to the file sequence length with ``mask=False``.
        """
        out: dict[str, np.ndarray] = {}
        for stream in self._mask_streams:
            # read the PadMaskWriter's outputs.<stream>.mask leaf (fed through the graph)
            mask = bundle.get(f"outputs.{stream}.mask").detach().cpu().numpy()
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
        ``pad_max``). Empty ``object_groups`` -> ``{}`` (the no-op path).
        """
        out: dict[str, np.ndarray] = {}
        for group in self._object_groups:
            arr = _join([self._pack_object_field(bundle, f) for f in group.fields])
            if group.name in self._seq_lengths and arr.ndim >= 2:
                arr = _pad_to(arr, self._seq_lengths[group.name])
            declared = self._object_shapes.get(group.name)
            if declared is not None and arr.shape[1:] != declared:
                raise ConfigError(
                    f"H5OutputSink: object group {group.name!r} fragment per-row shape "
                    f"{arr.shape[1:]} does not match the schema-declared per-row shape {declared} "
                    "— a reader `pad_max` narrower than the file width is unsupported for a "
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
        # InputCopyWriter(streams=None) means "every reader stream"
        # (the v1 default); resolve it now that the reader streams are known.
        if self._copy_all_tasked_streams:
            self.copy_inputs = {s: list(self._copy_variables.get(s, [])) for s in streams}
        if not self.copy_inputs:
            return
        if unknown := sorted(set(self.copy_inputs) - set(streams)):
            raise ConfigError(
                f"H5OutputSink: InputCopyWriter streams {unknown} are not reader streams — "
                f"streams are {list(streams)}"
            )
        self._copy_handle = h5py.File(source_path, "r", swmr=True, libver="latest")
        for stream in [s for s in streams if s in self.copy_inputs]:  # reader config order
            ds = self._copy_handle[group_datasets[stream]]
            file_fields = list(ds.dtype.names or ())
            fields = self.copy_inputs[stream] or file_fields
            if missing := sorted(set(fields) - set(file_fields)):
                raise ConfigError(
                    f"H5OutputSink: InputCopyWriter variables {missing} missing for stream "
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
        stream's file token length. `ConfigError` on: duplicate group name, a
        non-reader group shadowing a reader stream, a reader-stream group
        (``shape is None``) naming an unknown stream, or a shape token naming
        a stream with no file token length. Returns ``{group -> trailing
        shape}`` for non-reader groups only.
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
                if isinstance(dim, str) and dim not in self._seq_lengths:
                    raise ConfigError(
                        f"H5OutputSink: object group {group.name!r} shape token {dim!r} is "
                        f"not a sequence stream — sequence streams are "
                        f"{sorted(self._seq_lengths)}"
                    )
                trailing.append(self._seq_lengths[dim] if isinstance(dim, str) else int(dim))
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
                if (other := owners.get((stream, descr[0]))) is not None:
                    raise ConfigError(
                        f"column {descr[0]!r} in group {stream!r} is declared by {other} AND "
                        f"{who} — rename one output (collision check)"
                    )
                owners[stream, descr[0]] = who
                descrs.setdefault(stream, []).append(descr)

        for stream, (ds, fields) in self._copy_reads.items():
            copy_dtype = np.dtype([(f, ds.dtype[f]) for f in fields])
            _add(stream, copy_dtype, f"InputCopyWriter[{stream!r}]")
        for col in self._ensure_columns():
            _add(col.stream, col.np_dtype(self._run_name), f"output {col.key!r}")
        for stream in self._mask_streams:
            _add(stream, np.dtype([("mask", "?")]), f"pad mask[{stream!r}]")
        # object-group columns come last; same `_add` uniqueness/attribution
        # check as reader columns. Dead code when object_groups is empty.
        for group in self._object_groups:
            for fld in group.fields:
                who = f"object_groups[{group.name!r}].{fld.leaf!r}"
                _add(group.name, fld.np_dtype(self._run_name, self.half_precision), who)
        if not descrs:
            raise ConfigError("H5OutputSink declares no output columns at all")
        self._group_of = {stream: group_datasets.get(stream, stream) for stream in descrs}
        dtypes = {self._group_of[s]: np.dtype(descr) for s, descr in descrs.items()}
        shapes = {self._group_of[s]: self._group_shape(s, sequence_streams, total) for s in descrs}
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
        limit = ctx.num_test_batches[0] if ctx.num_test_batches else None
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
        if test_suff := getattr(dm, "test_suff", None):
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
