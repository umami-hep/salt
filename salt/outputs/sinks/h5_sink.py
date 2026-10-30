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
    KEY_SEP,
    Mode,
    TensorSpec,
    flatten_spec,
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


def _declared_seq_lengths(reader: Any) -> dict[str, int]:
    """Token length of each padded stream a reader declares as ``raw.<stream>`` ``(B, T)``.

    A reader with no h5py-openable source pads every jagged stream to a fixed,
    declared length, so its batch token length IS the file token length.
    """
    declare_io = getattr(reader, "declare_io", None)
    if not callable(declare_io):
        return {}
    out: dict[str, int] = {}
    for key, spec in flatten_spec(declare_io(Mode.TEST).produces).items():
        namespace, _, stream = key.partition(KEY_SEP)
        shape = spec.shape
        if (
            namespace == "raw"
            and KEY_SEP not in stream
            and shape is not None
            and len(shape) == 2
            and isinstance(shape[1], int)
        ):
            out[stream] = shape[1]
    return out


class H5OutputSink(RuntimeSink):
    """The H5 sink: a terminal graph node serialising ``outputs.*`` to the eval H5.

    Terminal `GraphModule` node: ``declare_io`` requires the demanded
    ``outputs.*`` leaves (+ ``meta.rows`` + each pad-mask stream) in TEST and
    produces nothing, so the planner keeps it in TEST and prunes it in
    FIT/VAL/ONNX.

    The file is created LAZILY, on the FIRST ``consume``, as a ftag `H5Writer`
    in DYNAMIC mode (no row total passed: every dataset is created empty with
    an unbounded row axis and resized on every write); later batches append.
    Input copies are not a source-file side channel — `InputCopyWriter` mints
    ordinary ``outputs.<stream>.<var>`` leaves with ``OutputColumn.dtype =
    None`` (the leaf's own dtype, learned from the first batch, never a
    declared one).

    Row-order contract (stated once, here): rows land in BATCH ARRIVAL order —
    source order on the map-style loader, the workers' round-robin arrival
    order on the streaming loader (deterministic for a fixed
    ``seed``/``num_workers``/corpus, but neither source order nor the order a
    different worker count would produce). A row is identified by its written
    columns (copies, task outputs, target labels), never by its position.

    The sink (not the producers) owns: structured-array packing of plain
    ``[B, C]``/``[B, L, C]`` leaves into ``{run_name}_{suffix}`` columns;
    per-token pad re-expansion to the FILE sequence length; optional boolean
    pad-mask columns (True = padded; truncated-away positions read False);
    and reader-stream -> FILE dataset naming. `writer_demand` adds, to the
    consumed ``outputs.*`` leaves, the source ``preds.*`` leaf of each feeding
    producer (so the TEST dead-preds gate sees those predictions as consumed),
    ``meta.rows``, and each pad-mask stream's mask.

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
        value; (at run setup) a column-name collision, an output leaf for an unknown
        stream, a pad-mask stream whose token length is unknown (no
        h5py-openable ``reader.h5_source`` and no declared fixed-length
        ``raw.<stream>``), or no bound producer minting a column for the
        selection. (At the first ``consume``) a per-token (sequence-stream)
        column — an input copy or a per-token task output — on a stream whose
        token length is unknown for the same reason.
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
        # per-test-run state (reset at open_schema); `_h5` is created lazily on
        # the first `consume`.
        self._h5: H5Writer | None = None
        self._rows_written = 0
        self._split_rows: int | None = None
        self._capped = False
        self._run_name = "salt"
        self._group_of: dict[str, str] = {}
        self._seq_lengths: dict[str, int] = {}
        # reader class name when the reader advertises no `h5_source` (None
        # otherwise) — named by the per-token-column ConfigError in consume.
        self._sourceless_reader: str | None = None
        # non-reader object-group -> trailing per-row shape, resolved at
        # open_schema from the configured object_groups ({} when none).
        self._object_shapes: dict[str, tuple[int, ...]] = {}
        self._mask_streams: tuple[str, ...] = ()
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
        ONNX name), and the section's mask writers contribute only when their
        ``modes:`` include ``export``. Call BEFORE `bind_output_section`.
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
        schema from the SECTION manifest in SECTION DECLARATION ORDER (the H5
        column-order authority). Only the `PadMaskWriter`'s streams need
        collecting here (they set the sink's ``write_pad_mask`` state) —
        `InputCopyWriter` mints ordinary `OutputColumn`s like any other
        producer (see the class docstring). A mask writer whose ``modes:``
        exclude this sink's selection mode contributes nothing.
        """
        # local: pad_mask_writer -> run_task_output -> salt.model.base would cycle
        from salt.outputs.pad_mask_writer import PadMaskWriter

        super().bind_output_section(section)
        mask_streams: list[str] = []
        mode = self._section_mode
        for writer in section.values():
            if isinstance(writer, PadMaskWriter) and writer.runs_in_mode(mode):
                mask_streams.extend(writer.streams)
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
        Mask writers contribute through `_merge_columns`, not here.
        `ConfigError` when no producer other than an input-copy writer
        (``dtype is None`` fields) mints a final column for the selection, or
        two leaves mint the same flat H5 column.
        """
        if self._columns_resolved:
            return self._columns
        columns = group_columns(
            self._filter_consumed(
                collect_manifest_fields(self._manifest_sources(), self._section_mode)
            ),
            attrgetter("h5_name" if self._section_mode is Mode.TEST else "resolved_onnx_name"),
        )
        # copy columns (`dtype is None`) keep their place in the column order but
        # do not satisfy this guard: a copy-only section must still hard-fail.
        if not any(col.dtype is not None for col in columns):
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
        """Resolve the eval-H5 schema BEFORE the first batch; the FILE itself is
        created LAZILY by the first `consume` (ftag dynamic mode).

        Resolves the reader/dataset, the sequence lengths (read from the source
        file, read-only, or from a source-less reader's declared fixed-length
        streams), the column-name/owner table
        (`_merge_columns`), the output path, and the advisory row total + a
        binding-cap flag (true only when the configured limit would stop the
        loop before the split total) that `flush` uses for its completeness
        warning.

        `ConfigError` on: missing ``ckpt_path``, foreign datamodule, unknown
        output template key, non-sequence pad-mask stream, column collision,
        or a pad-mask stream whose token length is unknown: a reader with no
        h5py-openable source (``reader.h5_source is None``) supplies it only by
        declaring a fixed-length ``raw.<stream>`` (`_declared_seq_lengths`).
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
        # key on the advertised CAPABILITY (`h5_source`), not the reader type: a
        # source-less reader takes the no-source path (task outputs only).
        h5_source = getattr(reader, "h5_source", None)
        self._sourceless_reader = type(reader).__name__ if h5_source is None else None
        if h5_source is None:
            # no file to read token lengths from: take them from the reader's declared
            # fixed (B, pad_max) streams
            self._seq_lengths = _declared_seq_lengths(reader)
            if missing := [s for s in self._mask_streams if s not in self._seq_lengths]:
                raise ConfigError(
                    "H5OutputSink: pad-mask columns need the file token length — an "
                    "H5StructuredReader-style reader exposing an h5py-openable source "
                    "(reader.h5_source), or a declared fixed-length raw.<stream> — "
                    f"{type(reader).__name__} provides neither for {missing}"
                )
            sequence_streams: tuple[str, ...] = tuple(self._seq_lengths)
            group_datasets: dict[str, str] = {}
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
        # empty object_groups -> {}, so the column merge below stays byte-identical
        # to a plain sink (the no-op path).
        self._object_shapes = self._resolve_object_shapes(streams)
        self._merge_columns(streams, group_datasets)
        self.output_path = self._output_path(ctx, dm, reader)
        self._h5 = None
        self._rows_written = 0
        n_rows = getattr(dset, "n_rows", None)
        self._split_rows = (
            int(n_rows()) if callable(n_rows) else (len(dset) if hasattr(dset, "__len__") else None)
        )
        # Lightning sets `num_test_batches` even when `limit_test_batches` is NOT
        # configured, so a finite `limit` alone does NOT mean truncation — only a
        # limit stopping the loop BEFORE the split total is a real cap; otherwise
        # `flush`'s completeness warning must still be allowed to fire.
        limit = ctx.num_test_batches[0] if ctx.num_test_batches else None
        batch_size = getattr(dm, "batch_size", None)
        cap_rows = (
            int(limit) * int(batch_size)
            if isinstance(limit, (int, float))
            and math.isfinite(limit)
            and limit > 0
            and isinstance(batch_size, int)
            and batch_size > 0
            else None
        )
        self._capped = (
            cap_rows is not None and self._split_rows is not None and (cap_rows < self._split_rows)
        )

    def consume(self, bundle: Bundle) -> None:
        """Serialise one batch of ``outputs.*`` (+ pad masks + object groups).

        Batch size ``n`` is the leading dim of the first resolved column's
        leaf. Packs each demanded leaf, builds pad-mask and object-group
        fragments, merges same-group fragments, and — on the FIRST batch —
        creates the dynamic `H5Writer` from the actual batch dtypes/shapes.
        `ConfigError` on a wrong-row-count fragment or a group missing from
        this batch.
        """
        cols = self._ensure_columns()
        n = int(bundle.get(cols[0].key).shape[0])
        fragments: dict[str, list[np.ndarray]] = {}
        # output columns (input copies included), pad masks, then object groups:
        # the join order `_merge_columns` lays out
        for part in (
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
                        f"expected {n}"
                    )
        data: dict[str, np.ndarray] = {}
        for stream, dst in self._group_of.items():
            if stream not in fragments:
                raise ConfigError(
                    f"H5OutputSink: no fragment for group {stream!r} in this batch — every "
                    "declared group must be present in every write (ftag dynamic-mode "
                    "requirement)"
                )
            data[dst] = _join(fragments[stream])
        if self._h5 is None:
            # no row total -> ftag dynamic mode: every dataset starts empty with an
            # unbounded row axis, resized per write. shuffle=False keeps each row's
            # copies/outputs/targets on one line
            self._h5 = H5Writer(
                dst=self.output_path,
                dtypes={g: arr.dtype for g, arr in data.items()},
                shapes={g: (0, *arr.shape[1:]) for g, arr in data.items()},
                shuffle=False,
                jets_name=next(iter(data)),  # batch-sizing group (v1 jets_name semantics)
                precision="half" if self.half_precision else "full",
                # h5py-native lzf (NOT ftag's default lz4 via hdf5plugin): plain
                # h5py reads the file back anywhere with zero setup
                compression="lzf",
            )
        self._h5.write(data)
        self._rows_written += n

    def flush(self) -> None:
        """Close the sink, warning on truncation; an empty run writes NO file."""
        if self._h5 is None:
            _LOG.warning(
                "H5OutputSink: no test batch reached the sink — no eval H5 written "
                f"at {self.output_path}"
            )
            return
        if (
            not self._capped
            and self._split_rows is not None
            and self._rows_written != self._split_rows
        ):
            _LOG.warning(
                f"H5OutputSink wrote {self._rows_written:,} rows but the test split has "
                f"{self._split_rows:,} — the loop ended early or a shard dropped rows"
            )
        self._h5.close()
        self._h5 = None
        console("-" * 100)
        console(f"Wrote eval file {self.output_path} ({self._rows_written:,} rows)")
        console("-" * 100)

    def close_if_open(self) -> None:
        """Idempotently close any open handle on an interrupted test.

        If ``consume`` raised mid-test, Lightning's ``on_test_end`` may not
        run, leaking the handle. Dynamic-mode `close()` has no completeness
        assertion, so this is identical to `flush`'s close and safe to call
        more than once (a no-op once already closed).
        """
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # -- per-batch fragment builders --------------------------------------------

    def _output_fragments(self, bundle: Bundle) -> dict[str, np.ndarray]:
        """Pack each demanded ``outputs.*`` leaf into its declared columns.

        Per-token (sequence-stream) leaves are re-expanded to the file
        sequence length; a 2-D ``[B, C]`` global leaf is packed directly. A
        column with ``dtype is None`` (an `InputCopyWriter` copy) takes the
        LEAF's own dtype instead of a declared descriptor.
        """
        frags: dict[str, list[np.ndarray]] = {}
        for col in self._ensure_columns():
            values = bundle.get(col.key).detach().cpu().numpy()
            if (
                self._sourceless_reader is not None
                and col.stream not in self._seq_lengths
                and (values.ndim >= 3 or (values.ndim == 2 and col.dtype is None))
            ):
                # a [B, T, C] leaf, or a [B, T] single-field copy: per-token, but the
                # reader gives no file token length (no source, no declared length)
                raise ConfigError(
                    f"H5OutputSink: column {col.key!r} on stream {col.stream!r} is "
                    "per-token (a sequence-stream input copy or per-token task output) "
                    "and needs the file token length — an H5StructuredReader-style "
                    "reader exposing an h5py-openable source (reader.h5_source), or a "
                    f"declared fixed-length raw.{col.stream} — {self._sourceless_reader} "
                    "provides neither; copy only global-stream variables with this reader"
                )
            # the producer leaf may be [B] / [B, L] (collapsed index columns) or
            # [B, C] / [B, L, C] — give u2s an explicit trailing channel axis so a
            # single-suffix column packs the same as the v1 eval path
            if values.ndim == 1 or (values.ndim == 2 and col.stream in self._seq_lengths):
                values = values[..., np.newaxis]
            values = np.ascontiguousarray(values)
            dtype = (
                np.dtype([(name, values.dtype) for name in col.column_names(self._run_name)])
                if col.dtype is None
                else col.np_dtype(self._run_name)
            )
            arr = u2s(values, dtype)
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

    # -- setup plumbing ----------------------------------------------------------

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

    def _merge_columns(self, streams: tuple[str, ...], group_datasets: Mapping[str, str]) -> None:
        """Validate column names/owners and resolve the FILE-dataset name per group.

        Column order: output columns in declaration order (input copies
        included — see the class docstring), then the pad mask, then the
        object-group columns. Output groups are named after the FILE dataset.
        A column collision or an output leaf for an unknown stream raises
        `ConfigError` naming both contributors. Dtypes/shapes are NOT resolved
        here — the first `consume` learns them from the actual batch.
        """
        owners: dict[tuple[str, str], str] = {}
        seen_streams: dict[str, None] = {}

        def _add(stream: str, names: list[str], who: str) -> None:
            if stream not in streams and stream not in self._object_shapes:
                raise ConfigError(
                    f"H5OutputSink: {who} targets unknown group {stream!r} — reader streams "
                    f"are {list(streams)} and object groups are {sorted(self._object_shapes)} "
                    "(declare a non-reader group's `shape` in object_groups)"
                )
            seen_streams.setdefault(stream, None)
            for field_name in names:
                if (other := owners.get((stream, field_name))) is not None:
                    raise ConfigError(
                        f"column {field_name!r} in group {stream!r} is declared by {other} AND "
                        f"{who} — rename one output (collision check)"
                    )
                owners[stream, field_name] = who

        for col in self._ensure_columns():
            _add(col.stream, col.column_names(self._run_name), f"output {col.key!r}")
        for stream in self._mask_streams:
            _add(stream, ["mask"], f"pad mask[{stream!r}]")
        # object-group columns come last; same `_add` uniqueness/attribution
        # check as reader columns. Dead code when object_groups is empty.
        for group in self._object_groups:
            for fld in group.fields:
                who = f"object_groups[{group.name!r}].{fld.leaf!r}"
                _add(group.name, fld.column_names(self._run_name), who)
        if not seen_streams:
            raise ConfigError("H5OutputSink declares no output columns at all")
        self._group_of = {stream: group_datasets.get(stream, stream) for stream in seen_streams}

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
