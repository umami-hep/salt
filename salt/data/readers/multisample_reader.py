"""`MultiSampleReader` — combines N labelled samples into one proportionally
stratified stream (largest-remainder interleave), injecting a per-event
sample label; each sub-read stays a contiguous per-file slice.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.readers.stream import pad_fill
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.planner import PlanStep
from salt.graph.spec import IO, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.schema import GroupSchema, Schema
from salt.utils.logging import get_logger

__all__ = ["MultiSampleReader", "SampleConfig"]

_LOG = get_logger(__name__)

_STAGE_KEYS = ("train", "val", "test")

_EXHAUSTION_LOG_EVERY = 50
"""Exhaustion logging emission cadence (1st call, then every 50th); the
served-rows counter itself updates on every `read()`."""


@dataclass
class SampleConfig:
    """One labelled sample in a `MultiSampleReader`.

    Parameters
    ----------
    name : str
        Sample name, unique across samples (used for error messages).
    label : int
        The event-level class index injected for every event of this sample
        (e.g. signal=1, background=0). Must be ``>= 0``.
    reader : Reader
        The sub-`Reader` that reads this sample. Its produced streams/fields must
        be identical across all samples (validated in `MultiSampleReader.prepare`).
    sources : Mapping[str, str | Path] | None, optional
        Per-stage source ``{train: ..., val: ..., test: ...}`` for this sample's
        sub-reader. ``None`` keeps the sub-reader's already-configured filename
        for every stage.
    """

    name: str
    label: int
    reader: Reader
    sources: Mapping[str, str | Path] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("SampleConfig: 'name' must be a non-empty string")
        if not isinstance(self.label, int) or isinstance(self.label, bool) or self.label < 0:
            raise ConfigError(
                f"SampleConfig {self.name!r}: 'label' must be a non-negative int, "
                f"got {self.label!r}"
            )
        if not isinstance(self.reader, Reader):
            raise ConfigError(
                f"SampleConfig {self.name!r}: 'reader' must be a Reader instance, got "
                f"{type(self.reader).__name__}"
            )
        if self.sources is not None:
            unknown = set(self.sources) - set(_STAGE_KEYS)
            if unknown:
                raise ConfigError(
                    f"SampleConfig {self.name!r}: unknown stage keys in 'sources' "
                    f"{sorted(unknown)} — expected a subset of {list(_STAGE_KEYS)}"
                )


class MultiSampleReader(Reader):
    """Combine N labelled samples into one proportionally-stratified stream.

    Wraps a list of sub-`Reader`s (one per sample) with identical produced
    schemas, builds a proportional round-robin interleaved combined index, reads
    contiguous global windows by decomposing them into the fewest per-sample
    contiguous sub-reads (`_segments`), and injects a per-event sample label as a
    scalar field on a chosen (global_object) stream. It is itself a `Reader`.

    Parameters
    ----------
    samples : Sequence[SampleConfig | Mapping]
        The labelled samples (``{name, label, reader, sources?}``).
    label_stream : str, optional
        The scalar (global_object) stream the injected per-event label is added
        to, by default ``"event"``. Must be a non-jagged stream produced by every
        sub-reader (validated in `prepare`).
    label_field : str, optional
        The field name of the injected label inside ``raw.<label_stream>``, by
        default ``"process"``. Must not collide with an existing sub-reader field.
    seed : int, optional
        Seed for the (deterministic) interleave order, by default 42.
    interleave_block : int, optional
        Rows emitted per turn of the round robin, by default 1 (row-granular —
        the historical behaviour, bit-for-bit). Larger values interleave in
        blocks of `interleave_block` rows, which leaves the per-batch sample
        proportions intact but CHANGES which rows land in which batch, so it is
        not a bitwise-neutral setting. Read granularity does not depend on it:
        `_segments` already coalesces each sample's contribution to a window
        into one sub-read per contiguous local stretch.

    Raises
    ------
    ConfigError
        On an empty / malformed samples list, duplicate sample names, a
        ``label_field`` that collides with an existing field, or an
        ``interleave_block`` below 1.
    SchemaError
        When the sub-readers' produced streams/fields/jaggedness are not identical,
        or ``label_stream`` is absent / not a scalar stream.
    """

    def __init__(
        self,
        samples: Sequence[SampleConfig | Mapping[str, Any]],
        label_stream: str = "event",
        label_field: str = "process",
        seed: int = 42,
        interleave_block: int = 1,
    ) -> None:
        super().__init__()
        if not samples:
            raise ConfigError("MultiSampleReader needs at least one sample")
        self.samples: list[SampleConfig] = [self._parse_sample(s) for s in samples]
        names = [s.name for s in self.samples]
        if len(set(names)) != len(names):
            raise ConfigError(f"MultiSampleReader: duplicate sample names {names}")
        self.label_stream = str(label_stream)
        self.label_field = str(label_field)
        self.seed = int(seed)
        self.interleave_block = int(interleave_block)
        if self.interleave_block < 1:
            raise ConfigError(
                f"MultiSampleReader: interleave_block must be >= 1, got {interleave_block!r}"
            )
        # No own on-disk schema artifact; built in prepare() from the sub-readers'.
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see __getstate__)
        self._lens: list[int] | None = None  # per-sample row count
        self._num_rows: int | None = None
        self._sample_of: np.ndarray | None = None  # global pos -> sample id
        self._local_of: np.ndarray | None = None  # global pos -> local row in that sample
        self._streams: tuple[str, ...] | None = None
        self._jagged: dict[str, bool] | None = None  # stream -> jagged?
        self._debug_reads = 0  # count of read() calls, for the exhaustion-log cadence
        # rows of each sample this worker has actually served; DEBUG-only, reset on bind
        self._served: dict[int, int] = {}

    @staticmethod
    def _parse_sample(cfg: SampleConfig | Mapping[str, Any]) -> SampleConfig:
        """Normalise one sample entry to a `SampleConfig`."""
        if isinstance(cfg, SampleConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - {"name", "label", "reader", "sources"}
        if unknown:
            raise ConfigError(
                f"sample: unknown config keys {sorted(unknown)} — expected "
                "name/label/reader/sources"
            )
        for required in ("name", "label", "reader"):
            if required not in cfg:
                raise ConfigError(f"sample: missing required key {required!r}")
        return SampleConfig(
            name=str(cfg["name"]),
            label=int(cfg["label"]),
            reader=cfg["reader"],
            sources=cfg.get("sources"),
        )

    @property
    def streams(self) -> tuple[str, ...]:
        """The combined stream names (identical across samples), in config order."""
        if self._streams is not None:
            return self._streams
        # config-only fast path: streams are the first sub-reader's (prepare()
        # asserts every sample agrees); this needs no file I/O.
        return self.samples[0].reader.streams

    @property
    def groups(self) -> Any:
        """Delegate the per-stream group config surface to the first sub-reader (all
        sub-readers share an identical schema).
        """
        return getattr(self.samples[0].reader, "groups", None)

    def declare_io(self, mode: Mode) -> IO:
        """Declare the merged sub-reader produces + the injected label field.

        Mirrors the (identical) sub-readers' ``raw.*``/``masks.*``/``meta.rows``;
        the injected ``raw.<label_stream>`` gains the ``label_field`` scalar.
        """
        base = self.samples[0].reader.declare_io(mode)
        flat = dict(flatten_spec(base.produces))
        key = f"raw.{self.label_stream}"
        if key not in flat:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} is not a produced "
                f"stream of the sub-readers ({sorted(self.samples[0].reader.streams)})"
            )
        spec = flat[key]
        if spec.shape != ("B",):
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} must be a SCALAR "
                f"(global_object [B]) stream to carry the injected event label, but its shape "
                f"is {spec.shape}"
            )
        # extend the label_stream's field set with the injected label field
        existing = tuple(spec.fields) if spec.fields is not None else ()
        if self.label_field in existing:
            raise ConfigError(
                f"MultiSampleReader: injected label_field {self.label_field!r} collides with an "
                f"existing field on stream {self.label_stream!r}"
            )
        flat[key] = TensorSpec(
            shape=spec.shape,
            kind=spec.kind,
            dtype=spec.dtype,
            modes=spec.modes,
            fields=(*existing, self.label_field) if spec.fields is not None else None,
        )
        return IO(produces=unflatten_spec(flat))

    def prepare(self) -> None:
        """Prepare every sub-reader, validate schema-compat, build the interleave (idempotent)."""
        if self._sample_of is not None:
            return
        for s in self.samples:
            s.reader.prepare()
        self._validate_schema_compat()
        self._build_index()

    def _produced_signature(self, reader: Reader) -> dict[str, Any]:
        """The reader's produced-stream signature for schema-compat comparison:
        ``{stream: (jagged, (field, dtype) pairs sorted by field)}``.
        """
        io = reader.declare_io(Mode.FIT)
        flat = flatten_spec(io.produces)
        sig: dict[str, Any] = {}
        for stream in reader.streams:
            key = f"raw.{stream}"
            if key not in flat:
                continue
            jagged = flat[key].shape != ("B",)
            gschema = reader.schema_group(stream)
            if gschema is None:
                # no schema artifact — compare on (jagged, field-name set) only
                fields: tuple[tuple[str, str], ...] = ()
            else:
                fields = tuple(sorted((f, str(np.dtype(dt))) for f, dt in gschema.fields.items()))
            sig[stream] = (jagged, fields)
        return sig

    def _validate_schema_compat(self) -> None:
        """Assert all sub-readers produce identical streams/fields/jaggedness (raises
        `SchemaError` on divergence, `ConfigError` if the injected label collides).
        """
        ref = self.samples[0]
        ref_sig = self._produced_signature(ref.reader)
        for s in self.samples[1:]:
            sig = self._produced_signature(s.reader)
            if set(sig) != set(ref_sig):
                raise SchemaError(
                    f"MultiSampleReader: sample {s.name!r} produces streams {sorted(sig)} but "
                    f"sample {ref.name!r} produces {sorted(ref_sig)} — sub-reader produced "
                    "streams must be IDENTICAL so batches concatenate"
                )
            for stream in ref_sig:
                if sig[stream] != ref_sig[stream]:
                    raise SchemaError(
                        f"MultiSampleReader: stream {stream!r} differs between sample "
                        f"{s.name!r} ({sig[stream]}) and {ref.name!r} ({ref_sig[stream]}) — "
                        "produced fields/dtypes/jaggedness must be identical"
                    )
        self._jagged = {stream: jagged for stream, (jagged, _f) in ref_sig.items()}
        self._streams = tuple(ref.reader.streams)
        if self.label_stream not in self._jagged:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} not a produced stream "
                f"(produced: {sorted(self._jagged)})"
            )
        if self._jagged[self.label_stream]:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} is a JAGGED/sequence "
                "stream; the injected event label must land on a SCALAR (global_object) stream"
            )
        ref_schema = ref.reader.schema
        if ref_schema is not None:
            groups: dict[str, GroupSchema] = {}
            for stream in self._streams:
                gs = ref.reader.schema_group(stream)
                fields = dict(gs.fields) if gs is not None else {}
                if stream == self.label_stream:
                    if self.label_field in fields:
                        raise ConfigError(
                            f"MultiSampleReader: injected label_field {self.label_field!r} "
                            f"collides with an existing field on {self.label_stream!r}"
                        )
                    fields[self.label_field] = "int64"
                groups[stream] = GroupSchema(fields=fields)
            self.schema = Schema(groups=groups)

    def _build_index(self) -> None:
        """Build the proportional round-robin interleaved combined index (largest-remainder
        apportionment): at each turn, emit `interleave_block` rows of the sample with the
        largest deficit ``(j+1)*n_i/N - emitted_i``; every prefix's per-sample counts stay
        within ±`interleave_block` of ideal (±1 at the default block of 1).
        """
        assert self._streams is not None
        lens = [len(s.reader) for s in self.samples]
        self._lens = lens
        n = sum(lens)
        self._num_rows = n
        sample_of = np.empty(n, dtype=np.int64)
        local_of = np.empty(n, dtype=np.int64)
        emitted = [0] * len(self.samples)
        cursor = [0] * len(self.samples)
        block = self.interleave_block
        # Largest-remainder round-robin. At step j (0-based), the ideal cumulative
        # count for sample i after emitting (j+1) items is (j+1)*n_i/N. We emit the
        # sample with the largest positive deficit, skipping exhausted samples.
        j = 0
        while j < n:
            best_id = -1
            best_deficit = -np.inf
            target = j + 1
            for i, n_i in enumerate(lens):
                if emitted[i] >= n_i:
                    continue  # sample exhausted
                deficit = target * (n_i / n) - emitted[i]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_id = i
            take = min(block, lens[best_id] - emitted[best_id], n - j)
            sample_of[j : j + take] = best_id
            local_of[j : j + take] = np.arange(cursor[best_id], cursor[best_id] + take)
            cursor[best_id] += take
            emitted[best_id] += take
            j += take
        self._sample_of = sample_of
        self._local_of = local_of

    def __len__(self) -> int:
        """Return the combined row count = Σ len(sub-reader)."""
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The merged schema's group for one stream (built in `prepare`)."""
        if self.schema is None:
            self.prepare()
        if self.schema is None or stream not in self.schema.groups:
            return None
        return self.schema.groups.get(stream)

    def config_fingerprint(self) -> dict[str, Any]:
        """The interleave config plus every sub-reader's own fingerprint.

        Recursive because a change inside one sample changes what this reader
        serves just as surely as a change out here does.
        """
        return {
            "label_stream": self.label_stream,
            "label_field": self.label_field,
            "interleave_block": self.interleave_block,
            "samples": [
                {
                    "name": s.name,
                    "label": s.label,
                    "reader": type(s.reader).__name__,
                    "config": s.reader.config_fingerprint(),
                }
                for s in self.samples
            ],
        }

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe, including the injected label."""
        if self.schema is None:
            self.prepare()
        if self.schema is None:
            return None
        return tuple(
            f"labels.{stream}.{field}"
            for stream, gs in self.schema.groups.items()
            for field in gs.fields
            if field != "valid"
        )

    def with_source(
        self,
        filename: str | Path,
        num: int = -1,
        vds_path: str | Path | None = None,
        stage: str | None = None,
    ) -> MultiSampleReader:
        """Clone for a stage, re-sourcing EACH sub-reader from its per-stage source
        (``filename`` is ignored — meaningless for N samples). A sample with no
        per-stage source but a requested `num` re-sources onto its own current file.
        """
        del filename
        new_samples: list[SampleConfig] = []
        for s in self.samples:
            sub = s.reader
            src: str | Path | None = None
            if s.sources is not None and stage is not None:
                src = s.sources.get(stage)  # type: ignore[assignment]
            if src is not None:
                sub = sub.with_source(filename=src, num=num, vds_path=vds_path, stage=stage)
            elif num != -1:
                # no per-stage source, but a row cap was requested — re-source the
                # sub-reader onto its OWN configured file with the cap. Use the
                # sub-reader's source_path when resolvable; else leave as-is.
                try:
                    cur = sub.source_path  # type: ignore[attr-defined]
                    sub = sub.with_source(filename=cur, num=num, vds_path=vds_path, stage=stage)
                except (AttributeError, NotImplementedError):
                    pass
            new_samples.append(
                SampleConfig(name=s.name, label=s.label, reader=sub, sources=s.sources)
            )
        clone = MultiSampleReader(
            samples=new_samples,
            label_stream=self.label_stream,
            label_field=self.label_field,
            seed=self.seed,
            interleave_block=self.interleave_block,
        )
        clone.name = self.name
        return clone

    def sources(self) -> list[Path]:
        """The de-duplicated union of every sub-reader's source files.

        A multi-sample reader has no file of its own — its data is the N
        sub-readers' files. Order: sub-reader order, then each sub-reader's own
        source order; duplicates (a file shared between samples) appear once.
        """
        seen: dict[str, Path] = {}
        for s in self.samples:
            for src in s.reader.sources():
                seen.setdefault(str(src), src)
        return list(seen.values())

    def restage(self, root: str | Path) -> MultiSampleReader:
        """Restage by delegating to EACH sub-reader recursively (no single file to copy);
        a fresh `MultiSampleReader` is built over the restaged sub-readers.
        """
        root = Path(root)
        new_samples = [
            SampleConfig(
                name=s.name,
                label=s.label,
                reader=s.reader.restage(root),
                sources=s.sources,
            )
            for s in self.samples
        ]
        clone = MultiSampleReader(
            samples=new_samples,
            label_stream=self.label_stream,
            label_field=self.label_field,
            seed=self.seed,
            interleave_block=self.interleave_block,
        )
        clone.name = self.name
        return clone

    def bind(self, ctx: WorkerCtx) -> None:
        """Prepare + forward demand-narrowing to every sub-reader (the injected
        ``label_field`` is stripped, since it is not a disk field).
        """
        self.prepare()
        # a re-bind is a new epoch/stage: reset the exhaustion counters
        self._served = {}
        self._debug_reads = 0
        # strip the injected (non-disk) label field from the demand we forward,
        # so a sub-reader never tries to read raw.<label_stream>.<label_field>.
        forwarded = dict(ctx.read_fields)
        if self.label_stream in forwarded:
            stream_demand = {
                f: who for f, who in forwarded[self.label_stream].items() if f != self.label_field
            }
            forwarded[self.label_stream] = stream_demand
        sub_ctx = WorkerCtx(
            mode=ctx.mode,
            read_fields=forwarded,
            seed=ctx.seed,
            worker_id=ctx.worker_id,
            num_workers=ctx.num_workers,
            step=ctx.step,
        )
        for s in self.samples:
            s.reader.bind(sub_ctx)

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Per-stream raw fields demanded, minus the injected (non-disk) label field."""
        out = super().read_fields(step)
        if self.label_stream in out:
            out[self.label_stream] = {
                f: who for f, who in out[self.label_stream].items() if f != self.label_field
            }
            if not out[self.label_stream]:
                del out[self.label_stream]
        return out

    def _runs(self, rows: slice) -> list[tuple[int, slice, int, int]]:
        """Decompose a global contiguous window into ordered per-sample runs (maximal
        stretches of consecutive positions mapped to the same sample with consecutive
        local rows, so each sub-read is contiguous); each run is
        ``(sample_id, local_slice, out_lo, out_hi)``.
        """
        assert self._sample_of is not None
        assert self._local_of is not None
        start, stop = rows.start, rows.stop
        runs: list[tuple[int, slice, int, int]] = []
        j = start
        while j < stop:
            sid = int(self._sample_of[j])
            loc0 = int(self._local_of[j])
            k = j + 1
            # extend while same sample AND local rows stay consecutive
            while (
                k < stop
                and int(self._sample_of[k]) == sid
                and int(self._local_of[k]) == loc0 + (k - j)
            ):
                k += 1
            runs.append((sid, slice(loc0, loc0 + (k - j)), j - start, k - start))
            j = k
        return runs

    def _segments(self, rows: slice) -> list[tuple[int, slice, np.ndarray]]:
        """Decompose a global window into the FEWEST contiguous per-sample sub-reads.

        Unlike `_runs` (which splits at every sample change — ~750 one-row
        sub-reads for a 1,000-row two-sample window), this exploits that the
        local rows one sample contributes to a window are consecutive
        (`_build_index` hands them out from a per-sample cursor): one sub-read
        per contiguous local stretch, scattered back to the interleaved output
        positions. Contiguity is DERIVED, never assumed — a non-consecutive
        stretch simply splits.

        Each segment is ``(sample_id, local_slice, out_positions)``, where
        ``out_positions`` index into ``[0, stop - start)``. Segments are ordered
        by their first output position, so the decomposition is deterministic.
        """
        assert self._sample_of is not None
        assert self._local_of is not None
        start, stop = rows.start, rows.stop
        window_sample = self._sample_of[start:stop]
        window_local = self._local_of[start:stop]
        segments: list[tuple[int, slice, np.ndarray]] = []
        for sid in np.unique(window_sample):
            pos = np.flatnonzero(window_sample == sid)
            loc = window_local[pos]
            breaks = np.flatnonzero(np.diff(loc) != 1) + 1
            for chunk in np.split(np.arange(pos.size), breaks):
                first = int(loc[chunk[0]])
                segments.append((int(sid), slice(first, first + chunk.size), pos[chunk]))
        segments.sort(key=lambda seg: int(seg[2][0]))
        return segments

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one combined contiguous window: per-segment sub-reads, scatter, inject label.

        Jagged streams pad to the max served ``T`` across the segments in this window.
        """
        if self._sample_of is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))  # standalone (tests)
        assert self._streams is not None
        assert self._jagged is not None
        start, stop = rows.start, rows.stop
        b = stop - start
        segments = self._segments(rows)

        # read every segment once (a segment yields all streams)
        seg_out: list[tuple[int, np.ndarray, dict[str, np.ndarray]]] = []
        for sid, local_slice, positions in segments:
            produced = self.samples[sid].reader.read(local_slice, mode)
            seg_out.append((sid, positions, produced))
        if _LOG.isEnabledFor(logging.DEBUG):
            self._log_exhaustion(segments)

        out: dict[str, np.ndarray] = {}
        for stream in self._streams:
            jagged = self._jagged[stream]
            raw_key = f"raw.{stream}"
            blocks = [(positions, sid, p[raw_key]) for sid, positions, p in seg_out]
            if jagged:
                out[raw_key], combined_valid = self._combine_jagged(blocks, b)
                out[f"masks.{stream}"] = ~combined_valid
            else:
                combined = self._combine_scalar(blocks, b, stream)
                out[raw_key] = combined
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([start, stop], dtype=np.int64)
        return out

    def row_blocks(self) -> list[RowBlock]:
        """Every sub-reader's blocks, tagged with its sample id.

        Deliberately does NOT build the global interleaved index (one int64 pair
        per corpus row — the object the streaming path exists to avoid holding
        per process); the per-shard proportional interleave is rebuilt from
        these blocks' row counts instead.
        """
        self.prepare()
        return [
            RowBlock(group=sid, start=block.start, stop=block.stop)
            for sid, sample in enumerate(self.samples)
            for block in sample.reader.row_blocks()
        ]

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        """Read one sub-reader's contiguous range and inject that sample's label.

        The single-sample case of `read`: one segment covering every output row,
        run through the same combine helpers so the produced dtypes (including
        the injected ``label_field``) are identical to the map-style path's.
        """
        self.prepare()
        assert self._streams is not None
        assert self._jagged is not None
        b = block.n_rows
        produced = self.samples[block.group].reader.read(slice(block.start, block.stop), mode)
        positions = np.arange(b, dtype=np.int64)
        out: dict[str, np.ndarray] = {}
        for stream in self._streams:
            raw_key = f"raw.{stream}"
            blocks = [(positions, block.group, produced[raw_key])]
            if self._jagged[stream]:
                out[raw_key], combined_valid = self._combine_jagged(blocks, b)
                out[f"masks.{stream}"] = ~combined_valid
            else:
                out[raw_key] = self._combine_scalar(blocks, b, stream)
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([block.start, block.stop], dtype=np.int64)
        return out

    def _log_exhaustion(self, segments: list[tuple[int, slice, np.ndarray]]) -> None:
        """Accumulate + (at cadence) log the per-worker, per-sample rows-served counter.

        Runs only behind the DEBUG guard in `read` (free at higher levels).
        Counts rows actually returned, never window positions (monotonic within
        an epoch even under ``shuffle=True``); per-worker (each worker owns its
        reader instance); reset by `bind`. Overlapping windows legitimately
        count twice — rows served, not distinct rows.
        """
        from torch.utils.data import get_worker_info

        self._debug_reads += 1
        for sid, _local_slice, positions in segments:
            self._served[sid] = self._served.get(sid, 0) + int(positions.size)
        if (self._debug_reads - 1) % _EXHAUSTION_LOG_EVERY != 0:
            return
        info = get_worker_info()
        worker = "main" if info is None else str(info.id)
        parts = []
        for sid, served in sorted(self._served.items()):
            total = self._lens[sid] if self._lens else 0
            if total <= 0:
                continue
            name = self.samples[sid].name
            parts.append(f"{name}={served}/{total} ({served / total:.1%})")
        if parts:
            _LOG.debug(
                "exhaustion [worker=%s] after %d reads: %s",
                worker,
                self._debug_reads,
                ", ".join(parts),
            )

    def _combine_scalar(
        self, blocks: list[tuple[np.ndarray, int, np.ndarray]], b: int, stream: str
    ) -> np.ndarray:
        """Scatter scalar ``(b_i,)`` blocks into a combined ``(B,)`` array at their output
        positions; for the ``label_stream`` an integer ``label_field`` is appended and filled
        per segment.
        """
        inject = stream == self.label_stream
        # field/dtype layout from the first block (identical across samples)
        ref = blocks[0][2]
        names = list(ref.dtype.names or ())
        dtype_fields = [(nm, ref.dtype[nm]) for nm in names]
        if inject:
            dtype_fields.append((self.label_field, np.dtype("int64")))
        combined = np.empty((b,), dtype=np.dtype(dtype_fields))
        for positions, sid, block in blocks:
            for nm in names:
                combined[nm][positions] = block[nm]
            if inject:
                combined[self.label_field][positions] = self.samples[sid].label
        return combined

    def _combine_jagged(
        self, blocks: list[tuple[np.ndarray, int, np.ndarray]], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scatter jagged ``(b_i, T_i)`` blocks into a combined ``(B, T)`` array at their
        output positions (T = max served multiplicity across segments; shorter blocks
        pad-extended via `pad_fill`).
        """
        ref = blocks[0][2]
        names = list(ref.dtype.names or ())
        t = max(int(block.shape[1]) for _pos, _sid, block in blocks)
        dtype_fields = [(nm, ref.dtype[nm]) for nm in names]
        combined = np.zeros((b, t), dtype=np.dtype(dtype_fields))
        # np.zeros already gives 0.0/0/False; only signed-int needs the -1 sentinel.
        for nm in names:
            fill = pad_fill(np.dtype(ref.dtype[nm]))
            if fill:  # non-zero/non-False fill (the signed-int -1 sentinel)
                combined[nm][:] = fill
        for positions, _sid, block in blocks:
            tb = block.shape[1]
            for nm in names:
                combined[nm][positions, :tb] = block[nm]
        valid = combined["valid"] if "valid" in names else np.zeros((b, t), dtype=bool)
        return combined, valid

    # -- pickling (fork is free; spawn re-binds in the worker) ----------------

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient index/probe state so the reader pickles under spawn."""
        state = self.__dict__.copy()
        state.update({
            "_lens": None,
            "_num_rows": None,
            "_sample_of": None,
            "_local_of": None,
            "_streams": None,
            "_jagged": None,
            "_debug_reads": 0,
            "_served": {},
            "schema": None,
        })
        return state
