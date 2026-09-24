"""`MultiSampleReader` — combines N labelled samples into one proportionally
stratified, per-event-labelled stream over sub-`Reader`s.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.readers.stream import pad_fill
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.planner import PlanStep
from salt.graph.spec import IO, Mode, flatten_spec, unflatten_spec
from salt.schema import GroupSchema, Schema
from salt.utils.logging import get_logger

__all__ = ["MultiSampleReader", "SampleConfig"]
_LOG = get_logger(__name__)
_STAGE_KEYS = ("train", "val", "test")
_EXHAUSTION_LOG_EVERY = 50
_Pth = str | Path
_SegOut = list[tuple[int, np.ndarray, dict[str, np.ndarray]]]


@dataclass
class SampleConfig:
    """One labelled sample in a `MultiSampleReader`.

    Parameters
    ----------
    name : str
        Sample name, unique across samples.
    label : int
        Event-level class index for this sample's events (``>= 0``).
    reader : Reader
        Sub-`Reader`; produced streams/fields must match every other sample.
    sources : Mapping[str, str | Path] | None, optional
        Per-stage source override, by default ``None`` (keep the configured file).
    """

    name: str
    label: int
    reader: Reader
    sources: Mapping[str, _Pth] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("SampleConfig: 'name' must be non-empty")
        if not isinstance(self.label, int) or isinstance(self.label, bool) or self.label < 0:
            raise ConfigError(f"SampleConfig {self.name!r}: 'label' must be a non-negative int")
        if not isinstance(self.reader, Reader):
            raise ConfigError(f"SampleConfig {self.name!r}: 'reader' must be a Reader")
        if self.sources is not None and (bad := set(self.sources) - set(_STAGE_KEYS)):
            raise ConfigError(f"SampleConfig {self.name!r}: bad stage keys {sorted(bad)}")


class MultiSampleReader(Reader):
    """Combine N labelled samples into one proportionally-stratified stream.

    Wraps sub-`Reader`s (one per sample, identical produced schemas) into a
    proportional round-robin interleaved index; reads decompose a window into
    the fewest per-sample contiguous sub-reads (`_segments`) and inject a
    per-event sample label as a scalar field on one (global_object) stream.

    Parameters
    ----------
    samples : Sequence[SampleConfig | Mapping]
        The labelled samples (``{name, label, reader, sources?}``).
    label_stream : str, optional
        Scalar stream the injected label is added to, by default ``"event"``.
    label_field : str, optional
        Field name of the injected label, by default ``"process"``.
    interleave_block : int, optional
        Rows per round-robin turn, by default 1 (larger values change row
        placement, not per-batch proportions).

    Raises
    ------
    ConfigError, SchemaError
        Bad config (empty/duplicate samples, collisions) or incompatible
        sub-reader schemas / label_stream, respectively.
    """

    def __init__(
        self,
        samples: Sequence[SampleConfig | Mapping[str, Any]],
        label_stream: str = "event",
        label_field: str = "process",
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
        self.interleave_block = int(interleave_block)
        if self.interleave_block < 1:
            raise ConfigError(f"interleave_block must be >= 1, got {interleave_block!r}")
        self.schema: Schema | None = None
        self._lens: list[int] | None = None
        self._num_rows: int | None = None
        self._sample_of: np.ndarray | None = None
        self._local_of: np.ndarray | None = None
        self._streams: tuple[str, ...] | None = None
        self._jagged: dict[str, bool] | None = None
        self._debug_reads: int = 0
        self._served: dict[int, int] = {}

    @staticmethod
    def _parse_sample(cfg: SampleConfig | Mapping[str, Any]) -> SampleConfig:
        """Normalise one sample entry to a `SampleConfig`."""
        if isinstance(cfg, SampleConfig):
            return cfg
        cfg = dict(cfg or {})
        if bad := set(cfg) - {"name", "label", "reader", "sources"}:
            raise ConfigError(f"sample: unknown config keys {sorted(bad)}")
        if missing := {"name", "label", "reader"} - set(cfg):
            raise ConfigError(f"sample: missing required keys {sorted(missing)}")
        return SampleConfig(str(cfg["name"]), int(cfg["label"]), cfg["reader"], cfg.get("sources"))

    @property
    def streams(self) -> tuple[str, ...]:
        return self._streams if self._streams is not None else self.samples[0].reader.streams

    @property
    def groups(self) -> Any:
        return getattr(self.samples[0].reader, "groups", None)

    def declare_io(self, mode: Mode) -> IO:
        """Merge the sub-readers' (identical) produces, adding the injected label field."""
        lf, ls = self.label_field, self.label_stream
        flat = dict(flatten_spec(self.samples[0].reader.declare_io(mode).produces))
        key = f"raw.{ls}"
        if key not in flat:
            raise SchemaError(f"label_stream {ls!r} not produced")
        spec = flat[key]
        if spec.shape != ("B",):
            raise SchemaError(f"label_stream {ls!r} must be SCALAR, got {spec.shape}")
        existing = tuple(spec.fields) if spec.fields is not None else ()
        if lf in existing:
            raise ConfigError(f"label_field {lf!r} collides on {ls!r}")
        new_fields = (*existing, lf) if spec.fields is not None else None
        flat[key] = replace(spec, fields=new_fields)
        return IO(produces=unflatten_spec(flat))

    def prepare(self) -> None:
        """Prepare every sub-reader, validate schema-compat, build the interleave (idempotent)."""
        if self._sample_of is not None:
            return
        for s in self.samples:
            s.reader.prepare()
        self._validate_schema_compat()
        self._build_index()

    def _validate_schema_compat(self) -> None:
        """Assert identical sub-reader signatures; build `self.schema` from the reference."""

        def sig_of(reader: Reader) -> dict[str, Any]:
            flat = flatten_spec(reader.declare_io(Mode.FIT).produces)
            out: dict[str, Any] = {}
            for stream in reader.streams:
                if (key := f"raw.{stream}") not in flat:
                    continue
                gs = reader.schema_group(stream)
                flds = gs.fields.items() if gs else ()
                fs = tuple(sorted((f, str(np.dtype(t))) for f, t in flds))
                out[stream] = (flat[key].shape != ("B",), fs)
            return out

        lf, ls = self.label_field, self.label_stream
        ref = self.samples[0]
        ref_sig = sig_of(ref.reader)
        for s in self.samples[1:]:
            sig = sig_of(s.reader)
            if set(sig) != set(ref_sig):
                raise SchemaError(f"streams differ: {s.name!r} vs {ref.name!r}")
            for stream in ref_sig:
                if sig[stream] != ref_sig[stream]:
                    raise SchemaError(f"{stream!r} not identical: {s.name!r} vs {ref.name!r}")
        self._jagged = {stream: jagged for stream, (jagged, _f) in ref_sig.items()}
        self._streams = tuple(ref.reader.streams)
        if ls not in self._jagged:
            raise SchemaError(f"label_stream {ls!r} not a produced stream")
        if self._jagged[ls]:
            raise SchemaError(f"label_stream {ls!r} is JAGGED, must be SCALAR")
        if ref.reader.schema is None:
            return
        groups: dict[str, GroupSchema] = {}
        for stream in self._streams:
            gs = ref.reader.schema_group(stream)
            fields = dict(gs.fields) if gs is not None else {}
            if stream == ls:
                if lf in fields:
                    raise ConfigError(f"label_field {lf!r} collides on {ls!r}")
                fields[lf] = "int64"
            groups[stream] = GroupSchema(fields=fields)
        self.schema = Schema(groups=groups)

    def _build_index(self) -> None:
        """Deterministic largest-remainder round-robin interleave over sub-reader lengths."""
        lens = self._lens = [len(s.reader) for s in self.samples]
        n = self._num_rows = sum(lens)
        sample_of, local_of = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        emitted, cursor = [0] * len(lens), [0] * len(lens)
        j = 0
        while j < n:
            best_id, best_deficit = -1, -np.inf
            for i, n_i in enumerate(lens):
                if emitted[i] >= n_i:
                    continue
                deficit = (j + 1) * (n_i / n) - emitted[i]
                if deficit > best_deficit:
                    best_id, best_deficit = i, deficit
            take = min(self.interleave_block, lens[best_id] - emitted[best_id], n - j)
            sample_of[j : j + take] = best_id
            local_of[j : j + take] = np.arange(cursor[best_id], cursor[best_id] + take)
            cursor[best_id] += take
            emitted[best_id] += take
            j += take
        self._sample_of, self._local_of = sample_of, local_of

    def __len__(self) -> int:
        self.prepare()
        return int(self._num_rows)  # type: ignore[arg-type]

    def schema_group(self, stream: str) -> GroupSchema | None:
        if self.schema is None:
            self.prepare()
        return self.schema.groups.get(stream) if self.schema is not None else None

    def config_fingerprint(self) -> dict[str, Any]:
        ls, lf, ib = self.label_stream, self.label_field, self.interleave_block
        samples = [
            {
                "name": s.name,
                "label": s.label,
                "reader": type(s.reader).__name__,
                "config": s.reader.config_fingerprint(),
            }
            for s in self.samples
        ]
        return {"label_stream": ls, "label_field": lf, "interleave_block": ib, "samples": samples}

    def label_universe(self) -> tuple[str, ...] | None:
        if self.schema is None:
            self.prepare()
        if self.schema is None:
            return None
        groups = self.schema.groups.items()
        return tuple(f"labels.{s}.{f}" for s, gs in groups for f in gs.fields if f != "valid")

    def with_source(
        self, filename: _Pth, num: int = -1, stage: str | None = None
    ) -> MultiSampleReader:
        """Clone for a stage, re-sourcing each sub-reader from its per-stage source."""
        del filename
        new_samples = []
        for s in self.samples:
            sub = s.reader
            src = s.sources.get(stage) if s.sources is not None and stage is not None else None  # type: ignore[assignment]
            if src is not None:
                sub = sub.with_source(filename=src, num=num, stage=stage)
            elif num != -1:
                with contextlib.suppress(AttributeError, NotImplementedError):
                    sub = sub.with_source(sub.source_path, num=num, stage=stage)  # type: ignore[attr-defined]
            new_samples.append(SampleConfig(s.name, s.label, reader=sub, sources=s.sources))
        return self._clone(new_samples)

    def sources(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for s in self.samples:
            for src in s.reader.sources():
                seen.setdefault(str(src), src)
        return list(seen.values())

    def restage(self, root: _Pth) -> MultiSampleReader:
        root = Path(root)
        return self._clone([
            SampleConfig(s.name, s.label, s.reader.restage(root), s.sources) for s in self.samples
        ])

    def _clone(self, new_samples: list[SampleConfig]) -> MultiSampleReader:
        clone = MultiSampleReader(
            new_samples, self.label_stream, self.label_field, interleave_block=self.interleave_block
        )
        clone.name = self.name
        return clone

    def bind(self, ctx: WorkerCtx) -> None:
        """Prepare + forward demand-narrowing to every sub-reader, stripping the injected label."""
        self.prepare()
        self._served, self._debug_reads = {}, 0
        lf = self.label_field
        forwarded = dict(ctx.read_fields)
        if (ls := self.label_stream) in forwarded:
            forwarded[ls] = {f: w for f, w in forwarded[ls].items() if f != lf}
        sub_ctx = WorkerCtx(ctx.mode, forwarded, ctx.seed, ctx.worker_id, ctx.num_workers, ctx.step)
        for s in self.samples:
            s.reader.bind(sub_ctx)

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        out = super().read_fields(step)
        if (ls := self.label_stream) in out:
            out[ls] = {f: w for f, w in out[ls].items() if f != self.label_field}
            if not out[ls]:
                del out[ls]
        return out

    def _segments(self, rows: slice) -> list[tuple[int, slice, np.ndarray]]:
        """Decompose a window into the fewest contiguous per-sample sub-reads."""
        assert self._sample_of is not None
        assert self._local_of is not None
        so, lo_ = self._sample_of, self._local_of  # type: ignore[assignment]
        ws, wl = so[rows.start : rows.stop], lo_[rows.start : rows.stop]
        segments: list[tuple[int, slice, np.ndarray]] = []
        for sid in np.unique(ws):
            pos = np.flatnonzero(ws == sid)
            loc = wl[pos]
            for chunk in np.split(np.arange(pos.size), np.flatnonzero(np.diff(loc) != 1) + 1):
                first = int(loc[chunk[0]])
                segments.append((int(sid), slice(first, first + chunk.size), pos[chunk]))
        segments.sort(key=lambda seg: int(seg[2][0]))
        return segments

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one combined contiguous window: per-segment sub-reads, scatter, inject label."""
        if self._sample_of is None:
            self.bind(WorkerCtx(mode, {}, 0))
        segments = self._segments(rows)
        seg_out = [(sid, pos, self.samples[sid].reader.read(sl, mode)) for sid, sl, pos in segments]
        if _LOG.isEnabledFor(logging.DEBUG):
            self._log_exhaustion(segments)
        return self._combine(seg_out, rows.stop - rows.start, mode, rows.start, rows.stop)

    def row_blocks(self) -> list[RowBlock]:
        self.prepare()
        return [
            RowBlock(sid, blk.start, blk.stop)
            for sid, s in enumerate(self.samples)
            for blk in s.reader.row_blocks()
        ]

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        """Read one sub-reader's contiguous range and inject that sample's label."""
        self.prepare()
        produced = self.samples[block.group].reader.read(slice(block.start, block.stop), mode)
        positions = np.arange(block.n_rows, dtype=np.int64)
        seg_out = [(block.group, positions, produced)]
        return self._combine(seg_out, block.n_rows, mode, block.start, block.stop)

    def _combine(
        self, seg_out: _SegOut, b: int, mode: Mode, lo: int, hi: int
    ) -> dict[str, np.ndarray]:
        """Scatter each segment into combined ``(B, ...)`` outputs; inject `label_field`."""
        assert self._jagged is not None
        out: dict[str, np.ndarray] = {}
        for stream in self._streams or ():
            raw_key = f"raw.{stream}"
            blocks = [(pos, sid, p[raw_key]) for sid, pos, p in seg_out]
            ref = blocks[0][2]
            names = list(ref.dtype.names or ())
            base = [(nm, ref.dtype[nm]) for nm in names]
            if self._jagged[stream]:
                t = max(int(blk.shape[1]) for _p, _s, blk in blocks)
                combined = np.zeros((b, t), dtype=np.dtype(base))
                for nm in names:
                    if fill := pad_fill(np.dtype(ref.dtype[nm])):
                        combined[nm][:] = fill
                for pos, _sid, blk in blocks:
                    for nm in names:
                        combined[nm][pos, : blk.shape[1]] = blk[nm]
                valid = combined["valid"] if "valid" in names else np.zeros((b, t), dtype=bool)
                out[raw_key], out[f"masks.{stream}"] = combined, ~valid
            else:
                if stream == self.label_stream:
                    base.append((self.label_field, np.dtype("int64")))
                combined = np.empty((b,), dtype=np.dtype(base))
                for pos, sid, blk in blocks:
                    for nm in names:
                        combined[nm][pos] = blk[nm]
                    if stream == self.label_stream:
                        combined[self.label_field][pos] = self.samples[sid].label
                out[raw_key] = combined
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([lo, hi], dtype=np.int64)
        return out

    def _log_exhaustion(self, segments: list[tuple[int, slice, np.ndarray]]) -> None:
        """Accumulate + (at cadence) log the per-worker, per-sample rows-served counter."""
        from torch.utils.data import get_worker_info

        self._debug_reads += 1
        n = self._debug_reads
        for sid, _sl, pos in segments:
            self._served[sid] = self._served.get(sid, 0) + int(pos.size)
        if (n - 1) % _EXHAUSTION_LOG_EVERY != 0:
            return
        worker = "main" if (info := get_worker_info()) is None else str(info.id)
        lens = self._lens or []
        parts = [
            f"{self.samples[sid].name}={served}/{lens[sid]} ({served / lens[sid]:.1%})"
            for sid, served in sorted(self._served.items())
            if lens[sid] > 0
        ]
        if parts:
            _LOG.debug("exhaustion [worker=%s] after %d reads: %s", worker, n, ", ".join(parts))

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient index/probe state so the reader pickles under spawn."""
        state = self.__dict__.copy()
        ks = ("_lens", "_num_rows", "_sample_of", "_local_of", "_streams", "_jagged", "schema")
        state |= dict.fromkeys(ks)
        state["_debug_reads"], state["_served"] = 0, {}
        return state
