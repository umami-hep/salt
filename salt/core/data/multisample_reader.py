"""`MultiSampleReader` — reader-agnostic proportional-stratified multi-sample reading (plan 02).

A `Reader` that combines N labelled samples — each read by ANY sub-`Reader`
(``EasyjetReader`` signal + ``EasyjetReader`` background now; ``H5StructuredReader``
/ a future ``PhysliteReader`` later) — into one stream, **proportionally
stratified** (a contiguous batch reflects each sample's share of the total), and
**injects a per-event sample label** the layer owns. First use: HH→4b (signal)
vs ttbar (background) **event-level** classification read DIRECTLY from per-sample
easyjet ROOT — no UPP/umami pre-resampling (design §2.4, §6.1).

This is the v2 modular-Reader boundary doing something v1 cannot: salt v1 relies
on UPP/umami to pre-resample all samples into ONE shuffled H5 with a per-object
label, then reads that one file. This layer does the mixing AT READ TIME from raw
per-sample files. Because it IS a `Reader`, ``Features`` / ``Labels`` /
``Normaliser`` / the model / ``salt2`` are all UNCHANGED — they cannot tell it
apart from a single-sample reader.

Config
------
``samples: [{name: signal, label: 1, reader: <sub-Reader>}, {name: background,
label: 0, reader: <sub-Reader>}, ...]``. Each ``reader`` is a full sub-`Reader`
instance (e.g. an ``EasyjetReader`` with its own files+branches). ``label`` is the
integer event-level class the layer injects for every event of that sample.
``label_stream`` (default ``"event"``) + ``label_field`` (default ``"process"``)
say WHERE the injected scalar label lands — it is added to the named SCALAR
(global_object) stream's structured array as a new integer field, so a downstream
``Labels`` processor exposes ``labels.<label_stream>.<label_field>`` for an
EVENT-level head with ZERO changes to ``Labels`` (``Labels.process`` reads
``raw.<stream>[field]``, processors.py:475).

Proportional interleave (the realization)
------------------------------------------
Pure global shuffle gives correct proportions but random disk access (slow). This
layer instead builds a **proportional round-robin interleave** of the combined
index at ``prepare`` time:

- each sample i contributes ``n_i`` rows, traversed in its own LOCAL order
  ``0..n_i-1`` (so per-sample reads stay contiguous-friendly slices);
- the combined index of length ``N = Σ n_i`` is filled by a **largest-remainder
  (Hamilton) round-robin**: walk global positions ``j = 0..N-1``; at each ``j`` pick
  the sample whose *running deficit* ``j·(n_i/N) - emitted_i`` is largest (ties →
  lowest sample id). This is the standard apportionment that keeps every prefix's
  per-sample counts within ``±1`` of the ideal ``j·n_i/N``.

**Per-window bound (the documented guarantee).** Because every prefix count
``emitted_i(j)`` satisfies ``|emitted_i(j) - j·n_i/N| < 1``, the count of sample i
in ANY contiguous window ``[a, b)`` (a batch is such a window) is
``emitted_i(b) - emitted_i(a)``, which lies within ``±2`` of the ideal
``(b-a)·n_i/N``. So a batch of size B has between ``floor(B·n_i/N) - 1`` and
``ceil(B·n_i/N) + 1`` events of sample i — bounded discrepancy, no long
single-class stretch, and the epoch-level mean is exactly the apportioned
``n_i``-share. A small-minority sample (expectation < 1 per batch) appears roughly
once every ``round(N/n_i)`` events, so it is present across the epoch and never
starved (its global positions are spread, not clustered).

The order is SEEDED at ``prepare`` and stable for the process lifetime. The
`Reader` interface has only ``prepare``/``bind``/``read`` (no epoch hook), so this
layer does NOT promise per-epoch reshuffling — that needs an explicit hook and is
a later enhancement. The dataloader's ``RandomBatchSampler`` still weakly shuffles
BATCH order on top, which preserves the per-batch composition.

read(slice)-only contract
-------------------------
The `Reader` API only guarantees ``read(slice, mode)`` — contiguous local slices,
NEVER scattered index arrays (sub-readers like ``EasyjetReader`` translate a slice
into ``entry_start``/``entry_stop`` file reads; a fancy-index read is not part of
the contract). So ``read(rows, mode)`` decomposes the global contiguous window
into an ORDERED list of ``(sample_id, contiguous local_slice)`` RUNS — each run is
a maximal stretch of consecutive global positions mapped to the same sample with
consecutive local rows — reads each run via its sub-reader's ``read(local_slice)``,
then CONCATENATES/REORDERS the per-run ``raw.<stream>`` / ``masks.<stream>`` blocks
back into combined batch (index) order. The injected ``process`` field is set per
run to that sample's label, so it is consistent with the index order and is a real
event-level scalar (never padded/masked away).

Stage binding — PER-READER stage-sourcing (plan 02 §"Stage binding")
--------------------------------------------------------------------
``GraphDataModule`` clones the reader prototype per stage via
``with_source(filename, num, vds_path)`` — ONE ``filename`` per stage
(datamodule.py:262). The plan's user-confirmed design makes the stage→data mapping
a PER-READER concern: each sample's sub-reader owns its own per-stage source. The
contract is evolved MINIMALLY and cleanly:

- ``Reader.with_source`` gains an OPTIONAL ``stage: str | None`` keyword (base ABC
  + H5StructuredReader + EasyjetReader accept-and-IGNORE it — the single-reader
  exp-01 path is byte-for-byte unchanged; ``stage`` is only meaningful here).
- ``GraphDataModule._make_dataset`` passes the stage string
  (``"train"``/``"val"``/``"test"``) it already knows.
- Each ``samples[i]`` MAY carry ``sources: {train: <src>, val: <src>, test: <src>}``
  (file path / dir / glob per stage). On ``with_source(filename, num, vds_path,
  stage)`` the `MultiSampleReader` IGNORES the single ``filename`` (it is
  meaningless for N samples) and re-sources EACH sub-reader from its own
  ``sources[stage]`` (falling back to the sub-reader's already-configured file when
  no per-stage source is given — useful for tests / a single-file smoke). The
  combined index is rebuilt for the new per-stage sub-readers. There is NO
  single-``train_file`` assumption — each sub-reader owns its sourcing, which is
  exactly why the multi-sample case fits cleanly. A cut-based per-stage selection
  (train/val by ``eventNumber`` parity) slots into a sub-reader WITHOUT changing
  this layer (the immediate next capability).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from salt.core.data.base import Reader, WorkerCtx
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.core.schema import GroupSchema, Schema

__all__ = ["MultiSampleReader", "SampleConfig"]

_LABEL_DTYPE = "int64"
"""Injected per-event sample label dtype (an event-level class index; int64 so a
downstream ``Labels`` int64-for-int policy is a no-op and a `ClassificationTaskModule`
consumes it directly)."""

# the stage keys the datamodule passes through with_source(stage=...)
_STAGE_KEYS = ("train", "val", "test")


@dataclass
class SampleConfig:
    """One labelled sample in a `MultiSampleReader` (plan 02).

    Parameters
    ----------
    name : str
        Human-readable sample name (``signal`` / ``background`` / ...). Must be
        unique across samples; used only for error messages and provenance.
    label : int
        The integer EVENT-level class index injected for every event of this
        sample (e.g. signal=1, background=0). Must be ``>= 0``.
    reader : Reader
        The sub-`Reader` that reads this sample (``EasyjetReader`` /
        ``H5StructuredReader`` / any `Reader`). Its PRODUCED streams/fields must be
        identical across all samples (validated in `MultiSampleReader.prepare`).
    sources : Mapping[str, str | Path] | None, optional
        Per-stage source for this sample's sub-reader:
        ``{train: ..., val: ..., test: ...}``. When the datamodule binds a stage
        the sub-reader is re-sourced from the matching entry. ``None`` (default)
        keeps the sub-reader's already-configured ``filename`` for every stage
        (single-file smoke / tests).
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
    """Combine N labelled samples into one proportionally-stratified stream (plan 02).

    Wraps a LIST of sub-`Reader`s (one per sample) with identical PRODUCED
    schemas, builds a proportional round-robin interleaved combined index, reads
    contiguous global windows by decomposing them into per-sample contiguous
    runs, and injects a per-event sample label as a scalar field on a chosen
    (global_object) stream. It is itself a `Reader`, so the rest of the v2
    pipeline is untouched.

    Parameters
    ----------
    samples : Sequence[SampleConfig | Mapping]
        The labelled samples (``{name, label, reader, sources?}``). At least two
        is the intended use (one is allowed — it degenerates to the sub-reader
        plus a constant injected label). Each ``reader`` is a full sub-`Reader`.
    label_stream : str, optional
        The SCALAR (global_object) stream the injected per-event label is added
        to, by default ``"event"``. Must be a non-jagged stream produced by every
        sub-reader (validated in `prepare`).
    label_field : str, optional
        The field name of the injected label inside ``raw.<label_stream>``, by
        default ``"process"``. Must not collide with an existing sub-reader field.
    seed : int, optional
        Seed for the (deterministic) interleave order, by default 42. The order is
        apportionment-deterministic; the seed only perturbs tie-breaking-free runs
        for reproducibility hooks and is otherwise a no-op on the composition.

    Raises
    ------
    ConfigError
        On an empty / malformed samples list, duplicate sample names, or a
        ``label_field`` that collides with an existing field.
    SchemaError
        When the sub-readers' PRODUCED streams/fields/jaggedness are not identical,
        or ``label_stream`` is absent / not a scalar stream.
    """

    def __init__(
        self,
        samples: Sequence[SampleConfig | Mapping[str, Any]],
        label_stream: str = "event",
        label_field: str = "process",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not samples:
            raise ConfigError("MultiSampleReader needs at least one sample (plan 02)")
        self.samples: list[SampleConfig] = [self._parse_sample(s) for s in samples]
        names = [s.name for s in self.samples]
        if len(set(names)) != len(names):
            raise ConfigError(f"MultiSampleReader: duplicate sample names {names}")
        self.label_stream = str(label_stream)
        self.label_field = str(label_field)
        self.seed = int(seed)
        # No own on-disk schema artifact; built in prepare() from the sub-readers'.
        self.schema: Schema | None = None
        # transient per-process state (never pickled, see __getstate__)
        self._lens: list[int] | None = None  # per-sample row count
        self._num_rows: int | None = None
        self._sample_of: np.ndarray | None = None  # global pos -> sample id
        self._local_of: np.ndarray | None = None  # global pos -> local row in that sample
        self._streams: tuple[str, ...] | None = None
        self._jagged: dict[str, bool] | None = None  # stream -> jagged?

    @staticmethod
    def _parse_sample(cfg: SampleConfig | Mapping[str, Any]) -> SampleConfig:
        """Normalise one sample entry to a `SampleConfig`.

        Returns
        -------
        SampleConfig
            The parsed sample config.

        Raises
        ------
        ConfigError
            On unknown keys or a missing required field.
        """
        if isinstance(cfg, SampleConfig):
            return cfg
        cfg = dict(cfg or {})
        unknown = set(cfg) - {"name", "label", "reader", "sources"}
        if unknown:
            raise ConfigError(
                f"sample: unknown config keys {sorted(unknown)} — expected "
                "name/label/reader/sources (plan 02)"
            )
        for required in ("name", "label", "reader"):
            if required not in cfg:
                raise ConfigError(f"sample: missing required key {required!r} (plan 02)")
        return SampleConfig(
            name=str(cfg["name"]),
            label=int(cfg["label"]),
            reader=cfg["reader"],
            sources=cfg.get("sources"),
        )

    # -- streams / declare_io (config-only where possible, design §2.2/§2.3) ---

    @property
    def streams(self) -> tuple[str, ...]:
        """The combined stream names (identical across samples), in config order.

        Returns
        -------
        tuple[str, ...]
            Stream names (the sub-readers' shared stream tuple).
        """
        if self._streams is not None:
            return self._streams
        # config-only fast path: the streams are the FIRST sub-reader's streams
        # (prepare() asserts every sample agrees). Sub-readers expose `streams`
        # without file I/O (it derives from their group config).
        return self.samples[0].reader.streams

    @property
    def groups(self) -> Any:
        """Delegate the per-stream group config surface to the first sub-reader.

        The writers callback (design §8) statically introspects a reader's
        ``groups[stream].global_object`` to split sequence vs global streams.
        Since all sub-readers share an identical produced schema, the FIRST
        sub-reader's ``groups`` is representative — the injected ``label_field``
        lands on an existing (scalar) stream, so it adds no new group. Exposing
        this keeps the multi-sample reader usable with the standard writers
        without any writer-side change.

        Returns
        -------
        Any
            The first sub-reader's ``groups`` mapping (group-config objects).
        """
        return getattr(self.samples[0].reader, "groups", None)

    def declare_io(self, mode: Mode) -> IO:
        """Declare the merged sub-reader produces + the injected label field.

        The produces mirror the (identical) sub-readers' ``raw.* / masks.* /
        meta.rows``; the injected ``raw.<label_stream>`` gains the ``label_field``
        scalar (so ``Labels`` can expose ``labels.<label_stream>.<label_field>``).

        Returns
        -------
        IO
            The declared interface.

        Raises
        ------
        ConfigError
            If the injected ``label_field`` collides with an existing field.
        SchemaError
            If ``label_stream`` is not a produced scalar (global_object) stream.
        """
        base = self.samples[0].reader.declare_io(mode)
        flat = dict(flatten_spec(base.produces))
        key = f"raw.{self.label_stream}"
        if key not in flat:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} is not a produced "
                f"stream of the sub-readers ({sorted(self.samples[0].reader.streams)}) (plan 02)"
            )
        spec = flat[key]
        if spec.shape != ("B",):
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} must be a SCALAR "
                f"(global_object [B]) stream to carry the injected event label, but its shape "
                f"is {spec.shape} (plan 02)"
            )
        # extend the label_stream's field set with the injected label field
        existing = tuple(spec.fields) if spec.fields is not None else ()
        if self.label_field in existing:
            raise ConfigError(
                f"MultiSampleReader: injected label_field {self.label_field!r} collides with an "
                f"existing field on stream {self.label_stream!r} (plan 02)"
            )
        flat[key] = TensorSpec(
            shape=spec.shape,
            kind=spec.kind,
            dtype=spec.dtype,
            modes=spec.modes,
            fields=(*existing, self.label_field) if spec.fields is not None else None,
        )
        return IO(produces=unflatten_spec(flat))

    # -- file-touching lifecycle hooks (design §2.3) --------------------------

    def prepare(self) -> None:
        """Prepare every sub-reader, validate schema-compat, build the interleave.

        Idempotent. Prepares each sub-reader (probing its files / row counts),
        ASSERTS that every sub-reader produces the IDENTICAL stream schema
        (streams + fields + jaggedness — a clear `SchemaError` on mismatch),
        validates the ``label_stream`` is a scalar stream, builds the merged
        `Schema` (with the injected label field), and constructs the proportional
        round-robin interleaved combined index. Propagates `SchemaError` (sub-reader
        schema mismatch / invalid ``label_stream``) and `ConfigError` (injected
        ``label_field`` collision) from `_validate_schema_compat`.
        """
        if self._sample_of is not None:
            return
        for s in self.samples:
            s.reader.prepare()
        self._validate_schema_compat()
        self._build_index()

    def _produced_signature(self, reader: Reader) -> dict[str, Any]:
        """The reader's PRODUCED-stream signature for schema-compat comparison.

        For each stream: ``(jagged?, ((field, dtype), ...))`` — derived from the
        reader's declared FIT IO (jaggedness from the ``raw.<stream>`` shape) and
        its per-stream `schema_group` (fields + dtypes). This is what must be
        IDENTICAL across samples so the per-sample blocks concatenate.

        Returns
        -------
        dict[str, tuple]
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
                fields = tuple(
                    sorted((f, str(np.dtype(dt))) for f, dt in gschema.fields.items())
                )
            sig[stream] = (jagged, fields)
        return sig

    def _validate_schema_compat(self) -> None:
        """Assert all sub-readers produce identical streams/fields/jaggedness.

        Raises
        ------
        ConfigError
            If the injected ``label_field`` collides with an existing schema field.
        SchemaError
            On any divergence (different stream set, field set, dtype, or
            jaggedness) — with a clear message naming the divergent samples.
        """
        ref = self.samples[0]
        ref_sig = self._produced_signature(ref.reader)
        for s in self.samples[1:]:
            sig = self._produced_signature(s.reader)
            if set(sig) != set(ref_sig):
                raise SchemaError(
                    f"MultiSampleReader: sample {s.name!r} produces streams {sorted(sig)} but "
                    f"sample {ref.name!r} produces {sorted(ref_sig)} — sub-reader produced "
                    "streams must be IDENTICAL so batches concatenate (plan 02)"
                )
            for stream in ref_sig:
                if sig[stream] != ref_sig[stream]:
                    raise SchemaError(
                        f"MultiSampleReader: stream {stream!r} differs between sample "
                        f"{s.name!r} ({sig[stream]}) and {ref.name!r} ({ref_sig[stream]}) — "
                        "produced fields/dtypes/jaggedness must be identical (plan 02)"
                    )
        # validate label_stream is a scalar (global_object) produced stream
        self._jagged = {stream: jagged for stream, (jagged, _f) in ref_sig.items()}
        self._streams = tuple(ref.reader.streams)
        if self.label_stream not in self._jagged:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} not a produced stream "
                f"(produced: {sorted(self._jagged)}) (plan 02)"
            )
        if self._jagged[self.label_stream]:
            raise SchemaError(
                f"MultiSampleReader: label_stream {self.label_stream!r} is a JAGGED/sequence "
                "stream; the injected event label must land on a SCALAR (global_object) stream "
                "(plan 02)"
            )
        # build the merged schema (ref sub-reader's, + the injected label field)
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
                            f"collides with an existing field on {self.label_stream!r} (plan 02)"
                        )
                    fields[self.label_field] = _LABEL_DTYPE
                groups[stream] = GroupSchema(fields=fields)
            self.schema = Schema(groups=groups)

    def _build_index(self) -> None:
        """Build the proportional round-robin interleaved combined index.

        Largest-remainder (Hamilton) apportionment: walk global positions
        ``j = 0..N-1``; at each ``j`` emit the sample whose deficit
        ``(j+1)·n_i/N - emitted_i`` is largest (ties → lowest id). The local row
        for each sample increments in its own order. Result: ``_sample_of[j]`` and
        ``_local_of[j]`` map every global position to ``(sample_id, local_row)``,
        with every prefix's per-sample counts within ``±1`` of the ideal.
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
        # Largest-remainder round-robin. At step j (0-based), the ideal cumulative
        # count for sample i after emitting (j+1) items is (j+1)*n_i/N. We emit the
        # sample with the largest positive deficit, skipping exhausted samples.
        for j in range(n):
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
            sample_of[j] = best_id
            local_of[j] = cursor[best_id]
            cursor[best_id] += 1
            emitted[best_id] += 1
        self._sample_of = sample_of
        self._local_of = local_of

    def __len__(self) -> int:
        """Return the combined row count = Σ len(sub-reader).

        Returns
        -------
        int
            The combined row count.
        """
        self.prepare()
        assert self._num_rows is not None
        return int(self._num_rows)

    def schema_group(self, stream: str) -> GroupSchema | None:
        """The merged schema's group for one stream (built in `prepare`).

        Returns
        -------
        GroupSchema | None
            The group schema (with the injected label field on the label
            stream), or None when no sub-reader has a schema artifact.
        """
        if self.schema is None:
            self.prepare()
        if self.schema is None or stream not in self.schema.groups:
            return None
        return self.schema.groups.get(stream)

    def label_universe(self) -> tuple[str, ...] | None:
        """The ``labels.<stream>.<field>`` universe, including the injected label.

        Returns
        -------
        tuple[str, ...] | None
            All schema-backed label keys (merged sub-reader fields + the injected
            ``labels.<label_stream>.<label_field>``), or None without a schema.
        """
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
        """Clone for a stage, re-sourcing EACH sub-reader from its per-stage source.

        The single ``filename`` the datamodule passes is MEANINGLESS for N samples
        (it would be one path for the whole multi-sample set), so it is IGNORED.
        Instead, for each sample, the sub-reader is re-sourced from
        ``sample.sources[stage]`` when present; otherwise the sub-reader keeps its
        already-configured source (single-file smoke / tests). ``num`` is applied
        per-sample (``-1`` = all). ``vds_path`` is forwarded to sub-readers that
        accept it.

        Parameters
        ----------
        filename : str | Path
            Ignored (the datamodule's single stage path; see above).
        num : int, optional
            Per-sample row cap (``-1`` = all), by default -1.
        vds_path : str | Path | None, optional
            Forwarded to sub-readers' ``with_source`` (H5 VDS path); ignored by
            ROOT sub-readers.
        stage : str | None, optional
            The stage key (``"train"``/``"val"``/``"test"``) the datamodule binds;
            selects each sample's ``sources[stage]``. None keeps each sub-reader's
            configured source.

        Returns
        -------
        MultiSampleReader
            A fresh, unbound multi-sample reader for the stage.
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
        )
        clone.name = self.name
        return clone

    # -- per-worker binding (design §2.3) -------------------------------------

    def bind(self, ctx: WorkerCtx) -> None:
        """Per-worker setup: prepare + forward demand-narrowing to every sub-reader.

        The combined read set is delegated: each sub-reader is bound with the
        SAME ``WorkerCtx`` (so its per-stream ``read_fields`` demand-narrowing
        applies). The injected ``label_field`` is NOT a disk field, so it is
        stripped from the demand forwarded for the ``label_stream`` (the layer
        produces it, not the sub-reader).
        """
        self.prepare()
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
        """Per-stream raw fields demanded — drop the injected label field.

        The default `DatasetModule.read_fields` collects ``raw.<stream>`` requires;
        for the label stream the injected ``label_field`` is produced by THIS layer
        (not on disk), so it must not be forwarded as a sub-reader read demand.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {field: demanding module}}`` with the injected label field
            removed from the label stream.
        """
        out = super().read_fields(step)
        if self.label_stream in out:
            out[self.label_stream] = {
                f: who for f, who in out[self.label_stream].items() if f != self.label_field
            }
            if not out[self.label_stream]:
                del out[self.label_stream]
        return out

    # -- the per-batch read (design §6.1) -------------------------------------

    def _runs(self, rows: slice) -> list[tuple[int, slice, int, int]]:
        """Decompose a global contiguous window into ordered per-sample runs.

        A RUN is a maximal stretch of consecutive global positions mapped to the
        SAME sample with CONSECUTIVE local rows (so the sub-reader receives a
        contiguous ``read(slice)`` — never a scattered index array). Returns the
        runs in GLOBAL (batch) order, each as ``(sample_id, local_slice, out_lo,
        out_hi)`` where ``[out_lo, out_hi)`` is the run's span in the output batch.

        Returns
        -------
        list[tuple[int, slice, int, int]]
            The ordered runs.
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

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Read one combined contiguous window: per-run sub-reads, reorder, inject label.

        Decomposes ``[rows.start, rows.stop)`` into ordered per-sample contiguous
        runs (`_runs`), reads each run via its sub-reader's ``read(local_slice)``,
        and CONCATENATES/REORDERS the per-run ``raw.<stream>`` / ``masks.<stream>``
        blocks into combined batch order. The injected ``raw.<label_stream>``
        gains the ``label_field`` set per run to that sample's label (a real
        event-level scalar, never padded). ``meta.rows`` is the global window in
        TEST.

        Jagged streams may have different served multiplicities ``T`` per
        sub-reader; the combined block uses the MAX ``T`` across the runs in this
        window and pads shorter blocks (float→0.0, masks→True/padded), so the
        concatenation is well-formed.

        Returns
        -------
        dict[str, np.ndarray]
            The produced keys, as a flat dotted dict.
        """
        if self._sample_of is None:
            self.bind(WorkerCtx(mode=mode, read_fields={}, seed=0))  # standalone (tests)
        assert self._streams is not None
        assert self._jagged is not None
        start, stop = rows.start, rows.stop
        b = stop - start
        runs = self._runs(rows)

        # read every run once (a run yields all streams)
        run_out: list[tuple[int, int, int, dict[str, np.ndarray]]] = []
        for sid, local_slice, out_lo, out_hi in runs:
            produced = self.samples[sid].reader.read(local_slice, mode)
            run_out.append((sid, out_lo, out_hi, produced))

        out: dict[str, np.ndarray] = {}
        for stream in self._streams:
            jagged = self._jagged[stream]
            raw_key = f"raw.{stream}"
            blocks = [(out_lo, out_hi, sid, p[raw_key]) for sid, out_lo, out_hi, p in run_out]
            if jagged:
                out[raw_key], combined_valid = self._combine_jagged(blocks, b)
                out[f"masks.{stream}"] = ~combined_valid
            else:
                combined = self._combine_scalar(blocks, b, stream)
                out[raw_key] = combined
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([start, stop], dtype=np.int64)
        return out

    def _combine_scalar(
        self, blocks: list[tuple[int, int, int, np.ndarray]], b: int, stream: str
    ) -> np.ndarray:
        """Concatenate scalar ``(b_i,)`` blocks into a combined ``(B,)`` array, in order.

        For the ``label_stream`` an integer ``label_field`` is appended to the
        structured dtype and filled per run with that sample's label.

        Returns
        -------
        np.ndarray
            The combined structured ``(B,)`` array.
        """
        inject = stream == self.label_stream
        # field/dtype layout from the first block (identical across samples)
        ref = blocks[0][3]
        names = list(ref.dtype.names or ())
        dtype_fields = [(nm, ref.dtype[nm]) for nm in names]
        if inject:
            dtype_fields.append((self.label_field, np.dtype(_LABEL_DTYPE)))
        combined = np.empty((b,), dtype=np.dtype(dtype_fields))
        for out_lo, out_hi, sid, block in blocks:
            for nm in names:
                combined[nm][out_lo:out_hi] = block[nm]
            if inject:
                combined[self.label_field][out_lo:out_hi] = self.samples[sid].label
        return combined

    def _combine_jagged(
        self, blocks: list[tuple[int, int, int, np.ndarray]], b: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate jagged ``(b_i, T_i)`` blocks into a combined ``(B, T)`` array.

        ``T`` = max served multiplicity across the runs in this window; shorter
        blocks are pad-extended (float→0.0, int→-1 sentinel, bool→False, ``valid``
        →False). Returns the combined structured ``(B, T)`` array and its
        ``valid`` ``(B, T)`` bool.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(combined (B, T) array, valid (B, T) bool)``.
        """
        ref = blocks[0][3]
        names = list(ref.dtype.names or ())
        t = max(int(block.shape[1]) for _lo, _hi, _sid, block in blocks)
        dtype_fields = [(nm, ref.dtype[nm]) for nm in names]
        combined = np.zeros((b, t), dtype=np.dtype(dtype_fields))
        # int (label) fields pad to -1 sentinel; bool/valid pad to False; float→0.0
        for nm in names:
            kind = np.dtype(ref.dtype[nm]).kind
            if kind == "i":
                combined[nm][:] = -1
        for out_lo, out_hi, _sid, block in blocks:
            tb = block.shape[1]
            for nm in names:
                combined[nm][out_lo:out_hi, :tb] = block[nm]
        valid = combined["valid"] if "valid" in names else np.zeros((b, t), dtype=bool)
        return combined, valid

    # -- pickling (fork is free; spawn re-binds in the worker) ----------------

    def __getstate__(self) -> dict[str, Any]:
        """Drop transient index/probe state so the reader pickles under spawn.

        Returns
        -------
        dict[str, Any]
            The picklable state (index reset; rebuilt at prepare).
        """
        state = self.__dict__.copy()
        state.update(
            {
                "_lens": None,
                "_num_rows": None,
                "_sample_of": None,
                "_local_of": None,
                "_streams": None,
                "_jagged": None,
                "schema": None,
            }
        )
        return state
