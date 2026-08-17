"""Regression tests for `salt.data.MultiSampleReader` (v2 dataloaders)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from salt.data.base import Reader, WorkerCtx
from salt.data.readers.multisample_reader import MultiSampleReader, SampleConfig
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema


# --------------------------------------------------------------------------- #
# A trivial in-memory STUB Reader (proves reader-agnosticism — NOT a UprootReader)
# --------------------------------------------------------------------------- #


class StubReader(Reader):
    """A minimal in-memory `Reader` over numpy arrays — for the agnostic gate."""

    def __init__(
        self,
        n: int,
        t: int = 4,
        offset: float = 0.0,
        seed: int = 0,
        fields: tuple[str, ...] = ("pt",),
        with_jets: bool = True,
        tag: str = "stub",
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.t = int(t)
        self.offset = float(offset)
        self.seed = int(seed)
        self._field_names = tuple(fields)
        self.with_jets = bool(with_jets)
        self.tag = str(tag)
        self._built = False
        self._event = None
        self._jets = None
        self._valid = None
        self.schema: Schema | None = None

    def _build(self) -> None:
        if self._built:
            return
        rng = np.random.default_rng(self.seed)
        n, t = self.n, self.t
        # scalar event stream: eventNumber identifies the (sample, local row)
        ev_dtype = [("eventNumber", "int64"), ("val", "float32")]
        event = np.empty((n,), dtype=np.dtype(ev_dtype))
        event["eventNumber"] = np.arange(n, dtype=np.int64) + int(self.offset)
        event["val"] = rng.uniform(0, 1, size=n).astype(np.float32) + self.offset
        self._event = event
        if self.with_jets:
            counts = rng.integers(0, t + 1, size=n)
            valid = np.arange(t)[None, :] < counts[:, None]
            j_dtype = [(f, "float32") for f in self._field_names] + [("valid", "bool")]
            jets = np.zeros((n, t), dtype=np.dtype(j_dtype))
            for f in self._field_names:
                jets[f] = rng.uniform(0, 1, size=(n, t)).astype(np.float32) + self.offset
            jets["valid"] = valid
            jets[~valid] = 0  # zero padded positions for float fields
            jets["valid"] = valid
            self._jets = jets
            self._valid = valid
        groups = {
            "event": GroupSchema(fields={"eventNumber": "int64", "val": "float32"}),
        }
        if self.with_jets:
            jfields = {f: "float32" for f in self._field_names}
            jfields["valid"] = "bool"
            groups["jets"] = GroupSchema(fields=jfields)
        self.schema = Schema(groups=groups)
        self._built = True

    @property
    def streams(self) -> tuple[str, ...]:
        return ("jets", "event") if self.with_jets else ("event",)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat: dict[str, TensorSpec] = {}
        if self.with_jets:
            flat["raw.jets"] = TensorSpec(shape=("B", self.t), kind="data")
            flat["masks.jets"] = TensorSpec(shape=("B", self.t), dtype="bool", kind="pad_mask")
        flat["raw.event"] = TensorSpec(shape=("B",), kind="data", fields=("eventNumber", "val"))
        flat["meta.rows"] = TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST)
        return IO(produces=unflatten_spec(flat))

    def prepare(self) -> None:
        self._build()

    def __len__(self) -> int:
        self._build()
        return self.n

    def schema_group(self, stream: str) -> GroupSchema | None:
        self._build()
        return self.schema.groups.get(stream) if self.schema is not None else None

    def bind(self, ctx: WorkerCtx) -> None:
        self._build()

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        self._build()
        out: dict[str, np.ndarray] = {}
        if self.with_jets:
            out["raw.jets"] = self._jets[rows].copy()
            out["masks.jets"] = ~self._valid[rows]
        out["raw.event"] = self._event[rows].copy()
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out

    def with_source(self, filename, num=-1, vds_path=None, stage=None):  # noqa: ANN001
        # the per-stage source IS a (seed, offset) pair encoded as a tuple/dict here,
        # so the stage-binding test can prove distinct stage sources are honoured.
        spec = filename
        if isinstance(spec, dict):
            seed = spec.get("seed", self.seed)
            offset = spec.get("offset", self.offset)
            n = spec.get("n", self.n)
        else:
            seed, offset, n = self.seed, self.offset, self.n
        clone = StubReader(
            n=n,
            t=self.t,
            offset=offset,
            seed=seed,
            fields=self._field_names,
            with_jets=self.with_jets,
            tag=f"{self.tag}:{stage}",
        )
        clone.name = self.name
        return clone


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _full_batches(reader: MultiSampleReader, b: int, mode: Mode = Mode.FIT):
    """Yield (batch_index, process_label_array) for each FULL contiguous batch."""
    n = len(reader)
    for start in range(0, n - b + 1, b):
        out = reader.read(slice(start, start + b), mode)
        yield start // b, out["raw.event"]["process"]


def _epoch_counts(reader: MultiSampleReader) -> Counter:
    """Count events per injected label across the whole epoch (one read)."""
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    return Counter(out["raw.event"]["process"].tolist())


# --------------------------------------------------------------------------- #
# 1. PROPORTIONAL stratification (reader-agnostic: uses StubReader)
# --------------------------------------------------------------------------- #


def test_proportional_1000_9000_per_window_bounded_and_epoch_mean() -> None:
    sig = StubReader(n=1000, seed=1, offset=0.0)
    bkg = StubReader(n=9000, seed=2, offset=100.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    assert len(reader) == 10000
    n = len(reader)
    b = 100
    ideal_sig = b * 1000 / n  # 10
    ideal_bkg = b * 9000 / n  # 90
    n_batches = 0
    for _bi, proc in _full_batches(reader, b):
        n_sig = int((proc == 1).sum())
        n_bkg = int((proc == 0).sum())
        assert n_sig + n_bkg == b
        # documented per-window bound: within ±2 of the ideal proportion
        assert abs(n_sig - ideal_sig) <= 2, f"batch {n_sig=} too far from {ideal_sig}"
        assert abs(n_bkg - ideal_bkg) <= 2, f"batch {n_bkg=} too far from {ideal_bkg}"
        # no long single-class stretch: both classes present every batch (here B>=ideal)
        assert n_sig >= 1 and n_bkg >= 1
        n_batches += 1
    assert n_batches == 100
    # epoch-level: exact apportioned share (every sample's rows emitted once)
    counts = _epoch_counts(reader)
    assert counts[1] == 1000
    assert counts[0] == 9000


@pytest.mark.parametrize(
    ("n_a", "n_b", "b"),
    [
        (5000, 5000, 100),  # 1:1
        (2500, 7500, 100),  # 1:3
    ],
)
def test_proportional_ratios(n_a: int, n_b: int, b: int) -> None:
    a = StubReader(n=n_a, seed=10, offset=0.0)
    bb = StubReader(n=n_b, seed=20, offset=100.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="a", label=0, reader=a),
            SampleConfig(name="b", label=1, reader=bb),
        ]
    )
    n = n_a + n_b
    ideal_a = b * n_a / n
    ideal_b = b * n_b / n
    for _bi, proc in _full_batches(reader, b):
        n_first = int((proc == 0).sum())
        n_second = int((proc == 1).sum())
        assert n_first + n_second == b
        assert abs(n_first - ideal_a) <= 2
        assert abs(n_second - ideal_b) <= 2
    counts = _epoch_counts(reader)
    assert counts[0] == n_a
    assert counts[1] == n_b


def test_small_minority_appears_across_epoch_but_not_every_batch() -> None:
    # minority expectation < 1 per batch: 50 minority in 10000, B=100 -> 0.5/batch
    minority = StubReader(n=50, seed=7, offset=0.0)
    majority = StubReader(n=9950, seed=8, offset=100.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="rare", label=1, reader=minority),
            SampleConfig(name="common", label=0, reader=majority),
        ]
    )
    n = len(reader)
    b = 100
    batches_with_minority = 0
    total_minority = 0
    for _bi, proc in _full_batches(reader, b):
        n_min = int((proc == 1).sum())
        total_minority += n_min
        # bounded: never more than ideal(0.5)+2 in any single batch
        assert n_min <= 2
        if n_min > 0:
            batches_with_minority += 1
    # the minority appears across the epoch (not starved) ...
    assert total_minority == 50
    # ... but NOT in (nearly) every batch (it is genuinely rare)
    assert batches_with_minority < n // b
    assert batches_with_minority >= 1
    del n


# --------------------------------------------------------------------------- #
# 2. LABEL injection consistent with the index
# --------------------------------------------------------------------------- #


def test_label_injection_matches_sample_of_index() -> None:
    sig = StubReader(n=300, seed=1, offset=0.0)
    bkg = StubReader(n=700, seed=2, offset=1000.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    proc = out["raw.event"]["process"]
    # the injected label must equal samples[sample_of[j]].label for every position
    sample_of = reader._sample_of
    labels = np.array([reader.samples[int(sid)].label for sid in sample_of])
    np.testing.assert_array_equal(proc, labels)
    # the injected process is a real event-level scalar (no mask, exactly N entries)
    assert proc.shape == (n,)
    assert "masks.event" not in out
    # consistency: the event scalar's offset distinguishes the source sample
    # (signal eventNumber < 1000, background eventNumber >= 1000 by construction)
    ev = out["raw.event"]["eventNumber"]
    assert np.all(ev[proc == 1] < 1000)
    assert np.all(ev[proc == 0] >= 1000)


def test_label_injection_in_partial_batch_windows() -> None:
    sig = StubReader(n=100, seed=1, offset=0.0)
    bkg = StubReader(n=300, seed=2, offset=1000.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    sample_of = reader._sample_of
    # arbitrary mid windows must still carry the right labels for those positions
    for lo, hi in [(0, 37), (37, 90), (113, 200), (250, 400)]:
        out = reader.read(slice(lo, hi), Mode.FIT)
        proc = out["raw.event"]["process"]
        expect = np.array([reader.samples[int(s)].label for s in sample_of[lo:hi]])
        np.testing.assert_array_equal(proc, expect)


# --------------------------------------------------------------------------- #
# 3. SCHEMA-compat mismatch -> clear SchemaError
# --------------------------------------------------------------------------- #


def test_schema_mismatch_different_streams_raises() -> None:
    a = StubReader(n=100, seed=1, with_jets=True)
    b = StubReader(n=100, seed=2, with_jets=False)  # no jets stream
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="a", label=0, reader=a),
            SampleConfig(name="b", label=1, reader=b),
        ]
    )
    with pytest.raises(SchemaError, match="streams"):
        reader.prepare()


def test_schema_mismatch_different_fields_raises() -> None:
    a = StubReader(n=100, seed=1, fields=("pt",))
    b = StubReader(n=100, seed=2, fields=("pt", "eta"))  # extra field
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="a", label=0, reader=a),
            SampleConfig(name="b", label=1, reader=b),
        ]
    )
    with pytest.raises(SchemaError, match="identical"):
        reader.prepare()


def test_label_stream_must_be_scalar() -> None:
    a = StubReader(n=100, seed=1)
    b = StubReader(n=100, seed=2)
    # point label_stream at the jagged 'jets' stream -> must fail
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="a", label=0, reader=a),
            SampleConfig(name="b", label=1, reader=b),
        ],
        label_stream="jets",
    )
    with pytest.raises(SchemaError, match="SCALAR|scalar|sequence"):
        reader.prepare()


# --------------------------------------------------------------------------- #
# 4. READER-AGNOSTIC + 5. ROUND-TRIP (StubReader)
# --------------------------------------------------------------------------- #


def test_reader_agnostic_roundtrip_against_subreaders() -> None:
    sig = StubReader(n=400, seed=1, offset=0.0)
    bkg = StubReader(n=600, seed=2, offset=100.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    # independent ground-truth sub-reader reads
    sig_raw = sig.read(slice(0, 400), Mode.FIT)
    bkg_raw = bkg.read(slice(0, 600), Mode.FIT)
    truth = {0: bkg_raw, 1: sig_raw}
    local_cursor = {0: 0, 1: 0}
    combined_event = out["raw.event"]
    combined_jets = out["raw.jets"]
    sample_of = reader._sample_of
    for j in range(n):
        lab = int(combined_event["process"][j])
        src = truth[lab]
        li = local_cursor[lab]
        local_cursor[lab] += 1
        # the combined event scalar must match the sub-reader's read at that local row
        assert combined_event["eventNumber"][j] == src["raw.event"]["eventNumber"][li]
        np.testing.assert_array_equal(combined_event["val"][j], src["raw.event"]["val"][li])
        # jagged jets match too (within each sub-reader's own T, padded to combined T)
        tb = src["raw.jets"].shape[1]
        np.testing.assert_array_equal(combined_jets["pt"][j, :tb], src["raw.jets"]["pt"][li])
        np.testing.assert_array_equal(
            combined_jets["valid"][j, :tb], src["raw.jets"]["valid"][li]
        )
    # sanity: the per-position label equals the index mapping
    assert [int(reader.samples[int(s)].label) for s in sample_of] == list(
        combined_event["process"]
    )


def test_runs_are_contiguous_local_slices() -> None:
    # the read(slice)-only contract: sub-readers only ever get contiguous slices
    sig = StubReader(n=300, seed=1)
    bkg = StubReader(n=700, seed=2)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    runs = reader._runs(slice(0, 250))
    # every run's local slice is a proper contiguous slice (stop > start, step None)
    for sid, local_slice, out_lo, out_hi in runs:
        assert isinstance(local_slice, slice)
        assert local_slice.step in (None, 1)
        assert local_slice.stop - local_slice.start == out_hi - out_lo
        assert local_slice.start >= 0
        del sid
    # the runs tile the output window exactly, in order
    assert runs[0][2] == 0
    assert runs[-1][3] == 250
    for k in range(1, len(runs)):
        assert runs[k][2] == runs[k - 1][3]


def _two_sample_reader(n_sig: int = 300, n_bkg: int = 700, **kwargs) -> MultiSampleReader:
    """A prepared two-sample reader over stubs, for the segment/interleave tests."""
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=StubReader(n=n_sig, seed=1)),
            SampleConfig(name="background", label=0, reader=StubReader(n=n_bkg, seed=2, offset=99)),
        ],
        **kwargs,
    )
    reader.prepare()
    return reader


def test_segments_cover_the_window_exactly_once() -> None:
    # every output position is claimed by exactly one segment, and each segment's
    # local slice is contiguous and the same length as its position list
    reader = _two_sample_reader()
    window = slice(120, 370)
    segments = reader._segments(window)
    claimed = np.concatenate([positions for _sid, _sl, positions in segments])
    np.testing.assert_array_equal(np.sort(claimed), np.arange(window.stop - window.start))
    for sid, local_slice, positions in segments:
        assert local_slice.step in (None, 1)
        assert local_slice.stop - local_slice.start == positions.size
        # the local rows a segment claims are exactly the index's own mapping
        np.testing.assert_array_equal(
            reader._local_of[window.start + positions],
            np.arange(local_slice.start, local_slice.stop),
        )
        np.testing.assert_array_equal(
            reader._sample_of[window.start + positions], np.full(positions.size, sid)
        )


def test_segments_coalesce_the_row_granular_interleave() -> None:
    # the whole point: a 1:1 interleave decomposes into ~250 runs but 2 segments
    reader = _two_sample_reader()
    window = slice(0, 250)
    assert len(reader._runs(window)) > 100
    assert len(reader._segments(window)) == len(reader.samples)


def test_segment_reads_match_run_reads_bit_for_bit() -> None:
    # coalescing must be invisible in the batch: same rows, same values
    reader = _two_sample_reader()
    out = reader.read(slice(0, 250), Mode.FIT)
    sig = reader.samples[0].reader
    bkg = reader.samples[1].reader
    truth = {1: sig.read(slice(0, 250), Mode.FIT), 0: bkg.read(slice(0, 250), Mode.FIT)}
    cursor = {0: 0, 1: 0}
    for j in range(250):
        lab = int(out["raw.event"]["process"][j])
        src = truth[lab]
        li = cursor[lab]
        cursor[lab] += 1
        assert out["raw.event"]["eventNumber"][j] == src["raw.event"]["eventNumber"][li]
        tb = src["raw.jets"].shape[1]
        np.testing.assert_array_equal(out["raw.jets"]["pt"][j, :tb], src["raw.jets"]["pt"][li])


def test_interleave_block_defaults_to_the_row_granular_index() -> None:
    # the default must be bit-for-bit the historical index
    row = _two_sample_reader()
    explicit = _two_sample_reader(interleave_block=1)
    np.testing.assert_array_equal(row._sample_of, explicit._sample_of)
    np.testing.assert_array_equal(row._local_of, explicit._local_of)


@pytest.mark.parametrize("block", [1, 16, 64])
def test_interleave_block_preserves_the_epoch_multiset_and_proportions(block: int) -> None:
    # a block interleave changes WHICH batch a row lands in, never the epoch's
    # content nor a batch's sample proportions beyond one block
    reader = _two_sample_reader(interleave_block=block)
    assert len(reader) == 1000
    for sid, expected in ((0, 300), (1, 700)):
        rows = reader._local_of[reader._sample_of == sid]
        np.testing.assert_array_equal(np.sort(rows), np.arange(expected))
    # a window's count is the difference of two prefix counts, and largest
    # remainder bounds each prefix to one block of ideal — hence 2 * block
    batch = 250
    for start in range(0, 1000, batch):
        n_sig = int((reader._sample_of[start : start + batch] == 0).sum())
        assert abs(n_sig - batch * 0.3) <= 2 * block


def test_interleave_block_below_one_is_refused() -> None:
    with pytest.raises(ConfigError, match="interleave_block"):
        MultiSampleReader(
            samples=[SampleConfig(name="a", label=0, reader=StubReader(n=10))],
            interleave_block=0,
        )


# --------------------------------------------------------------------------- #
# 6. STAGE-BINDING — train vs val bind DISTINCT per-sample sources
# --------------------------------------------------------------------------- #


def test_stage_binding_distinct_sources_per_sample() -> None:
    # each sample carries a per-stage source spec; train/val resolve to distinct data
    sig = StubReader(n=10, seed=0)
    bkg = StubReader(n=10, seed=0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(
                name="signal",
                label=1,
                reader=sig,
                sources={
                    "train": {"seed": 101, "offset": 0.0, "n": 200},
                    "val": {"seed": 102, "offset": 0.0, "n": 50},
                },
            ),
            SampleConfig(
                name="background",
                label=0,
                reader=bkg,
                sources={
                    "train": {"seed": 201, "offset": 1000.0, "n": 800},
                    "val": {"seed": 202, "offset": 1000.0, "n": 150},
                },
            ),
        ]
    )
    train = reader.with_source(filename="IGNORED", stage="train")
    val = reader.with_source(filename="IGNORED", stage="val")
    train.prepare()
    val.prepare()
    # distinct sizes per stage prove the per-stage source was honoured per sample
    assert len(train) == 1000  # 200 + 800
    assert len(val) == 200  # 50 + 150
    # the sub-readers themselves are DISTINCT instances per stage with the staged data
    assert train.samples[0].reader is not val.samples[0].reader
    assert train.samples[0].reader.n == 200
    assert val.samples[0].reader.n == 50
    assert train.samples[1].reader.n == 800
    assert val.samples[1].reader.n == 150
    # the staged seeds differ -> the actual event content differs train vs val
    t_out = train.read(slice(0, 10), Mode.FIT)
    v_out = val.read(slice(0, 10), Mode.FIT)
    # both stages still inject the right labels
    assert set(t_out["raw.event"]["process"].tolist()) <= {0, 1}
    assert set(v_out["raw.event"]["process"].tolist()) <= {0, 1}
    # tag carries the stage through with_source (provenance the layer set)
    assert train.samples[0].reader.tag.endswith("train")
    assert val.samples[0].reader.tag.endswith("val")


def test_declare_io_injects_label_field() -> None:
    sig = StubReader(n=10, seed=1)
    bkg = StubReader(n=10, seed=2)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="s", label=1, reader=sig),
            SampleConfig(name="b", label=0, reader=bkg),
        ]
    )
    io = reader.declare_io(Mode.FIT)
    from salt.graph.spec import flatten_spec

    flat = flatten_spec(io.produces)
    ev = flat["raw.event"]
    assert "process" in ev.fields
    assert "eventNumber" in ev.fields
    # label universe exposes the injected label as a labels.event.process key
    universe = reader.label_universe()
    assert "labels.event.process" in universe


def test_demand_narrowing_strips_injected_label_field() -> None:
    # the injected label field is NOT a disk field -> never forwarded to sub-readers
    from salt.graph.planner import PlanStep
    from types import MappingProxyType

    sig = StubReader(n=10, seed=1)
    bkg = StubReader(n=10, seed=2)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="s", label=1, reader=sig),
            SampleConfig(name="b", label=0, reader=bkg),
        ]
    )
    reader.name = "reader"
    step = PlanStep(
        name="reader",
        module=reader,
        requires=MappingProxyType(
            {
                "raw.event": TensorSpec(kind="data", fields=("process", "val")),
            }
        ),
        produces=MappingProxyType({}),
    )
    rf = reader.read_fields(step)
    # 'process' (injected) is stripped; 'val' (a real disk field) survives
    assert "process" not in rf.get("event", {})
    assert "val" in rf.get("event", {})


# --------------------------------------------------------------------------- #
# Bonus: wrap two UprootReaders on the synthetic ROOT fixtures (skips w/o deps)
# --------------------------------------------------------------------------- #

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.data import UprootGroupConfig, UprootReader  # noqa: E402
from salt.tests._fixtures.easyjet_minitree import (  # noqa: E402
    build_fixture_arrays,
    write_minitree,
)

_EJ_JET_BRANCHES = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
}
_EJ_EVENT_BRANCHES = {"eventNumber": "eventNumber"}


def _ej_groups(truncate: int = 8) -> dict:
    return {
        "jets": UprootGroupConfig(branches=dict(_EJ_JET_BRANCHES), jagged=True, pad_max=truncate),
        "event": UprootGroupConfig(branches=dict(_EJ_EVENT_BRANCHES), jagged=False),
    }


def _ej_reader(**kw):
    return UprootReader(tree="AnalysisMiniTree", unroll=None, **kw)


@pytest.fixture
def two_easyjet_files(tmp_path: Path) -> tuple[Path, Path]:
    sig = write_minitree(tmp_path / "sig.root", build_fixture_arrays(seed=1))
    bkg = write_minitree(tmp_path / "bkg.root", build_fixture_arrays(seed=2))
    return sig, bkg


def test_easyjet_wrapped_multisample_roundtrip_and_labels(
    two_easyjet_files: tuple[Path, Path],
) -> None:
    sig_path, bkg_path = two_easyjet_files
    sig = _ej_reader(groups=_ej_groups(truncate=8), filename=sig_path)
    bkg = _ej_reader(groups=_ej_groups(truncate=8), filename=bkg_path)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    n = len(reader)
    assert n == len(sig) + len(bkg)
    out = reader.read(slice(0, n), Mode.FIT)
    proc = out["raw.event"]["process"]
    # label injection on a real UprootReader-produced scalar event stream
    np.testing.assert_array_equal(
        proc, [reader.samples[int(s)].label for s in reader._sample_of]
    )
    # round-trip: combined jets pt matches the per-sample UprootReader reads
    sig_raw = sig.read(slice(0, len(sig)), Mode.FIT)
    bkg_raw = bkg.read(slice(0, len(bkg)), Mode.FIT)
    truth = {0: bkg_raw, 1: sig_raw}
    cursor = {0: 0, 1: 0}
    for j in range(n):
        lab = int(proc[j])
        li = cursor[lab]
        cursor[lab] += 1
        np.testing.assert_array_equal(
            out["raw.jets"]["pt"][j], truth[lab]["raw.jets"]["pt"][li]
        )
        assert out["raw.event"]["eventNumber"][j] == truth[lab]["raw.event"]["eventNumber"][li]


# --------------------------------------------------------------------------- #
# 7. reader-owned staging — sources() = union over sub-readers; restage()
#    delegates to each sub-reader recursively (the multi-SAMPLE multi-FILE case).
# --------------------------------------------------------------------------- #


def test_multisample_sources_is_union_over_subreaders(
    two_easyjet_files: tuple[Path, Path],
) -> None:
    """sources() is the de-duplicated union of every sub-reader's sources."""
    sig_path, bkg_path = two_easyjet_files
    sig = _ej_reader(groups=_ej_groups(), filename=sig_path)
    bkg = _ej_reader(groups=_ej_groups(), filename=bkg_path)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    assert reader.sources() == [sig_path, bkg_path]


def test_multisample_restage_delegates_recursively_and_roundtrips(
    two_easyjet_files: tuple[Path, Path], tmp_path: Path
) -> None:
    """restage() restages EACH sub-reader into root; combined read is byte-identical."""
    sig_path, bkg_path = two_easyjet_files
    sig = _ej_reader(groups=_ej_groups(truncate=8), filename=sig_path)
    bkg = _ej_reader(groups=_ej_groups(truncate=8), filename=bkg_path)
    orig = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    orig.prepare()
    n = len(orig)
    orig_out = orig.read(slice(0, n), Mode.FIT)

    root = tmp_path / "ms_stage"
    staged = orig.restage(root)
    # each sub-reader was restaged under the shared root (recursive delegation),
    # each into its OWN per-reader subdir so they don't glob each other's files
    assert all(root in p.parents for p in staged.sources())
    assert {p.name for p in staged.sources()} == {"sig.root", "bkg.root"}
    assert {p.name for p in root.rglob("*.root")} == {"sig.root", "bkg.root"}
    # originals survive
    assert sig_path.is_file() and bkg_path.is_file()

    staged.prepare()
    assert len(staged) == n
    staged_out = staged.read(slice(0, n), Mode.FIT)
    # the interleave is seeded identically (same config), so the composition matches
    np.testing.assert_array_equal(
        staged_out["raw.event"]["process"], orig_out["raw.event"]["process"]
    )
    np.testing.assert_array_equal(staged_out["raw.jets"]["pt"], orig_out["raw.jets"]["pt"])
    np.testing.assert_array_equal(
        staged_out["raw.event"]["eventNumber"], orig_out["raw.event"]["eventNumber"]
    )


# --------------------------------------------------------------------------- #
# 1b. MANY samples of very different lengths (K >> 2)
#
# The K=2 tests above fix the two-sample case; the training target is many
# samples whose sizes span orders of magnitude, so the same properties are
# asserted for K in {3, 8, 32} over size mixes spanning ~180x. Nothing here is
# a weaker assertion than the K=2 versions — same +/-2 per-window bound, same
# exact epoch multiset, same label-injection identity.
# --------------------------------------------------------------------------- #

# size mixes spanning ~2 orders of magnitude (50 ... 9,000)
_K3_SIZES = (50, 700, 9000)
_K8_SIZES = (50, 120, 300, 700, 1500, 3000, 5000, 9000)
_K32_SIZES = tuple(
    int(round(50 * (9000 / 50) ** (i / 31))) for i in range(32)
)  # log-spaced 50 -> 9000


def _many_sample_reader(sizes, interleave_block: int = 1) -> MultiSampleReader:
    """A MultiSampleReader over len(sizes) StubReaders; label i == sample i."""
    return MultiSampleReader(
        samples=[
            SampleConfig(
                name=f"s{i}",
                label=i,
                reader=StubReader(n=n, seed=100 + i, offset=1_000_000.0 * (i + 1)),
            )
            for i, n in enumerate(sizes)
        ],
        interleave_block=interleave_block,
    )


@pytest.mark.parametrize(
    ("sizes", "b"),
    [(_K3_SIZES, 100), (_K8_SIZES, 200), (_K32_SIZES, 500)],
    ids=["K3", "K8", "K32"],
)
def test_many_samples_per_window_bounded_and_epoch_exact(sizes, b) -> None:
    """Every sample stays within the documented per-window bound, at every K."""
    reader = _many_sample_reader(sizes)
    n = sum(sizes)
    assert len(reader) == n

    n_batches = 0
    for _bi, proc in _full_batches(reader, b):
        assert len(proc) == b
        for i, n_i in enumerate(sizes):
            ideal = b * n_i / n
            got = int((proc == i).sum())
            assert abs(got - ideal) <= 2, (
                f"K={len(sizes)} sample {i} (n={n_i}): batch had {got}, ideal {ideal:.3f}"
            )
        n_batches += 1
    assert n_batches == n // b

    # epoch-level: every sample's rows emitted exactly once, no more, no fewer
    counts = _epoch_counts(reader)
    for i, n_i in enumerate(sizes):
        assert counts[i] == n_i, f"sample {i}: epoch served {counts[i]}, expected {n_i}"
    assert sum(counts.values()) == n


@pytest.mark.parametrize(
    "sizes", [_K3_SIZES, _K8_SIZES, _K32_SIZES], ids=["K3", "K8", "K32"]
)
def test_many_samples_label_injection_matches_the_index(sizes) -> None:
    """The injected label identifies the owning sample for every row, at every K.

    StubReader stamps eventNumber = local row + offset, and each sample gets a
    distinct 1e6-spaced offset, so the label and the value are independently
    derived — a mislabelled row cannot agree with both.
    """
    reader = _many_sample_reader(sizes)
    out = reader.read(slice(0, len(reader)), Mode.FIT)
    proc = out["raw.event"]["process"]
    evno = out["raw.event"]["eventNumber"]

    for i, n_i in enumerate(sizes):
        mine = proc == i
        assert int(mine.sum()) == n_i
        local = evno[mine] - 1_000_000 * (i + 1)
        # each sample contributes exactly its own rows 0..n_i-1, in order
        np.testing.assert_array_equal(np.sort(local), np.arange(n_i))
        np.testing.assert_array_equal(local, np.arange(n_i))


@pytest.mark.parametrize(
    ("sizes", "b"),
    [(_K3_SIZES, 100), (_K8_SIZES, 200), (_K32_SIZES, 500)],
    ids=["K3", "K8", "K32"],
)
def test_many_samples_minorities_are_spread_not_starved(sizes, b) -> None:
    """The smallest samples appear across the epoch rather than bunching."""
    reader = _many_sample_reader(sizes)
    n = sum(sizes)
    n_batches = n // b
    smallest = min(range(len(sizes)), key=lambda i: sizes[i])

    seen_in = []
    for bi, proc in _full_batches(reader, b):
        if int((proc == smallest).sum()) > 0:
            seen_in.append(bi)

    assert seen_in, "smallest sample never appeared"
    # spread: its first and last appearance straddle most of the epoch, and it
    # is not confined to one contiguous run at the start
    assert seen_in[0] < n_batches * 0.2
    assert seen_in[-1] > n_batches * 0.8


@pytest.mark.parametrize("block", [1, 16, 64])
@pytest.mark.parametrize(
    ("sizes", "b"), [(_K8_SIZES, 200), (_K32_SIZES, 500)], ids=["K8", "K32"]
)
def test_many_samples_block_interleave_preserves_proportions(sizes, b, block) -> None:
    """A block interleave moves rows between batches, never the epoch content."""
    reader = _many_sample_reader(sizes, interleave_block=block)
    n = sum(sizes)

    for _bi, proc in _full_batches(reader, b):
        for i, n_i in enumerate(sizes):
            ideal = b * n_i / n
            got = int((proc == i).sum())
            # same bound the K=2 block test uses: one block of slack either way
            assert abs(got - ideal) <= 2 * block, (
                f"block={block} K={len(sizes)} sample {i}: {got} vs ideal {ideal:.3f}"
            )

    counts = _epoch_counts(reader)
    for i, n_i in enumerate(sizes):
        assert counts[i] == n_i
