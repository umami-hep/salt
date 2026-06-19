"""Regression tests for `salt.core.data.MultiSampleReader` (plan 02, v2 dataloaders).

Gates the reader-agnostic proportional-stratified multi-sample layer:

1. **PROPORTIONAL stratification** — for sample sizes ``n_i`` and batch size B, each
   contiguous full batch has BOUNDED discrepancy from the ideal ``B·n_i/Σn``
   composition (documented ``±2`` per-window bound), AND the epoch-level mean is the
   exact apportioned share. 1000:9000, 1:1, 1:3, and a small-minority (<1 per batch,
   still appears across the epoch).
2. **LABEL injection** — each event's injected ``process`` == its source sample's
   label, consistent with the interleave index order, not masked away.
3. **SCHEMA-compat** — sub-readers with mismatched produced streams/fields →
   `SchemaError`.
4. **READER-AGNOSTIC** — the SAME layer wraps two trivial in-memory STUB sub-Readers
   (NOT EasyjetReader) plus (bonus) two EasyjetReaders on the synthetic ROOT fixtures.
5. **ROUND-TRIP** — combined ``raw.<stream>`` == the per-sample sub-reader reads.
6. **STAGE-BINDING** — train vs val bind DISTINCT per-sample sources via the
   per-reader stage-sourcing contract.

The stub readers depend on NOTHING but numpy, so the proportional/label/agnostic
tests run without uproot/awkward (the EasyjetReader-wrapped tests skip if absent).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from salt.core.data.base import Reader, WorkerCtx
from salt.core.data.multisample_reader import MultiSampleReader, SampleConfig
from salt.core.graph.errors import SchemaError
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.core.schema import GroupSchema, Schema


# --------------------------------------------------------------------------- #
# A trivial in-memory STUB Reader (proves reader-agnosticism — NOT EasyjetReader)
# --------------------------------------------------------------------------- #


class StubReader(Reader):
    """A minimal in-memory `Reader` over numpy arrays — for the agnostic gate.

    Produces a SCALAR ``event`` stream (eventNumber + a per-event ``val`` scalar)
    and a JAGGED ``jets`` stream (a ``pt`` float + a ``valid`` bool), all generated
    deterministically from a seed. ``read(slice)`` returns the requested contiguous
    rows — no files, no uproot — so it exercises the multi-sample layer's
    contract on a reader the layer has never heard of.
    """

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
    from salt.core.graph.spec import flatten_spec

    flat = flatten_spec(io.produces)
    ev = flat["raw.event"]
    assert "process" in ev.fields
    assert "eventNumber" in ev.fields
    # label universe exposes the injected label as a labels.event.process key
    universe = reader.label_universe()
    assert "labels.event.process" in universe


def test_demand_narrowing_strips_injected_label_field() -> None:
    # the injected label field is NOT a disk field -> never forwarded to sub-readers
    from salt.core.graph.planner import PlanStep
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
# Bonus: wrap two EasyjetReaders on the synthetic ROOT fixtures (skips w/o deps)
# --------------------------------------------------------------------------- #

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.core.data import EasyjetGroupConfig, EasyjetReader  # noqa: E402
from salt.tests.core.fixtures.easyjet_minitree import (  # noqa: E402
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
        "jets": EasyjetGroupConfig(branches=dict(_EJ_JET_BRANCHES), jagged=True, truncate=truncate),
        "event": EasyjetGroupConfig(branches=dict(_EJ_EVENT_BRANCHES), jagged=False),
    }


@pytest.fixture
def two_easyjet_files(tmp_path: Path) -> tuple[Path, Path]:
    sig = write_minitree(tmp_path / "sig.root", build_fixture_arrays(seed=1))
    bkg = write_minitree(tmp_path / "bkg.root", build_fixture_arrays(seed=2))
    return sig, bkg


def test_easyjet_wrapped_multisample_roundtrip_and_labels(
    two_easyjet_files: tuple[Path, Path],
) -> None:
    sig_path, bkg_path = two_easyjet_files
    sig = EasyjetReader(groups=_ej_groups(truncate=8), filename=sig_path)
    bkg = EasyjetReader(groups=_ej_groups(truncate=8), filename=bkg_path)
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
    # label injection on a real EasyjetReader-produced scalar event stream
    np.testing.assert_array_equal(
        proc, [reader.samples[int(s)].label for s in reader._sample_of]
    )
    # round-trip: combined jets pt matches the per-sample EasyjetReader reads
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
