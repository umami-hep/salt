"""Reader-agnostic tests for `IterableGraphDataset` — stubs only, no uproot, no ROOT.

Every guarantee the streaming path claims is checked here against an in-memory
stub reader whose rows carry a unique id, so coverage, disjointness, epoch
multiset, stratification, determinism, termination and balance are all decided
by counting ids rather than by trusting a file format. Nothing in this module
imports `uproot`, `awkward`, `h5py` or ROOT — that is the point of it, and it is
also why it can never be silently skipped by a module-level ``importorskip``.
"""

from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pytest

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.iterable_dataset import IterableGraphDataset
from salt.data.manifest import CorpusManifest, ManifestEntry, build_manifest
from salt.data.processors.features import Features
from salt.data.readers.multisample_reader import MultiSampleReader, SampleConfig
from salt.data.sharding import interleave_plan, partition_blocks, shard_row_counts
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema

# --------------------------------------------------------------------------- #
# A trivial in-memory STUB Reader — reader-agnosticism is the contract here
# --------------------------------------------------------------------------- #


class BlockStubReader(Reader):
    """In-memory reader with declared block boundaries and per-row unique ids.

    ``uid`` is globally unique across samples (``offset + i``) so any set of
    emitted rows can be compared to the corpus as a multiset.
    """

    def __init__(
        self,
        n: int,
        t: int = 4,
        offset: int = 0,
        n_blocks: int = 1,
        seed: int = 0,
        with_jets: bool = True,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.t = int(t)
        self.offset = int(offset)
        self.n_blocks = int(n_blocks)
        self.seed = int(seed)
        self.with_jets = bool(with_jets)
        self._built = False
        self.schema: Schema | None = None

    def _build(self) -> None:
        if self._built:
            return
        rng = np.random.default_rng(self.seed)
        n, t = self.n, self.t
        event = np.empty((n,), dtype=np.dtype([("uid", "int64"), ("val", "float32")]))
        event["uid"] = np.arange(n, dtype=np.int64) + self.offset
        event["val"] = rng.uniform(0, 1, size=n).astype(np.float32)
        self._event = event
        groups = {"event": GroupSchema(fields={"uid": "int64", "val": "float32"})}
        if self.with_jets:
            counts = rng.integers(0, t + 1, size=n)
            valid = np.arange(t)[None, :] < counts[:, None]
            jets = np.zeros((n, t), dtype=np.dtype([("pt", "float32"), ("valid", "bool")]))
            jets["pt"] = rng.uniform(0, 1, size=(n, t)).astype(np.float32)
            jets["pt"][~valid] = 0
            jets["valid"] = valid
            self._jets, self._valid = jets, valid
            groups["jets"] = GroupSchema(fields={"pt": "float32", "valid": "bool"})
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
        flat["raw.event"] = TensorSpec(shape=("B",), kind="data", fields=("uid", "val"))
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

    def row_blocks(self) -> list[RowBlock]:
        """`n_blocks` near-equal blocks — the stand-in for 'one block per file'."""
        self._build()
        edges = [(i * self.n) // self.n_blocks for i in range(self.n_blocks + 1)]
        return [
            RowBlock(group=0, start=lo, stop=hi)
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            if hi > lo
        ]

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
        return self


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _multisample(sizes: list[int], t: int = 4, n_blocks: int = 3) -> MultiSampleReader:
    """A `MultiSampleReader` over stub samples with globally unique uids."""
    samples, offset = [], 0
    for i, n in enumerate(sizes):
        samples.append(
            SampleConfig(
                name=f"s{i}",
                label=i,
                reader=BlockStubReader(n=n, t=t, offset=offset, n_blocks=n_blocks, seed=i),
            )
        )
        offset += n
    return MultiSampleReader(samples=samples, label_stream="event", label_field="process")


def _dataset(reader: Reader, batch_size: int = 8, **kwargs) -> IterableGraphDataset:
    """A streaming dataset over `reader` producing the event uid + label."""
    modules = {
        "reader": reader,
        "feats": Features(variables={"event": ["val"]}),
    }
    kwargs.setdefault("block_rows", None)
    return IterableGraphDataset(
        modules,
        mode=Mode.FIT,
        sinks=["inputs.event"],
        batch_size=batch_size,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# pure sharding arithmetic (no dataset, no reader)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [1, 2, 3, 7, 64, 1000])
def test_intervals_cover_and_are_disjoint(n_shards: int) -> None:
    """Every row of every group lands in exactly one shard, at any shard count."""
    group_rows = [97, 5, 1000]
    blocks = [
        RowBlock(group=g, start=lo, stop=min(lo + 13, n))
        for g, n in enumerate(group_rows)
        for lo in range(0, n, 13)
    ]
    seen: Counter = Counter()
    for shard in range(n_shards):
        mine = partition_blocks(blocks, group_rows, n_shards, shard)
        for group, bs in mine.items():
            for block in bs:
                for row in range(block.start, block.stop):
                    seen[(group, row)] += 1
    expected = {(g, r) for g, n in enumerate(group_rows) for r in range(n)}
    assert set(seen) == expected
    assert set(seen.values()) == {1}


@pytest.mark.parametrize("n_shards", [2, 3, 8, 32])
def test_shard_balance_within_one_row_per_group(n_shards: int) -> None:
    """Interval sharding balances to <= 1 row per group per shard."""
    group_rows = [50, 743, 9000]
    counts = shard_row_counts(group_rows, n_shards)
    assert max(counts) - min(counts) <= len(group_rows)


def test_shard_count_may_exceed_block_count() -> None:
    """More shards than blocks is the general case, not an error (sub-file splitting)."""
    group_rows = [100]
    blocks = [RowBlock(group=0, start=0, stop=100)]  # ONE block, 50 shards
    non_empty = sum(bool(partition_blocks(blocks, group_rows, 50, s)) for s in range(50))
    assert non_empty == 50


def test_block_rows_caps_every_read() -> None:
    """No block handed to the reader exceeds `block_rows`."""
    blocks = [RowBlock(group=0, start=0, stop=1000)]
    mine = partition_blocks(blocks, [1000], 1, 0, block_rows=64)
    assert all(b.n_rows <= 64 for b in mine[0])
    assert sum(b.n_rows for b in mine[0]) == 1000


@pytest.mark.parametrize("n_shards,shard_id", [(0, 0), (2, 2), (2, -1)])
def test_partition_rejects_bad_shards(n_shards: int, shard_id: int) -> None:
    """An impossible (n_shards, shard_id) is a config error, not silent nonsense."""
    with pytest.raises(ConfigError):
        partition_blocks([RowBlock(0, 0, 10)], [10], n_shards, shard_id)


@pytest.mark.parametrize("k", [2, 3, 8, 32])
def test_interleave_plan_proportions(k: int) -> None:
    """Per-prefix group counts stay within `interleave_block` of ideal at every K."""
    rng = np.random.default_rng(k)
    counts = {g: int(n) for g, n in enumerate(rng.integers(50, 9000, size=k))}
    total = sum(counts.values())
    plan = interleave_plan(counts, interleave_block=1)
    assert plan.size == total
    assert Counter(plan.tolist()) == Counter({g: n for g, n in counts.items()})
    for prefix in (total // 4, total // 2, total):
        emitted = Counter(plan[:prefix].tolist())
        for group, n in counts.items():
            assert abs(emitted[group] - prefix * n / total) <= 1 + 1e-9


def test_interleave_block_widens_the_bound() -> None:
    """A larger `interleave_block` keeps proportions, within the wider stated bound."""
    counts = {0: 500, 1: 1500}
    plan = interleave_plan(counts, interleave_block=10)
    assert Counter(plan.tolist()) == Counter(counts)
    for prefix in (200, 1000, 2000):
        emitted = Counter(plan[:prefix].tolist())
        for group, n in counts.items():
            assert abs(emitted[group] - prefix * n / 2000) <= 10


# --------------------------------------------------------------------------- #
# streaming dataset: coverage, multiset, determinism, termination
# --------------------------------------------------------------------------- #


def _stream_uids(reader_factory, n_shards: int, batch_size: int = 8, **kwargs) -> list[list[int]]:
    """Per-shard uid lists over a simulated (world x workers) grid, in one process."""
    out = []
    for shard in range(n_shards):
        reader = reader_factory()
        dataset = _dataset(
            reader,
            batch_size=batch_size,
            world_size=n_shards,
            rank=shard,
            num_workers=1,
            worker_id=0,
            **kwargs,
        )
        out.append(_batch_uids(dataset))
    return out


def _batch_uids(dataset: IterableGraphDataset) -> list[int]:
    """Flatten a shard's stream into uids by reading the reader's raw output directly."""
    uids: list[int] = []
    for batch in _raw_batches(dataset):
        uids.extend(int(v) for v in batch["raw.event"]["uid"])
    return uids


def _raw_batches(dataset: IterableGraphDataset) -> list[dict]:
    """The reader-level batches the dataset assembles, before the torch boundary.

    Monkey-free: `_run_plan` is the only step between assembly and torch, so the
    assembled dict is captured by iterating with the plan short-circuited.
    """
    captured: list[dict] = []
    original = dataset._run_plan

    def capture(rows, raw=None):  # noqa: ANN001, ANN202
        if raw is not None:
            captured.append(raw)
        return original(rows, raw)

    dataset._run_plan = capture  # type: ignore[method-assign]
    for _ in dataset:
        pass
    dataset._run_plan = original  # type: ignore[method-assign]
    return captured


@pytest.mark.parametrize("grid", [(1, 1), (2, 1), (1, 4), (2, 3), (3, 5)])
def test_coverage_and_disjointness_over_a_worker_grid(grid: tuple[int, int]) -> None:
    """Every corpus row is yielded exactly once across all (rank, worker) shards."""
    world, workers = grid
    n_shards = world * workers
    sizes = [40, 130, 7]
    emitted: list[int] = []
    for shard in range(n_shards):
        dataset = _dataset(
            _multisample(sizes),
            batch_size=4,
            world_size=world,
            rank=shard // workers,
            num_workers=workers,
            worker_id=shard % workers,
            drop_last=False,
            shuffle=False,
        )
        emitted.extend(_batch_uids(dataset))
    assert Counter(emitted) == Counter(range(sum(sizes)))


def test_epoch_multiset_preserved_under_shuffle() -> None:
    """Shuffling changes order, never membership."""
    sizes = [40, 130, 7]
    plain = _batch_uids(_dataset(_multisample(sizes), batch_size=4, drop_last=False, shuffle=False))
    shuffled = _batch_uids(
        _dataset(_multisample(sizes), batch_size=4, drop_last=False, shuffle=True, seed=7)
    )
    assert Counter(plain) == Counter(shuffled)
    assert plain != shuffled


@pytest.mark.parametrize("k", [2, 3, 8, 32])
def test_per_batch_stratification_at_k_samples(k: int) -> None:
    """Every batch's per-sample counts stay within the documented bound, at every K.

    Sizes span two orders of magnitude, mirroring the verified `MultiSampleReader`
    suite — the point being that the streaming path re-derives the SAME
    apportionment per shard rather than inheriting a global index.
    """
    rng = np.random.default_rng(100 + k)
    sizes = [int(n) for n in rng.integers(50, 9000, size=k)]
    total = sum(sizes)
    batch_size = 64
    dataset = _dataset(
        _multisample(sizes), batch_size=batch_size, drop_last=True, shuffle=False
    )
    batches = _raw_batches(dataset)
    assert batches, "streamed no batches"
    for batch in batches:
        labels = Counter(int(v) for v in batch["raw.event"]["process"])
        assert sum(labels.values()) == batch_size
        for group, n in enumerate(sizes):
            # a batch is the difference of two prefixes, each within
            # +/- interleave_block of ideal, so the per-batch bound is 2x that
            assert abs(labels[group] - batch_size * n / total) <= 2


def test_determinism_same_seed_and_epoch() -> None:
    """Same (seed, epoch, shard) -> identical stream; a new epoch reorders the same rows."""
    sizes = [40, 130, 7]
    kw = {"batch_size": 4, "drop_last": False, "shuffle": True, "seed": 11}
    a = _batch_uids(_dataset(_multisample(sizes), epoch=0, **kw))
    b = _batch_uids(_dataset(_multisample(sizes), epoch=0, **kw))
    c = _batch_uids(_dataset(_multisample(sizes), epoch=1, **kw))
    assert a == b
    assert a != c
    assert Counter(a) == Counter(c)


def test_shards_are_disjoint_and_deterministic() -> None:
    """Different shards share no row, and each is reproducible on its own."""
    sizes = [200, 90]
    per_shard = _stream_uids(lambda: _multisample(sizes), n_shards=4, batch_size=4, drop_last=False)
    for i, a in enumerate(per_shard):
        for b in per_shard[i + 1 :]:
            assert not (set(a) & set(b))
    again = _stream_uids(lambda: _multisample(sizes), n_shards=4, batch_size=4, drop_last=False)
    assert per_shard == again


def test_even_termination_across_ranks() -> None:
    """With drop_last every shard emits the same number of batches, and the loss is bounded."""
    sizes = [37, 411, 5]
    n_shards, batch_size = 6, 8
    counts, emitted = [], 0
    for shard in range(n_shards):
        dataset = _dataset(
            _multisample(sizes),
            batch_size=batch_size,
            world_size=n_shards,
            rank=shard,
            num_workers=1,
            worker_id=0,
            drop_last=True,
        )
        uids = _batch_uids(dataset)
        assert len(uids) == dataset.n_batches() * batch_size == len(dataset) * batch_size
        counts.append(dataset.n_batches())
        emitted += len(uids)
    assert len(set(counts)) == 1, f"ranks disagree on batch count: {counts}"
    dropped = sum(sizes) - emitted
    assert 0 <= dropped <= n_shards * (batch_size + len(sizes))


def test_no_drop_last_streams_every_row() -> None:
    """`drop_last=False` keeps the ragged tail."""
    sizes = [37, 411, 5]
    uids = _batch_uids(_dataset(_multisample(sizes), batch_size=8, drop_last=False))
    assert Counter(uids) == Counter(range(sum(sizes)))


# --------------------------------------------------------------------------- #
# bounded buffers, fork safety, single-sample readers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_live", [1, 2, 3])
def test_max_live_streams_never_loses_rows(max_live: int) -> None:
    """Evicting a part-consumed block costs a re-read, never a row."""
    sizes = [60, 200, 45, 90]
    uids = _batch_uids(
        _dataset(
            _multisample(sizes),
            batch_size=8,
            drop_last=False,
            shuffle=False,
            max_live_streams=max_live,
        )
    )
    assert Counter(uids) == Counter(range(sum(sizes)))


def test_block_rows_bounds_every_reader_call() -> None:
    """No single `read_block` call exceeds `block_rows` rows."""
    sizes = [500, 300]
    dataset = _dataset(
        _multisample(sizes), batch_size=8, drop_last=False, shuffle=False, block_rows=32
    )
    seen: list[int] = []
    original = dataset._read_block

    def spy(block):  # noqa: ANN001, ANN202
        seen.append(block.n_rows)
        return original(block)

    dataset._read_block = spy  # type: ignore[method-assign]
    _batch_uids(dataset)
    assert seen and max(seen) <= 32


def test_nothing_open_before_iteration_and_pickle_roundtrip() -> None:
    """The dataset pickles as a DataLoader worker would, and only then reads."""
    dataset = _dataset(_multisample([30, 70]), batch_size=5, drop_last=False, shuffle=False)
    assert dataset._layout is None, "layout resolved at construction — would break fork safety"
    revived = pickle.loads(pickle.dumps(dataset))
    assert revived._layout is None
    assert Counter(_batch_uids(revived)) == Counter(range(100))


def test_single_sample_reader_needs_no_multisample_wrapper() -> None:
    """A plain reader streams through the default one-group path."""
    uids = _batch_uids(
        _dataset(
            BlockStubReader(n=250, n_blocks=4), batch_size=10, drop_last=False, shuffle=False
        )
    )
    assert uids == list(range(250))


def test_default_row_blocks_is_one_whole_block() -> None:
    """A reader that overrides nothing still streams correctly."""
    reader = BlockStubReader(n=64, n_blocks=1)
    blocks = Reader.row_blocks(reader)
    assert blocks == [RowBlock(group=0, start=0, stop=64)]


# --------------------------------------------------------------------------- #
# corpus manifest (T4) — planning without touching the corpus
# --------------------------------------------------------------------------- #


def test_manifest_round_trips_and_matches_the_reader(tmp_path) -> None:  # noqa: ANN001
    """A saved manifest reproduces the reader's blocks and row counts exactly."""
    reader = _multisample([40, 130, 7], n_blocks=3)
    built = build_manifest(reader)
    path = built.save(tmp_path / "corpus.json")
    loaded = CorpusManifest.load(path)
    assert loaded.blocks() == reader.row_blocks()
    assert loaded.group_rows == [40, 130, 7]
    assert loaded.n_rows == 177
    assert loaded.group_names == ["s0", "s1", "s2"]


def test_manifest_plans_the_same_shards_as_the_reader(tmp_path) -> None:  # noqa: ANN001
    """A manifest-planned shard stream is identical to a reader-planned one."""
    sizes = [40, 130, 7]
    manifest = build_manifest(_multisample(sizes)).save(tmp_path / "corpus.json")
    kw = {"batch_size": 4, "drop_last": False, "shuffle": False}
    from_reader = _batch_uids(_dataset(_multisample(sizes), **kw))
    from_manifest = _batch_uids(_dataset(_multisample(sizes), manifest=manifest, **kw))
    assert from_reader == from_manifest


def test_manifest_sizes_the_epoch_without_a_reader_index(tmp_path) -> None:  # noqa: ANN001
    """`n_batches` comes off the manifest — `row_blocks` is never called."""
    sizes = [40, 130, 7]
    manifest = build_manifest(_multisample(sizes)).save(tmp_path / "corpus.json")
    reader = _multisample(sizes)
    calls = []
    original = reader.row_blocks

    def spy():  # noqa: ANN202
        calls.append(1)
        return original()

    reader.row_blocks = spy  # type: ignore[method-assign]
    dataset = _dataset(reader, batch_size=4, drop_last=False, manifest=manifest)
    assert dataset.n_batches() == -(-177 // 4)
    assert calls == [], "the manifest path still asked the reader for its blocks"


def test_manifest_detects_a_changed_corpus(tmp_path) -> None:  # noqa: ANN001
    """Validation is stat-based: a resized file makes the manifest stale."""
    data = tmp_path / "f.root"
    data.write_bytes(b"x" * 100)
    st = data.stat()
    manifest = CorpusManifest(
        entries=[
            ManifestEntry(
                group=0, start=0, stop=10, path=str(data), size=st.st_size, mtime_ns=st.st_mtime_ns
            )
        ],
        schema_hash="abc",
    )
    assert manifest.validate(schema_hash="abc") == []
    data.write_bytes(b"x" * 200)
    problems = manifest.validate(schema_hash="abc")
    assert problems and "size changed" in problems[0]
    assert any("schema hash" in p for p in manifest.validate(schema_hash="different"))


def test_manifest_rejects_a_foreign_format_version(tmp_path) -> None:  # noqa: ANN001
    """A future/older artifact is a hard error, never a partial read."""
    path = tmp_path / "corpus.json"
    path.write_text('{"version": 999, "entries": [], "group_names": ["default"]}')
    with pytest.raises(ConfigError):
        CorpusManifest.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": 4, "block_rows": 0},
        {"batch_size": 4, "interleave_block": 0},
        {"batch_size": 4, "max_live_streams": 0},
        {"batch_size": 4, "world_size": 2, "rank": 5},
    ],
)
def test_rejects_impossible_configuration(kwargs: dict) -> None:
    """Bad knobs fail at construction (or at shard resolution), never silently."""
    with pytest.raises(ConfigError):
        dataset = IterableGraphDataset(
            {"reader": BlockStubReader(n=10), "feats": Features(variables={"event": ["val"]})},
            mode=Mode.FIT,
            sinks=["inputs.event"],
            **kwargs,
        )
        dataset.shard()
