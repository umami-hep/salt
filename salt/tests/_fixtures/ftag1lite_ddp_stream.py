"""Picklable harness for the FTAG1LITE streaming tests.

Everything a spawned rank or a forkserver DataLoader worker unpickles lives here and is
importable by name: the grouped in-memory stub reader, the two datamodule builders (stub
leg and synthetic-FTAG1LITE leg) and the per-rank `child_main` of the real 2-rank gloo +
forkserver tests; and the unit suite's interleaved stub, dataset builders and cell generators.
"""

from __future__ import annotations

import copy
import json
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.datamodule import SaltDataModule
from salt.data.processors.features import Features
from salt.data.readers.ftag1lite_reader import FTAG1LITEStreamingDataset, shard_assignment
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema
from salt.tests._fixtures.joint_ftag1lite_corpus import corpus_glob, make_reader

SIZES = [40, 130, 7, 61]
"""Rows per group: 4 uneven groups, 238 rows, uids ``0..237`` in global row order."""

BATCH = 8
N_BLOCKS = 3
POOL_ROWS = 4 * BATCH
"""The per_rank shuffle pool: a small whole number of batches."""

JET_VARS = ["pt", "jet_type"]
"""``inputs.jets`` columns of the FTAG1LITE leg. ``pt`` (a distinct integer per jet), the
``jet_type`` and the event's ``eventNumber`` identify one corpus row exactly in float32."""

_ROW = np.dtype([("uid", "int64"), ("val", "float32")])


def group_of(uid: int, sizes: Sequence[int] = SIZES) -> int:
    """The stub group holding global row `uid`."""
    return int(np.searchsorted(np.cumsum(sizes), uid, side="right"))


class GroupedStubReader(Reader):
    """In-memory grouped reader over the ``event`` stream; ``uid`` is the global row index.

    Blocks are in GLOBAL row coordinates (the `FTAG1LITEReader` convention): group ``g``
    covers ``[offset_g, offset_g + n_g)``. `shard_scope` / `shuffle_pool_rows` are plain
    attributes the streaming dataset reads, as on `FTAG1LITEReader`. No `files`/`_table`,
    so its manifest is sourceless and path-less.
    """

    def __init__(
        self,
        sizes: Sequence[int] = SIZES,
        n_blocks: int = N_BLOCKS,
        seed: int = 0,
        shard_scope: str = "per_rank",
        shuffle_pool_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.sizes = [int(n) for n in sizes]
        self.n_blocks = int(n_blocks)
        self.shard_scope = str(shard_scope)
        self.shuffle_pool_rows = None if shuffle_pool_rows is None else int(shuffle_pool_rows)
        self.offsets = [0, *np.cumsum(self.sizes).tolist()]
        rng = np.random.default_rng(seed)
        rows = np.empty((sum(self.sizes),), dtype=_ROW)
        rows["uid"] = np.arange(len(rows), dtype=np.int64)
        rows["val"] = rng.uniform(0, 1, size=len(rows)).astype(np.float32)
        self._rows = rows
        self.schema = Schema(
            groups={"event": GroupSchema(fields={"uid": "int64", "val": "float32"})}
        )

    @property
    def streams(self) -> tuple[str, ...]:
        return ("event",)

    @property
    def iterable_dataset_class(self) -> type:
        """The FTAG1LITE streaming dataset, as on `FTAG1LITEReader`."""
        return FTAG1LITEStreamingDataset

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat = {"raw.event": TensorSpec(shape=("B",), kind="data", fields=("uid", "val"))}
        return IO(produces=unflatten_spec(flat))

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def group_names(self) -> list[str]:
        """One name per group — what makes `build_manifest` record one group per entry."""
        return [f"g{i}" for i in range(len(self.sizes))]

    def row_blocks(self) -> list[RowBlock]:
        """`n_blocks` near-equal blocks per group, in global row coordinates, ascending."""
        blocks = []
        for group, n in enumerate(self.sizes):
            off = self.offsets[group]
            edges = [(i * n) // self.n_blocks for i in range(self.n_blocks + 1)]
            blocks.extend(
                RowBlock(group=group, start=off + lo, stop=off + hi)
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)
                if hi > lo
            )
        return blocks

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        """Global rows ``[start, stop)``; a block outside its own group is a layout bug."""
        del mode
        lo, hi = self.offsets[block.group], self.offsets[block.group + 1]
        if not lo <= block.start <= block.stop <= hi:
            raise ValueError(f"block {block} lies outside group {block.group} = [{lo}, {hi})")
        return {"raw.event": self._rows[block.start : block.stop].copy()}

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del mode
        return {"raw.event": self._rows[rows].copy()}

    def with_source(self, filename, num=-1, stage=None):
        """A copy (same data, name and streaming knobs) bound to `stage`."""
        del filename, num
        clone = copy.copy(self)
        clone.stage = stage
        return clone

    def sources(self) -> list[Path]:
        return []

    def prepare(self) -> None:
        pass

    def bind(self, ctx: WorkerCtx) -> None:
        del ctx

    def schema_group(self, stream: str) -> GroupSchema | None:
        return self.schema.groups.get(stream) if self.schema is not None else None


def _pool(scope: str) -> int | None:
    """``owner`` requires no shuffle pool; ``per_rank`` uses `POOL_ROWS`."""
    return None if scope == "owner" else POOL_ROWS


def _datamodule(
    reader: Reader,
    variables: dict[str, list[str]],
    source: str,
    out_dir: Path,
    *,
    shuffle: bool,
) -> SaltDataModule:
    """The shared streaming datamodule: batch 8, 2 persistent forkserver workers."""
    sinks = [f"inputs.{stream}" for stream in variables]
    return SaltDataModule(
        modules={"reader": reader, "feats": Features(variables=variables)},
        train_file=source,
        val_file=source,
        batch_size=BATCH,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        pin_memory=False,
        iterable=True,
        block_rows=None,
        max_live_streams=None,
        manifest={
            "train": str(out_dir / "train_manifest.json"),
            "val": str(out_dir / "val_manifest.json"),
        },
        sinks={Mode.FIT: sinks, Mode.VAL: sinks},
        seed=42,
        shuffle_stream=shuffle,
    )


def make_datamodule(out_dir: Path, *, scope: str, shuffle: bool) -> SaltDataModule:
    """A streaming datamodule over `GroupedStubReader` with reader scope `scope`."""
    reader = GroupedStubReader(shard_scope=scope, shuffle_pool_rows=_pool(scope))
    return _datamodule(
        reader, {"event": ["uid", "val"]}, str(out_dir / "unused"), out_dir, shuffle=shuffle
    )


def make_ftag1lite_datamodule(corpus_root: Path, out_dir: Path, *, scope: str) -> SaltDataModule:
    """A streaming datamodule over the synthetic FTAG1LITE corpus (``jets: both``)."""
    reader = make_reader(
        Path(corpus_root), jets="both", shard_scope=scope, shuffle_pool_rows=_pool(scope)
    )
    variables = {"jets": JET_VARS, "event": ["eventNumber"]}
    return _datamodule(reader, variables, corpus_glob(corpus_root), out_dir, shuffle=False)


def _stub_batches(loader) -> list[list[int]]:
    """Iterate `loader` fully; per-batch uids (column 0 of ``inputs.event``, float32-exact)."""
    return [[int(u) for u in batch["inputs"]["event"][:, 0].tolist()] for batch in loader]


def _ftag1lite_batches(loader) -> list[list[list[float]]]:
    """Iterate `loader` fully; per batch, per row ``[eventNumber, jet_type, pt]``."""
    out = []
    for batch in loader:
        jets = batch["inputs"]["jets"]
        event = batch["inputs"]["event"]
        cols = [event[:, 0], jets[:, JET_VARS.index("jet_type")], jets[:, JET_VARS.index("pt")]]
        out.append(torch.stack(cols, dim=1).tolist())
    return out


def child_main(
    rank: int,
    world_size: int,
    out_dir: str,
    scope: str,
    shuffle: bool = False,
    epochs: int = 1,
    corpus_root: str | None = None,
) -> None:
    """One rank: join the file-store gloo group, stream train/val, write ``rank{rank}.json``.

    `corpus_root` None runs the stub leg, else the FTAG1LITE leg over that corpus. With
    ``epochs > 1`` the SAME train loader is iterated again: its persistent workers must
    reshuffle on their own (no epoch hook is called). Any failure writes its traceback to
    ``rank{rank}.error.txt`` and re-raises, so the process exits non-zero.
    """
    try:
        torch.set_num_threads(1)
        import torch.distributed as dist

        dist.init_process_group(
            "gloo",
            init_method=f"file://{out_dir}/pg",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        sub = Path(out_dir) / f"rank{rank}"
        sub.mkdir(parents=True, exist_ok=True)
        if corpus_root is None:
            dm = make_datamodule(sub, scope=scope, shuffle=shuffle)
            batches = _stub_batches
        else:
            dm = make_ftag1lite_datamodule(Path(corpus_root), sub, scope=scope)
            batches = _ftag1lite_batches
        # no prepare_data: each rank builds its own manifests under its own dir
        dm.setup("fit")
        train = dm.train_dataloader()
        val = dm.val_dataloader()
        result: dict[str, Any] = {
            "train": {"len": len(train), "batches": batches(train)},
            "val": {"len": len(val), "batches": batches(val)},
            "epochs": {},
        }
        del val  # shuts its persistent workers down before the epoch loop
        result["epochs"]["0"] = result["train"]["batches"]
        for epoch in range(1, epochs):
            result["epochs"][str(epoch)] = batches(train)
        (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(result))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        (Path(out_dir) / f"rank{rank}.error.txt").write_text(traceback.format_exc())
        raise


# --------------------------------------------------------------------------- #
# unit-suite harness (test_ftag1lite_reader.py): an interleaved stub, dataset builders and
# seeded cell generators
# --------------------------------------------------------------------------- #

SEED = 42
"""Every unit-suite dataset's seed, so owner's permutation is reconstructible."""

_GRP_ROW = np.dtype([("uid", "int64"), ("grp", "int64")])


class InterleavedStubReader(Reader):
    """In-memory grouped reader: ``uid`` is the GLOBAL row, ``grp`` the group. Each group is cut
    into `n_blocks` blocks laid out round-robin, so a group's blocks are non-contiguous (as
    FTAG1LITE containers can be); `read_block` refuses a block outside its group's blocks.
    """

    def __init__(
        self,
        sizes: Sequence[int] = SIZES,
        n_blocks: int = N_BLOCKS,
        shard_scope: str = "per_rank",
        shuffle_pool_rows: int | None = 100_000,
    ) -> None:
        super().__init__()
        self.sizes = [int(n) for n in sizes]
        self.n_blocks = int(n_blocks)
        self.shard_scope = shard_scope
        self.shuffle_pool_rows = shuffle_pool_rows
        edges = [[(i * n) // self.n_blocks for i in range(self.n_blocks + 1)] for n in self.sizes]
        parts = [[int(d) for d in np.diff(e) if d > 0] for e in edges]
        blocks: list[RowBlock] = []
        start = 0
        for b in range(self.n_blocks):
            for group, lens in enumerate(parts):
                if b < len(lens):
                    blocks.append(RowBlock(group=group, start=start, stop=start + lens[b]))
                    start += lens[b]
        self._blocks = blocks
        rows = np.empty((start,), dtype=_GRP_ROW)
        rows["uid"] = np.arange(start, dtype=np.int64)
        for block in blocks:
            rows["grp"][block.start : block.stop] = block.group
        self._rows = rows
        self.schema = Schema(groups={"event": GroupSchema(fields={"uid": "int64", "grp": "int64"})})

    @property
    def streams(self) -> tuple[str, ...]:
        return ("event",)

    @property
    def group_names(self) -> list[str]:
        return [f"c{g}" for g in range(len(self.sizes))]

    @property
    def iterable_dataset_class(self) -> type:
        return FTAG1LITEStreamingDataset

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat = {
            "raw.event": TensorSpec(shape=("B",), kind="data", fields=("uid", "grp")),
            "meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST),
        }
        return IO(produces=unflatten_spec(flat))

    def __len__(self) -> int:
        return len(self._rows)

    def group_uids(self, group: int) -> np.ndarray:
        """Group `group`'s global rows in group-local order (its blocks, start-sorted)."""
        mine = sorted((b for b in self._blocks if b.group == group), key=lambda b: b.start)
        return np.concatenate([np.arange(b.start, b.stop) for b in mine])

    def row_blocks(self) -> list[RowBlock]:
        return list(self._blocks)

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        if not any(
            b.group == block.group and b.start <= block.start <= block.stop <= b.stop
            for b in self._blocks
        ):
            raise ValueError(f"{block} is not inside one of group {block.group}'s blocks")
        return self.read(slice(block.start, block.stop), mode)

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        out = {"raw.event": self._rows[rows].copy()}
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out

    def schema_group(self, stream: str) -> GroupSchema | None:
        return self.schema.groups.get(stream) if self.schema is not None else None


def _scoped(sizes: Sequence[int], scope: str) -> InterleavedStubReader:
    """A stub for `scope`: FIT ``owner`` needs a null pool, ``per_rank`` keeps the default."""
    pool = None if scope == "owner" else 100_000
    return InterleavedStubReader(sizes, shard_scope=scope, shuffle_pool_rows=pool)


def _dataset(
    reader: Reader, batch_size: int = 8, mode: Mode = Mode.FIT, **kwargs: Any
) -> FTAG1LITEStreamingDataset:
    """``inputs.event = (uid, grp)``; FIT shuffles + drops the tail, VAL/TEST keep order + tail."""
    kwargs.setdefault("block_rows", None)
    kwargs.setdefault("max_live_streams", None)
    kwargs.setdefault("seed", SEED)
    if mode != Mode.FIT:
        kwargs.setdefault("shuffle", False)
        kwargs.setdefault("drop_last", False)
    modules = {"reader": reader, "feats": Features(variables={"event": ["uid", "grp"]})}
    sinks = ["inputs.event", "meta.rows"] if mode == Mode.TEST else ["inputs.event"]
    return FTAG1LITEStreamingDataset(
        modules, mode=mode, sinks=sinks, batch_size=batch_size, **kwargs
    )


def _shard_dataset(
    reader: Reader, world: int, workers: int, shard: int, **kw: Any
) -> FTAG1LITEStreamingDataset:
    """The dataset of global shard `shard` on a (world x workers) grid, in this process."""
    rank, wid = divmod(shard, workers)
    return _dataset(reader, world_size=world, rank=rank, num_workers=workers, worker_id=wid, **kw)


def _batches(source) -> list[np.ndarray]:
    """Every batch's ``(uid, grp)`` rows as int64, read at the torch boundary."""
    return [batch["inputs"]["event"].numpy().astype(np.int64) for batch in source]


def _uids(batches: list[np.ndarray]) -> list[int]:
    return [int(u) for b in batches for u in b[:, 0]]


def _group_counts(batches: list[np.ndarray]) -> Counter:
    return Counter(int(g) for b in batches for g in b[:, 1])


def _assignment(
    sizes: Sequence[int], n_shards: int, scope: str, n_workers: int = 1
) -> list[dict[int, tuple[int, int]]]:
    return shard_assignment(sizes, n_shards, scope, seed=SEED, n_workers=n_workers)


def _expected_uids(reader: InterleavedStubReader, mapping: Mapping[int, tuple[int, int]]) -> set:
    """The global rows a shard's group-local intervals address (an independent mapping)."""
    return {int(u) for g, (lo, hi) in mapping.items() for u in reader.group_uids(g)[lo:hi]}


def _row_counts(assignment: Sequence[Mapping[int, tuple[int, int]]]) -> list[int]:
    return [sum(hi - lo for lo, hi in mapping.values()) for mapping in assignment]


def _random_group_rows(seed: int, n_groups: int, lo: int = 500, hi: int = 50_000) -> list[int]:
    return np.random.default_rng(seed).integers(lo, hi, size=n_groups).tolist()


def _owner_group_rows(seed: int, n_groups: int) -> list[int]:
    """`n_groups - 1` seeded 500..50,000-row groups plus one 45%-of-total group, seeded index."""
    rng = np.random.default_rng(seed)
    others = rng.integers(500, 50_000, size=n_groups - 1).tolist()
    big = round(0.45 / 0.55 * sum(others))
    others.insert(int(rng.integers(0, n_groups)), big)
    return others


_WINDOW_CHOICES = (3, 64, 1000)
_TOTAL_WINDOWS_BUDGET = 1_500_000


def _log_uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    x = int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi + 1))))))
    return int(np.clip(x, lo, hi))


def _random_window_schedule_fixtures(n: int, seed: int) -> list[tuple[dict[int, int], int, int]]:
    """`n` seeded cells: G in [1, 100], window 3/64/1000, log-uniform counts and T (budgeted)."""
    rng = np.random.default_rng(seed)
    fixtures: list[tuple[dict[int, int], int, int]] = []
    budget_left = _TOTAL_WINDOWS_BUDGET
    attempts = 0
    while len(fixtures) < n and attempts < n * 20:
        attempts += 1
        n_groups = int(rng.integers(1, 101))
        window = int(rng.choice(_WINDOW_CHOICES))
        lens = [_log_uniform_int(rng, 1, 10**6) for _ in range(n_groups)]
        total = sum(lens)
        t_windows = min(_log_uniform_int(rng, 1, 2000), total // window)
        t_windows = min(t_windows, max(1, budget_left // max(n - len(fixtures), 1)))
        if t_windows < 1 or t_windows * window > total:
            continue
        budget_left -= t_windows
        fixtures.append((dict(enumerate(lens)), t_windows * window, window))
    assert len(fixtures) >= n, f"only {len(fixtures)}/{n} fixtures in {attempts} attempts"
    return fixtures


def _adversarial_window_schedule_fixtures() -> list[tuple[str, dict[int, int], int, int]]:
    """Labelled boundary cells random draws rarely hit (``m_c``/``k`` ones: ``quotas == lens``)."""
    real_shape = [900_000, 1, 1, 1, *([137_500] * 7), 137_497]
    assert sum(real_shape) == 2_000_000
    return [
        ("mixed_m_zero_and_nonzero", dict(enumerate([98, 130, 111, 87, 64])), 490, 70),
        ("m_extremes_T_minus_1_and_1", dict(enumerate([41, 120, 119])), 280, 40),
        ("one_row_group_M_zero", dict(enumerate([1, 999_999])), 100, 10),
        ("single_atom", {0: 500}, 100, 10),
        ("k_zero_all_integer", dict(enumerate([40, 80, 120, 160, 200])), 600, 150),
        ("k_equals_g_minus_one", dict(enumerate([80, 80, 80, 70, 70, 70])), 450, 50),
        ("T_equals_one", dict(enumerate([500, 300, 150, 50])), 37, 37),
        ("window_equals_one", dict(enumerate([10_000, 5_000, 2_000])), 500, 1),
        ("plan_real_shape", dict(enumerate(real_shape)), 2_000_000, 1000),
    ]
