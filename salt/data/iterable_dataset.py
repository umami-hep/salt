"""`IterableGraphDataset` — file-sharded sequential streaming over any `Reader`,
with per-shard proportional interleaving and bounded buffers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from salt.data.base import RowBlock, SaltDatasetModule
from salt.data.manifest import CorpusManifest
from salt.data.plan_runner import _PlanRunner
from salt.data.readers.stream import pad_fill
from salt.data.sharding import interleave_plan, partition_blocks, shard_row_counts
from salt.graph.errors import ConfigError
from salt.graph.planner import Sinks
from salt.graph.spec import Mode

__all__ = ["DEFAULT_BLOCK_ROWS", "IterableGraphDataset"]

DEFAULT_BLOCK_ROWS = 16_384
"""Rows per reader call.

Measured, not chosen: on the FTAG1LITE corpus the read-range curve
(experiment 06, job 3966) is flat from a whole file down to ~3,400 entries
(~16k jets) — 9,052 vs 9,070 jets/s — and falls off below it (7,275 jets/s at
859 entries, 4,853 at 212). 16,384 rows sits on that knee, holding one read's
decompressed footprint near 77 MB mean / 179 MB max instead of the 302 / 690 MB
a whole-file read costs, which is what makes many concurrent shards affordable.
"""


class IterableGraphDataset(_PlanRunner, IterableDataset):
    """Streaming dataset: sequential blocks, sharded across ranks x workers.

    Reads large contiguous `RowBlock`s from the reader instead of one
    batch-sized window at a time, fills batches from them, and interleaves
    groups with the proportional stratification `MultiSampleReader` already
    uses. Downstream is unchanged — the same compiled plan produces the same
    batch object as `GraphDataset`; only the order and the read granularity
    differ.

    Sharding is by ROW INTERVAL, not by file: shard ``s`` of ``S`` takes
    ``[floor(s*N/S), floor((s+1)*N/S))`` of every group. Sub-file splitting is
    therefore the general case rather than a fallback, shard balance is within
    one row per group, and ``S`` may exceed the file count — which it must,
    since O(100-1000) GPUs times DataLoader workers is thousands of shards
    against a corpus of a few hundred files.

    Parameters
    ----------
    modules : dict[str, SaltDatasetModule]
        The dataset modules by instance name (exactly one `Reader`).
    mode : Mode
        The primary mode to compile for.
    sinks : Sinks
        The model boundary's demanded keys.
    batch_size : int
        Rows per emitted batch.
    seed : int, optional
        Base seed for the epoch shuffle and read-time augmentations, by
        default 42.
    epoch : int, optional
        Epoch index; with `seed` it fixes the whole stream, by default 0. The
        datamodule sets it from ``trainer.current_epoch``.
    shuffle : bool, optional
        Permute block order within the shard and row order within each batch,
        by default True. Multiset-preserving either way.
    drop_last : bool, optional
        Emit exactly ``min_shard_rows // batch_size`` batches so every rank
        steps the same number of times, by default True. ``False`` streams every
        row this shard owns, including a ragged final batch.
    block_rows : int | None, optional
        Maximum rows per reader call, by default `DEFAULT_BLOCK_ROWS`; ``None``
        reads each of the reader's own blocks whole.
    interleave_block : int, optional
        Rows per turn of the group round robin, by default 1. Every batch's
        per-group counts stay within this many rows of their ideal share.
    manifest : CorpusManifest | str | Path | None, optional
        A prebuilt `CorpusManifest` (or its path) to plan shards from. With one,
        `n_batches`/`__len__`/the partition need no reader index at all, so a
        rank can size its epoch without touching the corpus. ``None`` (default)
        asks the reader for its blocks, which resolves `prepare` — cheap with an
        index cache, expensive without one.
    max_live_streams : int | None, optional
        Maximum groups holding a resident block at once, by default 2. Bounds
        peak resident rows at ``max_live_streams * block_rows``; ``None``
        removes the bound (one live stream per group).
    world_size, rank, num_workers, worker_id : int | None, optional
        Shard resolution overrides. ``None`` (the default) detects them from
        `torch.distributed` and `torch.utils.data.get_worker_info`. Injecting
        them is what lets tests sweep a whole (ranks x workers) grid in one
        process with no DDP and no DataLoader.
    debug : bool, optional
        Assert no boundary leaf aliases a reader buffer, by default False.
    sink_origins : Mapping[str, str] | None, optional
        Demanded-key -> demander description, for error messages.

    Raises
    ------
    ConfigError
        On a non-positive `batch_size` / `block_rows` / `interleave_block` /
        `max_live_streams`, or an out-of-range shard override.
    """

    def __init__(
        self,
        modules: dict[str, SaltDatasetModule],
        mode: Mode,
        sinks: Sinks,
        batch_size: int,
        seed: int = 42,
        epoch: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        block_rows: int | None = DEFAULT_BLOCK_ROWS,
        interleave_block: int = 1,
        manifest: CorpusManifest | str | Path | None = None,
        max_live_streams: int | None = 2,
        world_size: int | None = None,
        rank: int | None = None,
        num_workers: int | None = None,
        worker_id: int | None = None,
        debug: bool = False,
        sink_origins: Mapping[str, str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {batch_size}")
        if block_rows is not None and block_rows < 1:
            raise ConfigError(f"block_rows must be >= 1, got {block_rows}")
        if interleave_block < 1:
            raise ConfigError(f"interleave_block must be >= 1, got {interleave_block}")
        if max_live_streams is not None and max_live_streams < 1:
            raise ConfigError(f"max_live_streams must be >= 1, got {max_live_streams}")
        super().__init__(modules, mode, sinks, seed=seed, debug=debug, sink_origins=sink_origins)
        self.batch_size = int(batch_size)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.block_rows = block_rows
        self.interleave_block = int(interleave_block)
        self.max_live_streams = max_live_streams
        # a path is kept as a path (picklable, and loaded post-fork like
        # everything else that touches the filesystem)
        self.manifest = manifest
        self._world_size = world_size
        self._rank = rank
        self._num_workers = num_workers
        self._worker_id = worker_id
        # transient per-process layout, resolved lazily in __iter__ (post-fork)
        self._layout: tuple[list[RowBlock], list[int]] | None = None

    # -- shard resolution -----------------------------------------------------

    def _dist(self) -> tuple[int, int]:
        """``(world_size, rank)`` — overrides first, then `torch.distributed`, else (1, 0)."""
        if self._world_size is not None or self._rank is not None:
            world = 1 if self._world_size is None else int(self._world_size)
            rank = 0 if self._rank is None else int(self._rank)
            if not 0 <= rank < world:
                raise ConfigError(f"rank {rank} out of range for world_size {world}")
            return world, rank
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size(), dist.get_rank()
        except (ImportError, RuntimeError):  # no distributed build / not initialised
            pass
        return 1, 0

    def _worker(self) -> tuple[int, int]:
        """``(num_workers, worker_id)`` — overrides first, then the DataLoader, else (1, 0)."""
        if self._num_workers is not None or self._worker_id is not None:
            n = 1 if self._num_workers is None else max(1, int(self._num_workers))
            wid = 0 if self._worker_id is None else int(self._worker_id)
            if not 0 <= wid < n:
                raise ConfigError(f"worker_id {wid} out of range for num_workers {n}")
            return n, wid
        info = get_worker_info()
        if info is None:
            return 1, 0
        return max(1, int(info.num_workers)), int(info.id)

    def shard(self) -> tuple[int, int]:
        """The calling process's ``(n_shards, shard_id)``.

        ``n_shards = world_size * num_workers`` and
        ``shard_id = rank * num_workers + worker_id``, so shard ids are unique
        and contiguous across the whole job.
        """
        world, rank = self._dist()
        n_workers, worker_id = self._worker()
        return world * n_workers, rank * n_workers + worker_id

    # -- layout ---------------------------------------------------------------

    def _manifest(self) -> CorpusManifest | None:
        """The configured manifest, loaded from disk on first use; ``None`` if unset."""
        if self.manifest is None:
            return None
        if not isinstance(self.manifest, CorpusManifest):
            self.manifest = CorpusManifest.load(self.manifest)
        return self.manifest

    def _resolve_layout(self) -> tuple[list[RowBlock], list[int]]:
        """``(blocks, group_rows)`` for the whole corpus, cached per process.

        From the manifest when one is configured — no reader index, no file
        opened. Otherwise from the reader's `row_blocks`, which resolves
        `prepare`; with the index cache in place that also opens no data file,
        but it does have to build or restore a per-row index the manifest makes
        unnecessary for planning.
        """
        if self._layout is None:
            manifest = self._manifest()
            if manifest is not None:
                self._layout = (manifest.blocks(), manifest.group_rows)
            else:
                blocks = list(self._reader.row_blocks())
                n_groups = 1 + max((b.group for b in blocks), default=0)
                group_rows = [0] * n_groups
                for block in blocks:
                    group_rows[block.group] += block.n_rows
                self._layout = (blocks, group_rows)
        return self._layout

    def n_batches(self) -> int:
        """Batches this shard emits — equal across shards under `drop_last`.

        Under `drop_last` every shard uses the SMALLEST shard's row count, so all
        ranks step identically without exchanging anything. The rows this costs
        are bounded by ``batch_size + G`` per shard for `G` groups: at most `G`
        from the ``<= 1``-row-per-group interval residual, plus the partial final
        batch.
        """
        _blocks, group_rows = self._resolve_layout()
        n_shards, shard_id = self.shard()
        per_shard = shard_row_counts(group_rows, n_shards)
        if self.drop_last:
            return min(per_shard) // self.batch_size
        return -(-per_shard[shard_id] // self.batch_size)  # ceil

    def __len__(self) -> int:
        """Batches this shard emits (exact, not an estimate)."""
        return self.n_batches()

    # -- iteration ------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Stream this shard's batches for the current ``(seed, epoch)``.

        Everything that touches a file happens here, after any fork: the layout
        resolve, the reader bind, and every read. Nothing unpicklable is held at
        construction, so the dataset survives a spawn-context DataLoader.
        """
        self._maybe_bind()
        blocks, group_rows = self._resolve_layout()
        n_shards, shard_id = self.shard()
        mine = partition_blocks(blocks, group_rows, n_shards, shard_id, block_rows=self.block_rows)
        if not mine:
            return
        rng = np.random.default_rng([self._seed, self.epoch, shard_id])
        if self.shuffle:
            for group_blocks in mine.values():
                rng.shuffle(group_blocks)  # type: ignore[arg-type]
        counts = {group: sum(b.n_rows for b in bs) for group, bs in mine.items()}
        n_rows = self.n_batches() * self.batch_size if self.drop_last else sum(counts.values())
        plan = interleave_plan(counts, self.interleave_block, n_rows=n_rows)
        yield from self._stream(mine, plan, rng)

    def _stream(
        self, mine: dict[int, list[RowBlock]], plan: np.ndarray, rng: np.random.Generator
    ) -> Iterator[dict[str, Any]]:
        """Emit batches, each filled in the group order `plan` dictates."""
        cursors = {group: _GroupCursor(blocks) for group, blocks in mine.items()}
        live: list[int] = []
        max_live = self.max_live_streams or len(cursors)
        for start in range(0, len(plan), self.batch_size):
            wanted = plan[start : start + self.batch_size]
            if wanted.size == 0:
                break
            chunks: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = []
            for group in np.unique(wanted):
                positions = np.flatnonzero(wanted == group)
                cursor = cursors[int(group)]
                live = self._admit(int(group), live, cursors, max_live)
                filled = 0
                for produced, rows in cursor.take(positions.size, self._read_block):
                    chunks.append((positions[filled : filled + rows.size], produced, rows))
                    filled += rows.size
            yield self._run_plan(
                slice(0, int(wanted.size)), raw=self._assemble(chunks, int(wanted.size), rng)
            )

    def _admit(
        self, group: int, live: list[int], cursors: dict[int, _GroupCursor], max_live: int
    ) -> list[int]:
        """Make `group` the most recently used live stream, releasing the LRU buffers."""
        live = [g for g in live if g != group]
        while len(live) >= max_live:
            cursors[live.pop(0)].release()
        live.append(group)
        return live

    def _read_block(self, block: RowBlock) -> dict[str, np.ndarray]:
        """One reader call for one block (the `read_block` seam)."""
        return self._reader.read_block(block, self._mode)

    # -- batch assembly -------------------------------------------------------

    def _assemble(
        self,
        chunks: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]],
        b: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Scatter ``(positions, produced, rows)`` chunks into one batch.

        Same shape of operation as `MultiSampleReader._combine_scalar` /
        `_combine_jagged`, and for the same reason: two samples may serve
        different multiplicities, so a jagged stream is allocated at the batch's
        maximum ``T`` and each chunk written into its own prefix. Scattering to
        `positions` (rather than concatenating) keeps the batch in the interleave
        plan's order, which is what makes the unshuffled stream inspectable.

        With `shuffle` on, one permutation is applied to every stream — ORDER
        only, never membership, so the plan's per-group proportions survive.
        """
        keys = list(chunks[0][1])
        # The shuffle is folded INTO the scatter rather than applied after it.
        # `combined[order]` would be a second full pass over every stream, on top
        # of the gather that materialises each chunk and the scatter that places
        # it — three passes where one suffices. Writing a value destined for
        # ``combined[p]`` straight to ``out[inv[p]]`` (where ``order[inv[p]] == p``)
        # produces the identical array in one.
        inv = None
        if self.shuffle:
            order = rng.permutation(b)
            inv = np.empty(b, dtype=np.int64)
            inv[order] = np.arange(b, dtype=np.int64)
        out: dict[str, np.ndarray] = {}
        for key in keys:
            if key == "meta.rows":
                continue  # per-block metadata; the batch's own is written below
            ref = chunks[0][1][key]
            if ref.ndim == 1:
                combined = np.zeros((b,), dtype=ref.dtype)
                for positions, produced, rows in chunks:
                    dest = positions if inv is None else inv[positions]
                    combined[dest] = _rows_of(produced[key], rows)
            else:
                t = max(int(produced[key].shape[1]) for _pos, produced, _rows in chunks)
                # a single chunk spanning the whole batch at full width overwrites
                # every slot, so the initial fill it would need is dead work
                covered = (
                    len(chunks) == 1
                    and chunks[0][0].size == b
                    and (int(chunks[0][1][key].shape[1]) == t)
                )
                if ref.dtype.names is None and ref.dtype == np.bool_ and not covered:
                    # a pad mask: True MEANS padded, so a slot no chunk writes
                    # (a shorter sample's T-extension) must start True, not False
                    combined = np.ones((b, t), dtype=ref.dtype)
                elif covered:
                    combined = np.empty((b, t), dtype=ref.dtype)
                else:
                    combined = np.zeros((b, t), dtype=ref.dtype)
                if not covered:
                    for name in ref.dtype.names or ():
                        fill = pad_fill(np.dtype(ref.dtype[name]))
                        if fill:  # signed-int -1 sentinel; zeros/False already right
                            combined[name][:] = fill
                for positions, produced, rows in chunks:
                    dest = positions if inv is None else inv[positions]
                    block = _rows_of(produced[key], rows)
                    tb = int(block.shape[1])
                    if ref.dtype.names:
                        for name in ref.dtype.names:
                            combined[name][dest, :tb] = block[name]
                    else:
                        combined[dest, :tb] = block
            out[key] = combined
        if self._mode == Mode.TEST:
            out["meta.rows"] = np.array([0, b], dtype=np.int64)
        return out


class _GroupCursor:
    """One group's position in its shard: a block list, and at most one resident block."""

    def __init__(self, blocks: list[RowBlock]) -> None:
        self._blocks = blocks
        self._next_block = 0
        self._buffer: dict[str, np.ndarray] | None = None
        self._offset = 0
        self._size = 0

    def empty(self) -> bool:
        """Whether the resident block is exhausted (or was never read)."""
        return self._buffer is None or self._offset >= self._size

    def release(self) -> None:
        """Drop the resident block — where the memory bound is actually enforced.

        A block evicted part-way through has its UNCONSUMED tail pushed back onto
        the queue, so eviction costs a re-read and never a row: without this the
        rows between the cursor and the block's end would be silently skipped,
        which is exactly the class of bug the coverage test exists to catch.
        """
        if self._buffer is not None and self._offset < self._size:
            block = self._blocks[self._next_block - 1]
            self._blocks[self._next_block - 1] = block.subrange(
                block.start + self._offset, block.stop
            )
            self._next_block -= 1
        self._buffer = None
        self._offset = self._size = 0

    def take(
        self, n: int, read: Callable[[RowBlock], dict[str, np.ndarray]]
    ) -> list[tuple[dict[str, np.ndarray], np.ndarray]]:
        """Up to `n` rows as ``(produced, row_index)`` pairs, reading blocks as needed.

        Returns fewer rows only when the group is exhausted, which the interleave
        plan never asks for — it is built from these same block row counts.
        """
        pairs: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        while n > 0:
            if self.empty():
                if self._next_block >= len(self._blocks):
                    break
                self.release()  # exhausted: pushes nothing back, just frees
                self._buffer = read(self._blocks[self._next_block])
                self._next_block += 1
                self._offset = 0
                self._size = _n_rows(self._buffer)
                continue
            take = min(n, self._size - self._offset)
            assert self._buffer is not None
            pairs.append((
                self._buffer,
                np.arange(self._offset, self._offset + take, dtype=np.int64),
            ))
            self._offset += take
            n -= take
        return pairs


def _rows_of(arr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``arr[rows]``, as a VIEW whenever `rows` is one contiguous ascending run.

    `_GroupCursor.take` always hands back ``np.arange(offset, offset + n)``, so
    the fancy index it names is really a slice — and taking it as a slice removes
    a full copy of every stream per chunk, before the scatter that copies it
    again. The run is verified rather than assumed (the span check short-circuits
    the common miss; `rows` is at most one batch, so the confirming pass is
    nothing against the megabytes it saves).
    """
    n = rows.size
    if n and int(rows[-1]) - int(rows[0]) + 1 == n and bool(np.all(np.diff(rows) == 1)):
        start = int(rows[0])
        return arr[start : start + n]
    return arr[rows]


def _n_rows(produced: Mapping[str, np.ndarray]) -> int:
    """Rows in a produced dict (every stream shares the row axis)."""
    for key, value in produced.items():
        if key.startswith("raw."):
            return int(value.shape[0])
    raise ConfigError(f"reader produced no raw.* stream: {sorted(produced)}")
