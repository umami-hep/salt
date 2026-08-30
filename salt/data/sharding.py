"""Pure sharding + interleave arithmetic for `IterableSaltDataset` — no reader, no
torch, no I/O, so the streaming guarantees are testable without a dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from salt.data.base import RowBlock
from salt.graph.errors import ConfigError

__all__ = ["interleave_plan", "partition_blocks", "shard_row_counts", "split_blocks"]


def _interval(index: int, n_shards: int, n_rows: int) -> tuple[int, int]:
    """Shard `index`'s half-open row interval of a group of `n_rows` rows.

    ``[floor(i*N/S), floor((i+1)*N/S))``. The intervals tile ``[0, N)`` exactly,
    so coverage and disjointness are properties of the arithmetic rather than
    things the caller has to check, and no shard can be empty for any reason
    other than the group genuinely having fewer rows than shards.
    """
    return (index * n_rows) // n_shards, ((index + 1) * n_rows) // n_shards


def split_blocks(blocks: Sequence[RowBlock], lo: int, hi: int) -> list[RowBlock]:
    """The parts of `blocks` (one group, ascending, contiguous) inside rows ``[lo, hi)``.

    Blocks straddling either edge are cut, which is what makes sub-file sharding
    the general case: a shard whose interval lands mid-file simply gets a
    shorter contiguous range, never a whole file it does not own.
    """
    out: list[RowBlock] = []
    for block in blocks:
        start, stop = max(block.start, lo), min(block.stop, hi)
        if stop > start:
            out.append(block.subrange(start, stop))
    return out


def partition_blocks(
    blocks: Sequence[RowBlock],
    group_rows: Sequence[int],
    n_shards: int,
    shard_id: int,
    block_rows: int | None = None,
) -> dict[int, list[RowBlock]]:
    """Blocks per group owned by one shard, from the interval partition.

    Every group is split independently, so each shard holds the same PROPORTION
    of every sample — the property the per-shard stratified interleave then
    rests on. Blocks longer than `block_rows` are chopped into consecutive
    pieces so one `read` never exceeds the measured buffer budget.

    Parameters
    ----------
    blocks : Sequence[RowBlock]
        Every group's blocks (as returned by `Reader.row_blocks`).
    group_rows : Sequence[int]
        Row count per group, indexed by group id.
    n_shards : int
        Total shards (world_size * num_workers).
    shard_id : int
        This shard, in ``[0, n_shards)``.
    block_rows : int | None, optional
        Maximum rows per read; ``None`` leaves blocks whole.

    Returns
    -------
    dict[int, list[RowBlock]]
        ``{group: blocks}`` for the groups this shard has rows in, each list
        ascending and disjoint from every other shard's.

    Raises
    ------
    ConfigError
        On a non-positive `n_shards`, an out-of-range `shard_id`, or a
        non-positive `block_rows`.
    """
    if n_shards < 1:
        raise ConfigError(f"n_shards must be >= 1, got {n_shards}")
    if not 0 <= shard_id < n_shards:
        raise ConfigError(f"shard_id {shard_id} out of range for {n_shards} shard(s)")
    if block_rows is not None and block_rows < 1:
        raise ConfigError(f"block_rows must be >= 1, got {block_rows}")
    by_group: dict[int, list[RowBlock]] = {}
    for block in blocks:
        by_group.setdefault(block.group, []).append(block)
    out: dict[int, list[RowBlock]] = {}
    for group, group_blocks in sorted(by_group.items()):
        n_rows = int(group_rows[group])
        lo, hi = _interval(shard_id, n_shards, n_rows)
        mine = split_blocks(sorted(group_blocks, key=lambda b: b.start), lo, hi)
        if block_rows is not None:
            mine = [
                piece
                for block in mine
                for piece in (
                    block.subrange(s, min(s + block_rows, block.stop))
                    for s in range(block.start, block.stop, block_rows)
                )
            ]
        if mine:
            out[group] = mine
    return out


def shard_row_counts(group_rows: Sequence[int], n_shards: int) -> list[int]:
    """Rows every shard receives, from the same interval arithmetic.

    Every rank can compute this locally, which is what lets the epoch length be
    agreed without a collective: no rank waits on a straggler because no rank
    has to ask.
    """
    return [
        sum(hi - lo for lo, hi in (_interval(shard, n_shards, int(n)) for n in group_rows))
        for shard in range(n_shards)
    ]


def interleave_plan(
    counts: dict[int, int], interleave_block: int = 1, n_rows: int | None = None
) -> np.ndarray:
    """Group id per output row, proportionally stratified by largest remainder.

    The same apportionment `MultiSampleReader._build_index` uses, applied to
    ONE shard's counts instead of the whole corpus — which is what makes it
    affordable to recompute per epoch. At every prefix, each group's emitted
    count stays within `interleave_block` of its ideal share.

    Parameters
    ----------
    counts : dict[int, int]
        Rows available per group in this shard.
    interleave_block : int, optional
        Rows emitted per turn of the round robin, by default 1.
    n_rows : int | None, optional
        Truncate the plan to this many rows (the drop-last epoch length);
        ``None`` emits every row.

    Returns
    -------
    np.ndarray
        ``(n,)`` int64 group ids.

    Raises
    ------
    ConfigError
        If `interleave_block` is below 1.
    """
    if interleave_block < 1:
        raise ConfigError(f"interleave_block must be >= 1, got {interleave_block}")
    groups = sorted(counts)
    lens = [int(counts[g]) for g in groups]
    total = sum(lens)
    n = total if n_rows is None else min(int(n_rows), total)
    plan = np.empty(n, dtype=np.int64)
    emitted = [0] * len(groups)
    j = 0
    while j < n:
        best_i, best_deficit = -1, -np.inf
        target = j + 1
        for i, n_i in enumerate(lens):
            if emitted[i] >= n_i:
                continue  # group exhausted
            deficit = target * (n_i / total) - emitted[i]
            if deficit > best_deficit:
                best_deficit, best_i = deficit, i
        take = min(interleave_block, lens[best_i] - emitted[best_i], n - j)
        plan[j : j + take] = groups[best_i]
        emitted[best_i] += take
        j += take
    return plan
