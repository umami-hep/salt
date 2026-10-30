"""The ONLY test that forks real DataLoader workers under a real process group: 2 spawned
gloo ranks, each streaming a `SaltDataModule` through 2 persistent forkserver workers.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from salt.tests._fixtures.ddp_stream import BATCH, N_ROWS, child_main

assert dist.is_available(), "torch.distributed is required: this test must FAIL, not skip"

CORPUS = Counter(range(N_ROWS))
WORLD = 2
JOIN_TIMEOUT_S = 240


def _error(tmp_path, rank: int) -> str:  # noqa: ANN001
    """The rank's recorded traceback, if it wrote one."""
    path = tmp_path / f"rank{rank}.error.txt"
    return path.read_text() if path.exists() else "(no rank error file)"


def _run_ranks(tmp_path) -> dict[int, dict]:  # noqa: ANN001
    """Run `WORLD` spawned `child_main` ranks under one join deadline; return their results."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=child_main, args=(r, WORLD, str(tmp_path))) for r in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
    hung = [r for r, p in enumerate(procs) if p.is_alive()]
    for r in hung:
        procs[r].kill()
        procs[r].join(10)
    if hung:
        details = "\n".join(f"rank {r}: {_error(tmp_path, r)}" for r in hung)
        pytest.fail(f"rank(s) {hung} still running after {JOIN_TIMEOUT_S} s, killed\n{details}")
    for r, p in enumerate(procs):
        if p.exitcode != 0:
            pytest.fail(f"rank {r} exited with code {p.exitcode}\n{_error(tmp_path, r)}")
    return {r: json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(WORLD)}


def _flat(batches: list[list[int]]) -> list[int]:
    """All uids of a batch list, in order."""
    return [u for batch in batches for u in batch]


def test_workers_shard_by_real_rank(tmp_path) -> None:  # noqa: ANN001
    """Each rank's forkserver workers shard as that rank of the real 2-rank group."""
    res = _run_ranks(tmp_path)

    val = {r: _flat(out["val"]["batches"]) for r, out in res.items()}
    union = Counter(u for uids in val.values() for u in uids)
    assert union == CORPUS, (
        f"RANK_SHARDING_BROKEN: every VAL corpus row must be yielded exactly once across "
        f"ranks, got {sum(union.values())} rows for a {N_ROWS}-row corpus (a rank sharding "
        "as rank 0 of 1 yields the whole corpus)"
    )
    overlap = set(val[0]) & set(val[1])
    assert overlap == set(), (
        f"RANK_SHARDING_BROKEN: VAL uid sets overlap across ranks ({len(overlap)} uids)"
    )
    for r, out in res.items():
        assert len(out["val"]["batches"]) == out["val"]["len"], (
            f"rank {r}: VAL yielded {len(out['val']['batches'])} batches, len(loader) says "
            f"{out['val']['len']}"
        )

    train = {r: _flat(out["train"]["batches"]) for r, out in res.items()}
    seen = Counter(u for uids in train.values() for u in uids)
    assert seen, "TRAIN yielded no rows on any rank"
    assert max(seen.values()) == 1, (
        f"RANK_SHARDING_BROKEN: a TRAIN uid was yielded more than once across ranks "
        f"(max multiplicity {max(seen.values())})"
    )
    for r, out in res.items():
        sizes = {len(b) for b in out["train"]["batches"]}
        assert sizes == {BATCH}, f"rank {r}: TRAIN batch sizes {sorted(sizes)}, expected {BATCH}"
    counts = {r: len(out["train"]["batches"]) for r, out in res.items()}
    assert counts[0] == counts[1], f"TRAIN batch counts differ across ranks: {counts}"
    for r, out in res.items():
        assert counts[r] == out["train"]["len"], (
            f"rank {r}: TRAIN yielded {counts[r]} batches, len(loader) says {out['train']['len']}"
        )
    overlap = set(train[0]) & set(train[1])
    assert overlap == set(), (
        f"RANK_SHARDING_BROKEN: TRAIN uid sets overlap across ranks ({len(overlap)} uids)"
    )
