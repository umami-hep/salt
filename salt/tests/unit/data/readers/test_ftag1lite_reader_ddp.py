"""`FTAG1LITEStreamingDataset` under a REAL process group: 2 spawned gloo ranks, each
streaming a `SaltDataModule` through 2 persistent forkserver DataLoader workers.

Two legs: the grouped in-memory stub (4 uneven groups) and the synthetic two-container
FTAG1LITE corpus (``jets: both``). Truncation policy: TRAIN drops the tail (full batches
only, equal step counts on every rank); VAL keeps the ragged tail and always streams
``per_rank``, so every VAL row is yielded exactly once across ranks.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from salt.tests._fixtures.ftag1lite_ddp_stream import BATCH, SIZES, child_main, group_of
from salt.tests._fixtures.joint_ftag1lite_corpus import (
    CONTAINERS,
    JointCorpusFacts,
    build_joint_corpus,
)

assert dist.is_available(), "torch.distributed is required: this test must FAIL, not skip"

WORLD = 2
JOIN_TIMEOUT_S = 240
STUB_CORPUS = Counter(range(sum(SIZES)))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> JointCorpusFacts:
    """One synthetic FTAG1LITE corpus for the module, written before any rank spawns."""
    return build_joint_corpus(tmp_path_factory.mktemp("ftag1lite_ddp_corpus"))


def _error(tmp_path, rank: int) -> str:
    """The rank's recorded traceback, if it wrote one."""
    path = tmp_path / f"rank{rank}.error.txt"
    return path.read_text() if path.exists() else "(no rank error file)"


def _run_ranks(
    tmp_path,
    *,
    scope: str,
    shuffle: bool = False,
    epochs: int = 1,
    corpus_root: str | None = None,
) -> dict[int, dict]:
    """Run `WORLD` spawned `child_main` ranks under one join deadline; return their results."""
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=child_main,
            args=(r, WORLD, str(tmp_path), scope, shuffle, epochs, corpus_root),
        )
        for r in range(WORLD)
    ]
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


def _uids(batches: list[list[int]]) -> list[int]:
    """All stub uids of a batch list, in order."""
    return [u for batch in batches for u in batch]


def _identities(batches: list[list[list[float]]]) -> list[tuple[int, int, float]]:
    """``(eventNumber, jet_type, pt)`` for every yielded FTAG1LITE row, in order."""
    return [(int(e), int(t), float(p)) for batch in batches for e, t, p in batch]


def _assert_shards(res: dict[int, dict], keys, expected_val: Counter, scope: str) -> None:
    """The shared VAL/TRAIN contract over the per-rank row keys `keys(batches)`."""
    val = {r: keys(out["val"]["batches"]) for r, out in res.items()}
    assert Counter(k for ks in val.values() for k in ks) == expected_val, (
        "every VAL row must be yielded exactly once across ranks (a rank sharding as "
        "rank 0 of 1 yields the whole corpus)"
    )
    assert set(val[0]) & set(val[1]) == set(), "VAL rows overlap across ranks"
    for r, out in res.items():
        assert len(out["val"]["batches"]) == out["val"]["len"], (
            f"rank {r}: VAL yielded {len(out['val']['batches'])} batches, len(loader) says "
            f"{out['val']['len']}"
        )

    train = {r: keys(out["train"]["batches"]) for r, out in res.items()}
    seen = Counter(k for ks in train.values() for k in ks)
    assert seen, "TRAIN yielded no rows on any rank"
    assert max(seen.values()) == 1, "a TRAIN row was yielded more than once across ranks"
    for r, out in res.items():
        sizes = {len(b) for b in out["train"]["batches"]}
        assert sizes == {BATCH}, f"rank {r}: TRAIN batch sizes {sorted(sizes)}, expected {BATCH}"
    counts = {r: len(out["train"]["batches"]) for r, out in res.items()}
    assert counts[0] == counts[1], f"TRAIN batch counts differ across ranks: {counts}"
    for r, out in res.items():
        assert counts[r] == out["train"]["len"], (
            f"rank {r}: TRAIN yielded {counts[r]} batches, len(loader) says {out['train']['len']}"
        )
    if scope == "per_rank":
        assert set(train[0]) & set(train[1]) == set(), "per_rank TRAIN rows overlap across ranks"


def _assert_ownership(
    train_groups: dict[int, set], val_groups: dict[int, set], val_sizes: Counter, scope: str
) -> None:
    """owner TRAIN: at most R-1 groups on more than one rank. VAL is per_rank under either
    configured scope, so every rank sees every group holding >= R VAL rows.
    """
    for r, groups in val_groups.items():
        missing = {g for g, n in val_sizes.items() if n >= WORLD} - groups
        assert not missing, f"rank {r}: VAL is not per_rank, groups {sorted(missing)} absent"
    if scope == "owner":
        ranks_of = Counter(g for groups in train_groups.values() for g in groups)
        shared = {g for g, n in ranks_of.items() if n > 1}
        assert len(shared) <= WORLD - 1, (
            f"owner TRAIN: groups {sorted(shared)} appear on more than one rank "
            f"(at most {WORLD - 1} may)"
        )
        assert all(train_groups.values()), f"owner TRAIN: a rank owns nothing {train_groups}"


@pytest.mark.parametrize("scope", ["per_rank", "owner"])
def test_workers_shard_by_real_rank(tmp_path, scope: str) -> None:
    """Each rank's forkserver workers shard as that rank of the real 2-rank group."""
    res = _run_ranks(tmp_path, scope=scope)
    _assert_shards(res, _uids, STUB_CORPUS, scope)
    train_groups = {
        r: {group_of(u) for u in _uids(out["train"]["batches"])} for r, out in res.items()
    }
    val_groups = {r: {group_of(u) for u in _uids(out["val"]["batches"])} for r, out in res.items()}
    _assert_ownership(train_groups, val_groups, Counter(dict(enumerate(SIZES))), scope)


def test_epoch_reshuffles_in_persistent_workers(tmp_path) -> None:
    """Iterating the SAME persistent loader again reshuffles inside the running workers."""
    res = _run_ranks(tmp_path, scope="per_rank", shuffle=True, epochs=2)
    for r, out in res.items():
        e0, e1 = out["epochs"]["0"], out["epochs"]["1"]
        assert e0 != e1, f"rank {r}: epoch 1 streamed epoch 0's exact batch order"
        assert len(e0) == len(e1) == out["train"]["len"], (
            f"rank {r}: epoch batch counts {len(e0)}, {len(e1)} vs len(loader) "
            f"{out['train']['len']}"
        )
        # drop_last truncates a per-epoch reshuffled order, so WHICH rows drop can
        # change: check membership against this rank's full row set (its per_rank,
        # untruncated VAL stream), not a cross-epoch multiset.
        own = set(_uids(out["val"]["batches"]))
        for epoch, batches in (("0", e0), ("1", e1)):
            assert {len(b) for b in batches} == {BATCH}, f"rank {r} epoch {epoch}: batch sizes"
            uids = Counter(_uids(batches))
            assert max(uids.values()) == 1, f"rank {r} epoch {epoch}: a uid repeated"
            assert set(uids) <= own, f"rank {r} epoch {epoch}: rows outside this rank's shard"


@pytest.mark.parametrize("scope", ["per_rank", "owner"])
def test_ftag1lite_reader_streams_disjoint_complete_shards(tmp_path, corpus, scope: str) -> None:
    """The synthetic FTAG1LITE corpus (container = parent dir) through both scopes."""
    rows = {stage: corpus.rows_for(stage) for stage in ("train", "val")}
    container_of = {}
    for stage_rows in rows.values():
        for row in stage_rows:
            container_of[row.identity] = corpus.files[row.file].parent.name
    assert set(container_of.values()) == set(CONTAINERS)
    expected = {stage: Counter(row.identity for row in rows[stage]) for stage in rows}
    for stage, ids in expected.items():
        assert ids and max(ids.values()) == 1, f"{stage}: corpus identities are not unique"

    res = _run_ranks(tmp_path, scope=scope, corpus_root=str(corpus.root))
    _assert_shards(res, _identities, expected["val"], scope)
    train = {r: _identities(out["train"]["batches"]) for r, out in res.items()}
    outside = {i for ids in train.values() for i in ids} - set(expected["train"])
    assert not outside, f"TRAIN yielded rows outside the train split: {sorted(outside)[:5]}"

    train_groups = {r: {container_of[i] for i in ids} for r, ids in train.items()}
    val_groups = {
        r: {container_of[i] for i in _identities(out["val"]["batches"])} for r, out in res.items()
    }
    val_sizes = Counter(container_of[i] for i in expected["val"])
    _assert_ownership(train_groups, val_groups, val_sizes, scope)
