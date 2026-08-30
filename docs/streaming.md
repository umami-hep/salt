# Streaming datasets

Salt ships two dataset styles over the same pipeline. Both compile the same
plan, run the same processors and produce the same batch object; they differ
only in how rows are addressed and how much of the corpus a reader process has
to know about.

| | `SaltDataset` (map-style, default) | `IterableSaltDataset` (streaming) |
|---|---|---|
| addressing | a sampler hands it contiguous row slices | it yields whole batches itself |
| read size | one batch (e.g. 1,000 rows) | one block (default 16,384 rows) |
| per-process index | the whole corpus, row-granular | per-block row counts only |
| sharding | `DistributedSampler` over a shared index | row intervals, resolved per (rank, worker) |
| order | weakly shuffled batches | shuffled blocks + within-batch permutation |

Use the map-style path by default. Reach for streaming when either of the two
things it fixes actually bites you:

1. **Read size.** A batch-sized window is a small read. On a ROOT corpus with
   many branches, the same bytes read in larger contiguous ranges are
   substantially faster, because one call's basket ranges coalesce.
2. **Scale.** The map-style index is proportional to corpus ROWS and exists once
   per reader process. Multiply by (ranks x dataloader workers) and there is a
   point past which it simply cannot be held. The streaming index is
   proportional to BLOCKS.

## Turning it on

```yaml
data:
  iterable: true
  block_rows: 16384
  max_live_streams: 2
  interleave_block: 1
  shuffle_stream: true
```

Nothing else changes: no sampler to configure, no per-rank settings. A shipped
overlay is at `salt/configs/readers/ftag1lite_streaming.yaml`.

## How sharding works

Each process resolves

```
n_shards = world_size * num_workers
shard_id = rank * num_workers + worker_id
```

and takes, **from every sample independently**, the row interval

```
[ floor(shard_id * N / n_shards),  floor((shard_id + 1) * N / n_shards) )
```

Three consequences worth knowing:

- **Coverage and disjointness are exact.** The intervals tile `[0, N)`, so every
  row is read by exactly one shard, once per epoch.
- **Sub-file splitting is the normal case.** A shard whose interval lands
  mid-file gets a shorter contiguous range. You do not need more files than
  shards, which matters: a few hundred GPUs times a few workers is thousands of
  shards against a corpus that may have a few hundred files.
- **Balance is within one row per sample per shard**, so with `drop_last` every
  rank runs the same number of steps without any collective — each computes
  `min(shard_rows) // batch_size` locally. No rank waits on a straggler.

## Blocks and buffers

A **block** is a reader's natural sequential read unit. `UprootReader` reports
one per file; any reader that overrides nothing reports one block for itself,
which is correct but coarse. Blocks only decide where reads may be split — never
which rows exist.

`block_rows` caps how many rows one `read` call covers, and it is the buffer
budget. Peak resident rows per worker is

```
max_live_streams * block_rows
```

plus the batch being assembled. Bigger blocks read faster up to a point and then
stop helping while continuing to cost memory, so measure before raising it. On
the FTAG1LITE corpus the curve is flat from a whole file down to ~3,400 tree
entries and falls off below; the 16,384-row default sits on that knee.

## Mixing samples

With a `MultiSampleReader`, each shard interleaves its samples using the same
proportional (largest-remainder) apportionment the map-style path uses — applied
to that shard's own row counts rather than to a global index. Every batch's
per-sample counts stay within `interleave_block` rows of the proportional share,
and `max_live_streams` bounds how many samples are open at once.

## Determinism

`(seed, epoch, shard_id)` fixes the stream. The same tuple gives a byte-identical
batch sequence; a different epoch gives a different order over the same rows;
different shards never overlap. The datamodule sets `epoch` from
`trainer.current_epoch` when it builds the loader.

Shuffling is multiset-preserving by construction: it permutes block order within
a shard and row order within a batch. It never changes which rows a batch
contains, so the per-sample proportion guarantee above survives it.

## The corpus manifest

Resolving a reader's blocks means resolving its index, which is the expensive
part of startup. With a manifest, shard assignment and epoch length are computed
without opening a single data file.

`manifest:` takes three values.

### `manifest: auto` — let the run build it

```yaml
data:
  iterable: true
  manifest: auto
```

The datamodule resolves a conventional path per stage, validates whatever is
there, and builds it when it is missing or stale. Building happens in Lightning's
`prepare_data()` hook, which runs on **global rank 0 only** and barriers every
other rank before `setup()` — so one process pays for the artifact and the rest
read it, with no lock and no stampede. Writes are atomic (temp file plus
`os.replace`), so a half-written manifest can never be read.

Where it lands:

1. next to the corpus — the common ancestor directory of the reader's resolved
   source files, when that directory is writable;
2. otherwise `$SALT_MANIFEST_CACHE`, or `~/.cache/salt/manifests/`.

Corpus directories on cluster storage are frequently read-only, so the fallback
is routine. The chosen path is logged either way. Filenames are keyed by a digest
of (source list, stage, row cap, reader class, **reader config**), so train and
val get their own manifests and two different corpora never collide in one
directory.

Staleness is decided without opening a data file:

- the artifact's format version;
- a `stat` per recorded file (size and mtime);
- the corpus's file list — the one change no per-file `stat` can see;
- the **reader's config fingerprint**, which is in the filename itself. The
  checks above all describe the files, and none of them sees a reader
  reconfigured over the *same* files. That matters because a changed `cuts:`
  changes row counts and therefore every block boundary, while leaving every
  file byte-identical. Config changes are a miss by construction: they key a
  different filename.
- the served schema's hash, whenever the caller already has a built schema and
  can supply it for nothing. (Computing one means opening files, which is the
  cost the artifact exists to avoid — so this check is a bonus, and the config
  fingerprint is the guarantee.)

A stale manifest is rebuilt, loudly.

The manifest also records the **served schema** (`{stream: {field: dtype}}`).
Plan compilation validates demanded fields against it, and both `schema_group`
and `label_universe` resolve `prepare()` when the reader has none — so without
this a run with a perfectly good manifest still opened one file per stage just
to re-learn field names the manifest already had. With it, a warm `setup()`
opens **no data file at all**. Reading still builds the index when a worker
actually reads: the manifest seeds the schema, not the row table.

!!! note "Format version 2"

    `MANIFEST_VERSION` is 2: the artifact now carries the served schema, and its
    filename key includes the reader's config. **Manifests written by an earlier
    salt are a miss.** Under `manifest: auto` that is invisible — the run
    rebuilds. An explicit `manifest: /path/...` **errors as stale**; rebuild it
    with `python -m salt.data.manifest`.

### `manifest: /path/to/corpus_manifest.json` — build it yourself

Recommended for a large corpus, and the only option that keeps the cost out of
the training job entirely:

```bash
python -m salt.data.manifest \
    --config salt/configs/readers/ftag1lite.yaml \
    --set data.train_file='/path/to/corpus/*/*.pool.root*' \
    --out /path/to/corpus_manifest.json
```

An explicit path must exist and load; it is applied to **every** streaming stage
of the run, so build it from the corpus those stages read, or use `auto` when
train and val are different file sets.

### `manifest:` unset — no artifact

Every reader process resolves its own index. Correct, and the thing that stops
scaling first.

## Writing a reader that streams well

Readers need no changes to work — the default `row_blocks()` reports one block
and everything is correct, just coarse. To let shards read contiguously, report
your storage's real boundaries:

```python
from salt.data.base import Reader, RowBlock


class MyReader(Reader):
    def row_blocks(self) -> list[RowBlock]:
        """Ascending, covering [0, len(self)) exactly once."""
        return [RowBlock(group=0, start=lo, stop=hi) for lo, hi in my_file_ranges()]
```

Override `read_block(block, mode)` only if a block needs something beyond a plain
row-slice `read` — `MultiSampleReader` does, because it maps the block's group to
a sub-reader and injects that sample's label.

## Caveats

- **`__len__` is the batch count for THIS shard**, not the corpus. That is the
  correct thing for a progress bar and for `max_steps`-free schedules, but it is
  not the dataset size.
- **Evaluation ordering.** Validation and test streams keep source order and the
  ragged tail, matching the writers' row-alignment contract. Do not enable
  shuffling for a stage whose outputs are consumed positionally.
- **Very high shard counts eventually hurt.** Per-shard read span is
  `rows / n_shards`; drive that below the knee and each read is small again. The
  fix is more data per shard, not a different setting.
