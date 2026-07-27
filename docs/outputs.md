# Outputs

This page is about everything salt writes when a model runs: the evaluation
HDF5 file, the ONNX export tuple, and how to add a format of your own.

If you just want to know *"what are the columns in my eval file and why are
they called that"*, read [What `salt test` writes](#what-salt-test-writes) and
[Where column names come from](#where-column-names-come-from). If you want to
change *which* outputs are written, read [Choosing what gets
written](#choosing-what-gets-written). If you want a new file format, jump to
[Writing your own sink](#writing-your-own-sink).

For the design rationale behind any of this (why sinks are graph nodes, how
demand-gating works) see [Architecture](architecture.md); this page is the
user-facing view.

## The output model in one picture

Salt's model is a graph. Outputs are the last two layers of it:

```
preds.*            raw predictions, loss-space          (task modules)
   |
   v
outputs.*          serialisation-ready leaves           (the `outputs:` section)
   |
   v
[ sink ]           bytes on disk                        (H5 / ONNX / yours)
```

1. Each **task** produces a raw `preds.<stream>.<task>` tensor. That is what
   the loss sees. It is *not* what gets written — a classification head's raw
   output is logits, not probabilities.
2. The top-level **`outputs:` section** says *what to serialise*. Its writers
   call each task's `get_output()`, which applies the eval conversion (softmax,
   de-scaling, argmax, union-find) using traceable torch ops, and write one
   `outputs.<stream>.<task>.<column>` leaf per column.
3. A **sink** is the terminal node that turns those leaves into a file. You
   never wire the usual sinks yourself: the command picks them (see below).

Two consequences worth internalising:

- **The section says WHAT, the command says WHERE.** The same `outputs:`
  section drives both the eval H5 (`salt test`) and the ONNX tuple
  (`salt export`) — one manifest, two destinations.
- **Nothing is written that nobody asked for.** The planner keeps alive
  exactly the producers some sink demands. A prediction that no sink consumes
  is a hard error, not a silent drop (see [Dead
  predictions](#dead-predictions)).

## What `salt test` writes

`salt test` writes **one HDF5 file**, next to the checkpoint it evaluated:

```
{checkpoint_dir}/{checkpoint_stem}__test_{sample}.h5
```

For example, evaluating `logs/GN2_20250101/ckpts/epoch=009-val_loss=0.64.ckpt`
on `pp_output_test_ttbar.h5` gives:

```
logs/GN2_20250101/ckpts/epoch=009-val_loss=0.64__test_ttbar.h5
```

`{sample}` comes from the test file's stem (the fourth `_`-separated field if
there are exactly four, else the whole stem), plus `--data.test_suff` if you
set one. The graph artifacts for the test plan are written into the same
directory.

The file's **groups** mirror your reader's streams — a `jets` group, a `tracks`
group, and so on, each a structured array with one row per jet (or event). Each
group's columns are, in this order:

| Block | Comes from | Example columns |
|---|---|---|
| input copies | `InputCopyWriter` in the `outputs:` section | `pt`, `eta`, `HadronConeExclTruthLabelID` |
| task outputs | `RunTaskOutput` → each task's `get_output()` | `GN2_pb`, `GN2_pc`, `GN2_pu` |
| target labels | the same tasks, `write_targets` on (the default) | `target_jets_classification` |
| pad mask | `PadMaskWriter` | `mask` (bool, `True` = padded) |
| object groups | the H5 sink's `object_groups` (MaskFormer) | the `objects` group, `HadronIndex` |

Per-token columns (anything on a sequence stream like `tracks`) are zero-padded
back out to the *source file's* sequence length, so the eval file lines up
row-for-row and token-for-token with the input file. A position that the model
truncated away reads `mask = False`.

!!! info "Only one device is supported for the test loop"

    Every sink enforces `trainer.world_size == 1` and raises a clear
    `ConfigError` otherwise. Multi-device test writing is out of scope, so a
    sink never needs a rank-zero guard.

## Where column names come from

Every eval column name is built from **one** declaration, so the eval file and
the ONNX outputs can never drift apart.

A task's `get_output()` returns `OutputField`s. Each field declares a bare
logical **suffix** — `pb`, `pc`, `pu`, `VertexIndex`, `HadronIndex` — and
nothing else about naming. The prefix is added by whoever is writing:

| Destination | Column / output name | Prefix source |
|---|---|---|
| eval H5 (`salt test`) | `{run_name}_{suffix}` | the `name:` field of your config |
| ONNX (`salt export`) | `{model_name}_{suffix}` | `export.model_name`, else the sanitised run name |

So a config with `name: GN2` and a classification task over classes `b, c, u`
gives eval columns `GN2_pb`, `GN2_pc`, `GN2_pu`. Export the same model as
`--name GN2v01` and the ONNX outputs are `GN2v01_pb`, `GN2v01_pc`, `GN2v01_pu`.
**Reordering `class_names` moves both together** — this is the single-source
naming rule, and it is why the v1 class of bug where eval columns and Athena
outputs disagreed is unrepresentable here.

Some columns are deliberately *not* prefixed, because they do not belong to a
model:

- **Target-label columns.** Every task with `write_targets: true` (the default)
  also writes the labels it was trained against, in TEST mode only:
  `target_{task}` for classification and vertexing, `target_{task}_{target}`
  for each regression target. These are the truth values, so a run-name prefix
  would be misleading. Padded / invalid positions read `-1` (integer labels) or
  `NaN` (regression targets). Export and inference are label-free — no target
  column is ever declared, demanded, or written there.
- **Input copies.** Copied straight from the source file with the source name
  and the source dtype.
- **`mask`.** The bool pad mask, one per sequence stream.

Two leaves minting the same flat column name in the same group is a hard error
naming both — a collision fails at run setup, not silently.

## Declaring outputs: the `outputs:` section

`outputs:` is a top-level, deep-mergeable section (like `callbacks:`) holding
an **ordered dict** of section writers. Dict order is column order within each
group. The standard v1-compatible layout:

```yaml
outputs:
  inputs_copy:
    class_path: salt.outputs.InputCopyWriter
    init_args:
      streams: [jets, tracks]
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args:
      tasks: [jets_classification, track_origin, track_vertexing]
  pad_mask:
    class_path: salt.outputs.PadMaskWriter
    init_args:
      streams: [tracks]
```

| Writer | Writes | Key arguments |
|---|---|---|
| `RunTaskOutput` | each named task's converted output columns (+ its target labels) | `tasks:` (ordered), `modes:` |
| `InputCopyWriter` | source-file variables re-read by row | `streams:` (`null` = every stream with a task), `variables:` to narrow |
| `PadMaskWriter` | a bool `mask` column per stream | `streams:` |

Every model config declares its own section; `base2.yaml` ships none. A
`salt test` config with no `outputs:` section is refused, and a config still
carrying the retired top-level `writers:` block fails with a migration error
pointing here.

### Modes: eval, export, or both

Each section writer takes a `modes:` list naming which destinations it
participates in — `test` (the eval H5) and/or `export` (the ONNX tuple).
Omitting `modes:` means both. Split tasks across two writers when they should
go to different places:

```yaml
outputs:
  jets_out:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [jets_classification]}          # test + export
  origin_out:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [track_origin], modes: [test]}  # eval H5 only
```

`InputCopyWriter` and `PadMaskWriter` never mint ONNX outputs (Athena supplies
the inputs, and a pad mask has no Athena consumer), but their `modes:` list
still decides whether they contribute columns to `salt inference`, which writes
the *export* selection to H5.

## Choosing what gets written

### Add a task's outputs

Name the task in a `RunTaskOutput`'s `tasks:` list. Position in the list is
position in the file:

```yaml
outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args:
      tasks: [jets_classification, track_origin]   # <- add here
```

### Remove a task's outputs

There are three levers, in increasing severity:

1. **Drop it from `tasks:`** — but see [Dead predictions](#dead-predictions):
   if the task still runs, you must also gate it out of the eval graph.
2. **`expose: [fit, val]` on the task** — the task stays trained (the loss is a
   FIT/VAL concern anyway) but its `preds.*` port is gated out of the TEST and
   ONNX plans. This is the right lever for a training-only auxiliary task.
3. **`--model.modules.<task>=null`** — delete the task entirely, weights and
   all.

### Turn off the target-label columns

Per task, on the task module:

```yaml
model:
  modules:
    jets_classification:
      init_args:
        write_targets: false
```

### Choose which input variables are copied

```yaml
outputs:
  inputs_copy:
    class_path: salt.outputs.InputCopyWriter
    init_args:
      streams: [jets, tracks]
      variables:
        jets: [pt, eta, HadronConeExclTruthLabelID]
        tracks: [truthOriginLabel]
```

A stream listed in `streams:` but absent from `variables:` copies every source
field. `streams: null` (the default) copies every stream that has a configured
task.

### Dead predictions

If a task produces a `preds.*` leaf that **no** sink consumes, salt refuses to
run:

```
[mode=TEST] key 'preds.tracks.track_origin' is produced but consumed by no sink
```

This is deliberate. Narrowing a `RunTaskOutput`'s `tasks:` list and forgetting
one is a mistake that would otherwise silently cost you columns. Fix it by
either putting the task back in `tasks:`, or gating it out of the eval graph
with `expose: [fit, val]`. The same error is reported statically by
`salt graph validate` and `salt graph deadcode`, so you can catch it without
running anything.

## ONNX export

`salt export` uses the same `outputs:` section, selecting the writers whose
`modes:` include `export`. Each task's `get_output(..., Mode.ONNX, ...)`
returns the Athena representation of the same fields — squeezed per-class
scalars and int8 argmax indices rather than the eval file's probability
columns — and the `OnnxExportSink` names them `{model_name}_{suffix}` and packs
them into the flat output tuple.

The ONNX namespace is flat: two leaves minting the same suffix is a hard error
naming both (resolve it with `export.rename:`). Inspect the whole manifest,
without a checkpoint, with:

```bash
salt export --manifest -c path/to/config.yaml
```

The same table is appended to `plan_onnx.txt` at export time. See
[ONNX Export](export.md) for the full export workflow and the Athena
validation steps.

## Extending: a new column

Two seams, by scope. Both are documented with a worked example in
[Architecture — Add a custom output column](architecture.md#add-a-custom-output-column-design-8):

- **A new column family for an existing task** — implement `get_output()` /
  `get_output_manifest()` on the task. Right when the column is a rendering of
  that task's prediction.
- **A column no task owns** — subclass `salt.outputs.OutputSectionWriter`,
  produce an `outputs.<stream>.<col>` leaf, and expose the manifest surface the
  sinks read. Right for anything computed from inputs/labels rather than
  predictions.

Neither of these needs a new sink — they add columns to the files you already
get.

## Writing your own sink

Write a sink when you want a **different file format**, not different columns.

A sink is a `salt.outputs.OutputSink`: simultaneously a terminal graph node and
a `lightning.Callback`. The Lightning hooks are already implemented on the base
and forward to five named lifecycle methods, so you never write a Lightning
hook yourself.

### The contract

| You provide | Called | Does |
|---|---|---|
| `name` | — | the graph-node instance name (a `callbacks:` dict key overrides it) |
| `declare_io(mode)` | at compile | declares which `outputs.*` leaves the sink needs; produces nothing |
| `open_schema(trainer)` | once, before the first batch | open the file, resolve the schema |
| `consume(bundle)` | once per test batch | read each required leaf with `bundle.get(key)` and append |
| `flush()` | once, after the last batch | close and report |
| `close_if_open()` | on every exit path, including a crash | idempotent cleanup |

`declare_io` is the important one. It is the *single* declaration that drives
both the planner (demand-gating keeps exactly the producers you require alive)
and `writer_demand` (the boundary demand). The two can therefore never
disagree.

Two predicates decide how the machinery treats your sink:

- `is_sink()` — `True` on the base, rarely overridden. Marks the node terminal,
  so the executor skips it in the per-batch forward loop.
- `is_test_sink()` — whether this is *the* TEST persistence sink. **Exactly one
  attached callback holds that role**; it anchors the TEST boundary demand. The
  base answers `True` whenever `declare_io(Mode.TEST).requires` is non-empty,
  which is correct for a primary sink (`H5OutputSink`) and for an ONNX-only
  sink (`OnnxExportSink`, whose TEST requires are empty). An **auxiliary** sink
  that runs alongside the H5 sink must override it to return `False`.

### How sinks get attached

The usual sinks are **implicit**: the command wires them over your section, so
you never name `H5OutputSink` or `OnnxExportSink` in a config.

| Command | Sink wired |
|---|---|
| `salt fit` | none |
| `salt test` | `H5OutputSink`, if any section writer runs in `test` |
| `salt graph` / `salt schema` / `salt export` | `H5OutputSink` and `OnnxExportSink`, so the static tooling sees what a real run would |

Wiring `H5OutputSink` yourself with an explicit `OutputColumn` table is a hard
error — that surface is retired, and the section replaced it.

Your own sink goes in `callbacks:`, and the command leaves it alone (it only
injects a sink type that is not already present):

```yaml
callbacks:
  my_sink:
    class_path: my_module.MyOutputSink
    init_args: {}
```

`class_path` is resolved with a normal import, so the module must be on
`sys.path` — run from the directory holding it, or set `PYTHONPATH` (with
apptainer: `--env PYTHONPATH=/path/to/dir`).

### Worked example: `JSONLOutputSink`

Salt ships a small, tested example of exactly this:
[`salt/outputs/jsonl_sink.py`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/outputs/jsonl_sink.py).
`JSONLOutputSink` writes the eval columns as newline-delimited JSON — one JSON
object per jet — beside the eval H5. It is deliberately minimal, but it is a
real sink that exercises the whole lifecycle, and it is the file to copy when
you write your own.

```yaml
callbacks:
  jsonl:
    class_path: salt.outputs.JSONLOutputSink
    init_args:
      columns: [GN2_pb, GN2_pc, GN2_pu]   # omit for every column the section mints
```

Run `salt test` with that stacked on your config (pass `--ckpt_path`
explicitly when you stack a second `--config`) and you get, next to the eval
H5:

```json
{"GN2_pb": 0.9231, "GN2_pc": 0.0512, "GN2_pu": 0.0257}
{"GN2_pb": 0.0143, "GN2_pc": 0.1120, "GN2_pu": 0.8737}
```

What it demonstrates, point by point — these are the things a real sink has to
get right, and the reasons they are not optional:

- **It derives its schema from the bound `outputs:` section**, not from its own
  config. `bind_output_section()` is called on every attached callback that
  exposes it, before any `declare_io` resolution; the sink then walks the
  section's `RunTaskOutput.manifest_fields(Mode.TEST)` for exactly the names,
  dtypes and order the H5 sink uses. Deriving from the section is what
  guarantees your file and the eval H5 agree.
- **It is an auxiliary sink** — `is_test_sink()` returns `False`, so
  `H5OutputSink` stays the demand anchor and you get *both* files from one
  `salt test`. Because its columns come from the same section, every leaf it
  reads is one the primary sink already demanded, so it never widens the plan.
  A sink meant to *replace* the H5 file rather than accompany it would leave
  `is_test_sink()` alone.
- **Deterministic output paths.** It renders the same `{ckpt_dir}` /
  `{ckpt_stem}` / `{sample}` template the H5 sink uses, so re-running an
  evaluation overwrites the same file rather than accumulating.
- **Explicit overwrite behaviour.** `overwrite: false` refuses to clobber an
  existing file instead of truncating it.
- **NumPy conversion is the sink's job.** Producers emit torch tensors, always;
  the sink calls `.detach().cpu().numpy()` and converts. Do not expect numpy
  from the graph.
- **NaN handling is explicit.** JSON has no `NaN` literal, and `json.dumps`
  would happily emit a non-standard `NaN` token that most parsers reject.
  Every non-finite float is mapped to `null`, and the dump then runs with
  `allow_nan=False` so a leak is a loud error rather than a corrupt file.
- **Cleanup is idempotent.** `close_if_open()` runs from `teardown` on every
  exit path, including an exception mid-test, and is safe to call twice.

### A minimal sink from scratch

If you do not need the section schema at all — say you just want a per-batch
summary — the whole thing is short:

```python
# row_count_sink.py
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.outputs import OutputSink


class RowCountSink(OutputSink):
    name = "row_count"

    def declare_io(self, mode: Mode) -> IO:
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"meta.rows": TensorSpec(shape=None, dtype="int64", kind="meta")}
        return IO(requires=unflatten_spec(req), produces={})

    def is_test_sink(self) -> bool:
        return False  # auxiliary: the H5 sink stays the demand anchor

    def open_schema(self, trainer):
        self.rows = 0

    def consume(self, bundle):
        start, stop = bundle.get("meta.rows")
        self.rows += int(stop) - int(start)

    def flush(self):
        print(f"saw {self.rows} rows")
```

Notes for sink authors:

- **Require kinds must match the producer.** `outputs.*` leaves take an
  unconstrained `TensorSpec(shape=None, dtype=None, kind="data")`;
  `meta.rows` is `kind="meta"`, `masks.<stream>` is `kind="pad_mask"`,
  `labels.*` is `kind="label"`. A mismatch fails the planner's kind-unify.
- **`meta.rows`** is the `[start, stop)` absolute row range of the batch. Use
  it if you need to align with the source file or detect a broken loop.
- **Per-token leaves are `[B, L]` / `[B, L, C]`** at the model's (possibly
  truncated) sequence length. The H5 sink re-expands them to the source file's
  length because a fixed HDF5 dataset shape demands it; a format without that
  constraint (like JSONL) need not.
- **A single-suffix leaf may be collapsed.** A column with one suffix can
  arrive as `[B]` / `[B, L]` rather than `[B, 1]` / `[B, L, 1]`. Handle both.

## Reference

| Class | Role |
|---|---|
| `salt.outputs.OutputSink` | base class for every sink — the extension point |
| `salt.outputs.H5OutputSink` | the eval-H5 sink (implicit on `salt test`) |
| `salt.outputs.OnnxExportSink` | the ONNX tuple sink (implicit on `salt export`) |
| `salt.outputs.JSONLOutputSink` | the worked example: newline-delimited JSON |
| `salt.outputs.OutputSectionWriter` | base class for an `outputs:` section writer |
| `salt.outputs.RunTaskOutput` | section writer running tasks' `get_output()` |
| `salt.outputs.InputCopyWriter` | section writer for source-file input copies |
| `salt.outputs.PadMaskWriter` | section writer for the bool pad-mask column |
| `salt.outputs.OutputField` | one declared output column (suffix, dtype, axis) |
| `salt.outputs.OutputColumn` | one `outputs.*` leaf's H5 column schema |
