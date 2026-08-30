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
set one.

A few graph artifacts land in the same directory, describing the plan that was
executed rather than the predictions: `plan_test.txt` (the executed TEST plan,
with the sink's demand table), `resolved_io.yaml` (every module's resolved
requires/produces), and `graph_test.*` / `graph_test_dataset.*` renders. They
are diagnostics — nothing downstream reads them.

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

    The generated adapter enforces `world_size == 1` for every sink and raises
    a clear `ConfigError` otherwise. Multi-device test writing is out of scope,
    so a sink never needs a rank-zero guard of its own.

## Where column names come from

Every eval column name is built from **one** declaration, so the eval file and
the ONNX outputs can never drift apart.

A task's `get_output()` returns `OutputField`s. Each field declares a bare
logical **suffix** — `pb`, `pc`, `pu`, `VertexIndex`, `HadronIndex` — and
nothing else about naming. The prefix is added by whoever is writing:

| Destination | Column / output name | Prefix source |
|---|---|---|
| eval H5 (`salt test`) | `{run_name}_{suffix}` | the `name:` field of your config |
| ONNX (`salt export`) | `{model_name}_{suffix}` | the export sink's `model_name` init arg, else the sanitised run name |

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

`outputs:` is a top-level, deep-mergeable dict section: everything that leaves
the model is declared here, and nowhere else. It holds two kinds of entry.

**Writers** describe the columns. They are **ordered** — dict order is column
order within each group. The standard v1-compatible layout:

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

**Sinks** describe the destination — a file, the ONNX output tuple. You rarely
declare one (the usual sinks are wired by the command, see [How sinks get
attached](#how-sinks-get-attached)), but when you do it goes in this same
section:

```yaml
outputs:
  run_tasks: {class_path: salt.outputs.RunTaskOutput, init_args: {tasks: [...]}}
  jsonl:
    class_path: salt.outputs.JSONLOutputSink
    init_args:
      modes: [test]
```

A sink is **excluded from the ordering**: entries are partitioned by type, so
only the writers form the ordered column list and where a sink sits in the
section makes no difference. Put them wherever reads best — the shipped
configs put them last.

Setting an entry to `null` deletes it, which is how a stacked config drops a
writer or a sink it inherited.

Every model config declares its own section; `base.yaml` ships none. A
`salt test` config with no `outputs:` section is refused.

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
   ONNX plans, so the planner prunes it and no dead-predictions error fires.
   This is the right lever for a training-only auxiliary task.
3. **`--model.modules.<task>=null`** — delete the task entirely, weights and
   all.

`expose:` is an **`init_args` key on the task module**, not a top-level key:

```yaml
model:
  modules:
    track_origin:
      class_path: salt.model.modules.tasks.ClassificationTaskModule
      init_args:
        stream: tracks
        label: ftagTruthOriginLabel
        expose: [fit, val]     # <- trained, but never written or exported
```

Stacked as an overlay config you only need the keys you are changing:

```yaml
# no_track_outputs.yaml — stack as a second --config
model:
  modules:
    track_origin:
      init_args: {expose: [fit, val]}
```

To *un*-defer a task an earlier config deferred, set `expose: null` (the
default, meaning all modes).

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
naming both (resolve it with the export sink's `rename:`). Inspect the whole manifest,
without a checkpoint, with:

```bash
salt export --manifest -c path/to/config.yaml
```

The same table is appended to `plan_onnx.txt` at export time. See
[Export to ONNX](deployment/export.md) for the full export workflow and the Athena
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

A sink is a terminal graph node with a lifecycle. There are two base classes:

- **`salt.outputs.RuntimeSink`** — a node that consumes batches as they are
  produced. This is what you subclass to write a file.
- **`salt.outputs.Node`** — its declare-only parent, for a node with no
  run-time work at all. `OnnxExportSink` is the one case in the tree: export
  never runs a test loop, so naming the leaves at compile time is its whole
  job.

A sink is **not** a Lightning callback and never sees a `Trainer`. The
Lightning test loop reaches a `RuntimeSink` through an adapter salt generates
for it, and `salt inference` — which runs no Lightning at all — calls the same
methods directly. So you never write a Lightning hook, and no driver is
privileged.

### The contract

| You provide | Called | Does |
|---|---|---|
| `name` | — | the graph-node instance name (the `outputs:` dict key overrides it) |
| `allowed_modes` | — | the modes a config may select for this class; see [Modes](#modes-when-a-sink-runs) |
| `declare_io(mode)` | at compile | declares which `outputs.*` leaves the sink needs; produces nothing |
| `writer_demand(model_modules, reader)` | at compile | the same keys as a `{key: who-wants-it}` map, for error messages |
| `open_schema(ctx)` | once, before the first batch | open the file, resolve the schema |
| `consume(bundle)` | once per test batch | read each required leaf with `bundle.get(key)` and append |
| `flush()` | once, after the last batch | close and report |
| `close_if_open()` | on every exit path, including a crash | idempotent cleanup |

`open_schema` receives a **`SinkContext`**, not a trainer: a small frozen
record of the run facts a sink actually reads, which every driver can build
honestly.

| `SinkContext` field | Is |
|---|---|
| `run_name` | the run name, the prefix on output column names |
| `ckpt_path` | the checkpoint being evaluated (output paths template on it) |
| `datamodule` | the datamodule; `ctx.reader` is the shortcut to `test_dset.reader` |
| `num_test_batches` | per-dataloader batch counts, or `None` for "the whole dataset" |
| `world_size` | devices taking part; always 1 (multi-device TEST is out of scope) |

`declare_io` is the important one. It is the *single* declaration that drives
both the planner (demand-gating keeps exactly the producers you require alive)
and `writer_demand`. Always **generate** `writer_demand` from `declare_io`
rather than writing the key list twice — that is what the one-line
`flatten_spec(self.declare_io(Mode.TEST).requires)` in the examples below is
doing. `flatten_spec` turns the nested `{"outputs": {"jets": {...}}}` shape
into flat dotted keys; `unflatten_spec` is its inverse, used to build an `IO`
from a flat dict. The strings `writer_demand` maps to are only ever shown to a
human in a planner error, so any description naming your sink will do.

`consume(bundle)` receives the executed `Bundle`. `bundle.get(key)` returns a
**torch tensor** — always torch, never numpy — so converting is your job
(`.detach().cpu().numpy()`). `bundle.get("meta.rows")` is a length-2 int64
tensor, the `[start, stop)` row range of the batch.

The `TensorSpec` **kind** you require must match what the producer declares, or
the planner's kind-unify raises. For a sink there are only four you will ever
need:

| Key you require | `TensorSpec` |
|---|---|
| `outputs.<stream>.<task>.<col>` — a prediction column | `TensorSpec(shape=None, dtype=None, kind="data")` |
| `meta.rows` — the batch's `[start, stop)` row range | `TensorSpec(shape=None, dtype="int64", kind="meta")` |
| `masks.<stream>` — the bool pad mask | `TensorSpec(shape=None, dtype="bool", kind="pad_mask")` |
| `labels.<stream>.<name>` — a truth label | `TensorSpec(shape=None, dtype=None, kind="label")` |

Leave `shape` and `dtype` as `None` on the prediction columns: the sink
consumes whatever the producer emits and casts at write time, and constraining
the dtype here would conflict with the producer's own declaration.

Two predicates decide how the machinery treats your sink:

- `is_sink()` — `True` on the base; leave it alone. It tells the planner this
  node is terminal (it consumes and produces nothing), so the executor skips it
  in the per-batch forward loop and calls your `consume` instead of a `forward`,
  and the graph render gives it its own sink card. You would only override it to
  `False` if your class were really a producer, in which case it should subclass
  `OutputSectionWriter` rather than `RuntimeSink`.
- `is_test_sink()` — whether this is *the* TEST persistence sink. **Exactly one
  wired sink may hold that role** (two is a hard error at wiring); it anchors
  the TEST boundary demand. The base answers `True` whenever
  `declare_io(Mode.TEST).requires` is non-empty, which is correct for a primary
  sink (`H5OutputSink`) and for an ONNX-only node (`OnnxExportSink`, whose TEST
  requires are empty). An **auxiliary** sink that runs alongside the H5 sink
  must override it to return `False`.

### Modes: when a sink runs

Each sink class declares `allowed_modes`, the modes it may be configured for —
`{test}` for a `RuntimeSink`, `{onnx}` for `OnnxExportSink`. A config narrows
that with a `modes:` list, which must be a subset (anything else is a
`ConfigError` naming both sets).

`modes:` is load-bearing, not decorative: outside its effective modes a sink
declares empty IO, so the planner prunes it **and every producer that was kept
alive only for it**. Narrowing therefore changes the compiled plan — that is
the point of it.

Omitting `modes:` means `allowed_modes`, which for every shipped sink is
exactly what its `declare_io` already gated on, so omitting it changes nothing.

The vocabulary is the planner's own modes — `fit`, `val`, `test`, `onnx` —
plus `export` as a second spelling of `onnx`, because that is what a section
*writer*'s `modes:` list calls it and the two live in one section.

!!! note "`salt inference` is not a mode"

    `salt inference` drives a sink's lifecycle directly, with no planner mode
    and no registration, so it is outside what `modes:` selects. There is no
    mode name for it, deliberately.

An **auxiliary** sink (one whose `is_test_sink()` is `False`) must state
`modes:` explicitly when declared in the section. It rides alongside the
persistence sink rather than replacing it, so it says when it runs rather than
inheriting a default that would read as if it were the primary sink.

### Consumes: what a sink takes

`modes:` picks WHEN a sink runs; `consumes:` picks WHAT it takes. A sink
collects its outputs from every bound producer declaring `manifest_fields(mode)`
— the section's writers first, then the model's graph modules. `consumes:` is a
list of fnmatch patterns over the dotted leaf key that narrows that collection:

```yaml
outputs:
  jsonl:
    class_path: salt.outputs.JSONLOutputSink
    init_args:
      modes: [test]
      consumes: [outputs.jets.*]
```

Omitting it (the default) takes everything declared. A pattern matching none of
the available leaves is a `ConfigError` naming the pattern and listing the keys,
so a typo fails loudly; an empty list is a `ConfigError` too (to switch a sink
off, delete its entry with `<key>: null`). Narrowing composes with demand-gating
rather than replacing it — a leaf that no sink consumes is still the existing
dead-prediction hard error.

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

Declare a sink explicitly only when it carries a manifest the command cannot
guess (`salt/configs/MaskFormer.yaml` is the shipped example), or when it is
your own. Either way it goes in the `outputs:` section, and the command leaves
it alone — the implicit wiring only injects a sink type that is not already
present:

```yaml
outputs:
  run_tasks: {class_path: salt.outputs.RunTaskOutput, init_args: {tasks: [...]}}
  my_sink:
    class_path: my_module.MySink
    init_args:
      modes: [test]
```

`class_path` is resolved with a normal import, so the module must be on
`sys.path` — run from the directory holding it, or set `PYTHONPATH` (with
apptainer: `--env PYTHONPATH=/path/to/dir`).

### Worked example: `JSONLOutputSink`

Salt ships a small, tested example of exactly this:
[`salt/outputs/sinks/jsonl_sink.py`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/outputs/sinks/jsonl_sink.py).
`JSONLOutputSink` writes the eval columns as newline-delimited JSON — one JSON
object per jet — beside the eval H5. It is deliberately minimal, but it is a
real sink that exercises the whole lifecycle, and it is the file to copy when
you write your own.

```yaml
outputs:
  jsonl:
    class_path: salt.outputs.JSONLOutputSink
    init_args:
      modes: [test]                       # required: it is an auxiliary sink
      columns: [GN2_pb, GN2_pc, GN2_pu]   # omit for every column the section mints
```

Stack that on a config that already has an `outputs:` section and the entry
deep-merges into it, beside the writers.

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
  config. `bind_output_section()` is called on every wired sink that exposes
  it, before any `declare_io` resolution; the sink then walks the
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
- **NaN handling is explicit.** Padded and invalid positions are `NaN`, and
  your format may not have a `NaN` literal. In JSON's case `json.dumps` would
  emit a non-standard `NaN` token that most parsers reject, so the sink maps
  every non-finite float to `null` and then calls `json.dumps(...,
  allow_nan=False)` so anything that slipped through raises instead of silently
  writing a corrupt file. Whatever your format is, decide this explicitly.
- **Cleanup is idempotent.** `close_if_open()` runs on every exit path and is
  safe to call twice. That takes *two* Lightning hooks, not one: an exception
  raised inside the test loop unwinds past `teardown`, and `on_exception` is
  the only hook Lightning still calls there. The generated adapter wires both,
  so a crashed `salt test` still releases the handle — but only because
  `close_if_open` is genuinely idempotent, which is your side of the contract.

### Reading the section schema

Most sinks want the same columns as the eval H5. Three pieces give you that:

- **`bind_output_section(section)`** — implement this and the machinery calls it
  on your sink, before any `declare_io` resolution, with the ordered dict of
  `outputs:` section writers. Store it.
- **`writer.is_run_task_output()`** — True on the section writers that carry a
  column manifest (`RunTaskOutput` and anything else declaring itself one).
  Skip the writers where it is absent or False.
- **`writer.manifest_fields(Mode.TEST)`** — the value-free schema: a list of
  `(leaf_key, OutputField)` pairs in column order. `leaf_key` is the
  `outputs.<stream>.<task>.<col>` key you pass to `bundle.get()`; the
  `OutputField` carries `h5_name` (the bare suffix, `None` for an ONNX-only
  field), `dtype`, `axis` and `prefix` (False for label columns, which are not
  run-name prefixed).

The flat column name is then `f"{run_name}_{field.h5_name}"` when
`field.prefix` else `field.h5_name`, with `run_name` read from `ctx.run_name`
at `open_schema`.

### A second worked example: a CSV sink

Putting those together — a complete sink writing one CSV row per jet, with a
header derived from the section:

```python
# csv_sink.py
import csv
from pathlib import Path

import numpy as np
from salt.graph.spec import IO, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.outputs import RuntimeSink


class CSVOutputSink(RuntimeSink):
    name = "csv_output"

    def __init__(self, output: str = "{ckpt_dir}/{ckpt_stem}__test_{sample}.csv", modes=None):
        super().__init__(modes=modes)
        self.output = output
        self._section = None
        self._fields = []          # [(leaf_key, OutputField)]
        self._handle = None
        self._writer = None
        self._run_name = "salt"

    # -- schema, from the bound outputs: section ---------------------------

    def bind_output_section(self, section):
        self._section = section
        self._fields = [
            pair
            for w in section.values()
            if callable(getattr(w, "is_run_task_output", None)) and w.is_run_task_output()
            for pair in w.manifest_fields(Mode.TEST)
            if pair[1].h5_name is not None
        ]

    def _column_name(self, field):
        return f"{self._run_name}_{field.h5_name}" if field.prefix else field.h5_name

    # -- graph node --------------------------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {key: TensorSpec(shape=None, dtype=None, kind="data") for key, _ in self._fields}
        req["meta.rows"] = TensorSpec(shape=None, dtype="int64", kind="meta")
        return IO(requires=unflatten_spec(req), produces={})

    def is_test_sink(self) -> bool:
        return False  # auxiliary: H5OutputSink stays the demand anchor

    def writer_demand(self, model_modules, reader) -> dict[str, str]:
        keys = flatten_spec(self.declare_io(Mode.TEST).requires)
        return {key: f"sink 'CSVOutputSink' demanding {key}" for key in keys}

    # -- lifecycle ---------------------------------------------------------

    def open_schema(self, ctx) -> None:
        self._run_name = ctx.run_name
        ckpt = Path(ctx.ckpt_path)
        stem = Path(getattr(ctx.reader, "filename", None) or ckpt.stem).stem
        path = Path(self.output.format(
            ckpt_dir=str(ckpt.parent),
            ckpt_stem=ckpt.stem,
            sample=split[3] if len(split := stem.split("_")) == 4 else stem,
        ))
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow([self._column_name(f) for _, f in self._fields])

    def consume(self, bundle) -> None:
        rows = bundle.get("meta.rows")
        n = int(rows[1]) - int(rows[0])
        # one flat scalar column per field; a per-token field is averaged over
        # its valid positions so every column stays one CSV cell.
        cols = []
        for key, _field in self._fields:
            v = bundle.get(key).detach().cpu().numpy()
            cols.append(v if v.ndim == 1 else v.reshape(n, -1).mean(axis=1))
        for row in range(n):
            self._writer.writerow(["" if not np.isfinite(c[row]) else c[row] for c in cols])

    def flush(self) -> None:
        self.close_if_open()

    def close_if_open(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
```

```yaml
# csv.yaml — stack as a second --config on salt test (pass --ckpt_path too)
outputs:
  csv:
    class_path: csv_sink.CSVOutputSink
    init_args:
      modes: [test]
```

Note the two shape decisions this example has to make and states explicitly:
a per-token leaf is `[B, L]` / `[B, L, C]` and CSV has no nesting, so it is
reduced to one cell; and a non-finite float becomes the empty field rather than
the string `nan`. Your format will face the same two questions.

### A minimal sink from scratch

If you do not need the section schema at all — say you just want a per-batch
summary — the whole thing is short:

```python
# row_count_sink.py
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.logging import console
from salt.outputs import RuntimeSink


class RowCountSink(RuntimeSink):
    name = "row_count"

    def declare_io(self, mode: Mode) -> IO:
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"meta.rows": TensorSpec(shape=None, dtype="int64", kind="meta")}
        return IO(requires=unflatten_spec(req), produces={})

    def is_test_sink(self) -> bool:
        return False  # auxiliary: the H5 sink stays the demand anchor

    def open_schema(self, ctx):
        self.rows = 0

    def consume(self, bundle):
        start, stop = bundle.get("meta.rows")
        self.rows += int(stop) - int(start)

    def flush(self):
        console(f"saw {self.rows} rows")
```

Notes for sink authors:

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
| `salt.outputs.RuntimeSink` | base class for a sink with a lifecycle — the extension point |
| `salt.outputs.Node` | its declare-only parent, for a node with no run-time work |
| `salt.outputs.SinkContext` | the run facts `open_schema` receives |
| `salt.outputs.H5OutputSink` | the eval-H5 sink (implicit on `salt test`) |
| `salt.outputs.OnnxExportSink` | the ONNX tuple sink (implicit on `salt export`) |
| `salt.outputs.JSONLOutputSink` | the worked example: newline-delimited JSON |
| `salt.outputs.OutputSectionWriter` | base class for an `outputs:` section writer |
| `salt.outputs.RunTaskOutput` | section writer running tasks' `get_output()` |
| `salt.outputs.InputCopyWriter` | section writer for source-file input copies |
| `salt.outputs.PadMaskWriter` | section writer for the bool pad-mask column |
| `salt.outputs.OutputField` | one declared output column (suffix, dtype, axis) |
| `salt.outputs.OutputColumn` | one `outputs.*` leaf's H5 column schema |
