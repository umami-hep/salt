# How salt works

The modular salt v2 stack: graph-of-modules models, a demand-driven dataset
pipeline, and a jsonargparse YAML surface. Everything here is config-first:
modules declare their inputs/outputs (`declare_io`), plans are compiled
statically, and no data file is touched before the run starts. This page is
an orientation to the whole stack: what the pieces are, how a run flows
end to end, the v1/v2 parity rules, and where each part is documented in
full. For the module-authoring reference itself, start at
[Writing modules](modules/index.md).

## The pieces

A salt model is built from named modules wired together in YAML. Two kinds
participate:

- **Data modules** (`SaltDatasetModule`, e.g. a `Reader` or a `Processor`)
  turn files into batches.
- **Model modules** (`SaltModelModule`, e.g. an embed, an encoder, a task
  head) turn batches into predictions.

Both kinds declare the bundle keys they read and write, before any data is
touched, through `declare_io`. That declaration is what lets the planner
check, order, and prune the whole graph statically, and it is what
[Writing modules](modules/index.md) documents method by method.

Every module reads and writes a shared **bundle**: a flat dict of dotted
keys such as `inputs.jets`, `preds.tracks.track_origin`, and
`outputs.jets.n_tracks_valid`. The prefix says who may write a key (readers
write `raw.*`, processors write `inputs.*`/`labels.*`, model modules write
`preds.*`, and the `outputs:` section writes `outputs.*`). The full
namespace table lives on [Writing modules](modules/index.md).

## How a run flows

1. **Config assembly.** Your config stacks on `base.yaml` and any further
   `--config` files by deep merge, then CLI overrides apply. See
   [Config model](#config-model) below.
2. **Static compile.** salt builds one plan per mode (`fit`, `val`, `test`,
   `onnx`) from every module's declared `IO`, pruning anything nothing
   demands. `salt graph validate`/`plan`/`plot`/`why` inspect this plan
   without touching data. See [Static graph tooling](#static-graph-tooling).
3. **Data.** Readers and processors turn files into batches; only the
   columns the compiled plan actually demands are read.
4. **Model.** Model modules bind their layers to the resolved widths, then
   run forward per batch, producing `preds.*`.
5. **Outputs.** The `outputs:` section turns `preds.*` into named columns
   and hands them to a sink: the eval H5 for `salt test`, the ONNX tuple
   for `salt export`. See [Outputs and the `outputs:`
   section](#outputs-and-the-outputs-section).

## Parity-closure doctrine (v1 vs v2 comparisons)

The v1 stack (the `salt.models`, `salt.data`, `salt.utils`, `salt.callbacks`,
`salt.onnx`, `salt.main` and `salt.modelwrapper` of pin `29c67a1`) is being
deleted from `main`.
All v1↔v2 numerical parity was established and passed at the frozen commit
**`29c67a1`** (`29c67a186f01`), the last commit where both stacks coexist and
the parity suite (`parity_gn2`, the v1-vs-v2 fold/state-dict/ONNX tests) run
green.

**Doctrine:** comparisons against v1 (or a pinned upstream)
are done by `git checkout <pin>`. No frozen comparison artifacts (goldens,
specimens, vendored snapshots) live in the tree. For v1, everything needed
lives at `29c67a1` and passed there; the frozen pin is the single source of
truth for the v1 reference. Git history is the archive.

### Pins of record

| Comparison | Pin | Where |
|---|---|---|
| v1 ↔ v2 numerical parity | `29c67a1` (`29c67a186f01`) | this repo — `git checkout 29c67a1` |
| v2 MaskFormer ↔ upstream | `6570e85` | upstream salt — see checkout below |

The upstream MaskFormer equivalence closes at the upstream pin. To
reproduce the comparison:

```bash
git remote add upstream ssh://gitlab.cern.ch:7999/aft/algorithms/salt.git  # if absent
git fetch upstream
git checkout 6570e85   # the validated upstream MaskFormer reference
```

## Config model

See [How a config is assembled](configuration.md#how-a-config-is-assembled)
for the deep-merge rules, the CLI override spelling, null-deletion, and a
worked example of adding a task from an override file.

## Running a fit or a test

See [Quickstart](cli.md#quickstart) for a copy-paste dummy-file run, and
[salt test](cli.md#salt-test) for the invocation, checkpoint resolution, and
config-stacking rules.

## Outputs and the `outputs:` section

See [Declaring outputs: the `outputs:`
section](outputs.md#declaring-outputs-the-outputs-section) for what a writer
is, what a sink is, how column naming and `modes:` work, and the two seams
for adding a new column or a new file format.

## Inference

See [salt inference](cli.md#salt-inference) for the offline,
Athena-equivalent inference command: the export set, run label-free, on a
file with no checkpointed labels required.

## Static graph tooling

See [salt graph](cli.md#salt-graph) for `validate`, `plan`, `plot`, `why`,
and `deadcode`, all of which read config only and touch no data.

## Run-dir artifacts

See [What a run writes](cli.md#what-a-run-writes) for the per-stage plan
tables, `resolved_io.yaml`, and the graph renders every `salt fit`/`salt
test` writes.

## Metrics callbacks

`salt.callbacks.ConfusionMatrix` reads the step bundle
(`preds.<stream>.<task>` argmax vs `labels.<stream>.<label>`), resolves
stream/label/class names from the named task module, and logs to Comet at
each validation epoch end (values are also stashed on the callback:
`last_matrix`, `last_truth_labels`, `last_pred_labels`). Configure it like
any callback:

```yaml
callbacks:
  confusion_matrix:
    class_path: salt.callbacks.ConfusionMatrix
    init_args:
      task_name: jets_classification
```

Deferral note: callback-declared `requires` are NOT yet FIT/VAL plan sinks.
`ConfusionMatrix` works because tasks publish `preds.*` in all modes and stay
alive via their losses (its VAL labels are demanded by the task itself). A
callback demanding a key no task keeps alive would currently find its
producer demand-pruned; the sink wiring mirrors the TEST writer-demand
mechanism (see `SaltModule._model_sinks`).

## ONNX export

See [salt export](cli.md#salt-export) for the invocation, and [Export to
ONNX](deployment/export.md) for the export mechanics, the sink's init args,
and the Athena validation steps.

## Checkpoints and resume

See [Checkpoints and resume](training.md#checkpoints-and-resume) for
resuming a fit, and for loading a checkpoint saved before the de-core
namespace rename (`salt.core.*` to `salt.*`).

## Where everything else lives

| Topic | Page |
|---|---|
| Every `salt` command and flag | [Command line](cli.md) |
| Config structure and every settable key | [Configuration](configuration.md) |
| Running training, speeding it up, resuming it | [Training](training.md) |
| Warm-starting and fine-tuning a checkpoint | [Fine-tuning](finetuning.md) |
| What `salt test` produces and how to read it | [Evaluation](evaluation.md) |
| The `outputs:` section, sinks, and adding a column | [Outputs](outputs.md) |
| Writing your own data or model module | [Writing modules](modules/index.md) |
| Exporting to ONNX and deploying a model | [Export to ONNX](deployment/export.md) |
| Streaming datasets for large or ROOT-backed corpora | [Streaming datasets](streaming.md) |
| Reading `salt.step/*` profiler scopes | [Profiling](profiling.md) |

This page stays the one place that names all of the above; when a topic
moves, its heading here becomes a link rather than disappearing.
