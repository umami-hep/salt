# Profiling

Training has two halves and they need different tools.

- The **dataset** side (readers, processors, the numpy→torch boundary) is
  synchronous Python + `h5py` + numpy. Per-line timing is meaningful, so salt
  ships a `line_profiler` harness: `salt profile dataset`.
- The **model** side is asynchronous CUDA. Per-line Python timing *lies* — a line
  that enqueues a kernel costs nothing, and whichever line next synchronises
  inherits the blame. Use `torch.profiler`, which timestamps the kernels
  themselves: `salt.profiling.TorchProfilerCallback`.

Start with the cheap answer (`--trainer.profiler`), and only reach for the
other two when you need to know *which* module or *which* line.

!!! warning "A profiled run is not a benchmark"

    Every profiler here costs throughput — `advanced` (cProfile) and
    `torch.profiler` with `with_stack=True` cost a lot. Quote rates from
    unprofiled runs; use profiled runs for attribution only.

## 1. Lightning's built-in profilers

`salt fit` is a `LightningCLI`, so the trainer's profiler argument is already
exposed. Nothing to install, nothing to configure:

```bash
# per-Lightning-action wall time: dataloader next, forward, backward, optimizer
salt fit --config configs/GN3V00.yaml --trainer.profiler simple

# cProfile per function, sorted by cumulative time — the closest built-in
# to a line profile, and the right first look at CPU-side overhead
salt fit --config configs/GN3V00.yaml --trainer.profiler advanced
```

`simple` prints a table of Lightning's own action names — `run_training_batch`,
`[Strategy]…backward`, `[LightningModule]SaltModule.optimizer_step`,
`[_TrainingEpochLoop].train_dataloader_next` — with mean and total wall time
each. It is the fastest way to see whether you are dataloader-bound.

`advanced` adds a cProfile report per action. It is heavy — use
`--trainer.limit_train_batches 20` with it.

A third value, `--trainer.profiler pytorch`, wires Lightning's `PyTorchProfiler`.
It works, but its default schedule records only three steps and its constructor
arguments are awkward to reach from YAML, so salt ships its own callback below
for the GPU work.

## 2. Dataset side — `salt profile dataset`

```bash
pip install 'salt-ml[profile]'      # line_profiler is an optional dependency

salt profile dataset \
    --config configs/GN3V00.yaml \
    --batches 50 \
    --out profile/
```

The harness:

1. parses the config stack run-free (repeat `--config` to stack, `--set K=V` to
   override) and builds the datamodule,
2. **forces `num_workers=0`** so the profiled code runs in this process (a
   worker fork would hide all of it),
3. draws one batch through the *unprofiled* loader, then compares the nested
   keys/shapes/dtypes with the first profiled batch — if they differ the run
   fails rather than silently reporting a partial pipeline,
4. wraps the read path in a `LineProfiler` (patching the class attribute — no
   `@profile` decorators in the source), iterates `--batches` batches, and
   restores the originals.

Outputs land in `--out`: `dataset_profile.txt` (the classic annotated listing),
`dataset_profile.lprof` (for `python -m line_profiler`), and
`dataset_summary.json` (per-function totals and the ranked hot lines).

`dataset_profile.txt` is the standard `line_profiler` listing — one block per
wrapped function, with `Hits`, `Time`, `Per Hit`, `% Time` and the source line
alongside each other, so the H5 slab read, the constituent cuts, the
`structured_to_unstructured` copy and the numpy→torch conversion are separated
by line rather than by function.

The default function list is `salt.profiling.DEFAULT_DATASET_FUNCTIONS`
(`GraphDataset.__getitem__` / `_to_torch`, `H5StructuredReader.read` /
`_read_kept`, `ConstituentCuts.apply`, `Features.process`, `Labels.process`,
`Bundle.merge`). Extend or replace it:

```bash
# add your own module
salt profile dataset --config my.yaml \
    --extra-functions mypkg.processors.MyProcessor.process

# or profile a different set entirely
salt profile dataset --config my.yaml \
    --functions salt.data.processors.features.Features.process
```

Note that salt has **no collate step**: `batch_size=None` plus
`RandomBatchSampler` means the dataset returns whole batches, so the
numpy→torch conversion in `GraphDataset._to_torch` is what a conventional
pipeline would call collation.

## 3. Model side — `TorchProfilerCallback`

```bash
salt fit --config configs/GN3V00.yaml \
    --trainer.callbacks+=salt.profiling.TorchProfilerCallback \
    --trainer.callbacks.dirpath profile/ \
    --trainer.callbacks.tag eager_flash \
    --trainer.callbacks.wait 5 \
    --trainer.callbacks.warmup 5 \
    --trainer.callbacks.active 20
```

or, in a config file:

```yaml
callbacks:
  profiler:
    class_path: salt.profiling.TorchProfilerCallback
    init_args:
      dirpath: profile/
      tag: eager_flash
      wait: 5
      warmup: 5
      active: 20
      with_stack: true
      profile_memory: true
```

The `wait`/`warmup`/`active` schedule matters: it skips startup and — under
`--compile` — the 60–90 s compilation step, so the capture is steady-state.

Four artifacts per capture:

| file | what it answers |
| ---- | --------------- |
| `<tag>_key_averages.txt` | which CUDA kernels dominate |
| `<tag>_stacks.txt`, `<tag>_stacks.flame` | which source lines launched them (`with_stack`) |
| `<tag>_trace.json.gz` | the timeline — load in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) |
| `<tag>_summary.json` | forward time **per plan step**, backward, optimizer, top ops, profiled it/s |

### Per-module attribution

The callback enables `salt.graph.executor.record_steps`, which wraps each plan
step's module call in a `salt.step/<name>` profiler scope. The keys in
`forward_steps_us` are therefore your own config's module names —
`salt.step/encoder`, `salt.step/track_origin`, `salt.step/norm` — so the split
between encoder, task heads and normalisation is read straight off the summary
rather than reverse-engineered from aten ops.

The scope sits **outside** the module call. Salt compiles each graph module in
place (see [Compiled Models](configuration.md#compiled-models)), so the scope
introduces no graph break and the compiled region is byte-for-byte the same as
in an unprofiled run — the eager and compiled captures are directly comparable.

Backward work is *not* inside those scopes (it runs later, in the autograd
engine). The summary reports it separately as `backward_scope_device` (the
`salt.backward` scope around `loss.backward()`) and `autograd_engine_device`
(the sum over `autograd::engine::evaluate_function*`), and the optimizer as
`optimizer_device` (torch's own `Optimizer.step#<Class>.step` scope).

If you want the step scopes under Lightning's `PyTorchProfiler` instead, add
`salt.profiling.PlanStepScopes` as a callback — it turns them on for the run
and does nothing else.

### Measuring the profiler's own cost

`profiled_it_s` in the summary is the rate measured *inside* the active window.
Run the same cell once without the callback and divide: that ratio is the
profiler overhead, and it is the only honest way to report it.
