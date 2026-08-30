# Profiling

Training has two halves and they need different tools.

- The **dataset** side (readers, processors, the numpy→torch boundary) is
  synchronous Python + `h5py` + numpy. Per-line timing is meaningful, so salt
  ships a `line_profiler` harness: `salt profile dataset`.
- The **model** side is asynchronous CUDA. Per-line Python timing *lies* — a line
  that enqueues a kernel costs nothing, and whichever line next synchronises
  inherits the blame. Use `torch.profiler`, which timestamps the kernels
  themselves: `salt profile model`, or its callback
  `salt.profiling.TorchProfilerCallback` attached to a run you were doing anyway.

Both subcommands take the same `--config` stack, the same `--set K=V` overrides
and the same `--steps` (default **100**).

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
salt fit --config configs/GN3/GN3V00.yaml --trainer.profiler simple

# cProfile per function, sorted by cumulative time — the closest built-in
# to a line profile, and the right first look at CPU-side overhead
salt fit --config configs/GN3/GN3V00.yaml --trainer.profiler advanced
```

`simple` prints a table of Lightning's own action names — `run_training_batch`,
`[Strategy]…backward`, `[LightningModule]SaltModule.optimizer_step`,
`[_TrainingEpochLoop].train_dataloader_next` — with mean and total wall time
each. It is the fastest way to see whether you are dataloader-bound.

!!! warning "`optimizer_step` in that table is not the optimizer"

    Lightning calls `optimizer.step(closure)`, and the closure runs the forward
    and the backward. `[LightningModule]SaltModule.optimizer_step` therefore
    reports ~99 % of the batch — it *contains* `training_step` and `backward`
    rather than sitting beside them. Read the table as a nesting, not a
    partition. For an actual partition of GPU time, use the torch.profiler
    callback below.

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
    --config configs/GN3/GN3V00.yaml \
    --steps 100 \
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
   `@profile` decorators in the source), iterates `--steps` batches, and
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
(`SaltDataset.__getitem__` / `_to_torch`, `H5StructuredReader.read` /
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
numpy→torch conversion in `SaltDataset._to_torch` is what a conventional
pipeline would call collation.

## 3. Model side — `salt profile model`

```bash
salt profile model \
    --config configs/GN3/GN3V00.yaml \
    --steps 100 \
    --out profile/
```

The subcommand runs a **short, capped, throwaway fit** — one epoch of `--steps`
train batches, no logger, no checkpoints, no graph artifacts, no validation —
with `TorchProfilerCallback` attached, and then prints the per-op table, the
per-plan-step split and the forward/backward/optimizer buckets. `--set K=V`
overrides work exactly as they do for `salt profile dataset`, and `--compile`
profiles the compiled model.

The `--steps` budget is spent getting to steady state: the capture is its
**tail**. At the default 100 steps that is `wait 75, warmup 5, active 20` — the
last twenty batches. Pin any phase explicitly with `--wait` / `--warmup` /
`--active`; a schedule that would not fit inside `--steps` is rejected up front
rather than silently recording nothing.

Unlike the callback's own defaults, the subcommand runs with `with_stack` and
`profile_memory` **off** (see the warning below); `--with-stack`,
`--profile-memory` and `--record-shapes` turn them on.

### The callback directly

Use the callback when you want the profile of a run you were going to do
anyway, rather than a throwaway one:

```bash
salt fit --config configs/GN3/GN3V00.yaml \
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

!!! warning "`with_stack` is expensive, and some builds record nothing"

    `with_stack=True` is a request, not a guarantee. If torch returns events with
    empty stacks, `<tag>_stacks.txt` says so explicitly and
    `summary.json` sets `stacks_available: false` — the per-module
    `salt.step/*` rows are the attribution in that case.

    It also costs **host RAM**, not GPU memory: assembling the kineto result
    for a long window can allocate many GB and has been seen to die with
    `MemoryError: std::bad_alloc` inside `_disable_profiler()` at large batch
    sizes. If that happens, drop `with_stack`, shorten `active`, or ask the
    batch system for more RAM.

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

On GN3V00 / one A100 / `flash-varlen` the measured overhead was **1.5–1.7x at
batch 1000** and **1.07–1.11x at batch 4500–5000** — it is a per-step cost, so
it shrinks as the step grows.

## 4. Worked example: what this found on GN3V00

A single baseline pass over GN3V00 on one A100 80GB (`flash-varlen`, `16-mixed`,
eager and `--compile`, batch 1000 and each mode's guarded maximum) produced the
following. It is a useful calibration for what the harness can tell you.

??? abstract "Measured: where a GN3V00 step goes"

    Device time per step, split by the `salt.step/*` scopes plus the autograd
    engine and torch's optimizer scope. The split closes to **99–102 %** of the
    step time the *unprofiled* run actually took, so it accounts for the whole
    step.

    | cell | batch | forward | backward | optimizer |
    | --- | --- | --- | --- | --- |
    | eager | 1000 | 32.0 % (31.0 ms) | 46.8 % (45.3 ms) | **21.3 % (20.6 ms)** |
    | `--compile` | 1000 | 36.2 % (33.8 ms) | 41.4 % (38.7 ms) | **22.4 % (21.0 ms)** |
    | eager | 5000 | 31.9 % (109.6 ms) | 63.1 % (216.4 ms) | 5.0 % (17.2 ms) |
    | `--compile` | 4500 | 30.0 % (80.9 ms) | 63.3 % (170.6 ms) | 6.7 % (18.1 ms) |

    Two of the twelve configured modules are 73–88 % of the forward — the
    encoder and the vertexing head. Normalisation is 0.25 ms/step (0.3 %), and
    the four other task heads are under 2 ms/step each.

    Cross-checking against the Chrome trace (span on the GPU timeline vs the
    summed duration of the kernels actually launched) sharpens it further:

    | scope | span | kernel work | busy | kernels/step |
    | --- | --- | --- | --- | --- |
    | `Optimizer.step#Lion.step` | 20.6 ms | 1.33 ms | **6 %** | **595** |
    | `salt.step/encoder` | 13.0 ms | 11.7 ms | 90 % | 115 |
    | `salt.step/track_vertexing` | 9.0 ms | 6.8 ms | 75 % | 104 |

    The optimizer is **launch-bound**, not compute-bound: 595 tiny kernels whose
    work totals 1.3 ms occupy 20.6 ms of the step. That is a `foreach`-shaped
    problem, and it is batch-size independent — which is a large part of why a
    bigger batch helps so much.

??? success "…and what happened when that finding was acted on"

    The optimizer row above is the reason `salt.optim.Lion` exists. It issues the
    same arithmetic as the `lion-pytorch` reference — bit-identically, gated in
    `salt/tests/unit/test_optim.py` — through `torch._foreach_*`, so the launch
    count is per parameter *group* rather than per parameter.

    Re-running the same capture on the same GPU afterwards (`salt profile model
    --config configs/GN3/GN3V00.yaml --steps 100`, `flash-varlen`, batch 1000):

    | bucket | before | after |
    | --- | --- | --- |
    | forward (sum of `salt.step/*`) | 31.0 ms | 30.2 ms |
    | backward (autograd engine) | 45.3 ms | 45.6 ms |
    | **optimizer** | **20.6 ms (21 %)** | **2.9 ms (3.7 %)** |

    Forward and backward are unchanged to within a percent, which is the check
    that the fix went where the profile said it would. End-to-end that is 1.11x
    at batch 1000; at batch 5000 the same ~6-9 ms saving is 1.02x, because the
    saving is roughly constant and the step is four times longer.

The lesson generalises: the per-module split tells you *where*, and the
span-vs-kernel-work ratio tells you *which kind of fix* — fuse the launches when
the scope is idle-dominated, change the algorithm when it is busy. And the
follow-up capture is not optional: it is what distinguishes "the fix worked" from
"something else moved at the same time".
