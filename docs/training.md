# Training

Training runs through the `salt fit` CLI command; see [`salt fit`](cli.md#salt-fit)
for the full invocation and flag reference. Config values come from a merged YAML
stack, described in [how a config is assembled](configuration.md#how-a-config-is-assembled).
This page covers running a training well: device choice, checkpoints, reproducibility,
dataloading performance, batch systems and troubleshooting.

## Choosing devices

By default the config tries to use the first available GPU, but you can specify
which ones to use with the `--trainer.devices` flag. See
[Lightning's GPU docs](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus)
for the different ways to specify which GPUs to use.

??? warning "Check GPU usage before starting training"

    Check with `nvidia-smi` that any GPUs you use are not already in use by
    another user.

## Checkpoints and resume

Model checkpoints are saved per epoch under `ckpts/` next to the run's saved
config (`config.yaml`), named `epoch=<NNN>-loss=<value>.ckpt`; see
[what a run writes](cli.md#what-a-run-writes) for the full run-directory layout.
Checkpoints carry the resolved schema and per-mode plan hashes under the
top-level `salt_core` key.

Resuming with `--ckpt_path` restores the full training state, including the
optimiser. On resume, a FIT plan-hash mismatch is fatal: it means the graph or
dataset boundary changed. `Normaliser`/class-weight values come from the state
dict; the norm/class dicts are NOT re-read. Data-less loading (no checkpoint
file, just resolved modules) goes through `SaltModule.load_from_checkpoint(path,
modules=...)`.

For the exact `--ckpt_path` mechanics on each command, see [`salt fit`](cli.md#salt-fit)
and [`salt test`](cli.md#salt-test). Fine-tuning has its own warm-start-vs-resume
distinction (`--init_from` vs `--ckpt_path`); see
[Warm start vs. resume](finetuning.md#warm-start-vs-resume-init_from-vs-ckpt_path)
rather than duplicating it here.

The `OneCycleLR` scheduler fixes the total step count for its cycle up front.
Resuming a run past its original `--trainer.max_epochs` needs that schedule
re-armed: set `lrs.last_epoch: 0` in the config (or `--model.lrs.last_epoch 0`
on the command line for a single-stage run) alongside the new
`--trainer.max_epochs` so the total-step count is recomputed.

### Loading an older checkpoint

The v2 namespace was flattened from `salt.core.*` to `salt.*` (e.g.
`salt.core.nn.StreamEmbed` -> `salt.model.modules.StreamEmbed`,
`salt.core.SaltModule` -> `salt.model.SaltModule`). One loading policy applies
to everything:

- **Only current-format checkpoints/configs load**: class_paths must already
  name their flat `salt.*` home. There is no load-time remapping: a
  pre-rename checkpoint or saved `config.yaml` still embedding `salt.core.*`
  class_paths is **not loadable** and must be re-created from a current
  config (no converter is provided). The `salt_core` checkpoint **metadata
  key** is an unrelated plain dict key (not an import path) and was never
  part of this rename.
- **v1 checkpoints** (the `ModelWrapper` `model.pool_net.*` state-dict layout)
  are **not** directly loadable either; `SaltModule.on_load_checkpoint`
  rejects them with a clear error. Convert them offline with the v1->v2
  weight mapper (`map_v1_state_dict`, in git history at `fb90a7c`) or use
  them at the frozen pin `29c67a1` (see
  [Parity-closure doctrine](architecture.md#parity-closure-doctrine-v1-vs-v2-comparisons)).

## Random seeds

Training runs are reproducible thanks to the `seed_everything` key,
which is already set for you in the
[`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config.
This seeds all random number generators used in the training, so for
example weight initialisation and data shuffling happen deterministically.

??? info "Stochastic operations can still lead to divergences between training runs"

    For more info see [PyTorch's notes on randomness](https://pytorch.org/docs/stable/notes/randomness.html).

## Dataloading performance

Objects are loaded in weakly shuffled batches from the training file. This is
much more efficient than randomly accessing individual entries, which would
be prohibitively slow. Some other dataloading considerations are discussed
below.

### Worker counts

During training, data is loaded using worker processes. The number of
workers used is specified by `data.num_workers`. Increasing the worker count
speeds up training until you max out your GPU's processing capabilities.
Test different counts to find the optimal value, or just set this to the
number of CPUs on your machine.

??? info "Maximum worker counts"

    Find the number of CPUs available on your machine with

    ```bash
    cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
    ```

    You should not use more workers than this. If you use too few or too
    many workers, you will see a warning at the start of training.

### Fast disk access

Most HPC systems have dedicated fast storage. Loading training data from
these drives can significantly improve training times. To temporarily copy
training files into a target directory before training, use the
`--data.move_files_temp=/temp/path/` flag.

If you have enough RAM, you can load the training data into shared memory
before starting training by setting `move_files_temp` to a path under
`/dev/shm/<username>`.

??? warning "Ensure temporary files are removed"

    The code tries to remove the temporary files when the training is
    complete, but if the training is interrupted this may not happen. Double
    check whether you need to manually remove the temporary files to avoid
    clogging up your system's RAM.

## Running on a batch system

!!! tip "On CERN lxplus?"

    The [lxplus (CERN) setup tab](setup.md) ships a one-command environment
    setup plus a `salt-lxplus-gpu` helper that submits GPU jobs to the CERN
    HTCondor batch farm (`salt-lxplus-gpu submit <config>`) or opens an
    interactive GPU node (`salt-lxplus-gpu shell`). Start there rather than
    hand-writing submit files. See
    [`setup/lxplus_gpu.sub`]({{repo_url}}-/blob/main/setup/lxplus_gpu.sub) for
    the underlying HTCondor submit file.

Outside the lxplus helper above, salt ships no bundled Slurm or generic
HTCondor submitter; write your own submission script for other batch systems,
using [`salt fit`](cli.md#salt-fit) as the command to submit. The performance
levers below (the attention backend and `--compile`) matter whether you train
locally or via a batch system, and are usually set once in your config.

## torch.compile

`torch.compile()` traces the model into a graph and hands it to a backend
compiler (inductor), which can fuse kernels and cut Python overhead. Enable
it with the `--compile` flag on `salt fit`. The first step takes a while
(that is the compilation) and you may see warnings.

Salt compiles **each graph module in place** (`nn.Module.compile()`), not the
whole `SaltModule`: in salt there is no single `nn.Module` spanning the
forward, the graph is a `Plan` executed module-by-module. Compiling in place
keeps every module instance, plan step and `state_dict` key unchanged, so a
checkpoint written under `--compile` loads into an uncompiled model with no
repair step.

!!! warning "Measure before you enable it: compilation is not free, and the answer depends on the model"

    On a GN2-sized model `--compile` was **slower than eager on every
    attention backend except `torch-math`**. On a GN3-sized one it is a win
    on every backend measured. Both matrices are below, under
    [Attention backends](#attention-backends). It is worth trying, not worth
    assuming.

??? failure "If you see `g++` compile errors, you may need to update your compiler"

    Check your `g++`/`gcc` version with `g++ --version`. `torch.compile()`
    needs `gcc` version 10 or later.

    Install a more recent version with

    ```bash
    conda install -c conda-forge cxx-compiler
    ```

### Attention backends

Compilation interacts strongly with the attention backend (`attn_type`), so
the two must be chosen together: the tables below benchmark both dimensions
at once.

??? abstract "Measured: `--compile` x attention backend, GN2v2 on one A100 80GB"

    GN2v2 open-data (256/128, 4 layers, 8 heads), batch 1000, `16-mixed`,
    seed 42, 220 training steps per cell, torch 2.12.1+cu126. Rates exclude a
    20-step warmup; "first step" is the one-off compilation cost.

    | backend        | eager      | `--compile` | speedup | peak memory (eager) |
    | -------------- | ---------- | ----------- | ------- | -------------------- |
    | `torch-math`   | 19.1 it/s  | 23.1 it/s   | 1.21x   | 2.57 GB              |
    | `torch-flash`  | 18.9 it/s  | n/a         | n/a     | 2.57 GB              |
    | `torch-meff`   | 27.97 it/s | 25.2 it/s   | 0.90x   | 2.15 GB              |
    | `flash-varlen` | 27.96 it/s | 26.3 it/s   | 0.94x   | **1.36 GB**          |

    Read this carefully before enabling `--compile`:

    - **The fastest configuration is eager**, on either `torch-meff` or
      `flash-varlen` (~28 it/s). No compiled cell beats it.
    - Compilation only pays off on `torch-math`, and even then the result
      (23.1 it/s) is still slower than plain eager `flash-varlen`.
    - It also costs 35-65 s of compilation per run before the first step.
    - `torch-flash` measures identically to `torch-math` because PyTorch's
      flash SDPA kernel rejects padding masks and silently falls back to
      math: this is why the `torch-flash` numbers track `torch-math`
      throughout this page.
    - Peak memory is *not* unaffected by compiling (an older version of this
      page claimed it was): it moved by -6% to +3% depending on backend.

    The `flash-varlen` figure requires the unpad/repad seam to be excluded
    from the compiled region (`torch.compiler.disable` in
    `salt/utils/tensor_utils.py`). Without that, the same cell runs at
    20.4 it/s (0.73x): the boolean-mask index lowers to `aten.nonzero`, which
    inductor cannot lower on CUDA, and the resulting graph breaks and
    recompiles cost ~29% of throughput.

??? abstract "Measured: `--compile` x attention backend, GN3V00 on one A100 80GB"

    The GN2 verdict above does **not** carry over. GN3V00 (512/256, 4 layers,
    8 heads, 8 registers, two padded streams of 50 slots each, five task
    heads), batch 1000, `16-mixed`, seed 42, 220 training steps per cell,
    torch 2.12.1+cu126.

    | backend        | eager      | `--compile` | speedup | first step (compile) | peak memory (eager -> compile) |
    | -------------- | ---------- | ----------- | ------- | --------------------- | ------------------------------- |
    | `torch-math`   | 4.95 it/s  | 6.99 it/s   | 1.41x   | 88 s                  | 15.6 -> 13.4 GB                 |
    | `torch-meff`   | 7.46 it/s  | 8.82 it/s   | 1.18x   | 35 s                  | 13.1 -> 12.1 GB                 |
    | `flash-varlen` | 10.34 it/s | 10.87 it/s  | 1.05x   | 69 s                  | 8.6 -> 8.2 GB                   |

    - **The fastest configuration is `flash-varlen`**, compiled or not;
      compiled is fastest overall. Unlike GN2, `flash-varlen` here is a large
      *speed* win over the SDPA backends (1.4x over `torch-meff`, 2.1x over
      `torch-math`) as well as a memory win, because 64.8% of the padded slot
      budget is padding at this configuration, and that is exactly the work
      `flash-varlen` skips.
    - The `flash-varlen` speedup is small (+5%). It was measured against an
      in-job control (the same eager cell re-run last in the same
      allocation), which came out within 0.9% of the first, so the +5% is
      real but modest. Do not read a compile claim of this size from two
      separate jobs: run-to-run scatter across jobs on this benchmark is
      ~5-7%.
    - Compilation costs 35-90 s before the first step. On a fixed-work
      benchmark that is most of the gain; on a real multi-epoch training run
      it is noise.
    - Loss parity held everywhere (worst drift 0.02% against a 2% tolerance).

??? abstract "Measured: what actually goes fastest, GN3V00 on one A100 80GB"

    The table above holds batch size fixed at 1000 to isolate the backend.
    That is not how you would train. `flash-varlen` uses less than a quarter
    of the memory, so it also fits a much larger batch, and the batch is
    where most of the throughput is. Same model, same data, same job; rate
    measured over a fixed 250,000-jet budget per cell so jets/s is
    comparable across batch sizes.

    | configuration                              | batch | jets/s | 1.5M-jet epoch | peak memory | first step |
    | ------------------------------------------- | ----- | ------ | --------------- | ----------- | ---------- |
    | `torch-math`, eager (the shipped default)   | 1000  | 5,191  | 4.82 min        | 15.6 GB     | 10 s       |
    | `flash-varlen`, eager                       | 1000  | 11,375 | 2.20 min        | 8.6 GB      | 11 s       |
    | `flash-varlen`, eager                       | 5000  | 14,835 | 1.69 min        | 40.6 GB     | 12 s       |
    | `flash-varlen`, `--compile`                 | 5000  | 17,223 | **1.45 min**    | 38.6 GB     | 70 s       |

    **3.3x** end to end, and two thirds of it is free: switching the
    attention backend and raising the batch costs nothing but a config edit.
    The last step, `--compile`, buys a further 1.16x for a 70 s charge before
    the first batch, so it is worth it from roughly the seventh epoch onwards
    and clearly worth it over GN3V00's shipped 40.

    Note the memory column: the compiled cell is not just as fast as it can
    be, it also holds batch 5000 in **less** memory than the eager cell does.

    Practical recipe, in the order the wins arrive:

    1. `attn_type: flash-varlen`: 2.2x, and it frees the memory that funds
       step 2.
    2. Raise the batch until it stops helping: a further 1.3x here.
    3. `--compile` if you are training for more than ~7 epochs: a further
       1.16x.
    4. `optimizer: lion` rather than `lion-pytorch` (this is the default);
       see below.

    The optimizer is worth calling out because it is invisible in a backend
    table. Profiling put `lion-pytorch`'s per-parameter Python loop at
    20.6 ms of a 96 ms step for 1.33 ms of actual kernel work (595 launches
    at 6% GPU-busy). `salt.optim.Lion` issues the identical arithmetic
    through `torch._foreach_*`; the profiled optimizer span drops from
    **20.6 ms to 2.9 ms**, worth 1.11x at batch 1000 and 1.02-1.03x at batch
    5000 (the saving is roughly constant in absolute terms, so it matters
    most when the step is short). Parameters and `exp_avg` are
    **bit-identical** between the two, gated on both CPU and A100 in fp32,
    bf16 and fp16.

### Graph breaks

A compiled salt model is not one graph. Dynamo splits the trace wherever it
meets something it cannot capture, and each split costs the fusion across it.
Two splits are deliberate and permanent:

| Seam | Where | Why it cannot be captured |
| ---- | ----- | ------------------------- |
| flash-varlen unpad/repad | `salt/utils/tensor_utils.py` | boolean-mask index -> `aten.nonzero`, which inductor refuses to lower on CUDA |
| vertexing head | `VertexingTaskModule.head_forward` | compresses a `[B, N, N]` adjacency to one row per valid edge: both the allocation size and the indices are data-dependent |

Everything else is expected to capture. `salt/tests/unit/model/test_compile_regression.py`
is the gate: it replays a compiled plan module-by-module under
`torch._dynamo.explain` and fails on any graph break at a site that is not on
its checked-in allowlist, on any module that captures no graph at all, and on
an encoder that will not compile with `fullgraph=True`. It runs on CPU in CI.
If you add a `.item()`, a boolean mask, or a branch on a tensor value to a
module's `forward`, that test tells you.

It is worth the gate. A GN3V00 `--compile` + `flash-varlen` run previously
took **14 distinct break sites and 49 break events**, plus one hard fallback
where dynamo skipped the whole loss frame and ran it eagerly. The breaks were
four cheap habits (a generator inside a reduction, a bool read off a buffer,
a NaN check on a tensor value, and boolean stream selection where a slice
would do) and one head that cannot be traced at all. Removing them left
**3 sites, 8 events, no fallback** (the two seams above) and cut recompiles
from 44 to 18.

If you are chasing the remainder: the dominant recompile guard is
`GLOBAL_STATE changed: grad_mode`, which fires when the trainer flips between
training and validation. It settles once both variants are cached; it is not
a per-batch cost.

!!! warning "`--compile` has not been tested with multi-GPU training"

## Troubleshooting

If you encounter issues, as a first step you should try pulling the latest
updates from `main` to see if your problem has been resolved. If you need
more help you can post on
[mattermost](https://mattermost.web.cern.ch/aft-algs/channels/gnns).

### Slow training

Before changing anything, measure: [Profiling](profiling.md) covers the
built-in `--trainer.profiler simple|advanced`, the `salt profile dataset`
line profiler for the read path, and the `torch.profiler` callback that
splits GPU time across encoder, task heads and optimizer.

For a GN3-shaped model the three changes that matter most, in order, are the
attention backend, the batch size, and `--compile`: together **3.3x** on a
measured A100 benchmark, most of it from the first two. See
[torch.compile](#torchcompile) and [Attention backends](#attention-backends)
above for the numbers and the recipe.

This section contains some suggestions for speeding up trainings. Some
external advice can be found
[here](https://lightning.ai/docs/pytorch/stable/advanced/speed.html) and
[here](https://lightning.ai/docs/pytorch/stable/levels/intermediate_level_13.html).

If you are not producing a "final" version of your model (i.e. with maximum
possible performance), but instead are running some studies, you should
consider the following:

- Limit the training statistics (e.g. 20M samples)
- Reduce the number of epochs you train for (e.g. 20 epochs)
- Remove any auxiliary tasks
- [Compile the model](#torchcompile)

Other things you can always do:

- Use bfloat16 precision
- Use the flash attention backend for the [`Transformer` class](https://gitlab.cern.ch/svanstro/hepformer/-/blob/main/hepformer/models/transformer.py)
- Use the maximum possible [batch size](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html)
- Increase your effective batch size by [accumulating gradients](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients)
- Ensure you have enough [workers for dataloading](#worker-counts)
- Use newer GPUs if possible
- Use [multiple GPUs](#choosing-devices)
- Reduce the size of the model (in particular the number of layers)

### Confusing errors

You might see confusing/cryptic errors when running on the GPU, for example

```
../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [377,0,0], thread: [13,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

Often, if you instead run on the CPU you will get a much more helpful error
message. Use `--trainer.accelerator=cpu` to run on the CPU instead of the
GPU.

### NaNs

Salt automatically checks that your normalisation parameters are finite (see
`salt.model.modules.Normaliser`). You may still encounter `nan` values in
your outputs and losses. Here are some mitigation strategies you can try:

- Make sure you have pulled the latest changes from `main`.
- Make doubly sure that your inputs are finite, even after applying
  normalisation.
- Ensure you don't have unexpected non-finite labels.
- Try lowering your max learning rate (`lrs.max`, either the top-level
  `lrs:` block or a stage override under `training_schedule.stages.<name>.lrs`).
- If you apply very large loss weights in your task configs, these might
  contribute to large gradients, so you can try removing any loss weights
  provided to your task modules (`salt.model.modules.tasks`).
- Check your training precision: if you have done the above and still have
  problems, you can try `--trainer.precision=32` or
  `--trainer.precision=bf16-mixed`. See
  [Lightning's precision docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision)
  for more info.
- Apply [gradient clipping](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping)
  to negate the effects of exploding gradients.
- [Auto detect gradient anomalies](https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#detect-autograd-anomalies).
- If you are running on multiple GPUs, try running on a single GPU with
  `--trainer.devices=1`.
