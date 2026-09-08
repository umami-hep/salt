# Fine-tuning

Salt has two independent tools for adapting a pretrained checkpoint.
`--init_from` is a weights-only warm start: a fresh run whose weights are
seeded from a checkpoint, module by module, with per-module accounting.
`training_schedule:` is a staged training plan: freeze part of the network
for some epochs, then unfreeze, each stage with its own learning rate. This
page is the reference for both. The worked walkthrough is [Fine-tuning a
pretrained tagger](tutorials/finetuning.md); the GN3Large module map it
starts from is at [The model you start
from](tutorials/finetuning.md#the-model-you-start-from).

## Before you start

Fine-tuning needs an existing, working base config and its checkpoint.
Writing that base config is not this page's job: see [the minimum graph that
compiles](configuration.md#the-minimum-graph-that-compiles) for what one has
to contain. `salt/configs/gn2v2-opendata.yaml` is a complete, shipped,
account-free config you can train from scratch and then fine-tune against,
end to end, with no CERN account needed. If you have access to the
GN3Large bundle the tutorial below uses, start from that; otherwise, run the
same overlay pattern against `gn2v2-opendata.yaml` instead.

## Warm start vs. resume: `--init_from` vs `--ckpt_path`

These two flags both take a checkpoint and both feed weights into a fit, so they
are easy to confuse. They do opposite things and are **mutually exclusive**:
passing both is a hard `ConfigError`.

| | `--ckpt_path` (resume) | `--init_from` (warm start) |
|---|---|---|
| **Intent** | Continue an *interrupted* run | Start a *new* run from pretrained weights |
| **Trainer state** | Restored (epoch, optimizer, LR schedule, stage position) | Fresh (epoch 0, new optimizer) |
| **Epoch counter** | Continues where it stopped | Starts at 0 |
| **State-dict load** | Strict — architecture must match exactly | Prefix-filtered per module, with accounting |
| **Plan-hash gate** | Enforced (same architecture) | Informational only (architecture may have changed) |
| **Subcommand** | `fit` and `test` | `fit` only |

Use **`--ckpt_path`** when a job died at epoch 7 of 15 and you want epoch 8 to
carry on as if nothing happened. Use **`--init_from`** when you have a
*finished*, good model and want a *new* training run that begins from those
weights instead of random init, on new data or with the network surgically
modified.

### What `--init_from` accounts for

A warm start does not blindly `load_state_dict`. It classifies every module (by
its `net.<name>.*` state-dict prefix) into one of three buckets and logs the
result:

- **loaded**: the module exists in both the checkpoint and your config, with an
  identical key set, shapes, and dtypes. Its weights are copied in.
- **new**: the module is in your config but *not* in the checkpoint (e.g. a head
  you just added). Left at fresh random init and materialised normally.
- **dropped**: the module is in the checkpoint but *not* in your config. Skipped
  and logged.

There is a deliberate trap here: a module that is present in *both* but only
**partially** covered (its internal architecture changed: different layer
widths, a missing sub-key) is a hard `ConfigError`, not a silent partial load.
The fix is to **rename** the module: under a new name it has no checkpoint
counterpart, so the old weights are dropped and the renamed module is **new**
(fresh init), a clean swap through the same three buckets. The rule:
*warm-starting is per-module all-or-nothing*. Changing a module's shape is a
swap; declare it as one by renaming. ([What else can you
change?](#what-else-can-you-change) puts this rule to work across every kind
of surgery, and [worked example
4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb)
is its fullest instance: renaming `norm` to `norm_xbb` when both the model's
stream set and each stream's variables change.)

## The `training_schedule:` schema

`training_schedule:` is a **top-level** config key, a peer of `trainer:`,
`data:`, and `model:`, **not** nested inside `model:`. (A
`model.init_args.training_schedule` is rejected fail-loud with a `ConfigError`
naming this location.) The CLI injects it into the `SaltModule` constructor
before the model is built.

A schedule is a dict of named **stages**:

```yaml
training_schedule:
  stages:
    head_warmup:                      # stage name (your choice)
      epochs: 5                       # how long this stage runs
      trainable: [jets_classification]  # only these modules train; the rest freeze
      lrs:
        max: 1.0e-4                   # per-stage LR override (deep-merged over base lrs)
    full_finetune:
      frozen: []                      # freeze nothing — the whole network trains
      lrs:
        initial: 1.0e-7
        max: 1.0e-5
      # epochs omitted → this stage takes the *remaining* epochs
```

The per-stage keys, and there are exactly these, anything else is a config typo
rejected fail-loud:

- **`epochs`**: a positive integer. Every stage **except the last** must give an
  explicit `epochs`; the final stage may omit it to take the remainder of
  `trainer.max_epochs`. The explicit epochs must sum to `≤ max_epochs`, and an
  omitted final stage needs at least 1 epoch left over. With `max_epochs: 15` and
  `head_warmup.epochs: 5`, `full_finetune` runs the remaining **10** epochs.
- **`frozen`** / **`trainable`**: a list of `model.modules` names. Give **at most
  one**: `frozen` freezes exactly those modules; `trainable` freezes their
  *complement* (everything else). Setting both is a `ConfigError`. Neither freezes
  nothing (the whole network trains). The names must be real `model.modules` keys
  or it fails loud. A **frozen** module is excluded from the optimizer and put in
  `eval()` mode, so dropout and running statistics stop updating; its weights are
  held bitwise-immobile for the stage. Unfreezing restores it to `train()` and
  re-adds it to the optimizer.
- **`lrs`**: a mapping that **deep-merges over** the base config's `lrs:`. Only
  list the keys that differ per stage; the rest fall through. (Above, `head_warmup`
  keeps the base `initial`/`end`/`pct_start` and overrides only `max`.)
- **`optimizer`**: a per-stage optimizer name, if a stage needs a different one.
  Usually omitted so every stage shares the base `optimizer:`. The allowed
  values and their exact spelling are at [`model:`](configuration.md#model)
  in the configuration reference.
- **`order`**: pins execution position. Omit it and stages run in declaration
  order (the normal case).
- **`early_stop`**: an optional early-stopping criterion that ends the stage
  before its `epochs` cap (see [Per-stage early
  stopping](#per-stage-early-stopping-early_stop)).
- **`callbacks`**: an optional list of extra Lightning callbacks active only
  during this stage (see [Per-stage callbacks](#per-stage-callbacks-callbacks)).
- **`lr_scheduler`**: an optional LR-scheduler *class* for this stage, replacing
  the default OneCycleLR (see [Per-stage LR
  scheduler](#per-stage-lr-scheduler-lr_scheduler)).

### When a schedule error fires

Some `training_schedule` mistakes are caught before a training process ever
starts; a few need a trainer attached first. Knowing which is which saves a
wasted GPU allocation.

**Caught at parse time**, so `salt merge-config` and `--print_config` catch
them with no GPU, no data file and no checkpoint: an unknown top-level key
under `training_schedule` or an unknown per-stage key, both `frozen` and
`trainable` set on one stage, and an unknown module name inside `frozen:` /
`trainable:` (`salt/schedule.py:316-331`, `:353-367`, `:567-584`).

**Caught at fit setup, not at parse time**: a non-final stage that omits
`epochs`, an over-allocated epoch budget across stages, and a multi-stage
schedule under a non-finite `trainer.max_epochs`. All three come from
`TrainingSchedule.validate_epochs` (`salt/schedule.py:246`), called from
`SaltModule._apply_training_schedule` once the trainer is attached
(`salt/model/saltmodule.py:831`), because the check needs the trainer's
`max_epochs` and that value does not exist before then. `salt merge-config`
and `--print_config` only merge and print the config; neither attaches a
trainer, so neither one catches these three.

### How the pieces layer

Everything that **shapes the schedule** (`frozen`/`trainable`, `epochs`,
`optimizer`, `lrs`, `early_stop`) is a **stage key**, not a callback, and that is
deliberate. Those concerns need first-class integration with the stage machinery:
they drive the optimizer rebuild at each boundary, the per-stage LR envelope, the
checkpoint boundary records that make a resume stage-correct, and the
DDP freeze flip. A Lightning callback cannot reach into any of that.

`early_stop` is the sharpest example. It is **not** a Lightning `EarlyStopping`
callback, because that callback can only do one thing: kill the whole fit. A
per-stage criterion has to be able to end *this stage* and hand off to the next,
so it is a stage key the schedule owns, wired into the boundary logic.

A stage's own **`callbacks:`** are for the opposite kind of thing:
stage-scoped *instrumentation and side-effects* (a monitor, a diagnostic, a
per-stage checkpoint policy) that observe a stage without steering it.

By default every stage runs a `OneCycleLR` envelope; `lrs:` tunes its parameters
(`max`, `initial`, `end`, `pct_start`) per stage. A stage that needs a *different*
scheduler class entirely (a cosine warm-up, a plateau-driven finetune) declares
[`lr_scheduler:`](#per-stage-lr-scheduler-lr_scheduler); the chosen class is built
over that stage's optimizer at the boundary, in place of OneCycle.

### Stacking and deleting stages

Stages **deep-merge by name** across stacked `--config` files, exactly like
`callbacks:` or `modules:`. That is what makes a fine-tune config an *overlay*: you
save a base run's `config.yaml`, then supply a second `--config` that adds or
tweaks stages. A stage set to `null` is **deleted** from the merged schedule:

```yaml
# overlay.yaml — drop the warm-up, keep only a full fine-tune
training_schedule:
  stages:
    head_warmup: null        # delete the inherited warm-up stage
```

A single-stage schedule (or no `training_schedule:` at all) is the plain-training
path: one implicit `fit` stage, no freezing, LR straight from the top-level
`lrs:`. Staged and unstaged training share one code path, so adding a schedule
never perturbs a plain run.

!!! warning "Multi-stage schedules need a finite `max_epochs`"

    Per-stage epoch and LR-envelope allocation is computed against
    `trainer.max_epochs`. A multi-stage schedule with `max_epochs: -1` (infinite
    training) is a `ConfigError`: staged training must know how many epochs it is
    dividing up.

### Per-stage early stopping: `early_stop`

`early_stop` lets a stage end **before** its `epochs` cap once its monitored
metric stops improving. This is exactly the two blocks already shown in
[worked example 1's
schedule](tutorials/finetuning.md#worked-example-1-add-the-calo-stream):
`calo_warmup` has `patience: 1`, `full_finetune` has `patience: 2`.

```yaml
# from finetune_gn3large_add_calo.yaml — the shipped early_stop blocks
training_schedule:
  stages:
    calo_warmup:
      early_stop:
        monitor: val/loss               # required — a trainer.callback_metrics key
        mode: min                       # min (default) or max
        patience: 1                     # validation checks without improvement (default 3)
        min_delta: 0.0                  # minimum improvement to reset patience (default 0.0)
        check_finite: true              # stop on a non-finite monitor value (default true)
    full_finetune:
      early_stop:
        monitor: val/loss
        mode: min
        patience: 2
```

`epochs` stays the hard cap; the stage ends at whichever comes first. **On the
final stage** (`full_finetune` here) early-stopping **ends the fit**; on a
**non-final stage** (`calo_warmup`) the schedule instead **advances to the next
stage**.

- The check runs at **validation-epoch end**; `patience` counts validation checks,
  not raw training epochs. Validation must be enabled (a stage with `early_stop`
  under a trainer with validation disabled is a fail-loud `ConfigError`), and the
  `monitor` key must exist in `trainer.callback_metrics` (e.g. `val/loss`,
  `val/jets_classification_loss`) or it fails loud at the first check.
- An early-stopped stage truncates its LR envelope mid-curve; the next
  stage rebuilds its own OneCycle cleanly. Boundaries become data-dependent, so
  they are **recorded in the checkpoint**: a resume reconstructs the exact stage
  position and patience counters (a mid-stage resume continues patience
  identically).
- Under multi-GPU DDP the decision is **rank-synchronised** (all ranks transition at
  the same step), so the freeze flip and optimizer rebuild never desync.
- A config with no `early_stop` on any stage behaves bitwise-identically to
  before (checkpoints included); nothing changes unless you opt in.

### Per-stage callbacks: `callbacks`

A stage can also declare extra Lightning callbacks that are active **only**
during that stage: stage-scoped instrumentation layered on the
always-propagated top-level `callbacks:`. Extending
`finetune_gn3large_add_calo.yaml` to watch the warm-up's learning rate
specifically:

```yaml
# extend finetune_gn3large_add_calo.yaml — a warm-up-only LR monitor
training_schedule:
  stages:
    calo_warmup:
      callbacks:
        - class_path: lightning.pytorch.callbacks.LearningRateMonitor
          init_args: {logging_interval: step}
```

- Top-level (global) `callbacks:` (`ModelCheckpoint`, the logger, progress bar)
  **persist for the whole fit** and keep their cross-stage state (best-checkpoint
  tracking, logging). They are **never re-instantiated** at a stage boundary.
- A stage's own `callbacks:` are **instantiated fresh when the stage begins** (fresh
  state each time), receive hooks only while their stage is active, and are torn down
  when the stage ends. The effective set in a stage is *the persistent globals plus
  that stage's freshly-instantiated callbacks*.
- Every declared stage callback is import/instantiation-checked at **fit start**, so a
  bad `class_path` fails before training, not three stages in.

### Per-stage LR scheduler: `lr_scheduler`

By default each stage runs a `OneCycleLR` envelope. A stage can instead choose
a different scheduler **class**: the classic fine-tuning shape is a cosine
warm-up followed by a plateau-driven finetune that drops the LR whenever
`val/loss` stalls:

```yaml
# extend finetune_gn3large_add_calo.yaml — cosine warm-up, plateau finetune
training_schedule:
  stages:
    calo_warmup:
      lrs: {initial: 1.0e-4}          # 'initial' = the optimizer's base LR
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.CosineAnnealingLR
        init_args: {T_max: 3}
    full_finetune:
      lrs: {initial: 1.0e-5}
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
        init_args: {mode: min, factor: 0.5, patience: 2}
        monitor: val/loss             # REQUIRED for a metric-driven scheduler
```

- The class is instantiated over the stage's **freshly-rebuilt optimizer** at
  the boundary. You never pass an `optimizer` in `init_args` (it is injected for
  you; a user-supplied one is a `ConfigError`).
- Optional Lightning scheduler-config keys: **`interval`** (`epoch`, the default
  for a custom scheduler, or `step`), **`frequency`**, and **`monitor`**. A
  *metric-driven* scheduler (`ReduceLROnPlateau`) **requires `monitor`**: a
  fail-fast error at fit start otherwise. Its LR reductions ride on the same
  rank-synced monitor as `early_stop`, so they are consistent under multi-GPU
  DDP.
- With `lr_scheduler:` the OneCycle-only `lrs:` keys (`max`/`end`/`pct_start`) no
  longer apply; keep only `initial` (base LR) and `weight_decay` in that stage's
  `lrs:` override. Setting a OneCycle-only key alongside `lr_scheduler:` is a
  `ConfigError`.
- A stage may declare **both** `lr_scheduler` (e.g. plateau) and `early_stop` on the
  same monitor: the scheduler lowers the LR when the metric plateaus while
  `early_stop` independently advances the stage once its patience is exhausted.
- A config with **no** `lr_scheduler:` anywhere is bitwise-identical to before (every
  stage keeps its OneCycleLR).

## Previewing a merge

`salt merge-config` takes the same arguments as `salt fit`. It instantiates no
trainer, no data module, and no checkpoint; it only merges the stacked configs
and reports the result. For each stage in the merged `training_schedule:`, it
writes one freeze-graph PNG showing which modules are frozen and which train,
alongside the fully-merged YAML itself:

```bash
salt merge-config \
  --config <base> \
  --config <overlay> \
  --merged.output logs/merge-preview/<name>.yaml
```

`--merged.output <path>` creates any missing parent directories
(`salt/merge_config.py:80`), writes the merged config YAML at that path, and
writes one `<stem>_stage{NN}_{name}.dot` file plus a sibling `.png` per
`training_schedule` stage, in execution order. Pass `--merged.plots false` to
write only the `.dot` files, for a machine with no Graphviz `dot` binary
installed (`merge_config.py:39-46`, `:84`).

The names inside `frozen:` / `trainable:` are `model.init_args.modules`
config keys, checked against that dict at parse time (`salt/schedule.py:567`,
`:581-584`); at runtime they map to `net.<name>.*` state-dict prefixes
(`schedule.py:700`). They are never read from the checkpoint, so a name
that is valid in your config stays valid whatever the checkpoint contains.

Run it before any real fit to confirm a config surgery did what you intended.
See [The model you start
from](tutorials/finetuning.md#the-model-you-start-from) for the module map
every worked example edits.

## Reading the evidence

Three things confirm a staged fine-tune is doing what you asked.

1. **The trainable-parameter count jumps at the stage boundary**, and the
   optimizer is rebuilt exactly once. A parameter hash on the frozen modules
   stays bitwise unchanged across the whole frozen stage, confirming the
   freeze is real and not just intended.
2. **Each stage runs its own `OneCycleLR` envelope.** Two distinct humps, not
   one: per-stage allocation, not one envelope stretched across the run.
3. **`val/loss` across the stage boundary** shows whether the new stage is
   actually helping, or whether an early stop should have fired sooner.

The worked numbers for one real run are in [Reading the
evidence](tutorials/finetuning.md#reading-the-evidence) inside worked example 1.

## What else can you change?

The four worked examples in the tutorial cover the common single changes, but
the mechanism generalises. Every kind of surgery reduces to the same three
moves: add a module, delete one with `null`, or rename one to force a clean
swap, combined differently:

| Change | Mechanism | Warm-start cost | Where it's shown |
|---|---|---|---|
| Add an input stream | new normaliser (or extend an existing one's `streams:`) + new `StreamEmbed` + restate `concat.streams` | cheapest: only the new stream's own modules are new | [worked example 1](tutorials/finetuning.md#worked-example-1-add-the-calo-stream) |
| Add a task head | new task module reading an existing embedding (`pooled.global`, or a per-token stream) + restate `run_tasks.tasks` | cheap: nothing upstream of the head changes | [worked example 2](tutorials/finetuning.md#worked-example-2-add-a-jet-task-bc-charge-head) |
| Drop a task head | `null` the module + restate `run_tasks.tasks` (dropping the entry) | free: one dropped module, nothing new | *(the mirror image of worked example 2 — not run for real on this page)* |
| Add/drop a variable in a collection | rename that collection's normaliser (or its `streams:` set) + rename every embed whose input or context width changes | expensive: every embed conditioned on the changed collection is rebuilt | [worked example 3](tutorials/finetuning.md#worked-example-3-add-variables-to-an-existing-collection) |
| Swap the normaliser class | rename-to-swap: delete the old normaliser, declare the new class under a new name | as expensive as the class allows (a dict-fed swap is painless; a learned one needs retraining) | [worked example 4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb) |
| Change a head's classes | rename the head (its `class_names`/`label_map` changed shape, so it is dropped + a same-named-but-different head is new) | one dropped, one new; everything else untouched | *(the same rule [worked example 4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb) applies to `xbb_classification`)* |
| Same heads, new sample | nothing — no overlay at all, just `--init_from` plus a `training_schedule` with a head warm-up | free: every module loads unchanged | *(one paragraph below)* |
| New jet definition / domain | combine several of the above at once | the everyday case | [worked example 4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb) |
| Combine several | apply each rule independently, then read the combined accounting | sum of the individual costs | [worked example 4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb) |

**Same heads, new sample: no overlay at all.** If a new dataset needs no
input or task changes whatsoever (same streams, same variables, same class
scheme), there is nothing to declare in `model.modules` at all: point
`data.modules.input_samples` at the new files, add a `training_schedule`
purely to re-align the heads gently (a short head warm-up, then a full
fine-tune), and every one of the 11 modules loads unchanged. This is the
degenerate case of the table above, a fine-tune that changes zero modules.

### The partial-coverage rule

Add and drop are *clean* only for a module that appears on **one** side
(config-only becomes new, checkpoint-only becomes dropped). The trap is a
**retained** module, present on both sides under the **same name**, whose
internal shape changed. That is a hard `ConfigError`, never a silent partial
load (the rule from [What `--init_from` accounts
for](#what-init_from-accounts-for)). It shows up in two forms, both fixed the
same way: **rename the module** so it drops the old weights and fresh-inits:

- **A changed stream's variable list.** Edit a kept stream's variables and its
  `StreamEmbed`'s input width changes, and the `Normaliser`'s buffers for that
  stream change width too. Rename the affected module ([worked example
  3](tutorials/finetuning.md#worked-example-3-add-variables-to-an-existing-collection)
  does this to four modules at once, because every embed reads the widened
  stream as context).
- **A changed `Normaliser` stream set.** Adding *or* removing a stream from
  `norm.init_args.streams` changes the `Normaliser`'s `means_<stream>` /
  `stds_<stream>` buffer keys: a coverage mismatch. **Rename the norm module**
  and update any `frozen`/`trainable` spec that names it. The module produces
  its `normed.<stream>` keys regardless of its own name, so nothing downstream
  needs rewiring.

A `Normaliser` rename is **painless** when the replacement also reads from a
dict file: its buffers are not learned. They are loaded from `norm_dict.yaml`
at materialise time, so a renamed `Normaliser` rebuilds from the norm dict and
nothing trained is lost. [Worked example
4](tutorials/finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb)'s
rename goes one step further and swaps the **class** too (`Normaliser` →
`MaskedInputNormaliser`), because the Xbb sample ships no norm dict at all:
still a clean rename-to-swap, just with a learned-online replacement instead
of a dict-fed one. (A *learned* module renamed the same way fresh-inits and
must be retrained; that is the intended "rename-to-swap" for a real
architecture change.)

Preview any surgery with [`salt merge-config`](#previewing-a-merge): the
merged YAML plus the per-stage freeze graphs are the fastest way to confirm
the new module is present, the old one is gone, and each stage freezes the
modules you expect, checked against [the module
map](tutorials/finetuning.md#the-model-you-start-from). See also [Multi-GPU
fine-tuning](#multi-gpu-fine-tuning) for how a changing frozen set interacts
with DDP, and [`lr_scheduler`](#per-stage-lr-scheduler-lr_scheduler) for
tuning the LR shape of any of these stages.

## Multi-GPU fine-tuning

Staged fine-tuning on multiple GPUs (DDP) has one subtlety, and salt handles it
**automatically**: there is nothing to configure.

When you freeze modules in one stage and unfreeze them in a later stage, the set
of parameters that produce gradients *changes across stages*. Plain DDP fixes its
reducer's managed-parameter set once, at wrap time, and cannot cope with that: it
either errors ("parameters that were not used in producing the loss") or silently
lets ranks desync. Salt detects a schedule that changes its frozen set across
stages under a DDP-family strategy and, for exactly that case:

- enables `find_unused_parameters=True` (so the reducer tolerates the changing
  active set), and
- switches to a **reducer-safe freeze**: frozen modules keep `requires_grad=True`
  (so every parameter stays in the reducer across flips) but are held immobile by
  being excluded from the optimizer, put in `eval()`, and having their gradients
  cleared each step. The result is bitwise-immobile frozen weights *and* ranks
  that stay in sync across the boundary.

On a **single GPU**, or for a schedule whose frozen set never changes, salt keeps
the simpler `requires_grad`-based freeze, which is optimal and bitwise-parity with
plain training. In every case the config is identical: the same
`training_schedule:` runs on one GPU or many. You do not opt in.

## Resume mid-schedule

Fine-tune runs are still ordinary runs, so a staged fit resumes with `--ckpt_path`
like any other (see [Training → resuming](training.md)). The one schedule-aware
detail: **resume is stage-correct**. Because the stage that owns any given epoch is
a pure function of the epoch counter and the schedule, restoring the epoch restores
the stage. If a job dies at epoch 7 (inside `full_finetune`), resuming from an
epoch-7 checkpoint continues in `full_finetune` with the whole network trainable;
it does *not* restart the warm-up. (When a stage boundary moved because
`early_stop` fired, the boundary is read back from the checkpoint's records, so the
resumed stage position is exact.)

```bash
salt fit \
  --config <run_dir>/config.yaml \
  --ckpt_path <run_dir>/ckpts/epoch=007-loss=....ckpt
```

Resume respects **epoch boundaries**: restore from an epoch-boundary checkpoint,
not a mid-epoch one, so the stage math and the LR envelope pick up cleanly.
(`--ckpt_path` and `--init_from` remain mutually exclusive: resume continues a
run, warm-start begins one.)
