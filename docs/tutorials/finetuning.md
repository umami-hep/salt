# Fine-tuning a pretrained tagger

The earlier tutorials trained models from scratch. This one starts from a
**pretrained checkpoint** and adapts it — the everyday situation when a new MC
campaign lands, a specialised topology needs its own tagger, or you want to bolt
a new task head onto an existing backbone without paying for a full retrain.

Salt gives you two independent tools for this, and the tutorial's first job is to
keep them straight:

- **`--init_from`** — a *weights-only warm start*. A fresh run (epoch 0, fresh
  optimizer) whose weights are seeded from a checkpoint, module by module. This
  is fine-tuning.
- **`training_schedule:`** — a *staged* training plan: freeze part of the network
  for some epochs, then unfreeze, each stage with its own learning rate. This is
  how you fine-tune *gently* — warm up a head before disturbing a good backbone.

The worked examples use the converted **GN3Large** flavour tagger (86 M
parameters) as the pretrained model. You will not be able to run them verbatim
without that checkpoint and its training data, so the commands show
**placeholders** with the exact paths used to produce the numbers here given as
the concrete instance. The concepts, configs, and the honest lesson at the end
transfer to any pretrained salt model.

## Prerequisites

You need salt installed ([Setup](../setup.md)) and should be comfortable with the
config structure from [part 1](mnist.md) (the `model.modules` dict, task heads,
`class_path` wiring). The container notes from [part 1](mnist.md#prerequisites)
apply unchanged.

Fine-tuning needs two things a from-scratch run does not:

1. **A pretrained checkpoint** — a `.ckpt` saved by a previous `salt fit`, plus
   the resolved `config.yaml` that produced it (so the architecture matches).
2. **A target dataset** — what you are adapting *to*. It must expose the labels
   the model's task heads demand.

## Warm start vs. resume — `--init_from` vs `--ckpt_path`

These two flags both take a checkpoint and both feed weights into a fit, so they
are easy to confuse. They do opposite things and are **mutually exclusive** —
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
carry on as if nothing happened. Use **`--init_from`** when you have a *finished*,
good model and want a *new* training run — on new data, or with the network
surgically modified — that begins from those weights instead of random init.

### What `--init_from` accounts for

A warm start does not blindly `load_state_dict`. It classifies every module (by
its `net.<name>.*` state-dict prefix) into one of three buckets and logs the
result:

- **loaded** — the module exists in both the checkpoint and your config, with an
  identical key set, shapes, and dtypes. Its weights are copied in.
- **new** — the module is in your config but *not* in the checkpoint (e.g. a head
  you just added). Left at fresh random init and materialised normally.
- **dropped** — the module is in the checkpoint but *not* in your config. Skipped
  and logged.

There is a deliberate trap here: a module that is present in *both* but only
**partially** covered (its internal architecture changed — different layer
widths, a missing sub-key) is a hard `ConfigError`, not a silent partial load.
The fix is to **rename** the module: under a new name it has no checkpoint
counterpart, so the old weights simply **drop** and the renamed module is **new**
(fresh init) — a clean swap through the same three buckets. The rule:
*warm-starting is per-module all-or-nothing*. Changing a module's shape is a swap;
declare it as one by renaming. (The [Changing the inputs](#changing-the-inputs)
section puts this rule to work.)

## The `training_schedule:` schema

`training_schedule:` is a **top-level** config key — a peer of `trainer:`,
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

The per-stage keys — and there are exactly these, anything else is a config typo
rejected fail-loud:

- **`epochs`** — a positive integer. Every stage **except the last** must give an
  explicit `epochs`; the final stage may omit it to take the remainder of
  `trainer.max_epochs`. The explicit epochs must sum to `≤ max_epochs`, and an
  omitted final stage needs at least 1 epoch left over. With `max_epochs: 15` and
  `head_warmup.epochs: 5`, `full_finetune` runs the remaining **10** epochs.
- **`frozen`** / **`trainable`** — a list of `model.modules` names. Give **at most
  one**: `frozen` freezes exactly those modules; `trainable` freezes their
  *complement* (everything else). Setting both is a `ConfigError`. Neither freezes
  nothing (the whole network trains). The names must be real `model.modules` keys
  or it fails loud. A **frozen** module is excluded from the optimizer *and* put in
  `eval()` mode (so dropout and running statistics stop updating) — its weights are
  held bitwise-immobile for the stage. Unfreezing restores it to `train()` and
  re-adds it to the optimizer.
- **`lrs`** — a mapping that **deep-merges over** the base config's `lrs:`. Only
  list the keys that differ per stage; the rest fall through. (Above, `head_warmup`
  keeps the base `initial`/`end`/`pct_start` and overrides only `max`.)
- **`optimizer`** — a per-stage optimizer name, if a stage needs a different one.
  Usually omitted so every stage shares the base `optimizer:`.
- **`order`** — pins execution position. Omit it and stages run in declaration
  order (the normal case).
- **`early_stop`** — an optional early-stopping criterion that ends the stage before
  its `epochs` cap (see [below](#per-stage-early-stopping-early_stop)).
- **`callbacks`** — an optional list of extra Lightning callbacks active only during
  this stage (see [below](#per-stage-callbacks-callbacks)).
- **`lr_scheduler`** — an optional LR-scheduler *class* for this stage, replacing the
  default OneCycleLR (see [below](#per-stage-lr-scheduler-lr_scheduler)).

### How the pieces layer

Everything that **shapes the schedule** — `frozen`/`trainable`, `epochs`,
`optimizer`, `lrs`, `early_stop` — is a **stage key**, not a callback, and that is
deliberate. Those concerns need first-class integration with the stage machinery:
they drive the optimizer rebuild at each boundary, the per-stage LR envelope, the
checkpoint boundary records that make a resume stage-correct, and the
DDP freeze flip. A Lightning callback cannot reach into any of that.

`early_stop` is the sharpest example. It is **not** a Lightning `EarlyStopping`
callback, because that callback can only do one thing: kill the whole fit. A
per-stage criterion has to be able to end *this stage* and hand off to the next —
so it is a stage key the schedule owns, wired into the boundary logic.

A stage's own **`callbacks:`** are for the opposite kind of thing:
stage-scoped *instrumentation and side-effects* — a monitor, a diagnostic, a
per-stage checkpoint policy — that observe a stage without steering it.

By default every stage runs a `OneCycleLR` envelope; `lrs:` tunes its parameters
(`max`, `initial`, `end`, `pct_start`) per stage. A stage that needs a *different*
scheduler class entirely — a cosine warm-up, a plateau-driven finetune — declares
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
    training) is a `ConfigError` — staged training must know how many epochs it is
    dividing up.

## Worked example A — same-heads fine-tune of GN3Large

The goal: take the converted GN3Large tagger and adapt it to a new dataset while
keeping its existing task heads (flavour, track origin, vertexing, track type,
pT regression). The strategy is two-stage:

1. **`head_warmup`** (5 epochs) — freeze the entire backbone and all heads except
   the flavour classifier; let *only* `jets_classification` adjust to the new data
   at a moderate LR. This re-aligns the head without disturbing a good backbone.
2. **`full_finetune`** (remaining 10 epochs) — unfreeze everything and fine-tune
   the whole network at a much gentler LR.

The overlay is shipped as **`salt/configs/finetune_gn3large.yaml`** — a template
you stack on the saved run config (paths and any data block are yours to fill in):

```yaml
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]  # complement (backbone + other heads) frozen + eval()
      lrs:
        max: 1.0e-4                      # per-stage lrs deep-merges over the base config's lrs
    full_finetune:
      frozen: []                         # unfreeze everything; epochs omitted = remaining epochs
      lrs:
        initial: 1.0e-7
        max: 1.0e-5

trainer:
  max_epochs: 15                         # 5 warm-up + 10 full fine-tune
```

Point `data:` at the sample you are adapting to. The simplest way is to override
the file paths on the CLI (shown in the command below); alternatively add a `data:`
block to the overlay, e.g.:

```yaml
# in finetune_gn3large.yaml, alongside training_schedule:
data:
  train_file: /path/to/new_campaign/train.h5
  val_file: /path/to/new_campaign/val.h5
```

Everything else — the reader, features, labels, and the full model — is inherited
from the saved GN3Large config.

The command stacks the saved training config, the fine-tune overlay, and the
warm-start checkpoint:

```bash
salt fit \
  --config <gn3large_config_v2.yaml> \
  --config salt/configs/finetune_gn3large.yaml \
  --init_from <gn3large_converted.ckpt>
```

The concrete instance behind the numbers below (the study's demo, 60 k training
jets, one A100):

```bash
salt fit \
  --config /data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt/experiments/23_gn3large_v1_to_v2_conversion/outputs/converted/config_v2.yaml \
  --config salt/configs/finetune_gn3large.yaml \
  --init_from /data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt/experiments/23_gn3large_v1_to_v2_conversion/outputs/converted/converted.ckpt \
  --data.train_file <new_campaign/train.h5> \
  --data.val_file   <new_campaign/val.h5>
```

At startup, the warm-start accounting confirms the load. For this checkpoint all
11 parameter-bearing modules line up with the config, so the log reads
**11 loaded, 0 new, 0 dropped** — a clean same-architecture warm start.

## Preview the merge before you train — `salt merge-config`

A fine-tune stacks a base `config.yaml`, an overlay, and CLI overrides. Before you
spend GPU time on the run above, it is worth *seeing* exactly what those layers
merged into — and which modules each stage freezes. `salt merge-config` takes **the
same arguments as `salt fit`** and writes two things without instantiating a
trainer, reading data, or loading a checkpoint:

```bash
salt merge-config \
  --config <gn3large_config_v2.yaml> \
  --config salt/configs/finetune_gn3large.yaml \
  --init_from <gn3large_converted.ckpt> \
  --merged.output out/merged.yaml
```

1. **`out/merged.yaml`** — the fully-merged config, produced through the same salt
   config surface as `--print_config` (the same deep-merge, the same `base2.yaml`
   defaults, the same schedule relocation). This is the single source of truth for
   what will actually run: every default made explicit, every overlay applied. When a
   config declares no `training_schedule:`, the merged YAML makes the effective one
   explicit — a single `fit` stage (everything trainable) under a
   `# materialized by merge-config` marker, so plain-training runs read the same way
   as staged ones.
2. **One graph per stage** — `out/merged_stage00_head_warmup.png`,
   `out/merged_stage01_full_finetune.png`, … (numbered by execution order). Each is
   the model graph with that stage's **frozen** modules greyed out and badged, and a
   caption naming the stage, its frozen set, and its early-stop criterion and
   stage-callback count. This is the fastest way to confirm a
   `trainable:`/`frozen:` spec froze what you intended.

Pass `--merged.plots false` to write only the merged YAML and the `.dot` graph
sources (skips rasterisation, so no Graphviz `dot` binary is needed). The graphs
render the FIT-mode plan; only the freeze overlay differs between stages.

## Reading the evidence: LR envelopes, the stage boundary, the param jump

Three things are worth watching to confirm a staged fine-tune is doing what you
asked. All numbers below come from the demo run above.

!!! note "Provenance of these numbers"

    The figures in this section were produced by the study's demo experiment
    (exp 08), pinned to salt at commit `2419254`. The current branch head adds a
    reducer-safe DDP freeze fix (`8f89278`, see [Multi-GPU](#multi-gpu-fine-tuning))
    that does **not** change any single-GPU number here — the run was not
    re-executed at head.

**1. The trainable-parameter jump at the boundary.** In `head_warmup` only the
flavour head trains — 8 parameter tensors. At epoch 5 the schedule crosses into
`full_finetune` and the whole network unfreezes — 155 tensors. The optimizer is
rebuilt **exactly once**, at that boundary:

| Epoch range | Stage | Trainable tensors |
|---|---|---|
| 0–4 | `head_warmup` | 8 (flavour head only) |
| 5–14 | `full_finetune` | 155 (whole network) |

A parameter-hash check confirms the freeze is real, not just intended: across the
warm-up stage the backbone's hash is **bitwise unchanged** (`e99730b6…` →
`e99730b6…`), while the flavour head moves. The frozen modules are genuinely
immobile.

**2. Per-stage LR envelopes.** Each stage gets its *own* `OneCycleLR` envelope
spanning only that stage's steps — not one envelope stretched across the whole
run. The warm-up LR rises toward its `max: 1.0e-4` and anneals down; then the
full-finetune stage starts a *fresh* envelope rising to its gentler `max: 1.0e-5`
and anneals again. You see two distinct humps, not one — the sign that per-stage
allocation worked.

**3. The validation loss.** With the two envelopes and the freeze in place, the
warm-up drives `val/loss` from 3.65 down to 3.54, and the moment the backbone
unfreezes it drops sharply to **2.45** within one epoch. That is the fine-tune
working.

## The honest lesson: this run overfit

Here is what actually happened after epoch 6:

| Epoch | Stage | `val/loss` |
|---|---|---|
| 4 | warm-up ends | 3.54 |
| 5 | full fine-tune begins | 2.45 |
| **6** | **best** | **2.45** |
| 9 | | 2.86 |
| 12 | | 3.82 |
| 14 | final | **3.97** |

The best model appears at **epoch 6**, and from there `val/loss` climbs steadily
to **3.97 by epoch 14 — worse than where the fine-tune started** (epoch 0's 3.65),
let alone the best. An 86 M-parameter network fine-tuned on only 60 k jets for 10
full-network epochs memorised the training set. **The final-epoch checkpoint is
the worst one.**

This is the point of including the run, not a blemish to hide. Two lessons follow:

- **Never ship the last checkpoint blindly.** Select on validation loss.
  Salt's `Checkpoint` callback with `save_top_k: -1` keeps every epoch, so the
  best one (here `epoch=006…`) is on disk — use *that* for `salt test` and
  export, not `epoch=014…`. If you only keep a few, set `monitor_loss: val/loss`
  and `mode: min` so the ones you keep are the good ones.
- **Size the schedule to the data.** 10 full-finetune epochs was far too many for
  60 k jets. For a small target set, prefer fewer full-finetune epochs, a longer
  frozen warm-up, a gentler `max` LR, or — the cleanest fix — add a per-stage
  **`early_stop`** so the stage halts on its own when `val/loss` stops improving.
  The warm-up phase alone (backbone frozen) is much harder to overfit and is
  often most of the benefit.

The staged schedule did its job perfectly — the *sizing* was wrong, and the
evidence (best at epoch 6, monotonic climb after) tells you exactly that. The next
two sections extend this same example to fix it.

### Per-stage early stopping — `early_stop`

The overfit above is exactly what `early_stop` prevents. Give the `full_finetune`
stage a criterion and it stops **before** its 10-epoch cap once `val/loss` stops
improving — the run ends near epoch 6 instead of grinding on to 14:

```yaml
# extend finetune_gn3large.yaml — stop full_finetune when val/loss stalls
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]
      lrs: {max: 1.0e-4}
    full_finetune:
      frozen: []
      lrs: {initial: 1.0e-7, max: 1.0e-5}
      early_stop:
        monitor: val/loss               # required — a trainer.callback_metrics key
        mode: min                       # min (default) or max
        patience: 3                     # validation checks without improvement (default 3)
        min_delta: 0.0                  # minimum improvement to reset patience (default 0.0)
        check_finite: true              # stop on a non-finite monitor value (default true)
```

`epochs` stays the hard cap; the stage ends at whichever comes first. **On the
final stage** (as here) early-stopping **ends the fit**; on a **non-final stage**
the schedule instead **advances to the next stage**.

- The check runs at **validation-epoch end**; `patience` counts validation checks,
  not raw training epochs. Validation must be enabled (a stage with `early_stop`
  under a trainer with validation disabled is a fail-loud `ConfigError`), and the
  `monitor` key must exist in `trainer.callback_metrics` (e.g. `val/loss`,
  `val/jets_classification_loss`) or it fails loud at the first check.
- An early-stopped stage simply **truncates its LR envelope** mid-curve; the next
  stage rebuilds its own OneCycle cleanly. Boundaries become data-dependent, so they
  are **recorded in the checkpoint** — a resume reconstructs the exact stage position
  and patience counters (a mid-stage resume continues patience identically).
- Under multi-GPU DDP the decision is **rank-synchronised** (all ranks transition at
  the same step), so the freeze flip and optimizer rebuild never desync.
- A config with no `early_stop` on any stage behaves — and checkpoints —
  bitwise-identically to before; nothing changes unless you opt in.

### Per-stage callbacks — `callbacks`

A stage can also declare extra Lightning callbacks that are active **only** during
that stage — stage-scoped instrumentation layered on the always-propagated
top-level `callbacks:`. Here we watch the warm-up's learning rate specifically:

```yaml
# extend finetune_gn3large.yaml — a warm-up-only LR monitor
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]
      lrs: {max: 1.0e-4}
      callbacks:
        - class_path: lightning.pytorch.callbacks.LearningRateMonitor
          init_args: {logging_interval: step}
    full_finetune:
      frozen: []
      lrs: {initial: 1.0e-7, max: 1.0e-5}
```

- Top-level (global) `callbacks:` — `ModelCheckpoint`, the logger, progress bar —
  **persist for the whole fit** and keep their cross-stage state (best-checkpoint
  tracking, logging). They are **never re-instantiated** at a stage boundary.
- A stage's own `callbacks:` are **instantiated fresh when the stage begins** (fresh
  state each time), receive hooks only while their stage is active, and are torn down
  when the stage ends. The effective set in a stage is *the persistent globals plus
  that stage's freshly-instantiated callbacks*.
- Every declared stage callback is import/instantiation-checked at **fit start**, so a
  bad `class_path` fails before training, not three stages in.

### Per-stage LR scheduler — `lr_scheduler`

By default each stage runs a `OneCycleLR` envelope. A stage can instead choose a
different scheduler **class** — the classic fine-tuning shape is a cosine warm-up
followed by a plateau-driven finetune that drops the LR whenever `val/loss` stalls:

```yaml
# extend finetune_gn3large.yaml — cosine warm-up, plateau finetune
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]
      lrs: {initial: 1.0e-4}          # 'initial' = the optimizer's base LR
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.CosineAnnealingLR
        init_args: {T_max: 5}
    full_finetune:
      frozen: []
      lrs: {initial: 1.0e-5}
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
        init_args: {mode: min, factor: 0.5, patience: 2}
        monitor: val/loss             # REQUIRED for a metric-driven scheduler
```

- The class is instantiated over the stage's **freshly-rebuilt optimizer** at the
  boundary — you never pass an `optimizer` in `init_args` (it is injected for you; a
  user-supplied one is a `ConfigError`).
- Optional Lightning scheduler-config keys: **`interval`** (`epoch` — the default for
  a custom scheduler — or `step`), **`frequency`**, and **`monitor`**. A *metric-driven*
  scheduler (`ReduceLROnPlateau`) **requires `monitor`** — a fail-fast error at fit
  start otherwise. Its LR reductions ride on the same rank-synced monitor as
  `early_stop`, so they are consistent under multi-GPU DDP.
- With `lr_scheduler:` the OneCycle-only `lrs:` keys (`max`/`end`/`pct_start`) no
  longer apply; keep only `initial` (base LR) and `weight_decay` in that stage's
  `lrs:` override. Setting a OneCycle-only key alongside `lr_scheduler:` is a
  `ConfigError`.
- A stage may declare **both** `lr_scheduler` (e.g. plateau) and `early_stop` on the
  same monitor: the scheduler lowers the LR when the metric plateaus while
  `early_stop` independently advances the stage once its patience is exhausted.
- A config with **no** `lr_scheduler:` anywhere is bitwise-identical to before (every
  stage keeps its OneCycleLR).

## Worked example B — module surgery: add a new head

Fine-tuning does not have to mean re-training the *same* heads. A common need is
to **add a brand-new task head** to a pretrained backbone and train only it — a
new classifier reusing GN3Large's learned jet representation.

This is where `--init_from`'s per-module accounting earns its keep. You add the
new module to the config; the warm start finds no checkpoint weights for it and
reports it as **new** (fresh init), while every inherited module loads normally.

The overlay is shipped as **`salt/configs/finetune_gn3large_new_head.yaml`**. It
does two things. First, **module surgery** — deep-merge a new head into the model's
`modules:` dict (the other modules are inherited untouched):

```yaml
# 1. Add the new head (deep-merged into GN3Large's modules).
model:
  init_args:
    modules:
      large_r_jet_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: jets
          input: pooled.global          # reuse the same pooled jet embedding the flavour head uses
          label: large_r_flavour_label  # the new label your target dataset must provide
          class_names: [hbb, hcc, top, qcd]
          dense: {hidden_layers: [128, 64, 32], activation: SiLU}
          weight_source: null
```

Second, **a schedule that references the new head by name** — train only it while
the backbone and all original heads stay frozen, then a gentle full pass:

```yaml
# 2. Warm-start the new head with the backbone frozen.
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [large_r_jet_classification]  # backbone + original heads frozen + eval()
      lrs:
        max: 1.0e-4
    full_finetune:
      frozen: []                               # unfreeze everything; remaining epochs
      lrs:
        initial: 1.0e-7
        max: 1.0e-5

trainer:
  max_epochs: 15
```

Same command shape as example A:

```bash
salt fit \
  --config <gn3large_config_v2.yaml> \
  --config salt/configs/finetune_gn3large_new_head.yaml \
  --init_from <gn3large_converted.ckpt>
```

The startup accounting now reads differently: the inherited modules are **loaded**
from the checkpoint, and `large_r_jet_classification` is **new** — fresh random
init, no checkpoint counterpart. That single word in the log is your confirmation
the surgery landed as intended: the backbone was inherited, the head was born
fresh.

!!! note "The label must exist in your data"

    `label: large_r_flavour_label` is a demand on the dataset: the `Labels`
    module must be able to serve it. Point `data:` at a sample that carries that
    truth field (and, if you class-weight the head, set
    `weight_source: {from_class_dict: <path>}`). The names in the shipped config
    are placeholders — swap in your real label handle and categories.

## Changing the inputs

Surgery is not limited to task heads. Fine-tuning can also change what the model
*reads* — add or drop an input stream. The same per-module accounting makes it
safe: a new stream's embedding is **new**, a removed stream's is **dropped**, and
the shared encoder — which works on tokens, not streams — stays **loaded**.

The running example is a GN3-family two-stream body (`tracks` + `flows`) whose
model wiring names each stream in three places: one `StreamEmbed` per stream, the
`Concat` that fuses their tokens, and the `Normaliser` that scales each stream's
raw inputs. (See `salt/configs/GN3V00.yaml` for the full config.)

### Adding an input stream

Say you want to add an `electrons` stream. Four coordinated edits:

1. **A new `StreamEmbed` module** (model side), embedding the stream to the shared
   token width:

    ```yaml
    model:
      init_args:
        modules:
          electron_embed:
            class_path: salt.model.modules.StreamEmbed
            init_args:
              stream: electrons
              context: [normed.jets]
              out_dim: 512     # MUST equal the other streams' embed width — a Concat constraint
              dense: {hidden_layers: [512], activation: SiLU}
    ```

2. **Add the stream to the token concatenation** — `concat.init_args.streams`:

    ```yaml
    concat:
      init_args: {streams: [tracks, flows, electrons]}
    ```

    `Concat` is **parameter-free**, so this rewiring loads nothing and drops
    nothing on the warm start.

3. **Add the stream to the input normaliser** — `norm.init_args.streams`. This
   grows the `Normaliser`'s per-stream buffer set, which trips the partial-coverage
   rule and so needs a **rename** (see [below](#the-partial-coverage-rule)).

4. **Data side** — the stream must actually arrive. Add its group to the `reader`
   (`groups: {electrons: {...}}`), its variable list to `features`
   (`variables: {electrons: [...]}`), and its per-variable `{mean, std}` to the
   `norm_dict.yaml` you pass at fit time.

The pretrained **encoder loads untouched**: it operates on the concatenated token
sequence with weights shared across tokens, so a longer sequence needs no new
parameters — its `net.encoder.*` keys match the checkpoint exactly and it is
reported **loaded**. Pair the surgery with a `training_schedule` stage that trains
only the new embed against a frozen backbone, then optionally unfreezes:

```yaml
training_schedule:
  stages:
    embed_warmup:
      epochs: 5
      trainable: [electron_embed]   # everything else — the loaded backbone — frozen
      lrs: {max: 1.0e-4}
    full_finetune:
      frozen: []
```

### Removing an input stream

To drop a stream, delete its embed module and rewire the two stream lists:

```yaml
# overlay: drop the flows stream
model:
  init_args:
    modules:
      flow_embed: null                        # delete the module (deep-merge deletion)
      concat:
        init_args: {streams: [tracks]}        # remove flows from the token concat
      norm:
        init_args: {streams: [jets, tracks]}  # remove flows from the normaliser
```

(equivalently `--model.modules.flow_embed=null` on the CLI). On the warm start
`flow_embed` is in the checkpoint but not the config → **dropped** (its weights
skipped and logged). The graph recompiles; the FIT plan-hash changes, which is
**logged for information, not enforced**, on a warm start. As with adding a
stream, editing `norm.init_args.streams` trips the partial-coverage rule — rename
the norm module.

### The partial-coverage rule

Add and drop are *clean* only for a module that appears on **one** side
(config-only → new, checkpoint-only → dropped). The trap is a **retained** module
— present on both sides under the **same name** — whose internal shape changed.
That is a hard `ConfigError`, never a silent partial load (the rule from
[What `--init_from` accounts for](#what-init_from-accounts-for)). Input surgery
hits it in two ways, both fixed the same way — **rename the module** so it drops
the old weights and fresh-inits:

- **A changed stream's variable list.** Edit a kept stream's variables and its
  `StreamEmbed`'s input width changes — and the `Normaliser`'s buffers for that
  stream change width too. Rename the affected module.
- **A changed `Normaliser` stream set.** Adding *or* removing a stream from
  `norm.init_args.streams` changes the `Normaliser`'s `means_<stream>` /
  `stds_<stream>` buffer keys — a coverage mismatch. **Rename the norm module**
  (e.g. `norm` → `norm_ft`) and update any `frozen`/`trainable` spec that names it.
  The module produces its `normed.<stream>` keys regardless of its own name, so
  nothing downstream needs rewiring.

The `Normaliser` rename is **painless**: its buffers are not learned — they are
loaded from `norm_dict.yaml` at materialise time. Renamed, it is **new**, so it
simply rebuilds from the norm dict; nothing trained is lost. (A *learned* module
renamed the same way genuinely fresh-inits and must be retrained — that is the
intended "rename-to-swap" for a real architecture change.)

Preview either surgery with [`salt merge-config`](#preview-the-merge-before-you-train-salt-merge-config):
the merged YAML plus the per-stage freeze graphs are the fastest way to confirm
the new embed is present, the old one is gone, and each stage freezes the modules
you expect.

## Multi-GPU fine-tuning

Staged fine-tuning on multiple GPUs (DDP) has one subtlety, and salt handles it
**automatically** — there is nothing to configure.

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
plain training. In every case the config is identical — the same
`training_schedule:` runs on one GPU or many. You do not opt in.

## Resume mid-schedule

Fine-tune runs are still ordinary runs, so a staged fit resumes with `--ckpt_path`
like any other (see [Training → resuming](../training.md)). The one schedule-aware
detail: **resume is stage-correct**. Because the stage that owns any given epoch is
a pure function of the epoch counter and the schedule, restoring the epoch restores
the stage. If a job dies at epoch 7 (inside `full_finetune`), resuming from an
epoch-7 checkpoint continues in `full_finetune` with the whole network trainable —
it does *not* restart the warm-up. (When a stage boundary moved because
`early_stop` fired, the boundary is read back from the checkpoint's records, so the
resumed stage position is exact.)

```bash
salt fit \
  --config <run_dir>/config.yaml \
  --ckpt_path <run_dir>/ckpts/epoch=007-loss=....ckpt
```

Resume respects **epoch boundaries** — restore from an epoch-boundary checkpoint,
not a mid-epoch one, so the stage math and the LR envelope pick up cleanly.
(`--ckpt_path` and `--init_from` remain mutually exclusive: resume continues a
run, warm-start begins one.)

## What you just proved

- **Warm start and resume are different operations.** `--init_from` seeds a fresh
  run from pretrained weights with per-module accounting; `--ckpt_path` continues
  an interrupted one with full trainer state. They are mutually exclusive.
- **A fine-tune is an overlay, not a rewrite.** A short `training_schedule:` config
  stacked on a saved run config plus `--init_from` is the whole recipe — freeze,
  warm up, unfreeze, at per-stage learning rates.
- **The evidence is legible.** The trainable-tensor jump (8 → 155), the bitwise-frozen
  backbone, and the two per-stage LR envelopes each confirm a specific piece of the
  schedule actually happened — and `salt merge-config` shows all of it before you train.
- **Surgery is just config.** Deep-merge a module into `modules:` (a new head, a new
  input stream), name it in a stage's `trainable:`, and `--init_from` reports it as
  *new* — no code. Drop one and it reports *dropped*; change a retained module's
  shape and rename it to make the swap explicit.
- **The last checkpoint is not the best checkpoint.** The demo's `val/loss` bottomed
  at epoch 6 and climbed to a worse-than-start value by epoch 14. Selecting on
  validation loss — sizing the schedule to the data, or letting a per-stage
  `early_stop` halt it — is the difference between a fine-tune that helps and one
  that quietly overfits.
