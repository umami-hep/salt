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
declare it as one by renaming.

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

## Preview the merge before you train — `salt merge-config`

A fine-tune stacks a base `config.yaml`, an overlay, and CLI overrides. Before you
spend GPU time, it is worth *seeing* exactly what those layers merged into — and
which modules each stage freezes. `salt merge-config` takes **the same arguments as
`salt fit`** and writes two things without instantiating a trainer, reading data,
or loading a checkpoint:

```bash
salt merge-config \
  --config base_config.yaml \
  --config finetune_gn3large.yaml \
  --init_from /path/to/pretrained.ckpt \
  --merged.output out/merged.yaml
```

1. **`out/merged.yaml`** — the fully-merged config, produced through the same salt
   config surface as `--print_config` (the same deep-merge, the same `base2.yaml`
   defaults, the same schedule relocation). This is the single source of truth for
   what will actually run: every default made explicit, every overlay applied.
2. **One graph per stage** — `out/merged_stage00_head_warmup.png`,
   `out/merged_stage01_full_finetune.png`, … (numbered by execution order). Each is
   the model graph with that stage's **frozen** modules greyed out and badged, and a
   caption naming the stage and its frozen set. This is the fastest way to confirm a
   `trainable:`/`frozen:` spec froze what you intended.

Pass `--merged.plots false` to write only the merged YAML and the `.dot` graph
sources (skips rasterisation, so no Graphviz `dot` binary is needed). The graphs
render the FIT-mode plan; only the freeze overlay differs between stages.

## Worked example A — same-heads fine-tune of GN3Large

The goal: take the converted GN3Large tagger and adapt it to a new dataset while
keeping its existing task heads (flavour, track origin, vertexing, track type,
pT regression). The strategy is two-stage:

1. **`head_warmup`** (5 epochs) — freeze the entire backbone and all heads except
   the flavour classifier; let *only* `jets_classification` adjust to the new data
   at a moderate LR. This re-aligns the head without disturbing a good backbone.
2. **`full_finetune`** (remaining 10 epochs) — unfreeze everything and fine-tune
   the whole network at a much gentler LR.

The overlay config (`finetune_gn3large.yaml`):

```yaml
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]   # complement (everything else) is frozen + eval()
      lrs:
        max: 1.0e-4                       # per-stage lrs deep-merges over the base config's lrs
    full_finetune:
      frozen: []                          # unfreeze everything; epochs omitted = remaining epochs
      lrs:
        initial: 1.0e-7
        max: 1.0e-5

trainer:
  max_epochs: 15                          # 5 warm-up + 10 full fine-tune
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
  --config finetune_gn3large.yaml \
  --init_from <gn3large_converted.ckpt>
```

The concrete instance behind the numbers below (the study's demo, 60 k training
jets, one A100):

```bash
salt fit \
  --config /data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt/experiments/23_gn3large_v1_to_v2_conversion/outputs/converted/config_v2.yaml \
  --config finetune_gn3large.yaml \
  --init_from /data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt/experiments/23_gn3large_v1_to_v2_conversion/outputs/converted/converted.ckpt
```

At startup, the warm-start accounting confirms the load. For this checkpoint all
11 parameter-bearing modules line up with the config, so the log reads
**11 loaded, 0 new, 0 dropped** — a clean same-architecture warm start.

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
  frozen warm-up, a gentler `max` LR, or add an early-stopping callback that halts
  when `val/loss` stops improving. The warm-up phase alone (backbone frozen) is
  much harder to overfit and is often most of the benefit.

The staged schedule did its job perfectly — the *sizing* was wrong, and the
evidence (best at epoch 6, monotonic climb after) tells you exactly that.

## Worked example B — module surgery: add a new head

Fine-tuning does not have to mean re-training the *same* heads. A common need is
to **add a brand-new task head** to a pretrained backbone and train only it — a
new classifier reusing GN3Large's learned jet representation.

This is where `--init_from`'s per-module accounting earns its keep. You add the
new module to the config; the warm start finds no checkpoint weights for it and
reports it as **new** (fresh init), while every inherited module loads normally.

The overlay does two things. First, **module surgery** — deep-merge a new head
into the model's `modules:` dict (the other modules are inherited untouched):

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
          dense:
            hidden_layers: [128, 64, 32]
            activation: SiLU
          label: large_r_flavour_label   # the new label your target dataset must provide
          class_names: [hbb, hcc, top, qcd]
          weight_source: null
```

Second, **a schedule that references the new head by name** — train only it while
the backbone and all original heads stay frozen, then optionally a gentle full
pass:

```yaml
# 2. Warm-start the new head with the backbone frozen.
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [large_r_jet_classification]   # backbone + 5 original heads frozen + eval()
      lrs:
        max: 1.0e-4
    full_finetune:
      frozen: []                                 # unfreeze everything; remaining epochs
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
  --config finetune_gn3large_new_head.yaml \
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
    `weight_source: {from_class_dict: <path>}`). The names above are
    placeholders — swap in your real label handle and categories.

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
it does *not* restart the warm-up.

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
  schedule actually happened.
- **Adding a head is just config.** Deep-merge a module into `modules:`, name it in
  a stage's `trainable:`, and `--init_from` reports it as *new* — no code.
- **The last checkpoint is not the best checkpoint.** The demo's `val/loss` bottomed
  at epoch 6 and climbed to a worse-than-start value by epoch 14. Selecting on
  validation loss — and sizing the schedule to the data — is the difference between
  a fine-tune that helps and one that quietly overfits.
