# Event-level tagging with easyjet ROOT ntuples

Parts [1](mnist.md) and [2](mnist_cnn.md) trained on a fixed-length global
vector (flat MNIST pixels). In this tutorial the input is **variable-length**:
a per-event set of jets read straight from ROOT ntuples in the easyjet
`AnalysisMiniTree` format (the same tree layout FTAG analyses use), with no
salt code at all — every module is stock.

You will classify events into two classes from their jet content — real
all-hadronic ttbar versus real HH→4b events, not a constructed toy — by
embedding each jet, running a small transformer over the per-event jet set,
pooling it to one vector per event, and classifying. You will also compute a
**dumb-feature baseline** (jet multiplicity alone) alongside the network's
score — the tutorial's real lesson: a network beating a cheap feature is only
meaningful once you know what that cheap feature can already do on its own.

This tutorial trains on **real data**: a campaign-matched mc20e easyjet dump
of all-hadronic ttbar (label 0) and HH→4b (label 1) events, downloaded from a
public CERNBox share (~225 MB combined, one-time). These are two genuinely
distinct physical processes, so the classes differ for real physics reasons
rather than a hand-tuned toy gap.

The full copy-paste run (download the data, split it, train, evaluate)
takes a couple of minutes end to end — the one-time ~225 MB download and
the split step dominate; training itself is only about 20 seconds. A
slower network connection or a machine with fewer CPU cores will add time
to the download and to each epoch respectively.

## Prerequisites

Complete [part 1](mnist.md) first if you have not (the reader/model-module
contracts are assumed knowledge here). As in part 1:

```bash
git clone https://gitlab.cern.ch/aft/algorithms/salt.git
cd salt
pip install -e .
cd ..
mkdir event-classifier-tutorial && cd event-classifier-tutorial
export PYTHONPATH=$PWD
```

The container notes from part 1 apply unchanged. Reading ROOT files needs
salt's `root` extra: `pip install -e '.[root]'` from the salt clone (installs
`uproot`/`awkward`; `UprootReader` raises a clear `ImportError` naming this
extra if it is missing).

## 1. Download the data

Download the two ROOT ntuples (~225 MB combined) from the public CERNBox
share, and verify they landed intact before doing anything else with them:

```bash
mkdir -p data
curl -L --fail -o data/ttbar.root 'https://cernbox.cern.ch/s/hg97ylDXS5jZx8W/download?files=ttbar.root'
curl -L --fail -o data/hh4b.root 'https://cernbox.cern.ch/s/hg97ylDXS5jZx8W/download?files=hh4b.root'

ttbar_size=$(wc -c < data/ttbar.root)
hh4b_size=$(wc -c < data/hh4b.root)
if [ "$ttbar_size" != "120092329" ] || [ "$hh4b_size" != "104507713" ]; then
    echo "download size mismatch: ttbar=$ttbar_size hh4b=$hh4b_size" >&2
    exit 1
fi
echo "downloads OK"
```

```text
downloads OK
```

`curl -L --fail` follows the redirect but turns an HTTP error (or the wrong
URL shape silently returning a folder tarball or an HTML error page) into a
non-zero exit instead of saving it as if it were the ROOT file; the byte-count
check on top of that catches the case where the request "succeeds" but the
bytes are still wrong. A public, external-service download must not be
allowed to fail silently and carry on straight into training.

`ttbar.root` holds 102,030 events / 182 branches (all-hadronic ttbar, DSID
410471, mc20e). `hh4b.root` holds 79,434 events / 180 branches (HH→4b, DSID
603404, mc20e). Both come from the SAME nominal easyjet config
(`RunConfig-HH4b-Resolved.yaml`), so their branch schemas match.

Split each file into disjoint train/val/test ROOT files:

```python
"""Split the two downloaded ntuples into disjoint train/val/test ROOT files.

`eventNumber` is carried as provenance only -- never a model feature. The
per-sample DSID-derived bookkeeping branch and the ttbar-only config-artefact
branches are excluded: in a two-sample task, sample identity is exactly the
class label.

Entries are permuted once with a seeded shuffle (not contiguous ranges) to
guard against production-order artefacts in the merged ntuples; each slice is
then re-sorted ascending for read locality before writing.
"""

import numpy as np
import uproot

TREE = "AnalysisMiniTree"
JET_BRANCHES = [
    "recojet_antikt4PFlow_pt_NOSYS",
    "recojet_antikt4PFlow_eta",
    "recojet_antikt4PFlow_phi",
    "recojet_antikt4PFlow_m_NOSYS",
    "recojet_antikt4PFlow_GN2v01_pb",
    "recojet_antikt4PFlow_GN2v01_pc",
    "recojet_antikt4PFlow_GN2v01_pu",
]
BRANCHES = JET_BRANCHES + ["eventNumber"]

SOURCES = {"ttbar": "data/ttbar.root", "hh4b": "data/hh4b.root"}
N_TRAIN, N_VAL, N_TEST = 40000, 7000, 7000

for name, path in SOURCES.items():
    with uproot.open(f"{path}:{TREE}") as t:
        arrays = t.arrays(BRANCHES, library="ak")
    n = len(arrays[BRANCHES[0]])

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    slices = {
        "train": perm[:N_TRAIN],
        "val": perm[N_TRAIN : N_TRAIN + N_VAL],
        "test": perm[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST],
    }
    for stage, idx in slices.items():
        idx_sorted = np.sort(idx)
        cols = {b: arrays[b][idx_sorted] for b in BRANCHES}
        dest = f"data/{name}_{stage}.root"
        with uproot.recreate(dest) as f:
            f[TREE] = cols
        print(f"wrote {dest} ({len(idx_sorted)} events)")
```

```text
wrote data/ttbar_train.root (40000 events)
wrote data/ttbar_val.root (7000 events)
wrote data/ttbar_test.root (7000 events)
wrote data/hh4b_train.root (40000 events)
wrote data/hh4b_val.root (7000 events)
wrote data/hh4b_test.root (7000 events)
```

Six files land in `data/`: `{ttbar,hh4b}_{train,val,test}.root`, 40,000 train
/ 7,000 val / 7,000 test events per class — the same split sizes validated at
this scale in the study this tutorial is drawn from.

## 2. Write the config

The block below writes `config.yaml`:

```bash
cat > config.yaml <<'EOF'
name: EventTagger

data:
  batch_size: 1000
  num_workers: 0
  train_file: train
  val_file: val
  test_file: test
  modules:
    reader:
      class_path: salt.data.MultiSampleReader
      init_args:
        label_stream: event
        label_field: process
        seed: 42
        samples:
          - name: ttbar
            label: 0
            reader:
              class_path: salt.data.UprootReader
              init_args:
                tree: AnalysisMiniTree
                unroll: null
                groups: &groups
                  jets:
                    jagged: true
                    pad_max: 15
                    branches:
                      pt: recojet_antikt4PFlow_pt_NOSYS
                      eta: recojet_antikt4PFlow_eta
                      phi: recojet_antikt4PFlow_phi
                      m: recojet_antikt4PFlow_m_NOSYS
                      GN2v01_pb: recojet_antikt4PFlow_GN2v01_pb
                      GN2v01_pc: recojet_antikt4PFlow_GN2v01_pc
                      GN2v01_pu: recojet_antikt4PFlow_GN2v01_pu
                  event:
                    jagged: false
                    branches:
                      eventNumber: eventNumber
            sources:
              train: data/ttbar_train.root
              val: data/ttbar_val.root
              test: data/ttbar_test.root
          - name: hh4b
            label: 1
            reader:
              class_path: salt.data.UprootReader
              init_args:
                tree: AnalysisMiniTree
                unroll: null
                groups: *groups
            sources:
              train: data/hh4b_train.root
              val: data/hh4b_val.root
              test: data/hh4b_test.root
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt, eta, phi, m, GN2v01_pb, GN2v01_pc, GN2v01_pu]
    labels:
      class_path: salt.data.Labels
      init_args: {dtype_policy: int64-for-int}

model:
  class_path: salt.model.SaltModule
  init_args:
    lrs: {initial: 1.0e-5, max: 1.0e-3, end: 1.0e-5, pct_start: 0.1}
    optimizer: AdamW
    modules:
      norm:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args: {streams: [jets]}
      jet_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: jets
          out_dim: &embed_dim 32
          dense: {hidden_layers: [32], activation: ReLU}
      concat:
        class_path: salt.model.modules.Concat
        init_args: {streams: [jets]}
      encoder:
        class_path: salt.model.modules.TransformerEncoder
        init_args:
          dim: *embed_dim
          out_dim: 32
          num_layers: 2
          attention: {num_heads: 4, attn_type: torch-math}
          dense: {activation: ReLU, gated: false}
      pool:
        class_path: salt.model.modules.GlobalAttentionPooling
        init_args: {input: encoded.seq, out: pooled.global}
      event_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: event
          input: pooled.global
          label: process
          class_names: [ttbar, hh4b]
          dense: {hidden_layers: [32, 16], activation: ReLU}
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [event_classification], modes: [test]}

trainer:
  max_epochs: 6
  accelerator: cpu
  precision: 32-true
  logger: false
  default_root_dir: run
EOF
```

### `data:` — MultiSampleReader over two UprootReaders

`MultiSampleReader` wraps N labelled sub-readers (here two `UprootReader`s,
one per class, `ttbar` and `hh4b`, `unroll: null` so each row is an event)
and interleaves them proportionally to their size, injecting a per-event
integer label as `raw.event.process` (`label_stream`/`label_field`). Each
sample owns its own **per-stage sourcing** — the `sources:` block under each
sample, not the top-level `train_file`/`val_file`/`test_file` (those are
placeholder strings the datamodule only checks are non-empty; the real
per-stage paths live on each sample).

Both samples share one `groups:` block via a YAML anchor (`&groups`/`*groups`)
— the config's way of saying "same schema, different files". A real two-era
dataset (different production tags) would instead give each sample its own
`groups:` with different branch names on the right-hand side of `branches:`
— the reader contract does not care, since field names (`pt`, `eta`, ...) are
resolved to on-disk branch names per sample, independently. Here it happens to
also be literally true: `ttbar.root` and `hh4b.root` were dumped from the same
easyjet config, so the schemas genuinely match.

`jets` is `jagged: true` with `pad_max: 15` — the leading 15 jets per event
(events with fewer are padded, events with more are truncated), padded/masked.
`event` is `jagged: false` — one row per event, carrying only `eventNumber`
(provenance, never a feature) plus the reader-injected `process` label.

### `model:` — embed, encode, pool, classify

- **`norm`** (`MaskedInputNormaliser`) — as in [part 1](mnist.md#model), a
  self-normalising input layer; no precomputed `norm_dict` needed. Unlike
  part 1's global stream, `jets` is a **padded sequence** — the normaliser
  accumulates statistics from valid (non-padded) jets only.
- **`jet_embed`** (`StreamEmbed`) — per-jet dense embedding,
  `normed.jets [B, T, 7] → embed.jets [B, T, 32]`.
- **`concat`** (`Concat`) — with one stream this is a pass-through; it exists
  because `TransformerEncoder` reads from the `encoded.*` merged namespace,
  not `embed.*` directly (the concat step is what a multi-stream config uses
  to combine several embedded streams before the shared encoder).
- **`encoder`** (`TransformerEncoder`) — 2 layers of self-attention over the
  (up to 15) jets in each event, `torch-math` attention (CPU-safe; no
  flash-attn dependency).
- **`pool`** (`GlobalAttentionPooling`) — reduces the per-jet sequence
  `encoded.seq [B, T, 32]` to one vector per event, `pooled.global [B, 32]`
  (a learned attention pool, not a plain mean — the model decides which jets
  matter for the event-level decision).
- **`event_classification`** (`ClassificationTaskModule`) — a 2-class head on
  `pooled.global`. `label: process` is the demand that makes `Labels` serve
  the injected `labels.event.process` field from `MultiSampleReader`.

### `outputs:`

Only `RunTaskOutput` — no `InputCopyWriter`. Input-copying needs an
h5py-openable structured source the sink can re-open; `MultiSampleReader` (and
the whole uproot-reader family) has none, and `H5OutputSink` raises a clear
`ConfigError` if you try. That is not a problem here: the classification
task's TEST-mode output already emits both the class probabilities *and* the
`target_event_classification` truth column (same mechanism as
[part 1](mnist.md#outputs)), which is everything the ROC/AUC step below needs.

!!! warning "`salt graph validate` does not yet support this reader family"

    Unlike [part 1](mnist.md#5-validate-the-graph-before-training)'s custom
    `IdxReader`, `UprootReader`/`MultiSampleReader` build their schema by
    probing the source file's branches (`Reader.prepare()`), and `graph
    validate` calls this on the raw config-instantiated reader — before any
    file has been bound via `with_source` — which raises `reader 'unnamed'
    has no source file`. This is a genuine, currently-open gap for the whole
    uproot-reader family (tracked; not specific to this config). Skip
    straight to `salt fit` below — the plan validates the SAME graph at fit
    time, just with a real error location if something is wrong (a missing
    branch, a shape mismatch), rather than a static preflight.

## 3. Train

Unlike [part 1](mnist.md), every `class_path` above is a stock salt module —
there is no user reader to make importable this time, so (unlike part 1)
`export PYTHONPATH=$PWD` is not required. Just fit:

```bash
salt fit --config config.yaml
```

```text
Epoch 5: 100%|██████████| 80/80 [00:03<00:00, 21.98it/s, train/loss=0.237, val/loss=0.258]
`Trainer.fit` stopped: `max_epochs=6` reached.
salt fit artifacts: config.yaml in run
salt fit artifacts: checkpoints in run/ckpts
```

`run/` now contains the resolved `config.yaml`, per-epoch checkpoints in
`run/ckpts/`, and the graph plots. Six epochs over 80,000 train events (80
batches/epoch at batch size 1000) took about 20 seconds total on the
multi-core CPU this was measured on — training time scales with the number
of CPU cores available, so expect it to vary on your own machine.

## 4. Evaluate

```bash
salt test --config run/config.yaml --ckpt_path run/ckpts/epoch=005*.ckpt
```

```text
Wrote eval file run/ckpts/epoch=005-loss=0.25804__test_ttbar_test.h5
```

(the exact filename varies with the loss value and which sample the writer
names it after — the read-back below globs for `*__test_*.h5` rather than
hardcoding it, same as [part 1](mnist.md#7-evaluate)).

## 5. Read the eval H5, and compare against a dumb baseline

The eval file has one structured dataset per stream — here `event` (there is
no `jets` dataset: input-copying is unsupported for this reader family, see
above). Columns follow the same `{run_name}_p{class}` / `target_{task}`
convention introduced in [part 1](mnist.md#7-evaluate): `name: EventTagger`
+ `class_names: [ttbar, hh4b]` gives `EventTagger_pttbar` /
`EventTagger_phh4b`, and the task name `event_classification` gives
`target_event_classification`. `import hdf5plugin` (unused directly, hence
`# noqa: F401`) registers the HDF5 compression filter salt's `H5OutputSink`
writes with — without it, `h5py.File(...)` raises an "unknown filter"
error on read. Read the network's score and compute its AUC, then compute the
same metric for two **dumb baselines** read straight from the ROOT test files:
jet multiplicity, and the leading jet's pT alone.

```python
import glob

import awkward as ak
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import uproot


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic — no extra dependency needed."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


with h5py.File(glob.glob("run/ckpts/*__test_*.h5")[0]) as f:
    table = f["event"][:]
p_hh4b, target = table["EventTagger_phh4b"], table["target_event_classification"]
print(f"network AUC:  {auc(target, p_hh4b):.4f}")

y, njets, lead_pt = [], [], []
for label, name in [(0, "ttbar"), (1, "hh4b")]:
    with uproot.open(f"data/{name}_test.root:AnalysisMiniTree") as t:
        pt = t["recojet_antikt4PFlow_pt_NOSYS"].array(library="ak")
    njets.append(np.asarray(ak.num(pt, axis=1)))
    lead_pt.append(np.asarray(ak.max(pt, axis=1)))
    y.append(np.full(len(njets[-1]), label))
y = np.concatenate(y)
print(f"njets-only baseline AUC: {auc(y, np.concatenate(njets)):.4f}")
print(f"leading-jet-pT baseline AUC: {auc(y, np.concatenate(lead_pt)):.4f}")
```

```text
network AUC:  0.9589
njets-only baseline AUC: 0.3141
leading-jet-pT baseline AUC: 0.6764
```

With real, physically-distinct classes the margin is much larger than a
constructed toy pair would show: the network lands close to **0.96**. The
leading-jet-pT baseline reaches **0.68** — real, modest discriminating
power from a single
number. The njets-only baseline reports **0.31**: below the 0.5 midpoint, not
because jet count carries no information but because it points the other
way — all-hadronic ttbar events average *more* jets than events passing the
resolved HH→4b selection, so raw jet multiplicity anti-correlates with the
hh4b label under this `auc()` function's fixed orientation (0.31 is exactly
as informative as 0.69 would be, just the other way round). Either way,
neither cheap feature comes close to the network's combined use of all seven
jet features — including the GN2v01 b-tagging scores — across up to 15 jets
per event. ttbar and HH→4b differ far more than a constructed toy pair would,
and that real physics difference is what the network is actually finding.
The lesson is not "the network wins by a lot" for its own sake; it is that
you now *know* how much it wins by, against the cheapest things it could have
secretly been learning instead.

## What you just proved

- **The reader seam serves ROOT as readily as HDF5 or IDX.** `UprootReader`
  is a stock salt module — nothing here is a custom reader, unlike
  [part 1](mnist.md).
- **`MultiSampleReader` composes readers, not just streams.** Two structurally
  identical sub-readers over two disjoint file sets become one interleaved,
  proportionally-stratified dataset with an injected label — no per-sample
  code, only per-sample config.
- **A variable-length stream is still just config.** `jets` differs from part
  1's fixed `mnist` stream only in `jagged: true` — embedding, encoding, and
  pooling are the same `class_path` idiom throughout.
- **Beating a dumb baseline is the actual bar.** The njets-only and
  leading-pT baselines are not strawmen: they are the cheapest things a
  network could be secretly learning. Reporting them alongside the network's
  AUC is what turns "0.9x AUC" into a *meaningful* claim rather than a
  plausible-looking number. With real, physically-distinct classes the margin
  here is much larger than a constructed toy pair would show — that does not
  make the baseline check less important, it is what makes the network's
  score credible: you now know exactly how much of it the cheapest
  alternative explanation cannot account for.

## Next: deploy it

This config trains and evaluates, but ships no ONNX surface. To export the
model and run it outside salt — with a worked post-hoc inference script and the
semantics you have to match to keep the scores correct — see
[Deploy in easyjet](../deployment/easyjet.md).
