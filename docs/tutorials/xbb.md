# Boosted Xbb tagging

This tutorial trains a **boosted large-R jet tagger** — the Xbb family — on a
public 100k-jet sample. It is the large-R counterpart to
[part 4](gn2_opendata.md), which tags small-R jets: same framework, same
commands, different physics and a different label schema.

It is also the smallest *real physics* tutorial in the series. The whole
dataset is 690 MB and the config trains on CPU in minutes, so unlike part 4
you can run this one start to finish while reading.

## What "Xbb" means

At high transverse momentum a Higgs boson decaying to two b-quarks is
**boosted**: the two b-jets are no longer resolved as separate small-R jets
but merge into a single large-radius (R = 1.0) jet. Tagging that jet as
"H→bb" rather than as a QCD jet, a hadronically decaying top, or a W is the
Xbb problem. The signal lives in two places:

- **Substructure** — a two-prong jet has a different energy-flow pattern from
  a one-prong QCD jet or a three-prong top. The N-subjettiness ratio `Tau21`
  and the energy-correlation ratios `C2` / `D2` capture this.
- **Flavour** — a H→bb jet contains two b-hadron decays, so its constituent
  tracks carry large impact parameters. This is the same information GN2 uses,
  read from the tracks associated to the large-R jet.

A modern Xbb tagger uses both, which is what the model below does.

## Prerequisites

Complete [part 1](mnist.md) first if you have not. As there:

```bash
git clone https://gitlab.cern.ch/aft/algorithms/salt.git
cd salt
pip install -e .
cd ..
mkdir xbb-tutorial && cd xbb-tutorial
export PYTHONPATH=$PWD
```

Everything below assumes you stay in `xbb-tutorial/` — the config and the
plotting script both use paths relative to it. `PYTHONPATH=$PWD` lets salt
import any custom module you write here by `class_path`; this tutorial does not
need one, but the later exercises and [part 2](mnist_cnn.md) do.

## 1. Get the data

The sample is a public CERNBox share — no CERN account, no authentication:

```bash
export XBB_DATA=$PWD/data
mkdir -p $XBB_DATA
BASE=https://cernbox.cern.ch/remote.php/dav/public-files/t1WnJ8UMUgEycjp
curl -L -o $XBB_DATA/pp_output_train_small.h5      $BASE/pp_output_train_small.h5
curl -L -o $XBB_DATA/pp_output_val-full_0_100k.h5  $BASE/pp_output_val-full_0_100k.h5
curl -L -o $XBB_DATA/pp_output_test-full_0_100k.h5 $BASE/pp_output_test-full_0_100k.h5
```

Three files, 100,000 large-R jets each, 690 MB total. They are
[UPP](https://github.com/umami-hep/umami-preprocessing)-preprocessed, so they
are already resampled and shuffled and can be fed to salt directly.

??? info "If you are on lxplus or another CERN machine"

    The same files are staged on EOS, which is faster from inside CERN:

    ```
    /eos/user/n/npond/salt-data/finetuning/xbb-finetune/
    ```

    Recommended copy — this is a directory copy, so it lands as
    `xbb-finetune/` containing all three files:

    ```bash
    cd <where you keep data> && xrdcp -r root://eosuser.cern.ch//eos/user/n/npond/salt-data/finetuning/xbb-finetune/ .
    ```

    From a batch node — or from inside a container, where EOS FUSE is not
    visible — use xrootd rather than the POSIX path, as above.

    That EOS directory is not world-readable; the CERNBox link above is the
    one to share with people outside the group. If you are following the
    [fine-tuning tutorial](finetuning.md), its Prerequisites already fetch
    this folder for you.

!!! warning "The three files do not have identical schemas"

    `train` and `val` carry 37 jet fields and 26 track fields. The `test` file
    is a fuller dump — 106 jet fields, 31 track fields, and two extra groups.
    Everything this tutorial uses is present in all three, but do not assume a
    variable you find in `test` exists in `train`.

## 2. The label schema (read this before you plot anything)

Two different truth labels live in the `jets` group, and confusing them is the
single easiest way to produce a plausible-looking but wrong ROC curve.

- **`R10TruthLabel_R22v1`** is the ATLAS large-R truth label: the
  `LargeRJetTruthLabel::TypeEnum` from
  [`LargeRJetLabelEnum.h`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/AnalysisCommon/ParticleJetTools/ParticleJetTools/LargeRJetLabelEnum.h).
  It is fine-grained and its integers are *not* contiguous.
- **`flavour_label`** is the **training** label: a contiguous 0–5 index that
  UPP derived by merging the enum into the classes this tagger targets. This is
  what salt trains against.

Cross-tabulating the two over all 100,000 training jets gives the mapping —
this is measured from the file, not assumed:

| `flavour_label` | `R10TruthLabel_R22v1` | enum name(s) | jets | fraction |
|---|---|---|---|---|
| 0 | 11 | `Hbb` | 9,006 | 9.0% |
| 1 | 12 | `Hcc` | 8,397 | 8.4% |
| 2 | 16 | `HtautauHad` | 2,403 | 2.4% |
| 3 | 1, 6, 7 | `tqqb` + `Wqq_From_t` + `other_From_t` | 7,754 | 7.8% |
| 4 | 10 | `qcd` | 71,268 | 71.3% |
| 5 | 2 | `Wqq` | 1,172 | 1.2% |

Three things to take from that table:

1. **`flavour_label` 0 is Hbb, not QCD.** The class order is Higgs-first. If
   you assume the small-R convention (`0 = light`) every curve you draw will be
   wrong but will still look like a curve.
2. **Class 3 (top) is a merge of three enum values.** A top jet is labelled
   `tqqb` when the whole top is contained in the large-R jet, and
   `Wqq_From_t` / `other_From_t` when only part of it is. All three are "top"
   for tagging purposes. So the mapping is many-to-one — you cannot invert
   `flavour_label` back to `R10TruthLabel_R22v1`.
3. **The sample is extremely imbalanced**: 71% QCD, 1.2% Wqq. That is realistic
   (QCD is the background you actually have to reject), but it dominates
   training and it is why the config below is explicit about not silently
   reweighting.

!!! tip "Do this check on any new sample"

    Reproduce the table in three lines before trusting a label:

    ```python
    import h5py, numpy as np
    with h5py.File("data/pp_output_train_small.h5") as f:
        jets = f["jets"][:]
    print(np.unique(np.stack([jets["flavour_label"], jets["R10TruthLabel_R22v1"]]),
                    axis=1, return_counts=True))
    ```

### The rest of the schema

| Group | Shape | What it is |
|---|---|---|
| `jets` | (100k,) | large-R kinematics (`pt`, `eta`, `mass`), substructure (`Tau21`, `C2`, `D2`, `N2`, `M2`, `L2`), the two labels, ghost b/c-hadron counts, and reference scores from the shipped `GN2Xv01` / `GN3XPV01` / `GN2XTauV00` taggers |
| `tracks` | (100k, 100) | up to 100 associated tracks, 26 fields: impact parameters and their significances, pixel/SCT hit counts, plus `valid` |
| `flow` | (100k, 100) | particle-flow constituents (not used here) |
| `truth_hadrons` | (100k, 5) | truth b/c-hadrons, MaskFormer-style targets (not used here) |

Jet selection is already applied: pT 200–2100 GeV, mass 50–300 GeV, |eta| < 2.

!!! info "The reference scores are a free baseline"

    `GN2Xv01_phbb` and friends are the *existing* production taggers' outputs,
    stored per jet. You can plot them on the same axes as your own model for a
    like-for-like comparison — see the closing exercise.

## 3. The config

The sample ships **no `norm_dict.yaml`**. Rather than a blocker, this is a good
excuse to use `MaskedInputNormaliser`, which learns input means and variances
online from the valid (non-padded) objects instead of reading precomputed
constants from a file. Save this as `xbb.yaml`:

```yaml
name: XbbTutorial

data:
  batch_size: 500
  num_workers: 0
  modules:
    input_samples:
      class_path: salt.data.InputSamples
      init_args:
        files:
          train: data/pp_output_train_small.h5
          val: data/pp_output_val-full_0_100k.h5
          test: data/pp_output_test-full_0_100k.h5
        num: {train: -1, val: -1, test: -1}
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          jets: {global_object: true}
          tracks: {global_object: false, pad_max: 40}
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt, eta, mass, Tau21, C2, D2]
          tracks:
            [
              d0,
              z0SinTheta,
              dphi,
              deta,
              qOverP,
              lifetimeSignedD0Significance,
              lifetimeSignedZ0SinThetaSignificance,
              phiUncertainty,
              thetaUncertainty,
              qOverPUncertainty,
              numberOfPixelHits,
              numberOfSCTHits,
              numberOfInnermostPixelLayerHits,
              numberOfNextToInnermostPixelLayerHits,
              numberOfInnermostPixelLayerSharedHits,
              numberOfInnermostPixelLayerSplitHits,
              numberOfPixelSharedHits,
              numberOfPixelSplitHits,
              numberOfSCTSharedHits,
            ]
    labels:
      class_path: salt.data.Labels
      init_args: {dtype_policy: int64-for-int}

model:
  class_path: salt.model.SaltModule
  init_args:
    lrs: {initial: 1.0e-7, max: 5.0e-4, end: 1.0e-5, pct_start: 0.01, weight_decay: 1.0e-5}
    optimizer: AdamW
    modules:
      norm:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args:
          streams: [jets, tracks]
          global_object: jets
      track_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: tracks
          context: [normed.jets]
          out_dim: 128
          dense: {hidden_layers: [128], activation: ReLU}
      concat:
        class_path: salt.model.modules.Concat
        init_args: {streams: [tracks]}
      encoder:
        class_path: salt.model.modules.TransformerEncoder
        init_args:
          dim: 128
          out_dim: 128
          num_layers: 3
          attention: {num_heads: 4, attn_type: torch-math}
          dense: {activation: ReLU, gated: false}
      split:
        class_path: salt.model.modules.Split
        init_args: {streams: [tracks]}
      pool:
        class_path: salt.model.modules.GlobalAttentionPooling
        init_args: {input: encoded.seq, out: pooled.global}
      jets_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: jets
          input: pooled.global
          label: flavour_label
          class_names: [hbb, hcc, htautauhad, top, qcd, wqq]
          weight_source: null
          dense: {hidden_layers: [128, 64, 32], activation: ReLU}
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  inputs_copy:
    class_path: salt.outputs.InputCopyWriter
    init_args:
      streams: [jets]
      variables:
        jets: [pt, eta, mass, R10TruthLabel_R22v1, flavour_label]
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args:
      tasks: [jets_classification]
  onnx_export:
    class_path: salt.outputs.OnnxExportSink
    init_args:
      model_name: XbbTutorial # Athena name: no '_'/'-'
      inputs:
        - {port: inputs.jets, name: jet_features}
        - {port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks}

trainer:
  max_epochs: 20
  precision: 32-true
```

### The decisions in that file

**`class_names: [hbb, hcc, htautauhad, top, qcd, wqq]`** — this list is
positional. Entry *i* is what `flavour_label == i` means, so it is the table
from section 2 transcribed in order. Get it wrong and the model trains fine
while every output column is mislabelled. It also fixes the eval column names:
`XbbTutorial_phbb`, `XbbTutorial_phcc`, `XbbTutorial_phtautauhad`,
`XbbTutorial_ptop`, `XbbTutorial_pqcd`, `XbbTutorial_pwqq`.

**`MaskedInputNormaliser`** instead of `Normaliser` — no `norm_dict:` argument,
because it learns the statistics online. It always applies frozen running
buffers, and updates them from masked batch moments only while training, so
evaluation and ONNX export stay a pure affine transform.

**`weight_source: null`** — no class reweighting. With 71% QCD you will
certainly want to revisit this, and that is the point: the tutorial trains on
the raw imbalance so the imbalance is visible in the results rather than hidden
behind a choice you did not make. See the exercises.

**`pad_max: 40`** on tracks — the files store up to 100 tracks per jet; sequences
shorter than 40 are padded up to 40 and longer ones truncated down to 40; 40 is
the usual working point and keeps the attention cost down.

**The training knobs** are sized for a laptop, not tuned: `batch_size: 500` and
`max_epochs: 20` get 100k jets to converge in minutes on CPU;
`num_workers: 0` loads data in the training process, which is fastest at this
scale and avoids worker-pool memory on a shared machine (raise it for a real
dataset); `precision: 32-true` keeps full float32 because there is no GPU
speed-up to buy here. All four are worth revisiting before you take this
config anywhere near a real training set.

**`inputs_copy`** carries `R10TruthLabel_R22v1` and `flavour_label` through to
the eval file so the plotting script can select classes without reopening the
input.

**No auxiliary track tasks.** Part 4's GN2 config trains track-origin and
vertexing heads alongside the jet classifier. This one does not, to keep the
tutorial small — adding them is an exercise below.

## 4. Train

```bash
salt fit --config xbb.yaml
```

Recommended before the real run — this checks the whole pipeline in seconds on
CPU, and catches a bad path or a mistyped variable name before you wait:

```bash
salt fit --config xbb.yaml \
  --trainer.accelerator cpu --trainer.max_epochs 1 \
  --trainer.limit_train_batches 4 --trainer.limit_val_batches 2
```

### Finding your run

Each run creates its own directory, stamped with the start time:

```bash
ls logs/
```

```text
XbbTutorial_20260727-T140233/
```

Inside it are `config.yaml` (the fully merged config — this is what you pass to
`salt test`, not `xbb.yaml`) and `ckpts/`, which holds one checkpoint per
epoch, named with the epoch number and the validation loss:

```bash
ls logs/XbbTutorial_*/ckpts/
```

```text
epoch=017-loss=0.71032.ckpt  epoch=018-loss=0.70915.ckpt  epoch=019-loss=0.71284.ckpt
```

The **best** checkpoint is the one with the lowest loss in its filename. Pick
it out rather than eyeballing:

```bash
ls logs/XbbTutorial_*/ckpts/epoch=*.ckpt | sort -t= -k3 -g | head -1
```

Then glob for that epoch number rather than typing the loss value, which is
what every command below does:

```bash
ls logs/XbbTutorial_*/ckpts/epoch=018*.ckpt
```

## 5. Evaluate

```bash
salt test --config logs/XbbTutorial_*/config.yaml \
  --ckpt_path logs/XbbTutorial_*/ckpts/epoch=018*.ckpt
```

Substitute your own best epoch number. If you have more than one run in
`logs/`, spell out the timestamp instead of globbing it — `salt test` takes
exactly one config.

This writes one HDF5 file next to the checkpoint, named after it:
`epoch=018-loss=0.70915__test_pp_output_test-full_0_100k.h5`. Its `jets` group
contains the six probability columns named above, the copied input variables,
and `target_jets_classification` (the truth label the model was scored
against). See [Outputs](../outputs.md) if you want to know exactly where those
names come from or how to change them.

## 6. Performance plots

With six classes there is no single "the" discriminant. The simplest choice —
and the one used here — is to take the signal-class probability directly:

```
D_Hbb = XbbTutorial_phbb
D_Hcc = XbbTutorial_phcc
```

That is a deliberate simplification. A production Xbb tagger uses a
log-likelihood-ratio discriminant that lets you tune the trade-off between QCD
and top rejection, e.g.

```
D_Hbb = log( p_hbb / (f_top * p_top + (1 - f_top) * p_qcd) )
```

with `f_top` chosen for the analysis. The raw probability is `f_top` implicitly
fixed, which is fine for a first look and suboptimal for a real measurement.

Save this as `make_plots.py`:

It needs [puma](https://github.com/umami-hep/puma), the FTAG plotting package.
It is a salt dependency (`puma-hep==0.5.3`), so `pip install -e .` in the
prerequisites already gave it to you; install it directly if you are plotting
from a different environment.

Run this **after** `salt test` has finished — it globs for the eval file rather
than hardcoding the run timestamp, so it runs unedited from the tutorial
directory. An `IndexError` on the first line means either no eval file exists
yet or you are not in `xbb-tutorial/`; more than one run in `logs/` and it
picks an arbitrary one, so spell the path out if that happens:

```python
import glob

import h5py
import numpy as np
from puma import Histogram, HistogramPlot, Roc, RocPlot
from puma.metrics import calc_rej

EVAL = glob.glob("logs/XbbTutorial_*/ckpts/*__test_*.h5")[0]
MODEL = "XbbTutorial"  # must match `name:` in xbb.yaml — it prefixes every column
NUM_JETS = 100_000

# flavour_label, verified against R10TruthLabel_R22v1 (see section 2)
HBB, HCC, HTAUTAUHAD, TOP, QCD, WQQ = range(6)

with h5py.File(EVAL) as f:
    jets = f["jets"][:NUM_JETS]

flav = jets["flavour_label"]
is_hbb, is_hcc = flav == HBB, flav == HCC
is_qcd, is_top = flav == QCD, flav == TOP

# the signal-class probability as the discriminant (see the caveat above)
disc = {"Hbb": jets[f"{MODEL}_phbb"], "Hcc": jets[f"{MODEL}_phcc"]}
is_sig = {"Hbb": is_hbb, "Hcc": is_hcc}

sig_eff = np.linspace(0.4, 1, 100)

for name, d in disc.items():
    # --- discriminant distributions, one figure per signal ---
    plot = HistogramPlot(
        n_ratio_panels=0,
        ylabel="Normalised number of jets",
        xlabel=f"{name} discriminant",
        logy=True,
        bins=np.linspace(0, 1, 50),
        figsize=(6.5, 4.5),
        atlas_second_tag=r"$\sqrt{s}=13$ TeV, large-$R$ jets",
    )
    plot.add(Histogram(d[is_qcd], label="QCD jets"))
    plot.add(Histogram(d[is_top], label="Top jets"))
    plot.add(Histogram(d[is_sig[name]], label=f"{name} jets"))
    plot.draw()
    plot.savefig(f"disc_{name}.png", transparent=False)

    # --- ROC: signal efficiency vs QCD and top rejection ---
    rej_qcd = calc_rej(d[is_sig[name]], d[is_qcd], sig_eff)
    rej_top = calc_rej(d[is_sig[name]], d[is_top], sig_eff)
    roc = RocPlot(
        n_ratio_panels=0,
        ylabel="Background rejection",
        xlabel=f"{name} efficiency",
        atlas_second_tag=r"$\sqrt{s}=13$ TeV, large-$R$ jets",
        figsize=(6.5, 6),
        y_scale=1.4,
    )
    roc.add_roc(Roc(sig_eff, rej_qcd, n_test=int(is_qcd.sum()),
                    rej_class="qcd", signal_class=name, label="QCD rejection"))
    roc.add_roc(Roc(sig_eff, rej_top, n_test=int(is_top.sum()),
                    rej_class="top", signal_class=name, label="Top rejection"))
    roc.draw()
    roc.savefig(f"roc_{name}.png", transparent=False)
```

```bash
python make_plots.py
```

Four figures: `disc_Hbb.png`, `disc_Hcc.png`, `roc_Hbb.png`, `roc_Hcc.png`.

!!! warning "Select on `flavour_label`, not `R10TruthLabel_R22v1`"

    The script above selects classes with `flavour_label`, which is what the
    model was trained on. If you select with `R10TruthLabel_R22v1` instead you
    must remember that top is *three* enum values (1, 6, 7) — selecting only
    `== 1` silently drops the partially-contained top jets and flatters your
    top rejection.

## Exercises

??? question "1. Fix the class imbalance"

    71% of the sample is QCD and 1.2% is Wqq. Retrain with class weights and
    compare the ROCs.

    ??? success "Hint"

        `ClassificationTaskModule` takes a `weight_source`. The quickest route
        is an explicit loss with a per-class `weight` list, in `class_names`
        order:

        ```yaml
        jets_classification:
          init_args:
            loss:
              class_path: torch.nn.CrossEntropyLoss
              init_args:
                weight: [1.0, 1.1, 3.7, 1.2, 0.13, 7.7]
        ```

        Those are the inverse class frequencies from the table in section 2,
        normalised to a mean of 1. The alternative,
        `weight_source: {from_class_dict: <path>}`, reads the same numbers from
        a class-dict YAML — see [Configuration](../configuration.md) for that
        file's schema.

    ??? success "What to expect"

        Rejection of the rare classes (Wqq, Htautauhad) should improve while
        QCD rejection at fixed Hbb efficiency gets slightly worse — you are
        moving where the model spends its capacity, not creating information.

??? question "2. Are the tracks doing anything?"

    Train a substructure-only model and compare. Remove the `tracks` entry from
    `features.variables`, delete `track_embed` / `concat` / `encoder` / `split`
    / `pool`, and feed `normed.jets` straight into the classification head.

    ??? success "Hint"

        This is the same ablation pattern as part 4's exercise. Keep the run
        `name:` distinct so the eval columns do not collide, then plot both
        models on the same `RocPlot` by adding two `Roc` objects with
        `reference=True` on the baseline.

    ??? success "What to expect"

        Substructure alone separates two-prong from one-prong reasonably well,
        but cannot distinguish Hbb from Hcc — that difference is entirely in the
        track impact parameters. Expect the Hcc ROC to collapse much further
        than the Hbb one.

??? question "3. Beat the shipped taggers"

    The `jets` group carries `GN2Xv01_phbb`, `GN3XPV01_phbb` and
    `GN2XTauV00_phbb` — the production taggers' own scores on these jets. Add
    them to `inputs_copy` and plot them alongside your model.

    ??? success "Hint"

        Those columns pass through `salt test` untouched if you list them in
        the `InputCopyWriter`'s `variables.jets`, so no second evaluation is
        needed. Note `GN3XPV01` splits QCD into four sub-classes
        (`pqcdbb`/`pqcdbx`/`pqcdcx`/`pqcdll`) — sum them before comparing
        against your single `pqcd`.

    ??? success "What to expect"

        You will lose, comfortably. Those models are trained on tens of
        millions of jets with more inputs and auxiliary tasks; this one saw
        100k. The interesting question is how much of the gap closes when you
        add the auxiliary tasks and the flow constituents.

??? question "4. Add auxiliary tasks"

    Part 4's GN2 config trains track-origin and vertexing heads alongside the
    jet classifier. Port them here — the `tracks` group has
    `ftagTruthOriginLabel` and `ftagTruthVertexIndex`.

    ??? success "Hint"

        Copy the `track_origin` and `track_vertexing` module blocks from
        `salt/configs/gn2v2-opendata.yaml` (in the salt clone you made in the
        prerequisites), and add the task names to the
        `RunTaskOutput`'s `tasks:` list — otherwise the planner will refuse to
        run with a dead-predictions error, since the new heads would produce
        predictions no sink consumes.

## Where to go next

- **Start from a pretrained model instead of scratch.** This sample is the
  worked domain-shift target in [Fine-tuning a pretrained tagger](finetuning.md):
  the resolved-jet GN3Large model is adapted to these boosted jets by module
  surgery — swapping the input streams and the classification head, then
  warm-starting the shared encoder. That is the realistic way to train an Xbb
  tagger with only 100k jets. To warm-start this task from the pretrained
  GN3Large backbone instead of training from scratch, see
  [Fine-tuning → worked example 4](finetuning.md#worked-example-4-backbone-transfer-to-boosted-xbb).
- **Understand the eval file.** [Outputs](../outputs.md) explains where every
  column name comes from and how to add or remove them.
- **Ship the model to Athena.** [Export to ONNX](../deployment/export.md), and part 4's
  [export section](gn2_opendata.md#7-export-to-onnx).
