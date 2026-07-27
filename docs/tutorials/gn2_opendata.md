# GN2 jet tagging on open data

This tutorial trains a v2 GN2 tagger — tracks + jet features, multi-task
(flavour classification + two auxiliary track tasks), the production-scale
architecture — on the public ATLAS open dataset.

It supersedes the v1-era open-data tutorial that salt shipped until 2025 (the
one given as a workshop at
[FTAG 2023](https://indico.cern.ch/event/1311519/) and repeated in later
software tutorials). Same data, same physics; the `salt fit` / `salt test`
commands and the config anatomy differ because a v2 model is built from
composable modules instead of one monolithic config block. That page has been
removed rather than left to rot — it taught retired config surfaces
(`train_file`, `scale_dict`, `--data.move_files_temp`) that now fail on
contact. Its two genuinely useful exercises, the auxiliary-task ablation and
the ONNX walkthrough, live on as steps 6 and 7 below; `git log` has the rest.

Unlike parts [1](mnist.md)–[3](event_classifier.md), this dataset is real
(14 GB download, 13.5M training jets) and the full recipe is a genuine GPU
training job (tens of minutes to hours depending on hardware) — not
something to run start-to-finish on a laptop while reading. Each step below
gives the full-scale command **and** a CPU-safe quick-check variant that
exercises the same wiring on a few hundred jets in seconds, so you can verify
your setup before committing to the full run.

## Prerequisites

Complete [part 1](mnist.md) first if you have not. As in part 1:

```bash
git clone https://gitlab.cern.ch/aft/algorithms/salt.git
cd salt
pip install -e .
cd ..
mkdir gn2-opendata-tutorial && cd gn2-opendata-tutorial
export PYTHONPATH=$PWD
```

## 1. Get the data

The "Top quark pair events for heavy flavour tagging and vertexing at the
LHC" dataset (Delphes-simulated ATLAS-like detector response, mean pileup
50), from [Zenodo record 10371998](https://zenodo.org/records/10371998).
Unpacked size 28 GB; compressed download 14 GB.

```bash
export TUTORIAL_DATA=<path to directory>
mkdir -p $TUTORIAL_DATA
cd $TUTORIAL_DATA
curl -o $TUTORIAL_DATA/tutorialdata.zip "https://zenodo.org/api/records/10371998/files-archive"
unzip $TUTORIAL_DATA/tutorialdata.zip -d $TUTORIAL_DATA
rm $TUTORIAL_DATA/tutorialdata.zip
cd -
```

You get `pp_output_train.h5` (13.5M jets), `pp_output_val.h5` (1.35M jets),
`pp_output_test_ttbar.h5` (1.35M jets, no kinematic resampling), plus
`norm_dict.yaml` and `class_dict.yaml`. The Zenodo record documents each
file; how salt consumes them is what changes below.

!!! info "Reference-only in this doc's own validation"

    This tutorial's commands were validated end-to-end against a real GN2
    config, but the *download-and-train-40-epochs* recipe below is
    reference-only (14 GB download, GPU-scale training) — the validation
    experiment substitutes a local copy of `pp_output_val.h5` for the test
    file and runs the documented quick-check variant (small batches, one
    epoch, `num_workers=0`). See the validation experiment README for the
    full command-by-command classification.

## 2. The config: `gn2v2-opendata.yaml`

This is a **shipped** salt config (`salt/configs/gn2v2-opendata.yaml`) —
you do not write it, only point it at your data. It reproduces v1's `GN2.yaml`
architecture (256/128 dims, 4 transformer layers, 8 heads) built entirely from
stock v2 modules:

```yaml
name: GN2v2_opendata

data:
  batch_size: 1000
  num_workers: 8
  modules:
    input_samples:
      class_path: salt.data.InputSamples
      init_args:
        files:
          train: ${DATA_TRAIN_PATH}
          val: ${DATA_VAL_PATH}
          test: ${DATA_TEST_PATH}
        num: {train: -1, val: -1, test: -1}
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          jets: {global_object: true}
          tracks: {global_object: false, truncate: 40}
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt_btagJes, eta_btagJes]
          tracks: [d0, z0SinTheta, dphi, deta, qOverP, ...]  # 19 track variables
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
        class_path: salt.model.modules.Normaliser
        init_args:
          norm_dict: ${DATA_NORM_DICT_PATH}
          streams: [jets, tracks]
          global_object: jets
      track_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args: {stream: tracks, context: [normed.jets], out_dim: 256, dense: {hidden_layers: [256], activation: ReLU}}
      concat:
        class_path: salt.model.modules.Concat
        init_args: {streams: [tracks]}
      encoder:
        class_path: salt.model.modules.TransformerEncoder
        init_args: {dim: 256, out_dim: 128, num_layers: 4, attention: {num_heads: 8, attn_type: torch-math}, dense: {activation: ReLU, gated: false}}
      split:
        class_path: salt.model.modules.Split
        init_args: {streams: [tracks]}
      pool:
        class_path: salt.model.modules.GlobalAttentionPooling
        init_args: {input: encoded.seq, out: pooled.global}
      jets_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args: {stream: jets, input: pooled.global, label: flavour_label, class_names: [bjets, cjets, ujets, taujets], weight_source: null, dense: {hidden_layers: [128, 64, 32], activation: ReLU}}
      track_origin:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args: {stream: tracks, context: pooled.global, label: ftagTruthOriginLabel, class_names: [Pileup, Fake, Primary, FromB, FromBC, FromC, FromTau, OtherSecondary], weight: 0.5, weight_source: null, dense: {hidden_layers: [128, 64, 32], activation: ReLU}}
      track_vertexing:
        class_path: salt.model.modules.tasks.VertexingTaskModule
        init_args: {stream: tracks, label: ftagTruthVertexIndex, origin_label: ftagTruthOriginLabel, context: pooled.global, weight: 1.5, dense: {hidden_layers: [128, 64, 32], activation: ReLU}}
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  inputs_copy:
    class_path: salt.outputs.InputCopyWriter
    init_args: {streams: [jets, tracks]}
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [jets_classification, track_origin, track_vertexing]}
  pad_mask:
    class_path: salt.outputs.PadMaskWriter
    init_args: {streams: [tracks]}

trainer:
  max_epochs: 40
  precision: 32-true
```

(the full 19-variable track list and the `export:` ONNX-input block are
elided here for length — see the shipped file for the complete config.)

### What each piece is doing

- **`input_samples`** (`InputSamples`) — the data-sourcing module: per-stage
  file paths as `${DATA_*}` placeholders, overridden per run (below). This is
  the v2 replacement for v1's flat `train_file`/`val_file`/`test_file` config
  keys.
- **`reader`** (`H5StructuredReader`) — reads the pre-processed UPP-format H5:
  `jets` is `global_object: true` (one row per jet, like part 3's `event`
  stream), `tracks` is a padded sequence (`truncate: 40`), matching v1's track
  cap.
- **`norm`** (`Normaliser`, not `MaskedInputNormaliser`) — this dataset ships
  a precomputed `norm_dict.yaml` (means/stds from the *training* set only), so
  the config uses the fixed-dict normaliser, not the self-normalising one from
  parts [1](mnist.md#model)–[3](event_classifier.md#model-embed-encode-pool-classify).
  Either works; this config demonstrates the other option.
- **Three task heads**, matching v1's `GN2.yaml`: jet flavour classification
  (`jets_classification`), track origin classification, and track vertexing
  (edge classification over track pairs) — all context-conditioned on the
  same `pooled.global` jet representation, `LossSum` combining all three
  losses. `weight_source: null` on both classification tasks means v2 trains
  with unweighted cross-entropy here — v1's `class_dict`-based class weighting
  is opt-in via a top-level `--class_dict` CLI flag (fans out to every
  `ClassificationTaskModule`'s `weight_source`), not consumed by this config.
- **`outputs:`** — `InputCopyWriter` (unlike [part 3](event_classifier.md#outputs),
  `H5StructuredReader` DOES expose an h5py-openable source, so input-copying
  works), the three task heads' predictions, and a track pad-mask column.

## 3. Train

Copy the config so you can safely edit placeholders (never edit the shipped
file directly — it stays generic on purpose):

```bash
cp $(python -c "import salt, pathlib; print(pathlib.Path(salt.__file__).parent / 'configs/gn2v2-opendata.yaml')") config.yaml
```

Edit `config.yaml`, replacing the `${DATA_*}` placeholders with **literal**
paths into your download directory. YAML does NOT expand shell variables, so write the actual path (e.g.
`/home/you/tutorial-data/pp_output_train.h5`), not the literal string
`$TUTORIAL_DATA/...`:

```yaml
    input_samples:
      class_path: salt.data.InputSamples
      init_args:
        files:
          train: <path to directory>/pp_output_train.h5
          val: <path to directory>/pp_output_val.h5
          test: <path to directory>/pp_output_test_ttbar.h5
        num: {train: -1, val: -1, test: -1}
```

```yaml
      norm:
        class_path: salt.model.modules.Normaliser
        init_args:
          norm_dict: <path to directory>/norm_dict.yaml
```

!!! warning "Don't pass `files.train`/`files.val` as separate CLI dotted overrides"

    Unlike scalar options (`--trainer.max_epochs 5`), `files` is a single
    `dict[str, str]`-typed argument. Overriding two of its keys as two
    separate `--data.modules.input_samples.init_args.files.train X
    --....files.val Y` flags makes jsonargparse merge them into a
    `Namespace` before validating against the `dict` type hint — and it
    rejects the `Namespace`, with a parser error mentioning `Does not
    validate against any of the Union subtypes`. Edit the paths into the
    config file instead, as above.

Then fit — no CLI data overrides needed, everything is in the file:

```bash
salt fit --config config.yaml
```

That is the full-scale command (40 epochs, batch 1000, `num_workers: 8` as
shipped) — expect a GPU and a real training budget.

!!! tip "Quick-check variant (CPU, seconds, laptop-safe)"

    Before committing to the full run, verify the wiring on a handful of
    batches — same idea as v1's `--trainer.fast_dev_run 2`, spelled out
    explicitly here because `num_workers: 8` (the shipped default, sized for
    a training cluster) will over-subscribe a laptop's DataLoader workers:

    ```bash
    salt fit --config config.yaml \
      --data.num_workers 0 --data.batch_size 200 \
      --trainer.max_epochs 1 --trainer.limit_train_batches 5 --trainer.limit_val_batches 3 \
      --trainer.accelerator cpu --trainer.precision 32-true --trainer.logger false
    ```

    ```text
    Created training dataset with 1,500,000 entries
    Created validation dataset with 150,000 entries
      | Name                    | Type                     | Params
    --------------------------------------------------------------------
    0  | net                     | ModuleDict               | 2.3 M
    ...
    Epoch 0: 100%|██████████| 5/5 [00:07<00:00, 0.68it/s, train/loss=5.290, val/loss=5.140]
    `Trainer.fit` stopped: `max_epochs=1` reached.
    salt fit artifacts: config.yaml in run
    salt fit artifacts: checkpoints in run/ckpts
    ```

    Note `limit_train_batches`/`limit_val_batches` cap how many batches run,
    not the dataset size reported at startup — `InputSamples.num` (`-1` =
    all rows) still controls how much of the file is *indexed*, so the
    created training/validation dataset entry counts above are always the
    FULL file sizes regardless of the smoke-scale flags. The exact counts
    (1,500,000 / 150,000 above) are what this validation's own local copy of
    the dataset reports — a fresh Zenodo download is the full 13.5M-train /
    1.35M-val jets described in [Step 1](#1-get-the-data); expect that
    larger number instead, not a discrepancy to debug.

## 4. Evaluate

Evaluate on `pp_output_test_ttbar.h5` (no kinematic resampling — the honest
evaluation set). Because the `test:` path was already
set in `config.yaml`'s `input_samples` block, no extra flag is needed —
`salt test` needs only the saved config and a checkpoint, exactly as in
[part 1](mnist.md#7-evaluate):

The checkpoint filename embeds the epoch and validation loss (e.g.
`epoch=039-loss=1.02345.ckpt`) — glob for it rather than typing the exact
loss value, same as [part 1](mnist.md#7-evaluate):

```bash
salt test --config run/config.yaml --ckpt_path run/ckpts/epoch=039*.ckpt
```

```text
Created test dataset with 150,000 entries
Restoring states from the checkpoint path at run/ckpts/epoch=039-loss=<value>.ckpt
Testing DataLoader 0: 100%|██████████| 150/150 [05:24<00:00, 0.46it/s]
Wrote eval file run/ckpts/epoch=039-loss=<value>__test_pp_output_test_ttbar.h5
```

`H5OutputSink` writes the eval file next to the checkpoint, exactly as in
[parts 1–3](mnist.md#7-evaluate) — glob for it too:
`glob.glob("run/ckpts/*__test_*.h5")[0]`.
150,000 jets at batch 1000 (the shipped default) takes several minutes on
CPU — the smoke-scale quick-check from Step 3 does not shrink the test set
(there is no `--data.num_test` equivalent used here; the full test file is
always read at evaluation time). Run the real test set once, after you have
selected a checkpoint.

## 5. Performance plots with puma

The eval H5 columns follow the same `{run_name}_p{class}` / `target_{task}`
convention as [part 1](mnist.md#7-evaluate) — here
`GN2v2_opendata_pb`/`_pc`/`_pu`/`_ptau` and `target_jets_classification` (plus
`flavour_label`, copied straight from the input by `InputCopyWriter`). The
ROC plotting script below needs two things worth flagging:
`puma-hep` (pinned `0.5.3`) no longer ships
`puma.metrics.calc_rej` (replaced below with a five-line equivalent), and the
column names change per the table above.

!!! warning "Headless container: `import puma` needs a Tk stub"

    `puma.__init__` unconditionally imports `tkinter` and
    `matplotlib.backends.backend_tkagg` (even with `MPLBACKEND=Agg` set —
    the import happens at module load, not backend-selection time). A
    container/CI image with no system Tcl/Tk installed (no `libtk8.6.so`)
    fails on a bare `import puma` with `ImportError: libtk8.6.so: cannot
    open shared object file`. If you hit this, pre-register fake modules
    before importing puma (safe — nothing below ever opens a Tk window):

    ```python
    import sys, types
    sys.modules["tkinter"] = types.ModuleType("tkinter")
    fake_tkagg = types.ModuleType("matplotlib.backends.backend_tkagg")
    fake_tkagg.FigureCanvasTkAgg = type("FigureCanvasTkAgg", (), {})
    sys.modules["matplotlib.backends.backend_tkagg"] = fake_tkagg
    ```

```python
import glob

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from puma import Roc, RocPlot


def calc_rej(sig_disc: np.ndarray, bkg_disc: np.ndarray, sig_eff: np.ndarray) -> np.ndarray:
    """Background rejection (1/efficiency) at each target signal efficiency."""
    thresholds = np.quantile(sig_disc, 1 - np.asarray(sig_eff))
    bkg_eff = np.array([(bkg_disc >= t).mean() for t in thresholds])
    with np.errstate(divide="ignore"):
        return np.where(bkg_eff > 0, 1.0 / bkg_eff, np.inf)


with h5py.File(glob.glob("run/ckpts/*__test_*.h5")[0]) as f:
    table = f["jets"][:]

pb, pc, pu = table["GN2v2_opendata_pb"], table["GN2v2_opendata_pc"], table["GN2v2_opendata_pu"]
flavour = table["flavour_label"]

def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> float:
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

discs = np.apply_along_axis(disc_fct, 1, np.stack([pu, pc, pb], axis=1))
sig_eff = np.linspace(0.49, 1, 20)
# `flavour_label` on THIS dataset is already class_names INDEX order
# (0=b, 1=c, 2=u, 3=tau — verified via np.unique(flavour, return_counts=True):
# 35000/35000/70000/10000), NOT the general ATLAS 0/4/5/15 hadron-ID
# convention used for most ATLAS samples. Confirm this
# on any new dataset with `np.unique` before reusing either mapping — the
# wrong one either crashes (empty class, as here) or silently produces a
# plausible-but-wrong curve.
is_b, is_c, is_light = flavour == 0, flavour == 1, flavour == 2

plot_roc = RocPlot(
    n_ratio_panels=2, ylabel="Background rejection", xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets, tutorial sample, $f_c=0.018$",
    figsize=(6.5, 6), y_scale=1.4,
)
plot_roc.add_roc(
    Roc(sig_eff, calc_rej(discs[is_b], discs[is_light], sig_eff), n_test=int(is_light.sum()),
        rej_class="ujets", signal_class="bjets", label="GN2v2"),
    reference=True,
)
plot_roc.add_roc(
    Roc(sig_eff, calc_rej(discs[is_b], discs[is_c], sig_eff), n_test=int(is_c.sum()),
        rej_class="cjets", signal_class="bjets", label="GN2v2"),
    reference=True,
)
# puma-hep 0.5.3 requires rej_class_label explicitly when rej_class is a
# plain string rather than an ftag.Flavours Label enum; two reference ROCs
# need two ratio panels (n_ratio_panels=2 above).
plot_roc.set_ratio_class(1, "ujets", rej_class_label="ujets")
plot_roc.set_ratio_class(2, "cjets", rej_class_label="cjets")
plot_roc.draw()
plot_roc.savefig("roc.png", transparent=False)
```

`roc.png` now has the b-vs-light and b-vs-c rejection curves with ratio
panels. On the smoke-scale
recipe above (1 epoch, 5 train batches — wiring only, not a trained tagger)
this validation measured ujets rejection ≈2.3 and cjets rejection ≈1.9 at
77% b-efficiency; the full 40-epoch recipe on the full 13.5M-jet training
set is what produces GN2-competitive numbers.

## 6. Exercise: does the auxiliary supervision help?

The config you just trained has three heads: the flavour classifier plus two
auxiliary track tasks (`track_origin` and `track_vertexing`). Neither auxiliary
head produces a tagger output — they exist because supervising the track
representation is believed to make the *jet* classifier better. That is a claim
you can test, and doing so is the standard way to justify an auxiliary task.
This is an **ablation study**: remove a component, retrain under identical
conditions, and compare.

### Remove the auxiliary tasks

Copy your config and delete the two auxiliary heads. Three edits, and all three
are needed:

```yaml
# gn2v2-opendata-noaux.yaml
name: GN2v2_opendata_noaux        # 1. distinct run name, so eval columns do not collide

model:
  init_args:
    modules:
      track_origin: null          # 2. delete both auxiliary heads
      track_vertexing: null

outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args:
      tasks: [jets_classification]   # 3. and drop them from the outputs section
```

!!! warning "Skip edit 3 and it will not run"

    `RunTaskOutput` still naming a deleted task fails at config assembly. The
    opposite mistake — dropping the tasks from `outputs:` while leaving the
    heads in the model — fails too, with

    ```
    [mode=TEST] key 'preds.tracks.track_origin' is produced but consumed by no sink
    ```

    That is deliberate: salt refuses to train a head whose predictions nothing
    writes, rather than silently dropping columns. If you want to keep a head
    trained but unwritten, set `expose: [fit, val]` on it instead of deleting
    it. See [Outputs](../outputs.md#choosing-what-gets-written).

Then retrain and evaluate exactly as in steps 3–4, keeping every other setting
identical — an ablation is only meaningful if one thing changed:

```bash
salt fit  --config gn2v2-opendata.yaml --config gn2v2-opendata-noaux.yaml
salt test --config run-noaux/config.yaml --ckpt_path run-noaux/ckpts/epoch=039*.ckpt
```

### Compare the two

```python
# plot_roc_ablation.py
import glob

import h5py
import numpy as np
from puma import Roc, RocPlot
from puma.metrics import calc_rej

RUNS = {
    "GN2 (with aux tasks)": ("run/ckpts", "GN2v2_opendata"),
    "GN2 (no aux tasks)": ("run-noaux/ckpts", "GN2v2_opendata_noaux"),
}
REFERENCE = "GN2 (with aux tasks)"
NUM_JETS = 150_000
F_C = 0.018
# flavour_label on THIS dataset is class_names INDEX order (0=b, 1=c, 2=u,
# 3=tau) -- see the warning in step 5 before reusing this on another sample.
B, C, U = 0, 1, 2

sig_eff = np.linspace(0.49, 1, 20)
results = {}
for label, (ckpt_dir, run_name) in RUNS.items():
    with h5py.File(glob.glob(f"{ckpt_dir}/*__test_*.h5")[0]) as f:
        jets = f["jets"][:NUM_JETS]
    pb, pc, pu = (jets[f"{run_name}_p{x}"] for x in ("b", "c", "u"))
    disc = np.log(pb / (F_C * pc + (1 - F_C) * pu))
    flav = jets["flavour_label"]
    results[label] = {
        "ujets": calc_rej(disc[flav == B], disc[flav == U], sig_eff),
        "cjets": calc_rej(disc[flav == B], disc[flav == C], sig_eff),
        "n_u": int((flav == U).sum()),
        "n_c": int((flav == C).sum()),
    }

plot = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag=r"$\sqrt{s}=13$ TeV, $t\bar{t}$ jets" "\n" r"tutorial sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
for label, r in results.items():
    for rej_class, n in (("ujets", r["n_u"]), ("cjets", r["n_c"])):
        plot.add_roc(
            Roc(sig_eff, r[rej_class], n_test=n, rej_class=rej_class,
                signal_class="bjets", label=label),
            reference=(label == REFERENCE),
        )
plot.set_ratio_class(1, "ujets")
plot.set_ratio_class(2, "cjets")
plot.draw()
plot.savefig("roc_ablation.png", transparent=False)
```

Both models are evaluated on the same test file in the same order, so the
truth selection from either file is valid for both.

??? success "What to expect"

    The auxiliary tasks should help, but modestly — a few percent in light-jet
    rejection at fixed b-efficiency, more at the high-efficiency end where the
    track information matters most. If you run this at smoke scale (a handful
    of batches) the two curves will be indistinguishable noise: an ablation
    needs both legs trained to convergence to say anything.

## 7. Export to ONNX

To run your tagger in Athena it has to be exported to
[ONNX](https://onnxruntime.ai/). The export set is not a separate
configuration — it comes from the `export:` block plus the `outputs:` section
you already have, so the eval columns and the Athena outputs cannot drift
apart.

Preview the output names before exporting anything. This needs only the config,
not a checkpoint:

```bash
salt export --manifest -c run/config.yaml
```

```text
ONNX output manifest (folded conversion nodes, model_name=GN2v2opendata):
  GN2v2opendata_pb           float32  folded conversion node (outputs.* leaf)
  GN2v2opendata_pc           float32  folded conversion node (outputs.* leaf)
  ...
```

Then export:

```bash
salt export --ckpt_path run/ckpts/epoch=039*.ckpt --name GN2vXX
```

The config is inferred from the checkpoint's grandparent directory if you do
not pass `-c`. The model is written to `network.onnx` next to the run config
(override with `--output`, and pass `-o`/`--overwrite` to replace an existing
file), alongside `plan_onnx.txt` — the rendered ONNX plan and output manifest.

`--name` sets the prefix on every ONNX output (Athena forbids `_` and `-` in
it), so `--name GN2vXX` gives `GN2vXX_pb`, `GN2vXX_pc`, `GN2vXX_pu`. It
overrides `export.model_name` in the config.

!!! info "The torch-vs-ONNX check runs automatically"

    `salt export` sweeps sequence lengths 0–39, draws random inputs at each,
    and compares the eager model against onnxruntime. Float outputs must agree
    to `1e-4` (both rtol and atol, tunable with `--float-atol`) and contain no
    NaNs or exact zeros; int8 outputs such as track-origin indices must match
    **exactly**. Disable it with `--no-check` — but an inconsistent model is
    deliberately left on disk when the check fails, so you can debug it rather
    than having to re-export.

To inspect the metadata stored in the exported file:

```python
import json

import onnx

model = onnx.load("run/network.onnx")
meta = {p.key: p.value for p in model.metadata_props}
print(json.dumps(json.loads(meta["gnn_config"]), indent=2))
```

That payload holds the input variables and their normalisation constants, the
output names, the model name, and the plan hash.

!!! warning "v1 helper scripts are gone"

    Salt v1 shipped `to_onnx`, `get_onnx_metadata`, `compare_models` and
    `repair_ckpt` as separate console scripts. In v2 the only entry points are
    `salt` and `setup_mup`: `to_onnx` became `salt export`, metadata inspection
    is the snippet above (or Athena's own `get-onnx-metadata` binary), and
    there is **no v2 replacement for `compare_models`** — comparing salt scores
    against Athena-produced scores is a manual diff of the two H5 files.

Finally, validate the exported model inside Athena itself: see
[Athena Validation](../export.md#athena-validation).

## What changed since the v1 tutorial

| | v1 (the retired open-data tutorial) | v2 (this doc) |
|---|---|---|
| Config | one monolithic YAML with `class_path` model list | composable `modules:` — embed, encode, pool, task heads each a stock module |
| Data paths | edited directly into the config file | still edited directly into a copy of the shipped config (`input_samples.init_args.files.*`) — CLI dotted overrides of this dict-typed field are rejected by jsonargparse, see the warning box in Step 3 |
| Quick check | `--trainer.fast_dev_run 2` | same flag, plus `num_workers`/`batch_size` overrides (v2 configs ship cluster-sized worker counts by default) |
| Eval columns | `GN2_pu`/`GN2_pc`/`GN2_pb` | `GN2v2_opendata_pu`/`_pc`/`_pb` (`{run_name}_p{class}`, same convention as [part 1](mnist.md)) |
| Class weighting | `class_dict` wired into the config | opt-in `--class_dict` CLI flag, fans out to every classification task (this config leaves it unset) |
| ONNX export | `to_onnx` script | `salt export` (see [step 7](#7-export-to-onnx)); `get_onnx_metadata` / `compare_models` / `repair_ckpt` removed |
