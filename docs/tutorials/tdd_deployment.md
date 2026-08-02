# Deploy your tagger: from checkpoint to the training-dataset-dumper

You have trained a tagger and you want to see it produce scores on real
derivations. This tutorial takes you the whole way: `salt export` to ONNX, then
running that ONNX inside the
[training-dataset-dumper](https://gitlab.cern.ch/aft/algorithms/training-dataset-dumper)
(TDD), which serves it through Athena's `FlavorTagInference` — the same
inference code that runs in production reconstruction and derivations.

No TDD or Athena source changes are needed. Everything here is configuration.

!!! info "What this gets you, and what it does not"

    Running in the TDD exercises the **real Athena inference path**: Athena
    parses your `gnn_config` metadata, builds its input tensors from the xAOD,
    runs ONNX Runtime, and decorates jets with your scores. That is the step
    that catches export mistakes.

    It is **not** the same as deploying a recommended tagger. Getting a model
    into a production derivation additionally needs the file uploaded to
    groupdata, registration in the FTAG configuration, and CP sign-off. See
    [Towards production](#6-towards-production) at the end.

## The one thing to understand first

Athena is stricter than salt about names. It does not simply read whatever your
model declares and adapt — several names are **parsed by regex or looked up in
hard-coded tables**, and a mismatch is a hard failure, usually at the first
event or at configuration time.

Almost every problem in this workflow is a name problem. The rules below each
say *what* the rule is and *why it exists*, so you can reason about your own
case rather than pattern-matching.

## 1. Prerequisites

You need a trained checkpoint and its run config, and the run config needs an
`export:` block. If yours does not have one, you can stack a second config
carrying only that block (see step 2).

Your `export.model_name` is the **Athena-facing model name**. It becomes the
prefix of every decorated variable (`GN3EPCLV01_pb`, and so on).

!!! warning "`model_name` may not contain `_` or `-`"

    Athena derives the model name by splitting each ONNX output node name at
    the **first underscore** and requiring every output to agree on the prefix.
    An underscore inside the model name would split in the wrong place; a
    disagreement raises *"model names are not consistent between outputs"*.

    salt validates this for you at export time, so a bad name fails fast.

## 2. Export to ONNX

```bash
salt export \
    --ckpt_path /path/to/your/run/checkpoints/epoch=NNN-loss=X.XXX.ckpt \
    -c /path/to/your/run/config.yaml \
    --output my_tagger.onnx
```

`--check` is on by default: after writing the file, salt runs the traced torch
model and ONNX Runtime over a sweep of random inputs at many sequence lengths
and compares them.

!!! tip "Preview the outputs without a checkpoint"

    ```bash
    salt export --manifest -c /path/to/your/run/config.yaml
    ```

    This prints the full list of ONNX output names and dtypes that the export
    will produce. It resolves the `export:` block and the output manifest
    without loading any weights, so a malformed `export:` section fails in
    seconds instead of after a multi-minute trace. Run it first — you will also
    need this list in step 4.

### Choosing an agreement tolerance

`--float-atol` defaults to `1e-4`. That is the right bar for a realistic model.
Do not be alarmed if a tighter bar fails: float32 error accumulates through
every layer, so a large multi-layer transformer will legitimately show
disagreements of order `1e-5`–`1e-6` between torch and ONNX Runtime even when
the export is perfectly correct. Integer outputs (auxiliary track tasks) are
compared **exactly** and are unaffected by this setting.

Only tighten `--float-atol` if you have a small model and want a stricter
regression guard.

### Input naming: use the shipped configs' convention

This is the trap that costs the most time, because it fails at the first event
with an opaque `std::out_of_range`.

**For a `v1`-metadata model, Athena does not use your ONNX graph input names as
identifiers — it derives the names it will look up, itself.** The scalar node is
always `jet_features`, and each sequence is the stream name with its trailing
`s` removed, plus `_features`. So it expects `track_features`, `flow_features`,
`electron_features` — **singular**. Athena then looks up exactly those names in
the ONNX graph, and anything else is not found.

The shipped configs already use the correct form. Copy it:

```yaml
export:
  model_name: MyTagger # Athena name: no '_'/'-'
  inputs:
    - {port: inputs.jets, name: jet_features}
    - {port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks}
    - {port: inputs.flows, name: flow_features, sequence: true, dyn_axis: n_flows}
    - {port: inputs.electrons, name: electron_features, sequence: true, dyn_axis: n_electrons}
```

Include only the streams your model actually consumes.

!!! danger "Do not let `name:` default here"

    salt's own default for a stream called `tracks` is `tracks_features`
    (plural) — which Athena will never look up. Always write the singular form
    explicitly, exactly as the shipped configs do.

### Sequence node names: `track_selection` drives them

Separately from the graph input names, the `gnn_config` metadata carries a
**node name per sequence**, and Athena parses those with regexes to decide
which track selection and which sort order to apply.

You do not write these names by hand. salt derives them from
`export.track_selection`:

```yaml
export:
  track_selection: r22loose # must match the selection used in your training sample
```

which produces metadata node names of the form `tracks_r22loose_sd0sort`,
`flows_r22loose_sd0sort` and `electrons_r22default`.

!!! warning "An unparseable node name is a hard failure at configuration time"

    Athena matches the node name against a list of selection regexes *and* a
    list of sort-order regexes, and throws `std::logic_error` if either finds no
    match. A name like `tracks_features` matches neither, so the job dies before
    processing a single event.

    Set `track_selection` to the selection that was applied when your **training
    sample** was dumped. Getting this wrong is not just a naming issue — it
    means Athena feeds your model a different set of tracks than it was trained
    on.

### What ends up in the metadata

salt writes a single `gnn_config` JSON blob into the ONNX file's metadata. The
parts Athena reads are:

| Key | Meaning |
| --- | --- |
| `onnx_model_version` | metadata schema version (`v1`) |
| `model_name` | the output-name prefix |
| `inputs` | scalar (per-jet) node and its variables |
| `input_sequences` | one entry per constituent sequence, with its variables |
| `output_names` | every decorated variable the model produces |

One small convenience: salt strips a trailing `_btagJes` from **global**
variable names when writing the metadata, so a model trained on `pt_btagJes` /
`eta_btagJes` publishes `pt` / `eta` — which is what Athena's jet accessors
expect. You do not need to remap those.

## 3. Checkpoint caveats

`salt export` reconstructs the model from the checkpoint alone — there is no
datamodule involved. That means the checkpoint has to carry everything needed
to rebuild a **bound** model: the Lightning hyper-parameters, and salt's
resolved bind schema.

Checkpoints written by a normal `salt fit` run carry both, and export just
works.

!!! warning "Externally converted checkpoints may not be exportable"

    A checkpoint produced outside salt's own training loop — for example by a
    migration or conversion script that saves only a `state_dict` — can be
    missing this metadata. Symptoms:

    - `TypeError: __init__() missing 1 required positional argument: 'lrs'`
      — no Lightning hyper-parameters in the checkpoint.
    - `RuntimeError: Unexpected key(s) in state_dict: "net...."`, listing
      essentially every weight — no stored bind schema. salt's modules are
      shaped lazily, so without the schema the model is still **unbound** when
      the weights are loaded, and an unbound model has no parameters for them to
      match. This is *not* a sign that your weights are wrong.

    Such a checkpoint may still load fine under `salt test -c config.yaml`,
    because that path builds and binds the model from the config first. The
    remedy is to re-save the checkpoint through a config-driven path that binds
    the model and strict-loads the weights before export.

    This is a known limitation for converted checkpoints, not a step you should
    expect to perform for a normally-trained model.

## 4. Run it in the TDD

The TDD serves a single ONNX file through its `MultifoldTagger` block. See the
TDD's own
[configuration docs](https://gitlab.cern.ch/aft/algorithms/training-dataset-dumper/-/blob/main/docs/configuration.md)
for building and running the dumper itself; what follows is only the part
specific to your model.

Add a block like this to your dump config:

```json
{
    "block": "MultifoldTagger",
    "alg_name": "MyTaggerTest",
    "tagger_name": "GN3EPCLV01",
    "target": "Jet",
    "nn_paths": ["/abs/path/to/my_tagger.onnx"],
    "per_fold_defaults": {
        "/abs/path/to/my_tagger.onnx": {
            "MyTagger_pb": -1.0,
            "MyTagger_pc": -1.0,
            "MyTagger_pu": -1.0
        }
    },
    "default_zero_tracks": true,
    "remap": {
        "BTagTrackToJetAssociator": "TracksForBTagging",
        "FTagElectrons": "GhostFTagElectrons"
    }
}
```

and list every decorated output in the writer's variable block:

```json
{
    "variables": {
        "jet": {
            "floats": ["MyTagger_pb", "MyTagger_pc", "MyTagger_pu"]
        }
    }
}
```

Each field, and why it is the way it is:

**`nn_paths`** — a one-element list for a single (non-multifold) model.

**`per_fold_defaults`** — the key must be the **exact same string** you wrote in
`nn_paths[0]`. The block looks the path up as a dictionary key with no
normalisation, so a trailing-slash or relative-vs-absolute difference is a
`KeyError`. The values are what gets decorated on jets that the tagger does not
run on (see `default_zero_tracks`). A distinctive value like `-1.0` makes those
jets easy to spot and exclude when you look at the distributions.

**`tagger_name`** — **must be a name the FTAG configuration already knows.** It
is looked up in a hard-coded registry to decide which extra containers the
tagger needs, and an unknown name raises `KeyError`. It does **not** name your
model: your decorations are named from the ONNX output prefix, independently.
So pick the registered tagger whose *input requirements* match yours — for a
model consuming electrons, pick a registered electron-consuming tagger. Set it
explicitly; the automatic fallback tries to parse a name out of the file path
and will usually not produce a registry entry.

**`alg_name`** — set it explicitly. The default builds a name out of path
components, which produces something meaningless (and possibly colliding) for a
model outside the standard directory layout.

**`variables.jet.floats`** — every output you want written must be listed
verbatim. Take the names from `salt export --manifest`. Two caveats: outputs not
listed are silently not written, and **only per-jet float outputs belong here**.
Auxiliary per-track outputs (track origin, vertex index, and similar) are
per-constituent integer arrays, not jet floats, and listing them here will fail.

**`target`** — must be `"Jet"`.

**`remap`** — this maps EDM names the model expects onto the names present in
your input. Two things to know:

- The block normally supplies the track-associator remap for you, **but only if
  you leave `remap` unset entirely**. The moment you set it for any reason, you
  must restate `"BTagTrackToJetAssociator": "TracksForBTagging"` yourself.
- If your model consumes electrons, you need
  `"FTagElectrons": "GhostFTagElectrons"`. The electron loader reads a jet
  decoration under its own default name, while the dumper block that writes the
  jet→electron association writes it under the ghost-association name. Nothing
  bridges these automatically.

Keep the map minimal: an entry that is never consumed is a hard error at
initialisation, deliberately, to catch typos.

!!! warning "Block ordering: your tagger must run after its decorators"

    A tagger consuming electrons or muons depends on decorations written by
    other blocks earlier in the same job. When one config imports another, the
    **importing** config's blocks are placed *before* the imported ones — so
    naively adding your tagger to a config that imports a base config schedules
    it too early, and it aborts with a missing-aux-data error such as
    *"Attempt to retrieve nonexistent aux data item `::GhostFTagElectrons`"*.

    If you hit this, make sure your tagger block ends up **last** in the
    assembled block list — for example by adding it to a fully-resolved config
    rather than relying on import ordering.

### Your input derivation must contain the right content

A tagger can only run if its inputs exist in the file you dump.

- **Tracks-only taggers** are the least demanding.
- **Flow, electron and muon** inputs need the corresponding jet-side
  associations.
- **Track-lepton variables** (lepton ID, muon quality and related) are written
  by the *derivation*, not by the dumper — the dumper never schedules that
  algorithm itself, so if they are not already in the file, the tagger cannot
  run.

In practice this means **use an FTAG-flavour derivation** (FTAG1/FTAG2) for
anything beyond a simple tracks-only model. This is also what the dumper's own
offline b-tagging tests use. A PHYS derivation does not carry the
lepton/electron-association content that the richer taggers need.

## 5. Validate the output

Check the decorated columns rather than assuming. At minimum: every name from
`output_names` is present, all values are finite, probabilities lie in `[0, 1]`,
and the class probabilities sum to one per jet.

```python
import h5py
import numpy as np

CLASS_COLUMNS = ["MyTagger_pb", "MyTagger_pc", "MyTagger_pu"]
DEFAULT = -1.0  # per_fold_defaults value, for jets the tagger skipped

with h5py.File("output.h5") as f:
    jets = f["jets"]
    missing = [c for c in CLASS_COLUMNS if c not in jets.dtype.names]
    assert not missing, f"missing decorated columns: {missing}"

    scores = np.stack([jets[c] for c in CLASS_COLUMNS], axis=1)

assert np.isfinite(scores).all(), "non-finite scores"

# exclude jets that got the zero-track default
real = scores[~np.all(scores == DEFAULT, axis=1)]
assert ((real >= 0.0) & (real <= 1.0)).all(), "probability outside [0, 1]"
assert np.allclose(real.sum(axis=1), 1.0, atol=1e-3), "probabilities do not sum to 1"

print(f"OK — {len(real)} jets validated")
```

Those checks prove the plumbing. To show the scores are *meaningful*, split them
by truth flavour and compare: a b-tagging output should be clearly higher for
b-jets than for light jets. If the plumbing checks pass but the separation is
absent, suspect the input wiring — most often `track_selection` not matching the
selection used for your training sample.

!!! tip "Check your class ordering against truth"

    Output names come from the class names in your config, in order. If the
    checkpoint's class ordering differs from what the config declares, every
    check above still passes while the labels are permuted. A per-flavour mean
    of each score is a cheap way to catch it: the largest mean for b-jets should
    be the output you *named* for b-jets.

## 6. Towards production

Running in the TDD proves the model works through Athena's inference path. To go
further and have your tagger used in production reconstruction or derivations,
you additionally need to:

- upload the ONNX file to groupdata so it resolves by a calibration-area path
  rather than an absolute one;
- register the tagger in the FTAG configuration, including its input
  dependencies (this is what makes a `tagger_name` valid in step 4);
- follow the FTAG group's process for validation and recommendations.

Those steps live with the FTAG group rather than with salt — consult the FTAG
documentation and coordinate with the group before assuming a model is
deployable.

## See also

This page covers **jet-level** taggers, where Athena's `FlavorTagInference`
dictates the names and the metadata is parsed strictly. For an **event-level**
model consumed by your own code — where naming is free-form and you deploy the
file yourself — see
[Deploy an event-level tagger](event_deployment.md).
