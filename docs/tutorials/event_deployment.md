# Deploy an event-level tagger

The [event-level classifier tutorial](event_classifier.md) trains a model that
scores a whole *event* rather than a jet. This page takes that model the rest of
the way: add the export surface it does not ship with, export to ONNX, and run
the result outside salt.

It is the event-level counterpart to
[Deploy your tagger in the TDD](tdd_deployment.md). The two differ in one
important way, and it shapes everything below:

| | jet-level (FTAG) | event-level (this page) |
| --- | --- | --- |
| Consumer | Athena `FlavorTagInference` | your own code |
| Input names | fixed by the consumer, not by you | free-form, you choose |
| Metadata | parsed by regex, strict | read as documentation |
| Deployment | groupdata + FTAG registration | ship the file with your job |

Because there is no framework dictating names, an event-level model is
considerably easier to deploy — but nothing validates your wiring for you, so
the checks at the end matter more.

## 1. Add the export surface

The tutorial config trains and evaluates, but has no ONNX surface: no
conversion node, no export sink, no `export:` block. Three additions.

**A conversion node**, so the graph emits *probabilities* rather than raw
logits. This lives in `model.modules` alongside everything else, because it is
part of the traced graph:

```yaml
model:
  init_args:
    modules:
      event_probs:
        class_path: salt.outputs.ClassProbs
        init_args: {task: event_classification, stream: event}
```

**An export sink**, which names the ONNX outputs. The manifest is declared by
an `OnnxExportSink` — putting an `outputs:` list inside the `export:` block is
a hard error:

```yaml
callbacks:
  onnx_export:
    class_path: salt.outputs.OnnxExportSink
    init_args:
      outputs:
        - {key: outputs.event.event_classification, names: [pbackground, psignal]}
```

**The `export:` block**, which names the graph inputs:

```yaml
export:
  model_name: EventTagger
  inputs:
    - {port: inputs.jets, name: jet_kinematics, sequence: true, dyn_axis: n_jets}
```

### Naming is yours to choose here

The FTAG route has to use `track_features`, `flow_features` and friends because
Athena derives those names itself and looks them up in the graph. **No such
constraint applies to a bespoke consumer** — it reads the input name out of the
model file at load time. `name:` should therefore say what the tensor *is*.
`jet_kinematics` is a better name than `jets_features` when the payload is
`[pt, eta, phi, m]`.

`sequence: true` gives the input a dynamic length axis and no batch dimension —
the tensor is `[L, F]`, i.e. "this event's L jets, F variables each". A
`sequence: false` input would be a per-event global vector, `[1, F]`.

!!! note "Normalisation lives inside the graph"

    The tutorial model uses `MaskedInputNormaliser`, which learns its own input
    statistics during training and carries them in the checkpoint. They are
    traced into the ONNX graph, so **a consumer feeds raw, unnormalised
    values** — the same numbers that are in the ntuple. There is no norm dict
    to ship alongside the model.

## 2. Export

Preview first — this needs no checkpoint and fails in seconds on a malformed
export block:

```bash
salt export --manifest -c config.yaml
```

```text
ONNX output manifest (folded conversion nodes, model_name=EventTagger):
  EventTagger_pbackground  float32  folded conversion node (outputs.* leaf)
  EventTagger_psignal      float32  folded conversion node (outputs.* leaf)
```

Then export:

```bash
salt export \
    --ckpt_path run/ckpts/epoch=005-loss=0.44452.ckpt \
    -c config.yaml \
    --output EventTagger.onnx
```

`--check` is on by default and compares the traced torch model against ONNX
Runtime across a sweep of random inputs at many sequence lengths.

!!! tip "`salt export` works on ROOT-reader configs that `salt graph validate` cannot handle"

    The event-classifier tutorial notes that `salt graph validate` fails on the
    `UprootReader`/`MultiSampleReader` family, because it calls
    `Reader.prepare()` on a reader that has no file bound yet.

    Export is unaffected. It never builds a dataset — it reconstructs the model
    from the checkpoint and traces it — so it needs no source file and runs
    happily on exactly the config that `graph validate` rejects. You do not
    need to restructure your config to export it.

## 3. What the model file carries

salt writes a `gnn_config` JSON blob into the ONNX metadata. For a bespoke
consumer the useful keys are:

| Key | What you need it for |
| --- | --- |
| `model_name` | prefix of every output name |
| `input_sequences` | per stream: the node name and the **ordered** variable list |
| `inputs` | the same, for non-sequence (global) inputs |
| `output_names` | what each output tensor means |

The variable list is the contract that matters. It tells a consumer which
column is which, in order. Read it — do not hardcode it.

!!! warning "One thing the metadata does NOT record"

    It carries variable *names* (`pt`, `eta`, ...), not the ntuple *branches*
    they were read from — that mapping lived in the training reader's
    `branches:` block. A consumer must supply it. Keep the reader config
    alongside the model, or write the mapping into your consumer's own config.

## 4. Run it: the post-hoc route

The simplest deployment is no framework at all — read the ntuple, build the
tensor, run ONNX Runtime:

```python
import json

import awkward as ak
import numpy as np
import onnxruntime as ort
import uproot

BRANCHES = {  # metadata variable -> ntuple branch (see the warning above)
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "phi": "recojet_antikt4PFlow_phi",
    "m": "recojet_antikt4PFlow_m_NOSYS",
}
TRUNCATE = 10  # must match the reader's `truncate:` used in training

sess = ort.InferenceSession("EventTagger.onnx", providers=["CPUExecutionProvider"])
cfg = json.loads(sess.get_modelmeta().custom_metadata_map["gnn_config"])

variables = [v["name"] for v in cfg["input_sequences"][0]["variables"]]
output_names = cfg["output_names"]
graph_input = sess.get_inputs()[0].name

with uproot.open("events.root:AnalysisMiniTree") as tree:
    raw = tree.arrays([BRANCHES[v] for v in variables], library="ak")

cols = [ak.to_list(raw[BRANCHES[v]]) for v in variables]

scores = []
for i in range(len(raw)):
    rows = [c[i][:TRUNCATE] for c in cols]
    tensor = np.asarray(rows, dtype=np.float32).T.reshape(len(rows[0]), len(variables))
    out = sess.run(None, {graph_input: tensor})
    scores.append([float(np.asarray(o).reshape(-1)[0]) for o in out])

scores = np.asarray(scores)
for j, name in enumerate(output_names):
    print(f"{name}: mean {scores[:, j].mean():.4f}")
```

### Three semantics you must match, or the numbers are quietly wrong

The model runs fine on badly-built input; it just returns nonsense. None of
these are recorded in the metadata:

1. **Column order** follows the metadata variable list, *not* the ntuple's
   branch order. Build the tensor from `variables`, as above.
2. **Truncation** must match the `truncate:` your training reader used, and it
   takes the *first* N objects in file order — the reader does not re-sort, it
   relies on the ntuple already being pT-ordered.
3. **Do not pad.** The exported graph has a dynamic length axis, so feed each
   event at its true length. Padding is a *batching* device from training; feed
   padded rows here with no mask and they will pollute the attention pooling.

### Check it against salt

Do not take agreement on faith — run the same events through `salt test` and
compare. Point the config at a single sample so both sides see the same events
in the same order (a multi-sample reader interleaves, so row *i* would not be
event *i* of either file), then diff the score columns of the eval H5 against
your post-hoc array. They should agree to float32 precision.

```python
import h5py
import hdf5plugin  # noqa: F401 - registers the codecs salt's eval H5 uses
import numpy as np

with h5py.File("...__test_signal_test.h5") as f:
    salt = np.stack([f["event"][n] for n in output_names], axis=1)

print("worst |post-hoc - salt|:", np.abs(scores - salt).max())
```

A healthy result is a worst-case difference of order `1e-7`. If it is larger,
suspect the three semantics above before suspecting the export.

!!! note "`hdf5plugin` is not optional"

    salt writes its eval H5 with a blosc filter. Without `import hdf5plugin`
    the read fails with an unhelpful `can't open directory
    (.../plugin)` error rather than anything about compression.

## 5. Running inside a framework

The post-hoc route is enough for studies and for validating a model. Running
the score *inside* an analysis framework — so it is available to selections and
lands in the output ntuple with everything else — means an algorithm that owns
an ONNX Runtime session.

In easyjet, the pieces are already in place for this: the AthAnalysis release
carries the ONNX Runtime headers and library transitively, so an algorithm can
use `Ort::Session` directly with no extra CMake `find_package`. The existing
`BaselineVarsbbyyAlg` in `bbyyAnalysis` is a working reference for the plumbing
(model path resolution, session creation, input introspection, tensor build,
`session.Run`).

The shape of such an algorithm:

- resolve the model path and create the `Ort::Session` in `initialize()`;
- resolve **all** input/output names and shapes in `initialize()` too — the
  framework's coding rules bar string operations and string-keyed lookups from
  the event loop;
- in `execute()`, build the `[L, F]` tensor from the jet container in the
  metadata's variable order, run the session, and write the outputs as
  `EventInfo` decorations;
- recompute per systematic if the inputs are systematics-dependent.

A generic, reusable version of this belongs in the framework's core package
rather than in any one analysis package, so every analysis can point it at its
own model file through configuration.

!!! warning "Validate in-framework scores against the post-hoc route"

    Whatever framework you deploy in, run both routes over the same events and
    compare, exactly as in §4. It is the only check that catches a tensor built
    in the wrong order or a truncation mismatch inside the algorithm — both of
    which produce plausible-looking scores.

## What to take away

- An event-level model needs three config additions to become deployable: a
  conversion node, an export sink, and an `export:` block.
- Input naming is free-form for a bespoke consumer; use it to describe the
  tensor.
- The metadata's ordered variable list is the contract. The branch mapping,
  truncation and padding semantics are *not* in it — you must carry them.
- Always cross-check a deployed model against `salt test` on the same events.
