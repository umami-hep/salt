# Run it in Python

The simplest deployment is no framework at all: load the ONNX file, build the
input tensor yourself, and call ONNX Runtime. It needs `onnxruntime` and
whatever reads your input files — nothing else.

This page is the foundation for the whole section. The framework pages
([TDD](tdd.md), [easyjet](easyjet.md)) assume you have read it, because a
python consumer is also the *reference* you validate those frameworks against:
when an in-framework score and a python score agree on the same events, the
framework's tensor construction is correct.

## 1. Read the model's own description

Everything about the model comes out of the model. Do not hardcode any of it.

```python
import json

import onnxruntime as ort

sess = ort.InferenceSession("EventTagger.onnx", providers=["CPUExecutionProvider"])
cfg = json.loads(sess.get_modelmeta().custom_metadata_map["gnn_config"])

variables = [v["name"] for v in cfg["input_sequences"][0]["variables"]]
output_names = cfg["output_names"]
graph_input = sess.get_inputs()[0].name

print(graph_input, sess.get_inputs()[0].shape)  # e.g. jet_kinematics ['n_jets', 4]
print(variables)                                 # e.g. ['pt', 'eta', 'phi', 'm']
print(output_names)                              # e.g. ['..._pbackground', '..._psignal']
```

Two lists matter here and they are different things:

- `sess.get_inputs()[0].name` is the **graph** input name — the key
  `session.run()` wants.
- `cfg["input_sequences"][0]["variables"]` is the **ordered variable list** —
  which column of the tensor is which.

Read [Export to ONNX](export.md#what-the-model-file-carries) for the rest of
the `gnn_config` keys.

## 2. Supply what the metadata does not carry

Three things are part of the contract but are *not* in the file. You have to
state them, and getting any of them wrong produces plausible, wrong numbers
rather than an error.

```python
# (a) which branch each metadata variable came from
BRANCHES = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "phi": "recojet_antikt4PFlow_phi",
    "m": "recojet_antikt4PFlow_m_NOSYS",
}
# (b) units: this model was trained on GeV; the ntuple stores MeV
SCALES = [0.001, 1.0, 1.0, 0.001]
# (c) the truncation the training reader applied
TRUNCATE = 10
```

Keep the training reader's config alongside the model, or copy these into your
consumer's own config. They are as much a part of the deployment as the `.onnx`
file is.

## 3. Build the input tensor

This is the part that goes wrong. For a sequence input the tensor is
`[L, F]` — one row per object, columns in the metadata's variable order:

```python
import awkward as ak
import numpy as np
import uproot

with uproot.open("events.root:AnalysisMiniTree") as tree:
    raw = tree.arrays([BRANCHES[v] for v in variables], library="ak")

cols = [ak.to_list(raw[BRANCHES[v]]) for v in variables]
scales = np.asarray(SCALES, dtype=np.float32)

scores = np.zeros((len(raw), len(output_names)))
for i in range(len(raw)):
    rows = [c[i][:TRUNCATE] for c in cols]
    if not rows[0]:
        scores[i] = -1.0  # no objects: the model has nothing to pool over
        continue
    tensor = np.asarray(rows, dtype=np.float32).T.reshape(len(rows[0]), len(variables))
    out = sess.run(None, {graph_input: tensor * scales})
    scores[i] = [float(np.asarray(o).reshape(-1)[0]) for o in out]
```

### Ordering, truncation, padding

Three semantics have to match the training reader exactly.

**Column order** follows the metadata variable list, *not* the branch order in
your file. Building the tensor by iterating `variables` — as above — is what
guarantees this; building it by iterating the file's branches does not.

**Truncation** keeps the **first** `truncate` objects in container order. The
reader does not re-sort: it relies on the collection already being pT-ordered.
If your consumer sorts, or reads a differently-ordered collection, you have
silently changed the input.

**Do not pad.** This is the one that surprises people. The exported graph has a
*dynamic* length axis, so each event is fed at its true length and no padding is
needed. Padding is a **batching** device: during training, events of different
lengths have to be stacked into one rectangular tensor, so short events are
padded and a mask tells attention to ignore the pad rows. A per-event ONNX call
stacks nothing, so there is no pad and no mask — and if you pad anyway, those
zero rows are *real* objects as far as the graph is concerned. They will drag
the pooled representation towards whatever a zero-filled object looks like.

!!! note "Zero-length sequences"

    An event with no objects has nothing to pool. Feed it and you will get
    `NaN` at best. Decide on a sentinel, write it for those events, and make
    sure every consumer of the model uses the same one — otherwise your
    "agreement" check is comparing sentinels, not scores.

!!! note "Normalisation is inside the graph"

    Models that use a self-normalising input layer carry their input statistics
    in the checkpoint, and those are traced into the ONNX graph. A consumer
    therefore feeds **raw, unnormalised values** — the numbers as they appear in
    the file. There is no norm dict to ship alongside the model. (Unit scaling
    is a different thing: the graph normalises whatever scale it was trained
    on, so MeV-vs-GeV is still yours to get right.)

## 4. Check it against salt

Do not take agreement on faith. Run the same events through `salt test` and
diff the scores.

Point the config at a **single** sample so both sides see the same events in
the same order — a multi-sample reader interleaves, so row *i* would not be
event *i* of either file:

```bash
salt test -c config.yaml -c single_sample.yaml --ckpt_path <checkpoint>.ckpt
```

Then compare the eval H5's score columns against your array:

```python
import h5py
import hdf5plugin  # noqa: F401 - registers the codecs salt's eval H5 uses
import numpy as np

with h5py.File("...__test_signal_test.h5") as f:
    salt = np.stack([f["event"][n] for n in output_names], axis=1)

print("worst |python - salt|:", np.abs(scores - salt).max())
```

A healthy result is a worst-case difference of order `1e-7`; anything up to
about `1e-5` is still float32 noise. Larger than that is a wiring bug, and the
three semantics above are where to look — not the export.

!!! note "`hdf5plugin` is not optional"

    salt writes its eval H5 with a blosc filter. Without `import hdf5plugin`
    the read fails with an unhelpful `can't open directory (.../plugin)` error
    rather than anything about compression.

## Where next

A python consumer is the right tool for studies, for validating a model, and
for anything that runs over an existing ntuple. Once the score needs to be
available *during* production — to selections, and in the output ntuple
alongside everything else — it has to move into the framework:

- [Deploy in the TDD](tdd.md) for jet-level taggers in Athena;
- [Deploy in easyjet](easyjet.md) for event-level models in an analysis
  framework.

Both of those are validated against the route on this page.
