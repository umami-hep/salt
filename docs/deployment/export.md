# Export to ONNX

A trained checkpoint is a salt object: it only means something to salt. To use
a model anywhere else — Athena, an analysis framework, a plotting script — you
export it to [ONNX](https://onnxruntime.ai/), a portable graph format that
almost every runtime can load.

`salt export` produces **one file** that carries three things:

- the **graph** — the traced network, including any normalisation layers, so a
  consumer feeds raw values;
- the **outputs** — named scalars, one per class probability or regression
  target;
- the **metadata** — a `gnn_config` JSON blob describing the inputs, their
  order, and the output names.

The rest of this section is about what happens after that file exists:

| Page | Consumer |
| --- | --- |
| [Run it in Python](python.md) | your own script — onnxruntime directly |
| [Deploy in the TDD](tdd.md) | Athena `FlavorTagInference`, jet-level taggers |
| [Deploy in easyjet](easyjet.md) | an analysis framework, event-level models |

## Two kinds of consumer

Which page you need depends on something that shapes the whole job: whether a
framework is doing the wiring for you, or whether you are writing the consumer
yourself.

**Framework consumers** — Athena's `FlavorTagInference`, used by the
training-dataset-dumper and by reconstruction — are generic code that already
knows how to serve a jet tagger. It builds the input tensors, applies the track
selection, and writes the outputs as b-tagging decorations, all without you
writing any inference code. The price is that it dictates the names: it derives
the graph input names itself (`track_features`, `flow_features`, ...) and looks
them up in your graph. Get one wrong and the job throws. That is a *feature* —
the failure is loud, immediate, and points at the mistake.

**Self-written consumers** — a python analysis script, or an algorithm you add
to an analysis framework — have no such contract. Nothing derives names,
because nothing else knows what your model is. You read the input name out of
the file and feed a tensor you built yourself. Naming is free, there are no
traps to fall into... and nothing checks your work. Feed the columns in the
wrong order, or truncate differently from the training reader, and the model
still runs and still returns plausible numbers. They are simply wrong.

So the two routes fail in opposite ways, and that dictates the discipline:

| | framework consumer | self-written consumer |
| --- | --- | --- |
| Input names | fixed by the framework | free-form, you choose |
| Metadata | parsed strictly, throws on mismatch | read as documentation |
| Wiring | done for you | done by you |
| Typical failure | loud, at start-up | silent, wrong numbers |
| What protects you | the framework's own checks | a parity check you run |

!!! tip "Validate by parity, always"

    Whatever you deploy into, run the *same events* through the deployed model
    and through `salt test`, and compare the scores. Agreement to float32
    precision (order `1e-7`, worst case `1e-5`) is the only evidence that the
    deployment is correct. Every page in this section ends with this check,
    and it is not optional for a self-written consumer — it is the *only* thing
    standing between you and quietly wrong physics.

## What the config needs

Training and export are configured in the same file. A config that only trains
is missing three things.

### 1. A conversion node

The network's task heads emit raw logits. A consumer wants probabilities, so
the softmax has to live *inside* the exported graph. That means it is a graph
module like any other:

```yaml
model:
  init_args:
    modules:
      jet_probs:
        class_path: salt.outputs.ClassProbs
        init_args: {task: jets_classification, stream: jets}
```

### 2. An export sink

The sink assembles the **output manifest** — which graph values become ONNX
outputs, and what each one is called. It NAMES nothing itself: every module
minting `outputs.*` leaves declares its own names and dtypes through
`manifest_fields(mode)`, and the sink collects them. `ClassProbs` inherits
`pb`/`pc`/`pu` from its source task, `Combination` names itself, and
`MaskFormerObjects` derives `leading_objects_<target>` from the regression
task's `targets`. Each suffix is prefixed with the model name — `GN2v2_pb`,
`GN2v2_pc`, `GN2v2_pu`.

Most configs declare no sink at all: `salt export` wires an `OnnxExportSink`
over the section's export-mode leaves for you. Declare one in the top-level
`outputs:` section (or, for one deprecation window, under `callbacks:`) only
when the command would not wire it — e.g. a section that opts out of `export`
while the model graph mints the tuple:

```yaml
outputs:
  run_tasks: {class_path: salt.outputs.RunTaskOutput, init_args: {tasks: [...]}}
  onnx_export: {class_path: salt.outputs.OnnxExportSink}
```

An explicit `outputs:` leaf list inside `init_args` is a hard error — add
`manifest_fields(mode)` to the producer instead. To narrow what one sink takes,
give it `consumes:` (fnmatch patterns over the `outputs.*` leaf key).

A sink is excluded from the section's column ordering, so where it sits among
the writers makes no difference. `export:` has no `outputs:` key at all — that
block describes the graph's input signature and the model identity, and putting
a manifest in it is a hard error.

### 3. The `export:` block

The block names the graph *inputs* and the model:

```yaml
export:
  model_name: GN2v2
  track_selection: r22default
  inputs:
    - {port: inputs.jets}
    - {port: inputs.tracks, sequence: true}
```

| Key | Meaning |
| --- | --- |
| `model_name` | prefix of every output name, and the ONNX `doc_string` |
| `track_selection` | Athena-side track selection, used to derive metadata names |
| `inputs` | the graph inputs, **in positional order** |
| `rename` | manifest suffix renames, applied before `combine` |
| `combine` | new outputs built from existing ones inside the graph |

Each `inputs:` entry describes one tensor:

| Key | Default | Meaning |
| --- | --- | --- |
| `port` | — | the bundle key it feeds, `inputs.<stream>` |
| `name` | `<stream without trailing 's'>_features` | the ONNX graph input name |
| `sequence` | `false` | variable-length `[L, F]` rather than global `[1, F]` |
| `dyn_axis` | `n_<stream>` | name of the dynamic length axis |
| `athena_name` | derived | the `gnn_config` metadata name |
| `alias` | — | feed this port from another entry's tensor |

`sequence: true` gives the input a dynamic length axis and **no batch
dimension** — the tensor is `[L, F]`, "this object's L constituents, F
variables each". `sequence: false` is a per-object global vector, `[1, F]`.

### Naming rules

`model_name` **must not contain `_` or `-`**. Athena builds output variable
names by concatenating the model name with each output suffix, and a separator
inside the model name makes those names ambiguous. The run's `name:` is
unrestricted; only the export name is validated.

Beyond that, whether naming is yours to choose depends on the consumer:

- **For Athena** (`FlavorTagInference`, and so the TDD) the names are *not*
  yours. Athena derives them and looks them up in the graph. Copy the
  convention from a shipped config rather than inventing one —
  [Deploy in the TDD](tdd.md) covers every rule and every way it bites.
- **For a self-written consumer** the name is documentation. `jet_kinematics`
  says more than `jets_features` when the payload is `[pt, eta, phi, m]`.

??? warning "`track_selection` must match your training samples"

    The value must be one of the selections defined in Athena's
    `DataPrepUtilities.cxx`, and it must match the selection applied when the
    training samples were dumped. It also drives the sequence node names in the
    metadata, which Athena parses with a regular expression — an unmatched name
    is a configuration-time error, not a silent one.

## Preview the manifest

`--manifest` needs no checkpoint and fails in seconds on a malformed export
block, so run it first:

```bash
salt export --manifest -c config.yaml
```

```text
ONNX output manifest (folded conversion nodes, model_name=GN2v2):
  GN2v2_pb  float32  folded conversion node (outputs.* leaf)
  GN2v2_pc  float32  folded conversion node (outputs.* leaf)
  GN2v2_pu  float32  folded conversion node (outputs.* leaf)
```

## Export

```bash
salt export \
    --ckpt_path logs/<run>/ckpts/<checkpoint>.ckpt \
    -c config.yaml \
    --output GN2v2.onnx
```

`-c/--config` is repeatable and the configs deep-merge left to right, exactly
as `salt fit` stacks them. This is the supported way to add an `export:` block
to a run that was trained without one — keep the run config untouched and stack
a small export-only file on top. If you omit `-c` entirely, the config is
inferred from the checkpoint's run directory.

Other options worth knowing: `-n/--name` overrides `model_name`,
`--set KEY=VALUE` applies ad-hoc config overrides, and `-o/--overwrite`
replaces an existing file.

### The `--check` sweep

`--check` is **on by default**. It runs the traced torch model and the exported
ONNX graph over random inputs at many sequence lengths and compares them. Tune
it with `--trials` (draws per length), `--max-length` (lengths swept) and
`--float-atol` (tolerance, default `1e-4`). `--no-check` skips it.

This catches tracing bugs. It does *not* catch deployment bugs — it feeds both
sides the same tensor, so it can say nothing about whether *you* will build
that tensor correctly. That is what the parity checks in the following pages
are for.

!!! tip "Export works on configs that `salt graph validate` cannot handle"

    `salt graph validate` calls `Reader.prepare()`, which some readers cannot
    do without a file bound — the ROOT-reader family in particular.

    Export is unaffected. It never builds a dataset: it reconstructs the model
    from the checkpoint and traces it, so it needs no source file and runs
    happily on exactly the config `graph validate` rejects. You do not need to
    restructure a config in order to export it.

??? tip "Exporting a model trained with `--compile`"

    Nothing special is needed. salt compiles each graph module **in place**, so
    the module tree is unchanged and a checkpoint written under `--compile` has
    exactly the same `state_dict` keys as one written without it. Export it
    directly.

## What the model file carries

salt writes a `gnn_config` JSON blob into the ONNX metadata:

| Key | What it is for |
| --- | --- |
| `model_name` | prefix of every output name |
| `input_sequences` | per sequence stream: node name and the **ordered** variable list |
| `inputs` | the same, for global (non-sequence) inputs |
| `output_names` | what each output tensor means |
| `onnx_model_version` | which metadata schema Athena should parse |

**The ordered variable list is the contract that matters.** It tells a consumer
which column is which. Read it from the file; do not hardcode it.

Reading it takes five lines:

```python
import json
import onnxruntime as ort

sess = ort.InferenceSession("GN2v2.onnx", providers=["CPUExecutionProvider"])
cfg = json.loads(sess.get_modelmeta().custom_metadata_map["gnn_config"])
print(cfg["output_names"])
```

??? info "Athena ships its own metadata dumper"

    After setting up Athena, `get_onnx_metadata` prints the same blob from the
    command line, including the normalisation values for models that carry
    them.

!!! warning "Three things the metadata does NOT record"

    1. **Which branch each variable came from.** It carries variable *names*
       (`pt`, `eta`, ...); the mapping to ntuple branches lived in the training
       reader's `branches:` block. A consumer must supply it.
    2. **Units.** If the model was trained on GeV and your input serves MeV,
       nothing will tell you — the scores will just be wrong.
    3. **Padding and truncation semantics.** The training reader's `truncate:`
       and its padding convention are not in the file.

    All three are part of the deployment contract and all three fail silently.
    Carry them alongside the model, and check them by parity.

### Renaming and combining outputs

`rename:` maps manifest suffixes to new ones. `combine:` builds a *new* output
as a weighted sum of existing global float outputs, computed inside the traced
graph:

```yaml
export:
  rename: {poldnamebjet: pb, poldnamecjet: pc}
  combine:
    - {name: plight, inputs: {pquark: 1.0, pgluon: 1.0}}
    - {name: pquark, inputs: {pud: 0.5, pgluon: 1.5}}
```

`rename:` is applied first, so `combine:` refers to the new names.

## Validate the exported model

The `--check` sweep proves the export traced correctly. It does not prove the
*deployment* is correct, and those are different claims. Before trusting a
model anywhere, run it through its real consumer and compare against
`salt test` on the same events:

- self-written consumer → [Run it in Python](python.md#check-it-against-salt)
- Athena / TDD → [Deploy in the TDD](tdd.md)
- analysis framework → [Deploy in easyjet](easyjet.md)

??? info "What level of discrepancy is expected?"

    Agreement within `1e-6` on the output probabilities is the usual bar —
    approximately floating-point precision. One or two objects at `1e-5` is
    fine. Larger discrepancies almost always come from one of:

    - not dumping at full precision on the consumer side;
    - not running `salt test` with `--trainer.precision 32` when the model was
      trained at lower precision;
    - not writing salt evaluation scores at full precision;
    - a runtime optimisation in torch, e.g.
      [`set_float32_matmul_precision`](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision).

    A discrepancy far larger than that is not precision at all — it is a wiring
    bug, and the input tensor is where to look.

## Deploying FTAG models in Athena

For registering a tagger in central reconstruction, see
[the FTAG documentation](https://ftag.docs.cern.ch/reco_algs/taggers/deploy/).
For a step-by-step walkthrough of running one in the training-dataset-dumper,
see [Deploy in the TDD](tdd.md).
