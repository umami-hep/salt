# Deploy in easyjet

The [event-level classifier tutorial](../tutorials/event_classifier.md) trains
a model that scores a whole *event* rather than a jet. This page takes that
model into an analysis framework, so the score is computed during ntuple
production and lands in the output tree with everything else.

It is the event-level counterpart to [Deploy in the TDD](tdd.md), and it
differs from it in one way that shapes everything below: there is no generic
framework code waiting to serve your model. A jet tagger has
`FlavorTagInference` to do the wiring; an event-level model has whatever you
write. See [the two kinds of consumer](export.md#two-kinds-of-consumer) for why
that makes validation non-negotiable rather than nice to have.

This page assumes you have exported the model ([Export to ONNX](export.md)) and
can run it in python ([Run it in Python](python.md)) — the python route is what
you will validate the framework against.

## Do you actually need this?

Running post-hoc over an existing ntuple is simpler, needs no framework, and
answers most questions. Move the model into the framework when:

- the score has to drive a **selection** during production;
- the score must be recomputed **per systematic variation**, because its inputs
  vary;
- the ntuple is the deliverable, and adding a post-processing pass to every
  consumer of it is not acceptable.

Otherwise, stay in python.

## The algorithm

A generic, model-agnostic algorithm belongs in the framework's **core** package
rather than in any one analysis package, so every analysis can point it at its
own model through configuration. In easyjet that is `EasyjetHub`:

```
EasyjetHub/src/EventOnnxDecoratorAlg.{h,cxx}
EasyjetHub/python/algs/event_onnx_decorator_config.py
```

`EventOnnxDecoratorAlg` is an `AthReentrantAlgorithm` that reads a jet
container, builds a single `[n_jets, n_variables]` float tensor, runs an ONNX
Runtime session, and writes one `EventInfo` decoration per model output.

Nothing in it knows about any particular model. The graph input name, the
output names, and the expected number of feature columns are all read from the
ONNX file in `initialize()` and cross-checked against the configuration, so a
mis-configured feature list refuses to start rather than producing quietly
wrong scores.

!!! note "ONNX Runtime is already there"

    The AthAnalysis release ships the ONNX Runtime headers and library, and
    they arrive transitively through packages easyjet already links. An
    algorithm can use `Ort::Session` directly with **no extra CMake
    `find_package`** — the only addition this algorithm needed was
    `PathResolver`, for locating the model file.

## Configuring it

The algorithm is off unless a RunConfig turns it on. The whole contract between
the ntupler and the model is one block:

```yaml
event_onnx:
  enable: true
  modelPath: 'MyPackage/EventTagger.onnx'
  jets: ''                       # empty: the config's small-R jet container
  variables: ['pt', 'eta', 'phi', 'm']
  variableScales: [0.001, 1.0, 1.0, 0.001]
  truncate: 10
```

| Key | What it is |
| --- | --- |
| `modelPath` | absolute path, or `<Package>/<file>.onnx` via `PathResolver` |
| `jets` | jet container to feed; empty means the analysis' small-R jets |
| `variables` | column order the model was trained with |
| `variableScales` | per-column multiplier, applied before inference |
| `truncate` | keep at most this many jets, in container order |
| `outputDecorations` | rename the outputs; empty means use the graph's own names |

Three of these are exactly the things
[the metadata does not record](export.md#what-the-model-file-carries), restated
on the framework side:

- **`variables`** is the ordered variable list, copied from `gnn_config`.
- **`variableScales`** is the unit conversion. xAOD serves MeV; a model trained
  on GeV needs `0.001` on `pt` and `m`. Nothing detects this — the scores are
  simply wrong at the wrong scale.
- **`truncate`** must match the `truncate:` the training reader used.

Each `variables` entry is either a jet kinematic (`pt`, `eta`, `phi`, `m`, `e`,
`rapidity`) or the name of a float aux decoration on the jet, which the
algorithm reads through a systematics-aware decoration handle.

!!! warning "List-valued RunConfig keys are *appended*, not replaced"

    RunConfig fragments deep-merge, and for list values that merge
    concatenates. A non-empty default for `variables` in the base config would
    be appended to whatever your analysis sets, producing a doubled feature
    list. The shipped defaults are therefore `[]`, so yours is the only list
    that reaches the algorithm.

    The start-up cross-check catches it either way — the algorithm compares the
    configured column count against the graph's and refuses to run on a
    mismatch — which is precisely the argument for having that check.

### Getting the score into the ntuple

The decorations are ordinary `EventInfo` variables, so they are written out
like any other:

```yaml
ttree_output:
  extra_output_branches:
    - "EventInfo.EventTagger_pbackground_%SYS% -> EventTagger_pbackground_%SYS%"
    - "EventInfo.EventTagger_psignal_%SYS% -> EventTagger_psignal_%SYS%"
```

The decoration names come from the model's own ONNX output names, so they are
whatever `model_name` and the export sink produced. `%SYS%` resolves to
`_NOSYS` for the nominal and to the variation name for each systematic.

## What the algorithm does per event

The tensor build has to match the training reader exactly, and it follows the
same three rules as
[the python route](python.md#ordering-truncation-padding):

- **order** — one row per jet, columns in the configured `variables` order,
  row-major;
- **truncation** — the first `truncate` jets in *container* order; the
  algorithm does not re-sort, and neither did the reader;
- **no padding** — the graph's length axis is dynamic, so each event is fed at
  its true length.

Events with **no jets** are not run through the model at all: there is nothing
to pool over, so the decorations get a sentinel (`-1` by default) and the
session is skipped. A python consumer validating against this must apply the
same rule, or the comparison is over sentinels.

The whole loop is per systematic, because the jet container varies with the
systematics and so does the score:

```cpp
for (const auto &sys : m_systematicsList.systematicsVector()) {
  // retrieve EventInfo and the jets for THIS variation
  // build the tensor, run the session
  // m_scores[i].set(*event, value, sys);
}
```

With `doSystematics` false, only the nominal is evaluated.

### If you are writing the equivalent elsewhere

The shape of such an algorithm, in any framework:

- **Create the session in `initialize()`**, once, and resolve the model path
  there too.
- **Resolve every input and output name in `initialize()`** and cache them as C
  strings. Framework coding rules commonly bar string operations and
  string-keyed lookups from the event loop; this is why they are easy to obey —
  the names never change.
- **Cross-check the configuration against the graph in `initialize()`.**
  Comparing the configured column count against the graph's declared input
  width costs three lines and turns an entire class of silent-wrong-answer bugs
  into a start-up error.
- **In `execute()`**, build the `[L, F]` tensor, run the session, and write the
  outputs as event-level decorations.
- **Mind reentrancy.** `Ort::Session::Run()` is not `const`, but ONNX Runtime
  documents concurrent `Run()` calls on one session as safe provided each call
  owns its inputs and outputs — exactly this usage. A shared session with an
  explicit thread-safety annotation beats rebuilding one per event.

### Shipping the model file

Two options, suiting different stages:

- **An absolute path**, while the model is still moving. No build step, no
  repository churn; the algorithm accepts it directly.
- **A calibration-area path** (`<Package>/<file>.onnx`) for anything durable.
  Install the file with `atlas_install_data(data/*)` and `PathResolver` finds
  it. Model files are binaries — check the repository's conventions for large
  files before committing one.

## Validate it: the parity gate

This is the check that matters, and the output ntuple already contains both
halves of it: the jet kinematics the algorithm fed the model, and the score it
wrote back.

So read the ntuple, rebuild the tensor with an independent implementation (the
[python route](python.md)), run the model again, and compare against the score
branch:

```python
posthoc = ...          # scores from the python route, over the ntuple branches
inframework = tree["EventTagger_psignal_NOSYS"].array()
print("worst |in-framework - python|:", np.abs(posthoc - inframework).max())
```

Run it over two samples with genuinely different jet content, so that a mistake
which happens to cancel on one does not pass unnoticed.

!!! success "What a passing gate looks like"

    Over 20 000 events each of a four-b-jet signal sample and a top-pair
    background sample, the in-framework and python scores agreed to **exactly
    zero** difference — not float32 noise, bitwise identity.

    That is what to expect on this route, and it is a stronger result than the
    `1e-7` typical of a `salt test` comparison. Both sides read the *same*
    float32 values from the *same* file, build the *same* tensor, and hand it
    to the *same* runtime; there is nothing left to disagree about.

A non-zero result here is not a precision problem to be absorbed by a looser
tolerance. It means the two sides are not doing the same thing — check
ordering, truncation and scaling, in that order, before suspecting anything
else.

!!! note "The parity gate does not check the physics"

    It proves the framework computes what the model says, on the inputs you
    gave it. It says nothing about whether the model is any good, or whether
    the features you configured are the ones it was trained on in spirit as
    well as in name. Look at the score distributions too, on samples where you
    know roughly what to expect.

## What to take away

- An event-level model needs framework code that a jet tagger gets for free,
  and the generic version of that code belongs in the framework's core package.
- Everything model-specific is configuration: the model file, the ordered
  variable list, the unit scaling, the truncation.
- The ONNX file carries none of the variable-to-branch mapping, the units, or
  the truncation. Those live in your config, and they fail silently.
- Validate against an independent implementation on the same events. On this
  route the bar is exact agreement; anything else is a bug.
