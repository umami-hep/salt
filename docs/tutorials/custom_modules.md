# Write your own modules

The [MNIST](mnist.md) and [MNIST part 2](mnist_cnn.md) tutorials each wrote one
piece of custom code: a reader, then a model module. This tutorial extends
both, it does not replace them. It builds a second data pipeline (a
`Processor` and a second `Reader`) and a second model module, contrasts the
two `bind` methods directly, wires everything into one config, and adds a
task head. If you have not done the MNIST tutorials yet, do part 1 first;
several sections here point back at it instead of repeating it.

Read [`modules/index.md`](../modules/index.md) alongside this tutorial for the
shared vocabulary (bundle namespaces, `Mode`, `TensorSpec`), and
[`modules/model.md`](../modules/model.md) / [`modules/data.md`](../modules/data.md)
for the full method-by-method reference each new module here only summarises.

Because in-session execution is blocked for this tutorial's own build (no
`.venv`, sandboxed test runner), the code below has **not** been run as part
of writing this page. It is checked against the source of the classes it
subclasses, the same way the rest of this reference set is. Treat command
output blocks as the *shape* the command produces, not a captured transcript.

## 1. Set up your workspace

```bash
mkdir custom-modules-tutorial && cd custom-modules-tutorial
mkdir my_modules && touch my_modules/__init__.py
export PYTHONPATH=$PWD
```

`export PYTHONPATH=$PWD` is required, not optional, for every command below:
`class_path` resolution is a normal Python import, and salt does not add your
working directory to `sys.path` implicitly. Running from a container instead
of a host shell needs the same variable passed through explicitly, since a
container does not inherit your host environment:

```bash
apptainer exec --env PYTHONPATH=$PWD <image> salt ...
```

Now generate a small training file. This is the same recipe as
[cli.md Quickstart](../cli.md#quickstart); see that page if you want the full
explanation of what it writes. Paste this as-is:

```bash
mkdir data
python -c "
from pathlib import Path
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.testing.inputs import write_dummy_file
write_parity_norm_dict('data/norm_dict.yaml', 'data/class_dict.yaml')
write_dummy_file('data/train.h5', 'data/norm_dict.yaml')
"
```

This writes `data/train.h5`: 1000 synthetic jets with a `jets` group
carrying, among other fields, a float `pt` column and an integer
`flavour_label` column (values 0, 1, 2). Both are used below: the new
`Processor` reads `pt`, and the task head added in step 6 predicts
`flavour_label`.

## 2. Derive a new column: your first data module, a Processor

A `Processor` (`salt.data.base.Processor`) is required to transform one
batch: it reads bundle keys its `declare_io` declared as `requires` and
returns only the keys it declared as `produces`. It never touches a file.
The full method-by-method reference is [`modules/data.md`](../modules/data.md);
this section shows one built by doing.

The processor below reads the `pt` field out of `raw.jets` (the reader's
structured per-jet record) and derives one new column, `log_pt`, published
as a new one-column stream `inputs.jets_logpt`:

```python
"""A salt Processor deriving log1p(pt) as a new jets-derived column."""

from __future__ import annotations

import numpy as np

from salt.data.base import Processor
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec


class LogPt(Processor):
    """Derive ``inputs.jets_logpt`` (log1p of ``pt``) from ``raw.jets``."""

    def __init__(self, stream: str = "jets", out_stream: str = "jets_logpt") -> None:
        super().__init__()
        self.stream = stream
        self.out_stream = out_stream

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({
                f"raw.{self.stream}": TensorSpec(kind="data", fields=("pt",)),
            }),
            produces=unflatten_spec({
                f"inputs.{self.out_stream}": TensorSpec(
                    shape=("B", 1), dtype="float32", kind="data", fields=("log_pt",)
                ),
            }),
        )

    def process(self, batch, rows, mode) -> dict[str, np.ndarray]:
        del rows, mode
        raw = batch.get(f"raw.{self.stream}")
        log_pt = np.log1p(raw["pt"]).astype(np.float32)[:, None]
        return {f"inputs.{self.out_stream}": log_pt}
```

Save this as `my_modules/processors.py`.

Two things worth pointing out on a first `declare_io`:

- `requires` names `fields=("pt",)` on `raw.jets`, not the whole record. The
  reader's own `raw.jets` declaration carries no field list (it doesn't know
  yet which fields anyone wants), so each consumer names its own subset; this
  is what tells the reader which columns to actually read from disk.
- `produces` gives `inputs.jets_logpt` a **concrete** shape, `("B", 1)`, not a
  symbolic one, because a `Processor` computes its own output width itself:
  there is nothing here for the planner to unify.

Now check it without touching real data. Salt's graph tooling accepts a
lightweight config, a flat `modules:` mapping with no `data:`/`model:`
split, for exactly this kind of isolated check:

```yaml
# graph_check.yaml
modules:
  reader:
    class_path: salt.data.H5StructuredReader
    init_args:
      groups:
        jets: {global_object: true}
      filename: data/train.h5
  logpt:
    class_path: my_modules.processors.LogPt
```

This config has no `salt.data.Features` module, and that is specific to
this example, not a general pattern: `LogPt` reads `raw.jets` directly and
mints its own `inputs.jets_logpt`, so nothing else needs to produce
`inputs.*` here. A config with a normaliser or embed reading a
`Features`-derived column still needs `Features` in `data.modules`; see the
[minimum graph that compiles](../configuration.md#the-minimum-graph-that-compiles).

```bash
salt graph validate -c graph_check.yaml
```

This prints one `OK [mode=<MODE>] <N> steps, <M> edges, plan_hash=<hash>` line
per primary mode, the same shape as the MNIST tutorials' `salt graph
validate` output ([cli.md salt graph](../cli.md#salt-graph) has the full
format). With only a reader and a processor and nothing consuming
`inputs.jets_logpt` yet, expect a warning-level unconsumed-output finding
alongside the `OK` lines; that finding goes away once step 4's model module
consumes the key. That warning, and not a hard failure, is specific to
`graph_check.yaml` being a standalone toy `modules:` config with no
`sinks:`: the planner skips demand pruning entirely when no sink list is
given (`salt/graph/planner.py:460`), so every configured module compiles
and an unconsumed output only ever surfaces through the non-raising
`deadcode()` warning path. In a full trainer config, sinks are derived
(`loss.total` for FIT/VAL, `preds.*` and writer demand for TEST/ONNX), and
the same unconsumed output is a hard `AllModesDeadError` instead, not a
warning; wire a consumer before you add a module in that setting. See
[`modules/index.md#liveness-every-module-must-reach-a-sink`](../modules/index.md#liveness-every-module-must-reach-a-sink)
for both regimes side by side.

Now ask salt why the new key exists. This is the first real use of
`salt graph why` in these docs ([cli.md salt graph why](../cli.md#salt-graph-why)
has the full command reference):

```bash
salt graph why -c graph_check.yaml --key inputs.jets_logpt
```

For a key that is present, this prints four lines: the key itself, its
`producer:` (here, `logpt (LogPt)`), its `spec:` (`kind=data,
shape=('B', 1), dtype=float32`), and its `consumers:`. For this config that
last line reads `none`, because nothing reads `inputs.jets_logpt` yet. That
is expected at this point in the tutorial, not a bug: `salt graph why` is
telling you exactly what step 4 is about to fix.

## 3. A reader for a different format

The MNIST tutorial's [`IdxReader`](mnist.md#2-write-the-reader) is the fuller
worked example of everything a `Reader` implements: `streams`, `with_source`,
`prepare`/`__len__`, `bind(ctx)`, `declare_io`, `read`. This section is
deliberately smaller, a reader for a two-column CSV file, to show the same
six pieces once more without repeating that walkthrough. It is not wired
into the rest of this tutorial; treat it as a second reference example, and
read [`modules/data.md`](../modules/data.md) for what each method is required
to do and why.

```python
"""A salt Reader for a tiny two-feature CSV format: x,y,label per row."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from salt.data.base import Reader, WorkerCtx
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec


class CsvReader(Reader):
    """Serve a two-feature CSV file as a single 'event' stream."""

    def __init__(self, filename: str | Path | None = None) -> None:
        super().__init__()
        self.filename = Path(filename) if filename is not None else None
        self._rows: list[tuple[float, float, int]] | None = None

    @property
    def streams(self) -> tuple[str, ...]:
        return ("event",)

    def with_source(self, filename, num: int = -1, vds_path=None, stage=None) -> "CsvReader":
        del num, vds_path, stage  # single-file reader: nothing stage-specific
        clone = CsvReader(filename=filename)
        clone.name = self.name
        return clone

    def prepare(self) -> None:
        if self._rows is not None:
            return
        with open(self.filename, newline="") as f:
            self._rows = [
                (float(r["x"]), float(r["y"]), int(r["label"])) for r in csv.DictReader(f)
            ]

    def __len__(self) -> int:
        self.prepare()
        return len(self._rows)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat = {
            "raw.event": TensorSpec(shape=("B",), kind="data"),
            "inputs.event": TensorSpec(shape=("B", 2), dtype="float32", kind="data"),
        }
        return IO(produces=unflatten_spec(flat))

    def bind(self, ctx: WorkerCtx) -> None:
        del ctx
        self.prepare()

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del mode
        chunk = self._rows[rows]
        batch = np.empty(len(chunk), dtype=np.dtype([("label", "i8")]))
        batch["label"] = [r[2] for r in chunk]
        inputs = np.array([[r[0], r[1]] for r in chunk], dtype=np.float32)
        return {"raw.event": batch, "inputs.event": inputs}
```

The shape is identical to `IdxReader`: `prepare`/`bind` do the file-touching
work exactly once (main-process probing, then per-worker setup), `read`
returns exactly the two keys `declare_io` promised, for one contiguous row
slice, and `with_source` clones the configured prototype onto a concrete
file without opening it. The only real difference from `IdxReader` is the
on-disk format `prepare` parses.

`declare_io`'s hard-coded `("B", 2)` on `inputs.event` is deliberate, not a
shortcut: a reader may not open or probe the file to size itself inside
`declare_io` (that step is config-only, before any file is touched, see
[`modules/index.md`](../modules/index.md#the-compile-then-run-split)), so a
column count a real format only reveals on read has to come from an
`init_arg` instead; see [`modules/data.md`](../modules/data.md) for the
pattern.

`with_source`'s signature carries four arguments a production reader must
actually respect, above all `num`: it is the per-stage row cap from
`data.num_train`/`data.num_val`/`data.num_test`, and a clone that drops it
silently disables those settings. `CsvReader` above discards all four
(`del num, vds_path, stage`) because this reader has nothing stage-specific
to do with them; a reader over a real, larger corpus should not copy that
line as-is.

## 4. Your first model module

A model module is a `salt.model.base.SaltModelModule`. Its life is
config-only `__init__`, then `declare_io`, then `bind(schema)`, then
`forward`, the same four-phase shape [MNIST part 2](mnist_cnn.md#1-the-model-module-lifecycle)
walked through for `MnistCNN`. This one is smaller: a single `nn.Linear`
consuming the processor's new key.

```python
"""A model module embedding the derived jets_logpt feature."""

from __future__ import annotations

from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.modules import ResolvedSchema, SaltModelModule


class LogPtEmbed(SaltModelModule):
    """Embed ``normed.jets_logpt [B, F]`` as ``embed.jets_logpt [B, out_dim]``."""

    def __init__(self, stream: str, out_dim: int) -> None:
        super().__init__()
        self.stream = stream
        self.out_dim = out_dim
        self.linear: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({
                f"normed.{self.stream}": TensorSpec(shape=("B", "F:jets_logpt"), dtype="float32"),
            }),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(shape=("B", self.out_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        n_in = schema.width(f"normed.{self.stream}")
        self.linear = nn.Linear(n_in, self.out_dim)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        del mode
        x = b.get(f"normed.{self.stream}")
        assert self.linear is not None, "forward before bind()"
        return {f"embed.{self.stream}": self.linear(x)}
```

Save this as `my_modules/model.py`.

`declare_io` requires `normed.jets_logpt` with a **symbolic** last dim,
`"F:jets_logpt"`, because this module does not know that width yet; it
produces `embed.jets_logpt` with a **concrete** shape, `("B", out_dim)`,
because `out_dim` is a config value this module owns. `bind` then asks the
`ResolvedSchema` for the concrete width the planner resolved for
`normed.jets_logpt` and builds the one `nn.Linear` from it. No width is
written twice: `out_dim` is configured once, and the input width is derived,
never repeated in YAML.

### The two `bind` methods, side by side

You have now written both. They share a name and nothing else:

| | `SaltDatasetModule.bind` (data side) | `SaltModelModule.bind` (model side) |
|---|---|---|
| Signature | `bind(self, ctx: WorkerCtx) -> None` | `bind(self, schema: ResolvedSchema) -> None` |
| Called by | `SaltDataset`, once per (worker process, plan) | `bind_all`, once per model, after every mode's plan compiles |
| Purpose | per-worker lazy setup: open a file handle, allocate a buffer | build the `torch.nn` layers this module owns, sized from resolved widths |
| May touch files | yes, the only data-side hook that may | no |

`CsvReader.bind(ctx)` above opens nothing (it delegates to `prepare`, which
already ran); `LogPtEmbed.bind(schema)` builds an `nn.Linear`. Confusing the
two is an easy mistake because the name and the four-line shape match; the
argument type is the tell. Full detail: [`modules/model.md#bind`](../modules/model.md#bind)
and [`modules/data.md#bind`](../modules/data.md#bind).

## 5. Wire both into one config, and validate

Four pieces go into one config: the `Processor` and a `Labels` processor
under `data.modules`; the model module and a task head under
`model.init_args.modules`; and an `outputs:` section naming the task head
so it renders. `class_path` is a plain Python import path in every case, so
`my_modules.processors.LogPt` and `my_modules.model.LogPtEmbed` must be
importable, which is exactly what `export PYTHONPATH=$PWD` from step 1
arranges. The task head reads the embedding `LogPtEmbed` produces and
predicts the dummy file's `flavour_label` column; step 6 explains how it is
wired in full.

```yaml
# config.yaml
name: CustomModulesDemo

data:
  batch_size: 32
  num_workers: 0
  train_file: data/train.h5
  val_file: data/train.h5
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          jets: {global_object: true}
    logpt:
      class_path: my_modules.processors.LogPt
    labels:
      class_path: salt.data.Labels

model:
  class_path: salt.model.SaltModule
  init_args:
    lrs: {initial: 1.0e-4, max: 1.0e-3, end: 1.0e-5, pct_start: 0.1}
    optimizer: AdamW
    modules:
      norm:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args:
          streams: [jets_logpt]
          global_object: jets_logpt
      logpt_embed:
        class_path: my_modules.model.LogPtEmbed
        init_args:
          stream: jets_logpt
          out_dim: 8
      logpt_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: jets
          input: embed.jets_logpt
          sequence: false
          label: flavour_label
          class_names: ["0", "1", "2"]
          loss: torch.nn.CrossEntropyLoss
          dense: {hidden_layers: [16]}
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [logpt_classification]}

trainer:
  max_epochs: 1
  precision: 32-true
  logger: false
  default_root_dir: run
```

**This `norm` module only works because there is exactly one derived
stream.** `global_object` takes a single `str`
(`salt/model/modules/norm.py:275`), and that one stream is declared rank 2,
`("B", F)`, while every other stream named in `streams:` is declared rank 3,
`("B", T, F)` (`norm.py:305-315`). A second derived per-row stream cannot
join `streams: [jets_logpt]` above and share this instance: it needs its
own `MaskedInputNormaliser`, with itself as `global_object`, two instances
side by side rather than one list with two entries. See the shape rule and
the two-instance example on
[`modules/model.md#shipped-model-modules`](../modules/model.md#shipped-model-modules).

`labels: {class_path: salt.data.Labels}` needs no `init_args`: it learns
which streams it may serve from the reader at bind time, then narrows to
the `labels.<stream>.<label>` keys the task head below actually demands.
`loss: {class_path: salt.model.modules.LossSum}` is legal only because
`logpt_classification` now produces a `losses.*` key for it to collect;
without a task head, `LossSum` narrows to an empty list and
`SaltModule.__init__` raises a `ConfigError`.

Validate the whole pipeline:

```bash
salt graph validate -c config.yaml
```

Then render it as an image, exactly as [MNIST part 2](mnist_cnn.md#7-finale-render-the-graph)
does:

```bash
salt graph plot -c config.yaml -o graph.svg
```

`salt graph plot` writes a DOT sidecar next to the requested path, then
shells out to Graphviz `dot`. The rendered file is a PNG regardless of the
extension you pass (the `dot -Tpng` invocation names its output after
whatever path you gave), plus a sibling `graph.pdf`; pass `-o graph.png`
instead if you would rather the filename matched the format.

Finally, smoke-test the wiring against real data without training a full
run. The flag is always spelled with its namespace, `--trainer.fast_dev_run`,
never the bare form:

```bash
salt fit --config config.yaml --trainer.fast_dev_run 2
```

This runs two batches through the whole pipeline: reader, `LogPt`, `Labels`,
normaliser, `LogPtEmbed`, `logpt_classification`, `LossSum`, and Lightning's
optimizer step, then exits without writing checkpoints. It is the fastest
way to find a wiring mistake before committing to a real training run.

## 6. How the task head is wired

`logpt_classification` in the config above is a `ClassificationTaskModule`
(`salt.model.modules.tasks.ClassificationTaskModule`), one of the task
modules built on the shared `_TaskModuleBase`. It reads `embed.jets_logpt`
(`input: embed.jets_logpt`) and produces two bundle keys, both derived from
its `stream` and its config key name:

- `preds.jets.logpt_classification`, its prediction (`pred_key`, spelled
  `preds.<stream>.<instance-name>`, `salt/model/modules/tasks/base.py:96-99`)
- `losses.logpt_classification`, its loss term (`loss_key`, spelled
  `losses.<instance-name>`, `tasks/base.py:101-104`); `LossSum` collects it
  automatically because it is the only module producing under `losses.*`

`input:` is set explicitly because a task head's default `input_key` is
`encoded.<stream>` (`tasks/base.py:57`), the encoder's output for that
stream. `logpt_classification` reads an embedding instead of the encoder
output, so `input:` overrides the default to name `embed.jets_logpt`
directly.

The `outputs:` block in the config above tells `RunTaskOutput` which task
instances to render into the eval H5 and the ONNX manifest; see
[outputs.md](../outputs.md#declaring-outputs-the-outputs-section) for the
full mechanics of `get_output` and the manifest it builds.

### Where the label comes from

`label: flavour_label` names a column, not a file read you configure
directly. The demand-driven `Labels` processor, declared explicitly under
`data.modules` above (the same way [MNIST part 1](mnist.md#3-write-the-config)
declares it, not added for you), serves `labels.<stream>.<label>` keys, but
only the ones some task head actually demands; it reads them out of
`raw.<stream>`, and the dummy file's `jets` group already carries
`flavour_label`.

A task head's `label_key` is spelled `labels.<stream>.<label>`
(`tasks/base.py:106-109`), so its `stream` selects where the label comes
from: `logpt_classification` sets `stream: jets` because the reader serves
the stream `jets`, not `jets_logpt` (`Labels._parse_targets` rejects any
stream outside the reader's served set, `salt/data/processors/labels.py:131-140`).
`input:` names the bundle key the head's input tensor comes from instead,
`embed.jets_logpt`. The two need not match: the label is a jets-level truth
column, and the input is a derived per-jet embedding published under its
own stream name.

Ask salt to show all of this at once:

```bash
salt graph why -c config.yaml --key preds.jets.logpt_classification
```

This prints the same four-line shape as step 2's `salt graph why`, but for
the head's prediction rather than an unconsumed intermediate key: the key
itself, its `producer:` (`logpt_classification (ClassificationTaskModule)`),
its `spec:`, and its `consumers:`.

## 7. When it goes wrong

Three real error messages, and what each means.

**`has no declare_io()`** (`salt/model/base.py:69`). You get this by
subclassing `SaltModelModule` (or `SaltDatasetModule`) and forgetting to
override `declare_io`, for example if `LogPtEmbed` above had left that
method out entirely. The base implementation exists only to raise this,
naming your class:

```text
LogPtEmbed has no declare_io() — either it is a manifest-only outputs:
section writer (never entered into the plan) or it is missing a
declare_io() override
```

Fix: implement `declare_io`, returning an `IO` built from your module's
`requires`/`produces`.

**`no statically resolved width for bundle key`** (`salt/model/bind.py`). You
get this by calling `schema.width(key)` in `bind` for a key the planner
never resolved a concrete last-dim width for, for example a typo in the key
name, or a key whose shape is `None` everywhere it is declared:

```text
salt.model.bind.BindError: no statically resolved width for bundle key
'normed.jets_lopgt' — the key is either absent from the compiled plans or
its last dim never binds to a concrete size; nearest: normed.jets_logpt
```

Fix: check the key spelling against what your `declare_io` (or the
producing module's `declare_io`) actually declares; the nearest-key
suggestion usually names the real key directly.

**`data-graph entries must subclass SaltDatasetModule`** (`salt/data/datamodule.py:270`).
You get this by wiring a class under `data.modules` that is not a `Reader`
or `Processor`, for example if `LogPt` above had subclassed plain `object`
instead of `Processor`:

```text
salt.graph.errors.ConfigError: module 'logpt' (LogPt) is not a
SaltDatasetModule — data-graph entries must subclass SaltDatasetModule;
wrap or extend it
```

Fix: subclass `Reader` or `Processor` (both are `SaltDatasetModule`
subclasses), not a bare object or an unrelated base class.

## What you just proved

- **Two data modules, one shared base.** A `Processor` transforms an
  existing batch; a `Reader` originates one from disk. Both are
  `SaltDatasetModule` subclasses, both are checked the same way.
- **`declare_io` is a function of config, run before any data exists.**
  `salt graph validate` and `salt graph why` compiled and inspected the
  whole pipeline, reader through task head, without opening `data/train.h5`
  once.
- **The two `bind` methods are not the same method.** One runs per worker
  and may touch files; the other runs once per model and builds layers from
  resolved widths. Confusing them is the single most common naming trap in
  this API.
- **A working model is a graph of small, swappable pieces.** `LogPtEmbed`
  slotted in next to `StreamEmbed` and `MaskedInputNormaliser` the same way
  `MnistCNN` slotted in next to salt's own modules in part 2, wired by
  nothing more than a `class_path` pointing into your own workspace.
