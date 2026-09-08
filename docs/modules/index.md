# Writing modules

A salt module is a class instantiated from one `class_path` + `init_args` entry
in a config, named by the dict key it sits under. Before any data is read,
salt compiles every mode's static plan: it asks each configured module which
bundle keys it reads and writes, orders them, and prunes the ones nothing
demands. Only after that compile does training or evaluation touch a file.

There are two families of module. **Model-side** modules
(`SaltModelModule`, `salt/model/base.py:29`) are the `nn.Module` subclasses
that embed, encode, pool, compute a loss, or write a prediction; they live
under `model.init_args.modules` in a config. **Data-side** modules
(`SaltDatasetModule`, `salt/data/base.py:113`) are the readers and processors
that turn files into batches; they live under `data.modules`. This page
covers what both sides share. [`model.md`](model.md) covers
`SaltModelModule` method by method, and [`data.md`](data.md) covers
`SaltDatasetModule`, `Reader` and `Processor`.

Both families satisfy the same protocol: `GraphModule`
(`salt/graph/spec.py:416`) requires a `name` attribute plus a
`declare_io(mode)` method returning an `IO` of required and produced bundle
keys. `name` is not something you set: salt assigns it from the module's
config dict key before the compile starts. This protocol name does not
appear anywhere else in the docs, so it is worth remembering: whenever this
page says "a module", it means "something satisfying `GraphModule`".

## The compile-then-run split

Every mode gets its own plan, built in this order:

1. **Declare.** Every configured module's `declare_io(mode)` runs, once per
   mode. This step is config-only: no file is opened, no tensor is built, no
   network call happens. It is what makes `salt graph validate` and
   `salt graph why` usable before a run exists.
2. **Connect and prune.** The planner matches every require to a producer of
   the same `kind`, checks that every port is active in at least one mode,
   and removes any module nothing in that mode ultimately demands.
3. **Resolve widths.** Symbolic dims are unified across the surviving plan
   and `derived_widths` contributions are folded in to a fixpoint, producing
   the `ResolvedSchema` (model side only; see
   [`model.md#resolvedschema`](model.md#resolvedschema)).
4. **Bind.** Each surviving module's `bind` runs once, given whatever that
   side's second-phase argument is (`ResolvedSchema` for a model module,
   `WorkerCtx` for a data module). This is the first point either side may
   build something sized from the plan, and for a data module it is also the
   first point it may touch a file.
5. **Run.** `forward` (model side) or `read`/`process` (data side) runs once
   per batch, for as many batches as the run needs.

Everything in steps 1-3 happens for every mode, every time salt starts,
including a plain `salt fit`. This is why a config with a bad `class_path` or
a dangling required key fails immediately, before any data loading begins.

## The bundle

Every value a module reads or writes lives in one run-scoped bundle, addressed
by a dotted key (`KEY_SEP = "."`, `salt/graph/spec.py:36`). A key's first
component is its namespace, and the namespace says who is allowed to write it:

| Namespace | Written by | Holds |
|---|---|---|
| `raw.<stream>` | readers | the stream's fields exactly as read from disk |
| `inputs.<stream>` | `Features` (or a custom processor) | float32 materialised input columns |
| `masks.<stream>` | readers | the stream's boolean pad mask |
| `labels.<stream>.<field>` | `Labels` / `FtagLabeller` (or a custom processor) | per-jet or per-constituent truth labels |
| `meta.rows` | readers | the batch's `[start, stop)` row range |
| `normed.*` | a normaliser module | normalised model inputs |
| `embed.*` | an embedding module | per-stream embeddings |
| `encoded.*` | an encoder module | encoder outputs |
| `pooled.*` | a pooling module | pooled per-jet vectors |
| `preds.*` | a task module | predictions |
| `losses.*` | a task module | per-task scalar losses |
| `outputs.*` | an `outputs:` section writer | columns destined for the eval H5 / ONNX manifest |

A module may read from any namespace its `declare_io` requires, but it should
only ever produce into the namespace this table assigns it. The planner does
not enforce that ownership, so a module producing into the wrong namespace is
a design mistake the planner will not catch for you.

A normaliser or embed module that requires `inputs.<stream>` fails with
`ConnectivityError` when no `Features` processor (or your own producer) is
configured in `data.modules` to write that namespace; see
[Errors you will see](#errors-you-will-see) for the exact message.

## `Mode`

`Mode` (`salt/graph/spec.py:39`) is a `Flag`, not an enum: `FIT`, `VAL`,
`TEST`, `ONNX`, plus two combined aliases, `TRAINING = FIT | VAL` and
`ALL = FIT | VAL | TEST | ONNX`. Because it is a flag, test membership with
`&`, not `==`:

```python
if mode & Mode.TEST:
    ...
```

`PRIMARY_MODES` (`salt/graph/spec.py:50`) is the four-element tuple
`(Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX)` the planner iterates when it
compiles one plan per mode. A `TensorSpec`'s own `modes` field (below) says
which of those four modes a given port is active in; `TRAINING` and `ALL` are
convenience unions you use when declaring that field, not modes a plan itself
compiles for.

## `TensorSpec`

`TensorSpec` (`salt/graph/spec.py:242`) is the frozen dataclass every bundle
leaf is declared with. Its fields:

| Field | Meaning |
|---|---|
| `shape` | a tuple mixing concrete `int`s and symbolic dim strings (`"B"`, `"T:tracks"`); `None` means unconstrained. A bare string is rejected with `TypeError` — `shape="BT"` would otherwise iterate into two one-character dims by accident, so write `shape=("BT",)` if you really mean a single dim named `"BT"`. |
| `dtype` | the leaf's dtype name, or `None` to leave it to the producer. |
| `kind` | one of `"data"`, `"pad_mask"`, `"label"`, `"loss"`, `"meta"` (the `Kind` literal, `KINDS` tuple). A consumer port only binds a producer leaf of the same `kind` — declaring `kind="label"` on a require and finding only a `kind="data"` producer for that key is a compile-time mismatch, not a runtime one. |
| `modes` | a `Mode` flag saying which modes this port is active in, default `Mode.ALL`. A port active in no mode (`modes=Mode(0)`, or a computed union that resolves empty) raises `ValueError` at construction — a port nothing ever activates is dead code, not a valid declaration. |
| `optional` | whether a *require* may go unmet without failing the compile (a producer's own leaf is never optional). |
| `fields` | the last dimension's column names, in order, when this leaf carries named columns. Declaring `fields` means downstream code resolves a column by name (`schema.fields_of(key).index("pt")`), never by remembering its position in a YAML list. |

## Symbolic versus concrete dimensions

A shape entry is either a concrete `int` or a symbolic dim string built by
`sym_dim(family, qualifier=None)` (e.g. `sym_dim("T", "tracks") == "T:tracks"`,
`salt/graph/spec.py:186`); `is_symbolic_dim` and `split_symbolic_dim` are its
counterparts (checking, and pulling the family/qualifier back apart).

Declaring `shape=("B", "F:mnist")` asks the planner to unify your module's
width for that key with whatever the producer of `"F:mnist"` ultimately
resolves to, so your module's own output width need not be spelled out at
compile time. Declaring `shape=("B", 128)` does the opposite: it publishes a
concrete width of 128 that anything downstream can size a layer against
without waiting for a resolve pass. Most task heads and normalisers declare
concrete produced widths; embeddings and encoders that pass a variable-length
sequence through declare `shape=None` and let `derived_widths` (model side,
see [`model.md`](model.md)) publish the real number once it is known.

## `IO` and `unflatten_spec`

`declare_io` returns an `IO` (`salt/graph/spec.py:392`), a frozen dataclass
of two *nested* spec trees, `requires` and `produces`. A nested spec mirrors
the bundle's own dict-of-dicts shape, with `TensorSpec` values at the leaves;
`unflatten_spec` (`salt/graph/spec.py:356`) builds that nested tree from the
flat dotted dict you naturally write while coding a module, and
`flatten_spec` is its inverse. In practice, every module in this tree writes
the flat form and calls `unflatten_spec` at the `return` line:

```python
from salt.graph.spec import IO, TensorSpec, unflatten_spec

def declare_io(self, mode):
    del mode
    requires = {"inputs.jets": TensorSpec(shape=("B", 24), dtype="float32")}
    produces = {"embed.jets": TensorSpec(shape=None, dtype="float32")}
    return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))
```

which is equivalent to writing the nested form directly:

```python
return IO(
    requires={"inputs": {"jets": TensorSpec(shape=("B", 24), dtype="float32")}},
    produces={"embed": {"jets": TensorSpec(shape=None, dtype="float32")}},
)
```

`IO.__post_init__` eagerly flattens and validates both trees, so an invalid
key raises at the point `declare_io` returns, not later when the planner
happens to visit that module.

## Reading and writing the bundle

At run time the same dotted keys address torch tensors (model side) or numpy
arrays (data side). A model module reads with `b.get("normed.jets")` inside
`forward`; a data module reads with `batch.get("raw.jets")` inside `process`
or `read`. Whichever side you are on, return **exactly** the keys you
declared as `produces`, no more and no fewer.

The executor checks this on every merge, not only when `debug: true` is set
(`salt/graph/bundle.py:126-133`, enforced from `salt/graph/executor.py:184-188`).
A mismatch raises `DeclarationError`:

```text
module '{name}' returned keys that do not match its declared produces: missing={list} unexpected={list}
```

Returning something other than a dict raises the same class with a
different message:

```text
module '{name}' returned {type} — modules return their declared produces as a dict of newly produced keys
```

The same equality applies per mode: a produce gated to one mode must be
returned under that same gate, so if `declare_io` returns an empty `IO` for a
mode, `forward` must return `{}` for that mode too.

## Wiring a module into a config

A module entry in `model.init_args.modules` or `data.modules` is a
`class_path` + `init_args` pair:

```yaml
model:
  init_args:
    modules:
      jet_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: jets
          out_dim: 128
```

The dict key (`jet_embed` above) becomes `self.name` once salt constructs the
module. `class_path` is resolved with a normal Python import, so the module's
containing package must already be importable: for anything outside `salt`
itself, put its parent directory on `PYTHONPATH` before running `salt`, and
when running inside a container pass `apptainer exec --env
PYTHONPATH=/path/to/your/package ...` so the import resolves inside the
container's own process, not only in the shell that launched it.

`class_path` is not a filesystem path, and it is not the name of the script
you ran. Salt imports it, in salt's own process, so it must be a real dotted
import path to a module already on `PYTHONPATH`, for example
`my_modules.model.FeatureScaler`. A bare class name, a relative path, or the
script you ran yourself will never resolve, and `__main__.FeatureScaler` will
not either: salt's own `__main__` is the `salt` console entry point, not the
shell script or notebook that launched it, so `import_module("__main__")`
succeeds and only the attribute lookup then fails:

```text
module '__main__' has no attribute 'FeatureScaler'
```

(`salt/cli.py:118-120`). Three related failures come from the same resolver.
A `class_path` with no dot at all:

```text
class_path must be a dotted import path like 'pkg.mod.Class', got '{value}'
```

(`salt/cli.py:107-109`). An import that fails outright:

```text
cannot import module '{mod}' for class_path '{path}': {err}
```

(`salt/cli.py:113-116`). And a constructor that raises:

```text
instantiating '{path}' failed: {err}
```

(`salt/cli.py:121-124`). All four are `ConfigError`; a bare `ImportError` or
`AttributeError` never reaches you directly.

Which base class you subclass decides which config section your module goes
under, and salt checks it: a `data.modules` entry that is not a
`SaltDatasetModule`, and a `model.init_args.modules` entry that is not a
`SaltModelModule`, both fail at construction time with the errors quoted
below, not later when the plan compiles. There is no third base class and no
module that is valid under both sections.

## Checking a module without data

None of these needs a data file, a GPU, or a training run: every one of
them is a static check over the compiled plan.

- `salt graph validate -c cfg.yaml` checks connectivity: every require has a
  producer, every producer is demanded by something (unless it is a declared
  sink), and kinds agree across every binding.
- `salt graph why -c cfg.yaml --mode <mode> --key <dotted key>` answers who
  produces one key, who consumes it, and if it is missing, why. See
  [`cli.md#salt-graph-why`](../cli.md#salt-graph-why) for the full output
  shape.
- `salt graph plan -c cfg.yaml` prints the ordered plan table for a mode.
- `salt graph plot -c cfg.yaml -o graph.svg` renders the compiled graph via
  Graphviz.
- `--set KEY=VALUE` on any of the above supplies a required `init_arg`
  data-free, when a module needs one to construct but the value itself does
  not matter for the check you are running.

See [`cli.md#salt-graph`](../cli.md#salt-graph) for the full command
reference, including `deadcode` and `resolve`. The smallest set of modules a
trainer config must contain for these checks to even reach a compiled plan
is catalogued at
[`configuration.md#the-minimum-graph-that-compiles`](../configuration.md#the-minimum-graph-that-compiles).

## Errors you will see

These are the actual messages, quoted from the source that raises them,
ordered by how often you meet them rather than by which module raises them.
Interpolated parts are shown as `{placeholder}`; search for the fixed text
around them the next time one appears.

### Liveness: every module must reach a sink

A configured module that nothing ever consumes is treated differently
depending on what kind of config you are running, and the difference
matters.

**A standalone toy `modules:` config with no `sinks:`** (the shape
[`tutorials/custom_modules.md`](../tutorials/custom_modules.md) uses for
isolated checks) never prunes: `sink_list is None` short-circuits demand
pruning (`salt/graph/planner.py:460`), so every configured module stays
alive and compiles. An unconsumed output only surfaces as an info- or
warning-level finding from `deadcode()` (`salt/graph/planner.py:265-330`),
which never raises.

**A trainer config** always has sinks: `loss.total` anchors FIT/VAL, and
`preds.*` plus writer demand anchor TEST/ONNX (see the anchor errors below).
A module whose outputs reach no sink in **any** of the four primary modes is
a hard error, not a warning:

```text
module '{name}' ({Class}) is dead in every mode (FIT/VAL/TEST/ONNX): no port is active, or its outputs reach no sink in any mode — a configured module must do something; remove it or wire a consumer
```

(`AllModesDeadError`, `salt/graph/planner.py:1159-1163`.) Wire a consumer
before you add a module, or expect this the moment you move from a toy
`modules:` check to a real trainer config.

### The catalogue

**1. `ConnectivityError`** (`salt/graph/planner.py:1048-1051`, raised at
`:1064`). The most common failure: a require names a key nothing produces.

```text
[mode={MODE}] module '{name}' requires '{key}' — no module or source produces it.
```

A declared sink demanding a key nothing produces raises the same class with
a sink-flavoured head (`planner.py:1045`):

```text
[mode={MODE}] sink key '{key}' (demanded by {origin}) — no module or source produces it.
```

Cause: the producing module is missing from `data.modules` or
`model.init_args.modules` (most often a missing `Features` or `Labels`), or
the key name has a typo. The message lists near-miss suggestions and the
available keys in that mode; read those before guessing.

**2. `AllModesDeadError`**, the liveness error above. Cause: the module's
own outputs reach no sink in FIT, VAL, TEST or ONNX. Fix: wire a consumer,
or delete the module.

**3. `ShapeError`** (`salt/graph/planner.py:1002-1005`). A producer and a
consumer disagree about a key's rank.

```text
[mode={MODE}] rank mismatch on '{key}': producer '{p}' declares shape {tuple} but consumer '{c}' expects {tuple}
```

The way readers most often reach this: a `MaskedInputNormaliser`'s
`global_object` stream is declared rank 2 (`("B", F)`) while every other
stream in its `streams:` list is declared rank 3 (`("B", T, F)`); see the
shape rule on [`model.md#shipped-model-modules`](model.md#shipped-model-modules).
Fix: check which of your two streams is actually per-jet versus
per-constituent, and declare it accordingly.

**4. `NotImplementedError`**, `"{ClassName} has no declare_io()"`
(`salt/model/base.py:69`). Cause: a concrete model module that reaches the
plan never overrode `declare_io`, which is required of every
`SaltModelModule`. The one case that legitimately never raises this is a
manifest-only `outputs:` writer, which never enters the plan at all. Fix:
implement `declare_io`, returning an `IO` built from your module's own
requires and produces.

**5. `ConfigError`, an empty loss narrow** (`salt/model/modules/losses.py:105-108`).

```text
LossSum '{name}': narrowed to an empty loss-key list — no module declares a losses.* produce
```

Cause: a `LossSum` module with no task head in the config to collect from.
`SaltModule.__init__` narrows every `LossSum` before any `declare_io` runs
(`salt/model/saltmodule.py:246-247`), so this fires before the plan compiles
and before `salt graph validate` reaches the graph. Fix: add at least one
task module, or remove the `LossSum`.

**6. `ConfigError`, a missing training or evaluation anchor**
(`salt/model/saltmodule.py:665-667`, `:685-686`, `:693-695`). Three related
messages, one per anchor a plan compiles against:

```text
no module produces 'loss.total' in mode {MODE} — training plans anchor on it; add a LossSum module
```

```text
[mode=TEST] the configured writers consume nothing the model produces — check the outputs: section
```

```text
no module produces a 'preds.*' key in mode {MODE} — evaluation plans anchor on predictions
```

Cause: FIT/VAL need a `LossSum` producing `loss.total`; TEST/ONNX need at
least one task head producing a `preds.*` key that a writer actually
demands. Fix: add the missing `LossSum` or task head, or widen the
`outputs:` section's `tasks:` list.

**7. `DeclarationError`** (`salt/graph/bundle.py:126-133`, enforced from
`salt/graph/executor.py:184-188`, on every merge, not only under
`debug: true`).

```text
module '{name}' returned keys that do not match its declared produces: missing={list} unexpected={list}
```

A non-dict return raises the same class differently
(`salt/graph/executor.py:173-176`):

```text
module '{name}' returned {type} — modules return their declared produces as a dict of newly produced keys
```

Cause: your module's `forward`/`process` returned a different key set than
its `declare_io` promised for that mode, most often a produce that is gated
to a mode but returned unconditionally. Fix: match the return to the
mode-gated `IO`, exactly (see [above](#reading-and-writing-the-bundle)).

**8. `UndeclaredAccessError` / `MutationError`**, `model.init_args.debug: true`
only (`salt/graph/executor.py:323-333`, `:179-183`).

```text
[mode={MODE}] module '{name}' read bundle key '{key}' which is not in its declared requires — declared: {keys}. fix: amend '{name}'.declare_io to require '{key}' (optional=True if consumed-if-present), or drop the access
```

```text
[mode={MODE}] module '{name}' mutated bundle key '{key}' in place during debug execution — bundle leaves are read-only for modules; clone before mutating (e.g. b.get(...).clone()) and return new keys
```

Cause: your module read or wrote a key it never declared. Both checks are
debug-only, so a config without `debug: true` never raises them and the
mistake goes unnoticed until you turn debug on. Fix: amend `declare_io` to
require the key (`optional=True` if the read is conditional), or clone
before mutating.

**9. `ConfigError`, an empty `tasks` list** (`salt/outputs/run_task_output.py:126-129`,
a literal with no interpolation).

```text
RunTaskOutput needs a non-empty 'tasks' list — name the task instances whose get_output() fields this writer serialises
```

Cause: `tasks: []` has no placeholder form; a `RunTaskOutput` with nothing
to name is rejected at construction, before any graph work. Fix: name at
least one task instance, or drop the `outputs:` section entirely and check
the graph with the toy `modules:` format instead (see
[`configuration.md#the-minimum-graph-that-compiles`](../configuration.md#the-minimum-graph-that-compiles)
and [`outputs.md#declaring-outputs-the-outputs-section`](../outputs.md#declaring-outputs-the-outputs-section)).

**10. `ConfigError`, an unsupported optimizer** (`salt/model/saltmodule.py:258-260`).

```text
optimizer '{value}' is not supported — choose from ['AdamW', 'lion', 'lion-pytorch', 'HybridMuonAdamW']
```

The name is case-sensitive: `adam` is rejected, `AdamW` is not.

**11. `ConfigError`, `class_path` resolution**, four sites in `salt/cli.py`:
no dot at all (`:107-109`), an import that fails (`:113-116`), a missing
attribute (`:118-120`, what `__main__.X` produces), and a constructor that
raises (`:121-124`). See
[Wiring a module into a config](#wiring-a-module-into-a-config) for all
four messages side by side. Every one is a `ConfigError`; a bare
`ImportError` or `AttributeError` never reaches you.

**12. `KeyError` from `Bundle.get`** (`salt/graph/bundle.py:46-61`), three
distinct messages:

```text
bundle key '{k}' is a subtree, not a leaf; use subtree()
```

```text
bundle key '{k}' not found: '{blocker}' is a leaf, cannot descend
```

```text
bundle key '{k}' not found
```

`get` never returns `None` for a missing key; the probe for a require
declared `optional=True` is `key in b`, and `__contains__` calls
`split_key` first (`bundle.py:91-96`), so a malformed key raises
`TypeError` or `ValueError` from `salt/graph/spec.py:89-104` rather than
silently reading as `False`.

**13. `BindError`**, two distinct messages. A key with no resolved width
(`salt/model/bind.py:51-55`):

```text
no statically resolved width for bundle key '{key}' — the key is either absent from the compiled plans or its last dim never binds to a concrete size{hint}
```

And a symbolic dim resolved to two different sizes (`salt/model/bind.py:310-313`):

```text
symbolic dim '{dim}' resolves to {n} at {where} but {m} at {where2} — conflicting widths
```

Both are always a `bind`-time problem, never a `declare_io`-time one:
`declare_io` only *declares* a shape, symbolic or concrete, and the planner
only resolves symbols to concrete widths once every mode's plan is compiled
and `bind_all` runs (see [`model.md#bind_all`](model.md#bind_all)). The
first names the nearest known keys; the second is a real modelling conflict
between two modules' declared widths for the same key, not a typo in one
place only.

**14. `ConfigError`, the data-graph module type and reader-count checks**
(`salt/data/datamodule.py:268-271` and `:278-280`). A `data.modules` entry
that is not a `SaltDatasetModule`:

```text
module '{name}' ({Class}) is not a SaltDatasetModule — data-graph entries must subclass SaltDatasetModule; wrap or extend it
```

And `data.modules` must contain exactly one `Reader`:

```text
SaltDataModule needs exactly one Reader in modules, got {n} ({[names]})
```

Fix the first by checking the `class_path`; fix the second by adding or
removing a `Reader` entry until exactly one remains.

**15. `ConfigError`, name-based `origin_weighting` with no schema**
(`salt/model/modules/tasks/edge.py:178-183`).

```text
VertexingTaskModule '{name}': name-based origin_weighting needs the origin label's class names, but the schema artifact has no string-list attr '{origin_label}' on the '{stream}' group (config: model.modules.{name}.init_args.origin_weighting) — dump the schema with the origin class names, or use integer origin ids
```

Fix: dump the schema so the origin label's class names are available, or
switch `origin_weighting` to integer origin ids.

**16. The check that does *not* fire.** `check_class_names`
(`salt/model/saltmodule.py:1972-1999`) returns `0` immediately when the
reader has no callable `schema_group` (`:1982-1984`). Without a schema
artifact, a wrong-length or reordered `class_names` list raises **no error
at all** at config-validation time; it either crashes on the first training
batch (wrong length) or silently mislabels the head (right length, wrong
order). See the `class_names` rule on [`configuration.md`](../configuration.md)
and [`model.md#shipped-model-modules`](model.md#shipped-model-modules).

## See also

- [`model.md`](model.md): `SaltModelModule` method by method, `ResolvedSchema`,
  `bind_all`, adding a task head, and the full `init_args` reference for
  every [shipped model module](model.md#shipped-model-modules).
- [`data.md`](data.md): `SaltDatasetModule`, `Reader`, `Processor`,
  `SaltDataModule`, and the full `init_args` reference for every
  [shipped data module](data.md#shipped-data-modules).
- [`tutorials/custom_modules.md`](../tutorials/custom_modules.md): building
  one data module and one model module end to end.
- [`outputs.md`](../outputs.md): terminal sinks (`is_sink`,
  `SinkModule`) and the `outputs:` section, which this page does not cover.
- [`cli.md`](../cli.md): the full `salt graph` and `salt schema` command
  reference.
