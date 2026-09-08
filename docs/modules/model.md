# Model modules

`SaltModelModule` (`salt/model/base.py:29`) is required to be the base of
every `nn.Module` that participates in the compiled model graph: every
shipped embed, encoder, pooling layer, normaliser, loss and task head
subclasses it. It is not required of everything under `salt/model/nn/`:
raw torch building blocks such as `Dense` and `Attention` are not graph
participants themselves; they stay plain `nn.Module`, owned internally by a
`SaltModelModule` that constructs them at `bind`.

Import `SaltModelModule` from `salt.model.modules`, alongside `ResolvedSchema`,
`bind_all`, `materialise_all`, `resolve_bind_schema` and `BindError`, which
are all re-exported there too (`salt/model/modules/__init__.py`). The
canonical home of all six is `salt.model.base` / `salt.model.bind`; import
from `salt.model.modules` unless you have a reason to import the narrower
module directly.

## Lifecycle

A model module goes through the same five phases every model in the plan
goes through, described in full in
[`index.md#the-compile-then-run-split`](index.md#the-compile-then-run-split):

| Phase | Method | Runs |
|---|---|---|
| construct | `__init__` | once, from config, at parse time |
| declare | `declare_io(mode)` | once per mode, before any data exists |
| bind | `bind(schema)` | once per model, after every mode's plan compiles |
| materialise | `materialise()` | once, only before a fresh fit (skipped on checkpoint load) |
| run | `forward(b, mode)` | once per batch |

`SaltModelModule` is not a Python `ABC`. `declare_io` and `forward` are the
two methods a graph participant is expected to override, but neither is
decorated `@abstractmethod`: a manifest-only `outputs:` section writer never
enters the plan's forward loop, so it legitimately never overrides `forward`,
and the base's `declare_io` default exists precisely so that class does not
have to. Salt checks module validity with `isinstance(m, SaltModelModule)`
at construction time, not by probing for method presence.

### `__init__(self) -> None`

`salt/model/base.py:51`

**When salt calls it.** Once, when jsonargparse constructs the module from
its `class_path` + `init_args` entry, before the config's dict key is even
known.

**What you are required to do.** Record configuration onto `self` and
nothing else. Call `super().__init__()` first, since `SaltModelModule`
subclasses `nn.Module`.

**What you must not do.** Do not build any `nn.Module` layer, allocate a
tensor, or touch a file. The module does not yet know its input widths;
anything sized from a bundle key belongs in `bind`, not here.

**Minimal snippet:**

```python
class MyEmbed(SaltModelModule):
    def __init__(self, stream: str, out_dim: int) -> None:
        super().__init__()
        self.stream = stream
        self.out_dim = out_dim
        self.net: nn.Module | None = None  # built at bind
```

**Read as a real example:** `StreamEmbed.__init__`
(`salt/model/modules/stream_embed.py:64`) records `stream`, `out_dim` and
several optional sub-configs, and leaves `self.net = None` until `bind`.

### `declare_io(self, mode: Mode) -> IO`

`salt/model/base.py:56`

**When salt calls it.** Once per mode, during the declare phase, before any
data file is opened.

**What you are required to do.** Return the bundle keys this module reads
and writes *in `mode`*, as a pure function of `self`'s own configuration.
Build the return value with `IO(requires=unflatten_spec({...}),
produces=unflatten_spec({...}))` (see
[`index.md#io-and-unflatten_spec`](index.md#io-and-unflatten_spec)).

**What you must not do.** Do not open a file, make a network call, or build
a tensor. The `mode` argument tells you which of `FIT`/`VAL`/`TEST`/`ONNX`
you are being asked about; do not use it to trigger any side effect.

**Not abstract.** The base implementation raises `NotImplementedError`
naming the class (`salt/model/base.py:69`) rather than declaring the method
`@abstractmethod`, precisely so the one legitimate non-override, a
manifest-only `outputs:` writer that never enters the plan, does not have to
provide a dummy body.

**Minimal snippet:**

```python
def declare_io(self, mode: Mode) -> IO:
    del mode
    requires = {"inputs.jets": TensorSpec(shape=("B", 24), dtype="float32")}
    produces = {"embed.jets": TensorSpec(shape=None, dtype="float32")}
    return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))
```

**Read as a real example:** `StreamEmbed.declare_io`
(`salt/model/modules/stream_embed.py:139`) declares a symbolic input and a
`shape=None` produce, with the concrete output width supplied later by
`derived_widths`. `ClassificationTaskModule.declare_io`
(`salt/model/modules/tasks/classification.py:119`) declares a concrete
produce (`shape=("B", n_classes)`) instead, because a classification head's
output width is a config constant, not something the planner needs to
resolve.

### `bind`

`bind(self, schema: ResolvedSchema) -> None`, `salt/model/base.py:75`

**When salt calls it.** Once per model, after every mode's plan has compiled
and every symbolic width has resolved, via `bind_all` (see
[`bind_all`](#bind_all) below). Never called directly by a module author.

**What you are required to do.** Build this module's inner `torch.nn`
layers, sized from widths the planner resolved rather than repeated a second
time in config. Read a resolved width with `schema.width("normed.jets")` and
a resolved column-name tuple with `schema.fields_of(key)`.

**What you must not do.** Do not read a file here. The base implementation is
a no-op, so a module with no sized layers, a plain relabelling or plumbing
module, may omit `bind` entirely.

**Minimal snippet:**

```python
def bind(self, schema: ResolvedSchema) -> None:
    self.net = nn.Linear(schema.width("normed.jets"), self.out_dim)
```

**Read as a real example:** `StreamEmbed.bind`
(`salt/model/modules/stream_embed.py:166`), `Normaliser.bind`
(`salt/model/modules/norm.py:101`), `GlobalAttentionPooling.bind`
(`salt/model/modules/pooling.py:74`), `ClassificationTaskModule.bind`
(`salt/model/modules/tasks/classification.py:155`) and
`TransformerEncoder.bind` (`salt/model/modules/transformer_encoder.py:291`).

This is **not** the same method as the data-side `bind`. They share a name
and nothing else: see [`data.md#bind`](data.md#bind) for the data-side
signature, and the disambiguation table at the [bottom of this
page](#data-side-bind-vs-model-side-bind).

### `materialise(self) -> None`

`salt/model/base.py:85`

**When salt calls it.** Once, only before a fresh fit, after `bind` and
before the first `forward`. Skipped entirely on checkpoint load, because by
then the values this method would load already live in the checkpoint's
`state_dict`.

**What you are required to do.** Load any value that comes from a file, most
commonly normalisation constants read from a `norm_dict` artifact. This is
the only hook on `SaltModelModule` allowed to touch the filesystem beyond
config.

**What you must not do.** Do not build layers here; that is `bind`'s job.
Do not assume this method runs on checkpoint load; anything it sets must
also be captured correctly by the `state_dict` so resuming works without it.

**Minimal snippet:**

```python
def materialise(self) -> None:
    stats = yaml.safe_load(Path(self.norm_dict_path).read_text())
    self.means.copy_(torch.tensor(stats["mean"]))
```

**Read as a real example:** `Normaliser.materialise`
(`salt/model/modules/norm.py:189`), which loads a `norm_dict` YAML and copies
its mean/std arrays into buffers `bind` already allocated.

### `derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]`

`salt/model/base.py:93`

**Decide whether you need it.** You do not need `derived_widths` when the
width the planner has to resolve is either a concrete `int` in your
`declare_io` shape or a symbol you reuse from one of your requires: the
planner's union-find binds it on its own (`salt/graph/planner.py:961-978`).
You do need it when your produce carries a fresh symbol, `shape=None`, or a
width that is a function of several resolved inputs, for example a concat
module whose output width is the sum of its inputs' widths. A key is
"absent" from the resolved schema when nothing in the plan ever pins its
last dim to a concrete `int`: no `fields`, no concrete dim, and no
`derived_widths` contribution. Omitting `derived_widths` where it was
needed leaves the key absent, and every downstream `schema.width()` call on
it raises `BindError`.

**When salt calls it.** Repeatedly, during the resolve-widths phase, until a
fixpoint is reached: the planner asks every module with a callable
`derived_widths` for its contribution, folds in whatever is new, and asks
again until nothing changes (`salt/model/bind.py:209`).

**What you are required to do.** Return `{produced_key: width}` for any
produced key whose width is not a single symbol the planner can unify on its
own, for example a concat module whose output width is the sum of its
inputs' widths. The `widths` argument is whatever has resolved so far; a key
you need may not be in it yet on an early sweep.

**What you must not do.** Do not raise on a key that is not yet resolved:
just omit it from your return and let it come up again on the next sweep. Do
not return a width that disagrees with a value the planner already pinned
for the same key elsewhere; the fixpoint runner raises `BindError` naming
"conflicting widths" (`salt/model/bind.py:226`) when two contributions
disagree, and that is the planner catching a real modelling conflict, not
something to work around.

**Minimal snippet:**

```python
def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
    del widths
    return {"embed.jets": self.out_dim}
```

**Read as a real example:** `StreamEmbed.derived_widths`
(`salt/model/modules/stream_embed.py:161`) publishes its config-fixed
`out_dim` directly; `Concat.derived_widths`
(`salt/model/modules/plumbing.py:87`) sums its inputs' resolved widths.

### `forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]`

`salt/model/base.py:128`

**When salt calls it.** Once per batch, for every module the plan for the
active mode kept, in the plan's compiled order.

**What you are required to do.** Read only the bundle keys named by
`self.declare_io(mode).requires`, via `b.get(key)`. Return exactly the keys
named by `self.declare_io(mode).produces`, as a flat or nested dotted dict;
the executor accepts either spelling.

**What you must not do.** Do not read a key you did not declare as a
require: it may not exist in the plan you were compiled into, and even when
it does, an undeclared read defeats the whole point of static declaration.
Do not return a key you did not declare as a produce.

**Minimal snippet:**

```python
def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
    del mode
    x = b.get("normed.jets")
    return {"embed.jets": self.net(x)}
```

**Read as a real example:** any shipped task head, for instance
`ClassificationTaskModule.forward` (`salt/model/modules/tasks/classification.py:285`).

**Reading the bundle.** `b.get(key)` raises `KeyError` for a missing key, a
subtree, or a descent through an existing leaf; it never returns `None`
(`salt/graph/bundle.py:46-61`). Probe a require declared `optional=True`
with `key in b`, which calls `split_key` first, so a malformed key raises
`TypeError`/`ValueError` rather than reading as `False`
(`salt/graph/bundle.py:91-96`, `salt/graph/spec.py:89-104`). Under
`model.init_args.debug: true`, reading or probing a key you did not declare
raises `UndeclaredAccessError`, and mutating a bundle leaf in place raises
`MutationError` (`salt/graph/executor.py:323-333`, `:179-183`); the debug
checks run only under that flag, not by default.

A worked `optional=True` example, end to end: declare the key as optional in
`declare_io`, probe it with `in` in `forward`, and branch on the result.

```python
def declare_io(self, mode: Mode) -> IO:
    del mode
    requires = {
        "masks.registers": TensorSpec(
            shape=("B", "R"), dtype="bool", kind="pad_mask", optional=True
        ),
    }
    produces = {"pooled.global": TensorSpec(shape=None, dtype="float32")}
    return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
    del mode
    if "masks.registers" in b:
        registers = b.get("masks.registers")
        # fold register rows into the pooled vector
    else:
        registers = None
    ...
```

When no module in the plan produces `masks.registers` at all, the planner
drops the optional require entirely rather than failing the plan: this is
exactly how `GlobalAttentionPooling` stays compilable on an encoder-less
config that never produces register masks.

### `is_sink(self) -> bool`

`salt/model/base.py:111`

Always `False` for a `SaltModelModule`; the base default exists only so an
`isinstance` structural check never accidentally treats an unrelated model
module as a sink. Terminal sinks are a separate family, satisfying the
`SinkModule` protocol (`salt/graph/spec.py:432`) rather than
`SaltModelModule`, and are documented in
[`outputs.md`](../outputs.md#what-a-sink-must-provide). There is no snippet
here because there is nothing to override: a model module that needs to be a
sink is not a model module, it is an `outputs:` section writer.

A configured module whose outputs reach no sink in any mode is not silently
dropped: it raises `AllModesDeadError`. The full liveness rule, including
the toy-config regime where the same situation is only a warning, is at
[Liveness: every module must reach a
sink](index.md#liveness-every-module-must-reach-a-sink).

### `ResolvedSchema`

`salt/model/bind.py:24`

A frozen dataclass carrying two mappings, `widths` (dotted key to concrete
last-dim size) and `fields` (dotted key to a tuple of column names), built by
`resolve_bind_schema` (`salt/model/bind.py:75`) once every mode's plan has
compiled. It has two lookup methods, both raising `BindError`, with
nearest-key suggestions from the known set, when the key is absent:

- `width(key: str) -> int` (`salt/model/bind.py:37`)
- `fields_of(key: str) -> tuple[str, ...]` (`salt/model/bind.py:57`)

A key can be legitimately absent from either mapping. A width is absent when
nothing in the plan ever pins that key to a concrete last dimension, for
example a meta leaf, a scalar loss, or a shape that stays data-dependent all
the way through. Fields are absent whenever the producing declaration never
supplied a `fields` tuple, which is common for anything that is not a named
column block.

```python
def bind(self, schema: ResolvedSchema) -> None:
    width = schema.width("normed.jets")
    columns = schema.fields_of("normed.jets")
```

### `bind_all`

`bind_all(modules: Mapping[str, SaltModelModule | GraphModule], schema: ResolvedSchema) -> None`,
`salt/model/bind.py:256`

This symbol has zero documentation coverage in the tree before this page.

**When salt calls it.** Once per model, after `resolve_bind_schema` produces
the `ResolvedSchema`, from `SaltModule`'s own fit-setup code
(`salt.model.saltmodule`).

**What it is required to do.** Be the single place `bind` is called at all:
it walks the model's module dict in dict order and calls
`module.bind(schema)` on every entry that is a `SaltModelModule`, skipping a
folded terminal sink (which has no `bind`). A module author never calls
`bind_all`; salt does, exactly once, and a module's own `bind` should assume
it is being called that way rather than implement any re-entrancy guard of
its own.

**What you must not do.** Do not write your own loop that calls `bind` on a
module dict; use `bind_all` if you are wiring modules programmatically
outside the normal fit path (a test fixture is the only place this comes up
in the shipped tree).

`materialise_all(modules: Mapping[str, SaltModelModule | GraphModule]) -> None`
(`salt/model/bind.py:276`) is `bind_all`'s counterpart for the materialise
phase, walking the same `SaltModelModule` partition and calling
`module.materialise()` on each. `SaltModule` calls it once, only before a
fresh fit, immediately after `bind_all`.

```python
bind_all(model_modules, schema)
```

## Adding a task head

There are two routes, depending on whether an existing task class already
does what you need.

**Route 1: configure an existing task class.** `ClassificationTaskModule`,
`RegressionTaskModule` and `VertexingTaskModule`
(`salt/model/modules/tasks/`) cover jet/track classification, regression and
edge (vertexing) tasks respectively. Route 1 needs no Python at all; write
Python only for Route 2, or for a column no shipped task owns. Add the new
module under `model.init_args.modules`, then add its name to a
`RunTaskOutput` `tasks:` list in the top-level `outputs:` section so its
predictions are written to the eval H5 and its ONNX output declared. This is
the worked example from `docs/architecture.md`, unchanged since it was
written:

```yaml
# my_aux_task.yaml — stack with: --config salt/configs/gn2v2-opendata.yaml --config my_aux_task.yaml
model:
  init_args:
    modules:
      track_type:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: tracks
          context: pooled.global
          label: ftagTruthOriginLabel
          class_names: [Pileup, Fake, Primary, FromB, FromBC, FromC, FromTau, OtherSecondary]
          dense: {hidden_layers: [16], activation: ReLU}
```

`class_names` is index-aligned with the label's on-disk integer values and
sets the head width, because `output_size = len(class_names)`
(`salt/model/modules/tasks/classification.py:170`). A wrong-length list is
not caught by `salt graph validate`: `check_class_names`
(`salt/model/saltmodule.py:1972`) runs only when the reader carries a
`schema:` artifact, and returns immediately when it does not
(`:1982-1983`). Without one, a short list fails on the first training batch
with a `CrossEntropyLoss` index error, and a right-length but reordered list
silently mislabels the head with no error at all. Check the list against the
label column before you fit.

### Task modules

`salt.model.modules.tasks.ClassificationTaskModule` and
`salt.model.modules.ClassificationTaskModule` name the same class:
`salt.model.modules.tasks` re-exports the same three names the
`salt.model.modules` package exports directly
(`salt/model/modules/__init__.py:25-29`). Either spelling works in a
`class_path`; use whichever the surrounding config already uses.

Every task subclasses `_TaskModuleBase` (`salt/model/modules/tasks/base.py:28`),
which derives four keys from the task's own instance name in config, not
from an `init_arg`: `pred_key = preds.<stream>.<name>`
(`tasks/base.py:96`), `loss_key = losses.<name>` (`:101`),
`label_key = labels.<stream>.<label>` (`:106`), and
`input_key = input or encoded.<stream>` (`:57`).

#### `salt.model.modules.ClassificationTaskModule`

`salt/model/modules/tasks/classification.py:29`, `__init__`
`classification.py:44-59`

| `init_arg` | Type | Default |
|---|---|---|
| `stream` | `str` | required |
| `label` | `str` | required |
| `class_names` | `Sequence[str]` | required |
| `input` | `str \| None` | `None` |
| `context` | `str \| None` | `None` |
| `sequence` | `bool \| None` | `None` |
| `dense` | `dict \| None` | `None` |
| `loss` | `str \| dict \| None` | `None` |
| `weight` | `float` | `1.0` |
| `weight_source` | `Mapping[str, str] \| None` | `None` |
| `label_map` | `dict[int, int] \| None` | `None` |
| `expose` | `Sequence[str] \| None` | `None` |
| `write_targets` | `bool` | `True` |

`class_names` is checked non-empty and duplicate-free at construction
(`classification.py:98-106`). `sequence` is inferred when omitted, from
`input is None` (`:108`). `weight_source: {from_class_dict: <path>}`
allocates the cross-entropy `weight` buffer at `bind` and fills it at
`materialise`, the only file-touching hook this class defines
(`:251-283`). Default loss is `torch.nn.CrossEntropyLoss` (`:26`).

#### `salt.model.modules.RegressionTaskModule`

`salt/model/modules/tasks/regression.py:28`, `__init__`
`regression.py:68-87`

| `init_arg` | Type | Default |
|---|---|---|
| `stream` | `str` | required |
| `targets` | `str \| Sequence[str]` | required |
| `input` | `str \| None` | `None` |
| `context` | `str \| None` | `None` |
| `sequence` | `bool \| None` | `None` |
| `target_denominators` | `str \| Sequence[str] \| None` | `None` |
| `norm_params` | `Mapping \| None` | `None` |
| `scaler` | `Mapping[str, Mapping] \| None` | `None` |
| `custom_output_names` | `str \| Sequence[str] \| None` | `None` |
| `gaussian` | `bool` | `False` |
| `sample_weight` | `str \| None` | `None` |
| `publish_targets` | `bool` | `False` |
| `dense` | `dict \| None` | `None` |
| `loss` | `str \| dict \| None` | `None` |
| `weight` | `float` | `1.0` |
| `expose` | `Sequence[str] \| None` | `None` |
| `write_targets` | `bool` | `True` |

`target_denominators`, `norm_params` and `scaler` are mutually exclusive
scaling methods; passing none leaves targets raw. `sequence` is inferred
when omitted (`:156`). `gaussian: true` doubles the output width to
`2 * len(targets)` (means then stddevs) and switches the default loss from
`torch.nn.MSELoss` to `torch.nn.GaussianNLLLoss` (`:24-25`, selected
`:147-148`). When `publish_targets: true` the module produces
`targets.<stream>.<name>` instead of its loss key, because a
`MaskFormerMatchedLoss` then owns `losses.regression` for it.

#### `salt.model.modules.VertexingTaskModule`

`salt/model/modules/tasks/edge.py:27`, `__init__` `edge.py:43-57`

| `init_arg` | Type | Default |
|---|---|---|
| `stream` | `str` | required |
| `label` | `str` | required, must contain `"VertexIndex"` |
| `origin_label` | `str` | required |
| `input` | `str \| None` | `None` |
| `context` | `str \| None` | `None` |
| `dense` | `dict \| None` | `None` |
| `loss` | `str \| dict \| None` | `None` |
| `weight` | `float` | `1.0` |
| `origin_weighting` | `Mapping[str, Sequence[int \| str]] \| None` | `None` |
| `prefix_vertex_column` | `bool` | `False` |
| `expose` | `Sequence[str] \| None` | `None` |
| `write_targets` | `bool` | `True` |

This class has no `sequence` parameter. The internal default for
`origin_weighting` reproduces the fixed heavy/fake split
(`{"heavy": [3, 4, 5], "fake": [1]}`). Name-based `origin_weighting` (class
names instead of integer ids) is resolved against the dataset schema before
`bind`; a name-based config with no schema artifact for `origin_label` is a
`ConfigError`. Default loss is `torch.nn.BCEWithLogitsLoss(reduction="none")`
(`:21-24`).

Per-family default losses and how to override them (`loss:` on the task, or
a `{class_path, init_args}` mapping) are covered in
[`configuration.md`](../configuration.md).

All existing modules survive the merge; the new task's label is demanded
from the dataset automatically and its loss joins `losses.**` via the
wildcard collection. A task added to `model.init_args.modules` but not named
in a `RunTaskOutput` `tasks:` list hard-fails TEST compilation, because
nothing then produces the `preds.*` key that mode anchors on.
`RunTaskOutput.tasks` must also be non-empty; there is no placeholder form,
and `tasks: []` is rejected at construction. Both rules, with the literal
error text, are covered in full at
[Declaring outputs](../outputs.md#declaring-outputs-the-outputs-section). If
the head is a training-time regulariser that must not reach eval or Athena,
set `expose: [fit, val]` on the task so its prediction is pruned from the
TEST/ONNX plans while it still trains; to keep it in eval but out of the
ONNX manifest, list it in a `RunTaskOutput` with `modes: [test]` and check
with `salt export --manifest`.

**Route 2: subclass `_TaskModuleBase`.** `_TaskModuleBase`
(`salt/model/modules/tasks/base.py:28`) is what `ClassificationTaskModule`,
`RegressionTaskModule` and `VertexingTaskModule` all subclass, and it is
where you start if none of the three shapes fits. It is a private name
(leading underscore); documenting it as an extension point here does not
make it public, and renaming it is a separate code change outside the scope
of this page. Subclassing it means implementing:

- `get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]`
  (`salt/model/modules/tasks/base.py:131`): the per-batch output fields for
  the eval H5 / ONNX writer.
- `get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]`
  (`salt/model/modules/tasks/base.py:178`): the same fields' static
  manifest, usable with no data.
- `output_time_requires(self, mode: Mode) -> list[str]`
  (`salt/model/modules/tasks/base.py:203`): extra bundle keys `get_output`
  needs at write time beyond the ones `declare_io` already requires for the
  forward pass.

`_TaskModuleBase.__init__` also gives every task three keyed properties worth
knowing about: `pred_key` (`salt/model/modules/tasks/base.py:97`), `loss_key`
(`:102`) and `label_key` (`:107`), each derived from the module's own `name`.
`expose:` (parsed by `_parse_expose`, `salt/model/modules/tasks/base.py:239`)
restricts which modes the task's prediction is active in, and
`write_targets:` (constructor kwarg, default `True`) controls whether TEST
mode also gets the target-label fields alongside the prediction.

## Shipped model modules

Every configurable entry under `model.init_args.modules` is one of the
classes exported from `salt.model.modules`
(`salt/model/modules/__init__.py:34-59`, 23 names). Six of those names are
base and bind machinery already documented earlier on this page
(`SaltModelModule`, `ResolvedSchema`, `BindError`, `bind_all`,
`materialise_all`, `resolve_bind_schema`); the three task classes are
catalogued separately under [Task modules](#task-modules); two more are
nested sub-module configs, not graph participants in their own right, and
get one line each at the end of this section
(`FeaturewiseTransformation`, `PositionalEncoder`). Every remaining name
below is a real `model.init_args.modules` entry: `class_path`, the source
line of the class and of `__init__`, an `init_args` table, the bundle keys
it requires and produces, and which of `bind`, `materialise` and
`derived_widths` it defines.

Three things worth stating loudly, because attempts against these docs have
repeatedly gotten them wrong:

- Width keys are named `dim` and `out_dim`, never `width`.
- Head count is `attention: {num_heads: N}`, never `n_head`.
- Port keys are named `input` and `out`, never `input_name` or `out_dim`.

### `salt.model.modules.Normaliser`

`salt/model/modules/norm.py:26`, `__init__` `norm.py:40-45`

| `init_arg` | Type | Default |
|---|---|---|
| `norm_dict` | `str \| Path` | required |
| `streams` | `Sequence[str]` | required |
| `global_object` | `str \| None` | `None` |

Requires `inputs.<stream>` for every configured stream. Produces
`normed.<stream>` for every stream (`norm.py:93-99`). The `global_object`
stream is declared `("B", F)`; every other stream in `streams:` is declared
`("B", T, F)` (`norm.py:85-91`).

Defines `bind` (`norm.py:101-113`, allocates `means_<stream>` and
`stds_<stream>` buffers) and `materialise` (`norm.py:189-239`, reads the
`norm_dict` YAML; this is the module's only file-touching hook). Does not
define `derived_widths`.

### `salt.model.modules.MaskedInputNormaliser`

`salt/model/modules/norm.py:261`, `__init__` `norm.py:272-278`

| `init_arg` | Type | Default |
|---|---|---|
| `streams` | `Sequence[str]` | required |
| `global_object` | `str \| None` | `None` |
| `momentum` | `float \| None` | `0.1` |
| `eps` | `float` | `1e-5` |

This class takes **no** `norm_dict` and defines **no** `materialise`: it
learns its statistics online from valid, non-padded objects instead of
loading a precomputed dictionary. It is a different class from `Normaliser`
above, not an alternate constructor for it.

Requires `inputs.<stream>` for every stream, plus `masks.<stream>` for
every stream except `global_object` (`optional=True`, `modes=Mode.TRAINING`,
`norm.py:317-334`). Produces `normed.<stream>` for every stream.

Defines `bind` (`norm.py:336-347`, allocates running-mean/running-var
buffers). Does not define `materialise` or `derived_widths`.

Shape rule (`_spec`, `norm.py:305-315`): `global_object` is a single `str`,
and that one stream is declared rank-2 `("B", F)`; every other stream in
`streams:` is declared rank-3 `("B", T, F)`. Two derived per-row global
streams therefore cannot share one instance: each needs its own
`MaskedInputNormaliser` with itself as `global_object`.

```yaml
model:
  init_args:
    modules:
      norm_jets:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args: {streams: [jets_logpt], global_object: jets_logpt}
      norm_tracks_d0:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args: {streams: [tracks_signed_d0], global_object: tracks_signed_d0}
```

A single instance configured with `streams: [jets_logpt, tracks_signed_d0]`
and `global_object: jets_logpt` declares `tracks_signed_d0` at rank 3
(`("B", T, F)`) when it is actually rank 2, and the planner raises a
`ShapeError` the moment that stream's real shape is bound.

### `salt.model.modules.StreamEmbed`

`salt/model/modules/stream_embed.py:34`, `__init__`
`stream_embed.py:64-74`

| `init_arg` | Type | Default |
|---|---|---|
| `stream` | `str` | required |
| `out_dim` | `int` | required |
| `dense` | `dict[str, Any] \| None` | `None` |
| `context` | `Sequence[str]` | `()` |
| `input` | `str \| None` | `None` |
| `mup` | `bool` | `False` |
| `featurewise` | `dict[str, Any] \| None` | `None` |
| `pos_enc` | `dict[str, Any] \| None` | `None` |

Requires `input` (default `normed.<stream>`, `shape=None`), each key in
`context`, and `inputs.parameters` when `featurewise` is set. Produces
`embed.<stream>` (`shape=None`) (`stream_embed.py:139-159`).

Defines `derived_widths` (`stream_embed.py:161-164`, publishes
`{"embed.<stream>": out_dim}` directly, since `out_dim` is a config
constant) and `bind` (`:166-193`).

### `salt.model.modules.Concat`

`salt/model/modules/plumbing.py:27`, `__init__` `plumbing.py:40`

| `init_arg` | Type | Default |
|---|---|---|
| `streams` | `Sequence[str]` | required |
| `registers` | `int` | `0` |

A nonzero `registers` raises `ConfigError`: registers are internal to
`TransformerEncoder`, not `Concat`. Set the encoder's `num_registers`
instead.

Requires `embed.<stream>` and `masks.<stream>` for every configured stream.
Produces `seq.x`, `seq.mask`, `seq.layout` (a `kind="meta"` leaf holding
`{stream: (start, stop)}`, consumed by `Split`), and `seq.offsets`
(`modes=Mode.ONNX`) (`plumbing.py:63-85`).

Defines `derived_widths` (`:87-97`, takes `seq.x`'s width from the first
resolved `embed.<stream>` width, because `StreamEmbed`'s `shape=None`
produce gives the dim table nothing concrete to bind directly).

### `salt.model.modules.TransformerEncoder`

`salt/model/modules/transformer_encoder.py:33`, `__init__`
`transformer_encoder.py:78-94`

| `init_arg` | Type | Default |
|---|---|---|
| `dim` | `int` | required |
| `num_layers` | `int` | required |
| `attention` | `dict[str, Any]` | required |
| `out_dim` | `int \| None` | `None` |
| `dense` | `dict[str, Any] \| None` | `None` |
| `norm` | `str` | `"LayerNorm"` |
| `num_registers` | `int` | `1` |
| `norm_type` | `str` | `"pre"` |
| `drop_registers` | `bool` | `False` |
| `mup` | `bool` | `False` |
| `edges` | `str \| None` | `None` |
| `edge_embed_dim` | `int` | `0` |
| `update_edges` | `bool` | `False` |
| `featurewise` | `Sequence[dict[str, Any]] \| None` | `None` |

`attention` is a plain mapping that must contain `num_heads`; the config key
is `attention: {num_heads: N}`, never `n_head`
(`transformer_encoder.py:148-152`). `attn_type` is an optional key inside
`attention` (default `"torch-math"`); every other key besides `num_heads`
and `attn_type` is forwarded as `attn_kwargs`.

Requires `seq.x` `("B", T, dim)` and `seq.mask`, plus the `edges` key when
`edges` is set, plus `inputs.parameters` under a FiLM config. Produces
`encoded.seq` `("B", T', out_dim)` always, and `masks.registers`
`("B", num_registers)` unless `drop_registers: true`
(`transformer_encoder.py:259-289`).

Defines `bind` (`:291-317`, validates the resolved edge width and builds the
optional encoder/global FiLM sub-modules). Does not define
`derived_widths`.

### `salt.model.modules.Split`

`salt/model/modules/plumbing.py:120`, `__init__` `plumbing.py:134`

| `init_arg` | Type | Default |
|---|---|---|
| `streams` | `Sequence[str]` | required |

Requires `encoded.seq`, `seq.layout`, and `seq.offsets`
(`modes=Mode.ONNX`). Produces `encoded.<stream>` for every configured stream
(`plumbing.py:149-165`).

### `salt.model.modules.GlobalAttentionPooling`

`salt/model/modules/pooling.py:25`, `__init__` `pooling.py:41`

| `init_arg` | Type | Default |
|---|---|---|
| `input` | `str` | `"encoded.seq"` |
| `out` | `str` | `"pooled.global"` |

Both are defaulted. The port keys on this class are `input` and `out`, not
`input_name` or `out_dim`.

Requires `input`, `seq.mask`, and `masks.registers` (`optional=True`, absent
on an encoder-less wiring). Produces `out` (`pooling.py:49-72`).

Defines `bind` (`:74-76`, builds `nn.Linear(schema.width(input), 1)` for the
attention gate).

### `salt.model.modules.LossSum`

`salt/model/modules/losses.py:23`, `__init__` `losses.py:36-40`

| `init_arg` | Type | Default |
|---|---|---|
| `losses` | `Sequence[str] \| None` | `None` |
| `weights` | `Mapping[str, float] \| None` | `None` |

You never list your tasks in `losses:`; they are collected automatically.
`declare_io` is TRAINING-only: outside `FIT`/`VAL` it returns an empty `IO`.
Inside `FIT`/`VAL` it requires every narrowed `losses.<task>` key
(`kind="loss"`) and produces `loss.total` (`losses.py:112-126`).

A `LossSum` with no task head in the graph is not a no-op: `SaltModule.__init__`
calls `narrow` on it before any plan compiles, and an empty collected
key list raises `ConfigError`.

Defines no `bind`, `materialise` or `derived_widths`.

### Compact reference: the remaining eight

| `class_path` | Source | `init_args` | Requires / produces / hooks |
|---|---|---|---|
| `salt.model.modules.VectorConcat` | `plumbing.py:186`, `__init__` `:199` | `inputs: Sequence[str]` (required); `out: str = "pooled.global"` | requires each key in `inputs` (own width symbol); produces `out`; `derived_widths` sums the input widths (`:244-248`) |
| `salt.model.modules.LossGLS` | `losses.py:139`, `__init__` `:157` | same signature as `LossSum` | subclasses `LossSum`, sharing `declare_io` and the `losses.**` wildcard; rejects any per-loss `weights` entry other than `1.0`, since the geometric mean is only meaningful when no task is pre-scaled |
| `salt.model.modules.EdgeFeatures` | `edge_embed.py:146`, `__init__` `:162-183` | `stream: str` (required); `features: Sequence[str]` (required); `out: str \| None = None`; `input: str \| None = None` | requires `input` (default `inputs.<stream>`) and `masks.<stream>`; produces `out` (default `edges.<stream>`, shape `("B", T, T, E)`); `bind` yes (`:227-231`) |
| `salt.model.modules.EdgeEmbed` | `edge_embed.py:242`, `__init__` `:257-264` | `stream: str` (required); `out_dim: int` (required); `dense: dict \| None = None`; `input: str \| None = None`; `out: str \| None = None` | requires `input` (default `edges.<stream>`); produces `out` (default `edges.<stream>_emb`); `bind` yes (`:311-315`) |
| `salt.model.modules.MaskDecoder` | `maskdecoder.py:50`, `__init__` `:61-71` | `embed_dim: int` (required); `num_queries: int` (required); `num_layers: int` (required); `class_net: Mapping[str, Any]` (required); `md: Mapping \| None = None`; `mask_net: Mapping \| None = None`; `input: str = "encoded.seq"`; `out_stream: str = "objects"` | produces `<out_stream>.{embed,class_logits,class_probs,masks}` (`:163-189`); `bind` yes (`:191-199`) |
| `salt.model.modules.MaskFormerMatchedLoss` | `maskformer_matched_loss.py:33`, `__init__` `:76-84` | `num_classes: int` (required); `num_queries: int` (required); `loss_weights: Mapping[str, float]` (required); `matcher_weights: Mapping \| None = None`; `null_class_weight: float = 0.5`; `class_weights: list[float] \| None = None`; `input_stream: str = "objects"` | TRAINING-only `declare_io` (`:165-229`) |
| `salt.model.modules.FeaturewiseTransformation` | `salt/model/nn/featurewise.py:18` | nested config, no `class_path` entry of its own | a `featurewise:` block on `StreamEmbed` or `TransformerEncoder`; never a `model.init_args.modules` entry |
| `salt.model.modules.PositionalEncoder` | `salt/model/nn/posenc.py` | nested config, no `class_path` entry of its own | a `pos_enc:` block on `StreamEmbed`; never a `model.init_args.modules` entry |

## Data-side `bind` vs. model-side `bind`

The two methods share a name and nothing else. Landing on this page or
[`data.md`](data.md) from a search for "bind", check which one you actually
have:

| | Data-side `bind` | Model-side `bind` |
|---|---|---|
| Signature | `bind(self, ctx: WorkerCtx) -> None` | `bind(self, schema: ResolvedSchema) -> None` |
| Defined on | `SaltDatasetModule` (`salt/data/base.py:138`) | `SaltModelModule` (`salt/model/base.py:75`) |
| Called by | `SaltDataset`, once per (worker process, plan) | `bind_all`, once per model, after every mode's plan compiles |
| Typical use | open a file handle, allocate a per-worker read buffer | build `torch.nn` layers sized from resolved widths |
| May touch a file | yes; the only data-module hook that may | no |

See [`data.md#bind`](data.md#bind) for the data-side entry in full.
