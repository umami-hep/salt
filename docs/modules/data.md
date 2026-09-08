# Data modules

**`SaltDataModule`** (`salt/data/datamodule.py:143`) is the Lightning
`LightningDataModule` salt provides: you configure it under `data:`, but you
never subclass it. **`SaltDatasetModule`** (`salt/data/base.py:113`) is the
base class of the things you put *inside* it, listed under `data.modules`.
Neither name appears anywhere else in the docs before this page.

## `SaltDataModule`

This symbol has zero documentation coverage in the tree before this page.

**What it is required to do.** Turn the configured `data.modules` dict into
per-stage dataloaders. The `modules:` dict must contain exactly one `Reader`
prototype plus any number of processors, or construction raises `ConfigError`
(the `SaltDatasetModule` type check at `salt/data/datamodule.py:268`, the
single-`Reader` guard at `:278`). A `None` entry deletes a module, which is
how a config layered with `--config` removes something an earlier layer
added.

**Constructor kwargs**, from the class docstring:

| kwarg | Meaning |
|---|---|
| `modules` | dataset modules by instance name; exactly one `Reader` plus any processors |
| `train_file` / `val_file` / `test_file` | per-stage source path (wildcards trigger VDS creation) |
| `batch_size` | rows per contiguous batch slab, default 1000 |
| `num_workers` | dataloader worker processes, default 0 |
| `num_train` / `num_val` / `num_test` | row counts per stage; `-1` means all |
| `test_suff` | suffix appended to the eval-file `{sample}` name by the writer callback |
| `move_files_temp` | opt-in staging root (e.g. `/dev/shm/<user>/tmp`); when set, `setup('fit')` restages the fit reader's files there and `teardown('fit')` removes the root; `None` leaves the read path byte-identical |
| `train_vds_path` / `val_vds_path` / `test_vds_path` | explicit VDS output paths for wildcard files |
| `sinks` | per-mode model-boundary demand (`inputs.*` / `masks.*` / `labels.*` / `meta.rows` keys); usually set later via `set_sinks`, not at construction |
| `pin_memory` | pin host memory for faster GPU transfer, default `True` |
| `persistent_workers` | keep worker processes (and their H5 handles) alive between epochs, default `True` |
| `prefetch_factor` | batches prefetched per worker; `None` derives it from the map-style or streaming path |
| `multiprocessing_context` | worker start method (`fork` / `spawn` / `forkserver`) |
| `seed` | base augmentation seed for non-worker reads, default 42 |
| `debug` | enable the boundary non-aliasing assertion |
| `iterable` | build the streaming `IterableSaltDataset` instead of the map-style `SaltDataset`, default `False` |
| `block_rows` | rows per reader call when `iterable`, default 16,384 |
| `interleave_block` | rows per turn of the per-shard sample round robin when `iterable`, default 1 |
| `max_live_streams` | samples holding a resident block at once when `iterable`, default 2 |
| `manifest` | `CorpusManifest` path(s), required when `iterable`; a mapping gives one path per stage |
| `shuffle_stream` | shuffle block order and within-batch row order on the fit streaming loader, default `True` |

The streaming-specific kwargs (`iterable`, `block_rows`, `interleave_block`,
`max_live_streams`, `manifest`, `shuffle_stream`) are covered in full in
[`streaming.md`](../streaming.md); this table gives their one-line meaning
only.

```yaml
data:
  train_file: /data/train_*.h5
  val_file: /data/val.h5
  batch_size: 2000
  num_workers: 8
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args: {groups: {jets: {global_object: true}, tracks: {global_object: false}}}
    features:
      class_path: salt.data.Features
      init_args: {variables: {jets: [pt, eta], tracks: [dphi, deta]}}
    labels:
      class_path: salt.data.Labels
```

`train_file`/`val_file`/`test_file` are the legacy per-stage source kwargs,
still accepted but superseded by `salt.data.InputSamples`
(`data.modules.input_samples`); see [the minimum graph that
compiles](../configuration.md#the-minimum-graph-that-compiles) for the
equivalent config written the current way.

## `SaltDatasetModule`

This symbol has zero documentation coverage in the tree before this page.

**What it is required to be.** The base of every data-graph participant. It
supplies `name` (the `GraphModule` protocol member) and the abstract
`declare_io`, plus four optional hooks covered below: `bind`,
`declare_setup_io`, `setup` and `teardown`. In practice you almost never
subclass `SaltDatasetModule` directly; you subclass `Reader` or `Processor`
(both below), which already implement the parts every reader or every
processor needs.

### `declare_io(self, mode: Mode) -> IO`

`salt/data/base.py:135`, `@abstractmethod`

**When salt calls it.** Once per mode, during the declare phase, before any
data file is opened, exactly like the model-side method of the same name.

**What you are required to do.** Return the bundle keys this module reads
and writes in `mode`, as a pure function of config. A `Reader` requires
nothing (`requires={}`) and produces `raw.<stream>`, `masks.<stream>` and
`meta.rows` only; it never produces `inputs.*`
(`salt/data/readers/reader.py:316-329`,
`salt/data/readers/uproot_reader.py:454-471`). `salt.data.Features` is the
only shipped producer of `inputs.<stream>`
(`salt/data/processors/features.py:79-91`); a normaliser or embed requiring
`inputs.*` with no `Features` in `data.modules` fails with
`ConnectivityError`. A `Processor` both requires and produces.

**What you must not do.** Do not open a data file, make a network call, or
build a numpy array here.

**Minimal snippet:**

```python
def declare_io(self, mode: Mode) -> IO:
    del mode
    requires = {"raw.jets": TensorSpec(kind="data", fields=("pt", "eta"))}
    produces = {"inputs.jets": TensorSpec(dtype="float32", kind="data", fields=("pt", "eta"))}
    return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))
```

**Read as a real example:** `H5StructuredReader.declare_io`
(`salt/data/readers/reader.py:316`) is a source node (`requires={}`);
`Features.declare_io` (`salt/data/processors/features.py:79`) and
`Labels.declare_io` (`salt/data/processors/labels.py:107`) both require and
produce.

**Declaring a width you can only learn from the file.** `declare_io` runs
before any data file is opened, which raises a real question for a
self-describing format: how do you declare a stream's width or column list
when the only place that information lives is the file itself? Two answers,
both used by shipped modules:

- Take the column list as an `init_arg`. `Features(variables={stream:
  [names]})` is the pattern: the config states the columns explicitly, and
  `declare_io` builds the `fields` tuple from `self.variables` with no file
  access at all.
- Omit `fields` and declare a concrete last dimension instead, when you know
  the width but not the column names.

Do not declare `TensorSpec(fields=())` and expect an error to catch a
mistake later. It is accepted silently
(`salt/graph/spec.py:257-299`): the length check there only fires when the
last dim is a concrete `int` (`:295-298`), and a `shape=None` spec skips it
entirely. `bind` then resolves that stream's width to **zero**: with no
shape observation to bind a width, the declared column count becomes the
width (`salt/model/bind.py:150-153`), and an empty `fields` tuple has a
column count of zero. The model builds without complaint and is wrong, and
nothing downstream raises to say so.

### `bind`

`bind(self, ctx: WorkerCtx) -> None`, `salt/data/base.py:138`

**When salt calls it.** Once per (worker process, plan), by `SaltDataset`,
before the first batch that worker reads.

**What you are required to do.** Only implement this if the module needs
per-worker setup: opening a file handle, allocating a reusable buffer. The
default is a no-op. This is **the only place a data module may touch data
files at serving time**. Anything reading from disk once training has
started belongs here, not in `read` or `process`, unless it is reading via a
handle this method already opened.

**What you must not do.** Do not build a `torch.nn` layer here; that is the
model-side `bind`'s job, not this one, and the two are otherwise unrelated
despite the shared name (see the disambiguation box at the [bottom of this
page](#model-side-bind-vs-data-side-bind)).

**`WorkerCtx` fields** (`salt/data/base.py:93`), the argument this `bind`
receives:

| Field | Meaning |
|---|---|
| `mode` | the plan's primary mode for this worker |
| `read_fields` | the demand-narrowed per-stream read set, `{stream: {field: demanding module name}}` |
| `seed` | the per-worker seed, for read-time augmentations |
| `worker_id` | this worker's index, default 0 |
| `num_workers` | total worker count, default 0 |
| `step` | this module's own compiled plan step, or `None`; wildcard producers such as `Labels` use it to learn their narrowed key set |

**Minimal snippet:**

```python
def bind(self, ctx: WorkerCtx) -> None:
    self._handle = h5py.File(self.filename, "r")
    self._rng = np.random.default_rng(ctx.seed)
```

**Read as a real example:** `H5StructuredReader.bind`
(`salt/data/readers/reader.py:430`) opens the file handle and allocates
demand-narrowed per-group buffers; `Labels.bind`
(`salt/data/processors/labels.py:140`) uses `ctx.step` to learn which
`labels.<stream>.<field>` keys it was narrowed to, with no file I/O at all.

!!! note "Two different `bind` methods"

    Data-side: `bind(self, ctx: WorkerCtx) -> None` (`SaltDatasetModule`,
    `salt/data/base.py:138`). Model-side:
    `bind(self, schema: ResolvedSchema) -> None` (`SaltModelModule`,
    `salt/model/base.py:75`, see [`model.md#bind`](model.md#bind)). They take
    different arguments, run at different times, and a module is only ever
    one kind or the other.

### `read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]`

`salt/data/base.py:178`

The default derives each stream's demanded fields from this module's bound
`raw.<stream>` requires, so most modules never override it. Override only
when the field demand is knowable only after narrowing a wildcard produce:
`Labels.read_fields` (`salt/data/processors/labels.py:145`) is the shipped
example, mapping each narrowed `labels.<stream>.<field>` key back to the
`raw.<stream>.<field>` read it needs.

### `declare_setup_io` / `setup` / `teardown`

`declare_setup_io` (`salt/data/base.py:147`) has zero documentation coverage
in the tree before this page.

These three are the setup-time face of a data module, run once per stage
inside `datamodule.setup(stage)` rather than once per batch. All three
default to empty or identity, so most modules never touch them.

- `declare_setup_io(self, stage: SetupStage) -> SetupIO` is the setup-time
  analogue of `declare_io`: a pure function of config, no data files, no
  tensors. A non-empty return for some stage is what marks a module as
  **setup-participating**; it does not change what the per-batch `declare_io`
  face reports.
- `setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle`
  (`salt/data/base.py:158`) is the one sanctioned point a module may mutate
  the setup-time bundle. It may touch the filesystem (resolving a wildcard,
  building an index), reads its declared setup-`requires` off `ctx`, and
  merges back only its declared setup-`produces`. It runs identically on
  every DDP rank, so it must not depend on anything rank-specific to stay
  correct.
- `teardown(self, ctx: SetupBundle, stage: SetupStage) -> None`
  (`salt/data/base.py:169`) is the symmetric cleanup hook, called from
  `datamodule.teardown(stage)`, guarded so it only fires for a stage the
  module actually set up.

**Minimal snippet:**

```python
def declare_setup_io(self, stage: SetupStage) -> SetupIO:
    if stage != "fit":
        return SetupIO()
    return SetupIO(produces=unflatten_source_spec(
        {"source.reader.fit.pattern": SourceSpec(kind="path", stages=("fit",))}
    ))

def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
    if stage != "fit":
        return ctx
    out = {"source.reader.fit.pattern": str(self.resolved_path)}
    ctx.merge(canonical_produced(out, set(out), self.name), who=self.name, expected=set(out))
    return ctx
```

**Read as a real example:** `InputSamples`
(`salt/data/input_samples.py:124`, `:143`) is the shipped source of
per-stage file patterns; `VDS` (`salt/data/readers/vds.py:289`, `:304`)
resolves a wildcard pattern into a concrete virtual-dataset path at setup
time.

## `Reader`

`salt/data/base.py:196`

**What it is required to do.** Turn a source file into a flat dotted dict of
numpy arrays for one contiguous batch slice. It is the source node of the
data graph: `declare_io` always returns `requires={}`.

| Member | Kind | Required to do | Override? |
|---|---|---|---|
| `streams` (`:245`) | abstract property | name the streams this reader serves, in config order | always |
| `__len__` (`:256`) | abstract | return the row count served | always |
| `read(rows, mode)` (`:260`) | abstract | read one contiguous batch, return produced keys | always |
| `prepare()` (`:248`) | concrete, default no-op | main-process file probing (VDS resolution, row counts); idempotent, called lazily by `__len__` | only if there is something to probe |
| `with_source(...)` (`:401`) | concrete, default raises `NotImplementedError` | clone this reader onto another source file, config-only, signature `with_source(self, filename, num=-1, vds_path=None, stage=None) -> Reader` (`salt/data/base.py:401-407`) | required if this reader is used as a `SaltDataModule` prototype |
| `sources()` (`:428`) | concrete, default introspects a `filename`/`files` attribute | list the concrete on-disk file(s) this reader will read | only when sources are not one such attribute (`MultiSampleReader`) |
| `restage(root)` (`:442`) | concrete, default copies each `sources()` file via `with_source` | clone this reader reading staged copies under `root` | only for a reader with more than one source |
| `row_blocks()` (`:265`) | concrete, default one whole-reader block | the reader's natural sequential read units for streaming | only when storage has a coarser natural grain (`UprootReader` per file) |
| `read_block(block, mode)` (`:278`) | concrete, default the plain row-slice `read` | read one `RowBlock` | only for a multi-sample reader mapping a block to a sub-reader |
| `schema_group(stream)` (`:360`) | concrete, default `None` | the schema for one served stream, for static validation | only if a schema artifact is configured |
| `label_universe()` (`:380`) | concrete, default `None` | the `labels.<stream>.<field>` universe, for wildcard narrowing validation | only with a schema |
| `config_fingerprint()` (`:369`) | concrete, default `{}` | everything about config that changes served rows/fields, without opening a file | only if the reader supports cached artifacts keyed on this |
| `h5_source` (`:390`) | property, default `None` | the h5py-openable file, when this reader has one | only for an H5-backed reader |
| `aliases(array)` (`:522`) | concrete, default `False` | whether `array` shares memory with a reusable buffer, for the `debug` boundary check | only if the reader has reusable buffers to check against |

`with_source` is the one method every `SaltDataModule` prototype reader must
implement. `SaltDataModule` calls it once per stage
(`salt/data/datamodule.py:563-564`), passing all four arguments:
`filename` is the resolved per-stage source; `num` is the per-stage row cap
taken from `data.num_train`/`num_val`/`num_test`, so a hand-written clone
that drops `num` silently disables those settings; `stage` selects the
per-split cuts, mapped from the Lightning stage via `_STAGE_OF_MODE`
(`salt/data/datamodule.py:118`); `vds_path` is the resolved virtual-dataset
path for a wildcard source.

Plus the class attributes `schema` (`Schema | None`, default `None`), `cuts`
(`GlobalObjectCuts | None`, sample-axis row eligibility, default `None`),
`constituent_cuts` (`dict[str, ConstituentCuts]`, per-stream, within-row
selection), `stage` (`str | None`, the bound `"train"`/`"val"`/`"test"`
split), `vds_capable` (`bool`, default `False`, whether this reader can build
an h5py virtual dataset) and `incompatible_with` (`tuple[str, ...]`, setup
modules this reader must not coexist with, checked by name so the named class
need not exist yet).

**Arrays returned by `read` may alias reusable buffers.** A reader is free to
reuse its own scratch memory across batches for throughput; anything that
survives to the torch boundary must be copied first. `Features` is the one
place this copy is guaranteed to happen (`np.may_share_memory` guard,
`salt/data/processors/features.py:103`), which is why `raw.*` itself never
crosses into a tensor directly.

**Read as a real example:** `H5StructuredReader`
(`salt/data/readers/reader.py`), `UprootReader`
(`salt/data/readers/uproot_reader.py`), `MultiSampleReader`
(`salt/data/readers/multisample_reader.py`), and the tutorial reader built
from scratch in [`tutorials/mnist.md`](../tutorials/mnist.md).

## `Processor`

`salt/data/base.py:533`

This symbol has zero documentation coverage as a base-class name in the tree
before this page; concrete processors such as `Features` and `Labels` are
named, but the base itself never is.

**What it is required to do.** Transform one batch. Implement
`process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]`
(`salt/data/base.py:544`, `@abstractmethod`): read declared requires via
`batch.get(key)`, and return only newly produced keys.

**What you must not do.** Do not return a key you did not declare in
`declare_io(mode).produces`. Do not hand back an array that aliases a reader
buffer without copying it first, for anything that will reach the torch
boundary.

**Minimal snippet:**

```python
class DoubledPt(Processor):
    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({"raw.jets": TensorSpec(kind="data", fields=("pt",))}),
            produces=unflatten_spec({"inputs.jets_pt2": TensorSpec(dtype="float32", kind="data")}),
        )

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del rows, mode
        pt = batch.get("raw.jets")["pt"]
        return {"inputs.jets_pt2": (pt * 2).astype(np.float32)}
```

**Read as a real example:** `Features`
(`salt/data/processors/features.py`), `Labels`
(`salt/data/processors/labels.py`), `FtagLabeller`
(`salt/data/processors/ftag_labeller.py`), `MultiTarget`
(`salt/data/processors/multi_target.py`) and `MaskFormerTargets`
(`salt/data/processors/maskformer_targets.py`).

## Shipped data modules

`salt/data/__init__.py`'s `__all__` holds 34 names. Ten of them are legal
`data.modules` entries; everything else is a base class, the datamodule
itself, a config or runtime dataclass, a constant, a function, or an
internal runtime object. The [Reference](#reference) table below classifies
every one. This section catalogues the ten real entries: `class_path`,
source line, `init_args`, the bundle keys each requires and produces, and
which hooks it defines.

`salt.data.SaltDataset` is the internal map-style dataset `SaltDataModule`
builds from the configured `data.modules`
(`salt/data/dataset.py:16`); it is never itself a `data.modules` entry, and
putting it there fails the `isinstance(module, SaltDatasetModule)` check at
`salt/data/datamodule.py:267-271`. The same holds for
`salt.data.IterableSaltDataset`, the streaming counterpart, and for
`salt.data.SaltDataModule` itself, which configures `data.modules` but is
never one of its own entries.

### `salt.data.InputSamples`

`salt/data/input_samples.py:24`, `__init__` `:46-50`

| `init_arg` | Type | Default |
|---|---|---|
| `files` | `dict[str, str \| Path]` | required |
| `num` | `dict[str, int] \| None` | `None` |

`files` keys are restricted to `train`/`val`/`test`; an empty `files` dict
or a key outside that set raises `ValueError` (`:52-62`). `num` gives the
per-stage row cap, `-1` meaning all.

This is a setup-only module: `declare_io` returns an empty `IO` for every
mode (`:108-113`); `declare_setup_io` produces
`source.<reader>.<stage>.pattern` per active stage, plus the whole-dict
`artifacts.<reader>.num` scalar once, on the first active stage (`:124-137`).
It defines no `bind`.

```yaml
data:
  modules:
    input_samples:
      class_path: salt.data.InputSamples
      init_args:
        files: {train: /data/train_*.h5, val: /data/val.h5, test: /data/test.h5}
        num: {train: -1, val: -1, test: -1}
```

### `salt.data.H5StructuredReader`

`salt/data/readers/reader.py:56`, `__init__` `:123-134`

| `init_arg` | Type | Default |
|---|---|---|
| `groups` | `Mapping[str, GroupConfig \| Mapping \| None]` | required |
| `schema` | `Schema \| str \| Path \| None` | `None` |
| `filename` | `str \| Path \| None` | `None` |
| `num` | `int` | `-1` |
| `constituent_cuts` | `Mapping[str, Any] \| None` | `None` |
| `cuts` | `GlobalObjectCuts \| None` | `None` |
| `stage` | `str \| None` | `None` |
| `transforms` | `Sequence[Callable] \| None` | `None` |
| `vds_path` | `str \| Path \| None` | `None` |

Requires nothing (`requires={}`, it is the source node). Produces
`raw.<stream>` (`("B",)` for a `global_object` group, else `("B", T)`),
`masks.<stream>` (bool, non-global groups only), and `meta.rows` (int64,
`modes=Mode.TEST`) (`:316-329`). **Never `inputs.*`.**

Defines `bind(ctx)` (`:430`), `prepare()` (`:331`) and `with_source(...)`
(`:291-314`).

Each entry in `groups` is a `GroupConfig` (`dataset`, `pad_max`,
`global_object`, all optional): `dataset` maps to a differently-named H5
dataset, `pad_max` caps a jagged stream to N rows (padding shorter rows,
truncating longer ones), and `global_object` declares a `[B, F]` stream with
no pad mask.

```yaml
data:
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          jets: {global_object: true}
          tracks: {global_object: false, pad_max: 40}
```

### `salt.data.Features`

`salt/data/processors/features.py:30`, `__init__` `:59-64`

| `init_arg` | Type | Default |
|---|---|---|
| `variables` | `Mapping[str, Sequence[str]]` | required |
| `non_finite_to_num` | `bool` | `False` |
| `ignore_finite_checks` | `bool` | `False` |

`variables` maps a stream name to an **ordered** list of column names; an
empty or duplicate-containing list raises `ConfigError` (`:66-74`). This is
the one place input column order is defined.

Requires `raw.<stream>` (with `fields`) and optionally `masks.<stream>`.
Produces `inputs.<stream>` float32, with the same `fields` (`:79-91`). This
is the only shipped producer of `inputs.*`; see the correction above at
[`declare_io`](#declare_ioself-mode-mode-io).

```yaml
data:
  modules:
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt, eta, GN2v01_pb, GN2v01_pc, GN2v01_pu]
          tracks: [dphi, deta, qOverP, IP3D_signed_d0_significance]
```

### `salt.data.Labels`

`salt/data/processors/labels.py:17`, `__init__` `:72-78`

| `init_arg` | Type | Default |
|---|---|---|
| `streams` | `Sequence[str] \| None` | `None`, defaults to the reader's streams via `bind_streams` |
| `dtype_policy` | `Literal["int64-for-int", "file"]` | `"int64-for-int"` |
| `valid_ranges` | `Mapping[str, Sequence[int]] \| None` | `None` |
| `recover_malformed` | `bool` | `False` |

Every parameter has a default, so `labels: {class_path: salt.data.Labels}`
with no `init_args` is a complete entry.

Requires `raw.<stream>` per resolved stream. Produces the wildcard
`labels.**`, which the planner narrows to the concrete
`labels.<stream>.<label>` keys the task heads demand (`:107-119`).

Defines `bind(ctx)` (`:140-143`, learns its narrowed targets from
`ctx.step`) and `read_fields` (`:145-152`, maps each narrowed
`labels.<stream>.<label>` key back to the `raw.<stream>.<label>` read it
needs). `_parse_targets` (`:121-138`) raises `ConfigError` for a narrowed
key that does not split into exactly `labels.<stream>.<label>` (`:126-130`)
and for a stream outside the resolved set (`:132-136`).

### Compact reference: the remaining six

| `class_path` | Source | `init_args` | Requires / produces / hooks |
|---|---|---|---|
| `salt.data.VDS` | `readers/vds.py:222`, `__init__` `:251` | `out: dict[str, str \| Path] \| None = None` | setup-only: `declare_io` empty (`:272-277`); `declare_setup_io` requires `source.<reader>.<stage>.pattern` and produces `source.<reader>.<stage>.vds_path` (`:289-302`); `setup` builds a real VDS only when the reader is `vds_capable` and the pattern is a wildcard, otherwise passes the pattern through unchanged (`:304-327`) |
| `salt.data.FtagLabeller` | `processors/ftag_labeller.py:17`, `__init__` `:70-77` | `stream: str = "jets"`; `label: str = "flavour_label"`; `class_names: Sequence[str] \| None = None` (empty or `None` raises `ConfigError`, `:87-91`); `require_labels: bool = True`; `dtype_policy = "int64-for-int"` | produces a **concrete** `labels.<stream>.<label>` (`:102-112`), which beats `Labels`' wildcard; `read_fields` overridden (`:114-122`) |
| `salt.data.UprootReader` | `readers/uproot_reader.py:200`, `__init__` `:261-271` | `groups` (required); `filename=None`; `tree: str = "CollectionTree"`; `unroll: str \| None = None`; `num: int = -1`; `cuts=None`; `constituent_cuts=None`; `stage=None`; `index_cache: str \| Path \| None = None` | same produced key families as `H5StructuredReader` (`:454-471`); its `with_source` (`:581-593`) accepts `vds_path` for API parity only and discards it, because ROOT has no VDS |
| `salt.data.MultiSampleReader` | `readers/multisample_reader.py:82`, `__init__` `:124-131` | `samples: Sequence[SampleConfig \| Mapping]` (required); `label_stream: str = "event"`; `label_field: str = "process"`; `seed: int = 42`; `interleave_block: int = 1` | mirrors its sub-readers' `raw.*`/`masks.*`/`meta.rows`, extending the scalar `label_stream`'s field list with the injected `label_field` (`:198-233`); its `with_source` ignores `filename` and re-sources each sub-reader from its own per-stage `SampleConfig.sources` (`:402-442`) |
| `salt.data.MultiTarget` | `processors/multi_target.py:26`, `__init__` `:77` | `replacements: Sequence[Mapping[str, Any]]` (required) | requires `labels.<stream>.{sel_label,source}` and produces `labels.<stream>.<output>` float32, all `modes=Mode.TRAINING` (`:138-164`) |
| `salt.data.MaskFormerTargets` | `processors/maskformer_targets.py:35`, `__init__` `:134-149` | `object_class`, `object_id`, `constituent_id`, `class_map`, `object_stream`, `constituent_stream` (all required); then `regression_targets=None`, `cuts=None`, `sort_by=None`, `sort_descending=True`, `pv_class=0`, `max_objects=None`, `max_lxy_mm=None`, `lxy_field="Lxy"` | produces `labels.objects.object_class` (`[B,M]` int64), `labels.objects.masks` (`[B,M,T]` bool) and `labels.objects.<target>` per regression target (`:299-334`) |

## Reference

Every name below is exported from `salt/data/__init__.py`'s `__all__`
(34 names). The third column says what kind of thing it is, since a base
class, a config dataclass, a constant, a function and a runtime object all
sit in the same namespace as the ten real `data.modules` entries, and only
those ten belong under `data.modules` in a config.

| Class | Role | Legal `data.modules` entry? |
|---|---|---|
| `salt.data.SaltDataModule` | the Lightning datamodule you configure, never subclass | no, it is the datamodule itself |
| `salt.data.SaltDatasetModule` | base of everything under `data.modules` | no, base class |
| `salt.data.Reader` | base of a source node: file to `raw.*` batch dict | no, base class |
| `salt.data.Processor` | base of a batch transform | no, base class |
| `salt.data.WorkerCtx` | the per-worker context `bind(ctx)` receives | no, runtime dataclass |
| `salt.data.RowBlock` | one contiguous sequential read unit, for streaming | no, runtime dataclass |
| `salt.data.H5StructuredReader` | the structured-H5 reader | **yes** |
| `salt.data.UprootReader` | the ROOT/uproot reader | **yes** |
| `salt.data.MultiSampleReader` | proportionally stratified multi-sample reader | **yes** |
| `salt.data.VDS` | resolves a wildcard file pattern into a virtual dataset | **yes** |
| `salt.data.InputSamples` | the per-stage file-pattern setup module | **yes** |
| `salt.data.Features` | `raw.*` to `inputs.*` float32 materialisation | **yes** |
| `salt.data.Labels` | `raw.*` to `labels.**` truth extraction | **yes** |
| `salt.data.FtagLabeller` | on-the-fly label relabelling | **yes** |
| `salt.data.MultiTarget` | multiple regression targets from one stream | **yes** |
| `salt.data.MaskFormerTargets` | MaskFormer object-selection targets | **yes** |
| `salt.data.SaltDataset` | the map-style dataset `SaltDataModule` builds internally | no, internal runtime object; see [Shipped data modules](#shipped-data-modules) |
| `salt.data.IterableSaltDataset` | the streaming counterpart of `SaltDataset` | no, internal runtime object |
| `salt.data.GroupConfig` | one `H5StructuredReader` stream's group configuration | no, config dataclass (nested inside `groups:`) |
| `salt.data.StreamConfig` | the shared cut/sort/pad configuration for a jagged stream | no, config dataclass |
| `salt.data.UprootGroupConfig` | one `UprootReader` stream's group configuration | no, config dataclass (nested inside `groups:`) |
| `salt.data.SampleConfig` | one `MultiSampleReader` sample's sub-reader configuration | no, config dataclass (nested inside `samples:`) |
| `salt.data.ConstituentCuts` | a per-stream within-row (constituent) cut | no, config dataclass |
| `salt.data.Cut` | one named cut expression | no, config dataclass |
| `salt.data.GlobalObjectCuts` | sample-axis row eligibility, evaluated once in `prepare` | no, config dataclass |
| `salt.data.OffsetIndex` | a reader's cached row-to-file offset index | no, runtime dataclass |
| `salt.data.ManifestEntry` | one shard entry inside a `CorpusManifest` | no, runtime dataclass |
| `salt.data.CorpusManifest` | the streaming-mode shard-planning artifact | no, runtime dataclass; passed via `manifest:`, not `data.modules` |
| `salt.data.DEFAULT_BLOCK_ROWS` | the default `block_rows` value for streaming | no, constant |
| `salt.data.MODEL_VISIBLE_NAMESPACES` | the bundle namespaces a model may read | no, constant |
| `salt.data.build_manifest` | builds a `CorpusManifest` from a set of source files | no, function |
| `salt.data.create_vds` | builds a concrete h5py virtual dataset file | no, function, called by `VDS.setup` |
| `salt.data.default_vds_path` | the default output path `VDS` writes to when `out` is unset | no, function |
| `salt.data.has_wildcard` | whether a path string contains a glob wildcard | no, function |

Verified against `salt/data/__init__.py`'s `__all__`.

## Model-side `bind` vs. data-side `bind`

The two methods share a name and nothing else. Landing on this page or
[`model.md`](model.md) from a search for "bind", check which one you
actually have:

| | Data-side `bind` | Model-side `bind` |
|---|---|---|
| Signature | `bind(self, ctx: WorkerCtx) -> None` | `bind(self, schema: ResolvedSchema) -> None` |
| Defined on | `SaltDatasetModule` (`salt/data/base.py:138`) | `SaltModelModule` (`salt/model/base.py:75`) |
| Called by | `SaltDataset`, once per (worker process, plan) | `bind_all`, once per model, after every mode's plan compiles |
| Typical use | open a file handle, allocate a per-worker read buffer | build `torch.nn` layers sized from resolved widths |
| May touch a file | yes; the only data-module hook that may | no |

See [`model.md#bind`](model.md#bind) for the model-side entry in full.
