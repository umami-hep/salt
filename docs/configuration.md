# Configuration

This page is the settings reference: how a salt config is assembled from several files, what each top-level block does, how to delete or override a key, and the full set of `data:` and `model:` keys that do not have their own page. `outputs:` has its own page ([Outputs](outputs.md)), the module reference lives under [Writing modules](modules/index.md), and everything about running a training (`salt fit`, resuming, `torch.compile`, dataloader performance) is on [Training](training.md).

## How a config is assembled

Every `salt fit` / `salt test` invocation auto-loads `salt/configs/base.yaml` first; your own config (or configs) stack on top of it, so you only need to state what differs from the defaults (trainer, callbacks, seed).

**What `base.yaml` already gives you.** Every run auto-loads `seed_everything: 42`, Lightning trainer defaults (`accelerator: auto`, `devices: 1`, a `CometLogger`, `log_every_n_steps: 50`), and five callbacks: `checkpoint`, `progress`, `lr_monitor`, `model_summary`, `artifacts` (see the callbacks table below). None of this needs restating in your own config; override only the keys that differ.

A config file can list further files to merge underneath it with a top-level `include:` key, before jsonargparse ever sees the file:

```yaml
include: [../base_model.yaml, ../base_data.yaml]

model:
  init_args:
    modules:
      encoder:
        init_args: {num_layers: 8}
```

`include:` must sit at the file's top level, not nested under `data:` or `model:`. Each entry is resolved as an absolute path if given as one, otherwise tried first against the including file's own directory and then against the shipped `salt/configs/` root, so a config can name a sibling without knowing where the caller keeps it. An included file's own `include:` is expanded the same way, depth-first, with a cycle raising an error naming the chain. The including file's own keys win over anything it includes.

Stacking multiple `--config` flags on the command line does the same kind of merge: `salt fit --config a.yaml --config b.yaml` deep-merges the two files' dict-typed sections key-by-key, later file wins per key, and keys neither file mentions are untouched. This differs from stock jsonargparse, which replaces a whole dict-typed section wholesale when a later file touches any key inside it; salt's parser unions instead, so `b.yaml` can add one task to `model.init_args.modules` without silently dropping every other module `a.yaml` declared.

A key is deleted with `null`: `--model.init_args.modules.track_vertexing=null` on the command line, or `track_vertexing: null` in an override file, removes that module. CLI overrides otherwise use the dotted spelling of the YAML path you would otherwise write, and combine with `--config` in the same merge:

```bash
salt fit --config salt/configs/gn2v2-opendata.yaml \
  --model.init_args.modules.encoder.init_args.num_layers=3
```

To see the fully-resolved config after every include, merge and override has been applied, without training anything, see [`--print_config`](cli.md#previewing-the-resolved-config-print_config) in the command-line reference.

### Worked example: add a task from an override file

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

`class_names` is index-aligned with the label's on-disk integer values and sets the head width, because `output_size = len(class_names)` (`salt/model/modules/tasks/classification.py:170`). A wrong-length list is not caught by `salt graph validate`: `check_class_names` (`salt/model/saltmodule.py:1970`) runs only when the reader carries a `schema:` artifact, and returns immediately when it does not (`:1979-1980`). Without one, a short list fails on the first training batch with a `CrossEntropyLoss` index error, and a right-length but reordered list silently mislabels the head with no error at all. Check the list against the label column before you fit.

Every module the base config already declared survives the merge unchanged, and the new task's label is demanded from the dataset automatically. The task produces its loss at `losses.<its config key>` (`salt/model/modules/tasks/base.py:104`); the `LossSum` module already present in the base config narrows the framework's `losses.**` wildcard to every such key and sums them into the single scalar `loss.total` that FIT and VAL anchor on. You do not list the new task anywhere inside `LossSum` yourself, it is collected automatically. A config with no `LossSum` module fails FIT with the `ConfigError` that names the empty loss-key narrow, see [the minimum graph that compiles](#the-minimum-graph-that-compiles). Use `LossSum`'s `weights:` init arg only to re-weight one task's loss against another's, or set `weight:` on the task itself; either changes only the weighting, never which losses get collected. To persist its predictions in the eval file and declare its ONNX output, add the task's name to a `RunTaskOutput` `tasks:` list in the top-level `outputs:` section, one line, see [Declaring outputs](outputs.md#declaring-outputs-the-outputs-section). If the head is a training-time regulariser that must never reach eval or Athena, set `expose: [fit, val]` on the task instead: its prediction is pruned out of the TEST/ONNX plans while it keeps training.

### Task-head keys: `input`, `context` and `sequence`

Every task class accepts these three keys with the same meaning:

| Key | Meaning | Default |
| --- | --- | --- |
| `input` | the bundle key the head reads its features from | `encoded.<stream>` (`salt/model/modules/tasks/base.py:57`) |
| `context` | an extra conditioning vector concatenated into the head's `Dense`, for example `pooled.global` for a track head conditioned on the jet | unset |
| `sequence` | whether the head predicts per-token (`true`) or once per jet (`false`) | inferred as `input is None` when unset (`classification.py:108`, `regression.py:156`) |

A jet head that reads a pooled vector such as `pooled.global` must set `input:` explicitly. A per-token track head must leave `input:` unset and pass any jet-level conditioning through `context:` instead: setting `input:` on it silently converts it to a global head, because `sequence` then infers to `false`, the prediction shape changes from one row per constituent to one row per jet, the pad-mask require is dropped, and the H5 column axis changes. Set `sequence: true` explicitly only for a head that needs both an explicit `input:` and per-token predictions. `VertexingTaskModule` has no `sequence` parameter at all (`salt/model/modules/tasks/edge.py:43-57`); its `input`/`context` behave the same as the other two task classes.

When `context:` is set, the head's internal `Dense` gains an extra `context_size` input equal to the context vector's resolved width (`classification.py:171`), so the head's parameter count grows with the context stream, not just with `input:`.

## The minimum graph that compiles

A trainer config needs a specific minimum set of pieces before it compiles in any of the four primary modes (FIT, VAL, TEST, ONNX). Elsewhere on this page and on the [module reference](modules/index.md) pages you find how to write your own module; this section names what has to be *present* before any of that matters, because a config that is missing one of these pieces produces a specific, named error rather than a vague failure.

The block below is derived from the shipped, account-free `salt/configs/gn2v2-opendata.yaml`, trimmed to the smallest classification graph that still compiles. Nothing here is schematic: every entry is a real module with real `init_args`, taken from that file. Replace the file paths, the norm-dict path, and the input variable names with values from your own dataset and it runs as is.

```yaml
data:
  modules:
    input_samples:
      class_path: salt.data.InputSamples
      init_args:
        files: {train: ${DATA_TRAIN_PATH}, val: ${DATA_VAL_PATH}, test: ${DATA_TEST_PATH}}
        num: {train: -1, val: -1, test: -1}
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          jets: {global_object: true}
          tracks: {global_object: false, pad_max: 40}
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt_btagJes, eta_btagJes]
          tracks: [d0, z0SinTheta, qOverP]
    labels:
      class_path: salt.data.Labels

model:
  class_path: salt.model.SaltModule
  init_args:
    lrs: {initial: 1.0e-7, max: 5.0e-4, end: 1.0e-5, pct_start: 0.01}
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
        init_args:
          stream: tracks
          context: [normed.jets]
          out_dim: 64
          dense: {hidden_layers: [64], activation: ReLU}
      concat:
        class_path: salt.model.modules.Concat
        init_args: {streams: [tracks]}
      encoder:
        class_path: salt.model.modules.TransformerEncoder
        init_args:
          dim: 64
          out_dim: 64
          num_layers: 2
          attention: {num_heads: 4}
      pool:
        class_path: salt.model.modules.GlobalAttentionPooling
        init_args: {input: encoded.seq, out: pooled.global}
      jets_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: jets
          input: pooled.global
          label: flavour_label
          class_names: [bjets, cjets, ujets, taujets]
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args:
      tasks: [jets_classification]
```

`salt/configs/gn2v2-opendata.yaml` is the complete, runnable, production-scale version of this same graph, carrying two further task heads (`track_origin`, `track_vertexing`) and an ONNX export sink that this trimmed-down graph omits. That file also carries a `split` module (`salt.model.modules.Split`) feeding `encoded.tracks` to its per-track heads; the trimmed graph above has no track-level head, so `split` is left out rather than kept as a dead module.

Leaving any one of these pieces out raises a specific, named error:

| Omit | What happens |
| --- | --- |
| `salt.data.Features` | nothing produces `inputs.*`; the normaliser fails with `ConnectivityError` |
| `salt.data.Labels` (and no `FtagLabeller`) | nothing produces `labels.<stream>.<label>`; the task head fails with `ConnectivityError` |
| the reader, or a second one | `ConfigError`: `SaltDataModule needs exactly one Reader in modules, got N ([names])` |
| any task head | `LossSum` narrows to an empty loss-key list; `ConfigError` at `SaltModule.__init__` |
| `LossSum` | `ConfigError`: no module produces `loss.total` in mode FIT |
| the `outputs:` section, or a task missing from `RunTaskOutput.tasks` | TEST compilation fails; `ConfigError`: no module produces a `preds.*` key in mode TEST |
| `tasks: []` as a placeholder | `ConfigError` at construction: `RunTaskOutput` needs a non-empty `tasks` list (full rule at [Declaring outputs](outputs.md#declaring-outputs-the-outputs-section)) |
| `class_path: salt.model.SaltModule` under `model:` | the model is parsed in subclass mode (`salt/main.py:452`), so a bare `init_args` block is rejected |
| a module whose output nothing consumes | `AllModesDeadError` in a trainer config; only a warning in a toy `modules:` config |

`RunTaskOutput.tasks` must name at least one existing task instance. There is no empty or placeholder form for it, and a config with no task head cannot declare an `outputs:` section at all, because there is nothing for that section to name. Check a graph with no task head using the toy `modules:` format instead (below), not by writing an empty `outputs:` section. The full `RunTaskOutput` rule, including the merge-replaces-lists warning, is at [Declaring outputs](outputs.md#declaring-outputs-the-outputs-section).

Which pieces matter depends on the mode. FIT and VAL only need the data half (a reader, `Features`, `Labels`) plus a `LossSum` producing `loss.total`, because those two modes anchor on that key. TEST and ONNX additionally need at least one task head and an `outputs:` section naming it, because those two modes anchor on a `preds.*` key instead, see [`salt graph`](cli.md#salt-graph) for the command that checks each mode on its own.

Three further things worth knowing about this graph:

- `data:` is not parsed in subclass mode, so it needs no `class_path`; `model:` is, so it does (`salt/main.py:452`). Every shipped config carries `class_path: salt.model.SaltModule`, for example `gn2v2-opendata.yaml:58`.
- `train_file` / `val_file` / `test_file` are the legacy path superseded by `InputSamples`. When no explicit `InputSamples` module is configured and one of those legacy keys is set, `SaltDataModule` synthesises one automatically (`salt/data/datamodule.py:361-399`); see [Shipped data modules](modules/data.md#shipped-data-modules) for the full mechanism.
- `salt.data.SaltDataset` is the internal runtime object the datamodule builds for you; it is never a `data.modules` entry.

**Checking a graph without data.** `salt graph validate -c cfg.yaml` compiles all four modes against the config alone, no data file or checkpoint read; add `--mode fit` to check only the data half when there is not yet a task head or an `outputs:` section, see [`salt graph`](cli.md#salt-graph). Every shipped module's full `init_args` are catalogued at [Shipped model modules](modules/model.md#shipped-model-modules) and [Shipped data modules](modules/data.md#shipped-data-modules).

This is why building a graph piece by piece works well in practice, running `salt graph validate --mode fit` after each of the first three steps and a bare `salt graph validate` (no `--mode`, checking all four modes) after the fourth:

1. Add the reader alone.
2. Add `Features` and `Labels`.
3. Add the model modules, one at a time, ending at `LossSum`.
4. Add a task head and the `outputs:` section naming it.

Each step either passes or names exactly the missing piece from the table above, which is a shorter loop than writing the whole graph first and debugging one error message against an unfamiliar config.

## The blocks of a config

A salt config has up to six top-level blocks. `base.yaml` supplies defaults for most of them; a run config states only what it changes.

### `data:`

Configures `SaltDataModule`. `data.modules` is a dict of named `SaltDatasetModule`s, exactly one `Reader` plus any number of processors; a `null` entry deletes a module. The reader and processor surfaces (`groups:`, `variables:`, `label_map`, and so on) are covered section by section below. The datamodule's own constructor keys, straight from its docstring:

| Key | Meaning |
| --- | --- |
| `train_file` / `val_file` / `test_file` | per-stage input file path; a wildcard filename triggers VDS creation (see below) |
| `batch_size` | rows per contiguous batch slab, default `1000` |
| `num_workers` | dataloader worker processes, default `0` |
| `num_train` / `num_val` / `num_test` | row counts per stage; `-1` means all |
| `test_suff` | suffix appended to the eval-file `{sample}` name by the writer callback |
| `move_files_temp` | opt-in staging root (for example `/dev/shm/<user>/tmp`); each per-stage reader restages its own file(s) there at `setup` and the root is removed at `teardown`; ignored under `fast_dev_run` |
| `train_vds_path` / `val_vds_path` / `test_vds_path` | explicit VDS output paths for wildcard files |
| `pin_memory` | pin host memory for faster GPU transfer, default `true` |
| `persistent_workers` | keep worker processes and their H5 handles alive between epochs, default `true` |
| `prefetch_factor` | batches prefetched per worker; unset derives one (`2` for the map-style path, more for `iterable`, see [Dataloading performance](training.md#dataloading-performance)) |
| `multiprocessing_context` | worker start method: `fork`, `spawn`, or `forkserver` |
| `seed` | base augmentation seed for non-worker reads, default `42` |
| `debug` | enable the boundary non-aliasing assertion |
| `iterable` | stream a large corpus with `IterableSaltDataset` instead of the map-style dataset, default `false`; the streaming-specific keys below only matter when this is on |
| `block_rows` | rows per reader call when `iterable`, default `16384` |
| `interleave_block` | rows per turn of the per-shard round robin when `iterable`, default `1` |
| `max_live_streams` | samples holding a resident block at once when `iterable`, default `2` |
| `manifest` | `CorpusManifest` path or per-stage mapping, required when `iterable` |
| `shuffle_stream` | shuffle block order and within-batch row order on the fit streaming loader, default `true` |

The streaming keys (`iterable` onward) are covered in full at [Dataloading performance](training.md#dataloading-performance).

### `model:`

Configures `SaltModule` (`class_path: salt.model.SaltModule`). Its `init_args` carry `lrs`, `optimizer`, `mup` (see [muP](#mup) below), and `modules`, a dict of named `SaltModelModule`s in the same `{name: {class_path, init_args}}` shape as `data.modules`. Every class you can put here, and how to write your own, is the [module reference](modules/index.md); the full `init_args` of every shipped module are catalogued at [Shipped model modules](modules/model.md#shipped-model-modules).

`optimizer:` accepts exactly four values, and the match is case-sensitive: `AdamW` (the default), `lion`, `lion-pytorch`, `HybridMuonAdamW` (`salt/model/saltmodule.py:86`). Any other value, including a differently-cased spelling such as `adam`, is rejected at `SaltModule.__init__` with:

```
optimizer 'adam' is not supported — choose from ['AdamW', 'lion', 'lion-pytorch', 'HybridMuonAdamW']
```

(`salt/model/saltmodule.py:258-260`.)

`optimizer:` is not the last word when `mup:` is also configured: with `mup` set, the optimizer is swapped to `mup.optim.MuAdamW` regardless of whatever name `optimizer:` names (`salt/model/saltmodule.py:189-190`). See [muP](#mup) below for the rest of that mechanism.

A task's `loss:` key is optional. Each task family falls back to its own default when `loss:` is unset:

| Task class | Default loss | Where |
| --- | --- | --- |
| `ClassificationTaskModule` | `torch.nn.CrossEntropyLoss` | `salt/model/modules/tasks/classification.py:26` |
| `RegressionTaskModule` | `torch.nn.MSELoss`, or `torch.nn.GaussianNLLLoss` when `gaussian: true` | `tasks/regression.py:24-25`, selected at `:147-148` |
| `VertexingTaskModule` | `torch.nn.BCEWithLogitsLoss(reduction="none")` | `tasks/edge.py:21-24` |

Set `loss:` to override this: either a bare `torch.nn` class name (`loss: MSELoss`), which resolves under `torch.nn` because a dotless name is looked up there (`tasks/base.py:283-299`), or a `{class_path, init_args}` mapping for a loss that needs constructor arguments or lives outside `torch.nn`, for example:

```yaml
loss: {class_path: torch.nn.SmoothL1Loss, init_args: {beta: 0.5}}
```

### `outputs:`

A dict of named output sinks and writers controlling what `salt test` writes to the eval H5 and what `salt export` puts in the ONNX graph. This is fully documented at [Declaring outputs](outputs.md#declaring-outputs-the-outputs-section); this page does not restate it.

### `trainer:`

Passed straight through to Lightning's `Trainer`. `base.yaml` sets `accelerator: auto`, `devices: 1`, a `CometLogger`, and `log_every_n_steps: 50`. `trainer.callbacks` is reserved for stock Lightning callbacks; salt's own callbacks belong under `callbacks:` below, not here. `--trainer.default_root_dir` and the run-directory layout it controls are covered at [`salt fit`](cli.md#salt-fit).

### `callbacks:`

A dict of named callbacks, deep-merged the same way as `data.modules` and `model.init_args.modules`; a `null` entry removes one. `base.yaml` ships:

| Key | Class | Purpose |
| --- | --- | --- |
| `checkpoint` | `salt.callbacks.Checkpoint` | writes `epoch=NNN-loss=<val/loss>.ckpt` under `ckpts/`, monitoring the metric named by its own `monitor_loss` init arg (default `val/loss`) — this is salt's parameter name, not Lightning's `monitor`. Keep the `loss=` filename stem in sync with any `fname_string` override: `salt test` run without `--ckpt_path` resolves the best epoch by globbing `{ckpts,checkpoints}/*.ckpt` and parsing that stem. |
| `progress` | `salt.callbacks.ProgressBar` | training progress bar |
| `lr_monitor` | `lightning.pytorch.callbacks.LearningRateMonitor` | logs the learning rate; dropped automatically when no logger is attached (it hard-raises on a logger-less trainer), or delete it explicitly with `lr_monitor: null` |
| `model_summary` | `lightning.pytorch.callbacks.ModelSummary` | prints the module tree at fit start |
| `artifacts` | `salt.callbacks.GraphArtifacts` | writes `plan_<mode>.txt` and `graph_<stage>.{dot,svg}` into the trainer log dir at fit/test start |

### `name:`

A plain string naming the run, default `"salt"` when unset. It becomes the Comet experiment name (set as `experiment_name` on the logger's `init_args`, or the `COMET_EXPERIMENT_NAME` environment variable on a Comet build whose constructor no longer declares that parameter) and is also written into the logger's `dict_kwargs.name`.

## Deleting and overriding

A module or callback entry is deleted with `null`: `--model.init_args.modules.track_vertexing=null`, or the equivalent in an override file. This works because `data.modules`, `model.init_args.modules` and `callbacks:` are each filtered at assembly time (by `SaltDataModule`, `SaltModule`, and the CLI respectively): a `None` entry is dropped before the module dict reaches its consumer.

That deletion mechanism is specific to those three dicts. It does not extend to a nested key inside one module's own `init_args`. Because `--config` stacking (and `include:`) unions dict-typed sections key-by-key rather than replacing them, an overlay that restates only part of a nested dict does not drop the base's other entries under that same key: `groups: {jets: {...}}` in an override adds or updates the `jets` group in the reader's `groups:` dict, but any `tracks` or `flows` group the base config declared survives untouched, merged straight in. The same holds for `variables:` under `Features`. Setting a group to `null` does not delete it either; `GroupConfig` treats a `None` (or empty) value as "use the defaults for this stream", so `groups: {tracks: null}` keeps the `tracks` group with its dataset name defaulted to `tracks`, not remove it. To drop a group or a variable that an inherited config declared, stop inheriting that block: write a config that does not `include:` the file whose `groups:`/`variables:` you want to shrink, or delete the whole reader/processor module by name and declare a fresh one under a new name with the smaller dict.

Renaming a module to a new key is also how you force a clean swap under `--init_from` warm-starting, and the general pattern behind every kind of fine-tuning surgery: a module present in both the checkpoint and your config but only partially compatible (different width, a missing sub-key) is a hard error, while a renamed module has no checkpoint counterpart at all, so the old weights are dropped and the new name is treated as a fresh, randomly-initialised module. See [What else can you change?](finetuning.md#what-else-can-you-change) for the full cost table across every kind of module surgery.

### Wildcard files and the VDS

Training files can grow large enough that they are split into several smaller ones. Point `train_file` (or `val_file` / `test_file`) at a glob pattern and salt reads them as one:

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
```

A wildcard filename triggers Virtual Dataset (VDS) creation, using the VDS support in [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools). The VDS is an HDF5 file of external links into the real member files, so the reader sees one contiguous dataset. It is built once (a `FileLock` plus a `.done` marker keep concurrent workers or DDP ranks from racing the build) and rebuilt automatically if any member file is newer than the existing VDS. By default it lands next to the wildcard, at a sibling directory: the pattern `pp_output_train_split_*.h5` writes to `pp_output_train_split_vds/vds.h5`. Give it an explicit path instead with `train_vds_path`:

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
  train_vds_path: /path/to/something/else/my_train_vds.h5
```

The same applies per stage:

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
  train_vds_path: /path/to/something/else/my_train_vds.h5
  val_file: /path/to/somewhere/pp_output_val_split_*.h5
  val_vds_path: /path/to/something/else/my_val_vds_file.h5
  test_file: /path/to/somewhere/pp_output_test_split_*.h5
  test_vds_path: /path/to/something/else/my_test_vds_file.h5
```

### Choosing input variables

Training files are structured arrays, so the variables that end up in the model are whichever ones you list, by name, in the `Features` processor's `variables:` key, one list per stream:

```yaml
data:
  modules:
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          jets: [pt_btagJes, eta_btagJes]
          tracks: [d0, z0SinTheta, dphi, deta, ...]
```

??? warning "Don't train on truth information"

    `Features` has no way to tell a truth variable from an input variable; if
    you list one, it is treated as an input. Keep truth information out of
    `variables:` and specify it as a task's `label` instead.

The number of variables listed for a stream sets the `[B, T, F]` (or `[B, F]` for a `global_object` stream) width the planner resolves for `inputs.<stream>`. A downstream module's own output width, such as `StreamEmbed`'s `out_dim`, is a separate, independently-configured number; the planner infers and checks the input width at `bind` time from the resolved schema, it is not something you set to match `variables:` by hand. Heterogeneous models with more than one input type are covered at [More than one input stream](#more-than-one-input-stream) below.

### Naming streams and groups

By default a stream's name in the config is also the H5 dataset name the reader reads. To read from a differently-named dataset, set `dataset:` on that stream's entry in the reader's `groups:` dict:

```yaml
data:
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          tracks: {global_object: false, dataset: tracks_ghost}
```

Here the stream is called `tracks` everywhere else in the config (in `variables:`, in a `StreamEmbed`'s `stream:`, and so on), but the reader pulls it from the H5 dataset named `tracks_ghost`. Leaving `dataset:` unset (or the whole group value empty, `tracks: {}`) defaults it to the stream name.

### Limiting constituents per row

`GroupConfig.pad_max` caps a stream at N constituents per row: shorter sequences are padded up to N, longer ones truncated down to N. Set it per stream on the reader:

```yaml
data:
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          tracks: {pad_max: 10}
```

### Remapping labels

Task modules can remap on-disk label values to a smaller or reordered set with `label_map`, useful when the values are not already `0, 1, 2, ...`. For example, to train on the raw `HadronConeExclTruthLabelID` PDG-style ids instead of the pre-mapped `flavour_label`:

```yaml
jets_classification:
  class_path: salt.model.modules.ClassificationTaskModule
  init_args:
    stream: jets
    input: pooled.global
    label: HadronConeExclTruthLabelID
    label_map: {0: 0, 4: 1, 5: 2}
    class_names: [ujets, cjets, bjets]
    dense: {hidden_layers: [128, 64, 32], activation: SiLU}
```

`class_names` is required whenever `label_map` is set; the head width is `len(class_names)`. Using `flavour_label` as the label needs no `label_map` at all, since the file already carries `0, 1, 2, ...`. See `salt/configs/GN3/GN3_Charge.yaml` for a real ~300-entry `label_map`.

### Deriving finer labels

To relabel on the fly, for example splitting a preprocessed `qcd` class into finer subclasses, or deriving `flavour_label` for a test file that was never run through UPP, wire the `salt.data.FtagLabeller` processor as its own `data.modules` entry:

```yaml
data:
  modules:
    labeller:
      class_path: salt.data.FtagLabeller
      init_args:
        stream: jets
        label: flavour_label
        require_labels: true
        class_names: [htautauhad, hbb, hcc, top, qcdbb, qcdbx, qcdcx, qcdll, Wqq]
```

`class_names` lists the derived classes in label-index order and is required. `require_labels: true` (the default) raises if any object matches none of the classes; `false` drops unmatched objects instead, so the derived label array can end up shorter than the batch. `FtagLabeller` becomes the sole producer of `labels.<stream>.<label>`, so a task's `class_names` must match this processor's `class_names` index-for-index. See `salt/configs/GN3X.yaml` for the real config this example is drawn from (GN3X derives its boosted-Higgs classes from `R10TruthLabel_R22v1` plus ghost-hadron counts, since they are not a precomputed column).

### Multiple regression targets

`MultiTarget` replaces or creates a label conditionally, row by row, from a list of rules evaluated in order:

```yaml
data:
  modules:
    multi_target:
      class_path: salt.data.MultiTarget
      init_args:
        replacements:
          - stream: jets
            sel_label: HadronConeExclTruthLabelID
            op: "=="
            value: 15
            source: HadronConeExclTruthLabelPt
            custom_target: pt_label_handle
          - stream: jets
            sel_label: HadronConeExclTruthLabelID
            op: "!="
            value: 15
            source: pt
            custom_target: pt_label_handle
```

Each rule needs `stream`, `sel_label`, `op` (one of `== != >= <= > <`), `value`, `source`, and exactly one of `target:` (replace an existing label) or `custom_target:` (create a new one, `nan`-filled where no rule matches). Multiple rules may target the same output; they apply in order over a running array, which is how the example above builds one `pt_label_handle` covering both the tau and non-tau cases. A regression task then reads it like any other label:

```yaml
reg_multi_target:
  class_path: salt.model.modules.tasks.RegressionTaskModule
  init_args:
    stream: jets
    input: pooled.global
    targets: pt_label_handle
    norm_params: {mean: 1.0, std: 1.0}
    loss: MSELoss
    dense: {hidden_layers: [128, 64, 32], activation: SiLU}
    # MultiTarget labels exist only in FIT|VAL, so TEST cannot demand or dump them
    write_targets: false
```

The full example, including the model and outputs blocks, is `salt/configs/regression/regression_multi_target.yaml`.

### Reading from S3

`salt.utils.file_utils` can read training data and configs from an S3 bucket. Set up your own bucket and keys with the [CERN OpenStack project](https://clouddocs.web.cern.ch/index.html), then add a `config_s3` block under `data:`:

```yaml
data:
  config_s3:
    use_S3: false        # true if this run needs S3 access at all
    download_S3: false   # true to download download_files locally before training
    pubKey:               # public key
    secKey:               # private key
    url: https://s3.cern.ch
    bucket:                # bucket name
    download_path:         # local path the files are downloaded to
    download_files:        # keys under data: to fetch, e.g.
      - train_file
      - val_file
      - norm_dict
      - class_dict
```

There is no `salt` subcommand that reads `config_s3` automatically today; call `import_data_S3` yourself before `salt fit`/`salt test`. It downloads every file in `download_files` to `download_path` in parallel, rewrites their paths in the config, and writes the patched config to `<download_path>/local_base.yaml`:

```bash
python -c "from salt.utils.file_utils import import_data_S3; \
  print(import_data_S3('salt/configs/gn2v2-opendata.yaml'))"
```

Pass the printed path to `salt fit --config <that path>`. The S3 client libraries (`boto3`, `s3fs`, `s3path`) ship in the `muP` pip extra (`pip install 'salt-ml[muP]'`), not a dedicated `s3` extra.

If you also want the trainer itself to write checkpoints and configs to S3, point `trainer.default_root_dir` at an S3 URL and swap the logger, since the default `CometLogger` does not work with an S3 root:

```yaml
trainer:
  default_root_dir: s3://BUCKET/FOLDER
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
```

### Global-object features

A stream can be a per-object vector rather than a padded sequence: set `global_object: true` on it in the reader's `groups:`, and list its variables under `Features` like any other stream. By default such a stream is concatenated as context onto every other stream's `StreamEmbed`, before the encoder (see `context:` in [More than one input stream](#more-than-one-input-stream)). To instead concatenate it onto the pooled representation, after the encoder, give it its own `Normaliser` and combine with `VectorConcat`:

```yaml
data:
  modules:
    reader:
      class_path: salt.data.H5StructuredReader
      init_args:
        groups:
          global: {global_object: true}
    features:
      class_path: salt.data.Features
      init_args:
        variables:
          global: [softMuon_pt, softMuon_dR, ...]

model:
  init_args:
    modules:
      norm_global:
        class_path: salt.model.modules.Normaliser
        init_args: {streams: [global], global_object: global}
      pool:
        class_path: salt.model.modules.GlobalAttentionPooling
        init_args: {input: encoded.seq, out: pooled.global}
      vconcat:
        # concat order fixes the output feature order: pooled first, global last
        class_path: salt.model.modules.VectorConcat
        init_args: {inputs: [pooled.global, normed.global], out: vconcat.global}
      jets_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args: {input: vconcat.global, ...}
```

`VectorConcat`'s output width is `sum` of its inputs' widths, resolved at `bind`. The full worked example is `salt/configs/GN2/GN2emu.yaml`.

### Edge features

`salt.model.modules.EdgeFeatures` builds pairwise edge features between constituents on the fly from raw (un-normalised) track variables. Currently implemented:

- `dR` = log(sqrt(deta^2 + dphi^2)), requires `phi`, `eta`
- `kt` = log(min(pt) * sqrt(deta^2 + dphi^2)), requires `pt`
- `z` = log(min(pt) / sum(pt)), requires `pt`, `phi`, `eta`
- `isSelfLoop`: 1 if the edge is a self-connection, 0 otherwise
- `subjetIndex`: 1 if both tracks are in the same subjet, 0 otherwise; requires `subjetIndex`
- `mass`: the pairwise invariant mass; requires `pt`, `eta`, `phi`, `energy`

```yaml
model:
  init_args:
    modules:
      edge_features:
        class_path: salt.model.modules.EdgeFeatures
        init_args:
          stream: tracks
          features: [dR, z, kt, subjetIndex, isSelfLoop]
      edge_embed:
        class_path: salt.model.modules.EdgeEmbed
        init_args:
          stream: tracks
          out_dim: 32
          dense: {hidden_layers: [32], activation: SiLU}
```

`features` names cannot repeat and must come from the list above; the variables each one requires are checked at `bind` against the resolved input fields. See `salt/configs/GN2/GN2XE.yaml` for the full config.

### Vertexing origin weighting

`salt.model.modules.tasks.VertexingTaskModule`'s per-edge loss needs to know which
track-origin classes count as heavy-flavour and which count as fake, so it can weight
edges accordingly. The `origin_weighting` init arg supplies this split; leaving it unset
falls back to the tagger default, `{"heavy": [3, 4, 5], "fake": [1]}`:

```yaml
model:
  init_args:
    modules:
      track_vertexing:
        class_path: salt.model.modules.tasks.VertexingTaskModule
        init_args:
          origin_label: ftagTruthOriginLabel
          origin_weighting: {heavy: [3, 4, 5], fake: [1]}
```

Each of `heavy`/`fake` is a list of either integer origin ids or origin class-name
strings, never a mix of the two within one list; any key other than `heavy`/`fake` is
rejected. Class names are resolved to ids at fit/test setup, against the origin label's
class-name attr in the dataset's schema artifact. A name-based `origin_weighting` used
without such a schema artifact therefore fails at bind, with a `ConfigError` naming the
`origin_weighting` config key; dump a schema that carries the origin class names (see
[`salt schema dump`](cli.md#salt-schema-dump)) or write integer origin ids instead.
`origin_label:` names the label the weighting reads (`labels.<stream>.<origin_label>`).

### More than one input stream

When a model reads more than one input type, give each stream its own `StreamEmbed` under a distinct module name, then combine the embedded streams with `Concat` before the encoder:

```yaml
model:
  init_args:
    modules:
      track_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: tracks
          context: [normed.jets]
          out_dim: &embed_dim 512
          dense: {hidden_layers: [512], activation: SiLU}
      flow_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: flows
          context: [normed.jets]
          out_dim: *embed_dim   # Concat requires equal embed width
          dense: {hidden_layers: [512], activation: SiLU}
      concat:
        class_path: salt.model.modules.Concat
        init_args: {streams: [tracks, flows]}
```

Each `StreamEmbed`'s `out_dim` must match across streams that feed the same `Concat`. The full config is `salt/configs/GN3/GN3V00.yaml`.

## muP

Salt is compatible with the muTransfer technique outlined in the paper [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466).

`salt mup-shapes` generates the `shape_path` artifact used below, `salt mup-coord-check` runs the coordinate check against it, and the `setup_mup` console script is a thin alias that forwards to `salt mup-shapes`. The live config surface is `mup: true` on the modules being scaled plus `model.init_args.mup.apply_to` (see `salt/configs/GN2/GN2_muP.yaml`), not the `mup_config` key the walkthrough below names.

### Setup

To setup mup, the model configuration (e.g., `GN2.yaml`) has to include the following extra-configuration setup to be placed under the `config.model` (e.g., after the model's other settings and before `model.model`):

```yaml
mup_config:
    shape_path: my_path_to_a_folder_for_shape
    embed_dim:
      apply_to: [init_nets, encoder]
      parameter_name: [output_size, embed_dim]
      parameter_base: 128
      parameter_delta: 4
```

Such that the `base` (`delta`) models are instantiated with the parameters highlighted in `parameter_name`, respectively corresponding to the module `apply_to`, taking the value `parameter_base` (`parameter_delta`). The `storeshapes` file will be placed at the path `shape_path` or, if this parameter is not set, at `./temp_mup/` with the `base` and `delta` models as well as their configuration (useful to debug they were correctly setup).

To run a GN2 training with mup, you also need to specify in `encoder` (and the `init_nets` if it is affected) config that it should be in `mup` configuration with the following boolean parameters:

- for `init_nets` (only if changing embedding dim):

```yaml
init_nets:
    - input_name: tracks
        dense_config:
            ...
            mup: True
```

- for `encoder`:

```yaml
encoder:
    class_path: salt.model.modules.TransformerEncoder
    init_args:
        ...
        mup: True
```

### Run

To run mup, you must instantiate a GN2 model into the Maximal Update Parametrisation (mup). To do this, you must follow the following steps, which are further detailed next.

- step 1: create `storeshapes` file using a model config file with mup configuration:

```bash
setup_mup -config GN2/GN2.yaml
```

- step 2: run a mup training normally with the model config with mup configuration:

```bash
salt fit --config GN2/GN2.yaml
```

The config file `GN2_mup.yaml` gives an example of a valid configuration file for mup.

A gentle introduction to mup is available in this [talk](https://indico.cern.ch/event/1339085/#3-mup-for-gn2-hyperparameter-o).

Important note: mup has been implemented to scale the transformer encoder (and init_nets if the embedding is changed). The last layer in the scaling __must__ be the out-projecting of the encoder (controlled with `out_dim`), which in particular must be set!

**Step 1:**

To leverage the existing [mup library](https://github.com/microsoft/mup), a `base` and `delta` models have to be instantiated using the `main_mup` script to generate a `storeshapes` file to be passed to the mup library. Note that you __must__ vary a parameter between the `base` and `delta` models, as this will define the dimension to muTransfer along (embedding dimension and num_heads are supported). This script is installed with salt and callable under the name `setup_mup`. For example, run:

```bash
setup_mup -c GN2/GN2.yaml
```

Where the `GN2.yaml` is your usual model configuration file, endowed with the following extra-configuration setup to be placed under the `config.model` (e.g., after the model's other settings and before `model.model`):

```yaml
mup_config:
    shape_path: my_path_to_a_folder_for_shape
    embed_dim:
      apply_to: [init_nets, encoder]
      parameter_name: [output_size, embed_dim]
      parameter_base: 128
      parameter_delta: 4
```

The `setup_mup` script will instantiate a `base` (`delta`) model with the parameters highlighted in `parameter_name`, respectively corresponding to the module `apply_to`, taking the value `parameter_base` (`parameter_delta`). The `storeshapes` file will be placed at the path `shape_path` or, if this parameter is not set, at `./temp_mup/` with the `base` and `delta` models as well as their configuration (useful to debug they were correctly setup). Note: currently supporting the num_heads & embedding size of the transformer `encoder`, with the latter being also relevant to `init_nets`. Both the base and delta value have to be divided by your chosen `num_heads`!

**Step 2:**

With step 1 creating a `storeshapes` under the path `shape_path` or the default `./temp_mup`, you can now turn to training a GN2 models with your desired widths. The model will have to load the `storeshapes` in the initialiser of `ModelWrapper`, and you must make sure the model has the mup_config passed to it with, in particular, the right path to the `storeshapes` (easiest is to not change the config w.r.t. base and delta model initialisation).

To run a GN2 training with mup, you also need to specify in `encoder` (and the `init_nets` if it is affected) config that it should be in `mup` configuration with the following boolean parameters:
- for `init_nets` (only if changing embedding dim):
```yaml
init_nets:
    - input_name: tracks
        dense_config:
            ...
            mup: True
```
- for `encoder`:
```yaml
encoder:
    class_path: salt.model.modules.TransformerEncoder
    init_args:
        ...
        mup: True
```

If correctly setup, you can just run a salt training in the usual way:
```bash
salt fit --config GN2/GN2.yaml
```

You are now training a mup-GN2!



