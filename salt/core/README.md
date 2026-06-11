# salt v2 (`salt.core`) — the `salt2` surface

The modular salt v2 stack: graph-of-modules models, a demand-driven dataset
pipeline, and a jsonargparse YAML surface. Everything here is config-first:
modules declare their inputs/outputs (`declare_io`), plans are compiled
statically, and no data file is touched before the run starts (design §2.3).

> Design doc references (`§…`) point at the v2 design document
> (`plans/02_design-doc.md` in the modularise-salt study).

## Quickstart: train GN2v2 on a dummy file

From the repo root (use `python -m salt.core.main` wherever the `salt2`
console script is not installed):

```bash
# generate a dummy training file + norm dict (python console):
#   from salt.tests.core.gn2_fixture import write_parity_norm_dict
#   from salt.utils.inputs import write_dummy_file
#   write_parity_norm_dict("/tmp/v2/norm_dict.yaml", "/tmp/v2/class_dict.yaml")
#   write_dummy_file("/tmp/v2/train.h5", "/tmp/v2/norm_dict.yaml")

salt2 fit --config salt/core/configs/gn2v2-dummy.yaml \
  --data.train_file /tmp/v2/train.h5 \
  --data.val_file   /tmp/v2/train.h5 \
  --model.modules.norm.init_args.norm_dict /tmp/v2/norm_dict.yaml
```

Recommended extra: a **schema artifact** (`salt2 schema dump /tmp/v2/train.h5
-o /tmp/v2/schema.yaml`, design §2.6) passed as
`--data.modules.reader.init_args.schema /tmp/v2/schema.yaml`. With it, field
typos fail statically before any data is read, and the configured
`class_names` lists are cross-checked (set AND order) against the file's
label attrs.

**Where outputs land**: until the M6 run-dir layout, `config.yaml` and
`checkpoints/` are written into the trainer log dir — your cwd unless you
pass `--trainer.default_root_dir <dir>`. The end-of-fit message prints the
exact paths. A leftover `config.yaml` from a previous run is overwritten.

## Config model (design §5)

- `base2.yaml` is auto-loaded; your config stacks on top.
- `model.init_args.modules` and `data.modules` are **dicts of named
  modules** (`{name: {class_path, init_args}}`). The dict key is the
  instance name and the config address.
- `--config a.yaml --config b.yaml` **deep-merges dict leaves key-by-key**
  (later file wins per key; siblings survive — unlike stock jsonargparse).
- CLI overrides use natural dotted spelling:
  `--model.modules.encoder.init_args.num_layers=3`.
- **Delete by null**: `--model.modules.track_vertexing=null` (or
  `track_vertexing: null` in an override file) removes a module.
- `--print_config` shows the fully-resolved config.

### Worked example: add an aux task from an override file

```yaml
# my_aux_task.yaml — stack with: --config salt/core/configs/gn2v2-dummy.yaml --config my_aux_task.yaml
model:
  init_args:
    modules:
      track_type:
        class_path: salt.core.nn.tasks.ClassificationTaskModule
        init_args:
          stream: tracks
          context: pooled.global
          label: ftagTruthOriginLabel
          class_names: [a, b, c]
          dense: {hidden_layers: [16], activation: ReLU}
```

All existing modules survive the merge; the new task's label is demanded
from the dataset automatically and its loss joins `loss.total` via the
`losses.**` collection (design §3.3).

## Static graph tooling (design §4)

All `salt2 graph` subcommands accept BOTH the trainer configs above and the
small M1 toy-graph format, and never touch data:

```bash
salt2 graph validate -c salt/core/configs/gn2v2-dummy.yaml \
  --set model.modules.norm.init_args.norm_dict=unused.yaml
salt2 graph plan -c <cfg> --mode fit
salt2 graph plot -c <cfg> --mode fit -o graph.svg
salt2 graph why  -c <cfg> --mode fit --key encoded.seq
salt2 graph deadcode -c <cfg>
```

`--set KEY=VALUE` (repeatable) supplies required init_args **data-free** —
e.g. the `norm_dict` path that the shipped configs leave as a required
override is never read by static tooling, so any value works. `validate`
also runs the §2.6 class-names ↔ schema-attrs cross-check when the reader
has a schema artifact.

## Notable M2 surface notes

- `lrs_config:` is the design §5.1 `lrs:` block under its v1 name (rename is
  an M3 cleanup). Schema: `{initial, max, end, pct_start[, weight_decay,
  last_epoch]}` driving AdamW/lion/HybridMuonAdamW + OneCycleLR.
- `VertexingTaskModule.origin_weighting` takes integer origin ids in M2
  (defaults reproduce v1's `heavy: [3,4,5], fake: [1]`); the design's
  name-based form lands in M3.
- `writers:` parses but is ignored until the M3 writer modules.
- Loggers (Comet) and run dirs are M6; losses show on the stock progress
  bar meanwhile (`train/loss`, `train/<task>_loss`).

## Checkpoints and resume (design §2.3)

Checkpoints carry the resolved schema + per-mode plan hashes under the
top-level `salt_core` key. On resume, a FIT plan-hash mismatch is fatal
(the graph or dataset boundary changed); `Normaliser`/class-weight values
come from the state dict — the norm/class dicts are NOT re-read.
Data-less loading: `SaltModule.load_from_checkpoint(path, modules=...)`.
