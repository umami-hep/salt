# salt v2 (`salt.core`) — the `salt2` surface

The modular salt v2 stack: graph-of-modules models, a demand-driven dataset
pipeline, and a jsonargparse YAML surface. Everything here is config-first:
modules declare their inputs/outputs (`declare_io`), plans are compiled
statically, and no data file is touched before the run starts (design §2.3).

> Design doc references (`§…`) point at the v2 design document
> (`plans/02_design-doc.md` in the modularise-salt study).

## Quickstart: train GN2v2 on a dummy file

One-time setup: if the container's installed salt predates `salt/core/`,
`python -m salt.core.main` fails with `ModuleNotFoundError: No module named
'salt.core'` from any cwd except the repo root, and there is no `salt2`
script on PATH. `pip install -e .` (from the repo root, inside the
container) fixes both — afterwards `salt2` works from any directory.

```bash
# generate a dummy training file + norm dict (copy-paste):
python -c "
from pathlib import Path
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file
Path('/tmp/v2').mkdir(parents=True, exist_ok=True)
write_parity_norm_dict('/tmp/v2/norm_dict.yaml', '/tmp/v2/class_dict.yaml')
write_dummy_file('/tmp/v2/train.h5', '/tmp/v2/norm_dict.yaml')
"

salt2 fit --config salt/core/configs/gn2v2-dummy.yaml \
  --data.train_file /tmp/v2/train.h5 \
  --data.val_file   /tmp/v2/train.h5 \
  --model.modules.norm.init_args.norm_dict /tmp/v2/norm_dict.yaml \
  --trainer.default_root_dir /tmp/v2/run
```

Recommended extra: a **schema artifact** (`salt2 schema dump /tmp/v2/train.h5
-o /tmp/v2/schema.yaml`, design §2.6) passed as
`--data.modules.reader.init_args.schema /tmp/v2/schema.yaml`. With it, field
typos fail statically before any data is read, and the configured
`class_names` lists are cross-checked (set AND order) against the file's
label attrs.

**Where outputs land**: until the M6 run-dir layout, `config.yaml`,
`checkpoints/` and the fit plan/graph artifacts are written into the trainer
log dir — **your cwd unless you pass `--trainer.default_root_dir <dir>`**,
which is why the quickstart command above sets it (without it, a fit run
from the repo root drops ~8 files into the checkout). The end-of-fit message
prints the exact paths. A leftover `config.yaml` from a previous run is
overwritten. `salt2 test` artifacts land next to the checkpoint, with the
eval H5 (see below).

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
`losses.**` collection (design §3.3) — and the default `TaskWriter`
(streams=null) persists its predictions in eval automatically (design §8).

## Evaluation: `salt2 test` + prediction writers (design §8)

```bash
salt2 test --config <run_dir>/config.yaml \
  --ckpt_path <run_dir>/checkpoints/....ckpt \
  --data.test_file /tmp/v2/pp_output_test_ttbar.h5
```

v1 ergonomics kept: the saved run `config.yaml` is the recommended single
base — further `--config` override files and CLI flags stack on top of it
exactly as for `fit` (the custom-writer journey below relies on this); one
test file per call; logger off; single device forced. Without `--ckpt_path`
the v1 best-epoch glob picks the lowest `loss=` checkpoint from `ckpts/`
(v1 runs) or `checkpoints/` (v2 runs — `base2.yaml` names files
`epoch=NNN-loss=<val/loss>.ckpt` so the glob matches by construction) next
to the single `--config`. The output is ONE H5 next to the checkpoint:
`{ckpt_dir}/{ckpt_stem}__test_{sample}.h5` (`{sample}` from the v1
test-file stem heuristic + `--data.test_suff`); the test plan/graph
artifacts are written into the same directory.

Prediction writing is modular: the top-level `writers:` block (deep-
mergeable, like `callbacks:`) is assembled into one `WriterCallback`
owning a single ftag `H5Writer` sink. `base2.yaml` ships the v1 layout —
**dict order = per-group column order**:

```yaml
writers:
  output: "{ckpt_dir}/{ckpt_stem}__test_{sample}.h5"
  modules:
    inputs_copy: {class_path: salt.core.writers.InputCopyWriter}  # source columns, source dtypes
    tasks:       {class_path: salt.core.writers.TaskWriter}       # {run_name}_pb/... probs, VertexIndex
    pad_mask:    {class_path: salt.core.writers.PadMaskWriter}    # bool 'mask', True = padded
```

Writers declare their consumed bundle keys (`requires`), making them
first-class TEST-graph sinks: demand-gating keeps exactly the demanded
producers alive, and a produced `preds.*` key consumed by **no** writer is
a hard error (design §4.2) — narrowing `TaskWriter` streams and forgetting
a task fails loudly (naming the producing task's config address and the
narrowed writer's `streams` entry) instead of silently dropping columns.
The same error fires statically from `salt2 graph validate`/`deadcode`.
Deleting all writers is refused on the `salt2 test` path. A train-only aux
task currently needs `--model.modules.<task>=null` on the test invocation
(the design §4.2 per-task `expose: [fit, val]` opt-out is an M5 deferral).

### Add a custom output column (design §8)

Subclass `salt.core.writers.Writer` and add four YAML lines under
`writers.modules`. Complete worked example — one new `jets` column holding
the first track's signed d0, resolved **by name**:

```python
# my_writer.py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from salt.core.graph.spec import TensorSpec
from salt.core.writers import Writer

class FirstTrackD0Writer(Writer):
    DTYPE = np.dtype([("first_track_d0", "f4")])

    def requires(self, ctx):  # static demand — keeps producers alive (§8)
        return {"inputs.tracks": TensorSpec(dtype="float32", kind="data")}

    def columns(self, ctx):   # output schema, declared before any batch
        return {"jets": self.DTYPE}

    def write(self, bundle, rows):
        idx = self.ctx.feature_fields["inputs.tracks"].index("d0")
        vals = bundle.get("inputs.tracks").cpu().numpy()[:, 0, idx : idx + 1]
        return {"jets": u2s(vals.astype("f4"), self.DTYPE)}
```

```yaml
# add_writer.yaml — stack as a second --config on salt2 test
writers:
  modules:
    first_d0: {class_path: my_writer.FirstTrackD0Writer}
```

Notes for writer authors:

- **Importability**: `class_path` is resolved with a normal import, so the
  module must be on `sys.path` — run from the directory holding
  `my_writer.py` or set `PYTHONPATH` (with apptainer:
  `--env PYTHONPATH=/path/to/dir`).
- **`requires` kinds** for dataset-served keys: `inputs.<stream>` →
  `TensorSpec(dtype="float32", kind="data")`, `masks.<stream>` →
  `TensorSpec(dtype="bool", kind="pad_mask")`, `labels.<stream>.<name>` →
  `TensorSpec(kind="label")`, `meta.rows` → `TensorSpec(shape=(2,),
  dtype="int64", kind="meta", modes=Mode.TEST)`. Model-produced keys
  (`preds.*`) take an unconstrained `TensorSpec(shape=None, dtype=None)`.
- **`ctx.feature_fields`** maps bundle keys to their declared column names
  (e.g. `"inputs.tracks"` → the configured `Features` variable order), so
  feature columns are resolved by name, never by hard-coded index.
- `bundle.get("dotted.key")` reads the executed TEST bundle; per-token
  outputs must be padded to `ctx.seq_lengths[stream]` (see
  `salt.core.writers.modules._pad_to`).

## Static graph tooling (design §4)

All `salt2 graph` subcommands accept BOTH the trainer configs above and the
small M1 toy-graph format, and never touch data. Saved run `config.yaml`
files round-trip directly (`salt2 graph plot -c <run_dir>/config.yaml ...`
— the run-surface `ckpt_path` key is accepted and ignored). The parsed
`writers:` block enters the static TEST graph exactly as at runtime, so
`validate`/`deadcode` fire the dead-preds error for a narrowed writer set
before anything runs:

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
has a schema artifact, and reports module **preflights** (e.g. a missing
norm dict) as warnings — promotable with `--strict`; an actual
`salt2 fit` fails hard on the same check at fit start (fresh fits only —
resumes never re-read the norm/class dicts), with one `ConfigError`
covering every module before any value is materialised. Unconsumed
`preds.*` in FIT/VAL are **info**-level (the normal no-metric-callback
case, design §3.3) and never promoted, so `--strict` passes on a standard
tagger config.

`salt2 graph plot` renders with **matplotlib** (layered topological DAG,
namespace-coloured modules — the salt container has no graphviz binary) and
always writes the Graphviz `.dot` alongside for manual re-rendering.

### Run-dir artifacts (design §4.4)

Every `salt2 fit`/`salt2 test` writes, at stage start (rank zero):
`plan_fit.txt`+`plan_val.txt` / `plan_test.txt` (dataset + model plan
tables, with the narrowed label list, the demand-narrowed per-group read
columns, and — test — the writer-sinks table: writer instance → consumed
keys), `resolved_io.yaml` (machine-readable: per mode, the plan sources and
every module's flattened requires/produces with resolved specs) and
`graph_<stage>.{dot,svg}` (+ `graph_<stage>_dataset.{dot,svg}`). Fit
artifacts go into the trainer log dir; test artifacts go NEXT TO THE
CHECKPOINT, with the eval H5. Default-on via the `artifacts:` entry in
`base2.yaml` (`salt.core.callbacks.GraphArtifacts`); delete with
`--callbacks.artifacts=null`, retarget with
`--callbacks.artifacts.init_args.output_dir=...`. (Writer nodes in the
`graph plot` rendering itself are an M5 item — the plan-table writer-sinks
section is the current source of that answer.)

## Metrics callbacks

`salt.core.callbacks.ConfusionMatrix` is the v1 `ConfusionMatrixCallback`
port: it reads the step bundle (`preds.<stream>.<task>` argmax vs
`labels.<stream>.<label>`), resolves stream/label/class names from the named
task module, and logs to Comet at each validation epoch end (values are also
stashed on the callback: `last_matrix`, `last_truth_labels`,
`last_pred_labels`). Configure it like any callback:

```yaml
callbacks:
  confusion_matrix:
    class_path: salt.core.callbacks.ConfusionMatrix
    init_args:
      task_name: jets_classification
```

M5 deferral note (design §3.1/§3.4): callback-declared `requires` are NOT
yet FIT/VAL plan sinks — `ConfusionMatrix` works because tasks publish
`preds.*` in all modes and stay alive via their losses (its VAL labels are
demanded by the task itself). A callback demanding a key no task keeps
alive would currently find its producer demand-pruned; the sink wiring
mirrors the TEST writer-demand mechanism and lands in M5 with the
MaskFormer metrics (see `SaltModule._model_sinks`).

## Notable M2 surface notes

- `lrs_config:` is the design §5.1 `lrs:` block under its v1 name (rename is
  an M3 cleanup). Schema: `{initial, max, end, pct_start[, weight_decay,
  last_epoch]}` driving AdamW/lion/HybridMuonAdamW + OneCycleLR.
- `VertexingTaskModule.origin_weighting` takes integer origin ids in M2
  (defaults reproduce v1's `heavy: [3,4,5], fake: [1]`); the design's
  name-based form lands in M3.
- Loggers (Comet) and run dirs are M6; losses show on the stock progress
  bar meanwhile (`train/loss`, `train/<task>_loss`).
- `TaskWriter` writes the vertexing column as bare `VertexIndex` (i8) by
  default — the v1 byte-schema; the design §8 run-name prefix is opt-in via
  `prefix_vertex_column: true` (default polarity to be revisited when v1
  byte-parity gating retires — study CLAUDE.md TODO).
- Per-task `expose: [fit, val]` (design §4.2) is not implemented yet (M5);
  a train-only aux task needs `--model.modules.<task>=null` on each
  `salt2 test` invocation.

## Checkpoints and resume (design §2.3)

Checkpoints carry the resolved schema + per-mode plan hashes under the
top-level `salt_core` key. On resume, a FIT plan-hash mismatch is fatal
(the graph or dataset boundary changed); `Normaliser`/class-weight values
come from the state dict — the norm/class dicts are NOT re-read.
Data-less loading: `SaltModule.load_from_checkpoint(path, modules=...)`.

## Parity and gate harnesses (design §9.5)

Three standalone v1-vs-v2 harnesses, each `python -m` runnable, each writing
a `<gate>_report.json` + stdout table and exiting non-zero on failure:

- `python -m salt.core.parity_gn2 --outdir ...` — bitwise forward parity on
  a weight-shared GN2 (M1/M2 baseline; must stay exit 0).
- `python -m salt.core.gates_m2 {g1..g5} ...` — dataset parity, fit smoke,
  loss-curve parity, throughput, resume round-trip (plan 05).
- `python -m salt.core.gates_m3 {w1..w5} ...` — the writer/eval milestone
  (plan 06): `w1` v1-PredictionWriter vs `salt2 test` output-file parity on
  a dummy file (byte-schema + values; default batch 96 deliberately
  exercises the partial final batch; `w2` the same on a real file via
  `--file/--norm-dict`), `w3` negative controls (TEST dead-preds hard error
  + a corrupted comparison must fail), `w4` the custom-writer four-line
  override journey, `w5` ConfusionMatrix v1/v2 value parity (with a
  prediction-diversity non-degeneracy bar). Dummy-file gates generate their
  data into `--outdir` when no `--file` is given.
