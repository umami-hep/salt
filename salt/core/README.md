# salt v2 (`salt.core`) — the `salt2` surface

The modular salt v2 stack: graph-of-modules models, a demand-driven dataset
pipeline, and a jsonargparse YAML surface. Everything here is config-first:
modules declare their inputs/outputs (`declare_io`), plans are compiled
statically, and no data file is touched before the run starts (design §2.3).

> Design doc references (`§…`) point at the v2 design document
> (`plans/02_design-doc.md` in the modularise-salt study).

## Parity-closure doctrine (v1 vs v2 comparisons)

The v1 stack (`salt.models`, `salt.data`, `salt.utils`, `salt.callbacks`,
`salt.onnx`, `salt.main`, `salt.modelwrapper`) is being deleted from `main`.
All v1↔v2 numerical parity was established and passed at the frozen commit
**`29c67a1`** (`29c67a186f01`) — the last commit where both stacks coexist and
the parity gates (`parity_gn2`, the v1-vs-v2 fold/state-dict/ONNX tests) run
green.

**Doctrine (user decision):** comparisons against v1 (or a pinned upstream)
are done by `git checkout <pin>` — NO frozen comparison artifacts (goldens,
specimens, vendored snapshots) live in the tree. For v1, everything needed
lives at `29c67a1` and passed there; the frozen pin is the single source of
truth for the v1 reference. Git history is the archive.

### Pins of record

| Comparison | Pin | Where |
|---|---|---|
| v1 ↔ v2 numerical parity | `29c67a1` (`29c67a186f01`) | this repo — `git checkout 29c67a1` |
| v2 MaskFormer ↔ upstream | `6570e85` | upstream salt — see checkout below |

The upstream MaskFormer equivalence (established during the MFU wave) closes
at the upstream pin. To reproduce the comparison:

```bash
git remote add upstream ssh://gitlab.cern.ch:7999/aft/algorithms/salt.git  # if absent
git fetch upstream
git checkout 6570e85   # the validated upstream MaskFormer reference
```

### Closure evidence (plan 47 sweep, 2026-07-13)

The remaining in-tree frozen comparison artifacts were retired at this commit.
In each case the frozen artifact IS the reference output, so the final green
run at deletion time IS the final parity check — none was regenerated from the
pin (redundant by construction):

- **`salt/tests/_fixtures/gn2v2_dummy_oracle/`** (frozen `WriterCallback` eval
  H5 + ckpt): the byte-parity tests consuming it were green at `b8d81bd`
  (pipeline `#15271499`) and `fb90a7c` (pipeline `#15272018`). Retired test
  node ids:
  `salt/tests/integration/test_outputs_h5_parity.py::TestH5OutputWriterParity::{test_deferred_columns_present_in_oracle,test_groups_match,test_semantic_h5_parity}`
  and `::TestCutoverCliE2E::{test_cli_groups_match_oracle,test_cli_semantic_h5_parity}`.
  The file keeps the live single-leg CLI e2e checks (softmax-once, column
  presence) re-anchored to a live 1-epoch fit.
- **`salt/tests/_fixtures/upstream_mf_snapshot/`** +
  **`salt/tests/_fixtures/mf_writer_parity/upstream_6570e85_schema.json`**
  (vendored upstream MaskFormer modules + writer schema): the snapshot modules
  had zero consumers left; the schema-parity assertions in
  `salt/tests/unit/outputs/test_maskformer_fold_w6b.py::TestSchemaParityVsFixture`
  (green in the same pipelines) were re-anchored to first-principles literals
  and the upstream cross-checks retired. Closure at upstream pin `6570e85`.
  Also retired: the `/tmp`-golden ONNX contract tests
  (`test_onnx_fold_w2.py::test_folded_gn2v2_export_contract_matches_oracle`
  re-anchored to pinned literals;
  `test_onnx_fold_w3.py::test_maskformer_folded_contract_matches_oracle`
  dropped — its literal twin `test_maskformer_folded_export_contract` stays).
- **`map_v1_state_dict`** (v1→v2 checkpoint weight mapper,
  `salt/core/nn/state_dict.py`) + `salt/tests/_fixtures/v1_gn2_state_dict.json`
  + `salt/tests/unit/nn/test_state_dict.py::TestStateDictMapping::*`:
  v1-checkpoint loading is deliberately dropped — recoverable from history
  (`fb90a7c`) or usable at the pin `29c67a1`. `SaltModule.on_load_checkpoint`
  now detects the v1 (`ModelWrapper`) state-dict layout (`model.pool_net.*`
  keys) and raises an explicit `ConfigError` instead of a missing-keys cascade.

**Regeneration recipe** (only if the frozen-oracle check is ever wanted again):
`generate_oracle.py` was a throwaway script never committed — reconstruct it
from `provenance.json` and the retired oracle test, both in git history at
`93a29ed^` (`salt/tests/_fixtures/gn2v2_dummy_oracle/provenance.json`,
`salt/tests/integration/test_outputs_h5_parity.py`). Parameters recorded there:
commit `a9e2ac2`, salt-py314 container, `salt/core/configs/gn2v2-dummy.yaml`;
synthetic data from `write_dummy_file` (1000 jets × 40 tracks, module-level
`np.random.default_rng(42)`); training `max_epochs=1`, `limit_train_batches=2`,
`limit_val_batches=2`, `batch_size=100`, `seed_everything=42`; `N_TEST=300`.

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
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.core.testing.inputs import write_dummy_file
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
(streams=null, onnx=true) persists its predictions in eval AND declares
its ONNX output (`{model_name}_TrackType`, argmax int8) automatically —
zero extra config, the M4.5 unified-manifest journey. If the aux head is
a training-time regulariser that must NOT reach eval OR Athena, set
`expose: [fit, val]` on the task (design §4.2): its prediction is gated out
of the TEST/ONNX plans (the task is pruned there) while it keeps training.
To keep it in eval but out of Athena only, narrow the export surface instead
(`onnx_tasks:`/`onnx: false` on the TaskWriter) and check with
`salt2 export --manifest`.

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
to the single `--config` — when stacking a second `--config`, pass
`--ckpt_path` explicitly (the glob needs exactly one config to anchor on). The output is ONE H5 next to the checkpoint:
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
task opts out of eval with the per-task `expose: [fit, val]` config (design
§4.2): it stays trained (the loss is FIT/VAL anyway) while its `preds.*`
port is gated out of the TEST/ONNX plans, so the planner prunes the task and
the dead-preds error never fires. `--model.modules.<task>=null` (delete the
task entirely) remains the heavier alternative.

### Writers are the SINGLE output manifest (M4.5 unified manifest)

Since the M4.5 amendment, the same writer declarations drive BOTH output
modes — **TEST executes, ONNX declares**:

- **TEST**: exactly as above — `requires`/`columns`/`write` run per batch.
- **ONNX**: the writer is *read, never run* — `Writer.onnx_outputs(ctx)`
  returns its export-manifest entries (M4 `ExportOutput` objects:
  port + logical suffix(es) + registered reduce), and `salt2 export`
  assembles `export.outputs` from them. There is no `export.outputs`
  config section anymore (declaring one is a hard error pointing here).

Naming policy (amendment §5): writers declare logical **suffixes**; the
TEST column is `{run_name}_{suffix}` and the ONNX output is
`{export.model_name}_{suffix}` — one declaration, two prefixes. For
classification the suffix list is literally the `class_names`-derived list
both modes share, so reordering classes moves eval columns AND Athena
outputs together (the v1 eval-vs-ONNX vertex-naming drift class is
unrepresentable). Cross-mode suffix constants live in
`salt.core.outputs.names` (`VERTEX_INDEX` shared; the MaskFormer
`OBJECT_INDEX` MaskIndex/HadronIndex pair is a pinned, documented v1
divergence M5 must import).

Per-writer ONNX participation (`TaskWriter`):

```yaml
writers:
  modules:
    tasks:
      class_path: salt.core.writers.TaskWriter
      init_args:
        onnx: true                       # default — eval-only with false
        onnx_streams: [jets, tracks]     # stream-grained narrowing
        onnx_tasks: [jets_classification, track_vertexing]  # task-grained
        onnx_names: {track_origin: TrackLabel,              # str: aux rename
                     jets_classification: [pb, pcharm, plight]}  # list: per-class
```

`InputCopyWriter`/`PadMaskWriter` are **eval-only** by design (Athena
feeds the inputs; a pad-mask output has no consumer). The **export-only**
direction is `salt.core.writers.ExportOnlyWriter`: subclass it and
override `onnx_outputs` only — that explicit base class (the
`export_only` flag) is THE blessed way to declare an Athena output with
no eval analogue; a hand-rolled empty-columns writer with export entries
is rejected, as is a writer with no role in either mode. **Export math is
reduces only**: a custom writer's `write()` numpy never traces and never
runs inside Athena, and every manifest entry names a reduce from the
SHIPPED registry — `split_scalars`, `argmax`, `vertex_union_find`
(`salt.core.onnx.config.KNOWN_REDUCES`, implemented in
`salt.core.onnx.reduces`). A public registration surface for custom
reduces is an M5 deliverable; until it lands, export-only writers compose
the shipped reduces only.

The assembled ONNX manifest is a FLAT namespace: two writers declaring
one suffix is a hard error naming both (fix via `onnx_names:` or
`export.rename:`). Inspect everything with `salt2 export --manifest`,
`salt2 graph resolve [--annotate]`, or the manifest table appended to the
export-time `plan_onnx.txt`.

#### M4.5 amendment addendum — TaskWriter regression family (AM dec 4, settled M5)

`WriterCallback._validate_writer_roles` derives every writer's ONNX
manifest on the TEST demand-assembly path too (`per_writer_demand` calls it
during `salt2 test`), so a task family with a TEST representation but NO
export representation would trip `TaskWriter.onnx_outputs`'s
unsupported-family `ConfigError` during eval, not just at export. AM
decision 4 left the M5 implementer two options when `TaskWriter` gains
**regression**: keep the loud eval-time coupling, or derive only
export-representable families (skip + deadcode finding).

**Decision (M5 sub-wave A foundation): derive a regression export
representation — KEEP the loud error as the correct guard for genuinely
unrepresentable families, and make scalar regression a supported family in
BOTH modes so it never trips that error.** Rationale, grounded in AM dec 4
+ the AM pre-implementation check (§"Pre-implementation check"): global
scalar regression is *already export-representable in v1* — `RegressionTask`
has `output_names` (one `{model_name}_{target}` column per target,
`task.py:512-518`) and a `get_onnx` (one squeezed scalar per target via
`torch.split`, `task.py:625-642`); the AM check explicitly flags "a plain
global regression export (GN2X-class, one line in v1)" as the family that
must NOT become a custom-writer authoring task. So "derive-only-export-
representable" and "keep the loud error" are not in tension *for
regression*: regression simply joins classification + vertexing as a
representable family via its own `output_names`/`get_h5` (TEST) and
`onnx_outputs` (ONNX, a `split_scalars`-style per-target manifest entry) on
`RegressionTaskModule` (`salt/core/nn/tasks.py`; the per-family rendering
lives on the TASK now, mirroring v1's `task.py` placement — `TaskWriter` is
pure orchestration), built from ONE shared per-family suffix helper exactly
as the amendment §2.2 single-ownership rule requires. The loud
unsupported-family raises (`salt/core/nn/tasks.py`
`_TaskModuleBase.output_names`/`get_h5`/`onnx_outputs`) stay — they remain
the right error for a future family with no export math — but a regression
config no longer reaches them.

Consequence for A2: when `RegressionTaskModule` lands, the default
`TaskWriter` formats it in TEST and ONNX, so `salt2 test` demand assembly
does NOT crash on a regression config — the `_validate_writer_roles` check
passes (the writer has a non-empty TEST demand AND a non-empty manifest).
A regression task an author chooses NOT to export is still expressible
(`onnx: false` / `onnx_tasks`): `onnx_outputs` returns `[]` before the
family dispatch, so the eval-only shape (non-empty demand, empty manifest)
is legal and never trips the raise. The narrow "TEST-representable but
intentionally never export-representable" case is therefore served by
explicit ONNX narrowing, not by silently skipping families.

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

    def columns(self, ctx):  # output schema, declared before any batch
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

When stacking a second `--config` on `salt2 test`, pass `--ckpt_path`
explicitly: the best-checkpoint glob needs exactly ONE `--config` (it
looks next to the saved run config), so the no-`--ckpt_path` shortcut and
config stacking are mutually exclusive.

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
— the run-surface `ckpt_path` key is accepted and ignored). `-c` is
**repeatable** for trainer configs with the fit/export deep-merge semantics
— the base + override pattern (e.g. the add-an-aux-task journey above) is
statically inspectable without a prior fit; with `--annotate` the comment
block goes into the LAST `-c` file. The parsed
`writers:` block enters the static TEST graph exactly as at runtime, so
`validate`/`deadcode` fire the dead-preds error for a narrowed writer set
before anything runs. `validate` also runs the writer kind/dtype unification
for TEST (design §2.7/§8 — the same `WriterCallback.validate_specs` check
`salt2 test` setup runs): a writer declaring a require with a kind/dtype that
contradicts its producing leaf (e.g. `preds.jets.classification` as
`kind=label` where the task publishes `data`) is a hard error here, data-free,
not only at `salt2 test` setup:

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

**ONNX mode checks the unified manifest** (design §3.1/§4.1; M4.5): the
static ONNX sinks are the union of the writers' declared `onnx_outputs`
ports — exactly what `salt2 export` will trace — with errors attributed to
the declaring writer (`writers.modules.<name>`). The export-only half of
the `export:` block is validated as `salt2 export` does: an invalid
`export.model_name` (`_`/`-`) is an error-level `validate` finding
(`plan`/`plot`/`why --mode onnx` raise it), `rename:`/`combine:` are
checked against the assembled manifest, and a legacy config still carrying
`export.outputs` fails with the M4.5 migration error. A trainer config
without an `export:` block keeps the writer-derived sinks but WARNS that
inputs/model_name were unchecked (promoted under `--strict` — the §9.3
converter CI gate expects converted configs to carry the block).
Predictions narrowed out of the manifest (`onnx_streams`/`onnx_tasks`)
show up as info-level ONNX deadcode findings (the §4.2 export-pruning
story). Note the static ONNX plan/plot remain the **dataset-fed
approximation** (reader/features included, no reduces); the authoritative
rendering of the traced graph is the `plan_onnx.txt` that `salt2 export`
writes next to `network.onnx` — the two plan hashes legitimately differ,
and the CLI prints this caveat on `plan`/`plot --mode onnx`.

**`salt2 graph resolve -c <cfg> [--annotate]`** prints the writer-derived
output manifest — eval columns per writer/stream AND the full ONNX output
list (names, dtypes, reduces/combines) — and with `--annotate` writes it
into the config as a refreshable comment block (the §4.4 labels precedent
extended to the Athena surface): a reader of the YAML always sees what
eval writes and what Athena gets, without running anything.

`salt2 graph plot` writes the §4.3 Graphviz `.dot` (port-card signature
layout, namespace-coloured modules) and rasterises it to a PNG + sibling PDF
by shelling out to the **`dot`** binary baked into the salt container. There
is no matplotlib path; a missing `dot` is a clear actionable error (the `.dot`
is still written for manual re-rendering).

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

- `lrs:` is the design §5.1 OneCycleLR block (renamed from the v1 ModelWrapper
  `lrs_config:` kwarg — M3 cleanup, landed in M5 sub-wave D). Schema:
  `{initial, max, end, pct_start[, weight_decay, last_epoch]}` driving
  AdamW/lion/HybridMuonAdamW + OneCycleLR.
- `VertexingTaskModule.origin_weighting` takes integer origin ids OR class
  NAMES (defaults reproduce v1's `heavy: [3,4,5], fake: [1]`); names (the
  design §5.1 / GN3 origin surface) are resolved to ids at fit/test setup
  against the origin label's class-name attr in the schema artifact (M5
  sub-wave D). A name-based config without a schema artifact is a loud bind
  error.
- Loggers (Comet) and run dirs are M6; losses show on the stock progress
  bar meanwhile (`train/loss`, `train/<task>_loss`).
- `TaskWriter` writes the vertexing column as bare `VertexIndex` (i8) by
  default — the v1 byte-schema; the design §8 run-name prefix is opt-in via
  `prefix_vertex_column: true` (default polarity to be revisited when v1
  byte-parity gating retires — study CLAUDE.md TODO).
- Per-task `expose: [fit, val]` (design §4.2, M5 sub-wave D): a train-only
  aux task gates its `preds.*` port to the listed modes — `[fit, val]` prunes
  it from the TEST/ONNX plans (silencing the dead-preds error) while it keeps
  training. `--model.modules.<task>=null` (full deletion) is the alternative.

## ONNX export: `salt2 export` (design §7)

```bash
salt2 export --ckpt_path <run_dir>/checkpoints/epoch=...-loss=....ckpt
# config inferred at <run_dir>/config.yaml (pass -c to override); output
# defaults to <run_dir>/network.onnx (--output / -o/--overwrite to control)
```

The design §7 `--run-dir` spelling lands with the M6 run-dir layout; until
then `--ckpt_path` + the inferred sibling config reproduces the v1 contract
(`to_onnx.py:629-631`). `-c` is **repeatable** with the fit deep-merge
semantics — the supported way to export a run trained before the export
block existed (every v1 migrator until the M7 converter):

```bash
salt2 export --ckpt_path <ckpt> -c <run_dir>/config.yaml -c my_export_block.yaml
# my_export_block.yaml carries ONLY the export: block below
```

The export contract has two halves (M4.5 unified manifest, "one manifest
and a half"):

1. **The outputs come from the writers** (the same declarations that name
   the eval columns — see the writers section above). There is NO
   `export.outputs` section: a config declaring one fails with the M4.5
   migration error. Inspect the assembled manifest any time with
   `salt2 export --manifest -c <config>` (no checkpoint needed) or
   `salt2 graph resolve [--annotate]`.
2. **The export-only half** lives in the **`export:` block** (top-level,
   shipped in the gn2v2 configs; parsed by the normal salt2 surface so it
   round-trips through saved run configs):

```yaml
export:
  model_name: GN2v2 # no '_'/'-' — validated ONLY at export time; the run
  #                   name stays unrestricted (default: run name stripped)
  inputs:
    - { port: inputs.jets, name: jet_features } # [1, F] global
    - { port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks } # [L, F]
  # optional Athena-presentation post-processing of the writer manifest
  # (the v1 --rename/--combine_outputs features — readers of exotic
  # configs consult the writers AND these two keys):
  rename: { pu: plight } # old suffix -> new (existence-checked)
  combine: # new output = sum(scale * existing GLOBAL output)
    - { name: pbc, inputs: { pb: 0.5, pc: 0.5 } }
```

How it works (no data file is touched — config + checkpoint only):

- The **ONNX plan** is compiled with the writer-manifest ports as sinks —
  labels/losses/the writers' own TEST role are demand-pruned
  automatically; sources mirror the `Features` declaration
  (widths/columns from `variables:`).
- The **`OnnxAdapter`** wrapper assembles the bundle in-graph (sequences
  `[L, F]` -> `[1, L, F]` + all-valid pad masks — the v1 traced pattern),
  executes the frozen plan, and emits the flat output tuple. Every module
  recursively receives `set_export_mode()` (the encoder forces torch-math
  attention — the Athena-agreement requirement).
- **Reduces** (`split_scalars`, `argmax`, `vertex_union_find`) generate the
  output names/dtypes/dynamic axes from the manifest entries; union-find
  runs INSIDE the traced graph on the raw edge scores the vertexing task
  publishes in ONNX mode (design §3.3 per-family exception). MaskFormer
  reduces are M5.
- **`rename:`/`combine:`** post-process the manifest with v1 semantics:
  renames apply first (existence-checked), combined outputs are linear
  combinations of the (renamed) GLOBAL float outputs computed inside the
  traced graph, and they insert after the global entries but BEFORE the
  per-token aux entries — the v1 output order (`to_onnx.py:258-292`,
  `combine_insertion_index`). Both are recorded in `gnn_config`
  byte-compatibly with v1 (`combine_outputs`/`rename_outputs`).
- **Multi-stream trace-safety** (design risk 7, adjudicated): eager `Split`
  slicing is wrong under tracing with ≥2 dynamic sequence axes; in ONNX
  mode `Concat` publishes a `seq.offsets` boundary tensor and `Split`
  slices via `index_select` — proven on a two-axis (tracks, electrons)
  grid including zero-length streams (see the `Split` docstring).
- **`gnn_config` metadata** is bit-compatible with v1 on equivalent config
  (key set/order, `jet_var`/`*_sd0sort` input names, `_btagJes` strip on
  global variables, placeholders); the single additive key `plan_hash` is
  appended after the v1 set. `onnx_model_version` stays `v1`.
- The **checker** runs by default after export (`--no-check` to skip):
  eager-v2-torch vs onnxruntime over the v1 sweep (L=0..39 x `--trials`,
  including L=0), outputs addressed by name; v1 bars (float 1e-4 + no-NaN
  + no-exact-zero; int8 exact). `--float-atol 1e-6` for the gate bar.
- Aliases: `{port: inputs.global, alias: inputs.jets}` binds a port from
  another input's tensor (the GN3 global stream; clone when the `Features`
  declarations match, name-resolved `index_select` gather otherwise);
  exercised on a real GN3 model in M5.
- **Artifacts**: alongside `network.onnx` the exporter writes
  `plan_onnx.txt` — the §4.4 plan table of the graph Athena will actually
  run (union-find placement, `Split` `index_select` mechanism) PLUS the
  writer-derived output-manifest table. This is the authoritative ONNX
  plan; `salt2 graph plan --mode onnx` shows the dataset-fed static
  approximation.
- **Expected console output**: a clean export prints NO trace warnings.
  The torch `aten::index ... indices of type Byte` UserWarning (raised on
  the v1-verbatim bool-mask gathers in the union-find/zero-token tricks,
  identically noisy under v1's exporter) is deliberately suppressed around
  `torch.onnx.export` — the default-on sweep checker (incl. L=0) is the
  proof the traced graph is correct.

Programmatic surface for gates/tests (no checkpoint needed):
`salt.core.onnx.export_graph(modules, export_cfg, variables, path,
outputs=<manifest>)` (`outputs` = the writer-derived `ExportOutput` list,
e.g. `WriterCallback.onnx_manifest(...)` or `gates_m4.writer_manifest`) +
`salt.core.onnx.check_onnx(adapter, path, ...)`.

## Checkpoints and resume (design §2.3)

Checkpoints carry the resolved schema + per-mode plan hashes under the
top-level `salt_core` key. On resume, a FIT plan-hash mismatch is fatal
(the graph or dataset boundary changed); `Normaliser`/class-weight values
come from the state dict — the norm/class dicts are NOT re-read.
Data-less loading: `SaltModule.load_from_checkpoint(path, modules=...)`.

## Parity and gate harnesses (design §9.5)

Four standalone v1-vs-v2 harnesses, each `python -m` runnable, each writing
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
- `python -m salt.core.gates_m4 {o1..o5} --outdir ...` — the ONNX-export
  milestone (plan 07; NO data needed — fixtures are built in `--outdir`):
  `o1` v2-torch vs v2-ONNX over the full L=0..39 sweep incl. L=0 at the
  1e-6 bar (v1 ships 1e-4), `o2` v1-ONNX vs v2-ONNX output identity on the
  same weights (bitwise on the GN2 fixture) + IO/dynamic-axes/`gnn_config`
  contract equality, `o3` two independent dynamic sequence axes over a
  (L_trk, L_el) grid incl. zeros — the design risk-7 blocker, with a v1
  eager cross-check and the `Concat seq.offsets -> Split index_select`
  mechanism recorded from the compiled plan, `o4` VertexIndex triple
  identity (v1-ONNX == v2-ONNX == the v1 torch union-find chain,
  int8-exact, multi-vertex non-degeneracy bar), `o5` negative controls
  (corrupted weights fail the checker; `model_name` rule) + `gnn_config`
  byte-comparison vs v1 (13-key ordered prefix + additive trailing
  `plan_hash`). The untrained fixture heads are mean-centered before the
  weight transfer (`_decollapse_v1_heads`) so the int8 comparisons have
  discriminating power.
