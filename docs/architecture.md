# salt v2 — the `salt` surface

The modular salt v2 stack: graph-of-modules models, a demand-driven dataset
pipeline, and a jsonargparse YAML surface. Everything here is config-first:
modules declare their inputs/outputs (`declare_io`), plans are compiled
statically, and no data file is touched before the run starts.

## Parity-closure doctrine (v1 vs v2 comparisons)

The v1 stack (the `salt.models`, `salt.data`, `salt.utils`, `salt.callbacks`,
`salt.onnx`, `salt.main` and `salt.modelwrapper` of pin `29c67a1`) is being
deleted from `main`.
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

### Closure evidence

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
  `salt/model/state_dict.py`) + `salt/tests/_fixtures/v1_gn2_state_dict.json`
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
commit `a9e2ac2`, salt-py314 container, `salt/configs/gn2v2-opendata.yaml`;
synthetic data from `write_dummy_file` (1000 jets × 40 tracks, module-level
`np.random.default_rng(42)`); training `max_epochs=1`, `limit_train_batches=2`,
`limit_val_batches=2`, `batch_size=100`, `seed_everything=42`; `N_TEST=300`.

## Quickstart: train GN2v2 on a dummy file

One-time setup: if the container's installed salt predates the modular `salt` package,
`python -m salt.main` fails with `ModuleNotFoundError: No module named
'salt'` from any cwd except the repo root, and there is no `salt`
script on PATH. `pip install -e .` (from the repo root, inside the
container) fixes both — afterwards `salt` works from any directory.

```bash
# generate a dummy training file + norm dict (copy-paste):
python -c "
from pathlib import Path
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.testing.inputs import write_dummy_file
Path('/tmp/v2').mkdir(parents=True, exist_ok=True)
write_parity_norm_dict('/tmp/v2/norm_dict.yaml', '/tmp/v2/class_dict.yaml')
write_dummy_file('/tmp/v2/train.h5', '/tmp/v2/norm_dict.yaml')
"

salt fit --config salt/configs/gn2v2-opendata.yaml \
  --data.train_file /tmp/v2/train.h5 \
  --data.val_file   /tmp/v2/train.h5 \
  --model.modules.norm.init_args.norm_dict /tmp/v2/norm_dict.yaml \
  --trainer.default_root_dir /tmp/v2/run
```

Recommended extra: a **schema artifact** (`salt schema dump /tmp/v2/train.h5
-o /tmp/v2/schema.yaml`) passed as
`--data.modules.reader.init_args.schema /tmp/v2/schema.yaml`. With it, field
typos fail statically before any data is read, and the configured
`class_names` lists are cross-checked (set AND order) against the file's
label attrs.

**Where outputs land**: `config.yaml`, `checkpoints/` and the fit plan/graph
artifacts are written into the trainer log dir — **your cwd unless you pass
`--trainer.default_root_dir <dir>`**, which is why the quickstart command
above sets it (without it, a fit run from the repo root drops ~8 files into
the checkout). The end-of-fit message prints the exact paths. A leftover
`config.yaml` from a previous run is overwritten. `salt test` artifacts land
next to the checkpoint, with the eval H5 (see below).

## Config model

- `base.yaml` is auto-loaded; your config stacks on top.
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
          class_names: [a, b, c]
          dense: {hidden_layers: [16], activation: ReLU}
```

All existing modules survive the merge; the new task's label is demanded
from the dataset automatically and its loss joins `loss.total` via the
`losses.**` collection. To persist its predictions in eval and declare its
ONNX output, add the task name to a `RunTaskOutput`
`tasks:` list in the top-level `outputs:` section (one line — see the
evaluation section below); the per-mode rendering (prob columns in TEST,
`{model_name}_TrackType` argmax int8 in ONNX) comes from the task's own
`get_output`. If the aux head is a training-time regulariser that must NOT
reach eval OR Athena, set `expose: [fit, val]` on the task:
its prediction is gated out of the TEST/ONNX plans (the task is pruned
there) while it keeps training. To keep it in eval but out of Athena only,
list it in a `RunTaskOutput` with `modes: [test]` and check with
`salt export --manifest`.

## Evaluation: `salt test` + the `outputs:` section

```bash
salt test --config <run_dir>/config.yaml \
  --ckpt_path <run_dir>/checkpoints/....ckpt \
  --data.test_file /tmp/v2/pp_output_test_ttbar.h5
```

v1 ergonomics kept: the saved run `config.yaml` is the recommended single
base — further `--config` override files and CLI flags stack on top of it
exactly as for `fit` (the custom-output journey below relies on this); one
test file per call; logger off; single device forced. Without `--ckpt_path`
the v1 best-epoch glob picks the lowest `loss=` checkpoint from `ckpts/`
(v1 runs) or `checkpoints/` (v2 runs — `base.yaml` names files
`epoch=NNN-loss=<val/loss>.ckpt` so the glob matches by construction) next
to the single `--config` — when stacking a second `--config`, pass
`--ckpt_path` explicitly (the glob needs exactly one config to anchor on). The output is ONE H5 next to the checkpoint:
`{ckpt_dir}/{ckpt_stem}__test_{sample}.h5` (`{sample}` from the v1
test-file stem heuristic + `--data.test_suff`); the test plan/graph
artifacts are written into the same directory.

Prediction writing is declared in the top-level `outputs:` section, the
single deep-mergeable home for everything that leaves the model. It
carries two kinds of entry, partitioned by type at parse time:

- **writers** (`salt.outputs.OutputSectionWriter` graph modules) — an
  ORDERED dict saying WHAT is written and in which modes;
- **sinks** (`salt.outputs.Node` subclasses) — the destinations, held
  aside on the model rather than folded into the graph, and therefore
  EXCLUDED from the ordering.

Usually no sink is declared at all: the COMMAND wires the matching
implicit one over the section — `salt test` instantiates the H5 sink
(`salt.outputs.H5OutputSink`), and the ONNX parse folds an
`OnnxExportSink` naming the export-mode leaves. A config declares a sink
only when it carries a manifest the command cannot guess (`MaskFormer.yaml`)
or when it is a third-party one; the implicit wiring then leaves it alone.
A sink declared under `callbacks:` is accepted for one deprecation window.

Every model config defines its own section (`base.yaml` ships none) —
**writer dict order = per-group column order**, the v1 layout being:

```yaml
outputs:
  inputs_copy:
    class_path: salt.outputs.InputCopyWriter  # source columns, source dtypes
    init_args: {streams: [jets, tracks]}
  run_tasks:
    class_path: salt.outputs.RunTaskOutput    # {run_name}_pb/... probs, VertexIndex
    init_args: {tasks: [jets_classification, track_origin, track_vertexing]}
  pad_mask:
    class_path: salt.outputs.PadMaskWriter    # bool 'mask', True = padded
    init_args: {streams: [tracks]}
```

Section writers are first-class graph participants: `RunTaskOutput`
requires each listed task's raw `preds.*` leaf (plus the task's
`output_time_requires` — at minimum the stream pad mask), calls
`task.get_output(bundle, mode, run_name)` and writes one
`outputs.<stream>.<task>.<col>` leaf per returned `OutputField`; the
sinks consume ONLY `outputs.*` leaves. Demand-gating keeps exactly the
demanded producers alive, and a produced `preds.*` key consumed by **no**
sink is a hard error — narrowing a `RunTaskOutput` `tasks:`
list and forgetting a task fails loudly instead of silently dropping
columns. The same error fires statically from
`salt graph validate`/`deadcode`. A `salt test` config without an
`outputs:` section is refused, and a config still carrying the retired
top-level `writers:` block fails with a clean migration `ConfigError`. A
train-only aux task opts out of eval with the per-task
`expose: [fit, val]` config: it stays trained (the loss is
FIT/VAL anyway) while its `preds.*` port is gated out of the TEST/ONNX
plans, so the planner prunes the task and the dead-preds error never
fires. `--model.modules.<task>=null` (delete the task entirely) remains
the heavier alternative.

Per-writer mode participation is the `modes:` list: each
section writer declares the modes it runs in — `test` (eval H5) and/or
`export` (ONNX); omitted = both. A `modes: [test]` writer mints no ONNX
leaves; an export-only writer contributes no eval columns.
`gn2v2-opendata.yaml` splits its tasks across two writers to keep
`track_origin` H5-only:

```yaml
outputs:
  jets_out:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [jets_classification]}          # test + export (both)
  origin_out:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [track_origin], modes: [test]}  # eval H5 only
```

`InputCopyWriter`/`PadMaskWriter` mint no ONNX leaves (Athena feeds the
inputs; a pad-mask output has no Athena consumer) — but their `modes:`
list is NOT inert: it decides whether they add copy/mask columns to the
`salt inference` H5 (which writes the export selection; omitted `modes:`
= both, so they run there by default; `modes: [test]` keeps them
eval-only). The explicit H5 sink config surface is
RETIRED: wiring `H5OutputSink` with an explicit `OutputColumn` table is a
hard error pointing back at the section mechanism. A config MAY still
declare a sink instance explicitly when it needs a capability the
injected default cannot mint from the section: an `H5OutputSink`
carrying `object_groups` (a generic per-object H5 group on a different row axis,
used by MaskFormer to write `[B, n_objects]`-shaped outputs alongside the standard
per-jet columns), or an `OnnxExportSink` declared by name when the command would
not inject one. Its tuple is always collected from the producers' own
`manifest_fields(mode)` — an explicit leaf list is a hard error; `consumes:`
(fnmatch patterns over the leaf key) narrows what a sink takes. The object math itself
(`MaskFormerObjects` reconstructing matched objects from mask logits) lives in a
graph module and writes `outputs.*` leaves like any other; the sink has no MaskFormer
knowledge. `MaskFormer.yaml` shows both patterns in use. The command leaves
declared sinks alone and only injects a sink type that is not already present.

### The `outputs:` section is the SINGLE output manifest

The same section drives BOTH output modes — **TEST executes into the H5
sink, ONNX traces into the export tuple**. The per-mode split lives in
each task's `get_output(bundle, mode, run_name) -> list[OutputField]`
(and its value-free twin `get_output_manifest`, which gives the sinks the
column schema before any batch runs): in TEST it returns the eval columns
(softmaxed probs, de-scaled regression values, the int union-find
`VertexIndex`, plus the per-task `target_{task}` label columns — default
on, `write_targets: false` per task to disable); in ONNX it returns the
Athena outputs (squeezed per-class scalar probs, argmax int8 indices).
All conversions are traceable torch ops running INSIDE the graph — there
is no off-graph numpy step between the model and either sink.

Naming policy: fields declare logical **suffixes**; the TEST column is
`{run_name}_{suffix}` and the ONNX output is `{model_name}_{suffix}` (the
export sink's `model_name` init arg) — one declaration, two prefixes. For
classification the suffix list is
literally the `class_names`-derived list both modes share, so reordering
classes moves eval columns AND Athena outputs together (the v1
eval-vs-ONNX vertex-naming drift class is unrepresentable). Cross-mode
suffix constants live in `salt.outputs.output_schema` (`VERTEX_INDEX`,
shared by TEST and ONNX). The MaskFormer track-to-object index is the
single suffix `HadronIndex` in both the eval-H5 column and the ONNX
output (the eval column, ours to name, was unified onto the Athena name).

The assembled ONNX manifest is a FLAT namespace: two leaves minting one
suffix is a hard error naming both (fix via the sink's `rename:`). Inspect
everything with `salt export --manifest` (no checkpoint needed) or the
manifest table appended to the export-time `plan_onnx.txt`.

Every representable task family owns its export math ON THE TASK:
classification (`ClassProbs`/`SeqClassProbs`/`SeqClassIndex` in
`salt.outputs.task_output`), vertexing (in-graph union-find), and
regression (`RegressionTaskModule.get_output` — de-scaled f4 columns in
TEST, squeezed per-target scalars in ONNX) are all first-class in both
modes. The legacy per-task rendering surface (`get_h5` / `output_names` /
`onnx_outputs`) and the writer module family (`Writer` / `TaskWriter` /
`ExportOnlyWriter` / `WriterCallback` under the old `salt.core.writers`) were
retired; `salt/tests/unit/nn/test_get_output.py`
carries the re-anchored per-family oracles.

### Add a custom output column

Two extension seams, by scope:

- **A new column family for a task** — implement it on the task:
  `get_output(bundle, mode, run_name)` returns `OutputField`s (see
  `salt/outputs/task_output.py` for the shared per-family value
  helpers and `salt/model/modules/tasks/` for the shipped families). This is
  the right seam when the column is a rendering of a task's prediction.
- **A column not owned by any task** — write a custom section writer:
  subclass `salt.outputs.OutputSectionWriter`, declare + produce an
  `outputs.<stream>.<col>` leaf, and expose the manifest surface the dumb
  sinks discover (`is_run_task_output()` returning True, plus
  `manifest_fields(mode)` returning `(leaf_key, OutputField)` pairs — the
  value-free column schema the H5 sink reads before any batch runs).

Worked example — one new `jets` column counting each jet's valid tracks:

```python
# my_output.py
import torch
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.outputs import OutputField, OutputSectionWriter

_LEAF = "outputs.jets.n_tracks_valid"


class ValidTrackCountWriter(OutputSectionWriter):
    @staticmethod
    def _field(value=None):
        return OutputField(h5_name="n_tracks_valid", dtype="i4", axis="global",
                           prefix=False, value=value)

    def declare_io(self, mode: Mode) -> IO:
        if not (self.runs_in_mode(mode) and mode & Mode.TEST):
            return IO(requires={}, produces={})
        req = {"masks.tracks": TensorSpec(shape=None, dtype="bool", kind="pad_mask")}
        prod = {_LEAF: TensorSpec(shape=None, dtype=None, kind="data")}
        return IO(requires=unflatten_spec(req), produces=unflatten_spec(prod))

    def forward(self, b, mode: Mode):
        mask = b.get("masks.tracks")  # True = padded
        return {_LEAF: (~mask).sum(-1).to(torch.int32)}

    def is_run_task_output(self) -> bool:
        return True  # the dumb sinks source their column schema from these

    def manifest_fields(self, mode: Mode):
        if not (self.runs_in_mode(mode) and mode & Mode.TEST):
            return []
        return [(_LEAF, self._field())]
```

```yaml
# add_output.yaml — stack as a second --config on salt test
outputs:
  n_valid: {class_path: my_output.ValidTrackCountWriter, init_args: {modes: [test]}}
```

When stacking a second `--config` on `salt test`, pass `--ckpt_path`
explicitly: the best-checkpoint glob needs exactly ONE `--config` (it
looks next to the saved run config), so the no-`--ckpt_path` shortcut and
config stacking are mutually exclusive.

Notes for output authors:

- **Importability**: `class_path` is resolved with a normal import, so the
  module must be on `sys.path` — run from the directory holding
  `my_output.py` or set `PYTHONPATH` (with apptainer:
  `--env PYTHONPATH=/path/to/dir`).
- **`requires` kinds** for dataset-served keys: `inputs.<stream>` →
  `TensorSpec(dtype="float32", kind="data")`, `masks.<stream>` →
  `TensorSpec(dtype="bool", kind="pad_mask")`, `labels.<stream>.<name>` →
  `TensorSpec(kind="label")`, `meta.rows` → `TensorSpec(shape=(2,),
  dtype="int64", kind="meta", modes=Mode.TEST)`. Model-produced keys
  (`preds.*`) take an unconstrained `TensorSpec(shape=None, dtype=None)`.
- The section writer emits **torch values, never numpy**: the H5 sink
  packs each demanded `outputs.*` leaf itself (global `[B]`/`[B, C]` and
  per-token `[B, L]`/`[B, L, C]` shapes; per-token columns are zero-padded
  to the file sequence length by the sink — `h5_sink._pad_to`).
- Whole structured output groups (e.g. the MaskFormer `objects`/
  `object_masks` groups + the `tracks` HadronIndex column) go through the H5
  sink's declarative `object_groups` seam instead — a generic capability
  whose fields source arbitrary bundle leaves (no per-consumer knowledge in
  the sink). The MaskFormer object math lives ONLY in the `MaskFormerObjects`
  node (which mints the per-constituent index leaf — named `HadronIndex` in
  the shipped `MaskFormer.yaml` via `index_name`, single-source — in both TEST
  and ONNX); see `H5OutputSink(object_groups=[...])`.

## Inference: `salt inference` — the export set, offline

```bash
salt inference --ckpt_path <run_dir>/checkpoints/....ckpt \
  --data.test_file /tmp/v2/unlabelled.h5
# config inferred at <ckpt>/../../config.yaml (pass -c to override; -c stacks
# like fit); output defaults to {ckpt_dir}/{ckpt_stem}__inference_{sample}.h5
```

**`salt inference` == Athena semantics by construction.** The command's
TASK columns are STRICTLY the export output set written to H5 (no
separate config surface; the only other columns are the
export-mode `InputCopyWriter`/`PadMaskWriter` copy/mask columns, which
Athena never sees — see **Label-free** below): it compiles the SAME
`Mode.ONNX` plan `salt
export` traces (the section's export-mode `OutputField` selection, via the
implicit `OnnxExportSink`) and executes it eagerly per jet through the
`OnnxAdapter` — the exact eager reference the post-export `check_onnx`
sweep certifies against onnxruntime. Each jet is fed Athena-style (valid
tokens only, all-valid mask), so the H5 values match what Athena computes
from the exported network, at the checker tolerance. One H5 column per
ONNX tuple output, named `{run_name}_{suffix}` for tuple output
`{model_name}_{suffix}` (e.g. the seq-classification argmax lands as one
int8 `TrackOrigin` column, not per-class probs); per-token columns are
zero-padded to the file length, with the `mask` column marking pads.

**Label-free.** The dataset demand is derived from the export sink's `inputs`
alone (feature ports + pad masks + `meta.rows`) — no `labels.*` key is ever
demanded, so the command runs unchanged on a label-stripped file (the
`Labels` producer narrows to nothing; export-mode `get_output` is
label-free on the task side). No `target_{task}` columns are
written. `InputCopyWriter`/`PadMaskWriter` participate iff their `modes:`
include `export` (the default) — input copies re-read source columns by
row, which is file content, not label demand. On a labelled file a
copy-all `InputCopyWriter` therefore passes the label columns through
into the inference H5 verbatim: label-freedom is a DEMAND guarantee
(nothing is ever required of the file), not a redaction guarantee.
Restrict the writer's `variables:` (or strip the file) to keep labels
out of the output.

What governs participation is the single `modes:` surface: `export` in a
`RunTaskOutput`'s modes list puts its tasks in the ONNX tuple AND the
inference H5; `modes: [test]` keeps them eval-only. A config whose section
mints no export-mode field is refused (there is nothing Athena-visible to
write). Model-graph producers (the MaskFormer object reduces, a `Combination`)
declare ONNX-only fields, so they contribute to the tuple but not to a `salt
test` column. Note the eager
loop runs per jet (the ONNX-mode graph branches assume the Athena calling
convention) — for bulk labelled evaluation use `salt test`; this command
is the offline twin of the deployed network.

## Static graph tooling

All `salt graph` subcommands accept BOTH the trainer configs above and the
small toy-graph format, and never touch data. Saved run `config.yaml`
files round-trip directly (`salt graph plot -c <run_dir>/config.yaml ...`
— the run-surface `ckpt_path` key is accepted and ignored). `-c` is
**repeatable** for trainer configs with the fit/export deep-merge semantics
— the base + override pattern (e.g. the add-an-aux-task journey above) is
statically inspectable without a prior fit. The parsed
`outputs:` section (with its implicit command sinks) enters the static TEST
graph exactly as at runtime, so `validate`/`deadcode` fire the dead-preds
error for a narrowed section before anything runs, and the planner's
kind/dtype unification rejects a require that contradicts its producing
leaf (e.g. `preds.jets.classification` as `kind=label` where the task
publishes `data`) — data-free, not only at `salt test` setup:

```bash
salt graph validate -c salt/configs/gn2v2-opendata.yaml \
  --set model.modules.norm.init_args.norm_dict=unused.yaml
salt graph plan -c <cfg> --mode fit
salt graph plot -c <cfg> --mode fit -o graph.svg
salt graph why  -c <cfg> --mode fit --key encoded.seq
salt graph deadcode -c <cfg>
```

`--set KEY=VALUE` (repeatable) supplies required init_args **data-free** —
e.g. the `norm_dict` path that the shipped configs leave as a required
override is never read by static tooling, so any value works. `validate`
also runs the class-names ↔ schema-attrs cross-check when the reader
has a schema artifact, and reports module **preflights** (e.g. a missing
norm dict) as warnings — promotable with `--strict`; an actual
`salt fit` fails hard on the same check at fit start (fresh fits only —
resumes never re-read the norm/class dicts), with one `ConfigError`
covering every module before any value is materialised. Unconsumed
`preds.*` in FIT/VAL are **info**-level (the normal no-metric-callback
case) and never promoted, so `--strict` passes on a standard tagger config.

**ONNX mode checks the unified manifest**: the static ONNX sinks come from
the implicit `OnnxExportSink` folded over the
section's export-mode leaves — exactly what `salt export` will trace —
with errors attributed to the declaring section writer (`outputs.<name>`).
The export-only half of the contract — now the sink's `init_args` — is
validated as `salt export` does: an invalid `model_name` (`_`/`-`) is an
error-level `validate` finding (`plan`/`plot`/`why --mode onnx` raise it),
`rename:`/`combine:` are checked against the assembled manifest, and a
legacy config still carrying `export.outputs` fails with the migration
error. A trainer config whose sink declares neither `inputs` nor
`model_name` (and carries no deprecated top-level `export:` block either)
keeps the section-derived sinks but WARNS that inputs/model_name were
unchecked (promoted under `--strict` — the converter CI gate expects
converted configs to declare the sink's `init_args`). Predictions narrowed
out of the manifest (a
`modes: [test]` writer, or a task listed in no export-mode
`RunTaskOutput`) show up as info-level ONNX deadcode findings (the
export-pruning story). Note the static ONNX plan/plot remain the
**dataset-fed approximation** (reader/features included); the
authoritative rendering of the traced graph is the `plan_onnx.txt` that
`salt export` writes next to `network.onnx` — the two plan hashes
legitimately differ, and the CLI prints this caveat on
`plan`/`plot --mode onnx`.

**`salt graph resolve` was removed** with the `writers.modules` manifest
that was its data source: to see what eval writes and what Athena gets,
read the `outputs:` section directly, run
`salt export --manifest -c <cfg>` (no checkpoint needed), or consult the
manifest table in the export-time `plan_onnx.txt`.

`salt graph plot` writes the Graphviz `.dot` (port-card signature
layout, namespace-coloured modules) and rasterises it to a PNG + sibling PDF
by shelling out to the **`dot`** binary baked into the salt container. There
is no matplotlib path; a missing `dot` is a clear actionable error (the `.dot`
is still written for manual re-rendering).

### Run-dir artifacts

Every `salt fit`/`salt test` writes, at stage start (rank zero):
`plan_fit.txt`+`plan_val.txt` / `plan_test.txt` (dataset + model plan
tables, with the narrowed label list, the demand-narrowed per-group read
columns, and — test — the writer-sinks table: writer instance → consumed
keys), `resolved_io.yaml` (machine-readable: per mode, the plan sources and
every module's flattened requires/produces with resolved specs) and
`graph_<stage>.{dot,svg}` (+ `graph_<stage>_dataset.{dot,svg}`). Fit
artifacts go into the trainer log dir; test artifacts go NEXT TO THE
CHECKPOINT, with the eval H5. Default-on via the `artifacts:` entry in
`base.yaml` (`salt.callbacks.GraphArtifacts`); delete with
`--callbacks.artifacts=null`, retarget with
`--callbacks.artifacts.init_args.output_dir=...`. (Writer nodes are not
rendered in the `graph plot` output itself — the plan-table writer-sinks
section is the current source of that answer.)

## Metrics callbacks

`salt.callbacks.ConfusionMatrix` is the v1 `ConfusionMatrixCallback`
port: it reads the step bundle (`preds.<stream>.<task>` argmax vs
`labels.<stream>.<label>`), resolves stream/label/class names from the named
task module, and logs to Comet at each validation epoch end (values are also
stashed on the callback: `last_matrix`, `last_truth_labels`,
`last_pred_labels`). Configure it like any callback:

```yaml
callbacks:
  confusion_matrix:
    class_path: salt.callbacks.ConfusionMatrix
    init_args:
      task_name: jets_classification
```

Deferral note: callback-declared `requires` are NOT yet FIT/VAL plan sinks
— `ConfusionMatrix` works because tasks publish `preds.*` in all modes and
stay alive via their losses (its VAL labels are demanded by the task
itself). A callback demanding a key no task keeps alive would currently
find its producer demand-pruned; the sink wiring mirrors the TEST
writer-demand mechanism (see `SaltModule._model_sinks`).

## Notable surface notes

- `lrs:` is the OneCycleLR block (renamed from the v1 ModelWrapper
  `lrs_config:` kwarg). Schema:
  `{initial, max, end, pct_start[, weight_decay, last_epoch]}` driving
  AdamW/lion/HybridMuonAdamW + OneCycleLR.
- `VertexingTaskModule.origin_weighting` takes integer origin ids OR class
  NAMES (defaults reproduce v1's `heavy: [3,4,5], fake: [1]`); names (the
  GN3 origin surface) are resolved to ids at fit/test setup against the
  origin label's class-name attr in the schema artifact. A name-based
  config without a schema artifact is a loud bind error.
- Loggers (Comet) and run dirs are not wired yet; losses show on the stock
  progress bar meanwhile (`train/loss`, `train/<task>_loss`).
- The vertexing `get_output` writes the eval column as bare `VertexIndex`
  (i8) by default — the v1 byte-schema; the run-name prefix is opt-in via
  `VertexingTaskModule`'s `prefix_vertex_column: true` (default polarity to
  be revisited when v1 byte-parity gating retires).
- Per-task `expose: [fit, val]`: a train-only aux task gates its `preds.*`
  port to the listed modes — `[fit, val]` prunes it from the TEST/ONNX plans
  (silencing the dead-preds error) while it keeps training.
  `--model.modules.<task>=null` (full deletion) is the alternative.

## ONNX export: `salt export`

```bash
salt export --ckpt_path <run_dir>/checkpoints/epoch=...-loss=....ckpt
# config inferred at <run_dir>/config.yaml (pass -c to override); output
# defaults to <run_dir>/network.onnx (--output / -o/--overwrite to control)
```

`-c` is **repeatable** with the fit deep-merge semantics — the supported
way to export a run trained before the export sink's `init_args` were
declared:

```bash
salt export --ckpt_path <ckpt> -c <run_dir>/config.yaml -c my_export_sink.yaml
# my_export_sink.yaml carries ONLY the outputs.onnx_export sink entry below
```

The export contract has two halves ("one manifest and a half"):

1. **The outputs come from the `outputs:` section** (the same declarations
   that name the eval columns — see the evaluation section above): the
   export-mode selection is folded into the implicit `OnnxExportSink`.
   There is NO `export.outputs` section: a config declaring one fails with
   the migration error. Inspect the assembled manifest any time with
   `salt export --manifest -c <config>` (no checkpoint needed).
2. **The export-only half** lives in the **`OnnxExportSink`'s `init_args`**
   (declared under the top-level `outputs:` section; parsed by the normal
   salt surface so it round-trips through saved run configs):

```yaml
outputs:
  onnx_export:
    class_path: salt.outputs.OnnxExportSink
    init_args:
      model_name: GN2v2 # no '_'/'-' — validated ONLY at export time; the run
      #                   name stays unrestricted (default: run name stripped)
      inputs:
        - { port: inputs.jets, name: jet_features } # [1, F] global
        - { port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks } # [L, F]
      # optional Athena-presentation post-processing of the output manifest
      # (the v1 --rename/--combine_outputs features — readers of exotic
      # configs consult the outputs: section AND these two keys):
      rename: { pu: plight } # old suffix -> new (existence-checked)
      combine: # new output = sum(scale * existing GLOBAL output)
        - { name: pbc, inputs: { pb: 0.5, pc: 0.5 } }
```

   A top-level `export:` block setting these same five keys is a deprecated
   alias, kept for one release window (`DeprecationWarning` on use): each key
   it sets fills a field the sink itself left unset, and a key carried by
   both homes is a `ConfigError` naming the key and both homes.
   `salt/configs/MaskFormer.yaml` ships the sink form; `salt/configs/gn2v2-opendata.yaml`
   still ships the deprecated block, deliberately, as the alias-window proof.

How it works (no data file is touched — config + checkpoint only):

- The **ONNX plan** is compiled with the folded `OnnxExportSink`'s
  demanded `outputs.*` leaves as sinks — labels/losses/the section's
  TEST-only leaves are demand-pruned automatically; sources mirror the
  `Features` declaration (widths/columns from `variables:`).
- The **`OnnxAdapter`** wrapper assembles the bundle in-graph (sequences
  `[L, F]` -> `[1, L, F]` + all-valid pad masks — the v1 traced pattern),
  executes the frozen plan, and emits the flat output tuple. Every module
  recursively receives `set_export_mode()` (the encoder forces torch-math
  attention — the Athena-agreement requirement).
- **Per-family conversions** (squeezed per-class scalar probs, argmax
  int8 indices, union-find) run INSIDE the traced graph via each task's
  `get_output`; the folded sink does no math — it only NAMES the resulting
  `outputs.*` leaves (with their dtypes and dynamic axes) into the flat
  Athena tuple. Union-find runs on the raw edge scores the vertexing task
  publishes in ONNX mode (per-family exception).
- **`rename:`/`combine:`** post-process the manifest with v1 semantics:
  renames apply first (existence-checked), combined outputs are linear
  combinations of the (renamed) GLOBAL float outputs computed inside the
  traced graph, and they insert after the global entries but BEFORE the
  per-token aux entries — the v1 output order. The `OnnxExportSink` tuple
  itself follows its manifest sources' declaration order, which is not a
  contract: Athena consumes the outputs by name.
  Both are recorded in `gnn_config` byte-compatibly with v1
  (`combine_outputs`/`rename_outputs`).
- **Multi-stream trace-safety**: eager `Split` slicing is wrong under
  tracing with ≥2 dynamic sequence axes; in ONNX
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
  declarations match, name-resolved `index_select` gather otherwise).
- **Artifacts**: alongside `network.onnx` the exporter writes
  `plan_onnx.txt` — the plan table of the graph Athena will actually
  run (union-find placement, `Split` `index_select` mechanism) PLUS the
  output-manifest table. This is the authoritative ONNX plan;
  `salt graph plan --mode onnx` shows the dataset-fed static
  approximation.
- **Expected console output**: a clean export prints NO trace warnings.
  The torch `aten::index ... indices of type Byte` UserWarning (raised on
  the v1-verbatim bool-mask gathers in the union-find/zero-token tricks,
  identically noisy under v1's exporter) is deliberately suppressed around
  `torch.onnx.export` — the default-on sweep checker (incl. L=0) is the
  proof the traced graph is correct.

Programmatic surface for gates/tests (no checkpoint needed):
`salt.outputs.sinks.onnx.export_graph(modules, export_cfg, variables, path)` — the
output set derives from the folded `OnnxExportSink` in `modules`; passing
a legacy reduce-manifest `outputs=` list is a hard `ConfigError` — plus
`salt.outputs.sinks.onnx.check_onnx(adapter, path, ...)`.

## Checkpoints and resume

Checkpoints carry the resolved schema + per-mode plan hashes under the
top-level `salt_core` key. On resume, a FIT plan-hash mismatch is fatal
(the graph or dataset boundary changed); `Normaliser`/class-weight values
come from the state dict — the norm/class dicts are NOT re-read.
Data-less loading: `SaltModule.load_from_checkpoint(path, modules=...)`.

### Class-path compatibility across the de-core rename

The v2 namespace was flattened from `salt.core.*` to `salt.*` (e.g.
`salt.core.nn.StreamEmbed` → `salt.model.modules.StreamEmbed`,
`salt.core.SaltModule` → `salt.model.SaltModule`). Two loading policies:

- **Pre-rename v2 checkpoints** (and their saved `config.yaml`, which embed
  `salt.core.*` class_paths) **load unmodified.** A load-time remapper in
  `salt/main.py` (`_remap_class_path`, a longest-prefix `salt.core.* → salt.*`
  table) is applied at every class-path resolution site: the salt-owned
  `_resolve_class_path`/`_is_persistence_sink`, and — via a wrapper installed on
  jsonargparse's `import_object` — the top-level model **subclass resolution
  during config parse**. So `salt test --config <old_run>/config.yaml
  --ckpt_path …` just works. The `salt_core` checkpoint **metadata key** is a
  plain dict key (not an import path) and is intentionally unchanged.
  The remapper table + its tests (`salt/tests/unit/test_ckpt_compat.py`) are the
  only sanctioned `salt.core.*` strings in the codebase.

- **v1 checkpoints** (the `ModelWrapper` `model.pool_net.*` state-dict layout)
  are **not** directly loadable — `SaltModule.on_load_checkpoint` rejects them
  with a clear error. Convert them offline via the v1→v2 weight mapper
  (`map_v1_state_dict` / the `scripts/convert_v1_model.py` pattern) or use them
  at the frozen pin `29c67a1` (see [Parity-closure doctrine](#parity-closure-doctrine-v1-vs-v2-comparisons)).

## Parity and gate harnesses — RETIRED

The standalone v1-vs-v2 migration harnesses (`parity_gn2`, `gates_m2`,
`gates_m3`, `gates_m4`, `gates_m6` + their pytest wrappers) served the
v1→v2 migration and are retired per the parity-closure doctrine above —
they ran green at the frozen pin and live in git history (the m2/m3/m6
sweep closed at `1190d7f`; `git checkout 29c67a1` for the full v1-vs-v2
set). Live coverage of the same surfaces is the ordinary test suite:
`salt/tests/unit` (CPU-safe) and `salt/tests/integration` (the
end-to-end fit/test/export flows, e.g. `test_outputs_section.py`,
`test_onnx_export.py`, `test_regression_e2e.py`).
