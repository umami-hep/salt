# Command line reference

Every `salt` subcommand, gathered in one place: what it takes, what it
writes, and where the output lands. `salt fit` and `salt test` go through
the Lightning-based trainer CLI; `salt graph`, `salt schema`, `salt export`
and `salt inference` are trainer-free and touch no data beyond the file
they are explicitly pointed at. Config structure and settings (what goes
under `data:`/`model:`/`trainer:`) live on [Configuration](configuration.md);
this page is about the commands themselves.

## Quickstart

One-time setup: if the container's installed salt predates the modular
`salt` package, `python -m salt.main` fails with `ModuleNotFoundError: No
module named 'salt'` from any cwd except the repo root, and there is no
`salt` script on PATH. `pip install -e .` (from the repo root, inside the
container) fixes both; afterwards `salt` works from any directory.

```bash
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

Recommended extra: dump a schema artifact (`salt schema dump
/tmp/v2/train.h5 -o /tmp/v2/schema.yaml`, see [salt schema
dump](#salt-schema-dump) below) and pass it as
`--data.modules.reader.init_args.schema /tmp/v2/schema.yaml`. With it, a
field typo fails statically before any data is read, and the configured
`class_names` are cross-checked, by set and by order, against the file's
label attrs.

**Where outputs land.** `config.yaml`, `ckpts/` and the fit
plan/graph artifacts are written into the trainer log dir, which is your
cwd unless you pass `--trainer.default_root_dir <dir>`. That is why the
quickstart command above sets it: without it, a fit run from the repo root
drops around eight files into the checkout. The end-of-fit message prints
the exact paths, and a leftover `config.yaml` from a previous run in the
same directory is overwritten rather than left stale. `salt test` and
`salt inference` artifacts land next to the checkpoint they read, not in
`default_root_dir`; see [salt test](#salt-test) and [What a run
writes](#what-a-run-writes) below.

## salt fit

```bash
salt fit --config salt/configs/gn2v2-opendata.yaml
```

`base.yaml` is auto-loaded underneath whatever you pass; a config file only
needs to state what differs from it. `--config` is repeatable, and unlike
stock jsonargparse the merge is a deep merge on dict leaves, key by key,
with the later file winning per key rather than replacing a whole block:
`--config a.yaml --config b.yaml` keeps siblings that `b.yaml` never
mentions. CLI overrides use the same dotted spelling as the YAML tree,
e.g. `--model.modules.encoder.init_args.num_layers=3`, and a module or
callback entry is removed by setting it to `null`. The full mechanics of
this merge, `include:` expansion, and the null-deletion rule are
documented once, on [How a config is
assembled](configuration.md#how-a-config-is-assembled); this section is
about running `fit`, not about the merge itself.

Two flags worth knowing before your first run:

- `--trainer.fast_dev_run 2` runs two training and two validation batches
  and exits, with logging and checkpointing suppressed. Reach for it to
  check that a config parses and a fresh module wires up, before spending
  a real epoch on it. Always spell the flag `--trainer.fast_dev_run`
  in full; a bare, unnamespaced spelling is not a recognised flag.
- `--init_from <ckpt>` warm-starts a fresh run from a checkpoint's weights,
  loaded per module with a strict per-module accounting: retained modules
  must be fully covered by the checkpoint, modules in the config but absent
  from the checkpoint are freshly initialised, and checkpoint modules
  absent from the config are dropped (and logged). It is mutually
  exclusive with `--ckpt_path` on `fit`: `--ckpt_path` resumes a run
  (trainer state restored, a strict full-model weight load, a plan-hash
  check enforced), while `--init_from` starts a new run from a possibly
  surgically changed architecture. Passing both raises a `ConfigError`
  naming the conflict. See [Checkpoints and
  resume](training.md#checkpoints-and-resume) for resuming, and
  [Fine-tuning](finetuning.md) for the warm-start-versus-resume
  distinction and worked fine-tuning examples.

Run `salt fit --help` for the full flag list; most of what it prints
(`--model.*`, `--data.*`, `--trainer.*`, `--callbacks`, `--outputs`,
`--training_schedule`, `--compile`, `--class_dict`) is config surface
documented on [Configuration](configuration.md), not CLI mechanics.

## Previewing the resolved config: --print_config

This is required to print the config salt actually assembled, after every
source has been merged, and then exit without training or reading any
data. Reach for it when a stack of `--config` files did not do what you
expected, when you want to know which value won a merge, or when you want
to save one self-contained config file for later reuse.

Invocation: append `--print_config` to any `salt fit` or `salt test`
command line you would otherwise run.

```bash
salt fit --config salt/configs/gn2v2-opendata.yaml \
  --model.modules.encoder.init_args.num_layers=3 \
  --print_config > resolved.yaml
```

The command exits 0 after printing and instantiates nothing, so it needs
no data file and no GPU.

Salt's version of this flag is not stock jsonargparse. Salt intercepts
`--print_config` on the raw argv, strips it before the parser sees it, and
defers the dump until after `include:` expansion, the deep-merge fan-out,
the `training_schedule` relocation, and validation have all run. What you
see on stdout is the fully resolved config that would actually be used to
build the trainer, not an intermediate snapshot from partway through
parsing.

Two flag values are accepted, and they may be combined comma-separated:

- `--print_config=skip_default` omits any key still at its declared
  default.
- `--print_config=skip_null` omits any key whose resolved value is
  `null`.

Any other value after the `=` raises a `ConfigError` that names the two
supported flags, so a typo fails loudly instead of silently printing the
full dump.

The dump is ordered so that every `class_path`/`init_args` mapping lists
`class_path` first, even when the override that produced it only touched
`init_args`. This is the same serialisation pass used by the `config.yaml`
a fit run saves and by `salt merge-config`, which is why the YAML produced
by all three agrees key for key.

`salt merge-config` reaches the same merged config through the same
parser, but goes further: it also renders one static model-graph plot per
`training_schedule` stage with that stage's frozen modules marked. Its
merged YAML output is byte-identical to `salt fit [same args]
--print_config` for a config stack with no staged schedule. Reach for
`salt merge-config` when you also want the per-stage freeze graphs; reach
for `--print_config` when the YAML is all you need.

## salt test

```bash
salt test --config <run_dir>/config.yaml \
  --ckpt_path <run_dir>/ckpts/....ckpt \
  --data.test_file /tmp/v2/pp_output_test_ttbar.h5
```

The saved run `config.yaml` from the training directory is the recommended
single base for `salt test`: further `--config` override files and CLI
flags stack on top of it exactly as they do for `fit`. One test file per
call, the experiment logger is disabled, and only a single device is
used, forcing `--trainer.devices=1` with a logged notice if you pass more.
A config with no `outputs:` section (and no explicit callback-based
persistence sink) is rejected with a `ConfigError`, because `salt test`
would then have nowhere to write its predictions.

**Best-checkpoint selection.** Without `--ckpt_path`, salt globs for the
lowest `loss=`-named checkpoint under `ckpts/` (salt's own `Checkpoint`
callback default, `salt/callbacks/checkpoint.py`) or `checkpoints/`
(Lightning's `ModelCheckpoint` default) next to the single `--config` you
passed; `base.yaml` names checkpoint files
`epoch=NNN-loss=<val/loss>.ckpt` so this glob always matches a v2 run's
own output. This only works with exactly one `--config`: when you stack a
second `--config` on top of the run's saved one, pass `--ckpt_path`
explicitly, because the glob needs a single config file to anchor its
search directory on. See [Checkpoints and
resume](training.md#checkpoints-and-resume) for the general resume story,
and [Fine-tuning](finetuning.md) for the distinction between warm-starting
from a checkpoint's weights and resuming a run from one.

**Where the output lands.** `salt test` writes one H5 next to the
checkpoint it read:

```
{ckpt_dir}/{ckpt_stem}__test_{sample}.h5
```

`{sample}` comes from the test file's stem, with `--data.test_suff`
appended when it is set (so re-running against the same test file with a
different suffix never overwrites the previous output). The test plan and
graph artifacts (`plan_test.txt`, `resolved_io.yaml`, `graph_test.svg`)
land in the same directory as the eval H5; see [What a run
writes](#what-a-run-writes).

For what the eval H5 contains and how to add columns to it, see
[Evaluation](evaluation.md) and [Outputs](outputs.md).

## salt schema dump

This is required to read a training file's structure and write it out as a
YAML artifact, so that salt can check a config against the file's real
fields before any data is read. Reach for it once per dataset: with the
artifact wired in, a field typo in a config fails statically instead of at
the first batch, and the configured `class_names` are cross-checked
against the file's label attributes, both by set and by order.

Invocation:

```bash
salt schema dump <file.h5> -o <schema.yaml>
```

The input file is positional and `-o`/`--output` is required. This command
takes no `-c`/`--config` and no `--mode`; it only ever reads the one H5
file you name.

Wire the resulting artifact into a training config with
`--data.modules.reader.init_args.schema <schema.yaml>`, or the equivalent
`schema:` key under the reader's `init_args` in the config file.

**What it reads.** Every top-level structured dataset in the file becomes
a schema group carrying its field names, its dtypes, and its HDF5
attributes; the file's own global attributes are kept too. Non-structured
datasets and nested groups are ignored, so only the flat, structured
arrays salt actually reads participate.

**What it writes.** A YAML file with a `schema_version`, the file's
`attrs`, and a `groups:` mapping of `{name: {fields, attrs}}`. Groups are
written in alphabetical order by group name, while within each group the
field order is preserved as it appears in the file, so the artifact still
diffs cleanly under version control between two dumps of similar files.

Two things can surprise you here. A dataset or field name containing a
literal `.` is legal in HDF5 but cannot be addressed as a dotted bundle
key, so it is skipped with a warning on stderr rather than failing the
whole dump. And a file with no structured datasets at all still produces
an artifact, just an empty one, printing `WARNING: no structured datasets
found in <path>` on stderr rather than raising. A file that cannot be
opened as HDF5 raises a `SchemaError`; a missing input file is a plain
error exit.

On success the command prints:

```
wrote schema for N group(s) to <output>
```

## salt inference

```bash
salt inference --ckpt_path <run_dir>/ckpts/....ckpt \
  --data.test_file /tmp/v2/unlabelled.h5
# config inferred at <ckpt>/../../config.yaml (pass -c to override; -c stacks
# like fit); output defaults to {ckpt_dir}/{ckpt_stem}__inference_{sample}.h5
```

`salt inference` matches Athena's semantics by construction. The output
columns are strictly the export output set written to H5, with no separate
config surface of their own (the only other columns are the export-mode
`InputCopyWriter`/`PadMaskWriter` copy and mask columns, which Athena never
sees, see Label-free below): it compiles the same `Mode.ONNX` plan `salt
export` traces, via the implicit `OnnxExportSink`, and executes it eagerly
per jet through the `OnnxAdapter`, the exact eager reference the
post-export checker sweep certifies against onnxruntime. Each jet is fed
Athena-style (valid tokens only, an all-valid mask), so the H5 values match
what Athena computes from the exported network, at the checker's
tolerance. One H5 column is written per ONNX tuple output, named
`{run_name}_{suffix}` for a tuple output `{model_name}_{suffix}` (so a
seq-classification argmax lands as one int8 `TrackOrigin` column, not one
column per class); per-token columns are zero-padded to the file length,
with a `mask` column marking the pad positions.

**Label-free.** The dataset demand for this command is derived entirely
from the export sink's `inputs`: feature ports, pad masks, and
`meta.rows`. No `labels.*` key is ever demanded, so the command runs
unchanged on a label-stripped file (the `Labels` producer narrows to
nothing, and export-mode `get_output` is label-free on the task side). No
`target_{task}` columns are written. `InputCopyWriter`/`PadMaskWriter`
still participate when their `modes:` include `export` (the default),
because input copies re-read source columns by row, which is file content
rather than label demand; on a labelled file a copy-all `InputCopyWriter`
therefore passes the label columns through into the inference H5
verbatim. Label-freedom here is a guarantee about what the run *demands*
from the file, not a guarantee that labels are redacted from the output.
Restrict the writer's `variables:` list, or strip the file, to keep labels
out.

What governs whether a task's output appears here is the single `modes:`
surface on its section entry: `export` in a `RunTaskOutput`'s `modes:`
list puts its tasks in both the ONNX tuple and the inference H5;
`modes: [test]` keeps them eval-only and out of this command's output. A
config whose `outputs:` section mints no export-mode field is refused,
since there would be nothing Athena-visible to write. Model-graph
producers (the MaskFormer object reduces, a `Combination` node) can
declare ONNX-only fields, so they contribute to the exported tuple but
never to a `salt test` column. The eager loop runs one jet at a time,
because the export-mode graph branches assume the Athena calling
convention; for bulk labelled evaluation use `salt test` instead. This
command is the offline twin of the network Athena will run.

## salt graph

All `salt graph` subcommands accept both the trainer configs used by
`fit`/`test` and a small toy-graph format, and none of them touch data.
Saved run `config.yaml` files round-trip directly
(`salt graph plot -c <run_dir>/config.yaml ...`, the run-surface
`ckpt_path` key is accepted and ignored). `-c` is repeatable for trainer
configs, with the same deep-merge semantics as `salt fit`, so a base
config plus an override file is statically inspectable without ever
running a fit. `--set KEY=VALUE` (repeatable) supplies a required
`init_args` value data-free, e.g. the `norm_dict` path a shipped config
leaves as a required override is never read by the static tooling, so any
placeholder value works.

```bash
salt graph validate -c salt/configs/gn2v2-opendata.yaml \
  --set model.modules.norm.init_args.norm_dict=unused.yaml
salt graph plan -c <cfg> --mode fit
salt graph plot -c <cfg> --mode fit -o graph.svg
salt graph why  -c <cfg> --mode fit --key encoded.seq
salt graph deadcode -c <cfg>
```

With no `--mode` given, `validate` compiles all four primary modes: FIT, VAL,
TEST and ONNX (`salt/cli.py:608-613`, the mode loop at `:705` and `:758`).
TEST and ONNX both anchor on a `preds.*` key, so a config with no task head
fails those two modes with

```
no module produces a 'preds.*' key in mode {MODE} — evaluation plans anchor on predictions
```

and the command exits 1 (`cli.py:1283-1286`, `_fail` at `:617-619`). Check
only the data half, before a task head or an `outputs:` section exists, with
`--mode fit`:

```bash
salt graph validate -c cfg.yaml --mode fit
```

`validate` also runs the class-names-versus-schema-attrs cross-check when
the reader carries a schema artifact, and reports module preflight
problems (a missing norm dict, say) as warnings, promotable to errors with
`--strict`. A real `salt fit` still fails hard on the same check at fit
start on a fresh fit (a resume never re-reads the norm or class dicts),
raising one `ConfigError` that covers every affected module before any
value is materialised. Unconsumed `preds.*` keys in FIT/VAL are info-level
only (the ordinary no-metric-callback case) and are never promoted, so
`--strict` passes cleanly on a standard tagger config.

ONNX mode checks the unified manifest: the static ONNX sinks come from the
implicit `OnnxExportSink` folded over the section's export-mode leaves,
which is exactly what `salt export` will trace, with any error attributed
to the declaring section entry (`outputs.<name>`). Predictions narrowed
out of the manifest (a `modes: [test]` writer, or a task listed in no
export-mode `RunTaskOutput`) show up as info-level ONNX deadcode findings
rather than errors. The static ONNX plan and plot remain a dataset-fed
approximation of the traced graph (reader and feature modules are
included as if for a normal run): the authoritative rendering of what
`salt export` actually traced is the `plan_onnx.txt` it writes next to
`network.onnx`, and the two plan hashes are allowed to differ for that
reason. `salt graph plan`/`plot --mode onnx` print this caveat directly.

**`salt graph resolve` was removed** along with the `writers.modules`
manifest block that was its only data source. To see what an eval run
writes and what Athena receives, read the config's `outputs:` section
directly, run `salt export --manifest -c <cfg>` (no checkpoint needed), or
consult the manifest table in the export-time `plan_onnx.txt`.

`salt graph plot` writes the Graphviz `.dot` source (a port-card layout,
modules coloured by namespace) and rasterises it to a PNG with a sibling
PDF by shelling out to the `dot` binary baked into the salt container.
There is no matplotlib fallback path; a missing `dot` raises a clear,
actionable error, and the `.dot` file is still written so you can render
it manually later.

### salt graph why

This is required to answer one question about a single bundle key: who
produces it, who consumes it, and if it is missing, why. It reads config
only and touches no data, so it works before a run has ever happened, and
equally well against a saved run's own `config.yaml`.

Invocation:

```bash
salt graph why -c <config> --mode <mode> --key <dotted key>
```

`-c` is repeatable with the same deep-merge semantics as `fit`; `--mode`
defaults to `fit`; `--set KEY=VALUE` supplies a required `init_arg`
data-free, exactly as for the other `salt graph` subcommands. An
unparseable `--key` (one that does not split into dotted components)
raises a `ConfigError`.

**When the key is present**, the command prints four lines:

```
[mode=FIT] 'encoded.seq'
  producer:  encoder (TransformerEncoder)
  spec:      kind=data, shape=(None, None, 128), dtype=float32
  consumers: pool
```

The first line names the key and the mode it was resolved in. `producer:`
names either the plan's source set (printed as `<sources>` for a key that
comes straight from the reader) or `<module name> (<ClassName>)` for a
key a module produces. `spec:` prints the resolved `kind`, `shape`, and
`dtype` of that key's `TensorSpec`. `consumers:` lists every module that
reads the key, sorted by name; when nothing reads it, the line instead
reads as below, pointing you at the command that lists every such dead
output at once:

```
consumers: none — dead output in mode <MODE> (see salt graph deadcode)
```

**When the key is absent**, the command prints `[mode=<MODE>] '<key>' is
not in the plan.` followed by whichever of these applies to your case:

- The key is produced, but the module that produces it is demand-pruned
  out of this mode's plan entirely. The output names the prune reason and
  adds a `fix:` line telling you to add a sink or consumer that demands
  the key in this mode, or to remove the module if it serves no purpose
  here.
- The key is produced only in some other mode, and is mode-gated out of
  the one you asked about. The output names which modes it does exist in.
- The key matches a wildcard producer's pattern, but nothing demands it in
  this mode: wildcard producers only materialise the specific keys
  something downstream asks for, so an unmatched pattern hit is not an
  error, just an unmaterialised possibility.

Exit codes: `0` when the key was explained (present or one of the absent
cases above), `1` when the key is unknown everywhere in the plan, in which
case the command's stderr output carries nearest-key suggestions drawn
from every key it does know about.

Reach for `salt graph why` when a column is missing from your eval file,
when a `BindError` names a key you were sure existed, or to confirm that a
new module's produced key is actually consumed by anything before you go
looking for a bug in its `forward`.

## salt export

```bash
salt export --ckpt_path <run_dir>/ckpts/epoch=...-loss=....ckpt
# config inferred at <run_dir>/config.yaml; output defaults to <run_dir>/network.onnx
```

`-c`/`--config` is repeatable with the same deep-merge semantics as `salt
fit`, which is the supported way to export a run that was trained before
its export sink's `init_args` were declared:

```bash
salt export --ckpt_path <ckpt> -c <run_dir>/config.yaml -c my_export_sink.yaml
# my_export_sink.yaml carries ONLY the outputs.onnx_export sink entry
```

`salt export --manifest` prints the assembled output manifest (the full
ONNX names, dtypes, and any `rename`/`combine` post-processing) and exits
without needing a checkpoint at all; reach for it when you want to check
what a config will export before you have a trained model to export it
from. `--no-check` skips the default-on torch-vs-ONNX sweep checker that
otherwise runs after every export (the checker draws random sequence
lengths, compares eager torch against onnxruntime, and is what actually
certifies the exported graph is correct). `--output`/`-o` controls where
the `.onnx` file is written; the default path sits next to the config that
was inferred or passed with `-c`.

Everything else about `salt export`, the two-part manifest model, the
`rename`/`combine` post-processing keys, the aliasing mechanism, and how
to validate an exported model, is documented on [Export to
ONNX](deployment/export.md); this section is the invocation only.

## What a run writes

Every `salt fit`/`salt test` run writes, at stage start on rank zero:
`plan_fit.txt`+`plan_val.txt` (or `plan_test.txt` for a test run), which
are the dataset and model plan tables, including the narrowed label list,
the demand-narrowed per-group read columns, and, for a test run, the
writer-sinks table (writer instance to consumed keys); `resolved_io.yaml`,
a machine-readable dump of, per mode, the plan's sources and every
module's flattened requires/produces with resolved specs; and
`graph_<stage>.{dot,svg}` plus `graph_<stage>_dataset.{dot,svg}`, the
rendered graph images.

Fit artifacts go into the trainer log dir (see [Where outputs
land](#quickstart) above); test artifacts go next to the checkpoint, in
the same directory as the eval H5. This is default-on via the `artifacts:`
entry in `base.yaml` (`salt.callbacks.GraphArtifacts`); delete it with
`--callbacks.artifacts=null`, or retarget it with
`--callbacks.artifacts.init_args.output_dir=...`. Writer and sink nodes
are not themselves drawn in the `graph plot` output; the plan table's
writer-sinks section is the current source of that information for a test
run.
