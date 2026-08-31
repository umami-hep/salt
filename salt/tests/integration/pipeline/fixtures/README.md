# `pipeline/fixtures/` — one hand-curated YAML per shipped config

Every non-`base.yaml` config under `salt/configs/` has exactly one fixture
file here, and every fixture file names exactly one config — bidirectional
completeness is enforced by a test in `test_pipeline.py`. `base.yaml` is the
single hardcoded exemption (it is auto-loaded machinery, never a model in
its own right).

**Filename stem = row id** — the pytest parametrize id, the artifact-cache
key (`{ckpt:NAME}`/`{config:NAME}` chained lookups), and the value passed to
the CI GPU job's `--pipeline-row=<id>` selector. For a fragment fixture the
stem is the fragment config's own basename (no `.yaml`).

## Schema

```yaml
config: <relpath under salt/configs/>.yaml
gpu: true                     # optional; row runs in the GPU CI job
fit: false                    # or an args block; default true
fit:
  args: ["--trainer.max_epochs=1"]
eval:
  expected_outputs:           # group -> [column, ...], eval-H5 vocabulary
    jets: [pb, pc, pu]
onnx:
  expected_outputs: [pb, pc, pu]   # ordered ONNX output-tuple names
inference:
  expected_outputs:           # same file-dataset vocabulary as eval
    jets: [pb, pc, pu]
xfail:                        # leg-scoped, reason mandatory
  fit: "reason..."
  eval: "reason..."
  export: "reason..."
  compile_plot: "reason..."
fragment: <value>             # fragment files only: "paired" | "included" | free text
```

`expected_outputs` semantics: **containment, not equality** for `eval`/`onnx`
(a declared name must be present; extra columns are fine and expected) —
except the inference ONNX-name gate, which checks **exact tuple order**.

## No regeneration script — ever

These files are **hand-curated oracles**, not machine-generated snapshots.
Seed or update an entry by actually running `salt test`/`salt export` and
reading what came out, then editing the YAML by hand. **Never write a
regeneration script for this directory** — this carries forward the study's
"GOLDEN SNAPSHOTS ARE GONE — EXPECTED_OUTPUTS is the oracle" ruling
(user, 2026-08-30) verbatim to these fixtures, per plan 04. Accepted
trade-off: undeclared drift (a rename nobody wrote down) is not
auto-detected.
