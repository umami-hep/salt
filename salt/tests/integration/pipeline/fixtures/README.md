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
  expected_outputs:           # ordered ONNX output-tuple names (exact tuple order),
                              # NOT the eval/file-dataset vocabulary
    [GN3V00_pb, GN3V00_pc, GN3V00_ps, GN3V00_pud, GN3V00_pg, GN3V00_ptau,
     GN3V00_ptFromTruthDressedWZJet, GN3V00_TrackOrigin, GN3V00_VertexIndex,
     GN3V00_TrackType]
xfail:                        # leg-scoped, reason mandatory
  fit: "reason..."
  eval: "reason..."
  export: "reason..."
  compile_plot: "reason..."
stack: [<relpath under salt/configs/>.yaml, ...]  # optional; shipped configs
                              # stacked before this row's own config (compile+plot
                              # and fit) — for overlay templates whose base cannot
                              # itself be a producer row
fragment: <value>             # fragment files only: "paired" | "included" | free text
```

`expected_outputs` semantics: **containment, not equality** for `eval`/`onnx`
(a declared name must be present; extra columns are fine and expected) —
except the inference ONNX-name gate, which checks **exact tuple order**.

## No real `--init_from` warm start in CI (currently)

`finetune_same_heads.yaml` — the fixture for the old `finetune_gn3large.yaml`
overlay — was a GPU row that ran a real `--init_from` warm start onto
`gn3v00_base`'s checkpoint. Both the overlay and its fixture were deleted
(plan 18): the checkpoint-chaining producer/consumer relationship they
exercised no longer exists anywhere in the tree.

The overlays that later replaced it (`finetune_gn3large_add_calo`,
`finetune_gn3large_add_charge_head`, `finetune_gn3large_add_jet_vars`,
`finetune_gn3large_xbb_transfer`, `gn3large_base`) moved to
`docs/tutorials/configs/finetuning/` and their `stack:`-overlay pipeline
fixtures were deleted along with them — they are tutorial material, not
shipped configs. **No branch-CI row exercises a real `--init_from` warm
start.** The overlays' static partitions and accounting (loaded/new/dropped
modules) are covered by `salt/tests/unit/test_finetune_configs.py` instead.

## No regeneration script — ever

These files are **hand-curated oracles**, not machine-generated snapshots.
Seed or update an entry by actually running `salt test`/`salt export` and
reading what came out, then editing the YAML by hand. **Never write a
regeneration script for this directory** — this carries forward the study's
"GOLDEN SNAPSHOTS ARE GONE — EXPECTED_OUTPUTS is the oracle" ruling
(user, 2026-08-30) verbatim to these fixtures, per plan 04. Accepted
trade-off: undeclared drift (a rename nobody wrote down) is not
auto-detected.
