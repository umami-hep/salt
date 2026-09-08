#!/usr/bin/env bash
#
# (Re)generates the nine per-stage "freeze graph" .dot/.png pairs embedded in
# docs/tutorials/finetuning.md — `salt merge-config` renders of the base config
# alone plus the four overlay stacks below — into docs/assets/finetuning/, never
# the .pdf siblings or merged_*.yaml (large and path-bearing, not doc assets).
#
# Single source of truth for
# salt/tests/unit/test_finetune_configs.py::TestFreezeGraphs, which reproduces
# the DOT from the shipped configs and byte-diffs it against the committed .dot
# files. Uses `python -m salt.main`, not the `salt` console script: a test
# container may carry an editable install of another checkout on PATH.
# Re-run after any change to docs/tutorials/configs/finetuning/*.yaml or the
# renderer (salt/graph/render.py, salt/cli.py's `_render_with_dot`).

set -euo pipefail

ASSETS_DIR="docs/assets/finetuning"
BASE="docs/tutorials/configs/finetuning/gn3large_base.yaml"

# ordered list of (stem overlay) pairs — overlay is empty for the base alone
STACKS=(
    "gn3large "
    "merged_calo finetune_gn3large_add_calo.yaml"
    "merged_charge finetune_gn3large_add_charge_head.yaml"
    "merged_jetvar finetune_gn3large_add_jet_vars.yaml"
    "merged_xbb finetune_gn3large_xbb_transfer.yaml"
)

usage() {
    cat >&2 <<'USAGE'
usage: finetuning_render_graphs.sh [--check]

  (no args)   regenerate the nine freeze graphs (base alone + four overlay
              stacks) and copy the .dot + .png pairs into
              docs/assets/finetuning/ (overwriting what is there)
  --check     regenerate into a scratch dir and verify the .dot files are
              byte-identical to the committed assets (and the .png siblings
              exist) — exits 1 on any mismatch, 0 otherwise

Must be run from the salt repo root.
USAGE
}

MODE="generate"
if [[ $# -eq 0 ]]; then
    MODE="generate"
elif [[ $# -eq 1 && "$1" == "--check" ]]; then
    MODE="check"
else
    usage
    exit 2
fi

if [[ ! -f "salt/main.py" || ! -f "docs/tutorials/finetuning.md" ]]; then
    echo "error: run this script from the salt repo root" \
        "(expected ./salt/main.py and ./docs/tutorials/finetuning.md to exist)" >&2
    exit 2
fi

if ! command -v dot >/dev/null 2>&1; then
    echo "error: the Graphviz 'dot' binary was not found on PATH — cannot" \
        "render the freeze-graph PNGs. Install Graphviz (the shipped salt" \
        "containers already include it, so hitting this there is" \
        "unexpected). See https://ftag-salt.docs.cern.ch/setup/#install-graphviz" >&2
    exit 3
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# into $WORK, never the repo: the merged YAML and .pdf siblings are scratch
for stack in "${STACKS[@]}"; do
    read -r stem overlay <<<"$stack"
    if [[ -z "$overlay" ]]; then
        python -m salt.main merge-config \
            --config "$BASE" \
            --merged.output "$WORK/${stem}.yaml"
    else
        python -m salt.main merge-config \
            --config "$BASE" \
            --config "docs/tutorials/configs/finetuning/${overlay}" \
            --merged.output "$WORK/${stem}.yaml"
    fi
done

# collect the per-stage .dot siblings merge-config wrote (glob, not hardcoded
# stage names, so a schedule rename is picked up automatically)
shopt -s nullglob
dot_files=("$WORK"/gn3large_stage*.dot "$WORK"/merged_calo_stage*.dot \
    "$WORK"/merged_charge_stage*.dot "$WORK"/merged_jetvar_stage*.dot \
    "$WORK"/merged_xbb_stage*.dot)
shopt -u nullglob

if [[ ${#dot_files[@]} -ne 9 ]]; then
    echo "error: expected 9 freeze-graph .dot files (1 for the base alone +" \
        "2 stages x 4 overlays), found ${#dot_files[@]} in $WORK:" \
        "${dot_files[*]-<none>}" >&2
    exit 4
fi

for dot_path in "${dot_files[@]}"; do
    png_path="${dot_path%.dot}.png"
    if [[ ! -f "$png_path" ]]; then
        echo "error: merge-config did not produce the expected PNG for" \
            "$dot_path (looked for $png_path)" >&2
        exit 4
    fi
done

if [[ "$MODE" == "generate" ]]; then
    mkdir -p "$ASSETS_DIR"
    copied=()
    for dot_path in "${dot_files[@]}"; do
        base="$(basename "$dot_path")"
        png_base="${base%.dot}.png"
        cp "$dot_path" "$ASSETS_DIR/$base"
        cp "${dot_path%.dot}.png" "$ASSETS_DIR/$png_base"
        copied+=("$ASSETS_DIR/$base" "$ASSETS_DIR/$png_base")
    done
    echo "wrote:"
    printf '  %s\n' "${copied[@]}"
    exit 0
fi

# --check: byte-diff each .dot against its committed asset and assert the .png
# sibling exists (PNG bytes are graphviz-version dependent, never compared)
mismatched=()
for dot_path in "${dot_files[@]}"; do
    base="$(basename "$dot_path")"
    committed="$ASSETS_DIR/$base"
    if [[ ! -f "$committed" ]]; then
        mismatched+=("$base (missing from $ASSETS_DIR)")
    elif ! cmp -s "$dot_path" "$committed"; then
        mismatched+=("$base")
    fi

    png_base="${base%.dot}.png"
    if [[ ! -f "$ASSETS_DIR/$png_base" ]]; then
        mismatched+=("$png_base (missing from $ASSETS_DIR)")
    fi
done

if [[ ${#mismatched[@]} -gt 0 ]]; then
    echo "freeze-graph check FAILED — mismatched/missing files:" >&2
    printf '  %s\n' "${mismatched[@]}" >&2
    exit 1
fi

echo "freeze-graph check OK — all 9 .dot files byte-identical, all 9 .png present"
exit 0
