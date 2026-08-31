# shellcheck shell=bash
# setup/setup_uv.sh — generic (non-lxplus) uv-based local install of salt.
#
#   SOURCE this (do NOT execute it):
#       source setup/setup_uv.sh
#
# Installs uv if missing, runs `uv sync` in the repo root, activates the default
# `.venv`. Idempotent — re-sourcing just re-activates.
#
# Storage: everything stays inside the repo (uv under setup/.uv-bin/, venv at
# the repo-root `.venv/`). Deliberately unlike setup/setup_lxplus.sh, which uses
# ~/.local/bin plus an AFS-workspace/EOS venv because lxplus needs shared,
# batch-worker-visible storage; a laptop/workstation checkout does not, and the
# user asked for a fully self-contained install ("install it inside salt").
#
# The py-lap-solver cp314 sdist-build fix (scikit-build-core<0.8) lives in
# pyproject.toml's [tool.uv] build-constraint-dependencies — plain `uv sync`
# picks it up automatically.

# Guard: must be sourced, not executed (we activate a venv in the caller's shell).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, do not run it:  source setup/setup_uv.sh" >&2
    exit 1
fi

_salt_uv_setup() {
    local setup_dir repo_root
    setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    repo_root="$(dirname -- "$setup_dir")"

    # --- fast path: already installed -> just re-activate ---
    if [[ -x "$repo_root/.venv/bin/python" ]] \
        && "$repo_root/.venv/bin/python" -c "import salt.main" 2>/dev/null; then
        # shellcheck disable=SC1091
        source "$repo_root/.venv/bin/activate" || return 1
        echo "salt env already installed — re-activated ($repo_root/.venv)."
        echo "Verify: python -m salt.main --help"
        return 0
    fi

    # --- install uv if missing, SCOPED INSIDE THE REPO (not ~/.local/bin) ---
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv (repo-local, $setup_dir/.uv-bin)..."
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$setup_dir/.uv-bin" UV_NO_MODIFY_PATH=1 sh \
            || { echo "ERROR: uv install failed" >&2; return 1; }
        export PATH="$setup_dir/.uv-bin:$PATH"
        hash -r
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not on PATH after install (expected in $setup_dir/.uv-bin)." >&2
        return 1
    fi

    # --- install salt + deps into the repo-root default .venv ---
    echo "Installing salt and dependencies with 'uv sync' (this can take a while)..."
    ( cd "$repo_root" && uv sync ) || return 1

    # shellcheck disable=SC1091
    source "$repo_root/.venv/bin/activate" || return 1

    cat <<EOF

==================================================================
 salt environment ready.
   repo = $repo_root
   venv = $repo_root/.venv  (activated)

 Verify:
   python -m salt.main --help

 Re-source any time to re-activate:
   source setup/setup_uv.sh
==================================================================
EOF
    return 0
}

_salt_uv_setup
unset -f _salt_uv_setup
