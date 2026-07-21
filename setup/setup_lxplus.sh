# shellcheck shell=bash
# setup/setup_lxplus.sh — set up salt on CERN lxplus, with the (large) CUDA venv
# on shared, batch-worker-visible storage.
#
#   SOURCE this on an lxplus login node (do NOT execute it):
#       source setup/setup_lxplus.sh
#
# Re-sourcing just re-activates the environment (idempotent). It installs uv if
# missing, creates a Python 3.14 venv (salt's required interpreter), runs
# `uv sync`, adds setup/ to PATH (so `salt-lxplus-gpu` is available), and prints
# next-step instructions.
#
# Storage: the venv holds the CUDA torch wheels (~3–5 GB), which do NOT fit in
# the 10 GB AFS home quota. The script picks a location in this order:
#   1. $SALT_LXPLUS_DIR                  (your explicit override — always wins)
#   2. /afs/cern.ch/work/<i>/<user>      (AFS workspace, ~100 GB — best latency)
#   3. /eos/user/<i>/<user>             (EOS home — works, FUSE is slower)
#   4. AFS home                          (only if `fs listquota` shows >=8 GB free)
#   5. otherwise: fail with guidance
# It never uses /tmp (node-local — invisible to the batch worker your job lands on).
#
# No AFS workspace? Request one (free, ~100 GB) at the CERN Resources Portal
# (https://resources.web.cern.ch → Services → AFS Workspaces); it is the best
# experience. EOS home works out of the box in the meantime.

# Guard: must be sourced, not executed (we export env + activate a venv).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, do not run it:  source setup/setup_lxplus.sh" >&2
    exit 1
fi

_salt_lxplus_setup() {
    local setup_dir repo_root user initial
    setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    repo_root="$(dirname -- "$setup_dir")"
    user="${USER:-$(id -un)}"
    initial="${user:0:1}"

    # --- sanity: are we actually on lxplus / a CERN node? ---
    if [[ "$(hostname -f 2>/dev/null)" != *.cern.ch ]]; then
        echo "WARNING: this host does not look like lxplus (hostname: $(hostname -f 2>/dev/null))." >&2
        echo "         setup_lxplus.sh targets CERN lxplus; continuing anyway." >&2
    fi

    # --- choose the storage root that will hold .venv ---
    local afs_work="/afs/cern.ch/work/${initial}/${user}"
    local eos_home="/eos/user/${initial}/${user}"
    local venv_home=""
    if [[ -n "${SALT_LXPLUS_DIR:-}" ]]; then
        venv_home="$SALT_LXPLUS_DIR"
    elif [[ -d "$afs_work" ]]; then
        venv_home="$afs_work/salt"
    elif [[ -d "$eos_home" ]]; then
        venv_home="$eos_home/salt"
        echo "NOTE: using EOS home ($venv_home). EOS is FUSE-mounted, so venv creation" >&2
        echo "      and 'uv sync' are slower than on an AFS workspace, but functional." >&2
        echo "      For best performance request an AFS workspace at the CERN Resources Portal." >&2
    else
        # AFS home last resort — only if it genuinely has room for the CUDA wheels.
        local free_kb free_gb
        free_kb="$(fs listquota "$HOME" 2>/dev/null | awk 'NR==2 {print $2-$3}')"
        free_gb=$(( ${free_kb:-0} / 1024 / 1024 ))
        if [[ "${free_kb:-0}" -gt 0 && "$free_gb" -ge 8 ]]; then
            venv_home="$HOME/salt-venv"
            echo "NOTE: no AFS workspace or EOS home found; using AFS home ($venv_home," >&2
            echo "      ~${free_gb} GB free). This works but AFS home quota is tight." >&2
        else
            echo "ERROR: nowhere with enough space for the CUDA torch venv (~5 GB)." >&2
            echo "       - No AFS workspace at $afs_work" >&2
            echo "       - No EOS home at $eos_home" >&2
            echo "       - AFS home has only ~${free_gb} GB free" >&2
            echo "       Fix one of:" >&2
            echo "         * Request an AFS workspace (free, ~100 GB) at the CERN Resources Portal," >&2
            echo "           then clone salt under $afs_work and re-source." >&2
            echo "         * Or set SALT_LXPLUS_DIR to a directory with >=8 GB free and re-source." >&2
            return 1
        fi
    fi

    # Never node-local /tmp — a batch worker cannot see it.
    case "$venv_home" in
        /tmp/*|/tmp)
            echo "ERROR: SALT_LXPLUS_DIR is under /tmp, which is node-local and invisible to" >&2
            echo "       HTCondor batch workers. Use AFS workspace or EOS instead." >&2
            return 1
            ;;
    esac

    mkdir -p "$venv_home" || { echo "ERROR: cannot create $venv_home" >&2; return 1; }
    export SALT_LXPLUS_DIR="$venv_home"
    export SALT_REPO_DIR="$repo_root"

    # The uv-managed interpreter must sit on the shared FS next to the venv, so a
    # batch worker can resolve the venv's python symlink into it.
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SALT_LXPLUS_DIR/.uv-python}"
    case "$SALT_LXPLUS_DIR" in
        /eos/*)
            # EOS FUSE has unreliable POSIX file locking, which makes uv's cache
            # (many small-file writes + lockfiles) both slow and prone to lock
            # timeouts. Put the *throwaway* cache on node-local /tmp (real locks,
            # fast); the venv itself stays on EOS. Cache and venv are then on
            # different filesystems, so hardlinks are impossible → force copy mode.
            export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${user}-uv-cache}"
            export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
            ;;
        *)
            # Keep uv's heavy cache off the tiny AFS home; same FS as the venv.
            export UV_CACHE_DIR="${UV_CACHE_DIR:-$SALT_LXPLUS_DIR/.uv-cache}"
            ;;
    esac

    # --- fast path: already installed -> just re-activate ---
    if [[ -x "$SALT_LXPLUS_DIR/.venv/bin/python" ]] \
        && "$SALT_LXPLUS_DIR/.venv/bin/python" -c "import salt.main" 2>/dev/null; then
        # shellcheck disable=SC1091
        source "$SALT_LXPLUS_DIR/.venv/bin/activate" || return 1
        export PATH="$setup_dir:$PATH"
        echo "salt env already installed — re-activated ($SALT_LXPLUS_DIR/.venv)."
        echo "Verify: python -m salt.main --help"
        return 0
    fi

    # --- install uv if missing ---
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "ERROR: uv install failed" >&2; return 1; }
        export PATH="$HOME/.local/bin:$PATH"
        hash -r
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not on PATH after install (expected in ~/.local/bin)." >&2
        return 1
    fi

    # --- create the venv (Python 3.14 — salt's requires-python) ---
    if [[ ! -d "$SALT_LXPLUS_DIR/.venv" ]]; then
        echo "Creating venv at $SALT_LXPLUS_DIR/.venv (Python 3.14)..."
        uv venv --python 3.14 "$SALT_LXPLUS_DIR/.venv" || return 1
    fi

    # py-lap-solver 0.1.4 (a salt dependency) ships wheels only up to cp312, but
    # salt pins Python 3.14, so it is built from sdist — and that build fails with
    # scikit-build-core >= 0.8 ("Use cmake.version instead of cmake.minimum-version").
    # Constrain the build backend until py-lap-solver ships cp314 wheels or fixed
    # metadata (then this file + UV_BUILD_CONSTRAINT can be removed).
    local bc_file="$SALT_LXPLUS_DIR/.salt-build-constraints.txt"
    printf 'scikit-build-core<0.8\n' > "$bc_file"
    export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$bc_file}"

    # --- install salt + deps into that venv (CUDA wheels install fine on the
    #     login node even without a GPU; the GPU only matters at run time) ---
    echo "Installing salt and dependencies with 'uv sync' (this can take a while)..."
    ( cd "$repo_root" && UV_PROJECT_ENVIRONMENT="$SALT_LXPLUS_DIR/.venv" uv sync ) || return 1

    # shellcheck disable=SC1091
    source "$SALT_LXPLUS_DIR/.venv/bin/activate" || return 1
    export PATH="$setup_dir:$PATH"

    cat <<EOF

==================================================================
 salt lxplus environment ready.
   repo            = $SALT_REPO_DIR
   SALT_LXPLUS_DIR = $SALT_LXPLUS_DIR
   venv            = $SALT_LXPLUS_DIR/.venv  (activated)

 Verify:
   python -m salt.main --help

 GPU access — CERN HTCondor batch farm (the sanctioned GPU route):
   salt-lxplus-gpu shell [flavour]           interactive GPU node
   salt-lxplus-gpu submit <config> [flavour] batch GPU training
   salt-lxplus-gpu status                    your condor jobs
   (This venv/SIF live on EOS, but the submit-file artifacts — the job
    executable + logs — go to AFS home: standard schedds reject /eos paths
    inside the submit file. salt-lxplus-gpu handles that split for you.)

 Re-source any time to re-activate:
   source setup/setup_lxplus.sh
==================================================================
EOF
    return 0
}

_salt_lxplus_setup
unset -f _salt_lxplus_setup
