# shellcheck shell=bash
# setup/setup_lxplus.sh — set up salt on CERN lxplus, with the (large) CUDA venv
# on shared, batch-worker-visible storage, mirrored into a /tmp-cached copy for
# fast interactive activation.
#
#   SOURCE this on an lxplus login node (do NOT execute it):
#       source setup/setup_lxplus.sh
#       source setup/setup_lxplus.sh --reinstall   # force a full rebuild
#
# Re-sourcing just re-activates the environment (idempotent, and fast after the
# first run — see "Dual-copy model" below). It installs uv if missing, creates a
# Python 3.14 venv (salt's required interpreter), runs `uv sync`, adds setup/ to
# PATH (so `salt-lxplus-gpu` is available), and prints next-step instructions.
#
# Dual-copy model: the durable, batch-worker-visible venv lives at
# $SALT_LXPLUS_DIR/.venv (AFS work / EOS home / AFS home — see below); batch jobs
# (salt-lxplus-gpu) read it directly and are unaffected by anything here.
# Interactive logins instead activate a second, node-local copy at
# /tmp/<user>-salt-venv, built once and re-extracted from a tarball cache
# ($SALT_LXPLUS_DIR/.venv-cache.tar, keyed by a pyproject.toml hash) on every
# subsequent login — avoiding a slow rebuild/`uv sync` against EOS/AFS on each
# interactive shell.
#
# Why tar-moving a uv venv is safe: `.venv/bin/python` is a symlink to the
# absolute, shared $UV_PYTHON_INSTALL_DIR interpreter on durable storage, so it
# resolves correctly regardless of where the venv directory itself is relocated.
# The `.venv/bin/*` console-script shebangs (e.g. a bare `salt`) go stale after a
# move — the supported invocation `python -m salt.main` never uses them, so do
# NOT try to "fix" the shebangs. `.venv/bin/activate` also hardcodes an absolute
# VIRTUAL_ENV by default, which would otherwise point at the wrong location
# (the /tmp build dir, from the durable copy's perspective) after the tar
# move/extract — this is why the venv is created with `uv venv --relocatable`:
# the generated activate then derives VIRTUAL_ENV from its own path at source
# time, so both the /tmp copy and the durable copy activate correctly regardless
# of where the tarball was extracted.
#
# From the /tmp-cached copy, always prefer `python -m salt.main` over the bare
# `salt` command (see above).
#
# Storage: the venv holds the CUDA torch wheels (~3–5 GB), which do NOT fit in
# the 10 GB AFS home quota. The script picks a location in this order:
#   1. $SALT_LXPLUS_DIR                  (your explicit override — always wins)
#   2. /afs/cern.ch/work/<i>/<user>      (AFS workspace, ~100 GB — best latency)
#   3. /eos/user/<i>/<user>             (EOS home — works, FUSE is slower)
#   4. AFS home                          (only if `fs listquota` shows >=8 GB free)
#   5. otherwise: fail with guidance
# $SALT_LXPLUS_DIR itself never uses /tmp (node-local — invisible to the batch
# worker your job lands on); the /tmp mirror above is a separate, deliberate
# interactive-only fast path, not a relocation of the durable venv.
#
# No AFS workspace? Request one (free, ~100 GB) at the CERN Resources Portal
# (https://resources.web.cern.ch → Services → AFS Workspaces); it is the best
# experience. EOS home works out of the box in the meantime.

# Guard: must be sourced, not executed (we export env + activate a venv).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, do not run it:  source setup/setup_lxplus.sh" >&2
    exit 1
fi

# /tmp is shared across every user on the node. Before trusting or reusing any
# path under it, verify it is owned by the current UID and not group/world
# writable — refuse rather than silently proceed on untrusted shared state
# (same posture as enforce-inner-clone-protection.sh elsewhere in this repo).
# Returns: 0 = exists and safely owned, 1 = exists but untrusted, 2 = absent.
_salt_lxplus_tmp_owned() {
    local path="$1" owner perms group_digit other_digit
    [[ -e "$path" || -L "$path" ]] || return 2
    read -r owner perms < <(stat -c '%u %a' "$path" 2>/dev/null) || return 1
    [[ -n "$owner" && "$owner" == "$(id -u)" ]] || return 1
    group_digit="${perms: -2:1}"
    other_digit="${perms: -1}"
    (( (group_digit & 2) == 0 )) || return 1
    (( (other_digit & 2) == 0 )) || return 1
    return 0
}

_salt_lxplus_setup() {
    local reinstall=0
    case "${1:-}" in
        "") ;;
        --reinstall) reinstall=1 ;;
        *)
            echo "ERROR: usage: source setup/setup_lxplus.sh [--reinstall]" >&2
            return 1
            ;;
    esac

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

    # --- /tmp-cached fast path (interactive only) ---
    local pyproject="$repo_root/pyproject.toml"
    [[ -f "$pyproject" ]] || { echo "ERROR: cannot find $pyproject" >&2; return 1; }
    local current_hash
    current_hash="$(sha256sum "$pyproject" | awk '{print $1}')"

    local local_venv="/tmp/${user}-salt-venv"
    local local_hash_file="${local_venv}.hash"

    if [[ $reinstall -eq 1 ]]; then
        _salt_lxplus_tmp_owned "$local_venv"
        local reinstall_venv_rc=$?
        if [[ $reinstall_venv_rc -eq 1 ]]; then
            echo "ERROR: --reinstall refused: $local_venv exists but is not safely owned" >&2
            echo "       (must be owned by $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
            return 1
        fi
        _salt_lxplus_tmp_owned "$local_hash_file"
        local reinstall_hash_rc=$?
        if [[ $reinstall_hash_rc -eq 1 ]]; then
            echo "ERROR: --reinstall refused: $local_hash_file exists but is not safely owned" >&2
            echo "       (must be owned by $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
            return 1
        fi
        echo "NOTE: --reinstall requested — wiping cached venvs and rebuilding from scratch." >&2
        rm -rf "$local_venv" "$local_hash_file"
        rm -f "$SALT_LXPLUS_DIR/.venv-cache.tar" "$SALT_LXPLUS_DIR/.venv-cache.hash"
    fi

    local tier="" local_hash=""
    if [[ $reinstall -eq 0 ]]; then
        _salt_lxplus_tmp_owned "$local_venv"
        local venv_rc=$?
        if [[ $venv_rc -eq 1 ]]; then
            echo "ERROR: $local_venv exists but is not safely owned (must be owned by" >&2
            echo "       $(id -un) and not group/world-writable). Someone/something else may own it," >&2
            echo "       or it may be a hostile pre-created path. Investigate before re-sourcing." >&2
            return 1
        fi
        if [[ $venv_rc -eq 0 ]]; then
            _salt_lxplus_tmp_owned "$local_hash_file"
            local hash_rc=$?
            if [[ $hash_rc -eq 1 ]]; then
                echo "ERROR: $local_hash_file exists but is not safely owned (must be owned by" >&2
                echo "       $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
                return 1
            elif [[ $hash_rc -eq 0 ]]; then
                local_hash="$(<"$local_hash_file")"
            fi
            if [[ "$local_hash" == "$current_hash" && -x "$local_venv/bin/python" ]] \
                && "$local_venv/bin/python" -c "import salt.main" 2>/dev/null; then
                tier="warmest"
            fi
        fi
    fi

    if [[ -z "$tier" && $reinstall -eq 0 ]]; then
        local durable_hash=""
        [[ -f "$SALT_LXPLUS_DIR/.venv-cache.hash" ]] && durable_hash="$(<"$SALT_LXPLUS_DIR/.venv-cache.hash")"
        if [[ "$durable_hash" == "$current_hash" && -f "$SALT_LXPLUS_DIR/.venv-cache.tar" ]]; then
            tier="warm"
        fi
    fi

    if [[ "$tier" == "warmest" ]]; then
        # shellcheck disable=SC1091
        source "$local_venv/bin/activate" || return 1
        export PATH="$setup_dir:$PATH"
        echo "salt env WARMEST — /tmp copy already installed and current, re-activated ($local_venv)."
        echo "Skipped the full install entirely (no uv sync this run)."
        echo "Verify: python -m salt.main --help   (prefer this over bare 'salt' from the /tmp copy)"
        return 0
    fi

    if [[ "$tier" == "warm" ]]; then
        echo "salt env WARM — restoring /tmp copy from durable tarball cache (no full install needed)..."
        rm -rf "$local_venv"
        local tmp_tar="/tmp/${user}-salt-venv-cache.$$.tar"
        _salt_lxplus_tmp_owned "$tmp_tar"
        local tmp_tar_rc=$?
        if [[ $tmp_tar_rc -eq 1 ]]; then
            echo "ERROR: $tmp_tar exists but is not safely owned (must be owned by" >&2
            echo "       $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
            return 1
        fi
        cp "$SALT_LXPLUS_DIR/.venv-cache.tar" "$tmp_tar" \
            || { echo "ERROR: failed to copy venv cache tarball to /tmp" >&2; rm -f "$tmp_tar"; return 1; }
        mkdir -p "$local_venv" || { echo "ERROR: cannot create $local_venv" >&2; rm -f "$tmp_tar"; return 1; }
        tar -C "$local_venv" -xf "$tmp_tar" \
            || { echo "ERROR: failed to extract venv cache tarball" >&2; rm -f "$tmp_tar"; return 1; }
        rm -f "$tmp_tar"
        _salt_lxplus_tmp_owned "$local_hash_file"
        local warm_hash_rc=$?
        if [[ $warm_hash_rc -eq 1 ]]; then
            echo "ERROR: $local_hash_file exists but is not safely owned (must be owned by" >&2
            echo "       $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
            return 1
        fi
        echo "$current_hash" > "$local_hash_file"
        # shellcheck disable=SC1091
        source "$local_venv/bin/activate" || return 1
        export PATH="$setup_dir:$PATH"
        echo "salt env WARM — restored from durable cache, re-activated ($local_venv)."
        echo "Verify: python -m salt.main --help   (prefer this over bare 'salt' from the /tmp copy)"
        return 0
    fi

    # --- COLD: no matching cache anywhere, pyproject.toml changed, or --reinstall ---

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

    # --- create the venv directly at the /tmp copy (Python 3.14 — salt's requires-python) ---
    rm -rf "$local_venv"
    echo "Creating venv at $local_venv (Python 3.14)..."
    uv venv --relocatable --python 3.14 "$local_venv" || return 1

    # py-lap-solver's cp314 sdist-build workaround now lives in pyproject.toml
    # ([tool.uv] build-constraint-dependencies) — nothing extra needed here.

    # --- install salt + deps into the /tmp venv (CUDA wheels install fine on the
    #     login node even without a GPU; the GPU only matters at run time) ---
    echo "Installing salt and dependencies with 'uv sync' (this can take a while)..."
    ( cd "$repo_root" && UV_PROJECT_ENVIRONMENT="$local_venv" uv sync ) || return 1

    # --- publish to durable storage, atomically, hash-last (readers trust the
    #     hash file as the commit point — never publish it before the tarball). ---
    echo "Publishing venv cache to durable storage ($SALT_LXPLUS_DIR)..."
    local tar_tmp="$SALT_LXPLUS_DIR/.venv-cache.tar.$$.tmp"
    tar -C "$local_venv" -cf "$tar_tmp" . \
        || { echo "ERROR: failed to tar venv for durable cache" >&2; rm -f "$tar_tmp"; return 1; }
    mv "$tar_tmp" "$SALT_LXPLUS_DIR/.venv-cache.tar" \
        || { echo "ERROR: failed to publish venv cache tarball" >&2; rm -f "$tar_tmp"; return 1; }

    local hash_tmp="$SALT_LXPLUS_DIR/.venv-cache.hash.$$.tmp"
    echo "$current_hash" > "$hash_tmp" \
        || { echo "ERROR: failed to write venv cache hash" >&2; rm -f "$hash_tmp"; return 1; }
    mv "$hash_tmp" "$SALT_LXPLUS_DIR/.venv-cache.hash" \
        || { echo "ERROR: failed to publish venv cache hash" >&2; rm -f "$hash_tmp"; return 1; }

    # Refresh the durable, batch-visible venv from the same tarball (slow: many
    # small writes on EOS/AFS — same cost class as today's uv sync, but now only
    # on a cache miss/rebuild, not on every login). Swap via rename, never an
    # in-place overwrite of individual files.
    local durable_new="$SALT_LXPLUS_DIR/.venv.new.$$"
    local durable_old="$SALT_LXPLUS_DIR/.venv.old.$$"
    mkdir -p "$durable_new" || { echo "ERROR: failed to create $durable_new" >&2; return 1; }
    tar -C "$durable_new" -xf "$SALT_LXPLUS_DIR/.venv-cache.tar" \
        || { echo "ERROR: failed to extract venv cache into $durable_new" >&2; rm -rf "$durable_new"; return 1; }
    if [[ -e "$SALT_LXPLUS_DIR/.venv" ]]; then
        mv "$SALT_LXPLUS_DIR/.venv" "$durable_old" \
            || { echo "ERROR: failed to move aside existing $SALT_LXPLUS_DIR/.venv" >&2; rm -rf "$durable_new"; return 1; }
    fi
    mv "$durable_new" "$SALT_LXPLUS_DIR/.venv" \
        || { echo "ERROR: failed to swap in refreshed $SALT_LXPLUS_DIR/.venv" >&2; return 1; }
    rm -rf "$durable_old"

    _salt_lxplus_tmp_owned "$local_hash_file"
    local cold_hash_rc=$?
    if [[ $cold_hash_rc -eq 1 ]]; then
        echo "ERROR: $local_hash_file exists but is not safely owned (must be owned by" >&2
        echo "       $(id -un) and not group/world-writable). Investigate before re-sourcing." >&2
        return 1
    fi
    echo "$current_hash" > "$local_hash_file"

    # shellcheck disable=SC1091
    source "$local_venv/bin/activate" || return 1
    export PATH="$setup_dir:$PATH"

    cat <<EOF

==================================================================
 salt lxplus environment ready — tier: COLD (full install ran).
   repo             = $SALT_REPO_DIR
   SALT_LXPLUS_DIR  = $SALT_LXPLUS_DIR   (durable, batch-worker-visible)
   active venv      = $local_venv   (activated, /tmp-cached)
   durable venv     = $SALT_LXPLUS_DIR/.venv   (refreshed for batch jobs)
   cache tarball    = $SALT_LXPLUS_DIR/.venv-cache.tar

 Verify:
   python -m salt.main --help
   (prefer this over the bare 'salt' command when running from the /tmp copy —
    its console-script shebangs go stale across cache moves; 'python -m
    salt.main' never depends on them.)

 Re-source any time to re-activate (fast after this first run):
   source setup/setup_lxplus.sh
 Force a full rebuild (e.g. after a dependency bump the hash didn't catch):
   source setup/setup_lxplus.sh --reinstall

 GPU access — CERN HTCondor batch farm (the sanctioned GPU route):
   salt-lxplus-gpu shell [flavour]           interactive GPU node
   salt-lxplus-gpu submit <config> [flavour] batch GPU training
   salt-lxplus-gpu status                    your condor jobs
   (This venv/SIF live on EOS, but the submit-file artifacts — the job
    executable + logs — go to AFS home: standard schedds reject /eos paths
    inside the submit file. salt-lxplus-gpu handles that split for you.)
==================================================================
EOF
    return 0
}

_salt_lxplus_setup "$@"
unset -f _salt_lxplus_setup _salt_lxplus_tmp_owned
