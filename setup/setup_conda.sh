# shellcheck shell=bash
# setup/setup_conda.sh — set up salt in a repo-local conda (miniforge) env.
#
#   SOURCE this (do NOT execute it):
#       source setup/setup_conda.sh
#
# Bootstraps miniforge into a repo-local `conda/` prefix if missing, creates a
# `salt` env with Python 3.14 (salt's `requires-python`) and installs salt into
# it. Idempotent — re-sourcing just re-activates.
#
# Generic: no lxplus/CERN storage logic. See setup/setup_lxplus.sh for the CERN
# batch-farm variant (AFS/EOS quotas, HTCondor worker visibility) and
# setup/setup_uv.sh for the uv equivalent.

# Must be sourced, not executed — we activate a conda env in the caller's shell.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, do not run it:  source setup/setup_conda.sh" >&2
    exit 1
fi

_salt_conda_setup() {
    local setup_dir repo_root
    setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    repo_root="$(dirname -- "$setup_dir")"

    # Prefix anchored to the repo, not $PWD — sourcing from anywhere must work.
    local conda_install="${repo_root}/conda"

    # --- bootstrap miniforge into $conda_install if missing ---
    if [[ ! -d "$conda_install" ]]; then
        local conda_repo="https://github.com/conda-forge/miniforge/releases/latest/download"
        local installer

        case "${OSTYPE:-}" in
            darwin*)
                case "$(uname -m)" in
                    arm64)  installer="Miniforge3-MacOSX-arm64.sh"  ;;
                    x86_64) installer="Miniforge3-MacOSX-x86_64.sh" ;;
                    *) echo "ERROR: unsupported macOS arch: $(uname -m)" >&2; return 1 ;;
                esac
                ;;
            linux*)
                case "$(uname -m)" in
                    x86_64)  installer="Miniforge3-Linux-x86_64.sh"  ;;
                    aarch64) installer="Miniforge3-Linux-aarch64.sh" ;;
                    *) echo "ERROR: unsupported Linux arch: $(uname -m)" >&2; return 1 ;;
                esac
                ;;
            *)
                echo "ERROR: operating system not supported (OSTYPE=${OSTYPE:-unknown})." >&2
                return 1
                ;;
        esac

        local installer_path="/tmp/${installer}"
        echo "Downloading ${installer}..."
        curl -fsSL -o "$installer_path" "${conda_repo}/${installer}" || {
            echo "ERROR: failed to download ${installer}" >&2
            return 1
        }

        echo "Installing miniforge to ${conda_install}..."
        bash "$installer_path" -b -p "$conda_install" || {
            echo "ERROR: miniforge installer failed" >&2
            rm -f "$installer_path"
            return 1
        }
        rm -f "$installer_path"
    fi

    # shellcheck disable=SC1091
    source "${conda_install}/etc/profile.d/conda.sh" || return 1

    # --- fast path: env 'salt' already installed and importable -> just activate ---
    local env_python="${conda_install}/envs/salt/bin/python"
    if [[ -x "$env_python" ]] && "$env_python" -c "import salt.main" 2>/dev/null; then
        conda activate salt || return 1
        echo "salt env already installed — re-activated (${conda_install}/envs/salt)."
        echo "Verify: python -m salt.main --help"
        return 0
    fi

    # --- create (or reuse) the 'salt' env with Python 3.14 ---
    # 3.14 is required, not a preference: salt pins `requires-python =
    # ">=3.14,<3.15"`; conda-forge ships it (verified 3.14.4-3.14.7, linux-64).
    if [[ ! -d "${conda_install}/envs/salt" ]]; then
        echo "Creating conda env 'salt' (Python 3.14)..."
        conda create -y -n salt python=3.14 || return 1
    fi
    conda activate salt || return 1

    # --- install salt + deps into the env ---
    # py-lap-solver 0.1.4 has no cp314 wheel, so pip builds its sdist — whose
    # metadata uses `cmake.minimum-version`, rejected by scikit-build-core>=0.8.
    # PIP_CONSTRAINT does NOT fix this: verified 2026-08-31 (pip 26.2.1) that it
    # never reaches pip's isolated build env — the build still pulled >=0.8.
    # Instead, per setup/Dockerfile (~line 137): pre-install scikit-build-core<0.8
    # plus cmake/ninja/pyproject-metadata/pathspec/pybind11[global] (needed
    # because --no-build-isolation skips pip's per-build env bootstrap), then
    # install py-lap-solver with --no-build-isolation. The `pip install -e .`
    # below then finds it satisfied and never touches the sdist again.
    echo "Installing py-lap-solver's build-time deps (scikit-build-core<0.8 + friends)..."
    python -m pip install 'scikit-build-core<0.8' cmake ninja pyproject-metadata pathspec 'pybind11[global]' || return 1

    echo "Building py-lap-solver from sdist (--no-build-isolation; no cp314 wheel exists)..."
    python -m pip install --no-build-isolation 'py-lap-solver>=0.1.4' || return 1

    echo "Installing salt (editable)..."
    ( cd "$repo_root" && python -m pip install -e . ) || return 1

    cat <<EOF

==================================================================
 salt conda environment ready.
   repo            = ${repo_root}
   conda prefix    = ${conda_install}
   env             = salt (Python 3.14, activated)

 Verify:
   python -m salt.main --help

 Re-source any time to re-activate:
   source setup/setup_conda.sh
==================================================================
EOF
    return 0
}

_salt_conda_setup
unset -f _salt_conda_setup
