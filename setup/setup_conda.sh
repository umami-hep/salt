#!/usr/bin/env bash
set -euo pipefail

# Local install prefix (change if you prefer)
CONDA_INSTALL="${PWD}/conda"

if [[ ! -d "$CONDA_INSTALL" ]]; then
  CONDA_REPO="https://github.com/conda-forge/miniforge/releases/latest/download"

  # Select installer
  case "${OSTYPE:-}" in
    darwin*)  # macOS
      case "$(uname -m)" in
        arm64)  CONDA_INSTALLER="Miniforge3-MacOSX-arm64.sh"   ;;
        x86_64) CONDA_INSTALLER="Miniforge3-MacOSX-x86_64.sh" ;;
        *) echo "Unsupported macOS arch: $(uname -m)"; exit 1 ;;
      esac
      ;;
    linux*)
      case "$(uname -m)" in
        x86_64)  CONDA_INSTALLER="Miniforge3-Linux-x86_64.sh"  ;;
        aarch64) CONDA_INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
        *) echo "Unsupported Linux arch: $(uname -m)"; exit 1  ;;
      esac
      ;;
    *)
      echo "Operating system not supported. Setup not possible."
      exit 1
      ;;
  esac

  # Download installer
  echo "Downloading ${CONDA_INSTALLER}…"
  curl -fsSL -o "${CONDA_INSTALLER}" "${CONDA_REPO}/${CONDA_INSTALLER}"

  # Install non-interactively to local prefix
  bash "${CONDA_INSTALLER}" -b -p "${CONDA_INSTALL}"

  # Cleanup installer
  rm -f "${CONDA_INSTALLER}"
fi

# Activate base environment
# (Miniforge ships mamba by default; conda activation works the same)
# Either of these works; keep one:
# shellcheck disable=SC1091
source "${CONDA_INSTALL}/bin/activate"
# Alternatively:
# source "${CONDA_INSTALL}/etc/profile.d/conda.sh" && conda activate
