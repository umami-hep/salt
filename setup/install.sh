#!/usr/bin/env bash
set -euo pipefail

# Set the local path to which everything is installed
SALTDIR="$(dirname "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)")"
export SALTDIR

export PYTHONUSERBASE="${SALTDIR}/python_install"

# Safe expansions: only add colon + old value if it exists
PYTHON_VERSION="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="${PYTHONUSERBASE}/lib/${PYTHON_VERSION}/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PYTHONUSERBASE}/bin${PATH:+:${PATH}}"

# Setup the environment for salt
cd "${SALTDIR}/"

# Remove existing installation
rm -rf "${PYTHONUSERBASE}"
rm -rf "${SALTDIR}/"*.egg-info

# Create the new directory
mkdir -p "${PYTHONUSERBASE}"

# Bootstrap uv into PYTHONUSERBASE, then install Salt into the same project environment.
python -m pip install --user uv
UV_PROJECT_ENVIRONMENT="${PYTHONUSERBASE}" uv sync --python "$(command -v python)" --group dev --extra muP --extra flash
