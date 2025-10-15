#!/usr/bin/env bash
set -euo pipefail

# Set the local path to which everything is installed
SALTDIR="$(dirname "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)")"
export SALTDIR

export PYTHONUSERBASE="${SALTDIR}/python_install"

# Safe expansions: only add colon + old value if it exists
export PYTHONPATH="${PYTHONUSERBASE}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PYTHONUSERBASE}/bin${PATH:+:${PATH}}"

# Setup the environment for salt
cd "${SALTDIR}/"

# Remove existing installation
rm -rf "${PYTHONUSERBASE}"
rm -rf "${SALTDIR}/"*.egg-info

# Create the new directory
mkdir -p "${PYTHONUSERBASE}"

# Install Salt (pip --user respects PYTHONUSERBASE)
python -m pip install --user -e '.[dev,muP,flash]'