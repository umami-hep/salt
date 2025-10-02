#!/bin/bash
# Set the local path to which everything is installed
export SALTDIR="$(dirname "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)")"
export PYTHONUSERBASE=${SETUP_DIR}/python_install
export PYTHONPATH=${PYTHONUSERBASE}:${PYTHONPATH}
export PATH=${PYTHONUSERBASE}/bin:${PATH}

# Setup the envirement for salt
cd ${SALTDIR}/

# Remove existing installation
rm -rf ${PYTHONUSERBASE}
rm -rf ${SALTDIR}/*.egg-info

# Create the new directory
mkdir -p ${PYTHONUSERBASE}

# Install Salt
python -m pip install --user -e .[dev,test,muP,flash]