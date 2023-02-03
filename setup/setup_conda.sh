#!/bin/bash

# borrowed from:
# https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/ftag-tracking-studies/-/blob/master/setup_conda.sh

# install mamba locally if it doesn't already exist
CONDA_INSTALL=$PWD/conda/

if [[ ! -d "${CONDA_INSTALL}" ]]; then
  CONDA_REPOSITORY=https://github.com/conda-forge/miniforge/releases/latest/download
  # installation for macOS (including support for M1 MacBooks)
  if [[ $OSTYPE == 'darwin'* ]]; then
    MAC_TYPE="$(uname -m)"
    if [[ $MAC_TYPE == 'arm64' ]]; then
      CONDA_INSTALLER="Mambaforge-MacOSX-arm64.sh"
    else
      CONDA_INSTALLER="Mambaforge-MacOSX-x86_64.sh"
    fi
  # installation for linux
  elif [[ $OSTYPE == 'linux'* ]]; then
    CONDA_INSTALLER="Mambaforge-Linux-x86_64.sh"
  # other operating system not supported
  else
    echo "Operating system not supported. Setup not possible."
    exit 1
  fi
  # install mamba to local directory
  curl -L -O ${CONDA_REPOSITORY}/${CONDA_INSTALLER}
  bash ${CONDA_INSTALLER} -b -p ${CONDA_INSTALL}
  rm ${CONDA_INSTALLER}
fi

# set up conda environment
source ${CONDA_INSTALL}/bin/activate
