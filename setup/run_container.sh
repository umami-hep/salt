#! /bin/bash

singularity shell \
    -e --nv --bind $PWD --bind /nfs --bind /tmp \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/salt:latest

