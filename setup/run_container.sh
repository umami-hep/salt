#! /bin/bash

# `latest` is an alias for the V2H image (cu126, sm_50..sm_90: V100 through
# Hopper). On a Blackwell card (RTX 40xx/50xx, B100/B200) swap it for `salt:T2B`
# — otherwise every kernel launch fails with "no kernel image is available for
# execution on the device" at your first real operation, NOT at startup.
# Run ./setup/valid-container-versions if you are unsure which you need.
SALT_TAG="${SALT_TAG:-latest}"

singularity shell \
    -e --nv --bind $PWD --bind /nfs --bind /tmp \
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/salt:${SALT_TAG}"

