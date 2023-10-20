# Introduction

Welcome to the Salt framework!

Salt is a general-purpose framework to train **multi-modal**, **multi-task** models.
It was developed for state-of-the art jet flavour tagging algorithms such as GN1 and GN2, but can be applied much more widely.

!!! example "The code is hosted on the CERN GitLab: [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt)"

### Features

- Fully based on [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
- Support for multiple YAML-configurable [input modalities][salt.models.InitNet] and [output tasks][task-heads].
- ONNX export support to use trained models in Athena.
- Easily extensible: you can implement your own custom models.
- Well documented and CI tested.

### Getting Started

Below are some helpful links to get you started:

!!! info "You can find out more about flavour tagging algorithms at the [FTAG docs](https://ftag.docs.cern.ch/)"

!!! question "There is a [channel](https://mattermost.web.cern.ch/aft-algs/channels/gnns) for the framework in the [FTAG Mattermost workspace](https://mattermost.web.cern.ch/signup_user_complete/?id=1wicts5csjd49kg7uymwwt9aho&md=link&sbr=su)"

!!! abstract "A tutorial on how to use Salt is provided at the [FTAG docs page](https://ftag.docs.cern.ch/software/tutorials/tutorial-salt/)"

!!! tip "An API reference for the framework is available [here](api/data)"

!!! note "Salt is based on work from two existing frameworks [[1]](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/GNNJetTagger) [[2]](https://gitlab.cern.ch/mleigh/flavour_tagging/)"


