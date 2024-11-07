# Introduction

Welcome to the Salt framework!

Salt is a general-purpose framework to train **multi-modal**, **multi-task** models.
It was developed for state-of-the art jet flavour tagging algorithms such as [GN1](https://ftag-docs.docs.cern.ch/algorithms/taggers/GN1/) and [GN2](https://ftag-docs.docs.cern.ch/algorithms/taggers/GN2/), but can be applied much more widely.
For example, you could use Salt to classify or regress properties of objects or events, or all these things at once!

!!! example "The code is hosted on the CERN GitLab: [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt)"

### Features

- Built on [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
- Support for multiple YAML-configurable [input modalities][salt.models.InitNet] and [output tasks][task-heads].
- ONNX export support to use trained models in [Athena](https://gitlab.cern.ch/atlas/athena/).
- Easily extensible: you can implement your own custom dataloaders and models.
- Documented and tested.

### Getting Started

Below are some helpful links to get you started:

!!! info "You can find out more about flavour tagging algorithms at the [FTAG docs](https://ftag.docs.cern.ch/)"

!!! question "There is a [channel](https://mattermost.web.cern.ch/aft-algs/channels/gnns) for the framework in the [FTAG Mattermost workspace](https://mattermost.web.cern.ch/signup_user_complete/?id=1wicts5csjd49kg7uymwwt9aho&md=link&sbr=su)"

!!! abstract "A tutorial on how to use Salt can be found [here](tutorial.md) and [here](tutorial-Xbb.md)"

!!! note "[Contributions](contributing) are welcome! Check out [existing issues](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/issues) for inspiration, or open your own"

!!! tip "You can become a Salt expert by checking out the [API reference](api/data)"

### Current Usage

Salt is currently used for the following projects:

- [Jet flavour tagging](https://ftag-docs.docs.cern.ch/algorithms/taggers/GN2/)
- [Boosted $X \rightarrow bb$ tagging](https://cds.cern.ch/record/2866601)
- [Tau ID](https://indico.cern.ch/event/1280531/timetable/?view=standard#153-new-identification-and-tag)
- [b-jet energy calibration](https://indico.cern.ch/event/1280531/timetable/?view=standard#131-b-jet-regression-effort-su)
- [Primary vertexing](https://indico.cern.ch/event/1311519/timetable/?view=standard#25-leveraging-the-ftag-softwar)
- [LLP vertexing](https://indico.cern.ch/event/1311519/timetable/?view=standard#25-leveraging-the-ftag-softwar)
- [Prompt lepton veto](https://indico.cern.ch/event/1341494/contributions/5647850/attachments/2742923/4771964/PLIV_ftagTPLT_20231030_pgadow.pdf)
- [multitop analysis](https://indico.cern.ch/event/1344154/contributions/5657769/subcontributions/452377/attachments/2754554/4795867/metsai_20231115.pdf)
- [PU rejection using hits](https://its.cern.ch/jira/browse/ATR-28390)
- [e/gamma calibration](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/EGAM-2023-01/)

The framework is originally based on work from two previously existing projects: [[1]](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/GNNJetTagger) [[2]](https://gitlab.cern.ch/mleigh/flavour_tagging/).
