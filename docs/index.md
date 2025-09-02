# Introduction

Welcome to the Salt framework!

Salt is a general-purpose framework for training **multi-modal**, **multi-task** models for high energy physics.

Salt supports arbitrary combinations of tasks including object classification and regression, and set-to-set reconstruction via edge classification and segmentation.
Salt was developed for state-of-the art jet flavour tagging algorithms such as [GN1](https://ftag-docs.docs.cern.ch/algorithms/taggers/GN1/) and [GN2](https://ftag-docs.docs.cern.ch/algorithms/taggers/GN2/), but can be applied much more widely, as seen [below](#current-usage).

!!! example "The code is hosted on the CERN GitLab: [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt)"

### Features

- Built on [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
- Support for multiple YAML-configurable [input modalities][salt.models.InitNet] and [output tasks][task-heads].
- ONNX export support to use trained models in C++ environments like [Athena](https://gitlab.cern.ch/atlas/athena/).
- Easily extensible: you can implement your own custom dataloaders and models.
- Documented and tested.

### Getting Started

Below are some helpful links to get you started:

!!! info "You can find out more about flavour tagging algorithms at the [FTAG docs](https://ftag.docs.cern.ch/)"

!!! question "There is a [channel](https://mattermost.web.cern.ch/aft-algs/channels/gnns) for the framework in the [FTAG Mattermost workspace](https://mattermost.web.cern.ch/signup_user_complete/?id=1wicts5csjd49kg7uymwwt9aho&md=link&sbr=su) (for active CERN users only)"

!!! abstract "A tutorial on using Salt with open data can be found [here](tutorial.md). A tutorial using internal CERN data is [also available](tutorial-Xbb.md)"

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
- [Pileup rejection using hits](https://its.cern.ch/jira/browse/ATR-28390)
- [e/gamma calibration](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/EGAM-2023-01/)
- [Pileup jet rejection with GNJVT](https://indico.cern.ch/event/1465013/contributions/6168106/attachments/2962190/5210357/GN-JVT%20Nov%206th%202024.pdf)


## Statement of Need

High energy physics research increasingly requires sophisticated machine learning tools to address complex data analysis challenges, for example identifying jets from bottom and charm quarks through their distinctive decay signatures in particle detectors. Salt meets this need by providing a versatile and high-performance machine learning framework that supports various tasks including object classification, regression, and set-to-set reconstruction, enabling physicists to effectively analyse complex particle collision signatures such as charged particle trajectories, decay vertices, and jets.


## Citing

If you use this software, please cite our article in the Journal of Open Source Software: [10.21105/joss.07217](https://joss.theoj.org/papers/10.21105/joss.07217).

```bibtex
@article{salt2025,
  author = {Jackson Barr and Diptaparna Biswas and Maxence Draguet and Philipp Gadow and Emil Haines and Osama Karkout and Dmitrii Kobylianskii and Wei Sheng Lai and Matthew Leigh and Nicholas Luongo and Ivan Oleksiyuk and Nikita Pond and Sébastien Rettie and Andrius Vaitkus and Samuel Van Stroud and Johannes Wagner},
  title = {Salt: Multimodal Multitask Machine Learning for High Energy Physics},
  journal = {Journal of Open Source Software},
  volume = {10},
  issue = {112},
  year = {2025},
  doi = {10.21105/joss.07217},
  url = {https://joss.theoj.org/papers/10.21105/joss.07217},
  issn = {2475-9066}
}
```