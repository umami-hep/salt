[![docs](https://img.shields.io/badge/info-documentation-informational)](https://ftag-salt.docs.cern.ch/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07217/status.svg)](https://doi.org/10.21105/joss.07217)
[![pipeline](https://gitlab.cern.ch/npond/salt/badges/main/pipeline.svg)](https://gitlab.cern.ch/npond/salt/-/pipelines?ref=main)
[![weekly uv](https://gitlab.cern.ch/npond/salt/-/jobs/artifacts/main/raw/badges/weekly-uv.svg?job=weekly-badge)](https://gitlab.cern.ch/npond/salt/-/pipeline_schedules)
[![weekly conda](https://gitlab.cern.ch/npond/salt/-/jobs/artifacts/main/raw/badges/weekly-conda.svg?job=weekly-badge)](https://gitlab.cern.ch/npond/salt/-/pipeline_schedules)

# Salt

This is the home of Salt, a framework for training multi-model and multi-task models models in the style of GN2.

The `pipeline` badge is the latest pipeline on `main` (normally a push; a scheduled run can also be the latest). The `weekly` badges come from a scheduled pipeline that, every Monday, cold-installs salt with `setup/setup_uv.sh` and `setup/setup_conda.sh` on a plain `python:3.14` image and runs the full CPU test suite on each; they show the latest result and its date.

Documentation is available [here](https://ftag-salt.docs.cern.ch/).

If you use this software, please cite our article in the Journal of Open Source Software.

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