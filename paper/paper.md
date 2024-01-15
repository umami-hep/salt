---
title: 'Salt: Multimodal Mutltask ML Models for High Energy Physics'

tags:
  - Python
  - high energy physics
  - machine learning
  - jet physics
  - flavour tagging

authors:
  - name: Jackson Barr
    orcid: 0000-0002-9752-9204 
    affiliation: 1
  - name: Diptaparna Biswas
    affiliation: 2
  - name: Maxence Draguet
    affiliation: 3
  - name: Philipp Gadow
    orcid: 0000-0003-4475-6734
    affiliation: 5
  - name: Dmitrii Kobylianskii
    orcid: 0009-0002-0070-5900
    affiliation: 10
  - name: Ivan Oleksiyuk
    orcid: 0000-0002-4784-6340
    affiliation: 4
  - name: Nikita Pond
    orcid: 0000-0002-5966-0332
    affiliation: 1
  - name: Sébastien Rettie
    orcid: 0000-0002-7092-3893
    affiliation: 5
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    affiliation: 1
  - name: Andrius Vaitkus
    affiliation: 1
  - name: Johannes Wagner
    affiliation: 13

affiliations:
 - name: University College London, United Kingdom
   index: 1
 - name: University of Siegen
   index: 2
 - name: University of Oxford, United Kingdom
   index: 3
 - name: Université de Genève, Switzerland
   index: 4
 - name: European Laboratory for Particle Physics CERN, Switzerland
   index: 5
 - name: Technical University of Munich, Germany
   index: 7
 - name: SLAC National Accelerator Laboratory, United States of America
   index: 8
 - name: Nikhef National Institute for Subatomic Physics and University of Amsterdam, Netherlands
   index: 9
 - name: Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Israel
   index: 10
 - name: INFN Genova and Universita' di Genova, Italy
   index: 12
 - name: University of California, Berkeley
   index: 13

date: 15 Janurary 2024
bibliography: paper.bib

---

# Summary

`Salt` is a Python package developed for the high energy physics community which streamlines the training of multimodal, multitask machine learning (ML) models.
`Salt` simplifies the creation and deployment of advanced ML models, making them more accessible and promoting shared best practices.
It also provides extensive customization options for model training and architecture to suit a variety of high energy physics applications.

Some key features of `Salt` are listed below:

- Based on established frameworks: `Salt` is built upon PyTorch [@pytorch] and Lightning [@lightning] for maximum performance and scalability with minimal boilerplate code.
- Multimodal, multitask models: `Salt` provides an efficient transformer backbone with support for multimodal inputs and edge features. It also provides various task head modules for classification, regression, and vertex reconstruction tasks. Any combination of these can be used to flexibly define models for multitask learning problems.
- Customisable and extensible: `Salt` supports full customisation of training and model configuration through YAML config files. Its modular design allows for the easy integration of custom dataloaders, layers, and models.
- Train at scale: `Salt` can handle large volumes of data with efficient HDF5 [@hdf5:2023] dataloaders. It also includes multi-GPU support from Lightning, enabling distributed training.
- Deployment ready: The package facilitates ONNX serialization for integrating models into C++ based software environments.


# Statement of need

In high energy physics research the reliance on ML for data analysis and object classification is growing [@Guest:2018; @Cagnotta:2022].
The Salt package meets this growing need by providing a versatile, formant, and user-friendly tool for developing advanced ML models.
`Salt` was originally developed to train state of the art flavour tagging models at the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008].
Flavour tagging, the identification of jets from bottom and charm quarks, plays a crucial role in analysing ATLAS collision data. This process is key for precision Standard Model measurements, particularly in the characterisation of the Higgs bosons, and for investigating new phenomena.
The unique characteristics of hadrons containing bottom and charm quarks – such as their long lifetimes, high mass, and high decay multiplicity – create distinct signatures in particle detectors which can be effectively exploited by ML algorithms.


# Model Architecture

Salt enables the training of multimodal, multitask models, as depicted in \autoref{fig:salt-arch}.
The architecture is designed to take advantage of multiple input modalities, which, in the context of jet classification, might include global features of the jet, and constituents such as tracks, calorimeter clusters, reconstructed leptons, or inner detector hits.
This architecture allows the model to leverage all the available detector information.
A unified encoder jointly processes these inputs, and the encoder outputs are then used for a configurable number of tasks.

![This diagram illustrates the flow of information within a model trained using `Salt`. Global features and inputs from multiple constituents feed into a unified encoder, which processes and integrates the information. The encoder then outputs to multiple task-specific modules, each tailored to a specific objective.\label{fig:salt-arch}](salt-arch.png){ width=80% }


# Related work

`Umami` [currently under JOSS review] is a related software package in use at ATLAS. 
It can be used for preprocessing data before training models with `Salt`.
`Umami` is missing several features provided by `Salt`, such as advanced transformer models, mutlimodal and multitask learning, and distributed training.
`Weaver` is an alternative package developed by members of the CMS collaboration [cite].


# Acknowledgements

The development of `Salt` is part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References
