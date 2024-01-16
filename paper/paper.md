---
title: 'Salt: Multimodal Multitask Machine Learning for High Energy Physics'

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
    orcid: 0000-0002-7543-3471
    affiliation: 2
  - name: Maxence Draguet
    orcid: 0000-0003-1530-0519
    affiliation: 3
  - name: Philipp Gadow
    orcid: 0000-0003-4475-6734
    affiliation: 4
  - name: Dmitrii Kobylianskii
    orcid: 0009-0002-0070-5900
    affiliation: 5
  - name: Ivan Oleksiyuk
    orcid: 0000-0002-4784-6340
    affiliation: 6
  - name: Nikita Pond
    orcid: 0000-0002-5966-0332
    affiliation: 1
  - name: Sébastien Rettie
    orcid: 0000-0002-7092-3893
    affiliation: 4
  - name: Andrius Vaitkus
    orcid: 0000-0002-0393-666X
    affiliation: 1
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    affiliation: 1
  - name: Johannes Wagner
    orcid: 0000-0002-5588-0020
    affiliation: 8

affiliations:
 - name: University College London, United Kingdom
   index: 1
 - name: University of Siegen
   index: 2
 - name: University of Oxford, United Kingdom
   index: 3
 - name: European Laboratory for Particle Physics CERN, Switzerland
   index: 4
 - name: Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Israel
   index: 5
 - name: Université de Genève, Switzerland
   index: 6
 - name: Technical University of Munich, Germany
   index: 7
 - name: University of California, Berkeley
   index: 8

date: 15 Janurary 2024
bibliography: paper.bib

---

# Summary

High energy physics studies the fundamental particles and forces that constitute the universe, often through experiments conducted in large particle accelerators such as the Large Hadron Collider (LHC) [@Evans:2008].
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
`Salt` meets this growing need by providing a versatile, performant, and user-friendly tool for developing advanced ML models.
`Salt` was originally developed to train state of the art flavour tagging models at the ATLAS experiment [@ATLAS:2008] at the LHC.
Flavour tagging, the identification of jets from bottom and charm quarks, plays a crucial role in analysing ATLAS collision data. This process is key for precision Standard Model measurements, particularly in the characterisation of the Higgs bosons, and for investigating new phenomena.
The unique characteristics of hadrons containing bottom and charm quarks – such as their long lifetimes, high mass, and high decay multiplicity – create distinct signatures in particle detectors which can be effectively exploited by ML algorithms.

While initially developed for flavour tagging, `Salt` has evolved into a flexible tool that can be used for a wide range of tasks, from object classification, regression of object properties, to vertex reconstruction and full event classification, demonstrating its broad applicability across various data analysis challenges in high energy physics.

# Model Architecture

Salt facilitates the training of multimodal and multitask models as depicted in \autoref{fig:salt-arch}.
The architecture is designed to take advantage of multiple input modalities, which, in the context of jet classification, might include global features of the jet and constituents of the jet such as tracks, calorimeter clusters, leptons, or inner detector hits.
This architecture allows the model to leverage all the available detector information, leading to improved performance.
A unified encoder jointly processes these inputs, and the encoder outputs are then used for a configurable number of tasks.
A concrete example of this architecture is in use at ATLAS [@GN1; @GN2X].

![This diagram illustrates the flow of information within a model trained using `Salt`. Global features and inputs from multiple constituents feed into a unified encoder, which processes and integrates the information. The encoder then outputs to multiple task-specific modules, each tailored to a specific objective.\label{fig:salt-arch}](salt-arch.png){ width=80% }


# Related work

`Umami` [currently under JOSS review] is a related software package in use at ATLAS. 
It can be used for preprocessing data before training models with `Salt`.
`Umami` is missing several features provided by `Salt`, such as advanced transformer models, multimodal and multitask learning, and distributed model training.


# Acknowledgements

The development of `Salt` is part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References
