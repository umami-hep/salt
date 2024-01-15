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
    orcid: NA
    affiliation: NA
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
  - name: Sebastien Rettie
    orcid: 0000-0002-7092-3893
    affiliation: 5
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    affiliation: 1
  - name: Andrius Vaitkus
    orcid: NA
    affiliation: 1
  - name: Johannes Wagner
    orcid: NA
    affiliation: NA

affiliations:
 - name: University College London, United Kingdom
   index: 1
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

date: 15 Janurary 2024
bibliography: paper.bib

---

# Summary

`Salt` is a Python [@Rossum:2009] package developed for the high energy physics community which streamlines the training of multimodal, multitask machine learning (ML) models.
Salt aims to make the deployment of advanced ML models more straightforward and allows for extensive customization in model training and architecture, catering to a wide variety of use cases.

Some key features of the package are listed below:

- Based on established frameworks: `Salt` is built upon PyTorch [@pytorch] and Lightning [@lightning] for maximum performance and scalability with minimal boilerplate code.
- Multimodal, multitask models: `Salt` provides an efficient transformer backbone with support for multimodal inputs and edge features. It also provides various task head modules for classification, regression, and vertex reconstruction tasks. Any combination of these can be used for multitask learning problems.
- Customisable and extensible: Salt supports full customisation of training parameters and model architecture through YAML config files. Its modular design allows for the implementation of custom dataloaders, layers, and models.
- Train at scale: `Salt` can handle large volumes of data with efficient HDF5 [@hdf5:2023] dataloaders. It also includes multi-GPU support from Lightning [@lightning], enabling distributed training.
- Deploy: Salt supports ONNX serialization, enabling model inference in existing software stacks based on `C++`.


# Statement of need

In high energy physics research the reliance on ML for data analysis and object classification is growing [@Guest:2018; @Cagnotta:2022].
`Salt` directly addresses this need by offering a flexible, performant, and user-friendly platform for training advanced ML models.
`Salt` was originally developed to train state of the art flavour tagging models [@GN2X] at the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008].
Flavour tagging is the identification jets originating from bottom and charm quarks, and a vital part of analysing collision data produced at ATLAS.
It is essential for precision measurements of the Standard Model, particularly in characterizing the properties of the Higgs boson and in exploring new phenomena.
The unique characteristics of hadrons containing bottom and charm quarks – such as their long lifetimes, high mass, and high decay multiplicity – create distinct signatures in particle detectors which can be effectively exploited by ML algorithms.


# Related work

`Umami` [currently under JOSS review] is a related software package in use at ATLAS. 
It can be used for preprocessing data before training models with `Salt`.
`Umami` is missing several features provided by `Salt`, such as advanced transformer models, mutlimodal and multitask learning, and distributed training.
`Weaver` is an alternative package developed by members of the CMS collaboration [cite].

# Acknowledgements

The development of `Salt` is part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References
