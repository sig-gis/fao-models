# fao-models

[![image](https://img.shields.io/pypi/v/fao-models.svg)](https://pypi.python.org/pypi/fao-models)
[![image](https://img.shields.io/conda/vn/conda-forge/fao-models.svg)](https://anaconda.org/conda-forge/fao-models)

[![image](https://pyup.io/repos/github/kyle-woodward/fao-models/shield.svg)](https://pyup.io/repos/github/kyle-woodward/fao-models)

Image Classification models for the FAO's Forest Resource Assessment data collection campaigns in Collect Earth Online.

The FAO FRA team is conducting its next round of global data collection surveys. They use Collect Earth Online and local trained interpreters. Quality of interpreter answers have been a concern in the past. Answers between multiple interpreters on the same plot may disagree for a variety of reasons, but there has not been a good way to flag certain plots or interpreters in an informed way. We are developing EO models to act as a 'source of truth' in a new CEO QA/QC feature, so that project admins can identify plots or interpreters who display high levels of disagreement with a 'source of truth' and/or other interpreters. 

This particular repo (see its cousin [sig-ssl4eo](https://github.com/sig-gis/sig-ssl4eo)) trains custom deep learning models for our specific task: image classification. That is, a single label for a given image. Our current fine-tuned model is just for forest/non-forest classification, but there is potential for further development beyond a binary forest/non-forest classification. 

-   Free software: GNU General Public License v3
-   Documentation: https://kyle-woodward.github.io/fao-models
    

### Setup

### Inference

### Creating Configs

### Training a Model
