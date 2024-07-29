<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
<p align="center">
  <a href="https://github.com/vevotx/vevo-scgpt-private">
    <picture>
      <img alt="vevo-therapeutics" src="./assets/vevo_logo.png" width="95%">
    </picture>
  </a>
</p>
<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Linter: Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/vevotx/vevo-scgpt-private/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
<br />

# MosaicFM

This is the internal codebase for the **MosaicFM** series of single-cell RNA-seq foundation models 
developed by Vevo Therapeutics. Our repository follows a similar structure to [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) 
and imports several utility functions from it. Please follow the developer guidelines if you are 
contributing to this repository. For main results and documentation, please refer to the results section. 
If you are looking to train or finetune a model on single-cell data, please refer to the training section.

## Hardware and Software Requirements

We have tested our code on NVIDIA A100 and H100 GPUs with CUDA 12.1. 
At the moment, we are also restricted to use a version of llm-foundry no later v0.6.0, since support for the triton 
implementation of flash-attention was removed in [v0.7.0](https://github.com/mosaicml/llm-foundry/releases/tag/v0.7.0).

We support launching runs on the MosaicML platform as well as on local machines through RunAI.

## Datasets
## Pre-trained Models
## Results
## Tutorials
## Developer guidelines
## Launching Jobs
## Acknowledgements