# py-salt: A python package for SALT: Standardized Audio event Label Taxonomy


This python package includes the code developed based on our paper: [**SALT: Standardized Audio event Label Taxonomy**](https://arxiv.org/abs/2409.11746).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Installation
First, ensure you have `conda` installed on your system. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).


### Step 2: Install extra dependencies with conda
Use conda to install any dependencies that are not available via pip. In this case, pygraphviz:
```bash
conda install --channel conda-forge pygraphviz
```

### Step 3: Install the package in editable mode
After installing the necessary dependencies, install the package in editable mode using pip:
```bash
cd py-salt && pip install -e .
```

## Usage

The package can be used to perform label aggregation or dataset exploration procedures for the publicly datasets that are mapped in SALT. The list of datasets is the following:

**Main Datasets:**

- AudioSet
- Freesound 50k
- ESC-50
- SINGA:PURA
- Urbansas
- SONYC
- UrbanSound8K
- MAVD-traffic
- CHiME Home
- MAESTRO Real
- MATS
- TUT Sound Events 2016
- TUT Sound Events 2017
- TAU NIGENS Spatial Sound Events 2020
- Archeo
- ReaLISED
- Starss22
- Starss23
- Nonspeech7k
- AnimalSound

**Secondary Datasets (subsets of the above):**

- AudioSet strong
- DESEDReal
- MAESTRO Synthetic
- IDMT-traffic
- TUT Rare Sound Events


## Examples

- [py-salt tutorial](/py-salt/notebooks/start_here.ipynb)
- [label aggregation tutorial](/py-salt/notebooks/label_aggregation.ipynb)
- [ESC-50 exploration tutorial](/py-salt/notebooks/esc50_exploration.ipynb)