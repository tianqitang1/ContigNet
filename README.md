# ContigNet: Phage-bacteria contig interaction prediction with convolutional neural network

[![Tests](https://github.com/tianqitang1/ContigNet/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/tianqitang1/ContigNet/actions/workflows/tests.yml)

The paper has been published at <https://doi.org/10.1093/bioinformatics/btac239>

Version: 1.0.1

Authors: Tianqi Tang, Shengwei Hou, Jed Fuhrman, Fengzhu Sun

Maintainer: Tianqi Tang tianqit@usc.edu

## Description

This is the repository containing the software ContigNet and related scripts for the paper "Phage-bacteria contig interaction prediction with convolutional neural network".

ContigNet is a deep learning based software for phage-host contig interaction prediction.
Traditional methods can work on contigs however the performance is poor.
Existing Deep learning based methods are not able to solve the particular question regarding interaction prediction between two contigs.

## Installation

### From PyPI (Recommended)

The software is available on PyPI. To install:

``` bash
pip install ContigNet
```

Or with [uv](https://github.com/astral-sh/uv) (faster):

``` bash
uv pip install ContigNet
```

### From Conda

``` bash
conda install -c bioconda contignet
```

### From Source

#### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. First, install uv:

``` bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install ContigNet from source:

``` bash
git clone https://github.com/tianqitang1/ContigNet
cd ContigNet
uv sync
```

This will create a virtual environment and install all dependencies automatically.

#### Using pip

Alternatively, you can use pip with a traditional virtual environment:

``` bash
git clone https://github.com/tianqitang1/ContigNet
cd ContigNet
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

```
Usage: ContigNet [OPTIONS]

  ContigNet: A deep learning based phage-host interaction prediction tool

  Predicts phage-host contig interactions using a convolutional neural
  network.

Options:
  --host-dir, -ho PATH   Directory containing host contig sequences in fasta
                         format  [default: demo/host_fasta]
  --virus-dir, -vi PATH  Directory containing virus contig sequences in fasta
                         format  [default: demo/virus_fasta]
  --output, -o PATH      Path to output file  [default: result.csv]
  --cpu                  Force using CPU if specified
  --show-preview         Show a preview table of top predictions
  --help                 Show this message and exit.
```

## Examples

### Test new contigs

Suppose the phage and host sequences are stored in ```phage``` and ```host``` directories respectively:

``` bash
ContigNet --host-dir host --virus-dir phage
```

The likelihood of each phage interacting with each host will be output to ```result.csv```.

### Show preview of results

To see the top 10 predictions in a nice table:

``` bash
ContigNet --host-dir host --virus-dir phage --show-preview
```

### Run with uv

If you installed from source with uv:

``` bash
uv run ContigNet --host-dir host --virus-dir phage
```

### Windows

On Windows, you can also run:

``` PowerShell
python -m ContigNet --host-dir host --virus-dir phage
```

<!-- ### Feature extractor mode -->

## Paper related

Browse ```training``` directory for the instructions of running the training and testing process for the paper.

## Copyright and License Information

Copyright (C) 2021 University of Southern California

Authors: Tianqi Tang, Shengwei Hou, Jed Fuhrman, Fengzhu Sun

This program is available under the terms of USC-RL v1.0.

Commercial users should contact Dr. Sun at fsun@usc.edu, copyright at the University of Southern California.
