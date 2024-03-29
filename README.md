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

The software is available at PyPI and Bioconda now, to install, run

``` bash
pip install ContigNet
```

or

``` bash
conda install -c bioconda contignet
```

To install the software from source, download and enter the repository by

``` bash
git clone https://github.com/tianqitang1/ContigNet
cd ContigNet
```

To install required dependencies a [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://conda.io/miniconda.html) installation is recommended for managing virtual environments. After a conda distribution is installed, create and activate a ```conda``` virtual environment with the following commands

 ``` bash
 conda create --name ContigNet
 conda activate ContigNet
 pip install .
 ```

## Usage

``` bash
usage: ContigNet [-h] [--host_dir HOST_DIR] [--virus_dir VIRUS_DIR]
                    [--output, -o OUTPUT] [--cpu]

ContigNet, a deep learning based phage-host interaction prediction tool

optional arguments:
  -h, --help            show this help message and exit
  --host_dir HOST_DIR   Directory containing host contig sequences in fasta
                        format (default: demo/host_fasta)
  --virus_dir VIRUS_DIR
                        Directory containing virus contig sequences in fasta
                        format (default: demo/virus_fasta)
  --output, -o OUTPUT   Path to output file (default: result.csv)
  --cpu                 Force using CPU if specified (default: False)
```

## Examples

### Test new contigs

Suppose the phage and host sequences are stored in ```phage``` and ```host``` directories respectively, running

``` bash
ContigNet --host_dir host --virus_dir phage
```

and the likelihood of each phage interacting with each host will be output to ```result.csv```.

For Windows machine, run

``` PowerShell
python -m ContigNet --host_dir host --virus_dir phage
```

<!-- ### Feature extractor mode -->

## Paper related

Browse ```training``` directory for the instructions of running the training and testing process for the paper.

## Copyright and License Information

Copyright (C) 2021 University of Southern California

Authors: Tianqi Tang, Shengwei Hou, Jed Fuhrman, Fengzhu Sun

This program is available under the terms of USC-RL v1.0.

Commercial users should contact Dr. Sun at fsun@usc.edu, copyright at the University of Southern California.
