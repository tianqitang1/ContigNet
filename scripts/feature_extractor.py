from fileinput import filelineno
import math
import os
import random
import sys
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
from scripts.train_model import load_host_onehot, train_test_split
from typing import List

sys.path.append("..")

import warnings

warnings.filterwarnings(
    "ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
)

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import time
import ContigNet.util as util
from tqdm import tqdm

from scripts.train_model import test, evaluate_performance, load_virus_onehot

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ContigNet Feature Extractor, extract features from sequences using ContigNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input, -i", dest="input_dir", help="Specifi the input directory storing sequences in FASTA format"
    )
    parser.add_argument(
        "--output, -o", dest="output_dir", help="Specify the path to the directory to store features for each sequence"
    )
    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")

    args = parser.parse_args()

    file_list = os.listdir(args.input_dir)
    file_path_list = [os.path.join(args.input_dir, file) for file in file_list]
    name_list = [os.path.splitext(file)[0] for file in file_list]

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        print(f'Using {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA is not available or CPU is explicitly selected for use. Using CPU.")
        device = torch.device("cpu")
    model = torch.load('final_model', map_location=device)
    model_cpu = torch.load('final_model', map_location='cpu')

    with torch.no_grad():
        model.eval()

        for i, name in enumerate(name_list):
            onehot = util.fasta2onehot(file_path_list[i])
            try:
                base_feature = model.base_channel1(torch.Tensor(onehot).to(device)[None, None, :, :]).cpu()
                codon_onehot = model.codon_transformer(torch.Tensor(onehot).to(device)[None, None, :, :])
                codon_feature = model.codon_channel1(codon_onehot).cpu()
            except RuntimeError:
                base_feature = model_cpu.base_channel1(torch.Tensor(onehot)[None, None, :, :])
                codon_onehot = model_cpu.codon_transformer(torch.Tensor(onehot)[None, None, :, :])
                codon_feature = model_cpu.codon_channel1(codon_onehot)
                "Running out of GPU memory"
            feature = torch.cat((base_feature, codon_feature), dim=1).numpy()
            np.save(os.path.join(args.output_dir, f'{name}.npy'), feature)
print("Finished")
