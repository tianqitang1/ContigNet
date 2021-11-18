import math
import os
import random
import sys
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
from train_model import load_host_onehot, train_test_split
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
import util
import seaborn

from train_model import test, evaluate_performance, load_virus_onehot

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ContigNet, a deep learning based phage-host interaction prediction tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host_dir", dest="host_dir", help="Directory containing host contig sequences in fasta format", default="demo/host_fasta"
    )
    parser.add_argument(
        "--virus_dir", dest="virus_dir", help="Directory containing virus contig sequences in fasta format", default="demo/virus_fasta"
    )

    parser.add_argument("--output, -o", dest="output", help="Path to output file", default='result.csv')

    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")


    args = parser.parse_args()

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        print(f'Using {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA is not available or CPU is explicitly selected for use. Using CPU.")
        device = torch.device("cpu")

    model = torch.load('model')

    host_list = os.listdir(args.host_dir)
    host_list.sort()
    host_name_list = [os.path.splitext(i)[0] for i in host_list]
    host_path_list = [os.path.join(args.host_dir, i) for i in host_list]
    virus_list = os.listdir(args.virus_dir)
    virus_list.sort()
    virus_name_list = [os.path.splitext(i)[0] for i in virus_list]
    virus_path_list = [os.path.join(args.virus_dir, i) for i in virus_list]

    # host_sequences = {host_name_list[i]: util.load}
    # virus_sequences = {}

    host_onehots = {host_name_list[i]: util.fasta2onehot(host_path_list[i]) for i in range(len(host_list))}
    virus_onehots = {virus_name_list[i]: util.fasta2onehot(virus_path_list[i]) for i in range(len(virus_list))}
    
    result_df = pd.DataFrame(np.zeros((len(host_list), len(virus_list))), host_name_list, virus_name_list)

    with torch.no_grad():
        model.eval()
        for host_name, host_onehot in host_onehots.items():
            for virus_name, virus_onehot in virus_onehots.items():
                try:
                    host_tensor = torch.Tensor(host_onehot).to(device)[None, None, :, :]
                    virus_tensor = torch.Tensor(virus_onehot).to(device)[None, None, :, :]
                    output = torch.sigmoid(model(host_tensor, virus_tensor)).cpu().numpy().flatten()[0]
                except RuntimeError: # Fallback in case of out of GPU memory
                    torch.cuda.empty_cache()
                    model = model.to('cpu')
                    host_tensor = torch.Tensor(host_onehot)[None, None, :, :]
                    virus_tensor = torch.Tensor(virus_onehot)[None, None, :, :]
                    output = torch.sigmoid(model(host_tensor, virus_tensor)).numpy().flatten()[0]
                result_df.loc[host_name, virus_name] = output
    result_df.to_csv(args.output)

    