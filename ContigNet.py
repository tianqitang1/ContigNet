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
        "--host_dir", dest="host_dir", help="Directory containing host one hot matrix", default="data/host_onehot"
    )
    parser.add_argument(
        "--virus_dir", dest="virus_dir", help="Directory containing virus one hot matrix", default="data/virus_onehot"
    )

    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")


    args = parser.parse_args()

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        if not args.no_cudnn_bench:
            torch.backends.cudnn.benchmark = True
        print(torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available or CPU is explicitly selected for use. Using CPU.")
        device = torch.device("cpu")

    model = torch.load('model')

    host_list = os.listdir(args.host_dir)
    host_list.sort()
    host_path_list = [os.path.join(args.host_dir, i) for i in host_list]
    virus_list = os.listdir(args.virus_dir)
    virus_list.sort()
    virus_path_list = [os.path.join(args.virus_dir, i) for i in virus_list]

    host_sequences = {}
    virus_sequences = {}

    host_onehots = {}
    virus_onehots = {}

    