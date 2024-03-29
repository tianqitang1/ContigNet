#!/usr/bin/env python

import os
import pandas as pd
import warnings
import numpy as np
import torch
from . import util, VirusCNN_siamese
from tqdm import tqdm
import pkgutil
from io import BytesIO
import argparse

warnings.filterwarnings(
    "ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
)


def main():
    parser = argparse.ArgumentParser(
        description="ContigNet, a deep learning based phage-host interaction prediction tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host_dir", '-ho',
        dest="host_dir",
        help="Directory containing host contig sequences in fasta format",
        default="demo/host_fasta",
    )
    parser.add_argument(
        "--virus_dir", '-vi',
        dest="virus_dir",
        help="Directory containing virus contig sequences in fasta format",
        default="demo/virus_fasta",
    )

    parser.add_argument("--output, -o", dest="output", help="Path to output file", default="result.csv")

    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        print(f"Using {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available or CPU is explicitly selected for use. Using CPU.")
        device = torch.device("cpu")

    model = VirusCNN_siamese.VirusCNN(share_weight=True).to(device)
    model.load_state_dict(torch.load(BytesIO(pkgutil.get_data("ContigNet", "models/model.dict")), map_location=device))

    host_list = os.listdir(args.host_dir)
    host_list.sort()
    host_name_list = [os.path.splitext(i)[0] for i in host_list]
    host_path_list = [os.path.join(args.host_dir, i) for i in host_list]
    virus_list = os.listdir(args.virus_dir)
    virus_list.sort()
    virus_name_list = [os.path.splitext(i)[0] for i in virus_list]
    virus_path_list = [os.path.join(args.virus_dir, i) for i in virus_list]

    # print(f'Loading fasta files')
    # host_onehots = {host_name_list[i]: util.fasta2onehot(host_path_list[i]) for i in tqdm(range(len(host_list)))}
    # virus_onehots = host_onehots
    # virus_onehots = {virus_name_list[i]: util.fasta2onehot(virus_path_list[i]) for i in tqdm(range(len(virus_list)))}

    result_df = pd.DataFrame(np.zeros((len(host_list), len(virus_list))), columns=virus_name_list, index=host_name_list)

    with torch.no_grad():
        model.eval()
        # for host_name, host_onehot in tqdm(host_onehots.items()):
        #     for virus_name, virus_onehot in virus_onehots.items():
        for i, host_fn in tqdm(enumerate(host_list)):
            host_name = host_name_list[i]
            host_path = host_path_list[i]
            host_onehot = util.fasta2onehot(host_path)
            for j, virus_fn in tqdm(enumerate(virus_list), leave=False):
                virus_name = virus_name_list[j]
                virus_path = virus_path_list[j]
                virus_onehot = util.fasta2onehot(virus_path)
                try:
                    host_tensor = torch.Tensor(host_onehot).to(device)[None, None, :, :]
                    virus_tensor = torch.Tensor(virus_onehot).to(device)[None, None, :, :]
                    if str(device) != "cpu":
                        output = torch.sigmoid(model(host_tensor, virus_tensor)).cpu().numpy().flatten()[0]
                    else:
                        output = torch.sigmoid(model(host_tensor, virus_tensor)).numpy().flatten()[0]
                except RuntimeError as e:  # Fallback in case of out of GPU memory
                    if "CUDA error: out of memory" in str(e):
                        torch.cuda.empty_cache()
                        model = model.to("cpu")
                        host_tensor = torch.Tensor(host_onehot)[None, None, :, :]
                        virus_tensor = torch.Tensor(virus_onehot)[None, None, :, :]
                        output = torch.sigmoid(model(host_tensor, virus_tensor)).numpy().flatten()[0]
                    else:
                        raise e
                result_df.loc[host_name, virus_name] = output
    result_df.to_csv(args.output)


if __name__ == "__main__":
    main()
