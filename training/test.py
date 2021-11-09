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
        description="Testing script for use on CARC", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--metadata_dir", dest="metadata_dir", help="Directory storing all metadata", default="data")
    parser.add_argument(
        "--host_dir", dest="host_dir", help="Directory containing host one hot matrix", default="data/host_onehot"
    )
    parser.add_argument(
        "--virus_dir", dest="virus_dir", help="Directory containing virus one hot matrix", default="data/virus_onehot"
    )
    parser.add_argument("--train_info_path", dest='train_info_path', default=None, type=str, help='Path to the directory containing both model and train-test split')
    parser.add_argument("--model_path", "-m", dest="model_path", help="Path to the trained model")
    parser.add_argument("--train_test_split_path", "-s", dest="train_test_split_path", help="Train test split path")
    parser.add_argument(
        "--taxon_rank",
        dest="taxon_rank",
        choices=["phylum", "class", "order", "family", "genus", "species"],
        default="species",
        help="The taxon rank used to determine negative virus-host pair, default: species",
    )
    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")
    parser.add_argument(
        "--no_cudnn_benchmark",
        dest="no_cudnn_bench",
        action="store_true",
        help="Disable CUDNN benchmark, by disabling it might improve reproducibility",
    )
    args = parser.parse_args()

    repeat = 1
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        if not args.no_cudnn_bench:
            torch.backends.cudnn.benchmark = True
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")

    if args.train_info_path:
        model = torch.load(os.path.join(args.train_info_path, 'final_model')).to(device)
        # model = torch.load(os.path.join(args.train_info_path, 'checkpoint', 'epoch_00499')).to(device)

        train_test_split = util.load_obj(os.path.join(args.train_info_path, 'train_test_split.pkl'))
    else:
        model = torch.load(args.model_path).to(device)
        train_test_split = util.load_obj(args.train_test_split_path)
    test_pair_list = train_test_split[1]
    host_taxid_to_fasta = util.load_obj(os.path.join(args.metadata_dir, "bacteria_taxid_to_filename.pkl"))

    virus_host_pair = util.load_obj(os.path.join(args.metadata_dir, "virus_host_pair.pkl"))
    virus_set, host_set = zip(*virus_host_pair)
    virus_set = set(virus_set)
    virus_list = list(virus_set)
    host_set = set(host_set)
    host_list = list(host_set)
    host_onehot = load_host_onehot(args.host_dir, host_set, host_taxid_to_fasta)
    virus_onehot = load_virus_onehot(args.virus_dir, virus_set)

    virus_hostset_mapping = util.tuple_list_to_dict_set(virus_host_pair)

    host_species_taxid_to_taxon_taxid = {}
    for host in host_set:
        host_species_taxid_to_taxon_taxid[host] = util.get_rank(host, args.taxon_rank)

    # possible_contig_length = [500, 1001, 2000, 3002, 4001, 5000]
    possible_contig_length = list(range(200, 5300, 300))
    # possible_contig_length = [200, 1000, 5000]
    metrics_all = []
    roauc = np.zeros((len(possible_contig_length), len(possible_contig_length)))
    for i, virus_contig_len in enumerate(possible_contig_length):
        for j, host_contig_len in enumerate(possible_contig_length):
            metrics_total = np.zeros(7)
            for _ in range(repeat):
                test_loss, test_target, test_output = test(
                    model,
                    args.host_dir,
                    host_onehot,
                    virus_onehot,
                    test_pair_list,
                    virus_contig_len,
                    host_contig_len,
                    100,
                    virus_hostset_mapping,
                    host_taxid_to_fasta,
                    host_species_taxid_to_taxon_taxid,
                    device=device
                )
                roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = evaluate_performance(test_target, test_output)
                metrics_total += [roc_auc, pr_auc, precision, recall, f1_score, acc, mcc]
                print(f'Virus Len: {virus_contig_len}')
                print(f'Host Len: {host_contig_len}')
                print(f'ROAUC:{roc_auc}')
                print(f'-------------------------------------------')
            metrics_total /= repeat
            metrics_all.append([virus_contig_len, host_contig_len, *metrics_total])
            roauc[i, j] = metrics_total[0]
    metrics_df = pd.DataFrame(metrics_all, columns=['virus_length', 'host_length', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score', 'acc', 'mcc'])
    fig, ax = plt.subplots(figsize=(10, 6))
    seaborn.set(font_scale=0.8)
    seaborn.heatmap(roauc, vmin=0.6, vmax=1, xticklabels=possible_contig_length, yticklabels=possible_contig_length, annot=True, ax=ax, cmap='RdYlGn')
    plt.xticks(rotation=90)
    plt.xlabel('Host Contig Length')
    plt.ylabel('Phage Contig Length')
    # plt.savefig('figures/mgv_499.pdf')
    # plt.savefig(os.path.join('figures', f"test_taxon_{args.taxon_rank}_{os.path.split(args.train_info_path)[-1]}.pdf"))
    # plt.savefig(os.path.join('figures', f"test_taxon_{args.taxon_rank}_plasmid_no_share.pdf"))

    # plt.clf()
    # mpl.style.use('default')
    # identical_length = [200, 1000, 5000]
    # labels = ['200bps', '1kbps', '5kbps'][::-1]
    # for i, l in enumerate(identical_length[::-1]):
    #     test_loss, test_target, test_output = test(
    #                 model,
    #                 args.host_dir,
    #                 host_onehot,
    #                 virus_onehot,
    #                 test_pair_list,
    #                 l,
    #                 l,
    #                 100,
    #                 virus_hostset_mapping,
    #                 host_taxid_to_fasta,
    #                 host_species_taxid_to_taxon_taxid,
    #                 device=device
    #             )
    #     fpr, tpr, _ = metrics.roc_curve(test_target, test_output)
    #     plt.plot(fpr, tpr, label=f'{labels[i]}, AUC={metrics.auc(fpr, tpr):.3f}')
    # plt.legend()
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.savefig('dl_contig.pdf')
"""
import os
os.chdir('d:/virus-host-interaction')
metrics_dfs = []
%run test.py --train_info_path data/training_info/2021-03-10-20_24_54_taxon_species_channel_both_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
%run test.py --train_info_path data/training_info/2021-03-11-22_41_44_taxon_species_channel_base_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
%run test.py --train_info_path data/training_info/2021-03-12-00_30_03_taxon_species_channel_codon_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
%run test.py --train_info_path data/training_info/2021-03-17-13_53_46_taxon_species_channel_both_lr0.0001_shareweight_False
metrics_dfs.append(metrics_df)
%run test.py --train_info_path data/training_info/2021-03-17-21_08_02_taxon_species_channel_both_lr0.0001_shareweight_False_revcomp_True
metrics_dfs.append(metrics_df)

%run test.py --metadata_dir data/plasmid_data/ --host_dir data/plasmid_data/host_onehot/ --virus_dir data/plasmid_data/plasmid_onehot/ --model_path data/training_info/2021-03-10-20_24_54_taxon_species_channel_both_lr0.0001_shareweight_True/final_model --train_test_split_path data/plasmid_data/train_test_split.pkl

roauc_taxon_level = []
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank phylum
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank class
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank order
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank family
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank genus
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_phylum_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-27-23_40_57 --taxon_rank species
roauc_taxon_level.append(roauc)

roauc_taxon_level = []
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank phylum
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank class
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank order
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank family
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank genus
roauc_taxon_level.append(roauc)
%run test.py --train_info_path data/training_info/taxon_class_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-04-28-21_55_15 --taxon_rank species
roauc_taxon_level.append(roauc)
roauc_taxon_level = [roauc.flatten() for roauc in roauc_taxon_level]
roauc_taxon_df = pd.DataFrame(np.vstack(roauc_taxon_level).T, columns=["phylum", "class", "order", "family", "genus", "species"])

import seaborn
seaborn.boxplot(data=roauc_taxon_df)

roauc_genus = []
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00099 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00199 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00299 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00399 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00499 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00599 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00699 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00799 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00899 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)
%run test.py -m data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/checkpoint/epoch_00999 -s data/training_info/taxon_genus_channel_both_lr0.0001_shareweight_True_revcomp_False_2021-05-01-03_06_24/train_test_split.pkl --taxon_rank genus
roauc_genus.append(roauc)

import os
os.chdir('d:/virus-host-interaction')
metrics_dfs = []
aucs = []
%run test.py --train_info_path data/training_info/2021-03-10-20_24_54_taxon_species_channel_both_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
aucs.append(roauc)
%run test.py --train_info_path data/training_info/2021-03-11-22_41_44_taxon_species_channel_base_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
aucs.append(roauc)
%run test.py --train_info_path data/training_info/2021-03-12-00_30_03_taxon_species_channel_codon_lr0.0001_shareweight_True
metrics_dfs.append(metrics_df)
aucs.append(roauc)
fig, ax = plt.subplots(figsize=(10, 6))
seaborn.set(font_scale=0.8)
seaborn.heatmap((aucs[0]-aucs[2])*100, vmin=-10, vmax=10, xticklabels=possible_contig_length, yticklabels=possible_contig_length, annot=True, ax=ax, cmap='RdYlGn', fmt='.1f', cbar_kws={'label':'ROAUC×100'})
plt.xticks(rotation=90)
plt.xlabel('Host Contig Length')
plt.ylabel('Phage Contig Length')
"""