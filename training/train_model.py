import math
import os
import random
import sys
from typing import List

import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from sklearn import metrics
import time
import ContigNet.util as util
import pandas as pd

from ContigNet.VirusCNN_siamese import VirusCNN

import argparse

sys.path.append("..")

warnings.filterwarnings(
    "ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
)


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


def train_test_split(virus_host_pair, train_ratio=0.8, host_taxon_level='genus', training_only=False):
    if training_only:
        return virus_host_pair, []
    train_boundary = math.floor(len(virus_host_pair) * train_ratio)
    init_train_pair_list = virus_host_pair[:train_boundary]
    init_test_pair_list = virus_host_pair[train_boundary:]
    virus_train_test_intersection = set(list(zip(*init_train_pair_list))[0]).intersection(
        set(list(zip(*init_test_pair_list))[0])
    )
    # host_train_test_intersection = set(list(zip(*init_train_pair_list))[1]).intersection(
    #     set(list(zip(*init_test_pair_list))[1])
    # )
    host_train_test_intersection = set()
    host_train_genus = set()
    host_test_genus = set()
    for host in set(list(zip(*init_train_pair_list))[1]):
        host_train_genus.add(util.get_rank(host, host_taxon_level))
    for host in set(list(zip(*init_test_pair_list))[1]):
        host_test_genus.add(util.get_rank(host, host_taxon_level))
    for host in set(list(zip(*init_train_pair_list))[1]):
        if util.get_rank(host, host_taxon_level) in host_test_genus:
            host_train_test_intersection.add(host)
    for host in set(list(zip(*init_test_pair_list))[1]):
        if util.get_rank(host, host_taxon_level) in host_train_genus:
            host_train_test_intersection.add(host)
    

    train_pair_list = [pair for pair in init_train_pair_list if (pair[0] not in virus_train_test_intersection) and
                       (pair[1] not in host_train_test_intersection)]
    test_pair_list = [pair for pair in init_train_pair_list if (pair[0] in virus_train_test_intersection) or
                      (pair[1] in host_train_test_intersection)]
    test_pair_list.extend(init_test_pair_list)

    return train_pair_list, test_pair_list


def load_virus_onehot(virus_dir, virus_list):
    """Load all virus onehot matrices into memory

    :param virus_dir: path to directory storing virus onehot matrices
    :type virus_dir: str
    :param virus_list:
    :type virus_list: Union[list, set]
    :return: dictionary mapping given virus name to onehot matrix
    :rtype: dict
    """
    virus_onehot_filename = [virus + ".fasta.npy" for virus in virus_list]
    virus_onehot = {}

    for filename in virus_onehot_filename:
        virus_onehot_id = filename[:-10]
        onehot = np.load(os.path.join(virus_dir, filename))
        onehot = onehot[:, :4]
        assert len(onehot.shape) == 2 and onehot.shape[1] == 4, "One hot shape incorrect"
        virus_onehot[virus_onehot_id] = onehot
    return virus_onehot


def load_host_onehot(host_dir, host_set, host_taxid_to_fasta):
    """

    :param host_dir: path to directory storing host onehot matrices
    :param host_set:
    :param host_taxid_to_fasta:
    :return:
    """
    host_onehot_filename = []
    for host in host_set:
        host_onehot_filename.extend([fasta + ".npy" for fasta in host_taxid_to_fasta[host]])
    host_onehot = {}

    for filename in host_onehot_filename:
        host_fn = filename[:-4]
        onehot = np.load(os.path.join(host_dir, filename)).astype(np.int8)
        onehot = onehot[:, :4]
        assert len(onehot.shape) == 2 and onehot.shape[1] == 4, "One hot shape incorrect"
        host_onehot[host_fn] = onehot
    return host_onehot


def sample_onehot(genome_onehot, length):
    """
    Sample a given length fragment from a genome onehot matrix, if the genome length is shorter than the given length,
    return a 0-padded matrix
    :param genome_onehot:
    :param length:
    :return:
    """
    genome_length = genome_onehot.shape[0]
    if genome_length > length:
        start_index = np.random.randint(0, genome_length - length)
        end_index = start_index + length
        onehot_sampled = genome_onehot[start_index:end_index, :]
    elif genome_length == length:
        onehot_sampled = genome_onehot
    else:
        onehot_sampled = np.vstack((genome_onehot, np.zeros((length - genome_length, genome_onehot.shape[1]))))
    return onehot_sampled


def construct_batch(
        host_dir,
        host_onehot,
        virus_onehot,
        pair_list,
        index,
        virus_contig_len,
        host_contig_len,
        virus_hostset_mapping,
        host_taxid_to_fasta,
        host_species_taxid_to_taxon_taxid,
        host_list=None,
        low_memory=False,
):
    pairs = [pair_list[i] for i in index]
    virus_id_batch, host_id_batch = zip(*pairs)

    if not host_list:
        _, host_list = get_virus_host_list(pair_list)

    virus_onehot_batch = [virus_onehot[virus] for virus in virus_id_batch]
    virus_contig_onehot_batch_pos = []
    for onehot in virus_onehot_batch:
        onehot_contig = sample_onehot(onehot, virus_contig_len)
        virus_contig_onehot_batch_pos.append(onehot_contig)
    virus_contig_onehot_batch_pos = np.stack(virus_contig_onehot_batch_pos, axis=0)

    virus_contig_onehot_batch_neg = []
    for onehot in virus_onehot_batch:
        onehot_contig = sample_onehot(onehot, virus_contig_len)
        virus_contig_onehot_batch_neg.append(onehot_contig)
    virus_contig_onehot_batch_neg = np.stack(virus_contig_onehot_batch_neg, axis=0)

    virus_batch = np.vstack((virus_contig_onehot_batch_pos, virus_contig_onehot_batch_neg))

    host_fn_list_batch = [host_taxid_to_fasta[host] for host in host_id_batch]
    if low_memory:
        host_onehot_batch = load_host_onehot(host_dir, host_id_batch, host_taxid_to_fasta)
    host_contig_onehot_batch_pos = []
    for host_fn_list in host_fn_list_batch:
        fn = random.choice(host_fn_list)
        if low_memory:
            onehot = host_onehot_batch[fn]
        else:
            onehot = host_onehot[fn]
        onehot_contig = sample_onehot(onehot, host_contig_len)
        host_contig_onehot_batch_pos.append(onehot_contig)
    host_contig_onehot_batch_pos = np.stack(host_contig_onehot_batch_pos, axis=0)

    host_id_batch_neg = []
    for i, taxid in enumerate(host_id_batch):
        taxid_neg = random.choice(host_list)
        # while host_species_taxid_to_taxon_taxid[taxid] == host_species_taxid_to_taxon_taxid[taxid_neg]:
        while host_species_taxid_to_taxon_taxid[taxid_neg] in virus_hostset_mapping[virus_id_batch[i]]:
            taxid_neg = random.choice(host_list)
        host_id_batch_neg.append(taxid_neg)
    host_fn_list_batch = [host_taxid_to_fasta[host] for host in host_id_batch_neg]
    if low_memory:
        host_onehot_batch = load_host_onehot(host_dir, host_id_batch_neg, host_taxid_to_fasta)
    host_contig_onehot_batch_neg = []
    for host_fn_list in host_fn_list_batch:
        fn = random.choice(host_fn_list)
        if low_memory:
            onehot = host_onehot_batch[fn]
        else:
            onehot = host_onehot[fn]
        onehot_contig = sample_onehot(onehot, host_contig_len)
        host_contig_onehot_batch_neg.append(onehot_contig)
    host_contig_onehot_batch_neg = np.stack(host_contig_onehot_batch_neg, axis=0)

    host_batch = np.vstack((host_contig_onehot_batch_pos, host_contig_onehot_batch_neg))

    target_batch = np.zeros(virus_batch.shape[0])
    target_batch[: virus_contig_onehot_batch_pos.shape[0]] = 1

    shuffled_index = np.arange(virus_batch.shape[0])
    np.random.shuffle(shuffled_index)

    virus_batch = virus_batch[shuffled_index, :]
    host_batch = host_batch[shuffled_index, :]
    target_batch = target_batch[shuffled_index]

    return virus_batch, host_batch, target_batch


def train_epoch(model, host_dir, virus_onehot, train_pair_list, batch_size):
    model.train()

    epoch_train_loss = 0

    train_num = len(train_pair_list)
    index = list(range(train_num))
    random.shuffle(index)

    target_whole = []
    output_whole = []

    for batch_start in range(0, train_num, batch_size):
        batch_end = min(batch_start + batch_size, train_num)
        index_batch = index[batch_start:batch_end]
        virus_contig_len = random.choice(possible_contig_length)
        host_contig_len = random.choice(possible_contig_length)
        virus_batch, host_batch, target_batch = construct_batch(
            host_dir,
            host_onehot,
            virus_onehot,
            train_pair_list,
            index_batch,
            virus_contig_len,
            host_contig_len,
            virus_hostset_mapping,
            host_taxid_to_fasta,
            host_species_taxid_to_taxon_taxid,
            host_list=host_list,
        )
        target_whole.append(target_batch.flatten())
        virus_batch = torch.Tensor(virus_batch).to(device)[:, None, :, :]
        host_batch = torch.Tensor(host_batch).to(device)[:, None, :, :]
        target_batch = torch.Tensor(target_batch).to(device)[:, None]

        optimizer.zero_grad()
        with autocast():
            output = model(host_batch, virus_batch)
            loss = criterion(output, target_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_train_loss += loss.item()

        output_whole.append(output.cpu().detach().numpy().flatten())
    target_whole = np.concatenate(target_whole)
    output_whole = np.concatenate(output_whole)
    return epoch_train_loss, target_whole, output_whole


def test(
        model,
        host_dir,
        host_onehot,
        virus_onehot,
        test_pair_list,
        virus_contig_len,
        host_contig_len,
        batch_size,
        virus_hostset_mapping,
        host_taxid_to_fasta,
        host_species_taxid_to_taxon_taxid,
        device="cuda",
        criterion=nn.BCEWithLogitsLoss(),
):
    """Test the input model with the given virus-host pair list and virus and host contig lengths

    :param model: The neural network model for test
    :type model: nn.Module
    :param test_pair_list: The list of tuples indicating virus-host pairs
    :type test_pair_list: List
    :param virus_contig_len: The length used for virus contig
    :type virus_contig_len: int
    :param host_contig_len: The length of host contig
    :type host_contig_len: int
    :return: epoch_test_loss, target_whole, output whole, the latter two outputs are used for subsequent performance evaluations
    :rtype: float, np.array, np.array
    """
    with torch.no_grad():
        model.eval()
        criterion = criterion.to(device)

        epoch_test_loss = 0

        test_num = len(test_pair_list)
        index = list(range(test_num))

        target_whole = []
        output_whole = []

        _, test_host_list = get_virus_host_list(test_pair_list)

        for batch_start in range(0, test_num, batch_size):
            batch_end = min(batch_start + batch_size, test_num)
            index_batch = index[batch_start:batch_end]
            virus_batch, host_batch, target_batch = construct_batch(
                host_dir,
                host_onehot,
                virus_onehot,
                test_pair_list,
                index_batch,
                virus_contig_len,
                host_contig_len,
                virus_hostset_mapping,
                host_taxid_to_fasta,
                host_species_taxid_to_taxon_taxid,
                host_list=test_host_list,
            )
            target_whole.append(target_batch.flatten())
            virus_batch = torch.Tensor(virus_batch).to(device)[:, None, :, :]
            host_batch = torch.Tensor(host_batch).to(device)[:, None, :, :]
            target_batch = torch.Tensor(target_batch).to(device)[:, None]

            with autocast():
                output = model(host_batch, virus_batch)
                loss = criterion(output, target_batch)
            epoch_test_loss += loss.item()

            output_whole.append(output.cpu().detach().numpy().flatten())
        target_whole = np.concatenate(target_whole)
        output_whole = np.concatenate(output_whole)
        return epoch_test_loss, target_whole, output_whole


def run_test(host_dir, virus_onehot, contig_length, batch_size):
    """A helper method for running the test procedure

    :param contig_length: The identical contig length for virus and host
    :type contig_length: int
    """
    epoch_test_loss, test_target, test_output = test(
        model,
        host_dir,
        host_onehot,
        virus_onehot,
        test_pair_list,
        contig_length,
        contig_length,
        batch_size,
        virus_hostset_mapping,
        host_taxid_to_fasta,
        host_species_taxid_to_taxon_taxid,
    )
    epoch_test_loss = epoch_test_loss / test_num
    print(f"Contig Length: {contig_length}")
    print(f"Testing Loss: {epoch_test_loss}")
    roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = evaluate_performance(test_target, test_output)
    print(f"Testing ROC AUC: {roc_auc}")
    print(f"Testing F1 Score: {f1_score}")
    testing_record.append(
        [contig_length, contig_length, epoch_test_loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc]
    )
    return epoch_test_loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc


def evaluate_performance(target, output, threshold=0):
    """Given the target and output of model, return various metrics assessing the performance of the model

    :param target: True labels
    :type target: np.array
    :param output: Output from model
    :type output: np.array
    :param threshold:
    :type threshold: float
    :return: ROAUC, PRAUC, precision, recall, F1, accuracy, MCC
    :rtype: tuple
    """
    roc_auc = metrics.roc_auc_score(target, output)
    pr_auc = metrics.average_precision_score(target, output)
    output[output > 0] = 1
    output[output < 0] = 0
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(
        target, output, beta=1, average="binary"
    )
    acc = metrics.accuracy_score(target, output)
    mcc = metrics.matthews_corrcoef(target, output)
    return roc_auc, pr_auc, precision, recall, f1_score, acc, mcc


def save_run_summary():
    pass


def get_virus_host_set(pair_list):
    virus_set, host_set = zip(*pair_list)
    virus_set = set(virus_set)
    host_set = set(host_set)
    return virus_set, host_set


def get_virus_host_list(pair_list):
    virus_set, host_set = get_virus_host_set(pair_list)
    return list(virus_set), list(host_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--metadata_dir", dest="metadata_dir", help="Directory storing all metadata", default="data")
    parser.add_argument("--host_dir", dest="host_dir", help="Directory containing host one hot matrix")
    parser.add_argument("--virus_dir", dest="virus_dir", help="Directory containing virus one hot matrix")
    parser.add_argument(
        "--taxon_rank",
        dest="taxon_rank",
        choices=["phylum", "class", "order", "family", "genus", "species"],
        default="species",
        help="The taxon rank used to determine negative virus-host pair, default: species",
    )
    parser.add_argument(
        "--reverse_complement",
        dest="reverse_complement",
        action="store_true",
        help="Use reverse complement for encoding the sequence or not",
    )
    parser.add_argument(
        "--train_ratio", type=float, dest="train_ratio", default=0.8, help="Ratio of virus host pairs used for training"
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        dest="test_interval",
        default=1,
        help="Evaluate model on test set for every n epochs",
    )
    parser.add_argument("--epoch", type=int, dest="epoch_num", default=500, help="Epoch number for training")
    parser.add_argument("--learning_rate", "--lr", type=float, dest="lr", default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=32, help="Batch size")
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        choices=["Adagrad", "Adam"],
        default="Adam",
        help="Optimizer used for training the model",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        choices=["both", "base", "codon"],
        default="both",
        help="Which channel is used for the model, default using both channels",
    )
    parser.add_argument(
        "--share_weight",
        dest="share_weight",
        action="store_true",
        help="If let virus and host channel share the same weight in the model",
    )
    parser.add_argument("--no_logging", dest="no_logging", action="store_true", help="Turn off all logging functions")
    parser.add_argument(
        "--no_checkpoint", dest="no_checkpoint", action="store_true", help="If you want to save checkpoint for training"
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        type=int,
        default=1,
        help="Interval for saving training checkpoint",
    )
    parser.add_argument(
        "--continue",
        dest="continue_model_path",
        default=None,
        type=str,
        help="Path to the saved model to continue train the model",
    )
    parser.add_argument("--training_only", dest="training_only", action="store_true", help='Use whole dataset for training')
    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force using CPU if specified")
    parser.add_argument(
        "--no_cudnn_benchmark",
        dest="no_cudnn_bench",
        action="store_true",
        help="Disable CUDNN benchmark, by disabling it might improve reproducibility",
    )
    parser.add_argument(
        "--no_half_precision",
        dest="no_half_precision",
        action="store_true",
        help="Disable half precision training and evaluation with this option, program will slow down",
    )
    parser.add_argument(
        "--no_tensorboard", dest="no_tensorboard", action="store_true", help="Disable Tensorboard logging"
    )
    parser.add_argument(
        "--low_memory",
        dest="low_memory",
        action="store_true",
        help="Use low memory mode when the host size is very large",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Use debug mode")
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        if not args.no_cudnn_bench:
            torch.backends.cudnn.benchmark = True
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")

    if args.continue_model_path:
        model = torch.load(args.continue_model_path)
    else:
        model = VirusCNN(channel=args.channel, rev_comp=args.reverse_complement, share_weight=args.share_weight).to(
            device
        )

    metadata_dir = args.metadata_dir
    # Load virus host pairs
    # The list stores (virus RefSeqID, host species taxid)
    virus_host_pair = util.load_obj(os.path.join(metadata_dir, "virus_host_pair.pkl"))
    # The strains of a species taxid were randomly chosen, this loads the pre-chosen strain for each species
    host_taxid_to_fasta = util.load_obj(os.path.join(metadata_dir, "bacteria_taxid_to_filename.pkl"))

    if not args.no_logging:
        dir_name = f'taxon_{args.taxon_rank}_channel_{args.channel}_lr{args.lr}_shareweight_{args.share_weight}_revcomp_{args.reverse_complement}_{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())}'

        # Store debug log into different file
        if not args.debug:
            training_info_dir = os.path.join(metadata_dir, "training_info", dir_name)
        else:
            training_info_dir = os.path.join(metadata_dir, "training_info", "debug", dir_name)
        os.makedirs(training_info_dir, exist_ok=True)

        if not args.no_tensorboard:
            # If tensorboard log is not disabled
            if not args.debug:
                writer = CorrectedSummaryWriter(
                    os.path.join(metadata_dir, "training_info", "tensorboard_record", dir_name)
                )
            else:
                writer = CorrectedSummaryWriter(
                    os.path.join(metadata_dir, "training_info", "debug", "tensorboard_record", dir_name)
                )

            hparam = {
                "lr": args.lr,
                "optimizer": args.optimizer,
                "channel": args.channel,
                "rev_comp": args.reverse_complement,
                "share_weight": args.share_weight,
                "batch_size": args.batch_size,
            }
            writer.add_hparams(hparam, {"hparam/auc": 0})
    else:
        args.no_checkpoint = True

    if args.debug:
        # Only use a subset of the dataset when debugging
        virus_host_pair = virus_host_pair[:50]

    # Extract virus and host sets
    virus_set, host_set = zip(*virus_host_pair)
    virus_set = set(virus_set)
    virus_list = list(virus_set)
    host_set = set(host_set)
    host_list = list(host_set)

    # Load one hot matrices
    virus_onehot = load_virus_onehot(args.virus_dir, virus_set)
    if not args.low_memory:
        # When in large memory mode load all host onehot matrices into memory
        host_onehot = load_host_onehot(args.host_dir, host_set, host_taxid_to_fasta)

    # Split training and test set
    train_pair_list, test_pair_list = train_test_split(virus_host_pair, training_only=args.training_only)
    train_num = len(train_pair_list)
    test_num = len(test_pair_list)

    # Save the train test split for later testing
    if not args.no_logging:
        util.save_obj((train_pair_list, test_pair_list), os.path.join(training_info_dir, "train_test_split.pkl"))

    # Get the corresponding taxid at requested taxonomy level, the taxonomy level is specified as commandline argument
    # This is used to construct the negative set
    host_species_taxid_to_taxon_taxid = {}
    for host in host_set:
        host_species_taxid_to_taxon_taxid[host] = util.get_rank(host, args.taxon_rank)

    virus_hostset_mapping = util.tuple_list_to_dict_set(virus_host_pair)
    virus_hostset_mapping = {
        key: set(host_species_taxid_to_taxon_taxid[taxid] for taxid in virus_hostset_mapping[key])
        for key in virus_hostset_mapping.keys()
    }

    # Configure possible contig length for the model
    possible_contig_length = [200, 500, 1001, 2000, 3002, 4001, 5000]

    if not args.no_checkpoint:
        checkpoint_dir_path = os.path.join(training_info_dir, "checkpoint", )
        os.makedirs(checkpoint_dir_path, exist_ok=True)

    if not args.no_half_precision:
        scaler = GradScaler()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == "Adagrad":
        torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=5e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().to(device)

    epoch_num = args.epoch_num

    training_record = []
    testing_record = []

    for epoch in range(epoch_num):
        start_time = time.time()
        epoch_train_loss, train_target, train_output = train_epoch(
            model, args.host_dir, virus_onehot, train_pair_list, args.batch_size
        )
        end_time = time.time()
        epoch_train_loss = epoch_train_loss / train_num
        print(f"Epoch: {epoch + 1}")
        print(f"Epoch time: {end_time - start_time:.5f}s")
        print(f"Training Loss: {epoch_train_loss}")
        roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = evaluate_performance(train_target, train_output)
        print(f"Training ROC AUC: {roc_auc}")
        print(f"Training F1 Score: {f1_score}")
        training_record.append([epoch_train_loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc])
        if not args.no_logging and not args.no_tensorboard:
            writer.add_scalar("Train/loss", epoch_train_loss, epoch)
            writer.add_scalar("Train/roauc", roc_auc, epoch)
            writer.add_scalar("Train/prauc", pr_auc, epoch)
            writer.add_scalar("Train/acc", acc, epoch)
            writer.add_scalar("Train/F1", f1_score, epoch)
            writer.add_scalar("Train/MCC", mcc, epoch)

        if (epoch + 1) % args.test_interval == 0 and not args.training_only:
            # run_test(args.host_dir, virus_onehot, possible_contig_length[0], args.batch_size)
            # run_test(args.host_dir, virus_onehot, possible_contig_length[-1], args.batch_size)
            loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = run_test(
                args.host_dir, virus_onehot, 200, 200
            )
            if not args.no_logging and not args.no_tensorboard:
                writer.add_scalar("Test_200/loss", loss, epoch)
                writer.add_scalar("Test_200/roauc", roc_auc, epoch)
                writer.add_scalar("Test_200/prauc", pr_auc, epoch)
                writer.add_scalar("Test_200/acc", acc, epoch)
                writer.add_scalar("Test_200/F1", f1_score, epoch)
                writer.add_scalar("Test_200/MCC", mcc, epoch)
            loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = run_test(
                args.host_dir, virus_onehot, 5000, 5000
            )
            if not args.no_logging and not args.no_tensorboard:
                writer.add_scalar("Test_5000/loss", loss, epoch)
                writer.add_scalar("Test_5000/roauc", roc_auc, epoch)
                writer.add_scalar("Test_5000/prauc", pr_auc, epoch)
                writer.add_scalar("Test_5000/acc", acc, epoch)
                writer.add_scalar("Test_5000/F1", f1_score, epoch)
                writer.add_scalar("Test_5000/MCC", mcc, epoch)
            # test_metrics = {'loss':{}, 'roauc':{}, 'prauc':{}, 'acc':{}, 'f1':{}, 'mcc':{}}
            # for i in [200, 5000]:
            #     loss, roc_auc, pr_auc, precision, recall, f1_score, acc, mcc = run_test(args.host_dir, virus_onehot, i, i)
            #     test_metrics['loss'][str(i)] = loss
            #     test_metrics['roauc'][str(i)] = roc_auc
            #     test_metrics['prauc'][str(i)] = pr_auc
            #     test_metrics['acc'][str(i)] = acc
            #     test_metrics['f1'][str(i)] = f1_score
            #     test_metrics['mcc'][str(i)] = mcc
            # if not args.no_logging and not args.no_tensorboard:
            #     writer.add_scalars('Test/loss', test_metrics['loss'], epoch)
            #     writer.add_scalars('Test/roauc', test_metrics['roauc'], epoch)
            #     writer.add_scalars('Test/prauc', test_metrics['prauc'], epoch)
            #     writer.add_scalars('Test/acc', test_metrics['acc'], epoch)
            #     writer.add_scalars('Test/F1', test_metrics['f1'], epoch)
            #     writer.add_scalars('Test/MCC', test_metrics['mcc'], epoch)

        if not args.no_checkpoint and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir_path, f"epoch_{epoch:05d}")
            torch.save(model, checkpoint_path)

    if not args.no_logging:
        training_record = pd.DataFrame(
            training_record, columns=["Loss", "ROC AUC", "PR AUC", "Precision", "Recall", "F1", "Acc", "MCC"]
        )
        training_record.to_pickle(os.path.join(training_info_dir, "training_record.pkl"))
        if not args.training_only:
            testing_record = pd.DataFrame(
                testing_record,
                columns=["Virus Len", "Host Len", "Loss", "ROC AUC", "PR AUC", "Precision", "Recall", "F1", "Acc", "MCC"],
            )
            testing_record.to_pickle(os.path.join(training_info_dir, "testing_record.pkl"))
        # Save the final model
        checkpoint_path = os.path.join(training_info_dir, f"final_model")
        torch.save(model, checkpoint_path)
