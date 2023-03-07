from itertools import groupby
from operator import itemgetter
import os
import numpy as np
import csv
import pickle
import statsmodels.api as sm
from typing import List, Optional
import pandas as pd
from ete3 import NCBITaxa
from Bio import SeqIO
import math

ncbi = NCBITaxa()
NT_DICT = {"A": 0, "C": 1, "G": 2, "T": 3}


def save_obj(obj, name):
    """Save an object on given path

    :param obj: Object to be save
    :type obj: Any
    :param name: Path of the saved object
    :type name: str
    """
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str):
    """Load a saved object from hard disk

    :param name: Paht of the saved object
    :type name: str
    :return: The saved object
    :rtype: Any
    """
    with open(name, "rb") as f:
        return pickle.load(f)


def lowess_adjustment(y: List[np.ndarray], x, target):
    lowess = sm.nonparametric.lowess

    z = lowess(y[0], x, return_sorted=False)
    y_new = []
    for y_ in y:
        y_new.append(y_ + target - z)

    return y_new


def count_nucleotide(file):
    """Count occurrence of nucleotide in given file

    :param file: Path to fasta file
    :type file: str
    :return: Count of A, C, G, T in given file
    :rtype: np.ndarray
    """
    nucleotides = ["A", "C", "G", "T"]
    try:
        f = open(file)
    except UnicodeDecodeError:
        print(file)
        return np.ones(4)
    occ = np.zeros(4)
    try:
        for line in f.readlines():
            if line[0] == ">":
                continue
            l = line.upper()
            for i, n in enumerate(nucleotides):
                occ[i] += l.count(n)
        return occ
    except UnicodeDecodeError:
        print(file)
        return np.ones(4)


def split_fasta_by_size(input_path, output_dir_path, size=5000):
    """Split a given fasta file and save the split fasta into the output directory

    :param input_path: path to the fasta file
    :param output_dir_path: path of the output directory
    :param size:
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(input_path) as fasta_handle:
        whole_seq = ""
        for line in fasta_handle.readlines():
            if not line.startswith(">"):
                whole_seq += line
    split_num = np.ceil(len(whole_seq) / size).astype(int)
    filename_length = len(str(split_num))
    for i in range(split_num):
        output_fasta_filename = "{}.fa".format(str(i).zfill(filename_length))
        with open(os.path.join(output_dir_path, output_fasta_filename), "w") as output_handle:
            output_handle.write(">{}\n".format(i))
            if i != split_num - 1:
                output_handle.write(whole_seq[size * i : size * (i + 1)])
            else:
                output_handle.write(whole_seq[size * i :])


def remove_plasmid_seq(input_dir_path, output_dir_path):
    """
    Given a series of assemblies of bacteriea, remove sequence with plasmid keyword in their description
    :param input_dir_path:
    :param output_dir_path:
    :return:
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    fasta_filename_list = os.listdir(input_dir_path)

    def filter_fun(fn: str):
        return fn.endswith(".fa") or fn.endswith(".fasta") or fn.endswith(".fna")

    fasta_filename_list = filter(filter_fun, fasta_filename_list)
    for fasta_file in fasta_filename_list:
        input_fasta_path = os.path.join(input_dir_path, fasta_file)
        input_handle = open(input_fasta_path)
        output_fasta_path = os.path.join(output_dir_path, fasta_file)
        output_handle = open(output_fasta_path, "w")

        seq_iter = SeqIO.parse(input_handle, format="fasta")
        seq_list = []
        for seq in seq_iter:
            if "plasmid" not in seq.description.lower():
                seq_list.append(seq)
        SeqIO.write(seq_list, output_handle, "fasta")

        input_handle.close()
        output_handle.close()


from Bio.Blast.Applications import NcbiblastnCommandline
from io import StringIO


def blast_single(fasta_path, db_path):
    """Blast the given fasta file to a NCBI database and calculate the percentage of aligned regions

    :param fasta_path:
    :param db_path:
    :return:
    """

    def cal_perc(x):
        indicator = [0] * x[4]
        for i in range(len(x[2])):
            indicator[x[2][i] : x[3][i]] = [1] * (x[3][i] - x[2][i] + 1)
        return sum(indicator)

    blastn_cline = NcbiblastnCommandline(
        query=fasta_path, db=db_path, outfmt="6 qacc sacc qstart qend qlen", num_threads=8
    )
    stdout, stderr = blastn_cline()
    if stdout == "":
        blast_success = False
        return blast_success, None
    blastn_output = StringIO(stdout)
    df = pd.read_csv(blastn_output, sep="\t", header=None)

    seqacc_to_hostacc_dict = load_obj("seqacc_to_hostacc.pkl")
    df[1] = [seqacc_to_hostacc_dict[k] for k in list(df[1])]

    df_blast_positions = df.groupby([0, 1]).agg({2: lambda x: tuple(x - 1), 3: lambda x: tuple(x - 1), 4: min})
    df_blast_positions.index = df_blast_positions.index.droplevel()
    query_len = len(SeqIO.parse(fasta_path, "fasta").__next__())
    df_blast_perc = df_blast_positions.apply(lambda x: cal_perc(x), axis=1) / query_len
    sr_blast = df_blast_perc.groupby(level=0, sort=False).apply(sum)
    blast_success = True
    return blast_success, sr_blast


def blast_dir_to_db(query_dir, db_path):
    """
    Given query directory with fasta and subject directory with fasta,
    :param query_dir:
    :param subject_dir:
    :return:
    """
    query_list = os.listdir(query_dir)
    query_list.sort()
    blast_results = {}
    for q in query_list:
        print("Calculating blast score for ", q)
        query_path = os.path.join(query_dir, q)
        blast_single_result = blast_single(query_path, db_path)
        blast_results[q[:-3]] = blast_single_result
    return blast_results


def cosine_similarity(f1, f2):
    n1 = np.linalg.norm(f1, axis=1, keepdims=True)
    n2 = np.linalg.norm(f2, axis=1, keepdims=True)
    prod = np.dot(f1, f2.T)
    return prod / (np.dot(n1, n2.T))


def check_directory_content(directory_path):
    for file in os.listdir(directory_path):
        if not file.endswith(".fasta") or file.endswith(".fa"):
            raise RuntimeError("Contains non-fasta file within provided directory.")
    return True


def construct_data(index, *features):
    data = []
    for feature in features:
        data.append(feature[np.arange(feature.shape[0]), index, np.newaxis])
    return np.concatenate(data, axis=1)


def nt2index(nucleotide: str):
    """
    Given a nucleotide character, return its corresponding index.

    Parameters
    ----------
    nucleotide : str
        The nucleotide character to convert to an index.

    Returns
    -------
    int
        The index of the nucleotide. If the nucleotide is not A, C, G, or T,
        returns 4.

    """
    nucleotide = nucleotide.upper()
    return NT_DICT.get(nucleotide, 4)


def seq2intseq(seq: str, contig_length: Optional[int] = None) -> np.ndarray:
    """
    Convert genome sequence into indexed sequence

    Parameters
    ----------
    seq : str
        Genome sequence
    contig_length : int, optional
        If designated, sample a subsequence with length contig_length from the original sequence, defaults to None

    Returns
    -------
    np.ndarray
        Indexed sequence as a numpy array
    """
    if contig_length:
        if len(seq) <= contig_length:
            seq_int = np.zeros(contig_length)
            seq_int[: len(seq)] = list(map(nt2index, str(seq)))
        else:
            start_idx = np.random.randint(len(seq) - contig_length)
            seq_int = np.array(list(map(nt2index, str(seq[start_idx : start_idx + contig_length]))))
        return seq_int
    else:
        return np.array(list(map(nt2index, str(seq))))


def int2onehot(array: np.ndarray) -> np.ndarray:
    """
    Convert an integer array to one-hot encoded array.

    Parameters
    ----------
    array : np.ndarray
        A 1D integer array to be one-hot encoded.

    Returns
    -------
    np.ndarray
        A 2D one-hot encoded array.

    Examples
    --------
    >>> a = np.array([0, 1, 2, 3])
    >>> int2onehot(a)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    """
    # n = np.max(array) + 1
    n = 4
    return np.eye(int(n))[array.astype(int)]


def seq2onehot(seq: str, contig_length: Optional[int] = None) -> np.ndarray:
    """
    Convert a DNA sequence string to a one-hot encoded array.

    Parameters
    ----------
    seq : str
        A DNA sequence string to be one-hot encoded.
    contig_length : int or None, optional
        The length of the input contig. If provided, the input sequence
        will be padded with subsampled with this length, by default None.

    Returns
    -------
    np.ndarray
        A 2D one-hot encoded array.

    Examples
    --------
    >>> seq = 'ATCG'
    >>> seq2onehot(seq)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    """
    assert isinstance(seq, str), "Input should be str"
    int_seq = seq2intseq(seq, contig_length)
    onehot = int2onehot(int_seq)
    return onehot


def fasta2onehot(path: str, contig_length: Optional[int] = None) -> np.ndarray:
    """
    Convert a multi-sequence fasta file to a one-hot encoded array.

    Parameters
    ----------
    path : str
        The path to the input fasta file.
    contig_length : int or None, optional
        The length of the input contigs. If provided, the input sequences
        will be subsampled with this length, by default None.

    Returns
    -------
    np.ndarray
        A 2D one-hot encoded array.

    Examples
    --------
    >>> path = 'path/to/file.fasta'
    >>> fasta2onehot(path)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           ...
           [1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    """
    seq = [str(s.seq) for s in SeqIO.parse(path, "fasta")]
    seq = "".join(seq)
    return seq2onehot(seq, contig_length)


def get_rank(taxid, rank):
    """
    Given a taxonomic identifier and a requested taxonomy rank, returns the
    identifier of the taxonomy at the requested rank.

    Parameters
    ----------
    taxid : int
        The taxonomic identifier of the desired taxonomy.
    rank : str
        The requested taxonomy rank.

    Returns
    -------
    Union[int, None]
        The taxonomic identifier of the taxonomy at the requested rank. If the
        requested rank is not found, None is returned.

    Examples
    --------
    >>> get_rank(12345, 'species')
    54321
    >>> get_rank(67890, 'genus')
    9876
    """
    lineage = ncbi.get_lineage(taxid)
    ranks = ncbi.get_rank(lineage)
    ranks = {ranks[key]: key for key in ranks.keys()}
    if rank in ranks.keys():
        return ranks[rank]
    else:
        return None


def generate_negative_sample(taxid_list, taxid_pool, rank):
    taxid_ranked_list = [ncbi.get_lineage()]
    negative_list = []


def tuple_list_to_dict_set(l):
    """
    Given a tuple list, returns a dictionary with the first element of the tuple
    as key and set of the second element as value.
    Parameters
    ----------
    l : list of tuples
        The input list of tuples.

    Returns
    -------
    dict
        A dictionary with the first element of the tuple as key and set of the
        second element as value.

    Examples
    --------
    >>> l = [('a', 1), ('b', 2), ('a', 3)]
    >>> tuple_list_to_dict(l)
    {'a': {1, 3}, 'b': {2}}

    Notes
    -----
    If multiple tuples have the same first element, the second elements are
    added to the same set for that key.
    """
    return dict((k, set([v[1] for v in itr])) for k, itr in groupby(l, itemgetter(0)))

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
        host_train_genus.add(get_rank(host, host_taxon_level))
    for host in set(list(zip(*init_test_pair_list))[1]):
        host_test_genus.add(get_rank(host, host_taxon_level))
    for host in set(list(zip(*init_train_pair_list))[1]):
        if get_rank(host, host_taxon_level) in host_test_genus:
            host_train_test_intersection.add(host)
    for host in set(list(zip(*init_test_pair_list))[1]):
        if get_rank(host, host_taxon_level) in host_train_genus:
            host_train_test_intersection.add(host)