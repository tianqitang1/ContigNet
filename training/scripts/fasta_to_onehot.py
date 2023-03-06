import sys
sys.path.append('..')

import numpy as np
import ContigNet.util as util
import os
from Bio import SeqIO

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', dest='input_dir')
parser.add_argument('--output_dir', dest='output_dir')

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

fasta_ext = ('.fna', '.fasta', '.fa')

input_fasta = os.listdir(input_dir)
input_fasta = filter(lambda x: x.endswith(fasta_ext), input_fasta)

for fasta in input_fasta:
    fasta_path = os.path.join(input_dir, fasta)
    seq_records = list(SeqIO.parse(fasta_path, 'fasta'))
    seq = [str(record.seq) for record in seq_records if not 'plasmid' in record.description]
    seq = ''.join(seq)

    onehot = util.seq2onehot(seq)
    onehot = onehot[:, :4]

    output_onehot_fn = fasta + '.npy'
    output_onehot_path = os.path.join(output_dir, output_onehot_fn)
    
    np.save(output_onehot_path, onehot.astype(np.int8))