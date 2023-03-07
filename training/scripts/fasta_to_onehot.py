import sys
sys.path.append('..')

import numpy as np
import ContigNet.util as util
import os
from Bio import SeqIO

import argparse

def fasta_to_onehot_dir(input_dir, output_dir):

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

def fasta_to_onehot_file(input_path, output_path):
    seq_records = list(SeqIO.parse(input_path, 'fasta'))
    seq = [str(record.seq) for record in seq_records if not 'plasmid' in record.description]
    seq = ''.join(seq)

    onehot = util.seq2onehot(seq)
    onehot = onehot[:, :4]

    np.save(output_path, onehot.astype(np.int8))

def main(args):
    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        fasta_to_onehot_dir(args.input, args.output)
    else:
        fasta_to_onehot_file(args.input, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert fasta file to onehot encoding')
    parser.add_argument('input', help='Input fasta file or directory')
    parser.add_argument('output', help='Output onehot encoding file or directory')
    args = parser.parse_args()
