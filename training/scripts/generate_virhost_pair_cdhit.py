import sys
sys.path.append('..')

import util
import pandas as pd
import os

data_dir = os.path.join('..', 'data')

input_path = sys.argv[1]

with open(input_path, 'r') as f:
    in_cluster = False
    cluster_head = False
    virus_cluster_mapping = {}
    cluster_head_name = ''
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('>'):
            cluster_head = True
            continue
        else:
            name = line.split()[2].strip('>.')
            if cluster_head:
                cluster_head_name = name
                virus_cluster_mapping[name] = cluster_head_name
                cluster_head = False
            else:
                virus_cluster_mapping[name] = cluster_head_name

metadata_selected = util.load_obj('../data/metadata_selected.pkl')

# Parse GanBank summary
summary_path = '../data/assembly_summary.txt'
bacteria_summary = pd.read_table(summary_path)
bacteria_summary = bacteria_summary[['assembly_accession', 'taxid', 'species_taxid', 'seq_rel_date', 'asm_name', 'ftp_path']]
bacteria_taxid_to_ftp = bacteria_summary.groupby(['species_taxid'])['ftp_path'].apply(list)

# Find species that are available on GenBank
ncbi_taxid = set(bacteria_summary.species_taxid)

virus_host_pair = []
for _, row in metadata_selected.iterrows():
    for virus in row['refseq id'].split(', '):
        if row['host tax id'] in ncbi_taxid:
            virus_host_pair.append((virus_cluster_mapping[virus], int(row['host tax id'])))
util.save_obj(virus_host_pair, os.path.join(data_dir, 'virus_host_pair.pkl'))
