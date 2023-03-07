import sys
sys.path.append('..')

import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from ete3 import NCBITaxa
import ContigNet.util as util

import argparse

# parser = argparse.ArgumentParser(description='Preprocessing for running on CARC')

data_dir = os.path.join('..', 'data')

# Parse metadata
metadata = pd.read_table(os.path.join(data_dir, 'virushostdb.tsv'))
metadata.dropna(subset=['host tax id'], inplace=True)
metadata.reset_index(drop=True, inplace=True)

hostid_list = list(metadata['host tax id'])
hostid_list = list(set([i for i in hostid_list if not np.isnan(i)]))
hostid_list.sort()


# Select host at species level
ncbi = NCBITaxa()
ranks = ncbi.get_rank(hostid_list)

selected_ranks = {'species'}
taxid_selected = []
for rank in ranks.keys():
    if ranks[rank] in selected_ranks:
        taxid_selected.append(rank)
        
metadata_selected = metadata[metadata['host tax id'].isin(taxid_selected)]
util.save_obj(metadata_selected, os.path.join(data_dir, 'metadata_selected.pkl'))

# Generate list of files to download
import math
import random

from ContigNet.util import seq2intseq, int2onehot

# Parse GanBank summary
summary_path = '../data/assembly_summary.txt'
bacteria_summary = pd.read_table(summary_path)
bacteria_summary = bacteria_summary[['assembly_accession', 'taxid', 'species_taxid', 'seq_rel_date', 'asm_name', 'ftp_path']]
bacteria_taxid_to_ftp = bacteria_summary.groupby(['species_taxid'])['ftp_path'].apply(list)

# Find species that are available on GenBank
ncbi_taxid = set(bacteria_summary.species_taxid)
database_taxid = set(taxid_selected)
taxid_intersection = ncbi_taxid.intersection(database_taxid)
taxid_intersection_list = list(taxid_intersection)

link_list = []
bacteria_taxid_to_filename = {} # Given a bacteria taxid, return list of downloaded filename of the assemblies

max_file_num = 10 # Max number of randomly selected assemblies for a species
for taxid in taxid_intersection_list:
    random.shuffle(bacteria_taxid_to_ftp[taxid])
    bacteria_taxid_to_ftp[taxid] = bacteria_taxid_to_ftp[taxid][:max_file_num]
    link_list.extend(bacteria_taxid_to_ftp[taxid])

    bacteria_taxid_to_filename[taxid] = [f'{link.split("/")[-1]}_genomic.fna' for link in bacteria_taxid_to_ftp[taxid]]

# Write download links to file
with open(os.path.join(data_dir, 'ftp_links.txt'), 'w', newline='\n') as f:
    for link in link_list:
        file_link = f'{link}/{link.split("/")[-1]}_genomic.fna.gz'
        f.write(file_link)
        f.write('\n')  


# Save presorted virus and bacteria information          
util.save_obj(bacteria_taxid_to_filename, os.path.join(data_dir, 'bacteria_taxid_to_filename.pkl'))

# database_taxid = set(metadata_selected.taxon_species_id)