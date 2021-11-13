#!/bin/bash
python split_virus_fasta.py
sbatch cdhit.job
python generate_host_link_txt.py
cat ftp_links.txt | xargs -n 1 -P 12 wget -q -P ../data/host_fasta
python fasta_to_onehot.py --input_dir ../data/host_fasta/ --output_dir ../data/host_onehot/
python fasta_to_onehot.py --input_dir ../data/virus_fasta/ --output_dir ../data/virus_onehot/
python generate_virus_host_pair_cdhit.py ../data/clstr