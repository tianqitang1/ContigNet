# Training

## Data availability

The data used in the paper "Phage-bacteria contig interaction prediction with convolutional neural network" can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1VJARCYazlk7C9IbJF62mfP7vdTP8GwiE/view?usp=sharing). The downloaded archive file should be extracted to ```data``` directory.

## Prepeare your own data
You can also prepare the data from scratch. First create and enter the directory with
``` bash
mkdir data
cd data
```

Then download the Virus-Host DB files with
``` bash
wget https://www.genome.jp/ftp/db/virushostdb/virushostdb.tsv
wget https://www.genome.jp/ftp/db/virushostdb/virushostdb.formatted.genomic.fna.gz
tar -xzvf virushostdb.formatted.genomic.fna.gz
```

```scripts``` directory contains necessary scripts to process the data. 
First split the sequences into individual files for following processes.
``` bash
cd scripts
python split_virus_fasta.py --input {path/to/virushostdb.formatted.genomic.fna} --output {output/directory}
```

The virus sequences need to be clustered with [CD-HIT](http://weizhong-lab.ucsd.edu/cd-hit/) to remove redundancy, after installation, run
``` bash
./cd-hit-est -i {path/to/virushostdb.formatted.genomic.fna} -o {output/directory} -c 0.95 -G 0 -aS 0.5 -M 0 -T 32 -n 3
```

After clustering the phages, the original phage-host pairs should then be replaced with phage cluster-host, and a random phage is chosen to represent the whole cluster.
This step is done by running the following command
``` bash
python generage_virhost_pair_cdhit.py
```

With host taxid, 10 assemblies is randomly selected to represent the corresponding taxid.
First download the assembly summary from [NCBI](https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt) and put it to ```data``` directory, then run
``` bash
python generate_host_link_txt.py
cat ftp_links.txt | xargs -n 1 -P 12 wget -q -P ../data/host_fasta
```
to download the host assemblies.

The final step is to convert the sequence files to one-hot files so it can be used by the training program, which can be done by running
``` bash 
python fasta_to_onehot.py --input_dir {input} --output_dir {output}
```

## Training model

To train the model, run

``` bash
python train_model.py --host_dir data/host_onehot --virus_dir data/virus_onehot --train_ratio 0.8 --test_interval 50 --checkpoint_interval 100 --epoch 1000 --batch_size 64 --share_weight
```

Related information and models will be stored under ```training_info``` directory

## Testing model

To test ContigNet on validation set, run 
``` bash
python test.py --train_info_path data/training_info/{name_of_desired_run}
```