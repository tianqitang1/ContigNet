# Training

## Data preparation

The data used in the paper "Phage-bacteria contig interaction prediction with convolutional neural network" can be downloaded from [GoogleDrive](). The downloaded archive file should be released to ```data``` directory.

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