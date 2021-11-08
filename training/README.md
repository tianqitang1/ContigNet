# Training

## Data preparation

The 

## Training model

To train the model, run

``` bash
python train_model.py --host_dir data/host_onehot --virus_dir data/virus_onehot --train_ratio 0.8 --test_interval 50 --checkpoint_interval 100 --epoch 1000 --batch_size 64 --share_weight
```

## Testing model