## Training
### Hooktheory-A2S
To train a model on a dataset:
```bash
python train.py --ds_location /path/to/dataset --batch_size 8 --tokeniser word  --patience 5 --encoder cnn  --learning_rate 5e-5 --weight_decay 1e-3 --ff_dim_multiplier 4 --label_smoothing 0.1
```

To use the MuQ encoder, first preprocess the audio:
```bash
python preprocess_muq.py --input-dataset /path/to/dataset --output-path /path/to/output 
```
Then train with:
```bash
python train.py --ds_location /path/to/dataset --batch_size 8 --tokeniser word  --patience 5 --encoder preprocessed_muq  --learning_rate 5e-5 --weight_decay 1e-3 --ff_dim_multiplier 4 --label_smoothing 0.1
```

Valid batch sizes are 1, 2, 4, and 8. For batch sizes below 8, gradient accumulation is used automatically to maintain an effective batch size of 8.


### Quartets
For the quartets dataset use tokeniser original. This will perform the cleaning the original paper did on quartets dataset as well as using word level tokenisation.
```bash
python train.py --batch_size 8 --patience 5 --encoder preprocessed_muq --ds_location LOCATION --learning_rate 5e-5 --weight_decay 1e-3 --ff_dim_multiplier 4 --tokeniser original
```

## Metrics
To obtain metrics on a model for the quartets dataset run
```bash
python run_metrics_quartets.py --ds_location DS_LOCATION --checkpoint_path CHECKPOINT_PATH --encoder ENCODER
```

To obtain metrics on a model for the hooktheory-a2s dataset run
```bash
python run_metrics.py --ds_location DS_LOCATION --checkpoint_path CHECKPOINT_PATH --encoder ENCODER
```