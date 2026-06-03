## Docker setup
First run the docker container 
```bash
sudo -E docker run -it --network host --ipc=host   --rm     - -v  DS_LOCATION:DS_LOCATION a2s-transformer   bash
```
Once inside this docker container run 
```bash
pip install --upgrade     torch==2.6.0     torchvision==0.21.0     torchaudio==2.6.0     torchcodec==0.2     --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
pip install datasets==3.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com 
pip install muq
```

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

## Sampling
To run inference and view model predictions sample-by-sample:

For the Hooktheory-A2S dataset:
```bash
python sample.py --ds_location DS_LOCATION --checkpoint_path CHECKPOINT_PATH --tokeniser word > samples
```

For the Quartets dataset:
```bash
python sample.py --ds_location DS_LOCATION --checkpoint_path CHECKPOINT_PATH --tokeniser original > quartet_samples
```

This will iterate through the test set, printing ground truth and predicted tokens for each sample along with Verovio Humdrum Viewer links to visualise the output as sheet music. You can use the parameter --number to limit the number of samples printed,


## Dataset and checkpoint locations
#### normal hooktheory with cnn (1024 baseline)
dataset - /home/eoin/datasets/final_hooktheory   
checkpoint_path - /workspace/weights/cnn/final_hooktheory-v2.ckpt

#### hooktheory with preprocessed muq
ds_location - /home/eoin/datasets/final_hooktheory_precomputed_muq/   
checkpoint_path - /workspace/weights/preprocessed_muq/final_hooktheory_precomputed_muq-v1.ckpt 


#### hooktheory with augmentation + preprocessed muq
ds_location - /home/eoin/datasets/final_augmented_precomputed_muq/   
checkpoint_path- /workspace/weights/preprocessed_muq/final_augmented_precomputed_muq-v1.ckpt