## Docker setup
First run the docker container.
```bash
sudo -E docker run -it --network host --ipc=host --rm \
  -v /path/on/host/datasets:/workspace/datasets \
  -v /path/on/host/exports:/workspace/exports \
  a2s-transformer bash
```

Use bind mounts if you want datasets and exported `.krn` files to persist outside the container.
In `-v host_path:container_path`, the left side is the path on your machine and the right side is the path seen inside Docker.

When you run the scripts inside Docker:

- pass the container path to `--ds_location`, for example `/workspace/datasets/HOOKTHEORY_DATASET`
- pass a mounted container path to `--output_dir`, for example `/workspace/exports/hooktheory_run`

If `--output_dir` points somewhere that is not mounted, the files will stay inside the container filesystem and disappear when the container is removed.

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

To export quartets `.krn` predictions for later scoring run
```bash
python run_metrics_quartets.py \
  --ds_location /workspace/datasets/QUARTETS_DATASET \
  --checkpoint_path CHECKPOINT_PATH \
  --encoder ENCODER \
  --output_dir /workspace/exports/quartets_run \
  --compute_scores False
```

To obtain metrics on a model for the hooktheory-a2s dataset run
```bash
python run_metrics.py --ds_location DS_LOCATION --checkpoint_path CHECKPOINT_PATH --encoder ENCODER
```

To export hooktheory-a2s `.krn` predictions for later scoring run
```bash
python run_metrics.py \
  --ds_location /workspace/datasets/HOOKTHEORY_DATASET \
  --checkpoint_path CHECKPOINT_PATH \
  --encoder ENCODER \
  --output_dir /workspace/exports/hooktheory_run \
  --compute_scores False
```

If you only want predictions and do not want to save matching ground truth files:
```bash
python run_metrics.py \
  --ds_location /workspace/datasets/HOOKTHEORY_DATASET \
  --checkpoint_path CHECKPOINT_PATH \
  --encoder ENCODER \
  --output_dir /workspace/exports/hooktheory_pred_only \
  --compute_scores False \
  --save_ground_truth False
```

When `--output_dir` is set, predictions are written under `OUTPUT_DIR/pred`. If `--save_ground_truth` is enabled, matching reference files are also written under `OUTPUT_DIR/true`.

Example with Docker:
```bash
sudo -E docker run -it --network host --ipc=host --rm \
  -v /path/on/host/datasets:/workspace/datasets \
  -v /path/on/host/exports:/workspace/exports \
  a2s-transformer bash
```

Then inside the container run:
```bash
python run_metrics.py \
  --ds_location /workspace/datasets/HOOKTHEORY_DATASET \
  --checkpoint_path CHECKPOINT_PATH \
  --encoder ENCODER \
  --output_dir /workspace/exports/hooktheory_run \
  --compute_scores False
```

With this setup, files written inside the container to `/workspace/exports/hooktheory_run` will be preserved on the host under `/path/on/host/exports/hooktheory_run`.

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
