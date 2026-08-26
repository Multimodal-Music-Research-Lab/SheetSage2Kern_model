# Sheetsage-A2S

<div>
  <a href='#'><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white"></a>
  <a href='https://arxiv.org/abs/2608.06165'><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2608.06165-%23b31b1b?logo=arxiv&link=https%3A%2F%2Farxiv.org%2F"></a>
  <a href='https://huggingface.co/collections/MMR-Lab/sheetsage-a2s'><img alt="Static Badge" src="https://img.shields.io/badge/huggingface-HookKern-%23FFD21E?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2F"></a>
  <a href='https://pytorch.org/'><img alt="Static Badge" src="https://img.shields.io/badge/framework-PyTorch-%23EE4C2C?logo=pytorch"></a>
  <a href='LICENSE'><img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green"></a>
</div>


This is the official repository for the paper *"**Audio-to-Score Transcription using Pre-trained Features, Data Augmentation, and the New SheetSage-A2S Dataset**"*, which has recently been accepted by ACMMM 2026.

In this repo, the following are released:

- **HookKern**: a transformer for audio-to-score transcription that maps raw audio to Humdrum `**kern` notation.
- **Encoder variants**: a CNN encoder variant and a [MuQ](https://github.com/tencent-ailab/MuQ)-based variant.
- Training, evaluation, and inference code. 


## Overview

We introduce **SheetSage-A2S**, the first audio-to-score (A2S) dataset for popular music: 61 hours of real, commercially released audio paired with `**kern` lead-sheet encodings across 6,066 songs. Unlike existing A2S datasets, which rely on synthetic recordings of Western classical music, SheetSage-A2S is built from real audio and isolates the sung melody together with its chords.

We also propose an **A2S model** that improves on prior work with three changes to an autoregressive transformer decoder: a larger ff dim + pre-norm decoder, using [MuQ](https://github.com/tencent-ailab/MuQ) as an encoder in place of the CNN, and pitch/tempo data augmentation. The model reaches 4.98% symbol error rate on the classical Quartets collection and sets a 20.92% benchmark on SheetSage-A2S.

For more details, please refer to our [paper](https://arxiv.org/abs/2608.06165).

<div>
  <img src="images/model_comparison_spectrogram.svg" width="45%" alt="Model overview">
  <!-- <img src="images/results.png" width="45%" alt="Qualitative results"> -->
</div>

## Datasets & Checkpoints

Datasets and model checkpoints are hosted on the Hugging Face Hub: https://huggingface.co/collections/MMR-Lab/sheetsage-a2s. The model checkpoints are publicly available for inference. To train our model from scratch and/or conduct other customized experiments, you need to fill in a brief application form at Hugging Face Hub in order to access the datasets. 


> **Note:** if `huggingface.co` is unreachable, set the mirror endpoint before downloading:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### Different datasets

SheetSage-A2S is published in two forms. They contain the same songs and the same
train/val/test splits. 
https://huggingface.co/datasets/MMR-Lab/Sheetsage-A2S contains the kern scores along with the YouTube links and timestamps.
https://huggingface.co/datasets/MMR-Lab/Sheetsage-A2S-MuQ contains the dataset with MuQ preprocessed features as well as kern scores.

### Getting the data

Every script that takes `--ds_location` accepts either a local path or a Hub dataset id.

```bash
python train.py --ds_location MMR-Lab/Sheetsage-A2S-MuQ --encoder preprocessed_muq ...
```

To download once up front instead.

```bash
hf download MMR-Lab/Sheetsage-A2S-MuQ \
    --repo-type dataset \
    --local-dir /path/to/data/Sheetsage-A2S-MuQ
```

### Model checkpoints

We release two model checkpoints:

- [**SheetSage-A2S model**](https://huggingface.co/MMR-Lab/Sheetsage-A2S-model): our best-performing
model that was trained and evaluated on SheetSage-A2S.
- [**Quartets model**](https://huggingface.co/MMR-Lab/quartets-a2s-model): our best-performing model
that was trained and evaluated on Quartets.

`--checkpoint_path` accepts either a local checkpoint path or a Hugging Face model repository ID. Hub checkpoints are downloaded automatically into the local Hugging Face cache.

```bash
# Local checkpoint
python run_metrics.py --checkpoint_path /path/to/checkpoint.ckpt ...

# SheetSage-A2S checkpoint
python run_metrics.py --checkpoint_path MMR-Lab/Sheetsage-A2S-model ...

# Quartets checkpoint
python run_metrics.py --checkpoint_path MMR-Lab/quartets-a2s-model ...
```

## Usage

### Installation

**Option A — Docker (recommended).** 

```bash
docker build -t hookkern .
docker run -it --gpus all --ipc=host \
    -v /path/to/data:/data \
    hookkern bash
```

**Option B — pip.** Requires `python>=3.11`, a CUDA-capable GPU, and the system packages
`ffmpeg`, `fluidsynth`, and (for augmentation only) `rubberband-cli`.

```bash
pip install -r requirements.txt
```


### 1. Preprocess audio into MuQ features (optional)

Precompute MuQ features so training does not run the (frozen) MuQ encoder every step.
Required for the `preprocessed_muq` encoders. If you want to do this for the Sheetsage-A2S dataset please note that we release the already preprocessed MuQ features for this dataset.


```bash
python preprocess_muq.py --input-dataset /path/to/dataset --output-dir /path/to/output
```


> **Note:** MuQ features are always extracted in 32-bit precision. Although training otherwise
> runs in 16-bit mixed precision, MuQ is executed with autocasting disabled both here and
> on-the-fly during training if you skip this step, since half precision produces NaN values in
> its output.

### 2. Train

SheetSage-A2S (lead sheets) with the precomputed-MuQ encoder:

```bash
python train.py \
    --ds_location MMR-Lab/Sheetsage-A2S-MuQ \
    --encoder preprocessed_muq \
    --tokeniser word \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --ff_dim_multiplier 4 \
    --label_smoothing 0.1 \
    --patience 5
```


To use the raw-spectrogram CNN encoder instead, pass `--encoder cnn` 

For the **Quartets** dataset, use `--tokeniser original` (this applies the cleaning from
the original quartets paper together with word-level tokenisation):

```bash
python train.py \
    --ds_location /path/to/quartets \
    --encoder preprocessed_muq \
    --tokeniser original \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --ff_dim_multiplier 4 \
    --patience 5
```

> **Batch size** must be one of `1, 2, 4, 8`. For values below 8, gradient accumulation is
> applied automatically to keep an effective batch size of 8. Checkpoints are written to
> `weights/<encoder>/` and training is logged to Weights & Biases (`wandb login`, or set
> `WANDB_MODE=offline`).

Valid **encoders**: `cnn`, `muq`, `preprocessed_muq`.
Valid **tokenisers**: `word`, `medium`, `original`.

### 3. Evaluate

Compute metrics (Sym-ER, MV2H) on the test set. Both `--checkpoint_path` and `--ds_location`
accept a local path or a Hub id.

```bash
# Lead sheets (SheetSage-A2S)
python run_metrics.py \
    --checkpoint_path MMR-Lab/Sheetsage-A2S-model \
    --ds_location MMR-Lab/Sheetsage-A2S-MuQ \
    --encoder preprocessed_muq \
    --tokeniser word \
    --dataset_type lead_sheet

# String quartets
python run_metrics.py \
    --checkpoint_path /path/to/checkpoint.ckpt \
    --ds_location /path/to/quartets \
    --encoder preprocessed_muq \
    --tokeniser original \
    --dataset_type quartets
```

### 4. Sample / visualise predictions

Run inference sample-by-sample, printing ground-truth and predicted tokens together with
[Verovio Humdrum Viewer](https://verovio.humdrum.org/) links that render the output as sheet music.

```bash
python sample.py \
    --checkpoint_path MMR-Lab/Sheetsage-A2S-model \
    --ds_location MMR-Lab/Sheetsage-A2S-MuQ \
    --tokeniser word \
    --number 5
```

`--number` limits how many samples are printed.

### 5. Data augmentation (optional)

Pitch/tempo augmentation of the quartets dataset. Requires the Humdrum `transpose` tool and the
`rubberband` CLI on your `PATH`.

```bash
python augment_existing_hf_dataset.py --output-path /path/to/augmented
```

## Performance

Model performance on the SheetSage-A2S and Quartets datasets using the word-level tokenisation
scheme.
### SER (overall)

| Method | SheetSage-A2S | Quartets |
| ------ | :------------: | :------: |
| Baseline | 66.85 | 15.3 |
| + 1024 Pre-Norm | 53.7 | 8.48 |
| + MuQ encoder | 25.39 | 7.16 |
| + Augmented data | **20.92** | **4.98** |

### Per-spine SER and MV2H (SheetSage-A2S)

Per-spine SER (melody and chord spines) and MV2H on SheetSage-A2S. Lower is better for SER;
higher is better for MV2H.

| Variant | Normal SER | Melody SER | Chord SER | MV2H |
| ------- | :--------: | :--------: | :-------: | :--: |
| Baseline | 66.85 | 106.88 | 64.12 | 70.39 |
| + 1024 Pre-Norm | 53.7 | 90.56 | 51.8 | 72.62 |
| + MuQ | 25.39 | 46.21 | 26.76 | 79.38 |
| + Augmented | **20.92** | **38.62** | **22.28** | **85.05** |


## License

The code in this repository is released under the MIT license as found in the [LICENSE](LICENSE) file.

## Citation


```
@article{cummins2026audio,
  title={Audio-to-Score Transcription using Pre-trained Features, Data Augmentation, and the New SheetSage-A2S Dataset},
  author={Cummins, Eoin and Huang, Zhongyi and D'Hooge, Alexandre and Mo, Zhuoro and Ju, Yaolong},
  journal={arXiv preprint arXiv:2608.06165},
  year={2026}
}
```

## Acknowledgement
- [MuQ](https://github.com/tencent-ailab/MuQ) — pretrained music audio encoder.
- The kern tokeniser adapts code from [ISMIR-Jazzmus](https://github.com/JuanCarlosMartinezSevilla/ISMIR-Jazzmus).
- We also borrow code from [a2s-transformer](https://github.com/mariaalfaroc/a2s-transformer).
