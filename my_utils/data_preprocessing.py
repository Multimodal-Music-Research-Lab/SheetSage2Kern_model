import joblib
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from my_utils.consts import CNN_ENCODER, MUQ_ENCODER, PREPROCESSED_MUQ_ENCODER

MEMORY = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)
NUM_CHANNELS = 1
IMG_HEIGHT = NUM_FREQ_BINS = 195


def set_pad_index(index: int):
    global PAD_INDEX
    PAD_INDEX = index


def get_spectrogram_from_raw_audio(raw_audio: np.ndarray, sr: float) -> np.ndarray:
    new_sr = 22050
    y = librosa.resample(raw_audio, orig_sr=sr, target_sr=new_sr)

    stft_fmax = 2093
    stft_frequency_filter_max = (
        librosa.fft_frequencies(sr=new_sr, n_fft=2048) <= stft_fmax
    )

    stft = librosa.stft(y, hop_length=512, win_length=2048, window="hann")
    stft = stft[stft_frequency_filter_max]

    stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
    log_stft = ((1.0 / 80.0) * stft_db) + 1.0

    return log_stft


@MEMORY.cache
def preprocess_audio(
    raw_audio: np.ndarray, sr: float, dtype=torch.float32, encoder=CNN_ENCODER
) -> torch.Tensor:
    # Get spectrogram (already normalized)
    if encoder == CNN_ENCODER:
        x = get_spectrogram_from_raw_audio(raw_audio, sr)
    elif encoder == MUQ_ENCODER:
        x = raw_audio
        x = librosa.resample(raw_audio, orig_sr=sr, target_sr=24_000)
    else:
        raise ValueError()
    # Convert to PyTorch tensor
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)  # [1, freq_bins, time_frames]
    x = x.type(dtype=dtype)
    return x


################################# CTC PREPROCESSING:


# def pad_batch_audios(x, dtype=torch.float32):
#     max_width = max(x, key=lambda sample: sample.shape[-1]).shape[-1]
#     x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[-1])) for i in x], dim=0)
#     x = x.type(dtype=dtype)
#     return x
def pad_batch_audios(
    x,
    feature_type,
    dtype=torch.float32,
):
    if feature_type == PREPROCESSED_MUQ_ENCODER:
        max_time = max(sample.shape[0] for sample in x)
        x = torch.stack(
            [F.pad(i, pad=(0, 0, 0, max_time - i.shape[0])) for i in x], dim=0
        )
    else:
        max_width = max(x, key=lambda sample: sample.shape[-1]).shape[-1]
        x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[-1])) for i in x], dim=0)
    x = x.to(dtype=dtype)
    return x


def pad_batch_transcripts(x, dtype=torch.int32):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack(
        [F.pad(i, pad=(0, max_length - i.shape[0]), value=PAD_INDEX) for i in x], dim=0
    )
    x = x.type(dtype=dtype)
    return x


################################# AR PREPROCESSING:


def ar_batch_preparation(batch, feature_type=CNN_ENCODER):
    x, xl, y = zip(*batch)
    # Zero-pad audios to maximum batch audio width
    x = pad_batch_audios(x, feature_type, dtype=torch.float32)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Decoder input: transcript[:-1]
    y_in = [i[:-1] for i in y]
    y_in = pad_batch_transcripts(y_in, dtype=torch.int64)
    # Decoder target: transcript[1:]
    y_out = [i[1:] for i in y]
    y_out = pad_batch_transcripts(y_out, dtype=torch.int64)
    return x, xl, y_in, y_out
