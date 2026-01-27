import librosa
import numpy as np
import torch
from datasets import load_from_disk

from networks.transformer.muq_encoder import MuqEncoder

ORIGINAL_DATASET_PATH = "/home/eoin/hookKern_dataset"
OUTPUT_DATASET_PATH = "/home/eoin/hookKern_dataset_precomputed_muq"
DEVICE = "cuda:2"


def preprocess_audio(
    raw_audio: np.ndarray, sr: float, dtype=torch.float32
) -> torch.Tensor:
    x = raw_audio
    x = librosa.resample(raw_audio, orig_sr=sr, target_sr=24_000)
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)
    x = x.type(dtype=dtype)
    return x


def main():
    print("Loading Encoder...")
    encoder = MuqEncoder(device=DEVICE)

    ds = load_from_disk(ORIGINAL_DATASET_PATH)

    ds = ds.filter(
        lambda example: example["audio"]["sampling_rate"] * 1
        <= len(example["audio"]["array"])
        <= example["audio"]["sampling_rate"] * 60
    )

    def extract_features(batch):
        audio_tensor = preprocess_audio(
            batch["audio"]["array"], batch["audio"]["sampling_rate"]
        ).to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                features = encoder(audio_tensor)

        features = features.cpu()
        features = features.squeeze()

        return {"muq_features": features.numpy()}

    print("Mapping dataset to features...")
    feature_dataset = ds.map(
        extract_features,
        batched=False,
        remove_columns=["audio"],
        desc="Extracting MuQ Features",
    )
    print(feature_dataset)

    print(f"Saving to {OUTPUT_DATASET_PATH}...")
    feature_dataset.save_to_disk(OUTPUT_DATASET_PATH)
    print("Done")


if __name__ == "__main__":
    main()
