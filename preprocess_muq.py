import librosa
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from muq import MuQ

ORIGINAL_DATASET_PATH = "/home/eoin/sheetsage_dataset_trimmed"
OUTPUT_DATASET_PATH = "/home/eoin/sheetsage_precomputed_muq"
OUTPUT_DATASET_PATH = "/home/eoin/quartets_precomputed_muq"
DEVICE = "cuda:3"


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

    # ds = load_from_disk(ORIGINAL_DATASET_PATH)
    ds = load_dataset(f"PRAIG/Quartets-quartets")

    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.eval()
    muq = muq.to(DEVICE)

    for param in muq.parameters():
        param.requires_grad = False

    def extract_features(batch):
        audio_tensor = preprocess_audio(
            batch["audio"]["array"], batch["audio"]["sampling_rate"]
        ).to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                features = muq(
                    audio_tensor, output_hidden_states=False
                ).last_hidden_state

        features = features.cpu()
        features = features.squeeze()
        return {"muq_features": features.numpy(), "muq_length": features.shape[0]}

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
