import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from muq import MuQ

DEVICE = "cuda"
SUFFIX = "precomputed_muq"


def preprocess_audio(
    raw_audio: np.ndarray, sr: float, dtype=torch.float32
) -> torch.Tensor:
    # x = raw_audio
    x = librosa.resample(raw_audio, orig_sr=sr, target_sr=24_000)
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)
    x = x.type(dtype=dtype)
    return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute MuQ features for a dataset."
    )
    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Local dataset path (for --from-disk) or HF dataset name (for --no-from-disk).",
    )
    parser.add_argument(
        "--from-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load dataset with load_from_disk (default: true). Use --no-from-disk for load_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the feature dataset. Required when using --no-from-disk.",
    )
    args = parser.parse_args()

    if not args.from_disk and args.output_dir is None:
        parser.error("--output-dir is required when using --no-from-disk.")

    return args


def output_path_for_input(input_dataset: str, from_disk: bool, output_dir: str) -> Path:
    input_path = Path(input_dataset)
    dataset_name = input_path.name if input_path.name else input_path.parent.name
    output_name = f"{dataset_name}_{SUFFIX}"

    if from_disk:
        return input_path.parent / output_name
    return Path(output_dir) / Path(output_name)


def main():
    args = parse_args()
    print("Loading Encoder...")

    if args.from_disk:
        ds = load_from_disk(args.input_dataset)
    else:
        ds = load_dataset(args.input_dataset)
    output_dataset_path = output_path_for_input(
        args.input_dataset, args.from_disk, args.output_dir
    )

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
                layer = muq(audio_tensor, output_hidden_states=False).last_hidden_state

        layer = layer.cpu()
        layer = layer.squeeze(0)
        return {"features": layer.numpy(), "length": layer.shape[0]}

    print("Mapping dataset to features...")
    feature_dataset = ds.map(
        extract_features,
        batched=False,
        remove_columns=["audio"],
        desc="Extracting MuQ Features",
    )
    print(feature_dataset)

    print(f"Saving to {output_dataset_path}...")
    feature_dataset.save_to_disk(str(output_dataset_path))
    print("Done")


if __name__ == "__main__":
    main()
