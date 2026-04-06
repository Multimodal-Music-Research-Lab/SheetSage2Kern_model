import argparse
import os
import random
import re
import shutil
import subprocess
import tempfile

import pyrubberband as pyrb
from datasets import load_dataset

SEED = 67
OUTPUT_DATASET_PATH = "/home/eoin/datasets/quartets_augmented_6x"

TRANSPOSITIONS = [-3, -2, -1, 1, 2, 3]
SPEEDS = [0.9, 0.95, 1, 1.05, 1.1]


SEMITONE_CANDIDATES = {
    1: ["m2", "A1"],
    2: ["M2", "d3"],
    3: ["m3", "A2"],
    -1: ["-m2", "-A1"],
    -2: ["-M2", "-d3"],
    -3: ["-m3", "-A2"],
}

TRIPLE_ACC_RE = re.compile(r"(#{3,}|-{3,})")
BARE_DURATION_RE = re.compile(r"^\d+\.?$")


def bad_kern_output(kern_text):
    for line in kern_text.splitlines():
        if not line or line.startswith(("!", "*", "=")):
            continue
        for tok in line.split():
            if TRIPLE_ACC_RE.search(tok):
                return True
            if tok == ".":
                continue
            if BARE_DURATION_RE.fullmatch(tok):
                return True
    return False


def run_transpose(kern_string, interval):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".krn", delete=False, encoding="utf-8"
    ) as tmp_in:
        tmp_in.write(kern_string)
        tmp_in_path = tmp_in.name
    try:
        result = subprocess.run(
            ["transpose", "-q", "-t", interval, tmp_in_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    finally:
        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)


def transpose_kern_string(kern_string, semitones):
    if semitones == 0:
        return kern_string

    if shutil.which("transpose") is None:
        raise RuntimeError("Humdrum 'transpose' not found in PATH.")

    for interval in SEMITONE_CANDIDATES[semitones]:
        out = run_transpose(kern_string, interval)
        if not bad_kern_output(out):
            return out

    raise ValueError()


def modify_kern_tempo_strict(kern_string, speed_factor):
    lines = kern_string.splitlines()

    target_line_idx = 6

    # assert lines[6].count("*MM") == 4 or  lines[6].count("*MM") == 8, kern_string

    def replace_match(match):
        original_mm = int(match.group(1))
        new_mm = int(round(original_mm * speed_factor))
        return f"*MM{new_mm}"

    lines[target_line_idx] = re.sub(r"\*MM(\d+)", replace_match, lines[target_line_idx])

    return "\n".join(lines)


def augment_batch(batch):
    new_batch = {"audio": [], "transcript": []}

    for i in range(len(batch["audio"])):
        orig_audio = batch["audio"][i]["array"]
        orig_sr = batch["audio"][i]["sampling_rate"]
        orig_kern = batch["transcript"][i]
        new_batch["audio"].append(batch["audio"][i])
        new_batch["transcript"].append(orig_kern)

        for semi in TRANSPOSITIONS:
            y_shifted = pyrb.pitch_shift(orig_audio, orig_sr, semi)
            kern_shifted = transpose_kern_string(orig_kern, semi)
            s = random.choice(SPEEDS)

            y_speed = pyrb.time_stretch(y_shifted, orig_sr, s)
            kern_shifted = modify_kern_tempo_strict(kern_shifted, s)
            if kern_shifted is not None:
                new_batch["audio"].append({"array": y_speed, "sampling_rate": orig_sr})
                new_batch["transcript"].append(kern_shifted)

    return new_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment a HuggingFace dataset with pitch/tempo variations."
    )
    parser.add_argument(
        "--output-path", required=True, help="Output path for the augmented dataset."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    print("Augmenting Train split...")
    ds = load_dataset("PRAIG/quartets-quartets")

    if "train" in ds:
        augmented_train = ds["train"].map(
            augment_batch,
            batched=True,
            batch_size=4,
            remove_columns=ds["train"].column_names,
            num_proc=4,
            features=ds["train"].features,
        )
        ds["train"] = augmented_train

    # print(f"Saving to {OUTPUT_DATASET_PATH}...")

    print(f"Saving to {args.output_path}...")
    ds.save_to_disk(args.output_path)
    # ds.save_to_disk(OUTPUT_DATASET_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
