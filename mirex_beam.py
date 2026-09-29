import warnings
from pathlib import Path

import fire
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from muq import MuQ
from tqdm import tqdm

from my_utils.consts import EOS_TOKEN, SOS_TOKEN
from my_utils.tokeniser import untokenize
from networks.transformer.model import A2STransformer


def extract_muq_features(audio_path, muq, device):
    audio, sample_rate = sf.read(str(audio_path))
    audio = librosa.to_mono(audio.T)

    if audio.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    audio = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=24_000,
    )
    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.autocast(device_type=device.type, enabled=False):
        features = muq(
            waveform,
            output_hidden_states=False,
        ).last_hidden_state

    return features  # [1, time, 1024]


def beam_search_decode(model, memory, beam_size=4, length_penalty=0.0):
    start = model.w2i[SOS_TOKEN]
    end = model.w2i[EOS_TOKEN]
    beams = [([start], 0.0, False)]

    def rank(beam):
        tokens, score, _ = beam
        length = max(1, len(tokens) - 1)
        return score / length**length_penalty

    for _ in range(model.max_seq_len):
        if all(finished for _, _, finished in beams):
            break

        candidates = []

        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            y_in = torch.tensor([tokens], dtype=torch.long, device=memory.device)
            logits = model.decoder(
                tgt=y_in,
                memory=memory,
                memory_len=None,
            )[0, :, -1]

            log_probs = F.log_softmax(logits, dim=-1)

            log_probs[start] = float("-inf")
            log_probs[model.w2i["<PAD>"]] = float("-inf")

            values, indices = torch.topk(log_probs, beam_size)

            for value, token in zip(values.tolist(), indices.tolist()):
                if value != float("-inf"):
                    candidates.append(
                        (
                            tokens + [token],
                            score + value,
                            token == end,
                        )
                    )

        if not candidates:
            raise RuntimeError("Beam search produced no valid candidates.")

        beams = sorted(candidates, key=rank, reverse=True)[:beam_size]

    completed = [beam for beam in beams if beam[2]]
    if not completed:
        warnings.warn("Maximum decoding length reached without EOS.")

    tokens, _, finished = max(completed or beams, key=rank)
    return tokens[1:-1] if finished else tokens[1:]


def tokens_to_kern(tokens, legacy=False, num_spines=4):
    if legacy:
        # Original Quartets tokenisation: one token per spine,
        # with <COR> between rows.
        rows, row = [], []

        for token in tokens:
            if token == "<COR>":
                if row:
                    rows.append(row)
                row = []
            elif token != "<COC>":
                row.append("." if token == "DOT" else token)

        if row:
            rows.append(row)
    else:
        rows = [
            line.split("\t") for line in untokenize(tokens).splitlines() if line.strip()
        ]

    if not rows:
        raise ValueError("The model predicted an empty score.")

    if all(field.startswith("**") for field in rows[0]):
        num_spines = len(rows[0])
    else:
        rows.insert(0, ["**kern"] * num_spines)

    for number, row in enumerate(rows, start=1):
        if len(row) != num_spines:
            raise ValueError(
                f"Score row {number} has {len(row)} fields; "
                f"expected {num_spines}. Inspect score reconstruction."
            )

        if any(field in {"*^", "*v", "*x", "*+"} for field in row):
            raise ValueError("This score writer supports fixed spines only.")

    if rows[-1] != ["*-"] * num_spines:
        rows.append(["*-"] * num_spines)

    return "\n".join("\t".join(row) for row in rows) + "\n"


@torch.inference_mode()
def evaluate(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    beam_size: int = 4,
    length_penalty: float = 0.0,
    num_spines: int = 4,
):
    input_folder = Path(input_dir).expanduser().resolve()
    output_folder = Path(output_dir).expanduser().resolve()

    if not input_folder.is_dir():
        raise NotADirectoryError(input_folder)

    if beam_size < 1 or length_penalty < 0 or num_spines < 1:
        raise ValueError("Invalid beam size, length penalty, or spine count.")

    extensions = {".flac", ".wav", ".mp3", ".ogg", ".aif", ".aiff"}
    files = sorted(
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not files:
        raise ValueError(f"No audio files found in {input_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = A2STransformer.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    if model.encoder_name not in {
        "preprocessed_muq",
        "preprocessed_muq_temporal_downsampling",
    }:
        raise ValueError("This script expects a precomputed-MuQ checkpoint.")

    model = model.float().eval().to(device)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").float().eval().to(device)

    if beam_size > len(model.w2i) - 2:
        raise ValueError("Beam size exceeds the usable vocabulary size.")

    i2w = {int(index): token for token, index in model.w2i.items()}
    legacy = "<COR>" in model.w2i

    output_folder.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(files, desc="Transcribing"):
        output_path = (
            output_folder / audio_path.relative_to(input_folder)
        ).with_suffix(".krn")

        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")

        features = extract_muq_features(audio_path, muq, device)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            memory = model.encoder(x=features)

            token_ids = beam_search_decode(
                model,
                memory,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        tokens = [i2w[index] for index in token_ids]

        try:
            score = tokens_to_kern(tokens, legacy, num_spines)
        except ValueError as error:
            raise ValueError(f"{audio_path.name}: {error}") from error

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(score, encoding="utf-8")

        del features, memory


if __name__ == "__main__":
    fire.Fire(evaluate)
