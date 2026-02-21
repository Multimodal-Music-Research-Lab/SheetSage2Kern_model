import json
import math
import os

import torch
from datasets import load_from_disk
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from my_utils.consts import (
    CNN_ENCODER,
    EOS_TOKEN,
    MUQ_ENCODER,
    PREPROCESSED_MUQ_ENCODER,
    SOS_TOKEN,
    SPLITS,
    VALIDATION_SPLIT,
)
from my_utils.data_preprocessing import (
    IMG_HEIGHT,
    ar_batch_preparation,
    preprocess_audio,
    set_pad_index,
)
from my_utils.tokeniser import GtParser
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION
from networks.transformer.muq_encoder import MUQ_DIMENSION_REDUCTION

KERN_COLUMN = "transcript"
DATASET_INFORMATION_FOLDER = "Dataset_information"


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name,
        ds_location,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
        encoder_name=CNN_ENCODER,
    ):
        super(ARDataModule, self).__init__()
        self.ds_location = ds_location
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Datasets
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.encoder_name = encoder_name

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    ds_location=self.ds_location,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                    encoder_name=self.encoder_name,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    ds_location=self.ds_location,
                    partition_type=VALIDATION_SPLIT,
                    use_voice_change_token=self.use_voice_change_token,
                    encoder_name=self.encoder_name,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    ds_location=self.ds_location,
                    partition_type="test",
                    use_voice_change_token=self.use_voice_change_token,
                    encoder_name=self.encoder_name,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,  # TODO temprorary
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader()

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_audio_len(self):
        try:
            return self.train_ds.max_audio_len
        except AttributeError:
            return self.test_ds.max_audio_len

    def get_max_encoder_output_len(self):
        return self.train_ds.max_encoder_output_len


####################################################################################################


class ARDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        ds_location: str,
        partition_type: str,
        use_voice_change_token: bool = False,
        encoder_name=CNN_ENCODER,
    ):
        self.ds_name = ds_name.lower()
        self.ds_location = ds_location
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
        self.encoder_name = encoder_name
        self.init(vocab_name="ar_w2i")
        self.max_seq_len += 1  # Add 1 for EOS_TOKEN

    def init(self, vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = GtParser()

        # Check partition type
        assert self.partition_type in SPLITS, (
            f"Invalid partition type: {self.partition_type}"
        )

        # Get audios and transcripts files
        if self.encoder_name == PREPROCESSED_MUQ_ENCODER:
            self.ds = load_from_disk(self.ds_location)[self.partition_type]
        else:
            self.ds = load_from_disk(self.ds_location)[self.partition_type].filter(
                lambda example: (
                    example["audio"]["sampling_rate"] * 1
                    <= len(example["audio"]["array"])
                    <= example["audio"]["sampling_rate"] * 60
                )
            )

        # Check and retrieve vocabulary
        vocab_folder = os.path.join(DATASET_INFORMATION_FOLDER, "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()
        # Modify the global PAD_INDEX to match w2i["<PAD>"]
        set_pad_index(self.w2i["<PAD>"])

        # Check and retrive max lengths
        # Set max_seq_len, max_audio_len
        max_lens_folder = os.path.join(DATASET_INFORMATION_FOLDER, "max_lens")
        os.makedirs(max_lens_folder, exist_ok=True)

        os.makedirs(os.path.join(max_lens_folder, self.encoder_name), exist_ok=True)
        max_lens_name = vocab_name

        self.max_lens_path = os.path.join(
            max_lens_folder, self.encoder_name, max_lens_name
        )
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_audio_len = max_lens["max_audio_len"]
        self.max_encoder_output_len = max_lens["max_encoder_output_len"]

    def __getitem__(self, idx):
        if self.encoder_name == PREPROCESSED_MUQ_ENCODER:
            x = torch.as_tensor(self.ds[idx]["muq_features"])
        else:
            x = preprocess_audio(
                raw_audio=self.ds[idx]["audio"]["array"],
                sr=self.ds[idx]["audio"]["sampling_rate"],
                dtype=torch.float32,
                encoder=self.encoder_name,
            )
        y = self.preprocess_transcript(text=self.ds[idx][KERN_COLUMN])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        elif self.partition_type == VALIDATION_SPLIT:
            return x, self.get_number_of_frames(x), y  # TODO temprorary
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert_text(text=text)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def make_vocabulary(self):
        full_ds = load_from_disk(self.ds_location)

        vocab = []
        for split in SPLITS:
            for text in full_ds[split][KERN_COLUMN]:
                transcript = self.krn_parser.convert_text(text=text)
                vocab.extend(transcript)
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def get_number_of_frames(self, audio):
        if self.encoder_name == CNN_ENCODER:
            return math.ceil(audio.shape[1] / HEIGHT_REDUCTION) * math.ceil(
                audio.shape[2] / WIDTH_REDUCTION
            )
        elif self.encoder_name == PREPROCESSED_MUQ_ENCODER:
            return audio.shape[0]
        elif self.encoder_name == MUQ_ENCODER:
            # Shape is [B,raw_audio_length]
            return math.floor(audio.shape[1] / MUQ_DIMENSION_REDUCTION)

    def __len__(self):
        return len(self.ds)

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def check_and_retrieve_max_lens(self):
        max_lens = {}

        if os.path.isfile(self.max_lens_path):
            with open(self.max_lens_path, "r") as file:
                max_lens = json.load(file)
        else:
            max_lens = self.make_max_lens()
            with open(self.max_lens_path, "w") as file:
                json.dump(max_lens, file)

        return max_lens

    def make_max_lens(self):
        # Set the maximum lengths for the whole Dataset:
        # 1) Get the maximum transcript length
        # 2) Get the maximum audio length

        print("creating max lengths")
        max_seq_len = 0

        full_ds = load_from_disk(self.ds_location)
        max_audio_raw = None
        max_audio_len = 0
        max_duration = 0.0
        max_audio_sr = None
        max_preprocessed_muq = 0
        for split in SPLITS:
            print(f"Processing split: {split}")
            for sample in tqdm(full_ds[split], desc=f"{split}"):
                if self.encoder_name != PREPROCESSED_MUQ_ENCODER:
                    sr = sample["audio"]["sampling_rate"]
                    raw_audio = sample["audio"]["array"]
                    if not (sr <= len(raw_audio) <= sr * 60):
                        continue

                    dur = raw_audio.shape[0] / sr

                    if dur > max_duration:
                        max_duration = dur
                        max_audio_raw = raw_audio
                        max_audio_sr = sr
                else:
                    max_preprocessed_muq = max(
                        max_preprocessed_muq,
                        sample["muq_length"],
                    )
                # Max transcript length
                transcript = self.krn_parser.convert_text(text=sample[KERN_COLUMN])
                max_seq_len = max(max_seq_len, len(transcript))

        if self.encoder_name != PREPROCESSED_MUQ_ENCODER:
            audio = preprocess_audio(
                raw_audio=max_audio_raw,
                sr=max_audio_sr,
                dtype=torch.float32,
                encoder=self.encoder_name,
            )
            max_audio_len = audio.shape[-1]

        if self.encoder_name == PREPROCESSED_MUQ_ENCODER:
            max_encoder_output_len = max_preprocessed_muq
        elif self.encoder_name == MUQ_ENCODER:
            max_encoder_output_len = math.floor(max_audio_len / MUQ_DIMENSION_REDUCTION)
        else:
            max_encoder_output_len = math.ceil(
                IMG_HEIGHT / HEIGHT_REDUCTION
            ) * math.ceil(max_audio_len / WIDTH_REDUCTION)
        return {
            "max_seq_len": max_seq_len,
            "max_audio_len": max_audio_len,
            "max_encoder_output_len": max_encoder_output_len,
        }
