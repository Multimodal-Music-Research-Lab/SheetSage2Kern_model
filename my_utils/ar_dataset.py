import json
import math
import os

import torch
from datasets import load_from_disk
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from my_utils.data_preprocessing import (
    ar_batch_preparation,
    preprocess_audio,
    set_pad_index,
)
from my_utils.tokeniser import GtParser
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token
SPLITS = ["train", "validation", "test"]

LOCAL_DATASET_PATH = "/home/eoin/hooktheory_dataset"
DATASET_NAME = "HookKern"


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name=DATASET_NAME,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Datasets
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="validation",
                    use_voice_change_token=self.use_voice_change_token,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    use_voice_change_token=self.use_voice_change_token,
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


####################################################################################################


class ARDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        use_voice_change_token: bool = False,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
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
        self.ds = load_from_disk(LOCAL_DATASET_PATH)[self.partition_type]

        # Check and retrieve vocabulary
        vocab_folder = os.path.join("Quartets", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()
        # Modify the global PAD_INDEX to match w2i["<PAD>"]
        set_pad_index(self.w2i["<PAD>"])

        # Check and retrive max lengths
        # Set max_seq_len, max_audio_len and frame_multiplier_factor
        max_lens_folder = os.path.join("Quartets", "max_lens")
        os.makedirs(max_lens_folder, exist_ok=True)
        max_lens_name = vocab_name
        self.max_lens_path = os.path.join(max_lens_folder, max_lens_name)
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_audio_len = max_lens["max_audio_len"]
        self.frame_multiplier_factor = max_lens["max_frame_multiplier_factor"]

    def __getitem__(self, idx):
        x = preprocess_audio(
            raw_audio=self.ds[idx]["audio"]["array"],
            sr=self.ds[idx]["audio"]["sampling_rate"],
            dtype=torch.float32,
        )
        y = self.preprocess_transcript(text=self.ds[idx]["kern"])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        elif self.partition_type == "validation":
            return x, self.get_number_of_frames(x), y  # TODO temprorary
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert_text(text=text)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def make_vocabulary(self):
        full_ds = load_from_disk(LOCAL_DATASET_PATH)

        vocab = []
        for split in SPLITS:
            for text in full_ds[split]["kern"]:
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
        # audio is the output of preprocess_audio
        # audio.shape = [1, freq_bins, time_frames]
        return math.ceil(audio.shape[1] / HEIGHT_REDUCTION) * math.ceil(
            audio.shape[2] / WIDTH_REDUCTION
        )

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
        # Set the maximum lengths for the whole QUARTETS collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum audio length
        # 3) Get the frame multiplier factor so that
        # the frames input to the RNN are equal to the
        # length of the transcript, ensuring the CTC condition
        max_seq_len = 0

        full_ds = load_from_disk(LOCAL_DATASET_PATH)
        max_audio_raw = None
        max_duration = 0.0
        max_audio_sr = None
        for split in SPLITS:
            print(f"split: {split} starting")
            for sample in full_ds[split]:
                # Max transcript length
                transcript = self.krn_parser.convert_text(text=sample["kern"])
                max_seq_len = max(max_seq_len, len(transcript))

                sr = sample["audio"]["sampling_rate"]
                raw_audio = sample["audio"]["array"]

                dur = raw_audio.shape[0] / sr

                if dur > max_duration:
                    max_duration = dur
                    max_audio_raw = raw_audio
                    max_audio_sr = sr

        audio = preprocess_audio(
            raw_audio=max_audio_raw,
            sr=max_audio_sr,
            dtype=torch.float32,
        )
        max_audio_len = audio.shape[2]
        return {
            "max_seq_len": max_seq_len,
            "max_audio_len": max_audio_len,
            "max_frame_multiplier_factor": 1,
        }
