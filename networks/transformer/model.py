import math
import random

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.nn import CrossEntropyLoss
from torchinfo import summary

from my_utils.consts import (
    CNN_ENCODER,
    EOS_TOKEN,
    MUQ_ENCODER,
    PREPROCESSED_MUQ_ENCODER,
    SOS_TOKEN,
)
from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import compute_metrics
from networks.transformer.decoder import Decoder
from networks.transformer.encoder_modules import CnnEncoder
from networks.transformer.muq_encoder import MuqEncoder, MuqEncoderPreprocessed


class A2STransformer(LightningModule):
    def __init__(
        self,
        max_seq_len,
        max_audio_len,
        w2i,
        i2w,
        max_encoder_output_length,
        ytest_i2w=None,
        attn_window=-1,
        teacher_forcing_prob=0.5,
        encoder=CNN_ENCODER,
        lr=1e-4,
        weight_decay=0.0,
        ff_dim_multiplier=1,
        label_smoothing=0.0,
    ):
        super(A2STransformer, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.label_smoothing = label_smoothing
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.padding_idx = w2i["<PAD>"]
        # Model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_audio_len = max_audio_len
        self.max_seq_len = max_seq_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.encoder_name = encoder
        self.max_flattened_encoder_output_length = max_encoder_output_length

        if encoder == CNN_ENCODER:
            self.encoder = CnnEncoder(
                max_encoder_output_length=max_encoder_output_length,
                max_audio_len=max_audio_len,
            )
        elif encoder == MUQ_ENCODER:
            self.encoder = MuqEncoder(
                max_encoder_output_length=max_encoder_output_length,
                max_audio_len=max_audio_len,
            )
        elif encoder == PREPROCESSED_MUQ_ENCODER:
            self.encoder = MuqEncoderPreprocessed(
                max_encoder_output_length=max_encoder_output_length,
                max_audio_len=max_audio_len,
            )

        embedding_dim = self.encoder.get_output_dim()
        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=self.max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
            embedding_dim=embedding_dim,
            ff_dim=embedding_dim * ff_dim_multiplier,
        )
        self.summary()
        # Loss
        self.compute_loss = CrossEntropyLoss(
            ignore_index=self.padding_idx, label_smoothing=self.label_smoothing
        )
        # Predictions
        self.Y = []
        self.YHat = []

    def summary(self):
        print("Encoder")
        if self.encoder_name == CNN_ENCODER:
            summary(
                self.encoder,
                input_size=[1, NUM_CHANNELS, IMG_HEIGHT, self.max_audio_len],
            )

        print("Decoder")
        tgt_size = [1, self.max_seq_len]
        memory_size = [
            1,
            self.max_flattened_encoder_output_length,
            self.encoder.get_output_dim(),
        ]
        memory_len_size = [1]
        summary(
            self.decoder,
            input_size=[tgt_size, memory_size, memory_len_size],
            dtypes=[torch.int64, torch.float32, torch.int64],
        )

    def configure_optimizers(self):
        # return torch.optim.Adam(
        #     list(self.encoder.parameters()) + list(self.decoder.parameters()),
        #     lr=1e-4,
        #     amsgrad=False,
        # )
        return torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            amsgrad=False,
            weight_decay=self.weight_decay,
        )

    def forward(self, x, xl, y_in):
        x = self.encoder(x=x)

        # Decoder
        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl)

        return y_out_hat

    def apply_teacher_forcing(self, y):
        # y.shape = [batch_size, seq_len]
        y_errored = y.clone()
        # Create a random mask with the same shape as y_errored
        random_mask = (
            torch.rand_like(y_errored, dtype=torch.float) < self.teacher_forcing_prob
        )
        # Create a mask for non-padding tokens
        non_padding_mask = y != self.padding_idx
        # Combine the random mask and non-padding mask
        combined_mask = random_mask & non_padding_mask
        # Generate random indices for the entire matrix
        random_indices = torch.randint(
            0, len(self.w2i), y_errored.shape, device=y_errored.device
        )
        # Apply the random indices only where the combined mask is True
        y_errored = torch.where(combined_mask, random_indices, y_errored)
        return y_errored

    def training_step(self, batch, batch_idx):
        x, xl, y_in, y_out = batch
        y_in = self.apply_teacher_forcing(y_in)
        yhat = self.forward(x=x, xl=xl, y_in=y_in)
        loss = self.compute_loss(yhat, y_out)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, xl, y_in, y_out = batch
        yhat = self.forward(x=x, xl=xl, y_in=y_in)
        loss = self.compute_loss(yhat, y_out)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return  # TODO temp
        x, y = batch
        assert x.size(0) == 1, "Inference only supports batch_size = 1"

        # Encoder
        x = self.encoder(x=x)
        # Prepare for decoder
        # 2D PE + flatten + permute
        x = self.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        # Autoregressive decoding
        y_in = torch.tensor([self.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(x.device)
        yhat = []
        for _ in range(self.max_seq_len):
            y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=None)
            y_out_hat = y_out_hat[0, :, -1]  # Last token
            y_out_hat_token = y_out_hat.argmax(dim=-1).item()
            y_out_hat_word = self.i2w[y_out_hat_token]
            yhat.append(y_out_hat_word)
            if y_out_hat_word == EOS_TOKEN:
                break

            y_in = torch.cat(
                [y_in, torch.tensor([[y_out_hat_token]]).long().to(x.device)], dim=1
            )

        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.no_grad()
    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        return  # TODO temp
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    @torch.no_grad()
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
