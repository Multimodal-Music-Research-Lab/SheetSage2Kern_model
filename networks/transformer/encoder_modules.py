import math

import torch
import torch.nn as nn

from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION, Encoder


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_channels, max_height, max_width, dropout_p: float = 0.1):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos_h = torch.arange(max_height).unsqueeze(1)
        pos_w = torch.arange(max_width).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, num_channels // 2, 2) / num_channels)

        pe = torch.zeros(1, max_height, max_width, num_channels)
        pe[0, :, :, 0 : num_channels // 2 : 2] = (
            torch.sin(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, 1 : num_channels // 2 : 2] = (
            torch.cos(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, num_channels // 2 :: 2] = (
            torch.sin(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe[0, :, :, (num_channels // 2) + 1 :: 2] = (
            torch.cos(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe = pe.permute(0, 3, 1, 2).contiguous()
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape = [batch_size, num_channels, h, w]
        x = x + self.pe[:, :, : x.size(2), : x.size(3)]
        return self.dropout(x)


class AudioEncoderBase(nn.Module):
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CnnEncoder(AudioEncoderBase):
    def __init__(
        self,
        max_encoder_output_length,
        max_audio_len,
    ):
        super().__init__()
        self.pos_2d = PositionalEncoding2D(
            num_channels=256,
            max_height=math.ceil(IMG_HEIGHT / HEIGHT_REDUCTION),
            max_width=math.ceil(max_audio_len / WIDTH_REDUCTION),
        )
        self.encoder = Encoder(in_channels=NUM_CHANNELS)

    def get_output_dim(self) -> int:  # Embedding dimension
        return 256

    def forward(self, x):
        x = self.encoder(x=x)

        x = self.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x
