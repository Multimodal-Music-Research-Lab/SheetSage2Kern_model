import torch
import torch.nn as nn
from muq import MuQ

from networks.transformer.decoder import PositionalEncoding1D
from networks.transformer.encoder_modules import AudioEncoderBase

MUQ_DIMENSION_REDUCTION = 960
MUQ_OUTPUT_NUM_FEATURES = 1024

EMB_LAYER_SIZE = 256


class LinearBridge(nn.Module):
    def __init__(self, muq_dim=1024, model_dim=256):
        super().__init__()
        self.proj = nn.Linear(muq_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MuqEncoder(AudioEncoderBase):
    def __init__(
        self,
        max_encoder_output_length,
        max_audio_len,
    ):
        super().__init__()
        muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = muq.eval()
        for param in self.muq.parameters():
            param.requires_grad = False

        self.pe = PositionalEncoding1D(
            emb_dim=EMB_LAYER_SIZE, max_len=max_encoder_output_length
        )
        self.proj_layer = LinearBridge()

    def forward(self, x):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                enabled=False,  # TODO
            ):  # 16 bit precision causes nan errors
                x = x.to(torch.float32)
                x = self.muq(x, output_hidden_states=False).last_hidden_state
        x = self.proj_layer(x)
        x = self.pe(x)
        return x

    def get_output_dim(self) -> int:
        return EMB_LAYER_SIZE


class MuqEncoderPreprocessed(AudioEncoderBase):
    def __init__(
        self,
        max_encoder_output_length,
        max_audio_len,
    ):
        super().__init__()

        self.pe = PositionalEncoding1D(
            emb_dim=EMB_LAYER_SIZE, max_len=max_encoder_output_length
        )

        self.proj_layer = LinearBridge()

    def forward(self, x):
        x = self.proj_layer(x)
        x = self.pe(x)
        return x

    def get_output_dim(self) -> int:
        return EMB_LAYER_SIZE
