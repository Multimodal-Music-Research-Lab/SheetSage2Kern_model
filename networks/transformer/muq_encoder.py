import librosa
import torch
from muq import MuQ

from networks.transformer.decoder import PositionalEncoding1D
from networks.transformer.encoder_modules import AudioEncoderBase

MUQ_DIMENSION_REDUCTION = 960
MUQ_OUTPUT_NUM_FEATURES = 1024


class MuqEncoder(AudioEncoderBase):
    def __init__(
        self,
        max_encoder_output_length,
        max_audio_len,
    ):
        super().__init__()
        # This will automatically fetch the checkpoint from huggingface
        muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = muq.eval()
        for param in self.muq.parameters():
            param.requires_grad = False

        self.pe = PositionalEncoding1D(
            emb_dim=MUQ_OUTPUT_NUM_FEATURES, max_len=max_encoder_output_length
        )

    def forward(self, x):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                enabled=False,  # TODO
            ):  # 16 bit precision was causing nan errors TODO check again
                x = x.to(torch.float32)
                x = self.muq(x, output_hidden_states=False).last_hidden_state
                x = self.pe(x)
                return x

    def get_output_dim(self) -> int:
        return MUQ_OUTPUT_NUM_FEATURES


class MuqEncoderPreprocessed(AudioEncoderBase):
    def __init__(
        self,
        max_encoder_output_length,
        max_audio_len,
    ):
        super().__init__()

        self.pe = PositionalEncoding1D(
            emb_dim=MUQ_OUTPUT_NUM_FEATURES, max_len=max_encoder_output_length
        )

    def forward(self, x):
        x = self.pe(x)
        return x

    def get_output_dim(self) -> int:
        return MUQ_OUTPUT_NUM_FEATURES
