import logging

import torch
from pytorch_lightning import LightningModule

from samantha.utils.hdfs_tools import hdfs_torch_load

logger = logging.getLogger(__name__)


class SoundStreamModel(LightningModule):
    pass


class SoundStreamSpeech24k(SoundStreamModel):
    """JIT files are generated as follows:

    ```bash
    git clone git@code.byted.org:litang.frank/bytegen.git && cd bytegen
    git checkout wavevae_mrd

    # on V100 gpu:
    python3 models/soundstream/convert_to_ts.py -c config.yaml --ckpt_path 460k_ckpt.pyt -t all --eps 1e-4
    ```
    """

    n_quantizers: int = 12
    codebook_size: int = 1024
    frame_rate: int = 50
    sample_rate: int = 24000
    encoder_fp: str = "hdfs:///home/byte_speech_sv/models/soundstream/soundstream_speech_24k/soundstream_speech_24k_encoder.pt"  # noqa
    decoder_fp: str = "hdfs:///home/byte_speech_sv/models/soundstream/soundstream_speech_24k/soundstream_speech_24k_decoder.pt"  # noqa

    def __init__(self):
        super().__init__()
        logger.warn(f"Loading checkpoint from HDFS: {self.encoder_fp}...")
        self.encoder = hdfs_torch_load(
            self.encoder_fp, jit=True, map_location="cpu"
        ).eval()
        logger.warn(f"Loading checkpoint from HDFS: {self.decoder_fp}...")
        self.decoder = hdfs_torch_load(
            self.decoder_fp, jit=True, map_location="cpu"
        ).eval()

    def n_frames(self, n_seconds: int):
        return n_seconds * self.frame_rate

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode(self, waveform):
        if self.encoder.training:
            self.encoder.eval()

        if waveform.ndim == 3:
            waveform = waveform.squeeze(dim=1)
        _, _, quant = self.encoder(waveform.float())
        return torch.stack(quant, dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def decode(self, x):
        if self.decoder.training:
            self.decoder.eval()
        return self.decoder(x)

    def forward(self, audio):
        return self.encode(audio)
