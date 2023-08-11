import logging

import torch
from pytorch_lightning import LightningModule

from recipes.umm.modules.lit_module import BestRQMel
from samantha.utils.hdfs_tools import hdfs_lightning_load_from_checkpoint

logger = logging.getLogger(__name__)

class BestRQMelModel(LightningModule):

    sample_rate: int = 16000
    frame_rate: int = 25
    codebook_size: int = 32768
    # ckpt_path: str = "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/finetune_speech_melrecon_25hz_rq8x4096_vq32768x256/checkpoints/step=080000-tr_loss=4.0822-val_loss_0=5.1172.ckpt"  # noqa
    ckpt_path: str = "./models/finetune_speech_melrecon_25hz_rq8x4096_vq32768x256/step=080000-tr_loss=4.0822-val_loss_0=5.1172.ckpt"

    def __init__(self):
        super().__init__()
        logger.warn(f"Loading checkpoint from HDFS: {self.ckpt_path}...")
        self.model = hdfs_lightning_load_from_checkpoint(
            BestRQMel, self.ckpt_path
        ).eval()

    def n_frames(self, n_seconds: int):
        return n_seconds * self.frame_rate

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode(self, audio):
        if self.model.training:
            self.model.eval()

        audio = audio.squeeze(dim=1)
        quant_encoder_out, quant_idx, quant_loss = self.model.get_quant_idx(audio)
        return quant_idx

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)


class BestRQMelCTCModel(BestRQMelModel):

    ckpt_path: str = "hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm/finetune_speech_melrecon_ctc_25hz_vq32768x256/checkpoints/step=020000-tr_loss=0.3844-val_loss_0=1.3190.ckpt"  # noqa
