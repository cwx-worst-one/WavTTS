from dataclasses import dataclass

import torch
from einops import rearrange

from samantha.dataio.data_bucket import data_bucket
from samantha.models.base import LightningModuleBase


@dataclass
class MusicFMResult:
    x: torch.Tensor


class MusicFM(LightningModuleBase):
    sample_rate: int = 24000
    frame_rate: int = 25
    precision = torch.float16
    encoder_depth: int = 12

    def __init__(self, layer_idx: int = 6, finetune: bool = False):
        super().__init__()
        self.model_path: str = data_bucket(
            "models/musicfm/musicfm_25hz_playlist_330m_520k.pt"
        )
        self.stat_path: str = data_bucket("models/musicfm/playlist_classic_stats.json")
        self.layer_idx = layer_idx
        self.finetune = finetune

        # self.model = MusicFM25Hz(
        #     encoder_depth=self.encoder_depth,
        #     is_flash=False,  # TODO check equivalence
        #     stat_path=self.stat_path,
        #     model_path=self.model_path,
        # )

        if not finetune:
            self.freeze()
            self.model.eval()

    @property
    def output_dim(self):
        return self.model.linear.in_features

    @torch.cuda.amp.autocast(enabled=True, dtype=precision)
    def forward(self, x: torch.Tensor) -> MusicFMResult:
        if not self.finetune:
            self.model.eval()

        with torch.set_grad_enabled(self.finetune):
            out = self.model.get_latent(x, layer_ix=self.layer_idx)
            out = rearrange(out, "b t c -> b c t")
            return MusicFMResult(out)
