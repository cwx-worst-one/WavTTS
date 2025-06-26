import numpy as np
import pytorch_lightning as pl
import torch

from recipes.musiclm.models.compat.semantic_model import (
    SSLFrontend,
    w2v_bert_tokenization,
)

W2V_BERT_20_PATHS = (
    "/mnt/bn/audio-diffusion/pretrained_models/w2v-BERT/v2.0/semantic.jit.pt",
    "/mnt/bn/audio-diffusion/pretrained_models/w2v-BERT/v2.0/centroids_epoch_10.npy",
)
W2V_CONFORMER_21_PATHS = (
    "/mnt/bn/audio-diffusion/pretrained_models/w2v/2.1/semantic.jit.pt",
    "/mnt/bn/audio-diffusion/pretrained_models/w2v/2.1/centroids_epoch_10.npy",
)


class SemanticModel(pl.LightningModule):
    def __init__(
        self,
        semantic_model_fp: str = W2V_CONFORMER_21_PATHS[0],
        centroids_fp: str = W2V_CONFORMER_21_PATHS[1],
    ):
        super().__init__()
        self.w2v_frontend = SSLFrontend().eval()
        self.w2v_model = torch.jit.load(semantic_model_fp, map_location="cpu").eval()
        self.register_buffer("centers", torch.from_numpy(np.load(centroids_fp)))
        self.n_frames = 250
        self.codebook_size = 1024
        self.frame_rate = 25

    @torch.no_grad()
    def forward(self, x: torch.Tensor, padding: bool = False) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(dim=1)
        with torch.autocast(device_type="cuda", enabled=False):
            self.w2v_model = self.w2v_model.eval()
            semantic_tokens = w2v_bert_tokenization(
                self.w2v_frontend,
                self.w2v_model,
                x.float(),
                self.centers,
                device=x.device,
            )
            if padding:
                semantic_tokens_padded = torch.zeros(
                    semantic_tokens.shape[0],
                    self.n_frames,
                    dtype=semantic_tokens.dtype,
                    device=semantic_tokens.device,
                )
                semantic_tokens_padded[:, 1:248] = semantic_tokens
                semantic_tokens_padded[:, 0] = semantic_tokens[:, 0]
                semantic_tokens_padded[:, -1] = semantic_tokens[:, -1]
                semantic_tokens_padded[:, -2] = semantic_tokens[:, -1]
                return semantic_tokens_padded
            return semantic_tokens
