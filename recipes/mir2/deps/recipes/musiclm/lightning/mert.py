import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from vector_quantize_pytorch import ResidualVQ

from recipes.musiclm.lightning.base import Identity


class MERTModel(pl.LightningModule):
    def __init__(self, train_head: bool = False):
        super().__init__()
        self.train_head = train_head

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.model = self.model.eval()

        if train_head:
            self.head = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        else:
            self.head = Identity()

        self.sample_rate = 24000
        self.pretrain_context_sec = 5
        self.pretrain_context_samples = self.sample_rate * self.pretrain_context_sec

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Generates MERT embeddings from an audio tensor (sampled at 24kHz).
        MERT generates 25 representation layers (the last dimension).
        We can select the representation empirically, as each performs
        differently per downstream task, or learn an weighted average.
        More details at: https://huggingface.co/m-a-p/MERT-v1-330M

        Args:
            audio (torch.Tensor): Audio tensor

        Returns:
            torch.Tensor: MERT embeddings of shape
            (batch, sequence length, embedding dim, representation layer)
        """
        if audio.ndim == 2:
            audio = audio.unsqueeze(dim=1)

        inputs = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )["input_values"]
        inputs = inputs[0].to(audio.device)
        hiddens = []
        for i in inputs.split(self.pretrain_context_samples, dim=2):
            with torch.no_grad():
                self.model = self.model.eval()
                h = torch.stack(
                    self.model(
                        i.squeeze(dim=1), output_hidden_states=True
                    ).hidden_states
                )
                n_embd = h.shape[-1]
                h = rearrange(h, "r b s n_embd -> b r (s n_embd)")

            if self.train_head:
                h = self.head(h)
            else:
                h = h.mean(dim=1, keepdim=True)
            hiddens.append(rearrange(h, "b 1 (s n_embd) -> b s n_embd", n_embd=n_embd))
        hiddens = torch.cat(hiddens, dim=1)
        return hiddens

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.embed(audio)


class QuantizedMERTModel(MERTModel):
    def __init__(
        self,
        train_head: bool = False,
        n_codebooks: int = 1,
        codebook_size: int = 1024,
        rq_ema_decay: float = 0.95,
        train_rvq: bool = False,
        threshold_ema_dead_code: float = 0.0,
    ):
        super().__init__(train_head)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.train_rvq = train_rvq

        self.mert_emb_dim = 1024

        self.rq = ResidualVQ(
            dim=self.mert_emb_dim,
            num_quantizers=self.n_codebooks,
            codebook_size=self.codebook_size,
            decay=rq_ema_decay,
            commitment_weight=0,  # embeddings are frozen, no need for commitment loss
            kmeans_init=False,
            threshold_ema_dead_code=threshold_ema_dead_code,
            quantize_dropout=False,  # no quantize dropout
        )

    def quantize(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.train_rvq):
            self.rq.train(self.train_rvq)
            _, indices, _ = self.rq(latents)
        return indices

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Quantize the MERT embeddings, and also take the arithmetic mean on the
        time dimension to get a summary of the full sequence and reduce dimensionality.

        Args:s
            audio (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        latents = self.embed(audio)
        return self.quantize(latents)
