import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from samantha.nn.layers import WNConv1d
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

        self.init_codebook()

        self.rearrange_latents = Rearrange("b d t -> (b t) d")

    def init_codebook(self, min_distance: float = 1.0):
        min_bits = math.log2(self.codebook_size)
        logger.info(f"Initialising codebook with minimum entropy (bits): {min_bits}...")

        n_attempts = 0
        while True:
            n_attempts += 1
            self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

            entropy = self.entropy()
            bits = entropy.log2()

            if n_attempts % 100 == 0:
                logger.info(
                    f"Initialising codebook entropy (bits): {entropy.log2()} ({n_attempts} attempts)"
                )

            if abs(bits - min_bits) < min_distance:
                break

        logger.info(
            f"Codebook entropy (bits): {entropy.log2()} ({n_attempts} attempts)"
        )

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        indices = self.decode_pre_latents(z_e)
        z_q = self.decode_code(indices)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_latent = self.out_proj(z_q)
        return z_latent, commitment_loss, codebook_loss, indices, z_e, z_q

    def _forward(self, z: torch.Tensor):
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        indices = self.decode_pre_latents(z_e)
        z_q = self.decode_code(indices)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_latent = self.out_proj(z_q)
        return z_latent, indices, z_e, z_q

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_pre_latents(self, latents: torch.Tensor) -> torch.Tensor:
        encodings = self.rearrange_latents(latents)
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(latents.size(0), -1)
        return indices

    def get_codes_from_z(self, z: torch.Tensor) -> torch.Tensor:
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        indices = self.decode_pre_latents(z_e)
        return indices

    def get_post_latents_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.decode_code(codes)
        return self.out_proj(z_q)

    def get_post_latents_from_z(self, z: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_z(z)
        return self.get_post_latents_from_codes(codes)

    @torch.no_grad()
    def entropy(self):
        p = self.codebook.weight / self.codebook.weight.sum()
        p = p.clamp(1e-12)
        entropy = (-p * p.log()).sum()
        return entropy

    @torch.no_grad()
    def codebook_magnitudes(self) -> torch.Tensor:
        norm = self.codebook.weight.norm(p=2, dim=1)
        return norm

    @torch.no_grad()
    def codebook_parwise_distance(self):
        embeddings = self.codebook.weight.data
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

        min_distance = torch.min(
            pairwise_distances
            + torch.eye(pairwise_distances.shape[0], device=pairwise_distances.device)
            * pairwise_distances.max()
        )
        return {
            "mean_distance": pairwise_distances.mean(),
            "min_distance": min_distance,
            "max_distance": pairwise_distances.max(),
        }


class NSVQ(nn.Module):
    """
    Implementation of NSVQ
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.rearrange_latents = Rearrange("b d t -> (b t) d")
        self.eps = 1e-12

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        indices = self.decode_pre_latents(z_e)
        z_q = self.decode_code(indices)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        # codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        # commitment_loss = torch.zeros(z_q.shape[0])
        codebook_loss = torch.zeros(z_q.shape[0])

        noise = torch.randn_like(z_q, requires_grad=False)

        norm_res = (z_q - z_e).norm(p=2, dim=1)
        norm_noise = noise.norm(p=2, dim=1)

        vq_error = (norm_res / norm_noise + self.eps).unsqueeze(dim=1) * noise

        z_q = z_e + vq_error  # TODO: .detach() on z_e?

        z_latent = self.out_proj(z_q)
        return z_latent, commitment_loss, codebook_loss, indices, z_e, z_q

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_pre_latents(self, latents: torch.Tensor) -> torch.Tensor:
        encodings = self.rearrange_latents(latents)
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(latents.size(0), -1)
        return indices

    def get_codes_from_z(self, z: torch.Tensor) -> torch.Tensor:
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        indices = self.decode_pre_latents(z_e)
        return indices

    def get_post_latents_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.decode_code(codes)
        return self.out_proj(z_q)

    def get_post_latents_from_z(self, z: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_z(z)
        return self.get_post_latents_from_codes(codes)

    @torch.no_grad()
    def entropy(self):
        p = self.codebook.weight / self.codebook.weight.sum()
        p = p.clamp(1e-9)
        entropy = (-p * p.log()).sum()
        return entropy

    @torch.no_grad()
    def codebook_magnitudes(self) -> torch.Tensor:
        norm = self.codebook.weight.norm(p=2, dim=1)
        return norm

    @torch.no_grad()
    def codebook_parwise_distance(self):
        embeddings = self.codebook.weight.data
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

        min_distance = torch.min(
            pairwise_distances
            + torch.eye(pairwise_distances.shape[0], device=pairwise_distances.device)
            * pairwise_distances.max()
        )
        return {
            "mean_distance": pairwise_distances.mean(),
            "min_distance": min_distance,
            "max_distance": pairwise_distances.max(),
        }


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers: nn.ModuleList[VectorQuantize] = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    @property
    def total_codebook_dim(self) -> int:
        return sum([q.codebook_dim for q in self.quantizers])

    def dropout_quantizers(self, batch_size: int, device: torch.device):
        n_quantizers = torch.ones((batch_size,)) * self.n_codebooks + 1
        dropout = torch.randint(1, self.n_codebooks + 1, (batch_size,))
        n_dropout = int(batch_size * self.quantizer_dropout)
        n_quantizers[:n_dropout] = dropout[:n_dropout]
        return n_quantizers.to(device)

    def forward(self, z, n_quantizers: Optional[int] = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        latents = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        pre_latents = []
        z_q = []
        z_q_masked = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if self.training:
            n_quantizers = self.dropout_quantizers(z.shape[0], z.device)

        for i, quantizer in enumerate(self.quantizers):
            # if self.training is False and i >= n_quantizers:
            #     break

            latents_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i, z_q_i = (
                quantizer(residual)
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            latents = latents + latents_i * mask[:, None, None]
            residual = residual - latents_i

            z_q_i_masked = z_q_i * mask[:, None, None]

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            pre_latents.append(z_e_i)
            z_q.append(z_q_i)
            z_q_masked.append(z_q_i_masked)

        codes = torch.stack(codebook_indices, dim=1)
        pre_latents = torch.cat(pre_latents, dim=1)
        z_q = torch.cat(z_q, dim=1)
        z_q_masked = torch.cat(z_q_masked, dim=1)

        return (
            latents,
            codes,
            pre_latents,
            z_q,
            z_q_masked,
            commitment_loss,
            codebook_loss,
        )

    def get_codes_from_z(self, z: torch.Tensor) -> torch.Tensor:
        latents = 0
        residual = z
        codebook_indices = []
        for i, quantizer in enumerate(self.quantizers):
            latents_i, indices_i, z_e_i, z_q_i = quantizer._forward(residual)
            latents = latents + latents_i
            residual = residual - latents_i
            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=1)
        return codes

    def get_z(self, h) -> torch.Tensor:
        latents = 0
        residual = h
        z_e = []

        for i, quantizer in enumerate(self.quantizers):
            z_e_i = quantizer.forward_pre_z(residual)
            latents_i, _ = quantizer.latents_from_pre_z(z_e_i)
            latents = latents + latents_i
            residual = residual - latents_i
            z_e.append(z_e_i)

        z_e = torch.cat(z_e, dim=1)
        return z_e

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def get_post_latents_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q, _, codes = self.from_codes(codes)
        return z_q

    def z_from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the codebook's continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Bottleneck's continuous representation of input
        """
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
        return torch.cat(z_p, dim=1)

    def latents_from_pre_z(self, z_e: torch.Tensor):
        latents = 0.0
        z_q = []
        # TODO: variable codebook
        for i, quantizer in enumerate(self.quantizers):
            step = self.codebook_dim[i]
            min_i = i * step
            max_i = (i + 1) * step

            latents_i, z_q_i = quantizer.latents_from_pre_z(z_e[:, min_i:max_i])
            latents = latents + latents_i
            z_q.append(z_q_i)

        z_q = torch.cat(z_q, dim=1)
        return latents, z_q

    def latents_from_post_z(self, z_q: torch.Tensor):
        latents = 0.0
        # TODO: variable codebook
        for i, quantizer in enumerate(self.quantizers):
            step = self.codebook_dim[i]
            min_i = i * step
            max_i = (i + 1) * step
            latents_i = quantizer.out_proj(z_q[:, min_i:max_i])
            latents = latents + latents_i
        return latents


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)
