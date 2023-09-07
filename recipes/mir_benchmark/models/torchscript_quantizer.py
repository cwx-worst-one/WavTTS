import torch
from einops import rearrange
from torch import einsum, nn


class RandomProjectionQuantizer(nn.Module):
    """Random projection and codebook lookup module"""

    def __init__(self, input_dim, codebook_dim, codebook_size, seed=142):
        super().__init__()

        # random seed
        torch.manual_seed(seed)

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

        # input norm
        self.input_norm = nn.LayerNorm(input_dim)

    def codebook_lookup(self, x):
        # reshape
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)

        # reshape
        xq = rearrange(nearest_indices, "(b n) -> b n", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x):
        # always eval
        self.eval()

        # input norm
        x = self.input_norm(x)

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq


class BEST_RQ_RQ(nn.Module):
    """
    Simplified BEST-RQ with CNN + T5 Encoder
    Latent representation is quantized with a randomly initialized projection and codebook
    """

    def __init__(
        self,
        latent_dim=1024,
        codebook_dim=16,
        codebook_size=8192,
        torchscript_path="/mnt/bn/audio-diffusion/torchscript/best_rq/chromatic_80k.pt",
    ):
        super().__init__()

        # pretrained torchscript
        self.best_rq = torch.jit.load(torchscript_path)

        # random quantizer
        self.quantizer = RandomProjectionQuantizer(
            latent_dim, codebook_dim, codebook_size
        )

    def get_latent(self, x, layer_ix):
        _, hidden_states = self.best_rq(x)
        return self.quantizer(hidden_states[layer_ix])
