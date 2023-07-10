import torch
from einops import rearrange
from torch import einsum, nn

from recipes.best_rq.models.flash_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)


class RandomProjectionQuantizerV2(nn.Module):
    """Random projection and codebook lookup module"""

    def __init__(self, input_dim, codebook_dim, codebook_size, num_quantizers=1):
        super().__init__()

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(num_quantizers, codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

        # input norm
        self.input_norm = nn.LayerNorm(input_dim)

    def codebook_lookup(self, x):
        # reshape
        b = len(x)
        x = rearrange(x, "b n e -> (b n) e")

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=-1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=-1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=1)

        # reshape
        xq = rearrange(nearest_indices, "c (b n) -> b n c", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x):
        # input norm
        x = self.input_norm(x)

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, conv_layers, kernel_size=5, input_channel=1):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        assert len(conv_layers) == 2

        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channel,
                conv_layers[0],
                self.kernel_size,
                2,
                self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_layers[0],
                conv_layers[1],
                self.kernel_size,
                2,
                self.kernel_size // 2,
            ),
            nn.ReLU(),
        )
        self.conv_out_size = conv_layers[1] * (idim // 2 // 2)
        self.linear = nn.Linear(self.conv_out_size, odim)

    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, idim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.linear(x)
        return x


class BestRqV2(nn.Module):
    def __init__(
        self,
        codebook_dim=16,
        codebook_size=8192,
        n_mels=128,
        conv_dim=512,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_conv_pos_embeddings=5,
        max_len=750,
        n_softmax=1,
        is_causal=False,
    ):
        super().__init__()

        self.n_softmax = n_softmax
        self.codebook_size = codebook_size

        # random quantizer
        self.quantizer = RandomProjectionQuantizerV2(
            n_mels * 4, codebook_dim, self.codebook_size, self.n_softmax
        )

        # two convolution layers + one projection layer
        self.conv = Conv2dSubsampling(n_mels, hidden_size, (conv_dim, conv_dim))

        # w2v-conformer
        config = Wav2Vec2ConformerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            position_embeddings_type="rotary",
            max_source_positions=max_len,
            num_attention_heads=num_attention_heads,
            num_conv_pos_embeddings=num_conv_pos_embeddings,
        )
        self.w2v_conformer = Wav2Vec2ConformerEncoder(config, is_causal)

        # projection
        self.linear = nn.Linear(hidden_size, codebook_size * self.n_softmax)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        emb = self.w2v_conformer(x)["last_hidden_state"]
        logits = self.linear(emb)
        return logits

    @torch.no_grad()
    def get_latent(self, x, layer_ix=12):
        x = self.conv(x)
        emb = self.w2v_conformer(x, output_hidden_states=True)["hidden_states"]
        return emb[layer_ix]

    def forward(self, batch):
        feature = batch["feature"][:, :, :-1]
        bs = feature.size(0)
        masked_feature = batch["masked_feature"][:, :, :-1]
        masked_indices = batch["token_domain_masked_indices"]
        # get target tokens
        target_tokens = self.quantizer(
            rearrange(feature, "b f (t s) -> b t (s f)", s=4)
        )
        num_uni_code = (
            sum(
                [
                    len(target_tokens[i, :, j].unique())
                    for i in range(bs)
                    for j in range(self.n_softmax)
                ]
            )
            / bs
            / self.n_softmax
        )
        # masking
        logits = self.encoder(masked_feature)
        # return logits and loss
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(masked_logits, "n (p c) -> (n c) p", c=self.n_softmax)
        masked_tokens = target_tokens[tuple(masked_indices.t())]
        masked_tokens = rearrange(masked_tokens, "n c -> (n c)", c=self.n_softmax)
        loss = self.criterion(masked_logits, masked_tokens)
        accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return {
            "logits": logits,
            "loss": loss,
            "accu": accu,
            "num_uni_code": num_uni_code,
        }


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


class BestRq(nn.Module):
    def __init__(
        self,
        codebook_dim=16,
        codebook_size=8192,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=24,
        num_attention_heads=8,
        num_conv_pos_embeddings=5,
        max_len=750,
        is_causal=False,
    ):
        super().__init__()

        self.codebook_size = codebook_size

        # random quantizer
        self.quantizer = RandomProjectionQuantizer(
            n_mels * 4, codebook_dim, self.codebook_size
        )

        # two convolution layers + one projection layer
        self.conv = Conv2dSubsampling(n_mels, encoder_dim, (conv_dim, conv_dim))

        # w2v-conformer
        config = Wav2Vec2ConformerConfig(
            hidden_size=encoder_dim,
            num_hidden_layers=encoder_depth,
            position_embeddings_type="rotary",
            max_source_positions=max_len,
            num_attention_heads=num_attention_heads,
            num_conv_pos_embeddings=num_conv_pos_embeddings,
        )
        self.w2v_conformer = Wav2Vec2ConformerEncoder(config, is_causal)
        self.input_norm = nn.LayerNorm(n_mels)
        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        emb = self.w2v_conformer(x)["last_hidden_state"]
        logits = self.linear(emb)
        return logits

    @torch.no_grad()
    def get_latent(self, x, layer_ix=12):
        x = self.conv(x)
        emb = self.w2v_conformer(x, output_hidden_states=True)["hidden_states"]
        return emb[layer_ix]

    def forward(self, batch):
        feature = batch["feature"][:, :, :-1]
        bs = feature.size(0)
        masked_feature = batch["masked_feature"][:, :, :-1]
        masked_indices = batch["token_domain_masked_indices"]
        # get target tokens
        target_tokens = self.quantizer(
            rearrange(feature, "b f (t s) -> b t (s f)", s=4)
        )
        num_uni_code = (
            sum(
                [
                    len(target_tokens[i, :, j].unique())
                    for i in range(bs)
                    for j in range(self.n_softmax)
                ]
            )
            / bs
            / self.n_softmax
        )
        # masking
        logits = self.encoder(masked_feature)
        # return logits and loss
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(masked_logits, "n (p c) -> (n c) p", c=self.n_softmax)
        masked_tokens = target_tokens[tuple(masked_indices.t())]
        masked_tokens = rearrange(masked_tokens, "n c -> (n c)", c=self.n_softmax)
        loss = self.criterion(masked_logits, masked_tokens)
        accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return {
            "logits": logits,
            "loss": loss,
            "accu": accu,
            "num_uni_code": num_uni_code,
        }
