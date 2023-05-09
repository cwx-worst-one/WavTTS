import torch
import torch.nn.functional as F
from einops import repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from recipes.mae.models.mut import RMSNorm, Transformer


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        decoder_checkpointing=False,
        decoder_use_flash_attn=False,
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        encoder_dim = encoder.dim

        self.logmel_frontend = encoder.logmel_frontend
        self.to_patch_embedding = encoder.to_patch_embedding

        pixel_values_per_patch = encoder.patch_dim

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.pre_dec_norm = RMSNorm(encoder_dim)
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
            checkpointing=decoder_checkpointing,
            use_flash_attn=decoder_use_flash_attn,
        )
        self.decoder_rotary_emb = RotaryEmbedding(dim=int(decoder_dim / decoder_heads))
        # TODO: add norm befor to_pixels
        self.pixel_norm = RMSNorm(decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, audio):
        device = audio.device

        # to logmel spectrogram
        mel_spec = self.logmel_frontend["logmel"](audio)
        origin_mel_spec = nn.Identity()(mel_spec)
        # get patches
        for k, layer in self.to_patch_embedding.items():
            mel_spec = layer(mel_spec)
            if k == "to_patch":
                patches = nn.Identity()(mel_spec)

        tokens = nn.Identity()(mel_spec)
        batch, num_patches, _ = tokens.shape

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )
        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with transformer

        encoded_tokens = self.encoder.transformer(
            tokens, self.encoder.rotary_emb, (masked_indices, unmasked_indices)
        )

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        encoded_tokens = self.pre_dec_norm(encoded_tokens)
        unmasked_decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device, dtype=mask_tokens.dtype
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens.to(
            decoder_tokens.dtype
        )
        decoder_tokens[batch_range, masked_indices] = mask_tokens.to(
            decoder_tokens.dtype
        )
        decoded_tokens = self.decoder(decoder_tokens, self.decoder_rotary_emb)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(self.pixel_norm(mask_tokens))

        # calculate reconstruction loss
        # TODO: mae paper suggests predicting the norm patches

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss, origin_mel_spec, masked_indices, pred_pixel_values


if __name__ == "__main__":
    from recipes.mae.models.mut import MuT

    mut = MuT(
        spec_shape=(128, 1000),
        patch_shape=(128, 2),
        num_classes=1000,
        sample_rate=24000,
        dim=1024,
        depth=6,
        heads=8,
        dim_head=128,
        channels=1,
        mlp_dim=2048,
    )

    mae = MAE(
        encoder=mut,
        masking_ratio=0.75,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=6,  # anywhere from 1 to 8
    )
    dummy_input = torch.randn(3, 1, 240000)
    loss = mae(dummy_input)
    print(loss)
