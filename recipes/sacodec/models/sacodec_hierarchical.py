import random
import torch
import torchaudio
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from recipes.sacodec.models.components.spectral_ops import AMP_PHA_Spectrum
from recipes.sacodec.models.components.vae_bottleneck import VAEBottleneck
from recipes.sacodec.models.components.convnext import ConvNeXtBlock, ConvNextBackboneDownUp
from recipes.sacodec.models.components.conformer_next import ConformerNextBackboneDownUp
from recipes.sacodec.models.sacodec_umm import UMMLoss

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def upsample_head(input_dim, output_dim):
    return nn.Sequential(
        WNConv1d(input_dim, output_dim*2, kernel_size=1),
        nn.GELU(),
        WNConv1d(output_dim*2, output_dim, kernel_size=1),
    )

class VAEVectorizer(nn.Module):
    def __init__(self, decoder_dim: int, sub_dim: int, beta=1e-5):
        super().__init__()

        if decoder_dim == sub_dim:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            # self.in_proj = upsample_head(decoder_dim, sub_dim)
            self.out_proj = upsample_head(sub_dim, decoder_dim)
        self.bottleneck = VAEBottleneck(in_channels=decoder_dim, latent_dim=sub_dim, beta=beta)
        
    def forward(self, z):
        # z_emb = self.in_proj(z)  # z_e : (B x D x T)
        z_emb = z
        # TODO: should we use F.normalize before sending to bottleneck? See: ResidualVectorQuantize
        vae_latent, kl_loss = self.bottleneck(z_emb)
        # TODO: should we use mu instead of z_latent?

        # # TODO: figure out what this is for.
        # z_latent = (
        #     z_emb + (z_latent - z_emb).detach()
        # )  # noop in forward pass, straight-through gradient estimator in backward pass
        expand_latent = self.out_proj(vae_latent)

        return expand_latent, kl_loss, vae_latent

    def features_to_decoder_latents(self, z):
        return self.out_proj(z)
    

class ResidualVAEVectorizer(nn.Module):
    def __init__(self, decoder_dim: int, sub_dim: int, beta=1e-5, n_vectorizers=4, vector_dropout=False, skip_semantic_residual=False):
        super().__init__()
        if isinstance(sub_dim, list):
            assert len(sub_dim) == n_vectorizers, "sub_dim list must have length of n_vectorizers"
            sub_dims = sub_dim
        elif isinstance(sub_dim, int):
            sub_dims = [sub_dim] * n_vectorizers
        self.sub_dims = sub_dims
        self.vectorizers = nn.ModuleList(
            [
                VAEVectorizer(decoder_dim, sub_dim, beta)
                for sub_dim in sub_dims
            ]
        )
        self.vector_dropout = vector_dropout
        self.skip_semantic_residual = skip_semantic_residual

    def split_features(self, features, dim=1):
        """Splits features into residual sub dimensions"""
        return torch.split(features, self.sub_dims, dim=dim)
    
    def forward(self, audio_features):
        residual = audio_features
        kl_loss = 0
        hierarchical_latents = []
        decoder_latents = 0

        if self.training and self.vector_dropout:
            n_sub_dims = len(self.sub_dims)
            n_dropout = random.randint(self.vector_dropout, n_sub_dims+3)
        else:
            n_dropout = len(self.sub_dims) + 1

        for idx, vectorizer in enumerate(self.vectorizers):
            decoder_latent, kl, diffusion_latent = vectorizer(residual)
            kl_loss += kl
            if idx < n_dropout:
                decoder_latents += decoder_latent
            hierarchical_latents.append(diffusion_latent)
            if idx == 0 and self.skip_semantic_residual:
                pass
            else:
                residual = residual - decoder_latent
        # hierarchical_latents = torch.cat(hierarchical_latents, dim=1) # return list instead
        return decoder_latents, kl_loss, hierarchical_latents, n_dropout

class STFTEncoderVAEHierarchicalUMMLoss(nn.Module):
    def __init__(self, 
                 n_fft=1024, hop_length=256, win_length=None,
                 sample_rate=44100,
                 vae_dim=128, beta=1e-5, hidden_size=1536, audio_channels=2, block_layers=[1,4,4], even_pad=True,
                 kernel_size=7, vae_sub_dim=32, n_vectorizers=4, umm_version="umm_vq", vector_dropout=False,
                 skip_semantic_residual=False, umm_block_layers=0, vae_hz=50
                 ):
        super().__init__()
        self.audio_channels = audio_channels
        if win_length is None: win_length = n_fft
        
        self.vae_dim = vae_dim
        
        N_CH = audio_channels * 2
        pre_channels = n_fft // 4

        # Feature encoder
        self.spec_encoder = AMP_PHA_Spectrum(n_fft=n_fft, hop_length=hop_length, win_length=win_length, audio_channels=audio_channels)

        # Pre-encoder
        T_PAD = 2 if even_pad else 3
        self.pre_conv = weight_norm(nn.Conv2d(N_CH, N_CH, (7, 7), stride=(2, 2),
                                        padding=(2, T_PAD)))
        self.pre_norm = nn.LayerNorm(pre_channels * N_CH, eps=1e-6)
        self.pre_convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=pre_channels*N_CH,
                    intermediate_dim=pre_channels*N_CH,
                    layer_scale_init_value=1/16
                ) for _ in range(block_layers[0])
            ]
        )

        # Down 1
        self.conv1 = ConvNextBackboneDownUp(
            # TODO: move this all into a config
            input_channels=pre_channels * N_CH,
            dim=hidden_size,
            intermediate_dim=hidden_size*2,
            num_layers=block_layers[1],
            ratio=3,
            apply_final_layer_norm=False,
            kernel_size=kernel_size
        )

        # Down 2 - replace with conformer
        self.conformer2 = ConformerNextBackboneDownUp(
            input_channels=hidden_size,
            dim=hidden_size,
            intermediate_dim=hidden_size*2,
            num_layers=block_layers[2],
            ratio=3,
            apply_final_layer_norm=True
        )
        self.vectorizer = ResidualVAEVectorizer(
            decoder_dim=vae_dim, sub_dim=vae_sub_dim, beta=beta, n_vectorizers=n_vectorizers, 
            vector_dropout=vector_dropout, skip_semantic_residual=skip_semantic_residual
        )
        self.audio_post = nn.Linear(hidden_size, vae_dim)

        semantic_dim = self.vectorizer.sub_dims[0]
        if umm_version == "no_loss":
            def no_loss(*args, **kwargs): return {}
            self.umm_loss = no_loss
        else:
            self.umm_loss = UMMLoss(vae_dim=semantic_dim, hidden_size=hidden_size, umm_version=umm_version, block_layers=umm_block_layers, vae_hz=vae_hz) # first quantizer is semantic

    def audio_to_spec(self, audio):
        return self.spec_encoder(audio)

    def forward(self, logamp, pha, return_loss=True, audio_input_24k_mono=None, **kwargs):
        assert len(logamp.shape) == 4 and logamp.shape[1] == self.audio_channels, f"Invalid number of channels {logamp.shape}"
        
        features = torch.cat([logamp, pha], dim=1) # append 

        features = self.pre_conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L).transpose(1, 2) # -> B L C*N, for layer norm
        features = self.pre_norm(features)
        features = features.transpose(1, 2) # B C L
        
        for pre_convnext in self.pre_convnext:
            features = pre_convnext(features)

        features = self.conv1(features)
        # features = self.conv2(features) # B D L
        features = self.conformer2(features)

        audio_features = self.audio_post(features.transpose(1, 2)).transpose(1, 2) # B D L


        decoder_latents, kl_loss, hierarchical_latents, hierarchical_latents_list, n_vector_dropout = self.vectorizer(audio_features)
        semantic_features, *acoustic_features = self.vectorizer.split_features(hierarchical_latents)
        umm_loss_results = self.umm_loss(semantic_features, audio_input_24k_mono=audio_input_24k_mono, return_loss=return_loss)

        # input to decoder: residual + semantic 
        # vae: residual.
        # decoder_latents = B VAE_DIM L
        # hierarchical_latents = B x sub_dim*n_vectorizers x L
        return {
            **umm_loss_results, # umm_cosine_loss, umm_l1_loss, umm_features
            "latent": decoder_latents, # combined features for decoder
            "hierarchical_latents": hierarchical_latents, # B x D x L
            "hierarchical_latents_list": hierarchical_latents_list,
            "kl_loss": kl_loss,
            "features": features,
            'n_vector_dropout': n_vector_dropout,
        }
    
    def features_to_decoder_latents(self, latents, skip_idxs=None):
        features = self.vectorizer.split_features(latents)
        latents = 0
        for i, chunk in enumerate(features):
            if skip_idxs and i in skip_idxs:
                continue
            vectorizer = self.vectorizer.vectorizers[i]
            emb = vectorizer.features_to_decoder_latents(chunk)
            latents += emb
        return latents

    def normalize_features(self, features: torch.Tensor, mean, std):
        if mean == 0 and std == 1: return features
        features = self.vectorizer.split_features(features, dim=-1)
        features = [(f - m) / s for f, m, s in zip(features, mean, std)]
        return torch.cat(features, dim=-1)

    def denormalize_features(self, features: torch.Tensor, mean, std):
        if mean == 0 and std == 1: return features
        features = self.vectorizer.split_features(features, dim=-1)
        features = [f * s + m for f, m, s in zip(features, mean, std)]
        return torch.cat(features, dim=-1)