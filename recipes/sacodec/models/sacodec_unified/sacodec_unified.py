import os
from typing import List, Optional, Tuple
import torch
from torch import nn
import math
from torch.nn.utils.parametrizations import weight_norm
from recipes.sacodec.models.components.spectral_ops import AMP_PHA_Spectrum
from recipes.sacodec.models.components.vae_bottleneck import VAEBottleneck
from recipes.sacodec.models.components.istft_head import ISTFTHeadStereo

from recipes.sacodec.models.sacodec_unified.components import ConvNextBackbone, DownUpBlock, LayerNorm, ConvNextChannelReduction, ConformerConfig, ConformerNextBlock, TransposeLast
from recipes.sacodec.models.sacodec_unified.losses import BestRQMaskedLoss, UMMLoss
from recipes.sacodec.models.sacodec_unified.bottlenecks import VAEBottleneck, VAEBottleneckV2, VAEBottleneckV3, ResidualVAEVectorizer
# from recipes.sacodec.models.sacodec_umm import UMMLoss
from dataclasses import dataclass, asdict
import ast

@dataclass
class STFTEncoderUnifiedConfig:
    n_fft: int = 1024
    hop_length: int = 49
    vae_dim: int = 128
    beta: float = 1e-3
    hidden_size: int = 1536
    audio_channels: int = 2
    bottleneck_framerate: int = 50
    sample_rate = 44100


    ratio_block_dim: list[list[int]] = ((3, 3, 2), (2, 3, 4), (1536, 1536, 1536))
    scale_spatial_channels: bool = False # DCAE - scale channels by ratio
    shortcut: bool = True
    mlp_ratio: float = 3

    semantic_loss_type: str = "umm_rope" # umm_rope | umm | bestrq | no_loss
    semantic_alignment_position: str = "pre" # pre | post | h (hierarchical)
    semantic_n_blocks: int = 1
    bottleneck_type: str = "v1" # v1 | v2 | h (hierarchical)

    # hierarchcial settings
    sub_dims: list[int] = None
    vector_dropout: int = 1
    skip_semantic_residual: bool = False

    pre_embed_type: str = "post_linear"
    atan2_magnitude_threshold_ratio: float = 0.0
    use_grn: bool = False

### Block Layers
# 2 x 2 x 44100 <- Audio
# 2 x 4 x 512+1 x 900 <- spec + stack, preconv
# 2 x 2052 x 900 <- reshape
# 2 x 512 x 450 <- down1
# 2 x 512 x 150 <- down2
# 2 x 512 x 50 <- down3
# 2 x 128 x 50 <- bottleneck 
### Decoder
# 2 x 128 x 50
# 2 x 1536 x 150 <- up1
# 2 x 1536 x 450 <- up2
# 2 x 1536 x 900 <- up3
# 2 x 6 x 512+1 x 900 <- linear
# 2 x 2 x 44100 <- istft


class PreEmbed_PostLinear(nn.Module):
    # post linear
    def __init__(self, 
                 audio_channels, out_channels, n_fft, n_blocks=1
                 ):
        super().__init__()
        
        ## Pre convolution
        audio_feat_channels = audio_channels * 2 # audio x featrues
        pre_channels = (n_fft // 2 + 1) * audio_feat_channels # 2 audio channels
        self.conv = ConvNextBackbone(dim=audio_feat_channels, num_layers=n_blocks, use_2d=True)

        self.norm = LayerNorm(pre_channels, eps=1e-6, data_format="channels_first")
        self.linear = nn.Sequential(
            TransposeLast(), # B C L ->  B L C
            nn.Linear(pre_channels, out_channels*2),
            nn.GELU(),
            nn.Linear(out_channels*2, out_channels),
            TransposeLast(), # B C N_FFT L
        )

    def forward(self, logamp, pha):
        features = torch.concat([logamp, pha], dim=1) # stack - audio dim
        features = self.conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L) # -> B L C*N, for layer norm
        features = self.norm(features)
        features = self.linear(features)
        return features


class PreEmbed_PostLinearSimple(nn.Module):
    # post linear
    def __init__(self, 
                 audio_channels, out_channels, n_fft, n_blocks=1
                 ):
        super().__init__()
        
        ## Pre convolution
        audio_feat_channels = audio_channels * 2 # audio x featrues
        pre_channels = (n_fft // 2 + 1) * audio_feat_channels # 2 audio channels
        # self.conv = ConvNextBackbone(dim=audio_feat_channels, num_layers=n_blocks, use_2d=True)
        self.conv = nn.Conv2d(audio_channels * 2, audio_channels * 2, kernel_size=(7, 7), padding=(3, 3))

        self.norm = LayerNorm(pre_channels, eps=1e-6, data_format="channels_first")
        self.linear = nn.Sequential(
            TransposeLast(), # B C L ->  B L C
            nn.Linear(pre_channels, out_channels*2),
            nn.GELU(),
            nn.Linear(out_channels*2, out_channels),
            TransposeLast(), # B C N_FFT L
        )

    def forward(self, logamp, pha):
        features = torch.concat([logamp, pha], dim=1) # stack - audio dim
        features = self.conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L) # -> B L C*N, for layer norm
        features = self.norm(features)
        features = self.linear(features)
        return features


class PreEmbed_PostConv(nn.Module):
    # post linear
    def __init__(self, 
                 audio_channels, out_channels, n_fft, n_blocks=1
                 ):
        super().__init__()
        
        ## Pre convolution
        audio_feat_channels = audio_channels * 2 # audio x featrues
        pre_channels = (n_fft // 2 + 1) * audio_feat_channels # 2 audio channels
        self.conv = ConvNextBackbone(dim=audio_feat_channels, num_layers=n_blocks, use_2d=True)
        self.post_conv = ConvNextChannelReduction(pre_channels, out_channels)

    def forward(self, logamp, pha):
        features = torch.concat([logamp, pha], dim=1) # stack - audio dim
        features = self.conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L) # -> B L C*N, for layer norm
        features = self.post_conv(features)
        return features

class PreEmbed_PostConvSimple(nn.Module):
    # post linear
    def __init__(self, 
                 audio_channels, out_channels, n_fft, n_blocks=1
                 ):
        super().__init__()
        
        ## Pre convolution
        audio_feat_channels = audio_channels * 2 # audio x featrues
        pre_channels = (n_fft // 2 + 1) * audio_feat_channels # 2 audio channels
        # self.conv = ConvNextBackbone(dim=audio_feat_channels, num_layers=n_blocks, use_2d=True)
        self.post_conv = ConvNextChannelReduction(pre_channels, out_channels)

    def forward(self, logamp, pha):
        features = torch.concat([logamp, pha], dim=1) # stack - audio dim
        # features = self.conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L) # -> B L C*N, for layer norm
        features = self.post_conv(features)
        return features

class PreEmbed_V1(nn.Module):
    # post linear
    def __init__(self, 
                 audio_channels, out_channels, n_fft, n_blocks=1
                 ):
        super().__init__()
        
        ## Pre convolution
        audio_feat_channels = audio_channels * 2 # audio x featrues
        # pre_channels = (n_fft // 2 + 1) * audio_feat_channels # 2 audio channels
        pre_channels = n_fft // 4
        # self.conv = ConvNextBackbone(dim=audio_feat_channels, num_layers=n_blocks, use_2d=True)

        self.pre_conv = weight_norm(nn.Conv2d(audio_feat_channels, audio_feat_channels, (7, 7), stride=(2, 1),
                                        padding=(2, 3)))
        self.post_conv = ConvNextChannelReduction(pre_channels * audio_feat_channels, out_channels)

    def forward(self, logamp, pha):
        features = torch.concat([logamp, pha], dim=1) # stack - audio dim
        features = self.pre_conv(features)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L) # -> B L C*N, for layer norm
        features = self.post_conv(features)
        return features

class STFTEncoderUnified(nn.Module):
    def __init__(self, 
                 config: STFTEncoderUnifiedConfig,
                 ):
        super().__init__()
        self.config = config
        vae_dim = config.vae_dim
        audio_channels = config.audio_channels
        hop_length = config.hop_length
        n_fft = config.n_fft


        # Feature encoder
        self.spec_encoder = AMP_PHA_Spectrum(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, audio_channels=audio_channels, center=False, 
            atan2_magnitude_threshold_ratio=config.atan2_magnitude_threshold_ratio
        )

        ## Pre convolution
        pre_embed_out_channels = 1536
        if config.pre_embed_type == "post_linear":
            self.pre_embed = PreEmbed_PostLinear(audio_channels=audio_channels, out_channels=pre_embed_out_channels, n_fft=n_fft, n_blocks=1)
        if config.pre_embed_type == "post_linear_simple":
            self.pre_embed = PreEmbed_PostLinearSimple(audio_channels=audio_channels, out_channels=pre_embed_out_channels, n_fft=n_fft, n_blocks=1)
        elif config.pre_embed_type == "post_conv": # worse results. discard
            self.pre_embed = PreEmbed_PostConv(audio_channels=audio_channels, out_channels=pre_embed_out_channels, n_fft=n_fft, n_blocks=1)
        elif config.pre_embed_type == "post_conv_simple": # worse results. discard
            self.pre_embed = PreEmbed_PostConvSimple(audio_channels=audio_channels, out_channels=pre_embed_out_channels, n_fft=n_fft, n_blocks=1)
        elif config.pre_embed_type == "v1": # worse results. discard
            self.pre_embed = PreEmbed_V1(audio_channels=audio_channels, out_channels=pre_embed_out_channels, n_fft=n_fft, n_blocks=1)

        ratios, n_blocks, dims = config.ratio_block_dim
        in_channels = pre_embed_out_channels

        expected_framerate = config.sample_rate / hop_length / math.prod(ratios)
        assert config.bottleneck_framerate * expected_framerate, f"Mismatched frame rate {config.bottleneck_framerate}. Expected: {expected_framerate}"

        down_blocks = []
        for idx, (ratio, n_block, dim) in enumerate(zip(ratios, n_blocks, dims)):
            scaled_channels = dim
            out_channels = dim * ratio if config.scale_spatial_channels else dim
            channel_block = [ConvNextChannelReduction(in_channels, scaled_channels)] if in_channels != scaled_channels else []
            scaling_block = DownUpBlock(scaled_channels, out_channels=out_channels, ratio=ratio, shortcut=config.shortcut)
            if idx < len(config.ratio_block_dim) - 1: # conv
                base_block = ConvNextBackbone(dim=out_channels, num_layers=n_block, mlp_ratio=config.mlp_ratio, use_grn=config.use_grn)
            else: # conformer
                conformer_config = ConformerConfig(in_channels=out_channels, hidden_size=out_channels, num_hidden_layers=n_block, mlp_ratio=config.mlp_ratio)
                base_block = ConformerNextBlock(conformer_config, channels_first=True)
            
            block_layer = nn.Sequential(
                *channel_block,
                scaling_block,
                base_block
            )
            down_blocks.append(block_layer)
            in_channels = out_channels

        self.down_blocks = nn.ModuleList(down_blocks)

        # if self.config.bottleneck_type.startswith("h") or self.config.semantic_alignment_position == "h": 
            # assert self.config.semantic_alignment_position == self.config.semantic_alignment_position, "Both bottlenck type and alignment need to be hierarchical"
        if self.config.bottleneck_type == "v1":
            self.bottleneck = VAEBottleneck(in_channels=out_channels, latent_dim=vae_dim, beta=config.beta) # x2 for audio + semantic features
        if self.config.bottleneck_type == "v2":
            self.bottleneck = VAEBottleneckV2(in_channels=out_channels, latent_dim=vae_dim, beta=config.beta) # x2 for audio + semantic features
        if self.config.bottleneck_type == "v3":
            self.bottleneck = VAEBottleneckV3(in_channels=out_channels, latent_dim=vae_dim, beta=config.beta) # x2 for audio + semantic features
        elif self.config.bottleneck_type == "h" or self.config.bottleneck_type.startswith("h_"):
            if self.config.bottleneck_type.startswith("h_"):
                h_bottleneck_type = self.config.bottleneck_type.replace("h_", "")
            else:
                h_bottleneck_type = "v1"
            self.bottleneck = ResidualVAEVectorizer(
                in_channels=out_channels, latent_dim=vae_dim, sub_dims=config.sub_dims, beta=config.beta, 
                vector_dropout=config.vector_dropout, bottleneck_type=h_bottleneck_type,
                skip_semantic_residual=config.skip_semantic_residual
            )
        elif self.config.bottleneck_type == "vq":
            raise NotImplementedError("VQ bottleneck not supported yet")


        # TODO: handle hierarchical, pre align, post align
        if self.config.semantic_alignment_position == "pre":
            semantic_input_dim = out_channels
        elif self.config.semantic_alignment_position == "post":
            semantic_input_dim = vae_dim
        elif self.config.semantic_alignment_position == "h":
            semantic_input_dim = self.config.sub_dims[0]


        if config.semantic_loss_type == "bestrq":
            self.semantic_encoder = BestRQMaskedLoss(
                in_channels=semantic_input_dim,
                block_layers=config.semantic_n_blocks,
                num_codebooks=8,
                codebook_dim=16,
                mask_prob=0.2,
                vae_hz=config.bottleneck_framerate
            )
        elif config.semantic_loss_type == "no_loss":
            def no_loss(*args, **kwargs): return {}
            self.semantic_encoder = no_loss
        else:
            self.semantic_encoder = UMMLoss(
                in_channels=semantic_input_dim,
                block_layers=config.semantic_n_blocks,
                umm_version=config.semantic_loss_type,
                vae_hz=config.bottleneck_framerate
            )

    def audio_to_spec(self, audio):
        return self.spec_encoder(audio)

    def is_hierarchical(self):
        return self.config.bottleneck_type.startswith("h")
    
    def normalize_features(self, features: torch.Tensor, mean, std):
        if mean == None: return features
        if isinstance(mean, str):
            mean = ast.literal_eval(mean)
            std = ast.literal_eval(std)
        if mean == 0 and std == 1: return features
        if isinstance(self.bottleneck, ResidualVAEVectorizer):
            assert len(self.bottleneck.sub_dims) == len(mean), f"Mean and STD must be an array and match sub dims. Mean: {mean}, Sub dims: {self.bottleneck.sub_dims}"
            if torch.is_tensor(features): # passing in concat latent features. must split
                features = self.bottleneck.split_features(features, dim=-1)
                features = [(f - m) / s for f, m, s in zip(features, mean, std)]
                return torch.cat(features, dim=-1)
            else:
                return [(f - m) / s for f, m, s in zip(features, mean, std)]
        else:
            return (features - mean) / std

    def denormalize_features(self, features: torch.Tensor, mean, std):
        if mean == None: return features
        if isinstance(mean, str):
            mean = ast.literal_eval(mean)
            std = ast.literal_eval(std)
        if mean == 0 and std == 1: return features
        if isinstance(self.bottleneck, ResidualVAEVectorizer):
            assert len(self.bottleneck.sub_dims) == len(mean), f"Mean and STD must be an array and match sub dims. Mean: {mean}, Sub dims: {self.bottleneck.sub_dims}"
            features = self.bottleneck.split_features(features, dim=-1)
            features = [f * s + m for f, m, s in zip(features, mean, std)]
            return torch.cat(features, dim=-1)
        else:
            return features * std + mean


    def forward(self, logamp, pha, return_loss=True, audio_input_24k_mono=None, **kwargs):
        assert len(logamp.shape) == 4 and logamp.shape[1] == self.config.audio_channels, f"Invalid number of channels {logamp.shape}"

        features = self.pre_embed(logamp, pha) # -> B C*N L

        for idx, block_layer in enumerate(self.down_blocks):
            features = block_layer(features)
            # print(f"Features {idx}", features.shape)


        bottleneck_results = self.bottleneck(features) # bs x ch x seq
        latents = bottleneck_results["latents"]
        kl_loss = bottleneck_results["kl_loss"]

        if return_loss == False:
            return {
                **bottleneck_results,
                "latent": latents,
                "features": features,
                "kl_loss": kl_loss,
            }
        
        if self.config.semantic_alignment_position == "pre":
            semantic_input_features = features
        elif self.config.semantic_alignment_position == "post":
            semantic_input_features = latents
        elif self.config.semantic_alignment_position == "h":
            semantic_input_features, *remaining_audio_features = bottleneck_results["hierarchical_latents_list"]

        # print("Semantic input features", semantic_input_features.shape)

        umm_results = self.semantic_encoder(semantic_input_features, return_loss=return_loss, audio_input_24k_mono=audio_input_24k_mono)
        return {
            **bottleneck_results,
            **umm_results, # umm_cosine_loss, umm_l1_loss, umm_features, semantic_features # bs x ch x seq
            "latent": latents,
            "features": features,
            "kl_loss": kl_loss,
        }

    def features_to_decoder_latents(self, latents, skip_idxs=None):
        is_flattened_vae = not isinstance(self.bottleneck, ResidualVAEVectorizer)
        if is_flattened_vae:
            # just return. vae bottleneck and diffusion latents are same.
            return latents
        features = self.bottleneck.split_features(latents)
        latents = 0
        for i, chunk in enumerate(features):
            if skip_idxs and i in skip_idxs:
                continue
            vectorizer = self.bottleneck.vectorizers[i]
            emb = vectorizer.features_to_decoder_latents(chunk)
            latents += emb
        return latents


@dataclass
class ISTFTDecoderUnifiedConfig:
    n_fft: int = 1024
    hop_length: int = 49
    vae_dim: int = 128
    hidden_size: int = 1536
    final_hidden_size: int = 3072
    audio_channels: int = 2
    ratio_block_dim: list[list[int]] = ((2, 3, 3), (6, 4, 2), (1536, 1536, 3072))
    mlp_ratio: int = 3

    upsample_type: str = "transpose"
    shortcut: int = 1
    atan2_magnitude_threshold_ratio: float = 0.0

class ISTFTDecoderUnified(nn.Module):
    def __init__(self, config: ISTFTDecoderUnifiedConfig):
        super().__init__()

        mlp_ratio = config.mlp_ratio
        upsample_type = config.upsample_type
        ratios, n_blocks, dims = config.ratio_block_dim
        in_channels = config.vae_dim
        up_blocks = []
        for idx, (ratio, n_block, dim) in enumerate(zip(ratios, n_blocks, dims)):
            up_block = nn.Sequential(
                DownUpBlock(in_channels, out_channels=dim, ratio=1/ratio, upsample_type=upsample_type, shortcut=config.shortcut),
                ConvNextBackbone(dim=dim, num_layers=n_block, mlp_ratio=mlp_ratio)
            )
            up_blocks.append(up_block)
            in_channels = dim

        self.up_blocks = nn.Sequential(*up_blocks)
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.istft_head = ISTFTHeadStereo(
            dim=in_channels, n_fft=config.n_fft, hop_length=config.hop_length, audio_channels=config.audio_channels, padding="same", 
            atan2_magnitude_threshold_ratio=config.atan2_magnitude_threshold_ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for idx, block_layer in enumerate(self.up_blocks):
        #     x = block_layer(x)
        #     print(f"Decoder features {idx}", x.shape)
        x = self.up_blocks(x)
        x = self.norm(x)
        x = self.istft_head(x)
        return x
    
