import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from recipes.sacodec.models.components.spectral_ops import AMP_PHA_Spectrum
from recipes.sacodec.models.components.vae_bottleneck import VAEBottleneck
from recipes.sacodec.models.components.convnext import ConvNeXtBlock, ConvNextBackboneDownUp
from recipes.sacodec.models.components.istft_head import ISTFTHeadStereo, ISTFTHeadStereoVocos, ISTFTHeadStereoMusic2Latent


class STFTEncoderVAE(nn.Module):
    def __init__(self, 
                 n_fft=1024, hop_length=256, win_length=None,
                 vae_dim=128, beta=1e-5, hidden_size=1536, audio_channels=2, block_layers=[1,4,8], even_pad=True
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
            input_channels=pre_channels * N_CH,
            dim=hidden_size,
            intermediate_dim=hidden_size*2,
            num_layers=block_layers[1],
            ratio=3,
            apply_final_layer_norm=False
        )
        # Down 2
        self.conv2 = ConvNextBackboneDownUp(
            input_channels=hidden_size,
            dim=hidden_size,
            intermediate_dim=hidden_size*2,
            num_layers=block_layers[2],
            ratio=3,
            apply_final_layer_norm=False
        )

        self.bottleneck = VAEBottleneck(in_channels=hidden_size, latent_dim=vae_dim, beta=beta)

    def audio_to_spec(self, audio):
        return self.spec_encoder(audio)

    def forward(self, logamp, pha, return_dict=False, **kwargs):
        assert len(logamp.shape) == 4 and logamp.shape[1] == self.audio_channels, f"Invalid number of channels {logamp.shape}"
        
        features = torch.cat([logamp, pha], dim=1) # append 

        features = self.pre_conv(features)
        # print('Features', features.shape)
        B, C, N, L = features.shape
        features = features.reshape(B, C*N, L).transpose(1, 2) # -> B L C*N, for layer norm
        # print('Features', features.shape)
        features = self.pre_norm(features)
        features = features.transpose(1, 2) # B C L
        
        for pre_convnext in self.pre_convnext:
            features = pre_convnext(features)

        features = self.conv1(features)
        features = self.conv2(features)
        hidden_states_latents, kl = self.bottleneck(features) # bs x ch x seq

        if return_dict:
            return {
                "latent": hidden_states_latents,
                "features": features,
                "kl": kl,
            }
        return hidden_states_latents, kl
    
class ISTFTDecoder(nn.Module):
    def __init__(self, 
                 n_fft=1024, hop_length=256, win_length=None,
                 atan2_magnitude_threshold_ratio=0.0,
                 vae_dim=128, hidden_size=1536, audio_channels=2, block_layers=[8,4,1], even_pad=True, final_hidden_size=3072, istft_head="stereo"
        ):
        super().__init__()
        self.n_fft = n_fft
        self.audio_channels = audio_channels
        self.N_feat = 3 # logamp, p_I, p_R
        self.N_dim = n_fft // 2 + 1 # N frequencies
        # out_dim = self.N_dim * self.N_feat * self.audio_channels

        T_PAD = 2 if even_pad else 3

        self.upconv_1 = ConvNextBackboneDownUp(
            input_channels=vae_dim,
            dim=hidden_size,
            intermediate_dim=hidden_size*2,
            num_layers=block_layers[0],
            ratio=1/2,
            pre_embed_padding=T_PAD, # 3000 -> 3001
            apply_final_layer_norm=False
        )
        self.upconv_2 = ConvNextBackboneDownUp(
            input_channels=hidden_size,
            dim=hidden_size,
            intermediate_dim=hidden_size,
            num_layers=block_layers[1],
            ratio=1/3,
#             pre_embed_padding=(7 - 2) // 2, # 3000 -> 3001
            apply_final_layer_norm=True
        )
        self.upconv_3 = ConvNextBackboneDownUp(
            input_channels=hidden_size,
            dim=final_hidden_size,
            intermediate_dim=final_hidden_size,
            num_layers=block_layers[2],
            ratio=1/3,
#             pre_embed_padding=(7 - 2) // 2, # 3000 -> 3001
            apply_final_layer_norm=True
        )
        if istft_head == "vocos":
            self.istft_head = ISTFTHeadStereoVocos(dim=final_hidden_size, n_fft=n_fft, hop_length=hop_length, audio_channels=audio_channels)
        elif istft_head == "m2l":
            self.istft_head = ISTFTHeadStereoMusic2Latent(dim=final_hidden_size, n_fft=n_fft, hop_length=hop_length, audio_channels=audio_channels, normalize_spec=False, atan2_magnitude_threshold_ratio=atan2_magnitude_threshold_ratio)
        elif istft_head == "m2l_norm":
            self.istft_head = ISTFTHeadStereoMusic2Latent(dim=final_hidden_size, n_fft=n_fft, hop_length=hop_length, audio_channels=audio_channels, normalize_spec=True, atan2_magnitude_threshold_ratio=atan2_magnitude_threshold_ratio)
        else:
            self.istft_head = ISTFTHeadStereo(dim=final_hidden_size, n_fft=n_fft, hop_length=hop_length, audio_channels=audio_channels, atan2_magnitude_threshold_ratio=atan2_magnitude_threshold_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upconv_1(x)
        # print('X', x.shape)
        x = self.upconv_2(x)
        # print('X', x.shape)
        x = self.upconv_3(x)
        # print('X', x.shape)
        x = self.istft_head(x)
        return x