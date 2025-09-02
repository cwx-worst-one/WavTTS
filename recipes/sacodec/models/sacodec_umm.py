import os
import torch
import torchaudio
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from recipes.sacodec.models.components.spectral_ops import AMP_PHA_Spectrum
from recipes.sacodec.models.components.vae_bottleneck import (
    VAEBottleneck,
    VAEBottleneckV2,
    VAEBottleneckV3,
    VAEBottleneckV4,
    VAEBottleneckV5,
    VAEBottleneckConstSigma,
)
from recipes.sacodec.models.components.convnext import ConvNeXtBlock, ConvNextBackboneDownUp
from recipes.sacodec.models.components.conformer_next import ConformerConfig, ConformerNextBlock, ConformerNextBackboneDownUp
 
from recipes.umm.requires.model_initializer import init_stage3, ensure_hdfs_ckpt_is_local, local_zero_first
from recipes.umm.modules.lit_module import Stage3

class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class UMMLoss(nn.Module):
    def __init__(self, 
                 vae_dim=64, hidden_size=1536, block_layers=0, 
                 vae_dim_lowres=32,
                 sample_rate=44100, umm_version="umm", vae_hz=50
                 ):
        super().__init__()
        self.sample_rate = sample_rate

        self.umm_version = umm_version
        if umm_version == "umm" or umm_version == "umm_rope":
            umm_dim = 1024
        elif umm_version == "umm_vq":
            umm_dim = 32
        elif umm_version == "umm2": # legacy
            umm_dim = 1280
        else:
            umm_dim = 1024

        self.required_modules = {}

        if block_layers > 0:
            if vae_hz == 50:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size, kernel_size=3, stride=2, padding=1)] # B D L
            elif vae_hz == 25:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size, kernel_size=1, stride=1, padding=0)] # B D L
            elif vae_hz == 100:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size, kernel_size=5, stride=4, padding=1)] # B D L
            config = ConformerConfig(
                input_channels=hidden_size,
                hidden_size=hidden_size,
                intermediate_size=hidden_size*2,
                num_hidden_layers=block_layers,
                rope_enhance_pos=7500
            )
            semantic_encoder = ConformerNextBlock(config)
            self.alignment_head = nn.Sequential(
                *downsample_block, # B D L
                Transpose(), # B L D
                semantic_encoder,
                torch.nn.Linear(hidden_size, umm_dim),
                Transpose(), # B D L
            )
            if self.umm_version == "umm_continuous_and_vq":
                self.alignment_head_lores = nn.Sequential(
                    *downsample_block, # B D L
                    Transpose(), # B L D
                    semantic_encoder,
                    torch.nn.Linear(hidden_size, vae_dim_lowres),
                    Transpose(), # B D L
                )

        else:
            if vae_hz == 50:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size*2, kernel_size=3, stride=2, padding=1)] # B D L
            elif vae_hz == 25:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size*2, kernel_size=1, stride=1, padding=0)] # B D L
            elif vae_hz == 100:
                downsample_block = [torch.nn.Conv1d(vae_dim, hidden_size*2, kernel_size=5, stride=4, padding=1)] # B D L
            self.semantic_encoder = None
            self.alignment_head = nn.Sequential(
                *downsample_block,
                Transpose(), # B L D
                nn.GELU(),
                torch.nn.Linear(hidden_size*2, umm_dim),
                Transpose(), # B D L
            )
            if self.umm_version == "umm_continuous_and_vq":
                self.alignment_head_lores = nn.Sequential(
                    *downsample_block,
                    Transpose(), # B L D
                    nn.GELU(),
                    torch.nn.Linear(hidden_size*2, vae_dim_lowres),
                    Transpose(), # B D L
                )

    def load_umm(self, device):
        umm_version = self.umm_version
        if umm_version == "umm":
            local_path = '/mnt/bn/ashaw-us/repos/samantha/.module_cache/musiclm/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/step=0180000.ckpt'
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location="cpu")
        if umm_version == "umm_cn":
            hpath = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_2255mixedZHEN_2375Speech/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0180000.ckpt'
            cache_dir = ".module_cache/tokenizer"

            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)

            with local_zero_first():
                local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location='cpu')
        if umm_version == "umm_cn_rope":
            hpath = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_unified_tts_rope/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=220000.ckpt"
            cache_dir = ".module_cache/tokenizer"

            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)

            with local_zero_first():
                local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location='cpu')
        elif umm_version == "umm_rope":
            local_path = '/mnt/bn/ashaw-us/repos/samantha/.module_cache/musiclm/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased_EMAEntropy32768x32/step=220000.ckpt'
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location="cpu")
        elif umm_version == "umm_vq":
            local_path = '/mnt/bn/ashaw-us/repos/samantha/.module_cache/musiclm/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/step=0180000.ckpt'
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location="cpu")
        conformer_umm = conformer_umm.eval().to(device)
        conformer_umm.freeze()
        self.required_modules['conformer_umm'] = conformer_umm # wrapping in list to exclude from weight saving

    @property
    def conformer_umm(self):
        if 'conformer_umm' in self.required_modules:
            return self.required_modules['conformer_umm']
        return None  

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
    def get_conformer_umm_hidden(self, conformer_umm, input_audio):
        outputs = conformer_umm.model.wav2token_alloutputs(input_audio)
        hidden = outputs['hidden_states'].permute((0, 2, 1))
        return hidden
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
    def get_conformer_vq_hidden(self, conformer_umm, input_audio):
        outputs = conformer_umm.model.wav2token_alloutputs(input_audio)
        vq_hidden = outputs['vq_hidden_states'].permute((0, 2, 1))
        return vq_hidden

    def forward(self, encoder_features, return_loss=True, audio_input_24k_mono=None, low_res_latents=None):
        # assert len(input_audio.shape) == 3 and len(encoder_features.shape) == 3, f"Invalid number of channels {input_audio.shape}, {encoder_features.shape}"
        # encoder features = B, D, L
        umm_predicted_features = self.alignment_head(encoder_features)

        if not return_loss:
            return {
                'umm_features': umm_predicted_features, # bs x ch x seq
            }
        else:
            assert audio_input_24k_mono is not None, "Must pass audio_input_24k_mono as input for UMM training"

        # self.conformer_umm.to(input_audio.device)
        # lazy load conformer umm
        if self.conformer_umm is None:
            self.load_umm(audio_input_24k_mono.device)
        if self.umm_version in ["umm", "umm_cn", "umm_cn_rope", "umm_rope"]:
            umm_features = self.get_conformer_umm_hidden(self.conformer_umm, audio_input_24k_mono) # [2, 1024, 250],  B, D, L
        elif self.umm_version == "umm_vq":
            # TODO: switch this to categorical cross entropy and predict vq ids
            umm_features = self.get_conformer_vq_hidden(self.conformer_umm, audio_input_24k_mono) # B, D, L
        elif self.umm_version == "umm_continuous_and_vq":
            umm_features = self.get_conformer_umm_hidden(self.conformer_umm, audio_input_24k_mono) # [2, 1024, 250],  B, D, L
            umm_features_vq = self.get_conformer_vq_hidden(self.conformer_umm, audio_input_24k_mono) # B, D, L

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            cosine_loss = -torch.nn.functional.cosine_similarity(umm_predicted_features, umm_features, dim=1).mean()
            l1_loss = torch.nn.functional.l1_loss(umm_predicted_features, umm_features)

            if self.umm_version == "umm_continuous_and_vq":
                umm_predicted_features_low_res = self.alignment_head_lores(low_res_latents)
                cosine_loss += -torch.nn.functional.cosine_similarity(umm_predicted_features_low_res, umm_features_vq, dim=1).mean()
                l1_loss += torch.nn.functional.l1_loss(umm_predicted_features_low_res, umm_features_vq)
        # TODO: Try adding melspec and masked loss

        return {
            'umm_cosine_loss': cosine_loss,
            'umm_l1_loss': l1_loss,
            'umm_features': umm_predicted_features, # B, D, L
        }

class STFTEncoderVAEPostUMM(nn.Module):
    def __init__(self, 
                 n_fft=1024, hop_length=256, win_length=None,
                 sample_rate=44100,
                 atan2_magnitude_threshold_ratio=0.0,
                 vae_hz=50,
                 vae_dim_lores=32,
                 vae_dim=128, beta=1e-5, hidden_size=1536, audio_channels=2, block_layers=[1,4,4], umm_block_layers=1, even_pad=True,
                 kernel_size=7, umm_version="umm", final_block="conformer", bottleneck_version="v1", align_pre=False, normalize_spec=False
                 ):
        super().__init__()
        self.audio_channels = audio_channels
        self.bottleneck_version = bottleneck_version
        if win_length is None: win_length = n_fft

        # legacy block layers was hardcoded. hack is to fix this
        if isinstance(block_layers, int):
            block_layers = [1,4,4] if final_block == "conformer" else [1,4,8]
        
        self.vae_dim = vae_dim
        
        N_CH = audio_channels * 2
        pre_channels = n_fft // 4

        # Feature encoder
        self.spec_encoder = AMP_PHA_Spectrum(n_fft=n_fft, hop_length=hop_length, win_length=win_length, audio_channels=audio_channels, normalize_spec=normalize_spec, atan2_magnitude_threshold_ratio=atan2_magnitude_threshold_ratio)

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
        self.final_block = final_block
        if self.final_block == "conformer":
            self.conformer2 = ConformerNextBackboneDownUp(
                input_channels=hidden_size,
                dim=hidden_size,
                intermediate_dim=hidden_size*2,
                num_layers=block_layers[2],
                ratio=3,
                apply_final_layer_norm=True
            )
        else:
            self.conv2 = ConvNextBackboneDownUp(
                input_channels=hidden_size,
                dim=hidden_size,
                intermediate_dim=hidden_size*2,
                num_layers=block_layers[2],
                ratio=3,
                apply_final_layer_norm=True,
                kernel_size=kernel_size
            )

        self.align_pre = align_pre
        semantic_input_dim = hidden_size if self.align_pre else vae_dim
        self.semantic_encoder = UMMLoss(
            vae_dim=semantic_input_dim,
            hidden_size=hidden_size,
            block_layers=umm_block_layers,
            sample_rate=sample_rate,
            vae_hz=vae_hz,
            umm_version=umm_version
        )
        if bottleneck_version == "v2":
            self.bottleneck = VAEBottleneckV2(in_channels=hidden_size, latent_dim=vae_dim, beta=beta)
        elif bottleneck_version == "v3":
            self.bottleneck = VAEBottleneckV3(in_channels=hidden_size, latent_dim=vae_dim, beta=beta)
        elif bottleneck_version == "v4":
            self.bottleneck = VAEBottleneckV4(in_channels=hidden_size, latent_dim=vae_dim, beta=beta)
        elif bottleneck_version == "v5":
            self.bottleneck = VAEBottleneckV5(in_channels=hidden_size, latent_dim=vae_dim, beta=beta, latent_dim_lores=vae_dim_lores)
        elif bottleneck_version == "const_sigma":
            self.bottleneck = VAEBottleneckConstSigma(in_channels=hidden_size, latent_dim=vae_dim, beta=beta)
        else:
            self.bottleneck = VAEBottleneck(in_channels=hidden_size, latent_dim=vae_dim, beta=beta) # x2 for audio + semantic features

    def audio_to_spec(self, audio):
        return self.spec_encoder(audio)

    def forward(self, logamp, pha, return_loss=True, audio_input_24k_mono=None, **kwargs):
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

        if self.final_block == "conformer":
            features = self.conformer2(features)
        else: 
            features = self.conv2(features)


        hidden_states_latents, kl = self.bottleneck(features) # bs x ch x seq
        if self.bottleneck_version == "v5":
            residual_latents, low_res_latents = hidden_states_latents.chunk(2, dim=1)
            hidden_states_latents = residual_latents + low_res_latents
        else:
            low_res_latents = None

        if return_loss == False:
            return {
                "latent": hidden_states_latents,
                "features": features,
                "kl_loss": kl,
            }

        semantic_input_features = features if self.align_pre else hidden_states_latents

        umm_results = self.semantic_encoder(semantic_input_features, return_loss=return_loss, audio_input_24k_mono=audio_input_24k_mono, low_res_latents=low_res_latents)
        return {
            **umm_results, # umm_cosine_loss, umm_l1_loss, umm_features, semantic_features # bs x ch x seq
            "latent": hidden_states_latents,
            "features": features,
            "kl_loss": kl,
        }
    
