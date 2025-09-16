import os
import torch
from torch import nn
import torch.nn.functional as F
# from recipes.umm.transforms.speech import SpeechTransform # use later one
from recipes.sacodec.datasets.transforms_speech import SpeechTransform
from samantha.utils.hparams import DotDict
from recipes.umm.models.umm_mkii import RandomProjectionQuantizer as mkiiRPQ
from recipes.sacodec.models.sacodec_unified.components import TransposeLast, ConformerConfig, ConformerNextBlock, DownUpBlock
from recipes.sacodec.models.components.masked_loss import MaskedEmb, MaskedCrossEntropy


from recipes.umm.requires.model_initializer import init_stage3, ensure_hdfs_ckpt_is_local, local_zero_first
from recipes.umm.modules.lit_module import Stage3

class UMMLoss(nn.Module):
    def __init__(self, 
                 in_channels=64, block_layers=0, 
                 sample_rate=44100, umm_version="umm", vae_hz=50
                 ):
        super().__init__()
        self.sample_rate = sample_rate

        self.umm_version = umm_version
        if umm_version == "umm2": # legacy
            umm_dim = 1280
        elif umm_version.endswith("_vq"):
            umm_dim = 32
        else: # for umm / umm_rope hidden size
            umm_dim = 1024

        self.required_modules = {}

        hidden_size = max(in_channels, umm_dim)
        config = ConformerConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size*2,
            num_hidden_layers=block_layers,
            rope_enhance_pos=7500
        )
        semantic_encoder = ConformerNextBlock(config)

        umm_hz = 25
        self.alignment_head = nn.Sequential(
            DownUpBlock(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1, ratio=round(vae_hz/umm_hz), shortcut=False),
            semantic_encoder,
            TransposeLast(), # B L D
            torch.nn.Linear(hidden_size, umm_dim),
            TransposeLast(), # B D L
        )
    def load_umm(self, device):
        umm_version = self.umm_version
        umm_version = umm_version.replace("_vq", "") # strip out vq to match model name
        
        if umm_version == "umm":
            local_path = '/mnt/bn/ashaw-us/repos/samantha/.module_cache/musiclm/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/step=0180000.ckpt'
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location="cpu")
        elif umm_version == "umm_cn":
            hpath = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_2255mixedZHEN_2375Speech/umm_stage3_2255mixedZHEN_2375Speech_optim_mem_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0180000.ckpt'
            cache_dir = ".module_cache/tokenizer"

            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)

            with local_zero_first():
                local_path = ensure_hdfs_ckpt_is_local(hpath, cache_dir)
            conformer_umm = Stage3.load_from_checkpoint(local_path, map_location='cpu')
        elif umm_version == "umm_cn_rope":
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

        conformer_umm = conformer_umm.eval().to(device)
        conformer_umm.freeze()
        self.required_modules['conformer_umm'] = conformer_umm # wrapping in list to exclude from weight saving

    @property
    def conformer_umm(self):
        if 'conformer_umm' in self.required_modules:
            return self.required_modules['conformer_umm']
        return None  

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True)
    def get_conformer_umm_hidden(self, conformer_umm, input_audio):
        outputs = conformer_umm.model.wav2token_alloutputs(input_audio)
        hidden = outputs['hidden_states'].permute((0, 2, 1))
        return hidden
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True)
    def get_conformer_vq_hidden(self, conformer_umm, input_audio):
        outputs = conformer_umm.model.wav2token_alloutputs(input_audio)
        vq_hidden = outputs['vq_hidden_states'].permute((0, 2, 1))
        return vq_hidden

    def forward(self, encoder_features, return_loss=True, audio_input_24k_mono=None):
        # assert len(input_audio.shape) == 3 and len(encoder_features.shape) == 3, f"Invalid number of channels {input_audio.shape}, {encoder_features.shape}"
        # encoder features = B, D, L
        # print("Encoder features", encoder_features.shape)
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
        if self.umm_version.endswith("_vq"):
            # TODO: switch this to categorical cross entropy and predict vq ids
            umm_features = self.get_conformer_vq_hidden(self.conformer_umm, audio_input_24k_mono) # B, D, L
        else:
            umm_features = self.get_conformer_umm_hidden(self.conformer_umm, audio_input_24k_mono) # [2, 1024, 250],  B, D, L

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            cosine_loss = -torch.nn.functional.cosine_similarity(umm_predicted_features, umm_features, dim=1).mean()
            l1_loss = torch.nn.functional.l1_loss(umm_predicted_features, umm_features)

        # TODO: Try adding melspec and masked loss

        return {
            'umm_cosine_loss': cosine_loss,
            'umm_l1_loss': l1_loss,
            'umm_features': umm_predicted_features, # B, D, L
        }


### BestRQ Loss

class BestRQMaskedLoss(nn.Module):
    def __init__(self, 
                 in_channels=1024,
                 block_layers=4,
                 num_codebooks=8, codebook_size=2**13, codebook_dim=16,
                 mask_prob=0.1, vae_hz=50, masked_dur=0.4
                 ):
        super().__init__()
        ## MELSPEC
        umm_config = DotDict({
            "sample_rate": 24000,
            "frame_rate": 25,
            "n_mels": 128,
            "n_fft": 2048,
            "win_length": 2048,
            "hop_length": 240,
        })
        self.audio_transform = SpeechTransform(
            sample_rate=umm_config.sample_rate,
            n_mels=umm_config.n_mels,
            n_fft=umm_config.n_fft,
            win_length=umm_config.win_length,
            hop_length=umm_config.hop_length,
            f_min=0,
            f_max=umm_config.sample_rate // 2,
        )
        self.audio_transform.load_from_checkpoint('recipes/sacodec/configs/sacodec_unified/stats/sacodec_bestrq_stats_0509.pt')
        # self.audio_transform.load_from_checkpoint('/mnt/bn/lyrics-to-song/ashaw/models/bestrq_stats/sacodec_bestrq_stats_0509.pt')

        # self.quantizer = RandomProjectionQuantizer(
        #     # dim=in_channels,
        #     dim=umm_config.n_mels,
        #     codebook_size=codebook_size,
        #     codebook_dim=codebook_dim,
        #     num_codebooks=num_codebooks,
        #     norm=False # layernorm does not actually work here...
        # )
        rpq_config = DotDict({
            "rq_input_dim": umm_config.n_mels,
            "rq_codebook_size": codebook_size,
            "rq_codebook_dim": codebook_dim,
            "rq_codebook_num": num_codebooks,
        })
        self.quantizer = mkiiRPQ(rpq_config)



        hidden_size = max(in_channels, umm_config.n_mels) * 2
        config = ConformerConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size*2,
            num_hidden_layers=block_layers,
            rope_enhance_pos=7500
        )
        melspec_hz = 100
        ratio = vae_hz/melspec_hz
        if vae_hz == 10:
            self.up1 = nn.Sequential(
                DownUpBlock(in_channels=in_channels, out_channels=hidden_size, ratio=1/5, shortcut=False),
                ConformerNextBlock(config),
                DownUpBlock(in_channels=hidden_size, out_channels=hidden_size, ratio=1/2, shortcut=False),
                ConformerNextBlock(config)
            )
        else:
            self.up1 = nn.Sequential(
                DownUpBlock(in_channels=in_channels, out_channels=hidden_size, ratio=ratio, shortcut=False),
                ConformerNextBlock(config)
            )
        self.vae_hz = vae_hz

        masked_length = int(masked_dur * vae_hz) # 50hz, 0.4s
        self.mask_emb = MaskedEmb(n_embd=in_channels, length=masked_length) # masking intermediate layers after conv
        
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(num_codebooks)]
        )
        self.loss_func = MaskedCrossEntropy()
        self.mask_prob = mask_prob


    def forward(self, encoder_features, return_loss=True, audio_input_24k_mono=None):
        encoder_features = encoder_features.transpose(1, 2) # B x D x L -> B x L x D
        melspec_features = self.audio_transform(audio_input_24k_mono, normalize=True)
        # vq_indices = self.quantizer(melspec_features) # for lucidrains
        vq_indices = self.quantizer(melspec_features.reshape(-1, melspec_features.shape[-1]))
        
        masked_features, keep_mask = self.mask_emb(encoder_features, mask_prob=self.mask_prob)
        loss_mask = ~keep_mask # loss should be inverse of keep
        scale_factor = int(round(100 / self.vae_hz))
        loss_mask = F.interpolate(loss_mask.transpose(1, 2).float(), scale_factor=scale_factor, mode="nearest").bool() # scale up to 100hz to match semantic
        loss_mask = loss_mask.transpose(1, 2)
        semantic_features = self.up1(masked_features.transpose(1, 2)) # B x L x D -> B x D x L
        semantic_features = semantic_features.transpose(1, 2) # B x D x L -> B x L x D
        logits = [h(semantic_features) for h in self.prediction_heads]

        
        codebook_losses = []
        codebook_accs = []
        quant_rates = []
        for idx, codebook_logits in enumerate(logits):
            codebook_index = vq_indices[..., idx]
            # codebook_logits = codebook_logits # for lucidrains
            codebook_logits = codebook_logits.reshape(-1, codebook_logits.shape[-1])
            codebook_loss = self.loss_func(codebook_logits, codebook_index, mask=loss_mask.squeeze(-1))
            codebook_losses.append(codebook_loss)
            # codebook_acc = (codebook_logits.argmax(dim=-1) == codebook_index)[loss_mask.squeeze(-1).bool()].float().mean() * 100 # lucidrains
            codebook_acc = (codebook_logits.argmax(dim=-1) == codebook_index)[loss_mask.view(-1).bool()].float().mean() * 100
            codebook_accs.append(codebook_acc)
            vq_unique = torch.unique(codebook_index, sorted=False)
            quant_rate = vq_unique.numel() / codebook_index.numel()
            quant_rates.append(quant_rate)
        codebook_quant_rate = sum(quant_rates) / len(quant_rates)
        loss = sum(codebook_losses) / len(codebook_losses)
        acc = sum(codebook_accs) / len(codebook_accs)
        return {
            "codebook_loss": loss,
            "codebook_acc": acc,
            "codebook_quant_rate": codebook_quant_rate,
        }

# #### TODO: enable mulan training
# class MulanLoss(nn.Module):
#     def __init__(self, 
#                  vae_dim=64, hidden_size=1024, block_layers=0, 
#                  sample_rate=44100
#                  ):
#         super().__init__()
#         self.sample_rate = sample_rate

#         self.mulan_ckpt: str = "/mnt/bn/audio-diffusion/ashaw/models/sstk_v9/mulan/mulan-step=005000-median_rank_1=61-kaggle-minimal.ckpt"
#         self.mulan_version: str = "sstkmae_v3"
#         mulan_dim = 1024
        
#         self.required_modules = {}

#         self.alignment_head = nn.Sequential(
#             torch.nn.Conv1d(vae_dim, hidden_size*2, kernel_size=3, stride=2, padding=1), # B D L
#             Transpose(), # B L D
#             nn.GELU(),
#             torch.nn.Linear(hidden_size*2, mulan_dim),
#             Transpose(), # B D L
#         )

#     def load_mulan(self, device):
#         mulan_requires = init_mulan(
#             hpath=self.mulan_ckpt,
#             local_rank=device.idx,
#             cache_dir='.module_cache/musiclm',
#             version=self.mulan_version
#         )
#         self.required_modules.update(mulan_requires)

#     def prepare_mulan_text(self, text):
#         embeds = get_mulan_embeds(
#             self.required_modules,
#             text,
#             data_type='text',                
#         )
#         return embeds
    
#     def prepare_mulan_audio(self, audio_input_24k_mono):
#         embeds = get_mulan_embeds(
#             self.required_modules,
#             audio_input_24k_mono,
#             data_type='music',
#             average=False
#         )
#         return embeds
    
#     def forward(self, encoder_features, return_loss=True, audio_input_24k_mono=None):
#         # assert len(input_audio.shape) == 3 and len(encoder_features.shape) == 3, f"Invalid number of channels {input_audio.shape}, {encoder_features.shape}"
#         # encoder features = B, D, L
#         mulan_predicted_features = self.alignment_head(encoder_features)

#         if not return_loss:
#             return {
#                 'mulan_features': mulan_predicted_features, # bs x ch x seq
#             }
#         else:
#             assert audio_input_24k_mono is not None, "Must pass audio_input_24k_mono as input for UMM training"

#         # self.conformer_umm.to(input_audio.device)
#         # lazy load conformer umm
#         if "mulan" not in self.required_modules:
#             self.load_mulan(audio_input_24k_mono.device)
#         mulan_features = self.prepare_mulan_audio(audio_input_24k_mono) # [2, 1024, 250],  B, D, L

#         cosine_loss = -torch.nn.functional.cosine_similarity(mulan_predicted_features, mulan_features, dim=1).mean()
#         l1_loss = torch.nn.functional.l1_loss(mulan_predicted_features, mulan_features)

#         # TODO: Try adding melspec and masked loss

#         return {
#             'mulan_cosine_loss': cosine_loss,
#             'mulan_l1_loss': l1_loss,
#             'mulan_features': mulan_predicted_features, # B, D, L
#         }