import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field

from .diffusion import Diffusion, VDiffusion
from .ECAPA_TDNN.model import ECAPA_TDNN_GN
from .util_layers import RMSNorm
from recipes.voicebox.modules.layers import FrontendEmbedding  
from recipes.voicebox.modules.based_ctiga_llama import ModelArgs as LLamaArgs, LLaMa
import logging
from recipes.voicebox.modules.loss import sequence_mask

logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    # frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    wordseg_embed_dim: int = 64
    n_wordseg: int = 8
    n_phone: int = 1000
    n_tone: int = 30

    # token
    n_token: int = 32768
    token_embed_dim: int = 512

    # encoder
    encoder_dim: int = 768
    encoder_n_layers: int = 8
    encoder_n_heads: int = 8

    use_token_vector: bool = False
    token_vector_dim: int = 32
    token_hidden_dim: int = 768 
    token_upscales: list = field(default_factory=lambda : [2, 2, 2]) 
    token_downscales: list = field(default_factory=lambda : [5]) 

    # diffusion
    schedule_type :  str = "linear"
    diffusion_type :  str = "noise" 
    loss_type: str = "l1"
    min_beta : float = 0.0001
    max_beta : float = 0.02
    timesteps :  int = 1000

    net_name: str =  "CondUNet"

    # dpd
    dpd_feature_dim: int = 512 # == spk_embed_dim
    dpd_cond_dim: int = 768 # == embedding_features
    dpd_cond_bn_dim: int = 768 # 
    dpd_num_blocks: int = 12
    dpd_segment_size: int = 16
    dpd_segment_stride: int = 8
    dpd_dropout: float = 0
    dpd_intra_seq2seq: str = "sru"
    dpd_inter_seq2seq: str = "roformer"

    # unet
    dim :  int = 1
    in_channels :  int = 100
    channels :  list = field(default_factory=lambda : [512, 1024, 1024])
    factors :  list = field(default_factory=lambda : [1, 2, 2])
    items :  list = field(default_factory=lambda : [1, 1, 1])
    attentions :  list = field(default_factory=lambda : [1 ,1 ,1])
    attention_features :  int = 1024
    attention_heads : int = 8
    embedding_features :  int = 768
    resnet_groups :  int = 8
    modulation_features :  int = 768
    embedding_max_length :  int = 10000
    out_channels :  int = 100
    aux_head_dim: int = None
    use_positional_embedding: bool=True

    # speaker encoder
    prompt_mel_dim: int = 100
    spk_e_dim: int = 1024
    spk_embed_dim : int = 512
    prompt_loss_weight: float = 0.2 # deprecated
    
    llama_provider: str = "ctiga"
    diffusion_use_ctiga: bool = False
    
    # features
    target: str = "mel" # ["mel", "bn"]
    prompt_feature: str = "mel" # ["mel", "bn"]
    ctx_feature: str = "mel" # ["mel", "bn"]

class LDM1(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.frontend_embed = FrontendEmbedding(
            hp.phone_embed_dim,
            hp.tone_embed_dim,
            hp.wordseg_embed_dim,
            n_phone=hp.n_phone,
            n_tone=hp.n_tone,
            n_wordseg=hp.n_wordseg,
            out_dim=hp.encoder_dim
        )
        llama_config = LLamaArgs(dim=hp.encoder_dim,
                n_layers=hp.encoder_n_layers, 
                n_heads=hp.encoder_n_heads,
                out_dim=hp.encoder_dim,
                causal=False)
        self.encoder = LLaMa(llama_config, "ctiga")

        self.fc1 = nn.Linear(hp.out_channels+hp.encoder_dim+hp.spk_embed_dim, hp.embedding_features,  bias=False)
        
        self.diffusion = VDiffusion(hp)

    def forward(self, inputs):
        token_embed = self.frontend_embed(inputs["frontend"])
        token_embed = self.encoder(token_embed, inputs["mel_lens"], attention_mask=inputs["mel_mask"])

        #print("mel_ctx", inputs["mel_ctx"].shape)
        #print("spk_emb", inputs["spk_emb"].shape)
        #print("token_embed", token_embed.shape)

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           inputs["spk_emb"].unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        crop_len = token_embed.shape[1] % 4
        if crop_len > 0:
            diffusion_cond = diffusion_cond[:, :-crop_len, :]
            inputs["mel"] = inputs["mel"][:, :-crop_len, :]
            inputs["mel_ctx_mask"] = inputs["mel_ctx_mask"][:, :-crop_len]

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                inputs["spk_emb"],
                inputs["mel"].transpose(1, 2),
                inputs["mel_ctx_mask"]
                )

    def inference(self, inputs, timesteps=25):
        token_embed = self.token_embedding(inputs["token"]).transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)

        clip_value = token_embed.shape[-1] % 4
        if clip_value > 0:
            token_embed = token_embed[:, :, :-clip_value]

        prompt_emb = self.prompt_encoder(inputs["prompt_mel"])
        return self.diffusion.inference(
                token_embed,
                prompt_emb,
                timesteps=timesteps,
                )

class LDM2(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.frontend_embed = FrontendEmbedding(
            hp.phone_embed_dim,
            hp.tone_embed_dim,
            hp.wordseg_embed_dim,
            n_phone=hp.n_phone,
            n_tone=hp.n_tone,
            n_wordseg=hp.n_wordseg,
            out_dim=hp.encoder_dim
        )
        llama_config = LLamaArgs(dim=hp.encoder_dim,
                n_layers=hp.encoder_n_layers, 
                n_heads=hp.encoder_n_heads,
                out_dim=hp.encoder_dim,
                causal=False)
        self.encoder = LLaMa(llama_config, "ctiga")

        self.prompt_encoder = nn.Sequential(
                ECAPA_TDNN_GN(hp.prompt_mel_dim, hp.spk_e_dim, hp.spk_embed_dim),
                nn.Softsign())

        self.fc1 = nn.Linear(hp.out_channels+hp.encoder_dim+hp.spk_embed_dim, hp.embedding_features,  bias=False)
        
        self.diffusion = VDiffusion(hp)

    def forward(self, inputs):
        token_embed = self.frontend_embed(inputs["frontend"])
        token_embed = self.encoder(token_embed, inputs["mel_lens"], attention_mask=inputs["mel_mask"])

        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        #print("mel_ctx", inputs["mel_ctx"].shape)
        #print("spk_emb", spk_emb.shape)
        #print("token_embed", token_embed.shape)

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        crop_len = token_embed.shape[1] % 4
        if crop_len > 0:
            diffusion_cond = diffusion_cond[:, :-crop_len, :]
            inputs["mel"] = inputs["mel"][:, :-crop_len, :]
            inputs["mel_ctx_mask"] = inputs["mel_ctx_mask"][:, :-crop_len]

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                inputs["mel"].transpose(1, 2),
                inputs["mel_ctx_mask"]
                )
    def inference(self, inputs, timesteps=25, eta=0.3):
        token_embed = self.frontend_embed(inputs["frontend"])
        token_embed = self.encoder(token_embed, inputs["mel_lens"], attention_mask=inputs["mel_mask"])

        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        clip_value = diffusion_cond.shape[1] % 4
        if clip_value > 0:
            diffusion_cond = diffusion_cond[:, :-clip_value, :]

        return self.diffusion.inference(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                timesteps=timesteps,
                eta=eta,
                #inpaint_x=inputs["prompt_mel"],
                )

# encoder 用于concat后的特征
class LDM2b(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.frontend_embed = FrontendEmbedding(
            hp.phone_embed_dim,
            hp.tone_embed_dim,
            hp.wordseg_embed_dim,
            n_phone=hp.n_phone,
            n_tone=hp.n_tone,
            n_wordseg=hp.n_wordseg,
            out_dim=hp.encoder_dim
        )

        self.prompt_encoder = nn.Sequential(
                ECAPA_TDNN_GN(hp.prompt_mel_dim, hp.spk_e_dim, hp.spk_embed_dim),
                nn.Softsign())

        self.fc1 = nn.Linear(hp.out_channels+hp.encoder_dim+hp.spk_embed_dim, hp.encoder_dim,  bias=False)

        llama_config = LLamaArgs(dim=hp.encoder_dim,
                n_layers=hp.encoder_n_layers, 
                n_heads=hp.encoder_n_heads,
                causal=False)
        self.encoder = LLaMa(llama_config, "ctiga")
        self.encoder_fc = nn.Linear(hp.encoder_dim, hp.embedding_features, bias=False)
        
        self.diffusion = VDiffusion(hp)

    def forward(self, inputs):
        token_embed = self.frontend_embed(inputs["frontend"])
        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        #print("mel_ctx", inputs["mel_ctx"].shape)
        #print("spk_emb", spk_emb.shape)
        #print("token_embed", token_embed.shape)

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        diffusion_cond = self.encoder(diffusion_cond, inputs["mel_lens"], attention_mask=inputs["mel_mask"])
        diffusion_cond = self.encoder_fc(diffusion_cond)

        crop_len = token_embed.shape[1] % 4
        if crop_len > 0:
            diffusion_cond = diffusion_cond[:, :-crop_len, :]
            inputs["mel"] = inputs["mel"][:, :-crop_len, :]
            inputs["mel_ctx_mask"] = inputs["mel_ctx_mask"][:, :-crop_len]

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                inputs["mel"].transpose(1, 2),
                inputs["mel_ctx_mask"]
                )

    # TODO: fix
    def inference(self, inputs, timesteps=25):
        token_embed = self.frontend_embed(inputs["frontend"])
        token_embed = self.encoder(token_embed, inputs["mel_lens"], attention_mask=inputs["mel_mask"])

        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        clip_value = diffusion_cond.shape[1] % 4
        if clip_value > 0:
            diffusion_cond = diffusion_cond[:, :-clip_value, :]

        return self.diffusion.inference(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                timesteps=timesteps,
                inpaint_x=inputs["prompt_mel"]
                )

class LDM3(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.act_fn = nn.GELU()

        if hp.use_token_vector:
            self.token_embedding = nn.Linear(hp.token_vector_dim, hp.token_embed_dim, bias=False)
            self.umm_pad_vector = nn.Parameter(torch.FloatTensor(1, hp.token_vector_dim))
        else:
            self.token_embedding = nn.Embedding(hp.n_token, hp.token_embed_dim)
        self.token_prenet = self.create_token_prenet(hp)

        self.prompt_encoder = nn.Sequential(
                ECAPA_TDNN_GN(hp.prompt_mel_dim, hp.spk_e_dim, hp.spk_embed_dim),
                nn.Softsign())

        self.fc1 = nn.Linear(hp.out_channels+hp.token_hidden_dim+hp.spk_embed_dim, hp.encoder_dim, bias=False)

        llama_config = LLamaArgs(dim=hp.encoder_dim,
                n_layers=hp.encoder_n_layers, 
                n_heads=hp.encoder_n_heads,
                causal=False)
        self.encoder = LLaMa(llama_config, hp.llama_provider)
        self.encoder_fc = nn.Linear(hp.encoder_dim, hp.embedding_features, bias=False)

        self.diffusion = VDiffusion(hp)

    def create_token_prenet(self, hp):
        token_prenet = nn.ModuleList([
            nn.Conv1d(hp.token_embed_dim, hp.token_hidden_dim, kernel_size=1)
            ])

        for scale in hp.token_upscales:
            token_prenet.append(nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="nearest"),
                nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=3, padding=1),
                self.act_fn,
                RMSNorm(hp.token_hidden_dim, feat_dim=1)
                ))
        for scale in hp.token_downscales:
            if scale == 1:
                conv = nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=1, stride=1)
            else:
                conv = nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=scale*2, stride=scale, padding=scale//2+scale%2)

            token_prenet.append(nn.Sequential(
                conv,
                self.act_fn,
                RMSNorm(hp.token_hidden_dim, feat_dim=1)
                ))

        return token_prenet


    def forward(self, inputs):
        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)

        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        diffusion_cond = self.encoder(diffusion_cond, diffusion_cond.shape[1],
                attention_mask=inputs["mel_mask"])
        diffusion_cond = self.encoder_fc(diffusion_cond)

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                inputs["mel"].transpose(1, 2),
                inputs["mel_ctx_mask"]
                )

    def inference(self, inputs, timesteps=25):
        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)

        spk_emb = self.prompt_encoder(inputs["prompt_mel"])

        diffusion_cond = torch.cat([
           token_embed, 
           inputs["mel_ctx"], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)

        diffusion_cond = self.encoder(diffusion_cond, diffusion_cond.shape[1])
        diffusion_cond = self.encoder_fc(diffusion_cond)

        clip_value = diffusion_cond.shape[1] % 4
        if clip_value > 0:
            diffusion_cond = diffusion_cond[:, :-clip_value, :]

        return self.diffusion.inference(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                timesteps=timesteps,
                )

class PrefixLDM3(LDM3):
    def __init__(self, hp):
        super().__init__(hp)
        self.frontend_embed = FrontendEmbedding(
            hp.phone_embed_dim,
            hp.tone_embed_dim,
            hp.wordseg_embed_dim,
            n_phone=hp.n_phone,
            n_tone=hp.n_tone,
            n_wordseg=hp.n_wordseg,
            out_dim=hp.encoder_dim
        )
        logger.info(f"prompt_feature: {self.hp.prompt_feature}")
        logger.info(f"cxt_feature: {self.hp.ctx_feature}")
        logger.info(f"target_feature: {self.hp.target}")

    def forward(self, inputs):
        text_embed = self.frontend_embed(inputs["frontend"]) # [B, T, 1024]
        text_lens = inputs["text_lens"]
        mel_lens = inputs["mel_lens"]

        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        diffusion_cond = torch.cat([
           token_embed, 
           inputs[ctx_feature], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)
        max_text_len = text_embed.shape[1]
        max_mel_len = diffusion_cond.shape[1]
        padded_diffusion_cond = torch.zeros(
            [B, inputs["text_mel_mask"].shape[1], diffusion_cond.shape[-1]], 
            device=device)
        for i in range(B):
            padded_diffusion_cond[i, :text_lens[i], :] = text_embed[i,:text_lens[i], :]
            padded_diffusion_cond[i, text_lens[i]:text_lens[i]+mel_lens[i], :] = diffusion_cond[i,:mel_lens[i],:]
        
        seq_mask = inputs["text_mel_mask"]
        encoder_output = self.encoder(padded_diffusion_cond, padded_diffusion_cond.shape[1], attention_mask=seq_mask)
        
        diffusion_cond_wotext = torch.zeros(
            [B, max_mel_len, encoder_output.shape[-1]],
            device=device)
        for i in range(B):
            diffusion_cond_wotext[i, :mel_lens[i], :] = encoder_output[i, text_lens[i]:text_lens[i]+mel_lens[i], :]
        
        diffusion_cond = self.encoder_fc(diffusion_cond_wotext)

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                inputs[self.hp.target].transpose(1, 2),
                mask=inputs["mel_ctx_mask"]
                )

    def inference(self, inputs, timesteps=25):
        text_embed = self.frontend_embed(inputs["frontend"]) # [B, T, 1024]

        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)

        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        diffusion_cond = torch.cat([
           token_embed, 
           inputs[ctx_feature], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)
        padded_diffusion_cond = torch.cat([text_embed, diffusion_cond], dim=1)
        encoder_output = self.encoder(padded_diffusion_cond, padded_diffusion_cond.shape[1])

        diffusion_cond_wotext = encoder_output[:, text_embed.shape[1]:text_embed.shape[1]+diffusion_cond.shape[1], :]
        
        diffusion_cond = self.encoder_fc(diffusion_cond_wotext)
        return self.diffusion.inference(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                timesteps=timesteps,
                )
    
    
class PrefixLDM3a(LDM3):
    def __init__(self, hp):
        super().__init__(hp)
        self.frontend_embed = FrontendEmbedding(
            hp.phone_embed_dim,
            hp.tone_embed_dim,
            hp.wordseg_embed_dim,
            n_phone=hp.n_phone,
            n_tone=hp.n_tone,
            n_wordseg=hp.n_wordseg,
            out_dim=hp.encoder_dim
        )
        self.text_encoder_fc = nn.Linear(hp.encoder_dim, hp.embedding_features, bias=False)
        logger.info(f"prompt_feature: {self.hp.prompt_feature}")
        logger.info(f"cxt_feature: {self.hp.ctx_feature}")
        logger.info(f"target_feature: {self.hp.target}")

    def forward(self, inputs):
        text_embed = self.frontend_embed(inputs["frontend"]) # [B, T, 1024]
        text_lens = inputs["text_lens"]
        mel_lens = inputs["mel_lens"]

        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])
        
        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        diffusion_cond = torch.cat([
           token_embed, 
           inputs[ctx_feature], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)
        max_text_len = text_embed.shape[1]
        max_mel_len = diffusion_cond.shape[1]
        padded_diffusion_cond = torch.zeros(
            [B, inputs["text_mel_mask"].shape[1], diffusion_cond.shape[-1]], 
            device=device)
        for i in range(B):
            padded_diffusion_cond[i, :text_lens[i], :] = text_embed[i,:text_lens[i], :]
            padded_diffusion_cond[i, text_lens[i]:text_lens[i]+mel_lens[i], :] = diffusion_cond[i,:mel_lens[i],:]
        
        seq_mask = inputs["text_mel_mask"]
        encoder_output = self.encoder(padded_diffusion_cond, padded_diffusion_cond.shape[1], attention_mask=seq_mask)

        
        diffusion_cond_wotext = torch.zeros(
            [B, max_mel_len, encoder_output.shape[-1]],
            device=device)
        text_cond = torch.zeros(
            [B, max_text_len, encoder_output.shape[-1]],
            device=device)

        for i in range(B):
            diffusion_cond_wotext[i, :mel_lens[i], :] = encoder_output[i, text_lens[i]:text_lens[i]+mel_lens[i], :]
            text_cond[i, :text_lens[i], :] = encoder_output[i, :text_lens[i], :]
        
        diffusion_cond = self.encoder_fc(diffusion_cond_wotext)
        text_cond = self.text_encoder_fc(text_cond)

        return self.diffusion(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                inputs[self.hp.target].transpose(1, 2),
                local_cond2=text_cond,
                mask=inputs["mel_mask"],
                mask2=inputs["text_mask"]
                )

    def inference(self, inputs, timesteps=25, sampler="ddim"):
        text_embed = self.frontend_embed(inputs["frontend"]) # [B, T, 1024]

        B, device = inputs["token"].size(0), inputs["token"].device
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2)

        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        diffusion_cond = torch.cat([
           token_embed, 
           inputs[ctx_feature], 
           spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)
           ], dim=-1)

        diffusion_cond = self.fc1(diffusion_cond)
        padded_diffusion_cond = torch.cat([text_embed, diffusion_cond], dim=1)
        encoder_output = self.encoder(padded_diffusion_cond, padded_diffusion_cond.shape[1])

        diffusion_cond = self.encoder_fc(encoder_output[:, text_embed.shape[1]:text_embed.shape[1]+diffusion_cond.shape[1], :])
        text_cond = self.text_encoder_fc(encoder_output[:, :text_embed.shape[1]])
        
        return self.diffusion.inference(
                diffusion_cond.transpose(1, 2),
                spk_emb,
                local_cond2=text_cond,
                timesteps=timesteps,
                sampler=sampler
                )

