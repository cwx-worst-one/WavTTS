
from dataclasses import dataclass
from torch import nn
import torch
import numpy as np
from recipes.voicebox.modules.based_ctiga_llama import LLaMa, ModelArgs
from recipes.voicebox.modules.layers import FrontendEmbedding
from recipes.voicebox.model.x_transformers import always, ScaledSinusoidalEmbedding, TransposeLast, SamePad
from recipes.voicebox.modules.cross_attention import FuseBlock


@dataclass
class VoiceBoxArgs(ModelArgs):
    causal: bool = False  # encoder.
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    wordseg_embed_dim: int = 64
    n_wordseg: int = 8
    n_phone: int = 1000
    n_tone: int = 30

    sem_dim: int = 512
    mel_dim: int = 80

    k: int = 31
    conv_pos_groups: int = 16
    num_pos_layers: int = 2
    post_emb_norm: bool = False
    emb_dropout: int = 0.
    use_abs_pos_emb: bool = True
    scaled_sinu_pos_emb: bool = False
    regression: bool = False
    condition_on_codec: bool = False
    codebook_num: int = 1024

    phone_enc: bool = False  # use it for phone encoder
    use_attn_mask: bool = False  # whether to use attention mask for the transformer encoder
    is_zy_embed: bool = False  # whether to use zy embedding
    fourier_scale: int = 16

    global_cond_dim: int = 0

    local_speaker_attention: bool = False
    fuse_hidden_channels: int = 256  # speaker conv btc c
    fuse_n_heads: int = 4
    fuse_p_dropout: int = 0.1


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=32, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class VoiceBox(LLaMa):
    def __init__(self, params: VoiceBoxArgs, provider="ctiga"):
        super().__init__(params, provider)
        self.init_weight_and_load_state(None)

        self.tok_embeddings = FrontendEmbedding(
            params.phone_embed_dim,
            params.tone_embed_dim,
            params.wordseg_embed_dim,
            n_phone=params.n_phone,
            n_tone=params.n_tone,
            n_wordseg=params.n_wordseg,
            out_dim=params.sem_dim
        )

        self.local_speaker_attention = params.local_speaker_attention
        if params.local_speaker_attention:
            self.fuse = FuseBlock(
                in_channels=params.sem_dim,
                hidden_channels=params.fuse_hidden_channels,
                out_channels=params.sem_dim,
                n_heads=params.fuse_n_heads,
                p_dropout=params.fuse_p_dropout
            )

        dim = params.dim  # dim for transformer input,1024
        self.time_emb = nn.Sequential(GaussianFourierProjection(embedding_size=dim//4, scale=params.fourier_scale),
                                      nn.Linear(dim//2, dim),
                                      nn.SiLU(),
                                      nn.Linear(dim, dim),  # 1024
                                      nn.SiLU())

        if not params.use_abs_pos_emb:
            self.pos_emb = always(0)
        elif params.scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            # used this one
            self.pos_emb = nn.Sequential(
                TransposeLast(),
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size=params.k,
                            padding=params.k // 2,
                            groups=params.conv_pos_groups,
                        ),
                        SamePad(params.k),
                        TransposeLast(),
                        nn.LayerNorm(dim, elementwise_affine=False),
                        TransposeLast(),
                        nn.GELU(),
                    )
                    for _ in range(params.num_pos_layers)
                ],
                TransposeLast(),
            )
        self.post_emb_norm = nn.LayerNorm(
            dim) if params.post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(params.emb_dropout)
        self.regression = params.regression

        if not self.regression:
            # wav
            self.project_in = nn.Linear(2*params.mel_dim+params.sem_dim+params.global_cond_dim, dim)
        else:
            self.project_in = nn.Linear(params.dim_in+params.phone_emb_dim, dim)

        self.project_out = nn.Linear(dim, params.mel_dim)

        # remove 1 frame
        self.adjust = nn.Conv1d(params.mel_dim, params.mel_dim, 6, padding=2, groups=params.mel_dim)
        

    def forward(self, x, t, ctx, frontend_inputs, mel_len, global_cond=None, bct_feat=None, text_len=None):
        """
        args:
            ctx: context, can be audio_ctx ([B, T, D]) for the audio model or duration_ctx ([B, T, 1]) for the duration model
            x: sample at flow step t, has the same shape as ctx, noisy ref
            zy: per-frame phone transcription for the audio model or per-frame duration sequence for the duration model, with shape [B, T] or [B, T, H]
            t: flow step, scalar
            codec: codec codes, with shape [B, T, C], C is the number of codec codes, it is 8 for encodec
            mask: mask used in attn_layers, with shape [B, T]. True represents the element is used for attention in the attn_layers, False means it is not used and usually it is used for padded elements. 
            is_zy_embed: whether zy is already embedded, if True, zy is already embedded and its shape should be [B, T, H]. If False, zy is not embedded and its shape should be [B, T]
        return:
            out: output of the transformer model, with shape [B, T, D] or [B, T, 1], has the sampe shape as ctx
        """
        # embed text
        z = self.tok_embeddings(frontend_inputs)  # B, T, sem
        # concate res for this step, ctx and text
        
        if self.local_speaker_attention:
            text_T = text_len.max()
            # 创建 [B, T] 形状的 mask tensor
            text_mask = torch.arange(text_T)[None, :].to(z.device) < text_len[:, None]
            bct_mask = torch.ones([bct_feat.shape[0], bct_feat.shape[2]]).to(z.device)
            lsa_output = self.fuse(z.transpose(1,2), text_mask.unsqueeze(1), bct_feat, bct_mask.unsqueeze(1))
            z = z + lsa_output.transpose(1,2)

        if global_cond is None:
            x = torch.cat((ctx, x, z), dim=-1)  # B ,T, 2D+sem
        else:
            global_cond = global_cond.unsqueeze(1).expand(-1, x.shape[1], -1) 
            x = torch.cat((ctx, x, z, global_cond), dim=-1)  # B ,T, 2D+sem+global_cond_dim
        x = self.project_in(x)  # b,t, hidden

        if not self.regression:
            t_emb = self.time_emb(t.squeeze(-1)).unsqueeze(1)  # B, 1, 1024
            x = torch.cat((t_emb, x), dim=1)  # B, T+1 ,1024

        x = x + self.pos_emb(x)  # B, T+1 ,hidden
        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)

        x = super().forward(x, mel_len+1)  # B, T+1 ,hidden
        out = self.project_out(x)  # B, T+1 ,80

        # remove one additional dim
        if not self.regression:
            out = out.transpose(1, 2)
            out = self.adjust(out)  # B,T,80
            out = out.transpose(1, 2)
        
        out = out.float()
        # if scale_by_sigma:
        #     out = out/t
        return out


if __name__ == "__main__":
    config = VoiceBoxArgs()
    frontend_inputs = {
        "phone": torch.randint(0, 1000, (32, 100)).to("cuda:0"),
        "tone": torch.randint(0, 30, (32, 100)).to("cuda:0"),
        "word_seg": torch.randint(0, 8, (32, 100)).to("cuda:0")
    }
    ref = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel_ctx = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel_len = torch.randint(0, 100, [32]).to("cuda:0")
    t = torch.rand((ref.shape[0], *([1] * (ref.dim() - 1))), device=ref.device, dtype=torch.bfloat16)
    model = VoiceBox(config).to("cuda:0").to(torch.bfloat16)
    ret_dict = model(ref, t, mel_ctx, frontend_inputs, mel_len)
    print(ret_dict['pred_mel'].shape)