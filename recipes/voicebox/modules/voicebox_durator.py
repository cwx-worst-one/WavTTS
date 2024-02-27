
from dataclasses import dataclass
from torch import nn
import torch
from recipes.voicebox.modules.based_ctiga_llama import LLaMa, ModelArgs
from recipes.voicebox.modules.layers import FrontendEmbedding
from recipes.voicebox.model.x_transformers import always, ScaledSinusoidalEmbedding, TransposeLast, SamePad


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
    duration_dim: int = 1

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
    n_duration: int = 128
    duration_embed_dim: int = 128


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

        self.tok_embeddings = FrontendEmbedding(
            params.phone_embed_dim,
            params.tone_embed_dim,
            params.wordseg_embed_dim,
            n_phone=params.n_phone,
            n_tone=params.n_tone,
            n_wordseg=params.n_wordseg,
            out_dim=params.sem_dim
        )

        self.duration_embeddings = nn.Embedding(
                params.n_duration, # duration token
                params.duration_embed_dim, 
                padding_idx=0)
        
        dim = params.dim  # dim for transformer input,1024

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
        self.project_in = nn.Linear(params.sem_dim + params.duration_embed_dim, dim, bias=False)
        self.project_out = nn.Linear(dim, params.duration_dim, bias=False)
        
        self.init_weight_and_load_state(None)


    def forward(self, ctx, frontend_inputs, mel_len):
        # embed text
        z = self.tok_embeddings(frontend_inputs)  # B, T, sem
        # embed duration context
        duration_embedding = self.duration_embeddings(ctx)

        x = torch.cat((duration_embedding, z), dim=-1)  # B ,T, 2D+sem
   
        x = self.project_in(x)  
        x = x + self.pos_emb(x)
        x = self.post_emb_norm(x)
        x = self.emb_dropout(x) # b, t, hidden

        x = super().forward(x, mel_len)  # b, t, hidden
        out = self.project_out(x)  # b, t, 1     
        out = out.float()
        return out

if __name__ == "__main__":
    config = VoiceBoxArgs()
    frontend_inputs = {
        "phone": torch.randint(0, 1000, (32, 100)).to("cuda:0"),
        "tone": torch.randint(0, 30, (32, 100)).to("cuda:0"),
        "word_seg": torch.randint(0, 8, (32, 100)).to("cuda:0")
    }
    mel_ctx = torch.randint(0, 128, (32, 100)).to("cuda:0")
    mel_len = torch.randint(0, 100, [32]).to("cuda:0")
    model = VoiceBox(config).to("cuda:0").to(torch.bfloat16)
    out = model(mel_ctx, frontend_inputs, mel_len)
    print(out.shape)