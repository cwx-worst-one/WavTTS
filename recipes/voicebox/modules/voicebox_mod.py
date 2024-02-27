
from recipes.text2semantic.modules.llama.based_ctiga_llama import LLaMa, ModelArgs
from dataclasses import dataclass
from recipes.text2semantic.modules.llama.layers import FrontendEmbedding
from torch import nn

@dataclass
class VoiceBoxArgs(ModelArgs):
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    n_phone: int = 1000
    n_tone: int = 30

    k: int = 31,
    conv_pos_groups: int = 16,
    num_pos_layers: int = 2,
    phone_num: int = 364,
    phone_emb_dim: int = 16,
    dim_in: int = None,
    dim_out: int = None,
    emb_dim: int = None,
    post_emb_norm: bool = False,
    emb_dropout: int = 0.,
    use_abs_pos_emb: bool = True,
    scaled_sinu_pos_emb: bool = False,
    regression: bool = False,
    condition_on_codec: bool = False,
    codebook_num: int = 1024,
    codec_emb_dim: int = 256,
    code_sel: int = 0, # which code to use for conditioning
    phone_enc: bool =False, # use it for phone encoder
    use_attn_mask: bool = False, # whether to use attention mask for the transformer encoder
    is_zy_embed: bool = False, # whether to use zy embedding
    fourier_scale: int = 16,

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class VoiceBox(LLaMa):
    def __init__(self, params: VoiceBoxArgs, provider="ctiga"):
        super().__init__(params, provider)
        self.init_weight_and_load_state(None)
        
        self.tok_embeddings = FrontendEmbedding(
                params.phone_embed_dim,
                params.tone_embed_dim,
                n_phone = params.n_phone,
                n_tone = params.n_tone,
                out_dim=params.dim
        )
        
        dim = params.dim

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

        self.time_emb = nn.Sequential(GaussianFourierProjection(embedding_size=dim//4, scale=params.fourier_scale), 
                                              nn.Linear(dim//2, dim),
                                              nn.SiLU(),
                                              nn.Linear(dim, dim),
                                              nn.SiLU())
        self.post_emb_norm = nn.LayerNorm(dim) if params.post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(params.emb_dropout)
        self.regression = params.regression
        self.condition_on_codec = params.condition_on_codec
        self.code_sel = params.code_sel
        self.phone_enc = params.phone_enc
        self.use_attn_mask = params.use_attn_mask

        if not self.regression:
            if self.condition_on_codec:# not using
                self.project_in = nn.Linear(2*params.dim_in+params.phone_emb_dim + params.codec_emb_dim, dim) if exists(params.dim_in) else nn.Identity()
            else:
                self.project_in = nn.Linear(2*params.dim_in+params.phone_emb_dim, dim) if exists(params.dim_in) else nn.Identity()
        else:
            if self.phone_enc:
                self.project_in = nn.Linear(params.phone_emb_dim, dim)
            else:
                self.project_in = nn.Linear(params.dim_in+params.phone_emb_dim, dim) if exists(params.dim_in) else nn.Identity()
        
        # self.attn_layers = attn_layers

        self.project_out = nn.Linear(dim, params.dim_out) if exists(params.dim_out) else nn.Identity()

        self.adjust = nn.Conv1d(params.dim_out, params.dim_out, 6, padding=2, groups=params.dim_out)

        self.is_zy_embed = params.is_zy_embed
        if params.is_zy_embed == False:
            self.phone_emb = nn.Embedding(params.phone_num, params.phone_emb_dim)
        if self.condition_on_codec:
            self.codec_emb = nn.Embedding(params.codebook_num, params.codec_emb_dim)

        # self.post_net = nn.Linear(1024, 80, bias=False)


    def forward(
        self,
        zy: torch.Tensor,
        ctx = None,
        x = None,
        t = None,
        codec = None,
        return_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        scale_by_sigma = False,
        **kwargs
    ):
        """
        args:
            ctx: context, can be audio_ctx ([B, T, D]) for the audio model or duration_ctx ([B, T, 1]) for the duration model
            x: sample at flow step t, has the same shape as ctx
            zy: per-frame phone transcription for the audio model or per-frame duration sequence for the duration model, with shape [B, T] or [B, T, H]
            t: flow step, scalar
            codec: codec codes, with shape [B, T, C], C is the number of codec codes, it is 8 for encodec
            mask: mask used in attn_layers, with shape [B, T]. True represents the element is used for attention in the attn_layers, False means it is not used and usually it is used for padded elements. 
            is_zy_embed: whether zy is already embedded, if True, zy is already embedded and its shape should be [B, T, H]. If False, zy is not embedded and its shape should be [B, T]
        return:
            out: output of the transformer model, with shape [B, T, D] or [B, T, 1], has the sampe shape as ctx
        """
        assert self.phone_enc or ctx is not None
        if not self.use_attn_mask:
            mask = None
        if self.condition_on_codec:
            c_emb = self.codec_emb(codec[:, :, self.code_sel])
        if self.is_zy_embed:
            z = zy
        else:
            z = self.phone_emb(zy)
        if self.regression:
            if self.phone_enc:
                x = z
            else:
                x = torch.cat((ctx, z), dim = -1)
        else:
            x = torch.cat((ctx, x, z), dim = -1)
        if self.condition_on_codec:
            x = torch.cat((x, c_emb), dim = -1)
        x = self.project_in(x)
        if not self.regression:
            t_emb = self.time_emb(t.squeeze(-1)).unsqueeze(1)
            x = torch.cat((t_emb, x), dim = 1)
            if self.use_attn_mask:
                mask = F.pad(mask, (1, 0), 'constant', True)
        x = x + self.pos_emb(x)

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings
        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        x = self.emb_dropout(x)

        # x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)
        x = super().forward(x, seqlen)

        out = self.project_out(x) if not return_embeddings else x

        if not self.regression:
            out = out.transpose(1, 2)
            out = self.adjust(out)
            out = out.transpose(1, 2)
        
        if scale_by_sigma:
            out = out/t

        # if return_intermediates:
        #     return out, intermediates

        ret_dict = {}
        ret_dict['pred_mel'] = out

        # if return_attn:
        #     attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
        #     ret_dict['attn_maps'] = attn_maps
                   
        return ret_dict
    # def forward(self, frontend_inputs, mel, seqlen):
    #     # transformer
        
    #     h =  self.tok_embeddings(frontend_inputs)
    #     h = super().forward(h, seqlen)
    #     h = self.post_net(h)
    #     ret_dict = {}
    #     ret_dict['pred_mel'] = h
    #     return ret_dict