import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from recipes.diffusion.models.lora import LoRALinearLayer

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        query_dim, 
        context_dim=None, 
        heads=8, 
        dim_head=64, 
        dropout = 0.,
        lora=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.lora = lora
        if lora:
            self.to_q_lora = LoRALinearLayer(query_dim, inner_dim, rank=8)
            self.to_k_lora = LoRALinearLayer(context_dim, inner_dim, rank=8)
            self.to_v_lora = LoRALinearLayer(context_dim, inner_dim, rank=8)
            self.to_out_lora = LoRALinearLayer(inner_dim, query_dim, rank=8)

    def forward(self, x, context=None, rotary_emb=None):
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        if self.lora:
            q = q + 1.0 * self.to_q_lora(x)
            k = k + 1.0 * self.to_k_lora(context)
            v = v + 1.0 * self.to_v_lora(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if rotary_emb is not None: 
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

        out = rearrange(out, "b h n d -> b n (h d)")

        if self.lora:
            out_lora = 1.0 * self.to_out_lora(out)

        out = self.to_out(out)
        
        if self.lora:
            out = out + out_lora

        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,  
        dim, 
        heads, 
        dim_head,
        depth=1,
        attn_dropout=0., 
        ff_dropout=0.,   
        use_checkpoint=False,
        context=False,
        lora=False,
    ):
        super(TransformerBlock, self).__init__()

        layers = nn.ModuleList([])
        for _ in range(depth):
            
            layer = nn.ModuleDict(
                    {
                        'norm_0': RMSNorm(dim),
                        'attn': Attention(
                            query_dim=dim, 
                            heads=heads, 
                            dim_head=dim_head, 
                            dropout=attn_dropout,
                            lora=lora
                        ),
                        'norm_1': RMSNorm(dim),
                        'ff': FeedForward(dim=dim, dropout=ff_dropout)
                    }
                )     
            if context:
                layer['norm_context'] = RMSNorm(dim)
            layers.append(layer)

        self.layers = layers
        self.use_checkpoint = use_checkpoint
        self.context = context
    
    def _forward_checkpoint(self, x, context=None, rotary_emb=None):
        for idx, transformer in enumerate(self.layers):
            x = checkpoint(
                transformer['attn'], 
                checkpoint(transformer['norm_0'], x, use_reentrant=False), 
                checkpoint(transformer['norm_context'], context, use_reentrant=False) if self.context else None,
                rotary_emb, 
                use_reentrant=False
            ) + x

            x = checkpoint(
                transformer['ff'], 
                checkpoint(transformer['norm_1'], x, use_reentrant=False), 
                use_reentrant=False
            ) + x

        return x

    def _forward(self, x, context=None, rotary_emb=None):
        for idx, transformer in enumerate(self.layers):
            x = transformer['attn'](
                transformer['norm_0'](x), 
                transformer['norm_context'](context) if self.context else None, 
                rotary_emb
            ) + x
            x = transformer['ff'](transformer['norm_1'](x)) + x
        return x
    
    def forward(self, x, context=None, rotary_emb=None):
        if self.use_checkpoint:
            return self._forward_checkpoint(x, context, rotary_emb)
        else:
            return self._forward(x, context, rotary_emb)

# dual-path blocks
class TNTBlocks(nn.Module):
    def __init__(self, 
            input_dim,
            context_dim,    
            fine_dim,
            fine_heads,
            fine_head_dim,
            coarse_dim, 
            coarse_heads,
            coarse_head_dim,
            depth=1, 
            dropout=0,
            semantic_cfg_prob=0.1,
            vc_cfg_prob=None,
            vc=False,
            use_checkpoint=False,
            lora=False,
        ):
        super().__init__()
        self.semantic_cfg_prob = semantic_cfg_prob
        self.vc_cfg_prob = vc_cfg_prob
        self.vc = vc

        # null embedding for cfg
        self.semantic_null_embedding = nn.Parameter(torch.randn(input_dim))
        if vc:
            self.vc_null_embedding = nn.Parameter(torch.randn(input_dim))
            # sep embedding for different condition
            self.sep_embedding = nn.Parameter(torch.randn(input_dim))

        self.fine_rotary_embedding = RotaryEmbedding(dim=fine_head_dim)
        self.coarse_rotary_embedding = RotaryEmbedding(dim=coarse_head_dim)

        layers = nn.ModuleList()
        for i in range(depth):
            get_cts = nn.Sequential(
                RMSNorm(coarse_dim),
                nn.Linear(coarse_dim, fine_dim),
                Rearrange('b t (n d) -> (b t) n d', n=1)
            )

            fine_to_coarse = nn.Sequential(
                RMSNorm(fine_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(fine_dim, coarse_dim),
            )

            layers.append(
                nn.ModuleList([
                    fine_to_coarse,
                    TransformerBlock(
                        dim=coarse_dim, 
                        heads=coarse_heads,
                        dim_head=coarse_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=False,
                    ),
                    TransformerBlock(
                        dim=fine_dim, 
                        heads=fine_heads, 
                        dim_head=fine_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=False,
                        context=True,
                        lora=lora,
                    ),
                    get_cts, 
                    TransformerBlock(
                        dim=fine_dim, 
                        heads=fine_heads, 
                        dim_head=fine_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=use_checkpoint
                    ),
                ]
            ))
        self.layers = layers

    def forward(self, 
        x, 
        time_emb, 
        semantic_context_emb, 
        semantic_force_cfg=None,
        vc_context_emb=None,
        vc_force_cfg=None,
    ):
        # input shape: b, d, fine_len, course_len
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        b, d, lf, lc = x.shape

        coarse_context = []
        # get mask for cfg
        semantic_cfg_prob = self.semantic_cfg_prob if semantic_force_cfg is None else semantic_force_cfg
        semantic_prob_keep_mask = prob_mask_like((b, 1, 1), 1. - semantic_cfg_prob, device=x.device)
        # get null embeddings for semantic context
        semantic_null_coarse_emb = repeat(self.semantic_null_embedding, 'd -> b l d', b=b, l=semantic_context_emb.shape[1])
        semantic_context_emb = torch.where(
            semantic_prob_keep_mask,
            semantic_context_emb,
            semantic_null_coarse_emb
        )
        coarse_context.append(semantic_context_emb)
        # vc context
        if self.vc:
            vc_cfg_prob = self.vc_cfg_prob if vc_force_cfg is None else vc_force_cfg
            vc_prob_keep_mask = prob_mask_like((b, 1, 1), 1. - vc_cfg_prob, device=x.device)
            vc_null_coarse_emb = repeat(self.vc_null_embedding, 'd -> b l d', b=b, l=vc_context_emb.shape[1])
            vc_context_emb = torch.where(
                vc_prob_keep_mask,
                vc_context_emb,
                vc_null_coarse_emb
            )
            sep_embedding = repeat(self.sep_embedding, 'd -> b l d', b=b, l=1)
            coarse_context.append(sep_embedding)
            coarse_context.append(vc_context_emb)
        # concat at dim1
        coarse_context = torch.cat(coarse_context, dim=1)

        time_emb = torch.mean(rearrange(time_emb, 'b d lf lc-> (b lc) lf d'), axis=1, keepdim=True)
        
        fine_emb = rearrange(x, 'b d lf lc -> (b lc) lf d')
        fine_emb = F.pad(fine_emb, (0, 0, 1, 0), value=0) # pad class token to the first positions
        cts = F.pad(time_emb, (0, 0, 0, lf), value=0)
        fine_emb = fine_emb + cts
        coarse_emb = 0

        for idx, (fine_to_coarse, coarse_transformer, context_cross_attn, get_cts, fine_transformer) in enumerate(self.layers):
            coarse_emb_residual = fine_to_coarse(fine_emb[:, 0:1])
            coarse_emb_residual = rearrange(coarse_emb_residual, '(b t) d -> b t d', b=b)
            coarse_emb = coarse_emb + coarse_emb_residual
            coarse_emb = coarse_transformer(coarse_emb, rotary_emb=self.coarse_rotary_embedding)
            # context cross-attn
            coarse_emb = context_cross_attn(coarse_emb, context=coarse_context, rotary_emb=self.coarse_rotary_embedding)

            cts = get_cts(coarse_emb)
            cts = F.pad(cts, (0, 0, 0, lf), value=0)
            fine_emb = fine_emb + cts
            fine_emb = fine_transformer(fine_emb, rotary_emb=self.fine_rotary_embedding)
        
        # remove cts and reshape
        output = fine_emb[:, 1:]
        output = rearrange(output, '(b lc) lf d -> b d lf lc', b=b, lc=lc)

        return output

# base module for deep DPT
class TNTDiffusionNetwork(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            context_dim=512,
            depth=8,
            segment_size=64,
            segment_stride=32,
            dropout=0,
            semantic_cfg_prob=0.1,
            use_checkpoint=False,
            vc=False,
            vc_cfg_prob=None,
            lora=False,
            consistency=False,
        ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.context_dim = context_dim

        self.segment_size = segment_size
        self.segment_stride = segment_stride

        self.dpp = DualPathProcessing(segment_size, segment_stride)
        self.time_slerp_points = nn.Embedding(2, 256)
        self.time_embed = nn.Sequential(
            RMSNorm(256),
            nn.Linear(256, feature_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(feature_dim, feature_dim),
        )
        if consistency:
            linear_0 = nn.Linear(256, feature_dim)
            linear_1 = nn.Linear(feature_dim, feature_dim)
            # zero init as described in the paper
            torch.nn.init.zeros_(linear_0.weight)
            torch.nn.init.zeros_(linear_0.bias)
            torch.nn.init.zeros_(linear_1.weight)
            torch.nn.init.zeros_(linear_1.bias)
            self.guidance_embed = nn.Sequential(
                RMSNorm(256),
                linear_0,
                nn.GELU(approximate='tanh'),
                linear_1,
            )
        
        if context_dim == 1:
            emb_first_layer = nn.Sequential(
                nn.Embedding(32_768, feature_dim),
                RMSNorm(context_dim),
            )
        else:
            emb_first_layer = nn.Sequential(
                RMSNorm(context_dim),
                nn.Linear(context_dim, feature_dim),
            )

        self.semantic_context_embed = nn.Sequential(
            emb_first_layer,
            nn.GELU(approximate='tanh'),
            nn.Linear(feature_dim, feature_dim),
        )
        self.vc = vc
        if vc:
            self.vc_context_embed = nn.Sequential(
                RMSNorm(32),
                nn.Linear(32, feature_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(feature_dim, feature_dim),
            )
        # bottleneck
        self.input_map = nn.Sequential(
            nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False),
        )

        # DPT model
        self.blocks = TNTBlocks(
            input_dim=feature_dim,
            context_dim=context_dim,
            fine_dim=feature_dim,
            fine_heads=8,
            fine_head_dim=int(feature_dim / 8),
            coarse_dim=feature_dim,
            coarse_heads=8,
            coarse_head_dim=int(feature_dim / 8),
            depth=depth,
            dropout=dropout,
            semantic_cfg_prob=semantic_cfg_prob,
            vc_cfg_prob=vc_cfg_prob,
            vc=vc,
            use_checkpoint=use_checkpoint,
            lora=lora,
        )
        
        self.output = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.input_dim, 1, bias=False)
        )

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32, device=torch.device('cpu')):
        """
        Generate embedding vectors for multiple 'w' inputs.

        Args:
        w: torch.Tensor: 1-D or 2-D tensor representing multiple values of 'w' for which to generate embedding vectors.
        embedding_dim: int: dimension of the embeddings to generate.
        dtype: data type of the generated embeddings.

        Returns:
        embedding vectors with shape `(len(w), embedding_dim)` for 1-D input 
        or `(w.shape[0], w.shape[1], embedding_dim)` for 2-D input.
        """
        w = w * 1000.

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000., device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)

        # Adjusting the operation for the potentially 2-D 'w'
        emb = w.to(dtype)[:, :, None] * emb[None, None, :] if len(w.shape) == 2 else w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Concatenating along the last dimension

        if embedding_dim % 2 == 1:  # zero pad if necessary
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb
   
    def forward(
        self, 
        x, 
        timesteps=None, 
        semantic_context=None, 
        semantic_force_cfg=None,
        vc_context=None,
        vc_force_cfg=None,
        guidance_scale=None,
    ):
        batch_size, input_dim, seq_length = x.shape
        if timesteps.ndim != 3:
            timesteps = timesteps.view(batch_size, 1, 1).repeat(1, 1, seq_length)
        # input: (B, D, T)
        # temporal embedding
        t_start_emb = self.time_slerp_points(torch.zeros([batch_size, 1], device=x.device, dtype=torch.long))
        t_end_emb = self.time_slerp_points(torch.ones([batch_size, 1], device=x.device, dtype=torch.long))
        low_norm = t_start_emb / torch.norm(t_start_emb, dim=-1, keepdim=True)
        high_norm = t_end_emb / torch.norm(t_end_emb, dim=-1, keepdim=True)
        omega = torch.acos((low_norm*high_norm).sum(-1, keepdim=True))
        so = torch.sin(omega)
        t_emb = (torch.sin((1.0-timesteps.view(batch_size, seq_length, 1))*omega) / so) * t_start_emb\
            + (torch.sin(timesteps.view(batch_size, seq_length, 1)*omega) / so) * t_end_emb
        t_emb = self.time_embed(t_emb)
        if exists(guidance_scale):
            guidance_scale = self.get_w_embedding(guidance_scale, embedding_dim=256, dtype=t_emb.dtype, device=t_emb.device)
            guidance_emb = self.guidance_embed(guidance_scale)
            guidance_emb = repeat(guidance_emb, 'b d -> b t d', t=seq_length)
            t_emb = t_emb + guidance_emb
    
        # context embedding
        semantic_context_emb = self.semantic_context_embed(semantic_context)
        if self.vc:
            vc_context_emb = self.vc_context_embed(vc_context)
        else:
            vc_context_emb = None
        
        t_emb = self.dpp.unfold(rearrange(t_emb, 'b t d -> b d t'))

        # split the encoder output into overlapped, longer segments
        x = self.input_map(x) 
        x = self.dpp.unfold(x)
        out = self.blocks(
            x, 
            time_emb=t_emb, 
            semantic_context_emb=semantic_context_emb,
            semantic_force_cfg=semantic_force_cfg,
            vc_context_emb=vc_context_emb,
            vc_force_cfg=vc_force_cfg,
        ).view(batch_size, self.feature_dim, self.segment_size, -1)  # b, d, lf, lc      

        # overlap-and-add of the outputs
        out = self.dpp.fold(out)  # B, N, T
        out = self.output(out)

        out = out.view(batch_size, input_dim, seq_length)

        return out

class DualPathProcessing(nn.Module):
    """
    Perform Dual-Path processing via overlap-add as in DPRNN [1].
    Args:
        chunk_size (int): Size of segmenting window.
        stride (int): segmentation hop size.
    References
        [1] Yi Luo, Zhuo Chen and Takuya Yoshioka. "Dual-path RNN: efficient
        long sequence modeling for time-domain single-channel speech separation"
        https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size, stride):
        super(DualPathProcessing, self).__init__()
        self.chunk_size = chunk_size
        self.stride = stride
        self.n_orig_frames = None

    def unfold(self, x):
        r"""
        Unfold the feature tensor from $(batch, channels, time)$ to
        $(batch, channels, chunksize, nchunks)$.
        Args:
            x (:class:`torch.Tensor`): feature tensor of shape $(batch, channels, time)$.
        Returns:
            :class:`torch.Tensor`: spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        """
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        
        # pad to be evenly divisible by chunk_size
        self.pad_len = 0
        if frames % self.chunk_size != 0:
            pad_len = self.chunk_size - frames % self.chunk_size
            x = torch.nn.functional.pad(x, (0, pad_len))
            self.pad_len = pad_len

        self.n_orig_frames = x.shape[-1]
        
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(0, 0),
            stride=(self.stride, 1),
        )
    
        unfolded = rearrange(unfolded, 'b (c l1) l2 -> b c l1 l2', c=chan)
        
        return unfolded

    def fold(self, x, output_size=None):
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.
        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.
        .. note:: `fold` caches the original length of the input.
        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = rearrange(x, 'b c l1 l2 -> b (c l1) l2')
        
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(0, 0),
            stride=(self.stride, 1),
        ).squeeze(-1)

        if self.pad_len != 0:
            x = x[..., :-self.pad_len] # remove padding

        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    model = TNTDiffusionNetwork(
        input_dim=32,
        feature_dim=256,
        context_dim=1,
        depth=2,
        segment_size=32,
        segment_stride=32,
        dropout=0,
        semantic_cfg_prob=0.1,
        vc_cfg_prob=0.1,
        vc=True,
        use_checkpoint=True,
        consistency=True,
    ).to('cuda')
    # for k, v in model.named_parameters():
    #     print(k)
    xt = torch.randn(16, 32, 3750).to('cuda')
    t = torch.randn(16, 1, 3750).to('cuda')
    semantic = torch.ones(16, 750).long().to('cuda')
    vc = torch.randn(16, 1, 512).float().to('cuda')
    guidance_scale = torch.rand(16).float().to('cuda')
    
    out = model(xt, t, semantic_context=semantic, vc_context=vc, guidance_scale=guidance_scale)
    print(out.shape)
