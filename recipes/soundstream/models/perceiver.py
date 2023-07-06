import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List, Tuple
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint

from recipes.soundstream.models.modules.ema_vqvae import EMAVectorQuantizer

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
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
            nn.GELU(),
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
        dropout = 0.
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

    def forward(self, x, context=None, rotary_emb=None):
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        if rotary_emb is not None: 
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

# perceiver block
class PerceiverBlock(nn.Module):
    def __init__(
        self, 
        depth,  
        input_dim, 
        latent_dim, 
        latent_heads, 
        dim_latent_head,
        block_ratio=[1, 2],
        attn_dropout=0., 
        ff_dropout=0.,   
        use_checkpoint=False 
    ):
        super(PerceiverBlock, self).__init__()

        assert len(block_ratio) == 2
        self.n_cross_attn, self.n_latent_attn = block_ratio

        layers = nn.ModuleList([])
        for i in range(depth):
            sub_layer = nn.ModuleList([])

            # cross attn 
            cross_attns = nn.ModuleList([])
            for j in range(self.n_cross_attn):
                cross_attns.append(
                    nn.ModuleDict(
                        {
                            'norm_0': RMSNorm(latent_dim),
                            'norm_context': RMSNorm(input_dim),
                            'attn': Attention(
                                query_dim=latent_dim, 
                                context_dim=input_dim, 
                                heads=latent_heads, 
                                dim_head=dim_latent_head, 
                                dropout=attn_dropout
                            ),
                            'norm_1': RMSNorm(latent_dim),
                            'ff': FeedForward(dim=latent_dim, dropout=ff_dropout)
                        }
                    ) 
                )
                
            sub_layer.append(cross_attns)

            # latent attn 
            latent_attns = nn.ModuleList([])
            for j in range(self.n_latent_attn):
                latent_attns.append(
                    nn.ModuleDict(
                        {
                            'norm_0': RMSNorm(latent_dim),
                            'attn': Attention(
                                query_dim=latent_dim, 
                                heads=latent_heads, 
                                dim_head=dim_latent_head, 
                                dropout=attn_dropout
                            ),
                            'norm_1': RMSNorm(latent_dim),
                            'ff': FeedForward(dim=latent_dim, dropout=ff_dropout)
                        }
                    )     
                )
            sub_layer.append(latent_attns)

            layers.append(sub_layer)

        self.layers = layers
        self.use_checkpoint = use_checkpoint
    
    def _forward_checkpoint(self, x, context, rotary_emb=None):
        for idx, sub_layers in enumerate(self.layers):
            cross_attns, latent_attns = sub_layers
            # corss attn
            for cross_attn in cross_attns:
                x, context = checkpoint(cross_attn['norm_0'], x), checkpoint(cross_attn['norm_context'], context)
                x = checkpoint(cross_attn['attn'], x, context, rotary_emb) + x
                x = checkpoint(cross_attn['norm_1'], x)
                x = checkpoint(cross_attn['ff'], x) + x

            # latent attn 
            for latent_attn in latent_attns:
                x = checkpoint(latent_attn['norm_0'], x)
                x = checkpoint(latent_attn['attn'], x, None, rotary_emb) + x
                x = checkpoint(latent_attn['norm_1'], x)
                x = checkpoint(latent_attn['ff'], x) + x

        return x

    def _forward(self, x, context, rotary_emb=None):
        for idx, sub_layers in enumerate(self.layers):
            cross_attns, latent_attns = sub_layers
            # corss attn
            for cross_attn in cross_attns:
                x, context = cross_attn['norm_0'](x), cross_attn['norm_context'](context)
                x = cross_attn['attn'](x, context, rotary_emb) + x
                x = cross_attn['norm_1'](x)
                x = cross_attn['ff'](x) + x

            # latent attn 
            for latent_attn in latent_attns:
                x = latent_attn['norm_0'](x)
                x = latent_attn['attn'](x, None, rotary_emb) + x
                x = latent_attn['norm_1'](x)
                x = latent_attn['ff'](x) + x

        return x
    
    def forward(self, x, context, rotary_emb=None):
        if self.use_checkpoint:
            return self._forward_checkpoint(x, context, rotary_emb)
        else:
            return self._forward(x, context, rotary_emb)


class Audio2Emb(nn.Module):
    def __init__(
        self,
        window_size=128,
        hop_size=64,
        emb_dim=64,
    ):
        '''
        apply slicing window on raw waveform and project to embedding space
        '''
        super(Audio2Emb, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = torch.nn.Parameter(torch.hann_window(window_size), requires_grad=False)

        self.norm = RMSNorm(window_size)
        self.to_emb = nn.Linear(window_size, emb_dim)
    

    def forward(self, x):

        # sliding window
        x = x.unfold(-1, self.window_size, self.hop_size)
        # apply window function
        x = x * self.window
        # flattent the channel dimension
        x = rearrange(x, 'b c l d -> b (c l) d')

        return self.to_emb(self.norm(x))

class Emb2Audio(nn.Module):
    def __init__(
        self,
        window_size=128,
        hop_size=64,
        emb_dim=64,
    ):
        '''
        apply overlap and add to reconstruct waveform from embedding space
        '''
        super(Emb2Audio, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = torch.nn.Parameter(torch.hann_window(window_size), requires_grad=False)

        self.norm = RMSNorm(emb_dim)
        self.to_emb = nn.Linear(emb_dim, window_size)

    def forward(self, x):
        # TODO: check the trick to output large value then norm
        x = torch.tanh(self.to_emb(self.norm(x)))
        # overlap and add
        # TODO: use torch internal function
        output = torch.zeros(
            (x.shape[0], 1, x.shape[1] * self.hop_size + self.window_size - self.hop_size), 
            device=x.device
        )
        for i in range(x.shape[1]):
            output[:, :, i*self.hop_size:i*self.hop_size+self.window_size] += (self.window * x[:, i:i+1, :])
        return output


class PerceiverRVQ(torch.nn.Module):
    def __init__(
        self,
        num_res=12,
        quant_token_num=1024,
        quant_token_dim=256,
        quant_beta=0.25,
        init_cluster_size=1,
        window_size=128,
        hop_size=64,
        input_emb_dim=64,
        encoder_latent_freq=200,
        encoder_latent_dim=64,
        encoder_latent_heads=8,
        encoder_depth=3,
        encoder_block_ratio=[1, 2],
        decoder_latent_dim=64,
        decoder_latent_heads=8,
        decoder_depth=3,
        decoder_block_ratio=[1, 2],
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=True,
        dist=False,

    ):
        super(PerceiverRVQ, self).__init__()

        # RVQ
        self.num_res = num_res
        self.quant_vaes = nn.ModuleList()
        if not isinstance(quant_token_num, (tuple, list)):
            quant_token_nums = [quant_token_num] * num_res
        else:
            quant_token_nums = quant_token_num
        for i in range(self.num_res):
            self.quant_vaes.append(
                EMAVectorQuantizer(
                    quant_token_num=quant_token_nums[i],
                    quant_token_dim=quant_token_dim,
                    quant_beta=quant_beta,
                    init_cluster_size=init_cluster_size,
                    dist=dist,
                )
            )

        # Auto encoder
        self.encoder_latent = nn.Parameter(torch.randn(encoder_latent_freq, encoder_latent_dim))
        self.decoder_latent = nn.Parameter(torch.randn(decoder_latent_dim))
        
        self.encoder_rotary_emb = RotaryEmbedding(dim=int(encoder_latent_dim / encoder_latent_heads))
        self.decoder_rotary_emb = RotaryEmbedding(dim=int(decoder_latent_dim / decoder_latent_heads))

        self.audio2emb = Audio2Emb(
            window_size=window_size,
            hop_size=hop_size,
            emb_dim=input_emb_dim,
        )
        self.emb2audio = Emb2Audio(
            window_size=window_size,
            hop_size=hop_size,
            emb_dim=decoder_latent_dim,
        )

        self.encoder = PerceiverBlock(
            depth=encoder_depth,  
            input_dim=input_emb_dim, 
            latent_dim=encoder_latent_dim, 
            latent_heads=encoder_latent_heads, 
            dim_latent_head=int(encoder_latent_dim / encoder_latent_heads),
            block_ratio=encoder_block_ratio,
            attn_dropout=attn_dropout, 
            ff_dropout=ff_dropout,   
            use_checkpoint=use_checkpoint 
        )
        self.decoder = PerceiverBlock(
            depth=decoder_depth,  
            input_dim=encoder_latent_dim, 
            latent_dim=decoder_latent_dim, 
            latent_heads=decoder_latent_heads, 
            dim_latent_head=int(decoder_latent_dim / decoder_latent_heads),
            block_ratio=decoder_block_ratio,
            attn_dropout=attn_dropout, 
            ff_dropout=ff_dropout,   
            use_checkpoint=use_checkpoint 
        )

    def quant(self, x, warmup=False):
        encoder_out = x
        quant_outs = []
        losses = []
        quant_indexs = []

        for quant_vae in self.quant_vaes:
            quant_out, loss, quant_index = quant_vae(encoder_out)
            quant_outs.append(quant_out)
            losses.append(loss)
            quant_indexs.append(quant_index)
            encoder_out = encoder_out - quant_out.detach()

        quant_out = sum(quant_outs)
        quant_loss = sum(losses)
        quant_index = quant_indexs
        return quant_out, quant_loss, quant_index
        
    def forward(self, x, warmup=False):
        # transform waveform to embedding space
        x = self.audio2emb(x)

        # repeat latent embedding to batch dimension
        latent_emb = repeat(self.encoder_latent, 'n d -> b n d', b=x.shape[0])

        # encoder
        encoder_out = self.encoder(
            latent_emb, 
            context=x, 
            rotary_emb=self.encoder_rotary_emb
        )

        encoder_out = rearrange(encoder_out, 'b l d -> b d l')
        quant_out, quant_loss, quant_index = self.quant(encoder_out)
        quant_out = rearrange(quant_out, 'b d l -> b l d')
        
        # repeat latent embedding to batch dimension and time dimension
        output_emb = repeat(self.decoder_latent, 'd -> b n d', b=x.shape[0], n=x.shape[1])

        # decoder
        decoder_out = self.decoder(
            output_emb, 
            context=quant_out, 
            rotary_emb=self.decoder_rotary_emb
        )
        # overlap add back to waveform
        decoder_out = self.emb2audio(decoder_out)

        return decoder_out, quant_loss, quant_index, encoder_out

    
if __name__ == '__main__':

    model = PerceiverRVQ(
        num_res=12,
        quant_token_num=1024,
        quant_token_dim=256,
        quant_beta=0.25,
        init_cluster_size=32,
        window_size=256,
        hop_size=128,
        input_emb_dim=256,
        encoder_latent_freq=10,
        encoder_latent_dim=256,
        encoder_latent_heads=8,
        encoder_depth=1,
        encoder_block_ratio=[1, 2],
        decoder_latent_dim=256,
        decoder_latent_heads=8,
        decoder_depth=1,
        decoder_block_ratio=[1, 2],
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=True
    )
    test_input = torch.randn(1, 48000*20)

    from recipes.soundstream.utils.audio_utils import pad_audio, enframe, deframe
    segment_samples = 48000
    hop_samples = 24000
    test_input = pad_audio(
        test_input, 
        segment_samples=segment_samples,
        hop_samples=hop_samples)
    test_input = enframe(
        test_input, 
        segment_samples=segment_samples, 
        hop_samples=hop_samples)

    out = model(test_input)

    test_input = deframe(
        test_input,
        hop_samples=hop_samples
    )
 

    encoder_cnt = 0
    for p in model.encoder.parameters():
        encoder_cnt += p.numel()
    
    decoder_cnt = 0
    for p in model.decoder.parameters():
        decoder_cnt += p.numel()

    print('encode parameters: ', encoder_cnt)
    print('decode parameters: ', decoder_cnt)

    