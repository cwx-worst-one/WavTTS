import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List, Tuple
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint

from recipes.soundstream.models.modules.ema_vqvae import EMAVectorQuantizer
from recipes.soundstream.utils.utils import get_padding, init_weights

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

class ConNeXt1DBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=int(kernel_size//2), groups=dim) # depthwise conv
        self.norm = RMSNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        
        x = input + self.drop_path(x)
        return x

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

class BasicBlock(nn.Module):
    def __init__(
        self,
        layout='ctd',
        kernel_size=[3],
        transformer_depth=1,
        dim=64, 
        heads=8, 
        dim_head=8,
        resample_dim=64,
        resample_rate=2,
        mode='downsample',
        attn_dropout=0., 
        ff_dropout=0.,   
        use_checkpoint=False 
    ):
        super(BasicBlock, self).__init__()

        assert mode in ['upsample', 'downsample'], 'mode must be either upsample or downsample'

        layers = nn.ModuleList([
            None, None, None
        ])
        # conv
        if 'c' in layout:
            convs = nn.ModuleList([])
            for k in kernel_size:
                convs.append(
                    ConNeXt1DBlock(dim, kernel_size=k)
                )
            layers[0] = convs
        # transformer
        if 't' in layout:
            transformer = nn.ModuleList([])
            for j in range(transformer_depth):
                transformer.append(
                    nn.ModuleDict(
                        {
                            'norm_0': RMSNorm(dim),
                            'attn': Attention(
                                query_dim=dim, 
                                heads=heads, 
                                dim_head=dim_head, 
                                dropout=attn_dropout
                            ),
                            'norm_1': RMSNorm(dim),
                            'ff': FeedForward(dim=dim, dropout=ff_dropout)
                        }
                    )     
                )
            layers[1] = transformer
        # downsample
        if 'd' in layout:
            if mode == 'downsample':
                c = nn.Conv1d(dim, resample_dim, kernel_size=resample_rate*2 + 1, stride=resample_rate, padding=resample_rate)
            elif mode == 'upsample':
                # transpose conv, output length is stride times original input length
                c = nn.ConvTranspose1d(dim, resample_dim, kernel_size=resample_rate, stride=resample_rate)

            resample_layer= nn.Sequential(
                Rearrange('b c l -> b l c'),
                RMSNorm(dim),
                Rearrange('b l c -> b c l'),
                c,
                
            )
            layers[2] = resample_layer

        self.layers = layers
        self.layout = layout
        self.use_checkpoint = use_checkpoint
    
    def _forward_checkpoint(self, x, rotary_emb=None):
        convs, transformers, resample = self.layers
        if 'c' in self.layout:
            for conv in convs:
                x = checkpoint(conv, x)
        x = rearrange(x, 'b c l -> b l c')
        if 't' in self.layout:
            for transformer in transformers:
                x = checkpoint(transformer['norm_0'], x)
                x = checkpoint(transformer['attn'], x, None, rotary_emb) + x
                x = checkpoint(transformer['norm_1'], x)
                x = checkpoint(transformer['ff'], x) + x
        x = rearrange(x, 'b l c -> b c l')
        if 'd' in self.layout:
            x = checkpoint(resample, x)

        return x

    def _forward(self, x, rotary_emb=None):
        convs, transformers, resample = self.layers
        if 'c' in self.layout:
            for conv in convs:
                x = conv(x)
        x = rearrange(x, 'b c l -> b l c')
        if 't' in self.layout:
            for transformer in transformers:
                x = transformer['norm_0'](x)
                x = transformer['attn'](x, None, rotary_emb) + x
                x = transformer['norm_1'](x)
                x = transformer['ff'](x) + x
        x = rearrange(x, 'b l c -> b c l')
        if 'd' in self.layout:
            x = resample(x)

        return x
    
    def forward(self, x, rotary_emb=None):
        if self.use_checkpoint:
            return self._forward_checkpoint(x, rotary_emb)
        else:
            return self._forward(x, rotary_emb)

class BasicModel(nn.Module):
    def __init__(
        self,
        layout=['cd', 'cd', 'cd', 'cd'],
        dim=[256, 256, 256, 256],
        kernel_size=[[3], [3], [3], [3]],
        transformer_depth=[1, 1, 1, 1],
        heads=[8, 8, 8, 8],
        dim_head=[32, 32, 32, 32],
        resample_dim=[256, 256, 256, 256],
        resample_rate=[2, 2, 2, 2],
        mode='downsample', 
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=False
    ):
        super(BasicModel, self).__init__()
        
        layers = nn.ModuleList([])
        for idx in range(len(layout)):
            block = BasicBlock(
                layout=layout[idx],
                kernel_size=kernel_size[idx],
                transformer_depth=transformer_depth[idx],
                dim=dim[idx],
                heads=heads[idx],
                dim_head=dim_head[idx],
                resample_dim=resample_dim[idx],
                resample_rate=resample_rate[idx],
                mode=mode,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_checkpoint=use_checkpoint
            )
            layers.append(block)
        self.layers = layers

    def forward(self, x, rotary_emb=None):
        for layer in self.layers:
            x = layer(x, rotary_emb)
        return x

class Audio2Emb(nn.Module):
    def __init__(
        self,
        emb_dim=64,
        kernel_size=59,
    ):
        '''
        apply slicing window on raw waveform and project to embedding space
        '''
        super(Audio2Emb, self).__init__()

        # pre conv
        self.pre_conv = nn.Sequential(
            nn.Conv1d(1, emb_dim, kernel_size=kernel_size, padding=int(kernel_size//2)),
            Rearrange('b c l -> b l c'),
            RMSNorm(emb_dim),
            Rearrange('b l c -> b c l'),
        )

    def forward(self, x):

        # pre conv
        x = self.pre_conv(x)

        return x

class Emb2Audio(nn.Module):
    def __init__(
        self,
        emb_dim=64,
        kernel_size=31,
    ):
        '''
        apply overlap and add to reconstruct waveform from embedding space
        '''
        super(Emb2Audio, self).__init__()

        self.to_audio = nn.Sequential(
            Rearrange('b c l -> b l c'),
            RMSNorm(emb_dim),
            Rearrange('b l c -> b c l'),
            nn.Conv1d(emb_dim, 1, kernel_size=kernel_size, padding=int(kernel_size//2)),
            nn.Tanh() 
        )

    def forward(self, x):
        # TODO: check the trick to output large value then norm
        x = self.to_audio(x)
        return x

class TransformerRVQ(torch.nn.Module):
    def __init__(
        self,
        num_res=12,
        quant_token_num=1024,
        quant_token_dim=256,
        quant_beta=0.25,
        init_cluster_size=1,
        window_size=128,
        hop_size=64,
        encoder_dim=64,
        encoder_heads=8,
        encoder_resample_rates=[2, 4, 8, 16],
        encoder_resample_interval=2,
        decoder_dim=64,
        decoder_heads=8,
        decoder_resample_rates=[2, 4, 8, 16],
        decoder_resample_interval=2,
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=True,
        dist=False,

    ):
        super(TransformerRVQ, self).__init__()

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
        self.encoder_rotary_emb = RotaryEmbedding(dim=int(encoder_dim / encoder_heads))
        self.decoder_rotary_emb = RotaryEmbedding(dim=int(decoder_dim / decoder_heads))

        self.audio2emb =Audio2Emb(
            emb_dim=32,
            kernel_size=59,
        )
        self.emb2audio = Emb2Audio(
            emb_dim=32,
            kernel_size=31,
        )

        self.encoder =  BasicModel(
            layout=['c', 'cd', 'c', 'cd', 'c', 'cd', 'c', 'cd'],
            dim=[32, 32, 64, 64, 128, 128, 256, 256],
            kernel_size=[
                [13, 13, 13], 
                [25, 25, 25], 
                [13, 13, 13], 
                [25, 25, 25],
                [13, 13, 13], 
                [25, 25, 25],
                [5, 5, 5], 
                [7, 7, 7],
            ],
            transformer_depth=[None, None, None, None, None, None, None, 3],
            heads=[None, None, None, None, None, None, None, 8],
            dim_head=[None, None, None, None, None, None, None, 32],
            resample_dim=[None, 64, None, 128, None, 256, None, 256],
            resample_rate=[1, 2, 1, 2, 1, 10, 1, 24],
            mode='downsample', 
            attn_dropout=0.,
            ff_dropout=0.,
            use_checkpoint=False
        )
        self.decoder = BasicModel(
            layout=['c', 'cd', 'c', 'cd', 'c', 'cd', 'c', 'cd'],
            dim=[256, 256, 128, 128, 64, 64, 32, 32],
            kernel_size=[
                [7, 7, 7], 
                [13, 13, 13], 
                [13, 13, 13], 
                [25, 25, 25],
                [13, 13, 13], 
                [25, 25, 25],
                [13, 13, 13], 
                [25, 25, 25],
            ],
            transformer_depth=[3, None, None, None, None, None, None, None],
            heads=[8, None, None, None, None, None, None, None],
            dim_head=[32, None, None, None, None, None, None, None],
            resample_dim=[None, 128, None, 64, None, 32, None, 32],
            resample_rate=[1, 20, 1, 6, 1, 4, 1, 2],
            mode='upsample', 
            attn_dropout=0.,
            ff_dropout=0.,
            use_checkpoint=False
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
        # encoder
        encoder_out = self.encoder(
            x, 
            rotary_emb=self.encoder_rotary_emb
        )
        quant_out, quant_loss, quant_index = self.quant(encoder_out)
        # decoder
        decoder_out = self.decoder(
            quant_out, 
            rotary_emb=self.decoder_rotary_emb
        )
        # overlap add back to waveform
        decoder_out = self.emb2audio(decoder_out)

        return decoder_out, quant_loss, quant_index, encoder_out

    
if __name__ == '__main__':

    model = TransformerRVQ(
        num_res=12,
        quant_token_num=1024,
        quant_token_dim=256,
        quant_beta=0.25,
        init_cluster_size=1,
        window_size=64,
        hop_size=32,
        encoder_dim=256,
        encoder_heads=8,
        encoder_resample_rates=[2, 5, 5],
        encoder_resample_interval=2,
        decoder_dim=256,
        decoder_heads=8,
        decoder_resample_rates=[5, 5, 2],
        decoder_resample_interval=2,
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=True,
        dist=False,
    )
    test_input = torch.randn(1, 48000*20)

    from recipes.soundstream.utils.audio_utils import pad_audio, enframe, deframe
    segment_samples = 48000
    hop_samples = int (segment_samples // 2)
    test_input = pad_audio(
        test_input, 
        segment_samples=segment_samples,
        hop_samples=hop_samples)
    test_input = enframe(
        test_input, 
        segment_samples=segment_samples, 
        hop_samples=hop_samples)
    print(test_input.shape)
    out = model(test_input)
    print(out[0].shape)

    test_input = deframe(
        test_input,
        hop_samples=hop_samples
    )
    print(test_input.shape)
 

    encoder_cnt = 0
    for p in model.encoder.parameters():
        encoder_cnt += p.numel()
    
    decoder_cnt = 0
    for p in model.decoder.parameters():
        decoder_cnt += p.numel()

    print('encode parameters: ', encoder_cnt)
    print('decode parameters: ', decoder_cnt)

    