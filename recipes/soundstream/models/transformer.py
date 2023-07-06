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

class TransformerBlock(nn.Module):
    def __init__(
        self,  
        dim, 
        heads, 
        dim_head,
        resample_rates,
        resample_interval=2,
        mode='downsample',
        attn_dropout=0., 
        ff_dropout=0.,   
        use_checkpoint=False 
    ):
        super(TransformerBlock, self).__init__()

        assert mode in ['upsample', 'downsample'], 'mode must be either upsample or downsample'
        self.mode = mode
        self.resample_rates = resample_rates

        layers = nn.ModuleList([])
        for resample_rate in resample_rates:
            sub_layer = nn.ModuleList([])
            # transformer
            transformer = nn.ModuleList([])
            for _ in range(resample_interval):
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
            resample = None
            if resample_rate != 1:
                if mode == 'downsample':
                    resample = nn.ModuleDict(
                        {
                            'norm': RMSNorm(dim),
                            'conv': nn.Conv1d(dim, dim, kernel_size=resample_rate*2 + 1, stride=resample_rate, padding=resample_rate),
                        }
                    )
                elif mode == 'upsample':
                    resample = nn.ModuleDict(
                        {
                            'norm': RMSNorm(dim),
                            'conv': nn.ConvTranspose1d(dim, dim, kernel_size=resample_rate, stride=resample_rate),
                        }
                    )
            
            sub_layer.append(transformer)
            sub_layer.append(resample)
            layers.append(sub_layer)

        self.layers = layers
        self.use_checkpoint = use_checkpoint
    
    def _forward_checkpoint(self, x, rotary_emb=None):
        for idx, sub_layers in enumerate(self.layers):
            transformers, resample = sub_layers
            # upsample
            if resample is not None and self.mode == 'upsample':
                x = checkpoint(resample['norm'], x, use_reentrant=False)
                x = rearrange(x, 'b n d -> b d n')
                x = checkpoint(resample['conv'], x, use_reentrant=False)
                x = rearrange(x, 'b d n -> b n d')

            for transformer in transformers:
                x = checkpoint(
                    transformer['attn'], 
                    checkpoint(transformer['norm_0'], x, use_reentrant=False), 
                    None, 
                    rotary_emb, 
                    use_reentrant=False
                ) + x
          
                x = checkpoint(
                    transformer['ff'], 
                    checkpoint(transformer['norm_1'], x, use_reentrant=False), 
                    use_reentrant=False
                ) + x

            # downsample
            if resample is not None and self.mode == 'downsample':
                x = checkpoint(resample['norm'], x, use_reentrant=False)
                x = rearrange(x, 'b n d -> b d n')
                x = checkpoint(resample['conv'], x, use_reentrant=False)
                x = rearrange(x, 'b d n -> b n d')

        return x

    def _forward(self, x, rotary_emb=None):
        for idx, sub_layers in enumerate(self.layers):
            transformers, resample = sub_layers
            # upsample
            if resample is not None and self.mode == 'upsample':
                x = resample['norm'](x)
                x = rearrange(x, 'b n d -> b d n')
                x = resample['conv'](x)
                x = rearrange(x, 'b d n -> b n d')

            for transformer in transformers:
                x = transformer['attn'](transformer['norm_0'](x), None, rotary_emb) + x
                x = transformer['ff'](x = transformer['norm_1'](x)) + x
            # downsample
            if resample is not None and self.mode == 'downsample':
                x = resample['norm'](x)
                x = rearrange(x, 'b n d -> b d n')
                x = resample['conv'](x)
                x = rearrange(x, 'b d n -> b n d')
        return x
    
    def forward(self, x, rotary_emb=None):
        if self.use_checkpoint:
            return self._forward_checkpoint(x, rotary_emb)
        else:
            return self._forward(x, rotary_emb)

class AudioProcessing(nn.Module):
    def __init__(
        self,
        window_size=128,
        hop_size=64,
        emb_dim=64,
        divisor=1,
    ):
        '''
        apply slicing window on raw waveform and project to embedding space
        '''
        super(AudioProcessing, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        # self.window = torch.nn.Parameter(torch.hann_window(window_size), requires_grad=False)
        self.divisor = divisor
        self.audio2emb_layer = nn.Sequential(
            RMSNorm(window_size),
            nn.Linear(window_size, emb_dim)
        )
        self.emb2audio_layer = nn.Sequential(
            RMSNorm(emb_dim),
            nn.Linear(emb_dim, window_size),
            nn.Tanh()
        )

    def audio2emb(self, x):
        # sliding window
        self.n_orig_len = x.shape[-1]
        x = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            padding=(self.hop_size, 0),
            stride=(self.hop_size, 1),
        )
        # # apply window function
        # x = x * self.window
        # flattent the channel dimension
        x = rearrange(x, 'b d l -> b l d')
        self.n_orig_frames = x.shape[1]
        # pad to be evenly divided by divisor
        self.pad_len = 0
        if x.shape[1] % self.divisor != 0:
            pad_len = self.divisor - x.shape[1] % self.divisor
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len), 'constant', 0)
            self.pad_len = pad_len
        
        return self.audio2emb_layer(x)

    def emb2audio(self, x):
        # remove padding
        if self.pad_len != 0:
            x = x[:, :-self.pad_len, :]
        # TODO: check the trick to output large value then norm
        x = self.emb2audio_layer(x)
        
        # overlap and add
        x = rearrange(x, 'b l d -> b d l')
        output = torch.nn.functional.fold(
            x,
            output_size=(self.n_orig_len, 1),
            kernel_size=(self.window_size, 1),
            padding=(self.hop_size, 0),
            stride=(self.hop_size, 1),
        )
        # assume 50 % overlap for now
        output /= float(self.window_size)/self.hop_size
        return output.squeeze(-1)


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

        self.audio_process = AudioProcessing(
            window_size=window_size,
            hop_size=hop_size,
            emb_dim=encoder_dim,
            divisor=torch.prod(torch.tensor(encoder_resample_rates))
        )
        
        self.encoder = TransformerBlock(
            dim=encoder_dim, 
            heads=encoder_heads, 
            dim_head=int(encoder_dim / encoder_heads),
            resample_rates=encoder_resample_rates,
            resample_interval=encoder_resample_interval,
            mode='downsample',
            attn_dropout=attn_dropout, 
            ff_dropout=ff_dropout,   
            use_checkpoint=use_checkpoint 
        )
        self.decoder = TransformerBlock(
            dim=decoder_dim, 
            heads=decoder_heads, 
            dim_head=int(decoder_dim / decoder_heads),
            resample_rates=decoder_resample_rates,
            resample_interval=decoder_resample_interval,
            mode='upsample',
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
        x = self.audio_process.audio2emb(x)
        
        # encoder
        encoder_out = self.encoder(
            x, 
            rotary_emb=self.encoder_rotary_emb
        )

        encoder_out = rearrange(encoder_out, 'b l d -> b d l')
        quant_out, quant_loss, quant_index = self.quant(encoder_out)
        quant_out = rearrange(quant_out, 'b d l -> b l d')

        # decoder
        decoder_out = self.decoder(
            quant_out, 
            rotary_emb=self.decoder_rotary_emb
        )
        # overlap add back to waveform
        decoder_out = self.audio_process.emb2audio(decoder_out)

        return decoder_out, quant_loss, quant_index, encoder_out

class TransformerKL(torch.nn.Module):
    def __init__(
        self,
        window_size=128,
        hop_size=64,
        latent_dim=64,
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
    ):
        super(TransformerKL, self).__init__()

        # Auto encoder
        self.encoder_rotary_emb = RotaryEmbedding(dim=int(encoder_dim / encoder_heads))
        self.decoder_rotary_emb = RotaryEmbedding(dim=int(decoder_dim / decoder_heads))

        self.audio_process = AudioProcessing(
            window_size=window_size,
            hop_size=hop_size,
            emb_dim=encoder_dim,
            divisor=torch.prod(torch.tensor(encoder_resample_rates))
        )
        self.vae_mean = nn.Linear(encoder_dim, latent_dim)
        self.vae_logvar = nn.Linear(encoder_dim, latent_dim)
        self.latent_to_feature = nn.Linear(latent_dim, decoder_dim)
        
        self.encoder = TransformerBlock(
            dim=encoder_dim, 
            heads=encoder_heads, 
            dim_head=int(encoder_dim / encoder_heads),
            resample_rates=encoder_resample_rates,
            resample_interval=encoder_resample_interval,
            mode='downsample',
            attn_dropout=attn_dropout, 
            ff_dropout=ff_dropout,   
            use_checkpoint=use_checkpoint 
        )
        self.decoder = TransformerBlock(
            dim=decoder_dim, 
            heads=decoder_heads, 
            dim_head=int(decoder_dim / decoder_heads),
            resample_rates=decoder_resample_rates,
            resample_interval=decoder_resample_interval,
            mode='upsample',
            attn_dropout=attn_dropout, 
            ff_dropout=ff_dropout,   
            use_checkpoint=use_checkpoint 
        )

    def sample(self, x, deterministic=False):
        mean = self.vae_mean(x)
        logvar = self.vae_logvar(x)
     
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        sample = self.latent_to_feature(sample)
        return sample, kl_loss.mean(), std.mean()

        
    def forward(self, x, warmup=False):
        # transform waveform to embedding space
        x = self.audio_process.audio2emb(x)
   
        # encoder
        encoder_out = self.encoder(
            x, 
            rotary_emb=self.encoder_rotary_emb
        )

        sample, kl_loss, std = self.sample(encoder_out)

        # decoder
        decoder_out = self.decoder(
            sample, 
            rotary_emb=self.decoder_rotary_emb
        )
        # overlap add back to waveform
        decoder_out = self.audio_process.emb2audio(decoder_out)

        return decoder_out, kl_loss, std

if __name__ == '__main__':

    model = TransformerKL(
        window_size=64,
        hop_size=32,
        latent_dim=64,
        encoder_dim=512,
        encoder_heads=8,
        encoder_resample_rates=[2, 4, 4, 1],
        encoder_resample_interval=3,
        decoder_dim=512,
        decoder_heads=8,
        decoder_resample_rates=[2, 4, 4, 1],
        decoder_resample_interval=3,
        attn_dropout=0.,
        ff_dropout=0.,
        use_checkpoint=True,
    )
    test_input = torch.randn(1, 24000*20)

    from recipes.soundstream.utils.audio_utils import pad_audio, enframe, deframe
    segment_samples = 24000#32 + 32*768
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
 

    encoder_cnt = 0
    for p in model.encoder.parameters():
        encoder_cnt += p.numel()
    
    decoder_cnt = 0
    for p in model.decoder.parameters():
        decoder_cnt += p.numel()

    print('encode parameters: ', encoder_cnt)
    print('decode parameters: ', decoder_cnt)

    
