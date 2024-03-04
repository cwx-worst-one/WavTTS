from torch import nn
from torch.nn import Linear
from torch.nn import functional as F

from recipes.umm.models.attention.simple_attention import SimpleAttention
from recipes.umm.models.voc_modules.pitch_predictor.model import ConvBlocks


class ConvStacksWithDownUpSampling(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_dim,
        output_dim,
        downsampling=1,
        upsampling=1,
        num_layers=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proj_in = nn.Conv1d(input_dim, hidden_size, 3, padding=1)
        self.conv_stacks1 = ConvBlocks(
            hidden_size,
            hidden_size,
            None,
            5,
            layers_in_block=2,
            num_layers=num_layers,
            norm_type="ln",
            dropout=0,
            post_net_kernel=3,
            is_BTC=False,
        )
        self.conv_stacks2 = ConvBlocks(
            hidden_size,
            output_dim,
            None,
            5,
            layers_in_block=2,
            num_layers=num_layers,
            norm_type="ln",
            dropout=0,
            post_net_kernel=3,
            is_BTC=False,
        )
        self.downsampling = downsampling
        self.upsampling = upsampling

    def forward(self, x, nonpadding=None):
        if nonpadding is None:
            nonpadding = (x.abs().sum(-1) > 0).float()[..., None]
        nonpadding = nonpadding.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.proj_in(x) * F.interpolate(
            nonpadding, size=x.shape[-1], mode="nearest"
        )
        x = self.conv_stacks1(
            x, F.interpolate(nonpadding, size=x.shape[-1], mode="nearest")
        )
        if self.downsampling > 1:
            x = F.avg_pool1d(x, self.downsampling)
        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling, mode="nearest")
        x = self.conv_stacks2(
            x, F.interpolate(nonpadding, size=x.shape[-1], mode="nearest")
        )
        x = x.transpose(1, 2)
        return x

    def get_flops(*args, **kwargs):
        """
        @hanoihantrakul 3Mar2024
        In order to make this compatible with old UMM interface, this methods needs to be defined.
        However, the actual number of flops does not matter.
        """
        return 0


class MultiRefTimbreEncoder(nn.Module):
    def __init__(self, hs, in_dims, q_hs):
        super().__init__()
        self.fg_spk_enc_hidden = hs
        self.timbre_mel_enc_in = Linear(in_dims, hs)
        self.timbre_q_enc = ConvBlocks(q_hs, q_hs, [1] * 2, 3, is_BTC=True)
        self.timbre_mel_enc1 = ConvBlocks(hs, hs, [1] * 5, 3, is_BTC=True)
        self.timbre_mel_ds = nn.Conv1d(hs, hs, stride=16, kernel_size=16)
        self.timbre_mel_enc2 = ConvBlocks(hs, hs * 2, [1] * 5, 3, is_BTC=True)
        self.timbre_attn = SimpleAttention(q_hs, hs, hs, 2)

    def forward(self, x_q, mels):
        # if self.is_intra_sent:
        # mels, ret['intra_sent_mel_mask'] = self.get_intra_sent_mel_seg(mels, mel_lengths, infer=infer)

        nonpadding_mel = (mels.abs().sum(-1) > 0).float()[:, :, None]
        nonpadding_q = (x_q.abs().sum(-1) > 0).float()[:, :, None]
        Q = self.timbre_q_enc(x_q) * nonpadding_q
        h_mels = self.timbre_mel_enc_in(mels) * nonpadding_mel
        h_mels = self.timbre_mel_enc1(h_mels) * nonpadding_mel
        h_mels = self.timbre_mel_ds(h_mels.transpose(1, 2)).transpose(1, 2)
        nonpadding_mel = F.pad(nonpadding_mel, [0, 0, 0, 16])
        nonpadding_mel = nonpadding_mel[:, ::16][:, : h_mels.shape[1]]
        h_mels = self.timbre_mel_enc2(h_mels) * nonpadding_mel
        K, V = (
            h_mels[..., : self.fg_spk_enc_hidden],
            h_mels[..., self.fg_spk_enc_hidden :],
        )
        attn_mask = (nonpadding_q[..., 0][:, :, None] > 0) & (
            nonpadding_mel[..., 0][:, None] > 0
        )
        attn_mask = 1 - attn_mask.float()
        out, _ = self.timbre_attn(Q, K, V, attn_mask=attn_mask)
        return out
