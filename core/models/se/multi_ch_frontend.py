import torch
from torch import nn
import torch.nn.functional as F
import pdb
import math
from math import sqrt
from core.models.se.conv_stft import ConvSTFT, ConviSTFT
from core.dataset.preprocess.kaldi_fbank import get_mel_banks
import scipy.io.wavfile as wav
import numpy as np
from scipy.linalg import toeplitz
from core.utils import get_local_rank
from core.models.utils import xavier_init


class MulChFrontend(nn.Module):
    """Multichannel frontend structure for ASR.

    Stft -> Neural Frontend -> Power-spec -> Mel-Fbank
    """

    def __init__(
        self,
        args=None,
        mic_num=6,
        fft_len=512,
        n_mel=80,
        frame_len=400,
        frame_shift=160,
        wav_key='mc_waveform',
        fbank_key='src',
    ):
        super().__init__()
        if args != None:
            mic_num = args.mic_num
            fft_len = args.fft_len
            n_mel = args.n_mel
            frame_len = args.frame_len
            frame_shift = args.frame_shift
            wav_key = args.wav_key
            fbank_key = args.fbank_key
        self.fft_len = fft_len
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.mic_num = mic_num
        self.wav_key = wav_key
        self.fbank_key = fbank_key
        fft_dim = fft_len // 2 + 1
        local_rank = get_local_rank()
        self.device = torch.device("cuda", local_rank)

        # stft module
        self.stft = ConvSTFT(frame_len, frame_shift, fft_len=fft_len, feature_type='complex').to(
            self.device
        )
        self.istft = ConviSTFT(frame_len, frame_shift, fft_len=512, feature_type='complex').to(
            self.device
        )
        # self.win = torch.hann_window(frame_len).to(self.device)

        # MC-DNN
        self.mcdnn = MCDNNlayer_v1_attn(fft_dim=fft_dim, mic_num=mic_num).to(self.device)

        # feature extraction network
        self.EPSILON = torch.tensor(torch.finfo(torch.float).eps).to(self.device)
        # self.EPSILON = 1e-8
        self.fe_layer = nn.Sequential(*[FELayer(n_mel, fft_dim, self.device), nn.ReLU()]).to(
            self.device
        )
        self.bn = nn.BatchNorm1d(n_mel, momentum=0.1, affine=True).to(self.device)

        for p in self.parameters():
            if len(p.size()) > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, batch):
        wav = batch[self.wav_key]
        bsz = wav.shape[0]
        wav = wav.reshape(bsz, self.mic_num, -1)

        # stft
        wav_stft = []
        for i in range(self.mic_num):
            wav_stft.append(self.stft(wav[:, i, :]))
            # wav_stft.append(torch.stft(wav[:,i,:],n_fft=self.fft_len,hop_length=self.frame_shift, win_length=self.frame_len, window=self.win, center=True,return_complex=False))
            # print(wav_stft[i].shape,wav_stft[i].dtype)
        wav_stft = torch.stack(wav_stft, dim=1)

        # MC DNN
        merge_spectrum = self.mcdnn(wav_stft)
        # nan_idx = torch.isnan(merge_spectrum)
        # merge_spectrum = torch.where(
        #     nan_idx == True, torch.zeros(merge_spectrum.shape).to(self.device), merge_spectrum
        # )
        batch['stft'] = wav_stft[:, 0, :, :, :].pow(2).sum(dim=-1)

        # FE
        # merge_spectrum = wav_stft[:,0,:,:,:].pow(2).sum(dim=-1).unsqueeze(2).contiguous()
        merge_spectrum = merge_spectrum.unsqueeze(2).contiguous()  # [B, T, 1, 257]
        mel_energies = torch.max(self.fe_layer(merge_spectrum), self.EPSILON)  # [B, T, 80]
        mel_energies = torch.transpose(mel_energies, 1, 2)
        fbank = self.bn(mel_energies)
        fbank = torch.transpose(fbank, 1, 2)

        # padding
        frame_len = fbank.shape[1]
        pad_right = 12 - frame_len % 12
        if pad_right > 0:
            fbank = torch.nn.functional.pad(fbank, [0, 0, 0, pad_right], 'constant', 0.0)
            fbank_mask = batch['src_mask']
            mask_len = fbank_mask.shape[1]
            frame_len = fbank.shape[1]
            mask_pad_right = max(frame_len - mask_len, 0)
            fbank_mask_pad = torch.nn.functional.pad(
                fbank_mask, [0, mask_pad_right], 'constant', 0.0
            )
            batch['src_mask'] = fbank_mask_pad

        ori_fbank = batch[self.fbank_key]
        batch['ori_src'] = ori_fbank
        batch[self.fbank_key] = fbank
        return batch


##############################################################################################################


class MCDNNlayer_v1_attn(nn.Module):
    def __init__(
        self, fft_dim=257, hidden=64, mic_num=8, num_head=4, kernel_size=[5, 5], padding=[2, 2]
    ):
        super().__init__()
        self.fft_dim = fft_dim
        self.mic_num = mic_num
        self.complex_conv = complex_conv2d(mic_num, mic_num, kernel_size, padding)
        self.complex_fc_in = complex_dense(fft_dim, hidden)
        self.conv_block = nn.Conv2d(mic_num, mic_num, kernel_size=kernel_size, padding=padding)
        self.channel_wise_attn = ChannelWiseSelfAttention_simple(
            mic_num, num_head, hidden, hidden, 0.1
        )
        self.cross_channel_attn = CrossChanneleAttention_simple(
            mic_num, num_head, hidden, hidden // num_head, hidden // num_head, hidden, 0.1
        )
        self.fc_out = nn.Linear(hidden, fft_dim)
        self.reset_parameters()

    def reset_parameters(self):
        xavier_init(self.complex_conv)
        xavier_init(self.complex_fc_in)
        xavier_init(self.conv_block)
        xavier_init(self.fc_out)
        xavier_init(self.channel_wise_attn)
        xavier_init(self.cross_channel_attn)

    def forward(self, x_in):
        # input [B, C, T, F, 2]
        # output [B,T,F]
        x = x_in.clone()
        x = self.complex_conv(x)
        x = self.complex_fc_in(x)
        x = (x**2).sum(dim=4)
        x = self.conv_block(x)
        x = self.channel_wise_attn(x)
        x = self.cross_channel_attn(x)
        x = self.fc_out(x)
        x = torch.softmax(x, dim=1).unsqueeze(-1)
        x = (x_in * x).sum(dim=1)
        x = x.pow(2).sum(dim=-1)
        return x


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(nn.functional.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    '''Multi-Head Attention module'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # permute之后若想用view聚合维度必须使用contiguous
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, n_channels, n_head, d_model, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn_list = nn.ModuleList()
        self.feed_forward_list = nn.ModuleList()
        for _ in range(n_channels):
            self.self_attn_list.append(MultiHeadAttention(n_head, d_model, d_k, d_v, dropout))
            self.feed_forward_list.append(PositionwiseFeedForward(d_model, d_ff, dropout))

    def forward(self, x):
        n_batch, n_channel, n_time, n_freq = x.shape
        output = []
        for i in range(n_channel):
            output_current, _ = self.self_attn_list[i](x[:, i, :, :], x[:, i, :, :], x[:, i, :, :])
            output_current = self.feed_forward_list[i](output_current)
            output.append(output_current)
        output = torch.stack(output, dim=1)
        return output


class ChannelWiseSelfAttention_simple(nn.Module):
    def __init__(self, n_channels, n_head, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        n_batch, n_channel, n_time, n_freq = x.shape
        x = x.permute(2, 1, 0, 3)
        output = []
        for i in range(n_channel):
            output_current, _ = self.self_attn(x[:, i, :, :], x[:, i, :, :], x[:, i, :, :])
            output_current = self.feed_forward(output_current)
            output.append(output_current)
        output = torch.stack(output, dim=1).permute(2, 1, 0, 3)
        return output


class CrossChanneleAttention(nn.Module):
    def __init__(self, n_channels, n_head, d_model, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.channel_embedding_list = nn.ModuleList()
        self.cross_attn_list = nn.ModuleList()
        self.feed_forward_list = nn.ModuleList()
        for _ in range(n_channels):
            self.channel_embedding_list.append(nn.Linear(d_model, d_model, bias=False))
            self.cross_attn_list.append(MultiHeadAttention(n_head, d_model, d_k, d_v, dropout))
            self.feed_forward_list.append(PositionwiseFeedForward(d_model, d_ff, dropout))

    def forward(self, x):
        n_batch, n_channel, n_time, n_freq = x.shape
        output = []
        for i in range(n_channel):
            kv_embedding = None
            for j in range(n_channel):
                if j == i:
                    pass
                if kv_embedding == None:
                    kv_embedding = self.channel_embedding_list[j](x[:, j, :, :])
                else:
                    kv_embedding = kv_embedding + self.channel_embedding_list[j](x[:, j, :, :])
            output_current, _ = self.cross_attn_list[i](x[:, i, :, :], kv_embedding, kv_embedding)
            output_current = self.feed_forward_list[i](output_current)
            output.append(output_current)
        output = torch.stack(output, dim=1)
        return output


class CrossChanneleAttention_simple(nn.Module):
    def __init__(self, n_channels, n_head, d_model, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.channel_embedding = nn.Linear(d_model, d_model, bias=False)
        self.cross_attn = MultiHeadAttention(n_head, n_channels * d_model, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(n_channels * d_model, d_ff, dropout)

    def forward(self, x):
        n_batch, n_channel, n_time, n_freq = x.shape
        x_emb = x.reshape(-1, n_time, n_freq)
        x_emb = self.channel_embedding(x_emb)
        x_emb = x_emb.reshape(n_batch, n_channel, n_time, n_freq).permute(0, 2, 1, 3)
        x_emb = x_emb.reshape(n_batch, n_time, -1)
        x = x.permute(0, 2, 1, 3).reshape(n_batch, n_time, -1)
        x = self.cross_attn(x, x_emb, x_emb)[0]
        x = self.feed_forward(x)
        x = x.reshape(n_batch, n_time, n_channel, -1).permute(0, 2, 1, 3)
        return x


class FELayer(nn.Module):
    def __init__(self, fbank_dim, fft_dim, device, sr=16000):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, fbank_dim, fft_dim))
        fft_len = (fft_dim - 1) * 2
        mel_energies, _ = get_mel_banks(fbank_dim, fft_len, sr, 20, 0, 100, -500, 1)  # (80,256)
        mel_energies = mel_energies.to(device)
        mel_energies = (
            F.pad(mel_energies, (0, 1), mode='constant', value=0).unsqueeze(0).unsqueeze(0)
        )  # (1,1,80,257)
        self.weight.data.copy_(mel_energies)
        self.weight.requires_grad = True
        self.mask = torch.zeros((self.weight.size())).float().to(device)
        self.mask[self.weight != 0.0] = 1.0
        self.mask.requires_grad = False

    def forward(self, x):  # b,f,1,257,
        return (x * self.weight * self.mask).sum(dim=-1)


class complex_conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        x_re_in = x[:, :, :, :, 0]
        x_im_in = x[:, :, :, :, 1]
        x_re = self.conv_re(x_re_in) - self.conv_im(x_im_in)
        x_im = self.conv_re(x_im_in) + self.conv_im(x_re_in)
        return torch.stack([x_re, x_im], dim=4)


class complex_dense(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.fc_re = nn.Linear(in_ch, out_ch, bias=bias)
        self.fc_im = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x):
        x_re_in = x[:, :, :, :, 0]
        x_im_in = x[:, :, :, :, 1]
        x_re = self.fc_re(x_re_in) - self.fc_im(x_im_in)
        x_im = self.fc_re(x_im_in) + self.fc_im(x_re_in)
        return torch.stack([x_re, x_im], dim=4)


class conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6()
        self.pool = nn.MaxPool2d([1, 3], stride=[1, 2], padding=[0, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.bn(x))
        x = self.pool(x)
        return x


class separable_conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, bias=False, last_act=nn.ReLU6()):
        super().__init__()
        self.conv_depth = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_ch
        )
        self.bn_depth = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU6()
        self.pool = nn.MaxPool2d([1, 3], stride=[1, 2], padding=[0, 1])
        self.conv_point = nn.Conv2d(in_ch, out_ch, kernel_size=[1, 1], padding=[0, 0], bias=bias)
        self.bn_point = nn.BatchNorm2d(out_ch)
        self.last_act = last_act

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.act(self.bn_depth(x))
        x = self.pool(x)
        x = self.conv_point(x)
        x = self.last_act(self.bn_point(x))
        return x


def xavier_init(module):
    '''xavier uniform'''
    for p in module.parameters():
        if len(p.size()) > 1:
            torch.nn.init.xavier_uniform_(p)
