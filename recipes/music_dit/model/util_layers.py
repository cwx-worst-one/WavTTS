import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random

from .functional import revgrad


class RMSNorm(nn.Module):
    def __init__(self, dim, feat_dim=-1, eps=1e-5):
        super().__init__()
        self.rms = dim**-0.5
        self.feat_dim = feat_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, unscaled=False):
        norm = torch.norm(x, dim=self.feat_dim, keepdim=True) * self.rms
        if unscaled:
            return x / norm.clamp(min=self.eps)
        g = self.scale
        if self.feat_dim != -1:
            while g.ndim <= self.feat_dim:
                g = g[None]
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
        return x / norm.clamp(min=self.eps) * g


class NearestUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        up_scale = 10
        self.conv = nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=2 * up_scale - 1,
                      padding_mode='zeros',
                      padding=up_scale - 1)

    def forward(self, x, size):
        x_size = x.size(2)
        x = F.interpolate(x, size=size, mode='nearest')
        x = self.conv(x)
        return x


class ResBlcok1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
                Function(lambda x: x.transpose(1, 2)),
                nn.LayerNorm(channels),
                Function(lambda x: x.transpose(1, 2)),
                nn.SiLU(inplace=True),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                Function(lambda x: x.transpose(1, 2)),
                nn.LayerNorm(channels),
                Function(lambda x: x.transpose(1, 2)),
                nn.SiLU(inplace=True),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1)
                )
        
    def forward(self, x):
        return x + self.conv_block(x)



class DownsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 scale):

        super(DownsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scale = scale
        self.skip_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.index = index
        layer = nn.Conv1d(input_size, 
                          output_size, 
                          kernel_size=scale * 2,
                          stride=scale, 
                          padding=scale // 2 + scale % 2)

        self.layer = nn.utils.weight_norm(layer)

    def forward(self, inputs):
        B, C, T = inputs.size()
        res = inputs[:, :, ::self.scale]
        skip = self.skip_conv(res)
        outputs = self.layer(inputs)
        outputs = outputs + skip
        return outputs


#From xin wang of nii
class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0, 
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    def __init__(self, samp_rate = 24000, harmonic_num = 0, 
                 sine_amp = 0.1, noise_std = 0.003,
                 voiced_threshold = 0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
    
    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv
            
    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],\
                              device = f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini        
        
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - 
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            
            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) \
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every 
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within 
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return  sines
    
    
    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """

        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, \
                                    device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx+1] = f0_buf[:, :, 0] * (idx+2)
                
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            
            # generate uv signal
            #uv = torch.ones(f0.shape)
            #uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)
            
            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            #.       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1-uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            
            # first: set the unvoiced part to 0 by uv
            # then: additive noise

            #sine_waves = sine_waves * uv + noise
            #TODO @ tianqiao.wave
            sine_waves = sine_waves * uv
        return sine_waves, uv, noise

class GRL(nn.Module):
    def __init__(self, alpha=1.):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self._alpha)

class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, resstack_depth=4, dilation=1):
        super().__init__()
        
        def get_padding(kernel_size, dilation):
            return int((kernel_size*dilation - dilation)/2)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=dilation, padding=get_padding(kernel_size, dilation))),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=dilation, padding=get_padding(kernel_size, dilation))),
            )
            for i in range(resstack_depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


def get_loop_up_embedding(n_symbols, embedding_dim):
    embedding = nn.Embedding(n_symbols, embedding_dim)
    val = 0.05
    embedding.weight.data.uniform_(-val, val)
    return embedding


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths)
    batch_size = lengths.shape[0]
    if batch_size != 1:
        enc_masks = torch.zeros(
            batch_size, max_len, dtype=torch.float).to(lengths.device)
        for e_id, src_len in enumerate(lengths):
            enc_masks[e_id, :src_len] = 1
    # for onnx conversion. ORT does not support pytorch tensor slice
    else:
        enc_masks = torch.ones(
            batch_size, max_len, dtype=torch.float).to(lengths.device)

    return enc_masks.bool()


def get_mel_mask(seq_len, max_len, mel_bins=80):
    # mel_shape: [B, T, n_mel]
    b_mask = get_mask_from_lengths(seq_len, max_len)
    mask = b_mask.float().unsqueeze(2)
    mask = mask.expand(-1, -1, mel_bins)
    return mask


def random_upsampling(x, enc_len, min_k=1, max_k=4):
    # x: B, T, C
    B = x.size()[0]
    T = x.size()[1]
    new_x = []
    new_enc_len = [0] * B

    for i in range(T):
        repeat = random.randint(min_k, max_k)
        for _ in range(repeat):
            new_x.append(x[:, i, :])
            for j in range(B):
                if i < enc_len[j]:
                    new_enc_len[j] += 1

    new_enc_len = np.array(new_enc_len)
    new_enc_len = torch.Tensor(new_enc_len).to(x.device).int()
    output = torch.stack(new_x, dim=1)
    return output, new_enc_len


def spec_augment(mel_spectrogram, frequency_masking_para=15,
                 time_masking_para=15, frequency_mask_num=1, time_mask_num=1):
    """Spec[B, C, T] augmentation Calculation Function.  
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    # warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)
    # for crop based, do not time_warp
    warped_mel_spectrogram = mel_spectrogram.clone()

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v-f)
        warped_mel_spectrogram[:, f0:f0+f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau-t)
        warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram


class LinearNorm(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 bias=True, 
                 w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 causal=False,
                 bias=True,
                w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation = dilation

        if padding is None:
            assert (kernel_size % 2 == 1)
            if causal:
                padding = int(dilation * (kernel_size - 1))
            else:
                padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        if self.causal and self.kernel_size > 1:
            conv_signal = conv_signal[:,:,:-self.dilation*(self.kernel_size-1)]
        return conv_signal



class Deconv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Deconv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - stride + 1) // 2  # from tests
        self.output_padding = 2 * self.padding - kernel_size + stride
        self.convTrans1d = nn.ConvTranspose1d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              output_padding=self.output_padding)

    def forward(self, inputs):
        # inputs: [B, channel, T]
        outputs = self.convTrans1d(inputs)
        return outputs


class ResSplitConv1DBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResSplitConv1DBlock, self).__init__()
        inner_out_channels = channels * 2
        padding = (kernel_size - 1) * dilation // 2
        self.channels = channels
        self.conv = nn.Conv1d(in_channels=channels,
                              out_channels=inner_out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=padding)

    def forward(self, inputs):
        # inputs: [B, channel, T]
        conv_out = self.conv(inputs)
        conv_out_l, conv_out_r = torch.split(conv_out, [self.channels, self.channels], dim=1)
        conv_out_r = torch.sigmoid(conv_out_r)
        conv_out_resi = conv_out_l * conv_out_r
        output = torch.add(inputs, conv_out_resi) * math.sqrt(0.5)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, rel_pos=None, mask=None):
        # q (k, v): (n*b) x lq x dk
        # for mask, if effective lengths is [1, 2, 3],
        # mask should be:
        # [[False, True, True],
        #  [False, False, True],
        #  [False, False, False]]

        attn = torch.bmm(q, k.transpose(1, 2))  # [n*b, lq, lk]
        if rel_pos is not None:
            # rel_pos: [lq, lk, n]
            pos = rel_pos.permute(2, 0, 1).contiguous()
            pos = pos.repeat(q.shape[0] // pos.shape[0], 1, 1)
            attn += pos

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, value=torch.tensor(-np.inf))

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, rel_pos=None, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, rel_pos=rel_pos, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)
        output = F.relu(self.fc(output))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size=(9, 1), dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0], padding=(fft_conv1d_kernel_size[0] - 1) // 2)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1], padding=(fft_conv1d_kernel_size[1] - 1) // 2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(mish(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel_size=(9, 1),
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size=fft_conv1d_kernel_size, dropout=dropout)

    def forward(self, enc_input, rel_pos=None, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, rel_pos=rel_pos, mask=slf_attn_mask)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class RelativePositionEmbedding(nn.Module):
    """
    https://arxiv.org/abs/1803.02155
    """

    def __init__(self, input_dim, output_dim):
        super(RelativePositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=output_dim)
        init_val = 0.05
        torch.nn.init.uniform_(self.embedding.weight, -init_val, init_val)

    def compute_position_ids(self, inputs):
        q, v = inputs  # q: [B, T, dim1], v:[B, T, dim2]
        q_idxs = torch.arange(0, q.shape[1])  # dtype should be long
        q_idxs = q_idxs.unsqueeze(1)  # [T, 1]
        v_idxs = torch.arange(0, v.shape[1])  # [T2]
        v_idxs = v_idxs.unsqueeze(0)  # [1, T]
        pos_ids = v_idxs - q_idxs  # [T, T]

        max_position = (self.input_dim - 1) // 2
        pos_ids = torch.clamp(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids

    def forward(self, inputs):
        pos_ids = self.compute_position_ids(inputs).long()
        pos_ids = pos_ids.to(self.embedding.weight.device)
        pos_embeded = self.embedding(pos_ids)
        return pos_embeded


class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    """
    Google T5, used in https://arxiv.org/abs/1910.10683;
    """

    def __init__(self, input_dim, output_dim, max_distance=128, bidirectional=True):
        super(RelativePositionEmbeddingT5, self).__init__(input_dim, output_dim)
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def compute_position_ids(self, inputs):
        q, v = inputs
        q_idxs = torch.arange(0, q.shape[1])  # dtype should be long
        q_idxs = q_idxs.unsqueeze(1)  # [T, 1], [T_q, 1]
        v_idxs = torch.arange(0, v.shape[1])  # [T2]
        v_idxs = v_idxs.unsqueeze(0)  # [1, T], [1, T_v]
        pos_ids = v_idxs - q_idxs  # [T, T], [T_q, T_v]
#         pos_ids.to(q.device)
        
        num_buckets = self.input_dim
        max_distance = self.max_distance
        relative_position = -pos_ids
        relative_buckets = 0
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position < 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets


class ZoneoutRNN(nn.Module):
    def __init__(self, forward_cell, zoneout_prob):
        super(ZoneoutRNN, self).__init__()
        self.forward_cell = forward_cell
        self.zoneout_prob = zoneout_prob

        if not isinstance(forward_cell, nn.RNNCellBase):
            raise TypeError("The cell is not a LSTMCell or GRUCell!")
        if isinstance(forward_cell, nn.LSTMCell):
            if not isinstance(zoneout_prob, tuple):
                raise TypeError("The LSTM zoneout_prob must be a tuple!")
        elif isinstance(forward_cell, nn.GRUCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The GRU zoneout_prob must be a float number!")
        elif isinstance(forward_cell, nn.RNNCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The RNN zoneout_prob must be a float number!")

    @property
    def hidden_size(self):
        return self.forward_cell.hidden_size
    @property
    def input_size(self):
        return self.forward_cell.input_size

    def forward(self, forward_input, forward_state):
        forward_new_state = self.forward_cell(forward_input, forward_state)

        if isinstance(self.forward_cell, nn.LSTMCell):
            forward_h, forward_c = forward_state
            forward_new_h, forward_new_c = forward_new_state

            zoneout_prob_h, zoneout_prob_c = self.zoneout_prob
            if self.training:
                forward_new_h = (1 - zoneout_prob_h) * F.dropout(
                                forward_new_h-forward_h, p=zoneout_prob_h,
                                training=self.training) + forward_h
                forward_new_c = (1 - zoneout_prob_c) * F.dropout(
                                forward_new_c-forward_c, p=zoneout_prob_c,
                                training=self.training) + forward_c
            else:
                forward_new_h = (1 - zoneout_prob_h) * forward_new_h + zoneout_prob_h * forward_h
                forward_new_c = (1 - zoneout_prob_c) * forward_new_c + zoneout_prob_c * forward_c
            forward_new_state = (forward_new_h, forward_new_c)
            forward_output = forward_new_h
 
        else:
            forward_h = forward_state
            forward_new_h = forward_new_state

            zoneout_prob_h = self.zoneout_prob

            if self.training:
                forward_new_h = (1 - zoneout_prob_h) * F.dropout(
                        forward_new_h-forward_h, p=zoneout_prob_h,
                        training=self.training) + forward_h
            else:
                forward_new_h = (1 - zoneout_prob_h) * forward_new_h + zoneout_prob_h * forward_h

            forward_new_state = forward_new_h
            forward_output = forward_new_h
        return forward_output, forward_new_state


class MaskedInstanceNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-7, const_std=0.5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.const_std = const_std

    def forward(self, x, uv):
        # B,1,T
        mean = torch.zeros((x.shape[0], self.num_features, 1), requires_grad=False).to(x.device)

        for c in range(self.num_features):
            for b in range(x.shape[0]):
                mean[b, c, 0] = (x[b, c, :]*uv[b, c, :]).sum() / (uv[b, c, :].sum() + self.eps)

        x = (x - mean)/self.const_std
        x *= uv
        return x



def time_jitter(x, time_jitter_prob=0.12):
    '''
        apply time jitter on time dimension
    '''
    b, t, c = x.size()
    left_prob = torch.rand(t)
    right_prob = torch.rand(t)

    for i in range(0,t):
        if i == 0:
            if right_prob[i] < time_jitter_prob:
                x[:,1,:] = x[:,0,:]
        elif i == t-1:
            if left_prob[i] < time_jitter_prob:
                x[:,-2,:] = x[:,-1,:]
        else:
            if right_prob[i] < time_jitter_prob:
                x[:,i+1,:] = x[:,i,:]
            if left_prob[i] < time_jitter_prob:
                x[:,i-1,:] = x[:,i,:]
    return x

class Function(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
