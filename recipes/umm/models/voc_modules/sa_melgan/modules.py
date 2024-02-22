import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from .utils import init_weights, get_padding
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import numpy as np

from .dataset import get_pitch_condition_torch


class Upsample(nn.Module):
    def __init__(self, mult, r):
        super(Upsample, self).__init__()
        self.r = r
        self.upsample = nn.Sequential(nn.Upsample(mode="nearest", scale_factor=r),
                                      nn.LeakyReLU(0.2),
                                      nn.ReflectionPad1d(3),
                                      nn.utils.weight_norm(nn.Conv1d(mult, mult // 2, kernel_size=7, stride=1))
                                      )
        r_kernel = r if r >= 5 else 5
        self.trans_upsample = nn.Sequential(nn.LeakyReLU(0.2),
                                            nn.utils.weight_norm(nn.ConvTranspose1d(mult, mult // 2,
                                                                                    kernel_size=r_kernel * 2, stride=r,
                                                                                    padding=r_kernel - r // 2,
                                                                                    output_padding=r % 2)
                                                                 ))

    def forward(self, x):
        x = torch.sin(x) + x
        out1 = self.upsample(x)
        out2 = self.trans_upsample(x)
        return out1 + out2


class Downsample(nn.Module):
    def __init__(self, mult, r):
        super(Downsample, self).__init__()
        self.r = r
        r_kernel = r if r >= 5 else 5
        self.trans_downsample = nn.Sequential(nn.LeakyReLU(0.2),
                                              nn.utils.weight_norm(nn.Conv1d(mult, mult * 2,
                                                                             kernel_size=r_kernel * 2, stride=r,
                                                                             padding=r_kernel - r // 2)
                                                                   ))

    def forward(self, x):
        out = self.trans_downsample(x)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_zero_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Audio2Mel(nn.Module):
    def __init__(
            self,
            hop_length=300,
            sampling_rate=24000,
            n_mel_channels=80,
            mel_fmin=0.,
            mel_fmax=None,
            frame_size=0.05,
            device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################

        self.n_fft = int(np.power(2., np.ceil(np.log(sampling_rate * frame_size) / np.log(2))))
        window = torch.hann_window(int(sampling_rate * frame_size)).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, self.n_fft, n_mel_channels, mel_fmin, mel_fmax
        )  # Mel filter (by librosa)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.hop_length = hop_length
        self.win_length = int(sampling_rate * frame_size)
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        fft = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(torch.clamp(real_part ** 2 + imag_part ** 2, min=1e-5))
        mel_output = torch.matmul(self.mel_basis, magnitude)

        log_mel_spec = 20 * torch.log10(torch.clamp(mel_output, min=1e-5)) - 20
        norm_mel = (log_mel_spec + 115.) / 115.
        mel_comp = torch.clamp(norm_mel * 8. - 4., -4., 4.)

        return mel_comp


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, dim_in=None):
        super().__init__()
        if dim_in is None:
            dim_in = dim

        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim_in, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim_in, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


'''
参照hifigan（https://arxiv.org/pdf/2010.05646.pdf）v2结构
多尺度主要是kernel_size不同，3组并行卷积模块，每个卷积模块内部采用不同的串行dilation size，且中间交叉正常无dilation卷积层
'''


class ResBlockMRFV2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlockMRFV2, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.2)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlockMRFV2Inter(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlockMRFV2Inter, self).__init__()
        self.block1 = ResBlockMRFV2(channels)
        self.block2 = ResBlockMRFV2(channels, 7)
        self.block3 = ResBlockMRFV2(channels, 11)

    def forward(self, x):
        xs = self.block1(x)
        xs += self.block2(x)
        xs += self.block3(x)
        x = xs / 3
        return x


'''
参照hifigan（https://arxiv.org/pdf/2010.05646.pdf）v3结构，相较于v2减少了卷积层数；
目前实验验证质量下降较大，暂时没有使用。
'''


class ResBlockMRFV3(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlockMRFV3, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.2)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, input_size_, ngf, n_residual_layers, num_band, args, ratios=[5, 5, 4, 3], onnx_export=False,
                 device='cpu'):
        super().__init__()
        self.hop_length = args.frame_shift
        self.args = args
        self.onnx_export = onnx_export
        res_layers = getattr(args, 'res_layers', 1)

        if self.args.use_pitch_condition:
            # -------------  Define downsample layers if using f0_condition  ----------------
            mult = 1
            model_down = []
            model_down += [
                nn.ReflectionPad1d(3),
                WNConv1d(args.dim_pitch_condition, mult * ngf, kernel_size=7, padding=0)
            ]

            for i, r in enumerate(reversed(ratios)):
                model_down += [Downsample(mult * ngf, r)]
                for j in reversed(range(n_residual_layers)):
                    model_down += [ResnetBlock(mult * ngf * 2, dilation=3 ** j) for _ in range(res_layers)]
                mult *= 2

            self.model_down = nn.Sequential(*model_down)

            # ------------- Between layer -------------
            # Match mel dim = downsample layer output dim
            dim_downsample_out = mult * ngf
            self.model_interval_linear = nn.Linear(input_size_, dim_downsample_out)

        # ------------- Define upsample layers ----------------
        mult = int(2 ** len(ratios))
        model_up = []
        if not self.args.use_pitch_condition:
            input_size = input_size_
        else:
            input_size = 2 * dim_downsample_out  # downsample_out + mel_linear
        model_up += [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model_up += [Upsample(mult * ngf, r)]
            if args.use_pitch_condition:
                # if use pitch_condition, the u-net res connection need dim to be doubled for first j in n_residual_layers
                model_up += [WNConv1d(mult * ngf, mult * ngf // 2, kernel_size=1)]
            model_up += [ResBlockMRFV2Inter(mult * ngf // 2)]
            mult //= 2

        model_up += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, num_band, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        if not args.use_tanh:
            model_up[-1] = nn.Conv1d(num_band, num_band, 1)
        model_up[-2].apply(weights_zero_init)

        self.model_up = nn.Sequential(*model_up)

        # ------------- Define MSG GAN layers ----------------
        # After each upsampling residual layer, use a dense layer to convert to target sampling rate audio
        if args.use_msg_gan:
            model_msg = []
            mult = int(2 ** len(ratios))
            for i, r in enumerate(ratios[:-1]):
                # The last ratio is not used, since the output of last upsample will be the finally output, no need msg connection
                model_msg += [nn.Linear(mult * ngf // 2, 1)]
                mult //= 2
            self.model_msg = nn.ModuleList(model_msg)

        # # ------------- Define Pitch Prediction layers ----------------
        # if args.use_pitch_prediction:
        #     assert self.args.use_pitch_condition
        #     self.model_pitch_predictor = Pitch_predictor(num_dim_mel=args.n_mel_channels)

        self.apply(weights_init)

    def forward(self, mel, pitch_condition=None, use_what_pitch='gt', step=None):
        # mel input: (batch_size, seq_num, 80)
        if self.onnx_export:
            mel = mel.transpose(1, 2)
            # on onnx, for engineering, mel input: (batch_size, 80, seq_num)
        additional_out = {}
        if getattr(self.args, 'all_noise', False):
            pitch_condition = torch.zeros_like(pitch_condition)

        # Downsample pipline
        if self.args.use_pitch_condition:
            pitch_condition = get_pitch_condition_torch(mels=mel.transpose(1, 2), pitch=pitch_condition,
                                                        frame_shift=self.args.frame_shift,
                                                        sampling_rate=self.args.sampling_rate,
                                                        mode=self.args.mode_pitch_condition,
                                                        onnx_export=self.onnx_export,
                                                        noise_index=self.args.noise_index)
            assert pitch_condition is not None
            unet_res = []
            x = pitch_condition
            for i, m in enumerate(self.model_down):
                if type(m) == Downsample:
                    unet_res.append(x)
                x = m(x)

        # Between Down and up
        if self.args.use_pitch_condition:
            mel_linear = self.model_interval_linear(mel.transpose(1, 2)).transpose(1, 2)
            x = torch.cat([mel_linear, x], 1)
        else:
            x = mel

        # Upsample pipline
        cnt_after_upsample = 0
        if self.args.use_msg_gan:
            x_msg_output = []

        for i, m in enumerate(self.model_up):
            x = m(x)

            if self.args.use_pitch_condition and type(m) == Upsample:
                unet_res_cur = unet_res[-cnt_after_upsample - 1]
                x = torch.cat([x, unet_res_cur], 1)
            if self.args.use_msg_gan and cnt_after_upsample > 0 and i + 1 < len(self.model_up) and type(
                    self.model_up[i + 1]) == Upsample:
                x_msg_output.append(self.model_msg[cnt_after_upsample - 1](x.transpose(1, 2)).transpose(1, 2))

            if type(m) == Upsample:
                cnt_after_upsample += 1

        if self.args.use_msg_gan:
            additional_out['x_msg_output'] = x_msg_output

        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, ratios_down, args):
        super().__init__()
        self.ratios_down = ratios_down
        self.args = args

        ndf = args.ndf
        model = nn.ModuleDict()
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        kf = 4  # kernel factor, experiential value
        for n_layer, ratio_cur in enumerate(ratios_down):
            nf_prev = nf
            nf = min(nf * args.downsamp_factor, 1024)

            model["layer_%d_downsample" % (n_layer + 1)] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=kf * 10 + 1,
                    stride=args.downsamp_factor,
                    padding=kf * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        model["layer_%d" % (len(ratios_down) + 1)] = nn.Sequential(
            WNConv1d(nf, nf // 2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (len(ratios_down) + 2)] = WNConv1d(
            nf // 2, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

        if args.use_msg_gan:
            model_msg_expand_wav = []
            model_msg_compressdim = []
            nf = ndf
            for i, ratio_cur in enumerate(ratios_down[:-1]):
                nf = min(nf * ratio_cur, 1024)
                model_msg_expand_wav.append(nn.Linear(1, nf))
                model_msg_compressdim.append(nn.Linear(2 * nf, nf))

            self.model_msg_expand_wav = nn.ModuleList(model_msg_expand_wav)
            self.model_msg_compressdim = nn.ModuleList(model_msg_compressdim)

    def forward(self, x, x_msg_output=None):
        results = []
        for i, (key, layer) in enumerate(self.model.items()):
            x = layer(x)
            if self.args.use_msg_gan and 'downsample' in key and i - 1 < len(x_msg_output):
                # downsample layer start from index 1
                index = i - 1
                msg_input_cur = list(reversed(x_msg_output))[index]
                msg_input_cur_expanded = self.model_msg_expand_wav[index](msg_input_cur.transpose(1, 2)).transpose(1, 2)
                x_cat = torch.cat([msg_input_cur_expanded, x], 1)
                x = self.model_msg_compressdim[index](x_cat.transpose(1, 2)).transpose(1, 2)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()
        ratios_down = list(reversed(args.up_sample))
        self.args = args

        self.model = nn.ModuleDict()
        for i in range(args.num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ratios_down,
                args
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x, x_msg_output=None):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, x_msg_output))
            x = self.downsample(x)
            if x_msg_output is not None:
                x_msg_output = [self.downsample(x) for x in x_msg_output]
        return results


# multi period discriminator submodule
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# multi period discriminator full module
class MultiPeriodDiscriminator(torch.nn.Module):
    '''
    guided by hifi-gan
    '''

    def __init__(self, device='cpu'):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


'''
参考论文，https://arxiv.org/pdf/2206.13404.pdf
multi-scale dilated convolution banks
two types of sub-modules: one captures the spectral feature’s changes over the time
axis and the other captures the relationship between each sub-band signal. These two sub-modules
are referred to as tSBD and fSBD
'''


class MSD_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=(5, 7, 11)):
        super(MSD_Block, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        ori_x = x
        for cidx, c in enumerate(self.convs):
            if cidx == 0:
                x = c(ori_x)
            else:
                x += c(ori_x)
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class tSBD(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilation=(5, 7, 11), fmap_depth=2):
        super(tSBD, self).__init__()
        self.fmap_depth = fmap_depth
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            MSD_Block(channels, 64, kernel_size, 1, dilation),
            MSD_Block(64, 128, kernel_size, 1, dilation),
            MSD_Block(128, 256, kernel_size, 1, dilation),
            MSD_Block(256, 256, kernel_size, 1, dilation),
            MSD_Block(256, 256, kernel_size, 1, dilation),
        ])
        self.conv_post = norm_f(Conv1d(256, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            if i >= self.fmap_depth:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class fSBD(torch.nn.Module):

    def __init__(self, channels, kernel_size, fmap_depth=2):
        super(fSBD, self).__init__()
        self.fmap_depth = fmap_depth
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            MSD_Block(channels, 32, kernel_size, 1, (1, 2, 3)),
            MSD_Block(32, 64, kernel_size, 1, (1, 2, 3)),
            MSD_Block(64, 128, kernel_size, 1, (1, 2, 3)),
            MSD_Block(128, 128, kernel_size, 1, (2, 3, 5)),
            MSD_Block(128, 128, kernel_size, 1, (2, 3, 5)),
        ])
        self.conv_post = norm_f(Conv1d(128, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        x = x.transpose(1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            if i >= self.fmap_depth:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class SubBandDiscriminator(nn.Module):

    def __init__(self, args, fmap_depth=2, device='cpu'):
        super().__init__()
        self.discriminators_t = nn.ModuleList([
            tSBD(6, 7, (5, 7, 11), fmap_depth=fmap_depth),
            tSBD(11, 5, (3, 5, 7), fmap_depth=fmap_depth),
            tSBD(16, 3, (1, 2, 3), fmap_depth=fmap_depth),
        ])
        self.discriminator_f = fSBD(args.frame_shift * args.seq_len // 64, 5, fmap_depth=fmap_depth)
        from mel2wav.pqmf import PQMF
        self.pqmf_t = PQMF(16, 256, 0.03, 10.0)
        self.pqmf_f = PQMF(64, 256, 0.1, 9.0)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # fsubband discriminator
        y_f = self.pqmf_f.analysis(y)  # (B, subbands, T // subbands)
        y_f_hat = self.pqmf_f.analysis(y_hat)
        y_d_r, fmap_r = self.discriminator_f(y_f)
        y_d_g, fmap_g = self.discriminator_f(y_f_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)

        # tsubband discriminator
        y_t = self.pqmf_t.analysis(y)  # (B, subbands, T // subbands)
        y_t_hat = self.pqmf_t.analysis(y_hat)
        eds = {0: 6, 1: 11, 2: 16}
        for i, d in enumerate(self.discriminators_t):
            y_d_r, fmap_r = d(y_t[:, :eds[i], :])
            y_d_g, fmap_g = d(y_t_hat[:, :eds[i], :])
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Pitch_predictor(torch.nn.Module):
    def __init__(self, num_dim_mel=80, dim=256, device='cpu'):
        super().__init__()

        self.dim = dim
        self.num_res_block = 4

        self.model = nn.Sequential(
            nn.ReflectionPad1d(3),
            WNConv1d(num_dim_mel, dim, kernel_size=7, padding=0)
        )

        # Pitch predictor
        model_pitch = []
        for i in range(self.num_res_block):
            model_pitch += [ResnetBlock(dim, dilation=i + 1)]
        model_pitch += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(dim, 1, kernel_size=7, padding=0)
        ]
        self.model_pitch = nn.Sequential(*model_pitch)

        # Voiced predictor
        model_voiced = []
        for i in range(self.num_res_block):
            model_voiced += [ResnetBlock(dim, dilation=i + 1)]
        model_voiced += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(dim, 1, kernel_size=7, padding=0)
        ]
        model_voiced += [nn.Sigmoid()]
        self.model_voiced = nn.Sequential(*model_voiced)

    def forward(self, mel):
        inter = self.model(mel)
        log_pitch = self.model_pitch(inter)
        voiced = self.model_voiced(inter)
        return log_pitch, voiced


class Speaker_predictor(torch.nn.Module):
    def __init__(self, speaker_num, device='cpu'):
        super().__init__()

        self.num_dim_mel = 80
        self.dim = 128
        self.sepaker_num = speaker_num

        self.audio2mel = Audio2Mel()
        self.model = nn.Sequential(
            nn.ReflectionPad1d(3),
            WNConv1d(self.num_dim_mel, self.dim, kernel_size=7, padding=0, stride=2),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(self.dim, self.dim * 2, kernel_size=7, padding=0, stride=2),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(self.dim * 2, self.dim * 4, kernel_size=7, padding=0, stride=2),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(self.dim * 4, self.dim * 4, kernel_size=7, padding=0, stride=2)
        )

        self.spk_predict = nn.Linear(self.dim * 4 * 5, self.sepaker_num)

    def forward(self, wav):
        mel = self.audio2mel(wav)[:, :, :-1]  # (B, 80, 80)
        inter = self.model(mel)  # (B, 512, 5)
        spk_out = self.spk_predict(inter.view(inter.shape[0], -1))
        return spk_out


def feature_loss(fmap_r, fmap_g):
    loss = 0
    num = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
            num += 1

    return loss / num


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    r_losses = 0.0
    g_losses = 0.0
    num = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        num += 1
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses += r_loss
        g_losses += g_loss

    return loss / num, r_losses / num, g_losses / num


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    num = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
        num += 1

    return loss / num, gen_losses


if __name__ == '__main__':
    model = Discriminator(3, 16, 4, 4)

    x = torch.randn(3, 1, 8196)

    scores = model(x)
    for score in scores:
        print(score[-1].shape)
