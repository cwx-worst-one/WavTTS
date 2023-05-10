'''
USM text encoder
'''
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

from core.models.pretrained.utils import *
from core.models.layers.embedding import AbsPositionalEncoding
from core.models.layers.feedforward_transformer import FFTBlock


def get_mask_from_lengths(lengths, max_len=None):
    '''generate mask matrix from length tensor'''
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def pad(input_ele, mel_max_length=None):
    '''pad tensor'''
    if mel_max_length:
        max_len = mel_max_length
    else:
        # pylint:disable=consider-using-generator
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = []
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class TextEncoder(nn.Module):
    """Text encoder"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_extractor = Encoder(args)
        self.resampler = Resampler(args)
        self.refiner = Decoder(args)
        self.mel_linear = nn.Linear(
            args.transformer.decoder_hidden,
            args.mel.n_mel_channels,
        )
        self.token_mask_prob = args.get('token_mask_prob', 0)
        self.token_mask_rand = args.get('token_mask_rand', 0)

    # pylint:disable=invalid-name
    def apply_mask(self, x, mask):
        '''apply mask for input tensor'''
        if self.token_mask_prob <= 0:
            return x, None

        B, L = x.shape
        mask_indices = compute_mask_indices(
            (B, L),
            mask,
            self.token_mask_prob,
            1,
            require_same_masks=False,
        )
        mask_indices = torch.from_numpy(mask_indices).to(x.device)
        if self.token_mask_rand:
            noise = torch.randint_like(x, self.args.tgt_vocab_size)
            x[mask_indices] = noise[mask_indices]
        else:
            x[mask_indices] = 0

        return x, mask_indices

    def forward(
        self,
        texts,
        src_masks,
        mel_masks=None,
        d_targets=None,
        mask=False,
    ):
        '''forward'''
        max_mel_len = mel_masks.shape[1] if mel_masks is not None else None

        if mask:
            texts, _ = self.apply_mask(texts, src_masks)

        output = self.embed_extractor(texts, src_masks)

        (output, log_d_predictions, d_rounded, mel_lens,) = self.resampler(
            output,
            src_masks,
            max_mel_len,
            d_targets,
        )

        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)
        output, mel_masks = self.refiner(output, mel_masks)
        output = self.mel_linear(output)

        return (
            output,
            log_d_predictions,
            d_rounded,
            mel_masks,
        )

    # pylint:disable=unused-variable
    def predict_duration(self, texts, src_masks):
        '''predict duration'''
        output = self.embed_extractor(texts, src_masks)

        (output, log_d_predictions, d_rounded, mel_lens,) = self.resampler(
            output,
            src_masks,
        )

        return d_rounded


class Encoder(nn.Module):
    '''Encoder'''

    def __init__(self, args):
        super().__init__()

        n_position = args.max_seq_len + 1
        n_src_vocab = args.tgt_vocab_size
        d_word_vec = args.transformer.encoder_hidden
        n_layers = args.transformer.encoder_layer
        n_head = args.transformer.encoder_head
        d_k = d_v = args.transformer.encoder_hidden // args.transformer.encoder_head
        d_model = args.transformer.encoder_hidden
        d_inner = args.transformer.conv_filter_size
        kernel_size = args.transformer.conv_kernel_size
        dropout = args.transformer.encoder_dropout

        self.max_seq_len = args.max_seq_len
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
        self.position_enc = AbsPositionalEncoding(d_word_vec, 0, n_position)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask):
        '''forward'''
        enc_output = self.src_word_emb(src_seq)
        enc_output = self.position_enc(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, mask=mask, slf_attn_mask=mask.unsqueeze(1))

        return enc_output


class Decoder(nn.Module):
    '''Decoder'''

    def __init__(self, args):
        '''init'''
        super().__init__()

        n_position = args.max_seq_len + 1
        d_word_vec = args.transformer.decoder_hidden
        n_layers = args.transformer.decoder_layer
        n_head = args.transformer.decoder_head
        d_k = d_v = args.transformer.decoder_hidden // args.transformer.decoder_head
        d_model = args.transformer.decoder_hidden
        d_inner = args.transformer.conv_filter_size
        kernel_size = args.transformer.conv_kernel_size
        dropout = args.transformer.decoder_dropout

        self.max_seq_len = args.max_seq_len
        self.d_model = d_model

        self.position_enc = AbsPositionalEncoding(d_word_vec, 0, n_position)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask):
        '''forward'''
        _, max_len = enc_seq.shape[0], enc_seq.shape[1]

        max_len = min(max_len, self.max_seq_len)
        dec_output = self.position_enc(enc_seq[:, :max_len])
        mask = mask[:, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, _ = dec_layer(dec_output, mask=mask, slf_attn_mask=mask.unsqueeze(1))

        return dec_output, mask


class Resampler(nn.Module):
    """Resampler: duration prediction & upsampling"""

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.fixed_duration = args.get('fixed_duration', -1)
        if self.fixed_duration <= 0:
            self.duration_predictor = DurationPredictor(args)
            self.length_regulator = LengthRegulator()

    # pylint:disable=invalid-name
    def forward(
        self,
        x,
        src_mask,
        max_len=None,
        duration_target=None,
        d_control=1.0,
    ):
        '''forward'''
        if self.fixed_duration > 0:
            B, _, H = x.shape
            x_repeat = x.unsqueeze(2).repeat(1, 1, self.fixed_duration, 1).view(B, -1, H)
            mel_len = (~src_mask).sum(-1) * self.fixed_duration
            return x_repeat, None, None, mel_len

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            mel_len,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def length_regulator(self, x, duration, max_len):
        '''length regulator'''
        output = []
        mel_len = []
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).cuda()

    def expand(self, batch, predicted):
        '''expand'''
        out = []

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        '''forward'''
        output, mel_len = self.length_regulator(x, duration, max_len)
        return output, mel_len


class DurationPredictor(nn.Module):
    """Duration Predictor"""

    def __init__(self, args):
        '''init'''
        super().__init__()

        self.input_size = args.transformer.encoder_hidden
        self.filter_size = args.duration_predictor.filter_size
        self.kernel = args.duration_predictor.kernel_size
        self.conv_output_size = args.duration_predictor.filter_size
        self.dropout = args.duration_predictor.dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        '''forward'''
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    # pylint:disable=unused-argument
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        '''forward'''
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
