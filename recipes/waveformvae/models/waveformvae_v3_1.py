import imp
import os
import json
import copy
import random
import math
import torch
from recipes.waveformvae.utils import commons
from recipes.waveformvae.models.waveformvae_modules import modules
from recipes.waveformvae.models.waveformvae_modules import attentions
# from recipes.waveformvae import monotonic_align
from recipes.waveformvae.utils import utils
import numpy as np
import os.path as osp
# import pyworld as pw

from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from recipes.waveformvae.utils.utils import AttrDict
from recipes.waveformvae.models.waveformvae_modules.bigvgan import BigVGAN
from einops import rearrange

try:
    from recipes.waveformvae.babble.utils import mfs, image_save
    from recipes.waveformvae.babble.datasets import phone_set, tone_set, prosody_set, wordcateg_set
    ELEMENTS_LEN_MAP = {
        'phones': len(phone_set),
        'tones': len(tone_set),
        'prosodies': len(prosody_set),
        'wordcateg': len(wordcateg_set),
        'word_categs': len(wordcateg_set),
    }
except:
    print("Only for testing!")
    ELEMENTS_LEN_MAP = {'phones': 148, 'tones': 13, 'prosodies': 7, 'wordcateg': 6, 'word_categs': 6}
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from recipes.waveformvae.utils.commons import init_weights, get_padding
from recipes.waveformvae.models.waveformvae_modules.mfd import MultiResolutionSTFTDiscriminator
from recipes.waveformvae.models.waveformvae_modules.pqmf import PQMF as PQMF_PWG
from recipes.waveformvae.utils.losses import kl_loss_standard, kl_loss_pp, kl_loss_vits
from recipes.waveformvae.utils.losses import kl_loss as kl_loss_no_form
# from recipes.waveformvae.models.waveformvae_modules.transformer import ProsodyDecoder as ProsodyGPT
# from recipes.waveformvae.models.waveformvae_modules.transformer import MelDecoder, SpeechDecoder
# from recipes.waveformvae.models.waveformvae_modules.transformer import get_pad_mask, get_subsequent_mask
# from recipes.waveformvae.pytorch_revgrad import RevGrad
from torchvision.transforms import ToTensor
from scipy.io.wavfile import read, write

import io
import PIL
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'monospace'
COLORS = [None, "tomato", "darkviolet", "blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
CMAPS = ['plasma', 'inferno', 'magma', 'cividis']


def extend2tuple(param, num, check=True):
    if isinstance(param, (list, tuple)):
        if check:
            assert len(param) == num
            param_list = list(param)
        else:
            assert len(param) > num
            param_list = list(param)[:num]
    else:
        param_list = [param] * num
    return tuple(param_list)


def split_text_line(text, category=1, max_words=25):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    category_len = len(text) // category
    texts = [text[i:i + category_len] for i in range(0, len(text), category_len)]
    history_pos = [0, ]
    end = False
    while not end:
        for i in range(max_words, max_words - max_words // 8, -1):
            pos = history_pos[-1] + i
            if pos < category_len:
                split_ok = True
                for text in texts:
                    if text[pos] != ' ':
                        split_ok = False
                        break
                if split_ok:
                    history_pos.append(pos)
                    break
            else:
                history_pos.append(category_len + 1)
                end = True
                break

    split_texts = []
    for i in range(1, len(history_pos)):
        for text in texts:
            split_texts.append(text[history_pos[i - 1]:history_pos[i]])
    return '\n'.join(split_texts)


def get_shape_2D(shapes_keywords, lengths_dict, batch_size, max_len=12000):
    shapes = []
    for shape_keyword in shapes_keywords:
        if shape_keyword is None or shape_keyword not in lengths_dict.keys():
            shape_tensor = torch.LongTensor([-1, ] * batch_size)
            shapes.append(shape_tensor)
        else:
            shapes.append(lengths_dict[shape_keyword])
    # defualt B T C Dataset
    if shapes is None or len(shapes) == 0:
        shapes = [torch.LongTensor([max_len, ] * batch_size), torch.LongTensor([-1, ] * batch_size)]
    return shapes


def plot_images(tensors, indice=None, labels=None, texts=None, split_text=False, num_split=2, max_words=80,
                shapes=None, width=10, height=4, align_direction='vertical', color_info=None, y_pos_info=None,
                y_lim_info=None, visual_methods=('show',), local_dir=None):
    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="C")
        ax.set_facecolor("None")
        return ax

    if isinstance(tensors, (list, tuple)):
        num_items = len(tensors)
    else:
        num_items = 1
    assert len(visual_methods) > 0, f"At least one visual methods should be provided"
    assert num_items % len(visual_methods) == 0
    tensors = extend2tuple(tensors, num_items)
    labels = extend2tuple(labels, num_items)
    buffer = io.BytesIO()
    if indice is None:
        indice = list(range(tensors[0].shape[0]))

    images = []

    for index in indice:
        tensors_save = []
        for tensor in tensors:
            tensor_plane = tensor[index]
            if shapes is not None:
                if tensor_plane.dim() == 2:
                    tensors_save.append(tensor_plane[:shapes[0][index].item(), :shapes[1][index].item()].detach().float())
                else:
                    tensors_save.append(tensor_plane[:shapes[1][index].item()].detach().float())
            else:
                tensors_save.append(tensor_plane.detach().float())
        if align_direction == 'vertical':
            groups = num_items // len(visual_methods)
            fig = plt.figure(figsize=(width, (groups + 1) * height))
            for k in range(groups):
                ax0 = fig.add_subplot(groups + 1, 1, k + 1)
                for i, visual_method in enumerate(visual_methods):
                    visual_method = visual_methods[i]
                    data = tensors_save[i + k * len(visual_methods)].cpu().numpy()
                    if i == 0:
                        ax = ax0
                        xlim = data.shape[1]
                    else:
                        ax = add_axis(fig, ax0)
                    if visual_method == 'show':
                        if color_info is None:
                            color = None
                        else:
                            color = color_info[i]
                        im = ax.imshow(data, cmap=cm.get_cmap(color), aspect='auto', interpolation='none')
                        fig.colorbar(mappable=im, shrink=0.65, orientation='vertical', ax=ax)
                        # ax.set_aspect(2.5, adjustable="box")
                        # ax.set_xlim(0, xlim)
                        ax.set_ylim(0, data.shape[0])
                        ax.set_title(labels[i + k * len(visual_methods)], fontsize="medium")
                        ax.tick_params(labelsize="x-small", left=True, labelleft=True)

                    elif visual_method == 'plot':
                        if color_info is None or color_info[i] is None:
                            color = COLORS[i]
                        else:
                            color = color_info[i]
                        ax.plot(data, color=color)
                        ax.set_xlim(0, xlim)
                        ax.set_ylim(0, y_lim_info[i])
                        if y_pos_info[i] == 'left':
                            ax.set_ylabel(labels[i + k * len(visual_methods)], color=color)
                            ax.yaxis.set_label_position("left")
                            ax.tick_params(
                                labelsize="x-small", colors=color, bottom=False, labelbottom=False
                            )
                        else:
                            ax.set_ylabel(labels[i], color=color)
                            ax.yaxis.set_label_position("right")
                            ax.tick_params(
                                labelsize="x-small",
                                colors=color,
                                bottom=False,
                                labelbottom=False,
                                left=False,
                                labelleft=False,
                                right=True,
                                labelright=True,
                            )
                    if i == 0:
                        ax0.set_anchor("C")
            if texts is not None:
                text = texts[index]
                if split_text:
                    text = split_text_line(text, category=num_split, max_words=max_words)
                # Set common labels
                fig.text(0.08, 0.1, text, ha='left', fontsize=12)
        buffer.seek(0)
        plt.savefig(buffer, format='jpg')
        buffer.seek(0)
        image = PIL.Image.open(buffer)
        image = ToTensor()(image)
        plt.close()
        images.append(image)
    images_tensor = torch.stack(images)
    if local_dir is not None:
        mfs.makedirs(local_dir)
        for i in range(images_tensor.size(0)):
            file_name = f"test_{i}.jpeg"
            file_path = osp.join(local_dir, file_name)
            image_save(tensor=images_tensor[i], path=file_path)
    return images_tensor


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, random_input=None, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)

        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            if random_input is None:
                z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            else:
                z = random_input[None, None, : x.size(2)].expand(x.size(0), 2, -1).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class StochasticDurationPredictor_prosody(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super(StochasticDurationPredictor_prosody, self).__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre_1 = nn.Conv1d(in_channels, filter_channels, 1)
        self.pre_2 = nn.Conv1d(in_channels, filter_channels, 1)

        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, prosody_hidden, x_mask, random_input=None, w=None, g=None, reverse=False, noise_scale=1.0):
        x = self.pre_1(x) + self.pre_2(prosody_hidden)
        if g is not None:
            x = x + self.cond(g)

        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            if random_input is None:
                z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            else:
                z = random_input[None, None, : x.size(2)].expand(x.size(0), 2, -1).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class DurationPredictor_prosody(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super(DurationPredictor_prosody, self).__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.relu = nn.ReLU()

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.pre_1 = nn.Conv1d(in_channels, in_channels, 1)
        self.pre_2 = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x, prosody_hidden, x_mask, g=None):
        x = self.pre_1(x) + self.pre_2(prosody_hidden)
        if g is not None:
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)

        x = self.relu(x)
        return x * x_mask


class GaussianUpsampling_DurationPredictor(nn.Module):
    def __init__(self, hidden_channels, cond_channels, kernel_size, dilation_rate, n_layers):
        super(GaussianUpsampling_DurationPredictor, self).__init__()
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.pre = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(cond_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, 2, 1)
        self.softplus = nn.Softplus()

    def forward(self, x, z, x_mask, y_lengths=None, gt_durations=None, fix_range=None):
        x = self.pre(x) + self.cond(z)
        x = self.enc(x, x_mask)
        x = self.proj(x) * x_mask
        x = self.softplus(x) * x_mask
        durations = x[:, 0, :]
        ranges = x[:, 1, :] + 1e-6
        if gt_durations is not None:
            attns = self.gaussian_upsample(gt_durations.squeeze(1), True, ranges, lengths=y_lengths, fix_range=fix_range)
        else:
            attns = self.gaussian_upsample(durations, False, ranges, lengths=y_lengths, fix_range=fix_range)
        return durations.unsqueeze(1), attns

    def gaussian_upsample(self, durations, gtd, ranges, lengths=None, fix_range=None):
        if lengths == None:
            lengths = torch.sum(durations, dim=1, keepdim=True)  # [B,1]
        else:
            if not gtd:
                durations = durations * (lengths / (durations.sum(-1) + 1e-6)).unsqueeze(-1)
        if fix_range is not None:
            ranges = fix_range * durations + 1e-6
        token_ends = torch.cumsum(durations, dim=1).float()  # [B,L]
        token_centers = (token_ends - 0.5 * durations).unsqueeze(2)  # [B,L,1]
        max_length = lengths.max().float().round().long()
        t = torch.arange(0, max_length).reshape(1, 1, max_length).to(durations.device)  # [1,1,T]

        sigma = ranges.unsqueeze(2)  # [B,L,1]
        mu = token_centers  # [B,L,1]
        molecular = torch.exp(-((t - mu) ** 2)/(sigma ** 2 + 1e-6))  # [B,L,T]
        denominator = torch.sum(molecular, dim=1, keepdim=True) + 1e-6  # [B,1,T]
        attn = molecular / denominator  # [B,L,T]

        return attn


class EmbeddingMapping(nn.Module):
    def __init__(self,
                 embedding_dict={'phones': 448, 'tones': 64, 'prosodies': 32, 'word_categs': 32},
                 combined_mode='concat',
                 padding_idx=0,
                 normalize=False):
        super().__init__()
        sanity_embedding_dict, output_num_embedding_dim = self.sanity_check(embedding_dict, combined_mode)
        module_dict = OrderedDict()
        for name, embedding_info in sanity_embedding_dict.items():
            num_embedding_dim = embedding_info['num_embedding_dim']
            num_embeddings = embedding_info['num_embeddings']
            if num_embedding_dim > 0:
                embedding = self.get_init_embedding(num_embeddings=num_embeddings, num_embedding_dim=num_embedding_dim,
                                                    padding_idx=padding_idx)
                module_dict[name] = embedding
        self.embedding_layers = nn.ModuleDict(module_dict)
        self.combined_mode = combined_mode
        self.output_num_embedding_dim = output_num_embedding_dim
        self.normalize = normalize

    def sanity_check(self, embedding_dict, combined_mode):
        sanity_embedding_dict = OrderedDict()
        assist_num_embedding_dim = 0
        for name, embedding_info in embedding_dict.items():
            if isinstance(embedding_info, int):
                assert name in ELEMENTS_LEN_MAP, f"The number of {name} should be provided."
                num_embedding_dim = embedding_info
                num_embeddings = ELEMENTS_LEN_MAP[name]
            elif isinstance(embedding_info, (list, tuple)):
                assert len(embedding_info) == 2, 'the embedding info should provide both only dimension and count'
                num_embedding_dim = embedding_info[0]
                num_embeddings = embedding_info[1]
            elif isinstance(embedding_info, dict):
                num_embedding_dim = embedding_info['num_embedding_dim']
                num_embeddings = embedding_info['num_embeddings']
            else:
                raise ValueError("EmbeddingMapping embedding info is not supported")
            sanity_embedding_dict[name] = dict(
                num_embeddings=num_embeddings,
                num_embedding_dim=num_embedding_dim,
            )
            if combined_mode is None:
                pass
            elif combined_mode.lower() == 'concat':
                assist_num_embedding_dim += num_embedding_dim
            elif combined_mode.lower() == 'add':
                if assist_num_embedding_dim == 0:
                    assist_num_embedding_dim = num_embedding_dim
                else:
                    assert assist_num_embedding_dim == num_embedding_dim, \
                        "Addition mode: all of num_embedding_dim should be sam_lightninge"
            else:
                raise ValueError("Multiple outputs of EmbeddingMapping should be concated or added, " +
                                 "but it got {combined_mode.lower()}")

        return sanity_embedding_dict, assist_num_embedding_dim

    def get_init_embedding(self, num_embeddings, num_embedding_dim, padding_idx=0):
        embedding = nn.Embedding(num_embeddings, num_embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(embedding.weight, 0.0, num_embedding_dim**-0.5)
        return embedding

    def forward(self, **kwargs):
        embedding_list = []
        for name, embedding_layer in self.embedding_layers.items():
            embedding = embedding_layer(kwargs[name])
            embedding_list.append(embedding)
        if len(embedding_list) > 1:
            if self.combined_mode == 'concat':
                mapping_output = torch.cat(embedding_list, dim=2)  # [B, T, dim1]
            elif self.combined_mode == 'add':
                mapping_output = torch.sum(embedding_list)
            else:
                mapping_output = embedding_list[0]
        else:
            mapping_output = embedding_list[0]
        if self.normalize:
            mapping_output = F.normalize(mapping_output)
        return mapping_output


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class TacolabelEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 embedding_dict={'phones': 448, 'tones': 64, 'prosodies': 32, 'word_categs': 32},
                 combined_mode='concat',
                 padding_idx=0,
                 normalize=False):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = EmbeddingMapping(embedding_dict, combined_mode, padding_idx, normalize)
        if self.emb.output_num_embedding_dim != hidden_channels:
            self.emb_proj = nn.Linear(self.emb.output_num_embedding_dim, hidden_channels)
        else:
            self.emb_proj = None
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, **kwargs):
        x_lengths = kwargs['phones_lengths']
        x = self.emb(**kwargs) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class TextEncoder_NoProb(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super(TextEncoder_NoProb, self).__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.pt2_upsampler = None
        self.mel_generator = None

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        x = x * x_mask
        return x, x_mask


class TextEncoder_NoProb_Tacolabel(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 embedding_dict={'phones': 448, 'tones': 64, 'prosodies': 32, 'word_categs': 32},
                 combined_mode='concat',
                 padding_idx=0,
                 normalize=False):
        super(TextEncoder_NoProb_Tacolabel, self).__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = EmbeddingMapping(embedding_dict=embedding_dict,
                                    combined_mode=combined_mode,
                                    padding_idx=padding_idx,
                                    normalize=normalize)
        if self.emb.output_num_embedding_dim != hidden_channels:
            self.emb_proj = nn.Linear(self.emb.output_num_embedding_dim, hidden_channels)
        else:
            self.emb_proj = None

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.pt2_upsampler = None
        self.mel_generator = None

    def forward(self, **kwargs):
        x_lengths = kwargs['phones_lengths']
        x = self.emb(**kwargs) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        x = x * x_mask
        return x, x_mask


class TextEmbedding(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_channels,
                 embedding_dict={'phones': 448, 'tones': 64, 'prosodies': 32, 'word_categs': 32},
                 combined_mode='concat',
                 padding_idx=0,
                 normalize=False):
        super(TextEmbedding, self).__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels

        self.emb = EmbeddingMapping(embedding_dict=embedding_dict,
                                    combined_mode=combined_mode,
                                    padding_idx=padding_idx,
                                    normalize=normalize)
        if self.emb.output_num_embedding_dim != hidden_channels:
            self.emb_proj = nn.Linear(self.emb.output_num_embedding_dim, hidden_channels)
        else:
            self.emb_proj = None

    def forward(self, **kwargs):
        x_lengths = kwargs['phones_lengths']
        x = self.emb(**kwargs) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = x * x_mask
        return x, x_mask


class ResidualCouplingBlock_Condition(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 affine=False,
                 gin_channels=0):
        super(ResidualCouplingBlock_Condition, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.affine = affine

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer_Condition(channels,
                                                                      hidden_channels,
                                                                      kernel_size,
                                                                      dilation_rate,
                                                                      n_layers,
                                                                      gin_channels=gin_channels,
                                                                      mean_only=True if not affine else False))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, c, g=None, reverse=False):
        logdet_tot = 0.0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, c, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, c, g=g, reverse=reverse)
        return x, logdet_tot


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 affine=False,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.affine = affine

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels,
                                                            hidden_channels,
                                                            kernel_size,
                                                            dilation_rate,
                                                            n_layers,
                                                            gin_channels=gin_channels,
                                                            mean_only=True if not affine else False))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        logdet_tot = 0.0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_tot


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ConditionPosteriorEncoder(nn.Module):
    def __init__(self,
                 spec_channels,
                 out_channels,
                 filter_channels,
                 hidden_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super(ConditionPosteriorEncoder, self).__init__()
        self.in_channels = spec_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.encoder = attentions.Decoder(hidden_channels,
                                          filter_channels,
                                          n_heads,
                                          n_layers,
                                          kernel_size,
                                          p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, c, c_mask):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.encoder(x, x_mask, h=c, h_mask=c_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ConditionPosteriorEncoder_PhoneLevel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ConditionPosteriorEncoder_PhoneLevel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.encoder = modules.WN(hidden_channels,
                                  kernel_size,
                                  dilation_rate,
                                  n_layers,
                                  gin_channels=0)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, c, x_lengths, attention, durations, phone_mask):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        c = self.cond(c) * x_mask
        x = x + c
        x = self.encoder(x, x_mask)

        attention_for_phone = attention.contiguous().transpose(1, 2)
        x = torch.bmm(x, attention_for_phone) * phone_mask
        durations = 1.0 / (durations + 1e-6)
        x = x * durations
        prosody_hidden = torch.detach(x)

        stats = self.proj(x) * phone_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))

        return z, m, logs, prosody_hidden, x_mask

    def infer(self, prosody_hidden, phone_mask):
        stats = self.proj(prosody_hidden) * phone_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z


class GradientReversalLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 alpha=1.):
        super(GradientReversalLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self._alpha = torch.tensor(alpha, requires_grad=False)

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.revgrad = RevGrad(alpha)

    def forward(self, x, x_mask):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        lin = self.proj(x) * x_mask
        return self.revgrad(lin)


class SineGen(nn.Module):

    def __init__(self,
                 sample_rate,
                 harmonic_num=0,
                 sine_amp=0.1,
                 noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False,
                 add_noise=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.add_noise = add_noise

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] -
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                                 device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            if self.add_noise:
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            else:
                noise_amp = uv * self.noise_std
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class JointF0(nn.Module):

    def __init__(self, sample_rate, mel_dim=80, f0_step=150):
        super().__init__()
        self.mel_dim = mel_dim
        self.f0_step = f0_step
        # self.wave_gen = SquareGen(sample_rate)
        self.wave_gen = SineGen(sample_rate, flag_for_pulse=True)

    def forward(self, mels, f0s=None):
        pred_log_f0s, pred_vuvs = 0, 0
        f0s = F.interpolate(f0s, scale_factor=self.f0_step, mode='nearest')
        f0s = f0s.transpose(1, 2)  # [B, D=1, T] -> [B, T, D=1]
        wave, _, _ = self.wave_gen(f0s)
        wave = wave.transpose(1, 2)
        # wave: [B, D=1, T]
        # pred_log_f0s: [B, D=1 , T]
        # pred_vuvs: [B, T] -> [B, T]
        return wave, pred_log_f0s, pred_vuvs


class F0Layer(nn.Module):

    def __init__(self, ch0, ch1, ch2, ch3, ch4, down_rates=[3, 4, 5, 5]):
        super().__init__()
        self.f0_layers = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, ch4, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch4, ch4, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch4, ch4, kernel_size=down_rates[0]*2-1, stride=down_rates[0], padding=down_rates[0]-1)),
                weight_norm(nn.Conv1d(ch4, ch3, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch3, ch3, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch3, ch3, kernel_size=down_rates[1]*2-1, stride=down_rates[1], padding=down_rates[1]-1)),
                weight_norm(nn.Conv1d(ch3, ch2, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch2, ch2, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch2, ch2, kernel_size=down_rates[2]*2-1, stride=down_rates[2], padding=down_rates[2]-1)),
                weight_norm(nn.Conv1d(ch2, ch1, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch1, ch1, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(
                    nn.Conv1d(ch1, ch1, kernel_size=down_rates[3]*2+1, stride=down_rates[3], padding=down_rates[3])),
                weight_norm(nn.Conv1d(ch1, ch0, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch0, ch0, kernel_size=3, padding=1)),
            ),
        ])

    def forward(self, sines):
        res = []
        for layer in self.f0_layers:
            sines = layer(sines)
            res.append(sines)
        res = res[::-1]
        return res


class Generator(nn.Module):
    def __init__(self,
                 initial_channel,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 use_sine=True,
                 gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsamples_rates = upsample_rates
        self.use_sine = use_sine

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        chs = []
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channels = upsample_initial_channel // (2 ** i)
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            if i == 0:
                chs.append(in_channels)
            chs.append(out_channels)
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(in_channels, out_channels, k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        if self.use_sine:
            self.joint_f0 = JointF0(24000, mel_dim=chs[0], f0_step=150)
            self.f0_layer = F0Layer(chs[0], chs[1], chs[2], chs[3], chs[4], [3, 4, 5, 5])

    def forward(self, x, g=None, f0s=None):
        x = self.conv_pre(x)
        if self.use_sine:
            wave, pred_log_f0s, pred_vuvs = self.joint_f0(x, f0s)
            f0_conds = self.f0_layer(wave)
        else:
            pred_log_f0s, pred_vuvs = 0, 0

        if g is not None:
            x = x + self.cond(g)
        if self.use_sine:
            x = x + f0_conds[0]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            if self.use_sine:
                x = x + f0_conds[i + 1]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x, pred_log_f0s, pred_vuvs

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class Generator_Original(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator_Original, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsamples_rates = upsample_rates

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class PreNet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(PreNet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([nn.Conv1d(in_size, out_size, 1, bias=False)
                                     for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class ConditionFusionLayer(nn.Module):
    def __init__(self, initial_channel, filter_channels, spec_channel, n_heads, n_layers, kernel_size, p_dropout):
        super(ConditionFusionLayer, self).__init__()

        self.condition_fusion_layer = attentions.Decoder(initial_channel,
                                                         filter_channels,
                                                         n_heads,
                                                         n_layers,
                                                         kernel_size,
                                                         p_dropout)
        self.proj = nn.Conv1d(initial_channel, 80, 1)

    def forward(self, x, x_lengths, c, c_lengths):
        x = self.condition_fusion_layer(x, x_lengths, c, c_lengths)
        mel = self.proj(x)
        return x, mel


class ConditionFusionLayer_WN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 gin_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ConditionFusionLayer_WN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, 80, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        mel = self.proj(x)
        return x, mel


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class PostNet(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 postnet_kernel_size,
                 postnet_n_convolutions):
        super(PostNet, self).__init__()
        self.pre = nn.Sequential(
            ConvNorm(in_channels,
                     out_channels,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(out_channels))

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80,
                         hidden_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hidden_channels))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hidden_channels,
                             hidden_channels,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hidden_channels))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hidden_channels, out_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(out_channels))
        )

    def forward(self, x, c):
        x = F.relu(self.pre(x))
        for i in range(len(self.convolutions) - 1):
            c = F.dropout(torch.tanh(self.convolutions[i](c)), 0.5, self.training)
        c = F.dropout(F.relu(self.convolutions[-1](c)), 0.5, self.training)
        out = x + c
        return out


class ConditionFusionLayer_WN_Add(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 spec_channels,
                 gin_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ConditionFusionLayer_WN_Add, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.spec_channels = spec_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.post_mel = nn.Conv1d(hidden_channels, 80, 1)

    def forward(self, x, c, x_mask, g=None):
        x = self.pre(x)
        c = self.cond(c)
        x = x + c
        x = x * x_mask
        x = self.enc(x, x_mask, g=g)
        mel = self.post_mel(x) * x_mask
        return x, mel


class PosteriorWaveEncoder(nn.Module):
    def __init__(self,
                 spec_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super(PosteriorWaveEncoder, self).__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.hidden_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs


class PosteriorWaveEncoder_SpeechGPT(nn.Module):
    def __init__(self,
                 spec_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super(PosteriorWaveEncoder_SpeechGPT, self).__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs


class CFL_Discriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(CFL_Discriminator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers)
        self.post = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        x = self.post(x)
        return x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        dicts = {
            2: {"subbands": 2, "taps": 62, "cutoff_ratio": 0.26699457, "beta": 9.0},
            3: {"subbands": 3, "taps": 62, "cutoff_ratio": 0.18366124, "beta": 9.0},
            5: {"subbands": 5, "taps": 72, "cutoff_ratio": 0.11463421, "beta": 9.0},
            7: {"subbands": 7, "taps": 82, "cutoff_ratio": 0.08427813, "beta": 9.0},
            11: {"subbands": 11, "taps": 92, "cutoff_ratio": 0.05690741, "beta": 9.0}
        }
        self.pqmf = PQMF_PWG(**dicts[self.period])

    def forward(self, x):
        fmap = []
        x = self.pqmf.analysis(x)  # [B, D, T]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class PQMF(nn.Module):

    def __init__(self, N, M, file_path="pqmf_hk_4_64.dat"):
        super().__init__()
        self.N = N  # nsubband
        self.M = M  # nfilter
        self.ana_conv_filter = nn.Conv1d(1,
                                         out_channels=N,
                                         kernel_size=M,
                                         stride=N,
                                         bias=False)
        data = np.reshape(np.fromfile(file_path, dtype=np.float32), (N, M))
        data = np.flipud(data.T).T
        gk = data.copy()
        data = np.reshape(data, (N, 1, M)).copy()
        dict_new = self.ana_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(data)
        self.ana_pad = nn.ConstantPad1d((M - N, 0), 0)
        self.ana_conv_filter.load_state_dict(dict_new)

        self.syn_pad = nn.ConstantPad1d((0, M // N - 1), 0)
        self.syn_conv_filter = nn.Conv1d(N,
                                         out_channels=N,
                                         kernel_size=M // N,
                                         stride=1,
                                         bias=False)
        gk = np.transpose(np.reshape(gk, (4, 16, 4)), (1, 0, 2)) * N
        gk = np.transpose(gk[::-1, :, :], (2, 1, 0)).copy()
        dict_new = self.syn_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(gk)
        self.syn_conv_filter.load_state_dict(dict_new)

        for param in self.parameters():
            param.requires_grad = False

    def analysis(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))

    def synthesis(self, inputs):
        return self.syn_conv_filter(self.syn_pad(inputs))

    def forward(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))


class MultiBandDiscriminator(torch.nn.Module):

    def __init__(self, use_spectral_norm=False, fmap_depth=0):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS()
        ])
        self.full_discriminator = DiscriminatorS()
        self.pqmf = PQMF_PWG(subbands=2, taps=62, cutoff_ratio=0.26699457, beta=9.0)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # full band discriminator
        y_d_r, fmap_r = self.full_discriminator(y)
        y_d_g, fmap_g = self.full_discriminator(y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)

        # multi band discriminator
        y = self.pqmf.analysis(y)
        y_hat = self.pqmf.analysis(y_hat)
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y[:, i:i + 1, :])
            y_d_g, fmap_g = d(y_hat[:, i:i + 1, :])
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, ifmbd=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.ifmbd = ifmbd
        periods = [2, 3, 5, 7, 11]

        self.extraD = MultiBandDiscriminator(use_spectral_norm=use_spectral_norm) if ifmbd else DiscriminatorS(use_spectral_norm=use_spectral_norm)
        self.mpd = nn.ModuleList([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        if self.ifmbd:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.extraD(y, y_hat)
        else:
            y_d_r, fmap_r = self.extraD(y)
            y_d_g, fmap_g = self.extraD(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.mpd):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32

        def convs(): return nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        win_length = self.window_length
        n_fft = win_length
        hop_length = int(self.window_length * self.hop_factor)

        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=False)  # [B, F, TT, 2]
        x = rearrange(x, "b f t c -> b c t f")
        x_bands = [x[..., b[0]: b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MBPRD(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, ifmbd=False):
        super(MBPRD, self).__init__()
        self.ifmbd = ifmbd
        periods = [2, 3, 5, 7, 11]
        fft_sizes = [2048, 1024, 512]
        bands = BANDS

        self.extraD = MultiBandDiscriminator(use_spectral_norm=use_spectral_norm) if ifmbd else DiscriminatorS(use_spectral_norm=use_spectral_norm)
        self.mpd = nn.ModuleList([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])
        self.mrd = nn.ModuleList([MRD(f, bands=bands) for f in fft_sizes])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        if self.ifmbd:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.extraD(y, y_hat)
        else:
            y_d_r, fmap_r = self.extraD(y)
            y_d_g, fmap_g = self.extraD(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.mpd):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.mrd):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Discriminator_FromBasisMelGAN(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(Discriminator_FromBasisMelGAN, self).__init__()
        self.msd = DiscriminatorS(use_spectral_norm=use_spectral_norm)
        self.mfd = MultiResolutionSTFTDiscriminator()

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        y_d_r, fmap_r = self.msd(y)
        y_d_g, fmap_g = self.msd(y_hat)
        outs_r = self.mfd(y)
        outs_g = self.mfd(y_hat)

        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
        for r, g in zip(outs_r, outs_g):
            y_d_rs.append(r[-1])
            y_d_gs.append(g[-1])
            fmap_rs.append(r[:-1])
            fmap_gs.append(g[:-1])

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        # print('z shape{},z_slice {},predict wav {}'.format(z.shape,z_slice.shape,o.shape))
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)


class SynthesizerTrnTacolabel(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 embedding_dict,
                 combined_mode,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 use_mas=True,
                 use_force=False,
                 **kwargs):

        super().__init__()
        self.embedding_dict = embedding_dict
        self.combined_mode = combined_mode
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp
        self.use_mas = use_mas
        self.use_force = use_force

        self.enc_p = TacolabelEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            embedding_dict=embedding_dict,
            combined_mode=combined_mode)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, **kwargs):
        y = kwargs['spec']
        y_lengths = kwargs['spec_lengths']
        sid = kwargs['speaker_ids']
        x, m_p, logs_p, x_mask = self.enc_p(**kwargs)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        # plot_images([m_p, m_q], indice=[0,1,2], labels=['prior', 'postior'], local_dir='visual')

        if self.use_mas:
            with torch.no_grad():
                # negative cross-entropy
                s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
                neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
                neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
                neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

                attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
                attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1), self.use_force).unsqueeze(1).detach()
            w = attn.sum(2)
        else:
            w = kwargs['durations']
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w, attn_mask).float().detach()
            w = w.unsqueeze(dim=1).half()

        if self.use_sdp:
            l_length = self.dp(x, x_mask, random_input=None, w=w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging
        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        mel = kwargs['mels']
        wav = kwargs['wav']
        return wav, mel, o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, data_dict, random_input=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(**data_dict)
        sid = data_dict['speaker_ids']
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, random_input=random_input, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        if random_input is None:
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        else:
            z_p = m_p + random_input[None, None, : m_p.shape[-1]].expand(m_p.shape[0], m_p.shape[1], -1) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)


class ProsodyPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ProsodyPredictor, self).__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.encoder = modules.WN(hidden_channels,
                                  kernel_size,
                                  dilation_rate,
                                  n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask):
        x = torch.detach(x)
        x = self.pre(x) * x_mask
        x = self.encoder(x, x_mask)
        x = self.proj(x) * x_mask
        return x


class ProsodyPredictor_KL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 gin_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ProsodyPredictor_KL, self).__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.encoder = modules.WN(hidden_channels,
                                  kernel_size,
                                  dilation_rate,
                                  n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = x * x_mask
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale)

        return z, m, logs


class ProsodyPredictor_Transformer_KL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 gin_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super(ProsodyPredictor_Transformer_KL, self).__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.encoder = attentions.Encoder(hidden_channels,
                                          filter_channels,
                                          n_heads,
                                          n_layers,
                                          kernel_size,
                                          p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None, noise_scale=1.0):
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = x * x_mask
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale)

        return z, m, logs


class F0PredictorV2(nn.ModuleList):
    def __init__(self, ir_dim=192, gin_channels=256):
        super().__init__()
        self.cond = nn.Conv1d(gin_channels, ir_dim, 1)
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(ir_dim, 128, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.f0_logit_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        self.f0_value_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x, g):
        x = x + self.cond(g)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.f0_predictor(x)
        pred_log_f0s = self.f0_value_layer(x).clamp(0)
        pred_f0s = torch.exp(pred_log_f0s) - 1.0
        pred_vuvs = self.f0_logit_layer(x)
        return pred_log_f0s, pred_vuvs.squeeze(1), pred_f0s.clamp(0)


class F0PredictorfromGTMel(nn.ModuleList):
    def __init__(self, ir_dim=80):
        super(F0PredictorfromGTMel, self).__init__()
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(ir_dim, 128, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )
        self.f0_logit_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        self.f0_value_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.f0_predictor(x)
        pred_f0s = 100 * self.f0_value_layer(x)
        pred_log_f0s = torch.log(pred_f0s.clamp(0) + 1)
        pred_vuvs = self.f0_logit_layer(x)
        return pred_log_f0s, pred_vuvs.squeeze(1), pred_f0s.clamp(0)


def random_select(e1, e2, p=0.5):
    assert e1.size(0) == e2.size(0)
    batch_size = e1.size(0)
    num_1 = int(batch_size * p)
    num_2 = batch_size - num_1
    index_1 = random.sample([i for i in range(batch_size)], num_1)
    index_2 = random.sample([i for i in range(batch_size)], num_2)
    index = index_1 + index_2
    e1_ = e1[index_1, :, :]
    e2_ = e2[index_2, :, :]
    return torch.cat([e1_, e2_], 0), index


def regular(linear, arange, p=0.15):
    len = arange[1] - arange[0]
    num = int(len * p)
    index = random.sample([i for i in range(arange[0], arange[1])], num)
    mask = torch.ones_like(linear).cpu()
    mask[:, index, :] = 0.0
    mask = mask.to(linear.device)
    linear = linear * mask
    return linear


class VIFSpeechTrn_Tacolabel(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 embedding_dict,
                 combined_mode,
                 padding_idx,
                 normalize,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_tacolabel=True,
                 affine=False,
                 sdp=False,
                 finetune=False,
                 speaker_id=-1,
                 infer=False,
                 f0_min=42.0,
                 f0_max=1000.0,
                 **kwargs):

        super(VIFSpeechTrn_Tacolabel, self).__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_tacolabel = use_tacolabel
        self.affine = affine
        self.sdp = sdp

        assert self.inter_channels == self.hidden_channels

        self.text_enc = \
            TextEncoder_NoProb_Tacolabel(n_vocab,
                                         hidden_channels,
                                         filter_channels,
                                         n_heads,
                                         n_layers,
                                         kernel_size,
                                         p_dropout,
                                         embedding_dict=embedding_dict,
                                         combined_mode=combined_mode,
                                         padding_idx=padding_idx,
                                         normalize=normalize) \
            if use_tacolabel else \
            TextEncoder_NoProb(n_vocab,
                               hidden_channels,
                               filter_channels,
                               n_heads,
                               n_layers,
                               kernel_size,
                               p_dropout)

        # Prosody Predictor
        self.prosody_predictor = ProsodyPredictor_Transformer_KL(in_channels=hidden_channels,
                                                                 out_channels=hidden_channels,
                                                                 hidden_channels=hidden_channels,
                                                                 filter_channels=filter_channels,
                                                                 gin_channels=gin_channels,
                                                                 n_heads=n_heads,
                                                                 n_layers=n_layers,
                                                                 kernel_size=kernel_size,
                                                                 p_dropout=p_dropout)

        self.prosody_transformer_decoder = ProsodyGPT(hidden_channels=hidden_channels,
                                                      n_layers=12,
                                                      n_head=n_heads,
                                                      d_k=hidden_channels // n_heads,
                                                      d_v=hidden_channels // n_heads,
                                                      d_model=hidden_channels,
                                                      d_inner=filter_channels,
                                                      n_position=1000)

        self.gu_duration_predictor = GaussianUpsampling_DurationPredictor(hidden_channels, hidden_channels, 5, 1, 8)

        self.acoustic_encoder = ConditionFusionLayer_WN_Add(in_channels=hidden_channels,
                                                            hidden_channels=hidden_channels,
                                                            spec_channels=spec_channels,
                                                            gin_channels=gin_channels,
                                                            kernel_size=5,
                                                            dilation_rate=1,
                                                            n_layers=12)

        self.wave_decoder = Generator_Original(hidden_channels,
                                               resblock,
                                               resblock_kernel_sizes,
                                               resblock_dilation_sizes,
                                               upsample_rates,
                                               upsample_initial_channel,
                                               upsample_kernel_sizes)

        self.prosody_enhancer = Generator_Original(initial_channel=hidden_channels,
                                                   resblock="2",
                                                   resblock_kernel_sizes=[3, 5, 7],
                                                   resblock_dilation_sizes=[[1, 2], [2, 6], [3, 12]],
                                                   upsample_rates=[10, 6, 5],
                                                   upsample_initial_channel=256,
                                                   upsample_kernel_sizes=[20, 12, 11])

        self.posterior_prosody_encoder = ConditionPosteriorEncoder_PhoneLevel(spec_channels,
                                                                              inter_channels,
                                                                              hidden_channels,
                                                                              5, 1, 8)

        self.prosody_flow = ResidualCouplingBlock(channels=inter_channels,
                                                  hidden_channels=hidden_channels,
                                                  kernel_size=5,
                                                  dilation_rate=1,
                                                  n_layers=4,
                                                  n_flows=4,
                                                  affine=affine,
                                                  gin_channels=0)

        self.wave_flow = ResidualCouplingBlock_Condition(channels=inter_channels,
                                                         hidden_channels=hidden_channels,
                                                         kernel_size=5,
                                                         dilation_rate=1,
                                                         n_layers=4,
                                                         n_flows=4,
                                                         affine=affine,
                                                         gin_channels=0)

        self.posterior_wave_encoder = PosteriorWaveEncoder(spec_channels=spec_channels,
                                                           hidden_channels=hidden_channels,
                                                           kernel_size=5,
                                                           dilation_rate=1,
                                                           n_layers=8,
                                                           gin_channels=0)

        self.finetune = finetune
        if self.finetune:
            print("[INFO]: Frozen text encoder...")
            for param in self.text_enc.parameters():
                param.requires_grad = False
            self.speaker_id = speaker_id

        if n_speakers > 1:
            self.emb_speaker = nn.Embedding(n_speakers, gin_channels)

    # def calculate_f0(self, wav, sample_rate, f0_ms=6.25, f0_floor=42.0, f0_ceil=1000.0, mode='harvest'):
    #     f0_fn = pw.harvest if mode == 'harvest' else pw.dio
    #     f0, t = f0_fn(wav.astype(np.float64),
    #                   sample_rate,
    #                   frame_period=f0_ms,
    #                   f0_floor=f0_floor,
    #                   f0_ceil=f0_ceil)
    #     f0 = pw.stonemask(wav.astype(np.float64), f0, t, sample_rate)
    #     return f0

    def forward(self, **kwargs):
        y = kwargs['spec']
        y_lengths = kwargs['spec_lengths']
        sid = kwargs['speaker_ids']
        if self.finetune and self.speaker_id != -1:
            sid[:] = self.speaker_id
            sid = sid.long()

        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**kwargs)
        else:
            x = kwargs["x"]
            x_lengths = kwargs["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        if self.n_speakers > 0:
            g = self.emb_speaker(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # Expand linguistic feature
        w = kwargs["durations"]
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w, attn_mask).float().detach().contiguous().squeeze(1).transpose(1, 2)
        w = w.unsqueeze(dim=1).float()

        # Expand
        x_hard_expand = torch.bmm(x, attn)

        # Condition
        z, m_q, logs_q, prosody_hidden, y_mask = self.posterior_prosody_encoder(y, x_hard_expand, y_lengths, attn, w, x_mask)

        # Predict prosody
        z_hat, m_pp, logs_pp = self.prosody_predictor(x.detach(), x_mask, g=g.detach())

        enc_output = x.contiguous().transpose(1, 2).detach()
        z_prosody = z.contiguous().transpose(1, 2).detach()
        z_prosody = torch.cat([torch.randn(z_prosody.size(0), 1, z_prosody.size(-1)).to(z_prosody.device), z_prosody[:, :-1, :]], dim=1)
        trg_mask = get_pad_mask(x_mask.squeeze(1), 0) & get_subsequent_mask(x_mask.squeeze(1))
        z_ptd, m_ptd, logs_ptd = self.prosody_transformer_decoder(z_prosody, enc_output, trg_mask)
        m_ptd = m_ptd.contiguous().transpose(1, 2)
        logs_ptd = logs_ptd.contiguous().transpose(1, 2)

        # Predict duration
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        w_hat, attn_hat = self.gu_duration_predictor(x, z, x_mask, y_lengths=y_lengths, gt_durations=w, fix_range=None)
        l_length = F.l1_loss(w, w_hat)

        # Flow to normal distribution
        z_p, logdet = self.prosody_flow(z, x_mask)

        # Expand
        x_expand = torch.bmm(x, attn_hat) * y_mask
        z_expand = torch.bmm(z, attn_hat) * y_mask

        # Random mask x_expand
        random_mask = torch.ones_like(x_expand)[:, :, :1]
        random_index = random.choices([i for i in range(random_mask.size(1))], k=int(random_mask.size(1) * 0.5))
        random_mask[:, random_index, :] = 0.
        x_expand = x_expand * random_mask

        # Fuse condition with z
        intermediate_representation, z_mel_hat = self.acoustic_encoder(x_expand, z_expand, y_mask, g=g)

        # Wave Flow
        z_pwe, m_pwe, logs_pwe = self.posterior_wave_encoder(y, y_mask)
        z_wf, logdet_wf = self.wave_flow(z_pwe, y_mask, intermediate_representation)

        # Random cut
        z_slice, ids_slice = commons.rand_slice_segments(z_pwe, y_lengths, self.segment_size)

        # Prosody enhanced
        ir_slice = commons.slice_segments(intermediate_representation, ids_slice, self.segment_size)
        o_aux = self.prosody_enhancer(ir_slice)

        # Generator
        o = self.wave_decoder(z_slice)

        return o, o_aux, ids_slice, x_mask, y_mask, z_mel_hat, (z_wf, logdet_wf, m_pwe, logs_pwe), (z_p, logdet, m_q, logs_q, m_pp, logs_pp, m_ptd, logs_ptd), l_length

    def infer(self, data_dict, sid=None, sid_pp=None, ones=0, test_vocoder=False, noise_scale=1.0, length_scale=1.0, max_len=None, use_gt_duration=False, reference=None, use_prosody_predictor=False, ifcuda=False, ref_audio=None, CAL_F0_FROM_WAV=False):
        if sid is None:
            sid = data_dict['speaker_ids']
        else:
            if ifcuda:
                sid = torch.Tensor([sid]).long().cuda()
            else:
                sid = torch.Tensor([sid]).long()

        if sid_pp is not None:
            if ifcuda:
                sid_pp = torch.Tensor([sid_pp]).long().cuda()
            else:
                sid_pp = torch.Tensor([sid_pp]).long()

        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**data_dict)
        else:
            x = data_dict["x"]
            x_lengths = data_dict["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        if self.n_speakers > 0:
            g = self.emb_speaker(sid).unsqueeze(-1)  # [b, h, 1]
            if sid_pp is not None:
                g_pp = self.emb_speaker(sid_pp).unsqueeze(-1)
            else:
                g_pp = g
        else:
            g = None

        # Prosody predictor
        z, _, _ = self.prosody_predictor(x, x_mask, g=g_pp, noise_scale=noise_scale)

        # Predict duration
        w, _ = self.gu_duration_predictor(x, z, x_mask)

        if use_gt_duration:
            w = data_dict["durations"].unsqueeze(1).float() * x_mask * length_scale
        else:
            w = w * x_mask * length_scale
        w_ceil = torch.round(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask).float().contiguous().squeeze(1).transpose(1, 2)

        # Expand
        x_expand = torch.bmm(x, attn)

        if not use_prosody_predictor:
            if reference is None:
                # Flow to normal distribution
                if ones == 0:
                    z_p = 0.0 + torch.zeros_like(x_mask.expand(-1, x_expand.size(1), -1)) * 1.0 * noise_scale
                    z, _ = self.prosody_flow(z_p, x_mask, reverse=True)
                elif ones == 1:
                    z_p = 0.0 + torch.ones_like(x_mask.expand(-1, x_expand.size(1), -1)) * 0.5 * noise_scale
                    z, _ = self.prosody_flow(z_p, x_mask, reverse=True)
                elif ones == -1:
                    z_p = 0.0 + torch.ones_like(x_mask.expand(-1, x_expand.size(1), -1)) * -0.5 * 1.0 * noise_scale
                    z, _ = self.prosody_flow(z_p, x_mask, reverse=True)
                elif ones == None:
                    z = self.generate_random_prosody(data_dict)
                else:
                    raise Exception("Unsupport ones!")
            else:
                if test_vocoder:
                    reference = reference[:, :, :y_lengths[0]]  # one batch
                    z_pwe, m_pwe, logs_pwe = self.posterior_wave_encoder(reference, y_mask)
                    o = self.wave_decoder(z_pwe)
                    return o, x_mask, y_mask, (None, None)
                else:
                    spec = reference[:, :, :y_lengths[0]]
                    w_ceil = w_ceil.squeeze(dim=1).float()
                    z_prosody, _, _, _ = self.extract_prosody(data_dict, spec, w_ceil)
                    z_prosody_regenerate = self.generate_prosody(x, z_prosody, data_dict)
                    o = self.synthesis_by_prosody(data_dict, sid, z_prosody_regenerate)
                    return o, _, _, (_, _)

        z_expand = torch.bmm(z, attn) * y_mask

        # Fuse condition with z
        intermediate_representation, mel = self.acoustic_encoder(x_expand, z_expand, y_mask, g=g)
        z_from_flow, _ = self.wave_flow(torch.randn_like(intermediate_representation), y_mask, intermediate_representation, reverse=True)

        o = self.wave_decoder(z_from_flow * y_mask)

        return o, x_mask, y_mask, (z, mel)

    def extract_prosody(self, tacolabel, spec, w):
        x, x_mask = self.text_enc(**tacolabel)
        y_lengths = torch.Tensor([spec.size(-1)]).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w, attn_mask).float().detach().contiguous().squeeze(1).transpose(1, 2)
        w = w.unsqueeze(dim=1).float()

        # Expand
        x_expand = torch.bmm(x, attn)

        # Extract Prosody
        z, m_q, logs_q, prosody_hidden, y_mask = self.posterior_prosody_encoder(spec, x_expand, y_lengths, attn, w, x_mask)

        return z, x, m_q, logs_q

    def synthesis_by_prosody(self, tacolabel, speaker_id, z_prosody):
        # Encode text
        x, x_mask = self.text_enc(**tacolabel)
        sid = torch.Tensor([speaker_id]).long()
        g = self.emb_speaker(sid).unsqueeze(-1)

        # Predict duration
        w, _ = self.gu_duration_predictor(x, z_prosody, x_mask)
        w_ceil = torch.round(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask).float().contiguous().squeeze(1).transpose(1, 2)

        x_expand = torch.bmm(x, attn)
        z_prosody = torch.bmm(z_prosody, attn) * y_mask

        # Fuse condition with z
        intermediate_representation, mel = self.acoustic_encoder(x_expand, z_prosody, y_mask, g=g)
        z_from_flow, _ = self.wave_flow(torch.randn_like(intermediate_representation), y_mask, intermediate_representation, reverse=True)

        # Wave decoder
        o = self.wave_decoder(z_from_flow * y_mask)

        return o

    def generate_prosody(self, x, z, tacolabel):
        enc_output = x.contiguous().transpose(1, 2)
        target_enc_output, _ = self.text_enc(**tacolabel)
        target_enc_output = target_enc_output.contiguous().transpose(1, 2)
        enc_output = torch.cat([enc_output, target_enc_output], dim=1)

        dec_output = z.contiguous().transpose(1, 2)
        dec_output = torch.cat([torch.randn(dec_output.size(0), 1, dec_output.size(2)).to(dec_output.device), dec_output], dim=1)

        z_prosody = self.prosody_transformer_decoder.infer(dec_output, enc_output)
        z_prosody = z_prosody.contiguous().transpose(1, 2)
        return z_prosody

    def generate_random_prosody(self, tacolabel):
        enc_output, _ = self.text_enc(**tacolabel)
        enc_output = enc_output.contiguous().transpose(1, 2)
        z_prosody = self.prosody_transformer_decoder.infer(None, enc_output)
        z_prosody = z_prosody.contiguous().transpose(1, 2)
        return z_prosody


def gen_duration(x_l, y_l):
    index_list = sorted(random.sample([i for i in range(y_l)], x_l - 1))
    index_list = [0] + index_list + [y_l]
    duration = [index_list[i + 1] - index_list[i] for i in range(len(index_list) - 1)]
    duration[-1] += 1
    return duration


class MelGPT(nn.Module):
    def __init__(self,
                 embedding_dict,
                 combined_mode,
                 padding_idx,
                 normalize,
                 n_vocab,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spec_channels,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 segment_size=9600 // 300,
                 use_tacolabel=True):
        super(MelGPT, self).__init__()

        self.segment_size = segment_size
        self.use_tacolabel = use_tacolabel
        self.hidden_channels = hidden_channels
        self.text_enc = \
            TextEmbedding(n_vocab,
                          hidden_channels,
                          embedding_dict=embedding_dict,
                          combined_mode=combined_mode,
                          padding_idx=padding_idx,
                          normalize=normalize) \
            if use_tacolabel else \
            TextEncoder_NoProb(n_vocab,
                               hidden_channels,
                               filter_channels,
                               n_heads,
                               n_layers,
                               kernel_size,
                               p_dropout)

        self.sequence_distribution_channels = 16

        self.mel_decoder = SpeechDecoder(in_channels=self.sequence_distribution_channels,
                                         hidden_channels=hidden_channels,
                                         out_channels=self.sequence_distribution_channels,
                                         n_layers=n_layers,
                                         n_head=n_heads,
                                         d_k=hidden_channels // n_heads,
                                         d_v=hidden_channels // n_heads,
                                         d_model=hidden_channels,
                                         d_inner=filter_channels,
                                         n_position=5000)

        self.posterior_wave_encoder = PosteriorWaveEncoder_SpeechGPT(spec_channels=spec_channels,
                                                                     hidden_channels=hidden_channels,
                                                                     out_channels=self.sequence_distribution_channels,
                                                                     kernel_size=5,
                                                                     dilation_rate=1,
                                                                     n_layers=8,
                                                                     gin_channels=0)

        self.flow = ResidualCouplingBlock(channels=self.sequence_distribution_channels,
                                          hidden_channels=hidden_channels,
                                          kernel_size=5,
                                          dilation_rate=1,
                                          n_layers=4,
                                          n_flows=4)

        self.vocoder = Generator_Original(initial_channel=self.sequence_distribution_channels,
                                          resblock=resblock,
                                          resblock_kernel_sizes=resblock_kernel_sizes,
                                          resblock_dilation_sizes=resblock_dilation_sizes,
                                          upsample_rates=upsample_rates,
                                          upsample_initial_channel=upsample_initial_channel,
                                          upsample_kernel_sizes=upsample_kernel_sizes)

        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**kwargs)
        else:
            x = kwargs["x"]
            x_lengths = kwargs["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        x_lengths = kwargs["phones_lengths"]
        y_lengths = kwargs["spec_lengths"]
        max_mel_length = y_lengths.max()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        src_mask = get_pad_mask(x_mask.squeeze(1), 0)
        trg_mask = get_pad_mask(y_mask.squeeze(1), 0) & get_subsequent_mask(y_mask.squeeze(1))

        # encode spectrogram
        spec = kwargs["spec"]
        z, m, logs = self.posterior_wave_encoder(spec, y_mask)
        m = m * y_mask
        logs = logs * y_mask
        z_p, logdet = self.flow(z, y_mask)
        kl_loss = kl_loss_no_form(z_p, logdet, logs, y_mask)

        # Random cut
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.vocoder(z_slice)

        token_target = []
        for i in range(x.size(0)):
            # token target
            token_target_one_batch = torch.zeros(y_lengths[i]).to(x.device)
            token_target_one_batch[-1] = 1
            token_target_one_batch = token_target_one_batch.long()
            token_target_one_batch = F.pad(token_target_one_batch, (0, max_mel_length - token_target_one_batch.size(-1)), value=1).long()
            token_target.append(token_target_one_batch)
        token_target = torch.stack(token_target, 0)

        trg_mask = get_pad_mask(y_mask.squeeze(1), 0) & get_subsequent_mask(y_mask.squeeze(1))

        dec_input = z.contiguous().transpose(1, 2)
        dec_input = torch.cat([torch.randn(dec_input.size(0), 1, dec_input.size(-1)).to(dec_input.device), dec_input[:, :-1, :]], dim=1)
        dec_input = dec_input.detach()
        memory = x.contiguous().transpose(1, 2)
        _, m_gpt, logs_gpt, token_output = self.mel_decoder(dec_input, trg_mask, memory, src_mask)
        token_output = token_output.contiguous().transpose(1, 2)

        kl_loss_gpt = kl_loss_pp(m_gpt, logs_gpt, m.detach(), logs.detach(), y_mask)
        ce_loss = self.ce_loss(token_output.reshape(-1, token_output.size(-1)), token_target.reshape(-1))

        return kl_loss_gpt, kl_loss, ce_loss, o, ids_slice

    def infer(self, data_dict):
        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**data_dict)
        else:
            x = data_dict["x"]
            x_lengths = data_dict["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        z_pred = []
        memory = x.contiguous().transpose(1, 2)
        src_mask = get_pad_mask(x_mask.squeeze(1), 0)
        dec_output = torch.randn(x.size(0), self.sequence_distribution_channels, 1).to(x.device)
        for i in range(1000):  # Fix length
            lengths = torch.Tensor([dec_output.size(-1)]).long()
            mask = torch.unsqueeze(commons.sequence_mask(lengths, dec_output.size(2)), 1).to(x.dtype)
            trg_mask = get_pad_mask(mask.squeeze(1), 0) & get_subsequent_mask(mask.squeeze(1))
            dec_output = dec_output.contiguous().transpose(1, 2)
            z, m, logs, token_output = self.mel_decoder(dec_output, trg_mask, memory, src_mask)
            # print(m.max().item(), m.min().item(), logs.max().item(), logs.min().item())
            z_pred.append(z[:, :, -1])
            dec_output = torch.cat([dec_output.contiguous().transpose(1, 2), z_pred[-1].unsqueeze(-1)], dim=-1)
            _, index = torch.max(token_output[0, :, -1], 0)
            if index.item() == 1:
                break
        z_pred = torch.stack(z_pred, dim=-1)
        o = self.vocoder(z_pred)
        return o

    def infer_from_mel(self, data_dict, spec):
        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**data_dict)
        else:
            x = data_dict["x"]
            x_lengths = data_dict["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        memory = x.contiguous().transpose(1, 2)
        src_mask = get_pad_mask(x_mask.squeeze(1), 0)
        y_mask = torch.unsqueeze(commons.sequence_mask(torch.Tensor([spec.size(-1)]).long(), None), 1).long().to(spec.device)
        z, _, _ = self.posterior_wave_encoder(spec, y_mask)
        dec_input = torch.cat([torch.randn(z.size(0), z.size(1), 1).to(x.device), z], dim=-1)
        lengths = torch.Tensor([dec_input.size(-1)]).long()
        mask = torch.unsqueeze(commons.sequence_mask(lengths, dec_input.size(2)), 1).to(x.dtype)
        trg_mask = get_pad_mask(mask.squeeze(1), 0) & get_subsequent_mask(mask.squeeze(1))
        dec_input = dec_input.contiguous().transpose(1, 2)
        z, _, _, _ = self.mel_decoder(dec_input, trg_mask, memory, src_mask)

        o = self.vocoder(z)
        return o

    def infer_vocoder(self, spec):
        y_mask = torch.unsqueeze(commons.sequence_mask(torch.Tensor([spec.size(-1)]).long(), None), 1).long().to(spec.device)
        z, _, _ = self.posterior_wave_encoder(spec, y_mask)
        o = self.vocoder(z)
        return o

    def infer_in_context_learning(self, data_dict, spec, phone_dict):
        # Encode text
        x, _ = self.text_enc(**data_dict)
        target_x, _ = self.text_enc(**phone_dict)
        x = torch.cat([x, target_x], dim=-1)

        x_mask = torch.unsqueeze(commons.sequence_mask(torch.Tensor([x.size(-1)]).long(), None), 1).long().to(x.device)
        y_mask = torch.unsqueeze(commons.sequence_mask(torch.Tensor([spec.size(-1)]).long(), None), 1).long().to(spec.device)

        memory = x.contiguous().transpose(1, 2)
        src_mask = get_pad_mask(x_mask.squeeze(1), 0)

        z, _, _ = self.posterior_wave_encoder(spec, y_mask)
        dec_output = torch.cat([torch.randn(z.size(0), z.size(1), 1).to(x.device), z], dim=-1)

        for i in range(1000):  # Fix length
            lengths = torch.Tensor([dec_output.size(-1)]).long()
            mask = torch.unsqueeze(commons.sequence_mask(lengths, dec_output.size(2)), 1).to(x.dtype)
            trg_mask = get_pad_mask(mask.squeeze(1), 0) & get_subsequent_mask(mask.squeeze(1))
            dec_output = dec_output.contiguous().transpose(1, 2)
            z, m, logs, token_output = self.mel_decoder(dec_output, trg_mask, memory, src_mask)
            dec_output = torch.cat([dec_output.contiguous().transpose(1, 2), z[:, :, -1:]], dim=-1)
            _, index = torch.max(token_output[0, :, -1], 0)
            if index.item() == 1:
                break
        o = self.vocoder(dec_output[:, :, 1:])
        return o


class Downsample(nn.Module):

    def __init__(self, down_scale, in_channels, out_channels):
        super().__init__()
        self.down_scale = down_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        self.conv = norm_f(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=2 * down_scale + 1,
                      stride=down_scale,
                      padding_mode='zeros',
                      padding=down_scale))

    def forward(self, x):
        x = self.conv(x)
        return x

    def remove_weight_norm(self):
        try:
            remove_weight_norm(self.conv)
        except:
            pass


class WVEncoder(torch.nn.Module):
    def __init__(self, out_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, downsample_rates, downsample_initial_channel):
        super(WVEncoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_downsamples = len(downsample_rates)
        self.downsamples_rates = downsample_rates
        self.out_channels = out_channels

        self.conv_pre = Conv1d(1, downsample_initial_channel, 59, 1, padding=29)

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.downs = nn.ModuleList()
        for i, u in enumerate(downsample_rates):
            self.downs.append(Downsample(u, downsample_initial_channel * (2**i), downsample_initial_channel * (2**(i+1))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = downsample_initial_channel * (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, out_channels * 2, 7, 1, padding=3, bias=False)

        self.dropout = nn.Dropout(0.5)

        self.downs.apply(init_weights)

    def forward(self, x, accer):
        # print(x.shape)
        # print(self.conv_pre)
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.downs[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        # accelerate
        x = x + self.dropout(accer)  # add dropout
        x = F.leaky_relu(x)
        stats = self.conv_post(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        # print('>>> only mean var 2.0 ')

        return z, m, logs

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class MelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MelPredictor, self).__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.net = modules.WN(hidden_channels=hidden_channels,
                              kernel_size=5,
                              dilation_rate=1,
                              n_layers=4,
                              p_dropout=0.1)
        self.out = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.pre(x)
        x_mask = torch.ones(x.size(0), 1, x.size(2)).long().to(x.device)
        x = self.net(x, x_mask)
        x = self.out(x)
        return x


class SpecAccelerator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(SpecAccelerator, self).__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.net_1 = modules.WN(hidden_channels=hidden_channels,
                                kernel_size=5,
                                dilation_rate=1,
                                n_layers=4,
                                p_dropout=0.1)
        self.down = Downsample(down_scale=2, in_channels=hidden_channels, out_channels=hidden_channels * 2)
        self.net_2 = modules.WN(hidden_channels=hidden_channels * 2,
                                kernel_size=5,
                                dilation_rate=1,
                                n_layers=4,
                                p_dropout=0.1)
        self.out = nn.Conv1d(hidden_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.pre(x)
        x_mask = torch.ones(x.size(0), 1, x.size(2)).long().to(x.device)
        x = self.net_1(x, x_mask)
        x = self.down(x)
        x_mask = torch.ones(x.size(0), 1, x.size(2)).long().to(x.device)
        x = self.net_2(x, x_mask)
        x = self.out(x)
        return x


class WaveformVAE(nn.Module):
    def __init__(self,
                 sd_channels,
                 spec_channels,
                 hidden_channels,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 downsample_rates,
                 downsample_initial_channel,
                 segment_size=9600):
        super(WaveformVAE, self).__init__()
        self.segment_size = segment_size
        self.down_size = 1
        for rate in downsample_rates:
            self.down_size *= rate

        with open(os.path.join("recipes", "waveformvae", "conf", "bigvgan.json")) as f:
            data = f.read()

        self.encoder = WVEncoder(out_channels=sd_channels,
                                 resblock=resblock,
                                 resblock_kernel_sizes=resblock_kernel_sizes,
                                 resblock_dilation_sizes=resblock_dilation_sizes,
                                 downsample_rates=downsample_rates,
                                 downsample_initial_channel=downsample_initial_channel)

        self.flow = ResidualCouplingBlock(channels=sd_channels,
                                          hidden_channels=hidden_channels,
                                          kernel_size=5,
                                          dilation_rate=1,
                                          n_layers=4,
                                          n_flows=8)

        self.mel_predictor = MelPredictor(sd_channels, 80, 256)
        self.spec_accelerator = SpecAccelerator(spec_channels, downsample_initial_channel * (2 ** len(downsample_rates)), 256)

        json_config = json.loads(data)
        h = AttrDict(json_config)
        h.num_mels = sd_channels
        self.vocoder = BigVGAN(h)

    def forward(self, x, spec):
        # encoder
        accer = self.spec_accelerator(spec)
        z, m, logs = self.encoder(x, accer)
        mel_z = self.mel_predictor(z)
        mask = torch.ones_like(z)[:, :1, :].long()
        z_p, logdet = self.flow(z, mask)
        kl_loss = kl_loss_no_form(z_p, logdet, logs, mask)

        # decoder
        out = self.vocoder(z)
        return out, mel_z, kl_loss, m, logs

    def infer(self, x, spec):
        # encoder
        accer = self.spec_accelerator(spec)
        z, m, logs = self.encoder(x, accer)
        # print(f"m mean: {m.mean().item()}, m std: {m.std().item()}, logs mean: {logs.mean().item()}, logs std: {logs.std().item()}")
        # decoder
        out = self.vocoder(z)
        return out


if __name__ == "__main__":
    # Single speaker testing
    from text import symbols
    print(f"Length of symbols is {len(symbols)}.")
    model = VIFSpeechTrn_Tacolabel(None,
                                   None,
                                   None,
                                   None,
                                   n_vocab=len(symbols),
                                   spec_channels=1024,
                                   segment_size=8192 // 256,
                                   inter_channels=192,
                                   hidden_channels=192,
                                   filter_channels=768,
                                   n_heads=2,
                                   n_layers=6,
                                   kernel_size=3,
                                   p_dropout=0.1,
                                   resblock="1",
                                   resblock_kernel_sizes=[3, 7, 11],
                                   resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                                   upsample_rates=[5, 5, 4, 3],
                                   upsample_initial_channel=512,
                                   upsample_kernel_sizes=[15, 15, 12, 9],
                                   n_speakers=1000,
                                   gin_channels=256,
                                   use_tacolabel=False,
                                   affine=False,
                                   sdp=False,
                                   finetune=True)
    num_of_param = utils.get_param_num(model)
    print(f"Number of parameter of model is {num_of_param}.")

    x_len_list = [6, 6, 6, 6, 6, 6, 9, 7]
    x = [[random.randint(1, len(symbols)-1) for _ in range(l)] for l in x_len_list]
    y_len_list = [i * 6 for i in x_len_list]
    x_durations = [gen_duration(x_l, y_l) for x_l, y_l in zip(x_len_list, y_len_list)]
    y = [torch.randn(l, 1024) for l in y_len_list]

    x_max_len = max(x_len_list)
    y_max_len = max(y_len_list)
    x_lengths = torch.Tensor(x_len_list).long()
    y_lengths = torch.Tensor(y_len_list).long()
    x = [e + [0 for _ in range(x_max_len - l)] for e, l in zip(x, x_len_list)]
    x = torch.Tensor(x).long()
    x_durations = [e + [0 for _ in range(x_max_len - l)] for e, l in zip(x_durations, x_len_list)]
    x_durations = torch.Tensor(x_durations).long()
    y = [torch.cat([e, torch.zeros(y_max_len - l, 1024)], 0) for e, l in zip(y, y_len_list)]
    y = torch.stack(y, 0).float().transpose(1, 2)
    f0s = torch.randn(y.size(0), y.size(-1) * 2)
    mel = torch.randn(y.size(0), 80, y.size(-1))

    sid = torch.Tensor([random.randint(1, 10) for _ in range(x_lengths.size(0))]).long()

    # Test forwarding
    data_dict = {
        "x": x,
        "x_lengths": x_lengths,
        "phones_lengths": x_lengths,
        "durations": x_durations,
        "spec": y,
        "spec_lengths": y_lengths,
        "speaker_ids": sid,
        "f0": f0s,
        "mels": mel
    }

    o, o_aux, ids_slice, x_mask, y_mask, z_mel_hat, \
        (z_wf, logdet_wf, m_pwe, logs_pwe), \
        (z_p, logdet, m_q, logs_q, m_pp, logs_pp, m_ptd, logs_ptd), \
        l_length = model(**data_dict)
    print(f"\nx size: {x.size()}")
    print(f"y size: {y.size()}")
    print(f"o size: {o.size()}")
    print(f"ids_slice: {ids_slice}")
    print(f"x, y mask size: {x_mask.size()} {y_mask.size()}")
    print(f"z_p size: {z_p.size()}")
    print(f"m_q size: {m_q.size()}")
    print(f"logs_q size: {logs_q.size()}")

    # Test inference
    x_lengths = torch.Tensor([100]).long()
    x = torch.Tensor([[random.randint(1, len(symbols)-1) for _ in range(x_lengths[0])]]).long()
    y_lengths = torch.Tensor([x_lengths[0] * 6]).long()
    with torch.no_grad():
        o, x_mask, y_mask, (z, mel) = model.infer({"x": x, "x_lengths": x_lengths}, sid=1, use_prosody_predictor=True)
    print(f"\no size: {o.size()}")
    print(f"x, y mask size: {x_mask.size()} {y_mask.size()}")
    print(f"z size: {z.size()}")
    print("Testing done.")

    try:
        discriminator = Discriminator_FromBasisMelGAN()
        y = torch.randn(2, 1, 12345)
        y_hat = torch.randn_like(y)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y, y_hat)
        for y_d_r, y_d_g in zip(y_d_rs, y_d_gs):
            print(f"y_r, y_g: {y_d_r.size()} {y_d_g.size()}")
        for f_r, f_g in zip(fmap_rs, fmap_gs):
            for r, g in zip(f_r, f_g):
                print(f"f_r, f_g: {r.size()} {g.size()}")
    except Exception as e:
        print(f"Not support MKL: {str(e)}")

    transformer_decoder = ProsodyGPT(hidden_channels=192,
                                     n_layers=8,
                                     n_head=2,
                                     d_k=192 // 2,
                                     d_v=192 // 2,
                                     d_model=192,
                                     d_inner=768,
                                     n_position=1000)
    dec_output = torch.randn(2, 10, 192)
    trg_seq = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).long()
    enc_output = torch.randn(2, 10, 192)
    trg_mask = get_pad_mask(trg_seq, 0) & get_subsequent_mask(trg_seq)
    dec_output, m, logs = transformer_decoder(dec_output, enc_output, trg_mask)
    print(f"dec_output: {dec_output.size()}")

    melgpt = MelGPT(None,
                    None,
                    None,
                    None,
                    n_vocab=len(symbols),
                    hidden_channels=384,
                    filter_channels=1536,
                    n_heads=12,
                    n_layers=12,
                    kernel_size=3,
                    p_dropout=0.1,
                    spec_channels=1024,
                    resblock="1",
                    resblock_kernel_sizes=[3, 7, 11],
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    upsample_rates=[5, 5, 4, 3],
                    upsample_initial_channel=512,
                    upsample_kernel_sizes=[15, 15, 12, 9],
                    use_tacolabel=False)
    num_of_param = utils.get_param_num(melgpt)
    print(f"Number of parameter of MelGPT is {num_of_param}.")
    kl_loss_gpt, kl_loss, ce_loss, o, ids_slice = melgpt(**data_dict)
    print(f"kl loss of GPT is {kl_loss_gpt.item()}.")
    data_dict = {
        "x": x[:1, :],
        "x_lengths": x_lengths[:1],
    }
    # o = melgpt.infer(data_dict)
    print(f"o is {o.size()}")

    waveform_vae = WaveformVAE(sd_channels=32,
                               hidden_channels=256,
                               resblock="1",
                               resblock_kernel_sizes=[3, 7, 11],
                               resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                               downsample_rates=[2, 2, 2, 3, 5, 5],
                               downsample_initial_channel=32)
    num_of_param = utils.get_param_num(waveform_vae)
    print(f"Number of parameter of Waveform VAE is {num_of_param}.")

    x = torch.randn(4, 1, 100000)
    length = torch.Tensor([100000, 100000, 100000, 100000]).long()
    out, ids_slice, kl_loss = waveform_vae(x, length)
