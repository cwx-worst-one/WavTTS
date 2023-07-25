from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from tqdm import tqdm

from .ssl_frontend import SSLFrontend, make_pad_mask


def hubert_cal_conv_out_dim(length):
    for _, k, s in eval('[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'):
        length = (length - k) // s + 1
    return length

def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

# Compute for each data point the closest center
def compute_codes(dataset, centers, device):
    num_points = dataset.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    inertia = 0
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
        inertia += distances[range(distances.shape[0]), min_ind].sum()
    return codes, inertia.item()

def log(t, eps=1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres=0.95):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

def sample(predict_logits, temp, mode="gumbel", device="cuda:0"):
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=1) # [b, d]
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample().unsqueeze(1).to(device)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits)
        samples = gumbel_sample(predict_logits, temp).unsqueeze(dim=1)
    else:
        raise NotImplementedError()
    return samples


class FrozenSSLEmbedder(nn.Module):
    
    """Uses the w2v-bert Conformer encoder for audio"""
    
    def __init__(self,
                 model='assets/hubert_v2/semantic.jit.pt',
                 config='assets/hubert_v2/config.yaml',
                 centers='assets/hubert_v2/centroids_epoch_143.npy',
                 device="cpu"):
        super().__init__()
        self.device = device
        self.frontend = SSLFrontend()
        config_args = yaml.safe_load(open(config))
        frontend_conf = config_args.get('frontend_conf', {})
        frontend = SSLFrontend(fs=frontend_conf.get('fs', 24000), 
                                n_fft=frontend_conf.get('n_fft', 1024),
                                win_length=frontend_conf.get('win_length', 600),
                                hop_length=frontend_conf.get('hop_length', 240),
                                )
        encoder_type = config_args.get('encoder')
        torch.set_num_threads(30)
        torch._C._jit_set_bailout_depth(0)
        self.model = torch.jit.load(model, map_location=torch.device('cpu'))
        self.model.to(device)
        self.model.eval()
        self.centers = torch.from_numpy(np.load(centers)).to(device)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
        #print("SSL Model Params: {:.4f}M".format(
        #    sum(p.numel() for p in self.model.parameters()) / 1e6))

    def forward(self, audio):
        return self.encode_audio(audio)

    def encode_audio(self, audios):
        if isinstance(audios, torch.Tensor):
            audio_lens = torch.tensor([audios.shape[-1],] * len(audios)).long()
        elif isinstance(audios, tuple) or isinstance(audios, list):
            audios, audio_lens = audios
        batch_size = len(audios)
        # audios = torchaudio.transforms.Resample(
        #     orig_freq=24000, new_freq=16000)(audios)
        resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)
        if audios.is_cuda:
            resampler = resampler.to(self.device)
        audios = resampler(audios)
        audio_lens = torch.tensor([audios.shape[-1],] * batch_size).long()
        feat_lens = hubert_cal_conv_out_dim(audio_lens)
        feat_masks = make_pad_mask(feat_lens)

        semantics, masks = self.model(audios.to(self.device),
                                      audio_lens.to(self.device),
                                      feat_masks.to(self.device))
        tokens, tlens = [], []
        for i in range(len(masks)):
            length = masks[i].sum()
            feat = semantics[i, :length].detach()
            codes, _ = compute_codes(feat, self.centers, self.device)
            tokens.append(codes.view(-1))
            #tlens.append(len(tokens[-1]))
        #for i, c in enumerate(tokens):
        #    if tlens[i] < 749:
        #        tokens[i] = torch.cat([c + 1, torch.zeros(749 - tlens[i], device=c.device)], 0)
        return torch.stack(tokens).to(self.device)#, tlens


class FrozenSemanticLMEmbedder(nn.Module):
    
    """Uses the SemanticLM encoder for text and audio"""
    
    def __init__(self,
            pretrained="assets/epoch=26-step=234000-accu=40.54.ckpt",
            mulan_sep=True,
            return_mulan=False,
            context_duration=10,
            use_ppg=False,
            device="cpu"):
        super().__init__()
        self.device = device
        self.mulan_sep = mulan_sep
        self.return_mulan = return_mulan
        self.context_duration = context_duration
        from recipes.audio_lm.lit_modules.v4_1.lit_semantic import SemanticModule
        pretrained = 'assets/semantic_sparse_mulan247_429M_epoch=22-step=88000-accu=36.77.ckpt'
        #from recipes.audio_lm.lit_modules.v5.lit_semantic import SemanticModule
        #pretrained = 'assets/epoch=40-step=284000-accu=42.47.ckpt'
        self.mulan_sep = True
        self.model = SemanticModule.load_from_checkpoint(pretrained, 'cpu').eval()
        #else:
        #    from recipes.audio_lm.lit_modules.v2.lit_semantic import SemanticModule
        #    self.mulan_sep = True
        #    self.model = SemanticModule.load_from_checkpoint(pretrained, 'cpu').eval()
        self.use_ppg = use_ppg
        self.model = self.model.model.to(device)
        self.mulan = FrozenMuLanEmbedder(version="247", return_rvq=True, device=device)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("Semantic Language Model Params: {:.4f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6))

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return self.encode_text(x)
        return self.encode_audio(x)

    def get_semantic_tokens(self, mulan_embeds, mulan_tokens, continue_from=None):
        batch_size = mulan_tokens.size(0)
        mulan_tokens = mulan_tokens + 1024 + 1 # [b, 12] # offset: w2-vert + EOS 1
        if self.mulan_sep:
            mulan_tokens = mulan_tokens + torch.arange(mulan_tokens.size(1)).to(mulan_embeds.device) * 1024
        eos_ids = torch.zeros([batch_size, 1], dtype=torch.long, device=mulan_embeds.device) + 1024 # offset: w2v-bert
        slice_range = []
        semantic_frame_rate = 25
        semantic_offset = -3
        stride_in_sec = 5
        beg = 0
        while True:
            end = beg + 10 * semantic_frame_rate + semantic_offset
            if end >= self.context_duration * semantic_frame_rate + semantic_offset:
                end = self.context_duration * semantic_frame_rate + semantic_offset
                beg = end - (10 * semantic_frame_rate + semantic_offset)
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += stride_in_sec * semantic_frame_rate
        prev_end = 0
        semantic_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            if cache_len == 0:
                input_tokens = torch.cat([mulan_tokens, eos_ids], dim=1)
            else:
                prefix_semantic_samples = semantic_samples[:, cur_beg: cur_beg + cache_len]
                input_tokens = torch.cat([mulan_tokens, eos_ids, prefix_semantic_samples], dim=1)
            past_key_values = None

            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for _ in pbar:
                pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
                semantic_outputs = self.model(input_tokens, past_key_values=past_key_values, use_cache=True)
                logits = semantic_outputs['logits'] # [b, t, d]
                predict_logits = logits[:, -1, 0 : 1024]
                samples = sample(predict_logits, temp=0.9, mode="gumbel")
                past_key_values = semantic_outputs['past_key_values']
                input_tokens = samples
                if semantic_samples is None:
                    semantic_samples = samples
                else:
                    semantic_samples = torch.cat([semantic_samples, samples], dim=1)
        return semantic_samples

    def encode_text(self, text, continue_from=None):
        embs, tokens = self.mulan.encode_text(text)
        return self.get_semantic_tokens(embs, tokens, continue_from)

    def encode_audio(self, audio):
        embs, tokens = self.mulan.encode_audio(audio)
        return self.get_semantic_tokens(embs, tokens)


