import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from torch import nn, Tensor
import math
from typing import Tuple
from einops import reduce, repeat
from tqdm import tqdm
import logging
import os
from einops import reduce, repeat, rearrange
from collections import OrderedDict
from apps.bigtts.audiogen.soundify.modules.blocks import NumberEmbedder, Repeat
from apps.bigtts.audiogen.soundify.modules.based_ctiga_llama import ModelArgs as LLamaArgs, LLaMa
from apps.bigtts.audiogen.soundify.modules.loss import sequence_mask

logger = logging.getLogger(__name__)


def load_vocoder_ckpt(model, checkpoint_path):
    print("setup vocoder")
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    if "state_dict" in checkpoint_dict:
        state_dict = checkpoint_dict['state_dict']
    else:
        state_dict = checkpoint_dict
    saved_state_dict = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split('.')
        parts = parts[1:]
        new_k = '.'.join(parts)
        if new_k in model.state_dict():
            saved_state_dict[new_k] = v
    if hasattr(model, 'module'):
        msg = model.module.load_state_dict(saved_state_dict)
    else:
        msg = model.load_state_dict(saved_state_dict)

    model.freeze_parameters()
    model.eval()
    print("setup vocoder DONE", msg)
    return model


def load_video_ckpt(model, checkpoint_path, freeze_video=True):
    print("setup video")

    checkpoint = torch.load(checkpoint_path)
    msg = model.load_state_dict(checkpoint)
    print("setup video DONE", msg)

    if freeze_video:
        model.freeze_parameters()
        model.eval()
        print("fix video encoder")
    else:
        print("train video encoder")

    return model


def rescale_noise_cfg(noise_cfg, noise_pred_cond, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class Distribution:

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):

    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class BernoulliDistribution(Distribution):

    def __init__(self, v1, v2):
        super().__init__()
        self.map = torch.tensor([v1, v2]).unsqueeze(0)

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        index = (torch.rand(num_samples, device=device) > 0.5).long()
        return self.map.repeat(num_samples, 1).to(device)[torch.arange(num_samples), index]


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class VideoProjector(nn.Module):

    def __init__(self, feat_dim=512, hidden_dim=768, mlp_depth=3, bias=False):
        super(VideoProjector, self).__init__()

        modules = [nn.Linear(feat_dim, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

        self.mlp = nn.Sequential(*modules)

    def forward(self, video_feats):
        return self.mlp(video_feats)


class Embedding(nn.Module):

    def __init__(self, modulation_features, num_layers: int = 2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(nn.Sequential(
            nn.Linear(modulation_features, modulation_features, bias=bias), nn.GELU()),
                          times=num_layers)

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


@dataclass
class ModelArgs:
    min_t: float = 0.0
    max_t: float = 1.0
    v2a_ratio: int = 10
    video_type: str = "cavp"
    video_feat_dim: int = 512
    in_channels: int = 128
    out_channels: int = 128
    local_cond_dim: int = 512
    time_embed_dim: int = 512
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    causal: bool = False
    llama_provider: str = 'ctiga'
    use_unet_style_skip_connect: bool = True
    use_qk_norm: str = 'head'
    bias: bool = False
    target_type: str = 'velocity'
    condition_type: str = 'prepend'
    video_repeat: int = 2
    drop_type: str = "video"


class Diffusion(nn.Module):

    def __init__(self,):
        super().__init__()

        hp = ModelArgs()
        self.hp = hp

        self.min_t = hp.min_t
        self.max_t = hp.max_t

        self.target_type = hp.target_type
        self.condition_type = hp.condition_type
        self.video_repeat = hp.video_repeat
        self.drop_type = hp.drop_type

        ############################################################################
        # time embedding
        self.time_embedding = Embedding(modulation_features=hp.time_embed_dim,
                                        num_layers=2,
                                        bias=hp.bias)

        self.speech_embedding = Embedding(modulation_features=hp.time_embed_dim,
                                          num_layers=2,
                                          bias=hp.bias)

        self.quality_embedding = Embedding(modulation_features=hp.time_embed_dim,
                                           num_layers=2,
                                           bias=hp.bias)

        self.music_embedding = Embedding(modulation_features=hp.time_embed_dim,
                                         num_layers=2,
                                         bias=hp.bias)

        # backbone
        llama_config = LLamaArgs(dim=hp.encoder_dim,
                                 n_layers=hp.encoder_n_layers,
                                 n_heads=hp.encoder_n_heads,
                                 causal=hp.causal,
                                 use_unet_style_skip_connect=hp.use_unet_style_skip_connect,
                                 use_qk_norm=hp.use_qk_norm)

        self.latent_encoder = LLaMa(llama_config, hp.llama_provider)

        self.latent_prenet = nn.Linear(hp.in_channels, hp.encoder_dim, bias=hp.bias)

        self.cond_prenet = nn.Linear(hp.time_embed_dim + hp.local_cond_dim,
                                     hp.encoder_dim,
                                     bias=hp.bias)

        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=hp.bias)

        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def cfg_drop(self, embed, prob=0.1):

        null_embed = torch.zeros_like(embed)

        B, device = embed.shape[0], embed.device
        mask = torch.bernoulli(torch.full([B, 1, 1], prob, device=device))
        embed = torch.where(mask.to(torch.bool), null_embed, embed)

        return embed

    def video_condition(self, video_embed):

        video_embed = F.normalize(video_embed, dim=-1)
        video_embed = torch.repeat_interleave(video_embed, self.video_repeat, dim=1)

        return video_embed

    def encoder(self, cond_embs, timesteps, latent, seq_mask=None):

        # time embedding
        time_emb = self.time_embedding(timesteps).unsqueeze(1)
        time_emb = time_emb.expand(cond_embs.shape[0], cond_embs.shape[1], -1)

        # concat condition.
        latent = self.latent_prenet(latent)
        cond = self.cond_prenet(torch.cat([time_emb, cond_embs], dim=-1))

        # add zeros
        latent = F.pad(latent, (0, 0, 3, 0), "constant", 0)

        x_noisy = latent + cond
        pred_v = self.latent_encoder(x_noisy, x_noisy.shape[1], attention_mask=seq_mask)

        pred_v = self.postnet(pred_v)

        # remove zeros
        pred_v = pred_v[:, 3::, :]

        return pred_v

    @torch.no_grad()
    def ddim_sample(self, video_embeds, step_num=25, cfg_w=7.5, norm_cfg=True):

        device = video_embeds.device
        frame_num = video_embeds.size(1)

        frame_num = frame_num * self.video_repeat

        latents = torch.randn([1, frame_num, self.hp.out_channels], device=device)

        self.max_t = 0.999
        sigmas = torch.linspace(self.max_t, self.min_t, step_num + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=latents.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        zero_value = torch.zeros(1, 1).to(device)
        one_value = torch.ones(1, 1).to(device)

        speech_embs = self.speech_embedding(zero_value).unsqueeze(1)
        music_embs = self.music_embedding(zero_value).unsqueeze(1)
        quality_embs = self.quality_embedding(one_value).unsqueeze(1)
        video_embeds = self.video_condition(video_embeds)

        cond_embs = torch.cat([speech_embs, music_embs, quality_embs, video_embeds], dim=1)

        null_embs = torch.zeros_like(cond_embs)
        cond_embs = torch.cat([cond_embs, null_embs], dim=0)

        for i in range(step_num):
            v_pred_cond, v_pred_uncond = self.encoder(cond_embs=cond_embs,
                                                      timesteps=sigmas[i],
                                                      latent=latents,
                                                      seq_mask=None).chunk(2)

            v_pred = cfg_w * v_pred_cond + (1 - cfg_w) * v_pred_uncond

            if norm_cfg:
                v_pred = rescale_noise_cfg(noise_cfg=v_pred,
                                           noise_pred_cond=v_pred_cond,
                                           guidance_rescale=0.7)

            x_pred = alphas[i] * latents - betas[i] * v_pred
            noise_pred = betas[i] * latents + alphas[i] * v_pred

            latents = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return latents


if __name__ == "__main__":
    config = ModelArgs()

    inputs = {}

    ######################## text ########################
    bts = 1
    max_duration = 10
    device = "cuda"

    video_fps = 8
    audio_fps = 32000

    max_video_len = int(video_fps * max_duration)
    max_audio_len = int(audio_fps * max_duration)

    video = torch.randn(bts, max_video_len, 3, 224, 224).to(device)  # mae
    audio = torch.randn(bts, 1, max_audio_len).to(device)
    speech = torch.randn(bts, 1).to(device)
    music = torch.randn(bts, 1).to(device)
    quality = torch.randn(bts, 1).to(device)

    video_lens = torch.randint(max_video_len // 2, max_video_len, [bts]).to(device)
    video_lens[0] = max_video_len
    audio_lens = (video_lens / video_fps * audio_fps)

    max_seq = max_video_len * 2 + 3
    seq_len = video_lens * 2 + 3
    mask = sequence_mask(seq_len, max_len=max_seq, device=device)

    inputs = {}
    inputs["audio"] = audio
    inputs["video"] = video
    inputs["speech"] = speech
    inputs["music"] = music
    inputs["quality"] = quality
    inputs["seqlen"] = seq_len
    inputs["mask"] = mask

    print(audio.shape, video.shape, max_seq, mask.shape, speech.shape, music.shape)

    from apps.bigtts.audiogen.soundify.modules.loss import MaskedMAELoss, MaskedMSELoss

    loss_funcs = MaskedMSELoss()
    with torch.autocast(device_type="cuda", enabled=True):
        model = Soundify(config).to("cuda:0")
        return_dict = model(inputs)

        pred_v = return_dict["pred_v"]
        target_v = return_dict["target_v"]

        mask = torch.ones(bts, max_seq).to(device)

        mask = mask[:, 3::]

        loss = loss_funcs(pred_v, target_v, mask)

        print(loss)

        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total {total_num/1024/1024} M, Trainable {trainable_num/1024/1024} M')

        resule = model.inference(inputs=inputs, step_num=50, sampler="ddim", cfg_w=7.5)
